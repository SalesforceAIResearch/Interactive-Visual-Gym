# Official-guidance audit — Qwen3.8-27B (`qwen3.8-27b`, port 18300) — 2026-09-18/19

**The question.** Does our serving setup match what Qwen and vLLM recommend for this checkpoint, item by item?
Where it differs, is the difference deliberate, or is it a fix that makes the model better or faster at no cost?

**How it was measured.** Every change was benched on the 24 genuine calls (`pass3_setup/bench.py`, one request in
flight, tags `qwen38_*_audit`), on one A100-80GB (GPU 0 or 1), vLLM 0.19.1. The columns are:
- median and p90 wall latency;
- *fixed* and *per-token*: a least-squares split `latency = fixed + per_token × completion_tokens`
  (`fit_latency.py`), so that a variant that merely talks more is not mistaken for a slower server;
- schema-valid replies out of 24;
- *spoke*: replies with a non-empty `text` (the reference spoke 15/24);
- *ref-hits*: on the 5 calls whose round-4 reference raised a flag, how many times the model raised the same error
  type. Those calls are #0 mis_prep, #2 wrong_order, #5 forget_next_step, #13 forget_next_step and #18 mis_prep.
  False flags on the other 19 calls are listed too.

`bench_audit.py --detect` prints this summary, and `bench_audit.py --profile-json` benches a PROFILES entry
in-process without editing the engine.

Sources:
- **[card]** the local copy of `huggingface.co/Qwen/Qwen3.8-27B`, `README.md` at snapshot `1d4bf0f`.
- **[recipe]** `recipes.vllm.ai/Qwen/Qwen3.8-27B`, from its source file
  `github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml` (updated 2026-09-14). The raw copy is
  `docs_raw/Qwen_Qwen3.8-27B.yaml` and the rendered page is `docs_raw/recipe_qwen38.txt`.
- **[cfg]** the checkpoint's `preprocessor_config.json` and `generation_config.json`.

## Item table

| # | item | ours (before) | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint | `Qwen/Qwen3.8-27B` bf16, snapshot 1d4bf0f | `Qwen/Qwen3.8-27B` is the `default` (bf16) variant. The FP8 and NVFP4 variants need FP8/FP4 hardware; `RedHatAI/Qwen3.8-27B-INT4` lists `supported_hardware: [h100, h200, b200, …]`, with no A100, and ships an FP8 KV-cache scheme | [recipe] `variants:` | **matches.** The only official variant that runs on sm_80 is bf16. INT4 is not recipe-supported on A100 and carries FP8 KV scales, so it was not adopted |
| 2 | engine | vLLM 0.19.1 (PyPI), transformers 5.15.0 | `min_vllm_version: "0.17.0"` and `transformers>=5.8.0`. The recipe's own runs used a 0.27-range image, and "`spec_decoding` in particular wants the gated-delta-net speculative fix (vllm#51812 / #51674), which no released tag carries yet" | [recipe] `model:` comments | **matches the minimum.** #51812 fixes GDN gates only when a batch mixes non-speculative and speculative rows. Our bench runs single-stream, so it does not apply there; see item 7 for the correctness check. #51674 is a Blackwell kernel |
| 2b | required flags | `--reasoning-parser qwen3`, `--limit-mm-per-prompt {"image":8}`, `--max-model-len 32768`, no `--trust-remote-code` | `--reasoning-parser qwen3` "is not optional in practice"; no remote code (no `auto_map`); 262,144 native context | [recipe] `features.reasoning`, `base_args: []` | **matches.** 32k is a deliberate cap: prompts are about 7.5k tokens |
| 3 | chat template / rendering | vLLM **auto-detected content format `string`** for this template. It then flattened our `[text, 8 images]` message to *all 8 images first, joined by `\n`, then the text*: the order we send was silently discarded, and 8 stray `\n` tokens were added | the HF processor renders the content parts in the order given, with no separators. The card's examples put media first | [card] "Image Input", "Instruct (or Non-Thinking) Mode"; `template_diff.py` | **FIX: `--chat-template-content-format openai`.** The server's token stream is now identical to the official processor's (6,838 = 6,838 tokens, checked for both orders) |
| 3b | content order | text first, then frames (the harness says "Below are your last 8 frames") — but see #3: the model actually saw images first | card examples: image, then question | [card] | **deviates on purpose, and now actually delivered.** Harness order was benched against card order with #3 fixed, below: text-first detects clearly more |
| 4 | visual input | 8 separate images at 1042×396 → smart-resize 1056×384 = 24×66 patches → **396 tokens/frame**, 3,168 for 8 | processor `size.shortest_edge 65,536 px`, `longest_edge 16,777,216 px`, patch 16, merge 2: our frames sit well inside the range, so no down-scaling. Video is supported ("By default, `fps=2`") | [cfg] `preprocessor_config.json`; [card] "Video Input" | **matches (images).** The card has no rule for short frame sequences. As video, the 8 frames would be merged in temporal pairs (`temporal_patch_size: 2`), so two frames 1.2 s apart would share one token grid. That halves the tokens but blurs exactly the per-tick change the assistant must judge, and it is sampled far below the 2 fps the model is used to. Kept as images; not benched (it needs a harness change) |
| 5 | sampling | T 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 (profile `qwen38-schema`) | "Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`" | [card] Best Practices §1 | **matches.** `presence_penalty` is honoured under MTP: vLLM 0.19.1's rejection sampler applies penalties (`v1/sample/rejection_sampler.py:260-318`). min_p 0 and repetition 1.0 are the defaults. The server's `generation_config` defaults (T 1.0 / top_p 0.95, the thinking set) are overridden by the request |
| 6 | thinking | off server-wide: `--default-chat-template-kwargs {"enable_thinking":false}` → template emits an empty `<think>\n\n</think>\n\n` | "Qwen3.8-27B will think by default … `chat_template_kwargs: {"enable_thinking": False}`"; server-wide via `--default-chat-template-kwargs` | [card] "Instruct (or Non-Thinking) Mode"; [recipe] "Thinking modes" | **matches.** Thinking is off for latency, which the game clock charges. The rendered tail was checked: `<|im_start|>assistant\n<think>\n\n</think>\n\n` |
| 7 | efficiency — speculative decoding | MTP, `num_speculative_tokens: 3` | "Native MTP head, 3 draft tokens" (default mode). `rtx_pro_6000` and `dgx_spark_gb10` override this to 5. DFlash2 is opt-in and needs vLLM ≥ 0.28 | [recipe] `features.spec_decoding` | **method matches; depth changed (FIX).** At depth 3 the measured mean acceptance length is 3.70 of 4. MTP roughly halves the call: see the table below (greedy 10.83 → 5.61 s; decode 26.9 → 69 tok/s). Greedy replies with MTP on vs off are byte-identical on 15/24 calls; the other 9 diverge only after 285–1,917 characters (bf16 tie flips from different GEMM shapes), with the same flags and the same spoke count. **FIX: depth 5**, the recipe's own override for two other GPUs: mean accepted length 5.20 of 6; median 4.73 s against 5.62–5.83 s; decode 86 against 68 tok/s; 24/24 valid |
| 7b | efficiency — other flags | prefix caching on (Mamba `align` mode, which vLLM marks experimental), CUDA graphs (default), FLASH_ATTN, default `max_num_batched_tokens` 2048, `max-num-seqs 2` | the recipe's NVIDIA builds add `--max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling --enable-prefix-caching` (NVFP4 variants); `--mm-encoder-tp-mode data` only matters for TP>1 | [recipe] `variants.nvfp4_nvidia.hardware_overrides` | **tested, not adopted.** Raising prefill chunks to 8192 did not help: fixed part 1.85 s vs 1.70 s, median 6.03 s vs 5.62–5.83 s |
| 8 | structured output | `response_format json_schema` (xgrammar) with the qwen3 reasoning parser | nothing model-specific; the recipe documents no guided-JSON caveat for this model | [recipe] | **matches.** With thinking off, `</think>` is already in the prompt, so the grammar applies from the first answer token. 24/24 schema-valid in every run |

## Bench — before / after (24 calls each; ref-hits out of the 5 reference-flag calls)

| variant | median | p90 | fixed | per-token | compl tok | valid | spoke | ref-hits (false) |
|---|---|---|---|---|---|---|---|---|
| **before**: string format (images silently first), MTP3 | 5.33 s | 8.83 s | 1.84 s | 14.6 ms | 239 | 24/24 | 10 | 0/5 (0) |
| before, greedy | 4.99 s | 7.89 s | 1.48 s | 14.7 ms | 229 | 24/24 | 9 | 1/5 (0) |
| fix #3, images first (= card order, exact tokens) | 5.36 / 5.07 s | 8.32 / 8.06 s | 1.95 / 1.54 s | 14.2 / 14.7 ms | 232 | 24/24 ×2 | 9 / 7 | 0/5, 1/5 (0) |
| **fix #3, text first (harness order) — adopted** | **5.83 / 5.62 s** | 8.89 / 9.16 s | **1.71 / 1.70 s** | 14.7 / 14.6 ms | 256 / 245 | **24/24 ×2** | **17 / 16** | **2/5, 1/5 (0)** |
| fix #3, text first, greedy | 5.61 s | 8.70 s | 1.68 s | 14.4 ms | 242 | 24/24 | 18 | 2/5 (0) |
| **+ MTP depth 5 — adopted** | **4.73 s** | **6.83 s** | 1.85 s | **11.6 ms** | 251 | **24/24** | 14 | 1/5 (0) |
| + `--max-num-batched-tokens 8192` (not adopted) | 6.03 s | 9.19 s | 1.85 s | 15.2 ms | 265 | 24/24 | 17 | 2/5 (0) |
| no MTP, text first (efficiency check) | 11.19 s | 19.90 s | 1.34 s | 37.6 ms | 257 | 24/24 | 15 | 2/5 (0) |
| no MTP, text first, greedy | 10.83 s | 19.76 s | 1.59 s | 37.2 ms | 242 | 24/24 | 18 | 2/5 (0) |

### Why the text-first fix is adopted even though the median rises by 0.3–0.5 s
- **The server is not slower.** The fixed part is unchanged or lower (1.70 s vs 1.84 s), and the per-token cost is
  the same (14.6 ms).
- **The extra time is extra speech.** Replies are 10–20 tokens longer because the model now speaks on 16–18 calls
  instead of 7–10. The reference spoke on 15.
- **Detection improves, and the improvement is concrete.** Over 5 text-first runs the model caught 9 of 25
  reference flags. Over 4 image-first runs it caught 2 of 20. There were no false flags either way.

The same two calls, run with the exact tokens and differing only in the order:

| call | images first | text first |
|---|---|---|
| #0 t16 | "Good catch. Now go to the tomato crate…" (flag none) | "Stop! You are chopping the CHEESE again. The recipe needs the tomato chopped and the cheese raw…" (flag `mis_prep`, same as the reference) |
| #5 t94 | "Good — keep walking to the serving pass…" (the cook is in fact still parked) | "You said you're walking there but you're still parked by the cheese crate. Use "go to the serving pass" right now…" (flag `forget_next_step`, same as the reference) |

With images first, the model trusts the conversation text that follows the frames. With the harness's order it
reads the frames the text points to.

## Changes

- **Serve script (applied)** `pass3_setup/serve_qwen38_27b.sh`:
  - add `--chat-template-content-format openai`;
  - change the speculative config to `{"method":"mtp","num_speculative_tokens":5}` (`MTP_K`, default 5).
- **PROFILES (orchestrator)**: no change. `qwen38-schema` already sends the card's non-thinking sampling, and it
  does **not** set `images_first`, which is the adopted order.
- **Benchmark flags**: unchanged (`--assist-profile qwen38-schema --assist-image-scale 1.0`).
- **Caution for anyone probing a live server**: `POST /tokenize` with images puts vLLM 0.19.1's multimodal cache out
  of sync. The next chat request carrying the same images fails with HTTP 500
  `AssertionError: Expected a cached item for mm_hash=…`. `template_diff_server.py` avoids this by perturbing one
  pixel. Never tokenize the benchmark's own frames against a production server.

## Pending
Nothing. The MTP-depth-5 run (`qwen38_textfirst_mtp5_audit`, one run, the first request sequence after a restart)
is the only single-run result. A repeat at leisure would firm up the detection column; the latency gain is well
outside run-to-run noise (per-token 11.6 ms vs 14.2–15.2 ms in all depth-3 runs).
