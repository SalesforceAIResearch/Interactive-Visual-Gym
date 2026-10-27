# Official-guidance audit — Qwen3-Omni-30B-A3B-Instruct (`qwen3-omni-30b-a3b`, port 18800) — 2026-09-19

**How it was measured.**
- **Method.** The method is the same as in `local_models/qwen/AUDIT.md`: the 24 genuine calls, one request in flight,
  and tags `qwen3omni_*_audit`.
- **Columns.** "fixed" and "per-token" come from the least-squares split
  `latency = fixed + per_token × completion_tokens`. "ref-hits" counts same-type flags on the 5 calls where the
  reference raised a flag.
- **Setup.** vLLM 0.19.1 serves the thinker only, on one A100-80GB.

**Sources.**
- **[card]** Local `README.md` of `Qwen/Qwen3-Omni-30B-A3B-Instruct` at snapshot `26291f7`.
- **[gh]** `github.com/QwenLM/Qwen3-Omni` README, fetched 2026-09-18 to `docs_raw/github_README.md`.
- **[cfg]** The checkpoint's `preprocessor_config.json`, `generation_config.json` and `chat_template.json`.
- **[recipe]** None. `vllm-project/recipes` has no Qwen3-Omni entry (`models/Qwen/Qwen3-Omni-30B-A3B-Instruct.yaml`
  returns 404, and the Qwen provider list has none).

## Item table

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint & precision | `Qwen/Qwen3-Omni-30B-A3B-Instruct` rev 26291f7, bf16, thinker only (talker shards 14–15 not downloaded) | Instruct / Thinking / Captioner only. No official quantized build. The single-GPU serve command uses `--dtype bfloat16` | [card] "vLLM serve" block | **matches** |
| 2 | engine & version | plain vLLM 0.19.1, `Qwen3OmniMoeThinkerForConditionalGeneration` | "We highly recommend using the latest vLLM-Omni … We also recommend using vLLM". "vLLM serve for Qwen3-Omni currently only supports the thinker model" | [gh] §vLLM Usage/Installation; [card] | **deviates on purpose.** vLLM-Omni exists for audio output (talker + code2wav). The benchmark consumes text only, and the thinker is the same model in both stacks |
| 2b | flags | `--dtype bfloat16 --max-model-len 32768 -tp 1`, `--limit-mm-per-prompt {"image":8,"video":0,"audio":0}`, prefix caching, `max-num-seqs 2` | `vllm serve … --dtype bfloat16 --max-model-len 32768 --allowed-local-media-path / -tp 1`. The offline example uses `max_num_seqs=8`, `limit_mm_per_prompt` 3 each | [card] | **matches.** `--allowed-local-media-path` is only needed for file:// media; we send data URLs |
| 3 | chat template | vLLM detected content format `openai` and renders the content parts in the order sent | HF processor + `chat_template.json` | `qwen/template_diff_server.py` | **matches.** Server tokens are identical to the official processor's (6,749 = 6,749), including the system block and `<|im_start|>assistant\n` |
| 3b | content order | text first, then the 8 frames (the harness says "Below are your last 8 frames") | evaluation setting: "the text should come **after** multimodal data in the sequence" | [card] "Setting for Evaluation → Input Sequence" | **deviates on purpose, benched** (below). Card order missed the textbook `wrong_order` at call #2 in both runs; harness order caught it in both runs, and in the setup run too. Latency was no better |
| 3c | system prompt | our supervisor system prompt | "No `system prompt` should be set for any evaluation benchmark". For interactive use the card *recommends* a system prompt | [card] | **deviates on purpose.** The eval rule is for academic benchmarks. The harness needs its instructions |
| 4 | visual input | 8 images, 1042×396 → 24×66 patches → **396 tokens/frame** | processor `min_pixels 3136`, `max_pixels 12,845,056`, patch 16, merge 2: no down-scaling at our size. Video eval uses `fps=2` | [cfg]; [card] eval settings | **matches (images).** As video, the 8 frames would be fused in temporal pairs and sampled well below the 2 fps the eval uses; kept as images |
| 5 | sampling | T 0.6, top_p 0.95, top_k 20 (`qwen3-omni-schema`) | vLLM example: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384`. Evaluation: "`Instruct` models use greedy decoding". `generation_config.json` carries talker settings only | [card] vLLM usage; eval settings; [cfg] | **matches (vLLM example).** Greedy (the eval setting) was benched: 2.59 s, 24/24, 1/5 hits, 2 false. No better, so T 0.6 is kept |
| 6 | thinking | none (Instruct) | the Instruct model has no thinking mode; Thinking is a separate repo | [card] | **matches** |
| 7 | efficiency | bf16 MoE (3B active), prefix caching, CUDA graphs, FLASH_ATTN, Triton fused-MoE | the card names no speed features for vLLM serve: no MTP or drafter for Omni | [card] | **matches.** One non-model item: vLLM warns "Using default MoE config … E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB" (no tuned kernel config ships for A100). Tuning it with vLLM's `benchmark_moe.py --tune` is optional vLLM-side work, not model guidance — **not done** |
| 8 | structured output | `json_schema` (xgrammar), no reasoning parser | no Omni-specific guidance | — | **matches.** 24/24 valid in every run |

## Bench (24 calls each)

| variant | median | p90 | fixed | per-token | compl | valid | spoke | ref-hits (false) |
|---|---|---|---|---|---|---|---|---|
| setup run 2026-09-18 (text first, T 0.6) | 3.11 s | 5.55 s | — | — | 259 | 24/24 | 24 | 1/5 (2) |
| **current: text first, T 0.6 — kept** | 2.24 s | 3.91 s | 0.08 s | 8.8 ms | 251 | 24/24 | 23 | 2/5 (1) |
| card order (images first), T 0.6 | 2.89 / 2.85 s | 5.35 / 5.21 s | 0.42 / 0.36 s | 9.3 / 9.3 ms | 255 / 263 | 24/24 ×2 | 23 / 24 | 1/5, 1/5 (1, 0) |
| greedy (card eval decoding), text first | 2.59 s | 4.60 s | 0.37 s | 8.9 ms | 243 | 24/24 | 24 | 1/5 (2) |

A first text-first run in this session (`qwen3omni_base_audit`: 5.78 s median) overlapped the `/tokenize` rendering
check on the same server and is discarded as contaminated. Its replies still match the pattern: 1/5 hits, and #2
was caught.

**Sanity read, call #2** (t32; the reference says "cheese went on before the chopped tomato").
- **Text first:** "The order is strict: tomato has to go on first. You've got the cheese on the plate now…",
  with flag `wrong_order`.
- **Images first:** "Good catch on the cheese. Now that the tomato is chopped, scoop it onto the plate first…", with
  no flag.

With images first, the model again follows the conversation text instead of the frames. This matches the
Qwen3.8-27B finding.

## Changes
- **Serve script:** no change.
- **PROFILES:** no change (`qwen3-omni-schema` stays).
- **Benchmark flags:** unchanged.

## Pending (optional)
- A tuned fused-MoE config for A100: vLLM's `benchmarks/kernels/benchmark_moe.py --tune` for E=128, N=768, bf16,
  batch sizes 1–16. It is not model guidance. The expected gain is only on the ~9 ms/token decode.
