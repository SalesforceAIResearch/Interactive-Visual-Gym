# Official-guidance audit — Muse Glimmer 30B (`museglimmer`, port 18200) — 2026-09-19

**How it was measured.**
- **Method.** The method is the same as in `local_models/qwen/AUDIT.md`: the 24 genuine calls, one request in flight,
  tags `muse_*_audit`, and profile `museglimmer-nothink-schema`.
- **Columns.** "fixed" and "per-token" come from the least-squares split
  `latency = fixed + per_token × completion_tokens`. "ref-hits" counts same-type flags on the 5 calls where the
  reference raised a flag.
- **Setup.** One A100-80GB, running vLLM nightly `0.27.2rc1.dev193+gaa9903490.cu129`.

**Sources.**
- **[card]** `huggingface.co/meta-models/Muse-Glimmer-30B` (`docs_raw/hf_Muse-Glimmer-30B.md`).
- **[draft]** `meta-models/Muse-Glimmer-30B-assistant` card.
- **[recipe]** `github.com/vllm-project/recipes/blob/main/models/meta-models/Muse-Glimmer-30B.yaml`, updated
  2026-09-09. It was re-fetched to `docs_raw/vllm_recipe_2026-09-18.yaml`; the older copy is `vllm_recipe.yaml`.
- **[rh]** The `RedHatAI/Muse-Glimmer-30B-INT4` card.
- **[cfg]** The meta-models checkpoint's `processor_config.json` and `generation_config.json`, downloaded for this
  audit.

## Item table

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint & precision | `RedHatAI/Muse-Glimmer-30B-INT4`: W4A16 GPTQ g128, with vision, embeddings and lm_head in bf16; Marlin kernels on A100 | BF16 `meta-models/Muse-Glimmer-30B` is "the reference precision". The recipe's other variants are FP8-block (needs FP8, which A100 lacks) and NVFP4 W4A4 (Blackwell-only). Meta's own 4-bit path is the GGUF K-Quants for llama.cpp, with **0.2 % / 1.0 %** average degradation over 15 benchmarks. The RedHat INT4 card publishes **no** accuracy numbers | [recipe] `variants`; [card] "Fitting the Model on Your Device"; [rh] | **deviates on purpose, benched.** BF16 fits one A100-80GB (60.4 GiB weights, 378k-token KV pool), so the official precision was served and benched below. It is +0.9 s median, and decode drops 185 → 115 tok/s. Detection is not demonstrably better: 3/10 hits with 3 false flags, against INT4's 2/10 hits with 0 false. INT4 is kept |
| 2 | engine & version | nightly `0.27.2rc1.dev193` (commit aa99034, 2026-08-18, after the model's merge PR #51655) | recipe now: `min_vllm_version: "0.28.0"`, docker `vllm/vllm-openai:v0.28.0` ("support merged in #51655 and shipped in v0.28.0") | [recipe] `model:` | **deviates (pre-release), accepted.** Since aa99034 the only behaviour-relevant change to `muse_glimmer.py`, its reasoning parser or `dflash.py` is #54373 (the DFlash drafter now takes its RoPE layout from its own config). That fix matters only when drafter and target layouts disagree; measured acceptance here (4.9 of 16 on INT4, 4.8 on BF16) shows a working drafter. The rest is typing, LoRA and CI. Upgrading to 0.28 needs a cu129 build (driver 570) and is **not done** |
| 2b | flags | `--generation-config auto`, `--reasoning-parser muse_glimmer`, `--max-model-len 32768`, `--max-num-seqs 2`, `--limit-mm-per-prompt {"image":8}`, prefix caching; no tool parser | `--generation-config auto` (stop tokens 200001/200008); `--reasoning-parser muse_glimmer`; the tool parser is for ATEM tool calls; with DFlash "replace the earlier sequence limit with `--max-num-seqs 32`" (a warm-up token-budget rule) | [recipe] `base_args`, `features`, "Speculative decoding" | **matches.** No tools are used. `max-num-seqs 2` × 16 query slots stays under the 2,048 scheduled-token budget, so the 32-seq rule is moot |
| 3 | chat template | server uses the RedHatAI repo's older ("Onyx") `chat_template.jinja`, with reasoning strength passed as `chat_template_kwargs` | meta-models `chat_template.jinja` | `template_diff.py`, `qwen/template_diff_server.py` | **matches.** For our exact request (system, text + 8 images, prefill `to=user`, `continue_final_message`) the server's tokens are identical to the official processor + official template (7,721 = 7,721). Offline, the two templates also render identically. They would differ only for a hand-written "Reasoning strength" line, which we do not use |
| 4 | visual input | 8 images at native size → 28×74 patches → **518 tokens/frame** | `max_image_tokens 4096`, patch 14, merge 2: no down-scaling at our size. "The model is not explicitly optimized for video; video input is processed as individual frames." The video processor would cap frames at 144 tokens and fuse frame pairs (8 frames → 532 tokens total, at about ¼ the spatial detail) | [card] "Considerations and Limitations"; [cfg] | **matches (images)** |
| 5 | sampling | T 1.0, top_p 0.95, top_k 64 | "temperature = 1.0, top_p = 0.95, top_k = 64". "Do not run it greedy" | [card] Best Practices; [recipe] "Sampling" | **matches** |
| 6 | thinking | **skipped.** The assistant turn is pre-opened as `to=user` (`continue_final_message`), with `reasoning_strength: low` passed but unused | reasoning is always on; the only control is `Reasoning strength: low/medium/high/xhigh`; "Use high or xhigh for … agentic tasks". There is no documented off switch | [card]; [recipe] `features.reasoning` | **deviates on purpose (latency).** With reasoning at `low`, a call costs about 2× (measured 7.6 s vs 3.7 s on the setup host; RESULTS.md). The game clock charges latency. Structured output then needs `--structured-outputs-config {"enable_in_reasoning":true}`, because the reasoning parser never sees the channel end (11/24 valid without it, 24/24 with it) |
| 7 | efficiency | DFlash drafter `meta-models/Muse-Glimmer-30B-assistant`, `num_speculative_tokens: 15` | "`num_speculative_tokens: 15` is fixed, not tuned"; "paired with the specific target it was distilled against" | [recipe] "Speculative decoding"; [draft] | **matches.** The drafter was distilled against the BF16 target, yet acceptance with the INT4 target is no worse (4.89 vs 4.76 mean accepted length) |
| 8 | structured output | `json_schema` via xgrammar, grammar enforced from the first token (`enable_in_reasoning`) | "do not enable guided JSON" applies to *tool calls* (ATEM XML). There is no guidance for JSON content | [recipe] `features.tool_calling` | **matches** (our JSON is message content, not a tool call). 24/24 in every run |

## Bench (24 calls each)

| variant | median | p90 | fixed | per-token | compl | valid | spoke | ref-hits (false) |
|---|---|---|---|---|---|---|---|---|
| **INT4 + DFlash (current) — kept** | 4.13 / 3.54 s | 5.33 / 4.27 s | 2.62 / 1.86 s | 5.6 / 5.4 ms | 253 / 284 | 24/24 ×2 | 18 / 21 | 1/5, 1/5 (0, 0) |
| BF16 (official precision) + DFlash | 5.07 / 4.43 s | 8.92 / 5.70 s | 2.26 / 1.86 s | 12.4 / 8.7 ms | 248 / 253 | 24/24 ×2 | 16 / 19 | 0/5, 3/5 (2, 1) |

The first INT4 run reproduces the 2026-09-16 setup run exactly (4.14 s, the same replies). vLLM's fixed start-up seed
makes the first request sequence after a restart repeat.

**Sanity read.**
- **Call #5** (t94, cook idle at the crate). INT4: "You're still at the cheese crate with the plate in hand … Go
  to the serving pass now". BF16: "You're still by the cheese crate … Go to the serving pass". Same content; only
  BF16 attached the flag.
- **Call #0** (t16, cheese mis-prep). Both precisions mention the stray cheese; only BF16 run 2 flagged it.

## Changes
- **Serve script:** no change. The comment now states the audited choice.
- **PROFILES:** no change.

## Pending (optional)
- vLLM 0.28.0 (the recipe's pinned release) as a cu129 build, if one exists for driver 570.
