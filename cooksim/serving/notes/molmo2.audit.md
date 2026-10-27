# Molmo2-8B — official-guidance audit (2026-09-18)

Served name `molmo2-8b`, port 18400, env `$ROOT/envs/molmo2` (vllm 0.19.1, torch 2.10.0, transformers 5.17.0),
weights `allenai/Molmo2-8B` @ `e28fa28`, bf16. (Molmo 2-O 7B shares the architecture, template and processor; see `../molmo2-o/AUDIT.md`.)

**Outcome.** One fix, taken from AllenAI's own vLLM inference code:
- **Sampling:** greedy decoding (temperature 0). This needs a PROFILES change.

The demo's other setting, `--max-num-batched-tokens 36864`, was tested and rejected. On this A100 it slows prefill by about 0.28 s: 1.72 s against 2.00 s median in two alternating rounds. The serve script is therefore unchanged.

The system-message fold is correct: Molmo2 has no system role, and the fold reproduces the official template exactly. Half-size frames are a deliberate deviation from the official 8-crop tiling, and full size was re-measured for this audit.

## Sources
- **[card]** https://huggingface.co/allenai/Molmo2-8B (README in the local snapshot):
  - Every example is a single `{"role": "user"}` message: text part first, then image or video parts.
  - Examples use `dtype="auto"` and `max_new_tokens=2048` (448 for multi-image QA).
  - No sampling parameters are given (plain `model.generate`, which is greedy because `generation_config.json` has no `do_sample`).
- **[repo]** https://github.com/allenai/molmo2 README, "Install vLLM (>= 0.15.0)": "Molmo2 is officially supported in vLLM starting from v0.15.0. Please install vLLM 0.15.0 or later." The "Convert Checkpoint" section says "`--override_max_model_len N`, N: 36864 for Molmo2-4B and Molmo2-8B, 65536 for Molmo2-O-7B".
- **[demo]** `scripts/unified_demo.py` in the same repo (AllenAI's vLLM inference script):
  - `SamplingParams(max_tokens=max_steps, temperature=0)`;
  - `LLM(..., dtype="bfloat16", limit_mm_per_prompt={"image": 6, "video": 1}, max_num_batched_tokens=36864)`;
  - messages are `[{"role": "user", "content": [text, *images]}]`, with no system message.
- **[sft]** `launch_scripts/sft.py` L270–276:
  - `data_formatter.message_format = "qwen3"`;
  - `data_formatter.system_prompt = "demo_or_style_v2"`;
  - `mm_preprocessor.image.max_multi_image_crops = 8`;
  - `mm_preprocessor.image.max_images = 5`.
- **[fmt]** `olmo/preprocessing/data_formatter.py` L1537–1634 and L1861–1878. The "system prompt" is a *style tag* (for example `"pointing:"`, or empty for chat/demo styles) that is prepended to the first user message: `with_system_prompt = prefix + " " + messages[0]`. There is no system role anywhere in training.
- **[tmpl]** the shipped `chat_template.jinja`:
  - it raises "Conversation roles must alternate user/assistant/…" when the first role is `system`;
  - it renders `Image i<|image|>` for every image before `<|im_start|>user`.
- **[pp]** `preprocessor_config.json`: `max_crops 8`, `overlap_margins [4,4]`, 378-px crops, `pooling 2×2`.
- **[vllm]** https://docs.vllm.ai/en/latest/models/supported_models/: the `Molmo2ForConditionalGeneration` row lists `allenai/Molmo2-8B` and `allenai/Molmo2-O-7B`. vLLM marks `molmo2` as a multimodal-prefix LM (`config/model.py` `MM_PREFIX_LM_MODELS`), gives image tokens bidirectional attention, and forces `--disable_chunked_mm_input` (log line: "Forcing --disable_chunked_mm_input for models with multimodal-bidirectional attention").

## Is there an official way to give Molmo2 instructions?
No. Molmo2 was never trained with a system role:
- The shipped template rejects one ([tmpl]).
- AllenAI's own vLLM demo sends a single user turn ([demo]).
- The training formatter's only "system prompt" is a style tag glued onto the front of the first user message ([fmt]).

So the only official slot for task instructions is the first user turn, and that is where our `molmo2_chat_template_system.jinja` puts the system text: `<system text>\n\n<user text>`.

**Verified.** `render_diff.py --fold-system` sent the benchmark's call 0 through the running server (`/tokenize`). It compared the result to the *official* template applied to the same conversation written as one user message, with the system text prepended. The prompt text is identical, including the `Image 1<|image|> … Image 8<|image|>` prefix before `<|im_start|>user` (`render_fold.txt`). The two remaining differences are both official processor behaviour:
- the server prepends the BOS token `<|im_end|>`, which `processing_molmo2.py` `insert_bos` also does;
- the server expands each image into its low-res and high-res token blocks.

The one free choice is the separator. We use `\n\n`, where the training formatter puts a single space between a style tag and the question. That rule is for one-word style tags, not paragraphs, so `\n\n` is kept. **Verdict: matches.**

## Item table
| # | item | ours (before) | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint / dtype | `allenai/Molmo2-8B` bf16 | same repo. The demo uses `dtype="bfloat16"`; the card uses `dtype="auto"`. No official quantized build exists. | [card], [demo] | **matches** |
| 2a | engine / version | vLLM 0.19.1 (native Molmo2 class) | vLLM ≥ 0.15.0 | [repo], [vllm] | **matches** |
| 2b | flags | `--trust-remote-code`, `--max-model-len 32768`, `limit-mm {"image":8,"video":0}` | the demo uses `trust_remote_code=True`, `limit_mm_per_prompt {"image":6,"video":1}`; the HF conversion uses max_model_len 36864 | [demo], [repo] | **matches in effect**. Our longest prompt is 13.7k tokens, so 32768 is enough. Image limit 8 is needed for the 8 frames. |
| 2c | multimodal-prefix attention | automatic (bidirectional image attention, chunked mm input disabled) | the model's design, handled natively by vLLM | [vllm] | **matches** |
| 3 | chat template / system message | custom fold template | no system role exists; instructions go in the user turn | [tmpl], [fmt], [demo] | **matches** (see the section above, byte-verified) |
| 4a | image tiling | frames pre-shrunk to 521×198 (`--assist-image-scale 0.5`), which gives 2 crops plus a global view per frame, 8.5k prompt tokens | full-resolution tiling with up to 8 crops per image, including in multi-image SFT: 1042×396 gives a 2×4 tiling plus global, 13.7k prompt tokens | [pp], [sft] | **deviates on purpose.** In a paired A/B, official full-size frames cost **+3.6 s** per call (7.1 s against 3.5 s median). Validity was the same at 24/24, and full size caught fewer reference errors (1 against 2 correct flags; it missed call 0). vLLM 0.19.1 cannot cap crops per request: `mm_processor_kwargs {"max_crops":2}` fails with an `assert sum(num_crops) == len(pixel_values)` 500 error, so pre-shrinking is the only way to reduce it. |
| 4b | number of images | 8 per call | SFT saw ≤ 5 images per example; AllenAI's evals allow up to 20 (`eval_molmo2.py` sets `max_images = 20`) | [sft] | **deviates** (the harness fixes 8 frames; within the eval range) |
| 4c | image position | text part first. The template moves all images before `<|im_start|>user` regardless of order, so our order has no effect. | text first, then images | [card], [demo] | **matches** |
| 5 | sampling | T 0.2 (profile `molmo2-schema`) | **greedy, temperature 0** | [demo] | **fixed**, adopted (see bench) |
| 6 | thinking | none | not a reasoning model | [card] | **matches** (not applicable) |
| 7a | scheduler budget | vLLM default. On A100, vLLM 0.19.1 uses **2048** for the API server (`engine/arg_utils.py`: "Setting large `max_num_batched_tokens` for A100 reduces throughput"). | `max_num_batched_tokens=36864` | [demo] | **deviates on purpose.** 36864 is slower here. Uncached prefill median was 1.72 / 1.73 s at 2048 against 2.01 / 2.00 s at 36864, over two alternating rounds of server restarts (p90 1.78 against 2.08 s). A single phase-2 probe had suggested the opposite (2.21 → 2.01 s), but that start ran under heavy host load. |
| 7b | other speed features | prefix caching, CUDA graphs (default), `max-num-seqs 2` | no drafter or MTP is published for Molmo2 | [card], [repo] | **matches** |
| 8 | structured output | `json_schema` (xgrammar) | not discussed officially. It works: 24/24. | – | **matches** (no known interaction; no reasoning parser involved) |

## Bench (24 real calls, `max_tokens 4000`)
Request-level variants use **interleaved A/B** (`ab_bench.py`): every call goes to every variant, in rotating order, on the same server. "Paired Δ" is the median over calls of (variant − before). Sequential runs of an identical config drifted by up to 1.8 s median on this shared host (`molmo2_t0_audit` 3.45 s against `molmo2_t0_mnbt_audit` 5.30 s), so only paired numbers are used to decide.

| run (tag, `_audit`) | server | frames | median | p90 | paired Δ | compl tok | schema ok | spoke | flags | correct flag (of 5) | length-cut |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **before**: `molmo2_ab_before` (T 0.2) | current | 0.5 | 4.01 s | 7.15 s | 0 | 261 | 24/24 | 22 | 2 | 1 | 0 |
| greedy `molmo2_ab_t0` | current | 0.5 | 3.50 s | 6.76 s | **−0.05 s** | 260 | 24/24 | 21 | 4 | 2 | 0 |
| greedy, official full tiling `molmo2_ab_t0_s1.0` | current | 1.0 | 7.11 s | 10.94 s | +3.62 s | 254 | 24/24 | 21 | 3 | 1 | 0 |
| before `molmo2_ab2_before` (T 0.2) | + mnbt 36864 | 0.5 | 3.31 s | 7.33 s | 0 | 246 | **23/24** | 20 | 2 | 1 | **1** |
| greedy `molmo2_ab2_t0` | + mnbt 36864 | 0.5 | 3.47 s | 6.79 s | −0.04 s | 260 | 24/24 | 21 | 4 | 2 | 0 |
| prefill only (max_tokens 1), calls 2–23, phase 2 (one start each, heavy host load) | current → + mnbt 36864 | 0.5 | 2.21 → 2.01 s | 4.12 → 2.15 s | – | – | – | – | – | – | – |
| prefill only, phase 3: alternating restarts 2048 / 36864 / 2048 / 36864 | 2048 vs 36864 | 0.5 | **1.72 / 1.73 vs 2.01 / 2.00 s** | 1.78 / 1.77 vs 2.09 / 2.08 s | – | – | – | – | – | – | – |

**After** = greedy on the unchanged server (`molmo2_ab_t0`): 3.50 s median, 6.76 s p90, paired −0.05 s, 24/24.

Sequential runs (for completeness): `molmo2_before_audit` 5.22 s, `molmo2_t0_audit` 3.45 s, `molmo2_t0_s1.0_audit` 6.97 s, `molmo2_t0_mnbt_audit` 5.30 s, `molmo2_before_mnbt_audit` 3.45 s. All were 24/24.

- **Greedy (temperature 0).** Greedy is AllenAI's own setting. It is not slower (paired −0.05 s in both A/B pairs) and was valid **24/24 in all 4 greedy runs**. T 0.2 ran away once: call 9 in `molmo2_ab2_before` repeated `subgoals_completed` up to the 4,000-token cap and returned no valid reply. Greedy also gives a deterministic assistant, which removes sampling noise from the pass@3 comparison.
- **`--max-num-batched-tokens 36864`: rejected.** This is AllenAI's demo value, and it lets the 8.5k-token prompt prefill in one step instead of five. Even so, it is about 0.28 s slower on every alternating restart. Molmo2-O shows the same, +0.39 s. A plausible cause is that vLLM's compiled graph covers batches of up to 2,048 tokens (`compile_ranges_endpoints [2048]`), so one big multimodal prefill step runs outside it; this was not verified. The single phase-2 probe that suggested a gain ran under heavy host load.
- **Sanity read** (greedy, half frames; reference = round-4 API reply):
  - [5] easy_hard_2 t94: "You're still standing by the cheese crate with the correct plate, not at the serving pass yet. Go to the serving pass, then serve." with `forget_next_step`. **Correct** in every run and variant.
  - [0] easy_hard_1 t16: `mis_prep` with "Wait, you grabbed cheese instead of a tomato!…". This text is a verbatim copy of the assistant's own t15 line in the conversation history, so it is echo, not fresh detection. T 0.2 echoes it in some runs only.
  - [2] easy_medium_1 t32, ref `wrong_order`: "You're back at the plate stack… Don't add it to the plate yet". A miss.
  - [13] hard_hard_1 t254, ref: go to the serving pass. It says "the potatoes are done and smoking in the middle fryer", an invented hazard.
  - [18] master_hard_4 t408, ref: onion picked up. It narrates the burrito order. A miss.
  - At full size the model still misses 2, 13 and 18, and loses the call-0 echo.

## Changes to apply
**Serve script:** no flag change. `pass3_setup/serve_molmo2.sh` only gains a comment recording why the demo's `max_num_batched_tokens` is not used.

**PROFILES** (engine, for the orchestrator). Change the entry both Molmo2 models use:
```python
# Molmo2-8B / Molmo2-O-7B: AllenAI's vLLM demo decodes greedily (scripts/unified_demo.py: temperature=0).
# Audit 2026-09-18: 24/24 in 4/4 runs, paired -0.05 s vs T 0.2 (which ran away to the cap once in 96 calls).
"molmo2-schema": dict(temperature=0.0, extra_body={}, prefill=None, reply_schema=True),
```
The benchmark flags are unchanged: `--assist-profile molmo2-schema --assist-max-tokens 4000 --assist-image-scale 0.5`.

## Pending / notes
- Official full-resolution tiling (`--assist-image-scale 1.0`) is available as an ablation at +3.6 s per call. The paired read shows no detection gain from it.
- vLLM 0.19.1 rejects a per-request `mm_processor_kwargs.max_crops`. A server-wide `--mm-processor-kwargs '{"max_crops":2}'` was not tried, because it is not an official configuration either.
- Never call `/tokenize` with images on a live server (vLLM 0.19.1 multimodal-cache bug; see `../glm46v-flash/AUDIT.md`).
