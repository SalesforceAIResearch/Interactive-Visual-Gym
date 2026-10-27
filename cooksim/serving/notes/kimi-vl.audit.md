# Kimi-VL-A3B-Thinking-2506 — official-guidance audit (2026-09-18)

Served name `kimi-vl-thinking`, port 18650, env `$ROOT/envs/kimi-vl-thinking` (vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0),
weights `moonshotai/Kimi-VL-A3B-Thinking-2506` @ `aa17309`, bf16.

**Outcome.** Three changes:
- **Server:** `--max-num-batched-tokens 32768`, the authors' own serve flag. Already in `serve_kimi_vl_thinking.sh`.
- **Request:** frames **before** the text, the authors' order. This needs a PROFILES change.
- **Request:** a **2,048-token thinking budget**, using vLLM's official `thinking_token_budget`. This needs a PROFILES change.
  - It is a deliberate deviation from the card's 32K output budget, which the game clock cannot afford.
  - It removes every no-answer run-away. The old setup lost 1–4 of 24 calls to 60-s run-aways that never reached the answer.

Sampling (T 0.8) already matched. Our reasoning-parser plugin stays: vLLM has no official Kimi-VL parser, and ours splits the output exactly as the authors' reference code does.

## Sources
- **[card]** https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506 (README in the local snapshot), "3.1 Inference with VLLM (recommended)":
  - `MAX_JOBS=4 pip install vllm==0.9.1 blobfile flash-attn --no-build-isolation` and "It is important to explicitly install flash-attn to avoid CUDA out-of-memory".
  - `LLM(model_path, trust_remote_code=True, max_num_seqs=8, max_model_len=131072, limit_mm_per_prompt={"image": 256})`.
  - `SamplingParams(max_tokens=32768, temperature=0.8)`.
  - `extract_thinking_and_summary(text, bot="◁think▷", eot="◁/think▷")`.
  - Messages are `[{"type":"image"}, …, {"type":"text"}]`: images first.
  - Intro: "supports 3.2 million total pixels in a single image".
- **[gh]** https://github.com/MoonshotAI/Kimi-VL README:
  - "For **Thinking models**, it is recommended to use `Temperature = 0.8`. For **Instruct models** … `Temperature = 0.2`."
  - "OpenAI-Compatible Server": `vllm serve moonshotai/Kimi-VL-A3B-Thinking-2506 --served-model-name kimi-vl-thinking-2506 --trust-remote-code --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-model-len 32768 --limit-mm-per-prompt image=64`.
- **[blog]** https://huggingface.co/blog/moonshotai/kimi-vl-a3b-thinking-2506 ("the recommended inference recipe", linked from the card):
  - "default sampling parameters for this thinking model (32K output token, temperature=0.8)".
  - OS-agent: "temperature to be set as 0.2 … max_tokens=8192".
  - Video: `resize_image_and_convert_to_pil(image, max_size=448)` per frame, a `HH:MM:SS` timestamp text part before each frame, and the prompt last.
- **[gencfg]** `generation_config.json`: `temperature: 0.6`. vLLM applies it as the server default, per the serve log: "overridden by the model's generation_config.json: {'temperature': 0.6}". Our requests send 0.8 explicitly.
- **[pp]** `preprocessor_config.json`: `in_token_limit 16384`, `patch_size 14`.
- **[vllm-r]** https://docs.vllm.ai/en/latest/features/reasoning_outputs/:
  - The supported-parser table has no Kimi-VL entry. `kimi_k2` is for Kimi-K2's own tokens.
  - "Token counting starts from `reasoning_start_str`. Once the reasoning token count reaches the configured `thinking_token_budget`, vLLM forces the model to produce `reasoning_end_str`."
- **[vllm-m]** https://docs.vllm.ai/en/latest/models/supported_models/: `KimiVLForConditionalGeneration` supports `moonshotai/Kimi-VL-A3B-Thinking`.

## Item table
| # | item | ours (before) | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint / dtype | `Kimi-VL-A3B-Thinking-2506` bf16 | same repo, `torch_dtype="auto"` (bf16). No official quantized build exists. | [card] | **matches** |
| 2a | engine / version | vLLM 0.19.1, native class, with flash-attn for the vision encoder (log: "Using AttentionBackendEnum.FLASH_ATTN for vit attention") | the card pins `vllm==0.9.1` plus flash-attn | [card], [vllm-m] | **deviates on purpose.** 0.9.1 predates reasoning-parser plugins, `thinking_token_budget` and structured output after reasoning, and 0.19.1 supports the model natively. The flash-attn concern (OOM from the SDPA vision path) is met: the ViT runs on FLASH_ATTN. |
| 2b | serve flags | `--trust-remote-code --max-model-len 32768 --max-num-seqs 2`, image limit 8, **no** batched-tokens flag (vLLM 0.19.1 on A100 defaults to 2048) | `--trust-remote-code --max-num-batched-tokens 32768 --max-model-len 32768 --limit-mm-per-prompt image=64` | [gh] | **fixed**: `--max-num-batched-tokens 32768` adopted. Uncached prefill went from 857 to **740 ms** median, faster on **22/22** calls. This was confirmed in two alternating restarts: 0.851 / 0.858 s against 0.754 / 0.743 s. (The same flag is *slower* on Molmo2, so this was checked per model.) The image limit of 8 is enough for our 8 frames. |
| 2c | reasoning parser | plugin `kimi_vl_reasoning_parser.py` (text-matched `◁think▷`/`◁/think▷`, since each marker is 5–6 tiktoken tokens) | none in vLLM. The authors split the output with `extract_thinking_and_summary` on exactly these two strings. | [vllm-r], [card] | **matches in effect.** The plugin applies the authors' split server-side, so `content` = summary and `reasoning_content` = thinking, and the reply schema applies after `◁/think▷`. No official alternative exists to switch to. |
| 3 | chat template | shipped template (it has a system branch) | same | [card] | **matches.** `render_textfirst.txt` / `render--images-first.txt`: the server `/tokenize` output equals the official `apply_chat_template` for the benchmark's call 0 (8,175 tokens, 8 media blocks), in both orders. |
| 4a | image resolution | native 1042×396 (about 518 tokens per frame) | single images up to 3.2 MP natively. For *video frames* the blog shrinks each frame to max side 448. | [card], [blog], [pp] | **matches** the image guidance. The 448-px video recipe was tested as `--assist-image-scale 0.43`: 5.1k prompt tokens, paired −1.9 s against before but not faster than images-first at full size, 22/24 valid, and 0 correct flags. **Not adopted.** |
| 4b | frame order / timestamps | text first, then frames, no per-frame timestamps | images first, then the prompt (card and blog). The blog also puts a timestamp text before each video frame. | [card], [blog] | **fixed**: images first adopted (see bench). Per-frame timestamps would need a harness change (the user text already says "one per tick, oldest first"), so they were left **pending**. |
| 5 | sampling | T 0.8 (the server default would be 0.6 from [gencfg]) | T 0.8 for Thinking models | [gh], [card], [blog] | **matches** |
| 6a | thinking budget | `max_tokens 4000` and no thinking budget: 1–4 of 24 calls hit the cap mid-thinking (about 60 s, no answer) | `max_tokens 32768` | [card], [blog] | **fixed, deviating on purpose from 32K:** `thinking_token_budget 2048` (vLLM's official mechanism) with `max_tokens 4000`. The result is **24/24**, with no length cut in 24 calls, max latency 30.8 s instead of 60.4 s, and paired −0.74 s. Every thinking run reached an answer. |
| 6b | thinking off (`kimi-vl-nothink-schema` ablation) | empty `◁think▷◁/think▷` prefill, T 0.8 | there is no official off switch. Moonshot's non-thinking model is Kimi-VL-A3B-Instruct, at T 0.2. | [gh] | **deviates** (ablation only, not official; not re-measured in this audit) |
| 7 | efficiency | prefix caching, CUDA graphs (piecewise; MLA via TRITON_MLA) | the card's `max_num_seqs 8` is a throughput setting. No drafter is published. | [card] | **matches** (we run one request at a time) |
| 8 | structured output | `json_schema`, applied after `◁/think▷` by the plugin | not discussed by the authors | – | **matches.** 24/24 with the budget; the no-answer failures were all length cuts, not schema errors. |

## Bench (24 real calls, T 0.8, full frames; interleaved A/B on the unchanged server)
"Paired Δ" is the median over calls of (variant − before).

| run (tag, `_audit`) | median | p90 | max | paired Δ | compl tok | schema ok | length-cut | spoke | flags | correct flag (of 5) |
|---|---|---|---|---|---|---|---|---|---|---|
| **before**: `kimivl_ab_before` (text first, no budget) | 11.01 s | 58.01 s | 60.4 s | 0 | 796 | **20/24** | 4 | 18 | 9 | 2 |
| images first `kimivl_ab_imgfirst` | 9.68 s | 22.29 s | 60.4 s | −2.89 s | 686 | 23/24 | 1 | 21 | 10 | 1 |
| images first, 448-px frames `kimivl_ab_imgfirst_s448` | 9.45 s | 26.07 s | 58.3 s | −1.94 s | 688 | 22/24 | 2 | 22 | 7 | 0 |
| **after**: images first + budget 2048 `kimivl_ab_imgfirst_budget2k` | 10.21 s | 29.35 s | **30.8 s** | −0.74 s | 671 | **24/24** | **0** | 22 | 12 | 0 |
| prefill only (max_tokens 1), calls 2–23: current → + mnbt 32768 | 0.857 → **0.740 s** | 0.940 → 0.809 s | – | −0.11 s (22/22 faster) | – | – | – | – | – | – |
| prefill only, alternating restarts 2048 / 32768 / 2048 / 32768 | **0.851 / 0.858 vs 0.754 / 0.743 s** | 0.93 / 0.96 vs 0.81 / 0.80 s | – | – | – | – | – | – | – | – |
| **final confirmation** (adopted serve script, interleaved): old profile `kimivl_final_before` | 11.32 s | 44.74 s | 60.1 s | 0 | 795 | 23/24 | 1 | 22 | 11 | 1 |
| **final confirmation**: adopted profile `kimivl_final_after` | 13.07 s | **29.47 s** | **34.8 s** | **−0.19 s** | 930 | **24/24** | **0** | 23 | 11 | 0 |

**Sequential phase-1 runs, for completeness.**

| tag (`_audit`) | server | median | p90 | schema ok |
|---|---|---|---|---|
| `kimivl_before` | current | 12.63 s | 26.1 s | 23/24 |
| `kimivl_imgfirst` | current | 12.31 s | 34.5 s | 24/24 |
| `kimivl_imgfirst_s448` | current | 9.10 s | 18.6 s | 23/24 |
| `kimivl_before_mnbt` | + mnbt 32768 | 10.07 s | 25.3 s | 24/24 |
| `kimivl_imgfirst_mnbt` | + mnbt 32768 | 10.85 s | 38.4 s | 23/24 |
| `kimivl_imgfirst_mt8k` (`max_tokens 8192`) | + mnbt 32768 | 10.37 s | 18.4 s | 24/24 |

Pooled over all runs:
- **text first, no budget:** 90/96 valid, counting the setup run plus three audit runs.
- **images first, no budget:** 94/96 valid, over four runs.
- **images first, budget 2048:** 24/24.

Raising `max_tokens` to 8192 only lengthens the tail. The budget bounds it.

**Final confirmation** on the adopted serve script: the adopted profile went 24/24 with 0 cut-offs, a p90 of 29.5 s against 44.7 s, a max of 34.8 s against 60.1 s, and a paired Δ of −0.19 s. The old profile lost call 18 to a 60-s run-away again. The median is higher (13.1 s against 11.3 s) because this sample thought longer (930 against 795 tokens). At T 0.8 that is sampling variation; the paired difference is the controlled number.

**How to read detection.** At the official T 0.8, Kimi's flags are very noisy on 24 calls: the correct-flag count moves between 0 and 2 across *identical* configs, and it raises 7–14 flags against the reference's 5. So no variant can be shown to detect better or worse. The changes are adopted on validity and latency, which is the rule's bar.

Sanity read, images first with budget 2048:
- [18] master_hard_4 t408, ref "Stop! You just picked up an onion…". It says: "There's a mistake: raw onion was added to your plate instead of cheese. Go to the trash bin in the bottom right corner to dump the plate's contents…". It sees the onion but raises no flag. The old setup ran out of tokens on this call (no reply) in 2 of 4 runs.
- [5] t94: "The Caprese plate is correctly assembled… Go to the serving pass and serve it now." The advice is correct but it raises no flag.
- [13] t254: "The fish is done and smoking in the top fryer…" (`leave_unattended`). This is an invented hazard, which every variant and the setup run also produce.
- [0] t16: "Great! You've realized you grabbed cheese instead of tomato. Trashing the cheese now is the right move." The error is seen, but no flag is raised in this run.

## Changes to apply
**Serve script** (already applied): `pass3_setup/serve_kimi_vl_thinking.sh` gains `--max-num-batched-tokens ${MNBT:-32768}`. The plugin and `--reasoning-config` are unchanged.

**PROFILES** (engine, for the orchestrator):
```python
# Kimi-VL-A3B-Thinking-2506: authors' order (images before the prompt) and T 0.8; thinking capped by vLLM's
# thinking_token_budget (the card's 32K budget cannot fit the game clock; without a cap 1-4/24 calls ran out of
# max_tokens mid-thought after ~60 s). Audit 2026-09-18: 24/24, 0 cut-offs, max 30.8 s, paired -0.74 s.
"kimi-vl-think-schema": dict(temperature=0.8, extra_body={"thinking_token_budget": 2048}, prefill=None,
                             reply_schema=True, images_first=True),
"kimi-vl-think256-schema": dict(temperature=0.8, extra_body={"thinking_token_budget": 256}, prefill=None,
                                reply_schema=True, images_first=True),
"kimi-vl-nothink-schema": dict(temperature=0.8,
                               extra_body={"add_generation_prompt": False, "continue_final_message": True},
                               prefill={"role": "assistant", "content": "◁think▷◁/think▷"},
                               reply_schema=True, images_first=True),
```
The benchmark flags are unchanged: `--assist-profile kimi-vl-think-schema --assist-max-tokens 4000 --assist-image-scale 1.0`.

For think256 and nothink, only the image order changes: it is the authors' order, and those two ablations were not re-benched. If the orchestrator wants those two ablations exactly as benched in setup, leave them as they are.

## Pending
- **Per-frame timestamp text** (the blog's video recipe) needs a harness change, so it is not tested.
- **Thinking-off ablation at the Instruct temperature** (0.2) is not tested.
