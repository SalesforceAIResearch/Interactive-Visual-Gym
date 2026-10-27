# InternVideo3-8B-Instruct (`internvideo3-8b`, port 18700): audit against the official guidance (2026-09-18/19)

**Verdict.** The shim follows the model card on checkpoint, dtype, chat template, resolution and the generation_config sampling. There are three findings.

1. **The runaways are not cured by the official settings.**
   - The official settings are the generation_config sampling (T 0.7 / top_p 0.8 / top_k 20) plus the card's `max_new_tokens=1024`.
   - On the 6 calls that had run away before, 4 replays each: **6/24 still run to the cap** (`iv3_stress_gencfg`).
   - Adding `repetition_penalty=1.05`, which no InternVideo3 source prescribes, gives **0/24** (`iv3_stress_rep105`) and 24/24 on the full bench, at a lower median (7.9 s against 11.2 s).
   - **Adopted as a deliberate deviation.**
2. **Video before the question.** Every official example places the video *before* the question; our profile sends the text first.
   - The official order was measured twice: 22/24 without the penalty, and 23/24 with it plus FA2. Both runs lose a call to a runaway on call 9. The text-first runs with the penalty were 24/24 twice.
   - **Not adopted.** It is a deliberate deviation, measured, although n is small.
3. **Attention backend.** The card and README load the model with **sdpa**; the authors' own eval scripts use **flash_attention_2**.
   - flash-attn was installed into `envs/internvideo3-8b` only.
   - Greedy A/B on 8 calls: same text on 6/8 (bf16 kernel numerics differ on 2), decode about 3% faster (43–45 against 44–47 ms per token).
   - Full bench with the penalty: 24/24, median 7.10 s against 7.91 s with sdpa.
   - **Adopted (`ATTN=flash_attention_2`).**

Sources:
- **[card]** HF card `yanziang/InternVideo3-8B-Instruct` (local `hf_cache/internvideo3-8b/.../README.md`) and the shipped `inference_demo.py`.
- **[gh]** github.com/OpenGVLab/InternVideo `InternVideo3/README.md` (Model Zoo links exactly this repo) and `InternVideo3/InternVideo3_eval/scripts/eval_*.sh`.
- **[cfg]** `generation_config.json`, `video_preprocessor_config.json`, `processing_internvideo3.py`, `video_processing_internvideo3.py`.
- **Missing pieces.**
  - The eval scripts' model class lives in an `lmms-eval` gitlink, and `.gitmodules` has no entry for it, so its sampling defaults and its `enable_thinking` handling could not be read.
  - There is no vLLM recipe or vLLM architecture for InternVideo3.

## Item table

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint | `yanziang/InternVideo3-8B-Instruct` rev c460291 | the only released checkpoint; [gh] Model Zoo points to it (`OpenGVLab/InternVideo3-*` does not exist) | [gh] README "Model Zoo" | matches |
| 1b | dtype | bf16 | `dtype=torch.bfloat16`; "Precision: BFloat16" | [card] Quickstart, Architecture table | matches (no quantized build exists) |
| 2 | engine | transformers 4.57.3 (trust_remote_code) behind our FastAPI shim | `pip install transformers>=4.57.3`; eval scripts pin `transformers==4.57.3` | [card]/[gh] Requirements; `eval_*.sh` | matches. vLLM has no architecture for it. |
| 3 | chat template | `processor.apply_chat_template` with the shipped `chat_template.jinja` (Qwen style, system branch) | same processor and template | [card] | matches. Render of call 0: `<|im_start|>system…<|im_end|>`, user text, then `<0.6 seconds><|vision_start|>[396 video tokens]<|vision_end|>` ×4, `<|im_start|>assistant\n`; 5,180 tokens. |
| 3b | order of video and text | **text first, then the clip.** The `internvideo3-schema` profile has no `images_first`, and the shim puts the clip where the first image was. | every example has `{"type":"video",…}` **then** `{"type":"text",…}` | [card] "Video Understanding"; [gh] README; `inference_demo.py` | **deviates on purpose (measured).** Video-first gave 22/24 (no penalty) and 23/24 (penalty + FA2), with runaways on call 9 in both. Text-first with the penalty gave 24/24 in both runs. Detection is 0 either way. |
| 4 | frames / fps / timestamps | the 8 tick frames as one clip; `fps = 1/1.2`, so timestamps `<0.6 s>…<7.8 s>`, one per temporal pair (Qwen3-VL temporal patch 2) | the video is sampled at `fps=4` (README, eval scripts) or `fps=1` (card/demo), with timestamps per temporal group; trained at "2fps/512 frames, then 4fps/2048 frames" | [gh] README; [card] Training Pipeline; `processing_internvideo3.py` `_calculate_timestamps` | **deviates by necessity.** The harness supplies one frame per 1.2-s tick, so no denser sampling is possible. Timestamps are truthful. Each temporal patch fuses two frames 1.2 s apart; at 4 fps it would fuse two frames 0.25 s apart. The shim has a `FRAMES=video_dup` option that duplicates each tick frame into its own patch (8 groups, 6.8k tokens); see Pending. |
| 4b | resolution | native 1042×396, 396 tokens per temporal group (no resize: 8×412k px is inside the processor's total budget 4,096…25.2M px) | README: `min_pixels = 128·2·32·32`, `max_pixels = 256·2·32·32` per frame (262k–524k px, i.e. 256–512 tokens per 2-frame group). Eval scripts: 256·32·32 per frame. Processor default: total budget up to 25.2M px. | [gh] README "Video Understanding"; `eval_videomme.sh`; `video_preprocessor_config.json` | matches the README range (412k px per frame). The eval scripts would downscale to about 262k px. |
| 5 | sampling | T 0.7 / top_p 0.8 / top_k 20 (profile `internvideo3-schema`) | `generation_config.json`: `do_sample true, temperature 0.7, top_p 0.8, top_k 20`. The card examples call `generate()` with no overrides (so the same values apply). No repetition/presence penalty anywhere. | [cfg]; [card] | matches, **plus a deliberate `repetition_penalty 1.05`** (below) |
| 5b | max tokens | 1024 (recommended flag `--assist-max-tokens 1024`) | `max_new_tokens=1024` in every example | [card]; [gh] | matches |
| 6 | thinking | none: the template has no `enable_thinking` branch, and no `<think>` appears | README and card examples pass `enable_thinking=True` to `apply_chat_template` (a no-op for this template). The eval scripts have separate `*_thinking.sh` variants passing `enable_thinking=true` to the lmms-eval class, whose code is not published. | [card]; [gh] scripts | matches the default (non-thinking) eval path. Thinking mode could not be reproduced: the switch lives in unreleased eval code. |
| 7 | attention / efficiency | was sdpa; **now flash_attention_2**; one request at a time, bf16 | card/README/demo: `attn_implementation="sdpa"`. Eval scripts: `attn_implementation=flash_attention_2`. No drafter, no vLLM. | [card]; [gh] `eval_*.sh` model_args | **fixed to the eval scripts' FA2.** Greedy text is identical on 6/8 calls, decode is about 3% faster, and the full bench has median 7.10 against 7.91 s at 24/24. |
| 8 | structured output | xgrammar logits processor on the reply schema | not addressed by the authors | — | matches. The grammar cannot bound the `subgoals_completed` array, which is where the loops happen. |

## Bench (24 real calls, `--max-tokens 1024 --image-scale 1.0`, one in flight, GPU 0, sdpa unless noted)

| tag | change | median | p90 | max | prompt | compl | schema ok | length-capped | spoke | flags (match ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| `iv3_audit_base` (**before**) | profile `internvideo3-schema` (gen-config sampling), text first | 11.16 s | 17.64 s | 28.6 s | 5,792 | 190 | 24/24 | 0 | 23 | 2 (0) |
| `iv3_audit_videofirst` | + `images_first` (official order) | 10.53 s | 25.29 s | 43.1 s | 5,793 | 215 | 22/24 | 2 | 21 | 0 (0) |
| `iv3_audit_rep105` | + `repetition_penalty 1.05` | **7.91 s** | **12.56 s** | 21.2 s | 5,792 | 171 | **24/24** | 0 | 19 | 5 (0) |
| `iv3_audit_fa2_rep105` (**after, adopted**) | rep 1.05 + FA2 | **7.10 s** | 14.96 s | 24.3 s | 5,792 | 157 | **24/24** | 0 | 15 | 3 (0) |
| `iv3_audit_after` | rep 1.05 + FA2 + `images_first` | 7.18 s | 14.84 s | 42.0 s | 5,793 | 157 | 23/24 | 1 (call 9) | 17 | 0 (0) |
| `iv3_speed_sdpa` / `iv3_speed_fa2` | greedy, 8 calls, cap 256 | 8.70 / 8.47 s | | | | | 6/8 each | 2 each | | same text 6/8 |

**The runaways are stochastic.** The runs above differ by random sampling, and the shim is not seeded, so one 24-call run cannot measure how often a runaway happens. The stress test replays the 6 calls that ran away in any earlier run (idx 7, 9, 15, 19, 20, 23), 4 times each:

| tag | request | runaways (hit 1024) | schema ok | median |
|---|---|---|---|---|
| `iv3_stress_gencfg` | official sampling, max 1024 | **6/24** (call 7 ×3, call 15 ×1, call 20 ×2) | 18/24 | 19.8 s |
| `iv3_stress_rep105` | + `repetition_penalty 1.05` | **0/24** | 24/24 | 9.2 s |

Greedy decoding loops as well: 2 of 8 calls ran to a 256-token cap at T 0 (`iv3_speed_sdpa`). Low temperature does not help.

**What a runaway costs.** Each one is 1,024 tokens, about 42 s at 41 ms per token, and it arrives as truncated JSON, which the engine reads as `{}`. `repetition_penalty 1.05` removes them and shortens the typical reply (171 against 190 tokens).
- Caveat: HF's repetition penalty also down-weights tokens that appear in the prompt.
- Even so, validity is 24/24 and the replies stay on task. Examples: "You placed raw steak into the pan. Now turn on the heat to cook it properly."; "You're at the serving pass with the correct plate. Serve it now…".

**Sanity read** (reference-flag calls; no variant catches a reference error):

| call | reference | before (base) | rep 1.05 | video first |
|---|---|---|---|---|
| easy_hard_1 t16 (mis_prep) | "You just chopped the cheese instead of trashing it!" | "go to the ingredient crate to get a raw tomato" | flag `leave_unattended`: "You left the cheese on the cutting board… It's now on fire!" (wrong) | "grab a clean plate" |
| easy_medium_1 t32 (wrong_order) | "cheese went on before the chopped tomato" | "Pick up a raw cheese…" | "Okay, you're clearing the plate…" | "I see you're clearing the plate…" |
| easy_hard_2 t94 (forget_next_step) | "you're still at the cheese crate with the correct finished plate. Go to the serving pass" | "You're still holding the plate with the correct order. Go to the serving pass" | "You're at the serving pass… Serve it now" | "**You are still standing by the cheese crate with the correct plate. Go to the serving pass, then serve it.**" (closest to the reference) |
| master_hard_4 t408 (mis_prep) | "you just picked up an onion" | "at the rice crate" | "holding a plate with a tortilla" | "next step is to cook the rice" |

Rep 1.05 raises more flags: 5, all `leave_unattended` or `repeat_step`, and none of them match the reference. 2 are plausible (smoking fryer at t128 and t68). Detection is not improved by any setting.

## Changes to apply
**Before → after** (24 calls):
- median 11.16 → **7.10 s**; p90 17.64 → 14.96 s
- 24/24 → 24/24 valid
- runaways on the prone calls 6/24 → 0/24

The FA2 row's p90 (14.96 s) is above the sdpa + penalty row (12.56 s). That comes from sampled reply length, not from the kernel: FA2 decodes 3% faster on identical greedy text. p90 against the before row still improves (17.64 → 14.96 s).

- **Engine PROFILES**:
```python
# InternVideo3-8B (audit 2026-09-19): generation_config sampling (T0.7/top_p0.8/top_k20) + repetition_penalty 1.05.
# The penalty is NOT in any InternVideo3 source: a deliberate deviation, because the official settings (gen-config
# sampling + the card's max_new_tokens 1024) still run away to the cap (6/24 on the 6 prone calls x4; 0/24 with 1.05).
# Text stays BEFORE the clip (the card puts video first): video-first measured 22/24 and 23/24 (runaways on call 9).
"internvideo3-schema": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 20, "repetition_penalty": 1.05},
                            prefill=None, reply_schema=True),
```
  Keep `--assist-max-tokens 1024`, which is the card's `max_new_tokens`.
- **Shim** (`internvideo3/shim_server.py`, now in place):
  - new env `ATTN` (default `sdpa`) and `FRAMES=video_dup`; default behaviour is unchanged.
  - The pre-audit copy is `internvideo3/logs/shim_server.py.pre_audit`.
- **Env**: `flash-attn 2.8.3` (cu12 torch2.8 cxx11abiTRUE cp312) and `einops 0.8.2` were installed into `envs/internvideo3-8b` only, with `uv pip --python`. `INSTALLER` is `uv`, and the import path is the env. `/opt/conda` was not touched: its flash_attn dist-info is dated 2026-02-25.
- **Serve script**: `serve_internvideo3.sh` now defaults to `ATTN=flash_attention_2`. `ATTN=sdpa` restores the card's setting.

## Pending
- `FRAMES=video_dup` is implemented but not benched. It puts each tick frame in its own temporal patch: 8 timestamped groups and 6.8k prompt tokens instead of 4 groups and 5.2k. Expected cost is about +0.3 s of prefill.
- Thinking mode (`*_thinking.sh`) cannot be reproduced, because the lmms-eval class that implements it is not published.
