# Local assistant models — setup report (pass@3 eval, 2026-09-15)

**How these were measured.** All benches replay the same 24 genuine assistant calls, one request at a time.
- `calls.jsonl` is built by `build_calls.py` from sweep-v5 round-4 records.
  - The system prompt is the record's `supervisor_system`.
  - The user text is a verbatim copy of `recovery_rollout.sup_turn`.
  - The frames are the 8 side-by-side 1042×396 JPEGs, re-rendered from the `.states.jsonl` dumps on SwiftShader. On the GPU renderer the WebGL context is lost and the scene comes out black, so every frame was brightness-checked; the dimmest pane scored 104/255.
- `bench.py` sends each call through the benchmark's own `tools/controller_eval.OpenAIBackend` (imported, not copied), after 2 untimed warm-up calls.
- "Schema" means `response_format = {"type":"json_schema", "strict":true, "schema": tools/assistant_reply_schema.json}`. That is identical to `reply_schema.json` here, and it is what the `*-schema` benchmark profiles send.
- "Schema ok" means the reply contains `cook_progress{subgoals_completed, current_subgoal, subgoals_future}`, `goal_change` and `text` (a string).
- "Spoke" counts replies with non-empty `text`; the round-4 API replies to the same 24 calls spoke 15/24.

Raw per-call rows are in `results/<tag>.json`; console logs are in `logs/`.

## MiniCPM-o 4.5 — GPU 2, port 18100, served name `minicpmo`

| | |
|---|---|
| stack | env `~/envs/minicpmo-vllm`: vllm 0.19.1, torch 2.10.0, transformers 4.57.6 |
| weights | `openbmb/MiniCPM-o-4_5-awq` (W4A16 AWQ LLM; vision/resampler bf16), 6.9 GiB loaded |
| server | `serve_minicpmo.sh` (defaults): `--max-model-len 16384 --max-num-seqs 2 --gpu-memory-utilization 0.85 --limit-mm-per-prompt {"image":8,"video":0,"audio":0} --mm-processor-kwargs {"max_slice_nums":1} --enable-prefix-caching --trust-remote-code` |
| thinking | off (`chat_template_kwargs.enable_thinking=false` in the profile) |

| variant (24 real calls) | median | p90 | prompt tok | completion tok | schema ok | spoke |
|---|---|---|---|---|---|---|
| AWQ, 1 slice, **no** schema (`json_object`) | 1.82 s | 2.95 s | 4,741 | 282 | 24/24 | 24 |
| **AWQ, 1 slice, schema — chosen** | **2.07 s** | **3.57 s** | 4,741 | 270 | **24/24** | 24 |
| AWQ, HD slicing (`max_slice_nums` 9), schema | 2.77 s | 4.33 s | 6,325 | 276 | 24/24 | 24 |
| bf16, 1 slice, schema | 3.57 s | 6.47 s | 4,741 | 267 | 24/24 | 24 |

**Choice.** AWQ with 1 slice.
- HD slicing adds 0.7 s and bf16 adds 1.5 s.
- Neither changed what the replies say: reading the same calls, the bf16 and HD-sliced replies make the same misses as AWQ.
- The schema costs about 0.25 s and changes nothing for this model, which already returned valid JSON 24/24 without it. It is kept for parity with the other schema profiles.

**Sanity read** (5 calls, reference = round-4 API reply). The model is fluent and always valid, but it narrates the next step every call and misses visible errors:
- easy_hard_1 t16: the reference says "you chopped the cheese instead of trashing it"; MiniCPM says "Now that you have the cheese out of the way, head to the plate stack…".
- easy_medium_1 t32: the reference calls a wrong order (cheese before tomato); MiniCPM says "Now you can get the raw cheese from the crate…".
- easy_hard_2 t94 (the cook is idle at the crate): MiniCPM says "You've successfully served the Caprese dish… Episode complete." That is false.
- It spoke on 24/24 calls against 15/24 for the reference, and set `flag` on 0/24.

Expect a high false-alarm/nag rate and little detection. That is the model, not the setup: HD slicing and bf16 did not change it.

**Benchmark flags**
```
--assist-backend openai --assist-model minicpmo --assist-base-url http://127.0.0.1:18100/v1 \
--assist-profile minicpmo-schema --assist-max-tokens 4000 --assist-image-scale 1.0
```
Launch: `CUDA_VISIBLE_DEVICES=2 nohup bash serve_minicpmo.sh > logs/serve_minicpmo_final.log 2>&1 &`. Cold start is about 5 min when weights are cached in the page cache; the first load took about 14 min.

## Molmo2-8B — GPU 3, port 18400, served name `molmo2-8b`

| | |
|---|---|
| stack | env `~/envs/molmo2` (new venv): vllm 0.19.1, torch 2.10.0, transformers 5.17.0; `HF_HOME=~/hf_cache/molmo2` |
| weights | `allenai/Molmo2-8B` bf16 (Qwen3-8B + SigLIP2), 16.1 GiB loaded. No official INT4 exists for the 8B model. |
| server | `serve_molmo2.sh`: `--trust-remote-code --max-model-len 32768 --max-num-seqs 2 --gpu-memory-utilization 0.90 --limit-mm-per-prompt {"image":8,"video":0} --enable-prefix-caching --chat-template molmo2_chat_template_system.jinja` |
| thinking | none (not a reasoning model) |

**Required fix.** The shipped chat template raises "Conversation roles must alternate user/assistant/…" on any `system` message, and every benchmark call sends one, so every call returned HTTP 400. `molmo2_chat_template_system.jinja` does two things:
- It prepends the system text to the first user turn.
- For conversations with no system message it renders byte-identically to the original (checked with jinja2).

The harness is unchanged.

| variant (24 real calls) | median | p90 | prompt tok | completion tok | schema ok | spoke |
|---|---|---|---|---|---|---|
| full frames, schema | 7.38 s | 10.93 s | 13,709 | 273 | 24/24 | 21 |
| **half frames (`image-scale 0.5`), schema — chosen for parity** | **5.10 s** | **8.45 s** | 8,485 | 262 | **24/24** | 22 |
| half frames, **no** schema (`json_object`) | 3.54 s | 6.89 s | 8,485 | 264 | 24/24 | 21 |

**Choice.** Half-size frames.
- Molmo2 tiles each 1042×396 frame into up to 8 crops, about 1,150 tokens per frame at full size. Halving the frames cuts the prompt from 13.7k to 8.5k tokens and saves 2.3 s per call.
- Replies were the same or slightly better. Of 6 calls read side by side, 2 had identical text, the same 3 flags were raised, and at master_medium_5 t178 the half-size reply warned "The rice in the pot is smoking!" where the full-size reply narrated.
- The schema constraint costs Molmo2 about 1.6 s and buys nothing measurable: 24/24 valid either way. I benched with it because the other models use it. Without it Molmo2 is 1.5 s faster, so whether to use it is the parent's parity call.
- After the image cut, the remaining time is mostly decode (about 265 tokens on an 8B bf16 model).

**Sanity read** (5 calls):
- It narrates the next step and asks clarifying things. It spoke 22/24 times against 15/24 for the reference.
- It raised 3 flags in 24 calls: `forget_next_step` (easy_hard_2 t94, cook idle at the crate: a correct call), `leave_unattended` and `mis_prep`.
- Misses: easy_hard_1 t16. The reference caught "chopped the cheese instead of trashing it"; Molmo2 said "Good, you're putting the cheese down…".
- It grounds places reasonably ("the fryer on the right wall"), noticeably better than MiniCPM-o.

At about 5 s median, each reply arrives about 5 ticks late under `COOKSIM_ASSIST_LATENCY=measured`.

**Benchmark flags.** There is no Molmo2 profile in `controller_eval.PROFILES`. Proposed profile: `"molmo2-schema": dict(temperature=0.2, extra_body={}, prefill=None, reply_schema=True)`. Molmo2 publishes no sampling guidance and ships no generation config, so this is the `default` profile's T=0.2 plus the schema, which is what was benched.
```
--assist-backend openai --assist-model molmo2-8b --assist-base-url http://127.0.0.1:18400/v1 \
--assist-profile molmo2-schema --assist-max-tokens 4000 --assist-image-scale 0.5
# without the schema (1.5 s faster, same 24/24 validity): --assist-profile default
```
Launch: `CUDA_VISIBLE_DEVICES=3 nohup bash serve_molmo2.sh > logs/serve_molmo2.log 2>&1 &`. The first load took about 5 min.

## Not set up
GLM-4.6V-Flash and StreamingVLM were not needed, because both lane models served correctly. vLLM 0.19.1 includes `glm4v`/`glm4_1v` model code if a fallback is wanted later.

## GLM-4.6V-Flash — attempted on GPU 1, ABANDONED 2026-09-16 00:50 UTC (not a model fault)

Stopped on the coordinator's deadline rule (serving *and* benched by 01:15 UTC or not at all). GPU 1 was left free and nothing was started in its place.

**What was verified before the stop** (all still valid for a later attempt):
- Weights: `zai-org/GLM-4.6V-Flash`, 20 GB bf16, downloaded to `~/hf_cache/glm` (14/14 files).
- Architecture `Glm4vForConditionalGeneration` (`model_type: glm4v`) is in the vLLM 0.19.1 registry — native support, no remote code, so `~/envs/molmo2` (vllm 0.19.1 / torch 2.10.0 / transformers 5.17.0) can serve it; the dedicated `~/envs/glm` venv was still installing when the attempt was stopped.
- Chat template: **no fix needed**, unlike Molmo2. It has an `m.role == 'system'` branch rendering `<|system|>`, checked by rendering it locally.
- Thinking toggle: `chat_template_kwargs={"enable_thinking": false}` appends `/nothink` to the user turn and prefills `<think></think>` after `<|assistant|>`. Verified by local render.
- Sampling from the model card: temperature 0.8, top_p 0.6, top_k 2, repetition_penalty 1.1.
- `serve_glm46v_flash.sh` is written and ready (port 18500, `VENV` overridable; defaults to `~/envs/molmo2`).

**Why it was abandoned.** The server process reached uninterruptible disk-wait and stayed there: after 11.75 min it had produced **0 bytes of log**, allocated no GPU memory, and was reading ~28 KB/s (`/proc/<pid>/io` read_bytes 173.1 → 173.5 MB over 15 s). vLLM had not finished importing, with 20 GB of weights still to load after that. The shared filesystem was saturated by the three concurrent timed-episode servers (GPU 3 at 100% util) plus the pip download. Nothing indicates a problem with GLM itself — the same box loaded these weights' neighbours in 5–14 min when less contended.

**To resume later** (needs a serving endpoint ~15–20 min before any bench):
```
CUDA_VISIBLE_DEVICES=<free gpu> nohup bash serve_glm46v_flash.sh > logs/serve_glm.log 2>&1 &
--assist-backend openai --assist-model glm-4.6v-flash --assist-base-url http://127.0.0.1:18500/v1 \
--assist-profile glm46v-flash-schema --assist-max-tokens 4000 --assist-image-scale 1.0
```
Profile to add to `controller_eval.PROFILES` (card sampling + thinking off + schema):
```python
"glm46v-flash-schema": dict(temperature=0.8, top_p=0.6,
    extra_body={"top_k": 2, "repetition_penalty": 1.1,
                "chat_template_kwargs": {"enable_thinking": False}},
    prefill=None, reply_schema=True),
```
No latency or schema-validity numbers were measured, so GLM has **no** benched result.
