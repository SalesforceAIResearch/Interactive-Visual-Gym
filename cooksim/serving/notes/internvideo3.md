# InternVideo3-8B-Instruct — setup (2026-09-18), served name `internvideo3-8b`, port 18700

**Status.** Served; both 24-call benches complete; GPU 1 freed (server killed by pid 19:28 UTC). The FRAMES=images comparison was not run.

**Main finding.** At T0.2 the model sometimes loops inside the JSON and runs to max_tokens, taking about 160 s per call. This happened on 3 of 24 calls.

## The model
- **Repo.** `yanziang/InternVideo3-8B-Instruct`: public, not gated, Apache-2.0, bf16, 18.7 GB, uploaded 2026-05-26. Paper: arXiv 2606.12195 (OpenGVLab / Shanghai AI Lab).
- **It is the official release.** The OpenGVLab/InternVideo GitHub README (InternVideo3 Model Zoo) links exactly this repo. `OpenGVLab/InternVideo3-*` returns 401: it does not exist or is private, even though the model card's code refers to it.
- **Architecture.** `InternVideo3ForConditionalGeneration`: a Qwen3-VL-8B vision tower (patch 16, temporal patch 2) plus a 36-layer LLM whose attention was converted to M²LA, a DeepSeek-style latent attention with kv_lora_rank 896.
  - It ships only as trust_remote_code transformers code.
  - vLLM 0.19.1 has no `internvideo3` architecture, which is why this setup uses a shim rather than vLLM.
- **Chat template.** Qwen style with a `system` branch, so no template fix was needed. It has no thinking toggle: `enable_thinking` is a no-op, and replies contained no `<think>` blocks.

## Stack
| | |
|---|---|
| env | `$ROOT/envs/internvideo3-8b` (uv venv, Python 3.12): torch 2.8.0+cu128, torchvision 0.23.0, transformers 4.57.3 (as the card pins), accelerate 1.15.0, xgrammar 0.2.7, fastapi 0.141.1, uvicorn 0.53.0, qwen-vl-utils 0.0.14 |
| weights | `HF_HOME=$ROOT/hf_cache/internvideo3-8b`, 18 GB, all 27 files except `.claude/` and `__pycache__/` |
| server | `$LM/internvideo3/shim_server.py`, launched by `$LM/pass3_setup/serve_internvideo3.sh` |
| launch | `cd $LM/pass3_setup && setsid nohup flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_internvideo3.sh' > logs/serve_internvideo3.log 2>&1 &` ; stop: `kill $(cat logs/serve_internvideo3.pid)` |
| load | 17 s, about 18.3 GB of GPU memory, no missing-weight warnings |

What the shim does:
- **Endpoints.** It serves `POST /v1/chat/completions`, `GET /v1/models` and `GET /health`. One request is in flight at a time (asyncio lock).
- **Inference.** Transformers `generate`, bf16, sdpa, as on the model card.
- **JSON output.** `response_format` of type `json_schema` is enforced with xgrammar, compiled once per schema. Type `json_object` is constrained to any JSON object. The constraint costs no measurable time: 5.52 s against 5.74 s free decoding on the same call.
- **Frames.** `FRAMES=video` is the default. The 8 frames go through the model's video processor as one clip, in order, with no resampling, timestamps 1.2 s apart and `do_sample_frames=False`. That gives 4 temporal groups × 396 tokens, about 5.2k prompt tokens in total.
  - The model card uses the video path for temporal input, which is why the clip is the default.
  - `FRAMES=images` sends 8 separate images instead, about 6.7k tokens. It is implemented but was not benched.

Fixes that were needed:
1. **Load from the local snapshot path.** With a repo id, transformers 4.57.3 calls `model_info` on the Hub even with `HF_HUB_OFFLINE=1` and crashes. The serve script therefore resolves the snapshot directory.
2. **Tokenizer warning is harmless.** The tokenizer warns about an "incorrect regex", but its output is byte-identical to the raw `tokenizer.json` on a full 3,552-token benchmark prompt.
3. **uv environment variables.** The shell exports `UV_SYSTEM_PYTHON=1`, so `uv pip install` has to be run with `--python <venv>/bin/python`, or it targets /opt/conda.

## Bench (24 real calls, one in flight, `bench.py`, max_tokens 4000, image-scale 1.0, clip mode)
| variant | median | p90 | prompt tok | completion tok | schema ok | runaway (length) | spoke | flags |
|---|---|---|---|---|---|---|---|---|
| T0.2 + schema (`--profile molmo2-schema`) | 10.06 s | 157.6 s | 5,792 | 220 | **21/24** | 3 | 21 | 3 |
| **T0.7 / top_p 0.8 / top_k 20 + schema (generation_config) — recommended** | **9.14 s** | **25.6 s** | 5,792 | 211 | **22/24** | 2 | 21 | 0 |

Reference round-4 API replies spoke 15/24.

**Latency.**
- Prefill of about 5.5k tokens takes about 0.9 s.
- Decode runs at about 39 ms per token. That is transformers' per-token Python overhead on an 8B model on a heavily loaded host (load average about 44); the model code already decodes through the absorbed-latent path.
- On non-runaway calls, latency is 5.3–30 s, and it scales with completion length: 115–668 tokens.
- **The model cannot keep pace.** Even its fast calls land 5 or more ticks late.

**Runaway loops.** Calls 7, 15 and 20 hit the 4,000-token cap at about 160 s each. The model keeps appending the same entries to `subgoals_completed`, for example repeated `{"completed_at": 434/438/442…, "subgoal": "chop(potato)"}` items. The schema allows an unbounded array, so the grammar cannot stop the loop.
- These are the three failed schema checks: the output is truncated JSON, so the engine sees `{}`.
- In a live episode, each runaway freezes the assistant for about 160 s.
- Mitigations for the orchestrator to choose from:
  - run with `--assist-max-tokens 1024`. The longest good reply was 668 tokens, so this caps a runaway at about 40 s;
  - and/or use sampling at T0.7, whose bench is pending;
  - and/or add `repetition_penalty`, which the shim passes through from `extra_body`.

## Sanity read (T0.2 + schema, clip; reference = round-4 API reply)
- **easy_hard_1 t16.** REF: "You just chopped the cheese instead of trashing it! …" (mis_prep). IV3: "Okay, you've corrected the cheese mistake. Now, please go to the plate stack…" (no flag). It misses the error and wrongly calls it fixed.
- **easy_medium_1 t32.** REF: "That's a wrong order: cheese went on before the chopped tomato… Dump the plate's contents…" (wrong_order). IV3: "I see you're at the cheese crate. Pick up a raw cheese slice and bring it to the trash. We'll need to start over with the correct order." (no flag). It half-sees the problem: it wants a restart, but gives a garbled repair.
- **easy_hard_2 t94** (cook idle at the cheese crate). REF: "You have not moved yet… Go to the serving pass now" (forget_next_step). IV3: "You have successfully plated the Caprese… Go to the serving pass and serve it now." (no flag). The instruction is right but the flag is missed.
- **master_hard_4 t408.** REF: "Stop! You just picked up an onion…" (mis_prep). IV3: "The first item you need is a raw tortilla. Head over to the tortilla crate…" (no flag). A miss, and its sense of progress is wrong.
- **master_medium_5 t178.** REF silent. IV3 went into a runaway: 4,000 tokens, 157 s, no reply.
- **Extra case, master_medium_5 t344.** REF silent. IV3 flagged `wrong_order`: "You added the tortilla to the plate out of order… dump the plate at the trash and rebuild…". This is probably a false alarm, since the reference saw nothing.

**Overall.** Unlike GLM-4.6V-Flash (0 flags), InternVideo3 does raise flags: 3/24 (2 × leave_unattended on smoking fryer or pan, which are plausible, and 1 × wrong_order). It is still verbose (21/24 spoke) and misses the reference's detections on the calls read.

## Proposed PROFILES entry and flags
Confirmed by the gencfg bench (22/24 valid, p90 25.6 s against 157.6 s at T0.2):
```python
# InternVideo3-8B (setup 2026-09-18): transformers shim (no vLLM arch), xgrammar-enforced schema; generation_config sampling.
"internvideo3-schema": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 20}, prefill=None, reply_schema=True),
```
- These are the model's own `generation_config.json` values, which follows the harness rule that the model's docs decide the sampling.
- At T0.2 the model loops (3/24 runaways). Low temperature is the classic trigger for repetition loops, and the model's defaults are T0.7 / top_k 20.
- The gencfg run still had 2/24 runaways (calls 15 and 19, about 160 s each) and raised no flags. To cut runaways further, add `"repetition_penalty": 1.05` to `extra_body` and re-bench.
```
--assist-backend openai --assist-model internvideo3-8b --assist-base-url http://127.0.0.1:18700/v1 \
--assist-profile internvideo3-schema --assist-max-tokens 1024 --assist-image-scale 1.0
```
`--assist-max-tokens 1024` rather than 4000 caps a runaway at about 40 s instead of 160 s. The longest valid reply seen was 668 tokens.

Raw rows: `$LM/pass3_setup/results/internvideo3_video_schema{,_gencfg}.json`, `results/iv3_smoke.json`. Logs: `$LM/pass3_setup/logs/bench_iv3_*.log`, `logs/serve_internvideo3.log`.
