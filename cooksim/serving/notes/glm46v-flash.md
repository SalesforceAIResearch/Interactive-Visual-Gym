# GLM-4.6V-Flash — setup (2026-09-18), served name `glm-4.6v-flash`, port 18500

**Status: ready.** Served with vLLM, benched on the 24 genuine assistant calls, GPU freed afterwards.
The 2026-09-16 attempt was abandoned on filesystem contention, not on a model fault. This time the server
was up 3 min after launch.

## Stack
| | |
|---|---|
| env | `$ROOT/envs/glm` (the dedicated venv the 09-16 attempt built): vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0 |
| weights | `zai-org/GLM-4.6V-Flash`, bf16, 20 GB on disk (`HF_HOME=$ROOT/hf_cache/glm`, 14/14 files). Loaded size 19.24 GiB. Weights load in 14 s, and the engine is ready 47 s after process start. |
| architecture | `Glm4vForConditionalGeneration`. vLLM supports it natively, with no remote code. |
| server | `$LM/pass3_setup/serve_glm46v_flash.sh`: `--max-model-len 32768 --max-num-seqs 2 --limit-mm-per-prompt {"image":8,"video":0} --gpu-memory-utilization 0.90 --enable-prefix-caching`, `VLLM_USE_FLASHINFER_SAMPLER=0` |
| launch | `cd $LM/pass3_setup && setsid nohup flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_glm46v_flash.sh' > logs/serve_glm.log 2>&1 &` ; stop: `kill $(cat logs/serve_glm.pid)` |

**One fix in this round.** The script's default `VENV` pointed at `$ROOT/envs/molmo2`, another model's env. Under the
one-env-per-model rule it now defaults to `$ROOT/envs/glm`. Both envs have the same vllm version. Nothing else changed, and no chat-template fix was needed:
the shipped template has a `system` branch.

## Bench (24 real calls, one in flight, `bench.py`, max_tokens 4000)
| variant | median | p90 | prompt tok | completion tok | schema ok | spoke | flags raised |
|---|---|---|---|---|---|---|---|
| **engine profile `glm46v-flash-schema`** (T 0.2, thinking off, schema), full frames | **4.46 s** | **7.77 s** | 8,290 | 245 | **24/24** | 23 | 0 |
| same profile, `--image-scale 0.5` | 3.81 s | 7.15 s | 5,210 | 244 | 24/24 | 24 | 0 |
| `--profile default --extra '{"chat_template_kwargs":{"enable_thinking":false}}'` (json_object, no schema) | 3.49 s | 6.86 s | 8,290 | 245 | 24/24 | 23 | 0 |
| engine profile + model-card sampling (T 0.8, top_p 0.6, top_k 2, rep 1.1), full frames | 3.64 s | 7.08 s | 8,290 | 255 | 24/24 | 20 | 1 (correct) |

Reference round-4 API replies to the same calls spoke 15/24.

**How to read the latencies.** Completion tokens are the same across variants (about 245), so the differences between rows are mostly
run-to-run noise. The host load average was about 44 during the runs, because other agents were building envs. The two
schema runs at full frames differ by 0.8 s with identical tokens. Thinking is really off: `reasoning_content` was
empty on 96/96 calls, and the completions are only the JSON. Latency is decode-bound, at roughly 245 tokens per call. Halving the
frames cuts the prompt from 8.3k to 5.2k tokens but saves only about 0.6 s.

**Pace.** At about 3.5–4.5 s median and about 7 s p90, each reply lands 3–4 ticks late at 1.2 s per tick. The model
cannot keep pace tick-by-tick. It is in the same band as Molmo2 (5.1 s) and slower than MiniCPM-o (2.1 s).

## Is the engine profile right?
It works as specified: 24/24 schema-valid, thinking off, no truncation. The one open question is sampling.
- The engine uses T 0.2. The model card recommends T 0.8 / top_p 0.6 / top_k 2 / repetition_penalty 1.1. With top_k 2, that setting is near-greedy anyway.
- Under card sampling, validity and latency were the same, and the model raised the only correct flag seen in any GLM run. At easy_medium_1 t32 it flagged `wrong_order` with "You've messed up the assembly order. Cheese shouldn't be on the plate yet.", where the reference says wrong_order. That is n=1, so it is weak evidence.
- By the harness's own rule ("each model's own docs decide the sampling"), I propose switching to card sampling:
```python
"glm46v-flash-schema": dict(temperature=0.8, top_p=0.6,
    extra_body={"top_k": 2, "repetition_penalty": 1.1,
                "chat_template_kwargs": {"enable_thinking": False}},
    prefill=None, reply_schema=True),
```
The current T 0.2 entry is also acceptable. It is what was benched as the headline row above.

**Benchmark flags**
```
--assist-backend openai --assist-model glm-4.6v-flash --assist-base-url http://127.0.0.1:18500/v1 \
--assist-profile glm46v-flash-schema --assist-max-tokens 4000 --assist-image-scale 1.0
# (--assist-image-scale 0.5 saves ~0.6 s with the same replies on the calls read; use it for parity with Molmo2 if wanted)
```

## Sanity read (5 calls, engine profile, full frames; reference = round-4 API reply)
The model is fluent, always valid and well grounded in places ("rice crate on the bottom left"). It narrates the next step and
misses visible errors. With the engine profile it raised no flag on any of the 24 calls.
- **easy_hard_1 t16.** REF: "You just chopped the cheese instead of trashing it! Please trash this chopped cheese, then grab a tomato to chop." (mis_prep). GLM: "Okay, now go to the tomato crate and pick up a tomato." (no flag). Miss.
- **easy_medium_1 t32.** REF: "That's a wrong order: cheese went on before the chopped tomato…" (wrong_order). GLM: "Okay, you're heading to the trash. That's the right first step to undo the mistake." (no flag). Miss. Card sampling caught it.
- **easy_hard_2 t94** (cook idle at the cheese crate). REF: "You have not moved yet — you're still at the cheese crate… Go to the serving pass now" (forget_next_step). GLM: "You're at the serving pass with the correct plate. Serve it now to complete the order." That is a false location claim.
- **master_hard_4 t408.** REF: "Stop! You just picked up an onion, but the Loaded Burrito recipe does not call for onion…" (mis_prep). GLM: "Go to the rice crate on the bottom left and pick up some rice." Miss.
- **master_medium_5 t178.** REF silent; GLM silent. Agree.

Expect the same profile as MiniCPM-o and Molmo2: it speaks nearly every call (23/24 against 15/24 for the reference), gives low detection and
nag-type false alarms.

Raw rows: `$LM/pass3_setup/results/glm46v_flash_{schema,schema_half,default_nothink,schema_cardsampling}.json`;
logs: `$LM/pass3_setup/logs/bench_glm_*.log`, `logs/serve_glm_0918.log`.
