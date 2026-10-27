# SETUP — Qwen3-Omni-30B-A3B-Instruct (`qwen3-omni-30b-a3b`), roster row 17

Status on 2026-09-18: **served, benched and verified**, and GPU 0 has been freed. It is served as bf16 on one A100-80GB with a 32k context. No quantization was needed.

    ROOT=$ROOT   LM=$ROOT/simulation_home/local_models

## Weights
- Repo: `Qwen/Qwen3-Omni-30B-A3B-Instruct`, revision `26291f793822fb6be9555850f06dfe95f2d7e695` (2025-09-22). It is public and not gated.
- No newer official revision exists. The Qwen HF org lists only Instruct, Thinking and Captioner for Qwen3-Omni, and no official AWQ or GPTQ build of it.
- Cache: `HF_HOME=$ROOT/hf_cache/qwen3-omni-30b-a3b`, 61 GB on disk. The loaded thinker takes 57.96 GiB on the GPU.
- Shards 14 and 15 (5.5 GB) were not downloaded:
  - `model.safetensors.index.json` shows they hold only `talker.*` and `code2wav.*` tensors.
  - vLLM skips those prefixes anyway.
  - Shard 13 is mixed thinker/talker and is kept.

      HF_HOME=$ROOT/hf_cache/qwen3-omni-30b-a3b hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
          --exclude model-00014-of-00015.safetensors model-00015-of-00015.safetensors

## Env
- `$ROOT/envs/qwen3-omni-30b-a3b` is a `python3 -m venv` built from /opt/conda python 3.12.12.
- It was installed with the venv's own pip: `$ROOT/envs/qwen3-omni-30b-a3b/bin/pip install vllm==0.19.1 openai pillow requests`.
- Result: vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0. Driver 570.124.06, CUDA 12.8.
- **Plain vLLM, not vLLM-Omni.**
  - vLLM 0.19.1 registers `Qwen3OmniMoeForConditionalGeneration`, which maps to `Qwen3OmniMoeThinkerForConditionalGeneration`.
  - It serves the thinker only, with `skip_prefixes=["talker.", "code2wav."]` at load.
  - The model card says the same: "vLLM serve for Qwen3-Omni currently only supports the thinker model".
  - So the output is text only and the talker costs no time or memory. That is exactly what the benchmark needs. vLLM-Omni would only be needed for audio output.
- UV hazard check (coordinator's warning):
  - This env never used `uv`. Its `vllm-0.19.1.dist-info/INSTALLER` reads `pip`, and it imports from the venv path.
  - /opt/conda **does** contain a `vllm-0.19.1.dist-info` (created 18:52 UTC, no INSTALLER file) and a `vllm/` package dir (last changed 19:06 UTC).
  - That was not written by this setup. My only install command was the venv pip above.
  - It is presumably another agent's `uv pip install` hitting `UV_SYSTEM_PYTHON=1`.

## Serve
`$LM/pass3_setup/serve_qwen3_omni.sh` runs `vllm serve` with these settings:
- `--dtype bfloat16 --max-model-len 32768 --max-num-seqs 2 --gpu-memory-utilization 0.92`
- `--limit-mm-per-prompt {"image":8,"video":0,"audio":0} --enable-prefix-caching`
- served name `qwen3-omni-30b-a3b`, port 18800, `HF_HUB_OFFLINE=1`.

    cd $LM/pass3_setup
    setsid nohup flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_qwen3_omni.sh' \
        > logs/serve_qwen3_omni.log 2>&1 < /dev/null &
    kill $(cat logs/serve_qwen3_omni.pid)      # stop (pid = the vllm APIServer; EngineCore exits with it)

- Memory: weights take 57.96 GiB, leaving 13.0 GiB of KV cache (142k tokens, 4.3× concurrency at 32k). The benchmark's ~16k-with-8-images need fits easily.
- Cold start: about 2.5 min, including 53 s of weight loading.
- No chat-template fix was needed. The template renders a system message natively, and the Instruct model has no thinking mode, so it needs no reasoning parser or `enable_thinking` switch.

## Bench (24 genuine calls, one in flight, `bench.py` via the benchmark's own OpenAIBackend)
The proposed profile is not yet in `controller_eval.PROFILES`, and the engine was not edited. `bench_profile.py` injects the profile into the process and then runs `bench.py` unchanged.

| variant | median | p90 | prompt tok | completion tok | schema ok | spoke | flags |
|---|---|---|---|---|---|---|---|
| **card sampling + schema, full frames — chosen** | **3.11 s** | **5.55 s** | 7,361 | 259 | **24/24** | 24 | 3 |
| card sampling + schema, half frames (0.5) | 2.75 s | 4.86 s | 4,961 | 285 | 24/24 | 23 | – |
| `--profile default` (T 0.2, json_object, no schema), full | 2.60 s | 4.57 s | 7,361 | 248 | 24/24 | 24 | – |

The round-4 reference replies spoke on 15/24 calls.

- Results are in `$LM/pass3_setup/results/qwen3omni_*.json` and `logs/bench_qwen3omni_*.log`.
- **Pace.** A tick is about 1.2 s, so the median reply arrives about 2–3 ticks late (p90 about 5 ticks). That is comparable to MiniCPM-o (2.1 s) and faster than Molmo2 (5.1 s).
  - Decode dominates: about 260 tokens of JSON on a 3B-active MoE.
  - With a poll every 2 ticks, the model cannot answer every poll in real time. Under `COOKSIM_ASSIST_LATENCY=measured` replies simply land later.
- **Choice.**
  - The schema costs about 0.5 s and buys nothing measurable, since the model is valid 24/24 without it. It is kept for parity with the other `*-schema` profiles.
  - Half frames save only 0.36 s here, because Qwen3-Omni's vision tokens are already cheap (about 400 per frame at full size, estimated from the full-vs-half prompt difference). Full frames are kept.

### Proposed PROFILES entry (sampling from the model card's vLLM example: T 0.6, top_p 0.95, top_k 20)
```python
"qwen3-omni-schema": dict(temperature=0.6, top_p=0.95, extra_body={"top_k": 20},
                          prefill=None, reply_schema=True),
```
### Benchmark flags
```
--assist-backend openai --assist-model qwen3-omni-30b-a3b --assist-base-url http://127.0.0.1:18800/v1 \
--assist-profile qwen3-omni-schema --assist-max-tokens 4000 --assist-image-scale 1.0
```

## Sanity read (reference = the round-4 API reply in calls.jsonl)
1. **easy_hard_1 t16.** The reference (Gemini 3.1 Pro) said "You just chopped the cheese instead of trashing it!"
   Omni said "Good, you got rid of the cheese. Now go to the plate stack…". This is a **miss**, the same miss MiniCPM-o and Molmo2 make.
2. **easy_medium_1 t32.** The reference (gpt-realtime) said "That's a wrong order: cheese went on before the chopped tomato…"
   Omni said "You've added the cheese to the plate before the tomato. That's the wrong order—tomato must go on first… start over with a clean plate." with `flag: wrong_order`. This is a **correct catch**, which neither MiniCPM-o nor Molmo2 made.
3. **easy_hard_2 t94.** The cook is still at the crate, unserved. The reference said "You have not moved yet… Go to the serving pass now."
   Omni said "You've successfully served the correct Caprese plate. Good job!" This is **false**, the same hallucination as MiniCPM-o.
4. **hard_hard_1 t254.** The reference said "…You just need to take it to the serving pass."
   Omni said "You've successfully assembled the entire Fish & Chips Platter in the correct order… You can proceed to serve it." This **agrees** with the reference.
5. **master_medium_5 t178.** The reference stayed silent.
   Omni said "The rice in the pot is smoking! Grab a plate… before it burns." It is plausibly right, though unrequested; Molmo2 said the same at half size.

Overall, Omni is fluent and always schema-valid, and it raised 3 flags in 24 calls: `wrong_order` twice and `leave_unattended` once.
- It speaks on every call (24/24, against 15/24 for the reference). Expect a nag/false-alarm rate like the other local models.
- It sometimes hallucinates completion ("served").
- It sometimes echoes an imagined cook utterance. At master_hard_2 t278 it said "Okay, good catch. You're right, the steak is just sitting on the counter…".

## Native duplex / proactive capability — what our harness would need (not implemented)
Qwen3-Omni's real-time mode is a **streaming thinker with a persistent context**:
- Audio arrives as 12.5 Hz AuT chunks, and video frames arrive with TMRoPE timestamps. Both are prefilled incrementally into one growing KV cache.
- The talker streams speech from the first chunk of thinker output.
- In the open release, "when to speak" is not a learned proactive head. Turn-taking and interruption (VAD, barge-in) sit in the application layer, as in Alibaba's Omni-Realtime API.

To use it natively instead of our stateless poll every 2 ticks, the harness would need four things:
1. **One session per episode.** Each tick appends only the new frame(s) as a video chunk, plus the cook's speech as audio (`audio` modality, now limited to 0). Re-sending the 8-frame window each time would go; prefix caching or vLLM-Omni's streaming input would keep the prefill incremental.
2. **A speak/silence decision every chunk.** This would come either from our JSON `text:""` convention, applied to a short per-chunk decode, or from an external trigger policy, because the model has no native "stay silent" behaviour.
3. **Barge-in semantics.** The harness must decide what happens when the cook speaks while the assistant's reply is still decoding or playing.
4. **For spoken output, vLLM-Omni** (thinker + talker + code2wav) with the talker's audio charged to the latency clock. The text-only thinker served here is enough for the current text-reply benchmark.
