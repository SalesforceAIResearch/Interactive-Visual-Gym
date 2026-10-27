# Proact-VL (roster row 19): served, but the reply JSON is built by the shim

Set up 2026-09-18 (UTC). Served name `proact-vl`, port 18950, benched on GPU 0, GPU freed afterwards.

## What it is
- Paper: *Proact-VL: A Proactive VideoLLM for Real-Time AI Companions*, arXiv 2603.03447 (ICML 2026).
- Code: github.com/microsoft/AnthropomorphicIntelligence, sub-folder `Proact-VL/`, cloned to `src/`.
- How it decides to speak:
  - Every 1-second video chunk ends with a `<|FLAG|>` token.
  - A small gated MLP response head (`state_proj`, read from layer −2) scores that token.
  - If sigmoid(score) > `state_threshold`, the model generates ≤12 tokens of speech for that second. Otherwise the silence continuation `" ..."` is prefilled.
  - Utterances run on across consecutive active seconds. This is the "transition smoothing".

## Weights: which checkpoint and why
- The Model Zoo checkpoints (`oaaoaa/proactvl_base_{liveccbase,qwen2vl,qwen2_5vl,qwen3vl}`) are **gated** (`gated: auto`, click-through licence, cc-by-nc-4.0). They need an HF token, and this machine has none, so they were not used.
- `quickstart.py` also says the release is "under compliance review".
- The same official collection (`huggingface.co/collections/oaaoaa/aicompanion`) contains **`oaaoaa/proact_lora_base_liveccbase_final`**, which is **not gated**. It is the authors' own upload: a LoRA adapter (r 32) on `chenjoya/LiveCC-7B-Base` (ungated; Qwen2-VL-7B architecture).
  - Its `modules_to_save` is `state_proj`, the trained FLAG head. The adapter header lists `base_model.model.state_proj.{gate,up,down}_proj.weight`.
  - It also carries the resized `embed_tokens` / `lm_head` base layers.
  - `config.json`: `model_type proact_model`, `active_layer_id −2`, `state_threshold 0.5`.
- **This is what is served.** It is the Proact-VL<sub>LiveCC-7B-Base</sub> recipe in its LoRA form (`script/train/lora_liveccbase.sh`), not the gated full fine-tune. Say so wherever results are reported.
- Downloaded to `HF_HOME=$ROOT/hf_cache/proact-vl`: LiveCC-7B-Base (16.6 GB bf16) plus the adapter (2.39 GB). About 18 GB on disk.

## Environment
- Env: `$ROOT/envs/proact-vl` (uv venv; Python 3.11.15), with:
  - torch 2.8.0+cu128 and torchvision 0.23.0
  - transformers 4.57.1, peft 0.18.1 (the adapter was saved with 0.18.1), accelerate
  - qwen-vl-utils 0.0.14, qwen-omni-utils 0.0.8
  - flash-attn 2.8.3 (prebuilt wheel `cu12torch2.8cxx11abiTRUE-cp311`)
  - fastapi / uvicorn
- The pins follow `src/Proact-VL/script/env/prepare_env.sh`. The one exception is the flash-attn ABI: pip torch 2.8 reports `_GLIBCXX_USE_CXX11_ABI=True`, so the abiTRUE wheel is the matching one.
- The repo has no setup.py. The shim puts `src/Proact-VL` on `sys.path`.
- **Environment hazard.** This shell exports `UV_SYSTEM_PYTHON=1`, so `uv pip install` with only `VIRTUAL_ENV` set installs into `/opt/conda`.
  - My first build attempt printed "Using Python 3.12.12 environment at: /opt/conda". I killed it before its first package finished; it logged no "Installed" line.
  - The torch 2.8.0 / transformers 4.57.3 now in `/opt/conda` came from another agent's concurrent install (pid 27558, `VIRTUAL_ENV=envs/internvideo3-8b`, started 18:51). It is not from this setup: 4.57.3 is not a version I requested.
  - All installs here were then redone with `uv pip install --python $ROOT/envs/proact-vl/bin/python` and `UV_SYSTEM_PYTHON` unset. The env was verified to contain its own packages.

## Serving
- Script: `$LM/pass3_setup/serve_proact_vl.sh`. The shim is `shim_server.py` in this directory.
```
cd $LM/pass3_setup
setsid nohup flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_proact_vl.sh' \
    > logs/serve_proact_vl.log 2>&1 < /dev/null &
kill $(cat logs/serve_proact_vl.pid)          # stop
```
- Startup is about 2.5 min: 10 s for the base weights, about 2 min for the PEFT load and merge. It uses about 17 GB of GPU memory.
- Knobs (environment variables):

| variable | default | what it sets |
|---|---|---|
| `PROACT_THRESHOLD` | 0.5 | FLAG threshold. 0.5 is the config / quickstart / eval-script default; the README's eval example uses 0.3. |
| `PROACT_NEW_CHUNKS` | 2 | How many of the newest chunks count as new. |
| `PROACT_QUERY_AT` | `new` | Where the benchmark user text goes: the first new chunk (`new`) or chunk 0 (`first`). |
| `PROACT_MAX_NEW` | 12 | Tokens generated per active second (native). |

**What a request does.** It is stateless; one request runs at a time. It reuses the model's own `Assistant` class (`prime_system_prompt`, `forward_user`, `forward_assistant`, `forward_custom_assistant`). Only the frame source is new.
1. The benchmark system message primes a fresh session.
2. Each of the ≤8 frames becomes one 1-s chunk.
   - Qwen2-VL video wants 2 frames per chunk (temporal patch 2), so the tick frame is duplicated.
   - It is resized with `smart_resize` inside the model's MIN/MAX_PIXELS: 1042×396 becomes 1036×392, about 518 tokens.
3. The user turn is built by `_make_user_prompt`'s logic: `<video><|query_start|>…<|query_end|><|FLAG|>`.
4. The benchmark user text is the query on the first new chunk. That is the native guidance protocol: a user query arrives at the second it is asked.
5. On every chunk the FLAG head decides natively. When it fires on a history chunk, the model's speech stays in the KV context but is not returned.
6. **Reply text** = the utterances the model produced on the last 2 chunks, i.e. the frames this poll adds. If the FLAG is silent on both, the text is `""`.
7. **JSON: shim-wrapped, not the model's own.** Proact-VL emits ≤12-token speech per second and has no way to produce the schema. The shim returns `{"cook_progress": {"subgoals_completed": [], "current_subgoal": {"hypotheses": [], "since": t}, "subgoals_future": {"remaining": [], "constraints": []}}, "goal_change": null, "text": <speech>, "flag": null}`.
   - `flag` is always null. The FLAG head says *when* to speak, not *which* error, so detection can only be graded from `text`.
8. Every response also carries `x_proact.trace`: per-chunk FLAG score, active, said. The same trace is logged to `logs/requests.jsonl`.

The shim accepts and ignores `response_format`, `temperature` and `max_tokens`. Sampling is the model's native setting: T 0.7, top_p 0.9, repetition 1.15, do_sample.

## Bench (24 genuine calls, `--profile default --max-tokens 4000 --image-scale 1.0`, GPU 0)

| variant | median | p90 | prompt tok | completion tok | schema ok | spoke |
|---|---|---|---|---|---|---|
| **query on first new chunk (chosen, native protocol)** `proact-vl_native_t50` | **4.34 s** | **4.64 s** | 8,379 | 44.5 | 24/24 (wrapped) | 24/24 |
| query on chunk 0, `proact-vl_queryfirst_t50` | 4.72 s | 6.80 s | 8,379 | 54 | 24/24 (wrapped) | 23/24 |

- Schema validity is 24/24 by construction, because the shim builds the JSON.
- The reference (round-4 API replies) spoke on 15/24 calls.
- **Pace.** The game runs at about 1.2 s per tick, so each reply lands about 3–4 ticks late. The model cannot keep pace in this stateless form: every call re-prefills about 8.4k tokens and runs up to 8 FLAG reads and generations.
  - Measured cost is about 0.54 s per 1-s chunk, including generation.
  - That suggests a real native session, which only adds 1 chunk per second, would keep pace at 1 fps.

**FLAG behaviour.** The head fires on almost every chunk once it has started (scores 0.5 → 0.9–1.0 from chunk 1 or 2 on; chunk 0 is about 0.3). So in effect it never chooses silence on this input.

## Sanity read (5 calls; reference = the round-4 API reply in calls.jsonl)
The model does not engage with the kitchen. It narrates or paraphrases the 13k-character supervisor system prompt, or echoes the transcript lines in the user text, in ~12-token pieces cut mid-sentence. This is a game-commentary LoRA facing a long, off-distribution rulebook prompt.

| call | reference reply | Proact-VL reply |
|---|---|---|
| easy_hard_1 t16 | "You just chopped the cheese instead of trashing it! Please trash this chopped cheese…" | "t15 cook: I'm chopping this tomato on the" (echo of the transcript) |
| easy_medium_1 t32 | "That's a wrong order: cheese went on before the chopped tomato…" | "t31 cook: Move to the fryer," |
| easy_hard_2 t94 | "…you're still at the cheese crate with the correct finished plate. Go to the serving pass now…" | "You are the kitchen supervisor, watching a single" |
| expert_hard_2 t128 | "Nice save—scooping it stopped the smoking. But the plate order is now broken…" | "You can always check the current timestamp" |
| master_hard_2 t278 | (silent) | "I'm sorry, but the steak is raw, and" (the closest it came to a kitchen observation) |

Moving the query to chunk 0 changed nothing material. Replies were still prompt fragments ("Your JOB is twofold: (1) DETECT", "mis_prep, omit_subgoal, repeat_step, search,"). Expect near-zero detection and a maximal nag / false-alarm rate. That is the model on this input, not a serving fault: the FLAG head loaded (the scores are structured, not ≈0.5 noise) and the trace follows the authors' `infer_one_chunk`.

## Proposed benchmark profile and flags
The shim ignores sampling and `response_format`, so the profile only needs to name the model. `default` works unchanged: `json_object` is accepted and ignored. For readability the orchestrator may add:
```python
# Proact-VL (LoRA on LiveCC-7B-Base, served by local_models/proact-vl/shim_server.py): the shim runs the model's
# own FLAG-head streaming loop over the frames and WRAPS its free-text speech into the reply schema; sampling is
# the model's native (T 0.7 / top_p 0.9 / rep 1.15) and any request sampling/response_format is ignored.
"proact-vl-shim": dict(temperature=None, extra_body={}, prefill=None),
```
```
--assist-backend openai --assist-model proact-vl --assist-base-url http://127.0.0.1:18950/v1 \
--assist-profile default --assist-max-tokens 4000 --assist-image-scale 1.0
```
Do not use `--assist-image-scale 0.5`. The frames are already at the model's native per-chunk pixel budget, and halving them only removes detail.

## What a native session would need from the harness
Proact-VL is built to hold one KV session per episode.
- The system prompt is primed once.
- One 1-s chunk (2 frames) is pushed per second of game time, with the cook's utterances as `<|query_start|>…<|query_end|>` (or as `<|history_start|>` co-speaker text) on the chunk where they occur.
- The FLAG head is read on every chunk. Nothing is materialised unless it fires. An utterance is the concatenation of consecutive active seconds, ending when the head drops.
- The model's own KV eviction (`max_kv_tokens` 16,384, 20% eviction, RoPE re-shift) handles long episodes.

From the harness that means:
- a `stream` backend: `push_chunk(frames, t, cook_speech)` → `{active, score, speech_piece}`, pushed each tick rather than polled every 2 ticks;
- latency charged from chunk arrival to the first speech token, with the FLAG read free;
- a sentence assembler that joins pieces across active seconds into one utterance;
- a separate small classifier or second-stage call if a `flag.error` type is wanted, because the head carries no error type.

The per-chunk cost measured here (about 0.5 s on an A100 for prefill + FLAG + ≤12 tokens) is within a 1 s budget.

## Files
- `shim_server.py`: the shim. `logs/requests.jsonl` holds the per-call FLAG traces.
- `$LM/pass3_setup/serve_proact_vl.sh`: the launcher.
- Results: `$LM/pass3_setup/results/proact-vl_native_t50.json` (chosen) and `proact-vl_queryfirst_t50.json`.
- Server logs: `$LM/pass3_setup/logs/serve_proact_vl*.log`.
- `src/`: the authors' code (`microsoft/AnthropomorphicIntelligence`).
