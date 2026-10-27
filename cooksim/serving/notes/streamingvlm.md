# StreamingVLM (roster row 20): served, but the reply JSON is built by the shim

Set up 2026-09-18 (UTC). Served name `streamingvlm`, port 19000, benched on GPU 0, GPU freed afterwards. The weights are released, so StreamOV (the fallback named in the roster) was not needed.

## What it is
- Paper: *StreamingVLM: Real-Time Understanding for Infinite Video Streams*, Xu et al. (MIT HAN Lab), arXiv 2510.09608.
- Code: github.com/mit-han-lab/streaming-vlm, cloned to `src/`.
- Weights: `mit-han-lab/StreamingVLM`. **Ungated**, no licence field. Qwen2.5-VL-7B fine-tune, `Qwen2_5_VLForConditionalGeneration`, bf16, 16.6 GB. Downloaded to `HF_HOME=$ROOT/hf_cache/streamingvlm`.
- How it streams:
  - Video and text are interleaved at 1-s chunks (FPS 2, so 2 frames per chunk). Each chunk is a user turn `Time=a-b s` + video; the assistant writes ≤20 tokens of commentary per chunk, ending in `" ..."`.
  - The KV cache is attention sinks (512 text tokens) + a 512-token text window + a 16-chunk vision window, with contiguous re-indexed RoPE ("shrink" mode).
  - This is text-out live commentary. It has **no separate speak/silent head**: staying silent means a chunk whose commentary is only the `" ..."` marker.

## Environment
- Env: `$ROOT/envs/streamingvlm` (uv venv; Python 3.11.15), with:
  - torch 2.7.1+cu126 and torchvision 0.22.1
  - **transformers 4.52.4**, flash-attn 2.8.3 (prebuilt wheel `cu12torch2.7cxx11abiTRUE-cp311`)
  - qwen_vl_utils 0.0.11, livecc-utils 0.0.2 (editable, from `src/streaming_vlm/livecc_utils`)
  - liger_kernel 0.6.1, decord, av ≥14, numpy 1.26.4, fastapi / uvicorn
- **Fix: transformers version.** `scripts/env_infer.sh` installs 4.51.3 last, but the patched attention code imports `make_flex_block_causal_mask` from `modeling_qwen2_5_vl`, which only exists from 4.52. The server failed to import on 4.51.3. 4.52.4 is the version pinned in `infer_requirements.txt`, and it works.
- **Fix: av.** The `av==14.4.0` pin has no cp311 wheel here and fails to build from source. It was unpinned; av is not on the served path.
- **Environment hazard.** `UV_SYSTEM_PYTHON=1` is exported in this shell.
  - My first build attempt was pointed at `/opt/conda` and was killed during download; it installed nothing.
  - Every install was then redone with `uv pip install --python $ROOT/envs/streamingvlm/bin/python` and `UV_SYSTEM_PYTHON` unset.
  - The env contains its own torch 2.7.1 and transformers 4.52.4 (verified by import).

## Serving
- Script: `$LM/pass3_setup/serve_streamingvlm.sh`. The shim is `shim_server.py` in this directory.
```
cd $LM/pass3_setup
setsid nohup flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_streamingvlm.sh' \
    > logs/serve_streamingvlm.log 2>&1 < /dev/null &      # (benched on gpu0 with /tmp/gpu0.lock)
kill $(cat logs/serve_streamingvlm.pid)
```
- Load takes about 30 s once the weights are in the page cache.

**What a request does.** It is stateless; one request runs at a time. It is `streaming_vlm/inference/inference.py:streaming_inference()` re-implemented line for line, with in-memory frames instead of a decord video reader.
1. `load_model_and_processor` → `convert_qwen2_5_to_streaming`, the authors' patched model.
2. `process_past_kv` runs every chunk, with sink 512, text window 512, vision window 16 and text rounds 16. With ≤8 chunks nothing is evicted.
3. `StreamingArgs(pos_mode="shrink")` is used.
4. Sampling is the defaults of `inference.py`: T 0.9, repetition 1.05, 20 tokens per chunk.
5. Each tick frame becomes one 1-s chunk.
   - The frame is duplicated to the FPS=2 clip and resized with the authors' `_spatial_resize_video`: 1042×396 becomes 1036×392, about 518 tokens.
   - The time label is `Time=(t−7+k).0-(t−6+k).0s`, taken from the call's `CURRENT TIMESTAMP`.
6. Chunk 0 is the native first turn: `[system] + previous text ("") + user[Time, video, query]`. The **query is the benchmark user text, verbatim**.
   - The benchmark system prompt goes in the `system` turn.
   - The native demo puts context in the `previous text` slot, but the text sink/window would cut that slot to 512+512 tokens and truncate the 3.3k-token rulebook, so the rulebook is kept out of it.
7. The model comments natively on all 8 chunks.
8. **Reply text** = the commentary of the last 2 chunks, with the `...` markers stripped. Those are the frames this poll adds, since the benchmark polls every 2 ticks (`SVLM_NEW_CHUNKS`). If the model is silent (`" ..."`) on both, the text is `""`.
9. **JSON: shim-wrapped, not the model's own.** Same wrapper as Proact-VL: empty `cook_progress` lists, `current_subgoal.since = t`, `goal_change null`, `flag null`, `text` = the commentary.
10. Every response carries `x_streamingvlm.trace` (raw per-chunk commentary), also logged to `logs/requests.jsonl`.

The shim ignores `response_format`, `temperature` and `max_tokens`.

**Can it produce the JSON itself? No.** I checked two ways.
- *Streaming mode.* In 2 of 24 calls it started writing the JSON (`{"cook_progress": {"subgoals_completed": [{"subgoal": "prep…`), but spread over 20-token chunks and broken off. At t32 the returned text was `{'hypothesis': {'assemble(bun)', 'assemble(meat)`.
- *Plain mode.* `probe/plain_json_probe.py` ran the unpatched weights as ordinary Qwen2.5-VL chat, greedy, 1024 new tokens, full prompt + 8 images, on calls 0, 2, 6 and 18. Every call returned just `" ..."` (0.9–2.2 s). The fine-tune has replaced instruction-following with the commentary format. Raw output: `probe/plain_json_probe.json`.

## Bench (24 genuine calls, `--profile default --max-tokens 4000 --image-scale 1.0`, GPU 0), tag `streamingvlm_native`

| median | p90 | prompt tok | completion tok | schema ok | spoke |
|---|---|---|---|---|---|
| **4.16 s** | **4.74 s** | 8,510 | 33.5 (all 8 chunks) | 24/24 (wrapped) | 11/24 |

- The reference spoke on 15/24 calls.
- **Pace.** At about 1.2 s per tick, each reply lands 3–4 ticks late. The model cannot keep pace in the stateless form, which re-prefills about 8.5k tokens and runs 8 generate calls.
- About 0.5 s per chunk was measured. The paper's claim of 8 FPS on an H100 is for its native incremental session, which only adds one chunk per second.

## Sanity read (5 calls; reference = the round-4 API reply)
The model is a sports-style live commentator.
- On most chunks it stays silent, and whole calls come back empty.
- When it speaks, it mostly talks *about the prompt* ("You choose your subgoal plan based on how many frames you've seen so far").
- Sometimes it role-plays the cook ("Now the heat is on for the pan, so I'll start placing the raw steak onto the board to be chopped right now.").
- It occasionally lands a relevant instruction, but often on a history chunk that the shim does not return.

| call | reference reply | StreamingVLM reply (new chunks) | notable history-chunk commentary |
|---|---|---|---|
| easy_hard_1 t16 | "You just chopped the cheese instead of trashing it! Please trash this chopped cheese…" | "based on how many frames you've seen so far, so" | "You choose your subgoal plan…" |
| easy_medium_1 t32 | "That's a wrong order: cheese went on before the chopped tomato…" | "{'hypothesis': {'assemble(bun)', 'assemble(meat)" | broken JSON over chunks 0–4 |
| expert_hard_2 t128 | "Nice save… But the plate order is now broken…" | "correct order and not putting" | "No. I have no idea why that would count as the…" |
| expert_medium_3 t214 | "You're at the smoking fryer with the steak plate. Scoop from the fryer now…" | "" (silent) | chunk 4: **"Toss both them cooked potatoes onto the steak plate."** |
| hard_medium_4 t68 | "You've got the chopped potato ready—go to the fryer… If the fryer doesn't start… turn it on with empty hands" | "" (silent) | chunks 4–5: **"Right. If the fryer doesn't start, turn it on with empty hands."** (paraphrasing the rulebook) |

- **Detection: none.** No reply names a visible mistake.
- **Mapping choice to review.** Its two most useful lines were spoken on history chunks, because the model speaks 1–4 s after the query arrives at chunk 0. With `SVLM_NEW_CHUNKS=2` the shim drops them.
  - The alternatives are to put the query on the first new chunk or to return the whole 8-chunk commentary.
  - Returning everything would re-say old lines on the next poll. That is why the native (new-chunks-only) choice was kept.

## Proposed benchmark profile and flags
`default` works unchanged: the shim ignores sampling and `response_format`. Optional named entry:
```python
# StreamingVLM (served by local_models/streamingvlm/shim_server.py): the authors' streaming loop over the frames
# (1 chunk per frame, commentary per second); the shim WRAPS the newest chunks' commentary into the reply schema.
"streamingvlm-shim": dict(temperature=None, extra_body={}, prefill=None),
```
```
--assist-backend openai --assist-model streamingvlm --assist-base-url http://127.0.0.1:19000/v1 \
--assist-profile default --assist-max-tokens 4000 --assist-image-scale 1.0
```

## What a native session would need from the harness
StreamingVLM's native loop needs:
- one session per episode;
- the system prompt and task once, as the chunk-0 query plus `previous text`;
- then **one user turn per second of game time**: `Time=…` + a 2-frame chunk, pushed incrementally into the persistent patched KV cache (never re-prefilled);
- the ≤20-token commentary read after each chunk, with `" ..."`-only turns treated as silence;
- the cook's speech inserted as text in the chunk where it occurs. The code supports extra user text per chunk; eviction deletes it with the text round;
- nothing else: the sink / window / RoPE re-shift handles 600-tick episodes without growth.

The harness side is:
- a `stream` backend that pushes a chunk per tick and collects per-chunk text;
- a sentence assembler, because utterances span chunks;
- latency charged per chunk from arrival to the first non-marker token;
- a second-stage classifier if `flag.error` is wanted, because the model has no structured output.

## Files
- `shim_server.py`: the shim. `logs/requests.jsonl` holds the per-chunk traces.
- `$LM/pass3_setup/serve_streamingvlm.sh`: the launcher.
- `probe/plain_json_probe.{py,json}`: the plain-mode JSON check.
- `$LM/pass3_setup/results/streamingvlm_native.json`: the bench result.
- `$LM/pass3_setup/logs/serve_streamingvlm.log`: the server log.
- `src/`: the authors' code.
