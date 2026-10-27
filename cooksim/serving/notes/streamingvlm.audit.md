# StreamingVLM (`streamingvlm`, port 19000): audit against the official guidance (2026-09-19)

**Verdict.** The shim reproduces the authors' streaming loop exactly. That covers:
- attention (FA2);
- 1-s chunks at FPS 2;
- the authors' resize, which gives the same 1036×392 as their reader;
- `Time=a-b s` turns;
- the sink 512 + window 512 text cache and the 16-chunk vision window;
- contiguous-RoPE "shrink";
- T 0.9, repetition 1.05, 20 tokens per chunk.

**Deviations from the official loop.**
1. **Frame rate (by necessity).** Each chunk holds one tick frame twice, and a "second" is really a 1.2-s tick.
2. **Prompt shape (fixed).** The authors' loop has **no system turn**. Context goes in the `previous text` slot (`inference.py --previous_text`, a one-line video description), with a short query in the first user turn. We had put the 13k-char rulebook in a system turn.
   - The official shape is `SVLM_PROMPT=native`: previous text = a one-line description of the two views + the recipe line; query = the benchmark user text in the first turn; no rulebook.
   - Effect: validity stays 24/24 (wrapped) and latency is equal (4.18 s against 4.21 s).
   - The model speaks on more calls (17/24 against 11/24), and a few more utterances describe the kitchen.
   - It remains mostly off-task. **Adopted** under the audit rule, but the evidence of better quality is weak. See the sanity read.

**Chunk window.** Returning more than the 2 newest chunks does **not** follow the authors' intent better; keep 2. Moving the query onto the new chunks makes things worse.

Sources:
- **[readme]** `src/readme.md` (github.com/mit-han-lab/streaming-vlm).
- **[inf]** `streaming_vlm/inference/inference.py`: constants l.34–45, `load_model_and_processor` l.70–85, `streaming_inference` l.180–520, `__main__` l.525–561.
- **[resize]** `livecc_utils/video_process_patch.py` `_spatial_resize_video` and `get_smart_resized_video_reader`.
- **[eval]** `scripts/eval_LiveSports3k-cc.sh`.
- **HF card.** `mit-han-lab/StreamingVLM` has no usage section beyond the repo link.

## Item table

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint | `mit-han-lab/StreamingVLM` rev 7ac1cef, bf16 | same (the `--model_path` default) | [inf] `__main__` | matches |
| 2 | engine | transformers 4.52.4 + the authors' patched Qwen2.5-VL (`convert_qwen2_5_to_streaming`) | `scripts/env_infer.sh` installs transformers 4.51.3 last, but the patched attention imports `make_flex_block_causal_mask`, which exists only from 4.52. `infer_requirements.txt` pins 4.52.4. | `scripts/env_infer.sh`; `infer_requirements.txt` | matches `infer_requirements.txt` (the 4.51.3 pin does not import; see SETUP.md) |
| 3 | turn format | chunk 0: `[previous text] + user[Time, video, query]`; later chunks: `user[Time, video]`, `text = "\n" + text[SYSTEM_PROMPT_OFFSET:]`; `prev_ids` concatenation and `<|im_end|>` repair | identical | [inf] l.365–395, 440–455 | matches |
| 3b | **system / context** | before: the benchmark rulebook as a **system** turn, `previous text` empty | **no system turn**; `previous_text` carries context (default: "THis is a video with title '…'. This is Highlight of Warriors…"); query default "Commentate on this match" | [inf] l.367 (`full_conversation_history = [{"role": "previous text"…}, {"role": "user"…}]`), `__main__` `--previous_text` | **fixed: `SVLM_PROMPT=native`.** No system turn. Previous text = "This is a video of a cooking game: top-down view of the kitchen (left) and the cook's first-person view (right). The cook was asked to make: …". The query is the benchmark user text. |
| 4 | fps / chunk | 1 chunk per tick frame, the frame duplicated to the FPS=2 clip, labelled `Time=(t−7+k)-(t−6+k)s` | `DEFAULT_CHUNK_DURATION = 1`; `_read_video_decord_plus(..., strict_fps=True, only_get_last_frame=int(chunk_duration*FPS))` with FPS 2, i.e. 2 distinct frames per second. README "Modify FPS" shows FPS can be changed via `QWENVL_FPS`. | [inf] l.35, 341; [readme] §Modify FPS | **deviates by necessity.** The harness supplies 1 frame per 1.2-s tick. |
| 4b | resolution | `_spatial_resize_video` on the 2-frame clip → 1036×392 (518 tokens per chunk) | the same call on each chunk; `MAX_PIXELS = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS/NFRAMES*2), 1.05·VIDEO_MIN_PIXELS)` = 602,112, which also gives 1036×392 for 1042×396 | [inf] l.44, 342; [resize] | matches (computed in the env for n = 2 and n = 32) |
| 5 | sampling | do_sample, T 0.9, repetition_penalty 1.05, `MAX_TOKEN_PER_DURATION` 20 per chunk | `DEFAULT_TEMPERATURE = 0.9`, `DEFAULT_REPETITION_PENALTY = 1.05`, `MAX_TOKEN_PER_DURATION = 20`; the eval script passes `--temperature 0.9 --repetition_penalty 1.05` | [inf] l.40–45; [eval] | matches |
| 6 | thinking | n/a | n/a | — | — |
| 7 | KV / cache policy | `process_past_kv` every chunk: text sink 512, text window 512, vision window 16 chunks, text rounds 16, `pos_mode="shrink"`; stateless per call (8 chunks, so nothing is evicted) | `DEFAULT_TEXT_SINK = 512`, `DEFAULT_TEXT_SLIDING_WINDOW = 512`, `DEFAULT_WINDOW_SIZE = 16`, `DEFAULT_TEXT_ROUND = 16`, `pos_mode="shrink"` | [inf] l.36–39, `__main__` | matches. The per-call re-prefill is the benchmark's stateless protocol. The paper's "8 FPS on H100" figure is for the persistent session. |
| 7b | attention | flash_attention_2 | `attn_implementation="flash_attention_2" if torch.cuda.is_available()` | [inf] l.74, 81 | matches |
| 8 | structured output | none: the shim wraps the commentary (`flag` null) | no JSON mode; the model is a live commentator | — | deviates by design (SETUP.md) |

## Bench (24 real calls, `--profile default --max-tokens 4000 --image-scale 1.0`, GPU 0; K = the 2 newest chunks returned)

| tag | prompt shape | query on | median | p90 | prompt tok | compl tok | schema ok | spoke |
|---|---|---|---|---|---|---|---|---|
| `svlm_audit_base` (**before**) | bench system turn | chunk 0 (native first turn) | 4.21 s | 4.78 s | 8,510 | 33.5 | 24/24 (wrapped) | 11/24 |
| `svlm_audit_qnew` | bench system turn | first new chunk | 4.14 s | 4.79 s | 8,510 | 33 | 24/24 | 11/24 |
| `svlm_audit_native` (**after**) | no system turn; previous text = view description + recipe | chunk 0 | 4.18 s | 4.71 s | 5,338 | 43 | 24/24 | **17/24** |
| `svlm_audit_native_qnew` | native | first new chunk | 3.89 s | 4.38 s | 5,338 | 37 | 24/24 | 7/24 |

Reference (round-4 API) spoke on 15/24.

**Sanity read.** Commentary over all 8 chunks, reference first. Kitchen-grounded lines are in bold.

| call | reference | before (bench) | after (native) |
|---|---|---|---|
| expert_hard_2 t128 | "Nice save—scooping it stopped the smoking. But the plate order is now broken…" | "No. I / have no idea / why that would / count as the / correct order…" | "**The last few moves were to put down the fried potato to stop the smoke, pick up a clean plate, and go to the fryer to scoop the potato.** … **Now, you can see the burn danger has passed, but you still need to replace the burne…**" |
| hard_hard_4 t188 | (silent) | "You might want to include subgoals with your output." | "**I put it into the fryer again. You need two potatoes cooked. The one in the fryer almost looks done.**" |
| master_hard_4 t408 | "Stop! You just picked up an onion…" | "The most recent frame from here." | "**I'm back with the rice, looking for the cooker to put them in. It's there, so I'll drop the rice into it…**" (role-plays the cook) |
| easy_hard_1 t16 | "You just chopped the cheese instead of trashing it!" | "You choose your subgoal plan based on how many frames you've seen so far" | "I know how hard that was because I tried this myself…" |
| easy_hard_2 t94 | "You have not moved yet — you're still at the cheese crate…" | "The last one is just finishing now." | "You can repeat these 8 frames as many times as necessary. … You don't need to submit all 30 seconds." |

- **Grounded utterances (hand count, all chunks).** About 7/24 calls before and about 10/24 after. It is still mostly meta-talk about the prompt ("A new line means a new frame.") or role-play of the cook.
- **Detection.** No variant names a visible mistake.
- **Bottom line.** This is a sports-commentary model facing an off-distribution task. The official prompt shape helps a little but does not change that.

## Chunk window: should the shim return more than the 2 newest chunks?

The same generations re-assembled with a wider window (`pass3_setup/audit_chunks.py`, no re-run):

| run | K = 2 (spoke / words per reply) | K = 4 | K = 8 |
|---|---|---|---|
| bench, query at chunk 0 | 11/24, 9.5 | 17/24, 11.6 | 20/24, 16.1 |
| bench, query on new chunk | 11/24, 8.5 | 21/24, 10.8 | 23/24, 16.2 |
| native, query at chunk 0 | 17/24, 8.4 | 21/24, 15.4 | 23/24, 23.8 |
| native, query on new chunk | 7/24, 4.4 | 21/24, 9.8 | 24/24, 18.8 |

**The authors' intended use.** One persistent session. Every second's commentary (or `" ..."` silence) is emitted once, at that second, and the query arrives once, in the first turn.

**What a wider window would do.** In our stateless replay, the 6 older chunks of each call cover seconds that the previous poll already returned (as its new chunks). Returning K > 2 re-emits those seconds: the text differs because it is regenerated, but it covers the same seconds.
- It raises "spoke" without raising the share of grounded content.
- The kitchen-word hits scale with the word count.

**What moving the query does.** Putting the query on the new chunks leaves the model only 2 s to react, and speech falls to 7/24. Its reply latency is 1–4 s, as SETUP.md found.

**Answer: keep K = 2 with the query at chunk 0.** That is the native first-turn placement and the once-per-second emission. If the orchestrator wants the model's reaction to the query itself, the right fix is a persistent session (SETUP.md §native session), not a wider window.

## Changes to apply
- **Serve script** (`pass3_setup/serve_streamingvlm.sh`, in place): `export SVLM_PROMPT=${SVLM_PROMPT:-native} SVLM_QUERY_AT=${SVLM_QUERY_AT:-first}`. `SVLM_PROMPT=bench` restores the system-turn rulebook. The pre-audit copy is `pass3_setup/logs/serve_streamingvlm.sh.pre_audit`.
- **Shim** (`streamingvlm/shim_server.py`, in place):
  - new `SVLM_PROMPT` (`bench` | `native`) and `SVLM_QUERY_AT` (`first` | `new`);
  - per-request overrides `svlm_prompt` / `svlm_query_at` in the request body, used only by the audit;
  - `cfg` is logged per request.
  - The pre-audit copy is `streamingvlm/logs/shim_server.py.pre_audit`.
- **Engine PROFILES**: none needed (`default`, or the SETUP.md `streamingvlm-shim` entry).
- **For the orchestrator.** With `native`, StreamingVLM gets the recipe line and the benchmark user text, not the rulebook. The same trade applies to Proact-VL's `ego4d` mode. Report it alongside the results.
