# MMDuet2 (roster row 18, replacing Streamo): served; silent on every benchmark call

Set up 2026-09-18 (UTC). Served name `mmduet2`, port 18900, benched on GPU 0, GPU freed afterwards.
Streamo's weights were never released (`../streamo/SETUP.md`). MMDuet2 replaces it in this roster slot, on the
user's approval.

## What it is
- Paper: *MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning*,
  arXiv 2512.06810.
- Code: github.com/yellow-binary-tree/MMDuet2 (commit 7bd91d5, 2026-01-14), cloned to `src/`.
- Weights: `wangyueqian/MMDuet2`, revision 3224235. **Ungated**; licence qwen-research. It is
  `Qwen2_5_VLForConditionalGeneration` (Qwen2.5-VL-3B-Instruct → SFT → multi-turn GRPO). The tensors are stored
  in bf16, 7.6 GB, and live in `HF_HOME=$ROOT/hf_cache/mmduet2`.
- **How it decides to speak.** The reply decision is plain text; there is no extra head and no threshold.
  - The conversation is a sequence of user turns. Each turn carries 2 frames, plus optional text that follows the
    frames.
  - After every turn the assistant writes either a reply or the literal string `NO REPLY`.
  - The system prompt teaches that rule: "… If the information in this segment of the video cannot answer the
    question, output "NO REPLY"."
  - During RL the reward traded answer quality and timeliness (PAUC) against repetition and replies outside the
    answer span.

## Environment
- Env: `$ROOT/envs/mmduet2` (uv venv, Python 3.10.19). It follows `src/demo/requirements.txt` exactly:
  - torch 2.4.0+cu121, torchvision 0.19.0
  - **transformers 4.49.0**, the version the checkpoint was saved with
  - flash-attn 2.6.3 (prebuilt wheel `cu123torch2.4cxx11abiFALSE-cp310`; pip torch 2.4 uses the pre-cxx11 ABI)
  - qwen-vl-utils 0.0.8 (with decord), six
  - Added for serving: fastapi 0.141, uvicorn 0.53, flask 3.1.3 (imported by the demo module),
    accelerate 1.15, numpy 1.26.4, pillow.
- **Fix: setuptools.** A uv venv ships without it. triton 3.0 imports it when flash-attn's rotary kernel loads, so
  the authors' model file failed to import until setuptools was installed.
- **Install hygiene.** Every install ran with `UV_SYSTEM_PYTHON` unset and
  `uv pip install --python $ROOT/envs/mmduet2/bin/python`. The build log shows only
  "Using Python 3.10.19 environment at: …/envs/mmduet2". torch, transformers and flash-attn import from the env,
  and `INSTALLER` is `uv`. Nothing was installed into /opt/conda.
  - The `transformers-4.57.3` in /opt/conda is dated 18:56, which is before this setup began (22:35). It is the
    other agent's install already noted in `../proact-vl/SETUP.md`.

## Serving
- Script: `$LM/pass3_setup/serve_mmduet2.sh`. The shim is `shim_server.py` in this directory.
```
cd $LM/pass3_setup
setsid nohup flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_mmduet2.sh' \
    > logs/serve_mmduet2.log 2>&1 < /dev/null &          # or gpu0 with /tmp/gpu0.lock
kill $(cat logs/serve_mmduet2.pid)                          # stop
```
- `wait_any_gpu.sh` in this directory does the same, but takes whichever GPU lock frees first. That is how the
  bench got GPU 0 while both GPUs were held.
- Startup takes about 15 s. It uses about 9 GB of GPU memory.

**The loop is the authors' own code, imported rather than copied.**
- `src/demo/api_server.py` provides `ProactiveInferenceAPIClient`, with `reset` / `add_text` / `add_image`.
- It runs on `src/proactive_eval/qwen2_5_vl`, the authors' Qwen2.5-VL class. Its `prepare_inputs_for_generation`
  sends only the new turn's pixels to the vision tower while the KV cache carries the history.
- The official defaults (system prompt, greedy decoding, max_new_tokens 512, 2 frames per turn, flash-attention-2,
  bf16) are read from the authors' argument dataclass, not retyped.

**What a request does.** Each request is stateless, and one runs at a time.
1. `reset()` sets the history to `[official system prompt]`.
2. The 8 frames (one per tick, oldest first) become **4 user turns of 2 frames each**. This matches the paper's
   "2 frames per user turn". At one frame per tick (≈1.2 s), it is close to the paper's recommended inference rate
   of one frame per second (§5.3, Table 7).
3. **Text placement follows the training format**: text comes after the frames in the same user turn.
   - The benchmark **system message** (the rulebook, i.e. the task) is the "question" on the **first** turn.
   - The benchmark **user text** (timestamp, previous JSON, conversation) goes on the **last** turn, the moment it
     is sent. This is the demo's `add_text` path.
   - The model generates after every turn, and its earlier outputs stay in the context, as in a live session.
4. **Frames** are pre-resized with qwen-vl-utils `smart_resize` to `max_pixels = 128·28·28`. That is "128 tokens
   per frame", the SFT/RL setting (paper §4.3; `train/scripts/train.sh` `MAX_PIXELS`).
   - 1042×396 becomes 504×168, which is **108 visual tokens per frame**. The prompt is about 5.1k tokens.
5. **Reply decision.** The last turn's output is returned; those are the 2 frames this poll adds.
   - An output of exactly `NO REPLY` becomes `text: ""`. That is `NO_REPLY_STR` in the RL reward and in the eval
     post-processing.
   - Any other text is spoken.
6. **JSON: shim-wrapped.**
   - MMDuet2 is trained to write plain-text replies. If its output parses as a JSON object with a string `text`,
     that object is used. This never happened with the adopted settings.
   - Otherwise the shim returns `{"cook_progress": {empty lists, since = t}, "goal_change": null, "text": <reply
     or "">, "flag": null}`.
   - `flag` is therefore always null. Detection can only be graded from `text`.
7. Every response carries `x_mmduet2.turns`, the raw output of all 4 turns. It is also logged to
   `logs/requests.jsonl`.

The shim accepts and ignores `response_format`, `temperature` and `max_tokens`. For probing only, a request may
carry `mmduet2_system` (`official`, `official_last` or `bench`), `mmduet2_max_pixels`, `mmduet2_sample`
(`[T, top_k]`) or `mmduet2_must_reply`. The benchmark never sends them.

Environment knobs in the serve script: `MMDUET2_SYSTEM` (official), `MMDUET2_MAX_PIXELS` (100352) and
`MMDUET2_FRAME_INTERVAL` (1.2 s; it only feeds the model's `time` bookkeeping).

## Bench (24 genuine calls, `--profile default --max-tokens 4000 --image-scale 1.0`, GPU 0)

| variant (tag) | median | p90 | prompt tok | completion tok | schema ok | spoke |
|---|---|---|---|---|---|---|
| **official, adopted** `mmduet2_official_t128` | **1.19 s** | **1.39 s** | 5,142 | 12 (4 × "NO REPLY") | 24/24 (wrapped) | **0/24** |
| same, re-run `mmduet2_official_t128_audit` | 1.08 s | 1.12 s | 5,142 | 12 | 24/24 | 0/24 |
| native pixels (518 tok/frame) `mmduet2_native_px_audit` | 1.77 s | 1.78 s | 8,422 | 12 | 24/24 | 0/24 |
| demo-frontend size (≤640 px side, ≈207 tok/frame) `mmduet2_demo640_audit` | 1.20 s | 1.24 s | 5,750 | 12 | 24/24 | 0/24 |
| rulebook moved to the last turn (with the user text) `mmduet2_textlast_audit` | 1.09 s | 1.12 s | 5,142 | 12 | 24/24 | 0/24 |
| sampling T 1.0 / top_k 40 (the code's optional path) `mmduet2_sample_audit` | 1.10 s | 1.13 s | 5,142 | 12 | 24/24 | 1/24 |
| "I must reply." suffix (eval-loader hook) `mmduet2_mustreply_probe` | 1.06 s | 1.09 s | 5,146 | 12 | 24/24 | 0/24 |
| benchmark system prompt in the system slot `mmduet2_benchsys_audit` | 5.57 s | 22.5 s | 5,087 | 117 | 24/24 | 24/24* |

- The reference (round-4 API replies) spoke on 15/24 calls.
- Schema validity is 24/24 by construction, because the shim builds or fills the JSON.
- \* In the benchmark-system-prompt variant, 16 of the 24 "spoken" replies are the literal `NO ACTION` or
  `NO ACTION REQUIRED`: the model's silence habit, reworded. 5 others are the model's own JSON written into
  a string, which degenerates into repeated `"assemble(potato)"` lists up to the 512-token cap.

**Pace.** The adopted setup takes about 1.1–1.2 s per call. The call covers 2 ticks (≈2.4 s), so **it keeps pace**
even stateless, re-prefilling about 5.1k tokens every call. It is the only shim-served model in the roster that
does.

**Reply behaviour.** Across all 7 official-prompt variants, the model chose `NO REPLY` on **727 of 728 turns**
(7 variants × 26 requests × 4 turns, counting bench warm-ups). It never speaks on this input.

**It is not a serving fault.** A smoke check used the same frames through the same shim with a short in-domain
question (`probe/smoke_short_question.txt`).
- Asked "What is the cook doing?", it answered on the turn it was asked, e.g. "The cook is preparing food, with a
  plate of cheese and a bowl of eggs on the counter, and a pot on the stove."
- Asked "Tell me whenever the cook makes a mistake.", it said `NO REPLY` on every turn.
- The model's training questions are short "what is happening" queries answered when the relevant content
  appears. A 13k-character supervisor rulebook that demands JSON and judgement is far outside that distribution,
  and its answer to it is silence.

## Sanity read (5 calls; reference = the round-4 API reply in calls.jsonl)

| call | reference reply | MMDuet2, adopted (official) | MMDuet2, benchmark system prompt (not adopted) |
|---|---|---|---|
| easy_hard_1 t16 (mis_prep) | "You just chopped the cheese instead of trashing it! Please trash this chopped cheese, then grab a tomato to chop." | "" (NO REPLY ×4) | "NO ACTION" |
| easy_medium_1 t32 (wrong_order) | "That's a wrong order: cheese went on before the chopped tomato… Dump the plate's contents at the trash…" | "" | "Just a quick guardrail: don't scoop cheese onto that plate yet. The tomato has to go on first, chopped, or you'll lock in the wrong order." (22 s) |
| easy_hard_2 t94 (forget_next_step) | "You have not moved yet — you're still at the cheese crate with the correct finished plate. Go to the serving pass now, then serve." | "" | "NO ACTION"; the sampled official variant said "Do not serve yet, as the order is incomplete…" (wrong: the plate was complete) |
| expert_medium_3 t214 | "You're at the smoking fryer with the steak plate. Scoop from the fryer now…" | "" | "NO ACTION" |
| hard_medium_4 t68 | "You've got the chopped potato ready—go to the fryer and put it in…" | "" | the model's JSON as a string, degenerating into repeated `assemble(potato)` (34.7 s) |

- **Detection: none** in the adopted setup. The model is silent on every call.
- In the non-official variant, the one error-adjacent line (t32) comes too late: the cheese is already on the
  plate. It is also largely paraphrased from the previous reply shown in the user text.

## Proposed benchmark profile and flags
The shim ignores sampling and `response_format`, so `default` works unchanged. For readability the orchestrator
may add:
```python
# MMDuet2 (Qwen2.5-VL-3B + proactive RL; served by local_models/mmduet2/shim_server.py): the authors' demo loop -- 2 frames
# per user turn, greedy, the model answers or says "NO REPLY" after every turn; rulebook = first-turn question, user
# text on the newest turn; 128 tokens/frame. The shim WRAPS the newest turn's text into the reply schema ("NO REPLY"
# -> text ""); flag is always null. Request sampling/response_format are ignored.
"mmduet2-shim": dict(temperature=None, extra_body={}, prefill=None),
```
```
--assist-backend openai --assist-model mmduet2 --assist-base-url http://127.0.0.1:18900/v1 \
--assist-profile default --assist-max-tokens 4000 --assist-image-scale 1.0
```
- Keep `--assist-image-scale 1.0`. The shim downsizes to the training budget itself, and 0.5 would only drop
  detail before that step.
- **Expectation for the orchestrator:** this arm will be a near-silent assistant, with no detections and no
  nags. That is a legitimate measurement of the model, not a setup defect. Report it that way.

## What a native (non-polled) session would need from the harness
MMDuet2's native mode is one growing conversation per episode.
- It opens with the official system prompt and the task as the first user question.
- Then comes **one user turn every 2 frames**. The authors recommend 1 frame per second at inference, so at one
  frame per tick that is a turn every 2 ticks. The cook's speech is appended as text to the turn in which it
  occurs.
- After every turn the model generates, and the output is either `NO REPLY` or an utterance. Both stay in the
  KV cache. There is no eviction in the released code, so a 600-tick episode grows to roughly
  300 turns × (2 × 108 + ~10) tokens ≈ 70k tokens.
  - That is within Qwen2.5-VL's 128k positions but past the RL context (90 frames).
  - Long episodes would need a sliding window or restarting sessions.

From the harness that means:
- a `stream` backend: `push_turn(frames[2], t, cook_speech)` → `{reply | None}`, called every 2 ticks on a
  persistent session instead of a stateless re-prefill;
- latency charged per turn: prefill of about 230 new tokens plus decoding, well under a second on an A100;
- `NO REPLY` read as silence;
- because the model has no structured output, a second-stage classifier (or a grader reading `text`) to obtain
  `flag.error`;
- a short in-domain question in place of the 13k-character rulebook, if the arm is meant to test the model's
  proactivity rather than its prompt-following. That would be a benchmark-design change, not a serving one.

## Files
- `shim_server.py`: the shim. `logs/requests.jsonl` holds the per-turn outputs of every call.
- `$LM/pass3_setup/serve_mmduet2.sh`: the launcher. `wait_any_gpu.sh`: an optional lock-waiting wrapper.
- `AUDIT.md`: the official-guidance checklist.
- `probe/render_template.py` and `probe/rendered_call0.txt`: the prompt as the official template renders it.
  `probe/run_variants.sh`: the variant runs. `probe/smoke_short_question.txt`: the loop smoke check.
- `data/`: the authors' eval input sample (`ego-frame_input_format.json`) and frame-extraction scripts, kept for
  reference.
- Results: `$LM/pass3_setup/results/mmduet2_*.json`. Server logs: `$LM/pass3_setup/logs/serve_mmduet2*.log`.
