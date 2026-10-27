# Official-guidance audit of every assistant we serve — shared checklist (2026-09-18)

Goal: each model runs the way its authors say gives its best quality AND efficiency. For every model,
find the OFFICIAL guidance — model card, the authors' repo README / inference scripts, the model's page in
the vLLM recipes (docs.vllm.ai/projects/recipes) or the serving engine the authors name — and compare our
setup to it item by item. Cite the source (URL + the exact line/section) for every item. "We think" is not
guidance.

Everything in local_models/CONSTRAINTS_2026-09-18.md still applies (isolation, GPU locks, ports, no engine
edits, never print keys). One addition, because it already happened once: **never let an install touch
/opt/conda.** This shell sets UV_SYSTEM_PYTHON=1; install only with `<env>/bin/pip …` or
`uv pip install --python <env>/bin/python …`, and check `INSTALLER`/import paths afterwards.

## What our harness sends (design against this)
One stateless call every 2 ticks (1 tick ≈ 1.2 s): a ~2–3k-token system prompt, a user turn with the
model's previous JSON + recent conversation + engine feedback, and 8 frames (each a side-by-side 1042×396
JPEG: top-down kitchen + first-person view), oldest → newest. It expects ONE JSON object back
(tools/assistant_reply_schema.json). Latency is charged to the game clock, so efficiency is quality here.
Current per-model setup: local_models/pass3_setup/serve_<name>.sh, local_models/pass3_setup/SETUP_REPORT.md,
local_models/<name>/SETUP.md, cook-bench-engine/cooksim_bench/controller_eval.py PROFILES, and the numbers in
local_models/pass3_setup/results/.

## Checklist, per model
1. **Checkpoint & precision** — the officially recommended checkpoint/revision; dtype; is our quantized
   build official, and does the card say what it costs? (A100 = sm_80: no FP8.)
2. **Engine & version** — the engine/version the authors support; required flags (trust-remote-code,
   chat template, reasoning parser, mm-processor kwargs, limit-mm-per-prompt, max-model-len).
3. **Chat template** — render our exact request through our server and through the official template /
   processor and diff them. System-message handling, generation prompt, special tokens.
4. **Visual input** — official min/max pixels or resolution, how many images/frames, video vs images, fps
   and timestamps. Are our 8 frames being resized/sliced the way the model was trained to see them?
5. **Sampling** — the official temperature / top_p / top_k / min_p / repetition or presence penalty for the
   mode we run (thinking vs non-thinking, instruct vs chat).
6. **Thinking / reasoning** — official way to switch it on/off, recommended budget, how reasoning must be
   parsed so the JSON is read from the answer only.
7. **Efficiency** — official speed features (MTP / EAGLE / DFlash drafters, prefix caching, CUDA graphs,
   attention backend, chunked prefill, max-num-seqs); anything the card says to turn on or off.
8. **Structured output** — does the official stack support guided JSON with this model (and with its
   reasoning parser)? Any known interaction?

## Deliverable
For each model a section in local_models/<name>/AUDIT.md: the item table (ours · official · source ·
verdict: matches / deviates-on-purpose (why) / fix). For every fix, re-run the 24-call bench before and
after (bench.py, same tag scheme with suffix _audit) and adopt it only if schema validity does not drop and
latency does not get worse — or if the official setting improves detection on the sanity read and you can
show it. Leave the adopted setup in the serve script (local_models/pass3_setup/) and write the exact engine
change (PROFILES entry / flags) in the report; the orchestrator applies engine changes. Free the GPU at
the end. If you run out of time, leave background jobs running with an auto-stop, and say exactly what is
pending.
