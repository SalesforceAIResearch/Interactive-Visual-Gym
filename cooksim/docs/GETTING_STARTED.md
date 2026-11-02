# Getting started

This page takes you from an empty machine to a scored episode: install, keys, the renderer, one episode, the record it writes, the first scoring pass, and the problems people hit on the way. Once this works, [EVALUATE_YOUR_MODEL.md](EVALUATE_YOUR_MODEL.md) covers putting your own model in the assistant seat and running the full protocol.

## 1. Install

CookSim needs Python 3.10 or newer, a headless Chromium (for the frames the assistant sees) and, optionally, ffmpeg (for films of episodes).

### With conda (recommended)

```bash
conda env create -f environment.yml
conda activate cooksim
```

The environment pins Python 3.11, installs ffmpeg, and runs `pip install -e .[anthropic,video,dev]` (the package plus the Claude client, the video tools and pytest).

### With pip only

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"               # core + test tools: engine, harness, Gemini and OpenAI clients, Playwright
pip install -e ".[anthropic,video]"    # optional: the Claude backend, films
```

### The headless browser

Frames are rendered by a WebGL page in a headless Chromium that Playwright drives. Install the browser and, on Linux, its shared libraries:

```bash
playwright install chromium
playwright install-deps chromium       # apt packages Chromium needs; requires sudo
```

If you cannot use sudo, ask your administrator for the libraries `playwright install-deps chromium --dry-run` lists.

### Check the installation

```bash
pytest -q          # engine and scoring unit tests, no API calls, about 15 s
cooksim-roster     # prints the assistant roster; proves the console scripts are on PATH
```

Every console script can also be run as a module: `python -m cooksim_bench.cli.episode --help`, `python -m cooksim_bench.cli.benchmark --help`, `python -m cooksim_bench.cli.score --help`, `python -m cooksim_bench.cli.case --help`, `python -m cooksim_bench.roster --help`.

## 2. API keys

| Variable | Needed for |
|---|---|
| `GEMINI_API_KEY` | **always**: the simulated cook is a Gemini model (`gemini-3.7-flash`); also Gemini assistants and the Gemini judges |
| `OPENAI_API_KEY` | GPT and GPT-realtime assistants, the GPT rubric judge, the cook's synthesized voice for native streaming seats |
| `ANTHROPIC_API_KEY` (+ `ANTHROPIC_WORKSPACE_ID` for user-scoped keys) | optional: the Claude assistant and the optional Claude judge |
| `DASHSCOPE_API_KEY` | optional: the Alibaba Model Studio assistants in the roster |

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```

The commands stop with `missing API key(s): ...` when a key they need is absent.

## 3. The renderer

Frame observations are drawn by `web/headless_render.html`, served by the CookSim server and captured by a headless Chromium that the harness launches itself. Start the server in its own terminal and leave it running:

```bash
cooksim-server --port 17773
```

The harness looks for `http://127.0.0.1:17773/static/headless_render.html`. To use another host or port, start the server there and set `COOKSIM_PAGE` to the full page URL. Every episode command checks the page first and refuses to start if it does not answer.

The browser uses the GPU through Vulkan when a usable Vulkan driver is installed; otherwise it falls back to software rendering (SwiftShader). The pixels are the same; software rendering is slower. `COOKSIM_FORCE_SWIFTSHADER=1` forces the software path.

## 4. Your first episode

```bash
cooksim-episode --latency 2 --case b3_easy_nops_medium_map_2 --persona baseline \
    --assistant gemini:gemini-3.8-flash --out runs/demo
```

What the flags mean:

| Flag | Meaning |
|---|---|
| `--case` | a case from the shipped suite (`cooksim-case list` shows the cells; names read `b3_<tier>_nops_<map>_<n>`) |
| `--persona` | the cook persona: `baseline`, `classic_novice`, `classic_expert`, ... (`cooksim-roster --personas`); default `baseline` |
| `--assistant` | `<backend>:<model>` (`gemini`, `openai`, `anthropic`, `live`, `realtime`, `live_native`, `realtime_native`) or a roster tag from `cooksim-roster` |
| `--out` | the run root; the record goes to `<out>/frames/<persona>/<tag>/<case>.json` |
| `--base-url`, `--profile` | for the `openai` backend: the endpoint and the request profile |
| `--cases` | a case file of your own instead of the shipped suite |
| `--latency` | `measured` (default: the model's thinking time is charged to the game clock) or a fixed number of ticks per reply, e.g. `2` |
| `--obs` | `frames` (default) or `agent+++` (frames plus an exact text readout of the kitchen) |
| `--dump-states` | also write the per-tick kitchen state next to the record (`<case>.states.jsonl`), needed for films and exact grading evidence |
| `--tag` | the record's directory name (default: the model name or the roster tag) |

The command sets the benchmark environment for you (scheduler, pace, latency charging, cook model) and prints progress tick by tick: sub-cases waiting on their anchors, the takeover being injected, the assistant's lines, the release. It ends with one line such as

```
b3_easy_nops_medium_map_2 x pv2_baseline [hardsoft]: outcome=won failed_at=solved injected=True released=False
record: runs/demo/frames/baseline/gemini-3.8-flash/b3_easy_nops_medium_map_2.json
```

An easy case takes a few minutes; a master case can take ten or more, mostly waiting on model calls. The benchmark environment can be overridden by exporting the variables before the command: `COOKSIM_SCHED=v5`, `COOKSIM_PACE=tight`, `COOKSIM_ASSIST_LATENCY=measured|<ticks>`, `COOKSIM_GRADER=cook`, `GEMINI_MODEL=gemini-3.7-flash`, `COOKSIM_CASES` (case file), `COOKSIM_GRID` (kitchen maps), `COOKSIM_PAGE` (renderer URL). The defaults are the ones every published number was produced with.

## 5. Reading the record

The record is one JSON object. The keys you will look at first:

| Key | Content |
|---|---|
| `outcome` | `won`, `wrong_serve`, `burned` or `timeout` |
| `ticks` | episode length in ticks (1 tick = 1.2 s of game time) |
| `failed_at` | `solved`, or the stage where the episode was lost (`detection`, `recovery`) |
| `conversation` | every line, in order: `speaker` (`human` = the cook, `vlm` = the assistant), `t`, `text`, `round`; cook entries carry the cook's own JSON (`move`, `plan_update`, `assistant_error_call`), assistant entries carry the full reply (`vlm_json`) and the flag verdict |
| `actions` | every command the cook issued: `cmd`, `t_issue`, `t_done`, `status` (`done`, `rejected`, ...) |
| `injections` | one entry per sub-case: `family` (`error`, `qa_probe`, `qa_proactive`, `plan_change`), `err`, `injected`, `released`, `skipped`, the window (`injected_at`, `window`, `released_at`), the assistant's flags inside it, `detected_at`, `prevented`, the cook's own verdict on each mistake call |
| `flags` | every typed flag the assistant raised: `t`, `error`, `verdict` (`valid`, `early`, `late`, false-alarm classes), the spoken text (`utt`) |
| `false_positives` | the flags that named a mistake that was not happening |
| `assist_latency` | one entry per reply: `t_issue`, `t_arrive`, `latency_s`, `ticks` charged, `capped` |
| `assistant_backend` | `model`, `calls`, `latency_ms_median`, `prompt_tokens`, `completion_tokens`, `thinking_tokens` |
| `cook_backend` | the model behind the cook |
| `composite` | the case's sub-cases (`subs`) and how many were reached |
| `heat_events` | food smoking or burning, with ticks |
| `plate_seq`, `plate_order`, `prepped` | what went on the plate, in order and state; what was prepped when |
| `persona_v2`, `questionnaire_v2` | the cook persona and the questionnaire answers the assistant was shown |
| `prompts` | the exact system prompts of both seats for this episode |
| `engine_rev` | a hash of the engine and harness files, so records from different code are never mixed |
| `eval` | the objective per-episode scores (`criteria`: task success, prevented, false alarms, talk volume, ...) |

With `--dump-states`, `<case>.states.jsonl` beside the record holds the full kitchen state at every tick.

## 6. Your first scoring pass

```bash
cooksim-score --run runs/demo --only objective
```

`objective` grades outcome, in-time success, efficiency, prevention timing and detection F1 from the records alone, with no API calls, and writes `runs/demo/eval_v2.json`. The full command runs five steps, each resume-safe (re-running grades only what is missing):

| Step | What it grades | Needs | Writes |
|---|---|---|---|
| `objective` | outcome, in-time success, efficiency, prevention timing, detection F1 | nothing | `<run>/eval_v2.json` |
| `qa` | the cook's questions: is each answer correct (LLM judge, Gemini by default; `JUDGE_MODEL` overrides) | `GEMINI_API_KEY` | `<run>/qa_regrade.json` |
| `proactive` | standing requests ("tell me when X is ready"): was the alert given, and on time (LLM judge) | `GEMINI_API_KEY` | `<run>/proactive_v2.json` |
| `rubric` | interaction quality: every assistant line graded by independent judges | `GEMINI_API_KEY`, `OPENAI_API_KEY` | `<run>/sq_v4/frames/.../<case>.json` |
| `table` | the seat table over runs 1-3 | nothing | `<run>/sq_v4/three_runs_five.json` |

```bash
cooksim-score --run runs/demo                       # all five steps
cooksim-score --run runs/demo --only qa rubric      # a subset
cooksim-score --run runs/demo --judges all          # add the Claude judge to the rubric (reported setting: gemini,gpt)
```

The rubric is described in [INTERACTION_QUALITY_GRADING.md](INTERACTION_QUALITY_GRADING.md); `--workers` sets how many judge calls run at once (default 64).

## 7. Common problems

**`the renderer is not answering at http://127.0.0.1:17773/static/headless_render.html`.** Start `cooksim-server --port 17773` in another terminal, or point `COOKSIM_PAGE` at the page URL of a server you started elsewhere. Note that `cooksim-server` without `--port` listens on 8000, which is not where the harness looks.

**Chromium fails to launch with a missing shared library (`libnss3.so`, `libatk-1.0.so.0`, `libgbm.so.1`, ...).** The browser's system libraries are not installed. Run `playwright install-deps chromium`, or install the packages it lists with your package manager.

**`Executable doesn't exist`** from Playwright. Run `playwright install chromium` inside the same environment the commands run in.

**Rate limits (`429`, `RESOURCE_EXHAUSTED`).** Every backend retries with a back-off, and a reply that still fails is recorded as silence for that turn, so rate limits cost the assistant score. Lower `--workers` on `cooksim-benchmark` (3-4 for API models, 2-3 for a locally served model) and keep the scoring judges' `--workers` within your quota.

**`missing API key(s): GEMINI_API_KEY`.** Export the key in the shell that runs the command. The cook always needs `GEMINI_API_KEY`; the other keys depend on the assistant and the judges (section 2).

**The episode is slow.** Each assistant call takes real seconds, and a reply in flight is charged to the game clock, so a slow model makes a long episode. Software rendering adds to it; a machine with a working Vulkan driver renders about an order of magnitude faster.

**A killed benchmark run.** Re-run the same `cooksim-benchmark` command: complete records under `<out>/frames` are skipped, so the run resumes where it stopped.
