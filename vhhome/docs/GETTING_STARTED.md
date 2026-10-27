# Getting started

Run the commands below from `interactive_visual_gym/vhhome/`.

This page takes you from an empty machine to a scored episode: install, keys, the simulator, one episode, the record it writes, the first scoring pass, the tests, and the problems people hit on the way. Once this works, [EVALUATE_YOUR_MODEL.md](EVALUATE_YOUR_MODEL.md) covers putting your own model in the assistant seat and running the full protocol.

## 1. Install

VH-Home needs Python 3.10 or newer, the VirtualHome v2.3.0 simulator (a Python client and a Linux Unity build), a virtual X display with software OpenGL for the simulator, and `ffmpeg` on the path if you want films of episodes.

### The Python package

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # engine, harness, the Gemini and OpenAI clients
pip install -e ".[anthropic]"          # optional: the Claude rubric judge
pip install -e ".[cluster]"            # optional: submitting the formal protocol to a cluster
pip install -e ".[dev]"                # optional: pytest
```

The package installs five console scripts: `vhhome-episode`, `vhhome-benchmark`, `vhhome-score`, `vhhome-case` and `vhhome-roster`. Each can also be run as a module: `python -m vhhome_bench.cli.episode --help`, `python -m vhhome_bench.cli.benchmark --help`, `python -m vhhome_bench.cli.score --help`, `python -m vhhome_bench.cli.case --help`, `python -m vhhome_bench.roster`.

### The simulator

```bash
bash simulator/install_virtualhome.sh --unity-archive /path/to/linux_exec.zip
```

Supply the VirtualHome **v2.3.0 Linux** archive. The upstream download is currently unavailable; the installer accepts a local archive or `VHHOME_UNITY_URL` and never substitutes a different simulator version. An existing installation is reused. See [simulator setup](SIMULATOR.md) for required system libraries.

The installer pins the Python client to a tested revision and recognizes both upstream client layouts automatically. `VHHOME_VIRTUALHOME` can point to an existing checkout. For offline tests only, run `bash simulator/install_virtualhome.sh --client-only`; no Unity archive is needed.

### Check the installation

```bash
vhhome-roster                   # prints the assistant roster; proves the console scripts are on PATH
python tests/test_engine.py     # the offline engine suite; no simulator, no API calls; ends with "0 failure(s)"
```

The engine suite imports the VirtualHome client, so run it after `install_virtualhome.sh`.

## 2. API keys

| Variable | Needed for |
|---|---|
| `GEMINI_API_KEY` | **always**: the simulated person is a Gemini model; also Gemini assistants and the gemini rubric judge |
| `OPENAI_API_KEY` | GPT and GPT-Realtime assistants, the gpt rubric judge, the person's synthesized voice for the native streaming seats |
| `ANTHROPIC_API_KEY` (+ `ANTHROPIC_WORKSPACE_ID` for a user-scoped key) | optional: the third rubric judge (`--judges all`) |
| `DASHSCOPE_API_KEY` | optional: the Alibaba Cloud assistants in the roster |

Keys are read from the environment first; what is missing there is read once from `~/.config/vhhome/keys.env`, a file of `KEY=value` lines (`VHHOME_KEYS_FILE` overrides the path). Nothing ever writes a key anywhere and no key is printed.

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```

Every command checks the keys it needs before it starts and stops with `missing GEMINI_API_KEY — export it or add it to ~/.config/vhhome/keys.env` when one is absent. The only commands that need no key are the offline ones (`vhhome-roster`, `vhhome-case`, the tests, `vhhome-score --only objective`).

## 3. The simulator

The person's apartment is rendered by a headless VirtualHome instance that answers HTTP on a port. Start one or more and leave them running:

```bash
bash simulator/launch_simulators.sh 2        # two instances, ports 8080 and 8081
```

The launcher prints `PORTS_READY 501 501` when both answer; HTTP 501 is expected for Unity's plain-GET health check. Instance `k` uses port `8080 + k` and display `:10 + k`; logs and process records are stored in `~/.cache/vhhome/instances/`. Use `--port`, `--display`, and `--state-dir` to isolate another installation. Occupied ports/displays are rejected, not killed. To stop these instances, run `bash simulator/launch_simulators.sh 2 --stop` with the same port and state-directory options. Run **one episode per instance at a time**. See [simulator setup](SIMULATOR.md) for details.

## 4. Your first episode

Start with the control assistant that never speaks; it needs only the person's key and shows you the whole machinery without an assistant model in the way:

```bash
vhhome-episode --case cx_tidy_missing_rep --persona baseline --assistant silent --out runs/demo
```

Then put a model in the seat:

```bash
vhhome-episode --case cx_tidy_missing_rep --persona baseline \
    --assistant gemini:gemini-3.8-flash --out runs/demo
```

`cx_tidy_missing_rep` is a case on the task `tidy_hard` ("Tidy up the house: switch off the lights on both table lamps, switch off the TV, and close the fridge"). Its two mistakes: the person switches off one bedroom lamp but forgets its twin in the other bedroom, and after the television it goes back to the fridge it already shut and opens it again. `vhhome-case show cx_tidy_missing_rep` prints the schedule and the plan it composes to.

What the flags mean:

| Flag | Meaning |
|---|---|
| `--case` | a case name (`vhhome-case list cases`), or `<task>__clean` for a run of that task with no mistakes |
| `--persona` | the person's persona (`vhhome-case list personas`); default `baseline` |
| `--assistant` | a roster tag from `vhhome-roster`, `silent`, `gemini:<model>`, `openai:<model>`, `live:<model>`, `realtime:<model>`, `live-native:<model>`, `realtime-native:<model>`; default `silent` |
| `--base-url`, `--profile` | for `openai:<model>`: the endpoint (any OpenAI-compatible server) and the request profile (`default`, `gpt5`, or a per-model profile) |
| `--image-scale`, `--max-tok` | downscale the frames sent to the assistant; its maximum output tokens per turn (default 4000) |
| `--port` | the simulator instance to use; default `8080` |
| `--out` | the run root; the record goes under `<out>/<tag>/`; default `runs/demo` |
| `--tag` | the record's directory name (default: the model name or the roster tag) |
| `--cases` | a case file written by `vhhome-case new`, in addition to the shipped cases |
| `--paradigm` | how a mistake is forced: `combo` (the benchmark setting: takeover plus an in-character prompt), `soft` (prompt only, no takeover), `hard` (silent takeover) |

The command sets the benchmark's episode settings for you and prints progress tick by tick: the world resetting the scene, the person's commands and the engine's acceptance or refusal of each, the assistant's lines and flags, the mistake being injected and released. It ends with a summary line of the form

```
stepin_combo_cx_tidy_missing_rep__baseline_p8080: outcome=won failed_at=... valid=... injected_at=... released=... compliant=... ticks=...
record: runs/demo/gemini-3.8-flash/gemini-3.8-flash__cx_tidy_missing_rep__baseline.json
```

Beside the record are `<same name>.log` (the full log) and `<same name>.mp4` (the first-person view beside the top-down map, tick by tick). An episode takes roughly 10 to 30 minutes: the person's model is called every turn, the assistant every 2 ticks, and every command is rendered by the simulator.

The episode settings can be overridden by exporting variables before the command: `GEMINI_MODEL` (the person's model; the default in the code is `gemini-3.1-pro-preview`, and every published number was produced with the default), `VH_GRADER=cook|engine` (whether the person's own judgement or the engine's flag verdict releases a forced mistake; default `cook`), `ASSIST_LATENCY=0` (an assistant whose replies are not charged to the clock, an ablation), `ASSIST_MAX_TICKS` (the latency cap, default 15), `STEPIN_WALL` (the wall-clock budget in seconds, 5400 in the benchmark), `STEPIN_CAP` (the tick cap, 520), `VH_NO_PROGRESS` (ticks without progress before the episode is `stuck`, 60), `IV_CAPTURE_DIR` (a directory into which every prompt and image both sides see is written). Keep the person's model fixed when comparing assistants: the person is part of the measurement.

## 5. Reading the record

The record is one JSON object. The keys you will look at first:

| Key | Content |
|---|---|
| `outcome` | `won`, `finished_wrong`, `stuck`, `timeout` or `silent_fail` |
| `ticks` | episode length in ticks (1 tick = 1 s of animation = 5 frames) |
| `engine_rev` | a hash of the engine and harness files, so records from different code are never mixed |
| `scenario`, `persona`, `episode`, `task` | the case, the persona, the task name, and the task instruction the person was given |
| `phases[]` | one entry per forced mistake: `err` (its class), `injected_at` (when the takeover began), `dev_live` (when the mistake physically happened), `armed_at` (when a correct claim was accepted), `released_at` and `released_reason` (`convinced`, `task_completed` or `block_exhausted`) |
| `phases_total`, `phases_caught`, `phases_prevented`, `phases_detected` | the per-episode counts the detection metric is built from |
| `flags[]` | every typed flag the assistant raised: `t`, `label` (the mistake name it gave), `verdict` (`valid`, `valid_alias`, `early_correct`, `early_alias`, `late`, `wrong_name`, `fp_preinjection`, `neutral_true`, ...), `phase` |
| `conversation[]` | every turn in order, `speaker` `human` (the person) or `vlm` (the assistant), with `t` and `text`; person entries carry the person's own `move` and `plan_update`, assistant entries the full JSON reply, and each turn an engine snapshot of the state |
| `action_log[]` | every command the person issued: the tick it was issued and done, `status` (`done`, `rejected`, `interrupted`) and the refusal reason |
| `assistant` | the assistant's statistics: calls, empty replies, retries, schema failures, tokens, latency |
| `assistant_latency[]` | one entry per reply: when it was called, the wall seconds it took, the ticks charged, whether the cap applied |
| `checkpoints` | the task's success checkpoints and whether each held at the end |
| `cook` | the person's own ledger of `assistant_error_call` judgements (none, correct, wrong) on the assistant's claims |
| `grader` | which grader released each mistake (`cook`, `engine`, or both) |

## 6. Your first scoring pass

```bash
vhhome-score --run runs/demo --only objective
```

`objective` grades outcome, in-time success, efficiency and detection from the records alone, with no API calls, and writes `runs/demo/metrics_table.json`. The full command runs three steps, each resume-safe:

| Step | What it grades | Needs | Writes |
|---|---|---|---|
| `objective` | task outcome, in-time success `S`, efficiency `EFF`, detection `DET` (prevention timing and flag F1) | nothing | `<run>/metrics_table.json` |
| `rubric` | interaction quality `AQ`: every conversation round graded by independent LLM judges | `GEMINI_API_KEY`, `OPENAI_API_KEY` | `<run>/sq_v4/<tag>/*.json` |
| `table` | the pooled table over all `--run` roots, mean ± half the range between runs, AQ with a bootstrap interval | nothing | `final_results.json` and `final_results.md` beside the first run root (or `--out`) |

```bash
vhhome-score --run runs/demo                        # all three steps
vhhome-score --run runs/demo --only rubric table    # a subset
vhhome-score --run runs/demo --judges all           # add the Claude judge to the rubric (reported setting: gemini,gpt)
vhhome-score --run runs/mine/r1 runs/mine/r2 runs/mine/r3   # three independent runs pooled into one table
```

Per-judge results are cached under `<run>/sq_v4/<tag>/`, so a rerun grades only what is missing. `--workers` sets how many judge calls run at once (default 32). The rubric is described in [INTERACTION_QUALITY_GRADING.md](INTERACTION_QUALITY_GRADING.md); the metrics in [BENCHMARK.md](BENCHMARK.md).

## 7. Running the tests

```bash
python tests/test_engine.py       # the offline engine suite: every mistake class at every legal anchor, no simulator, no API calls
python tests/verify_cases.py      # re-checks the 25 shipped cases against the composition rules; expects 25/25 admitted
```

`test_engine.py` ends with a line such as `=== N checks, 0 failure(s), M non-viable timing(s) ===`; `-v` shows every timing, not just the failures. `verify_cases.py` rewrites `benchmark/composite_ledger.json`. Both import the VirtualHome client, so `install_virtualhome.sh` must have run, but no simulator needs to be up.

## 8. Common problems

**The launcher reports a readiness failure.** Read `unity-<port>.log` and `xvfb-<port>.log` in its state directory. It waits up to three minutes per instance and stops the processes it started if startup fails. To restart one owned instance, run `ONLY_INDEX=k bash simulator/launch_simulators.sh N --stop`, then the same command without `--stop`.

**`GEMINI_API_KEY is not set: export it, or add a line to ~/.config/vhhome/keys.env`** (or `<tag>: missing GEMINI_API_KEY ...`). The simulated person is always a Gemini model, so every episode needs this key whatever the assistant. Export it in the shell that runs the command, or write it to the keys file (section 2).

**Two episodes on one port.** The simulator holds one scene; a second episode on the same port resets it under the first, and both records are garbage. Give each concurrent episode its own port (`--port` on `vhhome-episode`, `--engines 8080,8081,...` on `vhhome-benchmark`), and never more ports than running instances.

**`Unable to find a supported OpenGL core profile` in the Unity log.** The X display has no GLX visual. This reads like a GPU problem and is not: it happens when a private Mesa/Xvfb stack (`simulator/gl_stack_env.sh`, built by `build_gl_stack.sh`) is sourced on a machine that already has a working system stack, so that the private loader paths shadow `/usr` and a mismatched Xvfb comes up without GLX. Use one stack or the other: the launcher picks the system stack by itself when `/usr/bin/Xvfb` and the Mesa software rasterizer are present, so make sure your shell has not sourced `gl_stack_env.sh` on such a machine. See [SIMULATOR.md](SIMULATOR.md).

**Unity never opens its port on a machine with many cores.** Unity sizes its job-worker pool to the machine, and on a node with hundreds of CPUs it then never gets to open its HTTP port. The launcher passes `-job-worker-count` (`UNITY_JOB_WORKERS`, default 16) for this reason; if you start Unity by hand, pass it too.

**`[lane] KILLED <cell>: no output for 900s — the cell stays missing for a later fill job`.** `vhhome-benchmark` watches each episode's log and kills an episode whose log has not grown for `VH_EPISODE_STALL` seconds (default 900) or that has run longer than `VH_EPISODE_TIMEOUT` seconds (default 10800). A slow episode keeps writing and is left alone; a hung one is killed and its record is simply absent. Re-run the same `vhhome-benchmark` command: cells whose record exists are skipped, so the run resumes with the missing cell. If the same cell is killed again, read its `.log` beside where the record would be.

**Rate limits (`429`, `RESOURCE_EXHAUSTED`).** A reply that fails after retries is recorded as an empty turn, which costs the assistant. Use fewer simulator instances at once for API assistants, and keep the judges' `--workers` within your quota. A cell whose assistant calls were all transport failures (retries, no tokens ever returned) is moved to `<tag>/void/` instead of counting as data.

**A killed benchmark run.** Re-run the same `vhhome-benchmark` command: complete records under `<out>/[r<k>/]<tag>/` are skipped.
