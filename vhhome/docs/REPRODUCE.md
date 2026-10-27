# Reproducing the published table

The numbers in [results/README.md](../results/README.md) are produced by the commands below, nothing else. Expect
differences from run to run: the simulated person is a language model, so episodes are not deterministic; that is
why every seat is run three times and reported as the mean ± half the range between runs.

## 1. Setting

* cells `benchmark/cells.json`: 25 cases x 3 personas each = 75 cells, drawn once with seed 20260920, the same for
  every assistant; `benchmark/t_ref.json`: the measured clean-plan time per case, which sets the success cap;
* the person: the Gemini model that is the code default (`GEMINI_MODEL` unset, i.e. `gemini-3.1-pro-preview`),
  persona per cell, releasing a forced mistake on its own judgement of the assistant's claim (`VH_GRADER=cook`, the
  default);
* paradigm `combo` (takeover with an in-character prompt), the CLI default; wall budget 5400 s and tick cap 520
  per episode (`STEPIN_WALL`, `STEPIN_CAP`, set by the CLI);
* the assistant's measured reply latency charged to the clock (`ASSIST_LATENCY` unset);
* judges for the interaction-quality rubric: `gemini-3.7-flash` and `gpt-5.4-mini` (`--judges gemini,gpt`).

Keys: `GEMINI_API_KEY` (always), `OPENAI_API_KEY` (the gpt judge, the GPT seats, the person's voice for the native
seats), plus the keys of the assistants you run. Simulators must be up: `bash simulator/launch_simulators.sh N`
starts N instances on ports 8080 and up (HTTP 501 on a port means healthy); one episode per instance at a time.

## 2. One seat, three runs

```bash
for r in 1 2 3; do
  vhhome-benchmark --assistant gemini-3.8-flash --out runs/gemini --repeat $r --engines 8080,8081,8082,8083
done
vhhome-score --run runs/gemini/r1 runs/gemini/r2 runs/gemini/r3 --judges gemini,gpt
```

`vhhome-roster` prints every seat of the table with its backend; `--assistant <tag>` uses those exact settings
([ASSISTANTS.md](ASSISTANTS.md)). `--repeat k` puts run k under `<out>/r<k>/`; records are
`<out>/r<k>/<tag>/<tag>__<case>__<persona>.json`. A run is resumable: rerunning the same command skips the cells
whose record exists, and an episode that was killed (no log growth for 900 s, or over the 10800 s absolute cap) is
simply missing and is picked up by the rerun.

A locally served assistant needs its server first and must run on **one** simulator port, because its latency is
charged to the game clock and two episodes sharing the server would each be charged the other's compute:

```bash
export VHHOME_MODELS=$HOME
CUDA_VISIBLE_DEVICES=0 bash serving/serve.sh qwen3.8-27b &            # one GPU; wait for /v1/models
for r in 1 2 3; do
  vhhome-benchmark --assistant qwen3.8-27b --out runs/qwen --repeat $r --engines 8080
done
vhhome-score --run runs/qwen/r1 runs/qwen/r2 runs/qwen/r3 --judges gemini,gpt
```

The 19 seats of the published table are: `gpt-6-astra`, `gemini-3.1-pro-preview`, `gemini-3.8-flash`,
`gemini-3.8-live`, `gemini-3.8-live__native`, `gpt-realtime-2.1`, `gpt-realtime-2.1__native`, `qwen3.8-27b`,
`molmo2-8b`, `internvideo3-8b`, `glm-4.6v-flash`, `kimi-vl-thinking`, `museglimmer`, `minicpmo`,
`qwen3-omni-30b-a3b`, `mmduet2`, `proact-vl`, `streamingvlm`, `silent`. Several seats can share one `--out` root:
`vhhome-score` tables every `<tag>/` folder it finds under each run root.

## 3. Scoring

`vhhome-score --run <r1> [<r2> <r3>]` runs, in order, for each run root and then once over all of them:

| step | module | writes |
|---|---|---|
| `objective` | `vhhome_bench.scoring.results_table`, then `vhhome_bench.scoring.metrics_table` | `<run>/metrics_table.json`: outcome, in-time success S, efficiency EFF (TT, CC), detection DET (DT, F1, precision, recall), per seat; no API calls |
| `rubric` | `vhhome_bench.scoring.rubric --subset --judge <judges>`, then `vhhome_bench.scoring.rubric_table` | `<run>/sq_v4/<tag>/<episode>.json`, one per episode, with every judge's verdict per round; the arm table |
| `table` | `vhhome_bench.scoring.final_table` | `final_results.json` and `final_results.md` beside the first run root (or `--out DIR`): one row per seat, pooled over the runs |

`--only objective rubric table` picks steps; `--workers` is the number of parallel judge calls (32). The rubric
step resumes per judge: an episode already graded by a judge is not graded by it again, and `--judges all` adds the
third judge (`claude-haiku-5-5`; `pip install -e .[anthropic]`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_WORKSPACE_ID`
for a user-scoped key) to whatever is already graded. One grader per run root at a time (a lock file under
`<run>/sq_v4/`).

The pooled table is regenerated at any time from the run roots without regrading:

```bash
vhhome-score --run runs/gemini/r1 runs/gemini/r2 runs/gemini/r3 --only table
```

## 4. What varies between runs, and what the table reports

The person model is stochastic, so the same cell plays out differently in each run: where it walks, what it says,
how quickly it accepts a correct claim. The assistant's own sampling and its latency vary too. The three runs are
paired repeats of one fixed cell table (same cases, same personas), so the spread between them is the measurement's
own noise, and it is the point of running three: it says how much of a gap between two seats is real.

* `won`, `S`, `EFF`, `DET`, `DT`, `F1`: the mean over the runs, ± half the range between the runs (the largest
  minus the smallest run value, halved).
* `AQ`: the per-episode interaction-quality score pooled over all episodes of all runs (225 for a full seat), with
  a 2000-draw bootstrap 95% interval over episodes; the per-run means are shown beside it.
* Every record carries `engine_rev`, a stamp of the engine and harness code it was produced with. Compare only
  records with the same stamp; the published runs were produced on one stamp, and `vhhome_bench/check.py` prints
  the stamps a run root contains.
* The published AQ was graded by `gemini-3.7-flash` and `gpt-5.4-mini` on every round and by `claude-haiku-5-5` on
  most rounds (99%, 99% and 44% of the rounds of runs 1, 2 and 3); each round's score is the mean over the judges
  that answered it. A reproduction with `--judges gemini,gpt` grades every round with the same two judges and is
  comparable; `--judges all` reproduces the three-judge setting where the Claude key allows.

## 5. Run time and cost

An episode takes minutes, not seconds: typically 2 to 15 minutes of wall time, more for a long chore with a slow
assistant. Most of it is rendering (software OpenGL on the CPU; a simulator instance uses a few cores) and the
model calls (the person is asked about every tick, the assistant every 2 ticks). A seat is 75 episodes per run and
225 over the three runs. The native streaming seats run at wall-clock pace (at least 1.2 s per tick) and are the
slowest to produce.

API cost per episode: the person's calls (roughly one per tick, each with its recent frames and a map) plus the
assistant's (one per 2 ticks, 10 images each) over an episode of a few hundred ticks. Grading adds one judge call
per round per judge, where a round is one assistant line with its context; silent rounds are not graded, so a
terse assistant is cheap to grade and a talkative one is not.

Local assistants need one GPU each (an 80 GB card for the two largest, Qwen3.8-27B and Qwen3-Omni-30B) and run
one episode at a time, so a local seat's three runs are serial: budget days rather than hours per seat.

## 6. Running the protocol on a cluster

The published runs were orchestrated with the optional subpackage `vhhome_bench/cluster/` on a Beaker cluster
(`pip install -e .[cluster]` installs its `beaker-py` dependency). It is not needed to run an episode, a lane or
the scoring, and it is written for one site; treat it as a worked example rather than a turnkey tool.

* `plan.py` describes the protocol as phases: a pre-flight, then for each run a benchmark phase (every seat over
  the 75 cells) and a scoring phase; later runs wait for an explicit go-ahead.
* `orchestrate.py` is the daemon that walks the phases (`init`, `daemon`, `status`, `approve <phase>`, `render`),
  submitting jobs and keeping its state in one JSON file; every runner is resume-safe, so it can be killed and
  restarted at any point.
* `beaker_submit.py` submits one job per locally served seat (one GPU, one model server, one simulator instance
  started inside the job on its CPUs) and, as CPU-only jobs, the API seats; `fill_missing.py` submits jobs for the
  cells a run still lacks and nothing else; `verify_phase.py` checks that a phase's records are all present, carry
  the expected engine stamp and look right before the phase is marked done; `latency_ratio.py` measures the
  per-GPU generation-time ratios in `benchmark/gpu_latency_ratio.json`; `page.py` renders a progress page.
* Site settings come from environment variables, never from the code: `VHHOME_BEAKER_WORKSPACE`,
  `VHHOME_BEAKER_BUDGET`, `VHHOME_BEAKER_GPU_WORKSPACE`, `VHHOME_BEAKER_GPU_BUDGET`, `VHHOME_BEAKER_CLUSTER`,
  `VHHOME_BEAKER_IMAGE` (the container image the jobs run), `VHHOME_JOB_HOME` (the home directory inside a job)
  and `VHHOME_JOB_MOUNTS` (`name:path` pairs, comma-separated); the Beaker token comes from `BEAKER_TOKEN`.
  `VHHOME_PY` names the interpreter with the package and `beaker-py` installed.

Any other scheduler works the same way with the plain commands of section 2: one `vhhome-benchmark` lane per seat
and run, `--shard k/n` to split a lane over machines, and the same `--out` root for all of them so that
`vhhome-score` sees every seat.
