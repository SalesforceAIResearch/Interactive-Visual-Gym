# Reproducing the leaderboard

The numbers in [results/](../results/README.md) are produced by the commands below, nothing else. Expect
differences from run to run: the cook is an LLM, so episodes are not deterministic; that is why every seat is run
three times and reported as mean ± SD.

## 1. Setting

* suite `benchmark/cases_composite_v5.json`, kitchens `benchmark/grid.json`
* cook model `gemini-3.7-flash`, personas `baseline classic_novice classic_expert`
* measured latency charged to the clock; scheduler, pace and grader as `cooksim-episode` sets them
* judges: QA and proactive alerts `gemini-3.7-flash`; interaction-quality rubric `gemini-3.7-flash` and `gpt-5.4-mini`

Keys: `GEMINI_API_KEY`, `OPENAI_API_KEY` (and the keys of the assistants you run). The renderer must be up
(`cooksim-server --port 17773`).

## 2. One seat, three runs

```bash
for r in 1 2 3; do
  cooksim-benchmark --assistant gemini:gemini-3.8-flash --out runs/gemini --workers 4 --repeat $r
done
cooksim-score --run runs/gemini
```

`cooksim-roster` prints every seat of the leaderboard with its backend; `--assistant <tag>` uses those exact flags.
A run is resumable: rerunning the same command skips complete episodes. One seat-run is 150 episodes of 150–300
ticks; with an API assistant and 4 workers it takes 6–10 hours, dominated by the models' latency.

Local open-weight assistants need their server first ([ASSISTANTS.md](ASSISTANTS.md)):

```bash
bash serving/serve_qwen.sh &            # one GPU; waits until /v1/models answers
cooksim-benchmark --assistant qwen3.8-27b --out runs/qwen --workers 3 --repeat 1
```

## 3. Ablations

Fixed 2-tick latency (every seat, one run):

```bash
cooksim-benchmark --assistant gemini:gemini-3.8-flash --out runs/gemini_lat2 --latency 2
```

Cook-model ablation (90 cells, the assistant fixed to gemini-3.7-flash, four models in the cook seat):

```bash
python -m cooksim_bench.run_assist_bench --tag gpt-6-astra --sup-obs frames --paradigm hardsoft \
    --persona-v2 baseline classic_novice classic_expert --workers 3 \
    --cells-file benchmark/cook_ablation_cells.json --out-root runs/cookabl/frames --log-dir runs/cookabl/logs \
    --assist-backend gemini --assist-model gemini-3.7-flash \
    --cook-backend openai --cook-model gpt-6-astra --cook-base-url https://api.openai.com/v1 --cook-profile gpt5-cook
cooksim-score --run runs/cookabl
```

(`cooksim_bench/roster.py` lists the four cook entries with their flags under `COOK_ABLATION`.)

## 4. Scoring

`cooksim-score --run <run>` runs, in order:

| step | tool | writes |
|---|---|---|
| objective | `cooksim_bench.scoring.eval_v2_sweep` | `eval_v2.json`: outcome, in-time success, TT, CC, DT, detection |
| qa | `cooksim_bench.scoring.regrade_qa` | `qa_regrade.json` |
| proactive | `cooksim_bench.scoring.regrade_proactive_v2` | `proactive_v2.json` |
| rubric | `cooksim_bench.scoring.sq_v4_score --judge gemini,gpt` | `sq_v4/frames/<persona>/<tag>/<case>.json` |
| table | `cooksim_bench.scoring.sq_v4_three_runs` | `sq_v4/three_runs_five.json`, the seat table over runs 1–3 |

Every step resumes: a killed pass grades only what is missing. The leaderboard columns are read from `eval_v2.json`
(episode fields `v3.in_time_success`, `task_success`, `task_efficiency.TT/CC`, `assistance_quality.DT`, the
detection fields), `qa_regrade.json`, `proactive_v2.json` and `sq_v4/three_runs_five.json`.

## 5. Cost

Per seat-run: 150 episodes × (50–150 assistant calls) plus the cook's calls (≈ 1 per 2 ticks); grading adds one judge
call per question and request and two judge calls per assistant line for the rubric. Frame rendering is CPU-bound
(headless Chromium with software GL); 4 workers per 8 cores is a comfortable rate.
