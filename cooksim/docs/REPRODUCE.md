# Running the benchmark

Published comparisons are maintained on the [project leaderboard](https://hellomuffin.github.io/ive-benchmark/redesign/#leaderboard-section). This repository contains the simulation and evaluation code, without a duplicate results snapshot.

## Setup

Run these commands from `cooksim/` after completing [Getting started](GETTING_STARTED.md). Start the renderer with `cooksim-server --port 17773` and configure the provider keys needed by the assistant, simulated user, and judges.

Keep the suite (`benchmark/cases_composite_v5.json`), layouts (`benchmark/grid.json`), simulated-user model (`gemini-3.7-flash`), and personas (`baseline classic_novice classic_expert`) fixed across comparisons. Use polled interfaces for streaming assistants.

## Main setting: fixed two-tick delay

The main comparison does **not** charge measured inference latency. Every assistant response is delivered after two simulation ticks; this is not zero delay.

```bash
for r in 1 2 3; do
  cooksim-benchmark --assistant gemini:gemini-3.8-flash --latency 2 \
      --out runs/flash_fixed2 --workers 4 --repeat "$r" --dump-states
done
cooksim-score --run runs/flash_fixed2
```

## Measured-latency comparison

Use the same settings but replace fixed timing with measured inference time. Delivery delay is `ceil(inference_seconds / 1.2)`, capped at 30 ticks. Run separate rollouts because delayed guidance can change user behavior and task outcomes.

```bash
for r in 1 2 3; do
  cooksim-benchmark --assistant gemini:gemini-3.8-flash --latency measured \
      --out runs/flash_measured --workers 4 --repeat "$r" --dump-states
done
cooksim-score --run runs/flash_measured
```

Always pass `--latency` explicitly: without it, the CLI preserves `COOKSIM_ASSIST_LATENCY` if exported and otherwise defaults to measured latency for compatibility. `--latency 0` is a custom instantaneous setting, not the main benchmark. The same flags work with `cooksim-episode`.

## Models, output, and scoring

`cooksim-roster` lists configured assistants; `--assistant <backend>:<model>` connects another model. For a local model, start its server first ([Assistants](ASSISTANTS.md)) and use `--workers 1` so concurrent episodes do not inflate measured inference time.

Records are stored under `<out>/frames/<persona>/<tag>/<case>.json`; runs two and three use tags suffixed with `__r2` and `__r3`. Repeating a completed command resumes the run. Use different output roots for fixed and measured timing to avoid resuming the wrong condition.

`cooksim-score` computes objective metrics, grades questions and proactive requests, and grades interaction quality with Gemini and GPT. It preserves per-judge outputs in `<out>/sq_v4/`. See [Interaction quality](INTERACTION_QUALITY_GRADING.md) for grading and [Submissions](../../docs/submissions.md) for the official two-judge aggregation and fixed-timing submission track.

## Changing the simulated-user model

A separate user-model experiment can run the fixed 90-case selection with a different cook model. For example, with the visual assistant held at Gemini 3.7 Flash:

```bash
COOKSIM_SCHED=v5 COOKSIM_PACE=tight COOKSIM_GRADER=cook COOKSIM_ASSIST_LATENCY=2 python -m cooksim_bench.run_assist_bench \
    --tag gpt-6-astra --sup-obs frames --paradigm hardsoft \
    --persona-v2 baseline classic_novice classic_expert --workers 3 \
    --cells-file benchmark/cook_ablation_cells.json \
    --out-root runs/cookabl_fixed2/frames --log-dir runs/cookabl_fixed2/logs \
    --assist-backend gemini --assist-model gemini-3.7-flash \
    --cook-backend openai --cook-model gpt-6-astra \
    --cook-base-url https://api.openai.com/v1 --cook-profile gpt5-cook
```

Report this separately from the standard benchmark because the simulated-user model has changed.
