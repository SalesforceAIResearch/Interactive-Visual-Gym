# Running the benchmark

Published comparisons are maintained on the [project leaderboard](https://hellomuffin.github.io/ive-benchmark/redesign/#leaderboard-section). This repository contains the simulator and evaluation code, without a duplicate results snapshot.

## Setup

Run these commands from `vhhome/` after completing [Getting started](GETTING_STARTED.md). Start the simulator with `bash simulator/launch_simulators.sh 1`; the examples below use port 8080. Configure `GEMINI_API_KEY` for the simulated user and Gemini judge, `OPENAI_API_KEY` for the GPT judge, and the keys required by your assistant.

Use the 75 case–persona combinations in `benchmark/cells.json`, the same simulated-user model (`gemini-3.1-pro-preview`), and the same assistant configuration in both timing settings. Polled streaming interfaces belong to this comparison; native streaming is a separate configuration.

## Main setting: fixed two-tick delay

The main comparison does **not** charge measured inference latency. Every response is delivered after two simulation ticks; this is not zero delay.

```bash
for r in 1 2 3; do
  vhhome-benchmark --assistant gemini-3.8-flash --latency 2 \
      --out runs/flash_fixed2 --repeat "$r" --engines 8080
done
vhhome-score --run runs/flash_fixed2/r1 runs/flash_fixed2/r2 runs/flash_fixed2/r3 \
    --judges gemini,gpt
```

## Measured-latency comparison

Delivery delay is `ceil(inference_seconds / 1.2)`, capped at 15 ticks. `ASSIST_LATENCY_DIV=1` uses the observed inference time without hardware normalization. Run separate rollouts because delayed guidance changes subsequent user actions.

```bash
for r in 1 2 3; do
  ASSIST_LATENCY_DIV=1 vhhome-benchmark --assistant gemini-3.8-flash --latency measured \
      --out runs/flash_measured --repeat "$r" --engines 8080
done
vhhome-score --run runs/flash_measured/r1 runs/flash_measured/r2 runs/flash_measured/r3 \
    --judges gemini,gpt
```

The same flags work with `vhhome-episode`. Without `--latency`, the runner preserves `ASSIST_LATENCY` if exported and otherwise uses measured latency for compatibility. For direct module invocations, `ASSIST_LATENCY=2` selects fixed two ticks and `ASSIST_LATENCY=measured` selects measured latency. `ASSIST_LATENCY=0` / `--latency 0` is a custom instantaneous setting, not the main benchmark.

## Models and repeated runs

`vhhome-roster` lists configured models; [Evaluate your model](EVALUATE_YOUR_MODEL.md) describes additional backends. API assistants can use multiple simulator instances through `--engines 8080,8081,...`. Locally served models use one simulator instance and one episode at a time to avoid charging inference contention as model latency.

`--repeat k` stores records under `<out>/r<k>/<tag>/<tag>__<case>__<persona>.json`. Repeating a completed command resumes it. Keep the fixed and measured output roots separate. `--limit N` supports a small test; `--shard k/n` distributes a run across machines.

## Scoring and submissions

`vhhome-score` computes objective metrics, saves per-judge interaction-quality verdicts in `<run>/sq_v4/`, and writes local summary reports. Judging resumes per judge. Use `--judges gemini,gpt` consistently; additional judges change the aggregation.

The [submission pipeline](../../docs/submissions.md) collects official scores from the saved records and two-judge outputs, pooling rounds within each run and averaging the three run scores. The current submission track accepts **fixed two-tick timing**. Measured-delay runs are a separate experiment.

Saved trajectories can be rescored, but new model rollouts can differ. Keep run IDs, code revisions, case definitions, user models, timing policy, and judge outputs with each evaluation.
