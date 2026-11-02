# Evaluating assistants

The evaluated model acts as the **visual assistant**. It receives the environment's prescribed observations and conversation context, and returns guidance and structured fields. It does not execute the user's actions.

## Connect a model

Each engine supports configured hosted-model backends and a roster of named assistant configurations. The exact reply schema, supported backend, and streaming behavior are engine-specific.

| Environment | Integration guide | List configured models |
|---|---|---|
| CookSim | [Evaluate your model](../cooksim/docs/EVALUATE_YOUR_MODEL.md) | `cooksim-roster` |
| VH-Home | [Evaluate your model](../vhhome/docs/EVALUATE_YOUR_MODEL.md) | `vhhome-roster` |
| ScreenSim | [Evaluate your model](../screensim/docs/EVALUATE_YOUR_MODEL.md) | `screensim-roster` |

Keep the assistant model separate from the simulated-user model and the evaluation judges. Changing one does not imply changing the others.

## Run and score

The public command families share a naming convention:

| Operation | CookSim | VH-Home | ScreenSim |
|---|---|---|---|
| Inspect cases | `cooksim-case list` | `vhhome-case list` | `screensim-case --episodes` |
| Run one episode | `cooksim-episode` | `vhhome-episode` | `screensim-episode` |
| Run a benchmark | `cooksim-benchmark` | `vhhome-benchmark` | `screensim-benchmark` |
| Objective scoring | `cooksim-score --only objective` | `vhhome-score --only objective` | `screensim-score` |
| Interaction quality | `cooksim-score --only rubric` | `vhhome-score --only rubric` | `screensim-grade` |

The table lists command families, not complete invocations. Use the linked guides for required arguments. Existing engine commands and record formats are retained for compatibility.

## Response timing

**The main benchmark ignores measured inference latency by using a fixed two-tick response delay in CookSim and VH-Home. It does not use zero delay.** The measured-latency comparison instead converts inference time into simulation ticks. These are separate rollouts because response timing changes subsequent user actions.

| Setting | CookSim | VH-Home | ScreenSim |
|---|---|---|---|
| Main: fixed response timing | `--latency 2` | `--latency 2` | Existing fixed zero-tick, action–response setting |
| Measured-latency comparison | `--latency measured` | `--latency measured` | Reuse the same ScreenSim evaluation |
| Tick duration / delay cap | 1.2 seconds / 30 ticks | 1.2 seconds / 15 ticks | Not applicable |

For measured latency, the delay is `ceil(inference_seconds / 1.2)`, capped as above. In VH-Home, use `ASSIST_LATENCY_DIV=1` to charge the observed time without hardware normalization. Fixed timing always delivers after two simulation ticks regardless of inference duration; API calls still take wall-clock time to complete.

Run the following from the corresponding engine directory after setup. Each loop evaluates three independent runs; keep separate output roots for the two settings.

**CookSim — fixed two ticks:**

```bash
for r in 1 2 3; do
  cooksim-benchmark --assistant gemini:gemini-3.8-flash --latency 2 \
      --repeat "$r" --out runs/flash_fixed2 --workers 4
done
```

**CookSim — measured latency:**

```bash
for r in 1 2 3; do
  cooksim-benchmark --assistant gemini:gemini-3.8-flash --latency measured \
      --repeat "$r" --out runs/flash_measured --workers 4
done
```

**VH-Home — fixed two ticks:**

```bash
for r in 1 2 3; do
  vhhome-benchmark --assistant gemini-3.8-flash --latency 2 \
      --repeat "$r" --out runs/flash_fixed2 --engines 8080
done
```

**VH-Home — measured latency:**

```bash
for r in 1 2 3; do
  ASSIST_LATENCY_DIV=1 vhhome-benchmark --assistant gemini-3.8-flash --latency measured \
      --repeat "$r" --out runs/flash_measured --engines 8080
done
```

The same `--latency` flags work with `cooksim-episode` and `vhhome-episode`. For compatibility, omitting the flag retains the exported timing variable (`COOKSIM_ASSIST_LATENCY` / `ASSIST_LATENCY`), or measured latency when unset. Always pass the flag when reproducing a benchmark setting. `--latency 0` is a custom instantaneous setting, not the main benchmark. Use polled streaming backends for benchmark comparisons; native streaming is a separate configuration.

[Leaderboard submissions](submissions.md) currently accept the fixed-two-tick track. Keep measured-latency rollouts separate; do not label them as fixed timing or mix their records in the same submission.

## Output locations

| Environment | Records | Scoring input |
|---|---|---|
| CookSim | `<out>/frames/<persona>/<assistant>/<case>.json` | Run root supplied to `--run` |
| VH-Home | `<out>/<assistant>/<assistant>__<case>__<persona>.json` | One or more run roots supplied to `--run` |
| ScreenSim | Summary JSON at `--out`; full records in the accompanying `*.records.jsonl` | Directory containing record files, supplied to `--results` |

For ScreenSim, use an explicit JSON filename for the runner's `--out`, such as `runs/demo/composite_results_v2.json`. Pass the enclosing `runs/demo` directory to the scorer and grader.

## Repeated runs and reporting

Keep independent runs in distinct output locations. Reusing a completed run's directory resumes it rather than producing an independent replicate. Follow each engine's reproduction guide for its repeat mechanism.

Record the code revision, case selection, personas, assistant settings, simulated-user model, latency policy, and judge panel. Preserve native records and per-judge outputs; a summary score alone is not enough to reproduce an evaluation.

See [Metrics](metrics.md) for the distinction between deterministic rescoring and model-based grading.
