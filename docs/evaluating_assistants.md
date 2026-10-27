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
