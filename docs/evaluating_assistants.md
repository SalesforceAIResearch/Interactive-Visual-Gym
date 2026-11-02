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

## Claude assistants

All three engines accept `--assistant anthropic:<model-id>`. Install the engine's
`anthropic` extra and set `ANTHROPIC_API_KEY` (and `ANTHROPIC_WORKSPACE_ID` if required
by your account). `GEMINI_API_KEY` is still required for the simulated user;
`OPENAI_API_KEY` is also needed for the GPT evaluation judge.

```bash
# From the repository root, install the optional provider adapters.
python -m pip install -e './cooksim[anthropic]' -e './vhhome[anthropic]' -e './screensim[all]'

# Start the renderers/simulator as described in each engine's setup guide first.
cooksim-benchmark --assistant anthropic:YOUR_MODEL_ID --latency 2 --out runs/claude_cook --workers 4
vhhome-benchmark --assistant anthropic:YOUR_MODEL_ID --latency 2 --out runs/claude_vh --engines 8080
screensim-benchmark --assistant anthropic:YOUR_MODEL_ID --out runs/claude_screen --workers 4
```

Use a model ID available to your Anthropic account. The adapter uses the
[Messages API image format](https://platform.claude.com/docs/en/build-with-claude/vision),
low effort, and the engine's existing JSON-response prompt. It does not give Claude
additional structured-decoding assistance or repair its answers. API billing or
model-access errors must be resolved before running a benchmark.

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

## Paper-comparable scoring

Interaction quality is computed from Gemini and GPT verdicts separately: each judge's
factual-grounding gate multiplies the mean of its applicable category scores, then the
two judges are averaged. Rounds are pooled within a run; the three runs receive equal
weight. Merged-item diagnostics and historical cached `AQ` values are not this score.

- CookSim: `cooksim-score --run RUN_ROOT` writes the five-category results to
  `sq_v4/three_runs_five.json` and objective results to `paper_scores.json`.
  The full sixteen-item rubric is presented to both judges; only the ten items
  belonging to the five reported categories enter the paper score.
- VH-Home: `vhhome-score --run RUN1 RUN2 RUN3` writes `final_results.json` and
  `final_results.md`. Rubric judges receive the transcript and engine state, without
  rendered frames, matching the paper. Recall uses the corrected detection windows.
- ScreenSim: after `screensim-grade --results RUN` for each run, use
  `python -m screensim_bench.scoring.aq_contract RUN1 RUN2 RUN3 --model TAG --json quality.json`.

Success includes every recorded episode. Recall divides detected errors by triggered
errors within each run, then averages runs. Incomplete or failed judge calls need to be
regraded before reporting a two-judge score. The submission grader enforces both judges.

The release reorganizes engine modules into installable packages. Research-tree and
release-tree engine stamps therefore differ even when simulation behavior is identical;
compare the pinned suite, prompts, renderer, and engine sources instead. Submission
bundles must use the release's own source pins and protocol version.
