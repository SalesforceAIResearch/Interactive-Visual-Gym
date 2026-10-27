# Reproducing results

The benchmark comprises 30 designed cases, each evaluated with three personas: 90 episodes per assistant per run. Preserve the task definitions, persona assignment, simulated-user model, and assistant configuration.

## Run the benchmark

From `interactive_visual_gym/screensim/`, after completing [setup](GETTING_STARTED.md):

```bash
screensim-benchmark --assistant gemini-3.8-flash \
    --out runs/gemini/r1/composite_results_v2.json --workers 6
screensim-benchmark --assistant gemini-3.8-flash \
    --out runs/gemini/r2/composite_results_v2.json --workers 6
screensim-benchmark --assistant gemini-3.8-flash \
    --out runs/gemini/r3/composite_results_v2.json --workers 6
```

Reusing a completed run's directory resumes it; it does not create another independent run. The default simulated-user model is `gemini-3.7-flash`. Changing it or omitting the task's User Guide context changes the evaluated setting.

## Objective scores

The scorer reads directories containing full record JSONL files, not the summary JSON:

```bash
screensim-score --results runs/gemini/r1
screensim-score --results \
    runs/gemini/r1 runs/gemini/r2 runs/gemini/r3 \
    --out runs/gemini/objective_scores.json
screensim-score --three-runs --results \
    runs/gemini/r1 runs/gemini/r2 runs/gemini/r3 \
    --out runs/gemini/repeated_runs.json
```

The three-run command requires three distinct record directories and reports the existing across-run aggregation.

## Interaction quality

```bash
screensim-grade --results runs/gemini/r1 --judge gemini,gpt
screensim-grade --results runs/gemini/r2 --judge gemini,gpt
screensim-grade --results runs/gemini/r3 --judge gemini,gpt
```

Grading is resumable per judge. Keep the judge panel and rubric subset identical across compared runs. See [Interaction quality](INTERACTION_QUALITY_GRADING.md) for aggregation and cached judgments.

## Reproducibility

Saved engine trajectories support deterministic rescoring; new model rollouts and fresh judge calls can vary. Report the run IDs, number of episodes and scored events, judge panel, and code revision rather than assuming every rerun exactly reproduces an earlier table.

See [release notes](../../docs/release_notes.md) for compatibility and [benchmark protocol](../../docs/benchmark_protocol.md) for shared reporting terms.
