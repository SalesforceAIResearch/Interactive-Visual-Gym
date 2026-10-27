# Metrics

Interactive Visual Gym records task execution and conversation so assistance can be evaluated in context. Two distinct mechanisms are used: engine-derived measurements and model-judged interaction quality.

## Engine-derived measurements

Task success is checked against the environment's goal conditions. Completion budgets, error detection windows, prevention criteria, and false-alarm handling are specified by each engine. ScreenSim additionally checks that unrelated settings are preserved.

These measurements can be recomputed from saved records without calling an assistant. Keep their numerators and denominators alongside the reported rates, especially when scheduled events are not reached.

## Interaction quality

A judge evaluates assistant responses using the available execution evidence, conversation context, and user preferences. The five reported dimensions are factual grounding, situational relevance, actionable guidance, user intent uptake, and guidance conciseness.

The public grading commands default to the Gemini and GPT judges. Existing per-judge judgments can be reused; generating missing judgments makes API calls. Historical result files may have different judge panels, rubric subsets, or aggregation rules. Do not relabel or merge them without checking their provenance.

Use the environment-specific item-level rules and aggregation implementation when reproducing its reported results.

## Questions and proactive requests

Where an environment provides scheduled questions or requests, their correctness and timing are evaluated using its recorded execution history and grading procedure. These metrics are not interchangeable with interaction quality. Do not assign a QA score to an episode with no scored QA items.

## Detailed definitions and commands

| Environment | Metrics and denominators | Rubric and aggregation |
|---|---|---|
| CookSim | [Benchmark metrics](../cooksim/docs/BENCHMARK.md) | [Interaction quality](../cooksim/docs/INTERACTION_QUALITY_GRADING.md) |
| VH-Home | [Benchmark metrics](../vhhome/docs/BENCHMARK.md) | [Interaction quality](../vhhome/docs/INTERACTION_QUALITY_GRADING.md) |
| ScreenSim | [Benchmark metrics](../screensim/docs/BENCHMARK.md) | [Interaction quality](../screensim/docs/INTERACTION_QUALITY_GRADING.md) |

A repeat of a live model rollout is not expected to be byte-identical. Compatibility checks instead compare unchanged code and benchmark assets, deterministic engine tests, saved-record scoring, and command dispatch.
