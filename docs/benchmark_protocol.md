# Benchmark protocol

IVG-Bench instantiates Interactive Visual Gym across cooking, household activities, and mobile-interface tasks.

## Terminology

| Term | Meaning |
|---|---|
| Task | The goal to be completed in an environment |
| Case | A task configuration with its scheduled human events |
| Persona | The simulated user's behavioral configuration |
| Episode | One execution of a case with a persona and assistant |
| Run | One independent evaluation of the selected case–persona combinations |
| Assistant configuration | Model, backend, observation interface, and inference settings |

Some native records use `cell` for a case–persona combination and `seat` for an assistant configuration. ScreenSim also uses `episode` for a designed case identifier. These legacy names remain readable; shared documentation uses the terms above.

## Suite composition

| Environment | Cases | Personas per case | Episodes per assistant per run |
|---|---:|---:|---:|
| CookSim | 50 | 3 | 150 |
| VH-Home | 25 | 3 | 75 |
| ScreenSim | 30 | 3 | 90 |
| Total | 105 | — | 315 |

Scheduled events are attached to conditions on task progress. Different assistant/user trajectories can reach different events. Preserve scheduled, triggered, and scored event counts when reporting results; do not interpret an unreached event as an observed response.

## Configuration is part of the evaluation

Match the following before combining results:

- Case definitions, persona assignments, and code revision.
- Assistant backend, observations, response format, and polling/session policy.
- Simulated-user model and prompts.
- Measured, fixed, or zero assistant-delay setting where supported.
- Completion budgets, reference rollouts, and event allowances.
- Judge models, rubric subset, aggregation, and treatment of missing judgments.

This checkout contains legacy alerting-persona fields and environment-specific grading histories; see [Release notes](release_notes.md). Do not infer a paper configuration solely from a command's default.

## Reproduce or customize

| Environment | Suite definition | Reproduce results | Create cases |
|---|---|---|---|
| CookSim | [Benchmark](../cooksim/docs/BENCHMARK.md) | [Reproduce](../cooksim/docs/REPRODUCE.md) | [Generate cases](../cooksim/docs/GENERATE_CASES.md) |
| VH-Home | [Benchmark](../vhhome/docs/BENCHMARK.md) | [Reproduce](../vhhome/docs/REPRODUCE.md) | [Generate cases](../vhhome/docs/GENERATE_CASES.md) |
| ScreenSim | [Benchmark](../screensim/docs/BENCHMARK.md) | [Reproduce](../screensim/docs/REPRODUCE.md) | [Generate cases](../screensim/docs/GENERATE_CASES.md) |

Custom tasks and altered event schedules are useful experiments, but should be reported separately from the fixed benchmark.
