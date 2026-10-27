# ScreenSim

**Phone-setting tasks in an executable interface, with simulated users, scheduled human events, and a visual assistant.**

A simulated user navigates phone settings while the assistant observes screenshots and provides guidance. The engine schedules mistaken taps and checks both the requested goal and preservation of unrelated settings.

Part of [Interactive Visual Gym](../README.md). See the shared [benchmark protocol](../docs/benchmark_protocol.md) for definitions and the three-environment evaluation.

## Requirements and installation

Run the commands below from `interactive_visual_gym/screensim/` in a dedicated Python environment.

Python 3.10+ and Chromium. No separate simulator server is required. The simulated user and hosted assistants require provider credentials; see [Getting started](docs/GETTING_STARTED.md).

```bash
python -m pip install -e ".[all,dev]"
python -m playwright install chromium
```

Detailed instructions: [Getting started](docs/GETTING_STARTED.md).

## Run one episode

```bash
export GEMINI_API_KEY=...
screensim-episode --task hearing_setup \
    --assistant gemini-3.8-flash --out runs/demo/composite_results_v2.json
```

The summary is `runs/demo/composite_results_v2.json`; full episode records are in `runs/demo/composite_results_v2.records.jsonl`. Use `--episode` and `--persona` to select a specific combination, and `--dry-run` to check the selection without model calls.

## Evaluate an assistant

```bash
screensim-benchmark --assistant gemini-3.8-flash --out runs/gemini/r1/composite_results_v2.json
screensim-score --results runs/gemini/r1
# Interaction-quality grading makes API calls to the configured judges:
screensim-grade --results runs/gemini/r1 --judge gemini,gpt
```

Objective scoring reads saved records without model calls. Interaction-quality grading requires the configured judge credentials. Follow [Evaluate your model](docs/EVALUATE_YOUR_MODEL.md) for integration and [Reproduce results](docs/REPRODUCE.md) for independent runs and aggregation.

## Create custom cases

Start by inspecting the available tasks and event schedules:

```bash
screensim-case --episodes
screensim-case --task hearing_setup
```

See [Generate cases](docs/GENERATE_CASES.md) for supported customization and validation.

## Documentation

| Guide | Contents |
|---|---|
| [Getting started](docs/GETTING_STARTED.md) | Installation, credentials, first episode, and troubleshooting |
| [Evaluate your model](docs/EVALUATE_YOUR_MODEL.md) | Assistant interface and evaluation workflow |
| [Benchmark](docs/BENCHMARK.md) | Tasks, cases, personas, events, and metrics |
| [Reproduce results](docs/REPRODUCE.md) | Run configuration, repeats, and scoring |
| [Generate cases](docs/GENERATE_CASES.md) | Create and validate custom evaluation cases |
| [Assistants](docs/ASSISTANTS.md) | Supported model configurations |
| [Interaction quality](docs/INTERACTION_QUALITY_GRADING.md) | Judge inputs, rubric, and aggregation |
| [Engine](docs/ENGINE.md) | State, actions, rendering, and engine API |

[Task library and training split](docs/TASKS.md)

## Development

From the repository root, run `python scripts/check_release.py --engine screensim --tests` using this environment's Python. See the shared [development guide](../docs/developing.md) for testing boundaries and compatibility checks.

## Citation and license

Use the project [citation](../CITATION.cff). Apache-2.0; see [LICENSE](LICENSE).
