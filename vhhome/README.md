# VH-Home

**Household tasks in VirtualHome, with simulated users, scheduled human events, and a visual assistant.**

A simulated user completes a household routine in a furnished apartment. The assistant follows the user's observations and provides guidance while the engine tracks object states, task subgoals, and scheduled errors.

Part of [Interactive Visual Gym](../README.md). See the shared [benchmark protocol](../docs/benchmark_protocol.md) for definitions and the three-environment evaluation.

## Requirements and installation

Run the commands below from `interactive_visual_gym/vhhome/` in a dedicated Python environment.

Python 3.10+, the VirtualHome Python client and Unity build, and Linux display/OpenGL dependencies. Start a simulator in another terminal with `bash simulator/launch_simulators.sh 1`. See [simulator setup](docs/SIMULATOR.md) for software rendering and headless systems.

```bash
python -m pip install -e ".[dev]"
bash simulator/install_virtualhome.sh --unity-archive /path/to/linux_exec.zip
```

Detailed instructions: [Getting started](docs/GETTING_STARTED.md).

## Run one episode

```bash
export GEMINI_API_KEY=...
vhhome-episode --latency 2 --case cx_tidy_missing_rep --persona baseline \
    --assistant gemini:gemini-3.8-flash --out runs/demo
```

The episode record is written under `runs/demo/gemini-3.8-flash/`, with the case and persona in its filename.

## Evaluate an assistant

```bash
vhhome-benchmark --latency 2 --assistant gemini:gemini-3.8-flash --out runs/gemini --engines 8080 --repeat 1
vhhome-score --run runs/gemini/r1 --only objective
# Interaction-quality grading makes API calls to the configured judges:
vhhome-score --run runs/gemini/r1 --only rubric --judges gemini,gpt
```

Objective scoring reads saved records without model calls. Interaction-quality grading requires the configured judge credentials. Follow [Evaluate your model](docs/EVALUATE_YOUR_MODEL.md) for integration and [Reproduce results](docs/REPRODUCE.md) for independent runs and aggregation.

## Create custom cases

Start by inspecting the available tasks and event schedules:

```bash
vhhome-case list
vhhome-case plan tidy_hard
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

[Engine and episode overview](docs/OVERVIEW.md) · [VirtualHome installation](docs/SIMULATOR.md) · [Project leaderboard](https://hellomuffin.github.io/ive-benchmark/redesign/#leaderboard-section)

## Development

From the repository root, run `python scripts/check_release.py --engine vhhome --tests` using this environment's Python. See the shared [development guide](../docs/developing.md) for testing boundaries and compatibility checks.

## Citation and license

Use the project [citation](../CITATION.cff). MIT for this package; see [LICENSE](LICENSE). VirtualHome retains its own license.
