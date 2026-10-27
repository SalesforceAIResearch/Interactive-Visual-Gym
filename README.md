# Interactive Visual Gym

**Closed-loop evaluation and training of visual assistants with simulated users.**

Interactive Visual Gym pairs executable task environments with simulated users who plan, act, and converse with an assistant. The assistant observes the activity and provides guidance; only the user acts on the environment. Scheduled human events and user personas let researchers evaluate how assistance affects task execution and interaction quality.

[Project page](https://hellomuffin.github.io/ive-benchmark/redesign/) · [Getting started](docs/getting_started.md) · [Evaluate an assistant](docs/evaluating_assistants.md) · [Benchmark protocol](docs/benchmark_protocol.md)

## Environments

| Environment | Tasks | Simulation |
|---|---|---|
| [CookSim](cooksim/) | Prepare and serve recipes in an Overcooked-style kitchen | Kitchen engine with top-down and egocentric rendering |
| [VH-Home](vhhome/) | Complete household routines in furnished apartments | VirtualHome with egocentric observations and maps |
| [ScreenSim](screensim/) | Configure phone settings to satisfy user goals | Executable phone interface with screenshot rendering |

The IVG-Bench suite contains **105 cases**, evaluated with three personas per case: **150 cooking, 75 household, and 90 mobile-interface episodes per assistant per run**. A case specifies a task and scheduled events; a recorded episode is one execution with a particular user persona and assistant.

## Get started

Clone the repository, then follow the installation guide for your environment. Install from the checkout: each engine uses assets and benchmark definitions alongside its Python packages.

```bash
git clone https://github.com/hellomuffin/interactive_visual_gym.git
cd interactive_visual_gym
```

| | Install and run one episode | Evaluate your model | Create cases |
|---|---|---|---|
| CookSim | [Quickstart](cooksim/docs/GETTING_STARTED.md) | [Evaluation](cooksim/docs/EVALUATE_YOUR_MODEL.md) | [Case generation](cooksim/docs/GENERATE_CASES.md) |
| VH-Home | [Quickstart](vhhome/docs/GETTING_STARTED.md) | [Evaluation](vhhome/docs/EVALUATE_YOUR_MODEL.md) | [Case generation](vhhome/docs/GENERATE_CASES.md) |
| ScreenSim | [Quickstart](screensim/docs/GETTING_STARTED.md) | [Evaluation](screensim/docs/EVALUATE_YOUR_MODEL.md) | [Case generation](screensim/docs/GENERATE_CASES.md) |

The workflow is the same across environments: **inspect cases → run an episode → inspect the record → score → run the benchmark**. Installation, action vocabularies, and record formats remain engine-specific.

## Documentation

- [Getting started](docs/getting_started.md): requirements, installation, and your first run.
- [Evaluating assistants](docs/evaluating_assistants.md): model interfaces, run commands, and outputs.
- [Benchmark protocol](docs/benchmark_protocol.md): cases, personas, repeated runs, and configuration.
- [Metrics](docs/metrics.md): engine-derived scores and model-judged interaction quality.
- [Development](docs/developing.md): repository structure, tests, and compatibility checks.
- [Validation results](docs/validation.md): before/after tests and remaining release blockers.
- [Release notes](docs/release_notes.md): known limitations and configuration differences.

## Repository structure

```text
docs/                   Shared workflow and benchmark documentation
scripts/                Repository-wide validation
cooksim/                Cooking engine, benchmark harness, assets, and tests
vhhome/                 Household engine, benchmark harness, simulator setup, and tests
screensim/              Phone engine, benchmark harness, assets, and tests
```

Each engine is independently installable, with its own simulator dependencies and model-serving configuration.

## Citation

See [CITATION.cff](CITATION.cff) for the project title and authors.

## Contributing and licenses

See [CONTRIBUTING.md](CONTRIBUTING.md). The root, CookSim, and VH-Home code carry MIT licenses; ScreenSim carries an [Apache-2.0 license](screensim/LICENSE). External simulators, model weights, and other third-party assets retain their own terms.
