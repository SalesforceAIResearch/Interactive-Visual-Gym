# Interactive Visual Gym

**Closed-loop evaluation and training of visual assistants with simulated users.**

Interactive Visual Gym pairs executable task environments with simulated users who plan, act, and converse with an assistant. The assistant observes the activity and provides guidance; only the user acts on the environment. Scheduled human events and user personas let researchers evaluate how assistance affects task execution and interaction quality.

[Project page](https://hellomuffin.github.io/ive-benchmark/redesign/) · [Getting started](docs/getting_started.md) · [Evaluate an assistant](docs/evaluating_assistants.md) · **[Submit via pull request](#submit-via-pull-request)** · [Benchmark protocol](docs/benchmark_protocol.md)

## Submit via pull request

**Submit your assistant to IVG-Bench by opening a pull request in this repository.**

1. Run the benchmark and package your trajectories with `ivg-submit`.
2. Upload the archive as a GitHub Release asset and generate a submission request containing its URL and checksum. Include a pinned inference-code repository and runnable instructions in your submission metadata.
3. Open a PR adding **`submissions/requests/<submission-id>.json`**—not the trajectory archive or inference code itself.

PR validation checks the submitted records automatically. Maintainers then grade and review the results before approving leaderboard publication.

**[Submission guide: commands, required files, and review process →](docs/submissions.md)**

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
git clone https://github.com/SalesforceAIResearch/Interactive-Visual-Gym.git
cd Interactive-Visual-Gym
```

| | Install and run one episode | Evaluate your model | Create cases |
|---|---|---|---|
| CookSim | [Quickstart](cooksim/docs/GETTING_STARTED.md) | [Evaluation](cooksim/docs/EVALUATE_YOUR_MODEL.md) | [Case generation](cooksim/docs/GENERATE_CASES.md) |
| VH-Home | [Quickstart](vhhome/docs/GETTING_STARTED.md) | [Evaluation](vhhome/docs/EVALUATE_YOUR_MODEL.md) | [Case generation](vhhome/docs/GENERATE_CASES.md) |
| ScreenSim | [Quickstart](screensim/docs/GETTING_STARTED.md) | [Evaluation](screensim/docs/EVALUATE_YOUR_MODEL.md) | [Case generation](screensim/docs/GENERATE_CASES.md) |

The workflow is the same across environments: **inspect cases → run an episode → inspect the record → score → run the benchmark**. Installation, action vocabularies, and record formats remain engine-specific.

## Evaluation timing and results

The main benchmark uses **fixed two-tick delay** (`--latency 2`) in CookSim and VH-Home. To account for model inference latency, run a separate evaluation with **`--latency measured`**. ScreenSim keeps its fixed zero-tick action–response setting. See the [complete commands](docs/evaluating_assistants.md#response-timing).

Published scores live on the [project leaderboard](https://hellomuffin.github.io/ive-benchmark/redesign/#leaderboard-section) and in the paper. This repository provides the engines, evaluation code, and submission tools without maintaining another copy of benchmark scores.

## Documentation

- [Getting started](docs/getting_started.md): requirements, installation, and your first run.
- [Evaluating assistants](docs/evaluating_assistants.md): model interfaces, run commands, and outputs.
- [Benchmark protocol](docs/benchmark_protocol.md): cases, personas, repeated runs, and configuration.
- [Metrics](docs/metrics.md): engine-derived scores and model-judged interaction quality.
- [Submitting results](docs/submissions.md): package trajectories, request official scoring, and provide inference code for audit.
- [Development](docs/developing.md): repository structure, tests, and compatibility checks.
- [Validation results](docs/validation.md): before/after tests and remaining release blockers.
- [Release notes](docs/release_notes.md): known limitations and configuration differences.

## Repository structure

```text
docs/                   Shared workflow and benchmark documentation
scripts/                Repository-wide validation
ivg_submission/         Data-only submissions, trusted grading handoff, and reviewed leaderboard export
submissions/            Small submission requests and maintainer-reviewed results (not raw trajectories)
cooksim/                Cooking engine, benchmark harness, assets, and tests
vhhome/                 Household engine, benchmark harness, simulator setup, and tests
screensim/              Phone engine, benchmark harness, assets, and tests
```

Each engine is independently installable, with its own simulator dependencies and model-serving configuration.

## Citation

See [CITATION.cff](CITATION.cff) for the project title and authors.

## Contributing and licenses

See [CONTRIBUTING.md](CONTRIBUTING.md). This repository is released under the [Apache-2.0 license](LICENSE.txt). CookSim and VH-Home retain their included MIT licenses; ScreenSim carries an [Apache-2.0 license](screensim/LICENSE). External simulators, model weights, and other third-party assets retain their own terms.
