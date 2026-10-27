# Contributing

Thank you for contributing to Interactive Visual Gym. Start with the [development guide](docs/developing.md).

## Before changing code

Identify the affected environment and reproduce the issue with a small example. Keep engine behavior, benchmark definitions, user prompts, and scoring changes separate from documentation or packaging changes.

A change to a prompt, persona, event trigger, latency setting, or aggregation rule can change benchmark results. Describe it explicitly; do not update historical results as part of an unrelated cleanup.

## What to include

- **Bug reports:** environment, commit, command, relevant package versions, expected behavior, and the smallest useful log or episode record. Remove credentials and private endpoints before sharing.
- **Assistant integrations:** backend implementation, roster entry where appropriate, and a test of the request/response contract. Follow the environment's model-integration guide.
- **New cases:** task description, event schedule, persona configuration, and successful validation with that engine's case checks.
- **Results:** code revision, benchmark cases, assistant and user-model configuration, latency setting, run IDs, judge configuration, and source records.

## Validation

Run the shared checks from the repository root after installing the engines you are testing:

```bash
python scripts/check_release.py --docs-only
python scripts/check_release.py --engine cooksim --tests
python scripts/check_release.py --engine vhhome --tests
python scripts/check_release.py --engine screensim --tests
```

VH-Home's offline tests also require the VirtualHome Python client; they do not require a running Unity instance. The shared runner executes its assertion-accumulating test suite in an isolated copy so generated ledgers do not overwrite benchmark files.

These checks make no paid model calls. Passing them does not replace a live rollout test when changing the assistant interface or simulator integration.

Engine-specific guidance: [CookSim backend interface](cooksim/docs/ADD_A_BACKEND.md), [VH-Home](vhhome/CONTRIBUTING.md), [ScreenSim](screensim/CONTRIBUTING.md).
