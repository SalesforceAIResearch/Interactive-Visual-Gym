# Getting started

Choose an environment, install it from the checkout, and run a small episode before starting a full benchmark.

## Requirements

| | CookSim | VH-Home | ScreenSim |
|---|---|---|---|
| Python | 3.10+ | 3.10+ | 3.10+ |
| Rendering | Playwright Chromium and WebGL | VirtualHome Unity build and its Python client | Playwright Chromium |
| Additional setup | Start the CookSim server | Launch one Unity instance per concurrent episode | Browser starts through the renderer |
| GPU | Needed if you serve a model locally; renderer can use software rendering | Local model serving may need a GPU; software rendering is supported | Needed if you serve a model locally |
| API access | Simulated user, hosted assistant, and judges as selected | Simulated user, hosted assistant, and judges as selected | Simulated user, hosted assistant, and judges as selected |

Use a separate Python environment for each simulator when possible. Local model servers can have their own environments.

## Installation and first episode

Follow the complete guide for your chosen environment:

- [CookSim](../cooksim/docs/GETTING_STARTED.md)
- [VH-Home](../vhhome/docs/GETTING_STARTED.md)
- [ScreenSim](../screensim/docs/GETTING_STARTED.md)

These guides specify the working directory, required services, credentials, command, and output location. Do not assume that one engine's output layout or configuration variables apply to another.

## Check the installation without model calls

From the repository root, using the Python environment in which the selected package is installed:

```bash
python scripts/check_release.py --engine cooksim
python scripts/check_release.py --engine vhhome
python scripts/check_release.py --engine screensim
```

The checks validate documentation links, command help, and case/model listings. Add `--tests` to run the offline suites. They do not call assistants or judges.

## Continue

Once your first record is written, follow [Evaluating assistants](evaluating_assistants.md). Before comparing results with a published table, match the [benchmark configuration](benchmark_protocol.md) and review [release limitations](release_notes.md).
