# Development

## Repository organization

Each environment is a separate installable package:

```text
<environment>/
  README.md             Short introduction and executable quickstart
  pyproject.toml        Dependencies and console entry points
  <engine_package>/     World state, actions, rendering, and user interaction
  <benchmark_package>/  Rollouts, model adapters, scoring, and CLI
  benchmark/            Shipped case/reference data
  docs/                 Detailed environment-specific guides
  examples/             Small runnable examples
  tests/                Offline regression tests
```

CookSim and VH-Home also have `results/` and model-serving helpers. VH-Home has a `simulator/` directory for VirtualHome installation. These differences reflect actual dependencies, not separate public workflows.

## Documentation contract

All engine READMEs follow the same order: overview, requirements and installation, first episode, evaluation, custom cases, documentation, and license/citation.

Every engine provides:

- `docs/GETTING_STARTED.md`
- `docs/EVALUATE_YOUR_MODEL.md`
- `docs/BENCHMARK.md`
- `docs/REPRODUCE.md`
- `docs/GENERATE_CASES.md`
- `docs/ASSISTANTS.md`
- `docs/INTERACTION_QUALITY_GRADING.md`
- `docs/ENGINE.md`

Keep shared explanations in the root guides and implementation details in engine guides. Preserve old document links with short forwarding pages when renaming a guide.

## Run validation

Install the selected package with its development dependencies. Run commands below from the repository root:

```bash
python scripts/check_release.py --docs-only
python scripts/check_release.py --engine cooksim --tests
python scripts/check_release.py --engine vhhome --tests
python scripts/check_release.py --engine screensim --tests
python -m unittest discover -s tests -v
```

Without `--engine`, the runner checks all three engines. It runs each test suite separately to avoid collisions between identically named test modules. VH-Home's standalone verifier accumulates failures rather than raising pytest assertions; the runner checks its actual exit status in a temporary copy.

The GitHub Actions workflow checks documentation and all three offline suites on Python 3.10 and 3.12. The VH-Home job installs the pinned Python client with `--client-only`; no Unity binary is needed for its offline tests. Browser and live-provider tests are separate.

## Compare before and after a change

```bash
python scripts/check_release.py --snapshot /tmp/ivg-before.json
# Make the intended changes.
python scripts/check_release.py --compare /tmp/ivg-before.json --tests
```

Snapshots record runtime-file hashes and safe CLI output. Comparison fails if runtime files or the captured command behavior change. Use the same installed dependencies for both runs. A deliberate runtime change requires separate review and regression tests, not updating the baseline merely to obtain a passing result.

## Testing boundaries

Offline checks cover case construction, engine transitions, scoring helpers, schemas, and CLI discovery. Browser smoke checks require installed Chromium. Live rollouts additionally require provider credentials and, for VH-Home, a running Unity instance. Local model evaluation requires its model server.

Do not run paid evaluations as part of documentation checks. Do not claim that command help or mocked dispatch proves an end-to-end rollout works.

## Results and contributions

Preserve saved records, model settings, and per-judge evidence. Benchmark results are not regenerated during a documentation cleanup. See [Contributing](../CONTRIBUTING.md) and [release notes](release_notes.md).
