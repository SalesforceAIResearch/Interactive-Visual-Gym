# Release validation

Validated on 15 October 2026 against baseline commit `0276cd20e9cf0a3105b40d477fb77c86a8ee49a5` on Linux. The initial documentation cleanup was followed by fixes to installation, command forwarding, process management, and scoring entry points. Benchmark cases, prompts, engine transitions, and numerical scoring functions are preserved.

## Before/after checks

| Check | Baseline | Release checks |
|---|---|---|
| CookSim offline suite | 119 passed, 2 skipped | 123 passed, 2 skipped |
| ScreenSim offline suite | 206 passed, 3 skipped | 231 passed, 3 skipped |
| VH-Home engine checks | 591 passed | 591 passed |
| VH-Home benchmark case verifier | 25/25 admitted | 25/25 admitted |
| Public command help and listings | 23 passed | 23 passed |
| Release/installer regression tests | — | 19 passed |

The skipped tests depend on optional local artifacts or source corpora absent from the release checkout. VH-Home also reports 12 non-viable candidate timings in its broader engine audit; the fixed benchmark's 25 cases pass verification. Its tests ran in a temporary copy to avoid rewriting shipped ledgers.

Regression tests cover exact ScreenSim case/persona selection, unchanged full-benchmark ordering, provider/endpoint forwarding, environment credentials, resumability, three-run scoring, and nonzero failure status. CookSim tests cover endpoint overrides, automatic scoring, failure propagation, and immediate scoring of fresh records. Installer tests cover both VirtualHome client layouts, pinned fresh clones, versioned archives, idempotence, occupied ports, and owned-process cleanup. Documentation links are checked automatically.

All three packages were installed in fresh Python environments without system-site packages; all offline suites and 23 command checks pass on Python 3.10.19 and 3.12.12. A Git-index-only source export also passes on Python 3.12.12, excluding ignored local files from the test installation. This validates editable source installation, not standalone wheel installation. Python 3.10 checks additionally exercise timeout compatibility in all three adapters and the public VH-Home scoring command. CI is configured for both Python versions; hosted CI still needs to run after publication.

## Rendering and live integration

- **CookSim:** five HTTP routes return successfully; a WebSocket session initializes and returns a state update. In headless Chromium, starting a level produces advancing ticks and a canvas without JavaScript errors. The opening level-selection menu intentionally pauses the simulation.
- **ScreenSim:** headless Chromium renders a 412 × 892 phone observation; navigating from Settings to Wi-Fi changes the rendered image.
- **VH-Home:** a fresh pinned client checkout connects to the locally available v2.3.0 Unity build. An isolated instance passes HTTP readiness, scene reset, character placement, and rendered actions. Tests use a separate port/display and do not stop existing simulators.

Live provider smoke tests exercise Gemini user/assistant interaction in all three environments. ScreenSim completes an episode and produces objective scores plus eight rounds graded by both Gemini and GPT. CookSim's short live rollout produces objective scores, a QA judgment, two-judge interaction-quality scores, and a summary table. A separate scripted CookSim run completes 94 ticks and verifies the repaired automatic post-episode scoring path. The short CookSim and VH-Home live tests use explicit time/tick caps: they test integration, not benchmark performance or successful task completion.

VH-Home's capped rollout includes a triggered event, rendered navigation/manipulation, and two assistant responses. Objective scoring, two-judge grading, and table export all complete. A local HTTP test also verifies ScreenSim's custom-model routing, image payload, and response parsing without contacting an external provider.

Grader output is inspected for per-judge errors; successful process exit alone is not treated as proof that grading succeeded. These smoke results are kept separate from historical benchmark data.

## Scope and external prerequisites

The checks do not establish byte-identical stochastic model rollouts or validate every hosted endpoint, GPU model server, or streaming session. Numerical scoring functions are compared with the baseline separately from changes to their command entry points; old and new CookSim objective scorers also return identical full score dictionaries for two actual rollout records. The VirtualHome container definition is not build-tested here.

Fresh VirtualHome binary installation requires a supplied v2.3.0 archive: the old upstream download is unavailable. Archive validation is tested with fixtures; live rendering uses the installed binary. See [release notes](release_notes.md) for installation and compatibility details.

To repeat the checks, see [Development](developing.md).
