# Submission workflow validation

Checks performed on 1 November 2026 in the Salesforce release checkout, without paid judge calls or inference audits:

| Check | Result |
|---|---|
| Root tests (submission, delay, catalog, release checks, and installation) | 60 passed |
| CookSim offline tests, from the engine directory | 125 passed, 2 skipped |
| ScreenSim offline tests, from the engine directory | 231 passed, 3 skipped |
| VH-Home offline release checker | 591 engine checks and 25/25 composite cases passed |
| Public engine commands | All 23 help/listing checks passed |
| Real native records | One episode from each engine packaged, validated, and staged; original files unchanged |
| Native objective rescoring | Completed on each of those three staged records |
| Browser | Empty state, result rows, engine selection, coverage expansion, HTML escaping, mobile overflow and JavaScript error checks passed |
| Catalog and grading source | Corrected stale CookSim grader hash; source fingerprint and frozen-definition regression check passed |
| Static checks | Python compilation, documentation links and whitespace checks passed |

The port preserves Salesforce's existing license and governance files, historical result files, benchmark cases, user prompts, and action/state transitions. Existing runtime changes are limited to the upstream CookSim judge fixes and VH-Home fixed-delay support. Protocol v2 updates the grader source pin without changing membership or evaluation settings.

The submission tests include the full 945-record membership, missing/duplicate records, changed identities, incompatible latency/user models, hash mismatches, unsafe ZIP paths, symlinks, undeclared files, malformed JSON, failed or missing judges, preview exclusion, audit sampling, and approval/result hash binding. The aggregate collector is exercised with a synthetic fixture; those scores are never published.

The VH-Home delay regression tests exercise both native and pulled delivery expressions: the new fixed-two-tick mode is independent of wall latency; zero and measured/capped behavior are preserved. Engine mechanics, case definitions, and existing native metric formulas are not changed.

Tests run from an engine's directory, as its existing test setup expects. Invoking all engines' tests from the monorepo root can shadow their editable packages with outer namespace directories; the grader launches from its isolated workspace. The submission fingerprint resolves the actual installed package initializer.

This validates the local tooling, not a public hosted service. A full paid two-judge run, a real inference-code audit, public artifact intake on GitHub Actions, and deployment to the live project page still require launch validation. Do not label the smoke-test records as officially scored or audited.
