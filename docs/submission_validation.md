# Submission workflow validation

## 2 November 2026: external-assistant compatibility and protocol v4

The targeted release fixes pass 75 repository tests, 125 CookSim tests (2 skipped),
231 ScreenSim tests (3 skipped), 591 VH-Home engine checks, all 25 household composite
cases, and 23 public CLI checks. Syntax, documentation links, source pins, and whitespace
checks also pass. Provider request construction and routing are tested offline.

Saved-record comparisons against the paper snapshot cover:

- VH-Home: all 15 models, three runs, both timing settings (6,750 episodes). Success,
  event-weighted recall, prevention, overall quality, and the five quality categories match.
- ScreenSim: all 15 models and three runs (4,050 episodes). Success, recall, overall
  quality, and the five categories match.
- CookSim: the actual quality-table CLI matches all three Gemini 4 runs in both timing
  settings, including all five categories. Objective-denominator tests cover episodes
  with no triggered error and unequal numbers of triggered errors.

The comparisons recompute quality from the Gemini and GPT verdicts. Only Boolean item
verdicts enter category scores, matching the paper exporter; historical string-valued
N/A entries do not count as passes. The submission collector retains its stricter
requirement for valid Boolean/null verdicts from both judges.

No simulation mechanics, task membership, user models, termination policies, or timing
caps were changed. No new benchmark rollouts or paid judging runs were needed.

## Earlier 2 November 2026: response timing and protocol v3

That revision passed 65 repository tests, 125 CookSim tests (2 skipped), 231 ScreenSim tests (3 skipped), 591 VH-Home engine checks, all 25 composite household cases, and 23 public CLI checks. Documentation links, Python syntax, frozen source fingerprints, and whitespace checks pass. These are offline checks; no paid inference or judge calls were made.

Timing regressions exercise fixed two-tick, measured/capped, and zero-delay behavior. CLI tests verify that explicit fixed/measured settings reach the episode runner and override exported timing settings. The suite membership, engine transitions, user prompts, and scoring sources are unchanged; the refreshed catalog fingerprints reflect the CLI edits. Protocol v3 corrects the VH-Home simulated-user model to Gemini 3.1 Pro, matching the paper.

## 1 November 2026: initial submission workflow

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
