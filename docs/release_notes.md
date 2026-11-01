# Release notes and compatibility

## Public interfaces

The three environments retain their separate packages, command names, benchmark cases, user prompts, model rosters, scoring formulas, and saved-record formats. Shared guides cover installation, evaluation, reproduction, and development with consistent navigation. Existing result files are not rewritten.

The release is intended to be installed from an editable checkout. Standalone wheel installation has not been validated for the external assets and simulator dependencies.

## Release fixes

### Submission workflow

The root `ivg-submit` package adds data-only packaging, exact suite validation, trusted grading staging, strict two-judge score collection, sampled-audit plans, and hash-bound approval/export. PR intake does not execute contributor inference code or call judges. See [submissions](submissions.md) for service status and [operations](submission_operations.md) for launch requirements.

The submission aggregator gates each judge's round before averaging the two judge scores. It does not reuse historical merged-item or product-of-means aggregation. Native scoring interfaces and existing result files are unchanged.

VH-Home now accepts `ASSIST_LATENCY=2` for a fixed two-tick delay, matching the current research runner. Its existing `0` (zero) and `1`/`measured` settings retain their behavior. CookSim's quality grader now accepts `OPENAI_API_KEY` before falling back to the legacy key file.

CookSim's quality grader retries cached entries that contain a score but no valid item verdict, preserves documented not-applicable rounds, and supports targeted judge repairs. The Salesforce release uses submission protocol `ivg-bench-submissions-v2` to pin this corrected grader; its episode membership and evaluation settings match v1. A regression check ensures all frozen source hashes match the shipped files.

### ScreenSim public workflows

Explicit case, persona, provider, model, and endpoint selections now reach the rollout. `--dry-run` validates a selection without model calls. The runner source is included in Git rather than excluded with generated run directories.

The user client accepts environment credentials with legacy key-file fallbacks. Failed episodes produce a nonzero command status after their error records are saved.

The one-episode example correctly reads JSONL output. Three-run scoring requires three distinct record directories.

### ScreenSim output and scoring paths

`--out` accepts either a directory or a JSON summary filename. Objective scoring and interaction-quality grading consume the directory containing its `*.records.jsonl` files.

### CookSim scoring and configuration

Explicit endpoint/profile overrides are honored. Automatic post-episode scoring invokes the installed scoring modules, and complete, newly written records can be scored immediately.

### VirtualHome installation and launch

Both upstream client layouts are recognized; fresh client installs are pinned. Caller configuration overrides are honored and OpenCV is declared as a dependency. The launcher checks readiness and stops only processes it owns; it never kills unrelated listeners or removes their display locks.

VH-Home requires the VirtualHome v2.3.0 Linux Unity build. The upstream archive URL is unavailable, so fresh simulator installation requires a separately supplied archive or accessible `VHHOME_UNITY_URL`. The installer validates the version and optionally its SHA-256 checksum. Client-only offline tests do not require Unity. See [simulator setup](../vhhome/docs/SIMULATOR.md).

### Python and CI

Streaming adapters retain Python 3.10 support through a conditional timeout dependency. VH-Home's score printer no longer uses Python-3.12-only f-string syntax. CI covers all three engines on Python 3.10 and 3.12, including syntax checks for every shipped Python module. See [validation](validation.md) for executed tests and their scope.

## Historical configurations

### Persona configuration

All three implementations still contain an alerting preference in their persona definitions. Removing it from release behavior would change prompts and potentially results. This cleanup does not remove it or describe the code as a four-axis implementation.

### Judge provenance

The grading entry points default to Gemini and GPT. Some VH-Home historical result documentation describes a mixed two-/three-judge panel. Preserve that distinction when reproducing old tables; a documentation edit cannot make those results a uniform two-judge evaluation.

### Citation and licensing

Project citation metadata is shared at the repository root. The Salesforce release uses the root [Apache-2.0 license](../LICENSE.txt). CookSim and VH-Home retain their included MIT licenses; ScreenSim uses Apache-2.0. Third-party dependencies retain their respective licenses.
