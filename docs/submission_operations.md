# Submission operations

The submission package does not change simulator action/state transitions or rewrite historical results. The only runner change is VH-Home's explicit fixed-delay option, ported from the current research runner; its default measured and zero-delay settings remain unchanged.

## Separation of responsibilities

```text
Contributor: native trajectories + pinned inference repository
    → request PR → secretless validation
    → maintainer-controlled grading → score/evidence review
    → optional isolated inference audit
    → approved result on protected main → static leaderboard export
```

Do not run contributor inference code on a grading worker, a persistent self-hosted CI runner, or a machine with repository write access. Validation reads data only, uses the base branch's tooling, and has no judge credentials. This separation follows [GitHub's workflow security guidance](https://docs.github.com/en/actions/reference/security/secure-use).

## Before public launch

1. Verify the protocol against an accepted baseline, including the fixed-delay track and strict two-judge aggregation. The tools do not import historical leaderboard scores automatically.
2. Make the repository/code and required simulator distribution available under their licenses. The unavailable upstream VirtualHome Unity archive remains a separate [installation constraint](release_notes.md).
3. Enable Actions and require review for changes to `submissions/reviewed/`, the protocol, grader code, and publication workflows. Do not let contributors approve their own score files. Repository settings are not changed by these scripts.
4. Provision a disposable grading environment with the three editable engine packages, native dependencies, FFmpeg, and the two judge credentials. Restrict network access to required provider endpoints; keep media processing updated. A single trusted CPU grading worker is enough to start; inference audit hardware depends on the submitted model.
5. Set a per-submission API budget, concurrency limit, retry policy and retention policy. `grade` requires explicit paid-call opt-in, but does not enforce a dollar budget at the provider. Do not expose it as an unauthenticated public endpoint.
6. Configure project-page deployment to consume the reviewed JSON feed. The included workflow exports a site artifact; it does not replace or deploy the existing project website.

## Maintainer commands

Download a request's exact artifact and stage it in a new private directory:

```bash
ivg-submit fetch submissions/requests/my-model-v1.json --out my-model-v1.zip
ivg-submit stage my-model-v1.zip --work grading/my-model-v1
```

Staging copies only allowlisted native data. It never imports supplied scores or inference code. Inspect `job.json` and `validation.json`; missing provenance or visual evidence needs resolution before official approval. Some older native formats do not stamp all settings, so metadata declarations alone are not verification.

Install the graders in the same interpreter as `ivg-submit`, from the trusted checkout:

```bash
python -m pip install -e . -e cooksim -e vhhome -e 'screensim[all]'
ivg-submit grade --work grading/my-model-v1 --allow-paid-judging
ivg-submit collect --work grading/my-model-v1
```

Export `GEMINI_API_KEY` and `OPENAI_API_KEY` securely; never put them in a submission or Git. Grade logs stay in the private workspace. Resume with the same `grade` command: native graders reuse per-judge results. Source and input fingerprints prevent reuse after a code/data change. A successful grader process is insufficient: `collect` checks every expected round and both judges. Failed QA judgments also block collection.

Scoring trusts the submitted native execution evidence, not its summary scores. CookSim objectives are re-evaluated from the record; VH-Home recalculates detection windows; ScreenSim uses native per-beat outcomes and goal evidence. These are not cryptographic engine attestations. Falsifying an entire trace cannot be ruled out by rescoring it; use the separate inference audit.

The result gives per-engine/per-run values, per-category quality, run SD, event counts and evidence hashes. No cross-environment combined rank is invented. A silent assistant has no interaction-quality score rather than a perfect score.

## Audit and approval

Choose the audit after the submission is fixed:

```bash
ivg-submit audit-plan my-model-v1.zip --per-engine 5 --out audit-plan.json
```

The default sample contains five distinct cases per engine with sampled run/persona combinations. This command only makes a plan. Execute the supplied inference implementation in a disposable, resource-limited sandbox with no judge keys, no publication credentials, no unrestricted network, and no host workspace mounts. Use the official harness and a separate limited inference-provider credential if needed.

An audit report contains the plan's `manifest_sha256`, `status: "passed"`, the complete original plan under `plan`, and a `cases` list covering exactly the selected sample. Each case includes `engine`, `run`, `case`, `persona`, `passed: true`, and `evidence_sha256`, the SHA-256 of its retained audit evidence. Keep the full evidence privately. Do not mark an unexecuted audit as passed.

After inspecting all results and provenance, copy `result.json` into `submissions/reviewed/my-model-v1/result.json`, then bind approval to those exact bytes:

```bash
ivg-submit approve submissions/reviewed/my-model-v1/result.json \
  --reviewer YOUR_GITHUB_USERNAME --notes 'Protocol, coverage, evidence and scores reviewed.' \
  --out submissions/reviewed/my-model-v1/approval.json
```

Add `--audit-report audit-report.json` only for a completed audit. Approval files are a maintainer-controlled Git review record, not digital signatures. Protect their directory and review changes; a locally forged approval has no authority unless a maintainer merges it.

```bash
ivg-submit leaderboard --reviewed submissions/reviewed --out leaderboard/feed.json
ivg-submit protocol > leaderboard/protocol.json
python -m http.server 8080 --directory leaderboard
```

The export rejects modified results and incomplete runs. Merge approved results through review. The export workflow then generates a fresh site artifact. Do not publish private grading logs, API credentials, or participant data. This workflow concerns simulated benchmark episodes, not human-study records.

## Changing the benchmark

Create a new protocol/catalog version when membership, prompts, user models, delay policy, judges, or score aggregation changes. Keep old leaderboards identifiable by protocol. Rebuild the catalog with `PYTHONPATH=screensim python scripts/build_submission_catalog.py`, inspect the diff, and update its version before accepting new results. The generator is not an automatic migration of old submissions.
