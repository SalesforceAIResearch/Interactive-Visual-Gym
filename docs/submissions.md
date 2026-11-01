# Submit to IVG-Bench

Run the benchmark with your assistant, submit the native trajectories, and provide the inference code used to produce them. Maintainers recompute scores with the benchmark graders and review the submission before publication. You do not need to host a permanent model endpoint.

## 1. Match the submission protocol

The current submission protocol is `ivg-bench-submissions-v2`:

| Setting | CookSim | VH-Home | ScreenSim |
|---|---|---|---|
| Case–persona episodes per run | 150 | 75 | 90 |
| Independent runs | 3 | 3 | 3 |
| Assistant delay | Fixed 2 ticks | Fixed 2 ticks | Interaction steps |
| Assistant interface | Pulled | Pulled | Pulled |
| Simulated-user model | Gemini 3.7 Flash | Gemini 3.7 Flash | Gemini 3.7 Flash |

Use the existing [assistant integration guides](evaluating_assistants.md). For this track, pass `--latency 2` to `cooksim-benchmark`, and export `ASSIST_LATENCY=2 GEMINI_MODEL=gemini-3.7-flash` before running `vhhome-benchmark`. The latter matches the latest `lat2_p37` research runs, not the older VH-Home default. Do not select a native streaming adapter. ScreenSim retains its step-based setting.

The exact membership and definition hashes are in [catalog.json](../ivg_submission/catalog.json). Version 2 incorporates the CookSim judge-resume fix; membership, user prompts, timing, and aggregation are unchanged from version 1. Repackage existing records with the current tools rather than editing an old manifest. This version freezes the current release definitions, including their existing persona prompts; it does not retrospectively change the paper's experiments. Custom tasks, different user models, zero/measured latency, and native streaming belong in separate experimental tracks.

## 2. Package your records

From the repository root:

```bash
python -m pip install -e .
ivg-submit init --out submission-config.json
```

Edit the generated configuration with your model name/revision, inference repository and full commit SHA, hardware, actual engine commits, and native record locations. Each engine's `assistant_model_id` must match the ID in its records (CookSim `assistant_backend.model`, VH-Home `assistant.model`, ScreenSim `contestant`). `sources` lists one source for each engine/run. CookSim can use the same run root with three different model tags; VH-Home uses its `r1`, `r2`, and `r3` roots. ScreenSim sources are individual `*.records.jsonl` files, not summary files. Multiple shards may be listed, but duplicate episodes are rejected rather than silently replaced.

```bash
ivg-submit prepare --config submission-config.json --out submission.zip
ivg-submit validate submission.zip
```

The archive contains a manifest, all **945 native episode records**, and available adjacent state dumps/videos. It excludes inference programs, environment files, logs, and cached score files. Check your records for credentials or private information before sharing them. Do not replace failed episodes with a more favorable rerun; recover infrastructure failures according to the maintainer's retry policy.

For a small local test, use `prepare --preview` and `validate --allow-preview`. A preview cannot be submitted to the leaderboard.

## 3. Open a submission request

Upload `submission.zip` as a public GitHub Release asset in your own repository. Keep bundles below 2 GiB for the hosted intake. If your evidence exceeds this limit, arrange a larger transfer with the maintainers instead of dropping necessary evidence.

```bash
ivg-submit request submission.zip \
  --artifact-url https://github.com/YOUR_ORG/YOUR_REPO/releases/download/v1/submission.zip \
  --out submissions/requests/my-model-v1.json
```

Use the same ID as `submission_id` in your configuration. Open a pull request adding only that small request file. CI downloads the artifact on a GitHub-hosted runner, checks its hash, validates suite coverage and record consistency, and returns a validation report. It does not run your inference code or make paid judge calls. A green validation check is not a leaderboard approval.

Include runnable inference instructions at the pinned inference-code revision: installation, model weights or API model ID, launch command, response format, and hardware requirements. Never include API credentials. Maintainers may request access separately if code cannot be public.

## 4. Scoring and verification

Maintainers stage validated data in a clean workspace and run the trusted engine graders. Every assistant round requires judgments from **Gemini 3.7 Flash and GPT-5.4-mini**. Within each judge, an applicable category passes only if all its applicable items pass; a factual-grounding failure gates that judge's round score to zero. The round score averages the two judges. Round scores are pooled within each run; the leaderboard reports the mean and population standard deviation across the three run scores. This explicit aggregation must not be mixed with historical merged-verdict or product-of-means exports.

The result retains scheduled, reached/triggered, and scored counts where applicable. QA includes proactive requests. An absent denominator is reported as unavailable, not zero. Missing/failed judges block score collection; an entirely not-applicable judgment is tracked separately. The original records and per-judge evidence remain available for review.

Two verification labels are supported:

- **Officially scored:** maintainers graded and reviewed the supplied records. This does not establish that the supplied trajectories are authentic.
- **Sample-audited:** maintainers additionally reran a sampled set using the submitted inference implementation and reviewed protocol/behavior consistency. Sampling does not certify every episode or eliminate benchmark overfitting.

Reruns are stochastic: the audit checks permitted observations, configuration, behavior, and score consistency, not byte-identical dialogue. Submitted inference code runs separately from grading credentials and publication access.

## Current service status

The repository includes local packaging, validation, grading/review commands, automatic PR validation, and a static leaderboard export. There is **no public upload server or continuously running paid grading service**. Grading and approval are maintainer-run steps; automatic publication to the project page requires connecting the reviewed feed to its deployment. The repository's visibility and access settings are unchanged.

See [maintainer operations](submission_operations.md) for deployment and review, and [validation results](submission_validation.md) for the checks performed so far.
