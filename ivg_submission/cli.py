"""Public data packaging and trusted maintainer operations for IVG-Bench."""

import argparse
import json
import sys

from .common import Invalid, catalog, write_new


def parser():
    p = argparse.ArgumentParser(prog="ivg-submit", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "protocol",
        help="Print the versioned protocol and exact case/persona membership",
    )
    init = sub.add_parser("init", help="Write a submission configuration template")
    init.add_argument("--out", required=True)
    prepare = sub.add_parser(
        "prepare", help="Package existing native runs (no model calls)"
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument(
        "--preview",
        action="store_true",
        help="Allow an incomplete local test; never leaderboard-eligible",
    )
    validate = sub.add_parser(
        "validate", help="Validate a data-only ZIP without running code or calling APIs"
    )
    validate.add_argument("bundle")
    validate.add_argument("--allow-preview", action="store_true")
    request = sub.add_parser(
        "request", help="Create a small PR request referencing an uploaded bundle"
    )
    request.add_argument("bundle")
    request.add_argument("--artifact-url", required=True)
    request.add_argument("--out", required=True)
    fetch = sub.add_parser(
        "fetch", help="Download and validate a public GitHub Release artifact"
    )
    fetch.add_argument("request")
    fetch.add_argument("--out", required=True)
    stage = sub.add_parser(
        "stage", help="Maintainer: prepare a clean grading workspace"
    )
    stage.add_argument("bundle")
    stage.add_argument("--work", required=True)
    stage.add_argument("--allow-preview", action="store_true")
    grade = sub.add_parser(
        "grade", help="Maintainer: run/resume trusted native graders"
    )
    grade.add_argument("--work", required=True)
    grade.add_argument("--allow-paid-judging", action="store_true")
    collect = sub.add_parser(
        "collect", help="Maintainer: check every judge and export scores for review"
    )
    collect.add_argument("--work", required=True)
    audit = sub.add_parser(
        "audit-plan",
        help="Maintainer: choose an unpredictable, engine-stratified rerun sample",
    )
    audit.add_argument("bundle")
    audit.add_argument("--out", required=True)
    audit.add_argument("--per-engine", type=int, default=5)
    approve = sub.add_parser(
        "approve",
        help="Maintainer: bind a review decision to an immutable scored result",
    )
    approve.add_argument("result")
    approve.add_argument("--out", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--notes", required=True)
    approve.add_argument("--audit-report")
    publish = sub.add_parser(
        "leaderboard",
        help="Build a static leaderboard feed from maintainer-reviewed results",
    )
    publish.add_argument("--reviewed", required=True)
    publish.add_argument("--out", required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        if a.command == "protocol":
            result = catalog()
        elif a.command == "init":
            c = catalog()
            config = {
                "submission_id": "my-model-v1",
                "assistant": {
                    "name": "My model",
                    "revision": "version-or-checkpoint",
                    "family": "open-nonstreaming",
                    "inference_code": "https://github.com/ORG/REPO",
                    "inference_commit": "REPLACE_WITH_FULL_COMMIT_SHA",
                    "hardware": "GPU type/count, or hosted API",
                },
                "settings": {
                    e: {
                        **s,
                        "code_commit": "REPLACE_WITH_FULL_ENGINE_COMMIT_SHA",
                        "assistant_model_id": "REPLACE_WITH_NATIVE_MODEL_ID",
                    }
                    for e, s in c["settings"].items()
                },
                "sources": [
                    {
                        "engine": e,
                        "run": run,
                        "path": f"/absolute/path/to/{e}/run-{run}"
                        + ("/model.records.jsonl" if e == "screensim" else ""),
                        "tag": "my-model"
                        + (f"__r{run}" if e == "cooksim" and run > 1 else ""),
                    }
                    for e in c["engines"]
                    for run in c["runs"]
                ],
            }
            write_new(a.out, config)
            result = {
                "config": a.out,
                "next": "Set model identity, actual commits, and native record paths before preparing.",
            }
        elif a.command == "prepare":
            from .bundle import prepare

            result = prepare(a.config, a.out, a.preview)
        elif a.command == "validate":
            from .bundle import validate

            result = validate(a.bundle, a.allow_preview)
        elif a.command == "stage":
            from .grading import stage

            result = stage(a.bundle, a.work, a.allow_preview)
        elif a.command == "request":
            from .intake import make_request

            result = make_request(a.bundle, a.artifact_url, a.out)
        elif a.command == "fetch":
            from .intake import fetch_request

            result = fetch_request(a.request, a.out)
        elif a.command == "grade":
            from .grading import execute

            result = execute(a.work, a.allow_paid_judging)
        elif a.command == "collect":
            from .grading import collect

            result = collect(a.work)
        elif a.command == "audit-plan":
            from .publishing import audit_plan

            result = audit_plan(a.bundle, a.out, a.per_engine)
        elif a.command == "approve":
            from .publishing import approve

            result = approve(a.result, a.out, a.reviewer, a.notes, a.audit_report)
        else:
            from .publishing import build_ledger

            result = build_ledger(a.reviewed, a.out)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (Invalid, OSError, ValueError, KeyError) as exc:
        print(f"Submission operation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
