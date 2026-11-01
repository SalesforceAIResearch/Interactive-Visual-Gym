"""Reviewed results only. Validation and scoring cannot authorize publication."""

import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .bundle import manifest, validate
from .common import (
    SHA,
    catalog,
    digest,
    encoded,
    file_hash,
    read_json,
    require,
    safe_id,
    write_new,
)
from .grading import AGGREGATION


def audit_plan(bundle, out, per_engine=5, seed=None):
    validate(bundle)
    require(1 <= per_engine <= 105, "Invalid audit sample size")
    m = manifest(bundle)
    seed = seed or secrets.token_hex(32)
    selected = []
    for engine in catalog()["engines"]:
        # Stratify by engine; select distinct cases before picking the run/persona.
        cases = sorted(
            {e["case"] for e in m["episodes"] if e["engine"] == engine},
            key=lambda case: hashlib.sha256(
                f"{seed}:{engine}:{case}".encode()
            ).digest(),
        )
        for case in cases[:per_engine]:
            choices = [
                e for e in m["episodes"] if e["engine"] == engine and e["case"] == case
            ]
            e = min(
                choices,
                key=lambda x: hashlib.sha256(f"{seed}:{x['record']}".encode()).digest(),
            )
            selected.append(
                {k: e[k] for k in ("engine", "run", "case", "persona", "record")}
            )
    result = {
        "schema": 1,
        "submission_id": m["submission_id"],
        "manifest_sha256": digest(encoded(m)),
        "seed": seed,
        "selected": selected,
        "status": "planned-not-executed",
        "checks": [
            "Assistant sees only permitted observations and conversation.",
            "User model, prompts, delays, task/persona and run settings match the protocol.",
            "Run submitted inference code in a separate disposable sandbox, without judge credentials.",
            "Inspect behavior and score consistency; do not require stochastic reruns to be byte-identical.",
        ],
    }
    write_new(out, result)
    return result


def approve(result_path, out, reviewer, notes, audit_report=None):
    r = read_json(result_path)
    require(
        r.get("status") == "scored-awaiting-review" and r.get("scope") == "full",
        "Only fully scored results can be reviewed",
    )
    require(
        r.get("protocol") == catalog()["version"]
        and r.get("aggregation") == AGGREGATION
        and r.get("judges") == catalog()["judges"],
        "Score protocol or judge settings do not match",
    )
    require(
        not r.get("publication_blockers"), "Resolve scoring blockers before approval"
    )
    require(
        isinstance(reviewer, str)
        and reviewer.strip()
        and isinstance(notes, str)
        and notes.strip(),
        "Reviewer and review notes are required",
    )
    tier = "officially-scored"
    evidence = None
    if audit_report:
        audit = read_json(audit_report)
        require(
            audit.get("manifest_sha256") == r["manifest_sha256"]
            and audit.get("status") == "passed",
            "Audit did not pass or refers to different inputs",
        )
        plan = audit.get("plan") or {}
        require(
            plan.get("manifest_sha256") == r["manifest_sha256"] and plan.get("seed"),
            "Missing bound audit plan",
        )
        selected = {
            (x["engine"], x["run"], x["case"], x["persona"])
            for x in plan.get("selected", [])
        }
        require(
            selected and {x[0] for x in selected} == set(catalog()["engines"]),
            "Audit must cover every engine",
        )
        require(
            isinstance(audit.get("cases"), list)
            and len(audit["cases"]) == len(selected)
            and all(
                x.get("passed") is True
                and SHA.fullmatch(str(x.get("evidence_sha256", "")))
                for x in audit["cases"]
            ),
            "Missing audit evidence",
        )
        require(
            {(x["engine"], x["run"], x["case"], x["persona"]) for x in audit["cases"]}
            == selected,
            "Audit cases differ from the selected sample",
        )
        tier, evidence = "sample-audited", file_hash(audit_report)
    approval = {
        "schema": 1,
        "submission_id": r["submission_id"],
        "result_sha256": file_hash(result_path),
        "reviewer": reviewer,
        "review_notes": notes,
        "verification": tier,
        "audit_report_sha256": evidence,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    write_new(out, approval)
    return approval


def build_ledger(reviewed, out):
    """Read ONLY maintainer-reviewed files, not an incoming submission directory."""
    root = Path(reviewed)
    entries = []
    for approval_path in sorted(root.glob("*/approval.json")):
        approval = read_json(approval_path)
        result_path = approval_path.with_name("result.json")
        require(
            file_hash(result_path) == approval["result_sha256"],
            "Approved result changed",
        )
        r = read_json(result_path)
        require(
            r.get("scope") == "full"
            and r.get("status") == "scored-awaiting-review"
            and not r.get("publication_blockers"),
            "Incomplete result cannot be published",
        )
        require(
            r["submission_id"]
            == approval["submission_id"]
            == approval_path.parent.name,
            "Review identity mismatch",
        )
        require(
            approval["verification"] in ("officially-scored", "sample-audited"),
            "Unknown verification tier",
        )
        require(
            set(r["engines"]) == set(catalog()["engines"]), "Missing engine results"
        )
        for engine, scores in r["engines"].items():
            require(set(scores["runs"]) == {"1", "2", "3"}, "Missing independent run")
            require(
                all(
                    x["episodes"] == len(catalog()["engines"][engine])
                    for x in scores["runs"].values()
                ),
                "Incomplete run coverage",
            )
        entries.append(
            {
                "id": safe_id(r["submission_id"]),
                "assistant": r["assistant"],
                "protocol": r["protocol"],
                "aggregation": r["aggregation"],
                "engines": r["engines"],
                "verification": approval["verification"],
                "approved_at": approval["approved_at"],
                "result_sha256": approval["result_sha256"],
            }
        )
    result = {"schema": 1, "entries": entries}
    write_new(out, result)
    return {"published_entries": len(entries), "out": str(out)}
