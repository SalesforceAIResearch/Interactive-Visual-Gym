"""Run from the trusted base checkout, with PR requests supplied as DATA only."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivg_submission.intake import check_request, fetch_request
from ivg_submission.common import require


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--incoming", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    base, incoming = Path(a.base), Path(a.incoming)
    require(
        all(
            not p.is_symlink()
            for p in (incoming, incoming.parent, incoming.parent.parent)
        ),
        "Request directory must not contain symbolic-link ancestors",
    )
    changed = []
    for p in sorted(incoming.glob("*.json")):
        require(not p.is_symlink() and p.is_file(), "Request must be a regular file")
        previous = base / p.name
        if previous.exists() and previous.read_bytes() == p.read_bytes():
            continue
        require(
            not previous.exists(),
            "Requests are immutable; submit a new ID for a revised model",
        )
        changed.append(p)
    require(len(changed) <= 3, "At most three new submission requests per pull request")
    reports = []
    for p in changed:
        request = check_request(p)
        require(
            p.stem == request["submission_id"],
            "Request filename must match submission_id",
        )
        with tempfile.TemporaryDirectory(prefix="ivg-intake-") as tmp:
            report = fetch_request(p, Path(tmp) / "submission.zip")
            reports.append(report)
    Path(a.report).write_text(json.dumps(reports, indent=2) + "\n")
    print(
        f"Validated {len(reports)} new submission(s). No grading or inference code was executed."
    )


if __name__ == "__main__":
    main()
