"""Offline adapter smoke test on maintainer-provided native records. Never publishes results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivg_submission.bundle import prepare
from ivg_submission.common import (
    catalog,
    encoded,
    file_hash,
    loads,
    read_json,
    write_new,
)
from ivg_submission.grading import stage


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cooksim", required=True, type=Path)
    ap.add_argument("--vhhome", required=True, type=Path)
    ap.add_argument("--screensim", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    before = {str(p): file_hash(p) for p in (a.cooksim, a.vhhome, a.screensim)}
    records = {"cooksim": read_json(a.cooksim), "vhhome": read_json(a.vhhome)}
    with a.screensim.open("rb") as f:
        records["screensim"] = loads(f.readline())
    settings = {
        e: {**s, "code_commit": "0" * 40} for e, s in catalog()["settings"].items()
    }
    sources = []
    for e, r in records.items():
        root = a.out / "native" / e
        if e == "cooksim":
            settings[e]["assistant_model_id"] = r["assistant_backend"]["model"]
            path = (
                root
                / "frames"
                / r["persona"].removeprefix("pv2_")
                / "smoke"
                / f"{r['scenario']}.json"
            )
        elif e == "vhhome":
            settings[e]["assistant_model_id"] = r["assistant"]["model"]
            path = root / "smoke" / f"smoke__{r['scenario']}__{r['persona']}.json"
        else:
            settings[e]["assistant_model_id"] = r["contestant"]
            path = root / "smoke.records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            (json.dumps(r) + "\n").encode() if e == "screensim" else encoded(r)
        )
        sources.append(
            {
                "engine": e,
                "run": 1,
                "path": str(path if e == "screensim" else root),
                "tag": r["contestant"] if e == "screensim" else "smoke",
            }
        )
    config = {
        "submission_id": "adapter-smoke-only",
        "assistant": {
            "name": "Adapter smoke test (mixed models)",
            "revision": "not-a-submission",
            "family": "frontier-api",
            "hardware": "record replay only",
            "inference_code": "https://github.com/SalesforceAIResearch/Interactive-Visual-Gym",
            "inference_commit": "0" * 40,
        },
        "settings": settings,
        "sources": sources,
    }
    write_new(a.out / "config.json", config)
    report = prepare(a.out / "config.json", a.out / "preview.zip", partial=True)
    stage(a.out / "preview.zip", a.out / "staged", allow_preview=True)
    assert before == {str(p): file_hash(p) for p in (a.cooksim, a.vhhome, a.screensim)}
    write_new(a.out / "report.json", {**report, "originals_unchanged": True})
    print(json.dumps({**report, "originals_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
