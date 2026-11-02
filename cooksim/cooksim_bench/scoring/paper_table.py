"""Aggregate paper objective metrics from eval_v2.json, without filtering success by events."""
import json
import os
from pathlib import Path
from statistics import fmean, pstdev
from .sq_v4_three_runs import parse_tag


def aggregate(episodes):
    groups = {}
    for key, ep in episodes.items():
        parts = key.split("/")
        if len(parts) < 4 or parts[0] != "frames":
            continue
        model, run = parse_tag(parts[-2])
        groups.setdefault((model, run), []).append(ep)
    models = {}
    for (model, run), eps in sorted(groups.items()):
        if any(e["v3"].get("in_time_success") is None for e in eps):
            raise ValueError(f"{model} run {run}: missing task-success score; repair the objective grading")
        fired = sum(e["evidence"]["errors_fired"] for e in eps)
        detected = sum(e["v3"]["tp"] for e in eps)
        prevented = sum(e["evidence"].get("prevented", 0) for e in eps)
        models.setdefault(model, {"runs": {}})["runs"][str(run)] = {
            "episodes": len(eps), "success": fmean(e["v3"]["in_time_success"] for e in eps),
            "triggered": fired, "detected": detected,
            "recall": detected / fired if fired else None,
            "prevention": prevented / fired if fired else None,
        }
    for row in models.values():
        for key in ("success", "recall", "prevention"):
            vals = [v[key] for v in row["runs"].values() if v[key] is not None]
            row[key] = fmean(vals) if vals else None
            row[key + "_sd"] = pstdev(vals) if vals else None
    return {"aggregation": "all-episodes-success; event-weighted-recall; equal-run-mean", "models": models}


def main():
    root = Path(os.environ["ASSIST_ROOT"])
    out = aggregate(json.loads((root / "eval_v2.json").read_text())["episodes"])
    (root / "paper_scores.json").write_text(json.dumps(out, indent=2) + "\n")
    print("Wrote", root / "paper_scores.json")


if __name__ == "__main__":
    main()
