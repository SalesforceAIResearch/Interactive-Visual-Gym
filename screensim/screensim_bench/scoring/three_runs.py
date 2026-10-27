"""The benchmark table: strict success over the repeats, with the reliability columns.

S_mean  = mean over runs of the per-case strict success (the headline)
pass^3  = the case succeeded in EVERY run (reliability)
pass@3  = the case succeeded in at least one run (a footnote: nobody gets three tries at helping a person)
flips   = cases that did not come out the same way every run

Strict success is recomputed here from the stored fields so that the 2026-09-22 ruling applies to runs
made before it: text typed into a search box (settings._search / _apps_search) is how iOS finds a page,
not a change the owner did not ask for.

    python screensim_bench/scoring/three_runs.py tmp/bench_v1 tmp/bench_v2 tmp/bench_v3 [--out docs/bench_3runs.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics as st

SCRATCH = ("settings._search", "settings._apps_search")


def strict(r):
    extra = [p for p in (r.get("extra_changes") or []) if not p.startswith(SCRATCH)]
    return 1 if (r.get("goal_ok") and not extra and r.get("success_lenient")) else 0


def load(root):
    """The run, with any cell re-run on a fixed engine (<root>/../refix/<run>/) replacing the original."""
    out = collections.defaultdict(dict)
    fix = os.path.join(os.path.dirname(root), "refix", os.path.basename(root))
    for src in (root, fix):
        for f in glob.glob(os.path.join(src, "*.records.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if "error" in r:
                    continue
                out[r["contestant"]][(r["episode"], r["persona"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="docs/bench_3runs.json")
    a = ap.parse_args()
    runs = [load(x) for x in a.roots]
    rows = []
    for c in runs[0]:
        cases = sorted(set(runs[0][c]) & set.intersection(*[set(r.get(c, {})) for r in runs]) if len(runs) > 1
                       else set(runs[0][c]))
        if not cases:
            continue
        per = [[strict(r[c][k]) for r in runs] for k in cases]
        means = [sum(v) / len(v) for v in per]
        se = st.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
        rows.append({
            "assistant": c, "cases": len(cases), "runs": len(runs),
            "S_mean": st.mean(means), "SE": se,
            "per_run": [st.mean(v[i] for v in per) for i in range(len(runs))],
            "pass_hat": sum(1 for v in per if all(v)) / len(per),
            "pass_at": sum(1 for v in per if any(v)) / len(per),
            "flips": sum(1 for v in per if any(v) and not all(v)) / len(per),
            "S_lenient": st.mean([runs[0][c][k].get("success_lenient") or 0 for k in cases]),
            "caught": st.mean([x for r in runs for k in cases
                               if (x := r[c][k].get("recall")) is not None] or [float("nan")]),
            "FA_per_ep": st.mean([r[c][k].get("false_flags", 0) for r in runs for k in cases]),
        })
    rows.sort(key=lambda x: -x["S_mean"])
    if not rows:
        ap.error("No comparable episode records found in the supplied run directories")
    w = max(len(r["assistant"]) for r in rows)
    print(f"{'assistant':{w}} {'S_mean':>7} {'±SE':>5} " + " ".join(f"{'run'+str(i+1):>5}" for i in range(len(runs)))
          + f" {'pass^3':>7} {'pass@3':>7} {'flips':>6} {'caught':>7} {'FA/ep':>6}")
    for r in rows:
        print(f"{r['assistant']:{w}} {r['S_mean']:7.3f} {r['SE']:5.3f} "
              + " ".join(f"{x:5.2f}" for x in r["per_run"])
              + f" {r['pass_hat']:7.2f} {r['pass_at']:7.2f} {r['flips']:6.2f} {r['caught']:7.2f} {r['FA_per_ep']:6.2f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
