"""The three headline scores, per assistant, over the repeats.

  in-time success  S   the goal is met, nothing else on the phone changed, inside the tick cap
  efficiency       E   mean of action efficiency  A = reference ticks / ticks used  (capped at 1)
                              and conversation efficiency  C = 1 - talk turns / polls   (the record's cc)
  detection        D   mean of prevention  P = the engine's detection-timing score (1 caught and undone,
                              0.5 caught late, 0 missed)  and  F1 of the assistant's mistake calls

Every score is 0..1; episodes with no forced mistake that fired contribute to S and E only.
Runs are merged with any cell re-run on a fixed engine (tmp/refix/<run>/).

    python screensim_bench/scoring/scores.py tmp/bench_v1 tmp/bench_v2 tmp/bench_v3 [--out docs/bench_scores.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics as st
import sys


from screensim.interact.composite import design_composite_episodes
from screensim.interact.errors import dry_replay
from screensim.tasks import by_id

SCRATCH = ("settings._search", "settings._apps_search")
_REF = {}


def ref_ticks(task_id):
    """How many ticks the reference solution needs — the yardstick for action efficiency."""
    if task_id not in _REF:
        t = by_id(task_id)
        ir, _eps = design_composite_episodes(t, per_task=3)
        _REF[task_id] = dry_replay(t, ir.flat())["end_tick"]
    return _REF[task_id]


def strict(r):
    extra = [p for p in (r.get("extra_changes") or []) if not p.startswith(SCRATCH)]
    return 1 if (r.get("goal_ok") and not extra and r.get("success_lenient")) else 0


def load(root):
    out = collections.defaultdict(dict)
    fix = os.path.join(os.path.dirname(root), "refix", os.path.basename(root))
    for src in (root, fix):
        for f in glob.glob(os.path.join(src, "*.records.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if "error" not in r:
                    out[r["contestant"]][(r["episode"], r["persona"])] = r
    return out


def scores(r):
    a = min(1.0, ref_ticks(r["task"]) / max(1, r["final_tick"])) if r["final_tick"] else None
    c = r.get("cc")
    e = st.mean([x for x in (a, c) if x is not None]) if (a is not None or c is not None) else None
    p, f1 = r.get("dt"), r.get("f1")
    d = st.mean([x for x in (p, f1) if x is not None]) if r.get("n_errors") else None
    return {"S": strict(r), "A": a, "C": c, "E": e, "P": p, "F1": f1, "D": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="docs/bench_scores.json")
    a = ap.parse_args()
    runs = [load(x) for x in a.roots]
    rows = []
    for c in runs[0]:
        per = collections.defaultdict(list)
        won = collections.defaultdict(list)          # efficiency is only meaningful where the task succeeded:
        n = 0                                        # silence and giving up early both look "efficient"
        for run in runs:
            for k, r in run.get(c, {}).items():
                n += 1
                sc = scores(r)
                for key, v in sc.items():
                    if v is not None:
                        per[key].append(v)
                        if sc["S"]:
                            won[key].append(v)
        if not n:
            continue
        m = {k: st.mean(v) for k, v in per.items() if v}
        se = (st.stdev(per["S"]) / math.sqrt(len(per["S"]))) if len(per["S"]) > 1 else 0.0
        mw = {k: st.mean(v) for k, v in won.items() if v}
        rows.append({"assistant": c, "episodes": n, "S": float(m.get("S", 0.0)), "S_se": se,
                     "E": m.get("E"), "A": m.get("A"), "C": m.get("C"),
                     "E_won": mw.get("E"), "A_won": mw.get("A"), "C_won": mw.get("C"),
                     "D": m.get("D"), "P": m.get("P"), "F1": m.get("F1")})
    rows.sort(key=lambda r: -(r["S"] or 0))
    if not rows:
        ap.error("No usable episode records found in the supplied run directories")
    w = max(len(r["assistant"]) for r in rows)
    print(f"{'assistant':{w}} {'eps':>4} | {'SUCCESS':>8} {'±SE':>5} | {'EFF|won':>8} {'action':>7} {'conv':>6} "
          f"| {'EFF|all':>8} | {'DETECT':>7} {'prevent':>8} {'F1':>6}")
    for r in rows:
        def f(x):
            return f"{x:.2f}" if isinstance(x, float) else "  — "
        print(f"{r['assistant']:{w}} {r['episodes']:4} | {f(r['S']):>8} {r['S_se']:5.3f} | {f(r['E_won']):>8} "
              f"{f(r['A_won']):>7} {f(r['C_won']):>6} | {f(r['E']):>8} | {f(r['D']):>7} {f(r['P']):>8} {f(r['F1']):>6}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
