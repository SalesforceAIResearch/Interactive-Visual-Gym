#!/usr/bin/env python3
"""Split the task library into an RL training set and a held-out benchmark set, and prove they differ.

The benchmark set is frozen: the 11 HARD_TASKS the formal runs were scored on. The training set is
the hand-written RL library (bands 1-4). The two must not be the same task in different words, so
every training task is checked against every benchmark task on its GOAL — the (state path, value)
pairs its reference solution produces:

  leak        the training goal is a subset of a benchmark goal, or a superset of one. Either way the
              benchmark task is contained in the training task, which is memorisable. Rejected.
  overlap     some paths in common but neither contains the other. Kept, and reported — the phone has
              only so many switches, and sharing controls is what makes transfer possible.

    python screensim_bench/suite/split.py           # report + write screensim/task_splits.py
    python screensim_bench/suite/split.py --check   # report only
"""
from __future__ import annotations

import collections
import copy
import os
import sys


from screensim.core.engine import Device
from screensim.interact.composite import design_composite_episodes
from screensim.interact.errors import dry_replay
from screensim.tasks import HARD_TASKS, TASKS, by_id, resolve_dynamic

SCRATCH = ("settings._search", "settings._apps_search")


def _diff(a, b, pre=""):
    out = []
    if isinstance(a, dict):
        for k in a:
            out += _diff(a[k], (b or {}).get(k), f"{pre}.{k}" if pre else k)
    elif a != b:
        out.append((pre, repr(b)))
    return out


def goal_of(t):
    d = Device(t.seed)
    if t.start_app != "home":
        d.open_app(t.start_app)
    before = copy.deepcopy(d.st.data)
    for g in t.plan:
        d.act(resolve_dynamic(d, g))
    return {pv for pv in _diff(before, d.st.data) if not pv[0].startswith(SCRATCH)}


def main():
    bench = [by_id(h) for h in HARD_TASKS]
    train = [t for t in TASKS if getattr(t, "band", 0) > 0]
    B = {t.id: goal_of(t) for t in bench}
    leaks, overlaps, clean, seen = [], [], [], {}
    keep = []
    for t in train:
        G = goal_of(t)
        bad = None
        for bid, BG in B.items():
            if G and BG and (G <= BG or BG <= G):
                bad = bid
                break
        key = frozenset(G)
        if bad:
            leaks.append((t.id, bad))
        elif key in seen:
            leaks.append((t.id, f"duplicate of training task {seen[key]}"))
        else:
            seen[key] = t.id
            keep.append(t)
            shared = max((len(G & BG) for BG in B.values()), default=0)
            (overlaps if shared else clean).append((t.id, shared, len(G)))

    print(f"benchmark (held out): {len(bench)} tasks")
    print(f"RL training candidates: {len(train)} -> kept {len(keep)}, rejected {len(leaks)}")
    for i, why in leaks:
        print(f"    rejected {i:28} contains / is contained by {why}")
    print(f"\nkept tasks sharing at least one control with the benchmark: {len(overlaps)} "
          f"(fine — neither contains the other); sharing nothing: {len(clean)}")

    rows = collections.defaultdict(lambda: [0, 0])
    for t in keep:
        _ir, eps = design_composite_episodes(t, per_task=3)
        rows[t.band][0] += 1
        rows[t.band][1] += len(eps)
    bench_eps = sum(len(design_composite_episodes(t, per_task=3)[1]) for t in bench)
    print(f"\n{'set':24} {'group':10} {'tasks':>6} {'designed episodes':>18} {'x3 personas':>12}")
    tot_t = tot_e = 0
    for band in sorted(rows):
        n, e = rows[band]
        tot_t += n
        tot_e += e
        print(f"{'RL training':24} {'band ' + str(band):10} {n:6} {e:18} {e * 3:12}")
    print(f"{'RL training':24} {'ALL':10} {tot_t:6} {tot_e:18} {tot_e * 3:12}")
    print(f"{'benchmark (held out)':24} {'HARD':10} {len(bench):6} {bench_eps:18} {bench_eps * 3:12}")

    if "--check" in sys.argv:
        return
    out = ['"""Which tasks train and which test — written by screensim_bench/suite/split.py, do not edit by hand.',
           "",
           "BENCH_TEST is frozen: the tasks the formal benchmark runs were scored on. RL_TRAIN is the",
           "hand-written library minus any task whose goal contains, or is contained by, a benchmark",
           'task\'s goal."""', "",
           f"BENCH_TEST = {sorted(t.id for t in bench)!r}", "",
           f"RL_TRAIN = {sorted(t.id for t in keep)!r}", "",
           f"RL_TRAIN_BY_BAND = {({b: sorted(t.id for t in keep if t.band == b) for b in sorted(rows)})!r}",
           ""]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "screensim", "task_splits.py")
    open(path, "w").write("\n".join(out))
    print("\nwrote screensim/task_splits.py")


if __name__ == "__main__":
    main()
