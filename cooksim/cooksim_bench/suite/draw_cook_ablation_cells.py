#!/usr/bin/env python3
"""Draw the FIXED 30-episode subset for the cook-model ablation (2026-09-22).

The formal grid is 50 composite cases x 3 personas = 150 cells. The ablation runs 30 of
them, and all four cook models run EXACTLY the same 30, so any difference between cooks is
not a difference of episodes.

The draw is stratified so it is both wide and balanced:
  * 30 DISTINCT cases out of the 50 (a case never repeats, so case-specific quirks cannot
    dominate the sample),
  * each of the three personas appears exactly 10 times (the persona mix is identical for
    every cook),
  * the assignment of persona to case is itself random under that constraint.

    python -m cooksim_bench.suite.draw_cook_ablation_cells            # rewrites bench/v5/cook_ablation_cells.json
Deterministic: the seed is in SEED below and is recorded in the output file.
"""
import json
import os
import random
import sys

SEED = 20260922
PERSONAS = ["baseline", "classic_novice", "classic_expert"]
SUITE = "benchmark/cases_composite_v5.json"
OUT = "benchmark/cook_ablation_cells.json"
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
def draw(n_cases=30, seed=SEED):
    pool = json.load(open(os.path.join(ROOT, SUITE)))
    names = sorted(c["name"] for c in pool["cases"])
    rng = random.Random(seed)
    cases = sorted(rng.sample(names, n_cases))
    slots = PERSONAS * (n_cases // len(PERSONAS))
    rng.shuffle(slots)
    return names, [[c, p] for c, p in zip(cases, slots)]


def grow(cells, seed=SEED):
    """Ruling 2026-10-09 (user): 30 episodes are too few -> 90. The 30 drawn cells stay first and unchanged (every record
    already on disk remains valid); the 60 added cells are the SAME 30 cases under their two other personas, so the grown
    list is 30 distinct cases x all 3 personas (each persona exactly 30 times)."""
    rng = random.Random(seed + 1)
    extra = [[c, q] for c, p in cells for q in PERSONAS if q != p]
    rng.shuffle(extra)
    return cells + extra


def main():
    names, cells = draw()
    cells = grow(cells)
    out = {"schema": "cook-ablation-cells-v2", "seed": SEED, "suite": SUITE,
           "n_suite_cases": len(names), "personas": PERSONAS,
           "drawn_by": "python -m cooksim_bench.suite.draw_cook_ablation_cells",
           "note": "90 cells = 30 distinct cases x 3 personas; the first 30 are the 2026-09-22 draw (one persona per case), "
                   "the next 60 the same cases under their other two personas (2026-10-09); every cook model runs this exact list",
           "cells": cells}
    path = os.path.join(ROOT, OUT)
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {OUT}: {len(cells)} cells, {len(set(c for c, _ in cells))} distinct cases")
    for p in PERSONAS:
        print(f"  {p}: {sum(1 for _, q in cells if q == p)}")
    if "--print" in sys.argv:
        for c, p in cells:
            print(f"  {c:38s} {p}")


if __name__ == "__main__":
    main()
