#!/usr/bin/env python3
"""Efficiency reference: ticks of a PERFECT clean-plan run, per grid cell.

    COOKSIM_GRID=tmp/grid.json python -m cooksim_bench.suite.gen_ref_ticks [--out bench/ref_ticks.json]

Writes `{cell: ticks}` — the denominator-free half of eval_v2's total-time efficiency,
`TT = clip01(ref_ticks / ticks)`. A cell's reference is the recipe's own reference plan
executed by the REAL executor with no supervisor, no LLM, no errors and no decoys: the
fastest a flawless cook could finish, in the same tick units the live episodes report.

Why this file has to be generated rather than guessed: TT divides by it, so a wrong
reference silently rescales the efficiency score of every episode on that cell — and it
rescales them all in the same direction, which is exactly the kind of error that looks
like a finding ("this persona is 30% less efficient") instead of like a bug.

Three properties the generator enforces, because each is a way the reference can be
quietly wrong:

  * the run must end `won`. A reference plan that stalls or times out is not a reference,
    it is a broken cell; emitting its tick count would hand TT a number with no meaning.
  * no command may be rejected. A rejected step means the executor disagreed with the
    reference plan, so the "ideal" was never actually achieved.
  * the value is deterministic. build_episode pins seed=11 and no supervisor runs, so a
    re-run reproduces the file exactly; the generator asserts this by replaying twice.

Cells are keyed by the harness's snake_case name (`master_nops_medium_map`), not the
grid's display label (`master-nops / medium-map`), because that is what case files and
`intent_lib.CELLS` use — the name the page and the scorer will look a report up by.
Provenance (engine rev, grid hash, per-cell outcome) goes to a sibling `.meta.json` so
the scored file stays exactly the `{cell: ticks}` schema eval_v2 expects.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys

import os
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)

CAP = 800          # generous: the longest reference plan (burrito, 40 ops) lands ~171


def cell_key(display: str) -> str:
    """`master-nops / medium-map` -> `master_nops_medium_map` (the harness's name)."""
    return display.replace(" / ", "_").replace("-", "_").replace(" ", "")


def replay(cell: str, cap: int = CAP):
    """One clean reference run. Returns the full record, not just the number, so the
    caller can refuse a cell rather than emit a meaningless tick count."""
    from cooksim_bench.exp_interact import build_episode
    from cooksim.core.plan_ir import segment_plan
    # inject=False: decoy sources exist to make ERROR scenarios ambiguous. They never
    # shorten or lengthen a correct plan, and `_place_decoys` raises outright on maps with
    # no spare counter — so including them could fail a cell for a reason unrelated to its
    # reference. lang=False: no announcements, nothing to react to.
    state, ro = build_episode(None, "capable_collaborator", cell=cell,
                              lang=False, inject=False)
    flat = [c for n in segment_plan(state.recipe) for c in n.commands]
    ro.cmds = [(c, None) for c in flat]
    ro.cap = cap
    while ro.i < len(ro.cmds) or ro.ctrl.skill is not None:
        if not ro.step():
            break
    bad = [l for l in ro.log if l["status"] in ("rejected", "illegal-skipped")]
    return {"cell": cell, "ticks": ro.tick, "outcome": ro.outcome,
            "plan_len": len(flat), "rejected": len(bad),
            "ok": ro.outcome == "won" and not bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark/ref_ticks.json"))
    ap.add_argument("--grid", default=os.environ.get("COOKSIM_GRID",
                                                     os.path.join(ROOT, os.path.join(ROOT, "benchmark", "grid.json"))))
    ap.add_argument("--cells", nargs="*", default=None,
                    help="restrict to these harness cell names (default: every grid cell)")
    a = ap.parse_args()
    os.environ.setdefault("COOKSIM_GRID", a.grid)

    grid = json.load(open(a.grid))
    cells = [cell_key(c["cell"]) for c in grid["grid"]]
    if a.cells:
        cells = [c for c in cells if c in set(a.cells)]
    seen, ordered = set(), []
    for c in cells:                                   # stable order, no duplicates
        if c not in seen:
            seen.add(c); ordered.append(c)

    ticks, meta, failed = {}, [], []
    for cell in ordered:
        try:
            r = replay(cell)
        except Exception as e:                        # noqa: BLE001
            r = {"cell": cell, "ticks": None, "outcome": f"ERROR: {e}", "ok": False,
                 "plan_len": None, "rejected": None}
        # Determinism check: a reference that differs between two identical replays would
        # make every downstream TT unreproducible, so catch it here rather than in a sweep.
        if r["ok"]:
            r2 = replay(cell)
            if r2["ticks"] != r["ticks"]:
                r["ok"] = False
                r["outcome"] = f"NONDETERMINISTIC: {r['ticks']} vs {r2['ticks']}"
        print(f"{'OK  ' if r['ok'] else 'DROP'} {cell:<26} ticks={r['ticks']!s:<5} "
              f"outcome={r['outcome']} plan_len={r['plan_len']} rejected={r['rejected']}",
              flush=True)
        meta.append(r)
        if r["ok"]:
            ticks[cell] = r["ticks"]
        else:
            failed.append(cell)

    if not ticks:
        print("!! no cell produced a valid reference — refusing to write", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(ticks, open(a.out, "w"), indent=1, sort_keys=True)
    try:
        rev = subprocess.run([sys.executable, os.path.join(ROOT, "python -m cooksim_bench.engine_stamp")],
                             capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:                                 # noqa: BLE001
        rev = ""
    json.dump({"schema": "ref-ticks-meta-v1", "engine_rev": rev,
               "grid_sha256": hashlib.sha256(open(a.grid, "rb").read()).hexdigest()[:16],
               "method": "clean reference plan, real executor, no supervisor/LLM, "
                         "inject=False, lang=False, seed=11, replayed twice",
               "cap": CAP, "cells": meta, "failed": failed},
              open(a.out.replace(".json", ".meta.json"), "w"), indent=1)
    print(f"\nwrote {a.out} ({len(ticks)} cells)"
          + (f"; {len(failed)} DROPPED: {failed}" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
