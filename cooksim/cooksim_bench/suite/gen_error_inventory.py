#!/usr/bin/env python3
"""FULL inventory of realizable error instances per cell — the raw material for designing
each composite episode by hand. Unlike the pooled generator (which kept <=3 per label), this
keeps EVERY candidate that survives the replay purity gate, enumerated over a wider kwargs
space: every off-recipe source as the foreign item, every wrong ingredient for mis_prep,
every subgoal for omit/forget/premature/search, every ordered pair for wrong_order, every
component and loop count for leave_unattended.

For each instance it records what the pooled cases record (dev, window, outcome...) PLUS
the plate-anchor (m, plate key) the live harness would use, a plain-words description,
and whether the anchor collapses (unusable).

    $PY python -m cooksim_bench.suite.gen_error_inventory  -> tmp/interact_eval/error_inventory.json + .md
"""
import collections
import json
import os
from cooksim_bench.paths import ROOT
import sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.suite import gen_detect_cases_v2 as G  # noqa: E402
from cooksim_bench.suite import gen_hardmap_cases as H  # noqa: E402
from cooksim_bench.suite import gen_composite_errors as CE  # noqa: E402
from cooksim_bench.suite.gen_composite_cases import derive_anchor                      # noqa: E402
from cooksim_bench.exp_interact import recipe_of                                 # noqa: E402
from cooksim.core.plan_ir import segment_plan                      # noqa: E402

CELLS10 = CE.CELLS10


def wide_candidates(cell, off):
    rec = recipe_of(cell)
    plan = list(segment_plan(rec))
    sids = [n.sid for n in plan]
    asm = [n.sid for n in plan if n.type == "assemble"]
    comps = [n.component for n in plan if n.component]
    prepped = [n.component for n in plan if n.type == "prep"]
    cooked = [c for c, st in rec.contents if st == "cooked"]
    C = [(e, k, c) for (e, k, c) in G.candidates() if c == cell]
    seen = {(e, json.dumps(k, sort_keys=True)) for e, k, c in C}

    def add(e, k):
        key = (e, json.dumps(k, sort_keys=True))
        if key not in seen:
            seen.add(key)
            C.append((e, k, cell))
    # every off-recipe source as the foreign item (hard maps have several)
    for w in (off or []):
        for before in asm + ["serve"]:
            add("add_foreign", {"wrong": w, "before": before})
        for comp in dict.fromkeys(comps):
            if f"assemble({comp})" in sids:
                add("misidentify_assemble", {"component": comp, "wrong": w})
    # v3 (2026-08-27): misidentify_assemble with an ON-recipe wrong item (scoop the cheese where
    # the tomato goes) — the only other ruinous move a cold dish allows — and serve_raw on every
    # recipe with a cooked or chopped component, not just the burrito
    for comp in dict.fromkeys(comps):
        if f"assemble({comp})" in sids:
            for w in dict.fromkeys(comps):
                if w != comp:
                    add("misidentify_assemble", {"component": comp, "wrong": w})
    for comp, st_ in rec.contents:
        if st_ in ("cooked", "chopped"):
            add("serve_raw", {"component": comp})
    # mis_prep: every wrong ingredient present on the map (recipe items + off-recipe)
    others = list(dict.fromkeys(comps)) + list(off or []) + (["lettuce"] if G.CELLS[cell][1] else [])
    for comp in prepped:
        for w in others:
            if w != comp:
                add("mis_prep", {"component": comp, "wrong": w})
    # forget / premature over every subgoal boundary
    for after in ["fetch_plate"] + asm:
        if after != asm[-1]:
            add("premature_serve", {"after": after})
    # forget_next_step: every plan step whose NEXT step is real work. DESIGN RULE (user, 2026-09-15): forget_next_step never forge…
    # (full rule in gen_composite_errors.gate). Prep and ignition steps are candidates too: before
    # 2026-09-15 only plating boundaries were, so on hot recipes every mid-plating freeze burned and
    # the only survivor was the step before serve.
    from cooksim.core.plan_ir import segment_plan as _seg
    _sids = [n.sid for n in _seg(rec)]
    for after in _sids[:_sids.index("serve") - 1] if "serve" in _sids else _sids[:-1]:
        add("forget_next_step", {"after": after})
    for comp in cooked:
        for loops in (1, 2, 3):
            add("leave_unattended", {"component": comp, "loops": loops})
    return C


DESC = {
    "mis_prep": lambda k: f"preps {k['wrong']} where the recipe needs {k['component']}",
    "add_foreign": lambda k: f"scoops off-recipe {k['wrong']} onto the plate before {k['before']}",
    "misidentify_assemble": lambda k: f"plates {k['wrong']} in place of {k['component']}",
    "omit_subgoal": lambda k: f"skips {k['sid']} entirely",
    "premature_serve": lambda k: f"serves right after {k['after']}, leaving the rest undone",
    "wrong_order": lambda k: f"plates {k['b']} before {k['a']} (order matters here)",
    "repeat_step": lambda k: f"does {k['sid']} a second time, at {k['at']}",
    "forget_next_step": lambda k: f"goes idle after {k['after']} as if the plan were lost",
    "leave_unattended": lambda k: f"walks away from the {k['component']} on live heat ({k['loops']} busywork loops)",
    "serve_raw": lambda k: f"plates {k['component']} raw",
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/interact_eval/error_inventory.json")
    ap.add_argument("--cells", default=None, help="comma-separated subset of the ten cells")
    a = ap.parse_args()
    G.CELLS.update(H.NEW_CELLS)
    inv, rejected = [], []
    cells = [c for c in CELLS10 if not a.cells or c in a.cells.split(",")]
    for cell in cells:
        names, on, off = H.sources_of(cell)
        inject = G.CELLS[cell][1]
        cands = wide_candidates(cell, off)
        print(f"== {cell}: {len(cands)} candidates (off-recipe sources: {off})", flush=True)
        for i, (err, kwargs, c) in enumerate(cands, 1):
            try:
                st = G.replay(err, kwargs, cell, inject)
            except Exception as e:                                  # noqa: BLE001
                rejected.append((cell, err, kwargs, f"replay error: {e!r}"))
                continue
            why = CE.gate(err, kwargs, cell, st)
            if why:
                rejected.append((cell, err, kwargs, why))
                continue
            ident = st["identifiable"]
            whi = ((st["fire_tick"] if st["fire_tick"] is not None else st["outcome_tick"])
                   if err == "leave_unattended" else ident + G.WINDOW)
            case = dict(err=G.DETECT_LABEL.get(err, err), fn=err, kwargs=kwargs, cell=cell,
                        inject=inject, persona="capable_collaborator", dev=st["dev"],
                        identifiable=ident, window=[ident - 2, whi], outcome=st["outcome"],
                        outcome_tick=st["outcome_tick"], end=st["end"], max_smoke=st["max_smoke"],
                        family="error")
            try:
                m, key, t0key, plan_len = derive_anchor(case, cell, inject)
            except Exception as e:                                  # noqa: BLE001
                rejected.append((cell, err, kwargs, f"anchor error: {e!r}"))
                continue
            case.update(anchor_m=m, anchor_plate=list(key["plate"]), anchor_held=key["plate_held"], anchor_prep=list(key.get("prep") or ()),
                        collapses=bool(m > 0 and key == t0key), plan_len=plan_len,
                        desc=DESC[err](kwargs))
            case["name"] = f"inv_{case['err']}_{CE.ABBREV[cell]}_{len(inv)}"
            inv.append(case)
            if i % 20 == 0:
                print(f"   [{i}/{len(cands)}] {len(inv)} admissible so far", flush=True)
    usable = [c for c in inv if not c["collapses"]]
    json.dump({"schema": "error-inventory-v1", "cases": inv, "rejected": [list(r[:2]) + [r[2], str(r[3])] for r in rejected]},
              open(a.out, "w"), indent=1)
    with open(a.out[:-len(".json")] + ".md", "w") as f:
        f.write(f"# Error inventory — {len(inv)} verified instances ({len(usable)} with usable anchors), "
                f"{len(rejected)} rejected\n\n")
        for cell in cells:
            rows = [c for c in inv if c["cell"] == cell]
            f.write(f"## {cell} ({len(rows)} instances)\n\n| label | what the cook does | anchor m | plate at anchor | held | outcome | usable |\n|---|---|---|---|---|---|---|\n")
            for c in sorted(rows, key=lambda c: (c["anchor_m"], c["err"])):
                f.write(f"| {c['err']} | {c['desc']} | {c['anchor_m']} | {c['anchor_plate']} | {c['anchor_held']} | "
                        f"{c['outcome']}@{c['outcome_tick']} | {'' if c['collapses'] else 'yes'} |\n")
            f.write("\n")
        cnt = collections.Counter((c["cell"], c["err"]) for c in usable)
        f.write("## usable instances per cell x label\n\n")
        for cell in cells:
            f.write(f"- {cell}: " + ", ".join(f"{lab} {cnt[(cell, lab)]}" for lab in CE.LABELS) + "\n")
    print(f"inventory: {len(inv)} verified, {len(usable)} usable, {len(rejected)} rejected")


if __name__ == "__main__":
    main()
