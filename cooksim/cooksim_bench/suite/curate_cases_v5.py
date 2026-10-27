#!/usr/bin/env python3
"""Curate the composite suite for scheduler v5 -> tmp/interact_eval/cases_composite_v4.json.

    $PY python -m cooksim_bench.suite.curate_cases_v5 [--src tmp/interact_eval/cases_composite_v3.json] [--dst tmp/interact_eval/cases_composite_v4.json]

What it changes (every change is recorded in the case under "curation_v5"):
  * mis_prep whose `wrong` item is a recipe component processed ON its own path (chopping a potato
    that the recipe chops anyway) can never commit under any cook: `wrong` is replaced by an
    ingredient the map offers whose processing at the component's station is OFF its own path,
    preferring items not in the recipe at all.
  * everything else is left as designed; the anchors are derived at run time from the same
    `dev`/`trigger_after` fields, so no anchor field is rewritten here.
Run python -m cooksim_bench.suite.verify_cases_v5 --cases tmp/interact_eval/cases_composite_v4.json afterwards.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from cooksim_bench.paths import ROOT  # the CookSim release root

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))


def cell_info(cell, inject=True):
    """The kitchen exactly as the harness builds it for this case (decoy sources only when the
    case says inject=True — hard-map cases run the bare map)."""
    from cooksim_bench import exp_interact as X
    from cooksim.core.plan_ir import segment_plan
    from cooksim.core.enums import station_accepts, Terrain
    state, ro = X.build_episode(None, "capable_collaborator", cell=cell, inject=inject)
    g = ro.game
    sources = sorted({str(t.value).replace("_source", "") for row in g.layout.grid for t in row if str(t.value).endswith("_source")})
    sources = [s for s in sources if s != "plate"]
    recipe = [n for n, _ in state.recipe.contents]
    plan = list(segment_plan(state.recipe))
    prep = {n.component: list(n.commands) for n in plan if n.type == "prep"}
    lbl = {"pot": Terrain.POT, "pan": Terrain.PAN, "oven": Terrain.OVEN, "fryer": Terrain.FRYER}
    def station_of(comp):
        for c in prep.get(comp, []):
            for w in c.split():
                if w in lbl:
                    return w
        return None
    return {"sources": sources, "recipe": recipe, "prep": prep, "station_of": station_of,
            "accepts": lambda st, item: station_accepts(lbl[st], item, "raw") if st in lbl else False}


def off_path_wrong(info, component, wrong):
    """Does mis_prep(component -> wrong) process `wrong` off its own recipe path, and can the
    kitchen actually execute that processing (a station that accepts it, or a choppable item)?"""
    from cooksim.core.enums import choppable
    own = info["prep"].get(wrong, [])
    st = info["station_of"](component)
    if st and info["accepts"](st, wrong):
        return wrong not in info["recipe"] or info["station_of"](wrong) != st   # cooked where its path never goes
    # falls to the cutting board: must be choppable, and chopping must be off its own path
    if not choppable(wrong):
        return False
    return wrong not in info["recipe"] or "chop" not in own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="tmp/interact_eval/cases_composite_v3.json")
    ap.add_argument("--dst", default="tmp/interact_eval/cases_composite_v4.json")
    a = ap.parse_args()
    pool = json.load(open(os.path.join(ROOT, a.src)))
    infos = {}
    changes = []
    for c in pool["cases"]:
        cell = c["cell"]
        inj = bool(c["subs"][0].get("inject", True)) if c.get("subs") else True
        if (cell, inj) not in infos:
            infos[(cell, inj)] = cell_info(cell, inject=inj)
        info = infos[(cell, inj)]
        for s in c["subs"]:
            if s.get("err") == "repeat_step":
                # a repeated PLATING placed after the recipe's last plating step can never land
                # (the plate is complete): trigger it after the previous plating step instead
                kw = s.get("kwargs") or {}
                sid, at = kw.get("sid", ""), kw.get("at", "")
                asm = [f"assemble({n})" for n in info["recipe"]]
                if sid.startswith("assemble") and at == asm[-1] and len(asm) >= 2:
                    prev_at = asm[-2]
                    if asm.index(sid) < asm.index(prev_at):
                        s["kwargs"] = dict(kw, at=prev_at)
                        s.setdefault("curation_v5", []).append({"issue": f"repeat_step {sid} at {at}: the plate is complete at the trigger, the repeated scoop cannot land",
                                                               "action": f"at -> {prev_at}"})
                        changes.append((c["name"], sid, at, f"at={prev_at}"))
                continue
            if s.get("err") != "mis_prep":
                continue
            kw = s.get("kwargs") or {}
            comp, wrong = kw.get("component"), kw.get("wrong")
            if off_path_wrong(info, comp, wrong):
                continue
            # candidates: not in the recipe first, then recipe items processed off their path
            cands = [x for x in info["sources"] if x not in info["recipe"] and x != comp and off_path_wrong(info, comp, x)]
            cands += [x for x in info["recipe"] if x != comp and x != wrong and off_path_wrong(info, comp, x)]
            if not cands:
                # no ingredient on this map can be mis-prepped off its path: swap the error for a
                # ruinous one on the same component that this kitchen CAN execute — serve_raw
                # (the component plated straight from its crate), unless the case already has it
                others = {(x.get("err"), (x.get("kwargs") or {}).get("component")) for x in c["subs"] if x is not s}
                if ("serve_raw", comp) not in others:
                    s["err"] = "serve_raw"; s["fn"] = "serve_raw"; s["kwargs"] = {"component": comp}
                    s["name"] = s["name"].replace("mis_prep", "serve_raw")
                    s.setdefault("curation_v5", []).append({"issue": f"mis_prep {comp}->{wrong} is on-path and no off-path item exists on this map",
                                                           "action": f"error -> serve_raw({comp})"})
                    changes.append((c["name"], comp, wrong, f"serve_raw({comp})"))
                else:
                    s.setdefault("curation_v5", []).append({"issue": f"mis_prep {comp}->{wrong} is on-path and no replacement exists", "action": "kept"})
                    changes.append((c["name"], comp, wrong, None))
                continue
            new = cands[0]
            s["kwargs"] = dict(kw, wrong=new)
            s.setdefault("curation_v5", []).append({"issue": f"mis_prep {comp}->{wrong} processes {wrong} on its own path (never commits)",
                                                   "action": f"wrong -> {new}"})
            changes.append((c["name"], comp, wrong, new))
    pool["curation_v5"] = {"source": a.src, "changes": [{"case": n, "component": c_, "old": o, "new": w} for n, c_, o, w in changes]}
    json.dump(pool, open(os.path.join(ROOT, a.dst), "w"), indent=1)
    for n, c_, o, w in changes:
        print(f"{n}: {c_}->{o}  =>  {w}")
    print(f"wrote {a.dst} with {len(changes)} mis_prep changes")


if __name__ == "__main__":
    main()
