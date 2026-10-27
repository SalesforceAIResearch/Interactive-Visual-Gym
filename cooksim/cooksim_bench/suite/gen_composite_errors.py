#!/usr/bin/env python3
"""Expanded, replay-verified ERROR case pool over ALL TEN benchmark cells.

    COOKSIM_GRID=tmp/grid.json python -m cooksim_bench.suite.gen_composite_errors \
        --out tmp/interact_eval/cases_composite_errors.json \
        --ledger tmp/interact_eval/cases_composite_errors_ledger.md

This is the gen_detect_cases_v2 pipeline widened to the full controller_eval.CELLS10
matrix, not a parallel implementation: candidate enumeration, dry replay, purity gate,
identifiability tick and window arithmetic are IMPORTED from gen_detect_cases_v2 (and the
hard-map source discovery / foreign-item enumeration from gen_hardmap_cases) and called
unchanged. Neither module file is modified; the two runtime-only additions are

  1. THE CELLS. `gen_detect_cases_v2.CELLS` lacks the two 9x9 braided cells
     (hard_nops_hard_map, expert_nops_hard_map); `gen_hardmap_cases.NEW_CELLS` is merged
     into the registry at runtime, exactly as that script does for its own run.

  2. UP TO THREE cases per (label x cell), chosen for ANCHOR DIVERSITY: where a label has
     admissible candidates at several distinct window starts (window[0] = the first tick
     the assistant may be credited), the three picks are spread across the distinct
     anchors — earliest, middle, latest — so a composite episode can stage the same label
     at different plate stages. gen_hardmap_cases keeps only the single earliest.

The purity gate is the UNION of the two parents' gates: gen_hardmap_cases' outcome /
hazard / dev_failed / identifiability / omit-last-twin checks PLUS gen_detect_cases_v2's
E1 anchor-collapse check (a deviation that precedes all plate work has no admissible
injection anchor — the splice would fire at t0 and amputate the prep prefix).

Every (label, cell) pair that yields no admissible case is written to the ledger with a
one-line reason — a structural one when nothing could even be enumerated (wrong_order
needs an ORDERED recipe; add_foreign/misidentify need an off-recipe source), otherwise
the replay's own most common rejection. Nothing is dropped silently.
-> tmp/interact_eval/cases_composite_errors.json
"""
from __future__ import annotations
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, collections, json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

from cooksim_bench.suite import gen_detect_cases_v2 as G  # noqa: E402
from cooksim_bench.suite import gen_hardmap_cases as H  # noqa: E402
from cooksim_bench.exp_interact import recipe_of                                 # noqa: E402
from cooksim.core.plan_ir import segment_plan                      # noqa: E402

# The ten benchmark cells (controller_eval.CELLS10) and this pool's name shorthands.
CELLS10 = [f"{n}_nops_{m}_map"
           for n in ("easy", "medium", "hard", "expert", "master")
           for m in ("medium", "hard")]
ABBREV = {"easy_nops_medium_map": "easyM",   "easy_nops_hard_map": "easyH",
          "medium_nops_medium_map": "medM",  "medium_nops_hard_map": "medH",
          "hard_nops_medium_map": "hardM",   "hard_nops_hard_map": "hardH",
          "expert_nops_medium_map": "expM",  "expert_nops_hard_map": "expH",
          "master_nops_medium_map": "masM",  "master_nops_hard_map": "masH"}
LABELS = H.LABELS            # the nine detection labels after DETECT_LABEL folding
PER_LABEL = 3
N_WRONGS = 2                 # distinct foreign items to try per hard cell (pool width)


def foreign_extras(cell, off_recipe):
    """add_foreign / misidentify_assemble candidates for a map with REAL off-recipe
    sources (gen_hardmap_cases.candidates_for enumeration, widened to N_WRONGS distinct
    foreign items so the selection has anchor- and item-diverse material)."""
    out = []
    if not off_recipe:
        return out
    rec = recipe_of(cell)
    plan = list(segment_plan(rec))
    sids = [n.sid for n in plan]
    asm = [n.sid for n in plan if n.type == "assemble"]
    comps = [n.component for n in plan if n.component]
    for wrong in off_recipe[:N_WRONGS]:
        for before in asm + ["serve"]:
            out.append(("add_foreign", {"wrong": wrong, "before": before}, cell))
        for comp in dict.fromkeys(comps):
            if f"assemble({comp})" in sids:
                out.append(("misidentify_assemble", {"component": comp, "wrong": wrong}, cell))
    return out


def gate(err, kwargs, cell, st):
    """Rejection reason, or None if admissible. Union of the two parents' gates.

    The anchor-collapse check mirrors the LIVE harness backstop exactly
    (e2e_stepin_rollout: `anchor_collapsed = bool(m > 0 and ...)`): a case whose
    deviation is the very first command (dev == 0) has an EMPTY clean prefix — the
    injection at t0 is precisely the anchor the case describes and nothing is
    amputated, so the harness runs it. Only a dev > 0 case whose prefix does no
    plate work is refused live, and only those are rejected here. (G.replay's
    design-time probe marks dev == 0 as collapsed trivially — its while-loop never
    runs — which is stricter than the harness; gen_hardmap_cases skipped the check
    entirely, which is looser: it published serve_raw sheets with dev > 0 and no
    prior plate work that the live harness refuses.)"""
    if st.get("anchor_collapses") and st.get("dev"):
        return "anchor collapses to t0 (no plate event before the dev>0 deviation)"
    if st["outcome"] not in G.EXPECT[err]:
        return f"outcome {st['outcome']} not in {sorted(G.EXPECT[err])}"
    if st["dev"] is None:
        return "no deviation anchor (injection never fires)"
    if err != "leave_unattended" and st["fire_tick"] is not None:
        return f"secondary hazard: reached FIRE at t{st['fire_tick']}"
    if (err == "leave_unattended" and st["fire_tick"] is None
            and st["outcome"] != "burned"):
        return "never reached the hazard (no fire, no terminal burn)"
    if st.get("dev_failed"):
        return "deviated block failed to execute (illegal-skipped/rejected)"
    if st["identifiable"] is None:
        return "no identifiability tick (evidence moment absent from the log)"
    if err == "omit_subgoal":
        rec = recipe_of(cell)
        asm = [f"assemble({n})" for n, _ in rec.contents]
        if kwargs.get("sid") == asm[-1]:
            return "observational twin: omit(last assemble) == premature_serve forever"
    if err == "forget_next_step":
        # DESIGN RULE (user, 2026-09-15): forget_next_step never forgets the SERVE. A cook standing next to a finished plate that it 'forgot' to hand in is not a plausible slip; the forgotten step must be a real production or plating step (a prep, an ignition, fetching the plate, a non-final assemble). On hot recipes a freeze with food on the heat burns — that is leave_unattended, not a forget — so where a cell has no admissible mid-recipe point the error is simply unavailable there; never fall back to the point before serve.
        from cooksim.core.plan_ir import segment_plan
        sids = [n.sid for n in segment_plan(recipe_of(cell))]
        if kwargs.get("after") in ("serve", sids[sids.index("serve") - 1] if "serve" in sids else None):
            return "design rule: forget_next_step would forget the serve (a finished plate is not forgotten)"
    return None


def structural_reason(cell, lab, enumerated):
    """One-line reason when a label produced NO candidate at all on a cell."""
    rec = recipe_of(cell)
    inject = G.CELLS[cell][1]
    if lab == "wrong_order" and not rec.ordered:
        return "recipe is UNORDERED — wrong_order is undefined (no order to violate)"
    if lab == "add_foreign" and not inject:
        return ("no off-recipe source on this map and no lettuce+trash injection — "
                "nothing foreign exists to add")
    if lab == "leave_unattended" and not any(s == "cooked" for _, s in rec.contents):
        return "recipe has no cooked component — nothing can be left on heat"
    if lab == "serve_raw" and not any(s in ("cooked", "chopped") for _, s in rec.contents):
        return "recipe has no cooked/chopped component — raw IS the correct state"
    if lab == "mis_prep" and not any(n.type == "prep" for n in segment_plan(rec)):
        return "recipe has no prep step — nothing to mis-prep"
    return f"no candidate enumerated for {lab} on this recipe/map"


def spread_pick(pool):
    """Up to PER_LABEL cases, anchors (window[0]) spread earliest/middle/latest.
    Within one anchor the earliest-identifiable, kwargs-stable candidate wins; leftover
    slots (fewer distinct anchors than PER_LABEL) fill with distinct-kwargs candidates."""
    key = lambda p: (p["identifiable"], json.dumps(p["kwargs"], sort_keys=True))
    by_anchor = {}
    for p in sorted(pool, key=key):
        by_anchor.setdefault(p["window"][0], []).append(p)
    anchors = sorted(by_anchor)
    k = min(PER_LABEL, len(anchors))
    idx = [0] if k == 1 else [round(i * (len(anchors) - 1) / (k - 1)) for i in range(k)]
    sel = [by_anchor[anchors[i]][0] for i in dict.fromkeys(idx)]
    if len(sel) < PER_LABEL:                      # fewer anchors than slots: vary kwargs
        seen = {json.dumps(p["kwargs"], sort_keys=True) for p in sel}
        for p in sorted(pool, key=key):
            kk = json.dumps(p["kwargs"], sort_keys=True)
            if kk not in seen:
                sel.append(p); seen.add(kk)
            if len(sel) == PER_LABEL:
                break
    return sorted(sel, key=lambda p: p["window"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/interact_eval/cases_composite_errors.json")
    ap.add_argument("--ledger", default="tmp/interact_eval/cases_composite_errors_ledger.md")
    a = ap.parse_args()

    G.CELLS.update(H.NEW_CELLS)                   # runtime only; the module file is untouched

    base = [(e, k, c) for (e, k, c) in G.candidates() if c in CELLS10]
    srcinfo, cands = {}, []
    for cell in CELLS10:
        names, on, off = H.sources_of(cell)
        srcinfo[cell] = (names, on, off)
        cands += [(e, k, c) for (e, k, c) in base if c == cell]
        cands += foreign_extras(cell, off)
    enumerated = collections.Counter((c, G.DETECT_LABEL.get(e, e)) for e, k, c in cands)
    print(f"{len(cands)} candidates over {len(CELLS10)} cells", flush=True)

    picked, rejected = [], []
    for i, (err, kwargs, cell) in enumerate(cands, 1):
        inject = G.CELLS[cell][1]
        try:
            st = G.replay(err, kwargs, cell, inject)
        except Exception as e:                                     # noqa: BLE001
            rejected.append((err, kwargs, cell, f"replay error: {e!r}"))
            continue
        finally:
            if i % 25 == 0:
                print(f"  [{i}/{len(cands)}] replayed; {len(picked)} admissible", flush=True)
        why = gate(err, kwargs, cell, st)
        if why:
            rejected.append((err, kwargs, cell, why))
            continue
        ident = st["identifiable"]
        whi = ((st["fire_tick"] if st["fire_tick"] is not None else st["outcome_tick"])
               if err == "leave_unattended" else ident + G.WINDOW)
        picked.append(dict(err=G.DETECT_LABEL.get(err, err), fn=err, kwargs=kwargs,
                           cell=cell, inject=inject, persona="capable_collaborator",
                           dev=st["dev"], identifiable=ident, window=[ident - 2, whi],
                           outcome=st["outcome"], outcome_tick=st["outcome_tick"],
                           end=st["end"], max_smoke=st["max_smoke"]))

    # ---- selection: up to PER_LABEL per (cell x label), anchors spread
    final, missing = [], []
    for cell in CELLS10:
        for lab in LABELS:
            pool = [p for p in picked if p["cell"] == cell and p["err"] == lab]
            if not pool:
                if enumerated[(cell, lab)] == 0:
                    reason = structural_reason(cell, lab, enumerated)
                else:
                    top = collections.Counter(
                        w for (e, k, c, w) in rejected
                        if c == cell and G.DETECT_LABEL.get(e, e) == lab).most_common(1)
                    reason = (f"all {enumerated[(cell, lab)]} candidates rejected — "
                              f"most common: {top[0][0]}")
                missing.append((cell, lab, reason))
                continue
            for n, c in enumerate(spread_pick(pool), 1):
                c = dict(c)
                c["name"] = f"cx_{lab}_{ABBREV[cell]}{n}"
                c["family"] = "error"
                final.append(c)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    blob = {"schema": "composite-errors-v1", "cells": CELLS10, "labels": LABELS,
            "cases": final}
    json.dump(blob, open(a.out, "w"), indent=1)

    # ---- ledger
    stamp = os.popen(f"{sys.executable} python -m cooksim_bench.engine_stamp").read().strip() or "?"
    L = ["# Composite-error case pool — nine detection labels x ten benchmark cells",
         "",
         f"Generated by `python -m cooksim_bench.suite.gen_composite_errors` on engine `{stamp}`.",
         "Every candidate was dry-replayed through `gen_detect_cases_v2.replay` and kept "
         "only if it passed the union of that module's and `gen_hardmap_cases`' purity "
         "gates (expected-outcome purity, no secondary hazard, deviation anchor exists "
         "and does not collapse to t0, identifiability tick found). Up to three cases "
         "per (label, cell), anchors spread across distinct window starts so composite "
         "episodes can stage the same label at different plate stages. Every infeasible "
         "(label, cell) pair is listed with a reason — nothing is dropped silently.", ""]
    for cell in CELLS10:
        names, on, off = srcinfo[cell]
        rec = recipe_of(cell)
        L += [f"## `{cell}`", "",
              f"Recipe **{rec.name}** (ordered={rec.ordered}); "
              f"off-recipe sources: {', '.join(off) or 'none'}; "
              f"injection: {G.CELLS[cell][1]}.", "",
              "| label | cases | anchors (window[0]) | names |", "|---|---|---|---|"]
        for lab in LABELS:
            cs = [c for c in final if c["cell"] == cell and c["err"] == lab]
            if cs:
                L.append(f"| {lab} | {len(cs)} | "
                         f"{', '.join(str(c['window'][0]) for c in cs)} | "
                         f"{', '.join(c['name'] for c in cs)} |")
        infeas = [(l, r) for (c, l, r) in missing if c == cell]
        if infeas:
            L += ["", "Infeasible labels:", ""]
            for lab, reason in infeas:
                L.append(f"- **{lab}** — {reason}")
        L.append("")
    L += ["## All rejected candidates (full audit trail)", "",
          "| cell | fn | kwargs | reason |", "|---|---|---|---|"]
    for e, k, c, w in rejected:
        L.append(f"| `{c}` | {e} | `{json.dumps(k)}` | {w} |")
    open(a.ledger, "w").write("\n".join(L) + "\n")

    # ---- verification + summary table
    chk = json.load(open(a.out))
    assert all(c["cell"] in CELLS10 for c in chk["cases"]), "case with off-matrix cell"
    assert all(c["err"] in LABELS for c in chk["cases"])
    print(f"\nwrote {a.out} ({len(final)} cases) and {a.ledger} "
          f"({len(missing)} infeasible pairs, {len(rejected)} rejected candidates)")
    w = max(len(c) for c in CELLS10)
    head = " ".join(f"{l[:9]:>9}" for l in LABELS)
    print(f"{'cell':<{w}} {head}  total")
    for cell in CELLS10:
        row = [sum(1 for c in final if c["cell"] == cell and c["err"] == l) for l in LABELS]
        print(f"{cell:<{w}} " + " ".join(f"{n:>9}" for n in row) + f"  {sum(row):>5}")
    tot = [sum(1 for c in final if c["err"] == l) for l in LABELS]
    print(f"{'TOTAL':<{w}} " + " ".join(f"{n:>9}" for n in tot) + f"  {sum(tot):>5}")
    nowhere = [l for l, n in zip(LABELS, tot) if n == 0]
    if nowhere:
        print(f"labels infeasible EVERYWHERE: {nowhere}")


if __name__ == "__main__":
    main()
