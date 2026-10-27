#!/usr/bin/env python3
"""Case sheets for the three 9x9 braided-maze HARD cells, all nine detection labels.

    COOKSIM_GRID=tmp/grid.json python -m cooksim_bench.suite.gen_hardmap_cases \
        --out tmp/interact_eval/cases_hardmaps.json \
        --ledger tmp/interact_eval/cases_hardmaps_ledger.md

This is the gen_detect_cases_v2 pipeline, not a parallel implementation of it: the
candidate enumeration, the dry replay, the purity gate, the identifiability tick and the
window arithmetic are all IMPORTED from that module and called unchanged. Three things are
supplied here that the module cannot know:

  1. THE CELLS. `gen_detect_cases_v2.CELLS` is a hardcoded registry and the two new hard
     cells are not in it. They are injected into the registry at runtime rather than edited
     into the file, because running that module's own main() rewrites the shared
     tmp/interact_eval/cases_v2.json pool that the three published pages were generated
     from. Nothing shared is touched.

  2. A REAL off-recipe source for add_foreign / misidentify_assemble. The module gates both
     behind `CELLS[cell][1]` (the lettuce+trash INJECTION) and hardcodes wrong="lettuce".
     These maps carry six new distractor ingredients as REAL sources, so the foreign item
     no longer has to be injected — and on the burrito cell lettuce is ON-recipe, so the
     hardcoded choice would not even be foreign. Each cell picks a genuine off-recipe
     source, preferring one of the six new distractors so the new meshes actually appear
     on camera.

  3. ONE case per (label x cell). The module selects PER_TYPE=6 per label round-robin
     across ten cells, which is the wrong shape for a per-cell sweep.

Every (label, cell) pair that yields no admissible case is written to the ledger with the
replay's own rejection reason. Nothing is dropped silently.
"""
from __future__ import annotations
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, collections, hashlib, json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

from cooksim_bench.suite import gen_detect_cases_v2 as G  # noqa: E402
from cooksim_bench.exp_interact import build_episode, recipe_of                  # noqa: E402
from cooksim.core.plan_ir import segment_plan                      # noqa: E402

# The three target cells and their case-name shorthands. The second tuple element is the
# module's `inject` flag: FALSE everywhere here — these maps already carry real off-recipe
# sources, so injecting the synthetic lettuce+trash pair would add a source the map design
# did not ask for.
NEW_CELLS = {
    "hard_nops_hard_map":   ("hardM9", False),
    "expert_nops_hard_map": ("expM9", False),
    "master_nops_hard_map": ("masM9", False),
}
NEW_DISTRACTORS = ("lemon", "peas", "butter", "pepper", "avocado", "corn")

# The nine detection labels (the module's fn vocabulary after DETECT_LABEL folding).
LABELS = ["mis_prep", "wrong_order", "omit_subgoal", "serve_raw", "add_foreign",
          "repeat_step", "forget_next_step", "leave_unattended"]


def sources_of(cell):
    """Every *_source on the map, and which of them are off-recipe."""
    _st, ro = build_episode(None, "capable_collaborator", cell=cell, inject=False)
    terr = ro.game.render_state()["terrain"]
    names = collections.Counter()
    for row in terr:
        for t in row:
            if isinstance(t, str) and t.endswith("_source"):
                names[t[:-7]] += 1
    rec = recipe_of(cell)
    on = {n for n, _ in rec.contents}
    off = [n for n in names if n not in on and n != "plate"]
    # prefer a NEW distractor so the new meshes reach the camera; stable order otherwise
    off.sort(key=lambda n: (n not in NEW_DISTRACTORS, n))
    return names, on, off


def candidates_for(cell, off_recipe):
    """The module's enumeration for one cell, plus the two foreign-item behaviours that its
    injection gate would otherwise suppress on a map that has real off-recipe sources."""
    mine = [(e, k, c) for (e, k, c) in G.candidates() if c == cell]
    if not off_recipe:
        return mine
    rec = recipe_of(cell)
    plan = list(segment_plan(rec))
    sids = [n.sid for n in plan]
    asm = [n.sid for n in plan if n.type == "assemble"]
    comps = [n.component for n in plan if n.component]
    wrong = off_recipe[0]
    # add_foreign: scoop an off-recipe item onto the plate before each assemble, and before serve
    for before in asm + ["serve"]:
        mine.append(("add_foreign", {"wrong": wrong, "before": before}, cell))
    # misidentify_assemble (graded add_foreign): replace a component with the foreign item
    for comp in dict.fromkeys(comps):
        if f"assemble({comp})" in sids:
            mine.append(("misidentify_assemble", {"component": comp, "wrong": wrong}, cell))
    return mine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/interact_eval/cases_hardmaps.json")
    ap.add_argument("--ledger", default="tmp/interact_eval/cases_hardmaps_ledger.md")
    a = ap.parse_args()

    G.CELLS.update(NEW_CELLS)                     # runtime only; the module file is untouched

    picked, rejected = [], []
    srcinfo = {}
    for cell in NEW_CELLS:
        names, on, off = sources_of(cell)
        srcinfo[cell] = (names, on, off)
        print(f"=== {cell}: on-recipe={sorted(on)} off-recipe={off}", flush=True)
        for err, kwargs, c in candidates_for(cell, off):
            try:
                st = G.replay(err, kwargs, c, NEW_CELLS[cell][1])
            except Exception as e:                                 # noqa: BLE001
                rejected.append((err, kwargs, cell, f"replay error: {e!r}"))
                continue
            why = None
            if st["outcome"] not in G.EXPECT[err]:
                why = f"outcome {st['outcome']} not in {sorted(G.EXPECT[err])}"
            elif st["dev"] is None:
                why = "no deviation anchor (injection never fires)"
            elif err != "leave_unattended" and st["fire_tick"] is not None:
                why = f"secondary hazard: reached FIRE at t{st['fire_tick']}"
            elif (err == "leave_unattended" and st["fire_tick"] is None
                  and st["outcome"] != "burned"):
                # user ruling 2026-08-14: the terminal burn IS the manifested hazard —
                # the window then runs to the burn tick (outcome_tick), not to fire
                why = "never reached the hazard (no fire, no terminal burn)"
            elif st.get("dev_failed"):
                why = "deviated block failed to execute (illegal-skipped/rejected)"
            elif st["identifiable"] is None:
                why = "no identifiability tick (evidence moment absent from the log)"
            if why is None and err == "omit_subgoal":
                rec = recipe_of(cell)
                asm = [f"assemble({n})" for n, _ in rec.contents]
                if kwargs.get("sid") == asm[-1]:
                    why = "observational twin: omit(last assemble) == premature_serve forever"
            if why:
                rejected.append((err, kwargs, cell, why))
                continue
            ident = st["identifiable"]
            whi = ((st["fire_tick"] if st["fire_tick"] is not None else st["outcome_tick"])
                   if err == "leave_unattended" else ident + G.WINDOW)
            picked.append(dict(err=G.DETECT_LABEL.get(err, err), fn=err, kwargs=kwargs,
                               cell=cell, inject=NEW_CELLS[cell][1],
                               persona="capable_collaborator", dev=st["dev"],
                               identifiable=ident, window=[ident - 2, whi],
                               outcome=st["outcome"], outcome_tick=st["outcome_tick"],
                               end=st["end"], max_smoke=st["max_smoke"]))
        print(f"    admissible so far: {sum(1 for p in picked if p['cell']==cell)}", flush=True)

    # ---- one case per (label x cell): earliest identifiable tick wins (most episode left
    # to run after the evidence appears), which is also the most gradable.
    final, ledger_missing = [], []
    for cell in NEW_CELLS:
        for lab in LABELS:
            pool = [p for p in picked if p["cell"] == cell and p["err"] == lab]
            if not pool:
                reasons = collections.Counter(
                    w for (e, k, c, w) in rejected
                    if c == cell and G.DETECT_LABEL.get(e, e) == lab)
                ledger_missing.append((cell, lab, reasons))
                continue
            pool.sort(key=lambda p: (p["identifiable"], json.dumps(p["kwargs"], sort_keys=True)))
            best = dict(pool[0])
            best["name"] = f"hm_{lab}_{NEW_CELLS[cell][0]}"
            best["family"] = "error"
            final.append(best)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    blob = {"schema": "hardmap-cases-v1", "cells": list(NEW_CELLS),
            "labels": LABELS, "cases": final}
    json.dump(blob, open(a.out, "w"), indent=1)

    # ---- the written ledger
    L = ["# Admissibility ledger — nine detection labels x three 9x9 braided-maze cells",
         "", f"Generated by `python -m cooksim_bench.suite.gen_hardmap_cases` on engine "
         f"`{os.popen('python -m cooksim_bench.engine_stamp').read().strip() or '?'}`.",
         "Every candidate was dry-replayed through `gen_detect_cases_v2.replay` and kept "
         "only if it passed that module's own purity gate. A (label, cell) pair with no "
         "surviving candidate is listed below with the replay's reason — none is dropped "
         "silently.", ""]
    L.append("## Map sources")
    L.append("")
    L.append("| cell | recipe | ordered | on-recipe | off-recipe sources (foreign-item pool) |")
    L.append("|---|---|---|---|---|")
    for cell in NEW_CELLS:
        names, on, off = srcinfo[cell]
        rec = recipe_of(cell)
        dup = [n for n in on if names.get(n, 0) > 1]
        L.append(f"| `{cell}` | {rec.name} | {rec.ordered} | {', '.join(sorted(on))} "
                 f"(duplicated: {', '.join(sorted(dup)) or 'none'}) | {', '.join(off)} |")
    L.append("")
    L.append("## Admitted cases")
    L.append("")
    L.append("| cell | label | realization (fn) | kwargs | dev | identifiable | window | outcome | end |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in final:
        L.append(f"| `{c['cell']}` | {c['err']} | {c['fn']} | `{json.dumps(c['kwargs'])}` | "
                 f"{c['dev']} | {c['identifiable']} | {c['window']} | {c['outcome']} | {c['end']} |")
    L.append("")
    L.append(f"**{len(final)} admitted** of {len(LABELS) * len(NEW_CELLS)} (label x cell) pairs.")
    L.append("")
    L.append("## Inadmissible pairs, with reasons")
    L.append("")
    if not ledger_missing:
        L.append("None — every label is admissible on every cell.")
    for cell, lab, reasons in ledger_missing:
        L.append(f"### `{cell}` x **{lab}** — INADMISSIBLE")
        if not reasons:
            L.append("")
            L.append("- no candidate was even enumerated for this label on this cell "
                     "(the recipe offers no anchor for it — e.g. `wrong_order` requires an "
                     "ORDERED recipe, `misidentify`/`add_foreign` require an off-recipe source).")
        else:
            L.append("")
            for why, n in reasons.most_common():
                L.append(f"- {n} candidate(s) rejected: {why}")
        L.append("")
    L.append("## All rejected candidates (full audit trail)")
    L.append("")
    L.append("| cell | fn | kwargs | reason |")
    L.append("|---|---|---|---|")
    for e, k, c, w in rejected:
        L.append(f"| `{c}` | {e} | `{json.dumps(k)}` | {w} |")
    open(a.ledger, "w").write("\n".join(L) + "\n")

    print(f"\nwrote {a.out} ({len(final)} cases) and {a.ledger} "
          f"({len(ledger_missing)} inadmissible pairs, {len(rejected)} rejected candidates)")
    for cell in NEW_CELLS:
        got = [c["err"] for c in final if c["cell"] == cell]
        print(f"  {cell}: {len(got)}/9  {sorted(got)}")


if __name__ == "__main__":
    main()
