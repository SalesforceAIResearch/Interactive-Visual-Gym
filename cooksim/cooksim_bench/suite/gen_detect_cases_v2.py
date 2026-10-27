#!/usr/bin/env python3
"""Generate DEDICATED, confound-free detection cases per error type (detection sweep v2).

The v1 matrix reused interaction cases; some were impure — e.g. a forget freeze while meat
cooked also produced real SMOKE, so leave_unattended became a second correct answer and the
exact-label timing test lost its meaning. Here every candidate anchor is DRY-REPLAYED and
kept only if it is pure:
  - the episode ends in the outcome class the behavior is DESIGNED to cause;
  - no secondary hazard: a non-leave_unattended candidate is killed only if it reaches
    FIRE. (Until 2026-08-03 this rejected any smoke run of SMOKE_OK+ consecutive ticks.
    That is too strict: smoke is the WARNING phase — burn_time=30 ticks of it must pass
    before fire_time even starts — and a pan that smokes and is then scooped harms
    nothing, leaves the dish correct and the episode won. It also silently disqualified
    whole cells: the medium-map Cheeseburger clean plan routes the cook plates -> bun ->
    back, so the patty sits cooked for 16 ticks and EVERY error on the benchmark's anchor
    cell failed this check, which is why wrong_order had to be staged on the easy map.
    Fire is the real confound, because only then is leave_unattended a second true label.)
  - leave_unattended must reach the hazard: FIRE, or the terminal BURN that since W22
    ends the episode first (window runs to whichever tick manifests);
  - the timing window [dev-2, dev+30] is well-defined (dev anchored per behavior).
Selection keeps up to PER_TYPE cases per error, preferring anchor/recipe diversity.
-> tmp/interact_eval/cases_v2.json
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.exp_interact import build_episode, recipe_of
from cooksim.core.plan_ir import segment_plan

PER_TYPE = 6
# Detection LABEL vocabulary (10 names): the misidentify_assemble REALIZATION (replace
# physics) is graded as add_foreign — at the moment the foreign scoop happens the two are
# observationally identical, and flags are graded at the moment they are raised.
DETECT_LABEL = {"misidentify_assemble": "add_foreign",
                "premature_serve": "omit_subgoal"}  # tail-truncation = omission of the tail
SMOKE_OK = 6            # max tolerated consecutive smoking ticks for non-safety errors
WINDOW = 30

EXPECT = {
    "repeat_step": {"wrong_serve", "won"}, "premature_serve": {"wrong_serve"},
    "forget_next_step": {"timeout"}, "omit_subgoal": {"wrong_serve"},
    "wrong_order": {"wrong_serve"}, "misidentify_assemble": {"wrong_serve"},
    "mis_prep": {"wrong_serve"}, "add_foreign": {"wrong_serve"},
    # leave_unattended: since W22 made `burned` TERMINAL the episode ends at the burn,
    # strictly before fire — and detection after either is pointless (neither is
    # recoverable). User ruling 2026-08-14: the terminal burn IS the manifested hazard,
    # so `burned` is accepted alongside the (late-ignite-only) `fire`.
    "serve_raw": {"wrong_serve"},
    "leave_unattended": {"burned", "fire"},
}

CELLS = {"medium_nops_medium_map": ("med", True),      # Cheeseburger, ordered (lettuce+trash injected)
         "medium_nops_easy_map": ("medE", False),      # Cheeseburger on the easy map
         "medium_nops_hard_map": ("medH", False),      # Cheeseburger on the hard map
         "easy_nops_easy_map": ("easy", False),        # Caprese, unordered, NO heat at all
         "easy_nops_medium_map": ("easyM", False),     # Caprese on the medium map
         "easy_nops_hard_map": ("easyH", False),       # Caprese on the hard map
         "hard_nops_medium_map": ("hard", False),      # Pizza (baseline-smoky; leave_unattended only)
         "expert_nops_medium_map": ("exp", False),     # Steak frites, unordered (no foreign source)
         "master_nops_medium_map": ("mas", False),     # Loaded burrito, ordered (lettuce ON-recipe -> no inject)
         "master_nops_hard_map": ("masH", False)}      # Loaded burrito on the hard map


def candidates():
    """(err, kwargs, cell) candidates enumerated from each recipe's own plan."""
    C = []
    for cell in CELLS:
        try:
            rec = recipe_of(cell)
        except StopIteration:
            continue                     # cell absent from the active COOKSIM_GRID

        plan = list(segment_plan(rec))
        sids = [n.sid for n in plan]
        asm = [n.sid for n in plan if n.type == "assemble"]
        comps = {n.component: dict(rec.contents).get(n.component) for n in plan if n.component}
        cooked = [c for c, st in rec.contents if st == "cooked"]
        prepped = [n.component for n in plan if n.type == "prep"]
        others = list(comps)
        # repeat_step: only sids whose re-execution is PHYSICALLY VISIBLE. An assemble that
        # scoops from a STATION (chopped/cooked item) cannot repeat — the station is empty
        # after the first scoop, so the dup block just stands there waiting (looks exactly
        # like forget_next_step). Crate-source assembles (raw items) and preps re-execute.
        raw_asm = [f"assemble({n})" for n, st in rec.contents if st == "raw"]
        for i, s in enumerate(asm):
            if s not in raw_asm:
                continue
            for at in asm[i + 1:]:
                C.append(("repeat_step", {"sid": s, "at": at}, cell))
        if prepped:
            C.append(("repeat_step", {"sid": f"prep({prepped[0]})", "at": asm[-1]}, cell))
        # premature_serve: truncate after any pre-final subgoal that leaves the plate incomplete
        for after in ["fetch_plate"] + asm[:-1]:
            C.append(("premature_serve", {"after": after}, cell))
        # forget_next_step: freeze after any subgoal (validator kills smoke-confounded anchors)
        for after in ["fetch_plate"] + asm:
            if after != asm[-1] or True:
                C.append(("forget_next_step", {"after": after}, cell))
        # omit_subgoal: any assemble. (Omit-of-last used to be excluded as a twin of
        # premature_serve; with premature MERGED into omit_subgoal the twin dissolves —
        # both realizations legitimately carry the same label.)
        for sid in asm:
            C.append(("omit_subgoal", {"sid": sid}, cell))
        # wrong_order: ordered recipes only
        if rec.ordered:
            names = [s[9:-1] for s in asm]
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    C.append(("wrong_order", {"a": a, "b": b}, cell))
        # misidentify realization (replace physics; GRADED as add_foreign): only with an
        # OFF-RECIPE wrong — a recipe-item wrong is an immediate observational twin of repeat_step
        if CELLS[cell][1]:
            for comp in comps:
                if f"assemble({comp})" in sids:
                    C.append(("misidentify_assemble", {"component": comp, "wrong": "lettuce"}, cell))
        # mis_prep: wrong thing prepped at the same station
        for comp in prepped:
            wrongs = (["lettuce"] if CELLS[cell][1] else []) + [w for w in others if w != comp]
            for w in wrongs[:3]:
                C.append(("mis_prep", {"component": comp, "wrong": w}, cell))
        # add_foreign: needs an off-recipe source -> injected map only
        if CELLS[cell][1]:
            for before in asm + ["serve"]:
                C.append(("add_foreign", {"wrong": "lettuce", "before": before}, cell))
        # serve_raw: any component the recipe wants cooked/chopped, plated straight from source
        for comp, st in rec.contents:
            if st in ("cooked", "chopped"):
                C.append(("serve_raw", {"component": comp}, cell))
        # leave_unattended: any cooked component, escalating busywork
        for comp in cooked:
            for loops in (1, 2, 3):
                C.append(("leave_unattended", {"component": comp, "loops": loops}, cell))
    return C


def identifiable_tick(err, kwargs, state, ro, smoke_start):
    """The first tick the error is UNIQUELY identifiable — later than dev_tick (command
    issue) and often later than mere arrival, because plan-edit errors have observational
    TWINS early on: swap-vs-skip at the first out-of-order arrival, replace-vs-insert until
    the move AFTER the foreign scoop. The GT window must not demand the exact name before
    the sibling hypotheses are ruled out."""
    log, plan = ro.log, [n.sid for n in state.clean_ir]

    def arrive(pred, after=-1):
        return next((l["t_done"] for l in log if pred(l)
                     and l.get("t_done") is not None and l["t_done"] > after), None)
    if err in ("misidentify_assemble", "add_foreign"):
        # replace vs insert disambiguate at the NEXT move after the foreign scoop
        sc = arrive(lambda l: l["kind"] in ro.dev_sids and l["cmd"] == "scoop")
        t = arrive(lambda l: l["cmd"].startswith("go to"), after=sc or 10**9)
        t = (t + 1) if t is not None else (sc + 2 if sc else None)
    elif err == "serve_raw":
        t = arrive(lambda l: l["kind"] in ro.dev_sids and l["cmd"].startswith("go to"))
        t = t + 1 if t is not None else None
    elif err == "mis_prep":
        t = arrive(lambda l: l["kind"] in ro.dev_sids and l["cmd"] == "put down")
        t = t + 1 if t is not None else None
    elif err == "wrong_order":
        gotos = [l["t_done"] for l in log if l["kind"] in ro.dev_sids
                 and l["cmd"].startswith("go to") and l.get("t_done") is not None]
        t = gotos[1] + 1 if len(gotos) > 1 else (gotos[0] + 1 if gotos else None)
    elif err == "omit_subgoal":
        i = plan.index(kwargs["sid"])
        nxt2 = plan[i + 2] if i + 2 < len(plan) else "serve"
        t = arrive(lambda l: l["kind"] == nxt2 and l["cmd"].startswith("go to"))
        t = t + 1 if t is not None else None
    elif err == "repeat_step":
        t = arrive(lambda l: l["kind"] in ro.dev_sids and l["cmd"].startswith("go to"))
        t = t + 1 if t is not None else None
    elif err == "premature_serve":
        t = arrive(lambda l: l["cmd"].startswith("go to the serving"))
        t = t + 1 if t is not None else None
    elif err == "forget_next_step":
        t = (ro.dev_tick + 10) if ro.dev_tick is not None else None
    elif err == "leave_unattended":
        t = (smoke_start + 12) if smoke_start else None
    else:
        t = None
    return t


def replay(err, kwargs, cell, inject):
    state, ro = build_episode((err, dict(kwargs)) if err else None, "capable_collaborator",
                              cell=cell, inject=inject)
    # E8 (hard-map audit 2026-08-15): validate sheets under the SAME physics the live
    # episodes run under. The live harness relaxes burn/fire for human-paced play; this
    # replay used to run default timings, so a sheet could certify an outcome the live
    # run can never reach (every leave_unattended sheet declares `burned`, which at the
    # relaxed burn time needs ~600 ticks). Opt-in until the leave_unattended family is
    # re-designed for the relaxed physics — set COOKSIM_CASE_HUMAN_PACE=1 to enable.
    if os.environ.get("COOKSIM_CASE_HUMAN_PACE") == "1":
        from cooksim.core.config import apply_human_pace
        apply_human_pace(ro.game)
    smoke_run = max_smoke = 0
    fire_tick = smoke_start = None
    while ro.step():
        smoking = any(getattr(st, "smoking", False) for st in ro.game.stations.values())
        if smoking and smoke_start is None:
            smoke_start = ro.tick
        smoke_run = smoke_run + 1 if smoking else 0
        max_smoke = max(max_smoke, smoke_run)
        if fire_tick is None and any(getattr(st, "fire_elapsed", 0)
                                     for st in ro.game.stations.values()):
            fire_tick = ro.tick
    # ---- deviation anchor (window start)
    dev = ro.dev_tick
    if err == "omit_subgoal" and dev is None:
        # no deviated command executes; the observable moment is the plan STARTING the
        # subgoal after the omitted one (order visibly broken / item visibly missing)
        plan = [n.sid for n in segment_plan(state.recipe)]
        after = plan[plan.index(kwargs["sid"]) + 1]
        dev = next((l["t_issue"] for l in ro.log if l["kind"] == after), None)
    # ---- outcome-established tick (compute_outcome_ticks rules, condensed)
    def done(pred):
        return next((l["t_done"] for l in ro.log if l["status"] == "done" and pred(l)), None)
    if err in ("misidentify_assemble", "add_foreign", "serve_raw", "repeat_step", "wrong_order"):
        ot = done(lambda l: l["kind"] in ro.dev_sids and l["cmd"] == "scoop")
    elif err == "mis_prep":
        ot = done(lambda l: l["cmd"] == "scoop" and l["kind"].startswith("assemble("))
    elif err == "leave_unattended":
        ot = fire_tick
    else:
        ot = None
    if ot is None and ro.outcome != "won":
        ot = ro.tick
    ident = identifiable_tick(err, kwargs, state, ro, smoke_start) if err else None
    dev_failed = any(l["kind"] in ro.dev_sids and l["status"] in ("illegal-skipped", "rejected")
                     for l in ro.log)
    # E1 (user ruling 2026-08-15): does this case have an admissible ANCHOR? The harness
    # fires the injection on the first tick the live kitchen matches the plate-only
    # fingerprint of the reference replay at the trigger point. If no plate event happens
    # before the deviation, that fingerprint equals the EMPTY t0 kitchen, the injection
    # fires at tick 0 and the splice amputates the whole prep prefix. Detect it here, at
    # design time, by replaying the CLEAN plan to the deviation point and asking whether
    # the plate ever changed.
    anchor_collapses = False
    if dev is not None:
        from cooksim_bench.e2e_stepin_rollout import inject_key
        _st2, rro = build_episode(None, "capable_collaborator", cell=cell, inject=inject)
        t0_key = inject_key(rro.game, rro)
        while rro.tick < dev and rro.step():
            pass
        anchor_collapses = bool(inject_key(rro.game, rro) == t0_key)
    return dict(outcome=ro.outcome, dev=dev, identifiable=ident, dev_failed=dev_failed,
                outcome_tick=ot, anchor_collapses=anchor_collapses,
                max_smoke=max_smoke, fire_tick=fire_tick, end=ro.tick)


def main():
    picked, rejected = [], []
    per_type = {}
    for err, kwargs, cell in candidates():
        short, inject = CELLS[cell]
        try:
            st = replay(err, kwargs, cell, inject)
        except Exception as e:
            rejected.append((err, kwargs, cell, f"replay error: {e}"))
            continue
        why = None
        if st.get("anchor_collapses"):
            # E1 (user ruling 2026-08-15): a case whose deviation precedes ALL plate work
            # has no admissible anchor under the plate-only injection key — it would fire
            # at t0 and amputate the prep prefix, producing an unrunnable episode. Do not
            # design such cases; the harness refuses them as a backstop.
            why = "anchor collapses to t0 (no plate event before the deviation)"
        elif st["outcome"] not in EXPECT[err]:
            why = f"outcome {st['outcome']} not in {EXPECT[err]}"
        elif st["dev"] is None:
            why = "no deviation anchor"
        elif err != "leave_unattended" and st["fire_tick"] is not None:
            why = f"secondary hazard: reached FIRE at t{st['fire_tick']}"
        elif (err == "leave_unattended" and st["fire_tick"] is None
              and st["outcome"] != "burned"):
            why = "never reached the hazard (no fire, no terminal burn)"
        elif st.get("dev_failed"):
            why = "deviated block failed to execute (illegal-skipped/rejected) — realization flaw"
        elif st["identifiable"] is None:
            why = "no identifiability tick (evidence moment not found in log)"
        if why is None and err == "omit_subgoal":
            rec = recipe_of(cell)
            asm = [f"assemble({n})" for n, _ in rec.contents]
            if kwargs["sid"] == asm[-1]:
                why = "observational twin: omit(last assemble) == premature_serve forever"
        if why:
            rejected.append((err, kwargs, cell, why))
            continue
        ident = st["identifiable"]
        wlo, whi = ident - 2, (st["fire_tick"] if err == "leave_unattended"
                               else ident + WINDOW)
        picked.append(dict(err=DETECT_LABEL.get(err, err), fn=err, kwargs=kwargs,
                           cell=cell, inject=inject,
                           persona="capable_collaborator", dev=st["dev"],
                           identifiable=ident, window=[wlo, whi], outcome=st["outcome"],
                           outcome_tick=st["outcome_tick"], end=st["end"],
                           max_smoke=st["max_smoke"]))
    # ---- select up to PER_TYPE per error, round-robin over cells for diversity
    final = []
    labels = list(dict.fromkeys(DETECT_LABEL.get(e, e) for e in EXPECT))
    for err in labels:
        pool = [c for c in picked if c["err"] == err]
        by_cell = {}
        for c in pool:
            by_cell.setdefault(c["cell"], []).append(c)
        sel, i = [], 0
        while len(sel) < PER_TYPE and any(by_cell.values()):
            for cell in list(CELLS):
                if by_cell.get(cell) and len(sel) < PER_TYPE:
                    sel.append(by_cell[cell].pop(0))
            i += 1
        for j, c in enumerate(sel, 1):
            c["name"] = f"{err}_{CELLS[c['cell']][0]}_{j}"
        per_type[err] = len(sel)
        final.extend(sel)
    # ---- clean + announce cases (false-positive probes) — only kitchens whose CLEAN run is
    # itself hazard-free qualify (a baseline-smoky kitchen is not a fair FP probe)
    for cell in ["medium_nops_medium_map", "easy_nops_easy_map",
                 "medium_nops_hard_map", "easy_nops_medium_map"]:
        short, inject = CELLS[cell]
        st = replay(None, {}, cell, inject)
        # Same relaxation as the error gate: a clean run that merely smokes is still a fair
        # false-positive probe — the dish is correct and the episode is won, so anything the
        # assistant flags there IS a false alarm. Only fire makes silence the wrong answer.
        if st["fire_tick"] is not None:
            print(f"  clean_{short} dropped as FP probe: clean run reaches FIRE at "
                  f"t{st['fire_tick']}")
            continue
        final.append(dict(name=f"clean_{short}", err=None, kwargs={}, cell=cell, inject=inject,
                          persona="capable_collaborator", dev=None, window=None,
                          outcome=st["outcome"], outcome_tick=None, end=st["end"],
                          max_smoke=st["max_smoke"]))
    rec = recipe_of("medium_nops_medium_map")
    for i, anchor in enumerate(["assemble(bun)", "assemble(meat)"], 1):
        final.append(dict(name=f"announce_med_{i}", err=None, kwargs={}, cell="medium_nops_medium_map",
                          inject=True, persona="capable_collaborator", dev=None, window=None,
                          events=[{"after": anchor, "subtype": "ANNOUNCE_PLAN_CHANGE", "pause": 6,
                                   "params": {"ingredient": "lettuce", "state": "raw"}}],
                          outcome=None, outcome_tick=None, end=None, max_smoke=0))
    os.makedirs("tmp/interact_eval", exist_ok=True)
    json.dump({"cases": final,
               "rejected": [{"err": e, "kwargs": k, "cell": c, "why": w} for e, k, c, w in rejected]},
              open("tmp/interact_eval/cases_v2.json", "w"), indent=1)
    print(f"kept {len(final)} cases ({sum(per_type.values())} error + "
          f"{len(final) - sum(per_type.values())} clean/announce); rejected {len(rejected)}")
    for err, n in per_type.items():
        print(f"  {err:22} {n}")
    for e, k, c, w in rejected:
        print(f"  REJ {e:20} {c.split('_')[0]:6} {json.dumps(k):55} {w}")


if __name__ == "__main__":
    main()
