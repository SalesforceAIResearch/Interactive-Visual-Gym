#!/usr/bin/env python3
"""Generate + VERIFY non-error INTENT scenarios for the Cheeseburger task.

Everything in the benchmark so far assumes the cook made a PHYSICAL MISTAKE. These
scenarios assume it did not:

  plan_change — the cook deliberately reorders its own approach. The dish that reaches
                the pass is still exactly right, so any "you made a mistake" from the
                assistant is a FALSE ALARM, not a detection.
                Costs time, changes nothing about the delivery.
  qa_probe    — the cook asks a question about the scene the way a viewer questions a
                streaming video ("what's in the pan?", "how many plates?"). The answer is
                computable from the engine, so it can be graded exactly.

    python -m cooksim_bench.suite.gen_intent_cases --out tmp/interact_eval/cases_intent.json

VERIFICATION (why this file replays instead of just writing JSON): a scenario is only
admissible if the kitchen actually accepts it AND the order still completes. Each case is
replayed headlessly — clean prefix, then the intent sequence, then the rest of the clean
plan — and is kept only when every intent action reports `done` (nothing rejected or
skipped) and the episode still ends `won`. A scenario that cannot be executed, or that
breaks the delivery, is not an intentional plan change: it is an error in disguise.
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.exp_interact import build_episode                      # noqa: E402
from cooksim.core.plan_ir import segment_plan               # noqa: E402

CELL = "medium_nops_medium_map"          # the Cheeseburger cell
CAP = 400

# Intent sequences are written against the CLEAN plan of that cell:
#   0 go to the meat source   1 pick up      2 go to the empty pan   3 put down
#   4 turn on                 5 go to the plates  6 pick up
#   7 go to the bun source    8 scoop        9 go to the pan with the meat  10 scoop
#  11 go to the cheese source 12 scoop      13 go to the serving pass      14 serve
CASES = [
    # ---------------------------------------------------------------- plan_change
    # REDESIGN 2026-08-12 (user): plan_change must change the RECIPE or the ORDER —
    # "plate_first" and "park_and_check" measured nothing and are deleted. The flagship
    # plan_change is now a genuine recipe change: the cook announces an added ingredient.
    # `goal_extend` makes it legitimate BY CONSTRUCTION: the success criterion extends the
    # moment the intent fires (extend_goal — the announcement IS the act), so the extra
    # ingredient is what the customer now expects and any "that's wrong" is a false alarm.
    {"name": "plan_change_add_lettuce",
     "family": "plan_change",
     "trigger_after": 13,       # cheese scooped (index 12 done); only "go to pass, serve" remain
     "intent_label": "add some fresh lettuce on top before serving",
     "intent_seq": ["go to the lettuce source", "scoop"],
     "resume_from": 13,
     "goal_extend": {"ingredient": "lettuce", "state": "raw"},
     "rationale": "A real recipe change, announced. The goal itself is extended when the "
                  "intent fires, so the lettuce-topped burger is exactly what the order "
                  "now asks for — the assistant should update its model, not object."},
    {"name": "plan_change_other_pan",
     "family": "plan_change",
     # TRIGGER-SAFETY RULE (learned the hard way): the trigger state must itself encode
     # the precondition the intent needs. `plan_change_plate_first` fires at t0, where the
     # only precondition is "nothing has happened yet" — but under SOFT prompting nothing
     # forces the cook to act on the task before its already-queued clean plan runs, so by
     # the time it obeyed, its "plate first, then meat" had become "start a SECOND patty"
     # and the assistant was right to object. Here the trigger is "raw meat in hand, both
     # pans empty": if the cook races past it the injection is simply SKIPPED, and if it
     # fires the intent is legitimate by construction.
     "trigger_after": 2,
     "intent_label": "cook the meat in the FAR pan instead of the nearest one",
     "intent_seq": ["go to the far pan", "put down", "turn on"],  # was "go to (0,5)": coordinate grounding was removed 2026-08-03
     "resume_from": 5,
     "rationale": "Choosing a different station is the cook's call: the same patty is "
                  "cooked the same way, just somewhere else, and the served dish is "
                  "identical. Anything the assistant calls wrong here is invented."},
    # ---------------------------------------------------------------- qa_probe
    # REDESIGN 2026-08-12 (user): two new probe kinds beyond present-scene questions —
    # WHERE (retrieve the location of a particular station) and PAST (retrieve history:
    # did a subgoal finish, did the cook ever walk somewhere). PAST reads the rollout log,
    # so ground_truth() takes the rollout for these kinds.
    {"name": "qa_where_cheese",
     "family": "qa_probe",
     "trigger_after": 6,        # plate in hand, cheese is two subgoals away
     "question": "Quick one — where's the cheese kept in this kitchen?",
     "gt_kind": "where:cheese",
     "intent_seq": [], "resume_from": 6,
     "rationale": "Spatial retrieval: the answer is a fixed map fact the assistant can "
                  "read off its observation, graded against the layout."},
    {"name": "qa_past_prep",
     "family": "qa_probe",
     "trigger_after": 7,        # prep(meat) finished at clean index 4
     "question": "Wait — did I already get the meat in the pan and switch it on, or did "
                 "I forget?",
     "gt_kind": "past_subgoal:prep(meat)",
     "intent_seq": [], "resume_from": 7,
     "rationale": "Episodic memory: did a SUBGOAL finish. The truth is in the action "
                  "log, so a confidently wrong 'you forgot' is exactly gradable."},
    {"name": "qa_past_visited_trash",
     "family": "qa_probe",
     "trigger_after": 9,        # mid-assembly; the clean plan never visits the trash
     "question": "Did I stop by the trash bin at any point so far?",
     "gt_kind": "visited:trash",
     "intent_seq": [], "resume_from": 9,
     "rationale": "Episodic memory with a NO ground truth: the clean plan never goes "
                  "there, so agreeing with a false memory is a graded failure."},
    {"name": "qa_pan_contents",
     "family": "qa_probe",
     "trigger_after": 6,
     "question": "Hey — what's in the pan right now, and is it done yet?",
     "gt_kind": "pan_state",
     "intent_seq": [], "resume_from": 6,
     "rationale": "Scene question a viewer would ask of a streaming video; the answer is "
                  "in the engine, so it can be graded exactly."},
    {"name": "qa_plate_count",
     "family": "qa_probe",
     "trigger_after": 6,
     "question": "Quick question — how many plates are out in the kitchen right now?",
     "gt_kind": "plate_count",
     "intent_seq": [], "resume_from": 6,
     "rationale": "Counting question: perception, not recipe knowledge."},
    {"name": "qa_plate_contents",
     "family": "qa_probe",
     "trigger_after": 9,        # bun already scooped
     "question": "What have I got on my plate so far, in order?",
     "gt_kind": "plate_contents",
     "intent_seq": [], "resume_from": 9,
     "rationale": "Order-sensitive state question — exactly the fact the assistant must "
                  "track to judge the plating."},
]


# ------------------------------------------------------------------ ground truth
# Ground truth is read from game.render_state() — the SAME structure the renderer and the
# state dumps use. (A station object's own render_state() carries no "type" field, so an
# earlier version filtering on it silently reported an empty kitchen.)
def _render(game):
    return game.render_state() if hasattr(game, "render_state") else game


def gt_pan_state(game):
    out = []
    for d in (_render(game).get("stations") or []):
        if d.get("type") != "pan":
            continue
        names = [c["name"] for c in (d.get("contents") or [])]
        if names:
            out.append({"at": [d.get("x"), d.get("y")], "contents": names,
                        "status": d.get("status"),
                        "progress": round(float(d.get("progress") or 0), 2),
                        "smoking": bool(d.get("smoking"))})
    return {"pans_with_food": out,
            "done": all(p["status"] == "done" or p["progress"] >= 1.0 for p in out) if out
                    else None}


def _plates(game):
    rs = _render(game)
    found = []
    for p in (rs.get("players") or []):
        d = p.get("holding")
        if d and d.get("kind") == "plate":
            found.append(("held", [x["name"] for x in d.get("contents", [])]))
    for o in (rs.get("objects") or []):
        d = o.get("item")
        if d and d.get("kind") == "plate":
            found.append((f"counter({o.get('x')},{o.get('y')})",
                          [x["name"] for x in d.get("contents", [])]))
    return found


def gt_plate_count(game):
    f = _plates(game)
    return {"count": len(f), "where": [w for w, _ in f]}


def gt_plate_contents(game):
    f = _plates(game)
    held = [c for w, c in f if w == "held"]
    return {"held_plate_contents_in_order": held[0] if held else None,
            "all_plates": f}


GT = {"pan_state": gt_pan_state, "plate_count": gt_plate_count,
      "plate_contents": gt_plate_contents}


# ------------------------------------------------------- parametrised kinds (2026-08-12)
# "where:<target>" reads the layout; "past_subgoal:<sid>" and "visited:<target>" read the
# ROLLOUT LOG — history questions cannot be answered from a scene snapshot, which is why
# ground_truth() now takes the rollout as an optional third argument.
_TERRAIN_OF = {"cheese": "CHEESE_SOURCE", "meat": "MEAT_SOURCE", "bun": "BUN_SOURCE",
               "lettuce": "LETTUCE_SOURCE", "tomato": "TOMATO_SOURCE",
               "plates": "PLATE_SOURCE", "trash": "TRASH", "serving": "SERVING",
               "pass": "SERVING", "cutting_board": "CUTTING_BOARD", "pan": "PAN",
               "pot": "POT", "fryer": "FRYER", "oven": "OVEN"}


def _region(x, y, w, h):
    ver = "top" if y < h / 3 else ("bottom" if y >= 2 * h / 3 else "middle")
    hor = "left" if x < w / 3 else ("right" if x >= 2 * w / 3 else "middle")
    if ver == "middle" and hor == "middle":
        return "center"
    return f"{ver}-{hor}" if hor != "middle" else ver


def gt_where(game, ro, target):
    want = _TERRAIN_OF.get(target, target.upper())
    hits = []
    grid = game.layout.grid
    h, w = len(grid), len(grid[0])
    for y, row in enumerate(grid):
        for x, t in enumerate(row):
            if str(t).split(".")[-1] == want:
                hits.append({"at": [x, y], "region": _region(x, y, w, h)})
    return {"target": target, "found": hits,
            "answer": (hits[0]["region"] if hits else "nowhere — it does not exist here")}


def _world_says_done(game, sid):
    """Subgoal completion read from the WORLD, not the plan log (2026-08-26 sweep audit):
    a live cook that redoes a step after a recovery issues its own commands, which carry no
    plan sid, so the log said "assemble(bun): not finished" while the bun sat on the plate.
    Returns True when the world proves the step happened, else None (unknown)."""
    import re as _re
    m = _re.match(r"(\w+)\((\w+)\)(?:#\d+)?$", sid)
    kind, comp = (m.group(1), m.group(2)) if m else (sid, None)
    try:
        cook = game.cooks[0]
        held = cook.holding
        plate_items = []
        if held is not None and type(held).__name__ == "Plate":
            plate_items = [(c.name, str(getattr(c, "state", ""))) for c in held.contents]
        stations = list(game.stations.values()) if isinstance(getattr(game, "stations", None), dict) else []
        boards = [it for it in game.counter_items.values()] if hasattr(game, "counter_items") else []
        def on_plate(name, states=None):
            return any(n == name and (states is None or any(st in s for st in states)) for n, s in plate_items)
        if kind == "fetch_plate":
            return True if plate_items is not None and held is not None and type(held).__name__ == "Plate" else None
        if kind == "assemble":
            return True if on_plate(comp) else None
        if kind == "ignite":
            if on_plate(comp, ["cooked", "burnt"]):
                return True
            for st in stations:
                names = [c.name for c in getattr(st, "contents", [])]
                if comp in names and getattr(st, "status", "") in ("cooking", "cooked", "burnt"):
                    return True
            return None
        if kind == "prep":
            if on_plate(comp, ["chopped", "cooked", "burnt"]):
                return True
            for st in stations:
                if comp in [c.name for c in getattr(st, "contents", [])]:
                    return True
            for it in boards:
                n = getattr(it, "name", None)
                if n == comp and "chopped" in str(getattr(it, "state", "")).lower():
                    return True
            return None
    except Exception:                                     # noqa: BLE001
        return None
    return None


def gt_past_subgoal(game, ro, sid):
    log = list(getattr(ro, "log", []) or []) if ro is not None else []
    done_t = [l.get("t_done") for l in log
              if l.get("kind") == sid and l.get("status") == "done"]
    pending = False
    if ro is not None:
        pending = any(k == sid for _, k in list(ro.cmds[ro.i + 1:]))
    finished = bool(done_t) and not pending
    world = _world_says_done(game, sid)
    if world is True:
        finished = True
    return {"subgoal": sid, "finished": finished,
            "last_step_done_at": max(done_t) if done_t else None,
            "world_evidence": world}


def gt_visited(game, ro, target):
    """Did the cook walk over to <target>? Resolved against the world (qa_world.visits: the station the cook FACED
    after each finished walk), not a regex on the command words -- "go to board 1" is a cutting-board visit
    (audit A3, 2026-09-19). Targets that are not stations (e.g. a source) fall back to the engine's parser."""
    from cooksim_bench.scoring.qa_world import visits
    log = list(getattr(ro, "log", []) or []) if ro is not None else []
    vs = visits(log, getattr(ro, "world", None) if ro is not None else None, target.replace("_", " "),
                upto=getattr(ro, "tick", None) if ro is not None else None)
    return {"target": target, "visited": bool(vs),
            "first_visit_t": vs[0]["t"] if vs else None}


GT_PARAM = {"where": gt_where, "past_subgoal": gt_past_subgoal, "visited": gt_visited}
try:                                      # suite v3 question families (docs/SUITE_V3_DESIGN.md §3)
    from cooksim_bench.scoring.qa_v3_lib import GT_PARAM_V3
    GT_PARAM.update(GT_PARAM_V3)
except Exception:                                            # pragma: no cover
    pass

# Generalised readers for the non-pan kitchens. gt_pan_state filters on type == "pan", so
# Fish & Chips (two fryers) reported an empty kitchen, and Steak Frites / Loaded Burrito
# silently omitted whichever component was not in the pan — a qa_probe graded against that
# marked a CORRECT assistant answer wrong on four of the five recipes. Defined in
# python -m cooksim_bench.suite.intent_lib so the intent matrix and the live rollout grade against exactly the same
# function; imported lazily so this module still works standalone.
try:
    from cooksim_bench.suite.intent_lib import EXTRA_GT
    GT.update(EXTRA_GT)
except Exception:                                            # pragma: no cover
    pass


def ground_truth(kind, game, ro=None):
    """`kind` may be parametrised ("where:cheese", "past_subgoal:prep(meat)",
    "visited:trash") — those readers get the rollout too, because history questions
    cannot be answered from a scene snapshot. Plain kinds keep the old game-only path."""
    if ":" in kind:
        base, arg = kind.split(":", 1)
        if base in GT_PARAM:
            return GT_PARAM[base](game, ro, arg)
    return GT[kind](game)


# ------------------------------------------------------------------ verification
def verify(case):
    """Replay clean-prefix + intent + remaining-clean. A case is admissible only if the
    kitchen accepted every intent action AND the order still completes."""
    state, ro = build_episode(None, "capable_collaborator", cell=CELL, inject=True)
    if case.get("goal_extend"):
        # the announced recipe change extends the success criterion — replay under the
        # same extended goal the live episode will use, or verify() rejects a case that
        # is legitimate by construction
        from cooksim.core.lang_events import extend_goal
        ge = case["goal_extend"]
        extend_goal(ro.game, ge["ingredient"], ge.get("state", "raw"))
    flat = [c for n in segment_plan(state.recipe) for c in n.commands]
    m = case["trigger_after"]
    seq = list(case.get("intent_seq") or [])
    plan = flat[:m] + seq + flat[case["resume_from"]:]
    ro.cmds = [(c, None) for c in plan]
    ro.cap = CAP
    while ro.i < len(ro.cmds) or ro.ctrl.skill is not None:
        if not ro.step():
            break
    bad = [l for l in ro.log if l["status"] in ("rejected", "illegal-skipped")]
    # only the intent actions must be clean; a rejection anywhere breaks the claim that
    # this is a legal, deliberate alternative
    ok = ro.outcome == "won" and not bad
    return {"ok": ok, "outcome": ro.outcome, "ticks": ro.tick,
            "rejected": [{"cmd": l["cmd"], "status": l["status"]} for l in bad][:5],
            "plan_len": len(plan)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/interact_eval/cases_intent.json")
    a = ap.parse_args()
    kept, dropped = [], []
    for c in CASES:
        v = verify(c)
        rec = dict(c, cell=CELL, err="no_error", verification=v)
        print(f"{'KEEP' if v['ok'] else 'DROP'} {c['name']:<26} "
              f"outcome={v['outcome']} ticks={v['ticks']} rejected={len(v['rejected'])}"
              + ("" if v["ok"] else f"  {v['rejected']}"), flush=True)
        (kept if v["ok"] else dropped).append(rec)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"schema": "intent-cases-v1", "cell": CELL, "task": "Cheeseburger",
               "cases": kept, "dropped": dropped}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}: {len(kept)} verified, {len(dropped)} dropped")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
