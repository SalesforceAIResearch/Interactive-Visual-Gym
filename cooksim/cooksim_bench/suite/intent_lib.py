#!/usr/bin/env python3
"""Shared machinery for the NON-ERROR **intent** scenarios (plan_change / qa_probe).

Everything the intent work needs that is not allowed to live in the engine:

  * cell / recipe metadata and a safe `build()` (the hard-coded decoy injection in
    exp_interact.build_episode destroys a needed source on some maps — see SAFE_INJECT);
  * introspection of a cell's CLEAN plan: where hands are free, where the plate is held,
    which stations exist, where each subgoal starts;
  * `replay()` — the headless splice (clean prefix + intent_seq + clean tail from
    resume_from) with far richer diagnostics than gen_intent_cases.verify();
  * extended qa_probe ground truth (`EXTRA_GT`) that works on fryer/pot/board recipes,
    not only on the Cheeseburger's pan.

    COOKSIM_GRID=tmp/grid.json python -m cooksim_bench.suite.intent_lib --dump
"""
from __future__ import annotations

import os
import sys

from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, os.path.join(ROOT, "benchmark", "grid.json")))

from cooksim.core.items import Plate                                       # noqa: E402
from cooksim.core.plan_ir import segment_plan                              # noqa: E402
from cooksim_bench.exp_interact import build_episode                                     # noqa: E402

# ---------------------------------------------------------------- cells
CELLS = {
    "caprese":      "easy_nops_medium_map",
    "cheeseburger": "medium_nops_medium_map",
    "fishchips":    "hard_nops_medium_map",
    "steak":        "expert_nops_medium_map",
    "burrito":      "master_nops_medium_map",
}
RECIPE_NAME = {"caprese": "Caprese", "cheeseburger": "Cheeseburger",
               "fishchips": "Fish & Chips Platter", "steak": "Steak Frites",
               "burrito": "Loaded Burrito"}

# The decoy injection (a lettuce source + a bin) used to be hard-coded at (0,2)/(5,2), which
# are spare counters on the Cheeseburger map and NOT spare anywhere else — on Caprese it ate
# the only tomato source and made the dish impossible (ECR-1). exp_interact._place_decoys now
# only overwrites empty counters and skips a decoy the map already has, so every cell is safe
# to inject and every cell therefore has a TRASH and a LETTUCE_SOURCE for the side quests.
SAFE_INJECT = {k: True for k in
               ("caprese", "cheeseburger", "fishchips", "steak", "burrito")}


def relax_burn(ro):
    """Mirror the burn/fire relaxation e2e_stepin_rollout applies to every live episode.

    Human-paced episodes argue mid-cook, so the live harness multiplies the burn timers and
    the fire delay before anything runs. Verifying an intent case under the DEFAULT timers
    verifies the wrong physics: a side quest costs time by definition, and at default timers
    a 12-tick detour on Steak Frites sets the kitchen on fire — under the timers the case will
    actually run with, it does not. Kept byte-identical to the live block on purpose; if that
    block changes, this must change with it.
    """
    for k in ("pot_burn_time", "pan_burn_time", "oven_burn_time"):
        setattr(ro.game.config, k, 480)
    ro.game.config.fire_time = 120
    for st in ro.game.stations.values():          # stations snapshot config at construction
        if hasattr(st, "burn_time"):
            st.burn_time = 480
        if hasattr(st, "fire_time"):
            st.fire_time = 120
    return ro


def build(cell_key, persona="capable_collaborator", relax=True):
    """A fresh episode for a cell, under the same physics the live harness uses."""
    state, ro = build_episode(None, persona, cell=CELLS[cell_key],
                              inject=SAFE_INJECT[cell_key])
    if relax:
        relax_burn(ro)
    return state, ro, ro.game


def flat(state):
    """The clean plan as a flat list of (command, subgoal-id)."""
    return [(c, n.sid) for n in segment_plan(state.recipe) for c in n.commands]


def flat_cmds(state):
    return [c for c, _s in flat(state)]


def node_starts(state):
    """subgoal id -> index of its first command in the flat plan."""
    out, i = {}, 0
    for n in segment_plan(state.recipe):
        out[n.sid] = i
        i += len(n.commands)
    return out


def plate_index(state):
    """Index of 'go to the plates' in the flat plan (the fetch_plate node)."""
    return node_starts(state)["fetch_plate"]


# ---------------------------------------------------------------- driving
def drive(ro, upto=None, cap=3000):
    """Run the queue. True if it reached a resting state rather than running out of budget."""
    ro.cap = max(getattr(ro, "cap", 0), cap)
    for _ in range(cap):
        idle = ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
        if idle or (upto is not None and ro.i >= upto and ro.ctrl.skill is None):
            return True
        if not ro.step():
            return True
    return False


def held_kind(game):
    h = game.cooks[0].holding
    if h is None:
        return None
    return "plate" if isinstance(h, Plate) else "ingredient"


def hand_trace(cell_key):
    """What the cook is holding immediately AFTER flat-plan command i completes.

    Index i in the returned list is the state at `trigger_after = i + 1` (the splice puts the
    intent right after command i). Position 0 of the returned dict is the state before
    anything has run.
    """
    state, ro, g = build(cell_key)
    cmds = flat(state)
    ro.cmds = list(cmds)
    ro.i = -1
    ro.cap = 3000
    out = {0: None}
    for i in range(len(cmds)):
        drive(ro, upto=i, cap=3000)
        out[i + 1] = held_kind(g)
    return out


def stations_present(cell_key):
    """The set of terrain names on this cell's map (post-injection)."""
    _s, _ro, g = build(cell_key)
    return {t.name for row in g.layout.grid for t in row}


def free_boards(cell_key):
    """How many cutting boards the map has."""
    return sum(1 for row in build(cell_key)[2].layout.grid
               for t in row if t.name == "CUTTING_BOARD")


# ---------------------------------------------------------------- the splice
def spliced_plan(state, case):
    """clean prefix + intent_seq + clean tail. The exact plan e2e_stepin_rollout builds."""
    f = flat_cmds(state)
    return (f[:case["trigger_after"]] + list(case.get("intent_seq") or [])
            + f[case["resume_from"]:])


DETOUR = ["go to the serving pass", "go to the plates"]


def replay(case, cap=600, detour=None):
    """Replay a case headlessly and report everything a test could want to assert.

    Returns a dict with:
      settled    the queue reached a resting state (no livelock / budget exhaustion)
      outcome    'won' | 'timeout' | ...
      rejected   commands the kitchen refused or skipped as illegal
      intent_done  every intent command appeared as a completed action, in order,
                   at or after the trigger
      plate      the ordered contents of the served/held plate at the end
      ticks, plan_len, trigger_tick
    """
    cell_key = case["cell_key"]
    state, ro, g = build(cell_key)
    plan = spliced_plan(state, case)
    if detour:
        # LIVE-COOK PERTURBATION. inject_key() fires the intent on the PLATE and the station
        # contents, deliberately ignoring where the cook is standing — so in a live episode
        # the trigger state is reached from an arbitrary position, not from the tile the
        # clean prefix happens to end on. Walking the cook somewhere else first, without
        # touching anything, reproduces that: the kitchen is identical, the cook is not.
        m = case["trigger_after"]
        plan = plan[:m] + list(detour) + plan[m:]
    ro.cmds = [(c, None) for c in plan]
    ro.i = -1
    ro.cap = cap

    m = case["trigger_after"] + len(detour or ())
    seq = list(case.get("intent_seq") or [])
    # run the clean prefix (and any perturbation) first, so the trigger state can be
    # inspected and so `done_after_trigger` contains only the intent and the tail
    if m:
        drive(ro, upto=m - 1, cap=cap)
    trigger_state = {"holding": held_kind(g), "tick": ro.tick,
                     "plate": plate_contents_list(g)}
    # stop one command short of the end: the plate is READ before `serve` takes it away,
    # otherwise every successful episode reports an empty plate.
    drive(ro, upto=max(0, len(plan) - 2), cap=cap)
    plate_before_serve = plate_contents_list(g)
    settled = drive(ro, cap=cap)

    bad = [{"cmd": l["cmd"], "status": l["status"]} for l in ro.log
           if l["status"] in ("rejected", "illegal-skipped")]
    # subsequence match over COMPLETED commands issued at or after the trigger.
    # The winning `serve` is still logged 'running': the episode ends on the tick it fires,
    # so its t_done is never written back. A terminal outcome IS its completion.
    ok_last = ro.outcome in ("won", "lost", "wrong_serve")
    done = [l["cmd"] for l in ro.log
            if (l["status"] == "done"
                or (ok_last and l is ro.log[-1] and l["status"] == "running"))
            and (l.get("t_issue") or 0) >= (trigger_state["tick"] or 0)]
    i = 0
    for c in done:
        if i < len(seq) and c == seq[i]:
            i += 1
    return {"settled": settled, "outcome": ro.outcome, "ticks": ro.tick,
            "rejected": bad, "intent_done": (i >= len(seq)),
            "intent_matched": i, "intent_len": len(seq),
            "plate": plate_before_serve, "plan_len": len(plan),
            "trigger": trigger_state, "done_after_trigger": done}


# ---------------------------------------------------------------- the live trigger
def trigger_window(case, cap=900):
    """Where, and for how long, the LIVE injection would actually fire.

    e2e_stepin_rollout does not splice at a command index. It replays the clean prefix
    headlessly, takes `inject_key()` of the result, and then watches the live kitchen every
    tick until that key comes round again (or TRIGGER_DEADLINE / the plate moves past it).
    So a case is only usable live if that key is:

      REACHABLE   the clean plan passes through it at all;
      WIDE ENOUGH a live cook, which does not move in lockstep with the replay, has more than
                  an instant to be caught in it;
      UNAMBIGUOUS the FIRST time the key matches is the trigger point the case intends — if
                  an earlier, identical-looking state matches first, the intent fires there
                  instead, and the preconditions the sequence relies on do not hold yet.

    Returns first_tick / first_index / ticks / indices, all measured on the clean plan.
    """
    from cooksim_bench.e2e_stepin_rollout import inject_key            # the live rule, not a copy of it

    cell_key = case["cell_key"]
    m = case["trigger_after"]

    rstate, rro, rg = build(cell_key)
    rro.cmds, rro.i, rro.cap = flat(rstate)[:m], -1, cap
    drive(rro, cap=cap)
    ref = inject_key(rg)

    state, ro, g = build(cell_key)
    ro.cmds, ro.i, ro.cap = flat(state), -1, cap
    ticks, indices = [], set()
    for _ in range(cap):
        if inject_key(g) == ref:
            ticks.append(ro.tick)
            indices.add(ro.i + 1)
        if ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None:
            break
        if not ro.step():
            break
    return {"ref": ref, "ticks": len(ticks), "first_tick": ticks[0] if ticks else None,
            "indices": sorted(indices),
            "first_index": min(indices) if indices else None,
            "intended_index": m}


# ---------------------------------------------------------------- state readers
def _render(game):
    return game.render_state() if hasattr(game, "render_state") else game


def plate_contents_list(game):
    """Ordered contents of the plate in hand, else of any plate on a counter, else [].

    Works on a live game AND on a replayed state-dump frame (a plain dict): grading
    re-derives ground truth from the dump, where `game.cooks` does not exist.
    """
    if not hasattr(game, "cooks"):                    # state-dump frame
        rs = game
        for p in (rs.get("players") or []):
            d = p.get("holding")
            if d and d.get("kind") == "plate":
                return [c["name"] for c in (d.get("contents") or [])]
        for o in (rs.get("objects") or []):
            d = o.get("item")
            if d and d.get("kind") == "plate":
                return [c["name"] for c in (d.get("contents") or [])]
        return []
    h = game.cooks[0].holding
    if isinstance(h, Plate):
        return [c.name for c in h.contents]
    for it in game.counter_items.values():
        if isinstance(it, Plate):
            return [c.name for c in it.contents]
    return []


def served_plate(ro):
    """Ordered contents of what actually reached the pass, from the game's delivery record."""
    g = ro.game
    for attr in ("served", "deliveries", "delivered"):
        v = getattr(g, attr, None)
        if v:
            try:
                return [c["name"] if isinstance(c, dict) else getattr(c, "name", str(c))
                        for c in (v[-1].contents if hasattr(v[-1], "contents") else v[-1])]
            except Exception:  # noqa: BLE001
                pass
    return None


# ---------------------------------------------------------------- extended qa ground truth
# gen_intent_cases.gt_pan_state filters on type == "pan", so it reports an EMPTY kitchen on
# every recipe that cooks in a fryer or a pot (Fish & Chips, Steak Frites, Loaded Burrito)
# and on Caprese, which has no heat step at all. These generalise it.
HEAT = {"pan", "pot", "fryer", "oven"}


def gt_station_state(game):
    """Every heat station that holds food: what, where, how far along, is it done."""
    out = []
    for d in (_render(game).get("stations") or []):
        if d.get("type") not in HEAT:
            continue
        names = [c["name"] for c in (d.get("contents") or [])]
        if not names:
            continue
        out.append({"station": d.get("type"), "at": [d.get("x"), d.get("y")],
                    "contents": names, "status": d.get("status"),
                    "progress": round(float(d.get("progress") or 0), 2),
                    "smoking": bool(d.get("smoking"))})
    return {"cooking_stations_with_food": out,
            "any_food_cooking": bool(out),
            "all_done": (all(s["status"] == "done" or s["progress"] >= 1.0 for s in out)
                         if out else None)}


def gt_board_state(game):
    """Every cutting board and what is sitting on it (chopped or not)."""
    out = []
    for d in (_render(game).get("stations") or []):
        if d.get("type") not in ("cutting_board", "board"):
            continue
        names = [(c["name"], c.get("state")) for c in (d.get("contents") or [])]
        if not names and d.get("item"):
            names = [(d["item"]["name"], d["item"].get("state"))]
        out.append({"at": [d.get("x"), d.get("y")],
                    "contents": [{"name": n, "state": s} for n, s in names]})
    return {"boards": out, "boards_in_use": sum(1 for b in out if b["contents"])}


def gt_holding(game):
    """What is in the cook's hands.

    Reads through `_render()` rather than `game.cooks[0].holding`, because this reader is
    ALSO called on a replayed state-dump frame (eval_intent.regt), which is a plain dict —
    the live-object version raised `AttributeError: 'dict' object has no attribute 'cooks'`
    there and would have destroyed the stored ground truth of every `holding` probe.
    """
    players = _render(game).get("players") or []
    h = (players[0] if players else {}).get("holding")
    if not h:
        return {"holding": None}
    if h.get("kind") == "plate":
        return {"holding": "plate",
                "plate_contents_in_order": [c["name"] for c in (h.get("contents") or [])]}
    return {"holding": h.get("name"), "state": h.get("state")}


def _order_contents(game):
    """The order slip's required contents, in plating order. Dump-safe."""
    for o in (_render(game).get("orders") or []):
        cs = o.get("contents") or []
        if cs:
            return [c["name"] for c in cs]
    return []


def gt_remaining(game, recipe=None):
    """What is still missing from the plate, in plating order (perception + the order slip).

    `recipe` was previously the ONLY way to get `still_to_plate_in_order`, but the GT
    dispatcher only ever calls registered readers as `fn(game)` — so the recipe was always
    None and the interesting half of this reader was dead in every live rollout. The order
    slip is in the render state, so fall back to it when no recipe is passed.
    """
    on = plate_contents_list(game)
    want = [n for n, _s in recipe.contents] if recipe is not None else _order_contents(game)
    if not want:
        return {"on_plate_in_order": on}
    left, seen = [], list(on)
    for n in want:
        if n in seen:
            seen.remove(n)
        else:
            left.append(n)
    return {"on_plate_in_order": on, "still_to_plate_in_order": left}


EXTRA_GT = {
    "station_state": gt_station_state,
    "board_state": gt_board_state,
    "holding": gt_holding,
    "remaining": gt_remaining,
}


def ground_truth(kind, game, ro=None):
    """Extended table first, then whatever gen_intent_cases already knows.

    `ro` is forwarded because the parametrised history kinds (`past_subgoal:`, `visited:`)
    are answered from the rollout log, not from the scene.
    """
    if kind in EXTRA_GT:
        return EXTRA_GT[kind](game)
    from cooksim_bench.suite.gen_intent_cases import ground_truth as base
    return base(kind, game, ro)


# ------------------------------------------- re-deriving ground truth from a state dump
# eval_intent.regt() replays the per-tick state dump and recomputes ground truth from it,
# because the dump is the authoritative record of the kitchen. Two rules follow, and both
# were violated before 2026-08-12:
#
#   1. a dump frame is a plain dict, so every reader on this path must use dict access
#      (`_render()` handles both) — never `game.cooks` / `isinstance(x, Plate)`;
#   2. a dump records the SCENE, not the cook's execution log, so a history question
#      (`past_subgoal:`, `visited:`) simply cannot be re-derived from it. Recomputing one
#      from a dump does not produce a worse answer, it produces a WRONG answer that then
#      overwrites the correct live one.
HISTORY_KINDS = ("past_subgoal", "visited")


def kind_base(kind):
    """`"where:cheese"` -> `"where"`; a plain kind is its own base."""
    return kind.split(":", 1)[0] if kind and ":" in kind else (kind or "")


def is_history_kind(kind):
    return kind_base(kind) in HISTORY_KINDS


def gt_where_from_dump(frame, target):
    """`where:<target>` off a state-dump frame.

    The live reader walks `game.layout.grid` (enum tiles, `Tile.CHEESE_SOURCE`); the dump
    carries the same grid as `terrain`, lower-cased. NOTE the indexing: `terrain[y][x]`,
    matching the live reader's `for y, row in enumerate(grid)`. Transposing it returns a
    wrong-but-plausible tile type rather than an error, so it is pinned by a test.
    """
    from cooksim_bench.suite.gen_intent_cases import _TERRAIN_OF, _region
    want = _TERRAIN_OF.get(target, target.upper()).lower()
    grid = (frame or {}).get("terrain") or []
    if not grid:
        return None                          # nothing to re-derive from; keep the live value
    h, w = len(grid), len(grid[0])
    hits = [{"at": [x, y], "region": _region(x, y, w, h)}
            for y, row in enumerate(grid)
            for x, t in enumerate(row)
            if str(t).split(".")[-1].lower() == want]
    return {"target": target, "found": hits,
            "answer": (hits[0]["region"] if hits else "nowhere — it does not exist here")}


def ground_truth_from_dump(kind, frame):
    """Ground truth for `kind` from one state-dump frame, or None if it cannot be had.

    Returning None means "keep whatever the live harness recorded" — which is always the
    safe outcome, and the only correct one for the history kinds.
    """
    if is_history_kind(kind):
        return None
    base, _, arg = kind.partition(":")
    if base == "where":
        return gt_where_from_dump(frame, arg)
    try:
        return ground_truth(kind, frame)
    except Exception:                                            # noqa: BLE001
        return None


# ---------------------------------------------------------------- dump
def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = {}
    for k in CELLS:
        state, ro, g = build(k)
        f = flat(state)
        ht = hand_trace(k)
        out[k] = {
            "cell": CELLS[k], "recipe": RECIPE_NAME[k],
            "ordered": state.recipe.ordered,
            "contents": [list(c) for c in state.recipe.contents],
            "inject": SAFE_INJECT[k],
            "n_cmds": len(f),
            "plate_index": plate_index(state),
            "node_starts": node_starts(state),
            "boards": free_boards(k),
            "terrain": sorted(stations_present(k)),
            "plan": [{"i": i, "cmd": c, "sid": s, "holding_after": ht[i + 1]}
                     for i, (c, s) in enumerate(f)],
        }
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    for k, d in out.items():
        print(f"\n===== {k}  {d['recipe']}  ({d['cell']}) ordered={d['ordered']} "
              f"inject={d['inject']} boards={d['boards']}")
        print("  terrain:", " ".join(t for t in d["terrain"] if t not in ("FLOOR", "WALL")))
        print("  nodes:", d["node_starts"])
        for p in d["plan"]:
            print(f"   {p['i']:>3} {p['cmd']:<42} [{p['sid']}] hold={p['holding_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
