#!/usr/bin/env python3
"""Interactive episode harness (design doc §10): MultiRollout + the interaction system.

Runs a single-error episode with a supervisor in the loop:
  - supervisor.observe(ro) is polled every tick (the 'frame stream' seam — scripted stubs
    read state; a real VLM will read rendered frames through the same interface);
  - [v2] a FLAG_ERROR/SUGGEST reply must CONVINCE the human. With no human VLM in this
    scripted harness, the deterministic conviction stand-in is checks.flag_is_correct — a
    scripted human "checks its own record" via the oracle: a claim matching its real past
    convinces, anything else is pushed back. Convinced -> error_correction (deviated future
    cleared, cook frozen, world clock RUNNING); otherwise recorded false positive, nothing
    changes physically. (The real-VLM harness, interact_eval.py, uses the human VLM's actual
    accept/reject moves; there the oracle is scoring-only.)
  - [v2] ANNOUNCE_PLAN_CHANGE events apply THEMSELVES at fire time (apply_announced_change);
    the supervisor has no engine channel;
  - the human-side parse is stubbed: the structured suggestion payload on the reply stands in
    for the future human VLM's linguistic parse; the deterministic persona/ability filter
    (system.convert_parse) is applied exactly as it will be for the real one; a HIGH-solvability
    user self-repairs from a bare validated flag;
  - each tick in error_correction the resolve predicate is checked; on pass the stashed clean
    tail resumes.

    python -m cooksim_bench.exp_interact --error omit_subgoal --persona eager_novice
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

from cooksim.core.recipes import Recipe
from cooksim.core.plan_ir import segment_plan, compile_ir
from cooksim.core import error_lib as E
from cooksim.interact import system, checks
from cooksim.interact.state import UserState, PERSONAS, VLMReply
from cooksim.interact.protocol import PerfectSupervisor, WorstSupervisor
from cooksim_bench.exp_multi_error import MultiRollout
from cooksim_bench.exp_error_rollout import inject_source
from cooksim_bench.backends.gemini_stream_rgb import make_gen_game

MED = "medium_nops_medium_map"


def recipe_of(cell=MED):
    import re as _re
    data = json.load(open(os.environ["COOKSIM_GRID"]))
    sl = lambda s: _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    e = next(x for x in data["grid"] if sl(x["cell"]) == cell)
    return Recipe(id=e["recipe_id"], name=e["recipe_name"],
                  contents=tuple((n, s) for n, s in e["contents"]), base_reward=20, ordered=e["ordered"])


def _spare_counter_tiles(game, n, prefer=()):
    """Empty COUNTER tiles safe to repurpose as decoys, farthest from the cook first.

    `prefer` is tried first so a cell whose historical decoy coordinates are genuinely spare
    keeps them (the Cheeseburger map is baked into a lot of recorded episodes).
    """
    from cooksim.core.enums import Terrain
    grid = game.layout.grid
    h, w = len(grid), len(grid[0])
    def spare(x, y):
        return (0 <= x < w and 0 <= y < h and grid[y][x] == Terrain.COUNTER
                and game.counter_items.get((x, y)) is None)
    out = [p for p in prefer if spare(*p)]
    sx, sy = game.cooks[0].pos
    rest = sorted(((x, y) for y in range(h) for x in range(w)
                   if spare(x, y) and (x, y) not in out),
                  key=lambda p: -(abs(p[0] - sx) + abs(p[1] - sy)))
    return (out + rest)[:n]


def _place_decoys(game):
    """Add a lettuce decoy and a bin WITHOUT deleting anything the recipe needs.

    These used to be hard-coded at (0,2) and (5,2), which are spare counters on the
    Cheeseburger map and nothing of the sort elsewhere: on Caprese (5,2) is the ONLY tomato
    source, so every Caprese episode built with the default inject=True ended `wrong_serve`
    with the dish physically impossible — including the ones tests were asserting on. On
    Loaded Burrito the same pair ate a cutting board and one of two steak sources.

    Only empty counters are ever overwritten now, and a map that already has a bin or a lettuce
    source spends no counter on one.
    """
    from cooksim.core.enums import Terrain
    grid = game.layout.grid
    has = lambda t: any(c == t for row in grid for c in row)
    want = [(Terrain.LETTUCE_SOURCE, "lettuce")] if not has(Terrain.LETTUCE_SOURCE) else []
    if not has(Terrain.TRASH):
        want.append((Terrain.TRASH, "trash"))
    if not want:
        return
    spots = _spare_counter_tiles(game, len(want), prefer=[(0, 2), (5, 2)])
    if len(spots) < len(want):                     # never silently half-inject
        raise RuntimeError(f"no spare counter for decoys on this map (need {len(want)})")
    for (terr, name), (x, y) in zip(want, spots):
        inject_source(game, f"{name}@{x},{y}")


# User ruling R2 (2026-09-19): add_foreign always scoops an OFF-RECIPE ingredient, so every kitchen must offer one.
# Three kitchens had none (Caprese and Cheeseburger on their hard maps, which run bare; the Loaded Burrito on its
# medium map, whose lettuce decoy is a recipe item). The first entry of this list that is not in the recipe is added as
# a crate. All four are choppable, so the same crate also serves mis_prep (audit C9: a mis_prep's wrong item goes to
# the cutting board, never onto a lit station where it would burn untracked), and all four are rendered (web/js
# render.js / render3d.js) and listed by the generated rulebook (cooksim/interact/kitchen_rules.py reads the tiles).
OFF_RECIPE_DISTRACTORS = ("lettuce", "onion", "mushroom", "cheese")


# Suite v5 round 2 (D3, 2026-09-19): a map with a SINGLE off-recipe crate forced every case that needs two wrong items
# (a mis_prep and an add_foreign) to reuse it, and made "flag anything with lettuce" a cheap policy (medium tier: lettuce
# in 9/10 cases). Every kitchen now offers at least OFF_RECIPE_MIN off-recipe crates. The second crate prefers an item
# no heat station accepts (cheese: a mis-prepped one is left on the board, never cooked and burned untracked, audit D7),
# else the next distractor; a kitchen whose off-recipe crates are all cookable (or not choppable) gets one clean crate
# for mis_prep. COOKSIM_OFF_RECIPE_MIN=1 restores the round-1 maps.
OFF_RECIPE_MIN = int(os.environ.get("COOKSIM_OFF_RECIPE_MIN", "2"))
SECOND_DISTRACTORS = ("cheese", "onion", "mushroom", "lettuce")


def _clean_mis_prep_item(game, name):
    """D7 (round 2): a choppable item that no heat station of this kitchen accepts (raw or chopped) — a mis-prepped
    one is left on the board; an onion went into the pot and burned the episode (expert hard_4 c)."""
    from cooksim.core.enums import STATION_ACCEPTS, choppable
    kinds = {t for row in game.layout.grid for t in row}
    return choppable(name) and not any(n == name for st, acc in STATION_ACCEPTS.items() if st in kinds for n, _s in acc)


def _ensure_off_recipe_source(game, recipe):
    """R2: if fewer than OFF_RECIPE_MIN ingredient crates on this map are off the recipe, turn spare counters into
    distractor crates.

    A crate goes on an EMPTY counter that a cook can face (it has a walkable neighbour) — counters are never
    walkable, so a crate there cannot block a path — and, like the decoys, as far from the cook's start as possible so
    it does not crowd the working area. Returns the list of (name, (x, y)) added (empty when the map already had
    enough)."""
    from cooksim.core.enums import Terrain
    from cooksim.agents.heuristic import _adjacent_floor
    grid = game.layout.grid
    items = {n for n, _ in recipe.contents}
    added = []
    while True:
        srcs = {str(t.value)[:-len("_source")] for row in grid for t in row
                if str(t.value).endswith("_source") and t != Terrain.PLATE_SOURCE}
        off = srcs - items
        clean = {n for n in off if _clean_mis_prep_item(game, n)}
        if len(off) >= max(1, OFF_RECIPE_MIN) and (clean or OFF_RECIPE_MIN < 2):
            return added
        if not off:
            order = OFF_RECIPE_DISTRACTORS
        elif not clean:
            order = tuple(n for n in SECOND_DISTRACTORS if _clean_mis_prep_item(game, n))
        else:
            order = SECOND_DISTRACTORS
        name = next((n for n in order if n not in items and n not in srcs), None)
        if name is None:
            return added
        h, w = len(grid), len(grid[0])
        sx, sy = game.cooks[0].pos
        spots = sorted(((x, y) for y in range(h) for x in range(w)
                        if grid[y][x] == Terrain.COUNTER and game.counter_items.get((x, y)) is None
                        and _adjacent_floor(game, x, y)),
                       key=lambda p: (-(abs(p[0] - sx) + abs(p[1] - sy)), p[1], p[0]))
        if not spots:
            if off:
                return added                # one off-recipe crate is the hard minimum (R2); a second is best effort
            raise RuntimeError(f"R2: no reachable spare counter for an off-recipe {name} crate on this map")
        x, y = spots[0]
        inject_source(game, f"{name}@{x},{y}")
        added.append((name, (x, y)))


# Suite v5 round 2 (2026-09-19): the Fish & Chips medium map had ONE cutting board for a recipe that chops two potatoes;
# a mis_prep left its wrong item on it and the LLM cook stalled asking how to clear it until the episode timed out
# (pilot hard_medium_3 baseline; round-1 review hard medium_4: a 225-tick loop). A kitchen gets a second board on the
# spare counter nearest its existing board. COOKSIM_MIN_BOARDS=1 restores the round-1 maps.
MIN_BOARDS = int(os.environ.get("COOKSIM_MIN_BOARDS", "2"))


def _ensure_boards(game):
    from cooksim.core.enums import Terrain
    from cooksim.core.stations import CuttingBoard
    from cooksim.agents.heuristic import _adjacent_floor
    grid = game.layout.grid
    boards = [(x, y) for y, row in enumerate(grid) for x, t in enumerate(row) if t == Terrain.CUTTING_BOARD]
    added = []
    while boards and len(boards) < MIN_BOARDS:
        bx, by = boards[0]
        spots = sorted(((x, y) for y in range(len(grid)) for x in range(len(grid[0]))
                        if grid[y][x] == Terrain.COUNTER and game.counter_items.get((x, y)) is None
                        and _adjacent_floor(game, x, y)),
                       key=lambda p: (abs(p[0] - bx) + abs(p[1] - by), p[1], p[0]))
        if not spots:
            break
        x, y = spots[0]
        grid[y][x] = Terrain.CUTTING_BOARD
        game.stations[(x, y)] = CuttingBoard(game.config.chop_time)
        boards.append((x, y))
        added.append((x, y))
    return added


def build_episode(error_call, persona="capable_collaborator", cell=MED, lang=True, inject=True):
    """error_call = (name, kwargs) or None for a clean/announcement episode."""
    rec = recipe_of(cell)
    game, _ = make_gen_game(cell)
    if inject:
        _place_decoys(game)
    _ensure_off_recipe_source(game, rec)
    _ensure_boards(game)
    clean = list(segment_plan(rec))
    if error_call is None:
        ir, recs = list(segment_plan(rec)), []
    else:
        ir, recs = E.compose(list(segment_plan(rec)), [error_call])
    state = UserState(recipe=rec, clean_ir=clean, err_ir=ir, records=recs,
                      profile=PERSONAS[persona])
    ro = MultiRollout(game, compile_ir(ir), succeeds=False,
                      lang_events=E.lang_events_from(recs) if lang else None, seed=11)
    ro.cap = 400
    return state, ro


def run_episode(state, ro, supervisor, events=None):
    """Step to completion with the supervisor polled every tick. Returns a report."""
    report = {"gate": None, "false_positive": False, "entered_correction": False,
              "fix_source": None, "filter": None, "resolved_at": None, "announced_change": None}
    if events:
        ro.lang_pending.extend(events)
    ro.path = []                       # cook position per tick (supervisor observation input)
    seen_lang = 0
    while True:
        ro.path.append(tuple(ro.game.cooks[0].pos))
        reply = supervisor.observe(ro) if supervisor else None
        if reply is not None and reply.type not in ("FLAG_ERROR", "SUGGEST", "SILENT", None):
            state.utterance_log.append({"t": ro.tick, "speaker": "vlm", "text": reply.text})
        if reply is not None and reply.type in ("FLAG_ERROR", "SUGGEST"):
            state.utterance_log.append({"t": ro.tick, "speaker": "vlm", "text": reply.text})
            gate = checks.flag_is_correct(
                {"error": reply.error, "refers_to": reply.refers_to},
                state.records, lang_log=ro.lang_log, now=ro.tick)
            report["gate"] = gate
            if gate["accepted"]:
                system.enter_error_correction(ro, state)
                report["entered_correction"] = True
                parse = {"suggested_subgoals": reply.suggested_subgoals or [],
                         "suggested_actions": reply.suggested_actions or []}
                filt = system.convert_parse(parse, state.profile, state.recipe, ro.game)
                report["filter"] = {k: filt[k] for k in
                                    ("kept", "dropped_by_acceptance", "unusable_for_solvability",
                                     "anything_applicable")}
                if filt["anything_applicable"]:
                    system.apply_fix(ro, state, filt["commands"])
                    report["fix_source"] = "suggestion"
                elif state.profile.task_solvability == "high":
                    system.apply_fix(ro, state, system.self_repair_plan(state, ro.game))
                    report["fix_source"] = "self_repair"
            else:
                report["false_positive"] = True
        if state.mode == "error_correction" and state.correction is not None \
                and state.correction.get("fix_applied") and system.resolved(ro, state):
            system.exit_correction(ro, state)
            report["resolved_at"] = ro.tick
        if not ro.step():
            break
        if len(ro.lang_log) > seen_lang:
            ev = ro.lang_log[-1]
            seen_lang = len(ro.lang_log)
            if ev.get("subtype") == "ANNOUNCE_PLAN_CHANGE":
                # [v2] the announcement IS the change — applied the moment it fires
                report["announced_change"] = system.apply_announced_change(ro, state, ev)
    report.update(outcome=ro.outcome, total_time=ro.tick,
                  mode_at_end=state.mode)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--error", default="omit_subgoal")
    ap.add_argument("--kwargs", default='{"sid": "assemble(cheese)"}')
    ap.add_argument("--persona", default="capable_collaborator")
    ap.add_argument("--supervisor", choices=["perfect", "worst", "none"], default="perfect")
    ap.add_argument("--delay", type=int, default=0)
    args = ap.parse_args()
    state, ro = build_episode((args.error, json.loads(args.kwargs)), args.persona)
    sup = (PerfectSupervisor(state, delay=args.delay) if args.supervisor == "perfect"
           else WorstSupervisor() if args.supervisor == "worst" else None)
    rep = run_episode(state, ro, sup)
    print(json.dumps(rep, indent=2, default=str))
