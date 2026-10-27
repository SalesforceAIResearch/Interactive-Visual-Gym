#!/usr/bin/env python3
"""Every error type at every timing, on every recipe shape — headless.

Timing is as much a part of an error as its type: wrong_order swapping slots 1-2 is a
different manipulation from swapping 3-4, and the plans they produce fail in different ways.
A live episode costs 25 minutes and one LLM's mood; these run the same plans through the real
engine in milliseconds, so this is where engine bugs should be found.

Recipe shapes deliberately differ:
  Cheeseburger    ordered, 3 assembles, ONE heat group  -> prep(meat) lights the pan itself
  Loaded Burrito  ordered, 5 assembles, TWO heat groups -> LATE-IGNITE: prep only LOADS, and a
                                                           separate ignite(x) lights it later
  Fish & Chips    unordered, batched potato, two heat groups
  Caprese         unordered, chop-only, no heat at all

The late-ignite shape is the interesting one: between prep(x) and ignite(x) the ingredient is
sitting in a cold station, which is neither "absent" nor "on its way to cooked".

    COOKSIM_GRID=tmp/grid.json python -m pytest tests/test_error_matrix.py -q
"""
from __future__ import annotations
import itertools
import os
import sys

import pytest

from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
sys.path.insert(0, ROOT)

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, os.path.join(ROOT, "benchmark", "grid.json")))

if not os.path.exists(os.environ["COOKSIM_GRID"]):
    pytest.skip("needs tmp/grid.json", allow_module_level=True)

from cooksim.core import error_lib as E                                     # noqa: E402
from cooksim.core.enums import PrepState                                    # noqa: E402
from cooksim.core.items import Plate                                        # noqa: E402
from cooksim.core.plan_ir import compile_ir, segment_plan                   # noqa: E402
from cooksim.interact import system                                         # noqa: E402
from cooksim_bench.exp_interact import build_episode                                      # noqa: E402

CELLS = {
    "cheeseburger": "medium_nops_medium_map",
    "burrito": "master_nops_medium_map",
    "fishchips": "hard_nops_medium_map",
    "caprese": "easy_nops_medium_map",
}
ORDERED = {"cheeseburger", "burrito"}


# ------------------------------------------------------------------ helpers
def ep(cell_key):
    state, ro = build_episode(None, cell=CELLS[cell_key])
    return state, ro, ro.game


def sids(state, typ=None):
    return [n.sid for n in segment_plan(state.recipe) if typ is None or n.type == typ]


def compose(state, call):
    return E.compose(list(segment_plan(state.recipe)), [call])


def load_plan(ro, ir):
    ro.cmds = [(c, sid) for c, sid, _d in compile_ir(ir)]
    ro.i = -1
    return ro


def drive(ro, upto=None, cap=2000):
    """Run the queue. Returns True if it reached a resting state rather than running out."""
    ro.cap = max(getattr(ro, "cap", 0), cap)
    for _ in range(cap):
        idle = ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
        if idle or (upto is not None and ro.i >= upto and ro.ctrl.skill is None):
            return True
        if not ro.step():
            return True
    return False


def plate_of(game):
    h = game.cooks[0].holding
    if isinstance(h, Plate):
        return h
    for it in game.counter_items.values():
        if isinstance(it, Plate):
            return it
    return None


def plate_names(game):
    p = plate_of(game)
    return [c.name for c in p.contents] if p else []


def resolve(state, game, sid):
    # the skill marks steps naming a kind this map has several of (2026-09-11); the cook's plan
    # ingestion strips that reminder, so the executability checks below see what gets queued
    from cooksim.agents.highlevel import strip_grounding_reminder
    r = system.engine_call("resolve_subgoal", {"subgoal": sid.split("#")[0]}, state, game)
    if r.get("ok"):
        r["result"]["actions"] = [strip_grounding_reminder(a) for a in r["result"]["actions"]]
    return r


def block_is_groundable(game, acts, label):
    """A resolved block must be RUNNABLE from the current kitchen, not merely well-formed.

    Two ways a block silently cannot run, both found in the wild:
      * it navigates to "the <station> with the <x>" when nothing holds x and the block never
        fetches one (assemble(meat) after the patty was binned; ignite(rice) on a cold start)
      * it picks something up while the plate is in hand and never parks the plate first, so
        the pick-up is illegal-skipped and every command after it fails
    """
    fetched = {a.split("go to the ")[1].split(" source")[0]
               for a in acts if a.startswith("go to the") and a.endswith("source")}
    for a in acts:
        if a.startswith("go to the") and " with the " in a:
            comp = a.split(" with the ")[-1].strip()
            assert system.station_holding(game, comp) is not None or comp in fetched, (
                f"{label}: '{a}' cannot ground — nothing holds {comp} and the block never "
                f"fetches one: {acts}")
    if "pick up" in acts and system.plate_held(game):
        i = acts.index("pick up")
        assert any(x.startswith("go to the counter") for x in acts[:i]), (
            f"{label}: picks up at index {i} while holding the plate, with no park first: {acts}")


def assert_resolvable_everywhere(state, game, ir, label):
    """The invariant that matters, for EVERY subgoal type — not just assemble.

    Restricting this to assemble is what let the identical bug survive in ignite(): the check
    passed while `ignite(rice)` was handing back "go to the pot with the rice" for a kitchen
    with no rice in any pot.
    """
    for node in ir:
        base = node.sid.split("#")[0]
        r = resolve(state, game, base)
        if not r["ok"]:
            assert len(r["error"]) > 20, f"{label}: bare refusal for {base}"
            continue
        acts = r["result"]["actions"]
        if not acts:
            assert r["result"].get("already_satisfied"), (
                f"{label}: {base} resolved to an empty block with no reason given")
            continue
        block_is_groundable(game, acts, f"{label}:{base}")
        if base.startswith("assemble("):
            comp = base[len("assemble("):-1]
            if not system.ingredient_ready(game, comp,
                                           system.required_state(state.recipe, comp)):
                assert r["result"].get("prerequisite_inserted"), (
                    f"{label}: {base} returned a bare block with no usable {comp}: {acts}")


# ================================================================== wrong_order: EVERY pair
def _pairs(state):
    a = [s for s in sids(state, "assemble")]
    return list(itertools.combinations(a, 2))


@pytest.mark.parametrize("cell_key", sorted(ORDERED))
def test_wrong_order_every_pair_composes_and_stays_executable(cell_key):
    """Swapping slots 1-2 is not the same manipulation as swapping 4-5. Cover all of them."""
    state0, _, _ = ep(cell_key)
    pairs = _pairs(state0)
    assert len(pairs) >= 3, f"{cell_key} should have several assemble slots"
    for a, b in pairs:
        state, ro, g = ep(cell_key)
        ir, _ = compose(state, ("wrong_order", {"a": a[len("assemble("):-1],
                                                "b": b[len("assemble("):-1]}))
        order = [n.sid for n in ir if n.type == "assemble"]
        clean = [n.sid for n in segment_plan(state.recipe) if n.type == "assemble"]
        assert order != clean, f"{cell_key} swap {a}/{b} did not change the assembly order"
        assert sorted(order) == sorted(clean), "a swap must not add or drop a component"
        load_plan(ro, ir)
        assert drive(ro), f"{cell_key} swap {a}/{b} never reached a resting state"
        assert_resolvable_everywhere(state, g, ir, f"{cell_key}:swap{a}/{b}")


@pytest.mark.parametrize("cell_key", sorted(ORDERED))
def test_wrong_order_actually_reverses_the_two_slots(cell_key):
    state0, _, _ = ep(cell_key)
    for a, b in _pairs(state0):
        state, ro, g = ep(cell_key)
        ir, _ = compose(state, ("wrong_order", {"a": a[len("assemble("):-1],
                                                "b": b[len("assemble("):-1]}))
        order = [n.sid for n in ir if n.type == "assemble"]
        assert order.index(a) > order.index(b), f"{a} should now follow {b}"


# ================================================================== timing sweep, all errors
def _calls_for(state, cell_key):
    """Every error type, anchored at every legal point in THIS recipe's plan."""
    asm = sids(state, "assemble")
    preps = sids(state, "prep")
    comps = [s[len("assemble("):-1] for s in asm]
    out = []
    for i, s in enumerate(asm):
        out.append((f"premature_serve@{i}", ("premature_serve", {"after": s})))
        out.append((f"forget_next_step@{i}", ("forget_next_step", {"after": s})))
        out.append((f"omit_subgoal@{i}", ("omit_subgoal", {"sid": s})))
        out.append((f"add_foreign@{i}", ("add_foreign", {"wrong": "lettuce", "before": s})))
    for i, c in enumerate(comps):
        out.append((f"serve_raw@{c}", ("serve_raw", {"component": c})))
        out.append((f"misidentify@{c}", ("misidentify_assemble",
                                         {"component": c, "wrong": "lettuce"})))
    for p in preps:
        c = p[len("prep("):-1].split("#")[0]
        out.append((f"mis_prep@{c}", ("mis_prep", {"component": c, "wrong": "lettuce"})))
        out.append((f"leave_unattended@{c}", ("leave_unattended", {"component": c, "loops": 1})))
    # repeat_step: every (earlier, later) anchor pair
    for i in range(len(asm)):
        for j in range(i + 1, len(asm)):
            out.append((f"repeat_step@{i}->{j}", ("repeat_step", {"sid": asm[i], "at": asm[j]})))
    return out


@pytest.mark.parametrize("cell_key", sorted(CELLS))
def test_every_error_at_every_anchor_composes(cell_key):
    """Composition itself is a check: a transform that cannot find its anchor, or that produces
    a plan referencing a subgoal it just deleted, raises here rather than 25 minutes into a run."""
    state0, _, _ = ep(cell_key)
    calls = _calls_for(state0, cell_key)
    assert len(calls) >= 15, f"{cell_key}: only {len(calls)} anchors generated"
    failures = []
    for label, call in calls:
        state, _, _ = ep(cell_key)
        try:
            ir, recs = compose(state, call)
            assert ir, "empty plan"
            assert recs, "no deviation record"
        except Exception as e:
            failures.append(f"{label}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("cell_key", sorted(CELLS))
def test_every_error_at_every_anchor_reaches_a_resting_state(cell_key):
    """No composed plan may run forever. This is the livelock check, done without an LLM."""
    state0, _, _ = ep(cell_key)
    stuck = []
    for label, call in _calls_for(state0, cell_key):
        state, ro, g = ep(cell_key)
        try:
            ir, _ = compose(state, call)
        except Exception:
            continue                      # composition is covered by the test above
        load_plan(ro, ir)
        if not drive(ro, cap=2500):
            stuck.append(label)
    assert not stuck, f"{cell_key}: plans that never settled: {stuck}"


@pytest.mark.parametrize("cell_key", sorted(CELLS))
@pytest.mark.parametrize("frac", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         ids=["t0", "t20", "t40", "t60", "t80", "end"])
def test_assembles_resolve_at_every_point_of_every_error(cell_key, frac):
    """The prerequisite invariant, swept across BOTH axes the presenter asked for: which error,
    and how far into it."""
    state0, _, _ = ep(cell_key)
    for label, call in _calls_for(state0, cell_key):
        state, ro, g = ep(cell_key)
        try:
            ir, _ = compose(state, call)
        except Exception:
            continue
        load_plan(ro, ir)
        drive(ro, upto=max(0, int(len(ro.cmds) * frac) - 1), cap=2500)
        assert_resolvable_everywhere(state, g, ir, f"{cell_key}:{label}@{frac}")


# ================================================================== recovery-shaped states
@pytest.mark.parametrize("cell_key", sorted(CELLS))
def test_after_a_trash_every_component_can_be_remade(cell_key):
    """The live failure generalised: the cook bins its work mid-episode. Every remaining
    assembly must still be reachable, for every error type and every anchor."""
    state0, _, _ = ep(cell_key)
    for label, call in _calls_for(state0, cell_key)[::3]:      # stride: this one is expensive
        state, ro, g = ep(cell_key)
        try:
            ir, _ = compose(state, call)
        except Exception:
            continue
        load_plan(ro, ir)
        drive(ro, upto=int(len(ro.cmds) * 0.6), cap=2500)
        p = plate_of(g)
        if p is not None:
            p.contents.clear()
        for st in g.stations.values():
            if getattr(st, "contents", None):
                st.reset()
        assert_resolvable_everywhere(state, g, ir, f"{cell_key}:{label}:after-trash")


# ================================================================== the late-ignite shape
@pytest.mark.parametrize("cell_key", ["burrito", "fishchips"])
def test_loaded_but_unlit_station_is_lit_in_place_not_re_fetched(cell_key):
    """LATE-IGNITE recipes leave the ingredient in a COLD station between prep(x) and ignite(x).
    It is present, so the fix is to light it. Fetching a second one wastes the ingredient and can
    overflow the station's capacity — and the engine did exactly that until missing_step()
    started distinguishing absent / cold / cooking."""
    state, ro, g = ep(cell_key)
    ir = segment_plan(state.recipe)
    upto = sum(len(n.commands) for n in ir if n.type == "prep") - 1
    load_plan(ro, list(ir))
    drive(ro, upto=upto, cap=2500)

    cold = []
    for st in g.stations.values():
        d = st.render_state()
        if d.get("status") == "raw" and (d.get("contents") or []):
            cold.append(d["contents"][0]["name"])
    assert cold, f"{cell_key}: expected a loaded-but-unlit station after prep"

    for comp in cold:
        if system.required_state(state.recipe, comp) != "cooked":
            continue
        acts = resolve(state, g, f"assemble({comp})")["result"]["actions"]
        assert f"go to the {comp} source" not in acts, \
            f"{cell_key}: assemble({comp}) re-fetches a second {comp}: {acts}"
        assert "turn on" in acts, f"{cell_key}: assemble({comp}) never lights the station: {acts}"
        assert any(a.startswith("wait until") for a in acts), acts


@pytest.mark.parametrize("cell_key", ["cheeseburger", "burrito"])
def test_the_three_situations_each_produce_an_EXECUTABLE_block(cell_key):
    """absent / cold / cooking each get a different block. Run each one and require the
    component to actually reach the plate — a block that merely looks right is worth nothing."""
    for situation in ("absent", "cold", "cooking"):
        state, ro, g = ep(cell_key)
        ir = segment_plan(state.recipe)
        comp = next(n.component for n in ir
                    if n.type == "assemble"
                    and system.required_state(state.recipe, n.component) == "cooked")
        upto = sum(len(n.commands) for n in ir if n.type in ("prep", "ignite", "fetch_plate")) - 1
        load_plan(ro, list(ir))
        drive(ro, upto=upto, cap=2500)
        if plate_of(g) is None:                       # make sure a plate is in hand to scoop into
            continue
        found = system.station_holding(g, comp)
        if situation == "absent" and found:
            found[0].reset()
        elif situation == "cold" and found:
            found[0].status, found[0].timer = "raw", 0
        elif situation == "cooking" and found:
            found[0].status, found[0].timer = "cooking", 1

        r = resolve(state, g, f"assemble({comp})")
        assert r["ok"] is True, f"{cell_key}/{situation}: {r.get('error')}"
        load_plan(ro, [])
        ro.cmds = [(a, "fix") for a in r["result"]["actions"]]
        ro.i = -1
        drive(ro, cap=3000)
        assert comp in plate_names(g), (
            f"{cell_key}/{situation}: block did not deliver {comp}; "
            f"plate={plate_names(g)} actions={r['result']['actions']}")


# ================================================================== batched components
def test_batched_component_counts_as_one_instance():
    """Fish & Chips needs TWO potato portions cooked in ONE basket and plated with ONE scoop.
    A readiness check that counts instances rather than stations would ask for a second batch."""
    state, ro, g = ep("fishchips")
    ir = segment_plan(state.recipe)
    upto = sum(len(n.commands) for n in ir if n.type in ("prep", "ignite")) - 1
    load_plan(ro, list(ir))
    drive(ro, upto=upto, cap=2500)
    for st in g.stations.values():
        d = st.render_state()
        if d.get("contents") and d["contents"][0]["name"] == "potato":
            st.status, st.timer = "cooked", getattr(st, "cook_time", 1)
            for c in st.contents:
                c.state = PrepState.COOKED
    assert system.ingredient_ready(g, "potato", "cooked") is True
    from cooksim.core.items import Plate
    g.cooks[0].holding = Plate()          # isolate the ingredient prerequisite from the plate one
    r = resolve(state, g, "assemble(potato)")
    assert r["result"].get("prerequisite_inserted") is None, \
        "a cooked batch was mistaken for a missing ingredient"


# ================================================================== unordered recipes


# ================================================================== impossible must be SAID
def test_no_free_counter_is_refused_not_answered_with_a_dead_block():
    """Prep needs empty hands. _with_hands_free returns the UNWRAPPED block when there is
    nowhere to park the plate, and running that block accomplishes nothing: the pick-up is
    illegal-skipped, the rest fails/rejects, the plate stays empty and the subgoal is still
    outstanding. Same class of bug as the original livelock, just quieter."""
    from cooksim.core.enums import Terrain
    from cooksim.core.items import Ingredient
    state, ro, g = ep("cheeseburger")
    ir = segment_plan(state.recipe)
    load_plan(ro, list(ir))
    drive(ro, upto=sum(len(n.commands) for n in ir[:2]) - 1, cap=2500)
    for st in g.stations.values():
        if getattr(st, "contents", None):
            st.reset()                                   # destroy the only patty
    for y, row in enumerate(g.layout.grid):
        for x, t in enumerate(row):
            if t == Terrain.COUNTER:
                g.counter_items[(x, y)] = Ingredient(name="lettuce", state=PrepState.RAW)

    assert system.free_counter(g) is None
    r = resolve(state, g, "assemble(meat)")
    assert r["ok"] is False
    assert "counter" in r["error"] and "hands" in r["error"], r["error"]


def test_a_refusal_always_explains_itself():
    """Every refusal has to be actionable — a bare 'no' just gets re-asked."""
    state, ro, g = ep("cheeseburger")
    r = resolve(state, g, "assemble(unobtainium)")
    assert r["ok"] is False and len(r["error"]) > 20


def test_missing_step_never_returns_a_block_that_needs_hands_it_cannot_free():
    """Property form of the above, over every recipe and every component."""
    from cooksim.core.enums import Terrain
    from cooksim.core.items import Ingredient
    for key in sorted(CELLS):
        state, ro, g = ep(key)
        ir = segment_plan(state.recipe)
        load_plan(ro, list(ir))
        drive(ro, upto=sum(len(n.commands) for n in ir if n.type != "serve") - 1, cap=3000)
        for st in g.stations.values():
            if getattr(st, "contents", None):
                st.reset()
        for y, row in enumerate(g.layout.grid):
            for x, t in enumerate(row):
                if t == Terrain.COUNTER:
                    g.counter_items[(x, y)] = Ingredient(name="lettuce", state=PrepState.RAW)
        if not system.plate_held(g):
            continue
        for comp in [c for c, _s in state.recipe.contents]:
            need = system.missing_step(g, state.recipe, comp)
            if isinstance(need, list) and "pick up" in need:
                assert any(a.startswith("go to the counter") for a in need), (
                    f"{key}/{comp}: block picks something up but never frees the hands: {need}")


# ================================================================== ignite: same blindness
@pytest.mark.parametrize("cell_key", ["burrito", "fishchips"])
def test_ignite_of_a_missing_component_loads_it_first(cell_key):
    """`ignite(x)` grounds to "go to the <station> with the <x> / turn on", which cannot ground
    when nothing holds x — the command fails and the subgoal can never retire. Same class as the
    assemble bug, found by asking every template subgoal on an emptied kitchen."""
    state, ro, g = ep(cell_key)
    ignites = [n.sid for n in segment_plan(state.recipe) if n.type == "ignite"]
    assert ignites, f"{cell_key} has no ignite subgoal"
    for st in g.stations.values():
        if getattr(st, "contents", None):
            st.reset()
    for sid in ignites:
        comp = sid[len("ignite("):-1]
        r = resolve(state, g, sid)
        assert r["ok"] is True, r.get("error")
        acts = r["result"]["actions"]
        assert r["result"].get("prerequisite_inserted") == f"prep({comp})", acts
        assert f"go to the {comp} source" in acts, acts
        assert acts.count("turn on") == 1, f"double ignition: {acts}"


@pytest.mark.parametrize("cell_key", ["burrito", "fishchips"])
def test_ignite_is_left_alone_when_the_component_is_already_loaded(cell_key):
    state, ro, g = ep(cell_key)
    ir = segment_plan(state.recipe)
    load_plan(ro, list(ir))
    drive(ro, upto=sum(len(n.commands) for n in ir if n.type == "prep") - 1, cap=2500)
    for sid in [n.sid for n in ir if n.type == "ignite"]:
        r = resolve(state, g, sid)
        assert r["ok"] is True
        assert r["result"].get("prerequisite_inserted") is None, r["result"]["actions"]


def test_every_template_subgoal_resolves_on_an_emptied_kitchen():
    """Property sweep: ask for EVERY subgoal of every recipe with the stations wiped. Each must
    either come back executable or refuse with a reason. This is how the ignite bug surfaced."""
    for key in sorted(CELLS):
        state, ro, g = ep(key)
        for st in g.stations.values():
            if getattr(st, "contents", None):
                st.reset()
        for n in segment_plan(state.recipe):
            r = resolve(state, g, n.sid)
            if not r["ok"]:
                assert len(r["error"]) > 20, f"{key}/{n.sid}: bare refusal"
                continue
            acts = r["result"]["actions"]
            assert acts, f"{key}/{n.sid}: empty block"
            for a in acts:
                if a.startswith("go to the") and " with the " in a:
                    comp = a.split(" with the ")[-1].strip()
                    assert (system.station_holding(g, comp) is not None
                            or any(x == f"go to the {comp} source" for x in acts)), (
                        f"{key}/{n.sid}: '{a}' cannot ground — nothing holds {comp} and the "
                        f"block never fetches one: {acts}")


# ================================================================== hand state is a prerequisite
HANDS = ["empty", "plate", "ingredient"]


@pytest.mark.parametrize("cell_key", sorted(CELLS))
@pytest.mark.parametrize("hands", HANDS)
def test_every_subgoal_respects_what_is_already_in_the_cook_s_hands(cell_key, hands):
    """What the cook is holding is a prerequisite like any other, and it was unchecked:

      serve with empty hands      -> "go to the serving pass / serve" with nothing to hand over
      assemble with empty hands   -> scoops into thin air
      prep holding an ingredient  -> the pick-up is illegal-skipped (_with_hands_free only knew
                                     about a held PLATE)
      fetch_plate holding a plate -> picks up a second one

    Each was ok=True with a block that could not run — the same silent failure that burned a
    whole episode's wall clock.
    """
    from cooksim.core.items import Ingredient, Plate
    state, ro, g = ep(cell_key)
    g.cooks[0].holding = (None if hands == "empty" else
                          Plate() if hands == "plate" else
                          Ingredient(name="meat", state=PrepState.RAW))
    for n in segment_plan(state.recipe):
        r = resolve(state, g, n.sid)
        if not r["ok"]:
            assert len(r["error"]) > 20, f"{cell_key}/{hands}/{n.sid}: bare refusal"
            continue
        acts = r["result"]["actions"]
        if not acts:
            assert r["result"].get("already_satisfied")
            continue
        block_is_groundable(g, acts, f"{cell_key}/{hands}/{n.sid}")
        if n.sid.startswith("assemble("):
            # must end up holding the plate before the scoop
            if hands != "plate":
                assert "pick up" in acts, f"{cell_key}/{hands}/{n.sid}: never picks the plate up: {acts}"
                assert acts.index("pick up") < len(acts) - 1


@pytest.mark.parametrize("cell_key", sorted(CELLS))
def test_serve_is_refused_without_the_plate(cell_key):
    state, ro, g = ep(cell_key)
    g.cooks[0].holding = None
    r = resolve(state, g, "serve")
    assert r["ok"] is False and "plate" in r["error"]


def test_a_parked_plate_is_retrieved_rather_than_a_fresh_one_taken():
    """Picking a second plate off the dispenser abandons everything already plated on the first."""
    from cooksim.core.items import Plate
    from cooksim.core.enums import Terrain
    state, ro, g = ep("cheeseburger")
    g.cooks[0].holding = None
    spot = next((x, y) for y, row in enumerate(g.layout.grid)
                for x, t in enumerate(row) if t == Terrain.COUNTER)
    g.counter_items[spot] = Plate()
    acts = resolve(state, g, "assemble(bun)")["result"]["actions"]
    assert f"go to the counter ({spot[0]},{spot[1]})" in acts, acts
    assert "go to the plates" not in acts, acts


# ================================================================== case definitions vs map


# ================================================================== mis_prep on late-ignite
LATE_IGNITE = ["burrito", "fishchips"]


@pytest.mark.parametrize("cell_key", ["cheeseburger", "burrito", "caprese"])
def test_mis_prep_leaves_no_orphaned_ignite(cell_key):
    """LATE-IGNITE recipes split prep into "load the station" and a separate ignite(x) that
    lights it later. err_mis_prep replaced prep(x) but left ignite(x) behind, pointing at a
    station that now holds nothing — "go to the fryer with the fish" cannot ground and the whole
    tail of the plan collapses behind it (9 illegal-skipped / 5 rejected / 2 failed / no
    outcome). A confused cook lights the station it actually loaded."""
    state, ro, g = ep(cell_key)
    comp = next(n.component for n in segment_plan(state.recipe) if n.type == "prep")
    ir, _ = compose(state, ("mis_prep", {"component": comp, "wrong": "lettuce"}))
    sids_after = [n.sid for n in ir]
    assert f"prep({comp})" not in sids_after
    assert f"ignite({comp})" not in sids_after, (
        f"{cell_key}: ignite({comp}) survived a mis_prep that removed its component: {sids_after}")


@pytest.mark.parametrize("cell_key", ["cheeseburger", "burrito", "caprese"])
def test_mis_prep_still_reaches_a_definite_outcome(cell_key):
    """The error must produce a WRONG DISH, not a collapsed plan."""
    from collections import Counter
    state, ro, g = ep(cell_key)
    comp = next(n.component for n in segment_plan(state.recipe) if n.type == "prep")
    ir, _ = compose(state, ("mis_prep", {"component": comp, "wrong": "lettuce"}))
    load_plan(ro, ir)
    drive(ro, cap=3000)
    st = Counter(l.get("status") for l in ro.log)
    fails = st.get("rejected", 0) + st.get("illegal-skipped", 0)
    assert ro.outcome is not None, f"{cell_key}: mis_prep produced no outcome at all ({dict(st)})"
    assert fails <= 3, f"{cell_key}: mis_prep plan collapsed ({dict(st)})"


def test_known_limitation_mis_prep_needs_a_spare_cutting_board(monkeypatch):
    """Since 2026-09-19 every kitchen gets a second board (tools/exp_interact.py, suite v5 round 2),
    which removes this limitation on the benchmark maps; COOKSIM_MIN_BOARDS=1 restores the old
    one-board map, where it still holds.

Documents a real limitation rather than pretending it is fixed.

    When the heat station refuses the wrong ingredient, err_mis_prep routes it to the cutting
    board and LEAVES it there. Fish & Chips has exactly one board and its potato prep is a chain
    ingredient that needs the same board, so the wrong item blocks it and the plan collapses.
    The IR layer has no map access, so it cannot pick a free board or park the item — fixing it
    properly means giving the transform a kitchen. Until then, mis_prep is not used on
    single-board maps.
    """
    from collections import Counter
    from cooksim_bench import exp_interact as _xi
    monkeypatch.setattr(_xi, "MIN_BOARDS", 1)      # read at import time
    state, ro, g = ep("fishchips")
    boards = sum(1 for row in g.layout.grid for t in row if t.name == "CUTTING_BOARD")
    assert boards == 1, "if Fish & Chips gained a board, re-check whether this limitation still holds"
    ir, _ = compose(state, ("mis_prep", {"component": "fish", "wrong": "lettuce"}))
    load_plan(ro, ir)
    drive(ro, cap=3000)
    assert ro.outcome is None       # the documented breakage; flip this test when it is fixed




# ================================================================== speech vs. motion
def test_a_walk_continues_while_the_cook_speaks():
    """Freezing the cook outright for the length of its own line cost a chatty persona 74-88%
    of every episode (novice_stepwise/frames: 198 of 267 ticks frozen, 69 usable), and the
    assistant then correctly reported "you are not moving" — which drew another reply, which
    froze it again. Talking does not stop your legs."""
    from cooksim_bench import exp_error_rollout as X
    state, ro, g = ep("burrito")
    load_plan(ro, list(segment_plan(state.recipe)))
    ro.cap = 600
    ro.step()
    assert isinstance(ro.ctrl.skill, X._GoTo), "test needs a walk in flight"
    start = tuple(g.cooks[0].pos)
    ro._pause = 6
    seen = []
    for _ in range(6):
        ro.step()
        seen.append(tuple(g.cooks[0].pos))
    assert any(p != start for p in seen), f"cook did not move while speaking: {seen}"


def test_a_manipulation_still_stops_while_the_cook_speaks():
    """The parity rationale survives for ACTIONS: you stop to chop while you explain."""
    from cooksim_bench import exp_error_rollout as X
    state, ro, g = ep("cheeseburger")
    load_plan(ro, list(segment_plan(state.recipe)))
    ro.cap = 400
    for _ in range(6):
        ro.step()
        if ro.ctrl.skill is not None and not isinstance(ro.ctrl.skill, X._GoTo):
            break
    assert ro.ctrl.skill is not None and not isinstance(ro.ctrl.skill, X._GoTo), \
        "test needs a manipulation in flight"
    held = g.cooks[0].holding
    ro._pause = 5
    for _ in range(5):
        ro.step()
    assert g.cooks[0].holding is held, "a manipulation ran while the cook was speaking"


def test_the_pause_never_clears_the_queue_or_the_skill():
    """The queue survives a pause untouched — this was the thing to rule out first."""
    state, ro, g = ep("cheeseburger")
    load_plan(ro, list(segment_plan(state.recipe)))
    ro.cap = 400
    for _ in range(4):
        ro.step()
    cmds_before, i_before = list(ro.cmds), ro.i
    ro._pause = 5
    for _ in range(5):
        ro.step()
    assert [c for c, _ in ro.cmds] == [c for c, _ in cmds_before], "pause rewrote the queue"
    assert ro.i == i_before, f"pause advanced the cursor {i_before} -> {ro.i}"
