#!/usr/bin/env python3
"""Suite v5 designer: rebuild the 50-case composite benchmark so that every case is well designed on the engine with
world-read question truths and reference-free firing (main 262ac44).

    $PY python -m cooksim_bench.suite.design_composite_v5 [--base bench/v5/cases_composite_v4.json] [--out bench/v5/cases_composite_v5.json]
                                     [--inventory bench/v5/error_inventory_v5.json] [--seed 5]

Lineage. v3 = python -m cooksim_bench.suite.design_composite_v3 (random admissible sets from the replay-checked error inventory, questions
by content hardness); v4 = python -m cooksim_bench.suite.curate_cases_v5 (mis_prep wrong items) + the 2026-09-15/17 redesigns and
python -m cooksim_bench.suite.backfill_after_search_delete (rules R1-R11 learned from never-firing sub-cases). This designer keeps the v4
case ids, cells, kitchens flags and tiers, re-picks every case's errors from the inventory under the rules below, and
re-picks its questions, standing request and plan change. v4 errors that still pass every rule are preferred (a case
changes only where a rule says it must, or where the diversity rules need it).

Error rules (each is a way a sub-case failed to fire, or read wrong, in the 2026-09-19 audit; tmp/suite_audit/):
  T   errors per tier 2/2/3/4/4, at least half ruinous; ruinous is the committed form's EFFECT (C6): mis_prep is not
      ruinous (the cook recovers the right item; the wrong one is left on a board), forget_next_step is not
      (time only); everything that leaves the served dish wrong, and the walk-away (smoke on the station), is
  L   one error LABEL per case (as since v3)
  R2  add_foreign (fn add_foreign or misidentify_assemble) scoops an OFF-RECIPE ingredient (ruling R2); every
      kitchen has one (cooksim_bench/exp_interact._ensure_off_recipe_source)
  C9  mis_prep's wrong item is off-recipe, choppable, and not taken raw by the component's own station, so it goes
      to the cutting board (a shrimp or egg loaded into a lit fryer/pan burned untracked)
  W   every foreign / mis_prep wrong item differs within a case (two mushroom mistakes read as one), and differs from
      the plan change's ingredient (C5)
  C1  one mistake per component / plate slot (slots(): the component, a swapped pair, an omission's item and the
      item it plates instead, a premature serve's missing items, the walk-away's food, a repeated item)
  C2  a premature serve is the last moment of its case: every other error, question, request and plan change comes
      strictly before it
  M   moments (moment()): no two body sub-cases share a moment or sit on adjacent ones (< MIN_GAP apart: back-to-back
      takeovers, or a walk-away starved by a takeover over its near-done window); a forget's 60-tick idle occupies
      FORGET_SPAN moments after it (food cooking through it smokes, which starves the walk-away and the plating
      mistakes on that food)
  WO  wrong_order(a, b): b is plated first, so b must be raw or chopped (never cooked food waiting on the heat: the
      smoke ban starved wrong_order(tortilla, rice)), and b is the item right after a or raw, never the last item
  SR  serve_raw only on items the cook plates before they are done: chopped items and the second-stage potatoes;
      never the first cooked item (fish, steak, meat, rice: cooked before the plate is taken, 10% fired)
  O   omit_subgoal(sid X) only where X's next item is ready when X is next (it plates it): not on hard/expert
  RP  repeat_step not at the last item (R5b); on the 5-item burrito the duplicate fills the plate, so no plating
      mistake at or after the lettuce may follow a repeat
  F   forget_next_step never after the second-to-last item (R9) and never forgets the serve
  D   diversity (C4): at most MAX_T1 cases per tier open with a t1 mistake (mis_prep, or wrong_order on a first chopped
      item); no two cases of one cell share an error set; per-tier label caps
Voice rules:
  V1  0 validator errors (python -m cooksim_bench.scoring.validate_suite_questions) and no recipe-determined order question (T1)
  V2  the progress floor of every question/request/plan change is at most len(recipe)-2 plated items and needs only
      prep an LLM cook does early (EARLY_PREP): burrito questions used to wait for chopped lettuce and fired on the
      walk to the pass
  V3  the question's own precondition must be reachable with the plate still two items short (after_ref of the
      second-to-last item, or an order question about the last plating, only hold at a complete plate)
  C7  no question about an item a mistake of the same case replaces, swaps, skips or repeats; no standing request
      about the walk-away's food or an item a mis_prep/serve_raw targets
  S10 the standing request is listed before any other voice sub-case, and no ready_in question on its item

Round 2 (2026-09-19, second hand review tmp/suite_audit/review_v5_*.md, FIX_PLAN_R2.md section D; engine d6e7fd4 +
f672601). Rules added or tightened; the suite is regenerated with `--prev` = the round-1 suite (old errors preferred
where still legal) and the ledger gains a round-2 section:
  R-LU  leave_unattended is NOT ruinous (user ruling): the ≥half-ruinous rule is met by the other errors. Where the tier's
        rules make that impossible it is relaxed per tier (MIN_RUINOUS) and the ledger says so: on Fish & Chips and
        Steak Frites the only ruinous kinds left are one add_foreign-label mistake and a premature serve after the first
        item, which C1/C2+/B forbid together with any third mistake.
  E9    every walk-away carries an EXPLICIT prep_floor/plate_floor (lu_floor): nothing for the first item the LLM cook
        lights (meat, fish, steak), the batch for the potatoes, rice plated for the burrito steak — never the
        reference run's prep, which an LLM cook that lights the first item early never has while it is near done
  D1    no serve_raw(potato): it fires only when the cook plates before the fryer is done
  D2    no two cases of a TIER share an error set up to the foreign item / loop count / potato index, with an
        add_foreign before X counted as a misidentify of X (same committed world)
  D3/L/W one error LABEL per case and one wrong item per case, never relaxed (flag credit is by name only; two
        add_foreign-labelled mistakes, or two mistakes with one item, cannot be told apart and the second commits as a
        phantom); every kitchen has two off-recipe crates (cooksim_bench/exp_interact.OFF_RECIPE_MIN) and the inventory rows are
        copied to the new crate (synth_wrong_items)
  D4    no repeat_step(bun) (a second bun reads as the top bun; the cook then serves without the cheese)
  D5    no add_foreign before a raw first item (lettuce before the bun made the cook drop the bun)
  D6    master variety: no walk-away on the first rice; at most ROW_CAP cases of a tier share one mistake (wrong_order
        (steak, tomato) and forget-after-steak were in all ten)
  D7    a mis_prep's wrong item is accepted by no heat station of its kitchen (a chopped onion went into a pot and
        burned the episode); no `meat` beside a steak (same icon), no pepper (a grinder on the plate)
  CAP   recipe items + extra items (add_foreign-label, repeat) + a plan change's item <= plate capacity 5; nothing
        that needs a plate slot (a forget included: it waits on a full plate) after the burrito's extra item
  WO    a swap advances two moments at once (next(a) .. plated(b)): no other mistake within MIN_GAP of that span
  F     forget_next_step only where nothing lit can smoke inside its freeze (the engine now waits for that): no forget
        in the burger (the meat is lit first and the only forget anchors left are at the plate), and on Fish & Chips /
        Steak Frites the potato forgets are low-yield
  D8    count/visit questions only once the plate is started (the counted thing has had time to happen); order
        questions only about pairs the LLM cooks really do in either order (VARIABLE_ORDER, measured on the 100
        round-1 real-cook episodes); no after_ref question (its answer is the recipe's next item); visit questions
        balanced yes/no; no next-step question in a case with a forget (the engine holds it while a forget is armed);
        questions placed away from the mistakes' moments (the voice slot is shut during and around takeovers)
  PC    a plan change fires with the plate started (it is "on top before serving")
  BOARD every kitchen has two cutting boards (cooksim_bench/exp_interact.MIN_BOARDS; the Fish & Chips medium map had one and a
        mis_prep's leftover blocked it until the episode timed out); a mis_prep never on a one-board kitchen
  TICK  the burrito's moment table is on a tick scale (tick/10) measured on the round-2 pilot episodes; the ordinal
        table put every plating moment one unit from the next and wrongly placed the tomato chop before the plate
  G     a brief moment right after a forget's freeze is lost in the takeover gap: >= 1.5 apart unless a chop/load lies
        between; a STATE moment (forget, premature serve, serve-time add_foreign, raw serve) only needs 0.5
Round 3 (2026-09-19, third hand review tmp/suite_audit/review_r3_*.md, FIX_PLAN_R3 D9-D17; engine F1-F10 + ruling R10:
the cook gets a turn at every takeover release). `--prev` = the round-2 suite (git:386ccb1); ledger section "Round 3";
DESIGN_ROUND=2 restores the round-2 rules.
  D9    spacing at the LLM cook's pace: after a PLATING takeover a state moment needs a full plating step (MIN_GAP) and a
        brief one PLATE_STEP_GAP; never "before X" with "after X" (medium_m1); never a swap that plates the last item with
        a serve-time mistake (medium_m4); no forget-after-plate behind a block that fetched the plate (easy_m3)
  D10   voice: a question is placed at its LIVE moment (plate floor AND prep floor AND its own precondition), pushed past
        any takeover (+G after it, VOICE_QUIET before it); kept only if that effective moment comes before the serve (and
        before a premature serve); ready_in/elapsed about X are dropped when a takeover follows X's ignition or a mis_prep
        of another item comes while X cooks. Requests: the engine now makes a request that is ready on a takeover's tick
        first (e2e_stepin_rollout schedule_v5), so they are no longer starved by the mistake that follows the ignition
  D11   no foreign item / misidentify at the fish or steak (the smoke ban holds it while that food smokes, 5 of 8 missed)
  D12   engine: wrong_order makes its own chopped b (e2e_stepin_rollout live_dev_block)
  D13   "after X" and "before next(X)" share one anchor (forget after the tomato + foreign at the lettuce): forbidden
  D14   hard / expert / master keep >= 1 ROBUST ruinous mistake (is_robust_ruinous: the serve-time foreign item, a premature
        serve, the burrito's last-item foreign item); easy / medium keep >= half ruinous
  D15   no longest_on / burn_first (one candidate in practice); "what did you tell me" only with the plate started and
        after the first mistake; visit questions only about a station whose answer the recipe does not decide (not the
        first-lit item's station, not one the recipe never uses; the trash only after a mistake that leaves something to
        throw away); past_done asked once the first cooked item can be done (engine: past_done needs something lit,
        elapsed needs its item on the heat)
  D16   per-tier caps on cases containing a mistake kind (FN_CASE_CAP) and on the opening mistake (OPENING_CAP), and cases
        that open alike ask different questions
  D17   the burrito steak walk-away is allowed again (rice-plated floor); the rice walk-away stays (engine F1/R10: the cook
        gets its turn at the release)
Structural limits found (reported in the ledger's round-2 section): Steak Frites and Fish & Chips reach at most one
ruinous error in most sets (see R-LU); the burrito's 4-mistake sets all contain the swap or the rice walk-away (the
steak walk-away is starved by the rice smoke, 0/8 in the pilot), so D6 is met by caps (HARD_CAP), not by absence.
"""
import argparse
import collections
import itertools
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from cooksim_bench.paths import ROOT  # the CookSim release root

os.chdir(ROOT)
os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

from cooksim_bench.suite import design_composite_v3 as D3  # noqa: E402
from cooksim_bench.suite import gen_intent_cases as GI  # noqa: E402
from cooksim_bench.scoring import qa_v3_lib as Q  # noqa: E402
import functools                                                   # noqa: E402
from cooksim_bench.exp_interact import recipe_of as _recipe_of                   # noqa: E402

recipe_of = functools.lru_cache(maxsize=None)(_recipe_of)

TIER_N = {"easy": 2, "medium": 2, "hard": 3, "expert": 4, "master": 4}
# C6: ruinous = the committed form loses the dish (wrong plate / smoke on the station) by itself
RUINOUS_FN = {"misidentify_assemble", "add_foreign", "serve_raw", "wrong_order", "omit_subgoal", "premature_serve",
              "repeat_step"}           # R-LU (user ruling 2026-09-19): leave_unattended is not ruinous
# R-LU: the least number of ruinous errors per case. ceil(n/2) everywhere it is reachable; Fish & Chips and Steak
# Frites cannot reach it without a premature serve after the first item in EVERY case (see the module docstring)
MIN_RUINOUS = {"easy": 1, "medium": 1, "hard": 1, "expert": 1, "master": 1}
# measured / expected yield with the LLM cook: serve_raw(tomato) — the burrito's tomato is chopped before the steak is
# plated in 11/13 round-1 episodes; the potato walk-away needs the fish/steak off the heat or not yet smoking; a
# potato forget needs nothing lit within its freeze (engine E4)
LOW_YIELD = {("forget_next_step", "steak@master"): 1, ("wrong_order", "steak"): 1}
ROW_CAP = 5                  # D6: at most this many of a tier's 10 cases share one (normalised) mistake
ROW_CAP_SPECIAL = {("master", "wrong_order"): 5, ("master", "leave_unattended"): 4}
HARD_CAP = {("master", "leave_unattended"): 6, ("master", "wrong_order"): 5}   # D6, never relaxed
if os.environ.get("DESIGN_ROUND", "3") in ("3", "4", "5"):
    # D17 (round 3): the rice walk-away is gone from the burrito; the cap was about it (D6: the first-rice walk-away in
    # every case). The steak walk-away (rice-plated floor) is in every master set the round-3 rules leave (ledger)
    HARD_CAP[("master", "leave_unattended")] = 10
# D8: order-question pairs the LLM cooks did in BOTH orders (each order in >= 25% of the decided round-1 real-cook
# episodes, tmp/suite_v5/llm_final_full, 20 per tier, world truths via qa_world). Every other pair was decided by the
# recipe in practice (e.g. "rice on the heat before chopping the steak": 19/20 the same).
VARIABLE_ORDER = {
    "hard": {"order:ignite(potato)|assemble(fish)", "order:prep(potato)|ignite(fish)", "order:prep(potato)|assemble(fish)"},
    "expert": {"order:ignite(potato)|assemble(steak)", "order:prep(potato)|ignite(steak)"},
    "master": {"order:ignite(steak)|assemble(rice)", "order:ignite(steak)|assemble(tortilla)",
               "order:prep(steak)|assemble(rice)", "order:prep(steak)|assemble(tortilla)",
               "order:prep(tomato)|assemble(rice)", "order:prep(tomato)|assemble(tortilla)"},
}
BAD_FOREIGN = {"pepper"}     # D7: renders as a pepper grinder on the plate
ICON_TWINS = {"steak": {"meat"}, "meat": {"steak"}}   # D7: same icon as a recipe item
MIN_GAP = 1.0            # moments closer than this are the same moment (back-to-back takeovers)
LU_GAP = 1.3             # a walk-away's near-done window needs a little more room
FORGET_SPAN = 2.0        # a forget's 60-tick idle covers this many moments after it (6.0 on the burrito's tick scale)
MAX_T1 = 3               # at most this many cases per tier open with a t1 mistake (C4)
LABEL_CASES_CAP = 6      # no label appears in more than this many of a tier's 10 cases
EARLY_PREP = {"easy": {"tomato"}, "medium": {"meat"}, "hard": {"fish", "potato"}, "expert": {"steak", "potato"},
              "master": {"rice", "steak"}}
Q_PER_CASE = 2
N_PLAN_CHANGES = 6
# ---- round 3 (FIX_PLAN_R3 D9-D17; engine F1-F10 + R10). ROUND3=False reproduces the round-2 rules.
ROUND3 = os.environ.get("DESIGN_ROUND", "3") in ("3", "4", "5")
PLATE_STEP_GAP = 1.5       # D9: a brief moment after a PLATING takeover (release + G at the LLM cook's pace)
FN_CASE_CAP = {("hard", "mis_prep"): 4, ("expert", "forget_next_step@steak"): 5,
               ("master", "wrong_order"): 4}   # D16
OPENING_CAP = 3            # D16: at most this many of a tier's cases open with the same (normalised) mistake
PROACTIVE_TARGET = 32


# ----------------------------------------------------------------------------------------------- recipe helpers
def tier_of(cell):
    return cell.split("_")[0]


@functools.lru_cache(maxsize=None)
def _items_t(cell):
    return tuple(n for n, _ in recipe_of(cell).contents)


def recipe_items(cell):
    return list(_items_t(cell))


def _old_recipe_items(cell):
    return [n for n, _ in recipe_of(cell).contents]


@functools.lru_cache(maxsize=None)
def want_of(cell):
    return {n: getattr(s, "value", s) for n, s in recipe_of(cell).contents}


def comp_of_sid(v):
    v = v or ""
    return v[v.find("(") + 1:v.rfind(")")].split("#")[0] if "(" in v else None


@functools.lru_cache(maxsize=None)
def moment_table(cell):
    """The order in which an LLM cook reaches each moment of this recipe (from the audited LLM episodes): prep and
    loading first (cooked items lit at load), the first cooked item done before the plate is taken, then the plate
    built in recipe order. Returns {token: position}."""
    items = recipe_items(cell)
    want = want_of(cell)
    t = {"reach": 0.0}
    tier = tier_of(cell)
    if tier == "easy":
        t.update({"chop(tomato)": 1.0, "plate0": 2.0, "next(tomato)": 2.0, "next(cheese)": 3.0, "plated(tomato)": 3.0,
                  "plated(cheese)": 4.0, "serve": 4.0})
    elif tier == "medium":
        t.update({"load(meat)": 1.0, "plate0": 2.0, "next(bun)": 2.0, "plated(bun)": 2.5, "done(meat)": 3.2,
                  "next(meat)": 3.6, "plated(meat)": 4.6, "next(cheese)": 4.6, "plated(cheese)": 5.6, "serve": 5.6})
    elif tier in ("hard", "expert"):
        f = items[0]
        if f == "steak":
            t["chop(steak)"] = 0.5      # Steak Frites: the steak is chopped first thing, then goes into the pan
        t.update({f"load({f})": 1.0, "chop(potato)": 2.0, "load(potato)": 3.0, f"done({f})": 4.0, "plate0": 4.6,
                  f"next({f})": 5.0, f"plated({f})": 6.0, "next(potato)": 6.0, "done(potato)": 6.5,
                  "plated(potato)": 7.8, "serve": 7.8})
    else:   # master: tortilla, rice, steak, tomato, lettuce
        # round 2: the burrito is on a TICK scale (tick / 10, medians of the 13 round-2 pilot episodes on this engine,
        # tmp/suite_v5r2/llm_pilot*): the rice lit at t20, done t42; the steak reached t23, lit ~t60, done ~t85; the
        # plate from ~t90 (tortilla t80-123, rice t89-121, steak t96-138); the tomato chopped AFTER the steak is plated
        # (a wrong_order(steak, tomato) found it unchopped 4/5) and plated t165-303; the lettuce chopped last. On the
        # ordinal scale every plating moment was one unit from the next and no 4-mistake set without the swap passed.
        t.update({"load(rice)": 2.0, "done(rice)": 4.2, "chop(steak)": 3.0, "load(steak)": 6.0, "done(steak)": 8.5,
                  "plate0": 9.0, "next(tortilla)": 9.0, "plated(tortilla)": 9.5, "next(rice)": 9.5,
                  "plated(rice)": 11.0, "next(steak)": 11.0, "plated(steak)": 12.0, "next(tomato)": 12.0,
                  "chop(tomato)": 15.0, "plated(tomato)": 20.0, "next(lettuce)": 20.0, "chop(lettuce)": 23.0,
                  "plated(lettuce)": 25.0, "serve": 25.0})
    for x in items:
        if want[x] == "cooked":
            t.setdefault(f"load({x})", 1.0)
        if want[x] == "chopped":
            t.setdefault(f"chop({x})", 1.0)
    return t


def moment(row, cell):
    """(position, token) of an error's live moment (e2e_stepin_rollout.live_trigger) on this recipe."""
    T = moment_table(cell)
    items = recipe_items(cell)
    want = want_of(cell)
    fn, kw = row["fn"], row["kwargs"]

    def at(tok):
        return T[tok], tok
    if fn == "mis_prep":
        x = kw["component"]
        first = next(i for i in items if want[i] in ("cooked", "chopped"))
        if x == first:
            return at("reach")                   # the cook reaches for it first thing (t1)
        if tier_of(cell) in ("hard", "expert") and x == "potato":
            # round 2 (LLM pilot): the cook reaches for the potato right after lighting the fish/steak, and the ~20-tick
            # chop block covers that item's near-done window (a walk-away on it was READY-BLOCKED 4/4)
            return T[f"done({first})"] - 0.5, f"reach({x})"
        tok = f"chop({x})" if f"chop({x})" in T else f"load({x})"
        return T[tok] - 0.5, f"reach({x})"
    if fn == "leave_unattended":
        c = kw["component"]
        _pf, plf = lu_floor(cell, c)
        if plf:
            # E9: the explicit plate floor holds the walk-away until that many items are plated
            p = max(T[f"done({c})"], T.get(f"plated({items[plf - 1]})", T[f"done({c})"]) - 0.4)
            return p, f"done({c})+plated({items[plf - 1]})"
        return at(f"done({c})")
    if fn == "forget_next_step":
        a = kw.get("after", "")
        x = comp_of_sid(a)
        if a == "fetch_plate":
            return at("plate0")
        if a.startswith("assemble("):
            return at(f"plated({x})")
        if a.startswith("ignite("):
            return at(f"load({x})")
        if x == "potato" and tier_of(cell) in ("hard", "expert"):
            # round 2: the forget waits until nothing on the heat can smoke in its freeze (engine E4) — in the LLM
            # pilot it fired as the fish/steak left the heat (t76, t133), before the potatoes were lit
            return T[f"plated({items[0]})"] - 0.6, "prep(potato)+nothing lit"
        # prep(X): chopping for a chopped item or a potato (chopped, then fried); loading for the rest
        return at(f"chop({x})") if f"chop({x})" in T else at(f"load({x})")
    if fn == "wrong_order":
        a = kw["a"]
        if items[0] == a and want[a] in ("cooked", "chopped"):
            return at("reach")                   # fires on the still-empty plate: the block fetches the plate
        return at(f"next({a})")
    if fn == "omit_subgoal":
        x = comp_of_sid(kw["sid"])
        nxt = items[items.index(x) + 1] if items.index(x) + 1 < len(items) else None
        if items[0] == x and want[x] in ("cooked", "chopped"):
            p = T.get(f"chop({x})", T.get(f"done({x})", 1.0)) + 0.5
            return p, f"ready({x})"
        if nxt and want.get(nxt) == "cooked":
            return max(T[f"next({x})"], T[f"done({nxt})"]), f"next({x})+ready({nxt})"
        return at(f"next({x})")
    if fn == "premature_serve":
        a = kw["after"]
        return (at("plate0") if a == "fetch_plate" else at(f"plated({comp_of_sid(a)})"))
    if fn == "repeat_step":
        return at(f"plated({comp_of_sid(kw['at'])})")
    if fn == "add_foreign" and kw.get("before") == "serve":
        return at("serve")
    x = kw.get("component") or comp_of_sid(kw.get("before"))
    if fn in ("misidentify_assemble", "add_foreign") and f"chop({x})" in T and T[f"chop({x})"] > T[f"next({x})"]:
        # the plating mistakes wait for the cook heading to plate x (heading_to_plate): after x is chopped
        return max(T[f"next({x})"], T[f"chop({x})"] + 0.8), f"chopped({x})+heading"
    return at(f"next({x})")


def span(row, cell):
    """(start, end) moments an error's takeover occupies. WO (round 2, master X2): a swap plates b and a in one block,
    so the plate is through both when it releases — its span runs from next(a) to the later of the two platings."""
    p, _tok = moment(row, cell)
    if row["fn"] == "wrong_order":
        T = moment_table(cell)
        a, b = row["kwargs"]["a"], row["kwargs"]["b"]
        items = recipe_items(cell)
        if ROUND3 and want_of(cell).get(b) == "chopped":
            # D12 (round 3): the block makes the chopped b itself and plates b then a at once
            return p, p + (2.5 if tier_of(cell) == "master" else 1.0)
        if ROUND3 and items[0] == a and want_of(cell).get(a) in ("cooked", "chopped") and _tok == "reach":
            # the swap onto the empty plate fires before a is made: the block plates b alone (live_dev_block drops a's
            # scoop) and the cook makes a afterwards — a short takeover
            return p, p + 0.5
        if items.index(b) == items.index(a) + 1:
            return p, max(p, T.get(f"plated({a})", p), T.get(f"plated({b})", p))
        # a raw b from later in the recipe (the burger's cheese): the items between a and b still come in order
        return p, max(p, T.get(f"plated({a})", p))
    return p, p


FORGET_NEXT_GAP = 1.5
STATE_GAP = 0.5
# round 2 (engine E4): a forget waits until no food on the heat can smoke inside its freeze, so the designer's own
# forget-smoke rules (M forget lets X smoke, F the stall) are the engine's job now; its moment is modelled as the
# first moment with nothing lit (moment())
ENGINE_FORGET_GUARD = True


def is_state(row):
    fn, kw = row["fn"], row["kwargs"]
    return fn in ("forget_next_step", "premature_serve", "serve_raw", "repeat_step") or (
        fn == "add_foreign" and kw.get("before") == "serve")


def is_brief(row, cell):
    """A mistake whose live moment lasts only a few ticks (the cook heading to plate an item, reaching for a crate), as
    opposed to a state that holds until the cook changes it (a plate through X, a finished plate, food near done)."""
    fn, kw = row["fn"], row["kwargs"]
    if fn in ("misidentify_assemble", "mis_prep"):
        return True
    if fn == "add_foreign":
        return kw.get("before") != "serve"
    if fn == "omit_subgoal":
        return True
    if fn == "wrong_order":
        items = recipe_items(cell)
        return not (items[0] == kw["a"])
    return False


def lu_floor(cell, c):
    """E9 (round-2 decision): the walk-away's explicit (prep_floor, plate_floor), chosen so the food is still cooking
    when the floor is met in the orders LLM cooks really use:
      * the first item the LLM cook lights (burger meat, fish, Steak Frites steak): no floor — it is near done long
        before anything else is prepped (the reference prep was 'everything prepped': 0/14 fired);
      * the potato batch: both potatoes in processing (the batch goes on the heat together);
      * the burrito steak: rice in the pot and the steak prepped, and tortilla + rice plated — the rice is off the heat
        (plated), so it is not smoking beside the steak and the smoke ban does not hold the walk-away."""
    tier = tier_of(cell)
    if tier == "master" and c == "steak":
        return {"rice": 1, "steak": 1}, 2
    if tier == "master" and c == "rice":
        # the rice is the first thing lit: no floor (a floor of the steak prepped fired 1/3 in the round-2 pilot: the
        # steak is chopped after the rice is done). D6 is met by capping it (ROW_CAP_SPECIAL): the burrito steak
        # walk-away is starved by the rice smoke (0/8)
        return {}, 0
    if c == "potato":
        # the LLM cooks fry the potatoes one at a time (round-2 pilot: a floor of both prepped fired 1/6)
        return {"potato": 1}, 0
    return {}, 0


def slots(row, cell):
    """C1: the plate slots / components a mistake acts on."""
    items = recipe_items(cell)
    fn, kw = row["fn"], row["kwargs"]
    if fn in ("mis_prep", "serve_raw", "misidentify_assemble", "leave_unattended"):
        return {kw["component"]}
    if fn == "add_foreign":
        return {"serve"} if kw.get("before") == "serve" else {comp_of_sid(kw["before"])}
    if fn == "wrong_order":
        return {kw["a"], kw["b"]}
    if fn == "omit_subgoal":
        x = comp_of_sid(kw["sid"])
        i = items.index(x)
        return {x} | ({items[i + 1]} if i + 1 < len(items) else {"serve"})
    if fn == "premature_serve":
        return {"serve"}                      # it walks the plate to the pass; it acts on no component
    if fn == "repeat_step":
        return {comp_of_sid(kw["sid"])}
    return set()                              # forget_next_step: time only


def wrong_item(row):
    return row["kwargs"].get("wrong") if row["fn"] in ("add_foreign", "misidentify_assemble", "mis_prep") else None


def is_ruinous(row):
    return row["fn"] in RUINOUS_FN


def opens_t1(row, cell):
    return moment(row, cell)[0] == 0.0


# ----------------------------------------------------------------------------------------------- single-row rules
def row_ok(row, cell):
    """None if an inventory row may be used at all in this cell, else the rule it breaks."""
    from cooksim.core.enums import choppable, STATION_ACCEPTS, Terrain
    items = recipe_items(cell)
    want = want_of(cell)
    fn, kw = row["fn"], row["kwargs"]
    if row.get("collapses"):
        return "anchor collapses to t0"
    if fn in ("add_foreign", "misidentify_assemble") and kw.get("wrong") in items:
        return "R2 the foreign item is a recipe item"
    if fn == "mis_prep":
        w, c = kw.get("wrong"), kw.get("component")
        if w in items:
            return "C9/R8 the wrong item is a recipe item"
        if not choppable(w):
            return "C9 the wrong item cannot be chopped"
        st = {"fish": Terrain.FRYER, "meat": Terrain.PAN, "rice": Terrain.POT, "steak": Terrain.PAN,
              "potato": Terrain.FRYER}.get(c)
        if st is not None and ((w, "raw") in STATION_ACCEPTS.get(st, set()) or (w, "chopped") in STATION_ACCEPTS.get(st, set())):
            # the cook believes the wrong item IS the component (the takeover's belief task) and carries it on to the
            # component's station: a chopped onion went into the rice pot, was lit and burned (LLM pilot, master_medium_2)
            return "C9 the component's station takes the wrong item (raw or chopped): it would be cooked and burn untracked"
    if fn == "wrong_order":
        a, b = kw["a"], kw["b"]
        if want[b] == "cooked":
            return "WO the item plated first is cooked food waiting on the heat"
        if b == items[-1] and items.count(b) == 1 and want[b] != "raw":
            return "WO the swapped-in item is the recipe's last item"
        if want[b] != "raw" and items.index(b) != items.index(a) + 1:
            return "WO a chopped swapped-in item must be the one right after a"
    if fn == "serve_raw":
        c = kw["component"]
        if want[c] == "cooked" and items.index(c) == items.index(next(x for x in items if want[x] == "cooked")) \
                and c != "potato":
            return "SR the first cooked item is cooked before the plate is taken"
        if want[c] == "cooked" and tier_of(cell) == "master":
            return "SR burrito rice/steak are cooked long before plating"
        if want[c] == "chopped" and items[0] == c:
            return "SR a first chopped item is chopped before the plate is taken"
    if fn == "omit_subgoal":
        if tier_of(cell) in ("hard", "expert"):
            return "O the potato is never ready while the fish/steak is still next"
        x = comp_of_sid(kw["sid"])
        nxt = items[items.index(x) + 1] if items.index(x) + 1 < len(items) else None
        if nxt and want[nxt] == "cooked":
            return "O the omission plates the next item, which is cooked: the cook plates this one before it is ready"
        if x == items[-2] and want[items[-1]] != "raw":
            return "R11 the omission plates the last item next, which still needs prep"
    # ---- round 2 (FIX_PLAN_R2 D1-D7)
    if fn == "serve_raw" and kw.get("component") == "potato":
        return "D1 serve_raw(potato) fires only when the cook plates before the fryer is done"
    if fn == "repeat_step" and comp_of_sid(kw.get("sid")) == "bun":
        return "D4 a repeated bun reads as the top bun (the cook then serves without the cheese)"
    if fn == "add_foreign" and kw.get("before", "").startswith("assemble(") and comp_of_sid(kw["before"]) == items[0] \
            and want[items[0]] == "raw":
        return "D5 a foreign item before a raw first item makes the cook drop that item"
    w_ = wrong_item(row)
    if w_ and fn in ("add_foreign", "misidentify_assemble") and (w_ in BAD_FOREIGN or ICON_TWINS.get(w_, set()) & set(items)
                                                                 or any(w_ in ICON_TWINS.get(i, set()) for i in items)):
        return f"D7 the foreign {w_} is not a clean off-recipe food here (icon twin of a recipe item, or not a food)"
    if fn == "mis_prep" and want.get(kw.get("component")) != "chopped" and kw.get("component") not in ("steak", "potato"):
        # (steak and potato are chopped, then cooked)
        return ("D7 the mis_prep block CHOPS the wrong item: for an item the recipe never chops (fish, rice, meat) that "
                "is a different process, not a confusion at that step (master F6.2)")
    if fn == "mis_prep" and kw.get("wrong") in heat_takes(cell):
        return "D7 a heat station of this kitchen takes the mis-prepped item: the cook cooks it and it burns untracked"
    if len(items) >= 5 and fn == "repeat_step":
        return "CAP a repeated item fills the 5-slot plate: the last item can never go on (an unlabelled omission)"
    if len(items) >= 5 and fn in ("add_foreign", "misidentify_assemble") and \
            (kw.get("component") or comp_of_sid(kw.get("before"))) != items[-1]:
        return "CAP on the full 5-slot plate an extra item before the last one leaves the last one unplated (unlabelled)"
    if fn == "forget_next_step" and tier_of(cell) == "master" and str(kw.get("after", "")).startswith("prep("):
        return ("F the burrito rice is lit first and a forget waits until nothing is on the heat (engine E4): a "
                "prep-stage forget fires only after the steak is plated — it is forget-after-steak")
    if fn == "leave_unattended" and tier_of(cell) == "master" and kw.get("component") == "steak" and not ROUND3:
        return ("LU the burrito steak finishes while the rice smokes (LLM pilot: rice smoke from t67-68, steak done "
                "t73-109, rice plated t80-120: 0/8 fired, the smoke ban holds every walk-away)")
    # ---- round 3 (FIX_PLAN_R3 D11, D17)
    if ROUND3 and tier_of(cell) == "master" and ((fn == "leave_unattended" and kw.get("component") == "rice")
                                                 or (fn == "mis_prep" and kw.get("component") == "steak")):
        # D17: the round-3 pilot (engine F1 + R10, the cook turns at every release) still burned 6/7 baseline burritos
        # whose early takeover sat in the rice window (rice done t42, smoke t67, fire t126): the Flash baseline cook
        # finishes its steak prep on the release turn. Nothing takes the body before the rice can be plated.
        return ("D17 an early takeover on the burrito's rice window (the walk-away on the rice, the steak mis_prep): "
                "the baseline cook still burned the rice after the release (round-3 pilot 6/7)")
    if ROUND3 and fn in ("add_foreign", "misidentify_assemble") and tier_of(cell) in ("hard", "expert") and \
            (kw.get("component") or comp_of_sid(kw.get("before"))) == items[0]:
        return ("D11 a foreign item at the fish/steak waits for a plate while that food smokes (20/20 fish smoked before "
                "the plate): the smoke ban holds it until the fish/steak is plated (missed 5 of 8 runs)")
    if fn == "mis_prep" and n_boards(cell) < 2:
        return ("C9 a one-board kitchen: the mis-prepped item blocks the only board (LLM pilot, hard_medium_3 baseline: "
                "the cook asked how to clear it until the episode timed out)")
    if tier_of(cell) == "master" and fn in ("mis_prep", "forget_next_step") and \
            (kw.get("component") in ("tomato", "lettuce") or comp_of_sid(kw.get("after")) in ("tomato", "lettuce")) and \
            not (kw.get("after") or "").startswith("assemble(") and not (fn == "mis_prep" and kw.get("component") == "tomato"):
        # (round 2: mis_prep(tomato) is allowed — it fires whenever the cook reaches for the tomato, which the LLM cooks
        # do around the start of plating (moment table chop(tomato)); the set rules keep other mistakes away from it)
        return "LC LLM cooks chop the burrito's tomato and lettuce late, often mid-plating: the moment is not placeable"
    if fn == "premature_serve" and kw.get("after") == "fetch_plate":
        return "C2 a serve right after the plate is the same moment as the first item"
    if fn == "repeat_step":
        at = comp_of_sid(kw.get("at"))
        if at == items[-1] or (at and items.index(at) >= len(items) - 1):
            return "R5b a repeat at the last item cannot land"
    if fn == "forget_next_step":
        a = kw.get("after", "")
        x = comp_of_sid(a)
        if a.startswith("ignite(") or (a.startswith("prep(") and want.get(x) == "cooked" and x not in ("steak", "potato")):
            return "F a stall that begins as the food goes on the heat is a walk-away (it smokes during the idle)"
        if a.startswith("assemble(") and comp_of_sid(a) in items[-2:] and not (
                comp_of_sid(a) == items[-2] and want[items[-1]] == "chopped" and len(items) >= 5):
            # (round 2: not on the burrito, whose last item, the lettuce, is still to be chopped at that point — the
            # forgotten step is the chop, never the serve; the engine refuses a forget on a name-complete plate)
            return "R9 forgetting after the second-to-last item collapses into 'plate complete'"
    # (round 1 banned the potato and burrito-steak walk-aways: the earlier item smoked beside them. Round 2 gives them an
    # explicit floor (lu_floor) instead; the potato ones stay LOW_YIELD.)
    if fn == "leave_unattended" and kw.get("loops", 1) < 2 and tier_of(cell) != "master":
        return "LU one loop is shorter than the smoke onset on this recipe"
    if fn == "forget_next_step" and tier_of(cell) == "medium":
        return "F the burger's meat is lit first: every forget waits for it and only plate-stage anchors are left"
    return None


_HEAT = {}


def n_boards(cell):
    from cooksim_bench import exp_interact as X
    _st, ro = X.build_episode(None, cell=cell, inject=not cell.endswith("hard_map"))
    from cooksim.core.enums import Terrain
    return sum(1 for row in ro.game.layout.grid for t in row if t == Terrain.CUTTING_BOARD)


def heat_takes(cell):
    """D7: every item some heat station of this cell's kitchen accepts (raw or chopped)."""
    if cell not in _HEAT:
        from cooksim.core.enums import STATION_ACCEPTS
        from cooksim_bench import exp_interact as X
        _st, ro = X.build_episode(None, cell=cell, inject=not cell.endswith("hard_map"))
        kinds = {t for row in ro.game.layout.grid for t in row}
        _HEAT[cell] = {n for st, acc in STATION_ACCEPTS.items() if st in kinds for n, _s in acc}
    return _HEAT[cell]


# ----------------------------------------------------------------------------------------------- set rules
def set_violation(rows, cell, relaxed_labels=False, relaxed_wrong=None):
    """None if this set of error rows is admissible together, else the first rule it breaks."""
    items = recipe_items(cell)
    # L / W (round 2, D3): one error LABEL and one wrong item per case, never relaxed. Flag credit is by name only, so
    # two add_foreign-labelled mistakes (a misidentify and a serve-time add_foreign) cannot be told apart, and a second
    # mistake with the same foreign item committed as a phantom (expert medium_3, hard medium_5). The relaxed_* flags
    # are kept for callers of round 1 and are ignored.
    labels = [r["err"] for r in rows]
    if len(set(labels)) < len(labels):
        return "L one label per case"
    ws = [wrong_item(r) for r in rows if wrong_item(r)]
    if len(set(ws)) < len(ws):
        return "W a wrong item used twice"
    # C1 (round 2): a mis_prep acts in the prep phase (the wrong item is chopped and left on a board; the cook recovers
    # the right one), so it may share its component with a later PLATING mistake; never with another prep-phase one
    seen = set()
    for r in rows:
        if r["fn"] == "mis_prep":
            continue
        s = slots(r, cell)
        if s & seen:
            return f"C1 slot {sorted(s & seen)} targeted twice"
        seen |= s
    mp_ = [r for r in rows if r["fn"] == "mis_prep"]
    if len(mp_) > 1:
        return "C1 two mis_preps"
    for r in mp_:
        c_ = r["kwargs"]["component"]
        if any(x is not r and x["fn"] == "serve_raw" and x["kwargs"].get("component") == c_ for x in rows):
            return f"C1 slot ['{c_}'] mis_prep and a raw serve of it"
    # M with spans (round 2, WO): a swap occupies next(a)..plated(b): nothing else inside that span or within MIN_GAP
    # before it; the next mistake may come at its end (the engine's takeover gap G gives the cook its turn)
    sp = sorted((span(r, cell), i) for i, r in enumerate(rows))
    for ((a1, e1), i1), ((a2, e2), i2) in zip(sp, sp[1:]):
        # a STATE trigger (a forget, a premature serve, the serve-time add_foreign, a raw serve: the moment holds until
        # the cook changes it) is only delayed by the engine's takeover gap, never lost; a BRIEF or window trigger
        # (heading to plate, reaching for a crate, a walk-away's near-done window) needs the full gap (round 2)
        need = STATE_GAP if (is_state(rows[i2]) and rows[i1]["fn"] != "wrong_order") else MIN_GAP
        if ROUND3 and is_plating(rows[i1]):
            # D9 (round 3): after a PLATING takeover the LLM cook plates the next item within the takeover gap G
            # (easy_m3 forget, medium_m1 premature, medium_m4 B add_foreign lost inside G): a state moment needs a full
            # plating step, a brief one a plating step plus G — counted from the END of the takeover's span (round-3
            # pilot, medium_m1: the bun/cheese swap ran t20-47 over the whole meat cook; the meat was plated at once)
            need = max(need, PLATE_STEP_GAP if is_brief(rows[i2], cell) else MIN_GAP)
            _k2 = rows[i2]["kwargs"]
            _prep_after = (rows[i2]["fn"] == "forget_next_step" and str(_k2.get("after", "")).startswith("prep(")
                           and moment_table(cell).get(f"chop({comp_of_sid(_k2['after'])})",
                                                      moment_table(cell).get(f"load({comp_of_sid(_k2['after'])})", -1)) >= e1 - 1e-6)
            if a2 - e1 < need - 1e-6 and not _prep_after:     # (a forget after a prep the cook still has to do)
                return f"D9 {rows[i2]['fn']}@{a2} comes within a plating step (+G) of the {rows[i1]['fn']} takeover's end @{e1}"
        if a2 - a1 < need - 1e-6 or (e1 > a1 and a2 < e1 - 1e-6):
            return f"M {rows[i1]['fn']}@{a1}-{e1} and {rows[i2]['fn']}@{a2} share a moment"
        # G (round 2): the cook resumes a forget's freeze exactly where it stopped, so the next moment comes within the
        # engine's takeover gap (G=10 ticks) — a BRIEF moment (the cook heading to plate an item, reaching for a crate)
        # is passed before the gap ends (verifier r1, easy_hard_3: misidentify(tomato) after forget(prep(tomato)) 0/3)
        if tier_of(cell) == "master" and rows[i1]["fn"] == "mis_prep" and not is_state(rows[i2]) and a2 - a1 < 2.2:
            return f"M the mis_prep block (~20 ticks) covers the {rows[i2]['fn']} moment"
        T_ = moment_table(cell)
        work_between = any(a1 + 1e-6 < v < a2 - 1e-6 and k.startswith(("chop(", "load(")) for k, v in T_.items())
        if rows[i1]["fn"] == "forget_next_step" and is_brief(rows[i2], cell) and a2 - a1 < FORGET_NEXT_GAP - 1e-6 \
                and not work_between:       # (a chop or a load between them takes the cook past the gap)
            return f"G a brief {rows[i2]['fn']} moment right after the forget's freeze is lost in the takeover gap"
    # CAP (round 2): the plate holds 5; the served dish must be able to hold every recipe item plus every extra item
    extra_n = sum(1 for r in rows if r["fn"] in ("add_foreign", "misidentify_assemble", "repeat_step"))
    if len(items) < 5 and len(items) + extra_n > 5:
        return "CAP the recipe and the extra items exceed the plate's 5 slots"
    prem = [r for r in rows if r["fn"] == "premature_serve"]
    if len(prem) > 1:
        return "C2 two premature serves"
    if prem:
        pp = moment(prem[0], cell)[0]
        if any(moment(r, cell)[0] >= pp for r in rows if r is not prem[0]):
            return "C2 a premature serve must be the last moment"
    T = moment_table(cell)
    for f in rows:
        if f["fn"] != "forget_next_step":
            continue
        pf = moment(f, cell)[0]
        for r in rows:
            if r is f or r["fn"] == "mis_prep":
                continue
            # food already on the heat when the 60-tick idle starts is done and smoking by its end: a walk-away or a
            # plating mistake on it is starved (the smoke ban / the walk-away's near-done window passes)
            x = r["kwargs"].get("component") or r["kwargs"].get("a") or comp_of_sid(r["kwargs"].get("sid") or r["kwargs"].get("before"))
            if x and f"done({x})" in T and T.get(f"load({x})", 0) < pf and pf < T[f"done({x})"] <= pf + FORGET_SPAN + 0.5 \
                    and not ENGINE_FORGET_GUARD:
                return f"M forget@{pf} lets the {x} smoke before {r['fn']}"
    # a stall that lets food already on the heat sit done for long before its plating is an untracked walk-away
    for f in rows:
        if f["fn"] != "forget_next_step":
            continue
        pf = moment(f, cell)[0]
        for x in items:
            if want_of(cell)[x] == "cooked" and T.get(f"load({x})", 99) < pf < T[f"done({x})"] + 1e-6 \
                    and T[f"next({x})"] - T[f"done({x})"] > 2 and not ENGINE_FORGET_GUARD:
                return f"F the stall leaves the {x} done on the heat long before it is plated"
    # SR+: a raw unit scooped from the crate stands in for one unit's processing: the cook then processes one unit fewer,
    # so "after the second potato is chopped" may never come (LLM check r19, expert_medium_5: 2/2 "potato #2 not in
    # processing yet")
    for r in rows:
        if r["fn"] == "serve_raw":
            x = r["kwargs"]["component"]
            for f in rows:
                a = (f["kwargs"].get("after") or "") if f["fn"] == "forget_next_step" else ""
                if a.startswith(f"prep({x})#"):
                    return f"SR+ the forget waits for a second {x} that the raw scoop replaces"
    # CAP: the 5-item burrito plate holds exactly the dish; one extra item (add_foreign before an item, a repeat)
    # leaves no room for the lettuce, so nothing may need to land after it, and there is room for one extra only
    if len(items) >= 5:
        extra = [r for r in rows if (r["fn"] == "add_foreign" and r["kwargs"].get("before") != "serve") or r["fn"] == "repeat_step"]
        if len(extra) > 1:
            return "CAP two extra items on the 5-item plate"
        if extra:
            pe = moment(extra[0], cell)[0]
            for r in rows:
                if r is extra[0]:
                    continue
                pr = moment(r, cell)[0]
                late = r["fn"] in ("misidentify_assemble", "serve_raw", "omit_subgoal", "add_foreign", "wrong_order", "premature_serve")
                if late and pr > pe and pr >= T[f"next({items[-1]})"]:
                    return "CAP the extra item fills the plate before this plating mistake"
                # round 2 (master F1.3): a forget after the extra waits on a full plate (engine E4) and never fires
                if r["fn"] == "forget_next_step" and pr > pe:
                    return "CAP a forget after the extra item waits on a full plate"
    lu = [r for r in rows if r["fn"] == "leave_unattended"]
    for L in lu:
        c = L["kwargs"]["component"]
        for r in rows:
            if r is not L and r["fn"] not in ("forget_next_step", "mis_prep") + (("premature_serve",) if ROUND3 else ()) and c in {r["kwargs"].get(k) for k in ("component", "a", "b", "wrong")} | {comp_of_sid(r["kwargs"].get(k)) for k in ("sid", "before", "at")}:
                return f"R1 the walk-away's {c} is also another mistake's item"
    # LU: the walk-away needs its near-done window free (no takeover within LU_GAP moments) and, once its food smokes,
    # the rescue window up to that food's own plating (a takeover there keeps the cook from the smoking pan: the
    # scripted cooks burned the rice that way)
    for L in lu:
        c = L["kwargs"]["component"]
        pl = moment(L, cell)[0]
        for r in rows:
            if r is L or r["fn"] == "mis_prep" and moment(r, cell)[0] == 0.0:
                continue
            pr = moment(r, cell)[0]
            if abs(pr - pl) < LU_GAP - 1e-6:
                return f"LU a {r['fn']} takeover sits in the walk-away's near-done window"
            # only a swap leaves a plate the cook keeps building; any other mistake there (an extra, raw or foreign item,
            # a stall, a detour) sends the cook into a repair — trash and rebuild — while the pan burns (scripted cooks
            # burned the burrito steak exactly so, verify r11)
            if f"next({c})" in T and pl < pr <= T[f"next({c})"] + 1e-6 and r["fn"] != "wrong_order":
                return f"LU a {r['fn']} takeover sits in the {c}'s rescue window (smoking until it is plated)"
    # C2+: a premature serve after X ends the episode as soon as X is plated, which a plate-first cook reaches before
    # anything about a later item: every other mistake must be about X or the items before it
    for P in prem:
        _x = comp_of_sid(P["kwargs"]["after"])
        ix = items.index(_x) if _x in items else -1
        for r in rows:
            if r is P:
                continue
            if r["fn"] == "mis_prep" and moment(r, cell)[0] < moment(P, cell)[0] - MIN_GAP:
                continue                    # round 2: a prep-phase mistake long before the serve is reached anyway
            its = {r["kwargs"].get(k) for k in ("component", "a", "b")} | {comp_of_sid(r["kwargs"].get(k)) for k in ("sid", "before", "at", "after")}
            its = {i for i in its if i in items}
            if r["fn"] == "add_foreign" and r["kwargs"].get("before") == "serve":
                return "C2 an add_foreign at the serve after a premature serve"
            if any(items.index(i) > ix for i in its):
                return f"C2 {r['fn']} is about an item after the premature serve's {items[ix]}"
            if r["fn"] == "wrong_order":
                return "C2 a swapped plate is complete (or out of order) when the premature serve's moment comes"
    # (a swap leaves every item correctly on the plate once its block has run — B1 — so "through Y" still holds for
    # an LLM cook; the scripted verifier's cook trashes a swapped plate, which is its own repair policy) — except when
    # the swapped-in b is the recipe's last item and Y the one before it: "through Y" is then a COMPLETE plate, where a
    # repeat, a premature serve or a forget can no longer land (LLM check r18, medium_medium_5: 2/2 "plate complete")
    for W in [r for r in rows if r["fn"] == "wrong_order"]:
        ib = items.index(W["kwargs"]["b"])
        for r in rows:
            k = r["kwargs"]
            y = comp_of_sid(k.get("at")) if r["fn"] == "repeat_step" else (
                comp_of_sid(k.get("after")) if r["fn"] in ("premature_serve", "forget_next_step") and (k.get("after") or "").startswith("assemble(") else None)
            if y in items and ib == len(items) - 1 and items.index(y) == len(items) - 2:
                return f"WO {r['fn']} waits for the plate through {y}, which the swap has already completed"
    # B: a mistake that leaves item X missing from the plate (X replaced, raw, or skipped) keeps X "next to plate"
    # until the cook re-plates it (next_to_plate counts correct units): every later plating moment waits for that, so
    # it must be the case's last plating moment (the serve-time add_foreign fires on the walk to the pass anyway)
    for b in rows:
        if b["fn"] not in ("misidentify_assemble", "serve_raw", "omit_subgoal"):
            continue
        x = b["kwargs"].get("component") or comp_of_sid(b["kwargs"].get("sid"))
        ix = items.index(x)
        for r in rows:
            if r is b:
                continue
            fn, kw = r["fn"], r["kwargs"]
            later = None
            if fn in ("misidentify_assemble", "serve_raw", "omit_subgoal"):
                later = items.index(kw.get("component") or comp_of_sid(kw.get("sid"))) > ix
            elif fn == "add_foreign" and kw.get("before") != "serve":
                later = items.index(comp_of_sid(kw["before"])) > ix
            elif fn == "wrong_order":
                later = items.index(kw["a"]) > ix
            elif fn == "premature_serve":
                later = items.index(comp_of_sid(kw["after"])) >= ix
            elif fn == "repeat_step":
                later = items.index(comp_of_sid(kw["at"])) >= ix
            elif fn == "forget_next_step" and kw.get("after", "").startswith("assemble("):
                later = items.index(comp_of_sid(kw["after"])) >= ix
            if later:
                return f"B {fn} waits for the {x} that {b['fn']} leaves off the plate"
    if ROUND3:
        # follow-up (round-3 final run, easy_medium_1 baseline: premature after the tomato lost): after a forget that
        # froze the cook WITH the plate in hand, the cook plates what is ready within the takeover gap — a following
        # "after X" state moment needs a full plating step plus G
        for f_ in rows:
            if f_["fn"] == "forget_next_step" and (f_["kwargs"].get("after") == "fetch_plate"
                                                    or str(f_["kwargs"].get("after", "")).startswith("assemble(")):
                pf_ = moment(f_, cell)[0]
                for r_ in rows:
                    if r_ is not f_ and after_item(r_) is not None and 0 <= moment(r_, cell)[0] - pf_ < PLATE_STEP_GAP - 1e-6:
                        return f"D9 {r_['fn']} after the {after_item(r_)} comes within a plating step (+G) of a plate-in-hand forget"
        why = round3_violation(rows, cell)
        if why:
            return why
    n_r = sum(1 for r in rows if is_ruinous(r))
    if n_r < MIN_RUINOUS[tier_of(cell)]:
        return f"T fewer than {MIN_RUINOUS[tier_of(cell)]} ruinous"
    return None


def is_plating(row):
    """A mistake whose takeover block scoops onto the plate (it advances the plate itself)."""
    fn, kw = row["fn"], row["kwargs"]
    if fn == "add_foreign":
        return kw.get("before") != "serve"
    return fn in ("misidentify_assemble", "serve_raw", "omit_subgoal", "wrong_order", "repeat_step")


def before_item(row):
    """The item a mistake takes over the plating OF (it fires when that item is next to plate)."""
    fn, kw = row["fn"], row["kwargs"]
    if fn == "add_foreign" and kw.get("before") != "serve":
        return comp_of_sid(kw["before"])
    if fn in ("misidentify_assemble", "serve_raw"):
        return kw.get("component")
    if fn == "wrong_order":
        return kw.get("a")
    if fn == "omit_subgoal":
        return comp_of_sid(kw.get("sid"))
    return None


def after_item(row):
    """The item a mistake waits to see PLATED (it fires once the plate is through that item)."""
    fn, kw = row["fn"], row["kwargs"]
    if fn in ("premature_serve", "forget_next_step") and str(kw.get("after", "")).startswith("assemble("):
        return comp_of_sid(kw["after"])
    if fn == "repeat_step":
        return comp_of_sid(kw.get("at"))
    return None


def is_robust_ruinous(row, cell):
    """D14 (round 3): a ruinous mistake whose moment every cook reaches — the serve-time foreign item, a premature serve
    after a plated item, and on the 5-item burrito the foreign item at the last item (the lettuce). Not a foreign item
    before an early item (the smoke ban holds it: hard medium_4 C, hard_4 B won with no ruinous error), not serve_raw of
    an item a careful cook chops first (master hard_3 C won), not the steak/tomato swap (missed 4/8)."""
    fn, kw = row["fn"], row["kwargs"]
    items = recipe_items(cell)
    if fn == "add_foreign" and kw.get("before") == "serve":
        return True
    if fn == "premature_serve" and kw.get("after") != "fetch_plate":
        return True
    if fn in ("add_foreign", "misidentify_assemble") and len(items) >= 5 and before_item(row) == items[-1]:
        return True
    return False


def round3_violation(rows, cell):
    """The round-3 set rules (FIX_PLAN_R3 D9, D13, D14)."""
    items = recipe_items(cell)
    tier = tier_of(cell)
    T = moment_table(cell)
    # D9 / D13: "before X" and "after X" (medium_m1: add_foreign before the meat, premature after it — the cook plates
    # the meat inside G); "after X" and "before next(X)" share one anchor (master: forget after the tomato and the
    # foreign item at the lettuce, 6 cases; the foreign item won and the forget met a full plate)
    for r1 in rows:
        a = after_item(r1)
        if a is None or a not in items:
            continue
        nxt = items[items.index(a) + 1] if items.index(a) + 1 < len(items) else None
        for r2 in rows:
            if r2 is r1:
                continue
            b = before_item(r2) if r2["fn"] != "wrong_order" else None   # a swap plates its items itself
            if b is not None and b == a:
                return f"D9 {r2['fn']} before the {a} and {r1['fn']} after it (the cook plates the {a} inside G)"
            if b is not None and b == nxt:
                return f"D13 {r1['fn']} after the {a} and {r2['fn']} at the {nxt} share one anchor"
    # D9: a block that completes the plate itself leaves no moment for a serve-time mistake (medium_m4 B: wrong_order
    # (meat, cheese) plated the last item; the cook served 9 ticks after the release)
    for r in rows:
        if r["fn"] == "wrong_order" and items[-1] in (r["kwargs"]["a"], r["kwargs"]["b"]) and items.count(items[-1]) == 1:
            if any(x["fn"] == "add_foreign" and x["kwargs"].get("before") == "serve" for x in rows):
                return "D9 a swap that plates the last item completes the plate: no serve-time mistake after it"
    # D9: forget(after fetch_plate) after a block that fetches the plate itself (easy_m3: the omission fetched the
    # plate, so the forget's moment was inside the takeover)
    for f in rows:
        if f["fn"] == "forget_next_step" and f["kwargs"].get("after") == "fetch_plate":
            pf = moment(f, cell)[0]
            if any(is_plating(r) and moment(r, cell)[0] <= pf + 1e-6 for r in rows if r is not f):
                return "D9 a plating block before the plate is fetched: the forget-after-plate moment is inside it"
    # D14: hard / expert / master keep >= 1 ROBUST ruinous mistake (easy / medium keep >= half ruinous)
    if tier in ("hard", "expert", "master") and not any(is_robust_ruinous(r, cell) for r in rows):
        return "D14 no robust ruinous mistake (the serve-time foreign item, a premature serve, the burrito's last item)"
    return None


# ----------------------------------------------------------------------------------------------- inventory
def load_inventory(path):
    inv = json.load(open(path))["cases"]
    return [x for x in inv if not x.get("collapses")]


def key_of(row):
    return (row["fn"], json.dumps(row["kwargs"], sort_keys=True))


def sg_norm(sig):
    """A case's error set with the details that do not change the design removed (the foreign item, the walk-away's
    loop count, which potato a forget follows), and (round 2, D2) an add_foreign before X written as a misidentify of X
    — the same committed world (master medium_3/medium_4): two cases equal under it are near-duplicates (C4)."""
    out = []
    for fn, kw in sig:
        kw = json.loads(kw)
        kw.pop("wrong", None); kw.pop("loops", None)
        if "after" in kw:
            kw["after"] = kw["after"].split("#")[0]
        if fn == "add_foreign" and str(kw.get("before", "")).startswith("assemble("):
            fn, kw = "misidentify_assemble", {"component": comp_of_sid(kw["before"])}
        out.append((fn, json.dumps(kw, sort_keys=True)))
    return tuple(sorted(out))


def row_norm(row):
    return sg_norm(((row["fn"], json.dumps(row["kwargs"], sort_keys=True)),))[0]


def low_yield(r):
    kw = r["kwargs"]
    c = kw.get("component") or kw.get("a") or comp_of_sid(kw.get("after"))
    if r["fn"] == "forget_next_step" and tier_of(r["cell"]) == "master" and str(kw.get("after", "")).startswith("prep("):
        c = f"{c}@master"             # the rice is lit first: a prep-stage forget waits until nothing is on the heat
    return LOW_YIELD.get((r["fn"], c), 0)


def fn_keys(rows):
    """D16: the mistake kinds a set uses, for the per-tier case caps (a forget is keyed by the item it follows)."""
    out = set()
    for r in rows:
        out.add(r["fn"])
        if r["fn"] == "forget_next_step":
            out.add(f"forget_next_step@{comp_of_sid(r['kwargs'].get('after')) or r['kwargs'].get('after')}")
    return out


def opening_of(rows, cell):
    """D16: the set's first mistake by moment (normalised: foreign item and loop count dropped)."""
    if not rows:
        return None
    r = min(rows, key=lambda x: moment(x, cell)[0])
    fn, kw = row_norm(r)
    return (fn, kw)


def pick_errors(case, pool, v4_keys, tier_state, cell_sigs, rng):
    """The admissible error set for one case that best keeps the tier diverse and its previous errors: EXHAUSTIVE
    search. Foreign-item rows are grouped by (function, slot) — the wrong item is assigned afterwards (distinct within
    the case, least used in the tier, the previous one when it is still legal) — so every combination of groups is tried.
    Hard constraints (round 2): the set rules, no tier near-duplicate (D2), no mistake in more than ROW_CAP of the
    tier's cases (D6). Score (lexicographic): t1-opener cap (C4), label cap, reuse of the same error pattern in the
    tier, low-yield rows, then kept previous rows, label/function use, wrong-item reuse, ruinous count."""
    cell = case["cell"]
    tier = tier_of(cell)
    n = TIER_N[tier]
    usable = [r for r in pool if row_ok(r, cell) is None]
    groups = collections.defaultdict(list)
    for r in usable:
        kw = {k: v for k, v in r["kwargs"].items() if k != "wrong"} if wrong_item(r) else r["kwargs"]
        groups[(r["fn"], json.dumps(kw, sort_keys=True))].append(r)
    keys = sorted(groups)
    v4_wrong = {}
    for k in v4_keys:
        kw = json.loads(k[1])
        if kw.get("wrong"):
            v4_wrong[(k[0], json.dumps({a: b for a, b in kw.items() if a != "wrong"}, sort_keys=True))] = kw["wrong"]

    def assign(combo):
        rows, used = [], set()
        for g in combo:
            opts = groups[g]
            if not wrong_item(opts[0]):
                rows.append(opts[0]); continue
            pref = sorted(opts, key=lambda r: (0 if r["kwargs"]["wrong"] == v4_wrong.get(g) else 1,
                                               tier_state["wrongs"][r["kwargs"]["wrong"]], r["kwargs"]["wrong"]))
            pick = next((r for r in pref if r["kwargs"]["wrong"] not in used), pref[0])
            used.add(pick["kwargs"]["wrong"])
            rows.append(pick)
        return rows
    for strict in (True, False):
        best, best_score = None, None
        for combo in itertools.combinations(keys, n):
            if len({json.loads(g[1]).get("component") for g in combo if g[0] == "leave_unattended"}) < \
                    sum(1 for g in combo if g[0] == "leave_unattended"):
                continue
            rows = assign(combo)
            if set_violation(rows, cell):
                continue
            sig = tuple(sorted(key_of(r) for r in rows))
            if sig in cell_sigs:
                continue
            cell_near = sum(1 for sg in cell_sigs if sg_norm(sg) == sg_norm(sig))
            if strict and cell_near:
                continue                    # D2: never two cases of one kitchen with the same design (when possible)
            near = sum(1 for sg in tier_state["sigs"] if sg == sg_norm(sig))
            capped = sum(1 for r in rows if tier_state["rows"][row_norm(r)] + 1 >
                         ROW_CAP_SPECIAL.get((tier, r["fn"]), ROW_CAP))
            if strict and (near or capped or sum(low_yield(r) for r in rows) > 1):
                continue
            if any(tier_state["rows"][row_norm(r)] + 1 > HARD_CAP.get((tier, r["fn"]), 99) for r in rows):
                continue                    # D6: never relaxed
            over_fn, over_open = 0, 0
            if ROUND3:
                # D16 (round 3): at most FN_CASE_CAP cases of a tier contain one mistake kind (hard: 8/10 opened with the
                # same mis_prep; expert: forget-after-steak 9/10) and at most OPENING_CAP cases open with one mistake
                over_fn = sum(1 for k in fn_keys(rows) if tier_state["fn_cases"][k] + 1 > FN_CASE_CAP.get((tier, k), 99))
                over_open = int(tier_state["openings"][opening_of(rows, cell)] + 1 > OPENING_CAP)
                if strict and (over_fn or over_open):
                    continue
            kept = sum(1 for r in rows if key_of(r) in v4_keys)
            over = sum(max(0, tier_state["labels"][l] + 1 - LABEL_CASES_CAP) for l in {r["err"] for r in rows})
            t1_over = max(0, tier_state["t1"] + (1 if any(opens_t1(r, cell) for r in rows) else 0) - MAX_T1)
            wrong_reuse = sum(tier_state["wrongs"][wrong_item(r)] for r in rows if wrong_item(r))
            use = sum(tier_state["labels"][r["err"]] for r in rows) + 2 * sum(tier_state["fns"][r["fn"]] for r in rows) \
                + 2 * sum(tier_state["rows"][row_norm(r)] for r in rows)
            n_r = sum(1 for r in rows if is_ruinous(r))
            pattern = tuple(sorted(r["fn"] for r in rows))
            low = sum(low_yield(r) for r in rows)
            short = max(0, math.ceil(n / 2) - n_r)          # R-LU: below half ruinous (allowed only per tier)
            score = (-60 * cell_near - 20 * near - 20 * capped - 10 * t1_over - 4 * over - 3 * tier_state["patterns"][pattern] - 3 * low
                     - 2 * short - 15 * over_fn - 15 * over_open,
                     3 * kept - 0.5 * use - 0.5 * wrong_reuse + 0.6 * n_r, rng.random())
            if best_score is None or score > best_score:
                best, best_score = rows, score
        if best is not None:
            if not strict:
                case.setdefault("_notes", []).append("no set passes the tier's duplicate / row-cap rules: best relaxed set")
            return best
    return None


# ----------------------------------------------------------------------------------------------- voice sub-cases
_REPLAY_PREP = {}


def prep_req_at(cell, m):
    """The prep a sub-case anchored at m waits for (e2e_stepin_rollout.derive_spec): the reference run's prep_units."""
    key = (cell, m)
    if key not in _REPLAY_PREP:
        from cooksim_bench.e2e_stepin_rollout import prep_units
        state, ro, _ = D3.replay(cell, m)
        _REPLAY_PREP[key] = {k: v for k, v in prep_units(ro.game, state.recipe).items() if v}
    return _REPLAY_PREP[key]


def floor_at(cell, m):
    return min(len(D3.key_at(cell, m)["plate"]), len(recipe_items(cell)) - 1)


def latest_floor(cell):
    """The most items that may be on the plate when a voice sub-case fires: two short of complete (V2/V3), except on
    the two-item Caprese, where one item plated still leaves the cheese scoop and the walk to the pass."""
    return max(1, len(recipe_items(cell)) - 2)


def prep_floor(cell, m, item=None):
    """V2: the prep a sub-case anchored at m waits for, written into the sub-case (e2e_stepin_rollout.derive_spec reads
    `prep_floor`): the reference prep at m restricted to what an LLM cook preps early — never the late-chopped burrito
    lettuce — or the item itself for a question/request about one item."""
    req = prep_req_at(cell, m)
    if item and item in req:
        return {item: 1}
    return {k: v for k, v in req.items() if k in EARLY_PREP[tier_of(cell)]}


def robust_anchor(cell, m, item=None):
    """V2: m's plate floor leaves the plate at least two items short of complete (one on the two-item Caprese)."""
    return floor_at(cell, m) <= latest_floor(cell)


def ask_moment(cell, m, kind, item=None):
    """The first reference anchor >= m where the question's own precondition holds (e2e_stepin_rollout.
    question_precondition), or None. V3: that moment must leave the plate two items short of complete."""
    from cooksim_bench.e2e_stepin_rollout import question_precondition
    _, _, flat = D3.replay(cell, 0)
    items = recipe_items(cell)
    for m2 in range(m, len(flat) + 1):
        state, ro, _ = D3.replay(cell, m2)
        plate = D3.key_at(cell, m2)["plate"]
        if len(plate) > latest_floor(cell):
            return None
        from cooksim_bench.e2e_stepin_rollout import prep_units
        pm = prep_units(ro.game, state.recipe)
        why = question_precondition(kind, item, ro.game, state, plated_ever=set(plate), live_plate=tuple(plate), prep_max=pm)
        if why is None:
            return m2
    return None


def q_items(kind):
    base, _, arg = kind.partition(":")
    out = set()
    if base in ("after_ref", "ready_in", "elapsed", "ready"):
        out.add(arg)
    if base == "order":
        for s in arg.split("|"):
            c = comp_of_sid(s)
            if c:
                out.add(c)
    return out


def q_blocked(kind, rows, cell):
    """C7: would a mistake of this case make the question's own premise impossible? A prep side ("did I chop the X")
    dies with a mis_prep of X; a plating side or an after_ref about X dies when X's plating is replaced, skipped,
    swapped or repeated; ready_in about the walk-away's food is left to the walk-away."""
    base, _, arg = kind.partition(":")
    prepped = {r["kwargs"]["component"] for r in rows if r["fn"] == "mis_prep"}
    plated = set()
    for r in rows:
        if r["fn"] in ("misidentify_assemble", "serve_raw", "omit_subgoal", "wrong_order", "repeat_step"):
            plated |= {x for x in slots(r, cell) if x != "serve"}
    if base == "order":
        for side in arg.split("|"):
            x = comp_of_sid(side)
            if side.startswith(("prep(", "ignite(")) and x in prepped:
                return True
            if side.startswith("assemble(") and x in plated:
                return True
        return False
    if base == "after_ref":
        items = recipe_items(cell)
        nxt = items[items.index(arg) + 1] if arg in items and items.index(arg) + 1 < len(items) else None
        return arg in plated or nxt in plated or any(r["fn"] == "add_foreign" and comp_of_sid(r["kwargs"].get("before")) == nxt for r in rows)
    return False


def replaced_items(rows, cell):
    """C7: items whose plating/processing a case's own mistakes replace, swap, skip or repeat."""
    out = set()
    for r in rows:
        if r["fn"] in ("forget_next_step", "leave_unattended", "add_foreign"):
            continue
        out |= {x for x in slots(r, cell) if x != "serve"}
    return out


OPEN_KINDS = collections.defaultdict(collections.Counter)     # D16: (tier, opening mistake) -> question kinds used
TIER_KINDS = collections.defaultdict(collections.Counter)     # D16: tier -> question kinds used (spread within a tier)
QUIET_GAP = 0.6             # round 2 (E3): a question's moment at least this far from every mistake's span
QUIET_GAP_R3 = 1.0          # round 3 (D10): after a takeover, its release plus G at the LLM cook's pace
QUIET_BEFORE_R3 = 0.4       # round 3: before a takeover (VOICE_QUIET ticks, and the "imminent" hold)


def last_item_next(cell):
    """Round 3: the moment the recipe's LAST unit is next to plate (one item left: the serve-time foreign item is then
    imminent and holds the voice). A batch item (two potatoes) is half-way between its first and its last plating."""
    T = moment_table(cell)
    items = recipe_items(cell)
    x = items[-1]
    a = T.get(f"next({x})", T["serve"])
    if items.count(x) > 1:
        return (a + T.get(f"plated({x})", T["serve"])) / 2.0
    return a


def station_of_item(cell, item):
    try:
        from cooksim.core.generator import cook_station_for
        return str(cook_station_for(recipe_of(cell).id, item)).split(".")[-1].lower()
    except Exception:                                       # noqa: BLE001
        return None


def cooking_window_blocked(x, rows, cell):
    """D10 (round 3): a request / ready-in / elapsed question about cooked item x is dead when a takeover covers x's
    cooking window: any mistake whose moment falls between x going on the heat and x done, and any mis_prep of ANOTHER
    item after x is lit (it fires when the cook reaches for that item, which LLM cooks do right after lighting x: hard
    fish request vs mis_prep(potato), master rice vs mis_prep(steak), steak vs mis_prep(tomato), 7/14 + 6 runs dead)."""
    T = moment_table(cell)
    if want_of(cell).get(x) != "cooked" or f"load({x})" not in T:
        return None
    lo, hi = T[f"load({x})"], T.get(f"done({x})", T[f"load({x})"] + 2.0)
    for r in rows:
        if r["fn"] == "leave_unattended" and r["kwargs"].get("component") == x:
            continue                                        # (no request about the walk-away's food: design_voice)
        p = moment(r, cell)[0]
        if lo - 0.2 <= p <= lo + 0.5 * (hi - lo):
            return f"a {r['fn']} takeover right after the {x} goes on the heat"
        if r["fn"] == "mis_prep" and r["kwargs"].get("component") != x and lo - 0.5 <= p <= hi:
            return f"mis_prep({r['kwargs'].get('component')}) fires as the cook moves on from lighting the {x}"
    return None


def voice_r3_block(c, rows, cell):
    """D10 / D15 (round 3): why a question candidate is not used in this case (None = usable)."""
    kind = c["kind"]
    base, _, arg = kind.partition(":")
    items = recipe_items(cell)
    if base in ("longest_on", "burn_first"):
        return ("D15 'which has been on the heat longest / will burn first' has one candidate in practice (4/4 Steak "
                "Frites asks, master hard_2): it degenerates to 'what is on the heat'")
    if base in ("ready_in", "elapsed") and arg in items:
        why = cooking_window_blocked(arg, rows, cell)
        if why:
            return "D10 " + why
    if base == "assistant_said":
        # "what did you tell me about X earlier?" at t11-17 with 0 assistant lines is empty (hard medium_1 / hard_1):
        # asked with the plate started (the engine's plate floor) and after the first mistake has had a full plating
        # step for the assistant to talk about it
        first = min([moment(r, cell)[0] for r in rows] or [99])
        if len(D3.key_at(cell, c["m"])["plate"]) < 1:
            return "D15 nothing to recall yet (plate floor 0: it fires in the prep phase)"
        if c["pos"] < first + MIN_GAP:
            return "D15 nothing to recall yet (asked before the first mistake)"
    if kind.startswith("happened:visit("):
        w = arg[len("visit("):-1] if arg.startswith("visit(") else arg
        cooked = [x for x in items if want_of(cell)[x] == "cooked"]
        used = {station_of_item(cell, x) for x in cooked}
        first_st = station_of_item(cell, cooked[0]) if cooked else None
        if w == "trash":
            commit = [r for r in rows if r["fn"] not in ("forget_next_step", "leave_unattended")]
            if not any(moment(r, cell)[0] < c["pos"] for r in commit):
                return "D15 trash visit with no mistake to throw away before it: the answer is 'no' from the recipe"
        elif w not in used:
            return f"D15 the recipe never uses the {w}: the answer is 'no' from the recipe"
        elif w == first_st:
            return f"D15 the {w} takes the first item lit: the answer is 'yes' from the recipe"
    return None


def design_voice(case, rows, fam_count, q_truths, rng, want_proactive, visit_truths=None):
    cell = case["cell"]
    tier = tier_of(cell)
    items = recipe_items(cell)
    want = want_of(cell)
    rec = recipe_of(cell)
    ordered = bool(getattr(rec, "ordered", False))
    ref = D3.REF.get(cell) or 100
    inject = case.get("inject", True)
    T = moment_table(cell)
    err_pos = sorted(moment(r, cell)[0] for r in rows)
    prem = [moment(r, cell)[0] for r in rows if r["fn"] == "premature_serve"]
    prem_pos = prem[0] if prem else None
    lu_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "leave_unattended"}
    raw_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "serve_raw"}
    blocked = replaced_items(rows, cell)
    first_error_m = 0
    _, _, flat = D3.replay(cell, 0)
    anchors = D3.question_anchors(cell, [])
    # reference plate size -> moment position: a question that fires with k items plated sits at next(items[k])
    def pos_of_plate(k):
        if k == 0:
            return T.get("plate0", 2.0) - 0.4
        return T.get(f"next({items[k]})", T["serve"]) if k < len(items) else T["serve"]

    def live_pos(c):
        """Round 3 (D10): where the question really fires with an LLM cook — when its plate floor AND its prep floor
        are reached (the engine's progress floor), and its own precondition holds. The round-2 position assumed every
        plate-0 question fires just before the plate is taken; the Fish & Chips Q8 with floor {fish: 1} fired at t11."""
        m_ = c["m"]
        k_ = len(D3.key_at(cell, m_)["plate"])
        base_, _, arg_ = c["kind"].partition(":")
        item_ = next(iter(q_items(c["kind"])), None) if base_ in ("ready_in", "elapsed") else None
        p_ = pos_of_plate(k_) if k_ > 0 else 0.3
        for x_ in prep_floor(cell, m_, item_):
            p_ = max(p_, T.get(f"chop({x_})", T.get(f"load({x_})", 0.0)) + 0.2)
        if base_ in ("ready_in", "elapsed") and f"load({arg_})" in T:
            p_ = max(p_, T[f"load({arg_})"] + 0.2)
        if base_ == "past_done":
            # D15: asked once the first cooked item can be done (asked at the put-down / ignition its truth was "no"
            # 10/10 in round 2)
            p_ = max(p_, min([T[f"done({x_})"] for x_ in items if f"done({x_})" in T] or [0.0]) - 0.3)
        if base_ == "missing":
            p_ = max(p_, T.get("plate0", 2.0))
        return p_

    cands = []
    for c in cell_question_cands(cell, bool(inject)):
        kind = c["kind"]
        base = kind.split(":")[0]
        if ROUND3:
            c = dict(c, pos=live_pos(c), pos_r2=c["pos"])
        if q_blocked(kind, rows, cell):
            continue                                            # C7
        if base == "ready_in" and q_items(kind) & lu_items:
            continue
        if prem_pos is not None and c["pos"] >= prem_pos:
            continue                                            # C2
        # C7 (serve_raw): the raw scoop replaces the item's processing — the cook may never cook it (LLM check r19,
        # hard_hard_4: every question waiting for "potato processed" went unasked); no question about its processing
        # or waiting for it
        if raw_items and raw_items & q_items(kind) and base in ("ready_in", "elapsed", "order"):
            continue
        if prem:
            # C2: asked before the premature serve in every cooking order: about X or earlier items only, and (round 2)
            # its prep floor is cut to X and the items before it (below) instead of dropping the question
            ix = items.index(comp_of_sid([r for r in rows if r["fn"] == "premature_serve"][0]["kwargs"]["after"]))
            if any(i in items and items.index(i) > ix for i in q_items(kind)):
                continue
        # D8/E3 (round 2): the engine holds a next-step question while a forget_next_step is armed (pending or live), so
        # in a forget case it is asked late or never; and its true answer during the freeze is the repair
        if base == "next_step" and any(r["fn"] == "forget_next_step" for r in rows):
            continue
        if ROUND3:
            why3 = voice_r3_block(c, rows, cell)
            if why3:
                continue
        cands.append(c)
    # round 2 (E3): the voice slot is shut during a takeover, G ticks after it and while one is imminent — a question
    # anchored on a mistake's moment is asked late or lost. Prefer candidates away from every mistake's span (and a
    # forget's freeze); the others only when no quiet one is left.
    spans = []
    for r in rows:
        a0, e0 = span(r, cell)
        if ROUND3:
            # round 3: a forget's freeze SHIFTS the rest of the plan (the cook resumes where it stopped), it does not
            # cover it; a premature serve holds the voice from the moment its item is next on the plate in hand
            # (plating_imminent) until the serve
            if r["fn"] == "premature_serve":
                x_ = comp_of_sid(r["kwargs"].get("after"))
                a0 = min(a0, T.get(f"next({x_})", a0))
            if r["fn"] == "add_foreign" and r["kwargs"].get("before") == "serve":
                # the serve-time foreign item is imminent (voice held) from the moment one item is left (round-3 pilot,
                # easy_h4: a question floored on the first item plated was held until the foreign scoop, then the serve)
                a0 = min(a0, last_item_next(cell))
            if r["fn"] == "omit_subgoal":
                # follow-up (round-3 final run, easy_h1 / easy_m3: 6 asks lost at the ceiling): the omission plates the
                # NEXT item, so after its release only the skipped item and the serve are left — nothing is askable
                e0 = T["serve"]
            spans.append((a0, e0))
            continue
        spans.append((a0, e0 + ((6.0 if tier == "master" else FORGET_SPAN) if r["fn"] == "forget_next_step" else 0.0)))

    def too_close(p_, a0, e0):
        if not ROUND3:
            return max(a0 - p_, p_ - e0, 0.0) < QUIET_GAP - 1e-6
        # round 3: asymmetric — a question may come shortly BEFORE a takeover (the engine keeps VOICE_QUIET ticks
        # between an ask and a takeover) but needs the release plus G AFTER one
        return (a0 - QUIET_BEFORE_R3 < p_ < e0 + QUIET_GAP_R3)

    def noisy(c):
        return any(too_close(c["pos"], a0, e0) for a0, e0 in spans)
    if ROUND3:
        # D10 / C2b (round 3): the voice slot is shut during a takeover, G ticks after its release and while one is
        # imminent, so a question whose floor is reached there is asked at the first quiet moment after it (its
        # effective moment). A question is kept only if that moment exists before the serve (the ceiling closes the
        # slot once the serve is queued) and before a premature serve: round 2 lost 1 in 3 designed questions whose
        # floor came after the last mistake or inside a forget's freeze.
        def effective(c):
            p_ = c["pos"]
            for _ in range(8):
                hit = [e0 for a0, e0 in spans if too_close(p_, a0, e0)]
                if not hit:
                    return p_
                p_ = max(hit) + QUIET_GAP_R3
            return None
        keep_ = []
        for c in cands:
            e_ = effective(c)
            if e_ is None or e_ > T["serve"] - 1.0 or (prem_pos is not None and e_ >= prem_pos - 0.5):
                continue
            base_, _, arg_ = c["kind"].partition(":")
            if base_ in ("ready_in", "elapsed") and e_ > T.get(f"done({arg_})", 99) - 0.3:
                continue                                        # expires before it can be asked
            keep_.append(dict(c, pos_eff=e_))
        cands = keep_
    # D8: visit questions balanced yes / no over the suite (round 1: 16/16 "no")
    vt = visit_truths if visit_truths is not None else collections.Counter()
    want_yes = vt["yes"] < vt["no"]

    def visit_bonus(c):
        if not c["kind"].startswith("happened:visit("):
            return 0.0
        yes = not D3.negative(c["gt"])
        return 0.8 if yes == want_yes else -0.8
    # choose 2 of different families, balancing the suite's family counts, spreading their ask moments
    qsubs = []
    fams = set()
    want_neg = (sum(q_truths) / max(1, len(q_truths))) < D3.NEG_SHARE
    ok_open = OPEN_KINDS[(tier, opening_of(rows, cell))] if ROUND3 else collections.Counter()
    # D16 (round 3): cases of a tier that open with the same mistake ask different questions (medium m2/m3 and h2/h4
    # were identical up to t53/t61: the same past-done question, then the same walk-away)
    tk = TIER_KINDS[tier] if ROUND3 else collections.Counter()
    order_c = sorted(cands, key=lambda c: (noisy(c), ok_open[c["kind"].split(":")[0]], tk[c["kind"].split(":")[0]] // 3, fam_count[c["fam"]], -(c["score"] + visit_bonus(c) + (0.4 if (want_neg and D3.negative(c["gt"])) else 0)), rng.random()))
    for c in order_c:
        if len(qsubs) == Q_PER_CASE:
            break
        if c["fam"] in fams:
            continue
        if any(q["kind"] == c["kind"] for q in qsubs):
            continue
        if any(abs(q["pos"] - c["pos"]) < 0.5 for q in qsubs) and len(order_c) > 4:
            continue                                            # spread the two asks (the voice gap)
        qsubs.append(c); fams.add(c["fam"])
        fam_count[c["fam"]] += 1
        ok_open[c["kind"].split(":")[0]] += 1
        tk[c["kind"].split(":")[0]] += 1
        q_truths.append(1 if D3.negative(c["gt"]) else 0)
        if c["kind"].startswith("happened:visit(") and visit_truths is not None:
            visit_truths["no" if D3.negative(c["gt"]) else "yes"] += 1
    after_prem = set()
    if prem:
        _ix = items.index(comp_of_sid([r for r in rows if r["fn"] == "premature_serve"][0]["kwargs"]["after"]))
        after_prem = {i for i in items[_ix + 1:] if i not in items[:_ix + 1]}
    out = []
    for c in qsubs:
        out.append({"name": f"{case['name']}_q_{c['kind'].split(':')[0]}_{c['m']}", "family": "qa_probe", "cell": cell,
                    "err": "no_error", "trigger_after": c["m"], "intent_seq": [], "resume_from": c["m"],
                    "question": c["text"], "gt_kind": c["kind"], "expected_gt": c["gt"], "qa_family": c["fam"],
                    "hardness": round(c["score"], 2), "hardness_why": c["why"], "ask_tick_clean": c["tick"],
                    "ask_m_clean": c["m_ask"], "inject": inject, "auto": False, "_pos": c["pos"],
                    "prep_floor": {k: v for k, v in prep_floor(cell, c["m"], next(iter(q_items(c["kind"])), None) if c["kind"].startswith(("ready_in", "elapsed")) else None).items()
                                   if k not in raw_items and k not in after_prem}})
    # ---- Q10 standing request, right after its item goes on the heat
    psub = None
    cooked = [x for x in dict.fromkeys(items) if want[x] == "cooked"]
    if cooked and want_proactive:
        bad = lu_items | raw_items \
            | {q for c in qsubs for q in q_items(c["kind"]) if c["kind"].startswith("ready_in")}
        choices = [x for x in cooked if x not in bad]
        if ROUND3:
            # follow-up (round-3 final run, hard_medium_1: ready:fish 0/2): a cook that preps the other item BEFORE
            # lighting the first one runs into its mis_prep first, and the takeover covers the whole cook of that item
            first_lit = cooked[0] if cooked else None
            if any(r["fn"] == "mis_prep" and r["kwargs"].get("component") != first_lit
                   and moment(r, cell)[0] <= T.get(f"done({first_lit})", 0.0) for r in rows):
                choices = [x for x in choices if x != first_lit]
        # (round 3, D10: a request ready on the same tick as a takeover is now made first by the engine, so the
        # takeovers that follow the ignition no longer starve it; cooking_window_blocked() stays for ready_in questions)
        rng.shuffle(choices)
        for item in choices:
            m_on = next((i + 1 for i, (cmd, sid) in enumerate(flat) if cmd == "turn on" and (sid or "").split("#")[0] in (f"prep({item})", f"ignite({item})")), None)
            if m_on is None:
                continue
            if not robust_anchor(cell, m_on, item=item):
                continue
            _, ro, _ = D3.replay(cell, m_on)
            gt = GI.ground_truth(f"ready:{item}", ro.game, ro)
            if gt.get("ticks_left", 0) < 5:
                continue
            p_pos = T.get(f"load({item})", 1.0) + 0.3
            if prem_pos is not None and T.get(f"done({item})", 99) >= prem_pos:
                continue
            multi = items.count(item) > 1
            psub = {"name": f"{case['name']}_p_ready_{item}_{m_on}", "family": "qa_proactive", "cell": cell, "err": "no_error",
                    "trigger_after": m_on, "intent_seq": [], "resume_from": m_on, "item": item,
                    "question": (f"Give me a shout the moment the {item} I just put on the heat is done, would you?" if multi
                                 else f"Give me a shout the moment the {item} is done, would you?"),
                    "gt_kind": f"ready:{item}", "expected_gt": gt, "qa_family": "Q10", "ask_tick_clean": ro.tick,
                    "inject": inject, "auto": False, "_pos": p_pos, "prep_floor": {item: 1}}
            fam_count["Q10"] += 1
            break
    return out, psub


def design_plan_change(case, rows, voice, intents, used_intents):
    """An announced plan change (never a mistake) at a robust early moment, not entangled with a mistake (C5): its
    ingredient is no mistake's wrong item and no slot a mistake touches, and it shares no moment with a mistake."""
    cell = case["cell"]
    items = recipe_items(cell)
    T = moment_table(cell)
    inject = case.get("inject", True)
    err_pos = sorted(moment(r, cell)[0] for r in rows)
    prem = [r for r in rows if r["fn"] == "premature_serve"]
    if prem:
        return None
    anchors = D3.question_anchors(cell, [])

    def pos_of_plate(k):
        if k == 0:
            return T.get("plate0", 2.0) - 0.4
        return T.get(f"next({items[k]})", T["serve"]) if k < len(items) else T["serve"]
    isub = None
    # CAP (round 2, hard_2: 3 items + noodles + meat + peas = 6): the plan change's item needs a free plate slot
    extra_n = sum(1 for r in rows if r["fn"] in ("add_foreign", "misidentify_assemble", "repeat_step"))
    if len(items) + extra_n + 1 > 5:
        return None
    wrongs = {wrong_item(r) for r in rows if wrong_item(r)}
    touched = set().union(*[slots(r, cell) for r in rows]) if rows else set()
    for it in intents:
        ing = (it.get("goal_extend") or {}).get("ingredient")
        if (cell, it["name"]) in used_intents or ing in wrongs or ing in touched:
            continue
        raw = {r["kwargs"]["component"] for r in rows if r["fn"] == "serve_raw"}

        def pfloor(m):
            # a serve_raw item may never be processed (its raw scoop stands in for a unit): never wait for it
            return {k: v for k, v in prep_floor(cell, m).items() if k not in raw}

        def pc_pos(m):
            # it fires when BOTH its plate floor and its prep floor are reached: the later of the two moments
            pp = [T.get(f"chop({x})", T.get(f"load({x})", 0.0)) for x in pfloor(m)]
            return max([pos_of_plate(len(D3.key_at(cell, m)["plate"]))] + pp) + 0.3
        # PC (round 2): "on top before serving" is announced with the plate started (hard_3 announced it at t13 and
        # neither cook added the lemon)
        ms = [a["m"] for a in anchors if robust_anchor(cell, a["m"]) and len(D3.key_at(cell, a["m"])["plate"]) >= 1]
        ms = [m for m in ms if all(abs(pc_pos(m) - p) >= QUIET_GAP - 1e-6 for p in err_pos)]   # round 2: the engine gaps the rest
        if not ms:
            continue
        # the latest such moment with something on the plate ("on top"), else the latest in the prep phase
        if ROUND3:
            # F7 / C3 (round 3): the latest moment put the announcement on the serve tick (easy_m2 CE: never carried
            # out; round-3 pilot: held at the ceiling). Take the EARLIEST moment with the plate started that is clear of
            # every takeover (VOICE_QUIET before one, its release + G after one) and a plating step before the serve.
            spans_ = []
            for r in rows:
                a0, e0 = span(r, cell)
                if r["fn"] == "premature_serve":
                    a0 = min(a0, T.get(f"next({comp_of_sid(r['kwargs'].get('after'))})", a0))
                if r["fn"] == "add_foreign" and r["kwargs"].get("before") == "serve":
                    a0 = min(a0, last_item_next(cell))
                spans_.append((a0, e0))
            ms = [m for m in ms if pc_pos(m) <= T["serve"] - 0.5
                  and not any(a0 - QUIET_BEFORE_R3 < pc_pos(m) < e0 + QUIET_GAP for a0, e0 in spans_)]
            if not ms:
                continue
            m_i = min(ms, key=lambda m: (pc_pos(m), m))
        else:
            m_i = max(ms, key=lambda m: (len(D3.key_at(cell, m)["plate"]) >= 1, m))
        p_i = pc_pos(m_i)
        isub = {"name": f"{case['name']}_i_plan_change_{m_i}", "family": "plan_change", "cell": cell, "err": "no_error",
                "trigger_after": m_i, "resume_from": m_i, "intent_label": it["intent_label"],
                "intent_seq": list(it["intent_seq"]), "duration": it.get("duration"),
                "clean_tick": D3.tick_at(cell, m_i), "rationale": it.get("rationale"), "inject": inject, "auto": False,
                "goal_extend": it["goal_extend"], "_pos": p_i, "prep_floor": pfloor(m_i)}
        used_intents.add((cell, it["name"]))
        break
    return isub


_CELL_Q = {}


def cell_question_cands(cell, inject):
    """Every (question, anchor) of a cell that passes V1-V3 and the content-hardness test, independent of the case's
    mistakes (those filters are applied per case). Cached per cell."""
    key = (cell, inject)
    if key in _CELL_Q:
        return _CELL_Q[key]
    items = recipe_items(cell)
    rec = recipe_of(cell)
    ordered = bool(getattr(rec, "ordered", False))
    ref = D3.REF.get(cell) or 100
    T = moment_table(cell)
    anchors = D3.question_anchors(cell, [])
    case = {"cell": cell, "inject": inject}

    def pos_of_plate(k):
        if k == 0:
            return T.get("plate0", 2.0) - 0.4
        return T.get(f"next({items[k]})", T["serve"]) if k < len(items) else T["serve"]
    templates = D3.question_templates(cell)
    # R3 (2026-09-19): visit questions are kept and scored (truth read off the world) — the v3 designer had dropped them
    from cooksim_bench.scoring.validate_suite_questions import check_sub
    for w in ("pot", "pan", "fryer", "oven", "trash"):
        kind = f"happened:visit({w})"
        q = f"Have I been over to the {w} at any point so far?"
        if not any(sev == "error" for _c, sev, _m in check_sub({"family": "qa_probe", "gt_kind": kind, "question": q}, case)):
            templates.append((kind, q))
    out = []
    for kind, text in templates:
        base = kind.split(":")[0]
        chk = check_sub({"family": "qa_probe", "gt_kind": kind, "question": text}, case)
        if any(sev == "error" for _c, sev, _m in chk) or any(c in ("M1", "T1") for c, _s, _m in chk):
            continue                                            # V1
        if base == "after_ref" and not ROUND4:
            continue                                            # D8: its answer is the recipe's next item
        if base == "order" and not ROUND4 and kind not in VARIABLE_ORDER.get(tier_of(cell), set()):
            continue                                            # D8: the LLM cooks do this pair in one order only
        if base == "order":
            # V2 for the event sides: only prep an LLM cook does early, and only platings two items short of complete
            bad = False
            for side in kind.split(":", 1)[1].split("|"):
                x = comp_of_sid(side)
                if side.startswith(("prep(", "ignite(")) and x not in EARLY_PREP[tier_of(cell)]:
                    bad = True
                if side.startswith("assemble(") and x in items and items.index(x) >= latest_floor(cell):
                    bad = True
            if bad:
                continue
        for a in anchors:
            m = a["m"]
            if not robust_anchor(cell, m, item=next(iter(q_items(kind)), None) if base == "ready_in" else None):
                continue                                        # V2
            if (base == "count" or kind.startswith("happened:visit(")) and len(D3.key_at(cell, m)["plate"]) < 1:
                continue                                        # D8: not before the counted thing can have happened
            m2 = ask_moment(cell, m, kind)
            if m2 is None:
                continue                                        # V3
            k2 = D3.key_at(cell, m2)
            _, ro, _ = D3.replay(cell, m2)
            try:
                gt = GI.ground_truth(kind, ro.game, ro)
            except Exception as e:                              # noqa: BLE001
                gt = {"error": repr(e)}
            if not D3.well_defined(kind, gt):
                continue
            if ROUND4 and base == "after_ref":
                # B10: after_ref is back. qa_v3_lib's hardness rejects it on the DESIGN-TIME replay, whose cook
                # plates the whole dish in a handful of ticks ("referent plated 1 ticks ago"); live, an LLM cook
                # plates 10-30 ticks apart. What the rule is really after is that the referent is not the last
                # thing that happened, which is a fact about the PLATE, not about ticks: at least one more item
                # has gone on since. The truth itself is read off the live plate history (gt_after_ref), so it is
                # right by construction whatever the cook did.
                pl_ = list(k2["plate"])
                ok = bool(gt.get("referent")) and arg_ref_ok(pl_, kind.split(":", 1)[1])
                sc, why = 0.95, f"plate {pl_}: the referent is {len(pl_) - pl_.index(kind.split(':', 1)[1]) - 1} platings back"
            else:
                ok, sc, why = Q.hardness(kind, gt, ro, ro.tick, ref, ordered=ordered, first_error_m=None, m=m2)
            if not ok:
                continue
            out.append({"kind": kind, "text": text, "m": m, "m_ask": m2, "gt": gt, "score": sc, "why": why,
                        "fam": Q.family_of(kind), "tick": ro.tick, "pos": pos_of_plate(len(k2["plate"]))})
    # one anchor per kind is enough choice: keep, per kind, the smallest trigger m for each ask moment
    best = {}
    for c in out:
        k = (c["kind"], c["m_ask"])
        if k not in best or c["m"] < best[k]["m"]:
            best[k] = c
    _CELL_Q[key] = list(best.values())
    print(f"[voice] {cell}: {len(_CELL_Q[key])} question candidates over {len({c['kind'] for c in _CELL_Q[key]})} kinds", flush=True)
    return _CELL_Q[key]


def arg_ref_ok(plate, arg):
    """B10: something was plated after the referent. (The round-4 real-cook check asked for TWO platings after it,
    which on the burrito is only reached as the plate completes: 12 of 16 asks were lost at the serve ceiling.
    One is enough here because round 4 only asks after_ref in a case whose own mistakes move that plating.)"""
    if arg not in plate:
        return False
    return len(plate) - (plate.index(arg) + 1) >= 1


def synth_mis_prep(by_cell):
    """mis_prep on the potato (chopped, then fried) and on the burrito's steak (chopped, then pan-fried): the wrong item
    is chopped on the board in its place. The template
    replay reports the anchor as collapsing (its reference cook reaches for the potato before any plate event), which
    says nothing about the live trigger (the cook reaching for the potato). Offsets copied from a mis_prep row."""
    from cooksim.core.enums import choppable
    tpl = next((r for rows in by_cell.values() for r in rows if r["fn"] == "mis_prep"), None)
    for cell, rows in by_cell.items():
        comp = {"hard": "potato", "expert": "potato", "master": "steak"}.get(tier_of(cell))
        if comp is None:
            continue
        items = recipe_items(cell)
        from cooksim_bench import exp_interact as X
        _st, ro = X.build_episode(None, cell=cell, inject=True if not cell.endswith("hard_map") else False)
        srcs = {str(t.value)[:-7] for row in ro.game.layout.grid for t in row if str(t.value).endswith("_source")}
        for w in sorted(srcs):
            if w in items or w == "plate" or not choppable(w):
                continue
            if any(r["fn"] == "mis_prep" and r["kwargs"] == {"component": comp, "wrong": w} for r in rows):
                continue
            r = dict(tpl)
            r.update(cell=cell, err="mis_prep", fn="mis_prep", kwargs={"component": comp, "wrong": w},
                     desc=f"preps {w} where the recipe needs {comp}", name=f"synth_mis_prep_{comp}_{w}_{cell}",
                     synth_from=tpl["name"], inject=not cell.endswith("hard_map"))
            rows.append(r)


def synth_wrong_items(by_cell):
    """Round 2 (D3): the kitchens gained a second off-recipe crate (cooksim_bench/exp_interact.OFF_RECIPE_MIN). The inventory
    was replayed before, so its foreign-item rows name only the old crate: copy every (function, slot) row to each
    off-recipe crate of the kitchen it does not name yet — the same act (walk to a crate, scoop / chop) with another
    item; the grading offsets are the template row's."""
    from cooksim.core.enums import choppable
    from cooksim_bench import exp_interact as X
    for cell, rows in by_cell.items():
        items = recipe_items(cell)
        _st, ro = X.build_episode(None, cell=cell, inject=not cell.endswith("hard_map"))
        srcs = sorted({str(t.value)[:-7] for row in ro.game.layout.grid for t in row if str(t.value).endswith("_source")}
                      - set(items) - {"plate"})
        groups = collections.defaultdict(list)
        for r in rows:
            if r["fn"] in ("add_foreign", "misidentify_assemble", "mis_prep") and r["kwargs"].get("wrong"):
                kw = {k: v for k, v in r["kwargs"].items() if k != "wrong"}
                groups[(r["fn"], json.dumps(kw, sort_keys=True))].append(r)
        for (fn, kws), grp in groups.items():
            have = {r["kwargs"]["wrong"] for r in grp}
            tpl = next((r for r in grp if r["kwargs"]["wrong"] not in items), grp[0])
            for w in srcs:
                if w in have or (fn == "mis_prep" and not choppable(w)):
                    continue
                r = dict(tpl)
                kw = dict(json.loads(kws), wrong=w)
                r.update(kwargs=kw, name=f"synth_{fn}_{w}_{cell}_{len(rows)}", synth_from=tpl["name"],
                         desc=(tpl.get("desc") or "").replace(tpl["kwargs"]["wrong"], w))
                rows.append(r)


def synth_rows(by_cell):
    """Live-plausible candidates the template replay gate cannot certify (its reference cook plates the potatoes only
    once both are done, so serve_raw(potato) 'times out' there): serve_raw on the second-stage potatoes of Fish & Chips
    and Steak Frites. The row copies the grading offsets (dev/identifiable/window) of the same cell's misidentify on the
    potato slot — the same act (a scoop from the potato crate onto the plate at the potato moment)."""
    for cell, rows in by_cell.items():
        if tier_of(cell) not in ("hard", "expert"):
            continue
        tpl = next((r for r in rows if r["fn"] == "misidentify_assemble" and r["kwargs"].get("component") == "potato"), None)
        if tpl is None or any(r["fn"] == "serve_raw" and r["kwargs"].get("component") == "potato" for r in rows):
            continue
        r = dict(tpl)
        r.update(err="serve_raw", fn="serve_raw", kwargs={"component": "potato"}, desc="plates potato raw",
                 name=f"synth_serve_raw_potato_{cell}", synth_from=tpl["name"])
        rows.append(r)


def removal_reasons(case, old_errs, rows, pool):
    """For every v4 error not kept: the rule that forbids it now (row rule, or the first set rule its v4 case breaks),
    else "re-picked for diversity" (it is legal but the tier's diversity/t1 caps chose another set)."""
    cell = case["cell"]
    kept = {key_of(r) for r in rows}
    old_rows = []
    for s_ in old_errs:
        k = ((s_.get("fn") or s_["err"]), json.dumps(s_["kwargs"], sort_keys=True))
        r = next((x for x in pool if key_of(x) == k), None)
        old_rows.append((s_, r))
    try:
        set_why = set_violation([r for _s, r in old_rows if r is not None], cell) if all(r is not None for _s, r in old_rows) else None
    except Exception as e:                                  # noqa: BLE001
        set_why = f"(unevaluable: {e!r})"
    out = {}
    for s_, r in old_rows:
        k = ((s_.get("fn") or s_["err"]), json.dumps(s_["kwargs"], sort_keys=True))
        if k in kept:
            continue
        label = f"{s_.get('fn')}{json.dumps(s_['kwargs'], separators=(',', ':'))}"
        if r is None:
            out[label] = row_ok({"fn": k[0], "kwargs": s_["kwargs"], "err": s_["err"]}, cell) or \
                "not in the v5 inventory (the replay gate rejects it on the v5 kitchens)"
        else:
            out[label] = row_ok(r, cell) or (f"v4 set breaks {set_why}" if set_why else "legal, re-picked for tier diversity (C4)")
    return out


# ----------------------------------------------------------------------------------------------- assembly
def error_sub(case, row, note):
    sub = {k: v for k, v in row.items() if not k.startswith("anchor_") and k not in ("collapses", "plan_len", "desc", "name")}
    sub.update(name=f"{case['name']}_{row['err']}", cell=case["cell"], inject=case.get("inject", True),
               persona="capable_collaborator", family="error", ruinous=is_ruinous(row))
    if note:
        sub["design_v5"] = note
    if row["fn"] == "leave_unattended":
        # E9 (round-2 decision): an explicit floor, never the reference run's prep (see lu_floor)
        pf, plf = lu_floor(case["cell"], row["kwargs"]["component"])
        sub["prep_floor"] = dict(pf)
        sub["plate_floor"] = int(plf)
        if ROUND3 and tier_of(case["cell"]) == "master" and row["kwargs"]["component"] == "steak":
            # D17 (round 3): the LLM cook's steak is done 18-22 ticks before the rice is plated (round-3 pilot, 3/3), so
            # the rice-plated floor is met after the +-6 near-done window; the walk-away may fire until 5 ticks before
            # the smoke (engine `lu_late`)
            sub["lu_late"] = 23      # follow-up: the round-3 final run found 21-22 ticks after done on the hard map
        
    sub["_pos"] = moment(row, case["cell"])[0]
    return sub



# =============================================================================================================
# Round 4 (2026-09-20): the BALANCED suite.
#
# User ruling: "注意一下 number of each error type 在整个 benchmark 中的 balancing … 长尾的 error type 都挺有意义的"
# — the round-3 suite put 84% of its 150 mistakes in four types (leave_unattended 34, forget_next_step 32,
# add_foreign 31, mis_prep 29) and left serve_raw and repeat_step at zero, wrong_order at one. Round 4 rebuilds
# the 50 cases from scratch against a per-type QUOTA instead of per-tier caps, on the fire-longtail engine
# (main a9285b1) where every plating mistake is FORCED and any sub-case may state its own prep/plate floor.
#
# What changed in the engine, and what the designer may do with it (python -m cooksim_bench.e2e_stepin_rollout):
#   * live_trigger: serve_raw / misidentify_assemble / add_foreign / wrong_order / repeat_step / omit_subgoal /
#     premature_serve no longer wait for the plate to be in the cook's hands, for the component to be the cook's
#     own next plating, or for the cook to be walking there. The takeover parks what the cook holds, fetches THE
#     dish plate, walks to the source and scoops. What is left is the mistake's own meaning:
#       serve_raw(x)          no unit of x is processed yet, and none finishes inside the walk to x's crate
#       misidentify(x) / add_foreign(before assemble(x))   x's slot is still open
#       add_foreign(serve)    the plate is complete (or the cook's serve is queued)
#       wrong_order(a, b)     neither a nor b plated; b ready, or a chop the block makes itself (D12)
#       repeat_step           `at` reached as a PROGRESS floor; the duplicate is scoopable (raw from the crate,
#                             chopped the block makes itself, cooked only when a spare cooked unit exists)
#       omit_subgoal(x)       x's slot open and the NEXT recipe item ready (the block plates that one)
#       premature_serve(x)    as many units on the plate as the recipe wants up to and including x
#     and the two nonsense guards (the dish is under way; the plate is not complete / the serve not queued).
#   * ANY sub-case may state `prep_floor` / `plate_floor`. That is the one knob for placing a forced mistake
#     mid-dish in a way that is reachable in EVERY cooking order (it counts the most items ever plated and the
#     most units ever in processing, never an exact reference plate). Round 4 states both on every forced
#     mistake — `prep_floor` must be written even when it is empty, or derive_spec falls back to the reference
#     run's prep, which an LLM cook that works in another order may never have.
#   * Grading windows are LIVE: they open at the first observable deviated act and close at the commit. The
#     inventory's dev/identifiable/window offsets are dead weight; a synthesised row may copy any template's.
#
# Round-4 rules (the round-1..3 rules that are not listed here stay in force unchanged):
#   B1  per-type quota. 10 fns x 14 = 140 mistakes (easy 2, medium 2, hard 3, expert 4, master 3 per case, the
#       user's tier counts). Accept 10-20 per type; no type above 20% of the suite or below 5%; no type in more
#       than R4_TIER_CASE_CAP of a tier's 10 cases (so no tier has a type in every case); no two cases anywhere
#       with the same design (normalised).
#   B2  every forced mistake carries an explicit (prep_floor, plate_floor) and its moment is READ OFF that floor,
#       not off the cook's plan: pos = the moment the floor's k-th item is plated / its prep is done. That is what
#       makes the long tail placeable at all.
#   B3  the type's own meaning bounds the floor: a slot-filling mistake (serve_raw, misidentify, add_foreign at an
#       item, wrong_order's a, an omission) must fire while that slot is open, so plate_floor <= the item's index;
#       serve_raw needs its item still unprocessed at the floor, with R4_RAW_MARGIN moments of room.
#   B4  CAP on the 5-slot plate is unchanged: on the burrito only the LAST item may be replaced or duplicated on
#       the plate (a 6th scoop is refused and the cook loops), so master's extra-item mistakes are at the lettuce
#       and its repeats are PREP repeats (which cost time and no slot).
#   B5  a prep repeat may not light a second unit: the burger's prep(meat) node ends in "turn on", so a repeated
#       burger prep leaves a second patty burning untracked. Only chop-only or unlit-station preps are repeated.
#   B6  forget_next_step is allowed after the second-to-last item when the LAST item is raw and ready (the burger's
#       cheese): the forgotten step is then that scoop, never the serve. R9's ban was about a plate the engine
#       reads as name-complete.
#   B7  omit_subgoal is allowed where the item after the skipped one is COOKED, as long as the omission's floor is
#       at or after that item's done moment (the engine waits for it anyway). That is what gives the burrito its
#       omissions (skip the tortilla, plate the rice).
#   B8  wrong_order may swap in a chopped item from later in the recipe (never the last one): the block makes the
#       chopped b itself (D12), and the items between a and b still come in order afterwards.
#   B9  robust ruinous (D14, re-read on the forced engine): a ruinous mistake whose floor every cook reaches in
#       every order. The forced plating mistakes qualify; a serve_raw qualifies only with the B3 margin; a
#       walk-away and a forget never do (not ruinous at all), and neither does a mistake the smoke ban can hold
#       (no foreign item at the fish / steak, D11).
#   B10 questions: 2 per case (100), spread over the eight families the templates can produce, and `order` (Q1)
#       and `after_ref` (Q7) are back. after_ref is read off the live plate history (qa_v3_lib.gt_after_ref), so
#       its truth is right by construction; what made it recipe-predictable was asking it where no mistake had
#       touched the plate, so round 4 asks it only where one has. `order` keeps the measured VARIABLE_ORDER pairs
#       and adds the pairs measured on the round-3 real-cook run (R4_ORDER_EXTRA).
#   B11 requests: R4_PROACTIVE_TARGET = 30, spread over items and tiers (at most R4_REQ_PER_TIER per tier).
#   B12 plan changes: a designed family of R4_PLAN_CHANGES = 11, on different items and tiers, announced by the
#       cook and checked on the served plate.
# =============================================================================================================
ROUND4 = os.environ.get("DESIGN_ROUND", "4") in ("4", "5")
R4_TIER_N = {"easy": 2, "medium": 2, "hard": 3, "expert": 4, "master": 3}      # user ruling: master down to 3
R4_FNS = ["mis_prep", "forget_next_step", "leave_unattended", "add_foreign", "misidentify_assemble",
          "serve_raw", "omit_subgoal", "premature_serve", "wrong_order", "repeat_step"]
R4_TIER_CASE_CAP = 8        # B1: no type in more than 8 of a tier's 10 cases
R4_RAW_MARGIN = 0.6         # B3: moments of room between a serve_raw's floor and its item's processing
                            # (one moment is ~25 ticks on the coarse easy/medium tables, 10 on the burrito's)
R4_PROACTIVE_TARGET = 30
R4_REQ_PER_TIER = 9
R4_PLAN_CHANGES = 11
R4_Q_TARGET = 2             # questions per case
R4_FAM_CAP = 14            # B10: no question family above ~15% of the ~100 questions
R4_Q_EXTRA = 14            # B10: cases that may carry a THIRD question (see the third-ask pass)
# B10: order pairs LLM cooks really do in both orders. The round-2 set was measured on the round-1 run; these are
# the pairs measured the same way on the 100 round-3 real-cook episodes (tmp/suite_v5r3/llm_final), each order in
# >= 25% of the episodes that decided it. Filled by cooksim_bench/measure_order_variability.py.
R4_ORDER_EXTRA = {}
_r4_order_path = os.path.join(ROOT, "benchmark/order_variability_r4.json")
if os.path.exists(_r4_order_path):
    R4_ORDER_EXTRA = {k: set(v) for k, v in json.load(open(_r4_order_path)).items()}


# =============================================================================================================
# ROUND 5 (2026-09-20). The round-4 hand review's four DESIGN items (docs/FIRING_AND_GRADING.md §6, "Left to the
# designer") plus the placements the round-5 real-cook run showed cannot fire. DESIGN_ROUND=4 restores round 4.
#
#   H2  FLOOR DOMINATION. The scheduler is reference-free: every pending sub-case is checked each tick and the one
#       that became READY FIRST takes the body slot. A later sub-case whose floor is reached earlier therefore
#       pre-empts an earlier one, which then waits for a moment the episode never comes back to
#       (`b3_expert_nops_medium_map_5` baseline: the walk-away fired at t64 and the premature serve at t111 while
#       `serve_raw{potato}@prep{potato:1}` sat unfired from t0 — the cook never chopped the potato). Round 5
#       therefore requires, for every pair i < j in moment order, that j's EFFECTIVE floor dominate i's: at least as
#       many items plated and at least as much prep. Both quantities are monotone in the engine (`max_plate_len`,
#       `prep_max`), so domination means j cannot become ready before i. The effective floor is the stated
#       (prep_floor, plate_floor) widened by the type's own anchor (a forget after `assemble(x)` needs the plate
#       through x; a prep repeat needs that prep) and by what the plate floor implies (to plate k real items the
#       cook has prepped items[:k]).
#   H6  CONSUMPTION. A sub-case whose STATED floor names x may not follow a mistake that consumes x. Round 4 had
#       this for serve_raw / misidentify / omit (rule SR+); round 5 adds `mis_prep{x}` (the cook preps the wrong
#       thing and x is never prepped: `b3_easy_nops_medium_map_4` C lost 2 of 3 sub-cases) and `wrong_order{a,x}`
#       (its block chops and plates x itself: `b3_master_nops_hard_map_2` was structurally a 2-error case).
#   H6b MIS_PREP IS THE FIRST MISTAKE. Its trigger is the cook reaching for the component's prep, which an LLM cook
#       does early whatever the moment table says; behind another takeover the prep has already happened and the
#       sub-case starves (5 of the 100 round-5 episodes: `the cook is not reaching for the potato yet`).
#   H9  NO UNLABELLED SECOND MISTAKE. Every recipe in the suite is `ordered`, so a `serve_raw{x}` placed at a plate
#       floor below x's own index puts x on the plate before the items that come before it and commits a
#       `wrong_order` no sub-case names (6 instances on Steak Frites). Its plate floor is raised to x's index.
#   R5-F A forget must keep a step that is not the serve. A forget one scoop from a complete plate, or any forget
#       behind a plating takeover on a two-item dish, meets `the only step left is the serve` (4 instances).
#   R5-LU The burrito steak walk-away is retired: 0/6 in the round-5 run and 0/8 in the round-2 pilot — by the time
#       the rice is plated the steak is off the heat. Master therefore carries no walk-away.
#   H8  DIVERSITY: a per-TIER cap on one placement (4 of 10 cases, was 6 suite-wide) and a penalty for two cases of
#       a tier whose designs differ in a single mistake (expert medium_1/2 differed only in the forget anchor).
#   H7  A question is not asked on the tick its referent event completes: `order`, `after_ref`, `elapsed` and
#       `ready_in` carry a floor R5_REF_GAP ticks after the last event they name.
# =============================================================================================================
ROUND5 = os.environ.get("DESIGN_ROUND", "5") == "5"
R5_ROW_TIER_CAP = 4         # H8: one placement in at most 4 of a tier's 10 cases
R5_REF_GAP = 10             # H7: ticks between a question's last referent event and its floor
R5_RAW_MARGIN_TICKS = 12    # B3, in ticks: the room between a serve_raw's floor and its item becoming ready
R5_DUP_TICKS = 35           # R5-RP: ticks a duplicate unit may sit on a lit station before the dish goes out
R5_TYPE_MIN, R5_TYPE_MAX = 10, 17     # B1: every realised type keeps 7-12% of the 140 mistakes
R5_RESTARTS = 60            # seeded kicks of the iterated local search over the assignment
# F6 (user ruling, 2026-09-20): the walk-away's late window widens to ~20 ticks past done and the forget freeze
# drops from 62 to ~35. With those constants the rule below is FREE -- measured on the candidate pools, expert
# keeps 8 admissible four-mistake sets and hard 28, exactly the counts with no rule at all -- whereas at 62/+-6
# it cost the expert cell 13 sets down to 2. It schedules the case forward (each takeover starts at its moment
# or when the previous one ends, whichever is later) and refuses a set in which a walk-away is pushed more than
# R5_LU_SLACK ticks past its own moment. That is the defect behind leave_unattended firing 15 of 34 in the
# round-5 real-cook run: in `b3_expert_nops_hard_map_4` the freeze ran t23-85, the prep slip behind it was
# displaced to t97-118, and that block sat exactly across the steak's near-done window (done t109).
R5_FS = os.environ.get("COOKSIM_FS_RULE", "1") == "1"
R5_FORGET_TICKS = 35        # the freeze, after the ruling (was 62)
R5_BLOCK_TICKS = 18         # a forced block that is not a freeze (measured median)
R5_LU_SLACK = 20            # how far a walk-away may be pushed past its moment (its late window, after the ruling)
# F4 follow-up: the burrito has NO mistake between the rice going done (moment 4.2) and the plate starting (9.0)
# -- the whole stretch where the rice sits in the pot while the cook chops and cooks the steak. Round 3 retired
# the rice walk-away (D17) because it fired 1 of 3 in the round-2 pilot against a +-6-tick near-done window. With
# a widened late window it is the only placement that fills that stretch, and it consumes nothing: it touches no
# plate slot and its own stated wait is prep(rice), which every case already has. Re-admitting it takes the
# burrito from 77 admissible three-mistake sets (44 distinct designs) to 94 (56), and in all 17 sets that carry
# it the walk-away is the case's FIRST mistake with the next one 5.3 moments (53 ticks) behind it, past the
# rice's smoke and the cook's own rescue. OFF until the engine's walk-away window actually widens.
R5_BURRITO_RICE_LU = os.environ.get("COOKSIM_BURRITO_RICE_LU", "1") == "1"


def r5_stated_prep(c, cell):
    """The prep a candidate explicitly WAITS for: its own prep_floor plus the prep its anchor names. This is what
    an earlier mistake can starve (H6); it never includes prep merely implied by a plate floor."""
    prep = {k: int(v) for k, v in (c.get("prep_floor") or {}).items()}
    fn, kw = c["fn"], c["kwargs"]
    if fn == "forget_next_step":
        a = str(kw.get("after", "")).split("#")[0]
        if a.startswith("prep("):
            x = comp_of_sid(a)
            if x:
                prep[x] = max(prep.get(x, 0), 1)
    elif fn == "repeat_step":
        sid = str(kw.get("sid", ""))
        if sid.startswith("prep("):
            x = comp_of_sid(sid)
            if x:
                prep[x] = max(prep.get(x, 0), 1)
    elif fn == "leave_unattended":
        x = kw.get("component")                 # to be near done on the heat it has been prepped and lit
        if x and want_of(cell).get(x) != "raw":
            prep[x] = max(prep.get(x, 0), 1)
    return prep


def r5_eff_floor(c, cell):
    """(items plated, {item: units prepped}) — the weakest world state in which this candidate can become ready.
    Both components are monotone in the engine, so a floor that dominates another is reached no earlier."""
    items, want = r4_items(cell), want_of(cell)
    plate = int(c["plate_floor"]) if c.get("plate_floor") is not None else 0
    prep = r5_stated_prep(c, cell)
    fn, kw = c["fn"], c["kwargs"]

    def through(sid):
        x = comp_of_sid(sid)
        return (items.index(x) + 1) if x in items else 0

    if fn in ("forget_next_step", "premature_serve") and str(kw.get("after", "")).startswith("assemble("):
        plate = max(plate, through(kw["after"]))
    elif fn == "repeat_step" and str(kw.get("at", "")).startswith("assemble("):
        plate = max(plate, through(kw["at"]))
    # what the plate floor itself implies: to have plated k items the cook has prepped the ones that need it
    for y in items[:plate]:
        if want.get(y) != "raw":
            prep[y] = max(prep.get(y, 0), 1)
    return plate, prep


def r5_floor_dominates(f2, f1):
    if f2[0] < f1[0]:
        return False
    return all(f2[1].get(x, 0) >= n for x, n in f1[1].items())


def r5_consumes(b, x):
    """Does mistake b leave component x unprepped / already used up for a later sub-case that waits on it?"""
    fn, kw = b["fn"], b["kwargs"]
    if fn in ("serve_raw", "misidentify_assemble") and kw.get("component") == x:
        return f"the earlier {fn} replaces it"
    if fn == "omit_subgoal" and comp_of_sid(kw.get("sid")) == x:
        return "the earlier omission skips it"
    if fn == "mis_prep" and kw.get("component") == x:
        return "the earlier mis_prep preps something else instead"
    if fn == "wrong_order" and kw.get("b") == x:
        return "the earlier swap preps and plates it itself"
    return None


def r5_is_terminal(c, cell):
    """A sub-case after which the dish is served: the episode has no room left for anything behind it. A premature
    serve carries the plate to the pass itself; a mistake that fires with the plate one item short is followed by
    the cook's own last scoop and its serve."""
    items = r4_items(cell)
    if c["fn"] == "premature_serve":
        return True
    if r5_eff_floor(c, cell)[0] >= len(items):
        return True                              # its floor is a plate that already holds the whole dish
    x = r4_slot_item(c, cell)
    return x is not None and x == items[-1] and items.count(x) == 1


def r5_nondominating(rows, cell):
    """The (later, earlier) pairs whose floors are not ordered: the later one can become ready first. H2 forbids
    this where the later one is TERMINAL; elsewhere it only reorders two takeovers, and the optimiser pays for it."""
    sp = sorted(rows, key=lambda r: (r["pos"], r["fn"]))
    fl = [r5_eff_floor(r, cell) for r in sp]
    out = []
    for j in range(1, len(sp)):
        for i in range(j):
            if not r5_floor_dominates(fl[j], fl[i]):
                out.append((sp[j], sp[i], fl[j], fl[i]))
    return out


def r5_set_violation(rows, cell):
    """The round-5 rules on a candidate set (None = admissible)."""
    items = r4_items(cell)
    sp = sorted(rows, key=lambda r: (r["pos"], r["fn"]))
    # H6b: a prep slip fires when the cook REACHES for that prep, which its moment now names directly
    # (`chop(x) - 0.5`, round 5) instead of the first item's done moment. The M gap then does the work: no earlier
    # takeover may still be running a full plating step before the cook gets there. On Fish & Chips and Steak
    # Frites that makes the potato slip the case's first mistake in practice; on the burrito, where the tomato is
    # chopped after the steak is plated, earlier mistakes are legal and the five starved instances of the
    # round-5 run were all H6 (a mistake that consumed the prep), not lateness.
    for later, earlier, f2, f1 in r5_nondominating(rows, cell):
        if r5_is_terminal(later, cell):
            return (f"H2 the {later['fn']} ends the dish at (plate>={f2[0]}, prep>={f2[1] or '{}'}), which does not "
                    f"dominate the earlier {earlier['fn']} floor (plate>={f1[0]}, prep>={f1[1] or '{}'})")
    # H6: a stated wait on a component an earlier mistake consumes. A mistake that itself waits for prep(x)
    # (`serve_raw{potato}` behind `potato: 1` — the cook chops, then the block scoops an unfried one) does not
    # consume it: the prep has already happened when the block runs.
    for j, r in enumerate(sp):
        for x, n in r5_stated_prep(r, cell).items():
            for b in sp[:j]:
                if r5_stated_prep(b, cell).get(x, 0) >= n:
                    continue
                why = r5_consumes(b, x)
                if why:
                    return f"H6 the {r['fn']} waits for a {x} and {why}"
    # F3: H9's plate floor counts every unit ever plated, not recipe items. A foreign or duplicate unit from an
    # earlier mistake therefore satisfies it one item early, and the raw unit goes down before the items that
    # belong in front of it after all (`b3_hard_nops_medium_map_3`: the misidentify put lettuce on the plate, the
    # raw potato followed at plate_floor 1, and the fish was plated behind both -- an unnamed wrong_order).
    for j, r in enumerate(sp):
        if r["fn"] != "serve_raw" or int(r.get("plate_floor") or 0) <= 0:
            continue
        for b in sp[:j]:
            if b["fn"] in ("add_foreign", "misidentify_assemble") or (
                    b["fn"] == "repeat_step" and str(b["kwargs"].get("sid", "")).startswith("assemble(")):
                return (f"F3 the raw {r['kwargs'].get('component')}'s plate floor counts the unit the earlier "
                        f"{b['fn']} puts on the plate, so it would go down out of order")
    if R5_FS:
        tpm = float(r4_tpm(cell))
        clock = -10.0
        for r in sp:
            start = max(r["pos"], clock + R4_MIN_GAP)
            if r["fn"] == "leave_unattended" and (start - r["pos"]) * tpm > R5_LU_SLACK:
                return (f"F6 the walk-away's near-done window is {(start - r['pos']) * tpm:.0f} ticks behind the "
                        f"takeovers in front of it (its late window is {R5_LU_SLACK})")
            clock = max(r["end"], start) + (R5_FORGET_TICKS if r["fn"] == "forget_next_step"
                                            else R5_BLOCK_TICKS) / tpm
    # R5-F: the forget must keep a step that is not the serve
    for r in sp:
        if r["fn"] != "forget_next_step":
            continue
        if r5_eff_floor(r, cell)[0] >= len(items) - 1 and any(o["pos"] < r["pos"] for o in sp):
            return "R5-F a forget one scoop from a complete plate sits behind another takeover"
        if len(items) <= 2 and any(o["pos"] < r["pos"] and r4_is_plating(o) for o in sp):
            return "R5-F on a two-item dish a forget behind a plating mistake meets a plate one scoop from done"
    return None


def r5_event_pos(cell, sid):
    """The moment a question's referent event happens on this cell's moment table."""
    T = moment_table(cell)
    x = comp_of_sid(sid) or sid
    if sid.startswith("assemble("):
        return T.get(f"plated({x})")
    if sid.startswith("prep("):
        return T.get(f"chop({x})", T.get(f"load({x})"))
    if sid.startswith("ignite("):
        return T.get(f"load({x})")
    return None


def r5_ref_plate_floor(cell, kind):
    """H7, the half the engine can be told: the PLATE the question waits for.

    A question opens on the first tick its floor and its precondition are both met, and an `order` question's
    precondition is "both events have happened" — so unless something later binds, it is asked on the very tick
    the second event completes (16 of the 21 order asks in the round-5 run, `t_b == asked_at`). A prep floor
    cannot help: it names the same events. A PLATE floor can. The cook plates in recipe order and cannot plate
    item k without having prepped and cooked it, so "the plate through the latest item the question names" is
    reached strictly after both events, with that item's cook and walk as the gap. Returns the least plate length
    a candidate's anchor must carry, or 0 when the family needs none."""
    items = r4_items(cell)
    base, _, arg = kind.partition(":")
    if base == "order":
        idx = [items.index(x) for x in (comp_of_sid(s_) for s_ in arg.split("|")) if x in items]
        return (max(idx) + 1) if idx else 0
    if base == "after_ref":
        # the answer is the plating AFTER the referent: the referent plus one more, and one past that to ask
        return (items.index(arg) + 2) if arg in items else 0
    return 0


def r5_ref_pos(cell, kind):
    """H7: the earliest moment a question may be asked so that its last referent event is R5_REF_GAP ticks old."""
    base, _, arg = kind.partition(":")
    T = moment_table(cell)
    ps = []
    if base == "order":
        ps = [p for p in (r5_event_pos(cell, s) for s in arg.split("|")) if p is not None]
    elif base == "after_ref":
        p = T.get(f"plated({arg})")
        ps = [p] if p is not None else []
    elif base in ("ready_in", "elapsed"):
        p = T.get(f"load({arg})")
        ps = [p] if p is not None else []
    if not ps:
        return 0.0
    return max(ps) + R5_REF_GAP / float(r4_tpm(cell))


def r4_items(cell):
    return recipe_items(cell)


def r4_plate_pos(cell, k):
    """The moment the plate holds k items (k = a plate_floor), on this cell's moment table."""
    T = moment_table(cell)
    items = r4_items(cell)
    if k <= 0:
        # a plate_floor of 0 asks for nothing: the forced block fetches the plate itself, so the mistake's moment
        # is the engine's "the dish is under way" guard (15 ticks, or the first prep / the first plate item)
        return 0.3
    if k >= len(items):
        return T["serve"]
    x = items[k]
    if x == items[k - 1]:                      # a batch item (the second potato): half-way through its platings
        return (T.get(f"next({x})", T["serve"]) + T.get(f"plated({x})", T["serve"])) / 2.0
    return T.get(f"next({x})", T["serve"])


def r4_prep_pos(cell, pf):
    """The moment a prep floor {item: n} is met."""
    T = moment_table(cell)
    out = 0.0
    for x in pf or {}:
        out = max(out, T.get(f"chop({x})", T.get(f"load({x})", 0.0)) + 0.2)
    return out


def r4_first_idx(cell, x):
    items = r4_items(cell)
    return items.index(x) if x in items else 0


def r4_last_pos(cell, x):
    """The plate position right after every unit of x is down (a premature serve's floor)."""
    items = r4_items(cell)
    return len(items) - items[::-1].index(x) if x in items else 0


def r4_processed_at(cell, x):
    """The moment the first unit of x becomes ready (chopped / cooked) — after this a serve_raw of it is dead."""
    T = moment_table(cell)
    w = want_of(cell).get(x)
    if w == "chopped":
        return T.get(f"chop({x})", 1.0)
    if w == "cooked":
        return T.get(f"done({x})", 4.0)
    return 0.0


def r4_safe_prep_repeat(cell, x):
    """B5: may prep(x) be repeated? Only when its node neither lights a station nor needs a busy one.

    R5-RP: "lights a station" was too narrow. `prep(potato)` on Fish & Chips and Steak Frites, `prep(rice)` and
    `prep(steak)` on the burrito END by putting the unit into a fryer / pot / pan that `ignite` has already lit,
    so a repeat leaves a SECOND unit cooking that no sub-case names: it reaches done, smokes, and in the scripted
    verifier it burned the kitchen twice (`b3_master_nops_hard_map_5`, steak cooked t191, smoke t252, fire t311,
    with the 48-tick rice repeat inside that window) and smoked inside seven other takeovers. Only a chop-only
    prep may be repeated."""
    from cooksim.core.plan_ir import segment_plan
    node = next((n for n in segment_plan(recipe_of(cell)) if n.sid == f"prep({x})"), None)
    if node is None:
        return False
    cmds = list(node.commands)
    if "turn on" in cmds:
        return False
    return True


def r5_prep_repeat_heats(cell, x):
    """Does prep(x) END by putting the unit on a heat station? Then a repeat leaves a duplicate cooking there."""
    from cooksim.core.plan_ir import segment_plan
    node = next((n for n in segment_plan(recipe_of(cell)) if n.sid == f"prep({x})"), None)
    if node is None:
        return False
    return any(w in str(c) for c in list(node.commands) for w in (" fryer", " pot", " pan", " oven"))


# ------------------------------------------------------------------------------- round 4: the candidate table
def r4_off_sources(cell):
    """The off-recipe crates of this cell's kitchen (D3: every kitchen has two)."""
    from cooksim_bench import exp_interact as X
    _st, ro = X.build_episode(None, cell=cell, inject=not cell.endswith("hard_map"))
    items = set(r4_items(cell))
    return sorted({str(t.value)[:-7] for row in ro.game.layout.grid for t in row
                   if str(t.value).endswith("_source")} - items - {"plate"})


def r4_mis_prep_items(cell):
    """B/C9/D7/D17: the components whose prep a mis_prep may replace on this recipe."""
    tier, items, want = tier_of(cell), r4_items(cell), want_of(cell)
    out = []
    for x in dict.fromkeys(items):
        if want[x] == "chopped" or x in ("steak", "potato"):
            if tier == "master" and x == "steak":
                continue                      # D17: the burrito's rice window
            if ROUND5 and tier == "master":
                # F4 / LC (round 3, dropped by the round-4 candidate table, re-measured in round 5): LLM cooks
                # chop the burrito's tomato and lettuce LATE and mid-plating. `mis_prep{tomato}` fired 2 of 6
                # real-cook runs (`the cook is not reaching for the tomato yet`), and `mis_prep{lettuce}`, whose
                # moment is 22.5 of 25, puts a 38-tick block in the window where the rice and the steak are
                # already done: it is the block inside two of the six burns. The burrito carries no prep slip.
                continue
            out.append(x)
    return out


def r4_cand(fn, kwargs, pos, *, prep_floor=None, plate_floor=None, end=None, brief=True, wrong=False,
            ruinous=False, robust=False, why="", err=None, lu_late=None, loops=None):
    if loops is not None:
        kwargs = dict(kwargs, loops=loops)
    return {"fn": fn, "err": err or {"misidentify_assemble": "add_foreign", "premature_serve": "omit_subgoal"}.get(fn, fn),
            "kwargs": dict(kwargs), "needs_wrong": wrong,
            "prep_floor": dict(prep_floor or {}), "plate_floor": (None if plate_floor is None else int(plate_floor)),
            "pos": float(pos), "end": float(end if end is not None else pos), "brief": bool(brief),
            "ruinous": bool(ruinous), "robust": bool(robust), "why": why, "lu_late": lu_late}


def r4_candidates(cell):
    """Every mistake round 4 will consider on this cell, with the floor that places it and the moment that floor
    is reached. One entry per (function, placement); the foreign item is assigned by the selector."""
    tier, items, want = tier_of(cell), r4_items(cell), want_of(cell)
    T = moment_table(cell)
    uniq = list(dict.fromkeys(items))
    cooked = [x for x in uniq if want[x] == "cooked"]
    last = items[-1]
    n_items = len(items)
    five = n_items >= 5
    out = []

    # ---- mis_prep: unchanged trigger (the cook reaching for the component). No floor: it is a prep-phase slip.
    for x in r4_mis_prep_items(cell):
        if x == items[0] and want[x] in ("cooked", "chopped"):
            p = 0.0
        elif tier in ("hard", "expert") and x == "potato" and not ROUND5:
            p = T[f"done({items[0]})"] - 0.5
        else:
            p = T.get(f"chop({x})", T.get(f"load({x})", 1.0)) - 0.5
        out.append(r4_cand("mis_prep", {"component": x}, p, wrong=True, why=f"preps the wrong thing for the {x}"))

    # ---- forget_next_step: a stall with nothing that can smoke through it (engine E4 decides; the moment table
    # models the first such moment). B6: after the second-to-last item when the last one is raw and ready.
    for a in r4_forget_anchors(cell):
        p = r4_forget_pos(cell, a)
        out.append(r4_cand("forget_next_step", {"after": a}, p, brief=False,
                           why=f"stalls after {a}"))

    # ---- leave_unattended: near-done window behind an explicit floor (E9 / D17)
    for x in cooked:
        if tier == "master" and x == "rice" and not R5_BURRITO_RICE_LU:
            continue                                   # D17
        if ROUND5 and tier == "master" and x != "rice":
            # R5-LU: by the time the rice is plated (its floor, E9) the steak is off the heat: 0/6 in the round-5
            # real-cook run, 0/8 in the round-2 pilot. The burrito carries no walk-away.
            continue
        pf, plf = lu_floor(cell, x)
        if plf:
            p = max(T[f"done({x})"], r4_plate_pos(cell, plf) - 0.4)
        else:
            p = T[f"done({x})"]
        lu_late = 23 if (tier == "master" and x == "steak") else None
        out.append(r4_cand("leave_unattended", {"component": x}, p, loops=2, prep_floor=pf, plate_floor=plf,
                           brief=False, lu_late=lu_late, why=f"walks away from the {x} near done"))

    # ---- the FORCED plating family. Each gets plate_floor = the slot's own index (B3) so it fires mid-dish in
    # any cooking order, and a prep floor that names only prep an LLM cook does early (V2).
    def pf_for(k, x=None):
        """The prep floor written beside a plate floor: empty at the start of the dish, else the early prep of
        this recipe that the cook must have done to plate k items."""
        pf = {}
        for y in items[:k]:
            if want[y] != "raw" and y in EARLY_PREP[tier]:
                pf[y] = max(pf.get(y, 0), 1)
        if x is not None and x in EARLY_PREP[tier] and want.get(x) != "raw" and k == 0:
            pass
        return pf

    for x in uniq:
        k = r4_first_idx(cell, x)
        # two placements per open slot where they differ: at the floor the slot opens ("early"), and behind the
        # component's own prep ("once it is prepped") — the cook has just made the thing and the block puts the
        # wrong one down anyway. Spreading the forced mistakes over the episode needs both.
        # S4 (self-review, master_hard_5: serve_raw(lettuce) and the last-item foreign items were skipped with
        # "the plate is already complete"): on a long recipe the cook plates in bursts 10-17 ticks apart, so a
        # floor set at the slot's own index is reached as the slot closes. Give it a plating step of room.
        if n_items >= 4:
            k = max(0, k - 1)
        floors = [{}]
        if want[x] != "raw" and f"{'chop' if want[x] == 'chopped' else 'load'}({x})" in T:
            floors.append({x: 1})
            if x == items[0]:
                # B5b: a mistake in the FIRST component's slot fired before the cook has prepped anything leaves
                # the cook's own plan stranded — it still walks to the board for a chopped tomato that was never
                # chopped and the scoop hangs 60 ticks (audit `scoop_hang`, easy_medium_3/4). Wait for the prep.
                floors = [{x: 1}]
        for pf_own in floors:
            pf = dict(pf_for(k)); pf.update(pf_own)
            p = max(r4_plate_pos(cell, k), r4_prep_pos(cell, pf))
            tag = " once it is prepped" if pf_own else ""
            if not (five and x != last):                       # B4 / CAP
                # D11 (round 3) banned a foreign item at the fish / steak because the smoke ban held it while that
                # food smoked. With the forced trigger the mistake no longer waits for the cook to walk to the
                # plate: at a floor well before that food is done nothing is smoking yet, so the ban now applies
                # only to a placement inside the last part of its cook.
                _d11 = (tier in ("hard", "expert") and x == items[0]
                        and not (k == 0 and p <= T.get(f"done({items[0]})", 99) - 0.8))
                if not _d11:
                    if not (want[x] == "raw" and x == items[0]):            # D5
                        out.append(r4_cand("add_foreign", {"before": f"assemble({x})"}, p, prep_floor=pf,
                                           plate_floor=k, wrong=True, ruinous=True, robust=True,
                                           why=f"an extra off-recipe item goes on before the {x}{tag}"))
                    out.append(r4_cand("misidentify_assemble", {"component": x}, p, prep_floor=pf,
                                       plate_floor=k, wrong=True, ruinous=True, robust=True,
                                       why=f"an off-recipe item goes into the {x}'s slot{tag}"))
        # serve_raw: the slot filled with a raw unit while nothing of x is processed. Its floor may never wait for
        # x's own prep (that is the thing it has to beat), so it is the plate floor and, for a cooked item, "x in
        # processing" — the pan is visibly still going while the raw one is scooped from the crate.
        if want[x] != "raw" and not (five and x not in (last, items[-2])):
            proc = r4_processed_at(cell, x)
            pf_ = {x: 1} if want[x] == "cooked" else {}
            k_ = k if five else 0                       # (k already carries the S4 step back)
            if ROUND5 and recipe_of(cell).ordered:
                # H9: every recipe here is ordered, so a raw unit dropped into the plate BEFORE the items that
                # come before it commits a wrong_order that no sub-case names. Wait for its own slot.
                k_ = max(k_, r4_first_idx(cell, x))
                if k_ >= n_items - 1:
                    # ... and the last slot is not a placement: "everything else plated" is one scoop from a
                    # finished dish, and the cook takes it (`serve_raw{lettuce}` on the burrito, 8 of 9 scripted
                    # runs archived `the cook's own serve is already queued`). Raw serves live in earlier slots.
                    continue
            p_raw = max(r4_plate_pos(cell, k_), r4_prep_pos(cell, pf_))
            # R5: the margin is a stretch of TIME, and the five moment tables are on different scales (the
            # burrito's is tick/10, the others ~25 ticks a moment — the same conversion R4_TICKS_PER_MOMENT makes
            # for the voice). 0.6 of an expert moment is 16 ticks, which refused every second-stage raw serve.
            _margin = min(R4_RAW_MARGIN, R5_RAW_MARGIN_TICKS / float(r4_tpm(cell))) if ROUND5 else R4_RAW_MARGIN
            if proc - p_raw >= _margin:
                out.append(r4_cand("serve_raw", {"component": x}, p_raw, prep_floor=pf_,
                                   plate_floor=k_, ruinous=True, robust=True,
                                   why=f"a raw {x} goes on the plate while no {x} is {want[x]} yet"))
    # add_foreign at the serve: the extra goes on a dish the cook thinks is done (no room on the 5-slot burrito)
    if not five:
        out.append(r4_cand("add_foreign", {"before": "serve"}, T["serve"], prep_floor={}, plate_floor=n_items,
                           wrong=True, brief=False, ruinous=True, robust=True,
                           why="an extra off-recipe item goes on the finished plate"))

    # ---- omit_subgoal: skip x, plate the next item (B7: the next item may be cooked if the floor is past its done)
    for x in uniq[:-1]:
        k = r4_first_idx(cell, x)
        nxt = items[items.index(x) + 1]
        if nxt == x:
            continue                                   # a batch item: "the next one" is the same food
        if want[nxt] == "chopped":
            continue                                   # B7: a chopped next item is not ready when the slot opens
        p = max(r4_plate_pos(cell, k), T.get(f"done({nxt})", 0.0) if want[nxt] == "cooked" else 0.0)
        if p >= T["serve"] - 0.5:
            continue
        if want[nxt] == "cooked" and p > T.get(f"plated({x})", T["serve"]) - 1.0:
            # S5 (self-review, master_medium_1): omit(rice) waits for a cooked steak, which the LLM cook reaches
            # at the same moment it plates the rice — the omission was skipped with "the plate is already
            # complete". The wait for the next item must end a full plating step before this slot closes.
            continue
        _own = ([{x: 1}] if x == items[0] else [{}, {x: 1}]) \
            if (want[x] != "raw" and f"{'chop' if want[x] == 'chopped' else 'load'}({x})" in T) else [{}]
        for pf_own in _own:
            pf = dict(pf_for(k)); pf.update(pf_own)
            p2 = max(p, r4_prep_pos(cell, pf))
            out.append(r4_cand("omit_subgoal", {"sid": f"assemble({x})"}, p2, prep_floor=pf, plate_floor=k,
                               ruinous=True, robust=True,
                               why=f"skips the {x}{' although it is ready' if pf_own else ''} and plates the {nxt} instead"))

    # ---- premature_serve: the plate goes to the pass with the dish short
    for x in uniq:
        pos_n = r4_last_pos(cell, x)
        if pos_n >= n_items:
            continue                                   # serving a complete plate is not premature
        p = r4_plate_pos(cell, pos_n)
        out.append(r4_cand("premature_serve", {"after": f"assemble({x})"}, p, prep_floor=pf_for(pos_n),
                           plate_floor=pos_n, brief=False, ruinous=True, robust=True,
                           why=f"carries the plate to the pass with only the {x} (and what came before) on it"))

    # ---- wrong_order: b goes on before a (B8: b raw, or a chop the block makes itself, never the last item)
    for i, a in enumerate(items[:-1]):
        for b in uniq[i + 1:]:
            if b == a or want[b] == "cooked":
                continue
            if b == last and want[b] != "raw":
                continue                               # WO
            if want[b] == "chopped" and five and items.index(b) - items.index(a) > 2:
                continue
            # B3b (round-4 real-cook check): a swap placed at plate_floor = idx(a) is reached only as the cook
            # plates a — the burrito cook puts the tortilla, the rice and the steak down 10-17 ticks apart, so the
            # swap lost the race 15 times in 24 runs ("the cook's own serve is already queued"). Its block plates
            # b and then a itself, so firing a plating step EARLIER costs nothing and is if anything the more
            # natural reading of "the cook thinks the b comes first".
            k = max(0, r4_first_idx(cell, a) - 1)
            p = r4_plate_pos(cell, k)
            end = max(p, T.get(f"plated({b})", p), T.get(f"plated({a})", p)) if items.index(b) == i + 1 else p + 1.0
            _own = ([{a: 1}] if a == items[0] else [{}, {a: 1}]) \
                if (want[a] != "raw" and f"{'chop' if want[a] == 'chopped' else 'load'}({a})" in T) else [{}]
            for pf_own in _own:
                pf = dict(pf_for(k)); pf.update(pf_own)
                p2 = max(p, r4_prep_pos(cell, pf))
                out.append(r4_cand("wrong_order", {"a": a, "b": b}, p2, prep_floor=pf, plate_floor=k,
                                   end=min(max(end, p2), p2 + 2.5), ruinous=True, robust=True,
                                   why=f"plates the {b} before the {a}"))

    # ---- repeat_step: a duplicate that READS as one. Plating duplicates only where a slot is free (B4);
    # prep duplicates only where the prep node lights nothing (B5).
    for i, c in enumerate(items[:-1]):
        if c in getattr(__import__("cooksim.core.error_lib", fromlist=["x"]), "REPEAT_DUPLICATE_INVISIBLE", set()):
            continue
        if want[c] == "cooked" or five:
            continue
        k = r4_first_idx(cell, c) + 1
        p = r4_plate_pos(cell, k)
        out.append(r4_cand("repeat_step", {"sid": f"assemble({c})", "at": f"assemble({c})"}, p,
                           prep_floor=pf_for(k), plate_floor=k, brief=False,
                           why=f"puts a second {c} on the plate"))
    if items.count(last) == 1 and want[last] == "raw" and last not in getattr(
            __import__("cooksim.core.error_lib", fromlist=["x"]), "REPEAT_DUPLICATE_INVISIBLE", set()) and not five:
        out.append(r4_cand("repeat_step", {"sid": f"assemble({last})", "at": f"assemble({last})"}, T["serve"],
                           prep_floor=pf_for(n_items), plate_floor=n_items, brief=False,
                           why=f"puts a second {last} on the finished plate"))
    for c in uniq:
        if not r4_safe_prep_repeat(cell, c):
            continue
        k = min(r4_first_idx(cell, c) + 1, n_items - 1) if c in items else 1
        anchor = items[max(0, k - 1)]
        p = max(r4_plate_pos(cell, k), r4_prep_pos(cell, {c: 1}) + 0.5)
        if p >= T["serve"] - 0.5:
            continue
        if ROUND5 and r5_prep_repeat_heats(cell, c) and (T["serve"] - p) * r4_tpm(cell) > R5_DUP_TICKS:
            # R5-RP: `prep(potato)` on Fish & Chips and Steak Frites, `prep(rice)` and `prep(steak)` on the burrito
            # END by putting the unit into a fryer / pot / pan that `ignite` has already lit, so a repeat leaves a
            # SECOND unit cooking that no sub-case names. It is only invisible when the dish goes out before that
            # unit can smoke: cook (~13 ticks) plus the smoke grace (25). Every smoke inside a takeover in the
            # scripted run was one of these (7 of 7: a second steak or fish in the pan), and the 48-tick rice
            # repeat of `b3_master_nops_hard_map_5` burned the kitchen twice.
            continue
        out.append(r4_cand("repeat_step", {"sid": f"prep({c})", "at": f"assemble({anchor})"}, p,
                           prep_floor={c: 1}, plate_floor=k, brief=False,
                           why=f"does the {c} prep a second time"))
    # B3: a mistake that fills or takes over a SLOT dies the moment the cook plates that item correctly, so its
    # floor has to be reached well before then. This is what keeps the omission off Fish & Chips and Steak Frites
    # (it waits for a cooked potato, which comes after the fish / steak is already down) and off the burger's bun.
    keep = []
    for c in out:
        x = r4_slot_item(c, cell)
        if x is not None:
            close = T.get(f"plated({x})", T["serve"])
            if c["pos"] > close - 0.3:
                continue
        keep.append(c)
    return keep


def r4_slot_item(c, cell):
    """The recipe item whose OPEN slot a candidate needs (None when it needs none)."""
    fn, kw = c["fn"], c["kwargs"]
    if fn in ("misidentify_assemble", "serve_raw"):
        return kw.get("component")
    if fn == "add_foreign":
        return None if kw.get("before") == "serve" else comp_of_sid(kw["before"])
    if fn == "omit_subgoal":
        return comp_of_sid(kw.get("sid"))
    if fn == "wrong_order":
        return kw.get("a")
    return None


def r4_forget_anchors(cell):
    """The forget anchors round 4 allows on this recipe (see B6 and the round-2 F rule)."""
    tier, items, want = tier_of(cell), r4_items(cell), want_of(cell)
    T = moment_table(cell)
    uniq = list(dict.fromkeys(items))
    out = []
    for x in uniq:
        if want[x] == "chopped" or (x in ("steak", "potato") and want[x] == "cooked"):
            if not (tier == "master" and x == "steak"):
                out.append(f"prep({x})")
            if tier in ("hard", "expert") and x == "potato":
                out.append("prep(potato)#2")
    if tier != "medium":
        out.append("fetch_plate")
    for i, x in enumerate(items):
        if i >= len(items) - 1 or x == items[i + 1]:
            continue
        if i == len(items) - 2 and not (want[items[-1]] == "raw" or (want[items[-1]] == "chopped" and len(items) >= 5)):
            continue                                   # R9 / B6
        nxt = items[i + 1]
        if want[nxt] == "cooked":
            continue                                   # the stall would sit beside food on the heat
        out.append(f"assemble({x})")
    seen, res = set(), []
    for a in out:
        if a in seen:
            continue
        seen.add(a)
        res.append(a)
    return res


def r4_forget_pos(cell, a):
    """The moment a forget after `a` can start (the engine holds it until nothing can smoke inside the freeze)."""
    tier, items = tier_of(cell), r4_items(cell)
    T = moment_table(cell)
    x = comp_of_sid(a)
    if a == "fetch_plate":
        return T.get("plate0", 2.0)
    if a.startswith("assemble("):
        return T.get(f"plated({x})", T["serve"])
    if x == "potato" and tier in ("hard", "expert"):
        return T[f"plated({items[0]})"] - 0.6
    p = T.get(f"chop({x})", T.get(f"load({x})", 1.0))
    if tier == "medium":
        p = max(p, T.get("plated(meat)", p))
    return p


# ------------------------------------------------------------------------------- round 4: set rules + selector
R4_MIN_GAP = 1.0            # a brief moment needs a full plating step after the previous takeover's end
R4_STATE_GAP = 0.5          # a state moment (a stall, a finished plate, a serve) is only delayed, never lost
R4_PLATE_STEP_GAP = 1.5     # D9: after a PLATING takeover the cook plates the next item inside the gap G
R4_LU_GAP = 1.3


def r4_is_plating(c):
    return c["fn"] in ("misidentify_assemble", "serve_raw", "omit_subgoal", "wrong_order", "repeat_step") or (
        c["fn"] == "add_foreign" and c["kwargs"].get("before") != "serve")


def r4_is_state(c):
    return not c["brief"]


def r4_slots(c, cell):
    items = r4_items(cell)
    fn, kw = c["fn"], c["kwargs"]
    if fn in ("mis_prep", "serve_raw", "misidentify_assemble", "leave_unattended"):
        return {kw["component"]}
    if fn == "add_foreign":
        return {"serve"} if kw.get("before") == "serve" else {comp_of_sid(kw["before"])}
    if fn == "wrong_order":
        return {kw["a"], kw["b"]}
    if fn == "omit_subgoal":
        x = comp_of_sid(kw["sid"])
        i = items.index(x)
        return {x} | ({items[i + 1]} if i + 1 < len(items) else {"serve"})
    if fn == "premature_serve":
        return {"serve"}
    if fn == "repeat_step":
        if ROUND5 and str(kw.get("sid", "")).startswith("prep("):
            # B5's own wording: a PREP repeat costs time and no plate slot. C1 is a rule about slots, so a prep
            # repeat of the walk-away's food (chop a second steak while the first sits in the pan) is not two
            # mistakes in one slot — it is two mistakes at two stations. Round 4 counted it as a collision and
            # took four of the expert cell's six four-mistake sets with it.
            return set()
        return {comp_of_sid(kw.get("sid"))}
    return set()


def r4_after_item(c):
    fn, kw = c["fn"], c["kwargs"]
    if fn in ("premature_serve", "forget_next_step") and str(kw.get("after", "")).startswith("assemble("):
        return comp_of_sid(kw["after"])
    if fn == "repeat_step":
        return comp_of_sid(kw.get("at"))
    return None


def r4_before_item(c):
    fn, kw = c["fn"], c["kwargs"]
    if fn == "add_foreign" and kw.get("before") != "serve":
        return comp_of_sid(kw["before"])
    if fn in ("misidentify_assemble", "serve_raw"):
        return kw.get("component")
    if fn == "wrong_order":
        return kw.get("a")
    if fn == "omit_subgoal":
        return comp_of_sid(kw.get("sid"))
    return None


def r4_leaves_off_plate(c):
    """A mistake after which a recipe item is still missing from the plate: every later plating moment waits for
    the cook to put it there, so it has to be the case's LAST plating moment (rule B of round 2)."""
    return c["fn"] in ("misidentify_assemble", "serve_raw", "omit_subgoal")


def r4_set_violation(rows, cell):
    """None if this set of round-4 candidates is admissible together, else the first rule it breaks."""
    items, want = r4_items(cell), want_of(cell)
    T = moment_table(cell)
    tier = tier_of(cell)
    labels = [r["err"] for r in rows]
    if len(set(labels)) < len(labels):
        return "L one error label per case"
    # C1: one mistake per plate slot (a mis_prep acts in the prep phase and may share with a later plating one)
    seen = set()
    for r in rows:
        if r["fn"] == "mis_prep":
            continue
        s = r4_slots(r, cell)
        if s & seen:
            return f"C1 slot {sorted(s & seen)} targeted twice"
        seen |= s
    if sum(1 for r in rows if r["fn"] == "mis_prep") > 1:
        return "C1 two mis_preps"
    for r in rows:
        if r["fn"] != "mis_prep":
            continue
        c_ = r["kwargs"]["component"]
        if any(x is not r and x["fn"] == "serve_raw" and x["kwargs"].get("component") == c_ for x in rows):
            return f"C1 mis_prep and a raw serve of the {c_}"
    # M / D9: spacing at the LLM cook's pace, measured from the previous takeover's END
    sp = sorted(rows, key=lambda r: (r["pos"], r["fn"]))
    for a, b in zip(sp, sp[1:]):
        need = R4_STATE_GAP if r4_is_state(b) else R4_MIN_GAP
        if b["fn"] == "premature_serve":
            # S3 (self-review, hard_hard_1): a premature serve queues the serve, which blocks every other
            # mistake's own trigger from that tick on. A forget 0.6 of a moment before it was skipped with "the
            # only step left is the serve". It needs the full plating step, not the state gap.
            need = max(need, R4_MIN_GAP)
        if r4_is_plating(a):
            need = max(need, R4_PLATE_STEP_GAP if b["brief"] else R4_MIN_GAP)
        if b["pos"] - a["end"] < need - 1e-6:
            return (f"M {b['fn']}@{b['pos']} comes {b['pos'] - a['end']:.2f} after the {a['fn']} takeover's end "
                    f"@{a['end']} (needs {need})")
    # CAP: the served dish plus every EXTRA unit must fit the 5-slot plate.
    def adds_unit(r):
        """Does this mistake leave a unit that has to coexist with the whole recipe? (B3: a wrong, raw or
        duplicate unit does NOT fill its slot, so the cook re-plates the right one behind it.) A mistake at the
        recipe's LAST item is terminal instead — nothing has to land behind it."""
        fn, kw, last_ = r["fn"], r["kwargs"], items[-1]
        if fn == "add_foreign":
            return True if kw.get("before") == "serve" else comp_of_sid(kw["before"]) != last_
        if fn in ("misidentify_assemble", "serve_raw"):
            return kw["component"] != last_
        if fn == "repeat_step" and str(kw.get("sid", "")).startswith("assemble("):
            return comp_of_sid(kw["sid"]) != last_
        return False
    if len(items) + sum(1 for r in rows if adds_unit(r)) > 5:
        return "CAP the recipe and the extra items exceed the plate's 5 slots"
    # C2 / C2+: a premature serve ends the episode — it is the last moment and nothing later than its item
    prem = [r for r in rows if r["fn"] == "premature_serve"]
    if len(prem) > 1:
        return "C2 two premature serves"
    if prem:
        P = prem[0]
        ix = items.index(comp_of_sid(P["kwargs"]["after"]))
        for r in rows:
            if r is P:
                continue
            if r["pos"] >= P["pos"]:
                return "C2 a premature serve must be the last moment"
            if r["fn"] == "add_foreign" and r["kwargs"].get("before") == "serve":
                return "C2 an add_foreign at the serve behind a premature serve"
            # (round 2's "nothing about a LATER item" is gone: with explicit floors the order of the moments is
            # the order of the floors, which the "last moment" rule and the M gaps already fix. A mistake about a
            # later item that fires at a LOWER floor — the raw potato scooped before the fish is plated — is
            # reached first in every cooking order, which is exactly what round 2 could not express.)
            if (r["plate_floor"] is not None and P["plate_floor"] is not None
                    and r["plate_floor"] >= P["plate_floor"] and r["fn"] != "mis_prep"):
                return f"C2 {r['fn']} shares the premature serve's plate floor ({P['plate_floor']})"
    # (round 2's rule B — "a mistake that leaves item X off the plate must be the case's last plating moment,
    # because every later plating moment waits for the cook to re-plate X" — is retired in round 4. It described
    # the pre-forcing triggers, which asked for the cook's OWN next plating; a forced mistake has a plate FLOOR
    # that counts the units ever on the plate, right or wrong, so a wrong or raw unit carries the floor rather
    # than blocking it. C1 still keeps two mistakes off one slot.)
    # D9 / D13: "before X" with "after X", and "after X" with "before next(X)", share one anchor
    for r1 in rows:
        a = r4_after_item(r1)
        if a is None or a not in items:
            continue
        nxt = items[items.index(a) + 1] if items.index(a) + 1 < len(items) else None
        for r2 in rows:
            if r2 is r1 or r2["fn"] == "wrong_order":
                continue
            b = r4_before_item(r2)
            # (gated on the two moments really being one plating step apart: with explicit floors a "before the
            # meat" mistake may sit at plate floor 0 and an "after the meat" one at plate floor 2 — two anchors a
            # whole cook apart, which round 3 could not tell from the collision it was banning)
            if abs(r2["pos"] - r1["pos"]) >= R4_PLATE_STEP_GAP - 1e-6:
                continue
            if b is not None and b == a:
                return f"D9 {r2['fn']} before the {a} and {r1['fn']} after it (the cook plates the {a} inside G)"
            if b is not None and nxt is not None and b == nxt:
                return f"D13 {r1['fn']} after the {a} and {r2['fn']} at the {nxt} share one anchor"
    # S6 (self-review, easy_medium_1: the forget after the tomato and the question both died at the ceiling
    # 18 ticks after the omission released): on a TWO-item recipe a mistake that puts an item on the plate
    # leaves the dish one scoop from complete, and any mistake that waits for the plate to be through an item
    # can no longer land. The moment gap cannot express this — the whole recipe is four moments long.
    if len(items) <= 2:
        for a_ in rows:
            if not r4_is_plating(a_):
                continue
            for b_ in rows:
                if b_ is not a_ and b_["pos"] > a_["pos"] and r4_after_item(b_) is not None:
                    return (f"S6 on a two-item dish the {b_['fn']} after the {r4_after_item(b_)} meets a complete "
                            f"plate one scoop after the {a_['fn']}")
    # SR+ (round 2, generalised in round 4): a mistake's own floor may not wait for the PREP of a component that
    # an earlier mistake of the same case fills with a raw or foreign unit — the cook may then never process it
    # (real-cook check: six easy sub-cases waited for "tomato: 1" all episode, live 0, behind a serve_raw of it)
    for r in rows:
        for x in (r["prep_floor"] or {}):
            for b in rows:
                if b is r or b["pos"] >= r["pos"]:
                    continue
                if b["fn"] in ("serve_raw", "misidentify_assemble") and b["kwargs"].get("component") == x:
                    return f"SR+ the {r['fn']} waits for a {x} the earlier {b['fn']} replaces"
                if b["fn"] == "omit_subgoal" and comp_of_sid(b["kwargs"].get("sid")) == x:
                    return f"SR+ the {r['fn']} waits for a {x} the earlier omission skips"
    # WO (round 2, kept): when the swapped-in b is the recipe's LAST item, "the plate through items[-2]" is a
    # COMPLETE plate once the swap has run — a forget, a premature serve or a repeat that waits for it then meets
    # a finished dish (the engine refuses the forget: audit `forget_complete`, medium_medium_2 plate_early t71).
    for W in [r for r in rows if r["fn"] == "wrong_order"]:
        b = W["kwargs"]["b"]
        if not (b == items[-1] and items.count(b) == 1):
            continue
        for r in rows:
            y = r4_after_item(r)
            if y in items and items.index(y) == len(items) - 2:
                return f"WO {r['fn']} waits for the plate through {y}, which the swap has already completed"
    # LU: the walk-away needs its near-done window and its rescue window free
    for L in [r for r in rows if r["fn"] == "leave_unattended"]:
        c_ = L["kwargs"]["component"]
        for r in rows:
            if r is L:
                continue
            if c_ in ({r["kwargs"].get(k) for k in ("component", "a", "b", "wrong")}
                      | {comp_of_sid(r["kwargs"].get(k)) for k in ("sid", "before", "at", "after")}):
                _prep_repeat = (ROUND5 and r["fn"] == "repeat_step"
                                and str(r["kwargs"].get("sid", "")).startswith("prep("))
                if r["fn"] not in ("forget_next_step", "mis_prep", "premature_serve") and not _prep_repeat:
                    return f"R1 the walk-away's {c_} is also the {r['fn']}'s item"
            if abs(r["pos"] - L["pos"]) < R4_LU_GAP - 1e-6:
                return f"LU a {r['fn']} takeover sits in the walk-away's near-done window"
            if f"next({c_})" in T and L["pos"] < r["pos"] <= T[f"next({c_})"] + 1e-6 and r["fn"] != "wrong_order":
                return f"LU a {r['fn']} takeover sits in the {c_}'s rescue window"
    # a plate-in-hand forget followed by an "after X" state moment inside one plating step (round-3 follow-up)
    for f_ in rows:
        if f_["fn"] == "forget_next_step" and (f_["kwargs"].get("after") == "fetch_plate"
                                               or str(f_["kwargs"].get("after", "")).startswith("assemble(")):
            for r_ in rows:
                if r_ is not f_ and r4_after_item(r_) is not None and 0 <= r_["pos"] - f_["pos"] < R4_PLATE_STEP_GAP - 1e-6:
                    return f"D9 {r_['fn']} after the {r4_after_item(r_)} within a plating step of a plate-in-hand forget"
    if ROUND5:
        why5 = r5_set_violation(rows, cell)
        if why5:
            return why5
    # T / D14: ruinous cover
    n_r = sum(1 for r in rows if r["ruinous"])
    if tier in ("easy", "medium"):
        if n_r < math.ceil(len(rows) / 2):
            return f"T fewer than {math.ceil(len(rows) / 2)} ruinous (easy/medium keep >= half)"
    else:
        if n_r < 1:
            return "T no ruinous mistake"
        if not any(r["robust"] and r["ruinous"] for r in rows):
            return "D14 no ROBUST ruinous mistake"
    return None


def r4_norm(rows):
    """A case's design with the details that do not change it removed (the foreign item, the loop count, which
    potato a forget follows, the floor variant): two cases equal under this are the same design."""
    out = []
    for r in rows:
        kw = {k: v for k, v in r["kwargs"].items() if k not in ("wrong", "loops")}
        if "after" in kw:
            kw["after"] = str(kw["after"]).split("#")[0]
        fn = r["fn"]
        if fn == "add_foreign" and str(kw.get("before", "")).startswith("assemble("):
            fn, kw = "misidentify_assemble", {"component": comp_of_sid(kw["before"])}
        out.append((fn, json.dumps(kw, sort_keys=True)))
    return tuple(sorted(out))


def r4_wrong_options(cell, fn, comp):
    """The off-recipe items this mistake may use in this kitchen (D3/D7/C9)."""
    from cooksim.core.enums import choppable
    items = r4_items(cell)
    out = []
    for w in r4_off_sources(cell):
        if w in BAD_FOREIGN or (ICON_TWINS.get(w, set()) & set(items)) or any(w in ICON_TWINS.get(i, set()) for i in items):
            continue
        if fn == "mis_prep":
            if not choppable(w) or w in heat_takes(cell):
                continue
        out.append(w)
    return out


def r4_enumerate(cell, n, rng):
    """Every admissible n-mistake set of this cell (the foreign items are assigned later)."""
    cands = r4_candidates(cell)
    # two candidates that are the same mistake at the same moment are one choice: keep the simpler floor
    best = {}
    for c in cands:
        k = (c["fn"], json.dumps(c["kwargs"], sort_keys=True), round(c["pos"], 2))
        if k not in best or len(best[k]["prep_floor"]) > len(c["prep_floor"]):
            best[k] = c
    cands = sorted(best.values(), key=lambda c: (c["pos"], c["fn"], json.dumps(c["kwargs"], sort_keys=True)))
    # a mistake that needs a foreign item is impossible where the kitchen has none of the right kind
    cands = [c for c in cands if not c["needs_wrong"] or r4_wrong_options(cell, c["fn"], None)]
    cands = [c for c in cands if not (c["fn"] == "mis_prep" and (not r4_wrong_options(cell, "mis_prep", None)
                                                                 or n_boards(cell) < 2))]
    out = []
    for combo in itertools.combinations(range(len(cands)), n):
        rows = [cands[i] for i in combo]
        if r4_set_violation(rows, cell):
            continue
        out.append(rows)
    return out


# ------------------------------------------------------------------------------- round 4: the quota optimiser
R4_TARGET_SHARE = 1.0 / len(R4_FNS)
R4_PATTERN_CAP = 3          # at most this many cases of a tier with the same multiset of functions
R4_OPENING_CAP = 3          # ... with the same opening mistake
R4_ROW_CAP = 6              # ... the same mistake at the same placement, suite-wide


def r4_row_key(r):
    kw = {k: v for k, v in r["kwargs"].items() if k not in ("wrong", "loops")}
    return (r["fn"], json.dumps(kw, sort_keys=True), r["plate_floor"], json.dumps(r["prep_floor"], sort_keys=True))


_R5_META = {}


def r5_meta(cell, si, rows):
    """Everything r4_cost needs about ONE candidate set, computed once. Round 5 added a per-tier pairwise term, a
    second normalisation per case and several descents: without this the cost evaluations live in json.dumps."""
    key = (cell, si)
    if key not in _R5_META:
        first = min(rows, key=lambda r: r["pos"])
        rowk = [r4_row_key(r) for r in rows]
        _R5_META[key] = {
            "fns": [r["fn"] for r in rows],
            "opening": (first["fn"], json.dumps({k: v for k, v in first["kwargs"].items() if k != "wrong"},
                                                sort_keys=True)),
            "rowk": rowk, "rowset": frozenset(rowk), "norm": r4_norm(rows),
            "ruin": sum(1 for r in rows if r["ruinous"]),
            "nondom": len(r5_nondominating(rows, cell)) if ROUND5 else 0,
        }
    return _R5_META[key]


R5_DESIGN_REPS = {}         # tier -> how many times one design may be reused (1 wherever the pool allows it)


def r4_cost(choice, pool, cases, target):
    """The suite-wide cost of one assignment (lower is better). Every term is a round-4 balance rule."""
    fn_n = collections.Counter()
    tier_fn = collections.Counter()
    pattern = collections.Counter()
    opening = collections.Counter()
    rowk = collections.Counter()
    tier_rowk = collections.Counter()
    tier_designs = collections.defaultdict(list)
    designs = collections.Counter()
    cell_designs = collections.Counter()
    ruin = 0
    nondom = 0
    for ci, si in enumerate(choice):
        c = cases[ci]
        tier = tier_of(c["cell"])
        m = r5_meta(c["cell"], si, pool[c["cell"]][si])
        fns = m["fns"]
        for f in fns:
            fn_n[f] += 1
        for f in set(fns):
            tier_fn[(tier, f)] += 1
        pattern[(tier, tuple(sorted(fns)))] += 1
        opening[(tier,) + m["opening"]] += 1
        for k_ in m["rowk"]:
            rowk[k_] += 1
            tier_rowk[(tier, k_)] += 1
        tier_designs[tier].append(m["rowset"])
        designs[(tier, m["norm"])] += 1
        cell_designs[(c["cell"], m["norm"])] += 1
        ruin += m["ruin"]
        nondom += m["nondom"]
    cost = 0.0
    for f in R4_FNS:
        cost += 9.0 * (fn_n[f] - target) ** 2
        if ROUND5:
            # B1's band made near-hard: 10-17 of the 140 (7-12%). Round 5's floors force every expert set to carry
            # the serve-time foreign item, so without this the suite drifts to add_foreign 22 / omit_subgoal 7.
            cost += 8000.0 * max(0, fn_n[f] - R5_TYPE_MAX) ** 2
            cost += 8000.0 * max(0, R5_TYPE_MIN - fn_n[f]) ** 2
    for (tier, f), n in tier_fn.items():
        cost += 30.0 * max(0, n - R4_TIER_CASE_CAP) ** 2
    for k, n in pattern.items():
        cost += 8.0 * max(0, n - R4_PATTERN_CAP) ** 2
    for k, n in opening.items():
        cost += 5.0 * max(0, n - R4_OPENING_CAP) ** 2
    for k, n in rowk.items():
        cost += 3.0 * max(0, n - R4_ROW_CAP) ** 2
    for k, n in designs.items():
        cost += 200.0 * max(0, n - 1) ** 2          # B1: no two cases with the same design
    if ROUND5:
        # B1, hardened. A design may be reused only as often as the tier's pool forces: R5_DESIGN_REPS is 1
        # wherever the tier has half again as many admissible designs as cases, 2 where it is tighter. The
        # penalty is a constraint, not a pull; a reuse may never be inside ONE kitchen, where the two episodes
        # would be the same thing twice, and across the two kitchens of a tier it is priced.
        for k, n in cell_designs.items():
            cost += 50000.0 * max(0, n - 1) ** 2
        for (t_, k), n in designs.items():
            cost += 100000.0 * max(0, n - R5_DESIGN_REPS.get(t_, 1)) ** 2
            cost += 3000.0 * max(0, n - 1) ** 2
    if ROUND5:
        # H2 (soft half): a non-terminal sub-case whose floor is not ordered with an earlier one's only swaps two
        # takeovers, but the swap costs the earlier one its moment when the clock is short. Prefer chains.
        cost += 4.0 * nondom
        # H8: the round-4 suite put four expert placements in 5 of 10 cases and let two cases of a tier differ in
        # a single mistake (expert medium_1/2: the same three, then forget(fetch_plate) vs forget(prep(potato))).
        for k, n in tier_rowk.items():
            # linear, not quadratic: on Steak Frites the serve-time foreign item is in every admissible set, so a
            # squared term there is a constant 7,200 that swamps every choice the assignment can actually make
            cost += 150.0 * max(0, n - R5_ROW_TIER_CAP)
        for lst in tier_designs.values():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if len(lst[i]) == len(lst[j]) and len(lst[i] & lst[j]) >= len(lst[i]) - 1:
                        cost += 60.0
    cost -= 0.5 * ruin
    # B10-B12: the room a mistake set leaves for the voice. Round 3 designed the mistakes first and then lost one
    # designed question in three to them; here the assignment itself pays for a case with no askable moment, and
    # for a suite that cannot host its 30 requests and 11 plan changes.
    nq = n_req = n_pc = 0
    rare = collections.Counter()
    pc_tier = collections.Counter()
    narrow = 0
    for ci, si in enumerate(choice):
        st = r4_stats(cases[ci]["cell"], si, pool[cases[ci]["cell"]][si])
        nq += st["nq"]
        n_req += int(st["req"])
        n_pc += int(st["pc"])
        narrow += max(0, 3 - len(st["fams"]))
        if st["pc"]:
            pc_tier[tier_of(cases[ci]["cell"])] += 1
        for f in st["fams"]:
            rare[f] += 1
    cost += 25.0 * (R4_Q_TARGET * len(cases) - nq)
    cost += 8.0 * max(0, R4_PROACTIVE_TARGET + 3 - n_req) ** 2
    cost += (150.0 if ROUND5 else 20.0) * max(0, R4_PLAN_CHANGES + 6 - n_pc) ** 2
    # B12: a family, not a leftover — every tier that CAN host an announced change (the burrito cannot: its five
    # components already fill the plate) keeps at least two cases able to take one
    for t_ in ("easy", "medium", "hard", "expert"):
        cost += 15.0 * max(0, 2 - pc_tier[t_]) ** 2
    for f in ("Q1", "Q2", "Q7"):
        cost += 2.0 * max(0, 12 - rare[f]) ** 2          # the families round 3 squeezed out
    # B10: a case whose takeovers leave only two askable families forces the suite onto the two kinds that are
    # always available (next step / how long has it been on) — the round-3 skew in miniature. Pay for it.
    cost += 6.0 * narrow
    for f in ("Q4", "Q5", "Q8"):
        cost += 1.0 * max(0, 30 - rare[f])
    return cost


def r4_assign(cases, rng, target=None):
    """Choose one admissible mistake set per case so the suite hits the per-type quota (B1). Greedy start, then
    a deterministic best-improvement local search over single-case swaps."""
    pool = {}
    for c in cases:
        cell = c["cell"]
        if cell not in pool:
            pool[cell] = r4_enumerate(cell, R4_TIER_N[tier_of(cell)], rng)
            print(f"[r4] {cell}: {len(pool[cell])} admissible sets", flush=True)
            if not pool[cell]:
                sys.exit(f"no admissible set for {cell}")
    if ROUND5:
        by_tier = collections.Counter(tier_of(c["cell"]) for c in cases)
        norms = collections.defaultdict(set)
        for c in cases:
            for rows in pool[c["cell"]]:
                norms[tier_of(c["cell"])].add(r4_norm(rows))
        R5_DESIGN_REPS.clear()
        for t_, n_ in by_tier.items():
            # one use each where the pool is half again the tier's case count; two where it is tighter — the
            # Caprese leaves 11 admissible two-mistake designs for 10 cases and Steak Frites 11 for 10, and with
            # no slack at all the per-type band cannot be met. Every reuse is still paid for.
            R5_DESIGN_REPS[t_] = max(1 if len(norms[t_]) >= 1.5 * n_ else 2, -(-n_ // max(1, len(norms[t_]))))
            print(f"[r4] {t_}: {len(norms[t_])} distinct designs for {n_} cases "
                  f"(a design may be used {R5_DESIGN_REPS[t_]}x)", flush=True)
    n_err = sum(R4_TIER_N[tier_of(c["cell"])] for c in cases)
    target = target if target is not None else n_err / float(len(R4_FNS))
    choice = [0] * len(cases)
    order = list(range(len(cases)))
    # greedy start
    for ci in order:
        best, best_c = 0, None
        for si in range(len(pool[cases[ci]["cell"]])):
            choice[ci] = si
            v = r4_cost(choice, pool, cases, target)
            if best_c is None or v < best_c:
                best, best_c = si, v
        choice[ci] = best
    def descend(ch):
        cur_ = r4_cost(ch, pool, cases, target)
        for _r in range(40):
            improved = False
            for ci in order:
                base_ = ch[ci]
                best_, bc_ = base_, cur_
                for si in range(len(pool[cases[ci]["cell"]])):
                    if si == base_:
                        continue
                    ch[ci] = si
                    v = r4_cost(ch, pool, cases, target)
                    if v < bc_ - 1e-9:
                        best_, bc_ = si, v
                ch[ci] = best_
                if best_ != base_:
                    cur_, improved = bc_, True
            if not improved:
                break
        return cur_
    cur = descend(choice)
    if ROUND5:
        # One greedy descent stops at 18 of a type where 17 is the ceiling: the band, the design-reuse limit and
        # the per-tier placement cap are coupled, and a best-improvement descent over single-case moves has plenty
        # of local minima. Seeded ITERATED local search — kick a few cases to random sets, descend again, keep the
        # better — reaches the band in about a minute. The rng is the run's, so the suite is reproducible.
        best_choice, best_cost = list(choice), cur
        for _k in range(R5_RESTARTS):
            ch = list(best_choice)
            for _ in range(rng.randint(2, 5)):
                ci = rng.randrange(len(cases))
                ch[ci] = rng.randrange(len(pool[cases[ci]["cell"]]))
            v = descend(ch)
            if v < best_cost - 1e-9:
                best_choice, best_cost = list(ch), v
        choice, cur = best_choice, best_cost
    print(f"[r4] assignment cost {cur:.1f}", flush=True)
    return [pool[cases[ci]["cell"]][choice[ci]] for ci in range(len(cases))], pool, choice


def r4_assign_wrongs(cases, sets, rng):
    """Give every foreign-item mistake its off-recipe ingredient: distinct inside the case, spread over the tier,
    never an icon twin or a non-food (D7), and for a mis_prep never something a heat station of the kitchen takes."""
    used = collections.Counter()
    out = []
    for c, rows in zip(cases, sets):
        cell = c["cell"]
        taken, new = set(), []
        for r in rows:
            r = dict(r, kwargs=dict(r["kwargs"]))
            if r["needs_wrong"]:
                opts = [w for w in r4_wrong_options(cell, r["fn"], r["kwargs"].get("component")) if w not in taken]
                if not opts:
                    sys.exit(f"{c['name']}: no off-recipe item left for {r['fn']}")
                w = min(opts, key=lambda w: (used[(cell, w)], used[w], w))
                r["kwargs"]["wrong"] = w
                taken.add(w)
                used[(cell, w)] += 1
                used[w] += 1
            new.append(r)
        out.append(new)
    return out


# ------------------------------------------------------------------------------- round 4: questions / requests
R4_FAMS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q7", "Q8", "Q9"]
# The moment tables are on different scales: the burrito's is tick/10 (measured), the other four are ordinal with
# roughly one plating step per unit. The engine's voice rules are in TICKS (TAKEOVER_GAP 10, VOICE_QUIET 3), so
# the designer's quiet windows are converted through the tier's own scale instead of a single number — round 3
# used 1.0 everywhere, which on the four-moment Caprese shut the voice for a quarter of the episode.
R4_TICKS_PER_MOMENT = {"easy": 28, "medium": 25, "hard": 26, "expert": 26, "master": 10}


def r4_tpm(cell):
    return R4_TICKS_PER_MOMENT[tier_of(cell)]


def r4_quiet_after(cell):
    return round(14.0 / r4_tpm(cell), 2)        # the release, the takeover gap G and a tick to speak


def r4_quiet_before(cell):
    return round(8.0 / r4_tpm(cell), 2)         # VOICE_QUIET plus the "imminent" hold


def r4_spans(rows, cell):
    """The moments the voice slot is shut, per mistake: its takeover, plus the window in which the engine already
    calls it imminent (a premature serve from the moment its item is next; the serve-time foreign item from the
    moment one item is left)."""
    T = moment_table(cell)
    out = []
    for r in rows:
        a0, e0 = r["pos"], r["end"]
        if r["fn"] == "premature_serve":
            a0 = min(a0, T.get(f"next({comp_of_sid(r['kwargs'].get('after'))})", a0))
        if r["fn"] == "add_foreign" and r["kwargs"].get("before") == "serve":
            a0 = min(a0, last_item_next(cell))
        if r["fn"] == "omit_subgoal":
            e0 = max(e0, a0 + 1.0)                 # its block plates the next item before it hands back
        if r["fn"] == "forget_next_step":
            e0 = max(e0, a0 + 0.5)
        out.append((a0, e0))
    return out


def r4_voice_block(c, rows, cell):
    """Why a question candidate is not used in this case (None = usable). The round-3 rules D10/D15 stay; B10
    adds the two families they had squeezed out, each with the condition that makes it non-trivial."""
    kind = c["kind"]
    base, _, arg = kind.partition(":")
    items = r4_items(cell)
    tier = tier_of(cell)
    touched = set()
    for r in rows:
        touched |= {x for x in r4_slots(r, cell) if x != "serve"}
        for k in ("component", "a", "b"):
            if r["kwargs"].get(k) in items:
                touched.add(r["kwargs"][k])
        for k in ("sid", "before", "at", "after"):
            x = comp_of_sid(r["kwargs"].get(k))
            if x in items:
                touched.add(x)
    if base in ("longest_on", "burn_first"):
        return "D15 one candidate in practice: it degenerates to 'what is on the heat'"
    if base in ("ready_in", "elapsed") and arg in items:
        why = r4_cooking_window_blocked(arg, rows, cell)
        if why:
            return "D10 " + why
    if base == "assistant_said":
        first = min([r["pos"] for r in rows] or [99])
        if len(D3.key_at(cell, c["m"])["plate"]) < 1:
            return "D15 nothing to recall yet (it fires in the prep phase)"
        if c["pos"] < first + R4_MIN_GAP:
            return "D15 nothing to recall yet (asked before the first mistake)"
    if kind.startswith("happened:visit(") or kind.startswith("count:visit("):
        w = arg[len("visit("):-1]
        cooked = [x for x in items if want_of(cell)[x] == "cooked"]
        used = {station_of_item(cell, x) for x in cooked}
        first_st = station_of_item(cell, cooked[0]) if cooked else None
        if w == "trash":
            commit = [r for r in rows if r["fn"] not in ("forget_next_step", "leave_unattended")]
            if not any(r["pos"] < c["pos"] for r in commit):
                return "D15 a trash visit with no mistake to throw away before it: the answer is 'no' from the recipe"
        elif base == "happened" and w not in used:
            return f"D15 the recipe never uses the {w}: the answer is 'no' from the recipe"
        elif base == "happened" and w == first_st:
            return f"D15 the {w} takes the first item lit: the answer is 'yes' from the recipe"
    if base == "order":
        # B10: kept when the LLM cooks really do the pair in both orders (measured, VARIABLE_ORDER), or when one
        # of its two sides is an item THIS case's mistakes move — a swap, an omission, a duplicate or a wrong
        # item changes the order the plate was really built in, and the truth is read off the world.
        sides = {comp_of_sid(x) for x in arg.split("|")}
        if kind not in VARIABLE_ORDER.get(tier, set()) and not (sides & touched):
            return "D8 the LLM cooks do this pair in one order only and no mistake of this case moves it"
    if ROUND5:
        # F1: B10 asks these families only where a mistake TOUCHES the referent. It must also refuse the case
        # where a mistake REMOVES it: an omission's item never reaches the plate and nothing after a premature
        # serve is ever plated, so the family's own precondition ("X on the plate with something after it",
        # "both events have happened") can never hold. All three `after_ref:tortilla` instances of the round-5
        # suite sat in cases whose omission skips the tortilla, and Q7 was asked 0 times in 6 real-cook runs.
        removed = set()
        for r in rows:
            if r["fn"] == "omit_subgoal":
                x_ = comp_of_sid(r["kwargs"].get("sid"))
                if x_ in items:
                    removed.add(x_)
            if r["fn"] == "premature_serve":
                a_ = comp_of_sid(r["kwargs"].get("after"))
                if a_ in items:
                    removed |= set(items[r4_last_pos(cell, a_):])
        named = set()
        if base == "after_ref":
            named.add(arg)
        elif base == "order":
            named |= {comp_of_sid(x_) for x_ in arg.split("|") if str(x_).startswith("assemble(")}
        elif base in ("ready_in", "elapsed", "missing"):
            named |= q_items(kind)
        bad = named & removed
        if bad:
            return f"F1 a mistake of this case removes the {sorted(bad)[0]} from the plate: the referent never happens"
    if base == "after_ref":
        # B10: what made it recipe-predictable was asking it where nothing had touched the plate. Asked in a case
        # whose mistakes reorder, replace, duplicate or skip a plating, the answer is a fact about THIS episode.
        nxt = items[items.index(arg) + 1] if arg in items and items.index(arg) + 1 < len(items) else None
        if not ({arg, nxt} & touched):
            return "D8 no mistake of this case touches the plating after the referent: the recipe answers it"
    return None


def r4_cooking_window_blocked(x, rows, cell):
    T = moment_table(cell)
    if want_of(cell).get(x) != "cooked" or f"load({x})" not in T:
        return None
    lo, hi = T[f"load({x})"], T.get(f"done({x})", T[f"load({x})"] + 2.0)
    for r in rows:
        if r["fn"] == "leave_unattended" and r["kwargs"].get("component") == x:
            continue
        p = r["pos"]
        if lo - 0.2 <= p <= lo + 0.5 * (hi - lo):
            return f"a {r['fn']} takeover right after the {x} goes on the heat"
        if r["fn"] == "mis_prep" and r["kwargs"].get("component") != x and lo - 0.5 <= p <= hi:
            return f"mis_prep({r['kwargs'].get('component')}) fires as the cook moves on from lighting the {x}"
    return None


def r4_voice_cands(case, rows):
    """Every question this case can still host, with its effective ask moment (B10)."""
    cell = case["cell"]
    tier = tier_of(cell)
    items = r4_items(cell)
    want = want_of(cell)
    inject = case.get("inject", True)
    T = moment_table(cell)
    _, _, flat = D3.replay(cell, 0)
    spans = r4_spans(rows, cell)
    prem = [r["pos"] for r in rows if r["fn"] == "premature_serve"]
    prem_pos = prem[0] if prem else None
    lu_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "leave_unattended"}
    raw_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "serve_raw"}

    def pos_of_plate(k):
        return r4_plate_pos(cell, k)

    def live_pos(c):
        m_ = c["m"]
        k_ = len(D3.key_at(cell, m_)["plate"])
        base_, _, arg_ = c["kind"].partition(":")
        item_ = next(iter(q_items(c["kind"])), None) if base_ in ("ready_in", "elapsed") else None
        p_ = pos_of_plate(k_) if k_ > 0 else 0.3
        for x_ in prep_floor(cell, m_, item_):
            p_ = max(p_, T.get(f"chop({x_})", T.get(f"load({x_})", 0.0)) + 0.2)
        if base_ in ("ready_in", "elapsed") and f"load({arg_})" in T:
            p_ = max(p_, T[f"load({arg_})"] + 0.2)
        if base_ == "past_done":
            p_ = max(p_, min([T[f"done({x_})"] for x_ in items if f"done({x_})" in T] or [0.0]) - 0.3)
        if base_ == "missing":
            p_ = max(p_, T.get("plate0", 2.0))
        if base_ == "after_ref":
            p_ = max(p_, pos_of_plate(max(2, k_)))
        if ROUND5:
            p_ = max(p_, r5_ref_pos(cell, c["kind"]))       # H7: never on the tick the referent event completes
        return p_

    qa_, qb_ = r4_quiet_after(cell), r4_quiet_before(cell)

    def too_close(p_, a0, e0):
        return (a0 - qb_ < p_ < e0 + qa_)

    def effective(p_):
        for _ in range(8):
            hit = [e0 for a0, e0 in spans if too_close(p_, a0, e0)]
            if not hit:
                return p_
            p_ = max(hit) + qa_
        return None

    cands = []
    for c in cell_question_cands(cell, bool(inject)):
        kind = c["kind"]
        base = kind.split(":")[0]
        c = dict(c, pos=live_pos(c), pos_r2=c["pos"])
        if base == "ready_in" and q_items(kind) & lu_items:
            continue
        if raw_items and raw_items & q_items(kind) and base in ("ready_in", "elapsed", "order"):
            continue
        if base == "next_step":
            # the engine holds a next-step question while a forget is PENDING or LIVE, so in a forget case it can
            # only be asked once that forget has fired and been released (round 3 dropped the kind outright)
            fg = [r for r in rows if r["fn"] == "forget_next_step"]
            if fg and live_pos(c) < max(r["end"] for r in fg) + r4_quiet_after(cell):
                c = dict(c, pos=max(r["end"] for r in fg) + r4_quiet_after(cell))
        if r4_voice_block(c, rows, cell):
            continue
        if ROUND5 and len(D3.key_at(cell, c["m"])["plate"]) < r5_ref_plate_floor(cell, kind):
            continue                    # H7: its referent event is not yet old at this anchor
        e_ = effective(c["pos"])
        ceil_ = T["serve"] - max(0.6, 20.0 / r4_tpm(cell))       # the serve ceiling (F7), in this tier's scale
        if e_ is None or e_ > ceil_ or (prem_pos is not None and e_ >= prem_pos - 0.5):
            continue
        base_, _, arg_ = kind.partition(":")
        if base_ in ("ready_in", "elapsed") and e_ > T.get(f"done({arg_})", 99) - 0.3:
            continue
        cands.append(dict(c, pos_eff=e_))
    return cands


def r4_pick_questions(case, rows, cands, fam_count, q_truths, visit_truths, req_state,
                      n_want=None, have=(), strict_cap=False):
    """Choose this case's two questions: the family the suite is furthest behind on first, never a family already
    at its share cap while another is free, and never a ready_in about the one item that could carry the case's
    standing request (round 3 lost 6 requests to exactly that collision)."""
    cell = case["cell"]
    tier = tier_of(cell)
    items = r4_items(cell)
    inject = case.get("inject", True)
    raw_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "serve_raw"}
    req_items = r4_req_items(rows, cell)
    if len(req_items) == 1 and req_state["n"] < R4_PROACTIVE_TARGET and req_state[tier] < R4_REQ_PER_TIER:
        cands = [c for c in cands if not (c["kind"].startswith("ready_in") and q_items(c["kind"]) & req_items)]
    vt = visit_truths
    want_yes = vt["yes"] < vt["no"]

    def visit_bonus(c):
        if not c["kind"].startswith("happened:visit("):
            return 0.0
        return 0.8 if (not D3.negative(c["gt"])) == want_yes else -0.8
    want_neg = (sum(q_truths) / max(1, len(q_truths))) < D3.NEG_SHARE
    n_want = R4_Q_TARGET if n_want is None else n_want
    qsubs, fams = [], {h["qa_family"] for h in have}
    have_kinds = {h["gt_kind"] for h in have}
    have_pos = [h["_pos"] for h in have]
    # B10: the family the suite is furthest behind on goes first
    order_c = sorted(cands, key=lambda c: (1 if fam_count[c["fam"]] >= R4_FAM_CAP else 0,
                                           fam_count[c["fam"]], TIER_KINDS[tier][c["kind"].split(":")[0]],
                                           -(c["score"] + visit_bonus(c) + (0.4 if (want_neg and D3.negative(c["gt"])) else 0)),
                                           c["kind"], c["m"]))
    def _take(c):
        qsubs.append(c)
        fams.add(c["fam"])
        fam_count[c["fam"]] += 1
        TIER_KINDS[tier][c["kind"].split(":")[0]] += 1
        q_truths.append(1 if D3.negative(c["gt"]) else 0)
        if c["kind"].startswith("happened:visit("):
            visit_truths["no" if D3.negative(c["gt"]) else "yes"] += 1
    # two passes: the first takes only families still under their share cap, the second fills a case that would
    # otherwise go short (a Caprese case can ask four families in all, and three of them are the common ones)
    for over_cap_ok in ((False,) if strict_cap else (False, True)):
        for c in order_c:
            if len(qsubs) == n_want:
                break
            if not over_cap_ok and fam_count[c["fam"]] >= R4_FAM_CAP:
                continue
            if c["fam"] in fams or c["kind"] in have_kinds or any(q["kind"] == c["kind"] for q in qsubs):
                continue
            gap_ = max(0.35, 10.0 / r4_tpm(cell))
            if any(abs(q["pos_eff"] - c["pos_eff"]) < gap_ for q in qsubs) or \
                    any(abs(hp - c["pos_eff"]) < gap_ for hp in have_pos):
                continue
            _take(c)
        if len(qsubs) == n_want:
            break
    out = []
    for c in qsubs:
        out.append({"name": f"{case['name']}_q_{c['kind'].split(':')[0]}_{c['m']}", "family": "qa_probe", "cell": cell,
                    "err": "no_error", "trigger_after": c["m"], "intent_seq": [], "resume_from": c["m"],
                    "question": c["text"], "gt_kind": c["kind"], "expected_gt": c["gt"], "qa_family": c["fam"],
                    "hardness": round(c["score"], 2), "hardness_why": c["why"], "ask_tick_clean": c["tick"],
                    "ask_m_clean": c["m_ask"], "inject": inject, "auto": False, "_pos": c["pos_eff"],
                    # S1 (self-review): one unit per item. A question's truth needs its event to have HAPPENED,
                    # not the whole batch: `prep {fish: 1, potato: 2}` was never reached in hard_hard_1 and the
                    # order question was lost. S2: and never before the cook has prepped anything at all — an
                    # "how many boards are free" asked at t0 answers itself (easy_hard_3, asked at tick 0).
                    "prep_floor": r4_question_floor(cell, c, raw_items)})
    return out


def r4_req_items(rows, cell):
    """B11 / C7: the cooked items this case could still hang a standing request on — never the walk-away's food,
    never one a serve_raw targets, never the first item lit when a mis_prep of another item covers its cook, and
    never one that finishes after a premature serve has ended the episode."""
    want = want_of(cell)
    items = r4_items(cell)
    T = moment_table(cell)
    cooked = [x for x in dict.fromkeys(items) if want[x] == "cooked"]
    if not cooked:
        return set()
    bad = {r["kwargs"]["component"] for r in rows if r["fn"] in ("leave_unattended", "serve_raw")}
    first_lit = cooked[0]
    if any(r["fn"] == "mis_prep" and r["kwargs"].get("component") != first_lit
           and r["pos"] <= T.get(f"done({first_lit})", 0.0) for r in rows):
        bad.add(first_lit)
    prem = [r["pos"] for r in rows if r["fn"] == "premature_serve"]
    return {x for x in cooked if x not in bad and not (prem and T.get(f"done({x})", 99) >= prem[0])
            and r4_request_anchor(cell, x) is not None}


def r4_question_floor(cell, c, raw_items):
    """S1 / S2 (round-4 self-review): the progress floor written beside a question."""
    item_ = next(iter(q_items(c["kind"])), None) if c["kind"].startswith(("ready_in", "elapsed")) else None
    pf = {k: 1 for k, v in prep_floor(cell, c["m"], item_).items() if k not in raw_items}
    if not pf and c["pos_eff"] > 0.4:
        first = next((x for x in r4_items(cell) if want_of(cell)[x] != "raw"), None)
        if first is not None and first not in raw_items:
            pf = {first: 1}
    return pf


def r4_design_request(case, rows, qsubs, fam_count, req_state):
    """B11: "tell me when the X is done", at the tick X goes on the heat."""
    cell = case["cell"]
    tier = tier_of(cell)
    items = r4_items(cell)
    if req_state["n"] >= R4_PROACTIVE_TARGET or req_state[tier] >= R4_REQ_PER_TIER:
        return None
    blocked = {q for c in qsubs for q in q_items(c["gt_kind"]) if c["gt_kind"].startswith("ready_in")}
    choices = sorted(r4_req_items(rows, cell) - blocked,
                     key=lambda x: (req_state["item"][x], items.index(x)))
    if not choices:
        return None
    item = choices[0]
    m_on, gt, tick = r4_request_anchor(cell, item)
    multi = items.count(item) > 1
    fam_count["Q10"] += 1
    req_state["n"] += 1
    req_state[tier] += 1
    req_state["item"][item] += 1
    return {"name": f"{case['name']}_p_ready_{item}_{m_on}", "family": "qa_proactive", "cell": cell,
            "err": "no_error", "trigger_after": m_on, "intent_seq": [], "resume_from": m_on, "item": item,
            "question": (f"Give me a shout the moment the {item} I just put on the heat is done, would you?"
                         if multi else f"Give me a shout the moment the {item} is done, would you?"),
            "gt_kind": f"ready:{item}", "expected_gt": gt, "qa_family": "Q10", "ask_tick_clean": tick,
            "inject": case.get("inject", True), "auto": False,
            "_pos": moment_table(cell).get(f"load({item})", 1.0) + 0.3, "prep_floor": {item: 1}}


# ---------------------------------------------------- round 4: what a mistake set leaves room for (voice/plan)
_R4_PC_ING = {}
_R4_INTENTS = {}
R4_PC_WORDS = {"lemon": "add a squeeze of lemon on top before serving",
               "butter": "add a knob of butter on top before serving",
               "peas": "add a spoon of peas on top before serving",
               "cheese": "add an extra helping of cheese on top before serving",
               "lettuce": "add some fresh lettuce on top before serving",
               "onion": "add some chopped onion on top before serving",
               "corn": "add a spoon of corn on top before serving",
               "avocado": "add some avocado on top before serving",
               "mushroom": "add some mushrooms on top before serving",
               "egg": "put a fried egg on top before serving",
               "shrimp": "add a few shrimp on top before serving"}


def r4_intents(cell):
    """B12: the announced plan changes available in this kitchen. The v3 library has 12 over eight kitchens (none
    for the burrito, whose five components already fill the plate), which is not enough to design a FAMILY of
    them: every kitchen's second off-recipe crate is added here as one more, built exactly like the library's
    (walk to the crate, scoop; the goal is extended and the cook announces it itself)."""
    if cell in _R4_INTENTS:
        return _R4_INTENTS[cell]
    try:
        its = [dict(it) for it in json.load(open(os.path.join(ROOT, "benchmark", "intents_v3.json")))["cases"] if it["cell"] == cell]
    except Exception:                                       # noqa: BLE001
        its = []
    have = {(it.get("goal_extend") or {}).get("ingredient") for it in its}
    tpl = its[0] if its else None
    for w in r4_off_sources(cell):
        if w in have or w in R4_PC_BAD_ITEMS or w not in R4_PC_WORDS:
            continue
        it = dict(tpl or {}, cell=cell, family="plan_change", name=f"plan_change_add_{w}",
                  intent_label=R4_PC_WORDS[w], intent_seq=[f"go to the {w} source", "scoop"],
                  goal_extend={"ingredient": w, "state": "raw"}, synth_r4=True)
        it.pop("verification", None)
        its.append(it)
    _R4_INTENTS[cell] = its
    return its


def r4_pc_ingredients(cell):
    if cell not in _R4_PC_ING:
        _R4_PC_ING[cell] = [(it.get("goal_extend") or {}).get("ingredient") for it in r4_intents(cell)
                            if (it.get("goal_extend") or {}).get("ingredient") not in R4_PC_BAD_ITEMS]
    return _R4_PC_ING[cell]


def r4_usable_questions(cell, rows):
    """How many of the cell's question candidates this mistake set still leaves askable (capped at 2), and their
    families. Precomputed per admissible set so the error assignment can be scored on the room it leaves for the
    voice: round 3 designed the mistakes first and lost one question in three to them."""
    T = moment_table(cell)
    spans = r4_spans(rows, cell)
    qa_, qb_ = r4_quiet_after(cell), r4_quiet_before(cell)
    items = r4_items(cell)
    prem = [r["pos"] for r in rows if r["fn"] == "premature_serve"]
    prem_pos = prem[0] if prem else None
    lu_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "leave_unattended"}
    raw_items = {r["kwargs"]["component"] for r in rows if r["fn"] == "serve_raw"}
    fg_end = max([r["end"] for r in rows if r["fn"] == "forget_next_step"] or [None] or [])  # noqa
    fg = [r for r in rows if r["fn"] == "forget_next_step"]
    ceil_ = T["serve"] - max(0.6, 20.0 / r4_tpm(cell))
    fams = set()
    for c in cell_question_cands(cell, not cell.endswith("hard_map")):
        kind = c["kind"]
        base, _, arg = kind.partition(":")
        p_ = r4_live_pos(cell, c, items, T)
        if base == "next_step" and fg:
            p_ = max(p_, max(r["end"] for r in fg) + qa_)
        if base == "ready_in" and q_items(kind) & lu_items:
            continue
        if raw_items and raw_items & q_items(kind) and base in ("ready_in", "elapsed", "order"):
            continue
        if r4_voice_block(dict(c, pos=p_), rows, cell):
            continue
        for _ in range(8):
            hit = [e0 for a0, e0 in spans if a0 - qb_ < p_ < e0 + qa_]
            if not hit:
                break
            p_ = max(hit) + qa_
        else:
            continue
        if p_ > ceil_ or (prem_pos is not None and p_ >= prem_pos - 0.5):
            continue
        if base in ("ready_in", "elapsed") and p_ > T.get(f"done({arg})", 99) - 0.3:
            continue
        fams.add(c["fam"])
    return min(R4_Q_TARGET, len(fams)), fams


def r4_live_pos(cell, c, items, T):
    m_ = c["m"]
    k_ = len(D3.key_at(cell, m_)["plate"])
    base_, _, arg_ = c["kind"].partition(":")
    item_ = next(iter(q_items(c["kind"])), None) if base_ in ("ready_in", "elapsed") else None
    p_ = r4_plate_pos(cell, k_) if k_ > 0 else 0.3
    for x_ in prep_floor(cell, m_, item_):
        p_ = max(p_, T.get(f"chop({x_})", T.get(f"load({x_})", 0.0)) + 0.2)
    if base_ in ("ready_in", "elapsed") and f"load({arg_})" in T:
        p_ = max(p_, T[f"load({arg_})"] + 0.2)
    if base_ == "past_done":
        p_ = max(p_, min([T[f"done({x_})"] for x_ in items if f"done({x_})" in T] or [0.0]) - 0.3)
    if base_ == "missing":
        p_ = max(p_, T.get("plate0", 2.0))
    if base_ == "after_ref":
        p_ = max(p_, r4_plate_pos(cell, max(2, k_)))
    return p_


def r4_req_capable(rows, cell):
    """B11: is a standing request still possible? (C7: never about the walk-away's food or a serve_raw's item.)"""
    want = want_of(cell)
    cooked = [x for x in dict.fromkeys(r4_items(cell)) if want[x] == "cooked"]
    if not cooked:
        return False
    bad = {r["kwargs"]["component"] for r in rows if r["fn"] in ("leave_unattended", "serve_raw")}
    first_lit = cooked[0]
    T = moment_table(cell)
    if any(r["fn"] == "mis_prep" and r["kwargs"].get("component") != first_lit
           and r["pos"] <= T.get(f"done({first_lit})", 0.0) for r in rows):
        bad = bad | {first_lit}
    prem = [r["pos"] for r in rows if r["fn"] == "premature_serve"]
    ok = [x for x in cooked if x not in bad and not (prem and T.get(f"done({x})", 99) >= prem[0])
          and r4_request_anchor(cell, x) is not None]
    return bool(ok)


_R4_REQ_ANCHOR = {}


def r4_request_anchor(cell, item):
    """The clean-run anchor at which "tell me when the X is done" is made (the tick X goes on the heat), or None
    when this kitchen cannot host the request for X."""
    key = (cell, item)
    if key not in _R4_REQ_ANCHOR:
        _, _, flat = D3.replay(cell, 0)
        m_on = next((i + 1 for i, (cmd, sid) in enumerate(flat)
                     if cmd == "turn on" and (sid or "").split("#")[0] in (f"prep({item})", f"ignite({item})")), None)
        res = None
        if m_on is not None and robust_anchor(cell, m_on, item=item):
            _, ro, _ = D3.replay(cell, m_on)
            gt = GI.ground_truth(f"ready:{item}", ro.game, ro)
            if gt.get("ticks_left", 0) >= 5:
                res = (m_on, gt, ro.tick)
        _R4_REQ_ANCHOR[key] = res
    return _R4_REQ_ANCHOR[key]


def r4_pc_moments(rows, cell):
    """B12: the clean-run anchors at which an announced plan change could be made in this case — the plate
    started, clear of every takeover (VOICE_QUIET before, the release plus G after) and a plating step before the
    serve. Empty when the case has no room for one."""
    items = r4_items(cell)
    T = moment_table(cell)
    if any(r["fn"] == "premature_serve" for r in rows):
        return []
    extra_n = sum(1 for r in rows if r["fn"] in ("add_foreign", "misidentify_assemble")
                  or (r["fn"] == "repeat_step" and str(r["kwargs"].get("sid", "")).startswith("assemble(")))
    if len(items) + extra_n + 1 > 5:
        return []
    spans = r4_spans(rows, cell)
    qa_, qb_ = r4_quiet_after(cell), r4_quiet_before(cell)
    raw = {r["kwargs"]["component"] for r in rows if r["fn"] == "serve_raw"}
    out = []
    for a in D3.question_anchors(cell, []):
        m = a["m"]
        if not robust_anchor(cell, m) or len(D3.key_at(cell, m)["plate"]) < 1:
            continue
        pf = {k: v for k, v in prep_floor(cell, m).items() if k not in raw}
        pp = [T.get(f"chop({x})", T.get(f"load({x})", 0.0)) for x in pf]
        pos = max([r4_plate_pos(cell, len(D3.key_at(cell, m)["plate"]))] + pp) + 0.3
        if pos > T["serve"] - max(0.5, 20.0 / r4_tpm(cell)):
            continue
        if any(a0 - qb_ < pos < e0 + qa_ for a0, e0 in spans):
            continue
        out.append((m, pos, pf))
    return out


def r4_pc_capable(rows, cell):
    """B12: is an announced plan change still possible in this case?"""
    ings = r4_pc_ingredients(cell)
    if not ings or not r4_pc_moments(rows, cell):
        return False
    touched = set().union(*[r4_slots(r, cell) for r in rows]) if rows else set()
    return any(i not in touched for i in ings)


_R4_STATS = {}


def r4_stats(cell, si, rows):
    key = (cell, si)
    if key not in _R4_STATS:
        nq, fams = r4_usable_questions(cell, rows)
        _R4_STATS[key] = {"nq": nq, "fams": fams, "req": r4_req_capable(rows, cell), "pc": r4_pc_capable(rows, cell)}
    return _R4_STATS[key]


# ------------------------------------------------------------------------------- round 4: plan changes + main
R4_PC_BAD_ITEMS = {"pepper"}     # D7: it renders as a grinder, not a food
R4_PC_PER_TIER = 4


def r4_design_plan_change(case, rows, intents, used_intents, pc_state):
    """B12: an announced plan change, designed as a family rather than as a leftover. It is never a mistake: the
    cook says it out loud, the engine extends the goal, and the served plate is checked for the extra item."""
    cell = case["cell"]
    tier = tier_of(cell)
    if pc_state["n"] >= R4_PLAN_CHANGES or pc_state[tier] >= R4_PC_PER_TIER:
        return None
    ms = r4_pc_moments(rows, cell)
    if not ms:
        return None
    wrongs = {r["kwargs"].get("wrong") for r in rows if r["kwargs"].get("wrong")}
    touched = set().union(*[r4_slots(r, cell) for r in rows]) if rows else set()
    m_i, p_i, pf = min(ms, key=lambda t: (t[1], t[0]))
    for it in sorted(intents, key=lambda i: pc_state["item"][(i.get("goal_extend") or {}).get("ingredient")]):
        ing = (it.get("goal_extend") or {}).get("ingredient")
        if (cell, it["name"]) in used_intents or ing in wrongs or ing in touched or ing in R4_PC_BAD_ITEMS:
            continue
        isub = {"name": f"{case['name']}_i_plan_change_{m_i}", "family": "plan_change", "cell": cell, "err": "no_error",
                "trigger_after": m_i, "resume_from": m_i, "intent_label": it["intent_label"],
                "intent_seq": list(it["intent_seq"]), "duration": it.get("duration"),
                "clean_tick": D3.tick_at(cell, m_i), "rationale": it.get("rationale"),
                "inject": case.get("inject", True), "auto": False, "goal_extend": it["goal_extend"],
                "_pos": p_i, "prep_floor": pf}
        used_intents.add((cell, it["name"]))
        pc_state["n"] += 1
        pc_state[tier] += 1
        pc_state["item"][ing] += 1
        return isub
    return None


_R4_TPL = {}


def r4_offsets(cell, fn, pool):
    """Grading offsets for a round-4 row: the inventory's, where it has this mistake on this kitchen. They are
    cosmetic since the 2026-09-20 live windows (the window opens at the first deviated act and closes at the
    commit), but the report still carries them."""
    key = (cell, fn)
    if key not in _R4_TPL:
        cand = [r for r in pool if r["fn"] == fn] or [r for r in pool]
        t = cand[0] if cand else None
        _R4_TPL[key] = ({k: t[k] for k in ("dev", "identifiable", "window", "outcome", "max_smoke") if k in t}
                        if t else {"dev": 0, "identifiable": 6, "window": [6, 38], "outcome": None, "max_smoke": 0})
    return dict(_R4_TPL[key])


def r4_error_sub(case, r, pool):
    sub = {"err": r["err"], "fn": r["fn"], "kwargs": dict(r["kwargs"]), "cell": case["cell"],
           "inject": case.get("inject", True), "persona": "capable_collaborator", "family": "error",
           "ruinous": bool(r["ruinous"]), "robust_ruinous": bool(r["ruinous"] and r["robust"]),
           "name": f"{case['name']}_{r['err']}", "design_r4": {"why": r["why"], "moment": round(r["pos"], 2)}}
    sub.update(r4_offsets(case["cell"], r["fn"], pool))
    # B2: the floor is ALWAYS written, prep_floor even when empty — derive_spec falls back to the reference run's
    # prep otherwise, and an LLM cook working in another order may never have it
    sub["prep_floor"] = dict(r["prep_floor"])
    if r["plate_floor"] is not None:
        sub["plate_floor"] = int(r["plate_floor"])
    if r.get("lu_late"):
        sub["lu_late"] = int(r["lu_late"])
    sub["_pos"] = r["pos"]
    return sub


def r4_main(a, base, prev_by):
    rng = random.Random(a.seed)
    inv = load_inventory(a.inventory)
    by_cell = collections.defaultdict(list)
    for r in inv:
        by_cell[r["cell"]].append(r)
    cases = base["cases"]
    sets, pool, choice = r4_assign(cases, rng)
    sets = r4_assign_wrongs(cases, sets, rng)
    intents = {c["cell"]: r4_intents(c["cell"]) for c in cases}
    fam_count, q_truths, visit_truths = collections.Counter(), [], collections.Counter()
    req_state = collections.Counter(); req_state["item"] = collections.Counter()
    pc_state = collections.Counter(); pc_state["item"] = collections.Counter()
    used_intents = set()
    designed = [{"case": c, "rows": rows, "cands": r4_voice_cands(c, rows), "qsubs": [], "psub": None, "isub": None}
                for c, rows in zip(cases, sets)]
    # B10: most constrained case first — a case with two askable families has no choice, so it must not find both
    # of them already at their share cap because an unconstrained case took them
    for d in sorted(designed, key=lambda d: (len({c["fam"] for c in d["cands"]}), d["case"]["name"])):
        d["qsubs"] = r4_pick_questions(d["case"], d["rows"], d["cands"], fam_count, q_truths, visit_truths, req_state)
    # B10: a THIRD question in the cases that can host one, from a family still under its share cap. The Caprese
    # and the cheeseburger can only ask four or five kinds at all, so two per case everywhere leaves the two
    # always-available families ("what is my next step", "how long has it been on") at a fifth of the suite; the
    # third asks are what bring the rarer families up and the suite to ~100.
    # requests after the questions (a ready_in question and a request about one item would be one alert twice)
    for d in sorted(designed, key=lambda d: (len(r4_req_items(d["rows"], d["case"]["cell"])), d["case"]["name"])):
        d["psub"] = r4_design_request(d["case"], d["rows"], d["qsubs"], fam_count, req_state)
    budget = R4_Q_EXTRA
    for d in sorted(designed, key=lambda d: (-len({c["fam"] for c in d["cands"]}), d["case"]["name"])):
        if budget <= 0:
            break
        cands_ = [c for c in d["cands"] if not (d["psub"] and c["kind"] == f"ready_in:{d['psub']['item']}")]
        more = r4_pick_questions(d["case"], d["rows"], cands_, fam_count, q_truths, visit_truths, req_state,
                                 n_want=1, have=d["qsubs"], strict_cap=True)
        if more:
            d["qsubs"] = d["qsubs"] + more
            budget -= 1
    # plan changes last, so they are placed around everything else (B12)
    for d in sorted(designed, key=lambda d: (tier_of(d["case"]["cell"]), d["case"]["name"])):
        d["isub"] = r4_design_plan_change(d["case"], d["rows"], intents.get(d["case"]["cell"], []),
                                          used_intents, pc_state)
    out_cases, table = [], []
    for d in designed:
        c, rows, qsubs, psub, isub = (d[k] for k in ("case", "rows", "qsubs", "psub", "isub"))
        esubs = [r4_error_sub(c, r, by_cell[c["cell"]]) for r in rows]
        subs = esubs + qsubs + ([psub] if psub else []) + ([isub] if isub else [])
        fam_rank = {"qa_proactive": 0, "qa_probe": 1, "plan_change": 2, "error": 3}
        subs.sort(key=lambda s: (s["_pos"], fam_rank[s["family"]]))
        voice = sorted([s for s in subs if s["family"] in ("qa_proactive", "qa_probe")],
                       key=lambda s: (0 if s["family"] == "qa_proactive" else 1, s["_pos"]))
        vi = iter(voice)
        subs = [next(vi) if s["family"] in ("qa_proactive", "qa_probe") else s for s in subs]
        for s in subs:
            s.pop("_pos", None)
        nc = {k: v for k, v in c.items() if k not in ("subs", "anchors", "order_pattern", "rationale", "n_errors",
                                                      "n_ruinous", "design_v5")}
        nc["subs"] = subs
        nc["n_errors"] = len(esubs)
        nc["n_ruinous"] = sum(1 for s in esubs if s["ruinous"])
        nc["order_pattern"] = "".join({"qa_probe": "Q", "qa_proactive": "P", "plan_change": "I"}.get(s["family"], "E")
                                      for s in subs)
        nc["rationale"] = "; ".join(f"{r['why']} (moment {r['pos']:.1f}, floor plate>={r['plate_floor']} prep>={r['prep_floor'] or '{}'})"
                                    for r in rows) + ". Questions: " + " / ".join(
            f"“{q['question']}” ({q['qa_family']})" for q in qsubs) + \
            (f". Standing request: “{psub['question']}”" if psub else "") + \
            (f". Plan change: {isub['intent_label']}" if isub else "")
        nc["design_r4"] = {"errors": len(rows), "robust_ruinous": sum(1 for s in esubs if s["robust_ruinous"])}
        out_cases.append(nc)
        old = [s for s in (prev_by.get(c["name"]) or c)["subs"] if s.get("family") == "error"]
        table.append((c["name"], tier_of(c["cell"]),
                      [f"{s.get('fn')}{json.dumps(s['kwargs'], separators=(',', ':'))}" for s in old],
                      [f"{r['fn']}{json.dumps(r['kwargs'], separators=(',', ':'))}"
                       f"@plate>={r['plate_floor']}" for r in rows],
                      [q["qa_family"] for q in qsubs], bool(psub), (isub or {}).get("intent_label")))
        print(f"{c['name']}: {[(r['fn'], r['kwargs']) for r in rows]} | q={[q['gt_kind'] for q in qsubs]} "
              f"p={psub and psub['item']} i={isub and isub['intent_label']}", flush=True)
    res = {"schema": "composite-v5", "cases": out_cases,
           "design_v5": {"designer": "python -m cooksim_bench.suite.design_composite_v5", "round": 4, "base": a.base,
                         "inventory": a.inventory, "seed": a.seed, "problems": []}}
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, ensure_ascii=True)
    os.replace(tmp, a.out)
    fn_n = collections.Counter(s["fn"] for c in out_cases for s in c["subs"] if s["family"] == "error")
    fams = collections.Counter(s["qa_family"] for c in out_cases for s in c["subs"] if s["family"] == "qa_probe")
    n_req = sum(1 for c in out_cases for s in c["subs"] if s["family"] == "qa_proactive")
    n_pc = sum(1 for c in out_cases for s in c["subs"] if s["family"] == "plan_change")
    r4_write_ledger(a, table, fn_n, fams, n_req, n_pc, out_cases)
    print(f"wrote {a.out}: {len(out_cases)} cases, {sum(fn_n.values())} errors "
          f"({sum(c['n_ruinous'] for c in out_cases)} ruinous), {sum(fams.values())} questions, "
          f"{n_req} requests, {n_pc} plan changes")
    print("errors   ", dict(sorted(fn_n.items(), key=lambda kv: -kv[1])))
    print("questions", dict(sorted(fams.items())))
    # FRONTIER (user ruling, 2026-09-20): the per-tier design-reuse limit is the HARD constraint and the per-type
    # band gives way to it, not the other way round -- six cross-kitchen repeats inside the limit are a smaller
    # distortion than eight beyond it, and the repeats are what a reader notices in the case table. The weights
    # that encode that: 100,000 per repeat beyond R5_DESIGN_REPS, 50,000 per repeat inside ONE kitchen, 8,000 per
    # unit outside the 10-17 band. Do not re-tune one without the other. Everything the run report needs is
    # printed here rather than re-derived by hand.
    nrm = collections.defaultdict(list)
    kit = collections.Counter()
    for c in out_cases:
        rows_ = [{"fn": s_["fn"], "kwargs": s_["kwargs"]} for s_ in c["subs"] if s_["family"] == "error"]
        k_ = r4_norm(rows_)
        nrm[(tier_of(c["cell"]), k_)].append(c["name"])
        kit[(c["cell"], k_)] += 1
    dup = {t_: sum(len(v) - 1 for (tt, _k), v in nrm.items() if tt == t_ and len(v) > 1)
           for t_ in ("easy", "medium", "hard", "expert", "master")}
    print(f"duplicates {sum(dup.values())} (by tier {dup}), allowance {dict(R5_DESIGN_REPS)}, "
          f"same-kitchen {sum(v - 1 for v in kit.values() if v > 1)}")
    over = {f: n for f, n in fn_n.items() if not R5_TYPE_MIN <= n <= R5_TYPE_MAX}
    print(f"out of the {R5_TYPE_MIN}-{R5_TYPE_MAX} band: {over or 'none'}")
    print(f"plan changes {n_pc} (design target {R4_PLAN_CHANGES}); requests {n_req} "
          f"(target {R4_PROACTIVE_TARGET})")
    lu_ = collections.Counter((tier_of(c["cell"]), s_["kwargs"].get("component"))
                              for c in out_cases for s_ in c["subs"]
                              if s_["family"] == "error" and s_["fn"] == "leave_unattended")
    print("walk-aways", dict(lu_))


def r4_write_ledger(a, table, fn_n, fams, n_req, n_pc, out_cases):
    head = open(a.ledger).read() if os.path.exists(a.ledger) else ""
    mark = "\n## " + (a.ledger_section if ROUND5 else "Round 4 (balanced)")
    if mark in head:
        head = head[:head.index(mark)]
    prev_lbl, this_lbl = ("round 4", "round 5") if ROUND5 else ("round 3", "round 4")
    notes = LEDGER_R5_NOTES if ROUND5 else LEDGER_R4_NOTES
    old3 = ({"forget_next_step": 17, "mis_prep": 16, "serve_raw": 15, "misidentify_assemble": 15,
             "add_foreign": 15, "repeat_step": 14, "leave_unattended": 14, "premature_serve": 12,
             "wrong_order": 12, "omit_subgoal": 10} if ROUND5 else
            {"leave_unattended": 34, "forget_next_step": 32, "add_foreign": 31, "mis_prep": 29,
             "premature_serve": 11, "misidentify_assemble": 6, "omit_subgoal": 6, "wrong_order": 1,
             "serve_raw": 0, "repeat_step": 0})
    old_f = ({"Q1": 13, "Q2": 8, "Q3": 20, "Q4": 8, "Q5": 12, "Q7": 7, "Q8": 9, "Q9": 17} if ROUND5 else
             {"Q1": 4, "Q2": 9, "Q3": 22, "Q4": 8, "Q5": 12, "Q7": 0, "Q8": 6, "Q9": 18})
    prev_tot, prev_q, prev_req, prev_pc = (140, 94, 29, 11) if ROUND5 else (150, 79, 21, 6)
    n_err = sum(fn_n.values())
    with open(a.ledger, "w") as f:
        f.write(head.rstrip("\n") + "\n" + mark + "\n\n")
        f.write(notes + "\n\n")
        f.write(f"### Per-type counts by REALISED type (old = the {prev_lbl} suite, new = this one)\n\n")
        f.write(f"| error type | {prev_lbl} | {this_lbl} | share |\n|---|---|---|---|\n")
        for fn in R4_FNS:
            f.write(f"| {fn} | {old3.get(fn, 0)} | {fn_n[fn]} | {100.0 * fn_n[fn] / max(1, n_err):.1f}% |\n")
        f.write(f"| **total** | {prev_tot} | {n_err} | |\n\n")
        f.write(f"### Question families\n\n| family | {prev_lbl} | {this_lbl} | share |\n|---|---|---|---|\n")
        nq = sum(fams.values())
        for fam in R4_FAMS:
            f.write(f"| {fam} | {old_f.get(fam, 0)} | {fams[fam]} | {100.0 * fams[fam] / max(1, nq):.1f}% |\n")
        f.write(f"| **total** | {prev_q} | {nq} | |\n\n")
        f.write(f"Standing requests (Q10): {prev_req} -> {n_req}. Plan changes: {prev_pc} -> {n_pc}.\n\n")
        f.write(f"### Per case\n\n| case | tier | {prev_lbl} errors | {this_lbl} errors (with plate floor) | Q | req | plan change |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for n, tier, o, nw, qf, pr, pc in table:
            f.write(f"| {n} | {tier} | {'<br>'.join(o)} | {'<br>'.join(nw)} | {', '.join(qf)} | "
                    f"{'yes' if pr else ''} | {pc or ''} |\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="benchmark/cases_composite_v4.json")
    ap.add_argument("--out", default="benchmark/cases_composite_v5.json")
    ap.add_argument("--inventory", default="benchmark/error_inventory_v5.json")
    ap.add_argument("--ledger", default="benchmark/cases_composite_v5_ledger.md")
    ap.add_argument("--overrides", default="benchmark/suite_v5_overrides.json",
                    help="hand decisions the designer cannot express (each with its reason)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--prev", default="git:96169c1:bench/v5/cases_composite_v5.json",
                    help="round 2: the previous suite (a path, or git:<rev>:<path>); its errors are the ones kept where "
                         "still legal and the ledger's round-2 section compares against it. 'none' = round-1 behaviour")
    ap.add_argument("--ledger-section", default="Round 5 (floors that dominate)",
                    help="the ledger heading this run (re)writes; everything above it is kept")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = json.load(open(a.base))
    prev = None
    if a.prev and a.prev != "none":
        if a.prev.startswith("git:"):
            import subprocess
            _g, rev, path = a.prev.split(":", 2)
            prev = json.loads(subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=ROOT))
        else:
            prev = json.load(open(a.prev))
    prev_by = {c["name"]: c for c in (prev or {}).get("cases", [])}
    if ROUND4:
        return r4_main(a, base, prev_by)
    inv = load_inventory(a.inventory)
    by_cell = collections.defaultdict(list)
    for r in inv:
        by_cell[r["cell"]].append(r)
    synth_mis_prep(by_cell)
    synth_wrong_items(by_cell)
    overrides = json.load(open(a.overrides)) if os.path.exists(a.overrides) else {"cases": {}}
    intents = collections.defaultdict(list)
    for it in json.load(open(os.path.join(ROOT, "benchmark", "intents_v3.json")))["cases"]:
        intents[it["cell"]].append(it)
    v4_plan_change = {c["name"] for c in (prev or base)["cases"] if any(s.get("family") == "plan_change" for s in c["subs"])}
    tier_state = collections.defaultdict(lambda: {"labels": collections.Counter(), "fns": collections.Counter(), "t1": 0,
                                                  "wrongs": collections.Counter(), "patterns": collections.Counter(),
                                                  "sigs": [], "rows": collections.Counter(),
                                                  "fn_cases": collections.Counter(), "openings": collections.Counter()})
    cell_sigs = collections.defaultdict(set)
    fam_count, q_truths, visit_truths = collections.Counter(), [], collections.Counter()
    used_intents = set()
    out_cases, problems, table = [], [], []
    n_pc = 0
    n_hot = sum(1 for c in base["cases"] if any(want_of(c["cell"])[x] == "cooked" for x in recipe_items(c["cell"])))
    designed = []
    for c in base["cases"]:
        cell = c["cell"]
        tier = tier_of(cell)
        old_errs = [s for s in (prev_by.get(c["name"]) or c)["subs"] if s.get("family") == "error"]
        v4_keys = {((s.get("fn") or s["err"]), json.dumps(s["kwargs"], sort_keys=True)) for s in old_errs}
        ov = overrides["cases"].get(c["name"], {})
        if ov.get("errors"):
            rows = []
            for e in ov["errors"]:
                r = next((x for x in by_cell[cell] if x["fn"] == e["fn"] and x["kwargs"] == e["kwargs"]), None)
                if r is None:
                    sys.exit(f"override {c['name']}: {e} not in the inventory")
                rows.append(r)
            why = set_violation(rows, cell)
            if why:
                problems.append((c["name"], f"override breaks {why}"))
        else:
            rows = pick_errors(c, by_cell[cell], v4_keys, tier_state[tier], cell_sigs[cell], rng)
        if rows is None:
            problems.append((c["name"], "no admissible error set"))
            rows = []
        st = tier_state[tier]
        for r in rows:
            st["labels"][r["err"]] += 1
            st["fns"][r["fn"]] += 1
            if wrong_item(r):
                st["wrongs"][wrong_item(r)] += 1
        if any(opens_t1(r, cell) for r in rows):
            st["t1"] += 1
        st["patterns"][tuple(sorted(r["fn"] for r in rows))] += 1
        for k in fn_keys(rows):
            st["fn_cases"][k] += 1
        st["openings"][opening_of(rows, cell)] += 1
        cell_sigs[cell].add(tuple(sorted(key_of(r) for r in rows)))
        st["sigs"].append(sg_norm(tuple(sorted(key_of(r) for r in rows))))
        for r in rows:
            st["rows"][row_norm(r)] += 1
        qsubs, psub = design_voice(c, rows, fam_count, q_truths, rng, want_proactive=True, visit_truths=visit_truths)
        designed.append({"case": c, "rows": rows, "qsubs": qsubs, "psub": psub, "isub": None, "old_errs": old_errs,
                         "v4_keys": v4_keys, "ov": ov})
    # plan changes: the v4 cases that had one first, then one more case per cell with an intent, up to N_PLAN_CHANGES
    order_pc = [d for d in designed if d["case"]["name"] in v4_plan_change] + \
               [d for d in designed if d["case"]["name"] not in v4_plan_change]
    per_cell = collections.Counter()
    for d in order_pc:
        if n_pc >= N_PLAN_CHANGES:
            break
        cell = d["case"]["cell"]
        if not intents.get(cell) or per_cell[cell] >= 2:
            continue
        d["isub"] = design_plan_change(d["case"], d["rows"], d["qsubs"], intents[cell], used_intents)
        if d["isub"]:
            n_pc += 1
            per_cell[cell] += 1
    for d in designed:
        c, rows, qsubs, psub, isub, old_errs, v4_keys, ov = (d[k] for k in ("case", "rows", "qsubs", "psub", "isub", "old_errs", "v4_keys", "ov"))
        esubs = []
        for r in rows:
            k = key_of(r)
            note = None if k in v4_keys else {"replaces": [f"{s.get('fn')}{json.dumps(s['kwargs'])}" for s in old_errs
                                                           if ((s.get("fn") or s["err"]), json.dumps(s["kwargs"], sort_keys=True)) not in {key_of(x) for x in rows}]}
            esubs.append(error_sub(c, r, note))
        subs = esubs + qsubs + ([psub] if psub else []) + ([isub] if isub else [])
        # order: by moment; the standing request before any other voice sub-case (S10); at a tie a voice sub-case first
        fam_rank = {"qa_proactive": 0, "qa_probe": 1, "plan_change": 2, "error": 3}
        subs.sort(key=lambda s: (s["_pos"], fam_rank[s["family"]]))
        voice = [s for s in subs if s["family"] in ("qa_proactive", "qa_probe")]
        voice.sort(key=lambda s: (0 if s["family"] == "qa_proactive" else 1, s["_pos"]))
        vi = iter(voice)
        subs = [next(vi) if s["family"] in ("qa_proactive", "qa_probe") else s for s in subs]
        for s in subs:
            s.pop("_pos", None)
        nc = {k: v for k, v in c.items() if k not in ("subs", "anchors", "order_pattern", "rationale", "n_errors", "n_ruinous")}
        nc["subs"] = subs
        nc["n_errors"] = len(esubs)
        nc["n_ruinous"] = sum(1 for s in esubs if s["ruinous"])
        nc["order_pattern"] = "".join({"qa_probe": "Q", "qa_proactive": "P", "plan_change": "I"}.get(s["family"], "E") for s in subs)
        nc["rationale"] = "; ".join(D3_desc(r) for r in rows) + ". Questions: " + " / ".join(
            f"“{q['question']}” ({q['qa_family']}, trigger m={q['trigger_after']}, asked at m≈{q['ask_m_clean']})" for q in qsubs) + \
            (f". Standing request: “{psub['question']}”" if psub else "") + (f". Plan change: {isub['intent_label']}" if isub else "")
        nc["design_v5"] = {"kept_from_v4": sum(1 for r in rows if key_of(r) in v4_keys), "errors": len(rows),
                           **({"override": ov.get("why")} if ov.get("errors") else {})}
        out_cases.append(nc)
        why_old = removal_reasons(c, old_errs, rows, by_cell[c["cell"]])
        nc["design_v5"]["v4_removed"] = why_old
        table.append((c["name"], [f"{s.get('fn')}{json.dumps(s['kwargs'], separators=(',', ':'))}" for s in old_errs],
                      [f"{r['fn']}{json.dumps(r['kwargs'], separators=(',', ':'))}" for r in rows],
                      [f"{k}: {v}" for k, v in why_old.items()]))
        print(f"{c['name']}: {[(r['fn'], r['kwargs']) for r in rows]} | q={[q['gt_kind'] for q in qsubs]} p={psub and psub['item']} i={isub and isub['intent_label']}", flush=True)
    res = {"schema": "composite-v5", "cases": out_cases,
           "design_v5": {"designer": "python -m cooksim_bench.suite.design_composite_v5", "base": a.base, "inventory": a.inventory,
                         "seed": a.seed, "problems": problems}}
    tmp = a.out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, ensure_ascii=True)
    os.replace(tmp, a.out)
    os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
    if prev is None:
        with open(a.ledger, "w") as f:
            f.write("| case | v4 errors | v5 errors | why each dropped v4 error went |\n|---|---|---|---|\n")
            for n, o, nw, why in table:
                f.write(f"| {n} | {'<br>'.join(o)} | {'<br>'.join(nw)} | {'<br>'.join(why)} |\n")
    else:
        # round 2: keep the ledger above this run's section, (re)write the section
        head = open(a.ledger).read() if os.path.exists(a.ledger) else ""
        mark = f"\n## {a.ledger_section}"
        if mark in head:
            head = head[:head.index(mark)]
        with open(a.ledger, "w") as f:
            f.write(head.rstrip("\n") + "\n" + mark + "\n\n")
            f.write(f"Designer `python -m cooksim_bench.suite.design_composite_v5 --prev {a.prev} --seed {a.seed}`. Old = the previous suite's "
                    f"errors, new = this run's; 'why' names the rule that removed each old error (or diversity). The "
                    f"rules are in the designer's module docstring ({'round-3' if ROUND3 else 'round-2'} block).\n\n")
            f.write((LEDGER_R3_NOTES if ROUND3 else LEDGER_R2_NOTES) + "\n\n")
            f.write("| case | old errors | new errors | why each dropped old error went |\n|---|---|---|---|\n")
            for n, o, nw, why in table:
                f.write(f"| {n} | {'<br>'.join(o)} | {'<br>'.join(nw)} | {'<br>'.join(why)} |\n")
    n_err = sum(c["n_errors"] for c in out_cases)
    print(f"wrote {a.out}: {len(out_cases)} cases, {n_err} errors ({sum(c['n_ruinous'] for c in out_cases)} ruinous), "
          f"{sum(1 for c in out_cases for s in c['subs'] if s['family'] == 'qa_probe')} questions, "
          f"{sum(1 for c in out_cases for s in c['subs'] if s['family'] == 'qa_proactive')} requests, "
          f"{sum(1 for c in out_cases for s in c['subs'] if s['family'] == 'plan_change')} plan changes; problems {problems}")
    print("families", dict(fam_count))


LEDGER_R2_NOTES = """Round-2 decisions and the limits the rules hit (FIX_PLAN_R2 D1-D8, user rulings of 2026-09-19):

* **leave_unattended is not ruinous.** The >=half-ruinous rule is kept where it is reachable (easy, medium: 1 of 2).
  It is **relaxed to >=1 ruinous** on hard (3 errors), expert and master (4 errors): with one error label per case,
  Fish & Chips and Steak Frites offer only two ruinous labels (add_foreign, and a premature serve after the first item),
  and C1/C2+/B forbid them together with a third mistake except in one set per tier; on the burrito every 2-ruinous set
  is built on the steak/tomato swap. The designer still prefers more ruinous errors when it can.
* **Walk-away floors (E9).** Explicit `prep_floor`/`plate_floor` on every walk-away: none for the first item lit (burger
  meat, fish, Steak Frites steak, burrito rice); one potato in processing for the potato walk-away (LLM cooks fry the
  potatoes one at a time: a two-potato floor fired 1/6 in the pilot). The burrito steak walk-away is dropped: the rice
  smokes from t67-68 and the steak is done t73-109, so the smoke ban held it 0/8 in the pilot; the rice walk-away is
  kept in at most 6 of the 10 master cases (D6 is met by a cap, not by absence).
* **Maps.** Every kitchen now has two off-recipe crates (the second is cheese where no heat station takes it, else
  onion) and two cutting boards (the Fish & Chips medium map had one: a mis_prep's leftover blocked it and the cook
  stalled until the episode timed out).
* **Duplicates (D2).** No two cases of one kitchen share a design where any alternative exists. The burrito hard map
  (hard_2, hard_4) and Steak Frites / Fish & Chips (same design on the two maps of a tier) repeat because the rules leave
  too few sets; they differ in the foreign item and the kitchen.
* **Questions (D8).** Count and visit questions only once the plate is started; order questions only about the pairs
  the LLM cooks do in both orders (VARIABLE_ORDER); no after_ref; visit truths balanced; no next-step question in a
  forget case. Cases whose last mistake is a premature serve after the first item keep one question (nothing else fits
  before the serve)."""


LEDGER_R3_NOTES = """Round-3 decisions (FIX_PLAN_R3 D9-D17 and user ruling R10; engine fixes F1-F10 in python -m cooksim_bench.e2e_stepin_rollout).
The rules are in the designer's module docstring (round-3 block). Pilots: 39 Gemini Flash episodes (silent assistant)
in tmp/suite_v5r3/llm_pilot{1,2,3}; scripted verifier runs in tmp/suite_v5r3/verify_try*.

* **Engine (R10, F1).** The cook gets a turn at every takeover release; a walk-away's displaced queue is no longer put
  back after its commit, and a kept segment that needed a dropped fetch is dropped too. Pilot: every release had a
  cook turn; the hard/expert walk-aways were all rescued (the fish/steak/potato plated before a burn).
* **Burrito (D17).** With the cook turning at the release, the baseline Flash cook still burned the rice in 6 of 7
  burritos whose early takeover sat in the rice window (the rice walk-away released at t67-68, the steak mis_prep
  t25-68): on its release turn it finished the steak prep. Both are gone from the burrito (row rule D17). The steak
  walk-away (floor: rice plated) never met its +-6-tick near-done window in the pilot (the steak is done 18-22 ticks
  before the rice is plated, 0/7), so it may fire up to 20 ticks after done (`lu_late`, 5 ticks before the smoke):
  4/5 fired in pilot 3, no burn. Three of five hard-map baseline burritos burned with NO takeover before t126 (pilot 2:
  the cook plates the rice after t126 on the long map) — a natural burn, accepted by ruling.
* **Burrito: one error pattern.** With the rice window closed, D13 (no forget after the tomato beside a lettuce
  mistake), C2 (a premature serve is last), B (a raw/replaced item blocks later platings) and CAP (5 slots) leave one
  4-mistake set per kitchen: {steak walk-away, forget after the steak, mis_prep(tomato -> cheese), and one robust
  ruinous: premature serve after the tomato, or a foreign / misidentified item at the lettuce}. The cases differ in
  the ruinous mistake, its foreign item, the questions and the request. Cheese is the only choppable crate no heat
  station takes in both burrito kitchens, so every burrito mis_prep uses it. In pilot 3 the tomato mis_prep was lost
  twice to the steak walk-away's smoke (the cook chops the tomato while the steak smokes; the smoke ban holds it).
* **Steak Frites (expert).** Four mistakes need four labels, and after D11 (no foreign item at the steak) and C2 the
  only labels left are mis_prep, forget, walk-away and the robust ruinous one (serve-time foreign item 8, premature
  serve after the steak 2): every expert case has all four; they differ in the item (steak / potato), the forget's
  anchor and the foreign item.
* **Fish & Chips (hard).** The robust ruinous mistake is the serve-time foreign item in all ten (a premature serve
  after the fish conflicts with every third mistake under C2); mis_prep(potato) is in 6/10 (only two non-mis_prep sets
  per kitchen pass the walk-away gap rules).
* **Burger (medium).** Four admissible sets per kitchen (the meat walk-away with premature / foreign / misidentify, or
  the meat omission with the serve-time foreign item); the bun/cheese swap is gone (D9: its takeover covers the meat's
  cook and the next plating moment is lost, round-3 pilot medium_m1). Cases that open with the walk-away share their
  first question (next step: the only kind askable before it).
* **Questions (D10, D15).** Each question sits at its LIVE moment (plate floor + prep floor + its own precondition),
  pushed past every takeover (VOICE_QUIET before, release + G after) and kept only if that moment comes before the
  serve / a premature serve (the serve-time foreign item holds the voice from the moment one item is left). A case may
  therefore carry 0-2 questions. Gone: longest_on, burn_first (one candidate), visit questions whose answer the recipe
  decides, "what did you tell me" before anything happened; past_done asked once the first cooked item can be done.
  Requests are made before a takeover that becomes ready on the same tick (engine)."""


LEDGER_R5_NOTES = """Round 5 (2026-09-20): the four DESIGN items the round-4 hand review left open
(docs/FIRING_AND_GRADING.md section 6, "Left to the designer"), plus every placement the round-5 real-cook run and
the scripted verifier showed cannot fire or commits a second, unnamed deviation. Engine: the round-5 fixes.
`DESIGN_ROUND=4` restores the round-4 suite exactly.

**The one thing round 4 got wrong about its own engine.** The scheduler is reference-free: every pending sub-case
is tested each tick and the one that became READY FIRST takes the body slot. The designer, however, ordered a
case's mistakes by a MOMENT read off a reference table. Where the two disagreed, a later-designed sub-case fired
first and the earlier one waited for a state the episode never came back to. In `b3_expert_nops_medium_map_5`
baseline the walk-away fired at t64 and the premature serve at t111 while `serve_raw{potato}` sat behind
`prep {potato: 1}` from t0: the cook plated the steak and served without ever chopping a potato.

**H2 - floors that dominate.** Every sub-case now has an EFFECTIVE floor: its stated `(prep_floor, plate_floor)`,
widened by what its own type waits for (a forget after `assemble(x)` needs the plate through x; a prep repeat
needs that prep; a walk-away needs its food prepped and lit) and by what the plate floor implies (to plate k
items the cook has prepped them). Both quantities only grow in the engine (`max_plate_len`, `prep_max`), so "B's
floor dominates A's" means B cannot become ready before A. The rule is HARD for a sub-case that ENDS the dish - a
premature serve, one whose floor is a plate that already holds everything, or one in the last single slot -
because no episode is left behind it. For the rest a pair out of order only swaps two takeovers, and the
assignment pays four points per such pair instead of refusing the set.

**H6 - consumption.** A sub-case may not wait for prep that the case's own earlier mistake destroys. Round 4 had
this for serve_raw / misidentify / omit (rule SR+). Round 5 adds `mis_prep{x}` - measured, not assumed: in
`b3_easy_nops_medium_map_3/4/5` the classic_expert cook chopped the lettuce it was handed at t1, plated it and
served, `prepped {lettuce: 17}`, tomato never - and `wrong_order{a, x}`, whose block chops and plates x itself. A
mistake that itself waits behind `prep(x)` does not consume it: `serve_raw{potato}` behind `potato: 1` scoops an
unfried potato AFTER the cook has chopped one, so a later sub-case may still wait on that prep. A walk-away's own
food counts as a stated wait, because it cannot be near done on the heat unless it was prepped.

**The prep slip is placed where the prep is.** `mis_prep`'s moment is now read off the prep it replaces
(`chop(x) - 0.5`) instead of the first item's done moment. On Fish & Chips and Steak Frites that is moment 1.5,
so the M gap makes it the case's first mistake in practice - which is what the five starved instances of the
round-5 run needed (`the cook is not reaching for the potato yet`, `b3_hard_nops_hard_map_3/4/5`). On the burrito
the tomato is chopped after the steak is plated, so a prep slip there may legitimately follow other mistakes.

**H9 - no unlabelled second mistake.** Every recipe in the suite is `ordered`. A `serve_raw{x}` at a plate floor
below x's own index puts x on the plate before the items that belong in front of it, so it commits a
`wrong_order` that no sub-case names, and an assistant that says "that is out of order" is right about the world
and wrong about the ledger (6 instances on Steak Frites). Its plate floor is now x's own index - which is also
what makes the raw unit legible: the steak is down, the potatoes are in the fryer, and a raw one goes on. Two
consequences: the B3 margin between that floor and the item becoming ready is now counted in TICKS (12) through
the tier's own scale, because the five moment tables are not on one scale (0.6 of an expert moment is 16 ticks
and refused every second-stage raw serve); and a raw serve may not take the LAST single slot, where "everything
else plated" is one scoop from a finished dish (`serve_raw{lettuce}` on the burrito: 8 of 9 scripted runs
archived `the cook's own serve is already queued`).

**R5-F - a forget must keep a step that is not the serve.** Four episodes ended with `the only step left is the
serve (a forget never forgets the serve)`. A forget whose floor is a plate one scoop from complete, or any forget
behind a plating takeover on the two-item Caprese, is refused at design time.

**R5-RP - a prep repeat may not leave a second unit cooking.** B5 banned a prep repeat whose node LIGHTS a
station. That was too narrow: `prep(potato)` on Fish & Chips and Steak Frites, and `prep(rice)` / `prep(steak)` on
the burrito, END by putting the unit into a fryer, pot or pan that `ignite` has already lit, so the repeat leaves
a SECOND unit cooking that no sub-case names. All seven "food smoked inside a takeover" events of the scripted
verifier were one of these (a second steak or fish in the pan), and the 48-tick rice repeat of
`b3_master_nops_hard_map_5` burned the kitchen in two of three cooking orders. Such a repeat is now allowed only
where the dish goes out before the duplicate can smoke - within 35 ticks of the serve, the unit's cook plus the
25-tick smoke grace - which keeps the potato repeat and retires the fish, steak and rice ones.

**R5-LU - the burrito walk-away is retired.** 0 of 6 in the round-5 run and 0 of 8 in the round-2 pilot: its floor
is "the rice plated", and by the time the rice is plated the steak is off the heat (`no steak cooking on the heat
yet`). Master carries no walk-away; the rice walk-away had already gone in round 3 (D17).

**H7 - a question is not asked on the tick its referent event completes.** Measured on the round-5 run: 16 of the
21 `order` asks had `asked_at == t_b`. A prep floor cannot fix it, because an order question's precondition IS
its two events. A PLATE floor can: the cook plates in recipe order and cannot plate item k without having prepped
and cooked it, so "the plate through the latest item the question names" is reached strictly after both events,
with that item's cook and walk as the gap. `order` now needs the plate through max(index of its two components)
and `after_ref` the referent plus two. It costs the two families most of their instances (Q1 13 -> 4, Q7 7 -> 3),
because on the three-item recipes that floor is the last plating but one: the alternative is an engine rule that
holds a question a fixed number of ticks after its referent, which is a ruling, not a placement.

**H8 - diversity, and what it cost.** One placement may appear in at most 4 of a tier's 10 cases (round 4: 6
suite-wide, with four expert placements in 5 of 10); the penalty is linear, because on Steak Frites the
serve-time foreign item is in nearly every admissible set and a squared term there swamps every choice the
assignment can actually make. Two cases of one KITCHEN may never share a design. Two cases of a tier may, and pay
3,000 for it. The per-type band (10-17 of the 140, 7-12%) is near-hard at 8,000 a unit, and the assignment is now
an iterated local search - a descent, then 60 seeded kicks - because the band, the reuse limit and the placement
cap are coupled and a single descent stops one over the ceiling.

**Structural limits (round 4's list still holds; what round 5 adds).**
* Steak Frites is the narrowest cell in the suite: three items, one early moment, H9's in-order raw serve and
  R5-RP's ban on the steak repeat leave **eight** admissible four-mistake designs for ten cases. Every one of
  them carries the serve-time foreign item or the walk-away, `premature_serve` no longer fits beside three other
  mistakes, and four expert designs are used twice - never twice in the same kitchen.
* The Caprese is the second narrowest: eleven two-mistake designs for ten cases.
* The suite therefore carries **eight** pairs of cases that share a mistake design across the two kitchens of
  their tier (round 4 had none). They differ in map, off-recipe crates, wrong items and questions. The
  alternative measured at the same seed was a suite with unique designs and three error types outside the band.
* `mis_prep` is gone from Fish & Chips and Steak Frites in any case that later waits on that prep, and from the
  burrito wherever an earlier mistake consumes the tomato or the lettuce.
* Plan changes fall from 11 to 7 and questions from 94 to 87: the round-5 sets are denser and later, which
  leaves fewer anchors clear of every takeover, and H7 removes most `order` and `after_ref` candidates.
"""


LEDGER_R4_NOTES = """Round 4 (2026-09-20): the suite rebuilt from scratch for BALANCE, on the fire-longtail engine
(main a9285b1, every plating mistake FORCED, any sub-case may state its own prep/plate floor).

User ruling: "\u6ce8\u610f\u4e00\u4e0b number of each error type \u5728\u6574\u4e2a benchmark \u4e2d\u7684 balancing \u2026 \u957f\u5c3e\u7684 error type \u90fd\u633a\u6709\u610f\u4e49\u7684" \u2014 balance the
number of each error type across the whole benchmark; the long-tail types are meaningful. Round 3 put 84% of its
150 mistakes in four types and left serve_raw and repeat_step at zero, wrong_order at one.

**What made the long tail placeable.** Until 2026-09-20 every plating mistake waited for three things at once: the
plate already in the cook's HANDS, the component being the cook's OWN next plating, and the cook walking there. A
self-planning LLM cook cooks everything before it fetches the plate, so serve_raw had no moment at all. The engine
now FORCES the block (it parks the hands, fetches the dish plate, walks to the source and scoops), and each
sub-case may state a `prep_floor` / `plate_floor` \u2014 a progress floor counting the most units ever prepped and the
most items ever on the plate, right or wrong. Round 4 states both on every forced mistake and reads the mistake's
moment off that floor, which is what lets the same type be placed early in one case and mid-dish in another, in
any cooking order.

**Errors per case by tier** 2 / 2 / 3 / 4 / 3 (the user's counts; master down from 4 to open up patterns) = 140
mistakes, so the per-type quota is 14, not 15. Accepted band 10-20, and no type is in more than 8 of a tier's 10
cases, so no tier has a type in every case.

**Structural limits (the targets that could not be met exactly, and why).**
* `omit_subgoal` and `wrong_order` are impossible on Fish & Chips and Steak Frites. Both recipes are
  first-cooked-item + two potatoes: an omission plates the NEXT recipe item, which is a potato that is only ready
  after the fish / steak is already on the plate (the omission's own slot has closed), and a swap needs a second
  item that is raw or a chop the block can make itself \u2014 a cooked potato is neither. They are carried by the
  Caprese, the cheeseburger and the burrito.
* `mis_prep` is impossible on the cheeseburger (the recipe never chops the meat, so replacing its prep is a
  different process, not a confusion at that step) and on the burrito's steak and rice (D17: an early takeover in
  the rice window burns the dish).
* `leave_unattended` is impossible on the Caprese (nothing is cooked).
* `forget_next_step` on the cheeseburger has one anchor only (after the meat is plated): the patty is on the heat
  from the first minute and the engine refuses a 60-tick freeze beside food that would smoke inside it.
* On the 5-slot burrito only the LAST item may be replaced or duplicated ON THE PLATE (a sixth scoop is refused
  and the cook loops), so its foreign / wrong-item mistakes sit at the lettuce and its duplicates are PREP
  repeats, which cost time and no slot.
* A **plan change** cannot go on the burrito at all: its five components already fill the plate, so "put X on top
  before serving" has nowhere to land. The intent library holds 12 announced changes over the other eight
  kitchens (one, a pepper grinder, is not a food on the plate), which is why round 4 has 11 and not more.
* **Q6 (reask)** is not generated. Its truth is "has the answer changed since you last asked?", which needs an
  earlier ask of the same kind in the same episode and a truth that moved; neither can be screened at design time
  (the hardness rule rejects every instance as "nothing changed since the first ask"). It stays at zero.
* **Q1 (order)** cannot be asked on the Caprese or the cheeseburger: every pair of their events is fixed by the
  recipe (the validator's T1), except plate-order pairs, which only a swap in the same case makes uncertain.
  **Q7 (after_ref)** needs the referent plus one more plating while the plate is still two items short of
  complete, which only the five-item burrito allows \u2014 so it is capped at one per burrito case.

**What the verification changed (in order).**
1. *Scripted verifier, first pass.* A swap that plates the recipe's LAST item completes the plate, so a forget /
   premature serve / repeat that waits for "the plate through the second-to-last item" met a finished dish (the
   round-2 WO rule, re-added). A slot mistake in the FIRST component's slot fired before the cook had prepped
   anything and left the cook's own plan walking to a board for a tomato that was never chopped (B5b). Food the
   takeover had put out of the cook's reach began to smoke inside the block (engine: `stranded_by_takeover`).
2. *Real-cook pilot, 100 Gemini Flash episodes.* `wrong_order` fired 9 times in 24: at plate_floor = idx(a) the
   floor is reached as the cook plates a, and the burrito cook plates 10-17 ticks apart \u2014 the swap lost the race
   15 times ("the cook's own serve is already queued"). Its floor is now one plating step earlier (B3b). Six easy
   sub-cases waited all episode for a prep a serve_raw of the same item had replaced (SR+, generalised).
   after_ref asked for two platings after the referent, which on the burrito is only reached as the plate
   completes (12 of 16 asks lost): one is enough. The stranded-food hold was made exempt from the 25-tick budget,
   as E10's plate path already is, because it was expiring INSIDE the block.
3. *Self-review of 11 transcripts, all five tiers (S1-S6).* A question's prep floor asked for the whole batch
   ("potato: 2") where its truth needs the event once; a question with no floor at all was asked at tick 0
   ("how many boards are free" before the cook had moved); a forget 0.6 of a moment before a premature serve was
   blocked by the queued serve; on the five-item burrito a slot mistake placed at its own index lost the same
   race the swap did; an omission whose next item is cooked reached its moment as the cook plated the item it
   was meant to skip; and on the two-item Caprese any "after X" mistake behind a plating one meets a complete
   plate one scoop later.

**Forced blocks and the plate in hand** (user check, 2026-09-20). Measured on the real-cook run, not read off the
code: of 48 forced plating mistakes that fired, 18 fired with the dish plate already in the cook's hands and
items already on it, and every one of those blocks walked straight from where the cook stood to the source and
scooped \u2014 no plate fetch, no second plate, no park. The only blocks that park are the two that physically need
the hands: a PREP repeat (it picks a raw unit up) and a swap that has to chop its own second unit, and both
retrieve THAT plate ("go to the counter with the plate"), never a fresh one. No trigger refuses because the
plate has items on it: every plate-related wait note is a progress floor asking for MORE, the mistake's own slot
already correctly filled, or the 5-item capacity.
"""


def D3_desc(r):
    from cooksim_bench.suite import gen_error_inventory as GE
    return GE.DESC[r["fn"]](r["kwargs"])


if __name__ == "__main__":
    main()
