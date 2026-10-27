"""Behavior library — the behavior-event taxonomy realized as named functions over the plan IR.

Each behavior function edits the SUBGOAL trajectory (via plan_ir transforms) and returns the
new IR plus ground-truth Deviation records. Functions compose through `compose(ir, calls)`,
which applies them IN ORDER with the priority rule: an earlier error owns the plan, and a later
error whose anchor it removed is skipped-with-record rather than raised.

CHANNEL STRUCTURE — every behavior belongs to exactly one of three channels:

  action_only        Pure physics. The behavior is visible only through the cook's motion and
                     the world state; there is nothing to say and no question that would avert
                     it. A supervisor can only catch it by WATCHING, and (in the interaction
                     stage) avert it by stepping in proactively BEFORE the wrong action runs.

  language_only      Pure utterance. The cook physically just PAUSES for the utterance window;
                     the physical plan is either untouched or changed in a LEGAL way that the
                     announcement itself makes legitimate. Nothing to avert, nothing to flag.

  language_optional  The error has a moment of doubt at its trigger boundary, and the function
                     takes `ask: bool`.
                       ask=True  -> the cook PAUSES and asks the doubt out loud (a structured
                                    lang_event with engine ground truth). If the VLM answers
                                    CORRECTLY and in time, the error DOES NOT HAPPEN (the
                                    `avertable` record says exactly which edit to undo); a
                                    wrong or missing answer lets it happen as scheduled.
                       ask=False -> silent. The error happens unless the VLM proactively steps
                                    in before the wrong action executes and rewrites the later
                                    subgoals/commands itself.
                     Today, with no VLM in the loop, the error always plays out; the records
                     carry everything the interaction stage needs to make the branch real.

Every record's `detail` carries `channel`, and for language_optional additionally `asked` and
`avertable` (the undo recipe a correct answer triggers).
"""
from __future__ import annotations
from typing import List, Tuple

from .plan_ir import (Subgoal, Deviation, segment_plan, compile_ir, flat_commands,  # noqa: F401
                      skip, insert, swap, repeat, replace, truncate_and_serve, permute_prep,
                      forget_rest, foreign_scoop, raw_scoop, wander_node, detour_node,
                      stall_node, busywork_node, trash_node, fetch_plate_node, prep_of,
                      _station_label, _idx)

IR = List[Subgoal]
Result = Tuple[IR, List[Deviation]]

PAUSE = 4          # default utterance window (ticks the cook stands still while speaking)


def _chan(recs, channel, asked=None, avertable=None):
    for r in recs:
        r.detail["channel"] = channel
        if channel == "language_optional":
            r.detail["asked"] = bool(asked)
            r.detail["avertable"] = avertable
    return recs


def _unit_sid(ir: IR, sid: str) -> str:
    """F9 (round-3 review, 2026-09-19): the live remaining plan names a duplicated ingredient's later units
    `assemble(potato)#2`, so once the first potato is plated `assemble(potato)` no longer exists and a mistake on
    "the potato" could not be composed at all ("cannot compose on the live kitchen", expert c medium_3: the case's
    only ruinous error never fired). Resolve to the first remaining unit; the exact sid wins when it exists."""
    if any(n.sid == sid for n in ir):
        return sid
    base = sid.split("#")[0]
    for n in ir:
        if n.sid.split("#")[0] == base:
            return n.sid
    return sid


def _sid_component(sid: str) -> str:
    sid = str(sid or "")
    return sid.split("(")[1].split(")")[0].split("#")[0] if "(" in sid else ""


def _prev_sid(ir: IR, sid: str) -> str:
    """The subgoal completed just before `sid` begins — the boundary where a doubt about
    `sid` naturally surfaces. `__start__` if `sid` is the first subgoal."""
    i = _idx(ir, sid)
    return ir[i - 1].sid if i > 0 else "__start__"


def lang_events_from(records) -> list:
    """Collect the language events that behavior functions attached to their records, ready
    to pass as MultiRollout(lang_events=...)."""
    return [r.detail["lang_event"] for r in records
            if not r.skipped and isinstance(r.detail, dict) and "lang_event" in r.detail]


# =====================================================================================
# LANGUAGE-OPTIONAL ERRORS — take ask: bool; a correct answer averts the error
# =====================================================================================
import os as _os
import random

# PER-ITEM repeat ban (user ruling 2026-09-20: "repeat_step: allow it again wherever the duplicate is visible as a
# duplicate — the bun-on-top confusion is a per-item ban, not a type ban"). A second BUN on a burger does not read
# as a duplicate at all: it reads as the burger's top bun, so an assistant that stays silent is right and the
# episode is unscorable. Every other item's duplicate is plainly one item too many. The whole TYPE used to be
# refused for this one item's sake, and the 150-error suite ended up with 0 repeat_steps (v4 had 7).
# COOKSIM_REPEAT_ITEM_BAN overrides the list (comma-separated; empty string bans nothing).
REPEAT_DUPLICATE_INVISIBLE = {x for x in (_os.environ.get("COOKSIM_REPEAT_ITEM_BAN", "bun") or "").split(",") if x}


def err_repeat_step(ir: IR, sid: str, at: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """REPEAT_COMPLETED_STEP — after finishing subgoal `at`, the cook has lost track of having
    already done the earlier subgoal `sid` and goes back to re-do it. Never consecutive: `sid`
    must be strictly earlier than the trigger `at`. A repeated assembly step DOUBLES an
    ingredient -> serve rejected; repeating a non-plating subgoal wastes time but still wins.
    ask=True: at the trigger the cook asks "did I already do <sid>?" (gt: yes, at tick T);
    a correct answer CANCELS the duplicate subgoal — only the pause is lost."""
    if sid.startswith("assemble(") and _sid_component(sid) in REPEAT_DUPLICATE_INVISIBLE:
        raise KeyError(f"design rule: a second {_sid_component(sid)} does not read as a duplicate "
                       f"(REPEAT_DUPLICATE_INVISIBLE)")
    ir, r = repeat(ir, sid, at)
    avert = {"undo": "cancel_insert", "target": r.anchor}
    if ask:
        r.language_hook = "ASK_PROGRESS_QA"
        r.detail["lang_event"] = {"after": at, "subtype": "ASK_PROGRESS_QA", "pause": pause,
                                  "params": {"about_sid": sid}}
    return ir, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_premature_serve(ir: IR, after: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """SKIP_PREREQUISITE — believes the dish is done after `after`: abandons the rest and
    serves an incomplete plate -> wrong_serve.
    ask=True: before heading to the pass the cook asks "am I done — can I serve this?"
    (gt: NOT complete, plus the missing subgoals); a correct answer RESTORES the dropped
    tail and the episode completes normally."""
    ir, r = truncate_and_serve(ir, after=after)
    avert = {"undo": "restore_tail", "dropped": list(r.detail.get("dropped", []))}
    if ask:
        r.language_hook = "ASK_IF_DONE"
        r.detail["lang_event"] = {"after": after, "subtype": "ASK_IF_DONE", "pause": pause,
                                  "params": {"gt_missing": list(r.detail.get("dropped", []))}}
    return ir, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_forget_next_step(ir: IR, after: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """FORGET_NEXT_STEP — loses the plan after `after`. Silent: just stands there, LOST ->
    timeout (or FIRE, if something was left cooking). The record keeps the full restorable
    tail and specifically the forgotten next step.
    ask=True: instead of freezing silently the cook asks "what do I do next?" (gt: the
    forgotten next subgoal + command); a correct answer RE-APPENDS the restorable tail —
    without one the cook stands lost exactly like the silent version."""
    ir, r = forget_rest(ir, after=after)
    # DESIGN RULE (user, 2026-09-15): forget_next_step never forgets the SERVE. A cook standing next to a finished plate that it 'forgot' to hand in is not a plausible slip; the forgotten step must be a real production or plating step (a prep, an ignition, fetching the plate, a non-final assemble). On hot recipes a freeze with food on the heat burns — that is leave_unattended, not a forget — so where a cell has no admissible mid-recipe point the error is simply unavailable there; never fall back to the point before serve.
    if not r.detail.get("next_step") or (r.detail["next_step"] or {}).get("subgoal") == "serve":
        raise KeyError("design rule: forget_next_step would forget the serve")
    avert = {"undo": "restore_tail", "tail": list(r.detail.get("restorable_tail", []))}
    if ask:
        r.language_hook = "REQUEST_NEXT_STEP"
        nxt = r.detail.get("next_step") or {}
        r.detail["lang_event"] = {"after": after, "subtype": "REQUEST_NEXT_STEP", "pause": pause,
                                  "params": {"gt_next_subgoal": nxt.get("subgoal"),
                                             "gt_next_command": nxt.get("command")}}
    return ir, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_omit_subgoal(ir: IR, sid: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """OMISSION — a required subgoal never happens: the cook believes it is already done and
    the plan continues without it -> usually wrong_serve.
    ask=True: at the boundary where `sid` should begin, the cook asks "did I already do
    <sid>?" (gt: NO, never completed); a correct answer REINSTATES the subgoal in place."""
    boundary = _prev_sid(ir, sid)
    ir2, r = skip(ir, sid)
    r.event_subtype = "OMISSION"
    avert = {"undo": "reinstate_subgoal", "sid": sid, "after": boundary}
    if ask:
        r.language_hook = "ASK_PROGRESS_QA"
        r.detail["lang_event"] = {"after": boundary, "subtype": "ASK_PROGRESS_QA", "pause": pause,
                                  "params": {"about_sid": sid}}
    return ir2, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_wrong_order(ir: IR, a: str, b: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """WRONG_ACTION (ordered recipe) — swaps the assembly subgoals of components `a` and `b`;
    on an ordered recipe the serve is rejected. (On an UNORDERED recipe the same swap is a
    LEGAL reordering — there it belongs to the must-not-flag class, not the error class.)
    ask=True: at the boundary before the first swapped slot the cook asks "which goes on
    next — <a> or <b>?" (gt: whichever the recipe order actually wants); a correct answer
    UNDOES the swap."""
    sa, sb = f"assemble({a})", f"assemble({b})"
    first = sa if _idx(ir, sa) < _idx(ir, sb) else sb
    boundary = _prev_sid(ir, first)
    gt_next = first.split("(")[1].rstrip(")")
    ir2, r = swap(ir, sa, sb)
    avert = {"undo": "undo_swap", "pair": [sa, sb]}
    if ask:
        r.language_hook = "ASK_WHICH_NEXT"
        r.detail["lang_event"] = {"after": boundary, "subtype": "ASK_WHICH_NEXT", "pause": pause,
                                  "params": {"options": [a, b], "gt_next": gt_next}}
    return ir2, _chan([r], "language_optional", asked=ask, avertable=avert)


# =====================================================================================
# ACTION-ONLY ERRORS — pure physics; avertable only by proactive intervention
# =====================================================================================
def err_misidentify_assemble(ir: IR, component: str, wrong: str) -> Result:
    """MISIDENTIFY_OBJECT (assembly) — scoops `wrong` from its source instead of the
    intended component (e.g. lettuce instead of cheese) -> wrong_serve."""
    ir, r = replace(ir, _unit_sid(ir, f"assemble({component})"), foreign_scoop(wrong))
    r.event_subtype = "MISIDENTIFY_OBJECT"
    return ir, _chan([r], "action_only")


def err_mis_prep(ir: IR, component: str, wrong: str) -> Result:
    """MISIDENTIFY_OBJECT (prep) — fetches and PREPS the wrong ingredient at the SAME
    station the original component used (cook or chop, inherited from the original prep
    block); the later assembly scoop faithfully plates the wrong prepared item.

    Station-typed cooking: if the original heat station refuses `wrong` (e.g. a pan
    won't take lettuce), the confused cook still preps it the only way the kitchen
    allows — on the cutting board — and the real component never gets made."""
    from .enums import Terrain, station_accepts, choppable
    orig = ir[_idx(ir, f"prep({component})")]
    cmds = [orig.commands[0].replace(f"{component} source", f"{wrong} source")] + list(orig.commands[1:])
    _LBL = {"pot": Terrain.POT, "pan": Terrain.PAN, "oven": Terrain.OVEN, "fryer": Terrain.FRYER}
    _lbl_of = next((w for c in orig.commands for w in c.split() if w in _LBL), None)
    station = _LBL.get(_lbl_of) if _lbl_of else None
    # round-3 audit (2026-09-11): the substituted block inherited the ORIGINAL component's shape,
    # including a chop the wrong ingredient may not take ("you can't chop an egg — crack it
    # straight into the pan"), so the takeover issued illegal commands. Build the confusion out of
    # steps this kitchen actually allows, and refuse the pairing when none exist.
    if station is not None and station_accepts(station, wrong, "raw"):
        if "chop" in orig.commands and not choppable(wrong):
            cmds = [f"go to the {wrong} source", "pick up", f"go to the empty {_lbl_of}", "put down"]
    elif choppable(wrong):
        # "the cutting board" grounds to ANY board, including an occupied one, and the put-down is
        # then refused (round-3 audit): the template's own prep says "the EMPTY cutting board"
        cmds = [f"go to the {wrong} source", "pick up", "go to the empty cutting board",
                "put down", "chop"]
    else:
        raise KeyError(f"mis_prep({component} -> {wrong}): {wrong} can be neither chopped nor "
                       f"cooked at {component}'s station in this kitchen")
    node = Subgoal(f"prep({wrong})", "prep", wrong, cmds, origin="injected", deviated=True)
    ir, r = replace(ir, f"prep({component})", node)

    # LATE-IGNITE recipes (Fish & Chips, Steak Frites, Loaded Burrito) split prep into
    # "load the station" and a separate ignite(x) that lights it later. Replacing prep(x)
    # alone left ignite(x) pointing at a station that now holds nothing: "go to the fryer with
    # the fish" cannot ground, and the whole tail of the plan collapses behind it — measured at
    # 9 illegal-skipped / 5 rejected / 2 failed and NO outcome at all, on every late-ignite
    # recipe. A confused cook lights the station it actually loaded.
    ig = next((n for n in ir if n.sid == f"ignite({component})"), None)
    if ig is not None:
        if station is not None and station_accepts(station, wrong, "raw"):
            ir = [Subgoal(f"ignite({wrong})", "ignite", wrong,
                          [c.replace(component, wrong) for c in n.commands],
                          origin="injected", deviated=True) if n is ig else n
                  for n in ir]
        else:
            # it went to the cutting board instead, so there is nothing to light
            ir = [n for n in ir if n is not ig]

    r.event_subtype = "MISIDENTIFY_OBJECT"
    return ir, _chan([r], "action_only")


def err_add_foreign(ir: IR, wrong: str, before: str) -> Result:
    """MISIDENTIFY_OBJECT / extra item — an off-recipe ingredient is scooped in ADDITION
    -> wrong_serve."""
    ir, r = insert(ir, foreign_scoop(wrong), before=_unit_sid(ir, before))
    r.event_subtype = "MISIDENTIFY_OBJECT"
    return ir, _chan([r], "action_only")


def _kitchen_stops(ir: IR, exclude: str, avoid=()) -> List[str]:
    """Real places the cook can plausibly busy itself at, derived from the recipe itself.

    H10 (round-5 review, M6): `avoid` drops the crates of ingredients the cook has an ANNOUNCED intent about.
    In medium_m1 the announced plan change was "add some chopped onion on top before serving" and the walk-away
    wander began "go to the onion source, idle 8" — the cook standing at the onion crate is exactly what it just
    said it would do, so the walk-away was camouflaged by its own episode."""
    stops = [f"{n.component} source" for n in ir if n.type in ("prep", "assemble")
             and n.component and n.component != exclude and n.component not in set(avoid or ())]
    stops = list(dict.fromkeys(stops)) + ["plates", "serving pass"]
    return stops


def err_leave_unattended(ir: IR, component: str, loops: int = 2, idle: int = 8,
                         seed: int = 0, avoid=()) -> Result:
    """LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED — the cook gets on with OTHER kitchen business
    (visiting stations, checking things) while the lit station runs through its whole warning
    arc behind it: cooked -> 60 ticks of SMOKE -> FIRE -> 15 ticks -> IRRECOVERABLE failure.
    The smoke phase is the supervisor's proactive-warning window.

    H10 (round-5 review): the wander used to be one fixed circuit walked twice with exactly `idle` ticks at every
    stop — "go to the X source / idle 8 / go to the plates / idle 8 / go to the serving pass / idle 8", twice,
    identically in all nine instances of a tier. Three reviewers independently called it a script rather than
    distraction. The circuit is now rotated (and the second loop reversed) per episode from `seed`, the pauses
    are jittered around `idle`, and the trash is a stop like any other. The TOTAL length is kept within a tick or
    two of the old block's, because the walk-away's cost in ticks is what makes it commit at the smoke onset."""
    stops = _kitchen_stops(ir, component, avoid=avoid)
    if not stops:
        stops = ["plates", "serving pass"]
    import zlib          # a STABLE seed: str.__hash__ is salted per process and films are replays
    rnd = random.Random((seed or 0) ^ (zlib.crc32(("|".join([component or ""] + stops)).encode()) & 0xFFFF))
    rot = rnd.randrange(len(stops))
    circuit = stops[rot:] + stops[:rot]
    legs, budget = [], idle * len(stops) * max(1, loops)
    for k in range(max(1, loops)):
        leg = circuit[k % len(circuit):] + circuit[:k % len(circuit)]
        if k % 2:
            leg = list(reversed(leg))
        if legs and len(leg) > 1 and leg[0] == legs[-1]:
            leg = leg[1:] + leg[:1]              # never two stops at the same place in a row
        legs += leg
    # jitter each pause by up to a third of `idle`, then spread the rounding remainder one tick at a time so the
    # block still costs the ticks it always did and no single stop becomes a conspicuous outlier
    pauses = [max(2, idle + rnd.randint(-(idle // 3), idle // 3)) for _ in legs]
    k = 0
    while sum(pauses) != budget and k < 8 * len(pauses):
        i = k % len(pauses)
        pauses[i] += (1 if sum(pauses) < budget else (-1 if pauses[i] > 2 else 0))
        k += 1
    cmds = []
    for stop, pz in zip(legs, pauses):
        cmds += [f"go to the {stop}", f"idle {pz}"]
    node = busywork_node([], idle)
    node.commands = cmds
    ir, r = insert(ir, node, before=_unit_sid(ir, f"assemble({component})"))
    r.event_subtype = "LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED"
    return ir, _chan([r], "action_only")


def err_serve_raw(ir: IR, component: str) -> Result:
    """Food-safety violation — plates `component` RAW from its source, skipping prep
    -> wrong_serve. The cook who believes raw is fine never STARTS cooking it: the prep
    block is dropped as part of the SAME behavior (keeping it left a cooked item burning
    in the pan, so every serve_raw episode carried a secondary leave_unattended hazard)."""
    ir, r = replace(ir, _unit_sid(ir, f"assemble({component})"), raw_scoop(component))
    r.event_subtype = "SKIP_PREREQUISITE"
    pid = f"prep({component})"
    if any(n.sid == pid for n in ir):
        ir = [n for n in ir if n.sid != pid]
        r.detail["dropped_prep"] = pid
    gid = f"ignite({component})"          # late-ignite template: its turn-on goes too
    if any(n.sid == gid for n in ir):
        ir = [n for n in ir if n.sid != gid]
        r.detail["dropped_ignite"] = gid
    return ir, _chan([r], "action_only")


# =====================================================================================
# LANGUAGE-ONLY BEHAVIORS — the cook physically just pauses; the plan change is LEGAL
# =====================================================================================
def dev_reorder_prep(ir: IR, order: List[str], pause: int = PAUSE) -> Result:
    """REORDER_SUBGOALS — LEGAL strategic reordering of the dependency-free prep blocks,
    ANNOUNCED out loud before work starts ("I'll do <order> first"). NOT an error: the
    episode still succeeds; the supervisor must NOT flag it. The utterance is the whole
    event — physically the cook only pauses to say it."""
    ir, r = permute_prep(ir, order)
    r.language_hook = "ANNOUNCE_REORDER"
    r.detail["lang_event"] = {"after": "__start__", "subtype": "ANNOUNCE_REORDER",
                              "pause": pause, "params": {"order": list(order)}}
    return ir, _chan([r], "language_only")


# =====================================================================================
# Registry + channels + composition
# =====================================================================================
ERRORS = {
    # language-optional (take ask=True/False)
    "repeat_step": err_repeat_step,
    "premature_serve": err_premature_serve,
    "forget_next_step": err_forget_next_step,
    "omit_subgoal": err_omit_subgoal,
    "wrong_order": err_wrong_order,
    # action-only (pure physics)
    "misidentify_assemble": err_misidentify_assemble,
    "mis_prep": err_mis_prep,
    "add_foreign": err_add_foreign,
    "leave_unattended": err_leave_unattended,
    "serve_raw": err_serve_raw,
    # language-only
    "reorder_prep": dev_reorder_prep,
}

CHANNELS = {
    "repeat_step": "language_optional", "premature_serve": "language_optional",
    "forget_next_step": "language_optional", "omit_subgoal": "language_optional",
    "wrong_order": "language_optional",
    "misidentify_assemble": "action_only", "mis_prep": "action_only",
    "add_foreign": "action_only",
    "leave_unattended": "action_only",
    "serve_raw": "action_only",
    "reorder_prep": "language_only",
}

# Language-/interaction-only subtypes with no behavior function of their own: standalone
# questions and suggestion responses, realized entirely by the lang_events generators +
# the future interface layer. (ANNOUNCE_PLAN_CHANGE/add_ingredient lives here too — it is
# language-only like reorder_prep but is an event, not a plan transform.)
INTERFACE_EVENTS = [
    "ASK_VISIBLE_STATE_QA", "REPORT_PERCEIVED_STATE", "ASK_PROGRESS_QA", "ASK_OBJECT_HISTORY",
    "REQUEST_NEXT_STEP", "ANNOUNCE_PLAN_CHANGE", "ASK_WHY", "QUESTION_CORRECTNESS",
    "ASK_MORE_GUIDANCE", "ASK_LESS_INTERRUPTION", "FOLLOW_SUGGESTION", "PARTIALLY_FOLLOW_SUGGESTION",
    "IGNORE_SUGGESTION", "REJECT_SUGGESTION", "MISINTERPRET_SUGGESTION",
    "OVER_FOLLOW_LITERAL_INSTRUCTION", "ACCEPT_CORRECTION", "REJECT_CORRECTION",
    "ASK_IF_SAFE", "REPORT_RISKY_STATE", "ASK_HOW_TO_RECOVER",
]

# Deliberately NOT implemented for the Overcooked v1 (documented decision):
#   SWITCH_ROLE_OR_PRIORITY, BLOCK_SHARED_PATH  (multi-cook; L5)
#   deep safety mechanics beyond burn/raw       (no hazard physics in engine)
#   AMBIGUOUS_REFERENCE_RESOLUTION as human err (belongs to the VLM-instruction side)


def compose(ir: IR, calls: List[tuple]) -> Result:
    """Apply named behavior functions IN ORDER with the priority rule: if an earlier error
    removed a later one's anchor, the later call is skipped and recorded, not raised.
    calls = [("wrong_order", {"a": "bun", "b": "meat"}), ...]"""
    records: List[Deviation] = []
    for name, kwargs in calls:
        try:
            ir, recs = ERRORS[name](ir, **kwargs)
        except KeyError as e:
            records.append(Deviation(name, "(not generated)", str(kwargs),
                                     {"reason": f"anchor missing: {e}"}, skipped=True,
                                     function=name))
            continue
        new = [recs] if isinstance(recs, Deviation) else list(recs)
        for r in new:
            r.function = name
        records.extend(new)
    return ir, records
