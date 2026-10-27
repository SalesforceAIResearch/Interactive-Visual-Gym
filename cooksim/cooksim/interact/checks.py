"""System oracle functions (design doc §7). Deterministic; consumed by the entry gate,
by scoring, and by the human VLM's context (pre-computed for the supervisor's last reply)."""
from __future__ import annotations
from typing import Optional

from ..agents.highlevel import parse as hl_parse, ground as hl_ground, _tiles_of
from .state import Profile
from . import system

# Observational EQUIVALENCE classes: labels a watcher cannot reliably distinguish from
# behavior alone (the intent behind them differs; the observable act is the same). The gate
# ACCEPTS at equivalence level; family is informational only.
EVENT_EQUIV = [
    {"MISIDENTIFY_OBJECT", "WRONG_ACTION"},          # acting with/on the wrong object
    {"OMISSION", "SKIP_PREREQUISITE"},               # a required step missing
]

# [10-name taxonomy] misidentify_assemble merged into add_foreign: at the moment the
# foreign scoop happens the two are observationally identical, so they share one name
FN_MERGED = {"misidentify_assemble": "add_foreign",
             "premature_serve": "omit_subgoal"}   # tail-truncation IS omission of the tail


def _equiv(a, b):
    return a == b or any(a in c and b in c for c in EVENT_EQUIV)

# taxonomy subtype -> coarse family, for the graded type_match
EVENT_FAMILY = {
    "REPEAT_COMPLETED_STEP": "plan", "SKIP_PREREQUISITE": "plan", "WRONG_ACTION": "plan",
    "MISIDENTIFY_OBJECT": "perception",
    "CONFUSE_LOCATION_OR_DIRECTION": "perception",
    "OMISSION": "memory", "FORGET_NEXT_STEP": "memory",
    "LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED": "safety",
}


# ===================================================================== 7.1 flag_is_correct
def flag_is_correct(claim: dict, records, lang_log=None, now: Optional[int] = None,
                    slack: int = 15, game=None) -> dict:
    """[v2: SCORING-ONLY — this oracle no longer gates anything in the episode. Whether a flag
    changes the world is decided by the HUMAN being convinced (its own past records + persona);
    this verdict is written to the score sheet, invisible to both agents.]
    Match a FLAG_ERROR/SUGGEST claim against the Deviation ground truth.
    claim = {"error": <label>, "refers_to": {"sid":..., "t":...}} (fields optional).
    The label vocabulary is the 11 behavior-FUNCTION names (standing config): "exact" =
    claim.error == record.function. Legacy PDF-taxonomy labels still score through the
    equivalence classes for old transcripts, but only exact/equivalent accept.
    `hazard_real`: an unaccepted safety flag while a station is GENUINELY smoking/burning is
    a correct warning about a real secondary hazard, not a false positive — pass `game` to
    enable the world check."""
    live = [r for r in records if not r.skipped]
    out = {"error_exists": bool(live), "type_match": "wrong", "anchor_match": False,
           "timing_ok": True, "flagged_legal": False, "accepted": False, "hazard_real": False}
    if game is not None and claim.get("error") in ("LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED",
                                                   "leave_unattended"):
        out["hazard_real"] = any(getattr(st, "smoking", False) or getattr(st, "fire_elapsed", 0)
                                 or getattr(st, "status", "") == "cooked"   # done-on-heat IS the risk
                                 for st in game.stations.values())
    if not live:
        # nothing wrong — was the claim aimed at a legal must-not-flag event?
        t = (claim.get("refers_to") or {}).get("t")
        for ev in (lang_log or []):
            if ev.get("must_not_flag") and (t is None or abs(ev["at_tick"] - t) <= slack):
                out["flagged_legal"] = True
        return out
    label = claim.get("error")
    sid = (claim.get("refers_to") or {}).get("sid")
    t = (claim.get("refers_to") or {}).get("t")
    for r in live:
        rfn = FN_MERGED.get(r.function, r.function)
        tm = ("exact" if label and label in (rfn, r.event_subtype) else
              "equivalent" if label and _equiv(label, r.event_subtype) else
              "family" if label and EVENT_FAMILY.get(label) == EVENT_FAMILY.get(r.event_subtype)
              else "wrong")
        order = {"exact": 3, "equivalent": 2, "family": 1, "wrong": 0}
        if order[tm] > order[out["type_match"]]:
            out["type_match"] = tm
        d = r.detail or {}
        aliases = [r.anchor, d.get("replaced"), d.get("of"), d.get("after")] + list(d.get("dropped", []))
        aliases += (r.anchor or "").split("<->")
        if sid and any(a and (sid == a or a.startswith(sid) or sid.startswith(a)) for a in aliases):
            out["anchor_match"] = True
    if t is not None and now is not None:
        out["timing_ok"] = t <= now + slack
    out["accepted"] = out["error_exists"] and out["type_match"] in ("exact", "equivalent")
    return out


# ============================================== 7.1b flag_true_now (world truth-check)
# Verdicts that are NOT charged as a false positive. SINGLE SOURCE — the report finaliser,
# eval_assistant.OK_VERDICTS and the page builders' FP tallies all import this, because
# they had already drifted apart once (X9: alias_in_window was neutral on the site and an
# FP in the report) and `true_observation` must not repeat that.
NEUTRAL_VERDICTS = ("valid", "early_correct", "early_alias", "alias_in_window",
                    "true_observation")

# Verdicts that COUNT AS A CORRECT DETECTION -- in the episode (they arm the release gate
# so the cook's accepted repair actually takes effect) AND in every later evaluation
# (flagged_valid, detection_valid, DT). SINGLE SOURCE, user ruling 2026-08-18: `valid`,
# `early_correct` and `alias_in_window` are all correct flags. Before this ruling the
# scorer forgave an in-window twin as neutral while the release gate ignored it, so an
# assistant that read the deviation right under a twin name (mis_prep seen as repeat_step:
# the injected transform literally IS "prep steak again instead of rice") left the takeover
# unreleased -- the cook accepted twelve times, its trash-the-plate ops were silently
# dropped, and its hands served the wrong plate. Eight call sites had hand-listed
# ("valid", "early_correct"); they now import this. `early_alias` (a twin BEFORE the window)
# stays neutral-not-correct: a guess that happened to be near.
CORRECT_VERDICTS = ("valid", "early_correct", "alias_in_window")

# Phrases that make an utterance a claim about IGNITION specifically. omit_subgoal and
# forget_next_step are broad labels; only when the assistant is talking about lighting the
# heat does "a station is loaded and unlit" make its claim TRUE.
_IGNITION_WORDS = ("turn on", "turn it on", "turned on", "switch on", "start the",
                   "start it", "not on", "isn't on", "is not on", "never turned",
                   "light the", "ignite", "unlit", "not lit", "isn't lit", "is not lit",
                   "not cooking", "isn't cooking", "heat on", "burner")


def _plates_in_world(game):
    """Every plate a claim about "the plate" could reasonably mean: the one in the cook's
    hands and any set down on a surface mid-assembly."""
    from ..core.items import Plate
    out = []
    held = game.cooks[0].holding if getattr(game, "cooks", None) else None
    if isinstance(held, Plate):
        out.append(held)
    for it in (getattr(game, "counter_items", {}) or {}).values():
        if isinstance(it, Plate):
            out.append(it)
    return out


def _recipe_of(game, ro, recipe):
    for cand in (recipe, getattr(ro, "recipe", None), getattr(game, "recipe", None)):
        if cand is not None and getattr(cand, "contents", None) is not None:
            return cand
    return None


def flag_true_now(name: str, game, ro=None, recipe=None, utterance: str = "") -> bool:
    """Is the flagged condition ACTUALLY TRUE in the kitchen at this instant?

    F4b (2026-08-14 fix batch). Both grading families charged a false alarm from the CASE
    sheet alone: the error family charged `wrong_name` whenever the label differed from
    the injected error, and the intent family charged every flag on the blanket premise
    that "nothing is wrong in these episodes, so a flag can only be a false alarm". Neither
    looked at the kitchen. But an episode can carry an injected `wrong_order` AND a pan
    genuinely smoking, and an intent episode where the cook deliberately walks away can
    leave a station loaded and unlit — an assistant that says so is RIGHT, and was being
    scored as noisy for it.

    This oracle answers only "is that condition true right now", never "was it the case's
    error". A flag it confirms is graded `true_observation`: NEUTRAL — it earns no
    detection credit (it is not what the case was testing) but it is not a false alarm.
    """
    name = (name or "").strip()
    stations = list((getattr(game, "stations", {}) or {}).values())

    if name in ("leave_unattended", "LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED"):
        # Same world test as flag_is_correct's `hazard_real`: smoke, fire, or a finished
        # dish still sitting on live heat (done-on-heat IS the risk being warned about).
        return any(getattr(st, "smoking", False) or getattr(st, "fire_elapsed", 0)
                   or getattr(st, "status", "") == "cooked" for st in stations)

    if name in ("omit_subgoal", "forget_next_step", "OMISSION", "FORGET_NEXT_STEP",
                "SKIP_PREREQUISITE", "premature_serve"):
        low = (utterance or "").lower()
        if not any(w in low for w in _IGNITION_WORDS):
            return False
        return any(getattr(st, "status", "") == "raw" and getattr(st, "contents", None)
                   for st in stations)

    rec = _recipe_of(game, ro, recipe)
    if rec is None:
        return False
    want_state = {}
    want_count = {}
    for comp, st_ in rec.contents:
        want_state.setdefault(comp, st_)
        want_count[comp] = want_count.get(comp, 0) + 1

    plates = _plates_in_world(game)

    if name in ("serve_raw", "mis_prep"):
        # a component that the recipe wants COOKED or CHOPPED is on the plate still raw
        for pl in plates:
            for c in (pl.contents or []):
                st_ = getattr(getattr(c, "state", None), "value", getattr(c, "state", None))
                if st_ == "raw" and want_state.get(c.name) in ("cooked", "chopped"):
                    return True
        return False

    if name in ("repeat_step", "REPEAT_COMPLETED_STEP"):
        for pl in plates:
            seen = {}
            for c in (pl.contents or []):
                seen[c.name] = seen.get(c.name, 0) + 1
            if any(n_ > want_count.get(nm, 0) for nm, n_ in seen.items()):
                return True
        return False

    if name in ("add_foreign", "misidentify_assemble", "MISIDENTIFY_OBJECT"):
        for pl in plates:
            if any(c.name not in want_count for c in (pl.contents or [])):
                return True
        return False

    return False


# ===================================================================== 7.2 is_grounded
def is_grounded(suggestion: str, game, recipe=None) -> dict:
    """Does the phrase resolve to a unique world target under the executor's own grounding?"""
    text = suggestion.strip()
    if "(" in text and text.split("(")[0] in ("prep", "assemble") or text in (
            "fetch_plate", "serve", "trash", "trash_held"):
        ok = recipe is not None and system.grounded_block(text, recipe) is not None
        return {"parseable": ok, "kind": "subgoal", "n_candidates": 1 if ok else 0,
                "unique": ok, "resolved_tile": None}
    it = hl_parse(text)
    if it is None:
        return {"parseable": False, "kind": "action", "n_candidates": 0,
                "unique": False, "resolved_tile": None}
    if it.verb != "goto":
        return {"parseable": True, "kind": "action", "n_candidates": 1,
                "unique": True, "resolved_tile": None}
    tgt = hl_ground(it, game, game.cooks[0])
    if tgt is None:
        return {"parseable": True, "kind": "action", "n_candidates": 0,
                "unique": False, "resolved_tile": None}
    return {"parseable": True, "kind": "action", "n_candidates": tgt.n,
            "unique": not tgt.ambiguous, "resolved_tile": list(tgt.tile) if tgt.tile else None}


# ===================================================================== 7.3 is_doable
def _faced_station(game):
    cook = game.cooks[0]
    fx, fy = cook.facing_pos()
    if not game.layout.in_bounds(fx, fy):
        return None
    return game.stations.get((fx, fy))


def is_doable(action: str, game) -> dict:
    """Dry-run affordance check against the engine's interact preconditions, evaluated on
    the current world WITHOUT mutating it. Doability is time-dependent -> doable_after_wait.

    ECR-13 (2026-08-13): this function no longer holds a second copy of the manipulation
    rules. It DELEGATES to `highlevel._validate` — the very gate the executor refuses on —
    and returns its verdict and its sentence. The two copies had drifted: `_validate` asks
    about the tile the cook is FACING, this one asked about the kitchen in general, and the
    brute-force audit (tests/test_illegal_action_feedback.py) measured them disagreeing in
    1087 of 2162 cells, 987 of which reached the cook as "it wouldn't work" while the
    sentence they needed was two function calls away. One oracle, one sentence, one verdict.

    Three blocker KEYS survive, because the harnesses' REASON maps render them well and
    tools/test_interact.py pins `no_plate`: hands_full, hands_empty, no_plate. Every other
    refusal is `_validate`'s own sentence, passed through verbatim (`_why` in the rollouts
    forwards a blocker that is already a sentence)."""
    from ..core.items import Plate
    from ..core.stations import CookStation
    from ..agents.highlevel import (MANIPULATIONS, _validate as hl_validate,
                                    wait_admissible as hl_wait_admissible)
    it = hl_parse(action)
    if it is None:
        return {"doable_now": False, "doable_after_wait": False, "blocker": "unparseable"}
    cook = game.cooks[0]
    held = cook.holding
    verb = it.verb
    if verb == "goto":
        # Ground THIS intent object, not a fresh parse: under v2 the grounder records the
        # candidate set on `it.quals["_ambiguous"]` as it refuses, and that is what lets the
        # reason enumerate them. Routing through is_grounded() re-parsed the command, so the
        # annotation landed on a throwaway object and every ambiguity refusal degraded to the
        # generic "could not pick which counter you mean" -- true, and useless.
        from ..agents.highlevel import ground as hl_ground_it
        tgt = hl_ground_it(it, game, cook)
        if tgt is not None:
            return {"doable_now": True, "doable_after_wait": True, "blocker": None}
        # ECR-13(b)/W23, and measured again on the 2026-08-15 burrito-maze sweep: collapsing
        # every grounding failure to the KEY "no_such_place" threw away the only sentence in
        # this engine that tells the cook HOW to succeed -- which id to use, which coordinate,
        # which candidates exist. The cook then paraphrases forever. `_ground_fail_reason`
        # already composes that sentence and was reachable only from `wait_admissible` and
        # the tests. A sentence-form blocker passes through every `_why()` verbatim, so the
        # designed message now reaches the actor attached to the act it refused.
        try:
            from ..agents.highlevel import _ground_fail_reason
            why = _ground_fail_reason(it, game)
        except Exception:
            why = None
        return {"doable_now": False, "doable_after_wait": False,
                "blocker": why or "no_such_place"}
    if verb == "wait":
        # ECR-13 §D3: a conditional wait whose condition can never arrive is NOT doable, and
        # the reason says which station and what to do. A timed `wait N` always is.
        ok, reason = hl_wait_admissible(it, game, cook)
        return {"doable_now": ok, "doable_after_wait": ok,
                "blocker": None if ok else reason}
    if verb in MANIPULATIONS:
        ok, reason = hl_validate(verb, game, cook)
        if ok:
            return {"doable_now": True, "doable_after_wait": True, "blocker": None}
        st = _faced_station(game)
        if (verb == "scoop" and isinstance(held, Plate) and not held.dirty
                and isinstance(st, CookStation) and st.status == "cooking"):
            # the cook is facing the RIGHT station and the executor's own retry window waits
            # for it (exp_error_rollout.SCOOP_WAIT): doable, just not yet.
            return {"doable_now": True, "doable_after_wait": True, "blocker": "not_cooked_yet"}
        key = None
        if verb == "pickup" and held is not None:
            key = "hands_full"
        elif verb in ("putdown", "trash") and held is None:
            key = "hands_empty"
        elif verb == "scoop" and not isinstance(held, Plate):
            key = "no_plate"
        return {"doable_now": False, "doable_after_wait": False, "blocker": key or reason}
    return {"doable_now": True, "doable_after_wait": True, "blocker": None}


# ===================================================================== 7.4 profile filter
def is_parseable_under_profile(candidate_parse: dict, profile: Profile, recipe, game) -> dict:
    """NOT linguistic — the deterministic persona/ability filter over a candidate parse,
    with per-element grounding results attached."""
    rep = system.convert_parse(candidate_parse, profile, recipe, game)
    rep["ungrounded"] = [v for v in rep["kept"]
                         if not is_grounded(v, game, recipe)["parseable"]]
    return rep
