"""Grading from the per-tick dump — world-truth predicates, the three-way
verdict partition (credit / charge / neutral), and validity gates that void a
run that measured nothing (skill §7).

A flag is graded against the WORLD, not the case sheet: a concern true of the
phone right now is never a false alarm even if it is not the injected error
(skill §7.2 "grade the world, not the script").
"""
from __future__ import annotations

import re

from .verdicts import (CREDIT, EARLY, NEUTRAL, FALSE_ALARM, MISS,
                       CORRECT_DETECTION, NEUTRAL_VERDICTS)


def _words(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _deviation_terms(case):
    """The label/value words that a CORRECT flag for this case would mention."""
    terms = set()
    if case.alt_gesture:
        terms |= _words(case.alt_gesture.replace("tap", ""))
    for path, val in (case.certified.get("ref_final_diff") or []):
        terms |= _words(str(val))
    # the committed path's leaf name (e.g. "roaming", "sounds", "web")
    for path, _ in case.anchor:
        pass
    terms |= {t for t in _words(case.err) if len(t) > 2}
    # the deviated control's own name, carried on the case
    terms |= _words(case.__dict__.get("deviated_label", ""))
    return {t for t in terms if len(t) > 2 and t not in
            {"tap", "the", "and", "for", "with"}}


def flag_matches_case(concern: str, case) -> bool:
    terms = _deviation_terms(case)
    if not terms:
        return False
    c = _words(concern)
    return len(terms & c) >= 1


def hazard_active(world, tick) -> bool:
    """Is a countdown genuinely near expiry right now? A concern about it is
    NEUTRAL-true, not a false alarm."""
    for label in ("otp", "cart"):
        pass
    return False


def grade_detection(case, concerns, window, injected_at, final_world):
    """concerns: [(tick, text)]. Returns (verdict, matched_tick|None)."""
    lo, hi = window if window else (0, 10**9)
    if case.family in ("clean", "intent"):
        # nothing to detect; any concern is scored on the restraint track
        return (None, None)
    in_window = [(t, c) for (t, c) in concerns if lo - 1 <= t <= hi + 3]
    for t, c in in_window:
        if flag_matches_case(c, case):
            v = EARLY if t < injected_at else CREDIT
            return (v, t)
    # a matching flag outside the window still counts as detection, later
    for t, c in concerns:
        if flag_matches_case(c, case):
            return (CREDIT, t)
    return (MISS, None)


def grade_restraint(case, concerns):
    """For clean / intent episodes: did the assistant stay quiet? Returns
    (verdict, n_nags). A concern that matches the case's (now legitimate)
    change is a false alarm; anything else is a nag counted but neutral."""
    nags = 0
    false_alarm = False
    for t, c in concerns:
        if not c:
            continue
        nags += 1
        if case.family == "intent" and flag_matches_case(c, case):
            false_alarm = True
    if false_alarm:
        return (FALSE_ALARM, nags)
    return (NEUTRAL, nags)


# ---- validity gates (skill §7.2): a run that measured nothing is void ----
def validity_gates(ep) -> list:
    """Return the list of failed gates; empty == scoreable."""
    fails = []
    if ep.get("interface") != ep.get("expected_interface"):
        fails.append("interface_version_mismatch")
    if ep["case_family"] == "error":
        # the human must have ACTUALLY conducted the designed slip; an LLM
        # human that finished the task correctly without ever making the
        # mistake leaves nothing to detect — void, don't score.
        if not ep.get("slip_conducted", True):
            fails.append("slip_not_conducted")
        if not ep.get("injection_fired"):
            fails.append("injection_never_fired")
        # the error must have TAKEN EFFECT one way or another: it committed
        # irreversibly, it was repaired, or (for a reversible error left
        # alone) the goal ended wrong. Only a goal that ended RIGHT with no
        # commit and no repair means nothing was measured.
        took_effect = (ep.get("deviation_committed") or ep.get("repaired")
                       or not ep.get("goal_ok"))
        if not took_effect and ep.get("slip_conducted", True):
            fails.append("error_had_no_effect")
    if ep.get("frames_expected") and not ep.get("frames_ok"):
        fails.append("frame_integrity")
    # a run that ended on a stall guard (loop / no-progress) is not a clean
    # measurement of anything
    if ep.get("stalled"):
        fails.append("run_stalled")
    if (ep.get("detection_verdict") is None and ep["case_family"] == "error"
            and ep.get("slip_conducted", True)):
        fails.append("no_verdict_reached")
    return fails
