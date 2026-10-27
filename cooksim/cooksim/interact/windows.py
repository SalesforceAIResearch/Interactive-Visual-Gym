"""LIVE GRADING WINDOWS (user ruling 2026-09-20).

The window a detection is graded against is a property of THE EPISODE THAT WAS RUN,
not of a headless scripted reference replay.

    lo = the first OBSERVABLE deviated act of the live run  (`report["dev"]`, the
         dev_live latch, with its two standing special cases kept: an idle-type
         deviation — forget_next_step, `__wander__` macros, a qa_probe question —
         opens at the injection, and a first deviated step that had already been
         executed when the injection landed also opens at the injection).

    hi = the tick the mistake COMMITS in the live run — the last moment a correct
         call could still have prevented the harm. "Commit" is the engine's own
         `plate_verdict()["error_committed"]` flip, i.e. exactly the notion the
         report already records as `committed_at` and the notion the max-hold
         release fires on (plate_committed / smoke_committed / the deviated block
         running to its end). If the mistake never commits — it was prevented, the
         takeover was released early, or the episode ended first — the window closes
         at the release / archive tick.

A flag inside [lo, hi] is `valid`; after hi it is `late` (a miss, charged in DT, not
a false alarm); before lo it keeps the early / early_correct / true_observation /
fp_preinjection treatment it has today.

The old behaviour — offsets copied from the scripted reference replay,
`lo = dev_live + case_offsets["lo"]`, `hi = dev_live + case_offsets["hi"]` — is
restored with COOKSIM_REFERENCE_WINDOWS=1.

Reports written by the live-window engine stamp `report_keys["window_rule"] =
"live"`; anything without that stamp was graded against the reference window and
every consumer here treats it as such, so the two can never be mixed silently
(that exact mixing was bug F4a, August 2026).
"""
from __future__ import annotations

import os

LIVE = "live"
REFERENCE = "reference"
OPEN = 10 ** 9                      # a window that has not closed yet


def mode_from_env(env=None):
    """The window rule this process will publish. Live is the default."""
    env = os.environ if env is None else env
    if env.get("COOKSIM_REFERENCE_WINDOWS") == "1":
        return REFERENCE
    return LIVE


def window_rule(rk):
    """The rule a RECORDED sub-case (or episode) was graded under. Unstamped == reference."""
    rk = rk or {}
    if rk.get("window_rule") == LIVE or rk.get("window_rule_run") == LIVE:
        return LIVE
    return REFERENCE


def open_window(lo, offsets, mode):
    """The window at the moment the deviation goes live."""
    if mode == REFERENCE:
        off = offsets or {"lo": 0, "hi": 0}
        return [lo + off["lo"], lo + off["hi"]]
    return [lo, OPEN]


def close_at(window, t, lo_floor=None):
    """Close an open window at tick `t` (never before its own lo)."""
    if not window or window[1] != OPEN:
        return window
    floor = window[0] if lo_floor is None else lo_floor
    window[1] = max(int(t), int(floor))
    return window


# --------------------------------------------------------------------------------
# OFFLINE re-derivation, for regrading runs that were recorded under the old rule.
# --------------------------------------------------------------------------------
def commit_tick(entry):
    """The tick the injected mistake committed in the recorded run, or None.

    `report_keys["committed_at"]` is the live measurement and is used whenever it is
    there. Older records (before the A5 commit poll) do not carry it; for those the
    commit is read off the release, which is what the engine itself releases ON:
      * max_hold `plate_committed` / `smoke_committed` — the release IS the commit tick;
      * max_hold `deviated_block_executed` — for the attention errors (forget_next_step,
        leave_unattended) the block running to its end IS the commit;
      * any other release where the plate verdict at release says the error committed —
        the commit is bounded by that tick.
    Measured against the 198 sub-cases of tmp/suite_v5r3/llm_followup that do carry a
    live `committed_at`, this reconstruction is exact for add_foreign and
    leave_unattended and one tick late for mis_prep / omit_subgoal / wrong_order
    (max error 1 tick, n=198).
    """
    rk = entry.get("report_keys") or {}
    if rk.get("committed_at") is not None:
        return int(rk["committed_at"]), "recorded"
    pv = entry.get("plate_verdict_at_release") or {}
    if pv.get("error_committed") is not True:
        return None, "not_committed"
    t = rk.get("released_at")
    if t is None:
        t = (rk.get("max_hold_release") or {}).get("t")
    if t is None:
        t = entry.get("archived_at")
    return (None, "not_committed") if t is None else (int(t), "from_release")


def live_window(entry, episode_ticks=None):
    """[lo, hi, why] for one recorded error sub-case, under the live rule."""
    rk = entry.get("report_keys") or {}
    lo = rk.get("dev")
    if lo is None:
        return None
    tc, how = commit_tick(entry)
    if tc is not None:
        return [lo, max(tc, lo), "commit:" + how]
    end = rk.get("released_at")
    why = "released"
    if end is None:
        end = entry.get("archived_at")
        why = "archived"
    if end is None:
        end = episode_ticks
        why = "episode_end"
    if end is None:
        return None
    return [lo, max(int(end), lo), why]


# --------------------------------------------------------------------------------
# Verdict re-derivation (the same ladder verdict_live applies, for a known window).
# --------------------------------------------------------------------------------
CORRECT = ("valid", "early_correct", "alias_in_window")


def regrade(flag, err, window, twins=(), injected_at=None, dev=None, old=None):
    """Re-grade ONE recorded flag against `window`. Only the verdicts that the window
    itself decides are touched: valid / late / early / early_correct / early_alias /
    alias_in_window. Everything else (false_alarm, true_observation, wrong_name,
    fp_preinjection, fp_skipped, invalid_label, intent_*) is a judgement about the
    NAME or about the world, not about timing, and is returned unchanged."""
    old = flag.get("verdict") if old is None else old
    if old not in ("valid", "late", "early", "early_correct", "early_alias",
                   "alias_in_window"):
        return old
    t, name = flag.get("t"), flag.get("error")
    if t is None or window is None:
        return old
    lo, hi = window
    if name == err:
        if lo <= t <= hi:
            return "valid"
        if t < lo:
            anchor = lo if injected_at is None else min(lo, injected_at)
            return "early_correct" if t >= anchor else "early"
        return "late"
    tw = set(twins or ())
    if name in tw:
        base = lo if dev is None else dev
        if base - 2 <= t < lo:
            return "early_alias"
        if lo <= t <= hi:
            return "alias_in_window"
    return old
