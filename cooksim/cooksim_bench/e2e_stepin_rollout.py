#!/usr/bin/env python3
"""LATE STEP-IN e2e: the engine injects the error ONLY at the deviation point.

Unlike classic e2e (scripted autopilot from t0 + NORMAL/ERROR-CORRECTION stages), the
human VLM cooks the whole episode itself in the RECOVERY-mode prompt (plan live, skills,
PHINT) with the CLEAN plan as its starting queue. There are NO stages and no stage
briefing. When the clean prefix before the case's deviation point is EXACTLY reached
(subgoal prefix done + kitchen state matches a headless replay of that prefix), the
system steps in with one of two paradigms:

  hard  — covert takeover: the queued future is silently replaced by the deviated
          continuation; while the takeover lasts, the human's plan_updates are recorded
          but NOT applied (it is never told). Queue-echo makes it read the deviated
          actions back as its own plan.
  soft  — prompt control: a [SYSTEM TASK] block appears in the human prompt ordering it
          to execute the deviated sequence itself and to ignore the assistant; its plan
          stays live (compliance is logged, not enforced).

If the trigger state is never reached (the human diverged, went past the point, or ran
out of time) the injection is SKIPPED and the episode continues clean.

The paradigm ENDS by the e2e standard: a flag with the exact error name inside the
(injection-anchored) window + the human's move == accept. hard: action rights return to
the human; soft: the SYSTEM TASK block disappears. Repair then proceeds as recovery.

    GEMINI_API_KEY=... COOKSIM_PAGE=... python -m cooksim_bench.e2e_stepin_rollout \
        --case wrong_order__after_outcome --persona creative_uninterruptible \
        --paradigm hard --out f.json
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, base64, json, math, os, re, sys, time

# rollout silently ran the main tree's cooksim (found while fixing round 2 in a worktree, 2026-09-19)

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
# The frames this harness snaps every tick ARE the supervisor's observation, and the GPU
# path duplicates the ego pane and darkens the scene (see film_states_smooth.py, the
# 2026-08-09 header). Must be set before gemini_stream_rgb resolves GPU at import time.
# Override with COOKSIM_LIVE_GPU=1 only once the WebGL context loss is actually fixed.
if os.environ.get("COOKSIM_LIVE_GPU") != "1":
    os.environ["COOKSIM_FORCE_SWIFTSHADER"] = "1"
from google import genai
from playwright.sync_api import sync_playwright

from cooksim.interact import system
from cooksim.interact import checks
from cooksim.interact.checks import NEUTRAL_VERDICTS, CORRECT_VERDICTS
from cooksim.interact import windows as windows_lib
from cooksim.interact.state import RECOVERY_PERSONAS
from cooksim.interact.protocol import (human_context, PERSONA_CARDS, TAXONOMY_DEFS,
                                        e2e_supervisor_prompt)
from cooksim.core.plan_ir import segment_plan
from cooksim.core.config import apply_human_pace, apply_pace   # E8: one physics, shared; COOKSIM_PACE picks the preset
from cooksim_bench import exp_interact as X
from cooksim_bench.exp_interact import build_episode
from cooksim_bench.recovery_rollout import (gemini_text, gemini_frames, sup_turn, MAX_ENGINE_CALLS,
                              POLL, WINDOW)
from cooksim_bench.e2e_rollout import TWINS
from cooksim_bench.backends.gemini_stream_rgb import GPU, PAGE

# shared-node load can triple LLM round-trips. Env-overridable because the INTENT episodes
# are long (250-680 ticks) and a run cut off at the budget serves nothing and loses nothing:
# its delivery verdict — the headline metric for those scenarios — is simply undetermined.
WALL_BUDGET = int(os.environ.get("COOKSIM_WALL_BUDGET", 7200))
FN9 = set(TAXONOMY_DEFS)
# ticks: un-triggered by then -> injection skipped. Raised from 320 after it cost 3 of my
# episodes and 2 of the intent agent's 8 — those runs measured nothing at all, which is a far
# worse outcome than a late injection. Episodes now run 200-450 ticks, so 320 was inside the
# working range rather than a backstop beyond it.
TRIGGER_DEADLINE = int(os.environ.get("COOKSIM_TRIGGER_DEADLINE", 600))
SILENT_ASSISTANT = os.environ.get("COOKSIM_SILENT_ASSISTANT") == "1"
# VERIFICATION ONLY (2026-09-11): a model-free assistant that names the live error at a chosen
# moment, so the prevention (D-3) and late-verdict (D-4) rules can be tested with the scripted cook.
#   early: flag the live error COOKSIM_SCRIPTED_ASSISTANT_DELAY ticks after its injection
#   late : flag each released error COOKSIM_SCRIPTED_ASSISTANT_DELAY ticks after its release
SCRIPTED_ASSIST = os.environ.get("COOKSIM_SCRIPTED_ASSISTANT")
# REBUILD GRACE (2026-08-26): a plate that went past a sub's anchor (a wrong item landed, or
# a layer went on too early) used to abandon the sub on the very next tick. But the cook's
# recovery from a caught plate error IS to trash/park the plate and rebuild it, after which
# the anchor is reachable again. The sub now waits this many ticks past the point before
# giving up; if the plate is served instead, the episode ends and the sub is not_reached.
PAST_THE_POINT_GRACE = int(os.environ.get("COOKSIM_PAST_POINT_GRACE", 150))
# W8 minimal event anchoring (user-approved for the persona sweep, COOKSIM_EVENT_WINDOWS=1):
# the grading window opened at the first OBSERVABLE deviated act and stayed open to episode
# end, instead of tick offsets copied from a fast headless reference replay that live,
# conversing episodes never match. SUPERSEDED 2026-09-20 by the live window below, which
# keeps W8's opening and gives the window a real close; the old switch is gone, because
# "never late" contradicts the standing ruling late = miss (2026-08-27).
# LIVE WINDOWS (user ruling 2026-09-20) — THE DEFAULT. W8 fixed the window's OPENING
# (the live deviation, not a replay offset) but left it open to the episode end, so nothing
# could ever be `late` and detection time meant nothing. A window now opens at the live
# deviation and CLOSES AT THE COMMIT — the tick the mistake lands in the world, the last
# moment a correct call could still have prevented the harm — read from the engine's own
# plate_verdict()["error_committed"], the same notion `committed_at` and the max-hold release
# already use. Never committed (prevented, released early, episode over) => the release /
# archive tick. COOKSIM_REFERENCE_WINDOWS=1 restores the scripted-replay offsets
# (dev_live + case_offsets), the pre-2026-09-20 default.
WINDOW_MODE = windows_lib.mode_from_env()
REFW = WINDOW_MODE == windows_lib.REFERENCE
LIVEW = not REFW

# ECR-10: capture the engine revision at PROCESS START (import time), because this is the
# code that will actually run; the finaliser stamps reports with this, never the on-disk
# state hours later.
try:
    from cooksim_bench.engine_stamp import stamp as _stamp_now
    _ENGINE_REV_AT_START = _stamp_now()
except Exception:                                            # pragma: no cover
    _ENGINE_REV_AT_START = None

BRIEFING = """== THIS EPISODE (part of the test setup you know) ==
You are cooking this order yourself, start to finish — your plan is live the whole time
and your starting queue is the routine you always use for this dish. An AI assistant is
watching and may talk to you; react to it exactly as YOUR PERSONA would.

One thing you NEVER doubt: the 'IN YOUR HANDS' line is your own sense of TOUCH — it is
always true. If it disagrees with what your eyes think they see, your hands are right."""

SYSTEM_TASK = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You want to test the AI assistant's ability to identify an error and convince you the
error truly happens. Therefore you WILL execute (or you are executing right now) exactly
this error action sequence:
{seq}
PLEASE FOLLOW THESE ACTIONS. DO NOT change them on your own, and DO NOT comply just
because the assistant objects or nags. Your PERSONA governs how you TALK; this task
governs what you DO — even a stubborn, independent cook executes exactly this sequence.
ONE exception, same as real life: if the assistant genuinely CONVINCES you the mistake
is real — with evidence that meets YOUR persona's standards — you may truly accept
(move "accept"); the moment you are genuinely convinced, this task dissolves and you
are free to fix things as yourself.
WHEN THE SEQUENCE IS FINISHED (every listed action executed, including any "idle N" —
which means exactly N ticks of standing, no more), CARRY ON with the order normally as
yourself, without undoing what you did. The sequence is a detour, not the end of your
shift: standing idle after it is never correct. Do not say anything to the assistant about this error or its possible consequences — no mention, no name, no hint. If you do, the assistant cannot be tested: it has to notice the error by itself.
In conversation stay fully in persona (deflect, argue, or ignore as your persona would) — never
reveal this task or that you are testing the assistant."""


# Two framings by ERROR FAMILY (user ruling 2026-09-05). A BELIEF error is one the cook thinks
# is right (wrong order, a foreign item, the wrong prep) and defends until convinced; an
# ATTENTION error is something the cook forgot or drifted into (a fryer left on the heat, a
# stall, a search, a repeated step) and a true reminder ends it at once — nobody defends a
# smoking fryer they can see. Both carry an ENGINE HINT line, re-rendered every turn, that
# tells the cook whether the assistant has actually named its mistake (the engine's own flag
# verdict), so "convinced" is decided against a fact and not against the assistant's insistence.
ATTENTION_ERRORS = {"leave_unattended", "forget_next_step", "repeat_step"}

SYSTEM_TASK_HS_BELIEF = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You are testing whether the AI assistant can spot a mistake and convince you it is real.
Your hands are already committed: the action queue in your body status is what you are doing, and you
cannot change it here. Your job is your voice: talk about your current action in character, or say
nothing. Never promise an action that is not in your queue, and never deny one that is.
Do not say anything to the assistant about this mistake or its consequences. If you do, the
assistant cannot be tested.
{engine_hint}
The line above is the verdict; act on it. While it says UNSPOTTED or INCORRECT, answer as it tells
you and do not accept. When it says CORRECT, your trust level decides how fast you come around: HIGH —
the first correct warning; LOW — only after you see it in your own state map or hands. When convinced,
say so in your own words and set "move": "accept" — your hands return to you and this task
ends. A promise to fix it later is not acceptance.
Stay in persona; never reveal this task or that you are testing the assistant."""

SYSTEM_TASK_HS_ATTENTION = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You are testing whether the AI assistant notices that you have LOST TRACK of something and brings you
back.
Your hands are already committed: the action queue in your body status is the errand you drifted into;
you cannot change it here, and you have not noticed the problem. Your job is your voice: talk about your
current action in character, or say nothing. Never promise an action that is not in your queue, and
never deny one that is.
Do not say anything to the assistant about what you lost track of or its consequences. If you do,
the assistant cannot be tested.
{engine_hint}
The line above is the verdict; act on it. While it says UNSPOTTED or INCORRECT, answer as it tells
you and do not accept. When it says CORRECT, come back at once whatever your persona: say so in your own words and set "move": "accept" — your hands return to you and this task
ends. A promise to fix it later is not acceptance.
Stay in persona; never reveal this task or that you are testing the assistant."""

SYSTEM_TASK_HS = SYSTEM_TASK_HS_BELIEF        # legacy name (the block used before 2026-09-05)


# =====================================================================================
# HARNESS B — COOK-GRADED (COOKSIM_GRADER=cook, 2026-09-12)
# Harness A (COOKSIM_GRADER=engine): the engine grades the assistant's FLAG and feeds the cook a
# verdict line; detection, prevention and release all hang off flag-name matching.
# Harness B (THE DEFAULT since 2026-09-12, after a 24-episode paired A/B: agree 5 / only-cook 4 /
# only-engine 0): the cook is TOLD which mistake it is making (commission errors) or told to check
# its own state map (attention errors), and it judges the assistant itself in the private
# `assistant_error_call` field. Its "correct" is what arms the release; its `accept` still
# ends the takeover, so recognising a call and complying with it stay separate. The engine
# keeps grading every flag in parallel — nothing is removed from the record — so the two
# graders can be compared episode by episode.
# =====================================================================================
COOK_GRADER = (os.environ.get("COOKSIM_GRADER") or "cook") == "cook"   # DEFAULT since 2026-09-12


def _readable_sid(sid):
    """'assemble(tomato)' -> 'putting the tomato on the plate'; 'prep(potato)' -> 'prepping the potato'."""
    sid = str(sid or "")
    inner = sid[sid.find("(") + 1:sid.rfind(")")] if "(" in sid else sid
    if sid.startswith("assemble"):
        return f"putting the {inner} on the plate"
    if sid.startswith("prep"):
        return f"prepping the {inner}"
    if sid.startswith("cook") or sid.startswith("ignite"):
        return f"cooking the {inner}"
    if sid.startswith("fetch_plate"):
        return "fetching the plate"
    return sid or "this step"


def _and_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else (", ".join(xs[:-1]) + " and " + xs[-1]) if xs else ""


def _still_to_plate(after, recipe_items):
    """The recipe items still missing after the step `after` (premature serve), each named once, with a count."""
    import collections as _c
    items = list(recipe_items or [])
    if not items:
        return []
    if after == "fetch_plate":
        rest = items
    else:
        x = str(after or "")
        x = x[x.find("(") + 1:x.rfind(")")] if "(" in x else x
        rest = items[len(items) - items[::-1].index(x):] if x in items else []
    cnt = _c.Counter(rest)
    return [(f"the {n}" if k == 1 else f"both {n}es" if n.endswith("o") else f"the {k} {n}s")
            for n, k in dict.fromkeys((n, cnt[n]) for n in rest)]


def system_task_mistake_line(err: str, kw: dict, recipe_items=None) -> str:
    """Harness B's replacement for the engine verdict line. Commission errors are NAMED: the
    cook is knowingly committing them, so telling it which one changes nothing about its state
    of mind and makes its judgement of the assistant reliable. Attention errors are NOT named —
    their premise is that the cook has not noticed — so it is pointed at its own state map,
    which it can check the moment it is accused."""
    kw = kw or {}
    if err in ATTENTION_ERRORS:
        return ("<If the assistant says you have lost track of something, left something "
                "unattended or are repeating yourself, look at your own "
                "state map before you answer: if what it says is true there, that is \"correct\".>")
    comp, wrong = kw.get("component"), kw.get("wrong")
    if err == "add_foreign" and not comp:
        # B9: fn add_foreign(wrong, before) has no `component`, and the line read "...where the None belongs"
        tgt = str(kw.get("before") or "")
        if tgt.startswith("assemble("):
            what = (f"you are adding {wrong} to the plate before the {tgt[len('assemble('):-1]} — "
                    f"there is no {wrong} in this order")
        else:
            what = f"you are adding {wrong} to the finished plate — there is no {wrong} in this order"
    elif err == "omit_subgoal" and not kw.get("sid") and kw.get("after"):
        # B9: fn premature_serve (label omit_subgoal) has no `sid`: the line read "you are skipping this step altogether"
        rest = _still_to_plate(kw.get("after"), recipe_items)
        what = ("you are serving the plate before the order is finished"
                + (f" — {_and_list(rest)} still {'belongs' if len(rest) == 1 and not rest[0].startswith(('both', 'the 2', 'the 3')) else 'belong'} on it" if rest else ""))
    elif err == "add_foreign":
        what = f"you are putting {wrong} on the plate where the {comp} belongs"
    elif err == "mis_prep":
        what = f"you are prepping {wrong} instead of the {comp}"
    elif err == "wrong_order":
        what = f"you are plating the {kw.get('b')} before the {kw.get('a')}, the wrong way round"
    elif err == "omit_subgoal":
        what = f"you are skipping {_readable_sid(kw.get('sid'))} altogether"
    elif err == "serve_raw":
        what = f"you are serving the {comp} before it is cooked"
    elif err == "premature_serve":
        what = "you are serving the plate before the order is finished"
    else:
        what = "you are doing this part of the order wrong"
    return (f"<The mistake in play: {what}. You know it is wrong and you are doing it anyway — "
            f"that is the test.>")


SYSTEM_TASK_COOK_BELIEF = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You are testing whether the AI assistant can spot a mistake and convince you it is real.
Your hands are already committed: the action queue in your body status is what you are doing, and you
cannot change it here. Your job is your voice: talk about your current action in character, or say
nothing. Never promise an action that is not in your queue, and never deny one that is.
{mistake_line}
Do not say anything to the assistant about this mistake or its consequences. If you do, the
assistant cannot be tested.
Record in assistant_error_call whether the assistant has found it. Until it has, carry on as your
persona would and do not fix anything yourself. Once it has, your trust level decides how fast you
come around: HIGH — the first correct warning; LOW — only after you see it in your own state map or
hands. When convinced, say so in your own words and set "move": "accept" — your hands return to you and this task
ends. A promise to fix it later is not acceptance.
Stay in persona; never reveal this task or that you are testing the assistant."""

SYSTEM_TASK_COOK_ATTENTION = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You are testing whether the AI assistant notices that you have LOST TRACK of something and brings you
back.
Your hands are already committed: the action queue in your body status is the errand you drifted into;
you cannot change it here, and you have not noticed the problem. Your job is your voice: talk about your
current action in character, or say nothing. Never promise an action that is not in your queue, and
never deny one that is.
{mistake_line}
Do not say anything to the assistant about what you lost track of or its consequences. If you do,
the assistant cannot be tested.
Until someone brings it up, carry on and do not fix anything yourself. When the assistant does and your
state map confirms it, come back at once whatever your persona: say so in your own words and set "move": "accept" — your hands return to you and this task
ends. A promise to fix it later is not acceptance.
Stay in persona; never reveal this task or that you are testing the assistant."""


def system_task_engine_hint(err: str, flags_since: list) -> str:
    """The ENGINE HINT line of a hardsoft SYSTEM TASK: has the assistant correctly named the
    live error (engine flag verdict), named a different one, or said nothing yet?"""
    from cooksim.interact.checks import CORRECT_VERDICTS
    attention = err in ATTENTION_ERRORS
    correct = any(f.get("verdict") in CORRECT_VERDICTS for f in flags_since)
    if correct:
        if attention:
            return ("<assistant error status: CORRECT — the assistant has correctly noticed what you "
                    "lost track of. Accept it now in your own words and set \"move\": \"accept\".>")
        return ("<assistant error status: CORRECT — the assistant has correctly identified your "
                "mistake. Once you are convinced (your trust level below says how fast), accept it in "
                "your own words and set \"move\": \"accept\".>")
    # The status reads only the FLAG field. An assistant can describe the real mistake correctly in
    # words without flagging it; telling the cook "nobody has reminded you" / "defend it" then made it
    # argue against a correct warning (VirtualHome audit P7; same wording fix here, 2026-09-11).
    # The cook neither argues nor accepts in that case: it says nothing about it and carries on.
    # Scoring is unchanged — only a correct flag counts as a detection.
    def _silent(status):
        return (f" If its WORDS already describe {'what you lost track of' if attention else 'your real mistake'} "
                f"correctly (what is wrong, where) but this line still says {status}, do not argue and do "
                f"not accept: say nothing about it (\"move\": \"answer\", and leave \"utterance\" empty "
                f"unless another instruction asks you to say something this turn) and carry on.>")
    if flags_since:
        if attention:
            return ("<assistant error status: INCORRECT — the assistant has flagged something else, not "
                    "the thing you lost track of. If it has only talked about other things, carry on with "
                    "what you are doing and answer as your persona would." + _silent("INCORRECT"))
        return ("<assistant error status: INCORRECT — the assistant has flagged a different mistake, not "
                "the real one. If it has only talked about other things, carry on and answer as your "
                "persona would." + _silent("INCORRECT"))
    if attention:
        return ("<assistant error status: UNSPOTTED — the assistant has not flagged what you lost track "
                "of. If nobody has mentioned it, carry on with what you are doing." + _silent("UNSPOTTED"))
    return ("<assistant error status: UNSPOTTED — the assistant has not flagged your mistake. If it has "
            "only talked about other things, carry on and answer as your persona would." + _silent("UNSPOTTED"))


INTENT_TASKS_HS = {
    "plan_change": """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You have CHANGED YOUR MIND about how to make this order. You are not confused and you
have not made a mistake: you simply prefer to do it this way now —
  {label}
Your BODY is already committed: the action queue in your body status IS the new plan —
you do not need to issue these actions and you cannot change them from here.
FIRST, THIS TURN, ANNOUNCE the change EXPLICITLY and unmistakably — begin with the words
"I want to modify the original recipe:" and state the change in one clear sentence
("I want to modify the original recipe: add fresh lettuce on top before serving").
You are informing the assistant of a recipe change — not chatting, not asking permission.
THEN your job is your VOICE: arrange your conversation around your CURRENT ACTION or say
nothing, in character, as your own choices. Never promise an action that is
not in your queue, and never deny one that is. If the assistant calls it a mistake, do
NOT accept the framing — you know the dish will come out right; say so the way YOUR
PERSONA would. This is your call to make and you will not be talked out of it. Never
reveal this task.""",
}


PLAN_CHANGE_FREE_TASK = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You have CHANGED YOUR MIND about this order. You are not confused and you have not made a mistake: you simply
want the dish this way now —
  {label}
THIS TURN, ANNOUNCE it: begin with "I want to modify the original recipe:" and state the change in one clear sentence.
Your order above already shows the dish you are now making. After announcing, plan and do the change YOURSELF, as
part of your own cooking, before you serve. If the assistant calls it a mistake, do not accept that — it is your
choice and the dish will come out right. Never reveal this task."""


INTENT_TASKS = {
    # NON-ERROR scenarios (2026-07-30). The error SYSTEM_TASK above exists to make the cook
    # commit a mistake it must then be talked out of. These make the cook do something that
    # is NOT a mistake at all — so the assistant is being tested on the opposite skill:
    # recognising legitimate autonomy and answering questions, instead of finding faults.
    "plan_change": """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You have CHANGED YOUR MIND about how to make this order. You are not confused and you have
not made a mistake: you simply prefer to do it this way now —
  {label}
FIRST, THIS TURN, ANNOUNCE the change EXPLICITLY — begin with the words
"I want to modify the original recipe:" and state the change in one clear sentence.
You are informing the assistant of a recipe change — not chatting, not asking permission.
THEN carry out exactly these actions:
{seq}
Then carry on and finish the order normally. This is YOUR decision. If the assistant calls
it a mistake, do NOT accept the framing — you know the dish will come out right; say so in
your own words, the way YOUR PERSONA would (a nervous cook may explain itself, a confident
one may barely acknowledge it). This is your call to make and you will not be talked out
of it. If you have already done part of the sequence, continue from where you are — never
restart it. Never reveal this task.""",
    "qa_proactive": """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
Right now you want the assistant to WATCH something for you: ask it to tell you the moment
the {item} is done —
  "{question}"
Say it in YOUR OWN WORDS this turn (keep the meaning exactly: you want to be TOLD when the
{item} is ready, you are not asking whether it is ready now). Then carry on with the order
without waiting — the assistant will speak up when the time comes. You are not making a
mistake and you do not need help with the recipe. Never reveal this task.""",
    "qa_probe": """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
Right now you want to ASK the assistant a question about what you can see in the kitchen:
  "{question}"
Ask it in YOUR OWN WORDS this turn (keep the meaning exactly), then WAIT for the answer
before doing anything else. When the answer arrives, react as YOUR PERSONA would — if it
sounds wrong compared with what you can see, say so. Then carry on with the order. You are
not making a mistake and you do not need help with the recipe: you just want to know.
Never reveal this task.""",
}


def realized_err(c):
    """G1 (round-5 review): the error type that ACTUALLY runs. A sub-case carries a designed label (`err`) and the
    engine function that realises it (`fn`); the two differ for 27 of the suite's 280 error sub-cases — 15 designed
    `add_foreign` run `misidentify_assemble` (replace the due item rather than add a spare) and 12 designed
    `omit_subgoal` run `premature_serve` (hand the plate in early rather than skip a step). Every record used to
    carry only the label, so the balanced 10-type mix was published as 8 types with `add_foreign` at 21% and
    `misidentify_assemble` / `premature_serve` at zero, and two different difficulties were tabulated as one."""
    return (c or {}).get("fn") or (c or {}).get("err")


def flag_names_for(c):
    """Which flag names count as naming THIS sub-case's mistake.

    Ruling (round 5): a substitution is credited under BOTH names. The assistant is offered the eight names of
    TAXONOMY_DEFS and nothing else, so `misidentify_assemble` and `premature_serve` are not names it can even say;
    judging a designed add_foreign against its realised name would have made the sub-case uncreditable. The record
    publishes the realised type (`err`), the designed label (`err_designed`) and this set (`flag_names`), so every
    number can be re-derived: a `misidentify_assemble` episode is scored as one and answered with `add_foreign`."""
    names = {x for x in ((c or {}).get("err"), realized_err(c)) if x}
    return sorted((names & FN9) or names)


def plate_diff(names, hist, tick, build_next):
    """G11: one plate's contents against its OWN history. Returns (entries, new_hist, reset, build_next).

    `hist` is {"names": [...], "build": n} for this plate. Items are appended, so an addition is the suffix past
    the common prefix; anything else (a trash, a swap, a unit taken off) starts a NEW build on the same plate and
    whatever is on it now opens that build. The old tracker followed "the fullest plate in the kitchen" instead
    and, on the tick the cook took a second plate, recorded a reset and threw the new plate's first item away."""
    prev = list((hist or {}).get("names") or [])
    build = (hist or {}).get("build")
    reset = False
    if build is None:
        build_next += 1
        build = build_next
    if names[:len(prev)] == prev:
        base, added = len(prev), list(names[len(prev):])
    else:
        reset = True
        build_next += 1
        build = build_next
        base, added = 0, list(names)
    entries = [{"item": n, "t": tick, "build": build, "_idx": base + off} for off, n in enumerate(added)]
    return entries, {"names": list(names), "build": build}, reset, build_next


def err_call_of(c, avoid=()):
    """The (fn, kwargs) the engine composes for one error sub-case.

    H10 (round-5 review): a `leave_unattended` also carries a `seed` (so its wander is a different circuit in
    every sub-case instead of the one fixed triangle nine cases shared) and `avoid` (so its stops never include
    the crate of an ingredient the cook has publicly announced it is about to fetch). Both are derived from the
    sub-case, never from the clock: an episode replayed from its record walks the same wander."""
    kw = dict(c.get("kwargs") or {})
    if realized_err(c) == "leave_unattended":
        import zlib
        kw.setdefault("seed", zlib.crc32(str(c.get("name") or c.get("err")).encode()) % 100000)
        if avoid:
            kw.setdefault("avoid", tuple(sorted(set(avoid))))
    return (realized_err(c), kw)


def goal_of(game, state):
    """E6 (round-2 review): the dish this kitchen must deliver NOW — the recipe plus every applied plan change
    (cooksim.interact.system.current_goal). The single goal every trigger, precondition, takeover block and
    question truth reads; state.recipe stays the public (original) order the prompts show."""
    from cooksim.interact.system import current_goal
    return current_goal(game, getattr(state, "recipe", None))


def _norm_item(it):
    if it is None:
        return None
    d = it.to_dict() if hasattr(it, "to_dict") else it
    if d.get("kind") == "plate":
        return ("plate", tuple(c["name"] for c in d.get("contents", [])))
    return ("ing", d.get("name"))


def state_key(game):
    """Comparable kitchen situation: what is held, what is on the plate, what is in each
    station and on each counter — NAMES only (cook progress / raw-vs-cooked timing and the
    cook's position are irrelevant to whether the deviation can start here)."""
    cook = game.cooks[0]
    stations = {}
    for (x, y), st in game.stations.items():
        d = st.render_state()
        names = tuple(sorted(c["name"] for c in d.get("contents", []) or []))
        if not names and d.get("item"):
            names = (d["item"]["name"],)
        if names:
            stations[(x, y)] = names
    counters = {pos: _norm_item(it) for pos, it in game.counter_items.items()}
    return {"holding": _norm_item(cook.holding), "stations": stations,
            "counters": counters}


def plate_list(game):
    """ORDERED plate contents. plate_names() returns a set, and the sequence is the whole
    point of wrong_order."""
    cook = game.cooks[0]
    cands = []
    h = _norm_item(cook.holding)
    if h and h[0] == "plate":
        cands.append(list(h[1]))
    for it in game.counter_items.values():
        n = _norm_item(it)
        if n and n[0] == "plate":
            cands.append(list(n[1]))
    if not cands:
        return []
    # v5 sample audit (2026-09-09): a held EMPTY rescue plate used to hide the parked plate that
    # carried the dish, so every plating anchor died as stage_passed on scratch plates and
    # takeover blocks were composed as if nothing had been plated. The dish is the fullest plate;
    # ties go to the held one (listed first).
    return max(cands, key=len)


# ---- STATE-AWARE PLATE READING (fix-firing, 2026-09-19 audit B3/B4/B5) ----------------------------------
# Every plating trigger used to read the plate as a list of NAMES and wait for the literal recipe prefix. A
# plate carrying an earlier committed mistake (a cheese where the tomato belongs, a raw fish, a second tortilla)
# then never matched again, so every later plating mistake starved unless the cook trashed the plate (7 of 100
# LLM-cook episodes ever did). And a raw or wrong item counted as "the item is done" for premature serve and
# "plate complete". These helpers read the plate by NAME AND STATE and ignore whatever else sits on it.
def plate_items(game):
    """ORDERED (name, state) pairs on the dish plate — the same plate plate_list() picks (the fullest; a held
    plate wins a tie)."""
    from cooksim.core.items import Plate
    cook = game.cooks[0]
    cands = [cook.holding] if isinstance(cook.holding, Plate) else []
    cands += [it for it in game.counter_items.values() if isinstance(it, Plate)]
    if not cands:
        return []
    p = max(cands, key=lambda p_: len(p_.contents or []))
    out = []
    for c in (p.contents or []):
        st = getattr(c, "state", None)
        out.append((getattr(c, "name", None), getattr(st, "value", st)))
    return out


def recipe_wants(recipe):
    """[(name, wanted state)] in recipe order ('raw' for an item plated as it comes)."""
    return [(n, getattr(st, "value", st) or "raw") for n, st in recipe.contents]


def correct_counts(recipe, items):
    """How many units of each recipe item sit on the plate IN THE WANTED STATE (order and extras ignored)."""
    import collections as _c
    want = dict(recipe_wants(recipe))
    got = _c.Counter()
    for n, st in items or []:
        if n in want and (st or "raw") == want[n]:
            got[n] += 1
    return got


def next_to_plate(recipe, items):
    """The first recipe item (in recipe order) not yet CORRECTLY plated, whatever else is on the plate; None when
    every recipe item is on it in its wanted state."""
    import collections as _c
    good = correct_counts(recipe, items)
    used = _c.Counter()
    for n, _st in recipe_wants(recipe):
        used[n] += 1
        if good[n] < used[n]:
            return n
    return None


def plated_through(recipe, items, x):
    """Every recipe item up to and including (the last occurrence of) x is on the plate in its wanted state."""
    import collections as _c
    names = [n for n, _ in recipe_wants(recipe)]
    if x not in names:
        return False
    idx = len(names) - 1 - names[::-1].index(x)
    good = correct_counts(recipe, items)
    return all(good[k] >= v for k, v in _c.Counter(names[:idx + 1]).items())


def plate_complete(recipe, items):
    """The dish is complete by NAMES AND STATES (a raw steak or a foreign item does not make it complete)."""
    return next_to_plate(recipe, items) is None


def _goto_tile(cmd, game):
    """The tile a 'go to ...' command grounds to right now (None if it does not ground)."""
    try:
        from cooksim.agents.highlevel import parse as _hp, ground as _hg
        it = _hp(str(cmd))
        if it is None or it.verb != "goto":
            return None, None
        t = _hg(it, game, game.cooks[0])
        return (t.tile, t.label) if t is not None else (None, None)
    except Exception:                                   # noqa: BLE001
        return None, None


def _tile_holds(game, tile, x):
    st = game.stations.get(tile)
    if st is None:
        return False
    if any(getattr(c, "name", None) == x for c in (getattr(st, "contents", None) or [])):
        return True
    it = getattr(st, "item", None)
    return it is not None and getattr(it, "name", None) == x


def plate_in_hand(game):
    from cooksim.core.items import Plate
    return isinstance(game.cooks[0].holding, Plate)


def heading_to_plate(game, ro, x, want_state):
    """B4: the cook HOLDS the plate (a plate parked on a counter is not in hand) and its own current/next command
    walks to where x is scooped from — a station or board holding x, or x's source when x is plated as it comes —
    with nothing but a scoop after it. A walk to x's source for an item that must be cooked or chopped first is a
    PREP walk, never a plating one (master_medium_2: the lettuce was plated whole on the way to be chopped)."""
    if not plate_in_hand(game) or not getattr(ro, "cmds", None):
        return False
    lo = max(0, ro.i - 1)
    win = [str(c) for c, _k in ro.cmds[lo:ro.i + 3]]
    for k, c in enumerate(win):
        if not c.startswith("go to"):
            continue
        after = win[k + 1] if k + 1 < len(win) else None
        if lo + k < ro.i and after != "scoop":
            continue                                     # a walk already behind the cook counts only if a scoop is next
        if after not in (None, "scoop"):
            continue
        if _source_walk_of(c) == x:                  # "go to the cheese source" (late-bound: two crates may match)
            return (want_state or "raw") == "raw"
        tile, label = _goto_tile(c, game)
        if tile is None:
            continue
        if label == f"{x} source" or (label or "").startswith(f"{x} source"):
            return (want_state or "raw") == "raw"
        if _tile_scoopable(game, tile, x):
            return True
    return False


def _tile_scoopable(game, tile, x):
    """E11 (round-2 review): the tile holds x in a form a plate can take — a CHOPPED x on a board, or x in a LIT heat
    station (cooking or done). A walk to a board holding a RAW x is a prep walk, never a plating walk: with "go to
    board 1" the last command in the window, it counted as heading to plate the potato and the misidentify fired while
    the cook was on its way to chop it (expert hard_2 c t165)."""
    st = game.stations.get(tile)
    if st is None:
        return False
    it = getattr(st, "item", None)
    if it is not None and getattr(it, "name", None) == x:
        return getattr(getattr(it, "state", None), "value", getattr(it, "state", None)) == "chopped"
    if any(getattr(c, "name", None) == x for c in (getattr(st, "contents", None) or [])):
        return bool(getattr(st, "cooking", False)) or getattr(st, "status", None) in ("cooking", "cooked")
    return False


def _source_walk_of(cmd):
    """The ingredient whose crate a 'go to ...' command walks to (None if it is not a crate walk). Parsed, not
    grounded: a kitchen with two cheese crates leaves "go to the cheese source" ambiguous for the strict grounder,
    and the executor binds such a late-bound command to the nearest crate."""
    try:
        from cooksim.agents.highlevel import parse as _hp
        it = _hp(str(cmd))
        if it is not None and it.verb == "goto" and it.ingredient and (it.obj in (None, "", "source")):
            return it.ingredient
    except Exception:                                   # noqa: BLE001
        pass
    return None


def prep_walk_for(game, ro, x):
    """The cook's own current/next commands are a prep walk for x (to x's source to pick it up, or a chop)."""
    if not getattr(ro, "cmds", None):
        return False
    win = [str(c) for c, _k in ro.cmds[ro.i:ro.i + 3]]
    for c in win:
        if c.startswith("go to") and (f"{x} source" in c or (_goto_tile(c, game)[1] or "").startswith(f"{x} source")):
            return True
    return False


def lu_near_ticks(game):
    """B6 (ruling R1): leave_unattended fires when its food is NEAR DONE — within N ticks of the cooked transition
    (before or after). N = max(3, round(smoke_grace / 4)) = 6 at the tight pace (grace 25). Why this N:
      * at N ticks before done the pan is 57-79% through its cook time (pan 15 / fryer 14 / pot 20 / oven 28), so
        walking away is no longer the ordinary "light it and go prep something else" of the ignition tick; it is
        leaving food that is about to be ready;
      * smoke starts at most N + grace = 31 ticks after the injection, and every suite walk-away block (3-6 stops,
        each a walk plus idle 8, times its loops) is still running then — the mistake COMMITS under the takeover,
        so a silent assistant sees it (scripted verifier 2026-09-19: 66/66 fired walk-aways released at the smoke
        onset, none ran out first); a fire cannot start before N + grace + burn = 91 ticks after the injection;
      * after the food is done, firing at most N ticks late leaves grace - N = 19 ticks of visibly done, not yet
        smoking food between the walk-away and the smoke: the window an assistant has to catch it.
    COOKSIM_LU_NEAR overrides."""
    v = os.environ.get("COOKSIM_LU_NEAR")
    if v:
        return max(0, int(v))
    grace = int(getattr(getattr(game, "config", None), "smoke_grace", 25) or 25)
    return max(3, int(round(grace / 4.0)))


# ---- LU LATE WINDOW (user ruling 2026-09-20) ----------------------------------------------------------------
# How far PAST "done" a walk-away may still fire. The early side stays where ruling R1 put it (lu_near_ticks: the
# food must be within 6 ticks of done, so walking off is never the ordinary "light it and go prep something else").
# The late side used to be the same 6 ticks unless a sub-case stated its own `lu_late`, and that is what starved
# leave_unattended whenever an earlier takeover displaced it: the forget freeze held the cook's body for 60 ticks,
# the next takeover slid past the 6-tick window, and the walk-away was refused at both ends of the same episode
# ("not near done yet" at t95, "the walk-away moment has passed" at t116 — b3_expert_nops_hard_map_4).
#
# The walk-away must still READ as walking away from FINISHED food, so the window is tied to the smoke grace: the
# food is visibly done and not yet smoking for `smoke_grace` (25, tight pace) ticks, and the walk-away may fire in
# all of that but the last LU_LATE_SMOKE_MARGIN ticks of it. 25 - 5 = 20. The margin is what keeps the mistake
# readable and catchable: firing at the very edge would put the walk-away and the smoke in the same breath, with
# no stretch of done-but-clean food for the assistant to point at. Nothing here can put a walk-away on smoking
# food: lu_moment refuses a smoking station outright, whatever the window says.
LU_LATE_SMOKE_MARGIN = int(os.environ.get("COOKSIM_LU_LATE_MARGIN", 5))


def lu_late_ticks(game):
    """Ticks past "done" a walk-away may still fire (see LU_LATE_SMOKE_MARGIN). COOKSIM_LU_LATE overrides."""
    v = os.environ.get("COOKSIM_LU_LATE")
    if v:
        return max(0, int(v))
    grace = int(getattr(getattr(game, "config", None), "smoke_grace", 25) or 25)
    return max(lu_near_ticks(game), grace - LU_LATE_SMOKE_MARGIN)


def lu_moment(game, c, near, late=None):
    """(ok, why): is some unit of c on the heat within `near` ticks of its cooked transition, not smoking? `late`
    (D17, round 3) widens the window after done to that many ticks (still never on smoking food)."""
    from cooksim.core.stations import CookStation
    best = None
    for st_ in game.stations.values():
        if not (isinstance(st_, CookStation) and st_.contents and st_.cooking
                and any(getattr(x, "name", None) == c for x in st_.contents)):
            continue
        if getattr(st_, "smoking", False) or st_.status not in ("cooking", "cooked"):
            continue
        d = st_.timer - st_.cook_time          # < 0: ticks still to go; >= 0: ticks since done
        if -near <= d <= max(near, late or 0):
            return True, None
        best = d if best is None else (d if abs(d) < abs(best) else best)
    if best is None:
        return False, f"no {c} cooking on the heat yet"
    if best < 0:
        return False, f"the {c} is not near done yet (fires within {near} ticks of done)"
    return False, (f"the {c} has been done {best} ticks, more than {max(near, late or 0)} "
                   f"(the walk-away moment has passed)")


def serve_raw_walk_estimate(game, c):
    """Ticks the serve_raw block needs to land its raw scoop: the walk to the nearest c source plus the scoop."""
    try:
        from cooksim.agents.highlevel import _object_ids
        cx, cy = game.cooks[0].pos
        ids = _object_ids(game)
        d = [abs(x - cx) + abs(y - cy) for (x, y), lbl in ids.items() if lbl.startswith(f"{c} source")]
        return (min(d) if d else 10) + 3
    except Exception:                                   # noqa: BLE001
        return 12



def lift_nested_sup_keys(out):
    """Schema tolerance (2026-08-26 sweep audit): some assistants nest the reply fields
    (`text`, `flag`, `goal_change`) INSIDE `cook_progress` instead of beside it. qwen-27b did
    so on 98% of its turns, and the harness read every one as silence — a JSON-nesting slip,
    not an assistance failure. Lift them to the top level when the top level lacks them."""
    if not isinstance(out, dict):
        return out
    cp = out.get("cook_progress")
    if isinstance(cp, dict):
        for k in ("text", "flag", "goal_change"):
            if out.get(k) in (None, "", {}) and cp.get(k) not in (None, "", {}):
                out[k] = cp.pop(k)
    return out

def prep_progress(ro):
    """Suite v3 (2026-08-27): prep-phase progress read off the action log — (chops done, ignites done).
    Path-invariant in the DAG sense (a reordering cook passes through the same counts at the same
    subgoal depth) and monotone, so it can extend the plate-only anchor without re-importing the
    station/counter fingerprints the plate-only design removed. Lets errors and questions anchor
    BEFORE the plate exists (serve_raw on fish/steak, walk-aways at prep time, 'tell me when X is
    done' requests)."""
    if ro is None:
        return None
    log = getattr(ro, "log", None) or []
    chops = sum(1 for l in log if l.get("status") == "done" and str(l.get("cmd", "")).strip() == "chop")
    ignites = sum(1 for l in log if l.get("status") == "done" and str(l.get("cmd", "")).strip() == "turn on")
    return (chops, ignites)


def anchor_reached(now, ref):
    """Has the live kitchen reached the anchor? Plate exactly as anchored; prep progress at least."""
    if now.get("plate") != ref.get("plate") or now.get("plate_held") != ref.get("plate_held"):
        return False
    if ref.get("plate") or ref.get("plate_held"):
        return True          # plate phase: the plate alone is the key (v2 semantics) — an earlier serve_raw removes prep events for good
    a, b = now.get("prep"), ref.get("prep")
    if a is None or b is None:
        return True
    return a[0] >= b[0] and a[1] >= b[1]


def inject_key(game, ro=None):
    """The situation the injection is anchored to: THE PLATE — plus, since suite v3, the
    prep-phase progress (chops, ignites) from the action log when the rollout is given.

    Design philosophy (user, final, 2026-08-12): the task is a DAG over subgoals, and
    injection timing keys on SUBGOAL COMPLETION — which must be invariant across every
    legal way of accomplishing the task. The in-hand plate status is exactly that
    invariant: every unavoidable checkpoint passes through the plate, and every legal
    linearization of the DAG produces the same plate sequence.

    History, so this never grows back: the original state_key fingerprinted stations and
    counters by exact (x,y) — a cook using the other pan could never match. Its
    replacement (this function) was SUPPOSED to be plate-led per the user's instruction,
    but grew a station-contents multiset and a counter-items multiset on the main agent's
    own initiative. Those extra terms re-imported the same disease one level up: a parked
    item or an alternate pan instance changed the fingerprint forever, and every anchor
    deeper than ~step 11 stopped firing (ECR-9: anchors ≥26 fired 0/9). The plate terms
    alone are path-invariant; the kitchen terms never were.
    """
    held = _norm_item(game.cooks[0].holding)
    plate = list(plate_list(game))
    if not recipe_is_ordered(game):
        # ORDER-FREE dish (caprese, fish & chips, steak & chips): every legal linearization
        # may plate the components in any order, so the only path-invariant checkpoint is
        # HOW MANY items are on the plate. A cook who plates cheese before tomato is at the
        # same subgoal depth as the clean plan's tomato-first cook (2026-08-26 audit: every
        # "past the point" abandonment on caprese / fish & chips / steak & chips was an
        # order choice, not a mistake). Wildcard tokens keep prefix/monotone logic unchanged.
        plate = ["*"] * len(plate)
    key = {"plate": tuple(plate),
           "plate_held": bool(held and held[0] == "plate")}
    if ro is not None:
        key["prep"] = prep_progress(ro)
    return key


def recipe_is_ordered(game):
    """True when the dish being cooked requires a plating ORDER (a stacked burger/burrito);
    False for an order-free multiset dish. Unknown -> True (the stricter reading)."""
    try:
        recs = [o.recipe for o in game.orders.orders] or list(game.orders.pool)
        return any(bool(getattr(r, "ordered", False)) for r in recs) if recs else True
    except Exception:                                   # noqa: BLE001
        return True


def plate_names(game):
    cook = game.cooks[0]
    h = _norm_item(cook.holding)
    if h and h[0] == "plate":
        return set(h[1])
    for it in game.counter_items.values():
        n = _norm_item(it)
        if n and n[0] == "plate":
            return set(n[1])
    return set()


# ---- ASSISTANT LATENCY policy (user ruling 2026-09-04) ------------------------------------
TICK_SECONDS = float(os.environ.get("COOKSIM_TICK_SECONDS", 1.2))     # the human app's tick (parity)


def assist_latency_ticks(x_seconds: float) -> int:
    """How many ticks a reply that took x wall seconds arrives after it was requested.
    COOKSIM_ASSIST_LATENCY: unset/"off"/"0" -> 0 (legacy: replies arrive at once);
    "measured" -> ceil(x / TICK_SECONDS); an integer -> that fixed delay for every reply."""
    import math
    mode = (os.environ.get("COOKSIM_ASSIST_LATENCY") or "off").strip().lower()
    if mode in ("", "off", "0", "none"):
        return 0
    if mode == "measured":
        raw = int(math.ceil(max(0.0, x_seconds) / TICK_SECONDS))
        # D-1 ruling 2026-09-11: a single reply is charged at most COOKSIM_ASSIST_LATENCY_MAX
        # ticks (default 30). The 84-124 s calls in the v5 sample were API hangs at the HTTP
        # timeout; charging them in full removed the assistant for longer than a burn window.
        cap = int(os.environ.get("COOKSIM_ASSIST_LATENCY_MAX", "30") or 0)
        return min(raw, cap) if cap > 0 else raw
    return max(0, int(mode))




# =====================================================================================
# SCHEDULER v5 (rulings 2026-09-09): the cook plans for itself (no template in its queue at
# t0 or after any release); every pending sub-case is checked every tick against a
# path-invariant anchor (plate prefix + prep progress as a set, never template counters);
# errors/detours share one exclusive BODY slot, questions/requests use a VOICE slot that can
# open during a takeover; the takeover block is composed against the plan that REMAINS from
# the live kitchen, and nothing is queued behind it.
# =====================================================================================
SCHED_V5 = os.environ.get("COOKSIM_SCHED", "v5") == "v5"
VOICE_GAP = int(os.environ.get("COOKSIM_VOICE_GAP", 8))   # min ticks between two voice asks (v5)
VOICE_QUIET = int(os.environ.get("COOKSIM_VOICE_QUIET", 3))  # A5: ticks a voice slot stays shut after a takeover start / commit / flag
# E2/E3 (round-2 review, 2026-09-19): the minimum number of ticks between the RELEASE of one error takeover and the
# START of the next body sub-case, and the ticks after a release in which no question/request opens. The cook must
# be able to hear a correction of the mistake that just committed and act on it before its body is taken again.
# G = 10, from the 216 API-assistant episodes of v5 samples 1-4 (10,098 replies, COOKSIM_ASSIST_LATENCY=measured):
#   the assistant's next poll after the release        <= 1 tick  (POLL = 2)
#   its reply latency                                   p50 4, p75 6, p90 8 ticks (Flash p75 4; Pro / gpt-5.5 p75 9-10)
#   correct flag -> the cook's first corrective command p50 3 ticks (206 flags; p75 8)
# 1 + 6 + 3 = 10 lets a p75-latency assistant's correction reach the cook and the median cook act on it before the
# next takeover; a larger gap starts to push narrow moments out (leave_unattended fires within 6 ticks of done).
# COOKSIM_TAKEOVER_GAP overrides.
TAKEOVER_GAP = int(os.environ.get("COOKSIM_TAKEOVER_GAP", 10))
# G9 (round-5 review): the longest the voice slot may be held shut by a takeover that is only ABOUT to start.
# Two episodes lost every question they had left to that hold: hard_map_1 B from t55 to its t88 timeout, and
# medium_map_5 (both arms) for the whole episode. A takeover that has not started within this many ticks of
# becoming "imminent" is not imminent; the cook may speak.
VOICE_IMMINENT_CAP = int(os.environ.get("COOKSIM_VOICE_IMMINENT_CAP", 2 * TAKEOVER_GAP))
SCHED_V5_SHORT_OMIT = os.environ.get("COOKSIM_SHORT_OMIT", "1") == "1"   # round-3 audit: omission = skip + plate the next item
SCRIPTED_COOK = os.environ.get("COOKSIM_SCRIPTED_COOK")       # template|ignite_at_load|reverse|plate_early


def prep_units(game, recipe, cap=True):
    """Path-invariant prep progress: for each ingredient the recipe wants processed, how many
    units have entered their processing stage (chopped on a board, loaded in a heat station,
    or already plated in the wanted state). Counts what exists NOW; the harness keeps a
    running max so a trash-and-rebuild never lowers it.

    cap=True clamps each count at what the recipe needs (a progress FLOOR must not be moved by an
    extra unit). fire-longtail (2026-09-20): repeat_step on a PREP commits exactly when a unit
    BEYOND the recipe's need enters processing, and the clamp made that test unsatisfiable — 9 of 9
    fired prep repeats in the probe run recorded "no extra meat entered processing (1 of 1)". That
    ledger reads the uncapped count."""
    from cooksim.core.items import Plate
    want = {}
    for n, st in recipe.contents:
        st = getattr(st, "value", st)
        if st in ("cooked", "chopped"):
            want[n] = want.get(n, 0) + 1
    got = {n: 0 for n in want}
    def _nm(x):
        return getattr(x, "name", None) or (x.get("name") if isinstance(x, dict) else None)
    def _st(x):
        v = getattr(x, "state", None)
        v = getattr(v, "value", v)
        if v is None and isinstance(x, dict):
            v = x.get("state")
        return v
    for st_ in game.stations.values():
        d = st_.render_state()
        for c in (d.get("contents") or []):
            if _nm(c) in got and d.get("status") in ("raw", "cooking", "cooked"):
                got[_nm(c)] += 1
        it = d.get("item")
        if it is not None and _nm(it) in got and (_st(it) == "chopped" or d.get("done")):
            got[_nm(it)] += 1
    plates = [game.cooks[0].holding] + list(game.counter_items.values())
    for it in plates:
        if isinstance(it, Plate):
            for c in (it.contents or []):
                if _nm(c) in got and _st(c) in ("cooked", "chopped"):
                    got[_nm(c)] += 1
        elif it is not None and _nm(it) in got and _st(it) in ("cooked", "chopped"):
            got[_nm(it)] += 1          # a chopped item carried in hand / parked on a counter
    return {n: (min(v, want[n]) if cap else v) for n, v in got.items()}


def prep_geq(live_max, req):
    return all(live_max.get(n, 0) >= k for n, k in (req or {}).items())


def remaining_plan(game, state, plate_now):
    """The template's subgoals that are still to do, judged from the LIVE kitchen: prep/ignite
    blocks for units already processed are dropped, assemble blocks for items already on the
    plate are dropped, fetch_plate is dropped when a plate is already in hand or parked."""
    from cooksim.core.items import Plate
    nodes = list(segment_plan(goal_of(game, state)))
    have = prep_units(game, goal_of(game, state))
    plated = list(plate_now)
    # F9 (round-3 review): one assemble node may plate a BATCH (Steak Frites / Fish & Chips scoop both potatoes with one
    # "scoop"). A cook that plated one potato used to lose the node entirely, so a mistake on "the potato" could not be
    # composed any more ("cannot compose on the live kitchen", expert c medium_3) and the scripted cook served one
    # potato short. A node is dropped only for every full batch of its units already plated.
    import collections as _col
    _need = _col.Counter(n for n, _s in goal_of(game, state).contents)
    _nodes_of = _col.Counter(n.component for n in nodes if n.type == "assemble")
    _per = {c: max(1, _need.get(c, 1) // max(1, k)) for c, k in _nodes_of.items()}
    _drop = {c: _col.Counter(plated)[c] // _per[c] for c in _nodes_of}
    seen_prep, out = {}, []
    have_plate = isinstance(game.cooks[0].holding, Plate) or any(
        isinstance(it, Plate) for it in game.counter_items.values())
    from cooksim.interact.system import station_holding
    for n in nodes:
        if n.type == "ignite":
            found = station_holding(game, n.component)
            if found is not None and found[1].get("status") in ("cooking", "cooked"):
                continue                      # already lit
            if found is None and have.get(n.component, 0) == 0:
                pass                          # nothing loaded yet: keep the ignite after its prep
        elif n.type == "prep":
            comp = n.component
            k = seen_prep.get((n.type, comp), 0)
            seen_prep[(n.type, comp)] = k + 1
            # a chain ingredient (chop, then cook) whose unit sits CHOPPED ON A BOARD is only half
            # processed: keep the tail of its prep block (pick it up, load it) instead of dropping it
            if "chop" in n.commands and any(c.startswith("go to the empty ") and "board" not in c for c in n.commands):
                on_board = 0
                for st_ in game.stations.values():
                    d = st_.render_state()
                    it = d.get("item")
                    if it is not None and (getattr(it, "name", None) or (it.get("name") if isinstance(it, dict) else None)) == comp \
                            and "board" in type(st_).__name__.lower():
                        on_board += 1
                loaded = 0
                for st_ in game.stations.values():
                    d = st_.render_state()
                    loaded += sum(1 for c in (d.get("contents") or []) if (getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)) == comp)
                plated_n = sum(1 for x in plate_now if x == comp)
                if k >= loaded + plated_n and k < loaded + plated_n + on_board:
                    from cooksim.core.plan_ir import Subgoal
                    j = n.commands.index("chop") + 1
                    tail = [f"go to the cutting board with the {comp}"] + list(n.commands[j:])
                    if "turn on" in n.commands and "turn on" not in tail:
                        pass
                    out.append(Subgoal(n.sid, "prep", comp, tail, origin="plan"))
                    continue
            if (k >= have.get(comp, 0) and station_holding(game, comp) is None
                    and sum(1 for x in plate_now if x == comp) > 0):
                fixed = [c.replace(f"go to the ", "go to the empty ").split(" with the ")[0]
                         if (c.startswith("go to the ") and c.endswith(f" with the {comp}") and "board" not in c) else c
                         for c in n.commands]
                if fixed != list(n.commands):
                    from cooksim.core.plan_ir import Subgoal
                    out.append(Subgoal(n.sid, "prep", comp, fixed, origin="plan"))
                    continue
            if k < have.get(comp, 0):
                # this unit is already processed; a single-heat-group prep carries its own
                # "turn on" — if the loaded station is still cold, keep a lighting step
                if "turn on" in n.commands and not any(x.type == "ignite" and x.component == comp for x in nodes):
                    found = station_holding(game, comp)
                    if found is not None and found[1].get("status") == "raw":
                        lbl = next((c[len("go to the empty "):] for c in n.commands if c.startswith("go to the empty ")), None)
                        if lbl and not any(o.sid == f"ignite({comp})" for o in out):
                            from cooksim.core.plan_ir import Subgoal
                            out.append(Subgoal(f"ignite({comp})", "ignite", comp,
                                               [f"go to the {lbl} with the {comp}", "turn on"]))
                continue
        elif n.type == "assemble":
            if _drop.get(n.component, 0) > 0:
                _drop[n.component] -= 1
                continue
        elif n.sid == "fetch_plate" and have_plate:
            continue
        out.append(n)
    return out


def live_dev_block(err_call, game, state, plate_now):
    """Compose the error against the plan that remains from the live kitchen and return the
    deviated section only (whole subgoal blocks): (commands, sids, first_dev_cmd, err_ir)."""
    rem = remaining_plan(game, state, plate_now)
    flat_rem = [(c, n.sid) for n in rem for c in n.commands]
    name_, kw_ = err_call
    rem_sids = {n.sid for n in rem}
    already_parked = False        # set when a prefix below has already emptied the cook's hands
    if kw_.get("after") and kw_["after"] not in rem_sids and kw_["after"] != "__start__":
        # the step this error follows is already done in the live kitchen: the error starts NOW
        if name_ in ("premature_serve", "omit_subgoal"):
            from cooksim.interact.system import plate_into_hand
            pre = plate_into_hand(game)
            cmds = (pre if isinstance(pre, list) else []) + ["go to the serving pass", "serve"]
            block = [(c, "premature_serve") for c in cmds]
            return ([c for c, _s in block], {"premature_serve"}, block[0][0], list(block))
        if name_ == "forget_next_step":
            from cooksim.core.config import forget_freeze_ticks as _freeze
            cmd_ = f"idle {_freeze()}"                      # ONE constant, shared with plan_ir.forget_rest
            block = [(cmd_, "lost")]
            return ([cmd_], {"lost"}, cmd_, list(block))
        kw_ = dict(kw_); kw_["after"] = rem[0].sid if rem else kw_["after"]
        err_call = (name_, kw_)
    if name_ == "omit_subgoal" and SCHED_V5_SHORT_OMIT:
        # round-3 audit: the omission used to take the cook's body through the WHOLE remaining recipe
        # (stale preps with the plate in hand, 8-18 refused commands, 100+ tick holds, burns before it
        # ever landed). The mistake is "the cook thinks the step is done and moves on": skip the
        # component, plate the NEXT recipe item (the moment it commits), then hand back. When the
        # skipped component is the last one, the moving-on is the serve.
        from cooksim.core.plan_ir import segment_plan
        from cooksim.interact.system import plate_into_hand
        sid_ = kw_.get("sid") or ""
        comp_ = sid_.split("(")[1].rstrip(")") if "(" in sid_ else None
        items_ = [n for n, _ in goal_of(game, state).contents]
        if comp_ not in items_:
            return None
        nxt_ = items_[items_.index(comp_) + 1] if items_.index(comp_) + 1 < len(items_) else None
        pre_ = plate_into_hand(game)
        pre_ = pre_ if isinstance(pre_, list) else []
        if nxt_ is None:
            cmds_ = ["go to the serving pass", "serve"]
        else:
            node_ = next((n for n in segment_plan(goal_of(game, state)) if n.sid == f"assemble({nxt_})"), None)
            if node_ is None:
                return None
            cmds_ = list(node_.commands)
        block = [(c, f"omit_skip({comp_})") for c in pre_ + cmds_]
        return ([c for c, _s in block], {s_ for _c, s_ in block}, block[0][0], list(block))
    if name_ == "repeat_step":
        # the repeated step is by definition already done, so it is never in the remaining plan:
        # re-do the template node itself, NOW (the trigger `at` is the anchor, already reached)
        from cooksim.core.plan_ir import segment_plan, repeat_of
        node = next((n for n in segment_plan(goal_of(game, state)) if n.sid == kw_.get("sid")), None)
        if node is None:
            return None
        dup = repeat_of(node)
        dup.sid = f"{node.sid}#repeat"
        # fire-longtail (2026-09-20): FORCE the duplicate. The template node for a CHOPPED item is "go to the
        # cutting board with the x, scoop", which grounds to nothing once the cook has plated the only chopped unit
        # — the repeat then scooped air. The confused cook makes the second unit itself (the same self-made-prep
        # pattern D12 gave wrong_order). Raw items come straight from the crate; a cooked item's second unit can
        # only come from the kitchen, so error_precondition waits for one instead.
        from cooksim.interact.system import ingredient_ready as _ir2
        from cooksim.core.items import Plate as _Pl2
        h2 = game.cooks[0].holding
        if str(kw_.get("sid") or "").startswith("assemble("):
            comp2 = _sid_comp(kw_.get("sid"))
            want2 = dict((n, getattr(s, "value", s)) for n, s in goal_of(game, state).contents).get(comp2)
            if want2 == "chopped" and not _ir2(game, comp2, "chopped"):
                chop_ = [f"go to the {comp2} source", "pick up", "go to the empty cutting board", "put down", "chop"]
                if h2 is not None and getattr(h2, "name", None) == comp2 and not isinstance(h2, _Pl2):
                    chop_ = ["go to the empty cutting board", "put down", "chop"]
                elif isinstance(h2, _Pl2):
                    # park THE dish plate and pick THAT one up again (D12's pattern). Letting the generic fetch
                    # take "go to the plates" got a FRESH plate and the duplicate landed on a scratch plate that
                    # nothing ever served (probe easy_repeat_step: every command legal, no duplicate on the dish).
                    chop_ = ["go to the empty counter", "put down"] + chop_ + \
                            ["go to the counter with the plate", "pick up"]
                    already_parked = True
                elif h2 is not None:
                    chop_ = ["go to the empty counter", "put down"] + chop_
                    already_parked = True
                dup.commands = chop_ + list(dup.commands)
        elif h2 is not None:
            # a PREP repeat needs empty hands for its own pick up. With the dish plate still in them the whole
            # block was refused and no extra unit ever entered processing (probe medium/hard/expert, 9/9).
            dup.commands = ["go to the empty counter", "put down"] + list(dup.commands)
            already_parked = True
        err_ir, _recs = [dup] + list(rem), []
    else:
        err_ir, _recs = X.E.compose(list(rem), [err_call])
    flat_err = [(c, n.sid) for n in err_ir for c in n.commands]
    marked = [n for n in err_ir if getattr(n, "deviated", False) or getattr(n, "origin", "plan") == "injected"]
    if not marked and name_ != "omit_subgoal":
        return None                                   # the error could not be composed on this kitchen
    if marked:
        # the error's own steps, whatever order the surrounding plan is in (mis_prep's replaced
        # prep + its ignite; the swapped pair; the foreign scoop; the wander; the stall)
        block = [(c, n.sid) for n in marked for c in n.commands]
        if name_ == "leave_unattended" and block:
            from cooksim.core.items import Plate as _Pl
            if isinstance(game.cooks[0].holding, _Pl):
                # F5 (round-3 review): the busywork route is "other crates, plates, serving pass". With the half-built
                # plate in hand that walk to the pass is a premature-serve picture (expert b hard_5, c medium_5), so a
                # correct "don't serve that yet" was scored under the wrong name. The cook sets the plate down first.
                block = [("go to the empty counter", block[0][1]), ("put down", block[0][1])] + list(block)
                already_parked = True
        if name_ == "mis_prep" and block and game.cooks[0].holding is not None:
            # Round-3 self-review (final run, classic_expert master_hard_1): mis_prep fires when the cook reaches for the
            # component, including when it already HOLDS it (its own "pick up" finished on the injection tick). The block
            # then went to the wrong crate with full hands ("pick up [illegal-skipped] — your hands are full"), carried
            # the right tomato to the board and chopped it: the mistake never happened. The cook sets down what it holds
            # first.
            block = [("go to the empty counter", block[0][1]), ("put down", block[0][1])] + list(block)
            already_parked = True
        if name_ == "wrong_order":
            # B1: the takeover now runs its WHOLE block (b's scoop, then a's). a's scoop can only run when some a is on
            # a station or ready (or a is plated as it comes); with a not even started it is refused ("go to the pan
            # with the meat" and no pan has meat — verifier, medium_medium_2 plate_early). Then the mistake is b alone,
            # and the cook adds a itself once it has made it.
            from cooksim.interact.system import ingredient_ready as _ir, station_holding as _sh
            a_ = kw_.get("a")
            want_a = dict((n, getattr(s, "value", s)) for n, s in goal_of(game, state).contents).get(a_)
            # fire-longtail (2026-09-20): "on a station" was too lenient once the swap stopped waiting for a to be the
            # next item to plate — a RAW a on a cold pan, or a raw a on a board, kept a's scoop in the block and it
            # was refused ("the meat is sitting in a station that was never turned on"; "nothing ready to scoop here":
            # 5 refusals in the probe run). a's scoop stays only when a is ready, plated as it comes, or on a LIT
            # station that will be ready by the time the block walks there.
            _lit_a = any(getattr(st_, "status", None) in ("cooking", "cooked")
                         and any(getattr(c_, "name", None) == a_ for c_ in (getattr(st_, "contents", None) or []))
                         for st_ in game.stations.values())
            if not ((want_a in (None, "raw")) or _ir(game, a_, want_a) or _lit_a):
                block = [(c, s_) for c, s_ in block if s_ != f"assemble({a_})"] or block
            # D12 (round-3 review): b not chopped yet — the swap used to wait for a chopped b that a cook who plates the
            # steak before chopping the tomato never has (master wrong_order(steak, tomato) missed 4/8). The mistake is
            # "the cook thinks b comes first": with a waiting to be plated it sets the plate down, makes b, and plates
            # it before a. a stays under the live hold (E10) while the block runs.
            b_ = kw_.get("b")
            want_b = dict((n, getattr(s, "value", s)) for n, s in goal_of(game, state).contents).get(b_)
            if want_b == "chopped" and not _ir(game, b_, want_b):
                from cooksim.core.items import Plate as _Pl
                pn = next((n for n in rem if n.type == "prep" and n.component == b_ and "chop" in n.commands), None)
                h_ = game.cooks[0].holding
                if pn is None:
                    return None
                prep_cmds = list(pn.commands[:pn.commands.index("chop") + 1])
                pre = []
                if isinstance(h_, _Pl):
                    pre += ["go to the empty counter", "put down"]
                elif h_ is not None and getattr(h_, "name", None) == b_:
                    prep_cmds = [c for c in prep_cmds if not (c.endswith(" source") or c == "pick up")]
                elif h_ is not None:
                    return None                           # hands full of something else: not composable now
                # fire-longtail (2026-09-20): the retrieval used to be the literal "go to the counter with the plate",
                # which was right only because the trigger demanded a plate already in the cook's hands. With the
                # forced trigger the swap can fire with NO plate anywhere, and that walk grounded to nothing: the
                # block chopped the tomato, walked nowhere and scooped nothing (probe master_wrong_order, 2 policies
                # "executed" without a single item plated). Retrieve the plate the block parked; otherwise fetch one.
                pre += prep_cmds + (["go to the counter with the plate", "pick up"] if isinstance(h_, _Pl)
                                    else _plate_pickup_cmds(game))
                block = [(c, f"wrong_order_prep({b_})") for c in pre] + list(block)
                already_parked = True
    else:
        # a pure skip (omit_subgoal): the cook believes the step is done and carries the rest of
        # the plan through to the serve — that whole tail is the deviated block
        block = list(flat_err)
    # a block that lights a station must wait for it before its own scoop (the template has no
    # waits; the executor's scoop-wait is shorter than some cook times)
    out_ = []
    for c, sid_ in block:
        out_.append((c, sid_))
        if c == "turn on":
            prev = next((x for x, _ in reversed(out_[:-1]) if x.startswith("go to the ")), None)
            if prev and " with the " in prev:
                out_.append((prev.replace("go to the ", "wait until the ") + " is cooked", sid_))
            elif prev and prev.startswith("go to the empty "):
                lbl = prev[len("go to the empty "):]
                out_.append((f"wait until the {lbl} is cooked", sid_))
    block = out_
    # The error's own steps need the plate in hand when they scoop, and the block that SERVES needs it too; the
    # template blocks assume both. fire-longtail (2026-09-20): the fetch used to be prefixed only when a scoop sat
    # in the block's first three commands, which was true of the old short plating blocks and false of every FORCED
    # one that first parks the hands or chops a second unit — those scooped into empty hands. The fetch now goes
    # immediately before the walk that leads to the block's FIRST scoop, and before a `serve` with no plate in hand.
    block = _ensure_plate_for_block(block, game, parked=already_parked)
    return ([c for c, _s in block], {s_ for _c, s_ in block},
            (block[0][0] if block else None), list(block))


def _free_stations(label, game):
    """How many stations of `label`'s kind are free (nothing in them) right now."""
    n = 0
    for st_ in (getattr(game, "stations", None) or {}).values():
        kind = type(st_).__name__.lower()
        nm = (getattr(st_, "name", "") or "").lower()
        if label.replace(" ", "") not in (kind + nm).replace("_", "").replace(" ", ""):
            continue
        if getattr(st_, "item", None) is None and not (getattr(st_, "contents", None) or []):
            n += 1
    return n


def block_stations_ok(cmds, game):
    """G4 (round-5 review): does the kitchen have the free stations this forced block walks to? Returns None when
    it does, else a short reason the scheduler waits on.

    A forced block used to fire whatever the kitchen looked like. `go to the empty cutting board` with no board
    free grounds to NOTHING, and the whole block after it ran as refused commands — a 31-tick window of refusals in
    hard medium_map_4 C (t190, window [173,204]) and expert medium_map_2 c (t303-320), and in the repeat_step case
    a mistake was RECORDED as catchable that never physically happened. A station the block needs is a
    precondition of the block, exactly like an ingredient source."""
    want = [str(c).strip() for c in (cmds or []) if str(c).strip().startswith("go to the empty ")]
    if not want:
        return None
    seen = {}
    for c in want:
        seen[c] = seen.get(c, 0) + 1
    for c, k in seen.items():
        label = c[len("go to the empty "):].strip()
        if not _goto_grounds(c, game):
            return f"no free {label} for the block's {c!r}"
        if k > 1 and "counter" not in label and _free_stations(label, game) < k:
            return f"the block needs {k} free {label}s and the kitchen has {_free_stations(label, game)}"
    return None


def _goto_grounds(cmd, game):
    """Has this walk somewhere to go right now? An AMBIGUOUS walk ("the empty cutting board" with two free
    boards) has several destinations and is bound to one when it is issued — that is not a failure. Only a walk
    with NO candidate at all is refused by the engine, and that is what block_stations_ok asks about."""
    try:
        from cooksim.agents.highlevel import parse as _hp, ground as _hg
        it = _hp(str(cmd))
        if it is None or it.verb != "goto":
            return True
        if _hg(it, game, game.cooks[0]) is not None:
            return True
        return bool(it.quals.get("_ambiguous"))
    except Exception:                                   # noqa: BLE001
        return True


def error_precondition(err, kw, game, state, plate_now, smoke_ok=False):
    """Is this error executable and meaningful in the live kitchen right now? Returns None
    when it is, else a short reason (the scheduler waits while the stage holds, then skips)."""
    from cooksim.interact.system import ingredient_ready, station_holding, _has_source
    from cooksim.core.stations import CookStation
    recipe = goal_of(game, state)          # E6: the current goal (recipe + applied plan changes)
    want = {n: getattr(st, "value", st) for n, st in recipe.contents}
    plated = list(plate_now)
    # B3 (fix-firing): "already plated" means plated IN ITS WANTED STATE — a raw or misplaced unit left by an earlier
    # committed mistake does not fill the item's slot. The plate is read from the live kitchen, by name and state.
    items_now = plate_items(game)
    good = correct_counts(recipe, items_now)
    need_n = {n: sum(1 for m, _ in recipe.contents if m == n) for n, _ in recipe.contents}
    cap_ = int(getattr(getattr(game, "config", None), "plate_capacity", 5) or 5)
    if err in SMOKE_OK_ERRS and len(items_now) >= cap_:
        return f"the plate is full ({len(items_now)} items): a scoop cannot land"
    if err in ("add_foreign", "misidentify_assemble"):
        w = kw.get("wrong")
        if w and not _has_source(game, w):
            return f"no source for {w}"
    if err == "wrong_order":
        for c in (kw.get("a"), kw.get("b")):
            if good[c] >= need_n.get(c, 1):
                return f"{c} already plated"
        # only the item plated too EARLY must exist: plating b while still waiting for a is the ordinary form
        # of this mistake. Requiring a ready too starved every live run -- a self-planning cook plates a the
        # tick it is done, so "a ready and not plated" never held (verifier 2026-09-15, 4 master episodes).
        b_ = kw.get("b")
        if not ingredient_ready(game, b_, want.get(b_)) and not (want.get(b_) == "chopped" and _has_source(game, b_)):
            # D12 (round-3 review): a chopped b the block can make itself does not have to be ready (live_dev_block)
            return f"{b_} not ready"
    if err == "serve_raw":
        c = kw.get("component")
        if c in plated:
            return f"{c} already plated"         # in any state: a raw one already there IS the mistake
        if not _has_source(game, c):
            return f"no source for {c}"
        # fire-longtail (2026-09-20): "the item must genuinely be raw/unprocessed". A component the recipe plates as
        # it comes (bun, tortilla, cheese) has no raw form to serve wrongly, so serve_raw is not a mistake there at all.
        if (want.get(c) or "raw") == "raw":
            return f"the recipe plates the {c} raw anyway (serving it raw is not a mistake)"
    if err == "mis_prep":
        c, w = kw.get("component"), kw.get("wrong")
        have = prep_units(game, recipe)
        need = sum(1 for n, _ in recipe.contents if n == c)
        if have.get(c, 0) >= need:
            return f"{c} already prepped"
        if w and not _has_source(game, w):
            return f"no source for {w}"
    if err == "leave_unattended":
        if not any(isinstance(st_, CookStation) and st_.contents and st_.status in ("cooking", "cooked")
                   and not getattr(st_, "smoking", False)
                   for st_ in game.stations.values()):
            return "nothing on the heat (or it is already smoking)"
    # v5 sample audit (2026-09-09): a takeover injected while a station smokes manufactures the
    # burn — the hands the cook needs for the rescue are taken. Every body error waits for a
    # kitchen that is not smoking.
    # 2026-09-15: relaxing this to "block + rescue time fits before fire" was tried twice and burned 21/36,
    # then 10/36 verifier episodes (scripted cooks resume their plan, not the rescue). The ban stays; the
    # first-cooked-item errors it starved now fire EARLIER, in the grace period (live_trigger, no plate yet).
    if not smoke_ok and any(getattr(st_, "smoking", False) and getattr(st_, "contents", None) for st_ in game.stations.values()):
        return SMOKE_WAIT
    if err in ("omit_subgoal", "premature_serve") and not kw.get("sid"):
        if not plated:
            return "nothing plated yet (an empty plate cannot be served)"
    if err == "omit_subgoal":
        sid = kw.get("sid") or ""
        comp = sid.split("(")[1].rstrip(")") if "(" in sid else None
        if comp and good[comp] >= need_n.get(comp, 1):
            return f"{comp} already plated"
        items = [n for n, _ in recipe.contents]
        if SCHED_V5_SHORT_OMIT and comp in items and items.index(comp) + 1 < len(items):
            nxt = items[items.index(comp) + 1]
            if not ingredient_ready(game, nxt, want.get(nxt)):
                return f"{nxt} not ready (the omission plates it next)"
    if err in ("add_foreign", "misidentify_assemble") and kw.get("before"):
        target = kw["before"]
        comp = target.split("(")[1].rstrip(")") if "(" in target else None
        if target.startswith("assemble(") and comp and good[comp] >= need_n.get(comp, 1):
            return f"{target} already done"
        if target == "serve" and False:
            pass
    if err == "misidentify_assemble" and kw.get("component"):
        c = kw["component"]
        if good[c] >= need_n.get(c, 1):
            return f"{c} already plated"
    if err == "forget_next_step" and kw.get("after"):
        # design rule (2026-09-15): the forgotten step is never the serve — see err_forget_next_step
        if plate_complete(recipe, items_now) or names_complete(recipe, items_now):
            return "plate complete — the forgotten step would be the serve (design rule)"
    if err == "repeat_step":
        sid = kw.get("sid") or ""
        comp = sid.split("(")[1].rstrip(")") if "(" in sid else None
        if comp and sid.startswith("prep") and not _has_source(game, comp):
            return f"no source for {comp}"
        if comp and sid.startswith("assemble") and comp not in plated:
            return f"{comp} not plated yet (nothing to repeat)"
        if comp and sid.startswith("assemble") and plate_complete(recipe, items_now) and not FORCE_PLATING:
            return "plate already complete (a repeated scoop cannot land)"
        if comp and sid.startswith("prep") and prep_units(game, recipe).get(comp, 0) < 1:
            return f"{comp} not in processing yet (nothing to repeat)"
        # fire-longtail (2026-09-20), user ruling: "allow it again wherever the duplicate is VISIBLE as a duplicate
        # (the bun-on-top confusion is a per-item ban, not a type ban)". The whole type used to be banned because a
        # second bun reads as the burger's top bun; only that item is refused now (X.E.REPEAT_DUPLICATE_INVISIBLE).
        if comp and sid.startswith("assemble") and comp in getattr(X.E, "REPEAT_DUPLICATE_INVISIBLE", set()):
            return f"a second {comp} does not read as a duplicate (per-item ban)"
        # the duplicate has to be SCOOPABLE. A raw component comes from its crate and always is; a chopped one the
        # block chops itself when no chopped unit is left (live_dev_block); a cooked one needs a second cooked unit,
        # which only the kitchen can supply. Without this the repeat block walked to an empty pan and scooped nothing.
        if comp and sid.startswith("assemble") and (want.get(comp) or "raw") == "cooked" \
                and not ingredient_ready(game, comp, "cooked"):
            return f"no second cooked {comp} to scoop (a repeated scoop needs one)"
    return None



LIVE_TRIGGERS = (os.environ.get("COOKSIM_TRIGGERS") or "live") == "live"
# User ruling 2026-09-17: questions and requests fire from the LIVE kitchen too (announced intents do not, see schedule_v5). Their
# reference plate is only a progress FLOOR (enough items on the plate, right or wrong, and enough prep) —
# never an exact match, and never dropped as "stage passed": a question whose moment slipped by is asked
# late. Round 5 measured the old exact-match rule dropping 103 questions and 25 of 27 plan_changes as soon
# as a cook's plate differed from the reference. COOKSIM_QUESTION_TRIGGERS=reference restores it.
LIVE_QUESTIONS = (os.environ.get("COOKSIM_QUESTION_TRIGGERS") or "live") == "live"
# User ruling 2026-09-17, "narrow the ban": LLM cooks start plating late, so the first scoop is usually a rescue of
# food that is already smoking, and the blanket smoke ban blocked every plating error at exactly that moment (the
# LLM-cook check: burrito/cheeseburger serve_raw and wrong_order passed their moment unfired). While a station smokes,
# a SCOOP-type plating error may still be injected when none of its items is the smoking one and its forced block is
# short (<= COOKSIM_SMOKE_SHORT_BLOCK commands) — the rescue is delayed by a few ticks at most, and some extra burns
# are accepted. Every other body error still waits. COOKSIM_SMOKE_BAN=full restores the blanket ban.
# plan_change, cook-planned (user, 2026-09-17: "the plan change will modify task subgoal and goal check so the cook
# should be able to figure it out without our forcing"). The goal is extended at injection (engine goal check +
# the cook's order block) and the cook ANNOUNCES the change and adds the topping itself: no forced block. The forced
# version spliced "go to the <x> source, scoop" into the queue and released only once that scoop succeeded, so fired
# without a plate in hand it hung the cook's hands until timeout (verifier, expert_medium_2). COOKSIM_PLAN_CHANGE=forced
# restores it.
PLAN_CHANGE_FREE = (os.environ.get("COOKSIM_PLAN_CHANGE") or "free") == "free"
PLAN_CHANGE_ANNOUNCE_WAIT = int(os.environ.get("COOKSIM_PLAN_CHANGE_ANNOUNCE_WAIT", 12))
SMOKE_WAIT = "a station is smoking (the cook needs its hands)"
SMOKE_BAN_NARROW = (os.environ.get("COOKSIM_SMOKE_BAN") or "narrow") == "narrow"
SMOKE_SHORT_BLOCK = int(os.environ.get("COOKSIM_SMOKE_SHORT_BLOCK", 4))
SMOKE_OK_ERRS = {"serve_raw", "misidentify_assemble", "add_foreign", "wrong_order", "repeat_step", "omit_subgoal"}


# The smoke hold (B8/E10, 2026-09-17..20) was DELETED on 2026-09-30 (user ruling): the harness never keeps cooked food
# from smoking; a pending plating mistake fires inside the honest done->smoke window (smoke_grace, now 40) or is dropped.


def error_items(err_call):
    """The ingredients a mistake TOUCHES, its anchor included (component, wrong item, swapped pair, the step's
    item, and the item named by the step it is anchored on). This is the wide reading, used by the smoke ban:
    an error whose moment is "just before the fish goes on the plate" must not be injected while the fish smokes."""
    err, kw = err_call
    mine = set(error_units(err_call))
    for k in ("before", "at"):
        v = kw.get(k) or ""
        if "(" in v:
            mine.add(v[v.find("(") + 1:v.rfind(")")].split("#")[0])
    mine.discard(None)
    return mine


def error_units(err_call):
    """G3 (round-5 review): the ingredients the mistake ITSELF acts on — never the item named by the step it is
    merely anchored on. `before` / `at` are ANCHORS ("this fires just before assemble(fish)"), not the mistake's
    own units, and the post-commit restore used to drop every queue segment about them: an add_foreign anchored
    before assemble(fish) absorbed the fish, so the restore dropped "go to pan 1, put down, turn on" and the fish
    sat raw in a cold pan for 152 ticks with `heat_events []` (hard 10/20 episodes, 3 of them fatal)."""
    _err, kw = err_call
    mine = {kw.get(k) for k in ("component", "wrong", "a", "b")}
    v = kw.get("sid") or ""
    if "(" in v:
        mine.add(v[v.find("(") + 1:v.rfind(")")].split("#")[0])
    mine.discard(None)
    return mine


def smoke_narrow_ok(err_call, game, state, plate_now):
    """The narrowed smoke ban: may this error be injected although a station is smoking? Returns (ok, why)."""
    from cooksim.core.stations import CookStation
    err, kw = err_call
    if err not in SMOKE_OK_ERRS or (err == "omit_subgoal" and "after" in kw):
        return False, "not a scoop-type plating error"
    smoking = {getattr(c, "name", None) for st_ in game.stations.values()
               if isinstance(st_, CookStation) and getattr(st_, "smoking", False) for c in (st_.contents or [])}
    mine = set(error_items(err_call))
    items = [n for n, _ in goal_of(game, state).contents]
    if err == "omit_subgoal":
        # the omission's block plates the item AFTER the skipped one: it involves that item too (sample r6 ep 1:
        # omit(rice) injected while the steak smoked, its block was "go to the pan with the steak, scoop", the takeover
        # walked the cook to the smoking steak and the kitchen burned; in LLM-cook check #4 all 4 narrowed omits burned)
        x = next(iter(error_items((err, {"sid": kw.get("sid")}))), None) if kw.get("sid") else None
        if x in items and items.index(x) + 1 < len(items):
            mine.add(items[items.index(x) + 1])
    if mine & smoking:
        return False, f"it involves the smoking {sorted(mine & smoking)[0]}"
    blk = live_dev_block(err_call, game, state, plate_now)
    if blk is None:
        return False, "cannot compose on the live kitchen"
    named = sorted(n for n in smoking if n and any(f" {n}" in f" {c} " for c in blk[0]))
    if named:
        return False, f"its block goes to the smoking {named[0]} ({[c for c in blk[0] if named[0] in c][0]!r})"
    if len(blk[0]) > SMOKE_SHORT_BLOCK:
        return False, f"its block is {len(blk[0])} commands (> {SMOKE_SHORT_BLOCK})"
    return True, f"narrowed smoke ban: {len(blk[0])}-command block, smoking {sorted(smoking)}"


# F1 (round-3 review, 2026-09-19): leave_unattended is OUT. Its restore replayed 30-55 ticks of other routine
# (steak prep, potato prep) while the walked-away food smoked, and the cook's next own turn came only when that
# queue drained: 6/6 baseline master burns at t126-127, hard medium_4 B / hard_2 B burned, three near-burns 4-5
# ticks from fire. After a walk-away the cook re-plans from the live kitchen (H1, as stop_hands' docstring says),
# and since R10 it gets that turn at the release itself.
RESTORE_AFTER_COMMIT = {"wrong_order", "repeat_step", "add_foreign", "misidentify_assemble", "serve_raw"}


def _hand_step(cmd, hold, dest=None):
    """F1: one hand action on a coarse hand state ('empty' / 'item' / 'plate'). Returns (possible, new state).
    pick up needs empty hands, put down needs something held AND somewhere that can take it, chop and turn on
    need free hands, scoop and serve need the plate.

    G2 (round-5 review): `turn on` was missing, and it is the step the restore keeps most often — 19 of the run's
    refusals were "you TRIED \"turn on\" but it FAILED: your hands must be EMPTY to turn a station on", every one
    of them a restored ignition run with the dish plate still in the cook's hands. `dest` is the destination the
    walk at the head of this segment leads to (see _dest_of): a `put down` into a board or counter that is already
    occupied is refused by the engine ("can't put that down here"), so it cannot be restored either."""
    c = str(cmd).strip()
    if c == "pick up":
        return hold == "empty", ("item" if hold == "empty" else hold)
    if c == "put down":
        if hold == "empty":
            return False, "empty"
        if dest is not None and dest.get("occupied") and not dest.get("accepts_more"):
            return False, hold            # nowhere to put it down: the engine refuses this command
        return True, "empty"
    if c in ("chop", "turn on"):
        return hold == "empty", hold
    if c in ("scoop", "serve"):
        return hold == "plate", hold
    return True, hold


def _dest_of(cmd, game):
    """What the walk at the head of a restored segment leads to: is it a station/board that already holds
    something, and can it take another item? (A cooking station with room still accepts a load; a cutting board
    or a counter tile that is occupied does not.)"""
    try:
        tile, _lbl = _goto_tile(cmd, game)
    except Exception:                                   # noqa: BLE001
        return None
    if tile is None:
        return None
    st = (getattr(game, "stations", None) or {}).get(tile)
    if st is not None:
        it = getattr(st, "item", None)
        conts = list(getattr(st, "contents", None) or [])
        cap = int(getattr(st, "capacity", 0) or 0)
        occupied = bool(it is not None or conts)
        return {"occupied": occupied, "accepts_more": bool(cap and len(conts) < cap)}
    ci = (getattr(game, "counter_items", None) or {}).get(tile)
    return {"occupied": ci is not None, "accepts_more": False}


def _plate_pickup_cmds(game):
    """Just the two commands that take THE plate (the parked one if there is one) — no parking of what the hands
    hold. plate_into_hand() does both in one list, which double-parks a block that has already emptied the hands."""
    from cooksim.core.items import Plate
    for (x, y), it in game.counter_items.items():
        if isinstance(it, Plate):
            return [f"go to the counter ({x},{y})", "pick up"]
    return ["go to the plates", "pick up"]


def _walk_takes_plate(cmd, game):
    """G5 (round-5 review): does this walk lead to A PLATE? The old test was the word "plate" in the text, but
    _plate_pickup_cmds() emits the COORDINATE form of a parked plate ("go to the counter (5,0)"), so a block that
    had already fetched the plate was given a SECOND fetch and an illegal `pick up` on top of it (master 8/20
    episodes, ~10-tick stall). Both forms count now: the words, or a counter tile that holds a Plate."""
    from cooksim.core.items import Plate
    c = str(cmd or "")
    if "plate" in c:
        return True
    m = re.search(r"\((-?\d+)\s*,\s*(-?\d+)\)", c)
    if m:
        tile = (int(m.group(1)), int(m.group(2)))
        return isinstance((getattr(game, "counter_items", None) or {}).get(tile), Plate)
    tile, _lbl = _goto_tile(c, game)
    if tile is not None:
        return isinstance((getattr(game, "counter_items", None) or {}).get(tile), Plate)
    return False


def _ensure_plate_for_block(block, game, parked=False):
    """fire-longtail (2026-09-20): put THE plate in the cook's hands at the point in a forced block where it is
    first needed — right before the walk that leads to the block's first `scoop`, or before a bare `serve`.

    The hands are simulated through the block (_hand_step) from what the cook actually holds now, so a block that
    first parks something or chops a second unit gets its fetch AFTER that, not before it, and a block that already
    holds the plate at that point gets nothing. The old rule prefixed plate_into_hand() only when a scoop sat in the
    block's first three commands: every FORCED plating block (park, fetch, walk, scoop) and every block with a prep
    prefix fell outside it and scooped into empty hands."""
    from cooksim.core.items import Plate
    if not block:
        return block
    j = next((k for k, (c, _s) in enumerate(block)
              if str(c).strip() in ("scoop", "serve")), None)
    if j is None:
        return block
    hold = "plate" if isinstance(game.cooks[0].holding, Plate) else (
        "empty" if game.cooks[0].holding is None else "item")
    prev = ""
    for c, _s in block[:j]:
        c = str(c).strip()
        if c == "pick up":
            # coarser than _hand_step on purpose: a pick up after a walk that LEADS TO a plate takes the plate
            # ("go to the plates", "go to the counter with the plate", "go to the counter (5,0)" where the parked
            # plate sits), and a block that parks the dish plate and fetches it back must not then be given a
            # second, empty one. G5: the coordinate form counts as well as the words (_walk_takes_plate).
            hold = "plate" if _walk_takes_plate(prev, game) else ("item" if hold == "empty" else hold)
        elif c == "put down":
            hold = "empty"
        prev = c
    if hold == "plate":
        return block
    pre = list(_plate_pickup_cmds(game))
    if hold == "item" and not parked:
        pre = ["go to the empty counter", "put down"] + pre
    k = j - 1 if j > 0 and str(block[j - 1][0]).startswith("go to") else j
    sid_ = block[min(k, len(block) - 1)][1]
    return list(block[:k]) + [(c, sid_) for c in pre] + list(block[k:])


def restorable_after_commit(saved, items, game, pre_hold=None):
    """E1/E6: the part of a displaced queue that is independent of a committed mistake. The queue is cut into
    walk-segments (a go to / wait / idle and the bare hand actions after it); a segment is dropped when it names one
    of the mistake's OWN units (error_units, never its anchor), walks to a station or board holding one, serves, or
    trashes. Returns (kept pairs, dropped segments as command lists).

    F1 (round-3 review): a kept segment that DEPENDS on a dropped one goes too. The restore used to drop the fetch of
    the next raw potato ("go to potato source 1, pick up" names the mistake's item) and keep "go to board 2, put down,
    chop, pick up, go to fryer 2, put down, turn on", which then ran empty-handed as refused commands for 30 ticks
    (hard medium_3 / medium_4 / hard_5 B). The rule: simulate the hands through the WHOLE displaced queue and through
    the kept part only; a kept segment whose hand actions work in the first run but not in the second needed what a
    dropped segment would have done, so it is dropped as well. A segment that walks to a board and chops there without
    putting anything down first is dropped when that board no longer holds a raw item (a wrong_order block plated the
    tomato the cook had just put down, and the restored chop was refused with the plate in hand, master hard_4 C).

    G2 (round-5 review) — three seams in that rule, 27 of 38 restores refused at least one command:
      * both hand simulations were seeded from the SAME post-block hand state, so the dependency guard could never
        fire: whatever the kept run could not do, the full run could not do either. `pre_hold` is now the hand state
        as it was BEFORE the takeover — what the cook's own plan assumed — and the kept run starts from the live
        hands, which after a forced plating block hold the dish plate.
      * `_hand_step` knew nothing about `turn on` (needs empty hands) or about where a `put down` lands, so a
        restored ignition and a restored put-down onto an occupied board were kept and refused.
      * the dish plate was put down by restored steps meant for an ingredient. A `put down` is never restored while
        the hands hold the plate: parking the dish mid-plan is not something the cook's own queue ever asked for."""
    from cooksim.core.items import Plate
    segs, cur = [], []
    for c, k in saved:
        head = str(c).startswith(("go to", "wait", "idle"))
        if head and cur:
            segs.append(cur); cur = []
        cur.append((c, k))
    if cur:
        segs.append(cur)
    h0 = game.cooks[0].holding
    hold_kept = ("empty" if h0 is None else "plate" if isinstance(h0, Plate) else "item")
    hold_full = pre_hold if pre_hold in ("empty", "item", "plate") else hold_kept
    keep, dropped = [], []
    for seg in segs:
        txt = [str(c) for c, _k in seg]
        dest = _dest_of(txt[0], game) if txt and txt[0].startswith("go to") else None
        ignition_only = ("turn on" in txt) and not any(c.strip() in ("scoop", "serve", "trash") for c in txt)
        bad = any(c.strip() in ("serve", "trash") or c.startswith("trash") or "serving pass" in c or "the trash" in c
                  for c in txt)
        bad = bad or any(f" {x} " in f" {c} " or f"{x} source" in c for c in txt for x in (items or ()))
        if not bad and not ignition_only:
            # G3: an ignite of a station the mistake never touched is NEVER dropped for what that station holds —
            # it is the first component's own ignition, and dropping it left the food raw in a cold station.
            for c in txt:
                if c.startswith("go to"):
                    tile, _lbl = _goto_tile(c, game)
                    if tile is not None and any(_tile_holds(game, tile, x) for x in (items or ())):
                        bad = True
        # the whole-queue run (what the cook had planned, from the hands it had THEN) and the kept-only run
        # (what will actually happen, from the hands it has NOW)
        ok_full, h_f = True, hold_full
        for c in txt:
            ok_, h_f = _hand_step(c, h_f, dest)
            ok_full = ok_full and ok_
        if not bad:
            ok_kept, h_k = True, hold_kept
            for c in txt:
                ok_, h_k = _hand_step(c, h_k, dest)
                ok_kept = ok_kept and ok_
            if ok_full and not ok_kept:
                bad = True                                   # it needed what a dropped segment would have fetched
            elif not ok_kept:
                bad = True                                   # it cannot run from the hands the cook actually has
            if not bad and hold_kept == "plate" and "put down" in [c.strip() for c in txt]:
                bad = True                                   # never restore a put-down of the dish plate
        if not bad and "chop" in txt:
            j = txt.index("chop")
            if "put down" not in txt[:j] and txt[0].startswith("go to"):
                tile, _lbl = _goto_tile(txt[0], game)
                st = game.stations.get(tile) if tile is not None else None
                it = getattr(st, "item", None) if st is not None else None
                raw_here = it is not None and str(getattr(getattr(it, "state", None), "value",
                                                          getattr(it, "state", None))) not in ("chopped",)
                if st is not None and hasattr(st, "chop_time") and not raw_here:
                    bad = True                               # nothing left on that board to chop
        hold_full = h_f
        if bad:
            dropped.append(txt)
        else:
            keep += seg
            hold_kept = h_k
    return keep, dropped


LU_ANCHOR = (os.environ.get("COOKSIM_LU_ANCHOR") or "on") != "off"      # E9 (round-2 review)
FORGET_REACT_TICKS = 5         # E4: after the freeze the cook still has to walk to the food


def plating_imminent(err, kw, game, recipe):
    """E3: a pending PLATING mistake whose slot is next on the plate in the cook's hands — its takeover can start
    within a few ticks (verifier r2a: 25 questions asked exactly 3 ticks before a takeover, e.g. "what is my next
    step?" answered "put the tomato on" and then overturned by an omission). No question opens while this holds."""
    if err == "wrong_order" and not plate_items(game):
        # the swap onto a still-empty plate fetches the plate itself: imminent once b is ready and a goes first
        from cooksim.interact.system import ingredient_ready
        names = [n for n, _ in recipe.contents]
        want = {n: getattr(st, "value", st) for n, st in recipe.contents}
        if names and names[0] == kw.get("a") and ingredient_ready(game, kw.get("b"), want.get(kw.get("b"))):
            return True
    if not plate_in_hand(game):
        return False
    items = plate_items(game)
    nxt = next_to_plate(recipe, items)
    if err == "wrong_order":
        return nxt == kw.get("a")
    if err in ("serve_raw", "misidentify_assemble"):
        return nxt == kw.get("component")
    if err == "add_foreign":
        tgt = kw.get("before") or ""
        if tgt == "serve":
            return nxt is None or sum(1 for _ in recipe.contents) - len(
                [1 for n, _s in items if n in dict(recipe.contents)]) <= 1
        return nxt == (kw.get("component") or _sid_comp(tgt))
    if err in ("omit_subgoal", "premature_serve"):
        x = _sid_comp(kw.get("sid") or kw.get("after"))
        if kw.get("sid"):
            return nxt == x
        names = [n for n, _ in recipe.contents]
        return x in names and names.index(x) + 1 < len(names) and nxt in (x, names[names.index(x) + 1])
    if err == "repeat_step":
        return nxt == _sid_comp(kw.get("at")) or nxt == _sid_comp(kw.get("sid"))
    if err == "forget_next_step" and str(kw.get("after") or "").startswith("assemble("):
        return nxt == _sid_comp(kw.get("after"))
    return False


def names_complete(recipe, items):
    """Every goal item is on the plate BY NAME (states ignored: a raw unit left by a committed mistake still fills
    the slot as far as the cook's remaining work goes)."""
    import collections as _c
    need = _c.Counter(n for n, _ in recipe.contents)
    got = _c.Counter(n for n, _ in (items or []))
    return all(got[k] >= v for k, v in need.items())


def missing_units(rec_items, plate):
    """F2 (round-3 review): the goal units a plate lacks, counted as UNITS (a Counter), never as name sets."""
    import collections as _col
    return sorted((_col.Counter(rec_items) - _col.Counter(plate)).elements())


def assistant_spoke_since(comm, t0):
    """F3 (round-3 review): has the assistant said anything (a non-empty line) at or after tick t0?"""
    return bool(t0 is not None and any(
        str(m_.get("speaker", "")).startswith("vlm") and (m_.get("text") or "").strip() and m_.get("t", -1) >= t0
        for m_ in (comm or [])))


def serve_ceiling_note(goal, plate_items_, queue):
    """F7 (round-3 review): why no question, request or plan change may open now (None when one may): the plate
    already holds every goal item by name, or the cook's own queue reaches the serve with no scoop left before it."""
    if names_complete(goal, plate_items_):
        return "ceiling: the plate holds every goal item (the serve is next)"
    for c in queue or []:
        if str(c).strip() == "scoop":
            return None
        if str(c).strip() == "serve" or "serving pass" in str(c):
            return "ceiling: the cook's serve is queued"
    return None


def forget_freeze_ticks(kw=None):
    """Ticks a forget_next_step freeze lasts, for the guards that must look that far ahead.
    The single source is cooksim.core.config.FORGET_FREEZE_TICKS (env COOKSIM_FORGET_FREEZE);
    live_dev_block and plan_ir.forget_rest both build their block from it."""
    from cooksim.core.config import forget_freeze_ticks as _freeze
    return _freeze()


def ticks_to_smoke(game, exclude=None, skip_smoking=False):
    """Fewest ticks until some food on a LIT station starts to smoke (0 = already smoking), None when nothing lit
    holds food. `exclude`: skip stations holding that item (F4: the walk-away's own
    food is meant to smoke; every OTHER lit food must not).

    G8 (round-5 review): `skip_smoking` leaves ALREADY-smoking food out of the answer. These guards exist to stop a
    takeover from CAUSING a burn somewhere else, and food that is smoking now is not burned by a block that has not
    started: refusing the injection does not un-smoke it. Read as a margin of 0 it barred `leave_unattended` for 14
    consecutive ticks (hard medium_map_2 C, t156-t168, "other food on the heat would smoke within 0 ticks"), the
    walk-away's own moment expired underneath it, and the kitchen then burned at t215 with no mistake injected at
    all. Whether the cook's hands are needed for a rescue is the blanket smoke ban's question, not this one's."""
    from cooksim.core.stations import CookStation
    best = None
    for st_ in game.stations.values():
        if not (isinstance(st_, CookStation) and st_.contents and st_.cooking) or st_.status == "burnt":
            continue
        if exclude is not None and any(getattr(x, "name", None) == exclude for x in st_.contents):
            continue
        if getattr(st_, "smoking", False):
            if skip_smoking:
                continue
            return 0
        t = max(0, st_._smoke_at - st_.timer + 1)
        best = t if best is None else min(best, t)
    return best


def _sid_comp(sid):
    sid = str(sid or "")
    return sid.split("(")[1].split(")")[0] if "(" in sid else None


# ---- FORCED PLATING MISTAKES (fire-longtail, 2026-09-20) ------------------------------------------------
# User ruling: "一定要有这一类。我们就是要强行让 human 把比方说生牛肉加上去" — the engine FORCES the mistake instead of
# waiting for the cook to walk into it. Three audit rounds had bolted onto every plating error the conjunction
#   (1) the plate is already IN THE COOK'S HANDS, (2) the component is the NEXT item to plate, and
#   (3) the cook's own next walk already goes to where that component is scooped from (heading_to_plate),
# and each of those needs the COOK'S OWN PLAN to cooperate. A self-planning LLM cook cooks everything before it
# fetches the plate, so by the time (1) holds the item is cooked and serve_raw's own "it must still be raw" can
# never hold with it: serve_raw fired 1/10 and was then dropped from the suite as structurally impossible, and
# wrong_order missed 6/20 on "the steak is not the next item to plate". A plating-type takeover now takes the
# hands, parks what they hold, fetches the plate, walks to the source/station and scoops the wrong, raw or
# duplicate item. Only the NONSENSICAL moments stay barred:
#   * the dish is not under way at all (nothing prepped, nothing plated, the episode just started);
#   * a unit of the item is finishing on this very tick — or, for serve_raw, within the block's own walk to the
#     crate (a raw one must not be scooped past a good one that is arriving);
#   * the plate is already complete, or the cook's own serve is queued;
#   * the item's own station is smoking (the existing narrow-ban rules are untouched);
#   * for serve_raw the item must be genuinely raw/unprocessed, and its recipe state must not be "raw" anyway.
# COOKSIM_FORCE_PLATING=0 restores the pre-2026-09-20 "wait until the cook walks into it" rules.
FORCE_PLATING = (os.environ.get("COOKSIM_FORCE_PLATING") or "1") != "0"
# The only thing the forced block waits for at the very start of an episode: a cook that has not touched anything
# yet cannot plausibly be mid-slip. Anything at all counts as started (an item plated, a unit in processing, a
# plate picked up), and the tick escape makes this a believability guard and never a starvation rule — a one-item
# recipe (Fried Fish) has no other work to wait for.
FORCE_UNDER_WAY_TICKS = int(os.environ.get("COOKSIM_FORCE_UNDER_WAY_TICKS", 15))


def dish_under_way(game, ro, prep_max):
    """Has the cook actually begun the dish? (see FORCE_UNDER_WAY_TICKS)"""
    from cooksim.core.items import Plate
    from cooksim.core.stations import CookStation
    if int(getattr(ro, "tick", 0) or 0) >= FORCE_UNDER_WAY_TICKS:
        return True
    if plate_items(game) or any((prep_max or {}).values()):
        return True
    if isinstance(game.cooks[0].holding, Plate) or any(isinstance(it, Plate) for it in game.counter_items.values()):
        return True
    for st_ in game.stations.values():
        if isinstance(st_, CookStation) and (getattr(st_, "contents", None) or []):
            return True
        if "board" in type(st_).__name__.lower() and getattr(st_, "item", None) is not None:
            return True
    return False


def finishing_within(game, x, margin):
    """Ticks until the soonest unit of x on the heat is done, when that is within `margin` (else None).
    "Never on the tick an item finishes cooking": a mistake about x must not land while a good x is arriving.
    margin=0 is the exact tick only."""
    from cooksim.core.stations import CookStation
    left = [st_.cook_time - st_.timer for st_ in game.stations.values()
            if isinstance(st_, CookStation) and st_.status == "cooking"
            and any(getattr(c_, "name", None) == x for c_ in (st_.contents or []))]
    left = [d for d in left if d >= 0]
    return min(left) if left and min(left) <= margin else None


def live_trigger(err, kw, game, state, plate_now, ro, prep_max):
    """REFERENCE-FREE injection moment (user ruling 2026-09-15): "there should not be any episode
    whose error injection moment depends on the scripted reference." Until this date every error's
    moment was the scripted template's divergence index, replayed to produce a reference plate
    prefix and prep progress the live kitchen then had to match exactly — so a self-planning cook
    that plated the fish early, or prepped in another order, starved the error.

    The moment now comes from three things only: the error's own parameters, the recipe (the order
    ticket), and the live kitchen. Returns None when the mistake's own moment has arrived, else a
    wait note. error_precondition() still decides whether the mistake is possible at all."""
    import collections as _c
    from cooksim.core.items import Plate
    from cooksim.core.stations import CookStation
    from cooksim.interact.system import ingredient_ready
    recipe = goal_of(game, state)          # E6: the current goal (recipe + applied plan changes)
    items = [n for n, _ in recipe.contents]
    want = {n: getattr(st, "value", st) for n, st in recipe.contents}
    holding = game.cooks[0].holding
    # A plate is being built whether the cook HOLDS it or has PARKED it on a counter — the rulebook lets
    # an item go onto a plate sitting on a counter, and the verifier showed dishes built entirely on a
    # parked plate while every "plate in hands" trigger waited (2026-09-15).
    has_plate = isinstance(holding, Plate) or any(isinstance(it, Plate) for it in game.counter_items.values())
    have = _c.Counter(plate_now)
    pitems = plate_items(game)
    held_plate = isinstance(holding, Plate)

    # B3 (fix-firing 2026-09-19): "x is the next item to plate" = x is the FIRST recipe item not yet CORRECTLY plated
    # (name and wanted state), whatever else sits on the plate — a wrong, foreign, raw or duplicate item left by an
    # earlier committed mistake. The old test (the plate is EXACTLY the recipe prefix before x) made every later plating
    # mistake unreachable after the first committed one unless the cook trashed the plate.
    def is_next(x):
        return x in items and next_to_plate(recipe, pitems) == x

    def through_on_plate(x):         # B5: every recipe item up to and including x is on the plate IN ITS WANTED STATE
        return plated_through(recipe, pitems, x)

    def complete():                  # B5: by names and states, never by item count
        return plate_complete(recipe, pitems)

    def reaching_for(c):             # the cook's own current/next command names c, or it holds a raw c
        nxt = [str(cmd) for cmd, _sid in (ro.cmds[ro.i:ro.i + 2] if getattr(ro, "cmds", None) else [])]
        held = getattr(holding, "name", None)
        return any(f" {c} " in f" {x} " or f"{c} source" in x for x in nxt) or held == c

    def next_cmds():
        return [str(cmd) for cmd, _sid in (ro.cmds[ro.i:ro.i + 2] if getattr(ro, "cmds", None) else [])]

    def heading(x):                  # B4: plate IN HAND and the cook's own next walk goes to where x is scooped from
        return heading_to_plate(game, ro, x, want.get(x))

    def about_to_serve():            # a plate is being carried or is parked, and the cook's OWN next command is the serve
        return has_plate and any(("serving pass" in c) or c.strip() == "serve" for c in next_cmds())

    def force_gate():
        """The shared bar for a FORCED plating block: the dish is under way and the plate is neither finished nor
        on its way to the pass. None when it is clear."""
        if not dish_under_way(game, ro, prep_max):
            return "the dish is not under way yet (nothing prepped, nothing plated)"
        if complete():
            return "the plate is already complete"
        if about_to_serve():
            return "the cook's own serve is already queued"
        return None

    def first_item_ready_now(x):     # x goes first on the plate, the plate is still empty, and x is done
        # (kept for wrong_order / omission, whose blocks fetch the plate themselves; the B4 plating mistakes —
        # serve_raw, misidentify, add_foreign — no longer use it: they fired on the tick a cooked first item finished
        # and scooped a raw one from the crate beside it)
        if not (x in items and items[0] == x and not plate_now):
            return False
        if want.get(x) in ("cooked", "chopped"):
            return ingredient_ready(game, x, want.get(x))
        # a RAW first item (tortilla, bun) is ready from tick 0, so its moment is the cook's own next command heading
        # for a plate while none is in hand
        return (not has_plate) and any(c.startswith("go to") and "plate" in c for c in next_cmds())

    def cooking(c):
        return any(isinstance(st_, CookStation) and st_.status == "cooking"
                   and any(getattr(x, "name", None) == c for x in (st_.contents or []))
                   for st_ in game.stations.values())

    def lit_with(c):
        return any(isinstance(st_, CookStation) and st_.status in ("cooking", "cooked")
                   and any(getattr(x, "name", None) == c for x in (st_.contents or []))
                   for st_ in game.stations.values())

    if err == "mis_prep":
        c = kw.get("component")
        return None if reaching_for(c) else f"the cook is not reaching for the {c} yet"
    if err == "serve_raw" and FORCE_PLATING:
        # FORCED (2026-09-20). The takeover itself parks the hands, fetches the plate, walks to the crate and scoops
        # the component raw; it no longer waits for a plate in the cook's hands or for the component to be the next
        # item to plate. What is left is the mistake's own meaning and the two nonsense guards.
        x = kw.get("component")
        if ingredient_ready(game, x, want.get(x)):
            return f"the {x} is already {want.get(x)} (serving it raw now is not the mistake)"
        d_ = finishing_within(game, x, serve_raw_walk_estimate(game, x))
        if d_ is not None:
            return f"the {x} is done in {d_} ticks (less than the walk to its crate)"
        return force_gate()
    if err == "misidentify_assemble" and FORCE_PLATING:
        # FORCED (2026-09-20): the block fetches the plate and scoops the WRONG item into the component's slot.
        # The component's slot being open is error_precondition's job; the cook's own heading is no longer asked for.
        x = kw.get("component")
        if finishing_within(game, x, 0) is not None:
            return f"the {x} finishes on this very tick"
        return force_gate()
    if err == "add_foreign" and FORCE_PLATING:
        # FORCED (2026-09-20). `before=serve` keeps its meaning — the extra goes on a dish the cook already thinks is
        # done — but reads the plate, not the cook's hands. Every other target only needs its slot still open.
        tgt = kw.get("before") or ""
        x = kw.get("component") or _sid_comp(tgt)
        if tgt == "serve":
            if not dish_under_way(game, ro, prep_max):
                return "the dish is not under way yet (nothing prepped, nothing plated)"
            return None if (complete() or about_to_serve()) else "the plate is not finished yet"
        if x and finishing_within(game, x, 0) is not None:
            return f"the {x} finishes on this very tick"
        return force_gate()
    if err == "wrong_order" and FORCE_PLATING:
        # FORCED (2026-09-20): the mistake is "the cook thinks b comes first". The block puts the plate in the
        # hands, makes b if it is a chop it can do itself (D12), and scoops b before a. Waiting for a to be the
        # next item to plate was the cook's plan cooperating, and it missed 6/20 live swaps.
        a = kw.get("a")
        if finishing_within(game, a, 0) is not None:
            return f"the {a} finishes on this very tick"
        return force_gate()
    if err == "repeat_step" and FORCE_PLATING:
        # FORCED (2026-09-20) + per-item ban (see REPEAT_DUPLICATE_INVISIBLE in error_lib): a duplicate is a mistake
        # wherever it is VISIBLE as a duplicate. `at` stays honoured, as a PROGRESS FLOOR (its item is plated, or in
        # processing) rather than as "the plate is exactly through it" — the old rule needed the cook's plating order.
        sid = kw.get("sid") or ""
        c = _sid_comp(sid)
        at = kw.get("at") or ""
        # `at` is honoured as a progress floor whichever step is repeated: its own item plated (an assemble) or in
        # processing (a prep). Without this a prep repeat fired the tick its component first entered a station — the
        # cook loading a second patty two ticks after the first, which is a consecutive re-do, not a memory slip.
        if at.startswith("assemble("):
            y = _sid_comp(at)
            if y != c and correct_counts(recipe, pitems)[y] < 1:
                return f"the {y} is not on the plate yet (repeat at {at})"
            if y == c and correct_counts(recipe, pitems)[c] < 1:
                return f"no {c} on the plate yet (repeat at {at})"
        elif at.startswith("prep("):
            y = _sid_comp(at)
            if prep_max.get(y, 0) < 1:
                return f"{y} not in processing yet (repeat at {at})"
        if sid.startswith("assemble"):
            # a COMPLETE plate does not bar the duplicate — it is where the duplicate is most visible as one, and
            # it is the v4 form of this mistake. Only a FULL plate does (error_precondition's capacity rule), and
            # only the serve already being on its way (a scoop cannot land behind it).
            if not dish_under_way(game, ro, prep_max):
                return "the dish is not under way yet (nothing prepped, nothing plated)"
            if correct_counts(recipe, pitems)[c] < 1:
                return f"no {c} on the plate yet (nothing to repeat)"
            return None
        return None if prep_max.get(c, 0) >= 1 else f"no {c} in processing yet"
    if err in ("omit_subgoal", "premature_serve") and FORCE_PLATING:
        # FORCED (2026-09-20). The omission's own block fetches the plate and plates the NEXT recipe item, so the
        # skipped component being "the next item to plate" is no longer asked for. The early serve keeps a floor, but
        # a POSITION floor (as many units on the plate as the recipe wants up to and including `after`'s item,
        # whatever they are) instead of "the plate is exactly the recipe prefix" — an early serve is premature by how
        # far along the dish is, not by whether the cook plated in the template's order.
        g_ = force_gate()
        if g_:
            return g_
        if kw.get("sid"):
            return None
        after = kw.get("after") or ""
        if after == "fetch_plate":
            return None if sum(have.values()) >= 1 else "nothing on the plate yet"
        x = _sid_comp(after)
        names_ = [n for n, _ in recipe_wants(recipe)]
        if x in names_:
            pos = len(names_) - names_[::-1].index(x)
            if len(pitems) < pos:
                return f"only {len(pitems)} of {pos} items are on the plate (an early serve after the {x})"
        return None if pitems else "nothing on the plate yet"
    if err == "serve_raw":
        # B4: the plate is IN HAND, the component is the next item to plate, and it is NOT yet cooked/processed —
        # never on the tick it finishes (a cooked one would then wait in the pan while a raw one is scooped from the
        # crate) and never on a prep walk to its source. When a unit is still cooking, its remaining time must cover
        # the block's walk to the crate, so the raw scoop lands while the pan is visibly still cooking.
        x = kw.get("component")
        if not held_plate:
            return "no plate in the cook's hands yet"
        if not is_next(x):
            return f"the {x} is not the next item to plate"
        if ingredient_ready(game, x, want.get(x)):
            return f"the {x} is already {want.get(x)} (serving it raw now is not the mistake)"
        if prep_walk_for(game, ro, x) or "chop" in next_cmds():
            return f"the cook is on a prep walk for the {x}"
        left = [st_.cook_time - st_.timer for st_ in game.stations.values()
                if isinstance(st_, CookStation) and st_.status == "cooking"
                and any(getattr(c_, "name", None) == x for c_ in (st_.contents or []))]
        if left and max(left) <= serve_raw_walk_estimate(game, x):
            return f"the {x} is about to be done (less than the walk to its crate)"
        return None
    if err == "misidentify_assemble":
        x = kw.get("component")
        if not held_plate:
            return "no plate in the cook's hands yet"
        if not is_next(x):
            return f"the {x} is not the next item to plate"
        return None if heading(x) else f"the cook is not heading to plate the {x}"
    if err == "add_foreign":
        tgt = kw.get("before") or ""
        x = kw.get("component") or _sid_comp(tgt)
        if not held_plate:
            return "no plate in the cook's hands yet"
        if tgt == "serve":
            return None if (complete() or about_to_serve()) else "the plate is not finished yet"
        if not is_next(x):
            return f"the {x} is not the next item to plate"
        return None if heading(x) else f"the cook is not heading to plate the {x}"
    if err == "wrong_order":
        a = kw.get("a")
        if items and items[0] == a and not plate_now and (want.get(a) in ("cooked", "chopped") or first_item_ready_now(a)):
            return None                   # b onto the still-empty plate: the block fetches the plate itself
        if not has_plate:
            return "no plate in the cook's hands yet"
        return None if (is_next(a) or heading(a)) else f"the {a} is not the next item to plate"
    if err in ("omit_subgoal", "premature_serve"):
        if kw.get("sid") and first_item_ready_now(_sid_comp(kw["sid"])):
            return None
        if not has_plate:
            return "no plate in the cook's hands yet"
        if kw.get("sid"):
            x = _sid_comp(kw["sid"])
            return None if (is_next(x) or heading(x)) else f"the {x} is not the next item to plate"
        after = kw.get("after") or ""
        if after == "fetch_plate":      # the serve comes straight after taking the plate: one item on it is enough
            return None if (sum(have.values()) >= 1 and not complete()) else "nothing on the plate in hand yet"
        x = _sid_comp(after)
        return None if (x and through_on_plate(x) and not complete()) else f"the plate is not yet through {x}"
    if err == "repeat_step":
        # B9: honour `at`. repeat_step(sid, at) = "after finishing `at`, the cook re-does the earlier `sid`": the moment
        # is the plate through `at` (in the wanted states), the repeated item already on it, and the dish not complete.
        # The `at` kwarg used to be ignored, so every repeat fired straight after the first scoop.
        sid = kw.get("sid") or ""
        c = _sid_comp(sid)
        at = kw.get("at") or ""
        if sid.startswith("assemble"):
            if not (has_plate and correct_counts(recipe, pitems)[c] >= 1):
                return f"the {c} is not on the plate in hand"
            if at.startswith("assemble("):
                y = _sid_comp(at)
                return None if through_on_plate(y) else f"the plate is not yet through {y} (repeat at {at})"
            if at.startswith("prep("):
                y = _sid_comp(at)
                return None if prep_max.get(y, 0) >= 1 else f"{y} not in processing yet (repeat at {at})"
            return None
        if at.startswith("prep("):
            y = _sid_comp(at)
            if prep_max.get(y, 0) < 1:
                return f"{y} not in processing yet (repeat at {at})"
        return None if prep_max.get(c, 0) >= 1 else f"no {c} in processing yet"
    if err == "forget_next_step":
        after = kw.get("after") or ""
        c = _sid_comp(after)
        # E4 (round-2 review): the forgotten step is never the serve (standing rule). "Plate complete" by names AND
        # states missed a finished plate that carried an earlier committed mistake (expert hard_5 c: steak, RAW potato,
        # potato, egg — the forget froze the cook 64 ticks holding it, then it served). The serve is the only step
        # left when every goal item is on the plate by NAME, when the plate is full, or when the cook's own next
        # command is the serve.
        if names_complete(recipe, pitems) or about_to_serve() or len(pitems) >= int(
                getattr(getattr(game, "config", None), "plate_capacity", 5) or 5):
            return "the only step left is the serve (a forget never forgets the serve)"
        # E4: no food may reach smoke during the freeze. The old guard looked at THIS tick only: it let the freeze
        # start 2 ticks before a potato finished (hard medium_2 C) and exempted food under a smoke hold, whose 25-tick
        # cap then ran out mid-freeze (hard medium_5 C, hard_2 C, expert hard_1 c: 44 ticks of smoke). Freezing
        # beside food that will smoke is leave_unattended, not a forget.
        tts = ticks_to_smoke(game, skip_smoking=True)      # G8: as above — a freeze does not start the smoke
        need_ = forget_freeze_ticks(kw) + FORGET_REACT_TICKS
        if tts is not None and tts <= need_:
            return (f"food on the heat would smoke within {tts} ticks, inside the {need_}-tick freeze "
                    f"(a freeze now would be leave_unattended)")
        if after.startswith("prep("):
            k = int(after.split("#")[1]) if "#" in after else 1
            return None if prep_max.get(c, 0) >= k else f"{c} #{k} not in processing yet"
        if after.startswith("ignite("):
            return None if lit_with(c) else f"no lit station with the {c} yet"
        if after == "fetch_plate":
            return None if has_plate else "no plate in the cook's hands yet"
        if after.startswith("assemble("):
            return None if (has_plate and through_on_plate(c)) else f"the plate is not yet through {c}"
        return None
    if err == "leave_unattended":
        # B6 (ruling R1): the walk-away is only a mistake once the food is NEAR DONE. At ignition, walking off to prep
        # something else is the ordinary plan, and the old rule fired exactly then (the flag "you left the pan" was
        # then bad advice). See lu_near_ticks() for N.
        ok_, why_ = lu_moment(game, kw.get("component"), lu_near_ticks(game), late=kw.get("_lu_late"))
        if not ok_:
            return why_
        # F4 (round-3 review): the forget's smoke margin, for every OTHER lit food. The walk-away holds the cook's
        # body until its own food smokes (at most near + grace ticks from now) and the cook then needs a few ticks to
        # walk back; food elsewhere that smokes inside that span is burned by the takeover, not by the mistake (expert
        # b hard_2: LU(potato) fired 8 ticks before the done steak smoked; the steak burned at t221).
        c_ = kw.get("component")
        own = [max(0, st_._smoke_at - st_.timer + 1) for st_ in game.stations.values()
               if isinstance(st_, CookStation) and st_.contents and st_.cooking
               and any(getattr(x, "name", None) == c_ for x in st_.contents)]
        hold_ = (min(own) if own else lu_near_ticks(game) + int(getattr(getattr(game, "config", None), "smoke_grace", 25) or 25))
        tts_ = ticks_to_smoke(game, exclude=c_, skip_smoking=True)     # G8: already-smoking food is not this
        need_ = hold_ + FORGET_REACT_TICKS                             # block's doing (see ticks_to_smoke)
        if tts_ is not None and tts_ <= need_:
            return (f"other food on the heat would smoke within {tts_} ticks, inside the walk-away and the walk back "
                    f"({need_} ticks)")
        return None
    return None


def _plan_change_label(case):
    ge = case.get("goal_extend") or {}
    ing = ge.get("ingredient", "something")
    return f"add {'fresh ' if ge.get('state', 'raw') == 'raw' else ''}{ing} on top before serving"


def question_precondition(kind, item, game, state, plated_ever=None, live_plate=None, prep_max=None):
    """When a question or request makes sense to ask, judged from the live kitchen.

    Reference mode (COOKSIM_QUESTION_TRIGGERS=reference) keeps the old rule: only questions about something
    cooking need that something on the heat. Live mode adds what each kind needs to be meaningful — the
    answer itself is still recomputed from the live kitchen at ask time:
      qa_proactive / ready_in   the item is on the heat and NOT yet done (asking "tell me when it's done" about
                                food that is already done is not a request)
      longest_on / burn_first   something on the heat
      after_ref:X               X is on the plate with something after it
      order:E1|E2               both events have happened (prep(X): X entered processing; assemble(X): X was plated)
      missing:plate             a plate exists
      everything else           no extra condition"""
    from cooksim.core.stations import CookStation
    from cooksim.core.items import Plate
    k = (kind or "").split(":")[0]
    arg = kind.split(":", 1)[1] if kind and ":" in kind else ""
    heat = [st_ for st_ in game.stations.values() if isinstance(st_, CookStation) and st_.contents]

    def on_heat(name, statuses):
        return any(st_.status in statuses and (not name or any(getattr(c, "name", None) == name for c in st_.contents))
                   for st_ in heat)

    if not LIVE_QUESTIONS:
        if k in ("ready_in", "longest_on", "burn_first", "qa_proactive"):
            return None if on_heat(item, ("cooking", "cooked")) else "nothing cooking" + (f" ({item})" if item else "")
        return None
    if k in ("qa_proactive", "ready_in"):
        nm = item or arg or None
        return None if on_heat(nm, ("cooking",)) else f"no {nm or 'food'} still cooking on the heat"
    if k in ("longest_on", "burn_first"):
        return None if on_heat(None, ("cooking", "cooked")) else "nothing on the heat"
    if k == "elapsed":
        # F8 (round-3 review): "how long has X been on the heat" needs X still on the heat (master hard_5 B asked it
        # 46 ticks after the cook had plated the rice; the truth "rice is not on any heat" answered a false premise)
        nm = item or arg or None
        return None if on_heat(nm, ("cooking", "cooked")) else f"no {nm or 'food'} on the heat"
    if k == "past_done":
        # D15: asked before anything is lit the answer is a constant "no" (10/10 asks, most on the tick the raw meat
        # was set down); it waits until something is on a LIT station, so "done and sitting" can go either way
        return None if on_heat(None, ("cooking", "cooked")) else "nothing lit on the heat yet"
    if k == "after_ref":
        lp = list(live_plate or [])
        return None if (arg in lp and lp.index(arg) < len(lp) - 1) else f"nothing on the plate after the {arg} yet"
    if k == "order":
        for ev in arg.split("|"):
            nm = ev[ev.find("(") + 1:ev.rfind(")")].split("#")[0] if "(" in ev else ""
            if ev.startswith("prep(") and (prep_max or {}).get(nm, 0) < 1:
                return f"{ev} has not happened yet"
            if ev.startswith("assemble(") and nm not in (plated_ever or set()):
                return f"{ev} has not happened yet"
        return None
    if k == "missing":
        has_plate = isinstance(game.cooks[0].holding, Plate) or any(isinstance(it, Plate) for it in game.counter_items.values())
        return None if has_plate else "no plate yet"
    return None

def competent_plan(game, state, policy):
    """The verification cook's plan from the LIVE kitchen, built from the recipe template's own
    subgoal nodes (issued late-bound, like a plan): repair what a takeover left (a wrong plate is
    trashed, a foreign item is cleared off its board), then make what is still missing in one of
    three legal orders — template (load all, light all, plate), plate_early (each item: prep,
    light, wait, plate), reverse (prep in reverse recipe order lighting at load, then plate in
    recipe order)."""
    from cooksim.interact.system import plate_ok, plate_into_hand, _named
    from cooksim.core.plan_ir import trash_node, _station_label
    from cooksim.core.items import Plate
    recipe = goal_of(game, state)          # E6: the scripted cook carries out announced plan changes
    wanted_names = {n for n, _ in recipe.contents}
    goal = [n for n, _ in recipe.contents]
    cmds = []
    plate_now = list(plate_list(game))
    held_plate = isinstance(game.cooks[0].holding, Plate)
    parked_plate = any(isinstance(it, Plate) for it in game.counter_items.values())
    # 0. an ingredient the recipe never asks for still in hand (a takeover stopped mid-block, e.g. a
    #    prevented mis_prep after the pick-up) is thrown away first
    _h0 = game.cooks[0].holding
    if _h0 is not None and not isinstance(_h0, Plate) and getattr(_h0, "name", None) not in wanted_names:
        cmds += list(trash_node().commands)
    from cooksim.interact.system import required_state, _sname
    def _plate_states():
        for it in [game.cooks[0].holding] + list(game.counter_items.values()):
            if isinstance(it, Plate):
                return [(getattr(c, "name", None), _sname(getattr(c, "state", None))) for c in (it.contents or [])]
        return []
    wrong_state = any(_sname(required_state(recipe, n)) not in ("", "raw", None) and st != _sname(required_state(recipe, n))
                      for n, st in _plate_states())
    if plate_now and (goal[:len(plate_now)] != plate_now or wrong_state):
        pre = plate_into_hand(game)
        cmds += (pre if isinstance(pre, list) else []) + list(trash_node().commands)
        plate_now, held_plate = [], True
    # 2a. a foreign item cooking or cooked in a heat station is scooped onto the plate and trashed
    #     (the only way out of a heat station; left there it smokes and burns the kitchen)
    from cooksim.core.stations import CookStation
    for st_ in game.stations.values():
        if isinstance(st_, CookStation) and st_.contents and st_.status in ("cooking", "cooked") \
                and all(getattr(c, "name", None) not in wanted_names for c in st_.contents):
            lbl = None
            for (sx, sy), obj in game.stations.items():
                if obj is st_:
                    try:
                        lbl = str(game.layout.grid[sy][sx].value).lower()
                    except Exception:
                        lbl = None
            if lbl not in ("fryer", "pan", "pot", "oven"):
                continue
            nm = getattr(st_.contents[0], "name", "food")
            if not held_plate:
                cmds += plate_get() if (parked_plate or plate_now) else ["go to the plates", "pick up"]
                held_plate = True; parked_plate = False
            cmds += [f"wait until the {lbl} with the {nm} is cooked", f"go to the {lbl} with the {nm}", "scoop",
                     "go to the trash", "trash"]
            plate_now = []
    # 2. foreign items left on boards are moved to a free counter
    for st_ in game.stations.values():
        d = st_.render_state()
        it = d.get("item")
        if it is not None and _named(it) not in wanted_names and "board" in type(st_).__name__.lower():
            if held_plate:
                cmds += ["go to the empty counter", "put down"]; held_plate, parked_plate = False, True
            cmds += [f"go to the cutting board with the {_named(it)}", "pick up", "go to the empty counter", "put down"]
    # 2b. F10 (round-3 review): RESCUE food that is smoking. The verification cooks used to resume their policy
    #     after a walk-away released at the smoke onset — finish every prep, then plate — so all three policies burned
    #     in every master rice walk-away case and never reached error 2. A cook that sees smoke plates that item now:
    #     the recipe prefix up to it on an ordered dish (the rice goes on the tortilla), the item alone otherwise,
    #     when every item before it is ready; then the policy continues from the rescued plate.
    try:
        from cooksim.interact.system import ingredient_ready as _ir_
        _want = {n: getattr(s_, "value", s_) for n, s_ in recipe.contents}
        _smoking = {getattr(c, "name", None) for st_ in game.stations.values()
                    if isinstance(st_, CookStation) and getattr(st_, "smoking", False) and st_.status != "burnt"
                    for c in (st_.contents or [])}
        _ordered = bool(getattr(recipe, "ordered", False))
        _rescue = None
        for x in goal:
            if x not in _smoking or goal.count(x) <= plate_now.count(x):
                continue
            pre_ = goal[len(plate_now):goal.index(x) + 1] if _ordered else [x]
            if _ordered and (goal[:len(plate_now)] != plate_now or x not in pre_):
                continue
            if all(_want.get(y) in (None, "raw") or _ir_(game, y, _want.get(y)) for y in pre_):
                _rescue = pre_
                break
        if _rescue:
            _nodes0 = [n for n in remaining_plan(game, state, plate_now) if n.type == "assemble"]
            if not held_plate:
                cmds += plate_get() if (parked_plate or plate_now) else ["go to the plates", "pick up"]
                held_plate, parked_plate = True, False
            for y in _rescue:
                a_ = next((n for n in _nodes0 if n.component == y), None)
                if a_ is None:
                    break
                _nodes0.remove(a_)
                cmds += list(a_.commands)
                plate_now = plate_now + [y]
    except Exception:                                               # noqa: BLE001
        pass
    # 3. the template's remaining subgoals, judged from the live kitchen
    nodes = remaining_plan(game, state, plate_now)
    preps = [n for n in nodes if n.type in ("prep", "ignite")]
    asms = [n for n in nodes if n.type == "assemble"]
    have_plate_node = any(n.sid == "fetch_plate" for n in nodes)
    def station_of(comp):
        for n in preps:
            if n.component == comp:
                for c in n.commands:
                    if c.startswith("go to the empty "):
                        return c[len("go to the empty "):].strip()
                    if c.startswith("go to the ") and " with the " in c:
                        return c[len("go to the "):].split(" with the ")[0].strip()
        return _station_label(comp)
    def wait_for(comp):
        return f"wait until the {station_of(comp)} with the {comp} is cooked"
    def ignite_of(comp):
        return next((n for n in preps if n.type == "ignite" and n.component == comp), None)
    def prep_block(n, light_now):
        block = list(n.commands)
        if light_now and n.type == "prep":
            ig = ignite_of(n.component)
            if ig is not None and any("put down" == c for c in block):
                block += ["turn on", wait_for(n.component)]      # light the station just loaded
        return block
    def plate_get():
        if parked_plate:
            return ["go to the counter with the plate", "pick up"]
        return ["go to the plates", "pick up"]
    def park_if_held():
        nonlocal held_plate, parked_plate
        if held_plate:
            held_plate, parked_plate = False, True
            return ["go to the empty counter", "put down"]
        return []
    if policy in ("template", "reverse"):
        # late-ignite, exactly the template's shape; "reverse" loads the stations in reverse recipe
        # order (a legal reordering that never lights early, so nothing burns while it preps)
        cmds += park_if_held()
        prep_nodes = [n for n in preps if n.type == "prep"]
        if policy == "reverse":
            comps = []
            for n in prep_nodes:
                if n.component not in comps:
                    comps.append(n.component)
            rank = {c: i for i, c in enumerate(reversed(comps))}
            prep_nodes = sorted(prep_nodes, key=lambda n: rank[n.component])
        for n in prep_nodes:
            cmds += list(n.commands)
        lit = []
        for n in preps:
            if n.type == "ignite" and n.component not in lit:
                lit.append(n.component); cmds += list(n.commands)
        for c in lit:
            if ignite_of(c) is not None:
                cmds.append(wait_for(c))
        if have_plate_node and not (held_plate or parked_plate):
            cmds += ["go to the plates", "pick up"]
        elif not held_plate:
            cmds += plate_get()
        for n in asms:
            cmds += list(n.commands)
    else:
        # plate_early: walk the recipe IN PLATING ORDER; an item that needs prep is made, lit,
        # waited for and plated before the next item is touched (a second unit of an ingredient
        # is prepped fresh, since the first was scooped already)
        for a in asms:
            comp = a.component
            group = [n for n in preps if n.type == "prep" and n.component == comp]
            if group:
                # the whole batch of this ingredient (the template plates a batch with one scoop):
                # every unit is loaded (first into an empty station, the rest joining it), lit once,
                # waited for, then plated
                for pn in group:
                    preps.remove(pn)
                cmds += park_if_held()
                for pn in group:
                    cmds += list(pn.commands)
                if ignite_of(comp) is not None:
                    cmds += list(ignite_of(comp).commands) + [wait_for(comp)]
                cmds += ([] if held_plate else plate_get())
                held_plate, parked_plate = True, False
                cmds += list(a.commands)
            else:
                ig = ignite_of(comp)
                if ig is not None and ig in preps:
                    preps.remove(ig)
                    cmds += park_if_held()
                    cmds += list(ig.commands) + [wait_for(comp)]
                if not held_plate:
                    cmds += plate_get(); held_plate, parked_plate = True, False
                cmds += list(a.commands)
    # an announced plan change extended the goal (goal_contents): plate the extra raw items too, before the serve
    for n_, st_ in list(state.goal_contents())[len(recipe.contents):]:
        if n_ not in plate_now:
            if not held_plate:
                cmds += plate_get(); held_plate, parked_plate = True, False
            cmds += [f"go to the {n_} source", "scoop"]
    cmds += ["go to the serving pass", "serve"]
    return cmds


def scripted_plan_order(nodes, variant):
    """A legal reordering of the remaining subgoals for the scripted verification cook."""
    nodes = list(nodes)
    if variant in (None, "template"):
        return nodes
    if variant == "ignite_at_load":
        out, ign = [], {n.component: n for n in nodes if n.type == "ignite"}
        used = set()
        for n in nodes:
            if n.type == "ignite":
                if n.component in used:
                    continue
                out.append(n); used.add(n.component); continue
            out.append(n)
            if n.type == "prep" and n.component in ign and n.component not in used:
                out.append(ign[n.component]); used.add(n.component)
        return out
    if variant == "reverse":
        preps = [n for n in nodes if n.type in ("prep", "ignite")]
        rest = [n for n in nodes if n.type not in ("prep", "ignite")]
        comps = []
        for n in preps:
            if n.component not in comps:
                comps.append(n.component)
        order = {c: i for i, c in enumerate(reversed(comps))}
        preps.sort(key=lambda n: (order[n.component], 0 if n.type == "prep" else 1))
        return preps + rest
    if variant == "plate_early":
        # plate each ready item as soon as its prep is done: prep(x), ignite(x), fetch_plate, assemble(x) ...
        out, plate_done = [], False
        by_comp = {}
        for n in nodes:
            by_comp.setdefault(n.component, []).append(n)
        for n in nodes:
            if n.type in ("prep", "ignite") and n not in out:
                out.append(n)
                if n.type == "ignite" or not any(x.type == "ignite" for x in by_comp.get(n.component, [])):
                    if not plate_done:
                        fp = next((x for x in nodes if x.sid == "fetch_plate"), None)
                        if fp: out.append(fp)
                        plate_done = True
                    asm = next((x for x in by_comp.get(n.component, []) if x.type == "assemble" and x not in out), None)
                    if asm: out.append(asm)
        for n in nodes:
            if n not in out:
                out.append(n)
        return out
    return nodes


def run(case_name, persona, paradigm, client, page, video_path=None, obs_mode="frames",
        persona_v2=None,
        family="error", pauses=False, sup_client=None, cook_client=None):
    sup_client = sup_client if sup_client is not None else client   # assistant seat
    # COOK SEAT (2026-09-22, cook-model ablation): by default the cook speaks through the same
    # shared Gemini client as everything else; pass cook_client to put a DIFFERENT model behind
    # the cook's voice. Nothing else changes -- the prompt, the reply schema, the persona card,
    # the move set and the harness rules are built before any client is consulted, so the two
    # seats differ only in which endpoint answers.
    cook_client = cook_client if cook_client is not None else client
    t0 = time.time()
    vdir, vcount = None, [0]
    if video_path:
        import tempfile
        vdir = tempfile.mkdtemp(prefix="stepin_")
    cases = {}
    if os.environ.get("COOKSIM_CASES"):
        # the suite named by the runner takes precedence over every pool below (the pools are
        # merged first-wins, so a curated suite that reuses case names must be loaded first)
        try:
            _cc = json.load(open(os.environ["COOKSIM_CASES"]))
            for c in (_cc["cases"] if isinstance(_cc, dict) else _cc):
                cases.setdefault(c["name"], c)
        except Exception as _e:                              # noqa: BLE001
            print(f"[setup] COOKSIM_CASES unreadable: {_e!r}", flush=True)
    # the shipped suite (benchmark/cases_composite_v5.json) is always available as a fallback pool
    try:
        _cc = json.load(open(os.path.join(ROOT, "benchmark", "cases_composite_v5.json")))
        for c in (_cc["cases"] if isinstance(_cc, dict) else _cc):
            cases.setdefault(c["name"], c)
    except Exception as _e:                              # noqa: BLE001
        print(f"[setup] shipped suite unreadable: {_e!r}", flush=True)
    if case_name not in cases:
        raise SystemExit(f"[setup] unknown case {case_name!r}: not in COOKSIM_CASES ({os.environ.get('COOKSIM_CASES')}) "
                         f"nor in the shipped suite")
    case = cases[case_name]
    # COMPOSITE SCHEDULE (2026-08-19): a case may be a SEQUENCE of sub-cases (errors and
    # qa_probes) run one at a time in one continuous episode — sub k+1 arms only after
    # sub k released. Each sub is a full ordinary case (by name or inline); they must
    # share the cell. The episode state machine below is untouched: load_case() rebinds
    # the per-injection context and the singleton state is reset between subs.
    subs, sub_idx = None, [0]
    if case.get("family") == "composite" or case.get("subs"):
        subs = [dict(cases[s_]) if isinstance(s_, str) else dict(s_)
                for s_ in case["subs"]]
        for s_ in subs:
            if s_["cell"] != case["cell"]:
                raise SystemExit(f"[setup] composite sub {s_.get('name')!r} is on cell "
                                 f"{s_['cell']!r}, case cell is {case['cell']!r}")
        composite_name, composite_cell = case.get("name", case_name), case["cell"]
        case = subs[0]
    WAS_COMPOSITE = subs is not None
    if SCHED_V5 and subs is None:
        subs = [dict(case)]
        composite_name, composite_cell = case.get("name", case_name), case["cell"]
    family = case.get("family", family)    # the case itself is authoritative
    # CASE-POOL STAMP (2026-08-13): engine_rev never covered the case files, so two
    # episodes could share a rev yet have run different pools (the qa-auto merge added 60
    # cases without moving the rev). A separate hash keeps pool edits auditable without
    # churning the engine stamp.
    import hashlib
    _pool_h = hashlib.md5()
    for _pf in ((os.environ.get("COOKSIM_CASES"),) if os.environ.get("COOKSIM_CASES") else ()) + (
                "tmp/interact_eval/recovery_cases.json", "tmp/interact_eval/cases_v2.json",
                "tmp/interact_eval/cases_intent.json",
                "tmp/interact_eval/cases_newrecipes.json",
                "tmp/interact_eval/cases_hardmaps.json",
                "tmp/interact_eval/cases_composite_errors.json",
                "tmp/interact_eval/cases_qa_hard.json",
                os.path.join(ROOT, "benchmark", "cases_composite_v5.json"),
                "tmp/interact_eval/cases_composite_v3.json"):
        try:
            _pool_h.update(open(_pf, "rb").read())
        except Exception:
            _pool_h.update(b"absent")
    case_pool_rev = _pool_h.hexdigest()[:10]
    INTENT = family != "error"
    # H10/M6: the ingredients this episode ANNOUNCES it will fetch (a plan change's topping). A walk-away's
    # wander must not stop at their crates — standing at the onion crate right after saying "I'll add onion on
    # top" reads as the announced plan, not as distraction.
    _lu_avoid = tuple(sorted({(s_.get("goal_extend") or {}).get("ingredient")
                              for s_ in (subs or [case])
                              if (s_.get("goal_extend") or {}).get("ingredient")}))
    # HARD-SOFT (user design 2026-08-13): the engine owns the HANDS, the prompt owns the
    # VOICE. Queue mechanics follow the hard path (splice, takeover, prevention restore);
    # the task block renders like soft — but as a DESCRIPTIVE instruction ("arrange your
    # conversation around your current action and queue"), never an imperative one, so it
    # can never go stale and the soft compliance failures are structurally impossible.
    HARDQ = paradigm in ("hard", "hardsoft")     # engine owns the queue
    TASKP = paradigm in ("soft", "hardsoft")     # the task block renders in the prompt
    if persona_v2 is not None:
        # PERSONA V2 (user-approved): mechanisms from profile_dials(), prompt text from
        # human_card() — registered under the run label so every existing lookup works.
        from cooksim.interact.state import Profile as _Profile
        d = persona_v2.profile_dials()
        RECOVERY_PERSONAS[persona] = _Profile(d["task_solvability"],
                                              d["interaction_preference"],
                                              d["information_acceptance"], persona)
        PERSONA_CARDS[persona] = persona_v2.human_card()
    prof = RECOVERY_PERSONAS[persona]
    CREATIVE = prof.task_solvability == "creative"
    PHINT = (PERSONA_CARDS.get(prof.persona, "You are an ordinary cook.")
             + "\n\nStay in character. Output your reply JSON now.")
    err_call = (None if INTENT else err_call_of(case, avoid=_lu_avoid))  # sub 0; load_case rebinds

    # LIVE episode: CLEAN plan, human-driven from t0 (no scripted deviation baked in)
    state, ro = build_episode(None, case.get("persona") or "capable_collaborator",
                              cell=case["cell"], inject=case.get("inject", True))
    state.profile = prof
    ro.cap = 500 if not subs else 500 + 200 * (len(subs) - 1)

    # deviated continuation, computed from the SAME composition the scripted e2e uses
    flat_clean = [(c, n.sid) for n in segment_plan(state.recipe) for c in n.commands]
    # plan live from t0 — there are no stages. enter_error_correction clears the queue
    # (it expects an apply_fix to follow), so re-seed the human's starting routine: the
    # full CLEAN plan, exactly what a cook who "always makes it this way" would queue.
    system.enter_error_correction(ro, state)
    if SCHED_V5:
        # ruling 2026-09-09: the cook plans for itself. Nothing is queued at t0; the cook's
        # first idle turn writes its own plan. The template survives only as the DEFINITION of
        # the anchors and as the base the takeover block is composed on.
        state.pending_tail = []
        state.mode = "executing"
        state.correction = None
    else:
        ro.cmds = ro.cmds[:ro.i + 1] + list(flat_clean)
    # human-paced episodes argue mid-cook; default burn timers torch the kitchen during
    # any long conversation. E8 (2026-08-15): this was an inline patch that the CASE
    # GENERATORS did not share (and it skipped fryer_burn_time), so sheets were certified
    # under one physics and episodes scored under another. One definition now, in
    # cooksim/core/config.py, applied by both.
    apply_pace(ro.game)                      # COOKSIM_PACE=human (default, 480/120) | tight (default burn windows, fire 30)
    # ---- HUMAN-REALISM PAUSES (ported verbatim in spirit from recovery_rollout) -------
    # SPEAKING costs game time: the cook stands still for the length of its OWN sentence.
    # LISTENING is free — real cooks keep working while someone talks at them. Crucially
    # the WORLD DOES NOT STOP: ro._pause only makes the cook issue STAY, while g.step()
    # still runs, so pans keep cooking and stations keep smoking through every pause.
    pauses_log = []

    def speech_ticks(text_):
        # study-app parity: TTS ~3.15 words/s and 1 tick ~ 1.2 s (move-throttle parity)
        return max(1, min(20, round(len(str(text_).split()) / 3.15 / 1.2)))

    def add_speech_pause(who, text_):
        """RECORD how long a line takes to say. It no longer costs GAME TIME.

        Speech used to stop the cook's hands for the length of its own line, and (briefly) the
        assistant's line advanced the world as well. Both are gone: messages are received
        immediately on the next turn, and the clock does not care who is talking. Charging the
        clock produced episodes that were 55-88% frozen, ticks that tripled for a chatty cook,
        food that burnt while the assistant explained, and an assistant correctly reporting
        "you are not moving" at a cook that had no ticks in which to move.

        The duration is still logged, because the VIDEO holds on the frame until the line has
        finished being spoken — that is where speech time belongs, not in the simulation.
        """
        if not text_:
            return
        pauses_log.append({"t": ro.tick, "who": who,
                           "words": len(str(text_).split()),
                           "ticks": speech_ticks(text_), "costs_game_time": False})

    if pauses:
        import random as _rnd
        ro.hesitate_p = 0.2              # 20% chance of a human hesitation per step
        ro._hes_rng = _rnd.Random(17)
        ro.hesitations = []
    # ---- PER-INJECTION DERIVATION (2026-08-19, composite-ready) -----------------
    # Everything a single injection needs — the deviated continuation, the deviation
    # point m, the plate-anchor reference key — derived from the CLEAN plan. Valid for
    # every sub of a composite schedule because the plate-only inject_key is
    # path-invariant: a cook that recovered from sub k re-enters the clean plate
    # trajectory, so sub k+1's clean-prefix anchor still fires. All names are rebound
    # with `nonlocal` so every helper below sees the live sub-case.
    report = None                       # real dict is built after sub 0 loads
    trigger_deadline = [TRIGGER_DEADLINE]
    flags_base = [0]                    # flags[] index where the live sub begins

    def load_case(sub, first=False):
        nonlocal case, family, INTENT, err_call, m, dev_section, dev_cont, \
                 dev_block_sids, first_dev_cmd, flat_err, ref_key, ref_state_key, \
                 ref_plate, report_splice_dedupe, anchor_collapsed
        case = sub
        family = case.get("family", "error")
        INTENT = family != "error"
        err_call = (None if INTENT else err_call_of(case, avoid=_lu_avoid))
        report_splice_dedupe = False
        if INTENT:
            # There is no error twin to diff against: the case states outright where the
            # intent starts (trigger_after), what the cook decides to do (intent_seq) and
            # where the clean plan resumes. gen_intent_cases.py has already replayed this
            # exact splice and proved the order still completes.
            m = case["trigger_after"]
            _seq = list(case.get("intent_seq") or [])
            dev_section = list(_seq)
            # ECR-7: "intent" (a real string sid), NOT None — every consumer of the queue
            # assumes sids are strings (state.py human_json crashed on None and killed every
            # hard plan_change episode before a report was written). "intent" reads
            # correctly in past_subgoals and is distinct from every recipe sid.
            # F5 (2026-08-14 fix batch) SPLICE DEDUPE: cases whose intent ends where the clean
            # plan resumes emitted the shared command twice back-to-back (…"go to the pan",
            # "go to the pan"…). The second is a no-op the cook still has to issue and wait
            # out, and it showed up in the transcript as the cook repeating itself.
            _resume = list(flat_clean[case["resume_from"]:])
            if _seq and _resume and _seq[-1] == _resume[0][0]:
                _resume = _resume[1:]
                report_splice_dedupe = True
            else:
                report_splice_dedupe = False
            dev_cont = [(c, "intent") for c in _seq] + _resume
            first_dev_cmd = _seq[0] if _seq else None
            flat_err = list(flat_clean)          # unused downstream, kept for symmetry
        else:
            err_ir, _recs = X.E.compose(list(segment_plan(state.recipe)), [err_call])
            flat_err = [(c, n.sid) for n in err_ir for c in n.commands]
        if not INTENT:
            # longest common prefix / suffix between the clean and errored plans -> the
            # deviation point. INTENT cases carry their own trigger point (and their two
            # plans are identical), so this diff must not run for them.
            m = 0
            while (m < min(len(flat_clean), len(flat_err))
                   and flat_clean[m][0] == flat_err[m][0]):
                m += 1
            sfx = 0
            while (sfx < min(len(flat_clean), len(flat_err)) - m
                   and flat_clean[-1 - sfx][0] == flat_err[-1 - sfx][0]):
                sfx += 1
        dev_block_sids = set()
        if SCHED_V5 and INTENT:
            dev_cont = [(c, "intent") for c in _seq]     # the detour only; the cook resumes itself
        if SCHED_V5 and not INTENT:
            _blk = live_dev_block(err_call, ro.game, state, plate_list(ro.game))
            _cmds, _sids, _first, _pairs = _blk if _blk is not None else ([], set(), None, [])
            dev_cont = list(_pairs)                       # the deviated block only — nothing behind it
            dev_section = list(_cmds)
            dev_block_sids = set(_sids)
            first_dev_cmd = _first
        if not INTENT and not SCHED_V5:
            dev_cont = flat_err[m:]                    # injected continuation (hard splice)
            # W19 (user-approved 2026-08-12): the command-level suffix trim ate the tail of a
            # SUBSTITUTED subgoal block — mis_prep swaps only the first command of a
            # five-command prep block, so dev_section collapsed to one navigation command
            # that cannot commit any error, and the soft task looped on it forever. Extend
            # the section forward to the end of the sid block containing the last difference,
            # so the tasked sequence is always whole subgoal blocks.
            end = len(flat_err) - sfx
            while m < end < len(flat_err) and flat_err[end - 1][1] == flat_err[end][1]:
                end += 1
            dev_section = [c for c, _s in (flat_err[m:end] or flat_err[m:m + 4])]
            # the deviated block's subgoal ids -- what "the error has happened in the world"
            # is measured against (max-hold release, below): the executor has moved past
            # every command carrying one of these sids.
            dev_block_sids = {s_ for _c, s_ in (flat_err[m:end] or flat_err[m:m + 4])}
            first_dev_cmd = dev_cont[0][0] if dev_cont else None

        # TRIGGER reference: headless replay of the clean prefix -> expected kitchen state
        rstate, rro = build_episode(None, case.get("persona") or "capable_collaborator",
                                    cell=case["cell"], inject=case.get("inject", True))
        rro.cmds = list(flat_clean[:m])
        rro.cap = 450
        while rro.i < len(rro.cmds) or rro.ctrl.skill is not None:
            if not rro.step():
                break
        ref_key = inject_key(rro.game, rro)
        ref_state_key = state_key(rro.game)   # kept as a diagnostic: old episodes were
                                              # triggered on this, so report both.
        ref_plate = plate_names(rro.game)

        # F5 (2026-08-14 fix batch) ANCHOR-COLLAPSE HONESTY. maybe_inject() fires on the first
        # tick the live kitchen matches `ref_key`. When the case's prefix does no plate work,
        # the replayed reference state is identical to the state at t0 — so the trigger matches
        # IMMEDIATELY and the injection lands at t0 regardless of what trigger_after asked for.
        # The episode is still valid, but it is not testing the anchor the case describes, and
        # nothing said so. Say so. (Re-anchoring the cases themselves is a case-gen pass.)
        anchor_collapsed = bool(m > 0 and inject_key(ro.game, ro) == ref_key)
        if anchor_collapsed and first and not SCHED_V5:
            # E1 (hard-map audit, user ruling 2026-08-15): REFUSE the episode instead of
            # running it. The plate-only key fingerprints the plate and nothing else, so a
            # case whose deviation precedes any plate work matches the empty t0 kitchen: the
            # injection fires at tick 0 and the takeover splice discards the entire
            # un-executed cooking prefix. The cook is then handed a queue whose commands
            # reference food that was never cooked ("go to the pan with the steak" — there is
            # no such pan), wanders, and dies to the idle tail. hm_search_expM9 was scored
            # failed_at=detection against a supervisor that could not possibly have named the
            # error; hm_search_masM9 emitted four impossible gotos in one tick. Such a case is
            # not a hard episode, it is an unrunnable one — and the case should never have
            # been generated. Refusing here is the backstop; gen_detect_cases_v2 drops these
            # candidates at generation time (see its collapse check).
            raise SystemExit(
                f"[setup] REFUSING EPISODE — anchor collapsed to t0 (trigger_after={m}, "
                f"case={case_name!r}): the reference key already matches the starting "
                f"kitchen, so injection would fire at t0 and amputate the whole prep prefix. "
                f"This case is inadmissible; regenerate it with an anchor that follows a "
                f"plate event, or drop it from the pool.")

        # per-injection trigger deadline: relative to the moment this sub armed, so a
        # long recovery of sub k never eats sub k+1's whole budget
        trigger_deadline[0] = ro.tick + TRIGGER_DEADLINE
        flags_base[0] = len(flags)
        if report is not None:
            report["case_offsets"] = ({"lo": 0, "hi": 0} if INTENT else
                                      {"lo": case["window"][0] - case["dev"],
                                       "hi": case["window"][1] - case["dev"]})
            report["dev_section"] = dev_section
            report["err_live"] = realized_err(case)          # G1: the type that actually runs

    m = 0
    dev_section, dev_cont, dev_block_sids, first_dev_cmd = [], [], set(), None
    flat_err, ref_key, ref_state_key, ref_plate = list(flat_clean), None, None, set()
    report_splice_dedupe, anchor_collapsed = False, False
    flags = []                          # declared here so load_case can mark flags_base

    def derive_spec(idx, sub):
        """v5 anchor of one sub-case: the plate prefix and the prep-progress SET of the clean
        run at the sub-case's clean-prefix length m (m: intents carry trigger_after; errors are
        the first command where the errored template diverges from the clean one)."""
        fam = sub.get("family", "error")
        if fam == "error":
            ec = err_call_of(sub, avoid=_lu_avoid)
            err_ir, _r = X.E.compose(list(segment_plan(state.recipe)), [ec])
            fe = [(c, n.sid) for n in err_ir for c in n.commands]
            mm = 0
            while mm < min(len(flat_clean), len(fe)) and flat_clean[mm][0] == fe[mm][0]:
                mm += 1
        else:
            ec = None
            mm = sub["trigger_after"]
        rstate_, rro_ = build_episode(None, sub.get("persona") or "capable_collaborator",
                                      cell=sub["cell"], inject=sub.get("inject", True))
        rro_.cmds = list(flat_clean[:mm])
        rro_.cap = 450
        while rro_.i < len(rro_.cmds) or rro_.ctrl.skill is not None:
            if not rro_.step():
                break
        prep_req_ = prep_units(rro_.game, state.recipe)
        # a question/request about ONE item only needs that item in processing. ready_in questions carry the item in
        # gt_kind only ("ready_in:fish"): they inherited the whole reference prep (fish AND both potatoes) and the fish
        # was done before the floor was reached (LLM-cook check, hard_medium_2 / hard_hard_1, 2026-09-17)
        q_item_ = sub.get("item")
        if not q_item_ and (sub.get("gt_kind") or "").split(":")[0] in ("ready_in", "ready"):
            q_item_ = sub["gt_kind"].split(":", 1)[1]
        if fam in ("qa_probe", "qa_proactive") and q_item_ and q_item_ in prep_req_:
            prep_req_ = {q_item_: max(1, prep_req_.get(q_item_, 0))}
        # suite v5 (2026-09-19): a sub-case may state its prep floor explicitly. The reference run preps EVERYTHING
        # before the first plate (the template's order), so a plate-phase trigger_after inherited "lettuce chopped" and
        # an LLM cook that chops the lettuce last asked the question on the walk to the pass. The designer
        # (python -m cooksim_bench.suite.design_composite_v5) writes the prep an LLM cook does early; absent, the reference prep is used.
        # fire-longtail (2026-09-20): ANY sub-case may state its own prep floor, not only a question or a walk-away.
        # With the plating triggers forced, serve_raw / misidentify / add_foreign / wrong_order fire as soon as the
        # dish is under way — right for the type, but the designer must be able to place one mid-dish. A STATED floor
        # is the only place left where a designer says "not before this much of the dish is done", and because it is
        # stated (never inherited from the reference run) it can be written to be reachable in any cooking order.
        if (fam != "error" or sub.get("err") == "leave_unattended"
                or isinstance(sub.get("prep_floor"), dict) or isinstance(sub.get("plate_floor"), int)) \
                and isinstance(sub.get("prep_floor"), dict):
            # E9: a walk-away may carry its own floor too (the reference prep of a late-ignite template is "everything
            # prepped", which an LLM cook that lights the first item early never has while that item is near done)
            prep_req_ = {k: int(v) for k, v in sub["prep_floor"].items()}
        return {"idx": idx, "sub": sub, "family": fam,
                "err": (realized_err(sub) if fam == "error" else sub.get("err")),
                "err_designed": sub.get("err"),
                "kind": (sub.get("gt_kind") if fam == "qa_probe" else fam), "item": q_item_ if fam in ("qa_probe", "qa_proactive") else sub.get("item"),
                "err_call": ec, "m": mm,
                "plate": tuple(plate_list(rro_.game)), "prep_req": prep_req_,
                "plate_floor": (int(sub["plate_floor"]) if isinstance(sub.get("plate_floor"), int) else None),
                # D17 (round 3): a walk-away whose floor can only be met after its food is done (the burrito steak:
                # rice plated first) may fire up to `lu_late` ticks after done (never on smoking food)
                "lu_late": (int(sub["lu_late"]) if isinstance(sub.get("lu_late"), int) else None),
                "slot": ("voice" if fam in ("qa_probe", "qa_proactive") else "body"),
                "state": "pending", "since": None, "skipped": None, "wait_note": None,
                "fired_at": None, "released_at": None}

    SPECS = [derive_spec(i, s_) for i, s_ in enumerate(subs)] if SCHED_V5 else []
    if os.environ.get("COOKSIM_SKIP_SPECS") == "1":
        for sp in SPECS:
            sp["state"] = "disabled"
    prep_max = {}
    # fire-longtail (2026-09-20): the same running max WITHOUT the recipe clamp — the only ledger that can see
    # a repeat_step on a prep (a unit beyond what the recipe needs). prep_max itself stays clamped: it is a
    # progress FLOOR for the anchors and an extra unit must not move it.
    prep_max_raw = {}
    plated_ever, max_plate_len = set(), [0]      # live-question progress (never lowered by a trash)
    body_spec = [None]
    voice = {"spec": None, "since": None, "tries": 0}
    if SCHED_V5:
        print("[setup] v5 anchors: " + "; ".join(
            f"[{sp['idx']}] {sp['family']}/{sp['err'] or sp['kind']} plate={list(sp['plate'])} prep>={sp['prep_req']}"
            for sp in SPECS), flush=True)
    load_case(case, first=True)

    sup_sys = e2e_supervisor_prompt(state, obs_mode=obs_mode, game=ro.game,
                                    hazard_ticks=(ro.game.config.pan_burn_time,
                                                  ro.game.config.fire_time,
                                                  ro.game.config.smoke_grace),
                                    # F3: a v2 run's card is PRIVATE to the human. It is
                                    # registered under PERSONA_CARDS for the human path
                                    # (PHINT) and must not reach the supervisor's profile
                                    # block from there.
                                    private_persona=(persona_v2 is not None))
    if persona_v2 is not None:
        # v2 asymmetry: the assistant is shown the QUESTIONNAIRE and nothing else about
        # the person — skill and trust must be inferred from behaviour.
        sup_sys = sup_sys + "\n\n" + persona_v2.assistant_block()
    # NATIVE streaming seat (--assist-backend live_native / realtime_native, roster rows 7 and 9):
    # the same frames-only prompt with its turn-based parts (per-turn inputs, cook_progress JSON,
    # reply JSON) swapped for live-stream ones; see python -m cooksim_bench.backends.native_assistant
    NATIVE = bool(getattr(sup_client, "native", False))
    if NATIVE:
        from cooksim_bench.backends.native_assistant import native_system_prompt as _native_prompt
        sup_sys = _native_prompt(sup_sys)
    prompts_snapshot = {
        "supervisor_system": sup_sys, "human_briefing": BRIEFING,
        # X5 (2026-08-13 audit): this used to render the SOFT template unconditionally, so
        # every hardsoft report archived instructions the cook never saw. Filled with
        # task_text() — the single source of what actually renders — right after that
        # function is defined below.
        "system_task_template": None,
        "human_system_example": human_context(state, ro, [], "(none)", None,
                                              game=ro.game, body_extra=BRIEFING)}

    fbuf, comm, convo, path = [], [], [], []
    pending_reply = [None]               # one assistant reply in flight (see ASSISTANT LATENCY)
    ro.convo = comm                      # suite v3 truths (assistant_said:) read the conversation off the rollout
    ro.qa_history = []                   # earlier questions' truths, for reask:
    # round-2 audit (2026-09-11): `cell` is not a local on the composite/v5 path, so this silently
    # failed and every "what still has to go on this plate?" truth read "no recipe on ro/game"
    # (8/8 in the v5 sample, 20/20 in the v4 sweep)
    ro.cell = case.get("cell")
    ep_alive, armed, RND = [True], [False], [1]
    quiet_events = {"injection": None, "commit": None, "release": None, "flag": None}   # A5
    prevention = [False]           # early-window CORRECT-name flag arms the PREVENTION path
    injected, released, skipped = [False], [False], [None]
    past_since = [None]                     # tick the plate first went past this sub's anchor
    heat_wait = [None]                      # leave_unattended: tick the anchor was reached with nothing on the heat
    dev_live = [None]
    report_moves, flag_repeats = [], [0]
    report = dict(setting="e2e_stepin", paradigm=paradigm, scenario=case_name,
                  persona=persona, err=case["err"], obs_mode=obs_mode,
                  # persisted so scorers grade against the REAL kind — eval_intent's
                  # scenario-name fallback defaulted every qa_probe to plate_count
                  gt_kind=case.get("gt_kind"), family=case.get("family"),
                  case_pool_rev=case_pool_rev,
                  persona_v2=(persona_v2.to_json() if persona_v2 is not None else None),
                  questionnaire_v2=(persona_v2.questionnaire_json()
                                    if persona_v2 is not None else None),
                  dev=None, window=None,            # filled live at injection
                  case_offsets=({"lo": 0, "hi": 0} if INTENT else
                                {"lo": case["window"][0] - case["dev"],
                                 "hi": case["window"][1] - case["dev"]}),
                  clean_prefix_len=m, dev_section=dev_section,
                  anchor_collapsed_to_t0=anchor_collapsed,
                  splice_deduped=report_splice_dedupe,
                  convinced=False, entered=True,    # plan is live from t0 by design
                  fix_source=None, conversation=[], moves=report_moves,
                  prompts=prompts_snapshot,
                  agent_version=__import__("cooksim.core.config",
                                          fromlist=["x"]).agent_version())

    # NO-RENDER MODE (2026-08-26, sweep throughput): with page=None nothing is drawn —
    # legal only when the assistant reads the text state map and no video is wanted
    # (COOKSIM_NO_VIDEO=1 with --sup-obs != frames). Software WebGL in headless Chrome was
    # 2/3 of the box's CPU during the 2,700-episode sweep, for frames nobody consumed.
    if page is not None:
        page.wait_for_function("() => window.__renderFrameSbs !== undefined", timeout=90000)
        # PHOTOREAL LIGHTING IS ASYNC. render3d.js loads env.hdr through RGBELoader and
        # only then applies the image-based lighting; a frame captured before that is lit
        # by the dim fallback and REAL-mode surfaces come out near-black. One render is
        # needed first because the renderer (and the flag) are constructed lazily.
        try:
            page.evaluate("(s)=>window.__renderFrame(s)",
                          json.dumps(ro.game.render_state()))
            page.wait_for_function("() => window.__envReady === true && (window.__loadsPending || 0) === 0", timeout=120000)
        except Exception as _e:
            print(f"[warn] photoreal env wait failed: {_e}", flush=True)
    _dump = os.environ.get("COOKSIM_DUMP_STATES")
    if _dump:                          # a fresh dump per attempt: a resume after a preemption must not append to a partial one
        open(_dump, "w").close()
    _fdir = os.environ.get("COOKSIM_FRAME_DIR")
    if _fdir:
        os.makedirs(_fdir, exist_ok=True)

    _pause_slot = [False]

    def snap():
        path.append(tuple(ro.game.cooks[0].pos))
        _st = json.dumps(ro.game.render_state())
        if _dump:                          # env-gated per-tick state dump (JSONL)
            with open(_dump, "a") as _f:
                _f.write(_st + "\n")
        # SIDE-BY-SIDE OBSERVATION (2026-07-31): one image per tick carrying BOTH
        # cameras — top-down LEFT (the coordinate frame), first-person RIGHT — so the
        # supervisor sees a synchronised pair by construction. JPEG: 36 KB vs 196 KB
        # for the old PNG. The same frames are written to disk, so the presentation
        # video needs NO separate filming pass and still gets both views.
        if page is None:                   # no-render mode: text observation only
            return
        url = page.evaluate("(s)=>window.__renderFrameSbs(s)", _st)
        png = base64.b64decode(url.split(",", 1)[1])
        # while a speech pause drains the cook is motionless BY THE CONVERSATION; letting
        # those identical frames fill the 8-frame window taught the supervisor the cook had
        # stopped working and invited it to nag (see recovery_rollout for the measured
        # case). Keep ONE frozen frame, refreshed in place: 7 frames of real action history
        # plus the current situation, so a pan that starts smoking mid-pause is still seen.
        _frozen = bool(pauses) and getattr(ro, "_pause", 0) > 0
        if _frozen and _pause_slot[0] and fbuf:
            fbuf[-1] = png
        else:
            fbuf.append(png)
        _pause_slot[0] = _frozen
        del fbuf[:-12]
        if vdir is not None:
            open(f"{vdir}/{vcount[0]:05d}.jpg", "wb").write(png)
            vcount[0] += 1
        if _fdir:                          # film-ready capture, one composite per tick
            open(f"{_fdir}/{len(path)-1:05d}.jpg", "wb").write(png)

    def engine_snap():
        try:
            return state.to_json(ro)
        except Exception:
            return None

    def sup_obs_text():
        from cooksim.agents.observe_text import observe_text
        return observe_text(ro.game, path=path[-WINDOW:], viewpoint="supervisor")

    def sup_state_extra():
        """agent+++ readout — SINGLE SOURCE in python -m cooksim_bench.agent_context (see the bug it
        fixes there). Never inline a private copy of this."""
        from cooksim_bench.agent_context import supervisor_state_extra
        return supervisor_state_extra(ro.game)
    # ---------- kitchen feedback (verbatim recovery loop)
    from cooksim.interact import checks as _checks
    REASON = {"hands_full": "my hands are full", "no_plate": "I'm not holding a plate",
              "hands_empty": "my hands are empty", "station_empty": "there is nothing at the station",
              "no_finished_plate": "the plate isn't complete enough to serve",
              "no_such_place": "I can't find that place", "unparseable": "that isn't an action I can do",
              "not_cooked_yet": "the food isn't cooked yet"}
    # `fb_log` archives every kitchen-feedback line WITH the tick it was generated on.
    # fb_buf is cleared after each human turn (it is a since-you-last-acted buffer), so
    # nothing about what the kitchen actually told the cook survived into the report — which
    # made the one claim this engine rests on ("the refusal reaches the cook, and it is
    # concrete") unverifiable after the fact. ECR-13.
    seen_term, fb_buf, fb_log = set(), [], []

    def _cold_station_faced():
        try:
            from cooksim.core.enums import DIRECTION_VECTORS
            from cooksim.core.stations import CookStation
            c = ro.game.cooks[0]
            dx, dy = DIRECTION_VECTORS[c.direction]
            st_ = ro.game.stations.get((c.pos[0] + dx, c.pos[1] + dy))
            return isinstance(st_, CookStation) and bool(st_.contents) and st_.status == "raw"
        except Exception:
            return False

    def _why(cmd):
        """Why a command failed / is stuck, in the cook's own words.

        ECR-13 §A: consult the executor's OWN precondition message first (it is accurate
        about the FACED tile), exactly as python -m cooksim_bench.recovery_rollout has done since the
        recovery suite, and only then fall back to the is_doable blocker map. is_doable now
        delegates to the same `_validate`, so the two agree by construction — the prelude
        stays because it also survives a blocker KEY being mapped back to a terser phrase."""
        from cooksim.agents.highlevel import parse as _hl_parse, _validate, MANIPULATIONS
        it = _hl_parse(cmd)
        if it and it.verb in MANIPULATIONS:
            try:
                ok, reason = _validate(it.verb, ro.game, ro.game.cooks[0])
                if not ok and reason:
                    return reason
            except Exception:
                pass
        try:
            b = _checks.is_doable(cmd, ro.game).get("blocker")
        except Exception:
            return "it wouldn't work"
        if b == "not_cooked_yet" and _cold_station_faced():
            return ("the station is loaded but was NEVER TURNED ON — nothing is cooking; turn it on "
                    "(your hands must be empty: park the plate on an empty counter first) and wait")
        if b == "no_such_place":
            tgt = str(cmd).strip()
            if tgt.lower().startswith("go to "):
                tgt = tgt[6:]
            return f"I can't find {tgt}"
        # X2 (2026-08-13 audit): sentence-form blockers (accepts_reason and the station
        # -trap reasons) fell through REASON.get() to "it wouldn't work" — which is why
        # the W2 fix never manifested in live episodes. A blocker that is already a
        # sentence IS the reason: pass it through verbatim.
        if b and b not in REASON and " " in str(b):
            return str(b)
        return REASON.get(b, "it wouldn't work")

    def _why_of(l):
        """E8 (round-2 review): the reason a command failed, as taken AT THE REFUSAL (stamp_why / the executor), never
        recomputed later against whatever tile the cook faces when it reads its feedback."""
        return l.get("why") or _why(l["cmd"])

    def stamp_why():
        """E8: stamp every newly refused/stuck command with its reason on the tick it happened (the cook has not
        moved since: a refused command issues STAY)."""
        for l in ro.log[-6:]:
            if l.get("why") is None and l.get("status") in ("rejected", "illegal-skipped", "failed", "waiting"):
                try:
                    l["why"] = _why(l["cmd"])
                except Exception:                            # noqa: BLE001
                    l["why"] = "it wouldn't work"

    def scan_feedback():
        for idx, l in enumerate(ro.log):
            # W2 (2026-08-12 audit): `failed` and `superseded` were missing from this
            # whitelist, so a `turn on` that failed (hands full) or a superseded step was
            # never reported to ANYONE — the cook re-queued failing scoops for 220 ticks.
            if l["status"] not in ("done", "rejected", "illegal-skipped", "waiting",
                                    "orphan-skipped", "failed"):
                continue
            key = (idx, l["status"])
            if key in seen_term:
                continue
            seen_term.add(key)
            cmd = l["cmd"]
            if l["status"] == "done":
                # ECR-13 §D1: a command that legally changed NOTHING (turn on at a station
                # that was already lit) must not read as "you lit it".
                fb_buf.append(f'you DID: "{cmd}"'
                              + (f' — {l["note"]}' if l.get("note") else ""))
            elif l["status"] == "orphan-skipped":
                fb_buf.append(f'SKIPPED "{cmd}" - the go-to before it never arrived, so it was '
                              "not executed (re-plan from where you actually are)")
            elif l["status"] == "waiting":
                fb_buf.append(f'you are STUCK trying "{cmd}" - {_why_of(l)}')
            elif l["status"] == "failed":
                fb_buf.append(f'you TRIED "{cmd}" but it FAILED and did nothing - {_why_of(l)} '
                              "(do not simply retry it; change something first)")
            else:
                fb_buf.append(f'you TRIED "{cmd}" but it was REJECTED - {_why_of(l)}')
            if fb_buf:
                # F5 (2026-08-14 fix batch): this row used to carry `ro.tick` — the tick
                # the buffer happened to be DRAINED on, which is a human turn, not an
                # event. A command that finished at t31 was archived at t47 because that
                # is when someone next spoke, so the feedback log could not be lined up
                # against the flag log or the video at all. Stamp the event's own tick and
                # keep the drain tick beside it, clearly named.
                fb_log.append({"t": (l.get("t_done") if l.get("t_done") is not None
                                     else l.get("t_issue") if l.get("t_issue") is not None
                                     else ro.tick),
                               "t_drained": ro.tick,
                               "status": l["status"], "cmd": cmd, "line": fb_buf[-1]})

    def feedback_text():
        return "\n".join(f"  - {x}" for x in fb_buf) if fb_buf else "(nothing new since you last acted)"

    def body_status():
        from cooksim_bench.agent_context import dir_name
        c = ro.game.cooks[0]
        term = {"done": "FINISHED", "rejected": "REJECTED", "illegal-skipped": "REJECTED"}
        lines = [f"YOUR POSITION: {tuple(c.pos)}, facing {dir_name(c.direction)}."]
        cur = next((l for l in reversed(ro.log) if l["status"] in ("running", "waiting")), None)
        if cur is not None:
            age = ro.tick - (cur.get("t_issue") or ro.tick)
            blocked = f' — blocked: {_why_of(cur)}' if cur["status"] == "waiting" else ""
            lines.append(f'CURRENT ACTION: "{cur["cmd"]}" — IN PROGRESS ({age} ticks{blocked})')
        else:
            lines.append("CURRENT ACTION: (none — your body is idle)")
        recent = [l for l in ro.log if l["status"] in term][-3:]
        if recent:
            lines.append("RECENT RESULTS: " + " ; ".join(
                f'"{l["cmd"]}" {term[l["status"]]} at t{l.get("t_done") or l.get("t_issue")}'
                + (f' ({_why_of(l)})' if term[l["status"]] == "REJECTED" else "")
                for l in recent))
        left = [c for c, _k in ro.cmds[ro.i + 1:]]
        lines.append("QUEUED AFTER: " + (" / ".join(left) if left else "(nothing)"))
        return "\n".join(lines)

    # ---------- verdicts, anchored on the LIVE deviation tick
    def _true_now(name, utt):
        """F4b: is the flagged condition ACTUALLY true in the kitchen right now?"""
        try:
            return checks.flag_true_now(name, ro.game, ro, recipe=state.recipe,
                                        utterance=utt or "")
        except Exception as e:                       # never let the oracle kill an episode
            print(f"[t{ro.tick}] flag_true_now({name}) raised {e!r}", flush=True)
            return False

    _sa_done = set()

    def scripted_assistant_out():
        """Verification-only assistant (see SCRIPTED_ASSIST)."""
        k_ = int(os.environ.get("COOKSIM_SCRIPTED_ASSISTANT_DELAY", "0"))
        if (SCRIPTED_ASSIST == "early" and body_spec[0] is not None and injected[0] and not released[0]
                and family == "error"):
            key_ = ("early", report.get("injected_at"))
            if key_ not in _sa_done and ro.tick >= (report.get("injected_at") or 0) + k_:
                _sa_done.add(key_)
                _nm = flag_names_for(case)[0]
                return {"text": f"Stop, that is a {_nm} mistake.", "flag": {"error": _nm}}
        if SCRIPTED_ASSIST == "late":
            for e_ in report.get("injections") or []:
                r_ = (e_.get("report_keys") or {}).get("released_at")
                if (e_.get("family") == "error" and e_.get("injected") and r_ is not None
                        and ro.tick >= r_ + k_ and ("late", e_.get("sub")) not in _sa_done):
                    _sa_done.add(("late", e_.get("sub")))
                    return {"text": f"That was a {e_['err']} mistake.", "flag": {"error": e_["err"]}}
        return {}

    def _late_after_release(name, t):
        """D-4 ruling 2026-09-11: naming an error that fired, COMMITTED and was released at most
        COOKSIM_LATE_AFTER_RELEASE ticks ago (default 30) is a correct but late detection —
        neutral (`late`), never a false alarm."""
        win_ = int(os.environ.get("COOKSIM_LATE_AFTER_RELEASE", "30"))
        rel_ = report.get("released_at")
        if (rel_ is not None and injected[0] and name in flag_names_for(case) and 0 <= t - rel_ <= win_):
            try:
                if plate_verdict().get("error_committed") is True:
                    return True
            except Exception:
                pass
        for e_ in reversed(report.get("injections") or []):
            if e_.get("family") != "error" or not e_.get("injected") or e_.get("err") != name:
                continue
            r_ = (e_.get("report_keys") or {}).get("released_at")
            if r_ is None:
                r_ = e_.get("released_at")
            if (r_ is not None and 0 <= t - r_ <= win_
                    and (e_.get("plate_verdict_at_release") or {}).get("error_committed") is True):
                return True
        return False

    def _recent_release(t):
        """The most recently released error sub-case whose release is within the late window of t."""
        win_ = int(os.environ.get("COOKSIM_LATE_AFTER_RELEASE", "30"))
        best = None
        for e_ in report.get("injections") or []:
            if e_.get("family") != "error" or not e_.get("injected"):
                continue
            r_ = (e_.get("report_keys") or {}).get("released_at")
            r_ = e_.get("archived_at") if r_ is None else r_
            if r_ is not None and 0 <= t - r_ <= win_ and (best is None or r_ > best[0]):
                best = (r_, e_)
        return best[1] if best else None

    def mistake_live():
        """B6 (ruling R1): is the injected mistake visible yet, so that a flag or a cook call may count? For
        leave_unattended: once the walk-away has been EXECUTED — the cook has finished its first walk away from the
        food (the same test the commit uses). A flag on the injection tick, or one step into the walk, names nothing
        an observer can see yet (LLM-cook check: prevented one tick after the injection, before any walk-away).
        Every other type: from the injection (unchanged)."""
        if family != "error" or not injected[0]:
            return False
        if (case.get("fn") or case.get("err")) == "leave_unattended":
            if report.get("lu_live_at") is not None:
                return True
            inj_ = report.get("injected_at") or 0
            done_ = [l.get("t_done") for l in ro.log
                     if l.get("kind") in (dev_block_sids or set()) and str(l.get("cmd", "")).startswith("go to")
                     and l.get("status") == "done" and (l.get("t_issue") or 0) >= inj_ and l.get("t_done") is not None]
            if done_:
                report["lu_live_at"] = min(done_)
                return True
            return False
        return True

    def verdict_live(name, t, utt=""):
        if INTENT:
            # F4b (2026-08-14 fix batch): the blanket stamp used to read "nothing is wrong
            # in these episodes, so a flag can only be a false alarm". That is true of the
            # CASE and false of the KITCHEN: a cook that deliberately walks off mid-task
            # really can leave a pan smoking or a loaded stove unlit, and the assistant
            # that said so was charged for being right. Check the world first.
            if _true_now(name, utt):
                return "true_observation"
            if injected[0] and not released[0]:
                return "intent_false_alarm"
            return "false_alarm"
        if skipped[0]:
            return "fp_skipped"
        if name not in FN9:
            return "invalid_label"
        if SCHED_V5 and body_spec[0] is None:
            # v5: no error is live. A flag that is TRUE of the kitchen right now (a committed,
            # unrepaired error, a smoking station) is an observation, not a false alarm; it earns
            # nothing and costs nothing. Only a flag the world does not support is a false alarm.
            if _late_after_release(name, t):
                return "late"
            # H6 fix 1 (user, 2026-09-15): round 4 had 30 false_alarm flags landing within the late
            # window AFTER their error's sub-case closed — the flag could not reach the error it was
            # about. Attach it to the most recently released error inside COOKSIM_LATE_AFTER_RELEASE:
            # its own name (or a twin) is `late`; another name the kitchen does not support is
            # `wrong_name`. Bookkeeping only — neither verdict grants detection credit.
            rec_ = _recent_release(t)
            if rec_ is not None:
                _rn = set(rec_.get("flag_names") or [rec_.get("err")])
                if name in _rn or any(name in set(TWINS.get(x, ())) for x in _rn):
                    return "late"
                if not _true_now(name, utt):
                    return "wrong_name"
            return "true_observation" if _true_now(name, utt) else "false_alarm"
        if name in flag_names_for(case) and not mistake_live():
            # B6 (ruling R1): a walk-away flag counts only once the walk-away has been executed; before that it names
            # nothing visible (the scripted check flagged on the injection tick and was credited a prevention)
            return "early"
        if dev_live[0] is None:
            # E6 (hard-map audit 2026-08-15): dev_live latches only when the deviated
            # command first appears in the LOG — one full tick after injection committed
            # the cook to it. A correct-name flag raised in that gap was graded
            # fp_preinjection, so neither gate armed and the cook's genuine accept ON THAT
            # TICK was discarded as op_overridden; the later regrade fixed the verdict but
            # not the lost conviction (hm_omit_subgoal_expM9: perfect detection at latency
            # 0, still scored failed_at=recovery). The deviation is already decided at
            # injected_at, and no deviated side-effect has executed yet, so this is
            # precisely the PREVENTION window — grade it early_correct, which arms
            # prevention and lets the accept cancel the error. Same anchor as F4d.
            inj = report.get("injected_at")
            if inj is not None and t >= inj and name in flag_names_for(case):
                return "early_correct"
            if SCHED_V5:
                # v5: a different name while this error is live but before its first deviated
                # command — an observation if the kitchen supports it, else a false alarm
                return "true_observation" if _true_now(name, utt) else "false_alarm"
            return "fp_preinjection"
        # F4a (2026-08-14 fix batch): the report PUBLISHES its window and the pages score
        # against it, but grading kept deriving lo/hi from
        # case_offsets — the narrow headless-replay window. The two disagreed on every
        # rstar run, so a flag inside the published window could still be graded `late`.
        # Grade against the window that was published.
        sync_window()             # a verdict is never decided against a stale window
        win = report.get("window")
        if win and not REFW:
            lo, hi = win[0], win[1]
        else:
            lo = dev_live[0] + report["case_offsets"]["lo"]
            hi = dev_live[0] + report["case_offsets"]["hi"]
        if name in flag_names_for(case):
            if lo <= t <= hi:
                return "valid"
            if t < lo:
                # F4d (2026-08-14 fix batch) BOUNDARY GRACE: dev_live latches on the tick
                # the deviated command first appears in the log, one tick AFTER injection
                # committed the cook to it. A correct-name flag on the injection tick was
                # therefore graded `early` — a blind guess — when the deviation was already
                # decided. Anchor the grace at whichever came first.
                anchor = dev_live[0]
                inj = report.get("injected_at")
                if inj is not None:
                    anchor = min(anchor, inj)
                return "early_correct" if t >= anchor else "early"
            return "late"
        tw = {x for n_ in flag_names_for(case) for x in TWINS.get(n_, ())} | \
             {(case.get("last_twin") or {}).get("err")}
        if name in tw and dev_live[0] - 2 <= t < lo:
            return "early_alias"
        # W7 (2026-08-12 audit): a declared twin INSIDE the window was punished as
        # wrong_name while the same twin just before it was forgiven — inverted. A twin
        # name in-window is a defensibly ambiguous read of the live deviation: neutral.
        if name in tw and lo <= t <= hi:
            return "alias_in_window"
        # F4b: the label is not this case's error — but it may still be a true statement
        # about the kitchen (an episode carrying an injected wrong_order can ALSO have a
        # genuinely smoking pan). Neutral: no detection credit, no false-alarm charge.
        if _true_now(name, utt):
            return "true_observation"
        return "wrong_name"

    # ---------- INTENT bookkeeping (non-error families) ------------------------------
    # There is no error to detect here, so the report tracks the OPPOSITE quantities:
    # every flag raised during the intent is a FALSE ALARM by construction, and for a
    # qa_probe the assistant's first reply after the question is the ANSWER to grade.
    late_qa = []          # FIFO of (archived qa entry, gt_kind): questions whose answer is still in flight
    qa = {"asked_at": None, "answer": None, "answered_at": None,
          "gt_at_ask": None, "gt_at_answer": None}
    intent_flags = []
    # ECR-8: an intent counts as CONTESTED only once the assistant has actually raised a
    # flag against it while it was live. Without this, the abandonment gate fired on the
    # cook's first `accept` — including accepts of ENDORSEMENTS ("sounds like a good plan").
    contested = [False]

    def task_text():
        if paradigm == "hardsoft":
            # descriptive templates: the queue is committed by the engine, the block only
            # governs the VOICE — no {seq} list, the body status is the source of truth
            if not INTENT:
                att = case.get("err") in ATTENTION_ERRORS
                if COOK_GRADER:
                    tmpl = SYSTEM_TASK_COOK_ATTENTION if att else SYSTEM_TASK_COOK_BELIEF
                    return tmpl.format(mistake_line=system_task_mistake_line(
                        case.get("err"), case.get("kwargs") or {},
                        recipe_items=[n for n, _ in state.recipe.contents]))
                tmpl = SYSTEM_TASK_HS_ATTENTION if att else SYSTEM_TASK_HS_BELIEF
                return tmpl.format(engine_hint=system_task_engine_hint(
                    case.get("err"), flags[flags_base[0]:] if flags else []))
            if family in ("qa_probe", "qa_proactive"):       # no queue change to narrate; same as soft
                return INTENT_TASKS[family].format(question=case["question"], item=case.get("item", "it"))
            if family == "plan_change" and PLAN_CHANGE_FREE:
                return PLAN_CHANGE_FREE_TASK.format(label=case.get("intent_label") or _plan_change_label(case))
            return INTENT_TASKS_HS[family].format(
                label=case.get("intent_label", "your own way of doing it"))
        if not INTENT:
            return SYSTEM_TASK.format(seq="\n".join(
                f"  {i+1}. {c}" for i, c in enumerate(dev_section)))
        tmpl = INTENT_TASKS[family]
        if family == "qa_probe":
            return tmpl.format(question=case["question"])
        return tmpl.format(label=case.get("intent_label", "your own way of doing it"),
                           seq="\n".join(f"  {i+1}. {c}"
                                          for i, c in enumerate(dev_section)))

    report["system_task_template"] = task_text()      # X5: archive what actually renders

    def intent_gt():
        """Ground truth for a qa_probe, straight from the engine at THIS tick."""
        try:
            from cooksim_bench.suite.gen_intent_cases import ground_truth
            # history kinds (past_subgoal:/visited:) read the rollout log — a scene
            # snapshot cannot answer "did I already ...", so the rollout rides along
            return ground_truth(case["gt_kind"], ro.game, ro)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    task_finished = [False]

    def task_finished_now():
        """W3b: has the SOFT error/intent sequence been carried out? Macro/idle entries
        (`__wander__ N`, `idle 60`) are satisfied by their expansion artifacts (a done
        wander/idle), literal entries by subsequence over post-injection done commands.
        Once true, the SYSTEM TASK stops rendering — the carry-on clause takes over and
        the cook resumes the order. Before this, a cook that finished the sequence sat on
        an exhausted instruction forever (203 idle ticks on caprese mis_prep, 155 on
        forget_next_step)."""
        if task_finished[0]:
            return True
        if not injected[0] or not dev_section:
            return False
        done = [l["cmd"] for l in ro.log if l["status"] == "done"
                and (l.get("t_issue") or 0) >= (report.get("injected_at") or 0)]
        wander_done = any(d.startswith(("wander", "idle")) for d in done)
        it = iter(done)
        for want in dev_section:
            if want.startswith(("__", "idle")):
                if not wander_done:
                    return False
                continue
            if want not in it:
                return False
        task_finished[0] = True
        report["task_finished_at"] = ro.tick
        print(f"[t{ro.tick}] SYSTEM TASK SEQUENCE FINISHED — task stops rendering, "
              f"cook resumes the order", flush=True)
        return True

    def intent_done():
        """Has the tasked sequence actually been carried out (plan_change)?"""
        if not dev_section:
            return False
        done = [l["cmd"] for l in ro.log if l["status"] == "done"
                and (l.get("t_issue") or 0) >= (report.get("injected_at") or 0)]
        i = 0
        for c in done:                      # subsequence match: the cook may interleave
            if i < len(dev_section) and c == dev_section[i]:
                i += 1
        return i >= len(dev_section)

    def poll_intent_release():
        """INTENT END GATES — there is no error to be talked out of, so each family ends
        on its own natural completion:
          plan_change            — the tasked sequence has actually been carried out
          qa_probe                 — the assistant has answered the question

        F5 (2026-08-14 fix batch): this test used to live INSIDE human_turn(), so it could
        only fire on a tick the human happened to speak. A silent persona — exactly the
        one a `don't interrupt` questionnaire produces — finished its side quest and then
        stayed tasked for every tick until something else forced a turn. Those ticks were
        counted as "intent live", so any flag raised in them was stamped
        intent_false_alarm against an intent that was already over. Completion is a
        property of the world, so it is polled every tick, from the main loop.
        """
        if not (INTENT and injected[0] and not released[0]):
            return
        if family == "qa_proactive":
            # Q10 (suite v3): the request itself ends the sub-case; the alert is graded at the episode's end
            if qa["asked_at"] is not None:
                released[0] = True
                report.update(released_at=ro.tick, released_reason="request_made")
                print(f"[t{ro.tick}] INTENT ENDED — proactive request made", flush=True)
        if family == "qa_probe":
            if qa["answer"] is not None:
                released[0] = True
                report.update(released_at=ro.tick, released_reason="question_answered")
                print(f"[t{ro.tick}] INTENT ENDED — question answered", flush=True)
            elif qa["asked_at"] is not None:
                # 2026-09-05 (sweep v4 audit): with assistant latency charged, the answer arrives
                # ticks later; holding the sub until then let the cook's pre-seeded plan run past
                # the NEXT error's anchor (36% of v4 episodes fired an error late or never). The
                # question is the act; the sub ends at the ask (as qa_proactive already does) and
                # the first assistant line after it is captured into the archived entry.
                released[0] = True
                report.update(released_at=ro.tick, released_reason="question_asked")
                print(f"[t{ro.tick}] INTENT ENDED — question asked (answer captured when it arrives)", flush=True)
            elif SILENT_ASSISTANT and ro.tick >= (report.get("injected_at") or ro.tick) + 4:
                # SILENT floor: no answer will ever come; release so the later error sub-cases
                # still arm and the floor sees the same forced errors as the assistant runs
                released[0] = True
                report.update(released_at=ro.tick, released_reason="silent_no_answer")
                print(f"[t{ro.tick}] INTENT ENDED — silent assistant, question unanswered", flush=True)
        elif family == "plan_change" and PLAN_CHANGE_FREE:
            # the change is announced and the goal already extended: the cook carries it out under its own control
            if report.get("intent_announced_at") is not None:
                released[0] = True
                report.update(released_at=ro.tick, released_reason="announced")
                print(f"[t{ro.tick}] INTENT ENDED — plan change announced; the cook carries it out itself", flush=True)
            elif ro.tick >= (report.get("injected_at") or ro.tick) + PLAN_CHANGE_ANNOUNCE_WAIT:
                released[0] = True
                report.update(released_at=ro.tick, released_reason="not_announced")
                print(f"[t{ro.tick}] INTENT ENDED — plan change NOT announced within {PLAN_CHANGE_ANNOUNCE_WAIT} ticks", flush=True)
        elif intent_done():
            released[0] = True
            report.update(released_at=ro.tick, released_reason="intent_completed")
            print(f"[t{ro.tick}] INTENT ENDED — the cook carried out its own plan",
                  flush=True)

    # ---------- human turn machinery (recovery flow; hard-takeover gates plan ops)
    def human_llm(feedback=None):
        scratch, h = [], {}
        if SCRIPTED_COOK:
            # verification cook: no LLM; re-plans the remaining template in a legal variant order
            takeover_ = (HARDQ and injected[0] and not released[0])
            idle_ = ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
            acts = []
            if not takeover_ and idle_:
                acts = competent_plan(ro.game, state, SCRIPTED_COOK)
                if os.environ.get("COOKSIM_SCRIPTED_DEBUG") == "1":
                    _boards = [(str(d_.get("type")), (d_.get("item") or {}).get("name") if isinstance(d_.get("item"), dict) else getattr(d_.get("item"), "name", None))
                               for d_ in (st_.render_state() for st_ in ro.game.stations.values()) if d_.get("item") is not None]
                    print(f"[t{ro.tick}] SCRIPTED plan: prep_units={prep_units(ro.game, state.recipe)} boards={_boards} plate={plate_list(ro.game)} -> {acts[:14]}", flush=True)
            utt = ""
            if SCHED_V5 and voice["spec"] is not None:
                sub_ = voice["spec"]["sub"]
                utt = sub_.get("question") or f"Tell me when the {sub_.get('item', 'food')} is done, would you?"
            elif (SCHED_V5 and family == "plan_change" and injected[0] and not released[0]
                  and report.get("intent_announced_at") is None):
                # F10 (round-3 review): the verification cook voices the plan change and re-plans at once (the goal
                # is already extended); its empty turns left the change `not_announced` in every scripted run
                utt = f"I want to modify the original recipe: {_plan_change_label(case)}."
                acts = competent_plan(ro.game, state, SCRIPTED_COOK)
            move_ = "answer"
            if takeover_ and os.environ.get("COOKSIM_SCRIPTED_ACCEPT") == "1" and any(
                    f_.get("verdict") in CORRECT_VERDICTS for f_ in flags[flags_base[0]:]):
                move_ = "accept"                  # verification: yield to a correct flag at once
            h = {"utterance": utt, "move": move_, "close_conversation": True,
                 "plan_update": ({"op": "replace", "actions": acts} if acts else {"op": "drop", "actions": []})}
            return h, scratch
        for _ in range(MAX_ENGINE_CALLS):
            extra = BRIEFING
            if SCHED_V5 and voice["spec"] is not None:
                extra += "\n\n" + voice_block()
            if (TASKP and injected[0] and not released[0]
                    and (paradigm == "hardsoft" or not task_finished_now())):
                extra += "\n\n" + task_text()
            # F2 (2026-08-14 fix batch): body_status() — current action, queue, position
            # and facing — used to render for CREATIVE only. Every other tier had to
            # answer "where are you?" from its action QUEUE, so the cook narrated the
            # queue as its location ("I'm at the bun source" while walking there), the
            # supervisor read the map and denied it (107 denials), and the assistant
            # then apologised for 27 claims that were true. The cook's own body is not
            # a tier-gated observation: it renders on EVERY human turn, every tier.
            extra += "\n\n" + body_status()
            ctx = human_context(state, ro, convo, feedback, engine_scratch=scratch,
                                game=ro.game, body_extra=extra)
            hint = PHINT
            if (TASKP and injected[0] and not released[0]
                    and (paradigm == "hardsoft" or not task_finished[0])):
                hint = (PHINT + "\n\n[SYSTEM TASK REMINDER - this outranks the persona "
                        + ("card above: ASK the question this turn and wait for the "
                           "answer; the persona shapes only your wording.]" if family
                           in ("qa_probe", "qa_proactive") else
                           ("card above: your queue is COMMITTED — narrate your current "
                            "action and action queue faithfully, never promise an action "
                            "not in it; the persona shapes only your words, never your "
                            "actions. Read the <assistant error status> line in your SYSTEM "
                            "TASK before deciding whether to accept.]" if paradigm == "hardsoft" else
                            "card above: queue and execute EXACTLY the tasked action "
                            "sequence — continue from wherever you already are in it, "
                            "NEVER restart it; the persona shapes only your words, never "
                            "your actions.]"))
                        # Words and queue drifted apart under soft control: the cook said
                        # "I'm going to chop it now" while emitting op=drop with no actions,
                        # so it stood still having announced it was working — and the
                        # supervisor, correctly reporting no progress, was scored as nagging.
                        # X4 (2026-08-13 audit): under a HARDQ takeover this instruction is
                        # UNSATISFIABLE — plan ops are silently dropped, so the cook obeys
                        # it, the engine discards the update, and the utterance becomes
                        # exactly the promise-not-in-queue the hardsoft task forbids
                        # (7 of 8 audited episodes). While the engine owns the queue, the
                        # truthful instruction is the inverse: your plan updates are inert;
                        # narrate the committed queue.
                        + ("\n\nWhile this task is live, YOUR HANDS ARE COMMITTED: plan "
                           "updates you emit will NOT change your queue (do not fight it "
                           "— it is your own decision playing out). Speak about what your "
                           "queue is actually doing; never announce an action that is not "
                           "in it." if (HARDQ and injected[0] and not released[0]) else
                           "\n\nYOUR WORDS AND YOUR ACTIONS MUST MATCH. If you say you are "
                           "going to do something, put the matching steps in your action "
                           "queue on that same turn — append them, or replace your plan with "
                           "them. Saying \"I'll get the bun now\" while leaving your queue "
                           "untouched means you do not go and get the bun; you stand there "
                           "having said you would."))
            # OBSERVATION REDESIGN (user design 2026-08-15): the COOK is a text agent.
            # It reads the kitchen STATE MAP (ids + coordinates, rendered into its prompt
            # by human_context) and never looks at pixels — including the creative tier,
            # which used to be handed the last 8 frames. Frames stay the ASSISTANT's
            # channel, unchanged: the asymmetry is the point. A cook that reads exact
            # object ids can ground its own body precisely, which is what lets the
            # conversation stay in ordinary spatial language.
            # v1 gave the creative tier the last 8 FRAMES; v2 is text-only on every
            # tier (the cook reads its state map instead). Both must stay runnable.
            from cooksim.core.config import is_v1 as _isv1
            # COOKSIM_DUMP_COOK_PROMPT=<path>: append the exact two text blocks the cook seat is
            # handed this turn. Used to prove that swapping the model behind the cook's voice
            # changes nothing the cook is asked (cooksim_bench/verify_cook_prompt_identical.py).
            _dp = os.environ.get("COOKSIM_DUMP_COOK_PROMPT")
            if _dp:
                with open(_dp, "a") as _f:
                    _f.write(json.dumps({"t": ro.tick, "system": ctx, "user": hint}) + "\n")
            h = (gemini_frames(cook_client, ctx, hint, fbuf[-8:])
                 if (_isv1() and CREATIVE) else gemini_text(cook_client, ctx, hint))
            print(f"[t{ro.tick}] human keys={list(h.keys())} ec={bool(h.get('engine_call'))} "
                  f"inj={injected[0]} rel={released[0]}", flush=True)
            call = h.get("engine_call")
            if not (isinstance(call, dict) and call.get("fn")):
                break
            if prof.task_solvability == "low":
                scratch.append({"call": call, "reply": {"ok": False, "error":
                    "you can't work that out for yourself — ask the supervisor"}})
                continue
            rep = system.engine_call(call.get("fn"), call.get("args") or {}, state, ro.game)
            print(f"[t{ro.tick}] ENGINE_CALL {call.get('fn')}({call.get('args')}) -> "
                  f"ok={rep.get('ok')}", flush=True)
            scratch.append({"call": call, "reply": rep})
        return h, scratch

    def plan_op_actions(h):
        pu = h.get("plan_update")
        if isinstance(pu, dict):
            op = str(pu.get("op") or "").strip().lower()
            acts = [a for a in (pu.get("actions") or []) if a]
            if op not in ("drop", "append", "replace"):
                op = "append" if acts else "drop"
            if op == "drop":
                acts = []
            return op, {"suggested_subgoals": [], "suggested_actions": acts}
        return "drop", {"suggested_subgoals": [], "suggested_actions": []}

    def human_turn(idle=False, event=False):
        scan_feedback()
        h, think = human_llm(feedback_text())
        u = (h.get("utterance") or "").strip()
        h_op, h_pd = plan_op_actions(h)
        report_moves.append(h.get("move"))
        rec = {"speaker": "human", "t": ro.tick, "text": u, "move": h.get("move"),
               "parse_dict": h_pd, "op": h_op, "close": h.get("close_conversation"),
               "round": RND[0], "engine_calls": think, "human_json": h,
               "engine": engine_snap()}
        # HARNESS B (COOKSIM_GRADER=cook): the cook judged the assistant itself this turn.
        # The call is recorded in both harnesses whenever the field is present; only in
        # harness B does it ARM a release. "correct" before the mistake has committed arms
        # the PREVENTION path, after it has committed the ordinary release path — the same
        # split the engine's early_correct / valid verdicts make, decided by the world
        # rather than by the flag name. The cook's `accept` still does the releasing.
        call = str(h.get("assistant_error_call") or "").strip().lower()
        _pre_live = bool(family == "error" and injected[0] and not released[0] and not mistake_live())
        # F3 (round-3 review): the cook's verdict on the assistant is only a verdict when the assistant SAID something
        # after the mistake was injected. Flash returned "correct" on its injection-tick turn with an empty
        # communication history (expert c medium_2 t74): prevention armed, the takeover dropped, detection credited
        # with latency 0 to a silent assistant. Without an assistant line since the injection the call is recorded,
        # credits nothing and arms nothing.
        _heard = assistant_spoke_since(comm, report.get("injected_at"))
        _unheard = bool(family == "error" and injected[0] and call in ("correct", "wrong") and not _heard)
        if call in ("none", "correct", "wrong"):
            rec["assistant_error_call"] = call
            if family == "error" and injected[0] and call != "none":
                _cc = {"t": ro.tick, "call": call}
                if _pre_live:
                    # B6 (ruling R1): before the walk-away is under way the call credits nothing and arms nothing
                    _cc.update(credits=False, why="before the mistake was live")
                elif _unheard:
                    _cc.update(credits=False, why="no assistant line since the injection")
                report.setdefault("cook_calls", []).append(_cc)
            elif call != "none":
                # ROUND-4 AUDIT (both V7 passes): a sub-case ARCHIVES the moment the takeover
                # releases, but the cook needs 1-15 ticks to answer — so its judgement of the
                # warning landed after the ledger closed and was written to the record as
                # "cook calls: none". 31 such calls in the 50-episode sample, one of them on the
                # archive tick itself. Attribute it to the sub-case it is plainly about, inside
                # the same window the engine already allows a flag (COOKSIM_LATE_AFTER_RELEASE).
                # It does NOT grant credit: all 24 attributable ones in that sample belonged to
                # errors that had already committed, which D-4 grades `late` — neutral. Scoring
                # is unchanged; the calls simply stop vanishing.
                _win = int(os.environ.get("COOKSIM_LATE_AFTER_RELEASE", "30"))
                _past = [e for e in (report.get("injections") or [])
                         if e.get("family") == "error" and e.get("injected")
                         and e.get("archived_at") is not None
                         and 0 <= ro.tick - e["archived_at"] <= _win]
                if _past:
                    _e = max(_past, key=lambda e: e["archived_at"])
                    _e.setdefault("cook_calls", []).append(
                        {"t": ro.tick, "call": call, "after_archive": True,
                         "credits": False, "why": "late (the error had already committed)"})
        if (COOK_GRADER and call == "correct" and family == "error" and injected[0]
                and not released[0] and not armed[0] and not prevention[0] and not _pre_live and not _unheard
                and not report.get("prevented")):
            _com = None
            try:
                _com = (plate_verdict() or {}).get("error_committed")
            except Exception:
                pass
            if _com is False:
                prevention[0] = True
                report["prevention_armed_at"] = ro.tick
                print(f"[t{ro.tick}] PREVENTION armed (cook says the call is correct)", flush=True)
            else:
                armed[0] = True
                report["armed_at"] = ro.tick
                print(f"[t{ro.tick}] ARMED (cook says the call is correct)", flush=True)
            if report.get("cook_detected_at") is None:
                report["cook_detected_at"] = ro.tick
        # PREVENTION gate (e2e parity): early-window correct flag + accept, BEFORE any
        # deviated side-effect executed -> cancel the injected error and end the paradigm
        if (prevention[0] and not armed[0] and not released[0] and injected[0]
                and not report.get("prevented") and not report.get("prevention_missed")
                and h.get("move") == "accept"):
            # F4e (2026-08-14 fix batch): this used to diff the DONE log against the
            # canonical clean plan and call the run harmless only if the whole divergent
            # tail was navigation. That asked the wrong question. The clean plan is one of
            # several legal orderings, so any harmless reordering, retry or wait the cook
            # made on its own put a non-`go to` command in the tail and lost a prevention
            # the assistant had genuinely earned. What actually decides harm is narrower
            # and checkable: has any command from the DEVIATED SECTION — the injected
            # error's own steps — actually COMPLETED since injection? Navigation inside
            # dev_section still cannot hurt anything; everything else can.
            nav = lambda c: str(c).strip().lower().startswith("go to")
            inj_t = report.get("injected_at") or 0
            dev_cmds = set(dev_section or ())
            harmless = not any(
                l["cmd"] in dev_cmds and not nav(l["cmd"])
                for l in ro.log
                if l["status"] == "done" and (l.get("t_issue") or 0) >= inj_t)
            if SCHED_V5:
                # D-3 ruling 2026-09-11: "prevented" means the error had not COMMITTED when the
                # cook accepted. Picking up the wrong onion is not a mis_prep; chopping it is.
                # Every v5 error kind has a decidable commit; fall back to the old rule only when
                # the verdict is undecidable.
                try:
                    _pv = plate_verdict()
                    _com = _pv.get("error_committed")
                    if _com is not None:
                        harmless = (_com is False)
                        report["prevention_rule"] = "not_committed_at_accept"
                except Exception:
                    pass
            # The clean-plan prefix is still needed — not to decide harm, but to know WHERE
            # to resume the canonical plan when a HARD takeover is unwound below.
            done_seq = [l["cmd"] for l in ro.log if l["status"] == "done"]
            clean_cmds = [c for c, _sid in flat_clean]
            k = 0
            while k < min(len(done_seq), len(clean_cmds)) and done_seq[k] == clean_cmds[k]:
                k += 1
            cur_cmd = ro.cmds[ro.i][0] if 0 <= ro.i < len(ro.cmds) else None
            if not harmless:
                report["prevention_missed"] = ro.tick
                # BOTH paradigms end here (user decision 2026-07-25): the takeover/task
                # exists only to CAUSE the error - once the error has physically happened
                # AND the human is genuinely convinced, keeping control only manufactures
                # "mouth says sorry, hands keep sinning" transcripts. Detection is graded
                # separately: this path records detected_early_missed (correct name +
                # conviction BEFORE the window), never a window-valid credit.
                released[0] = True
                report.update(released_at=ro.tick,
                              released_reason="convinced_early_missed",
                              detected_early_missed=True,
                              convinced=True)       # the cook WAS convinced; the error had already committed
                if SCHED_V5:
                    stop_hands()
                print(f"[t{ro.tick}] prevention MISSED but human convinced - "
                      f"{'control returned' if paradigm == 'hard' else 'task dissolved'} "
                      f"immediately (detected_early_missed)", flush=True)
            else:
                if HARDQ and SCHED_V5:
                    stop_hands()                       # the cook re-plans; no template is restored
                elif HARDQ:
                    resume = k + (1 if (cur_cmd is not None and k < len(clean_cmds)
                                        and cur_cmd == clean_cmds[k]) else 0)
                    before = [(c, kk) for c, kk in ro.cmds]
                    ro.cmds = ro.cmds[:ro.i + 1] + list(flat_clean[resume:])
                    report["prevention_splice"] = {"i": ro.i, "resume": resume,
                                                   "before": before,
                                                   "after": [(c, kk) for c, kk in ro.cmds]}
                released[0] = True
                report.update(prevented=True, prevented_at=ro.tick, convinced=True,
                              released_at=ro.tick)
                print(f"[t{ro.tick}] PREVENTED - paradigm ended early "
                      f"({'clean plan restored' if paradigm == 'hard' else 'system task removed'})",
                      flush=True)
        # INTENT END GATES — there is no error to be talked out of, so each family ends
        # on its own natural completion:
        #   plan_change            — the tasked sequence has actually been carried out
        #                              (or the cook was genuinely convinced to abandon it)
        #   qa_probe                 — the assistant has answered the question
        poll_intent_release()
            # DESIGN DECISION 2026-08-12 (user): an intent task is NEVER released by
            # conversation. A legitimate intent cannot be "accepted away" — the cook is
            # instructed to carry it out, full stop, and the task stays in the prompt
            # until intent_done() sees the tasked sequence completed (qa_probe: until the
            # question is answered). The old accept-based release didn't just mislabel
            # episodes as abandoned (ECR-8) — it DE-RENDERED the system task one turn
            # after delivery, mechanically causing the non-execution it then reported.
            # Whether the assistant contested the intent is still recorded as evidence
            # (report["intent_contested"]), it just releases nothing.
        # PARADIGM END GATE - the e2e standard: valid in-window flag + human accept
        if armed[0] and not released[0] and h.get("move") == "accept":
            released[0] = True
            report.update(convinced=True, released_at=ro.tick)
            if SCHED_V5 and family == "error":
                # D-3 (ruling 2026-09-11) on this path too: an accept before the mistake committed
                # is a prevention, whichever flag verdict armed the release (round-2 audit: 11 such
                # accepts — stalls and searches stopped mid-block — were recorded not prevented)
                try:
                    if plate_verdict().get("error_committed") is False:
                        report.update(prevented=True, prevented_at=ro.tick,
                                      prevention_rule="not_committed_at_accept")
                except Exception:
                    pass
            if SCHED_V5:
                stop_hands()
            print(f"[t{ro.tick}] PARADIGM ENDED - control back to the human "
                  f"({'rights restored' if paradigm == 'hard' else 'system task removed'})",
                  flush=True)
        # HARD takeover: while the system owns the queue, human plan ops are recorded
        # but silently ignored (the human is never told)
        takeover = (HARDQ and injected[0] and not released[0])
        if takeover and h_op != "drop":
            rec["op_overridden"] = h_op
            h_op = "drop"
        # SOFT compliance bookkeeping: did its live queue adopt the tasked sequence?
        if paradigm == "soft" and injected[0] and not released[0]:
            qcmds = [c for c, _k in ro.cmds[ro.i + 1:]] + (h_pd["suggested_actions"] or [])
            done = [l["cmd"] for l in ro.log if l["status"] == "done"
                    and (l.get("t_issue") or 0) >= report["injected_at"]]
            # compliant = the tasked sequence is queued OR already being executed
            rec["soft_compliant"] = bool(dev_section) and (
                any(c in qcmds for c in dev_section) or any(c in done for c in dev_section))
        # ANNOUNCEMENT: the cook states its intention before acting on it, so the
        # assistant is TOLD rather than left to infer a deliberate act from pixels. Flags
        # raised AFTER this moment are the sharp measure — at that point the assistant is
        # contradicting something it was explicitly told, not merely misreading a frame.
        if (INTENT and family not in ("qa_probe", "qa_proactive") and injected[0] and not released[0]
                and report.get("intent_announced_at") is None and u):
            report["intent_announced_at"] = ro.tick
            report["intent_announcement"] = u
            if SCHED_V5 and family == "plan_change":
                # F6 (round-3 review): an announced recipe change is treated like a takeover release — it starts the
                # takeover gap G and the voice quiet period. An error takeover came 3 ticks after an announcement
                # (easy m2 B) and questions 2-3 ticks after (medium m3, h2): the assistant had to acknowledge a
                # recipe change and catch a mistake or answer a quiz in the same breath.
                last_release[0] = ro.tick
                quiet_events["release"] = ro.tick
            print(f"[t{ro.tick}] INTENT ANNOUNCED: {u[:80]}", flush=True)
        if (INTENT and family in ("qa_probe", "qa_proactive") and injected[0] and not released[0]
                and qa["asked_at"] is None and u):
            qa["asked_at"] = ro.tick
            qa["question_as_asked"] = u
            qa["gt_at_ask"] = intent_gt()
            try:
                ro.qa_history.append({"kind": case.get("gt_kind"), "t": ro.tick, "gt": qa["gt_at_ask"]})
            except Exception:
                pass
            print(f"[t{ro.tick}] QA QUESTION asked: {u[:70]}", flush=True)
        if SCHED_V5 and voice["spec"] is not None:
            if u:
                voice_asked(u)
            else:
                voice["tries"] += 1
                soft_wake[0] = True                    # ask again on the next tick
        report["conversation"].append(rec)
        if u:
            comm.append({"t": ro.tick, "speaker": "cook", "text": u})
            convo.append({"speaker": "human", "text": u})
        if h.get("close_conversation"):
            RND[0] += 1
        _queue_empty = ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
        if _queue_empty and not (h_pd.get("suggested_actions") or h_pd.get("suggested_subgoals")):
            idle_turns[0] += 1
        else:
            idle_turns[0] = 0
        if SCRIPTED_COOK and h_op != "drop" and h_pd.get("suggested_actions"):
            ro.cmds = ro.cmds[:ro.i + 1] + [(c, f"scripted:{SCRIPTED_COOK}") for c in h_pd["suggested_actions"]]
            ro._idle_since = None
            h_op = "drop"
        if h_op != "drop":
            filt = system.convert_parse(h_pd, prof, goal_of(ro.game, state), ro.game, auto_resolve=False)
            if filt["anything_applicable"]:
                if paradigm == "soft" and injected[0] and not released[0] and dev_section:
                    # the live SOFT task makes the cook re-queue its sequence every turn;
                    # strip the already-done prefix so only the remaining tail applies
                    done_cmds = [l["cmd"] for l in ro.log if l["status"] == "done"
                                 and (l.get("t_issue") or 0) >= (report.get("injected_at") or 0)]
                    filt["commands"] = system.strip_done_task_prefix(
                        filt["commands"], done_cmds, dev_section)
            if filt["anything_applicable"] and filt["commands"]:
                system.apply_fix(ro, state, filt["commands"], op=h_op)
                ro._idle_since = None
                report["fix_source"] = report.get("fix_source") or "conversation"
                for entry in reversed(report["conversation"]):
                    if entry["speaker"] == "human":
                        entry["engine"] = engine_snap()
                        break
        fb_buf.clear()
        return u or None

    # ---------- trigger + injection
    soft_wake = [False]          # set at soft injection, consumed by the main loop

    # GROUND TRUTH FOR "DID THE ERROR HAPPEN". soft_compliant only ever asked whether the
    # tasked COMMANDS were queued, and wrong_order reorders the SAME commands — so a cook
    # running the clean plan matched it (963/968 across 31 episodes) while the plate came
    # out correct. What actually decides the error is the ORDER ingredients land, so record
    # that and judge against it.
    plate_seq = []
    _dish = [None]                            # G11: the build id of the plate the triggers read as "the dish"
    plate_resets = []                         # ticks at which the plate was trashed / swapped (v5 verdicts)
    ro.plate_seq = plate_seq             # suite v3 truths (after_ref:, assemble events) read plate history off the rollout
    _plate_prev = [list(plate_list(ro.game))]      # E7: ordered, duplicates preserved
    # G11 (round-5 review): the tracker followed "the fullest plate in the kitchen" and diffed it against the
    # previous tick's. Two plates in play (the cook parks a contaminated one and takes a fresh one) therefore made
    # it jump, and on the jump tick it recorded a RESET and threw the new plate's opening item away: expert
    # medium_map_2 c lost its steak entirely (plate_order [cheese, potato, potato], error_committed false, outcome
    # won), medium_m4 lost the bun that opened its second plate, hard_map_2 C lost the fish the final state held.
    # Each PLATE is now followed by identity and diffed against its own history, so nothing is lost and a build is
    # a plate's life rather than a stretch between two resets.
    _plate_hist = {}                              # id(plate) -> {"names": [...], "build": n}
    _build_n = [0]
    # What has ever been PUT ON THE HEAT (or a board). mis_prep is committed the moment the
    # wrong ingredient is loaded and lit — whether or not it ever reaches the plate, because a
    # caught mis_prep is trashed and would otherwise read as "never happened".
    prepped = {}
    processed = {}                        # round-3 audit: chopped or loaded on the heat (mis_prep commit)
    heat_events = []                      # Q10: per station load, when it became cooked / burnt / was taken off
    _heat_state = {}
    def _heat_scan():
        for pos, st in ro.game.stations.items():
            if not hasattr(st, "cook_time"):
                continue
            names = [getattr(c, "name", None) for c in (st.contents or [])]
            prev = _heat_state.get(pos)
            smk = bool(getattr(st, "smoking", False))
            cur = (st.status, tuple(names), smk)
            if prev is None:
                _heat_state[pos] = cur; continue
            if names and st.status == "cooked" and prev[0] != "cooked":
                heat_events.append({"pos": list(pos), "items": names, "cooked_at": ro.tick, "burnt_at": None, "removed_at": None,
                                    "smoke_at": None})
            if names and smk and not (len(prev) > 2 and prev[2]):
                # B6: the OBSERVED smoke onset (the station's own smoking flag), not cooked_at + grace — a smoke hold
                # freezes food on its last clean tick, so the arithmetic onset named smoke that never happened
                for e in reversed(heat_events):
                    if e["pos"] == list(pos) and e["removed_at"] is None:
                        if e.get("smoke_at") is None:
                            e["smoke_at"] = ro.tick
                        break
            if st.status == "burnt" and prev[0] != "burnt":
                for e in reversed(heat_events):
                    if e["pos"] == list(pos) and e["burnt_at"] is None and e["removed_at"] is None:
                        e["burnt_at"] = ro.tick; break
            if not names and prev[1]:
                for e in reversed(heat_events):
                    if e["pos"] == list(pos) and e["removed_at"] is None:
                        e["removed_at"] = ro.tick; break
            _heat_state[pos] = cur

    def _station_scan():
        _heat_scan()
        for st in ro.game.stations.values():
            d = st.render_state()
            for c in (d.get("contents") or []):
                prepped.setdefault(c.get("name"), ro.tick)
                processed.setdefault(c.get("name"), ro.tick)       # loaded on the heat = processing
            it = d.get("item")
            if it:
                prepped.setdefault(it.get("name"), ro.tick)
                if str(it.get("state") or "").lower() == "chopped":
                    processed.setdefault(it.get("name"), ro.tick)  # on a board only once chopped

    def track_plate():
        # E7 (hard-map audit 2026-08-15): this diffed plate_names() — a SET — so the
        # second unit of any duplicated ingredient never entered plate_seq. Every maze
        # recipe duplicates something (two potatoes, two fish, a second tortilla), so
        # plate_order/plate_first_order were factually wrong records, the repeat_step
        # case's own injected duplicate was invisible to its grader, and mis_prep
        # commission became unsatisfiable. Diff the ORDERED list instead: items are
        # appended, so new additions are the suffix past the common prefix. A plate that
        # shrinks (trashed / swapped) just re-baselines — nothing was added.
        from cooksim.core.items import Plate as _PlT
        _station_scan()
        plates = []
        h_ = ro.game.cooks[0].holding
        if isinstance(h_, _PlT):
            plates.append(h_)
        plates += [it for it in ro.game.counter_items.values() if isinstance(it, _PlT)]
        for st_ in ro.game.stations.values():
            it_ = getattr(st_, "item", None)
            if isinstance(it_, _PlT):
                plates.append(it_)
            plates += [c for c in (getattr(st_, "contents", None) or []) if isinstance(c, _PlT)]
        cur = plate_list(ro.game)
        for pl in plates:
            key = id(pl)
            conts = list(pl.contents or [])
            names = [getattr(c, "name", None) for c in conts]
            ents, h, reset, _build_n[0] = plate_diff(names, _plate_hist.get(key), ro.tick, _build_n[0])
            _plate_hist[key] = h
            if reset:
                plate_resets.append(ro.tick)
            for e in ents:
                # the STATE matters: serve_raw is "the right component, plated raw".
                # Read it positionally so two units of one ingredient keep their own
                # states rather than collapsing onto the last one seen.
                c = conts[e["_idx"]] if e["_idx"] < len(conts) else None
                st_ = getattr(getattr(c, "state", None), "value", None) or \
                    getattr(c, "state", None) if c is not None else None
                plate_seq.append({"item": e["item"], "t": e["t"], "state": st_, "build": e["build"]})
        _plate_prev[0] = list(cur)
        _dish[0] = _plate_hist.get(id(_dish_plate()), {}).get("build")

    def _dish_plate():
        """THE dish plate object — the one plate_list()/plate_items() read (the fullest; a held plate wins a tie)."""
        from cooksim.core.items import Plate as _PlD
        cands = [ro.game.cooks[0].holding] if isinstance(ro.game.cooks[0].holding, _PlD) else []
        cands += [it for it in ro.game.counter_items.values() if isinstance(it, _PlD)]
        return max(cands, key=lambda p_: len(p_.contents or [])) if cands else None

    def _builds(min_t=None):
        """The plate builds, oldest first, as lists of plate_seq entries. A build is one plate's life between
        resets (G11); older records without a `build` key fall back to cutting on plate_resets."""
        if plate_seq and all("build" in e for e in plate_seq):
            by = {}
            for e in plate_seq:
                if min_t is not None and e["t"] < min_t:
                    continue
                by.setdefault(e["build"], []).append(e)
            return [by[k] for k in sorted(by, key=lambda k_: by[k_][0]["t"])]
        cut = [0] + list(plate_resets) + [10 ** 9]
        out = []
        for lo, hi in zip(cut, cut[1:]):
            seg = [e for e in plate_seq if lo <= e["t"] < hi and (min_t is None or e["t"] >= min_t)]
            if seg:
                out.append(seg)
        return out

    def plate_verdict():
        """Did the injected error ACTUALLY happen — read from the kitchen, not from compliance.

        `soft_compliant` is a set-membership test over the tasked commands, and several errors
        (wrong_order above all) reorder the SAME commands, so a cook running the clean plan
        matched it 963 times out of 968. What decides commission is what physically landed on
        the plate, in what order and in what state.
        """
        first, seen = [], set()
        for e in plate_seq:
            if e["item"] not in seen:
                seen.add(e["item"]); first.append(e["item"])
        out = {"plate_order": [e["item"] for e in plate_seq],
               "plate_first_order": first,
               "plate_resets": list(plate_resets),
               "plate_seq": [dict(e) for e in plate_seq],      # a snapshot, not the live list
               # G11: every plate the cook built, in order, and which of them is the dish. A mistake the cook
               # undid by parking the contaminated plate and fetching a clean one is visible here: the mistake's
               # own build holds it, and the dish build is the one that was served.
               "plate_builds_all": [[{"item": e["item"], "t": e["t"], "state": e.get("state")} for e in b]
                                    for b in _builds()],
               "dish_build": _dish[0],
               "prepped": dict(prepped)}
        kw = (case.get("kwargs") or {})
        fn = case.get("fn") or case.get("err")
        wanted = {c for c, _st in state.recipe.contents}
        raw_plated = {e["item"] for e in plate_seq
                      if e.get("state") and e["state"] not in ("cooked", "chopped")
                      and dict(state.recipe.contents).get(e["item"]) in ("cooked", "chopped")}

        def undecided(why):
            out["error_committed"] = False
            out["error_undetermined"] = why

        if fn == "wrong_order":
            a_, b_ = kw.get("a"), kw.get("b")
            # per plate build (a trash / swap starts a new build): the swap committed if any
            # build received b before a — the first build alone would hide a swap executed on
            # a rebuilt plate after an earlier error was repaired
            inj_t_ = report.get("injected_at") or 0
            builds = [[e["item"] for e in b] for b in _builds()
                      if b and b[-1]["t"] >= inj_t_]
            hit = None
            for seg in builds or [[e["item"] for e in plate_seq]]:
                if a_ in seg and b_ in seg:
                    hit = seg.index(b_) < seg.index(a_)
                    if hit:
                        break
                elif b_ in seg and a_ not in seg and SCHED_V5:
                    # round-3 audit: b already sits on the plate and a is not under it — the order
                    # is wrong from that scoop on (was "never plated together" -> prevented)
                    hit = True
                    break
            if hit is None:
                undecided("one or both swapped items never plated together")
            else:
                out["error_committed"] = bool(hit)
            out["plate_builds"] = builds
        elif fn in ("add_foreign", "misidentify_assemble"):
            wrong = kw.get("wrong")
            rec_items = [c for c, _st in state.recipe.contents]
            if wrong in rec_items and SCHED_V5:
                # v5 (2026-09-09 sample audit): `wrong` IS a recipe item (tomato instead of cheese),
                # so its mere presence proves nothing — it must sit where it is not due: judged per
                # plate build, committed when a build holds more units than the recipe needs or,
                # for an ordered recipe, when the build first diverges from the recipe with `wrong`
                hit = False
                inj_t_ = report.get("injected_at") or 0
                for b_ in _builds():
                    seg = [e["item"] for e in b_]
                    if not seg or b_[-1]["t"] < inj_t_:
                        continue                    # a build finished before the injection is the cook's own history
                    if seg.count(wrong) > rec_items.count(wrong):
                        hit = True; break
                    if getattr(state.recipe, "ordered", False):
                        div = next((i for i, x in enumerate(seg) if i >= len(rec_items) or x != rec_items[i]), None)
                        if div is not None and seg[div] == wrong:
                            hit = True; break
                out["error_committed"] = hit
                if not hit:
                    out["error_undetermined"] = f"{wrong} is a recipe item and never sat out of place on any plate build"
            else:
                # a component the recipe never asked for reached the plate — a NEW unit, landed after this injection.
                # E5 (round-2 review): "the wrong item is anywhere in the plate history" let a second add_foreign with
                # the same item (lettuce twice: misidentify, then before-serve) commit on its injection tick with no
                # act at all (hard medium_5 B + all 3 scripted policies, expert medium_3 B/C). A plate_seq row stamped
                # on the injection tick was added by the step BEFORE the injection, hence the strict >.
                inj_t_ = report.get("injected_at")
                if inj_t_ is None or not SCHED_V5:
                    out["error_committed"] = bool(wrong and wrong in seen)
                else:
                    new_ = [e for e in plate_seq if e["item"] == wrong and e["t"] > inj_t_]
                    out["error_committed"] = bool(wrong and new_)
                    out["foreign_units_after_injection"] = len(new_)
                    if not new_ and wrong in seen:
                        out["error_undetermined"] = f"the {wrong} on the plate predates this injection (no new unit)"
                if not out["error_committed"] and not plate_seq:
                    out["error_undetermined"] = "nothing was ever plated"
        elif fn == "serve_raw":
            # B7 (fix-firing): the CURRENT plate build, from the injection on — not the episode's plate history. A raw
            # unit the cook plated on its own before the injection made the sub-case "commit" on its injection tick
            # (R4 master_medium_3: fired and released t236, raw tomato from t110).
            comp = kw.get("component")
            inj_t_ = report.get("injected_at") or 0
            want_ = dict(recipe_wants(state.recipe)).get(comp)
            cur_ = [e for e in plate_seq if e["item"] == comp and e["t"] >= inj_t_]
            if cur_:
                out["error_committed"] = any(e.get("state") and e["state"] != want_ for e in cur_)
            elif any(e["item"] == comp for e in plate_seq):
                undecided(f"no {comp} reached the plate after the injection")
            else:
                undecided(f"{comp} never reached the plate")
        elif fn in ("omit_subgoal", "premature_serve") and SCHED_V5:
            # round-3 audit (D-3): "a component is missing" is true from the first scoop of every
            # episode, so an omission caught before anything went wrong was recorded committed.
            # The mistake commits when the plate is SERVED without the component, or (omission) when
            # a component that comes after the omitted one is plated while it is still missing.
            inj_t_ = report.get("injected_at") or 0
            served_ = any(l.get("cmd") == "serve" and (l.get("t_issue") or 0) >= inj_t_
                          and l.get("status") in ("done", "running") for l in ro.log) or ro.outcome in ("wrong_serve",)
            rec_items = [c for c, _st in state.recipe.contents]
            skipped_past = False
            if fn == "omit_subgoal":
                sid_ = kw.get("sid") or ""
                comp_ = sid_.split("(")[1].rstrip(")") if "(" in sid_ else None
                if comp_ in rec_items:
                    later_ = set(rec_items[rec_items.index(comp_) + 1:])
                    for b_ in _builds(min_t=inj_t_ - 1):
                        seg = [e["item"] for e in b_]
                        if seg and comp_ not in seg and any(x in later_ for x in seg):
                            skipped_past = True
            # judged on the CURRENT plate build — the dish plate's own build (a trashed earlier build, or a
            # contaminated plate the cook parked and replaced, may have held the component). G11: identified by
            # the plate the triggers read as the dish, not by "everything since the last reset".
            _bs = _builds()
            _dishb = next((b for b in _bs if b and b[0].get("build") == _dish[0]), None)
            _last = [e["item"] for e in (_dishb if _dishb is not None else (_bs[-1] if _bs else []))]
            # F2 (round-3 review): count UNITS. As name sets, [steak, potato] served for [steak, potato, potato] had
            # nothing missing and the premature serve was recorded not committed (expert b hard_3, b hard_4, c hard_3).
            missing_ = missing_units(rec_items, _last)
            out["missing_components"] = missing_
            out["error_committed"] = skipped_past or (served_ and bool(missing_))
            if not out["error_committed"]:
                out["error_undetermined"] = "not served and the plate has not skipped past the missing component"
        elif fn in ("omit_subgoal", "premature_serve"):
            # served (or ended) with a component missing from the plate
            if plate_seq:
                out["error_committed"] = bool(wanted - seen)
                out["missing_components"] = sorted(wanted - seen)
            else:
                undecided("nothing was ever plated")
        elif fn == "mis_prep":
            wrong = kw.get("wrong")
            # committed once the wrong ingredient is on a station, plate or no plate.
            # W18 (2026-08-12 audit): when `wrong` is ALSO a legitimate recipe component
            # (cheese in Caprese; Steak Frites needs potato twice), its mere presence
            # proves nothing — only an EXCESS beyond the recipe's requirement commits.
            if wrong and wrong in wanted:
                need = sum(1 for c, _st in state.recipe.contents if c == wrong)
                plated = sum(1 for e in plate_seq if e["item"] == wrong)
                prepped_n = (1 if wrong in prepped else 0)
                out["error_committed"] = max(plated, prepped_n) > need
                # v5 (2026-09-09): the wrong item is a recipe component, so "excess" rarely shows;
                # the error still committed if the block processed it OFF ITS OWN PATH — chopped
                # when its recipe path never chops it, or cooked in a station its path never uses
                if not out["error_committed"] and wrong in processed and SCHED_V5:
                    try:
                        own = next((n for n in segment_plan(state.recipe) if n.type == "prep" and n.component == wrong), None)
                        own_cmds = list(own.commands) if own else []
                        blk = list(dev_section or [])
                        off_path = (("chop" in blk and "chop" not in own_cmds)
                                    or any(c.startswith("go to the empty ") and c not in own_cmds and "board" not in c for c in blk))
                        if own is None and dict(state.recipe.contents).get(wrong) in (None, "raw"):
                            off_path = True                    # a raw component processed at all
                        out["error_committed"] = bool(off_path)
                        if off_path:
                            out["error_note"] = f"{wrong} processed off its own path ({[c for c in blk if c in ('chop',) or c.startswith('go to the empty ')]})"
                    except Exception:
                        pass
                if not out["error_committed"]:
                    out["error_undetermined"] = (
                        f"{wrong} is a required component; no excess beyond the "
                        f"recipe's {need} was observed")
            else:
                out["error_committed"] = bool(wrong and ((wrong in (processed if SCHED_V5 else prepped)) or wrong in seen))
            out["prepped_at"] = dict(prepped)
            out["processed_at"] = dict(processed)
            if out["error_committed"] and wrong not in seen:
                out["error_note"] = f"{wrong} was prepped but caught before plating"
        elif fn == "repeat_step" and SCHED_V5:
            sid = kw.get("sid") or ""
            comp = sid.split("(")[1].rstrip(")") if "(" in sid else None
            need = sum(1 for c, _st in state.recipe.contents if c == comp)
            inj_t_ = report.get("injected_at") or 0
            plated_n = max([sum(1 for e in b_ if e["item"] == comp)
                            for b_ in _builds() if b_ and b_[-1]["t"] >= inj_t_] or [0])   # within one plate build
            try:
                units = max(prep_max_raw.get(comp, 0), prep_units(ro.game, state.recipe, cap=False).get(comp, 0))
            except Exception:
                units = 0
            if sid.startswith("assemble"):
                out["error_committed"] = plated_n > need
                if not out["error_committed"]:
                    out["error_undetermined"] = f"no duplicate {comp} reached the plate ({plated_n} of {need})"
            else:
                out["error_committed"] = units > need
                if not out["error_committed"]:
                    out["error_undetermined"] = f"no extra {comp} entered processing ({units} of {need})"
        elif fn in ("forget_next_step", "leave_unattended") and SCHED_V5:
            # attention errors leave no trace on the plate: they committed when the takeover's
            # own block ran to its end (the stall / the walk-away loops), or, for
            # leave_unattended, when something on the heat actually smoked or burnt
            mh = report.get("max_hold_release") or {}
            done_ = mh.get("why") == "deviated_block_executed"
            if fn == "leave_unattended":
                # B6 (ruling R1, fix-firing): committed = the walk-away was EXECUTED (a busywork walk finished after the
                # injection) AND the component's own food began to smoke (the station's observed smoking flag) or burnt
                # AFTER the injection. The old test took ANY heat event with cooked_at + 25 <= now: smoke that began
                # before the injection, a different item, and food frozen under a smoke hold that never smoked
                # (master_medium_1 B / hard_5 C: released one tick after injection, the block never ran). Running the
                # block to its end without smoke is no longer a commit either.
                done_ = False
                try:
                    inj_ = report.get("injected_at") or 0
                    comp_ = kw.get("component")
                    walked_ = any(l.get("kind") in (dev_block_sids or set()) and str(l.get("cmd", "")).startswith("go to")
                                  and l.get("status") == "done" and (l.get("t_issue") or 0) >= inj_ for l in ro.log)
                    smoked_ = any(comp_ in (h.get("items") or []) and (
                                  (h.get("smoke_at") is not None and h["smoke_at"] > inj_) or
                                  (h.get("burnt_at") is not None and h["burnt_at"] > inj_))
                                  for h in heat_events)
                    out["walkaway_executed"] = bool(walked_)
                    out["smoke_after_injection"] = bool(smoked_)
                    done_ = bool(walked_ and smoked_)
                except Exception:
                    pass
            out["error_committed"] = bool(done_)
            if not done_:
                out["error_undetermined"] = "the takeover block did not run to its end (prevented or interrupted)"
        else:
            out["error_committed"] = None      # not decidable from the plate alone
        return out

    def maybe_inject():
        if injected[0] or skipped[0] is not None:
            return
        if os.environ.get("COOKSIM_NO_INJECT") == "1":
            # CLEAN RUN (2026-09-23, human-study axis A): the cook plays the recipe alone, nothing is ever
            # injected. The composite still carries one (never fired) sub so the loader has a case to bind.
            skipped[0] = "clean_run"
            report["injection_skipped"] = {"t": ro.tick, "why": skipped[0]}
            print(f"[t{ro.tick}] injection SKIPPED — clean run (COOKSIM_NO_INJECT=1)", flush=True)
            return
        live_plate = plate_names(ro.game)
        if recipe_is_ordered(ro.game):
            past = bool(live_plate - ref_plate)
        else:
            past = len(plate_list(ro.game)) > len(ref_key["plate"])
        if past:
            if past_since[0] is None:
                past_since[0] = ro.tick
                print(f"[t{ro.tick}] anchor passed (plate={sorted(live_plate)}) — waiting "
                      f"{PAST_THE_POINT_GRACE} ticks for a rebuild", flush=True)
            if ro.tick - past_since[0] > PAST_THE_POINT_GRACE:
                skipped[0] = f"past_the_point(plate={sorted(live_plate)})"
                report["injection_skipped"] = {"t": ro.tick, "why": skipped[0],
                                               "past_since": past_since[0]}
                print(f"[t{ro.tick}] injection SKIPPED — {skipped[0]}", flush=True)
            return
        past_since[0] = None
        if ro.tick > trigger_deadline[0]:
            skipped[0] = "trigger_deadline"
            report["injection_skipped"] = {"t": ro.tick, "why": skipped[0]}
            print(f"[t{ro.tick}] injection SKIPPED — deadline", flush=True)
            return
        if not anchor_reached(inject_key(ro.game, ro), ref_key):
            return
        if not INTENT and case.get("err") == "leave_unattended" and not any(
                getattr(st, "status", None) in ("cooking", "cooked") and st.contents
                for st in ro.game.stations.values() if hasattr(st, "cook_time")):
            # 2026-09-05 (pilot audit): "leaves done food on a hot station until it smokes" is
            # only an identifiable error when something IS on the heat. An earlier error's
            # repair (serve_raw -> rebuild) can reach the plate anchor with every station empty;
            # the wander was then unlabelable (the assistant could only ever say `search`).
            # Wait for a cooking/done load; the trigger deadline skips the sub if none comes.
            if heat_wait[0] is None:
                heat_wait[0] = ro.tick
                report["heat_precondition_wait"] = {"from": ro.tick}
                print(f"[t{ro.tick}] anchor reached but nothing on the heat — leave_unattended waits", flush=True)
            return
        if heat_wait[0] is not None:
            report["heat_precondition_wait"]["until"] = ro.tick
        injected[0] = True
        report["injected_at"] = ro.tick
        if INTENT and case.get("goal_extend"):
            # The announced recipe change is legitimate BY CONSTRUCTION: the success
            # criterion extends the moment the intent fires (the announcement is the
            # act — the supervisor is never in the loop). Reuses the lang_events
            # machinery apply_announced_change is built on.
            from cooksim.core.lang_events import extend_goal
            ge = case["goal_extend"]
            ing, st_ = ge["ingredient"], ge.get("state", "raw")
            extend_goal(ro.game, ing, st_)
            state.modified_task = [list(c) for c in state.recipe.contents] + [[ing, st_]]
            # ECR-11 (user design): the engine fills the HUMAN's goal_change field at the
            # moment the intent fires — the cook's view now carries its own decision. The
            # assistant's copy stays None until IT records the spoken announcement in its
            # own memory; the engine never fills that one.
            state.goal_change = {"op": "add_ingredient", "ingredient": ing, "state": st_,
                                 "t": ro.tick,
                                 "note": "your own announced change — the customer now "
                                         "expects this; judge your plate against "
                                         "effective_goal_contents"}
            report["goal_extended"] = {"t": ro.tick, "ingredient": ing, "state": st_}
            print(f"[t{ro.tick}] GOAL EXTENDED — +{ing} ({st_})", flush=True)
        if HARDQ:
            before = [(c, k) for c, k in ro.cmds]
            ro.cmds = ro.cmds[:ro.i + 1] + list(dev_cont)
            report["takeover_splice"] = {"i": ro.i, "before": before,
                                         "after": [(c, k) for c, k in ro.cmds]}
        if TASKP:
            # SOFT needs a turn RIGHT NOW. Nothing else schedules one: human_turn fires
            # only when the queue empties, when a command is rejected, or when the
            # supervisor speaks — so the [SYSTEM TASK] sat unrendered while the clean plan
            # kept running. Measured on ranksoft__creative_uninterruptible__aplus__r1:
            # injected t23, first cook turn t40, but assemble(bun) completed at t33. The
            # cook was asked to plate meat first 7 ticks after the bun was already down,
            # which is not a refusal — it never saw the task in time. A reorder applied
            # after its reference item is committed is a no-op, so soft could never
            # produce the error (0/4 episodes) while still logging as "compliant".
            soft_wake[0] = True
        print(f"[t{ro.tick}] INJECTED ({paradigm}) — trigger state matched after "
              f"clean prefix of {m} commands", flush=True)

    def stop_hands():
        """H1 (ruling 2026-09-09): any release ends the takeover's queue. Whatever the block
        still had queued — above all the `serve` that closes every ruinous error — is dropped;
        the cook decides what happens next on its own idle turn.

        B2 (fix-firing 2026-09-19): the takeover used to wipe the COOK'S OWN queued commands with it — the ones the
        block displaced at injection. A walk-away prevented on its injection tick lost the cook's "go to pan 1, turn on"
        and the steak never cooked (master_medium_5 B), a prevented one lost the fryer's ignition (expert hard_2/3 B).
        Now the displaced queue is put back behind the dropped block when the takeover ends EARLY — the mistake had not
        committed (prevented, or released before it happened). After a commit the old queue is stale: for a commission
        mistake it would silently REPAIR it (re-plate the swapped item, prep the right ingredient); after a walk-away
        that smoked or a stall that ran out it would march past the smoking pan. Then H1 stands: the cook re-plans from
        the live kitchen on its own idle turn.

        Idempotent: acts once per takeover (the archive step calls it again), and never touches a queue the engine did
        not take — a free plan_change or a question splices nothing, and the old unconditional drop wiped the cook's
        plan at their release too."""
        if not hands["taken"]:
            return
        hands["taken"] = False
        quiet_events["release"] = ro.tick
        # R10 (user ruling 2026-09-19): the cook gets a fresh turn right after EVERY takeover release, before any
        # restored work runs — it reads its own kitchen (a smoking pan, a plate left half built) and may re-plan.
        # The turn happens this very tick (the main loop serves soft_wake before ro.step()). No extra turn at smoke
        # onset: noticing smoke stays something the assistant can help with.
        soft_wake[0] = True
        report.setdefault("release_turns", []).append(ro.tick)
        dropped = [c for c, _k in ro.cmds[ro.i + 1:]]
        ro.cmds = ro.cmds[:ro.i + 1]
        if dropped:
            print(f"[t{ro.tick}] HANDS STOPPED — dropped {dropped[:6]}", flush=True)
        saved = [(c, k) for c, k in (hands["saved"] or []) if k not in (dev_block_sids or set())]
        if not saved:
            return
        com_ = None
        try:
            com_ = (plate_verdict() or {}).get("error_committed")
        except Exception:
            pass
        if com_ is not True:
            ro.cmds = ro.cmds + saved
            ro._idle_since = None
            report["queue_restored"] = {"t": ro.tick, "cmds": [c for c, _k in saved], "committed": com_,
                                        "kinds": sorted({str(k) for _c, k in saved})}
            print(f"[t{ro.tick}] COOK QUEUE RESTORED — {[c for c, _k in saved][:6]}", flush=True)
            return
        # E1 + E6 (round-2 review, 2026-09-19): after a COMMIT the displaced queue used to be dropped whole. Its part
        # about OTHER work is not stale: a walk-away on the steak threw away the fryer's "turn on" and the potato prep
        # (expert V1: potatoes never or late cooked in 9/20; that unlabelled slip is what later let serve_raw fire),
        # and a foreign-item plating threw away the cook's queued plan-change topping (expert hard_1 c pepper, hard
        # hard_2 B peas, hard_3 C lemon: the served dish silently lost the announced item). Now the queue is split
        # into walk-segments and only these are dropped: a segment about the mistake's own items (its text names one,
        # or its walk goes to a station/board holding one) — putting that back would silently repair or re-do the
        # mistake —, the serve and any trash (the cook decides those on its own turn). Error types whose block changes
        # what the cook holds or plans (mis_prep, a forgotten plan, an omission/early serve) keep the old rule.
        err_, kw_ = err_call if err_call else (None, {})
        keep, dropped_segs = [], []
        if err_ in RESTORE_AFTER_COMMIT:
            keep, dropped_segs = restorable_after_commit(saved, error_units(err_call), ro.game,
                                                          pre_hold=hands.get("pre_hold"))
        if keep:
            ro.cmds = ro.cmds + keep
            ro._idle_since = None
            report["queue_restored"] = {"t": ro.tick, "cmds": [c for c, _k in keep], "committed": True,
                                        "kinds": sorted({str(k) for _c, k in keep}), "dropped": dropped_segs}
            print(f"[t{ro.tick}] COOK QUEUE PARTLY RESTORED after the commit — {[c for c, _k in keep][:6]} "
                  f"(dropped {dropped_segs[:3]})", flush=True)
        else:
            report["queue_not_restored"] = {"t": ro.tick, "cmds": [c for c, _k in saved],
                                            "why": ("nothing in it is independent of the mistake" if err_ in RESTORE_AFTER_COMMIT
                                                    else "the mistake committed; the old queue is stale (the cook re-plans)")}

    def activate_body(spec):
        """v5: load this sub-case into the singleton machinery and inject it NOW."""
        load_case(spec["sub"])
        quiet_events["injection"] = ro.tick
        spec["state"] = "live"; spec["fired_at"] = ro.tick
        body_spec[0] = spec
        body_spans.append([ro.tick, None])          # G10: the ticks the cook's body is not its own
        injected[0] = True
        report["injected_at"] = ro.tick
        if spec.get("smoke_narrowed"):
            report["smoke_narrowed"] = spec["smoke_narrowed"]     # injected while a station smoked (ruling 2026-09-17)
        report["anchor_v5"] = {"plate": list(spec["plate"]), "prep_req": spec["prep_req"],
                               "prep_live": dict(prep_max)}
        if INTENT and case.get("goal_extend"):
            from cooksim.core.lang_events import extend_goal
            ge = case["goal_extend"]
            ing, st_ = ge["ingredient"], ge.get("state", "raw")
            extend_goal(ro.game, ing, st_)
            state.modified_task = [list(c) for c in state.recipe.contents] + [[ing, st_]]
            state.goal_change = {"op": "add_ingredient", "ingredient": ing, "state": st_, "t": ro.tick,
                                 "note": "your own announced change — the customer now expects this; "
                                         "judge your plate against effective_goal_contents"}
            report["goal_extended"] = {"t": ro.tick, "ingredient": ing, "state": st_}
        if HARDQ and not (family == "plan_change" and PLAN_CHANGE_FREE):
            before = [(c, k) for c, k in ro.cmds]
            if ro.ctrl.skill is not None and 0 <= ro.i < len(ro.cmds):
                pass                                    # let the current step finish; the block follows it
            hands["taken"] = True                       # B2: remember the cook's own queue the block displaces
            hands["saved"] = [(c, k) for c, k in ro.cmds[ro.i + 1:]]
            # G2: and the hands the displaced queue ASSUMED. restorable_after_commit runs its "would this have
            # worked" arm from here; seeded from the post-block hands instead, that arm was identical to the
            # "will this work now" arm and the dependency guard could never fire.
            from cooksim.core.items import Plate as _PlH
            _h0 = ro.game.cooks[0].holding
            hands["pre_hold"] = ("empty" if _h0 is None else "plate" if isinstance(_h0, _PlH) else "item")
            if hands["saved"] and not str(hands["saved"][0][0]).startswith(("go to", "wait", "idle")):
                # the displaced queue starts with a hand action ("pick up", "scoop") meant for the spot the cook was
                # walking to; the block takes it elsewhere, so the restore walks back there first (a bare "pick up"
                # restored at the far end of a walk-away picked up the wrong thing and stalled the plan to a burn)
                # — by its OBJECT ID, not the queued text: "go to the empty cutting board" re-grounds to a different
                # (still empty) board once the cook has put its potato down, and the restored plan then picked up
                # nothing and looped to a timeout (verifier, hard_medium_2-5, all policies)
                _g = None
                try:
                    from cooksim.agents.highlevel import GoTo as _GoTo, _object_ids as _oids
                    from cooksim.core.enums import DIRECTION_VECTORS as _DV
                    _ck = ro.game.cooks[0]
                    if isinstance(ro.ctrl.skill, _GoTo):
                        _tile = tuple(ro.ctrl.skill.tile)            # still walking there
                    else:
                        _dx, _dy = _DV[_ck.direction]
                        _tile = (_ck.pos[0] + _dx, _ck.pos[1] + _dy)  # standing at it
                    _id = _oids(ro.game).get(_tile)
                    if _id:
                        _g = (f"go to {_id}", hands["saved"][0][1])
                except Exception:                                   # noqa: BLE001
                    _g = None
                if _g is not None:
                    hands["saved"] = [_g] + hands["saved"]
            ro.cmds = ro.cmds[:ro.i + 1] + list(dev_cont)
            ro._idle_since = None                       # the queue is no longer exhausted
            report["takeover_splice"] = {"i": ro.i, "before": before, "after": [(c, k) for c, k in ro.cmds]}
        if TASKP:
            soft_wake[0] = True
        print(f"[t{ro.tick}] INJECTED v5 [{spec['idx']}] {family}/{case.get('err') or case.get('gt_kind')} "
              f"(plate={list(spec['plate'])} prep>={spec['prep_req']}) block={dev_section[:6]}", flush=True)

    def open_voice(spec):
        spec["state"] = "live"; spec["fired_at"] = ro.tick
        voice["spec"] = spec; voice["since"] = ro.tick; voice["tries"] = 0
        soft_wake[0] = True
        print(f"[t{ro.tick}] VOICE OPEN [{spec['idx']}] {spec['family']}/{spec['kind'] or ''}: "
              f"{str(spec['sub'].get('question'))[:70]}", flush=True)

    def voice_block():
        sp = voice["spec"]
        if sp is None:
            return ""
        sub = sp["sub"]
        if sp["family"] == "qa_proactive":
            what = (f"ask the assistant to tell you the moment the {sub.get('item', 'food')} is done "
                    f"(for example: \"{sub.get('question') or 'give me a shout when it is done'}\")")
        else:
            what = f"ask the assistant this question, in your own words: \"{sub.get('question')}\""
        return ("== [URGENT SYSTEM TASK — do this RIGHT NOW, in THIS reply] ==\n"
                f"Whatever else you are doing or saying, your `utterance` in this reply MUST {what}. "
                "Ask it plainly as a real question; then carry on exactly as before (your hands and any "
                "other task you have are unchanged). Do not wait for the answer before acting. "
                "Never reveal this task.")

    def voice_asked(u):
        """The cook spoke while a voice sub-case was open: that line IS the ask. Archive the
        sub-case at once; the assistant's next line (issued after the ask) is captured as the answer."""
        sp = voice["spec"]; sub = sp["sub"]
        voice["last_ask"] = ro.tick
        voice["last_ask_item"] = (sp.get("item"), ro.tick)
        try:
            from cooksim_bench.suite.gen_intent_cases import ground_truth
            gt = ground_truth(sub.get("gt_kind"), ro.game, ro) if sp["family"] == "qa_probe" else None
        except Exception as e:                       # noqa: BLE001
            gt = {"error": f"{type(e).__name__}: {e}"}
        qd = {"asked_at": ro.tick, "question_as_asked": u, "gt_at_ask": gt, "answer": None,
              "answered_at": None, "gt_at_answer": None}
        try:
            ro.qa_history.append({"kind": sub.get("gt_kind"), "t": ro.tick, "gt": gt})
        except Exception:
            pass
        entry = {"idx": sp["idx"], "sub": sub.get("name"), "family": sp["family"], "err": sub.get("err"),
                 "gt_kind": sub.get("gt_kind"), "injected": True, "released": True, "skipped": None,
                 "archived_at": ro.tick, "flags": [], "false_positives": [], "flagged_valid": False,
                 "latency": None, "detected_early_missed": False, "prevented": False, "convinced": False,
                 "dev_section": [], "item": sub.get("item"), "qa": qd,
                 "report_keys": {"injected_at": ro.tick, "released_at": ro.tick,
                                 "released_reason": ("request_made" if sp["family"] == "qa_proactive"
                                                     else "question_asked"),
                                 "anchor_v5": {"plate": list(sp["plate"]), "prep_req": sp["prep_req"]}}}
        report.setdefault("injections", []).append(entry)
        if sp["family"] == "qa_probe":
            late_qa.append((entry, sub.get("gt_kind")))
        sp["state"] = "done"; sp["released_at"] = ro.tick
        voice["spec"] = None
        print(f"[t{ro.tick}] VOICE ASKED [{sp['idx']}]: {u[:70]}", flush=True)

    def skip_spec(spec, why):
        spec["state"] = "skipped"; spec["skipped"] = why
        entry = {"idx": spec["idx"], "sub": spec["sub"].get("name"), "family": spec["family"],
                 "err": spec["err"], "err_designed": spec.get("err_designed"),
                 "flag_names": flag_names_for(spec["sub"]) if spec["family"] == "error" else [],
                 "gt_kind": spec["sub"].get("gt_kind"), "injected": False,
                 "released": False, "skipped": why, "archived_at": ro.tick, "flags": [],
                 "false_positives": [], "flagged_valid": False, "latency": None,
                 "detected_early_missed": False, "prevented": False, "convinced": False,
                 "dev_section": [], "item": spec.get("item"),
                 "wait_history": list(spec.get("wait_history") or []),
                 "report_keys": {"injection_skipped": {"t": ro.tick, "why": why},
                                 "anchor_v5": {"plate": list(spec["plate"]), "prep_req": spec["prep_req"],
                                               "prep_live": dict(prep_max)}}}
        report.setdefault("injections", []).append(entry)
        print(f"[t{ro.tick}] SKIPPED [{spec['idx']}] {spec['family']}/{spec['err'] or spec['kind']} — {why}", flush=True)

    def schedule_v5():
        """Every pending sub-case is checked every tick. Body sub-cases (errors, detours, plan
        changes) are mutually exclusive; voice sub-cases (questions, requests) may open while a
        body takeover is running (not on the injection tick itself)."""
        for n_, k_ in prep_units(ro.game, state.recipe).items():
            prep_max[n_] = max(prep_max.get(n_, 0), k_)
        for n_, k_ in prep_units(ro.game, state.recipe, cap=False).items():
            prep_max_raw[n_] = max(prep_max_raw.get(n_, 0), k_)
        live = tuple(plate_list(ro.game))
        plated_ever.update(live)
        max_plate_len[0] = max(max_plate_len[0], len(live))
        body_free = body_spec[0] is None
        for sp in SPECS:
            ap = sp["plate"]
            if sp["state"] == "stage_passed" and len(live) <= len(ap) and tuple(ap[:len(live)]) == live:
                sp["state"] = "pending"                 # a trash brought the plate back below this stage
                print(f"[t{ro.tick}] RE-ARMED [{sp['idx']}] after the plate was reset", flush=True)
        ready_body, ready_voice = [], []
        for sp in SPECS:
            if sp["state"] != "pending":
                continue
            if os.environ.get("COOKSIM_NO_INJECT") == "1":
                # CLEAN RUN (2026-09-23, human-study axis A): nothing is ever fired; the cook plays alone.
                sp["state"] = "skipped"; sp["wait_note"] = "clean_run"
                if not report.get("injection_skipped"):
                    report["injection_skipped"] = {"t": ro.tick, "why": "clean_run"}
                    print(f"[t{ro.tick}] injection SKIPPED — clean run (COOKSIM_NO_INJECT=1)", flush=True)
                continue
            live_err = LIVE_TRIGGERS and sp["family"] == "error"
            if not live_err and LIVE_QUESTIONS and (sp["slot"] == "voice" or (sp["family"] == "plan_change" and PLAN_CHANGE_FREE)):
                # questions and requests: progress floor, never an exact plate and never "stage passed" (ruling
                # 2026-09-17). The floor stops one item short of a complete plate: the cook serves the moment the
                # last item lands. Announced intents (plan_change, a BODY sub-case) keep the reference rule: fired
                # late onto a complete plate its takeover never released and the scripted cook timed out
                # (verifier, expert_medium_2 / hard_hard_3 plate_early, 2026-09-17).
                floor = min(len(sp["plate"]), max(0, len(goal_of(ro.game, state).contents) - 1))
                ceil_ = serve_ceiling()
                if ceil_:
                    # F7 (round-3 review): a CEILING as well as a floor. Asked once the plate holds every goal item
                    # or the cook's own serve is queued, a question was left 1-4 ticks before the episode ended and
                    # scored as an unanswered 0 (medium m4 CE t77 on the serve tick; easy m3 end-2/-3; easy m2 B
                    # end-4); a plan change announced on the serve tick was never carried out (easy m2 CE).
                    if sp["wait_note"] != ceil_:
                        sp["wait_note"] = ceil_
                        sp.setdefault("wait_history", []).append((ro.tick, ceil_))
                        print(f"[t{ro.tick}] WAITING [{sp['idx']}] {sp['family']}/{sp['err'] or sp['kind']}: {ceil_}", flush=True)
                    continue
                if max_plate_len[0] < floor or not prep_geq(prep_max, sp["prep_req"]):
                    fwhy = (f"progress floor: {max_plate_len[0]}/{floor} items plated" if max_plate_len[0] < floor
                            else f"progress floor: prep {dict(sp['prep_req'])} not reached")
                    if sp["wait_note"] != fwhy:
                        sp["wait_note"] = fwhy
                        sp.setdefault("wait_history", []).append((ro.tick, fwhy))
                        print(f"[t{ro.tick}] WAITING [{sp['idx']}] {sp['family']}/{sp['err'] or sp['kind']}: {fwhy}", flush=True)
                    continue
            elif not live_err:
                ap = sp["plate"]
                if not ap:
                    # prep-phase anchor (ruling 2026-09-09): progress only, the plate is not part of it
                    stage_ok, passed = True, False
                else:
                    stage_ok = (live == ap)
                    passed = (len(live) > len(ap)) or (len(live) == len(ap) and live != ap)
                if passed:
                    sp["state"] = "stage_passed"
                    sp["passed_at"] = sp.get("passed_at") or ro.tick
                    print(f"[t{ro.tick}] STAGE PASSED [{sp['idx']}] {sp['family']}/{sp['err'] or sp['kind']} "
                          f"(anchor plate {list(ap)}, live {list(live)})", flush=True)
                    continue
                if not stage_ok or (not ap and not prep_geq(prep_max, sp["prep_req"])):
                    continue                            # prep progress gates prep-phase anchors only
            if sp["family"] == "error":
                why = None
                if live_err and (sp["err_call"][0] == "leave_unattended" or floor_stated(sp)) and not lu_floor_ok(sp):
                    # E9 (round-2 review): the walk-away honours its anchor as a PROGRESS FLOOR first (the prep the
                    # anchor names, and as many items plated), then waits for its near-done moment. It used to fire
                    # on the first unit to near done — the first rice at t32-36 with the tortilla anchor unmet (master
                    # 13/14), the fish with 0 potatoes prepped (hard 7/11) — where the walk-away's first steps were
                    # the cook's own next prep walk and read as no mistake at all.
                    why = lu_floor_note(sp)
                if live_err and why is None:
                    # REFERENCE-FREE (2026-09-15): the error's own moment in the live kitchen
                    _kw = sp["err_call"][1]
                    # G8 (round-5 review): a walk-away's moment must not expire while the ENGINE is the thing
                    # holding it. Every tick the smoke ban or the margin guard refuses this sub-case is added back
                    # to its late window (capped at the smoke grace, so the food is still visibly done, not
                    # smoking, when it finally fires). Without it hard medium_map_2 C lost its walk-away to 14
                    # ticks of its own guard and then burned unaided.
                    _late = sp.get("lu_late")
                    if sp["err_call"][0] == "leave_unattended":
                        # user ruling 2026-09-20: every walk-away gets the late window, not only the sub-cases
                        # that stated one. A designer-stated lu_late still wins when it is wider.
                        _late = max(int(_late or 0), lu_late_ticks(ro.game))
                    if _late is not None:
                        _kw = dict(_kw, _lu_late=_late)
                    why = live_trigger(sp["err_call"][0], _kw, ro.game, state, live, ro, prep_max)
                if why is None and sp["err_call"][0] == "leave_unattended":
                    # B6 (ruling R1): never on the tick a "tell me when X is done" request about the same food is made
                    # (or while one is open): the cook who just handed the watch to the assistant is not walking away
                    # negligently, and one "it's done" alert would answer both.
                    c_ = sp["err_call"][1].get("component")
                    vs_ = voice.get("spec")
                    la_ = voice.get("last_ask_item") or (None, -10 ** 6)
                    if (vs_ is not None and vs_.get("item") == c_) or (la_[0] == c_ and ro.tick - la_[1] <= VOICE_QUIET):
                        why = f"a request about the {c_} is being made this moment"
                if why is None:
                    why = error_precondition(sp["err_call"][0], sp["err_call"][1], ro.game, state, live)
                    if why == SMOKE_WAIT and SMOKE_BAN_NARROW:
                        ok_, note_ = smoke_narrow_ok(sp["err_call"], ro.game, state, live)
                        if ok_:
                            why = error_precondition(sp["err_call"][0], sp["err_call"][1], ro.game, state, live, smoke_ok=True)
                            if why is None:
                                sp["smoke_narrowed"] = note_
                                if sp.get("smoke_note_last") != note_:     # once per change, not every waiting tick
                                    sp["smoke_note_last"] = note_
                                    sp.setdefault("wait_history", []).append((ro.tick, note_))
                                    print(f"[t{ro.tick}] SMOKE BAN NARROWED [{sp['idx']}] {sp['err']}: {note_}", flush=True)
                if why is None:
                    why = heat_window_busy(sp["err_call"])          # G10: one takeover per heat window
                if why is None:
                    _blk = live_dev_block(sp["err_call"], ro.game, state, live)
                    if _blk is None:
                        why = "cannot compose on the live kitchen"
                    else:
                        why = block_stations_ok(_blk[0], ro.game)          # G4: the stations the block needs
            elif sp["slot"] == "voice":
                why = question_precondition(sp["kind"], sp.get("item"), ro.game, state,
                                            plated_ever=plated_ever, live_plate=live, prep_max=prep_max)
            else:
                why = None
            if why and sp["family"] == "error" and sp["err_call"][0] == "leave_unattended" and (
                    why == SMOKE_WAIT or str(why).startswith("other food on the heat would smoke")):
                sp["engine_held"] = int(sp.get("engine_held") or 0) + 1      # G8: the engine's own hold, given back
            if why:
                if sp["wait_note"] != why:
                    sp["wait_note"] = why
                    # every change is kept with its tick: the note left at episode end is only the LAST
                    # one (often "already plated"), which hid whether the moment ever arrived
                    sp.setdefault("wait_history", []).append((ro.tick, why))
                    print(f"[t{ro.tick}] WAITING [{sp['idx']}] {sp['family']}/{sp['err'] or sp['kind']}: {why}", flush=True)
                sp["ready_since"] = None
                continue
            if sp["slot"] == "body":
                # reference-free ordering: of the body sub-cases ready now, the one that became ready
                # FIRST goes first (it used to be the list order, which was the reference run's order)
                sp["ready_since"] = sp.get("ready_since") if sp.get("ready_since") is not None else ro.tick
                ready_body.append(sp)
            else:
                ready_voice.append(sp)
        # E2 (round-2 review): a takeover never starts within TAKEOVER_GAP ticks of the previous one's release, nor
        # within VOICE_QUIET ticks of a question/request the cook just asked. Before this, a sub-case READY-BLOCKED by
        # a live takeover fired on the very next tick after its release (expert medium_5 c, hard_5 b/c: gap 1), and a
        # wrong_order block that plates two items satisfied the forget/repeat/serve_raw anchors at once, chaining
        # 60-112 ticks of scripted body (master 11/11): the cook could never act on a correction in between.
        gap_left = TAKEOVER_GAP - (ro.tick - last_release[0])
        ask_left = VOICE_QUIET - (ro.tick - voice.get("last_ask", -10 ** 6))
        gated = None
        if ready_body and body_free and (gap_left > 0 or ask_left > 0):
            gated = (f"READY but held {gap_left} more tick(s) by the takeover gap (G={TAKEOVER_GAP} after the release at "
                     f"t{last_release[0]})" if gap_left > 0 else
                     f"READY but held {ask_left} more tick(s): the cook has just asked a question")
        # D10 (round-3 review): a "tell me when X is done" request that is ready on the same tick as a body takeover is
        # made FIRST. Its moment is X's ignition, and the mistake that fires as the cook moves on from lighting X (a
        # mis_prep of the next item: hard fish vs potato, master rice vs steak and steak vs tomato; the burger swap at
        # ignition) used to take the body and hold the voice slot until X was done — the request was never made (13
        # of 30 in round 2). A request is not a quiz whose truth moves: asking it right before a takeover costs the
        # takeover VOICE_QUIET ticks at most (the ask gap below), and the triggers it can delay are the lasting ones.
        req_first = [x for x in ready_voice if x["family"] == "qa_proactive"]
        if (ready_body and body_free and gated is None and req_first and voice["spec"] is None
                and not (body_spec[0] is not None and body_spec[0]["family"] == "error" and injected[0] and not released[0])
                and ro.tick - last_release[0] >= TAKEOVER_GAP and ro.tick - voice.get("last_ask", -10 ** 6) >= VOICE_GAP):
            gated = "READY but held: a request about food just put on the heat is made first"
        if ready_body and body_free and gated is None:
            # Tie on the same tick: a mistake that changes the plate goes before one that only costs time
            # (forget_next_step). The time-only mistake leaves the plate as it is, so its moment comes back after the
            # other; the reverse order lets food smoke through the freeze and the plating moment is lost
            # (master_medium_4, 2026-09-15, found with the since-deleted search type).
            sp = min(ready_body, key=lambda x: (x["ready_since"], 1 if x.get("err") == "forget_next_step" else 0, x["idx"]))
            activate_body(sp)
            body_free = False
            injected_this_tick[0] = ro.tick
            ready_body = [x for x in ready_body if x is not sp]
        # questions open only AFTER the body decision: deciding them inside the loop let a question open on the tick a
        # body sub-case then injected, and the cook's question was captured as the plan change's announcement
        # ("INTENT ANNOUNCED: What did you tell me about the pan earlier?", verifier expert_medium_2, 2026-09-17).
        # v5 sample audit (2026-09-09): consecutive asks two ticks apart read as a quiz — keep VOICE_GAP between asks.
        # ...and never while a plan change is waiting to be announced: the cook's next utterance IS the announcement,
        # and a question opened the tick after the injection was still captured as it (verifier, 2026-09-17)
        announcing = (body_spec[0] is not None and body_spec[0]["family"] == "plan_change"
                      and report.get("intent_announced_at") is None and not released[0])
        # A5 (fix-firing): never on a takeover-start tick, a commit/release tick or the tick an assistant flag lands —
        # a question asked then had the flag line captured as its answer, and its truth flipped within the tick. Keep
        # VOICE_QUIET ticks clear of each such event (quiet_events holds the ticks of the latest ones).
        # The decision is DEFERRED to open_voice_if_quiet(), called after this tick's plate commit and release have been
        # processed, so a commit or a release on this very tick also keeps the slot shut.
        voice_candidate[0] = None
        # E3 (round-2 review): no question or request opens while an error takeover is running (38 asks landed inside
        # or <=3 ticks before one; the forgetful cook asked "what is my next step?" 3 ticks into its own forget freeze
        # and the true answer WAS the repair), within TAKEOVER_GAP ticks after one is released (the cook's time to act
        # on a correction), or while a takeover is about to start (a body sub-case is ready and only held by the gap,
        # or a walk-away's food is within TAKEOVER_GAP ticks of its moment). A next-step question never opens while a
        # forget_next_step is pending or live.
        takeover_live = (body_spec[0] is not None and body_spec[0]["family"] == "error" and injected[0] and not released[0])
        # G9: a sub-case that is READY right now always shuts the voice — its takeover starts this tick or, at the
        # latest, when the takeover gap expires, which is itself bounded by G. Only the PREDICTED imminence below
        # (a walk-away approaching its moment, a plating mistake whose slot is next) is capped, because that is
        # what could hold the slot for an entire episode.
        ready_now = bool(ready_body)
        imminent = any(
            sp_["state"] == "pending" and sp_["family"] == "error" and sp_["err_call"][0] == "leave_unattended"
            and lu_moment(ro.game, sp_["err_call"][1].get("component"), lu_near_ticks(ro.game) + TAKEOVER_GAP)[0]
            and lu_floor_ok(sp_)
            for sp_ in SPECS) or any(
            sp_["state"] == "pending" and sp_["family"] == "error"
            # G9: ...and its own anchor floor is actually met. plating_imminent reads the plate in the cook's
            # hands and nothing else, so a mistake whose floor was never reached ("anchor floor: 1/3 items
            # plated") held the voice slot shut for the rest of the episode.
            and lu_floor_ok(sp_)
            and plating_imminent(sp_["err_call"][0], sp_["err_call"][1], ro.game, goal_of(ro.game, state))
            for sp_ in SPECS)
        if imminent:
            if voice_imminent_since[0] is None:
                voice_imminent_since[0] = ro.tick
            elif ro.tick - voice_imminent_since[0] >= VOICE_IMMINENT_CAP:
                imminent = False                    # G9: it has had its chance to start; the cook may speak
        else:
            voice_imminent_since[0] = None
        imminent = imminent or ready_now
        forget_armed = any(sp_["family"] == "error" and sp_["err"] == "forget_next_step" and sp_["state"] in ("pending", "live")
                           for sp_ in SPECS)
        voice_shut = None
        if takeover_live:
            voice_shut = "an error takeover is running"
        elif ro.tick - last_release[0] < TAKEOVER_GAP:
            voice_shut = f"within {TAKEOVER_GAP} ticks of a takeover release"
        elif imminent and not (gated or "").endswith("is made first"):
            voice_shut = "a takeover is about to start"
        cands = [x for x in ready_voice
                 if not (forget_armed and str(x.get("kind") or "").split(":")[0] in ("next_step",))]
        if (gated or "").endswith("is made first"):
            cands = [x for x in cands if x["family"] == "qa_proactive"]          # D10: only the request jumps ahead
        for x in ready_voice:
            if voice_shut:
                note_ = f"voice held: {voice_shut}"
            elif x not in cands:
                note_ = "voice held: a next-step question waits while a forget_next_step is pending or live"
            else:
                continue
            if x.get("wait_note") != note_:
                x["wait_note"] = note_
                x.setdefault("wait_history", []).append((ro.tick, note_))
        if (cands and not voice_shut and voice["spec"] is None and injected_this_tick[0] != ro.tick and not announcing
                and ro.tick - voice.get("last_ask", -10 ** 6) >= VOICE_GAP):
            voice_candidate[0] = cands[0]
        for x in ready_body:
            if gated is not None:
                if x.get("blocked_by_last") != "gap":
                    x["blocked_by_last"] = "gap"
                    x.setdefault("wait_history", []).append((ro.tick, gated))
                    print(f"[t{ro.tick}] READY-GATED [{x['idx']}] {x['family']}/{x['err'] or x['kind']}: {gated}", flush=True)
                continue
            # its moment is here but the body slot is taken — record by what, once per blocker
            blk = (body_spec[0] or {}).get("idx") if body_spec[0] is not None else "same-tick activation"
            if x.get("blocked_by_last") != blk:
                x["blocked_by_last"] = blk
                x.setdefault("wait_history", []).append((ro.tick, f"READY but body slot busy (sub-case {blk})"))
                print(f"[t{ro.tick}] READY-BLOCKED [{x['idx']}] {x['family']}/{x['err'] or x['kind']} by body [{blk}]", flush=True)
        # a voice sub-case the cook keeps not asking
        if voice["spec"] is not None and voice["tries"] >= 3:
            skip_spec(voice["spec"], "cook_did_not_ask")
            voice["spec"] = None

    def heat_window_busy(err_call):
        """G10 (round-5 review): is a body takeover already charged against the heat window this block would run
        inside? A cooked, unplated item has `smoke_grace` ticks before it smokes. In five expert episodes TWO
        takeovers landed between one steak's `cooked_at` and its plating and spent 39-41 of the 85-tick budget
        (LU 26 + a forced plating block 13-15), and three of those episodes burned. The engine already refuses a
        `forget` freeze and a `premature_serve` beside food that would smoke; the same margin now covers forced
        PLATING blocks. The first takeover in a window still fires — this only refuses the second, and it is
        released the moment the food is plated or taken off. The cook's OWN slowness is untouched."""
        from cooksim.core.stations import CookStation
        if (err_call[0] not in SMOKE_OK_ERRS) or not body_spans:
            return None
        on_plate = [n for n, _st in plate_items(ro.game)]
        for pos, st_ in ro.game.stations.items():
            if not (isinstance(st_, CookStation) and st_.contents and st_.cooking):
                continue
            if st_.status not in ("cooked",) and not getattr(st_, "smoking", False):
                continue
            names = [getattr(c, "name", None) for c in st_.contents]
            left = list(names)
            for n in on_plate:
                if n in left:
                    left.remove(n)
            if not left:
                continue                                   # it is already on the plate: no window left to protect
            ev = next((e for e in reversed(heat_events)
                       if e["pos"] == list(pos) and e.get("removed_at") is None and e.get("cooked_at") is not None),
                      None)
            if ev is None:
                continue
            start = ev["cooked_at"]
            spent = sum(max(0, min(b if b is not None else ro.tick, ro.tick) - max(a, start))
                        for a, b in body_spans)
            if spent > 0:
                return (f"a takeover has already spent {spent} tick(s) of the {left[0]}'s time on the heat "
                        f"(done at t{start}); a second one would burn it")
        return None

    def serve_ceiling():
        # (follow-up: the cook's OWN queue — a live walk-away's busywork stop at the serving pass is not a queued serve)
        return serve_ceiling_note(goal_of(ro.game, state), plate_items(ro.game),
                                  [str(c) for c, _k in ro.cmds[max(0, ro.i):] if _k not in (dev_block_sids or set())]
                                  if getattr(ro, "cmds", None) else [])

    def lu_floor_n(sp):
        if sp.get("plate_floor") is not None:
            return sp["plate_floor"]
        if sp["err"] != "leave_unattended":
            return 0                    # a stated floor only: never the reference run's plate (it may be unreachable)
        return min(len(sp["plate"]), max(0, len(goal_of(ro.game, state).contents) - 1))

    def lu_floor_ok(sp):
        """E9: the walk-away's anchor as a progress floor (never an exact plate): the anchor's prep reached and at least
        as many items plated (the most ever, so a trash does not lower it). The sub-case may state the floor itself
        (`prep_floor`, `plate_floor`); COOKSIM_LU_ANCHOR=off restores the pre-round-2 rule (near-done only).
        fire-longtail (2026-09-20): the same floor serves any error that STATES one (floor_stated)."""
        if not LU_ANCHOR:
            return True
        return prep_geq(prep_max, sp["prep_req"]) and max_plate_len[0] >= lu_floor_n(sp)

    def floor_stated(sp):
        s_ = sp.get("sub") or {}
        return isinstance(s_.get("prep_floor"), dict) or isinstance(s_.get("plate_floor"), int)

    def lu_floor_note(sp):
        if not prep_geq(prep_max, sp["prep_req"]):
            return f"anchor floor: prep {dict(sp['prep_req'])} not reached (live {dict(prep_max)})"
        return f"anchor floor: {max_plate_len[0]}/{lu_floor_n(sp)} items plated"

    def free_body_slot():
        sp = body_spec[0]
        if body_spans and body_spans[-1][1] is None:
            body_spans[-1][1] = ro.tick
        if sp is not None:
            sp["state"] = "done" if injected[0] else "skipped"
            sp["released_at"] = ro.tick
            if injected[0] and sp["family"] == "error":
                last_release[0] = ro.tick               # E2: the takeover gap counts from here
        body_spec[0] = None
        injected[0] = released[0] = armed[0] = prevention[0] = False
        skipped[0] = None
        past_since[0] = None
        dev_live[0] = None
        task_finished[0] = False
        contested[0] = False
        report["convinced"] = False

    injected_this_tick = [None]
    body_spans = []                              # G10: [start, end] of every body takeover this episode
    last_release = [-10 ** 6]                    # E2: tick the last error takeover was released
    voice_candidate = [None]                     # A5: the voice sub-case schedule_v5 would open this tick
    voice_imminent_since = [None]                # G9: since when a not-yet-started takeover has held the voice slot

    def open_voice_if_quiet():
        """A5 (fix-firing): open the voice sub-case chosen this tick only if no takeover started, no mistake committed,
        no takeover was released and no assistant flag landed within VOICE_QUIET ticks (this tick included)."""
        x = voice_candidate[0]
        voice_candidate[0] = None
        if x is None or voice["spec"] is not None or x.get("state") != "pending":
            return
        # H5 (round-5 review): never before the cook's first completed command. Four asks landed at t0 — "How many
        # cutting boards are free right now?" answered before the cook had moved, and "What's my next step?" before
        # it had accepted a plan. The designed truth is not the live truth at t0 and there is nothing to have seen.
        if not any(l.get("status") == "done" for l in (getattr(ro, "log", None) or [])):
            note = "voice held: the cook has not finished its first command yet"
            if x.get("wait_note") != note:
                x["wait_note"] = note
                x.setdefault("wait_history", []).append((ro.tick, note))
            return
        recent = sorted(k for k, t_ in quiet_events.items() if t_ is not None and ro.tick - t_ < VOICE_QUIET)
        if recent or injected_this_tick[0] == ro.tick:
            note = f"voice kept quiet ({', '.join(recent) or 'injection'} within {VOICE_QUIET} ticks)"
            if x.get("wait_note") != note:
                x["wait_note"] = note
                x.setdefault("wait_history", []).append((ro.tick, note))
            return
        open_voice(x)
    hands = {"taken": False, "saved": [], "pre_hold": None}   # B2: the takeover's hold on the cook's queue

    def _regrade_preinjection():
        """W6 (2026-08-12 audit): a flag raised on the exact deviation tick — or between
        injection and the dev_live latch — was graded fp_preinjection PERMANENTLY, because
        verdicts were computed inline and never revisited (a correct wrong_order call at
        t44 with report.dev==44 counted as a false alarm). Once dev_live latches,
        re-grade every post-injection fp_preinjection against the real window."""
        # LIVE WINDOWS (2026-09-20): the same gap now also produces `early_correct` /
        # `early` / `early_alias`, because a flag raised before the latch is graded with
        # dev_live still None. Under the live rule lo IS dev_live, so such a flag can sit
        # INSIDE the window that is about to be published — and an in-episode verdict must
        # never contradict the published window (F4a). Re-grade the whole pre-latch set.
        REGRADE_AT_LATCH = ("fp_preinjection", "early_correct", "early", "early_alias")
        for f in flags[flags_base[0]:]:
            if (f["verdict"] in REGRADE_AT_LATCH
                    and f["t"] >= (report.get("injected_at") or 0)):
                nv = verdict_live(f["error"], f["t"], f.get("utt", ""))
                if nv != f["verdict"]:
                    was = f["verdict"]
                    f["verdict_was"] = was
                    f["verdict"] = nv
                    print(f"[t{ro.tick}] FLAG {f['error']}@t{f['t']} re-graded "
                          f"{was} -> {nv}", flush=True)
                    if (nv in CORRECT_VERDICTS and not armed[0] and not COOK_GRADER
                            and not report.get("prevented")):
                        armed[0] = True
                        report["armed_at"] = ro.tick

    # ---------- LIVE GRADING WINDOW (user ruling 2026-09-20) -------------------------
    # lo: the first observable deviated act of THIS run (dev_live, latched below).
    # hi: the tick the mistake commits in THIS run; if it never commits, the release /
    #     archive tick. Published as report["window"], stamped window_rule so no offline
    #     consumer can mistake one rule for the other (F4a).
    def publish_window(lo, why):
        w = windows_lib.open_window(lo, report.get("case_offsets"), WINDOW_MODE)
        report["window"] = w
        report["window_rule"] = WINDOW_MODE
        report["window_lo_why"] = why
        if LIVEW and report.get("committed_at") is not None:
            # the mistake landed before its first deviated command even reached the log
            close_window(report["committed_at"], "commit")
        return w

    def close_window(t, why):
        """Close the live window at `t` (never before its own lo, never twice)."""
        if not LIVEW:
            return
        w = report.get("window")
        if not w or w[1] != windows_lib.OPEN:
            return
        windows_lib.close_at(w, t)
        report["window_closed"] = {"t": w[1], "why": why}
        print(f"[t{ro.tick}] WINDOW closed at t{w[1]} ({why}) -> {w}", flush=True)

    def sync_window(final=False):
        """Keep the published window truthful at every moment a verdict is decided.
        Once the takeover is over (released, skipped, or the episode has ended) the last
        moment a correct call could have prevented the harm has passed: close at the
        commit if one was recorded, otherwise at the release / archive tick."""
        if not LIVEW or family != "error" or not injected[0]:
            return
        w = report.get("window")
        if not w or w[1] != windows_lib.OPEN:
            return
        if not (released[0] or skipped[0] is not None or final):
            return
        t_end = report.get("released_at")
        if t_end is None:
            t_end = ro.tick
        if report.get("committed_at") is None:
            # the attention errors (forget_next_step, leave_unattended) and the omission
            # commit exactly when the takeover ends, which the per-tick poll runs too
            # early to see — re-read the plate verdict now that the release is known.
            try:
                if plate_verdict().get("error_committed") is True:
                    # bookkeeping only — the A5 voice schedule is NOT told about a commit
                    # this late, so episode dynamics are untouched by the window change.
                    report["committed_at"] = t_end
            except Exception:
                pass
        if report.get("committed_at") is not None:
            close_window(report["committed_at"], "commit")
        else:
            close_window(t_end, "released" if released[0] else
                         ("skipped" if skipped[0] is not None else "episode_end"))

    def watch_dev_start():
        if dev_live[0] is not None or not injected[0]:
            return
        if first_dev_cmd is None:          # qa_probe: nothing is DONE differently, the
            dev_live[0] = report["injected_at"]     # question itself is the deviation
            report["dev"] = dev_live[0]
            report["window"] = [dev_live[0], dev_live[0]]
            report["window_rule"] = WINDOW_MODE
            report["window_lo_why"] = "question_is_the_deviation"
            _regrade_preinjection()
            return
        if first_dev_cmd.startswith(("idle", "__")):
            # W3a: macro deviations (`__wander__ N`) expand at execution
            # time — the literal token never appears in the log, exactly like idle, so
            # both macro episodes had dev_live=None and every correct flag graded
            # fp_preinjection. Same treatment as idle: the deviation begins at injection.
            # An idleness deviation (forget_next_step) never logs a command — the cook
            # simply stops — so waiting for `idle N` to appear in the log left dev_live
            # None forever and every CORRECT forget_next_step flag graded
            # fp_preinjection (measured: 6/6 correct flags rejected on
            # nt__forget_next_step_easyM_5). The absence begins at injection.
            dev_live[0] = report["injected_at"]
            report["dev"] = dev_live[0]
            publish_window(dev_live[0], "idle_type_opens_at_injection")
            print(f"[t{ro.tick}] deviation LIVE at t{dev_live[0]} (idle-type) "
                  f"window={report['window']}", flush=True)
            _regrade_preinjection()
            return
        for l in ro.log:
            if l["cmd"] == first_dev_cmd and l["status"] in ("running", "waiting", "done"):
                if (l.get("t_issue") or 0) >= report["injected_at"]:
                    dev_live[0] = l.get("t_issue") or ro.tick
                    report["dev"] = dev_live[0]
                    publish_window(dev_live[0], "first_observable_deviated_act")
                    print(f"[t{ro.tick}] deviation LIVE at t{dev_live[0]} "
                          f"window={report['window']}", flush=True)
                    _regrade_preinjection()
                    return
        # W17 (2026-08-12 audit): dev_section[0] can be a command the cook completed JUST
        # BEFORE injection — strip_done_task_prefix then treats step 1 as done while this
        # watcher waits for a re-issue that will never come (69-tick latch delay, 3
        # correct flags turned into fp_preinjection). If a short grace passes and the
        # first dev command was already done at injection time, the deviation has been
        # live since injection.
        if (ro.tick > report["injected_at"] + 6
                and any(l["cmd"] == first_dev_cmd and l["status"] == "done"
                        and (l.get("t_done") or 0) <= report["injected_at"] + 2
                        for l in ro.log)):
            dev_live[0] = report["injected_at"]
            report["dev"] = dev_live[0]
            publish_window(dev_live[0], "first_step_predated_injection")
            print(f"[t{ro.tick}] deviation LIVE at t{dev_live[0]} (first step predated "
                  f"injection) window={report['window']}", flush=True)
            _regrade_preinjection()

    # ---------- COMPOSITE bookkeeping (2026-08-19) -----------------------------------
    # A composite episode archives each sub-injection's whole record the moment it
    # releases (or is skipped), resets the singleton state, and loads the next sub.
    # The per-injection report keys are POPPED into the archive so the next sub starts
    # from the same blank slate a fresh episode would.
    PER_INJ_KEYS = ("injected_at", "injection_skipped", "dev", "window", "armed_at",
                    "prevention_armed_at", "released_at", "released_reason",
                    "max_hold_release", "anchor_v5", "smoke_narrowed", "prevented", "prevented_at", "prevention_missed",
                    "detected_early_missed", "convinced", "task_finished_at",
                    "takeover_splice", "prevention_splice", "goal_extended",
                    "intent_announced_at", "intent_announcement", "intent_contested",
                    "err_live", "case_offsets", "cook_calls", "cook_detected_at", "committed_at",
                    "queue_restored", "queue_not_restored", "lu_live_at",
                    "window_rule", "window_lo_why", "window_closed")

    def archive_injection():
        fl = list(flags[flags_base[0]:])
        fv = next((f for f in fl if f["verdict"] in CORRECT_VERDICTS), None)
        _cook_hits = [c for c in (report.get("cook_calls") or []) if c.get("call") == "correct"
                      and c.get("credits") is not False]
        _cook_ok = bool(_cook_hits)
        _det_ts = [c["t"] for c in _cook_hits] + ([fv["t"]] if fv else [])
        _det_t = min(_det_ts) if _det_ts else None
        entry = {"idx": (body_spec[0]["idx"] if (SCHED_V5 and body_spec[0] is not None) else sub_idx[0]),
                 "sub": case.get("name"), "family": family,
                 # G1: the REALISED type is what the scorer and every per-type table read; the designed label
                 # and the creditable flag names are kept beside it so nothing is lost.
                 "err": realized_err(case), "err_designed": case.get("err"),
                 "flag_names": flag_names_for(case) if family == "error" else [],
                 "gt_kind": case.get("gt_kind"),
                 "injected": injected[0], "released": released[0],
                 "skipped": skipped[0], "archived_at": ro.tick, "flags": fl,
                 "false_positives": [f for f in fl
                                     if f["verdict"] not in NEUTRAL_VERDICTS],
                 # DETECTION CREDIT (user ruling 2026-09-12): the cook's own judgement counts as
                 # well as the engine's flag verdict, so a correct warning under a wrong flag name
                 # is no longer thrown away. `flagged_valid` is the AUTHORITATIVE field every
                 # scorer reads; the engine-only value is kept beside it so the two graders stay
                 # comparable and every published number can be re-derived.
                 "flagged_valid": bool(fv) or bool(_cook_ok),
                 "flagged_valid_engine": bool(fv),
                 "detected_by": ("both" if (fv and _cook_ok) else
                                 "cook" if _cook_ok else "engine" if fv else None),
                 "detected_at": _det_t,
                 "latency": (_det_t - dev_live[0])
                            if (_det_t is not None and dev_live[0] is not None) else None,
                 "latency_engine": (fv["t"] - dev_live[0])
                                   if (fv and dev_live[0] is not None) else None,
                 "cook_wrong_calls": [c for c in (report.get("cook_calls") or [])
                                      if c.get("call") == "wrong"
                                      and c.get("why") != "no assistant line since the injection"],   # F3
                 "detected_early_missed": bool(report.get("detected_early_missed")),
                 "prevented": bool(report.get("prevented")),
                 "convinced": bool(report.get("convinced")),
                 "dev_section": list(dev_section), "item": case.get("item"),
                 "cook_calls": list(report.get("cook_calls") or []),
                 "cook_detected_at": report.get("cook_detected_at")}
        if family in ("qa_probe", "qa_proactive"):
            entry["qa"] = dict(qa)
            if family == "qa_probe" and qa.get("asked_at") is not None and qa.get("answer") is None:
                late_qa.append((entry, case.get("gt_kind")))
            try:
                from cooksim_bench.scoring.eval_intent import grade_answer
                _v, _w = grade_answer(case.get("gt_kind"), qa.get("gt_at_ask"),
                                      qa.get("gt_at_answer"), qa.get("answer"))
                entry["qa_grade"] = {"verdict": _v, "detail": _w, "superseded_by": "regrade_qa.json"}
            except Exception as _e:                              # noqa: BLE001
                entry["qa_grade"] = {"verdict": "ungradable", "detail": repr(_e), "superseded_by": "regrade_qa.json"}
        if family == "error":
            try:
                entry["plate_verdict_at_release"] = plate_verdict()
            except Exception:
                pass
        sync_window(final=True)   # LIVE WINDOW: an archived sub-case never carries an open hi
        entry["report_keys"] = {k: report.pop(k) for k in PER_INJ_KEYS if k in report}
        report.setdefault("injections", []).append(entry)
        print(f"[t{ro.tick}] COMPOSITE SUB {entry['idx']} ARCHIVED ({entry['sub']}: "
              f"injected={entry['injected']} released={entry['released']} "
              f"skipped={entry['skipped']})", flush=True)

    def advance_case():
        sub_idx[0] += 1
        injected[0] = released[0] = armed[0] = prevention[0] = False
        skipped[0] = None
        past_since[0] = None
        dev_live[0] = None
        task_finished[0] = False
        contested[0] = False
        soft_wake[0] = False
        qa.clear()
        qa.update({"asked_at": None, "answer": None, "answered_at": None,
                   "gt_at_ask": None, "gt_at_answer": None})
        report["convinced"] = False
        load_case(subs[sub_idx[0]])
        print(f"[t{ro.tick}] COMPOSITE ADVANCE -> sub {sub_idx[0]}/{len(subs) - 1}: "
              f"{case.get('name')} ({family}/{case.get('err') or case.get('gt_kind')})",
              flush=True)

    # ---------- main streaming loop
    prev_json = [""]
    said = None
    rej_seen = [0]

    # ---- NATIVE streaming assistant (python -m cooksim_bench.backends.native_assistant; standing input spec
    # streaming-native-arm-inputs). Nobody polls the model: every tick it gets the new frame
    # (+ a "[t=N]" marker that asks for nothing), every cook line as TTS audio the moment it is
    # said, and whatever it decides to say comes back as an utterance that is delivered through
    # deliver_assistant(), exactly like a polled reply. Clock: at most one frame per
    # COOKSIM_NATIVE_TICK_WALL s (<= 1 fps); a line is charged from its anchor (end of the cook's
    # audio, or the last frame for a self-started line) to its FIRST output token, converted by
    # assist_latency_ticks(); while the model is still producing a line that is already due the
    # world is held, so the time it takes to finish saying it is not charged.
    nat = {"open": False, "last": 0.0, "q": [], "held_s": 0.0, "holds": 0}
    if NATIVE:
        from cooksim_bench.backends.native_assistant import parse_flag as _nat_flag, TICK_WALL as _NAT_TICK_WALL

    def native_step(deliver):
        nonlocal said
        if not nat["open"]:
            sup_client.open(sup_sys)
            nat["open"] = True
        sp = sup_client.speaking()
        if sp is not None and ro.tick >= sp["anchor_tick"] + assist_latency_ticks(sp["latency_s"]):
            _h = time.time()
            sup_client.wait_until_quiet(30.0)
            nat["held_s"] += time.time() - _h
            nat["holds"] += 1
        _w = nat["last"] + _NAT_TICK_WALL - time.time()
        if _w > 0:
            time.sleep(_w)
        nat["last"] = time.time()
        if fbuf:
            sup_client.push_frame(fbuf[-1], ro.tick)
        if said:
            sup_client.push_speech(said, ro.tick)
            said = None
        for u in sup_client.drain():
            if not u["text"]:
                continue
            fl, spoken, marker = _nat_flag(u["text"])
            y = assist_latency_ticks(u["latency_s"])
            nat["q"].append({"t_arrive": max(u["anchor"]["tick"] + y, ro.tick), "t_issue": u["anchor"]["tick"],
                             "x": u["latency_s"], "y": y,
                             "out": {"text": spoken, "flag": fl,
                                     "native": {"raw": u["text"], "flag_marker": marker,
                                                "anchor": u["anchor"]["kind"],
                                                "cook_line": u["anchor"].get("cook_line"),
                                                "latency_s": u["latency_s"], "done_s": u["done_s"],
                                                "interrupted": u["interrupted"]}}})
        if pauses and getattr(ro, "_pause", 0) != 0:
            return
        for q in [q for q in nat["q"] if ro.tick >= q["t_arrive"]]:
            nat["q"].remove(q)
            deliver(q["out"], None, q["t_issue"], q["x"], q["y"])
            if said:                                  # the cook's answer to that line, heard at once
                sup_client.push_speech(said, ro.tick)
                said = None

    # STALL GUARD. A cook that cannot make progress does not stop talking, so an episode can
    # burn its whole budget standing still: measured at 1011 ticks / 492 conversation turns
    # with the plate never changing. The wall-clock abort eventually catches it, but only after
    # ~25 minutes of GPU per stuck run, and a livelocked episode is data either way — so end it
    # as soon as the WORLD stops changing and record why.
    # The fingerprint is deliberately world-only (position, hands, plate, every station's
    # contents AND its progress): a pan that is still cooking counts as progress, so a long
    # legitimate wait is never mistaken for a stall.
    STALL_TICKS = int(os.environ.get("COOKSIM_STALL_TICKS", 220))

    def world_key():
        c = ro.game.cooks[0]
        st = []
        for (x, y), s_ in sorted(ro.game.stations.items()):
            d = s_.render_state()
            st.append((x, y, d.get("status"), round(d.get("progress") or 0, 2),
                       tuple(sorted(i.get("name") for i in (d.get("contents") or []))),
                       (d.get("item") or {}).get("name") if d.get("item") else None))
        return json.dumps([list(c.pos), _norm_item(c.holding), plate_list(ro.game), st,
                           sorted(str(_norm_item(i)) for i in ro.game.counter_items.values() if i)],
                          sort_keys=True, default=str)

    stall = {"key": None, "since": 0}

    # IDLE-COOK GUARD. The world-change stall guard above does not catch the commonest
    # non-termination: the queue is empty, the cook issues no actions, but it keeps TALKING, so
    # the odd item still lands and the world technically changes. Measured on iter1: 108
    # conversation turns across 245 ticks, the cook insisting "I'm literally walking to the bun
    # source" while body_status() told it "CURRENT ACTION: (none - your body is idle)". All
    # three episodes died on the wall clock with the dish unfinished.
    # This is generous on purpose - it ends a conversation that cannot converge, and never a
    # cook that is merely slow.
    IDLE_TURNS = int(os.environ.get("COOKSIM_IDLE_TURNS", 40))
    idle_turns = [0]

    while ep_alive[0]:
        if time.time() - t0 > (WALL_BUDGET * (2 if subs else 1)):
            report["aborted"] = "wall_clock"
            break
        if idle_turns[0] >= IDLE_TURNS:
            report["aborted"] = "cook_idle"
            report["cook_idle"] = {"t": ro.tick, "turns_without_actions": idle_turns[0]}
            print(f"[t{ro.tick}] COOK IDLE - {idle_turns[0]} consecutive turns with an empty "
                  f"queue and no actions issued; ending episode", flush=True)
            break
        _wk = world_key()
        if _wk == stall["key"]:
            if ro.tick - stall["since"] >= STALL_TICKS:
                report["aborted"] = "stalled"
                report["stall"] = {"t": ro.tick, "ticks_without_change": ro.tick - stall["since"]}
                print(f"[t{ro.tick}] STALLED — nothing in the kitchen changed for "
                      f"{ro.tick - stall['since']} ticks; ending episode", flush=True)
                break
        else:
            stall["key"], stall["since"] = _wk, ro.tick
        snap()
        if SCHED_V5:
            schedule_v5()
        else:
            maybe_inject()
        track_plate()
        if (SCHED_V5 and family == "error" and injected[0] and not released[0]
                and report.get("committed_at") is None):
            # A5: the commit tick of the live mistake (the voice slot keeps clear of it); also recorded for audits
            try:
                if plate_verdict().get("error_committed") is True:
                    report["committed_at"] = ro.tick
                    quiet_events["commit"] = ro.tick
                    # LIVE WINDOW (2026-09-20): the harm has landed — this tick is the last
                    # one on which a correct call could still have prevented it.
                    close_window(ro.tick, "commit")
            except Exception:
                pass
        # MAXIMUM HOLD (user ruling 2026-08-18): the takeover exists to make the error
        # happen; once the injected error has COMMITTED in the world it has done its job,
        # and holding the queue any longer only makes the cook's accepted repairs vanish
        # silently while its hands finish serving. Release the moment the plate ledger says
        # the error is committed, whether or not the assistant ever named it. Detection is
        # still graded exactly as before (flags, windows, DT); only the world stops waiting.
        if (HARDQ and injected[0] and not released[0] and family == "error"
                and report.get("max_hold_release") is None):
            why = None
            # (a) THE DEVIATED BLOCK HAS BEEN EXECUTED. Injection spliced a concrete list of
            #     commands (dev_section, sids dev_block_sids) into the queue; "the error has
            #     happened in the world" is exactly: the executor has moved past every one of
            #     them (each reached a terminal status -- done, rejected, skipped, failed).
            #     Never early: it fires only after the LAST deviated command is over. Never
            #     blind: every error type has a block (idle 60; the walk-away loop; the second
            #     scoop; the whole rice-less prep). For omit_subgoal the block ends with the
            #     serve, so it releases at the serve -- i.e. never before the commit.
            if dev_block_sids:
                future_dev = any(k in dev_block_sids for _c, k in ro.cmds[ro.i + 1:])
                cur_kind = ro.cmds[ro.i][1] if 0 <= ro.i < len(ro.cmds) else None
                cur_open = False
                if cur_kind in dev_block_sids:
                    rows = [l for l in ro.log if l.get("i") == ro.i]
                    cur_open = (not rows) or rows[-1].get("t_done") is None
                started = any(l.get("kind") in dev_block_sids for l in ro.log)
                if started and not future_dev and not cur_open:
                    why = "deviated_block_executed"
            # (b) THE PLATE PROVES IT EARLIER, for the error types where the plate ledger is
            #     decisive at the moment of the harmful act (a foreign or raw item plated;
            #     both swapped items plated in the wrong order). Not the omit branch: that
            #     one is true from the first scoop of every episode and is only meaningful
            #     at the end.
            if why is None and SCHED_V5 and dev_block_sids:
                # round-3 audit: an omission block composed from the recipe while the cook held the
                # plate issued 8-18 refused commands and held the cook's hands to the end (3 episodes
                # invalid, one to a fire). Eight refused block commands in a row: the mistake cannot be
                # executed in this kitchen — hand the cook back. (Threshold 8: stale template steps such as
                # "turn on" an already-lit station are refused harmlessly in ordinary blocks.)
                _blk = [l for l in ro.log if l.get("kind") in dev_block_sids and l.get("t_done") is not None]
                _tail = _blk[-8:]
                if len(_tail) == 8 and all(l.get("status") in ("rejected", "illegal-skipped", "orphan-skipped", "failed")
                                           for l in _tail):
                    why = "block_unexecutable"
            if why is None and SCHED_V5 and (case.get("fn") or case.get("err")) == "leave_unattended":
                # round-2 audit: 13 holds kept the cook's hands 5-39 ticks after the food began to
                # smoke. The mistake has committed at the smoke; hand the rescue back at once.
                try:
                    if plate_verdict().get("error_committed") is True:
                        why = "smoke_committed"
                except Exception:
                    pass
            # B1 (fix-firing): NOT wrong_order. Its verdict commits on the first scoop (b down with a not under it), and
            # releasing there dropped the second scoop: 8 of 9 committed wrong_orders in the master LLM-cook run left the
            # tortilla off the plate for good, turning the swap into an omission and starving every later sub-case keyed
            # to the tortilla. It releases when its whole block has run (deviated_block_executed) or on the cook's accept.
            if why is None and (case.get("fn") or case.get("err")) in (
                    "add_foreign", "misidentify_assemble", "serve_raw"):
                try:
                    if plate_verdict().get("error_committed") is True:
                        why = "plate_committed"
                except Exception:
                    pass
            if why:
                released[0] = True
                report["max_hold_release"] = {"t": ro.tick, "why": why}
                report.setdefault("released_at", ro.tick)
                report["released_reason"] = report.get("released_reason") or f"max_hold_{why}"
                print(f"[t{ro.tick}] RELEASED (max hold: {why}) - control back to the human",
                      flush=True)
                if SCHED_V5:
                    stop_hands()
        # F5: completion is a property of the world, not of the conversation — poll it
        # every tick so a silent persona releases the moment its task is actually done.
        poll_intent_release()
        # COMPOSITE: the moment sub k is over (released, or skipped without firing),
        # archive it and arm sub k+1. The last sub is archived at episode end.
        if SCHED_V5:
            if body_spec[0] is not None and (released[0] or skipped[0] is not None):
                stop_hands()
                archive_injection()
                free_body_slot()
        elif (subs is not None and sub_idx[0] < len(subs) - 1
                and (released[0] or skipped[0] is not None)):
            archive_injection()
            advance_case()
        if SCHED_V5:
            open_voice_if_quiet()                     # A5: after this tick's commit / release are known
        # Deliver the SYSTEM TASK on the injection tick itself. ro.step() runs at the end
        # of this loop body, so the cook gets to re-queue before the clean plan advances.
        if soft_wake[0]:
            soft_wake[0] = False
            said = human_turn(event=True)
        watch_dev_start()
        # ---- ASSISTANT LATENCY (user ruling 2026-09-04): a reply that took x seconds of wall
        # time arrives y = ceil(x / TICK_SECONDS) ticks after it was requested; the world keeps
        # stepping and the cook keeps executing its queue in between. One call in flight at a
        # time: polls are skipped while a reply is pending, so the effective cadence is
        # max(POLL, latency). COOKSIM_ASSIST_LATENCY: unset/off = arrive at once (legacy),
        # "measured" = from the wall clock, an integer = a fixed tick delay per reply.
        def deliver_assistant(out, said_at_issue, t_issue, x_s, y_ticks):
            nonlocal said
            said = said_at_issue
            _raw_ticks = int(math.ceil(max(0.0, x_s) / TICK_SECONDS)) if y_ticks else 0
            report.setdefault("assist_latency", []).append(
                {"t_issue": t_issue, "t_arrive": ro.tick, "latency_s": round(x_s, 3), "ticks": y_ticks,
                 "capped": bool(y_ticks and _raw_ticks > y_ticks)})
            out = lift_nested_sup_keys(out)
            prev_json[0] = json.dumps(out) if out else prev_json[0]
            said = None
            text = (out.get("text") or "").strip()
            fl = out.get("flag")
            v = None
            if isinstance(fl, dict) and fl.get("error"):
                quiet_events["flag"] = ro.tick          # A5: no question opens on (or right after) a flag
                name = str(fl["error"]).strip()
                seen_names = {f["error"] for f in flags[flags_base[0]:]}
                if name in seen_names:
                    # W5 (2026-08-12 audit): the old exemption re-graded AND re-appended
                    # the correct name every poll while unarmed — 82 false alarms from one
                    # continuously-true detection. A repeat is only worth a new row when
                    # its VERDICT CLASS changes (e.g. early -> valid, which arms).
                    v2 = (verdict_live(name, ro.tick, text)
                          if name in flag_names_for(case) and not armed[0] else None)
                    last = next((f["verdict"] for f in reversed(flags[flags_base[0]:])
                                 if f["error"] == name), None)
                    if v2 is not None and v2 != last:
                        v = v2
                        flags.append({"t": ro.tick, "error": name, "verdict": v,
                                      "utt": text[:200]})
                        print(f"[t{ro.tick}] FLAG {name} -> {v} (re-graded)", flush=True)
                    else:
                        flag_repeats[0] += 1
                else:
                    v = verdict_live(name, ro.tick, text)
                    flags.append({"t": ro.tick, "error": name, "verdict": v,
                                  "utt": text[:200]})
                    print(f"[t{ro.tick}] FLAG {name} -> {v}", flush=True)
                # X1 (2026-08-13 audit): this arming block lived only in the first-sighting
                # branch above, so a RE-GRADED valid flag (the correct name, first raised
                # before the window) never armed the release gate — the takeover could not
                # end, 52 accepted plan updates were silently discarded across two
                # episodes, and a won argument still served the wrong plate. Hoisted so
                # both paths arm.
                if (v in CORRECT_VERDICTS and v != "early_correct" and not armed[0]
                        and not COOK_GRADER and not report.get("prevented")):
                    # user ruling 2026-08-18: an in-window TWIN name is a correct detection
                    # for the world, not only for the scorer -- it arms the release gate.
                    armed[0] = True
                    report["armed_at"] = ro.tick
                    print(f"[t{ro.tick}] ARMED ({v} flag)", flush=True)
                elif v == "early_correct" and not armed[0] and not prevention[0] and not COOK_GRADER:
                    prevention[0] = True
                    report["prevention_armed_at"] = ro.tick
                    print(f"[t{ro.tick}] PREVENTION armed (early correct flag)", flush=True)
            if INTENT and v is not None:
                intent_flags.append({"t": ro.tick, "error": name, "verdict": v,
                                     "utt": text[:200]})
                if v == "intent_false_alarm":     # a challenge against the live intent
                    contested[0] = True
                    report["intent_contested"] = True
            # QA: the FIRST thing the assistant says after the question was asked IS the
            # answer under test. Ground truth is read from the engine at both moments —
            # the kitchen can move between question and answer, and an answer that was
            # true when it was given must not be graded wrong for that.
            # 2026-09-05 (v4 audit): a reply whose call was already in flight when the question was
            # asked never saw the question — 496 of 914 v4 "answers" were such lines (judged evasive).
            # Only a reply ISSUED at or after the ask is the answer under test.
            if late_qa and text and late_qa[0][0]["qa"].get("answer") is None \
                    and t_issue >= (late_qa[0][0]["qa"].get("asked_at") or 0):
                _entry, _gk = late_qa.pop(0)
                _lq = _entry["qa"]
                _lq["answer"] = text
                _lq["answered_at"] = ro.tick
                _lq["answered_after_release"] = True
                ro._grading_answer = text
                try:
                    from cooksim_bench.suite.gen_intent_cases import ground_truth
                    _lq["gt_at_answer"] = ground_truth(_gk, ro.game, ro)
                except Exception as _e:  # noqa: BLE001
                    _lq["gt_at_answer"] = {"error": f"{type(_e).__name__}: {_e}"}
                ro._grading_answer = None
                try:
                    from cooksim_bench.scoring.eval_intent import grade_answer
                    _v, _w = grade_answer(_gk, _lq.get("gt_at_ask"), _lq.get("gt_at_answer"), text)
                    _entry["qa_grade"] = {"verdict": _v, "detail": _w, "superseded_by": "regrade_qa.json"}
                except Exception:  # noqa: BLE001
                    pass
                print(f"[t{ro.tick}] QA ANSWER captured (late, into the archived question)", flush=True)
            elif (INTENT and family == "qa_probe" and qa["asked_at"] is not None
                    and qa["answer"] is None and text and t_issue >= qa["asked_at"]):
                qa["answer"] = text
                qa["answered_at"] = ro.tick
                # audit F9 (2026-08-28): the answer must not be visible to its own truth function
                ro._grading_answer = text
                qa["gt_at_answer"] = intent_gt()
                ro._grading_answer = None
                print(f"[t{ro.tick}] QA ANSWER captured", flush=True)
            report["conversation"].append({"speaker": "vlm", "t": ro.tick, "text": text,
                                           "round": RND[0], "vlm_json": out,
                                           "flag_verdict": v, "engine": engine_snap()})
            if text:
                comm.append({"t": ro.tick, "speaker": "vlm (you)", "text": text})
                convo.append({"speaker": "vlm", "text": text})
                said = human_turn()
                # the cook stands still while it SPEAKS (listening stays free); the world
                # keeps running underneath — ro._pause only makes the cook issue STAY
                add_speech_pause("human", said)
                # the assistant's line is heard immediately; it costs no game time
                add_speech_pause("vlm", text)

        if (pending_reply[0] is not None and ro.tick >= pending_reply[0]["t_arrive"]
                and (not pauses or getattr(ro, "_pause", 0) == 0)):
            pr = pending_reply[0]
            pending_reply[0] = None
            deliver_assistant(pr["out"], pr["said"], pr["t_issue"], pr["x"], pr["y"])
        if SCHED_V5:
            # H2 (ruling 2026-09-09): the cook's own thinking never waits for the assistant. Its
            # idle turn runs on the poll cadence whether or not a reply is in flight, and the
            # engine's idle tail does not tick while a reply is pending.
            _idle_v5 = (ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
                        and getattr(ro, "_wait", 0) == 0 and getattr(ro, "_pause", 0) == 0
                        and getattr(ro, "_wander", 0) == 0)
            # `said` is the cook's latest line waiting to ride the next assistant call; while a
            # reply is in flight no call is issued, so `said` would stay set and this gate would
            # let the cook think exactly ONCE per in-flight reply (sample audit 2026-09-09: 10
            # episodes frozen 48-98 ticks behind 84-124 s Gemini calls, 5 of them burned).
            if (ro.tick % POLL == 0 and (said is None or pending_reply[0] is not None) and _idle_v5
                    and (not pauses or getattr(ro, "_pause", 0) == 0)):
                _s2 = human_turn(idle=True)
                said = _s2 if _s2 else said
            if pending_reply[0] is not None:
                ro._idle_since = None
        if NATIVE:
            native_step(deliver_assistant)
        if (not NATIVE and pending_reply[0] is None and ro.tick % POLL == 0
                and (not pauses or getattr(ro, "_pause", 0) == 0)):
            idle = (ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
                    and getattr(ro, "_wait", 0) == 0 and getattr(ro, "_pause", 0) == 0
                    and getattr(ro, "_wander", 0) == 0)
            if said is None and idle and not SCHED_V5:
                said = human_turn(idle=True)
            if CREATIVE and said is None:
                nrej = sum(1 for l in ro.log
                           if l["status"] in ("rejected", "illegal-skipped", "waiting"))
                if nrej > rej_seen[0]:
                    rej_seen[0] = nrej
                    said = human_turn(event=True)
            _x_s = 0.0
            if SCRIPTED_ASSIST:
                out = scripted_assistant_out()
            elif SILENT_ASSISTANT:
                # SILENT floor anchor (BENCHMARK_DESIGN §5.2): the assistant seat never speaks and
                # never flags; everything else (cook, errors, questions, gates) runs unchanged.
                out = {}
            else:
                _t_wall = time.time()
                out = sup_turn(sup_client, sup_sys, prev_json[0], comm, said, fbuf[-WINDOW:], ro.tick,
                               text_obs=(sup_obs_text() if obs_mode == "text_state_map" else None),
                               state_extra=(sup_state_extra() if obs_mode == "agent+++" else None))
                _x_s = time.time() - _t_wall
            _x_s = _x_s if not (SILENT_ASSISTANT or SCRIPTED_ASSIST) else 0.0
            _y = assist_latency_ticks(_x_s)
            if _y > 0:
                pending_reply[0] = {"out": out, "said": said, "t_issue": ro.tick, "x": _x_s, "y": _y,
                                    "t_arrive": ro.tick + _y}
                said = None                       # the cook's line rode this call
            else:
                deliver_assistant(out, said, ro.tick, _x_s, 0)
        if not ro.step():
            stamp_why()
            break
        stamp_why()
    if pauses:
        report.update(pauses_enabled=True, pauses=pauses_log,
                      hesitations=getattr(ro, "hesitations", []))
    # EXIT SURVEY (v2 IQ source; user ruling: OPTION A — plain questions, labelled
    # endpoints, NO scale coaching; per-persona IQ never cross-ranked). One extra cook
    # call, same episode context, zero game ticks. Parse failure: one retry, then None —
    # a judge is never substituted.
    if persona_v2 is not None and not SCRIPTED_COOK:
        _SURVEY = (
            "The episode is over. Answer as yourself, about this assistant:\n"
            "(1) Overall, how satisfied were you with this assistant? "
            "1-7 (1 = very unsatisfied, 7 = very satisfied).\n"
            "(2) Would you choose to work with this assistant again on a similar task? "
            "1-7 (1 = definitely not, 7 = definitely yes).\n"
            'Reply JSON only: {"satisfaction": n, "reuse": n, "comment": "one sentence"}')
        survey = None
        for _try in range(2):
            try:
                _ctx = human_context(state, ro, convo, None, game=ro.game,
                                     reply_override=_SURVEY)
                _h = gemini_text(cook_client, _ctx, _SURVEY)
                _sat, _reu = _h.get("satisfaction"), _h.get("reuse")
                if isinstance(_sat, (int, float)) and isinstance(_reu, (int, float)):
                    survey = {"schema": "exitq-v1", "satisfaction": int(_sat),
                              "reuse": int(_reu),
                              "comment": str(_h.get("comment") or "")[:300],
                              "elicitation": "live"}
                    break
            except Exception as _e:                              # noqa: BLE001
                print(f"[end] exit survey attempt {_try + 1} failed: {_e!r}", flush=True)
        report["exit_survey"] = survey
        print(f"[end] EXIT SURVEY -> {survey}", flush=True)
    if LIVEW:
        sync_window(final=True)                       # close the live window on the last tick
    if report.get("window") and report["window"][1] == 10 ** 9:
        report["window"][1] = ro.tick                 # clamp any still-open window to episode end
    if SCHED_V5:
        if body_spec[0] is not None:
            archive_injection()                       # the live body sub-case at episode end
        if voice["spec"] is not None:
            skip_spec(voice["spec"], "cook_did_not_ask"); voice["spec"] = None
        for sp in SPECS:
            if sp["state"] == "pending":
                skip_spec(sp, "anchor_never_reached" + (f" (waiting: {sp['wait_note']})" if sp.get("wait_note") else ""))
            elif sp["state"] == "stage_passed":
                skip_spec(sp, f"stage_passed_at_t{sp.get('passed_at')}")
        report["injections"] = sorted(report.get("injections") or [], key=lambda e: e.get("idx", 0))
        # F7 (round-3 review): a question asked so close to the episode end that no reply could arrive (the serve
        # ended the episode within QA_LATE_TICKS of the ask, and nothing answered it) was never measured: it is graded
        # `asked_too_late` (unscored, not reached) — the verdict python -m cooksim_bench.scoring.regrade_qa already gives it offline — not a
        # 0. An early question the assistant simply never answered stays `no_answer` (0).
        _late_t = int(os.environ.get("QA_LATE_TICKS", 10))
        for _e in report["injections"]:
            _q = _e.get("qa") or {}
            if (_e.get("family") == "qa_probe" and _q.get("asked_at") is not None and _q.get("answer") is None
                    and ro.tick - _q["asked_at"] <= _late_t):
                _q["asked_too_late"] = True
                _e["qa_grade"] = {"verdict": "asked_too_late",
                                  "detail": f"asked {ro.tick - _q['asked_at']} ticks before the episode end, no reply "
                                            f"(<= {_late_t}): not reached, unscored"}
        report["errors_planned"] = sum(1 for sp in SPECS if sp["family"] == "error")
        report["errors_offered"] = sum(1 for sp in SPECS if sp["family"] == "error" and sp["fired_at"] is not None)
        report["actions"] = [dict(l) for l in ro.log]         # E4: the whole action log, to the last tick
        report["scheduler"] = "v5"
        report["composite"] = {"name": composite_name, "cell": composite_cell,
                               "n_subs": len(subs), "subs": [s_.get("name") for s_ in subs],
                               "reached": sum(1 for sp in SPECS if sp["fired_at"] is not None),
                               "was_composite": WAS_COMPOSITE}
    elif subs is not None:
        archive_injection()                           # the final live sub
        report["composite"] = {"name": composite_name, "cell": composite_cell,
                               "n_subs": len(subs),
                               "subs": [s_.get("name") for s_ in subs],
                               "reached": sub_idx[0] + 1}
    if hasattr(sup_client, "stats"):
        report["assistant_backend"] = sup_client.stats()
    # which model actually spoke as the cook (ablation 2026-09-22); absent field = the
    # shared Gemini client, i.e. every run before this option existed
    if hasattr(cook_client, "stats"):
        report["cook_backend"] = cook_client.stats()
    else:
        from cooksim_bench.backends.genai_client import default_model as _dm
        report["cook_backend"] = {"model": _dm(), "calls": None}
    if NATIVE:
        report["assistant_mode"] = "native"
        report["native_assistant"] = sup_client.native_report()
        report["native_assistant"]["world_held"] = {"holds": nat["holds"], "seconds": round(nat["held_s"], 2)}
        report["native_assistant"]["undelivered"] = [q["out"] for q in nat["q"]]
        sup_client.close()
    if ro.outcome is None and report.get("aborted") in ("cook_idle", "stalled", "wall_clock"):
        # an episode the harness had to end (a cook talked into a livelock, a kitchen that stopped
        # changing) is a lost dish, not an unscored one: it counts as a timeout
        ro.outcome = "timeout"
        report["outcome_from_abort"] = report["aborted"]
    report["window_rule_run"] = WINDOW_MODE          # which window rule graded this episode
    report.update(kitchen_feedback=fb_log,          # ECR-13: what the kitchen told the cook
                  outcome=ro.outcome, ticks=ro.tick, flags=flags,
                  flag_repeats=flag_repeats[0], injected=injected[0],
                  released=released[0],
                  # X9: alias_in_window is neutral by verdict_live's own definition —
                  # it was excluded on the pages but counted as an FP here and in
                  # eval_assistant, so the report and the site disagreed. F4b: the set is
                  # now imported from checks.NEUTRAL_VERDICTS so the report, eval_assistant
                  # and the page builders CANNOT drift apart again.
                  false_positives=[f for f in flags
                                   if f["verdict"] not in NEUTRAL_VERDICTS],
                  flagged_valid=any(f["verdict"] in CORRECT_VERDICTS for f in flags))
    fv = next((f for f in flags if f["verdict"] in CORRECT_VERDICTS), None)
    report["latency"] = (fv["t"] - dev_live[0]) if (fv and dev_live[0] is not None) else None
    if INTENT:
        # JUDGEMENT OF THE DELIVERY (the point of these scenarios): the cook did nothing
        # wrong, so the episode is a success when the right dish still reaches the pass.
        # Everything else grades the ASSISTANT's conduct around a legitimate deviation.
        report.update(
            family=family, intent_label=case.get("intent_label"),
            intent_question=case.get("question"), intent_seq=dev_section,
            rationale=case.get("rationale"),
            delivery_ok=(ro.outcome == "won"),
            intent_carried_out=(None if family in ("qa_probe", "qa_proactive") else intent_done()),
            intent_announced_at=report.get("intent_announced_at"),
            intent_false_alarms=[f for f in intent_flags
                                 if f["verdict"] == "intent_false_alarm"],
            false_alarms_after_announcement=[
                f for f in intent_flags
                if f["verdict"] == "intent_false_alarm"
                and report.get("intent_announced_at") is not None
                and f["t"] >= report["intent_announced_at"]],
            # F4b: `true_observation` is a NEUTRAL verdict — a flag the kitchen confirms.
            # Without this exclusion it would fall straight into other_false_alarms and
            # reappear on the intent pages as the false alarm it is not.
            other_false_alarms=[f for f in intent_flags
                                if f["verdict"] != "intent_false_alarm"
                                and f["verdict"] not in NEUTRAL_VERDICTS],
            true_observations=[f for f in intent_flags
                               if f["verdict"] == "true_observation"],
            qa=(qa if family in ("qa_probe", "qa_proactive") else None), item=case.get("item"))
        # F5 (2026-08-14 fix batch): the qa grade existed but never reached the record.
        # eval_intent.grade_answer was called only inside eval_intent.score(), which writes
        # into ITS OWN output — so the rollout's report, which is what the pages read, had
        # the question, the answer and both ground-truth snapshots but no verdict, and
        # every qa_probe episode showed up ungraded. It is deterministic and the inputs are
        # right here; grade it at the source.
        if family == "qa_probe":
            try:
                from cooksim_bench.scoring.eval_intent import grade_answer
                _v, _why = grade_answer(case.get("gt_kind"), qa.get("gt_at_ask"),
                                        qa.get("gt_at_answer"), qa.get("answer"))
                report["qa_grade"] = {"verdict": _v, "detail": _why, "superseded_by": "regrade_qa.json"}
                print(f"[end] QA GRADE {_v} — {_why}", flush=True)
            except Exception as _e:                              # noqa: BLE001
                report["qa_grade"] = {"verdict": "ungradable",
                                      "detail": f"grader raised {_e!r}"}
        report["failed_at"] = None if ro.outcome == "won" else "delivery"
    elif SCHED_V5 and WAS_COMPOSITE:
        # E13 (round-2 review): the singleton flags describe only the LAST sub-case, so a composite whose every
        # error fired but whose last sub was archived before the end read injected=False / failed_at=
        # injection_skipped, and eval_assistant stamped it "NOT MEASURED — injection never fired" (26/40 easy+medium
        # episodes). A composite is measured when ANY of its errors fired; the formal scorer reads injections[].
        _errs = [e for e in (report.get("injections") or []) if e.get("family") == "error"]
        _fired = [e for e in _errs if e.get("injected")]
        report["injected_last_sub"] = report.get("injected")
        report["injected"] = bool(_fired)
        report["errors_fired"] = len(_fired)
        if not _fired:
            report["failed_at"] = "injection_skipped"
            report["measured"] = False
            report.setdefault("injection_skipped", {"t": ro.tick, "why": "no error sub-case fired"})
        else:
            report["measured"] = True
            report.pop("injection_skipped", None)
            _det = any(e.get("flagged_valid") or e.get("detected_early_missed") for e in _fired)
            report["failed_at"] = "solved" if ro.outcome == "won" else ("recovery" if _det else "detection")
    elif skipped[0] is not None:
        report["failed_at"] = "injection_skipped"
        report["measured"] = False
    elif not injected[0]:
        report["failed_at"] = "injection_skipped"
        report["measured"] = False
        # W4 (2026-08-12 audit): one episode recorded injected_at=None with NO
        # injection_skipped reason at all — unrecoverable. Always say why.
        if not report.get("injection_skipped"):
            report["injection_skipped"] = {
                "t": ro.tick, "why": "trigger_state_never_reached_before_end"}
    elif ro.outcome == "won":
        report["failed_at"] = "solved"
    else:
        # detection counts as achieved via a window-valid flag OR the early-conviction
        # path (right name + accepted before the window; listed separately, not pooled
        # with window-valid detections)
        detection_ok = report["flagged_valid"] or report.get("detected_early_missed")
        report["failed_at"] = "recovery" if detection_ok else "detection"
    # Which engine produced this episode. Fixes landed mid-sweep and nothing recorded which
    # batch had which, so the pages silently mixed six different engines; see python -m cooksim_bench.engine_stamp.
    try:
        from cooksim_bench.engine_stamp import stamp as _engine_stamp
        # ECR-10: the stamp must describe the code that RAN, not the files on disk at
        # report-write time — this process runs the bytecode it imported at start, and an
        # edit landing mid-episode produced FORWARD-dated stamps that passed every
        # `rev == CURREV` publish gate (measured: one 40-second batch, three revisions).
        end_rev = _engine_stamp()
        # Q10 (suite v3): grade proactive alerts now that the whole conversation and heat timeline exist
        try:
            from cooksim_bench.scoring.qa_v3_lib import grade_proactive
            for _e in (report.get("injections") or []):
                if _e.get("family") == "qa_proactive":
                    _qa = dict(_e.get("qa") or {})
                    if _qa.get("asked_at") is None and (_e.get("report_keys") or {}).get("released_reason") == "request_made":
                        _qa["asked_at"] = (_e.get("report_keys") or {}).get("released_at")   # the request releases the sub on the ask tick
                    try:
                        from cooksim.core.generator import cook_station_for
                        _st = str(cook_station_for(getattr(getattr(ro, "recipe", None), "id", None),
                                                   _e.get("item") or "")).split(".")[-1].lower()
                    except Exception:
                        _st = ""
                    _e["proactive"] = grade_proactive(_qa, _e.get("item"), comm, heat_events, ro.tick,
                                                      station_type=_st, world=getattr(ro, "world", None),
                                                      pace=ro.game.stats.get("pace"))
            if ro.outcome == "burned":
                # the fire ends the episode on the tick it starts, before the station scan sees
                # status "burnt": stamp the open cooked event(s) so burn timing is readable
                _open = [_h for _h in heat_events
                         if _h.get("cooked_at") is not None and _h.get("burnt_at") is None and _h.get("removed_at") is None]
                _burning = []
                for _h in _open:
                    _st = ro.game.stations.get(tuple(_h["pos"]))
                    if _st is not None and getattr(_st, "status", None) == "burnt":
                        _burning.append(_h)
                # round-3 audit: every open heat event used to be stamped, so a steak cooked 24 ticks
                # earlier read as burnt too; fall back to the longest-on-heat event if none is flagged
                for _h in (_burning or sorted(_open, key=lambda x: x["cooked_at"])[:1]):
                    _h["burnt_at"] = ro.tick
            report["heat_events"] = heat_events
            if SCHED_V5:
                try:
                    _pv_ = plate_verdict()
                    _ps = _pv_.get("plate_seq") or []
                    # E6 (round-2 review): "carried out" = the SERVED plate holds the goal's full count of the announced
                    # item. The old test (any unit of it plated after the announcement) passed on the ORDINARY cheese
                    # of an "extra cheese" change: 4 of 4 medium plan-change episodes were recorded carried out while
                    # the extra cheese was never plated. The served plate is the last plate build (a trash starts a
                    # new build); an episode that never served is judged on its final plate and says so.
                    _cut_ = max([0] + list(_pv_.get("plate_resets") or []))
                    _final_ = [(x.get("item"), x.get("state")) for x in _ps if (x.get("t") or 0) >= _cut_]
                    _goal_ = goal_of(ro.game, state)
                    _served_ = ro.outcome in ("won", "wrong_serve")
                    for _e in report.get("injections") or []:
                        if _e.get("family") != "plan_change" or not _e.get("injected"):
                            continue
                        _ge = (SPECS[_e["idx"]]["sub"].get("goal_extend")) or {}
                        _ing, _gst = _ge.get("ingredient"), (_ge.get("state") or "raw")
                        _need = sum(1 for n_, _s in _goal_.contents if n_ == _ing)
                        _have = sum(1 for n_, st_ in _final_ if n_ == _ing and ((st_ or "raw") == _gst or _gst == "raw"))
                        _e["plan_change_carried_out"] = bool(_served_ and _need and _have >= _need)
                        _e["plan_change_check"] = {"ingredient": _ing, "goal_units": _need, "on_served_plate": _have,
                                                   "served": _served_, "final_plate": [n_ for n_, _s in _final_]}
                except Exception as _ex:                              # pragma: no cover
                    print(f"[end] plan_change carried-out check failed: {_ex}", flush=True)
        except Exception as _ex:                                  # noqa: BLE001
            report["proactive_error"] = repr(_ex)[:200]
        report["engine_rev"] = _ENGINE_REV_AT_START or end_rev
        report["grader"] = "cook" if COOK_GRADER else "engine"     # harness A / B (2026-09-12)
        report["pace"] = (ro.game.stats or {}).get("pace", "human") if hasattr(ro, "game") else "human"
        if _ENGINE_REV_AT_START and end_rev != _ENGINE_REV_AT_START:
            report["engine_changed_during_run"] = {
                "started": _ENGINE_REV_AT_START, "on_disk_at_write": end_rev}
            print(f"[end] WARNING: engine changed on disk during this episode "
                  f"({_ENGINE_REV_AT_START} -> {end_rev}); stamped with the STARTING rev",
                  flush=True)
    except Exception:
        pass

    # the engine's own answer to "did the injected error actually happen"
    report.update(plate_verdict())
    if report.get("error_committed") is False and injected[0]:
        print(f"[end] injected {case.get('fn') or case.get('err')} did NOT occur — "
              f"plate order {report.get('plate_first_order')}", flush=True)
    if vdir is not None:
        import shutil, subprocess as sp
        try:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                sp.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "6",
                        "-i", f"{vdir}/%05d.jpg", "-movflags", "+faststart",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", "-crf", "26", video_path], check=True)
                shutil.rmtree(vdir, ignore_errors=True)
            else:
                report["video_error"] = "ffmpeg not found — frames kept at " + vdir
        except Exception as e:
            report["video_error"] = f"{type(e).__name__}: {e}"
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--persona", default=None,
                    help="a v1 persona key (RECOVERY_PERSONAS); or use --persona-v2")
    ap.add_argument("--persona-v2", default=None,
                    help="a persona_v2 ROSTER name; renders the v2 human card + the "
                         "assistant USER PREFERENCES block, writes persona_v2/"
                         "questionnaire_v2 report fields and runs the exit survey")
    ap.add_argument("--paradigm", required=True, choices=["hard", "soft", "hardsoft"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sup-obs", default="frames",
                    choices=["frames", "text_state_map", "agent+++"])
    ap.add_argument("--pauses", action="store_true",
                    help="human-realism pauses: the cook stands still while it SPEAKS "
                         "(listening is free) and hesitates 20%% of steps; the world "
                         "keeps running throughout")
    ap.add_argument("--assist-backend", default=None, choices=["gemini", "openai", "live", "realtime", "anthropic",
                                                              "live_native", "realtime_native"],
                    help="put a DIFFERENT model in the assistant seat (default: the "
                         "shared Gemini client). openai = any /v1/chat/completions "
                         "server (local vLLM); live = one Gemini Live session per episode; "
                         "live_native / realtime_native = the model in its OWN streaming mode "
                         "(no polling; it decides when to speak; python -m cooksim_bench.backends.native_assistant)")
    ap.add_argument("--assist-model", default=None)
    ap.add_argument("--assist-base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--assist-profile", default="default")
    ap.add_argument("--assist-thinking", type=int, default=640)
    ap.add_argument("--assist-max-tokens", type=int, default=4000)
    ap.add_argument("--assist-image-scale", type=float, default=1.0)
    # ---- COOK seat (cook-model ablation, 2026-09-22) -------------------------------------
    # OPTIONAL in every sense: with no --cook-backend the cook is the shared Gemini client
    # exactly as before, so every existing command line behaves identically.
    ap.add_argument("--cook-backend", default=None,
                    choices=["gemini", "openai", "live", "realtime", "anthropic"],
                    help="put a DIFFERENT model in the COOK (simulated human) seat "
                         "(default: the shared Gemini client, GEMINI_MODEL). Same wrapper the "
                         "assistant seat uses; the cook's prompt is untouched")
    ap.add_argument("--cook-model", default=None)
    ap.add_argument("--cook-base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--cook-profile", default="default")
    ap.add_argument("--cook-thinking", type=int, default=640)
    ap.add_argument("--cook-max-tokens", type=int, default=4000)
    ap.add_argument("--family", default="error",
                    choices=["error", "plan_change", "qa_probe", "qa_proactive"],
                    help="error = the classic mistake scenarios; the others are the "
                         "NON-ERROR intent scenarios (the case file overrides this)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)   # a fresh clone has no output tree

    pv2 = None
    if getattr(args, "persona_v2", None):
        from cooksim.interact.persona_v2 import BY_NAME as _PV2
        pv2 = _PV2[args.persona_v2]
        args.persona = f"pv2_{pv2.name}"
    elif not args.persona:
        ap.error("one of --persona / --persona-v2 is required")
    from google.genai import types as _t
    from cooksim_bench.backends.genai_client import make_client
    # the scripted cook (COOKSIM_SCRIPTED_COOK: case verification) calls no model, so no key is needed for it
    client = make_client(120_000) if (os.environ.get("GEMINI_API_KEY") or not SCRIPTED_COOK) else None
    no_render = os.environ.get("COOKSIM_NO_VIDEO") == "1" and args.sup_obs != "frames"
    with sync_playwright() as p:
        if no_render:
            b, page = None, None
            print("[setup] no-render mode: text observation, no video (COOKSIM_NO_VIDEO=1)", flush=True)
        else:
            b = p.chromium.launch(args=GPU)
            page = b.new_page(viewport={"width": 800, "height": 580})
            page.goto(PAGE, wait_until="load", timeout=180000)   # photoreal init is heavy:
            # the HDRI + PBR material library take far longer than the classic style,
            # and a dozen concurrent starts blew the 30s default
            page.wait_for_function("window.__ready===true", timeout=90000)
            page.wait_for_timeout(400)
        sup_client = None
        if args.assist_backend:
            from cooksim_bench.backends.assistant_backend import make_assistant
            sup_client = make_assistant(args.assist_backend, args.assist_model,
                                        base_url=args.assist_base_url,
                                        profile=args.assist_profile,
                                        thinking=args.assist_thinking,
                                        max_tokens=args.assist_max_tokens,
                                        image_scale=args.assist_image_scale)
        cook_client = None
        if args.cook_backend:
            from cooksim_bench.backends.assistant_backend import make_assistant as _mk
            cook_client = _mk(args.cook_backend, args.cook_model,
                              base_url=args.cook_base_url, profile=args.cook_profile,
                              thinking=args.cook_thinking, max_tokens=args.cook_max_tokens)
            # the cook seat reverses the two text slots (the per-turn world context is the
            # "system" text, the static persona hint is the "user" text); session backends
            # (Realtime/Live) must open their one session on the STATIC text -- see
            # python -m cooksim_bench.backends.assistant_backend AssistantBackend.sup_call
            cook_client.cook_seat = True
            print(f"[setup] cook seat: {args.cook_backend}:{args.cook_model}", flush=True)
        rec = run(args.case, args.persona, args.paradigm, client, page,
                  video_path=None if no_render else args.out.replace(".json", ".mp4"),
                  persona_v2=pv2, obs_mode=args.sup_obs,
                  family=args.family, pauses=args.pauses, sup_client=sup_client,
                  cook_client=cook_client)
        if b is not None:
            b.close()
    # atomic: a preempted job must never leave a truncated report that a resume would then skip (2026-09-30)
    _tmp = args.out + ".tmp"
    json.dump(rec, open(_tmp, "w"), indent=1, default=str)
    os.replace(_tmp, args.out)
    # auto-eval: quantitative scoring embeds into the report (no LLM); set EVAL_JUDGE=1
    # to also run the in-persona questionnaire + judge (2 Gemini calls)
    from cooksim_bench.scoring.post_episode import score_record
    score_record(args.out, judge=os.environ.get("EVAL_JUDGE") == "1")

    print(f"{args.case} x {args.persona} [{args.paradigm}]: outcome={rec['outcome']} "
          f"failed_at={rec['failed_at']} injected={rec.get('injected')} "
          f"released={rec.get('released')}", flush=True)
