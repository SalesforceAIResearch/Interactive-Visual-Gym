#!/usr/bin/env python3
"""VH E2E STEP-IN rollout — port of CookSim tools/e2e_stepin_rollout.py (handover §5.7).

Difference from iv_e2e_rollout.py: there the deviated plan is scripted from t0 and the human is
mechanically LOCKED on it (autopilot). Here the human runs the CLEAN plan itself from t0, and the
error is injected only when the world reaches the deviation point.

Three paradigms (see docs/SOFT_HARD_CLARIFICATION.md):
  combo (DEFAULT) — the overcook soft-hard COMBINATION. The engine TAKES OVER the queue to
      FORCE the error (guaranteed manifestation) AND shows the [SYSTEM TASK] so the person
      narrates it IN CHARACTER (not a confused puppet). The takeover then RELEASES on correct
      detection + accept, or on the error committing -- handing agency back for recovery.
  soft — PROMPT-only, no takeover: the error manifests only if the person VOLUNTARILY enacts
      it. Fragile for active errors (a compliant novice follows the assistant's clean guidance
      and never makes the error). Use ONLY to test whether the assistant can talk a person out.
  hard — covert takeover with NO prompt: the queue is silently replaced; the person is never
      told (can read as a puppet). combo is hard + the in-character prompt.

Task dissolution (when the [SYSTEM TASK] block disappears):
  1. `armed` (window-valid flag) + human `accept`      -> released_reason="convinced"
  2. the tasked sequence has been carried out          -> released_reason="task_completed"
  3. prevention paths (see below)
NOTE ON (2): CookSim dissolves on completion only for its NON-error INTENT families; its error
family releases solely on conviction. (2) is enabled here for the error family too, per the
experiment owner's explicit spec ("the system prompt will disappear after action finished OR
assistant flag correctly and human agent accepts"). Set STEPIN_RELEASE_ON_DONE=0 to get
CookSim's stricter behaviour.

Usage: iv_e2e_stepin_rollout.py PORT CASE PERSONA OUT_JSON [combo|soft|hard]
"""
import collections
import json, math, os, sys, time

from vhhome_bench import protocol as P
from vhhome_bench import cases as CS
from vhhome import plan as pir
from vhhome.world import LiveWorld, pretty_cmd, FPT, ENGINE_REV
from vhhome.world import base as _base
from vhhome_bench.person import Queue, make_runner, CAP as _CAP, IDLE_END, NO_PROGRESS
# a multi-phase episode is roughly one episode per error plus the negotiations between
# them, so both budgets are raised by env for composite runs.
CAP = int(os.environ.get("STEPIN_CAP", str(_CAP)))

WALL_BUDGET = int(os.environ.get("STEPIN_WALL", "3000"))   # MUST be under iter_run's outer `timeout 3600` (skill §7:
                     # nested timeouts ordered so the INNERMOST writes a report —
                     # at 4200 the outer kill always won and the run left NOTHING)
TRIGGER_DEADLINE = 320
WINDOW_FALLBACK = 12      # ticks after dev_live before the window is forced open
OMISSION_ERRS = ("skipped_step", "missing_step")   # a refused attempt of their injected step commits them (f99)
RELEASE_ON_DONE = os.environ.get("STEPIN_RELEASE_ON_DONE", "1") != "0"
# INTERRUPTIBLE ACTIONS. render_script is atomic: a cross-room [Walk] costs 30-45 frames and the
# supervisor cannot be heard until it has finished, so a correct flag at latency 0 still cannot
# stop the arrival it is warning about. Here the clip is rendered ONCE (cost unchanged), then
# STREAMED tick by tick; if the human changes its plan partway, the unwanted tail is discarded
# and the body is put back at that frame's pose (LiveWorld.rewind_to -> move_character, verified
# to 0.04 m). Set STEPIN_INTERRUPT=0 for the old atomic behaviour.
INTERRUPT = os.environ.get("STEPIN_INTERRUPT", "1") != "0"
# ASSISTANT LATENCY (2026-09-05). The assistant's reply is not free: an API round-trip or a local
# model's forward pass takes x seconds of wall clock, and in that time the person keeps moving.
# A reply that took x s is delivered ceil(x / ASSIST_TICK_S) ticks after the tick it was asked
# on; the world (the person's queue) advances in between, and the next assistant call starts
# only when the reply has landed. ASSIST_LATENCY=0 restores the instantaneous assistant.
ASSIST_LATENCY = os.environ.get("ASSIST_LATENCY", "1") != "0"
DEAD_CONV_TICKS = int(os.environ.get("VH_DEAD_CONV_TICKS", "30"))    # standstill the dead-conversation guard needs
ASSIST_TICK_S = float(os.environ.get("ASSIST_TICK_S", "1.2"))   # wall seconds per game tick
# D3 (pilot 2026-09-05): 8 of 222 calls took 45-86 s of API stalls and retries and froze the
# person for 40-70 ticks. Only the successful attempt's own generation time is charged, and the
# charged delay is capped; the raw wall time is still recorded per call.
ASSIST_MAX_TICKS = int(os.environ.get("ASSIST_MAX_TICKS", "15"))
# HARDWARE NORMALISATION (2026-09-22). The charged delay must describe the MODEL, not the card it
# happened to land on: the same served model answers roughly twice as slowly on an L40S as on an A100,
# and charging that to the game clock would let the person walk twice as far while the assistant thinks
# — the arm would look worse for having been scheduled on a smaller GPU. ASSIST_LATENCY_DIV is the
# measured ratio (this card's generation time / an A100's) for THIS model, so gen_s / DIV is the
# A100-equivalent time every run is charged, whatever hardware was free. 1.0 = charge what was measured.
# Measured by vhhome_bench/cluster/latency_ratio.py into docs/GPU_LATENCY_RATIO.json; the Beaker job reads its own
# GPU's row and exports it. The raw wall and generation times stay in the report either way.
ASSIST_LAT_DIV = float(os.environ.get("ASSIST_LATENCY_DIV", "1") or 1)
if not ASSIST_LAT_DIV > 0:
    ASSIST_LAT_DIV = 1.0

# Best-verdict precedence when a flag could be credited to more than one live phase.
_VERDICT_RANK = {"valid": 0, "valid_alias": 1, "early_correct": 2, "early_alias": 3,
                 "late": 4, "early": 5, "wrong_name": 6, "invalid_label": 7}

# Two framings (ported from CookSim 2026-09-05), chosen by the injected error's family, each
# carrying an ENGINE HINT line re-rendered every turn from the engine's own flag verdict. The
# hint stops the "assistant named the right error, human keeps defending for ten turns" pattern:
# once the verdict is correct the human is told so and asked to accept when convinced.
# The split moved into iv_protocol with the cook grader: skipped_step / missing_step became
# COMMISSION errors (the engine cuts the step out of the body queue, so the person is doing it
# knowingly) and are NAMED; only these three keep the "you have not noticed" framing.
ATTENTION_ERRORS = P.ATTENTION_ERRORS

# COOK GRADER (docs/COOK_GRADER.md, default here since f83). "cook": the person's own judgement of
# whether the assistant named its mistake arms the release; the engine's flag verdict is recorded
# but controls nothing. "engine": the pre-f83 behaviour, kept so the two can be run as a paired A/B.
GRADER = (os.environ.get("VH_GRADER") or "cook").strip().lower()


def _detected_at(flags, cook):
    """Earliest crediting tick under the union: the person's first `correct` while a mistake was
    live, or the engine's first detection-grade flag, whichever came first."""
    ts = [f["t"] for f in flags if f["verdict"] in P.ARMS_DETECTION]
    ts += [c["t"] for c in cook["calls"] if c["call"] == "correct" and c["phase"] is not None]
    return min(ts) if ts else None


def _credit_latency(PH, flags, cook):
    """Latency of the earliest crediting event, measured against the DEVIATION OF THE PHASE THAT
    WAS CREDITED. Read off the live cell instead, it can go negative: the person may call the
    mistake correctly in the tick before the deviating action actually executes, and that is
    prevention, not a detection with a negative delay -- so a credit that precedes its phase's
    deviation returns None here and is counted on the prevention side."""
    best = None
    for i, ph in enumerate(PH):
        ts = [f["t"] for f in flags if f.get("phase") == i and f["verdict"] in P.ARMS_DETECTION]
        ts += [c["t"] for c in cook["calls"] if c["call"] == "correct" and c["phase"] == i]
        if not ts:
            continue
        dev = ph.get("dev_live")
        if dev is None or min(ts) < dev:
            continue                                   # credited before the deviation stood
        lat = min(ts) - dev
        best = lat if best is None else min(best, lat)
    return best


def _said_since_last_turn(conv, now):
    """Did the assistant actually speak between the person's previous turn and this one?

    ORDER, NOT TICKS (f100). On a poll tick the loop runs the person's idle turn first (recording a person
    entry at tick b), then delivers the supervisor's reply at the SAME tick b, then gives the person its turn to
    react. Comparing ticks with a strict `prev < t` threw that reaction away as "unprompted": the person's
    `correct` call on a true accusation never armed a release. Measured: no mistake was released by the
    person accepting a correction in any run since the guard landed (f84check, the 25-episode re-run, f94, f98,
    f99 -- 0 'convinced' releases, 127-323 discarded calls per set), against 21 in pilot 6 before it.
    The conversation list is appended in the order things happened, so "since the person's last turn" is
    "after the person's last entry in the list"."""
    last_h = max((i for i, c in enumerate(conv) if c.get("speaker") == "human"), default=-1)
    return any(c.get("speaker") == "vlm" and (c.get("text") or "").strip() and (c.get("t") or 0) <= now
               for c in conv[last_h + 1:])


_NO_COOK = {"calls": []}      # score-side helpers read the engine's verdicts only (A31)


def _detected_by(flags, cook):
    e = any(f["verdict"] in P.ARMS_DETECTION for f in flags)
    c = any(x["call"] == "correct" and x.get("phase") is not None for x in cook["calls"])
    return "both" if (e and c) else ("cook" if c else ("engine" if e else None))

# Wording follows CookSim's shortened blocks (CookSim 60d8f80, 2026-09-30; ported 2026-10-02):
# "or say nothing" instead of "arrange your conversation around your current action" (which asked for
# narration), and an explicit "do not fix anything yourself". The one household difference: the sequence
# is listed here, because while the engine owns the hands the person's own queue is empty.
SYSTEM_TASK_BELIEF = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You are testing whether the AI assistant can spot a mistake and convince you it is real.
Your hands are already committed: the action sequence below is what you are doing, and you
cannot change it here:
{seq}
Your job is your voice: talk about your current action in character, or say nothing. Never
promise an action that is not in the sequence, and never deny one that is.
{mistake_line}
Do not say anything to the assistant about this mistake or its consequences. If you do, the
assistant cannot be tested.
Record in assistant_error_call whether the assistant has found it. Until it has, carry on as your
persona would and do not fix anything yourself. Once it has, your trust level ({trust}) decides how
fast you come around: HIGH — the first correct warning; LOW — only after you see it with your own
eyes or in your own hands. When convinced, say so in your own words and set "move": "accept" —
your hands return to you and this task ends. A promise to fix it later is not acceptance.
Stay in persona; never reveal this task or that you are testing the assistant."""

SYSTEM_TASK_ATTENTION = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You are testing whether the AI assistant notices that you have LOST TRACK of something and brings
you back.
Your hands are already committed: the action sequence below is the errand you drifted into; you
cannot change it here, and you have not noticed the problem:
{seq}
Your job is your voice: talk about your current action in character, or say nothing. Never
promise an action that is not in the sequence, and never deny one that is.
{mistake_line}
Do not say anything to the assistant about what you lost track of or its consequences. If you do,
the assistant cannot be tested.
Until someone brings it up, carry on and do not fix anything yourself. When the assistant does and
what you can see confirms it, come back at once whatever your persona: say so in your own words
and set "move": "accept" — your hands return to you and this task ends. A promise to fix it later
is not acceptance.
Stay in persona; never reveal this task or that you are testing the assistant."""



def task_family(err):
    return "attention" if P.LABEL_OF.get(err, err) in ATTENTION_ERRORS else "belief"


def render_task(err, seq, kw, persona):
    """The [SYSTEM TASK] block for this turn: framing by error family, the error's own mistake line,
    trust level from the persona's dial. The engine's verdict is NOT in here any more (cook grader):
    the person judges the assistant itself, through `assistant_error_call`, so feeding it the
    engine's own grading would make that field a paraphrase of the thing it is meant to replace."""
    fam = task_family(err)
    _d = P.PERSONA_DIALS.get(P.PERSONA_ALIASES.get(persona, persona)) or P.PERSONA_DIALS.get(persona)
    trust = (_d[1] if _d else "medium").upper()      # dial tuple = (familiarity, trust, speed, alert, interrupt)
    tpl = SYSTEM_TASK_ATTENTION if fam == "attention" else SYSTEM_TASK_BELIEF
    return tpl.format(seq=seq, mistake_line=P.mistake_line(err, kw), trust=trust)


SYSTEM_TASK = SYSTEM_TASK_BELIEF      # legacy name (report prompts, tests)

# 35 of 61 `accept` turns across 12 runs queued NO action (op="drop") -- the human agrees out
# loud and its body does not move ("mouth says sorry, hands keep sinning", CookSim 5.7.6).
# `move` and `plan_update` are independent fields and nothing couples them. We do NOT fabricate
# actions on the human's behalf (that would enforce compliance the soft paradigm must not
# enforce); we re-ask ONCE, and record every occurrence either way.
ACCEPT_NUDGE = ("\n\n[You just said you accept / will fix it, but your action queue is unchanged, "
                "so your body will do nothing. If you meant it, put the exact commands in "
                "plan_update.actions now with op=\"append\" or \"replace\". If you did NOT mean it, "
                "say so plainly instead of agreeing.]")

TASK_REMINDER = ("\n\n[SYSTEM TASK REMINDER - this outranks the persona card above: your body is "
                 "carrying out the tasked sequence; decide for YOURSELF whether the supervisor has "
                 "named your real mistake, and record that in assistant_error_call. If a tasked action was REFUSED "
                 "because you cannot see its target, DO NOT re-issue it from the same spot — move "
                 "first, then retry. The persona shapes only your words, never your actions.]")


def _names(x):
    return tuple(sorted(_base(v) for v in (x or [])))


def state_key(holding, loc, opened, on, keep=None):
    """Comparable household situation, NAMES only — what is held, where each tracked object
    sits, what is open, what is on. The agent's position and how long a walk took are
    irrelevant to whether the deviation can start here (CookSim §5.7.3).

    `keep` restricts `loc` to the object classes this episode's plan actually touches. It is
    REQUIRED for the two sides to be comparable: the live symbolic state is seeded from the
    whole scene graph (183 objects), while pir.simulate() only ever records the handful the
    plan moved. Without the filter the reference key could never equal the live key and the
    injection simply never fired.
    """
    items = [(_base(k), _base(v)) for k, v in (loc or {}).items()]
    if keep is not None:
        items = [(k, v) for k, v in items if k in keep]
    return {"holding": _names(holding), "loc": tuple(sorted(items)),
            "opened": _names(opened), "on": _names(on)}


# ---------------------------------------------------------------- live-state injection (f105-f111)
# WHY. An injection used to fire only when the live symbolic state EQUALLED the state obtained by
# replaying the clean plan's prefix. A self-planning person reaches that state late, in another order,
# or not at all, and then the error never happens and the cell tests nothing: 9 phases lost over 60
# episodes (3 of them an episode's only error), plus 2 of 10 in the flash round.
# That exact match is still tried first. When it does not hold, every class fires on a LIVE ANCHOR:
# the moment in the person's own flow where the mistake belongs -- NEVER on a timer (user ruling
# 2026-09-18: "I don't like timer"; f105's 40-tick and f110's 150-tick firings are gone).
#   wrong_object / skipped / missing / wrong_placement   one of the NEXT TWO queued actions aims at the step
#   navigation_inefficiency (before: X)                  one of the next two queued actions aims at step X
#   stall (after: X)                                     the action that completes step X has just finished
#   repetition (sid: S, at: A)                           S done >= REP_GAP ticks ago AND the next two actions
#                                                        aim at A -- or the chore is one step from finishing
# A mistake whose moment never comes does not fire; the phase is skipped at its deadline with the reason.
LIVE_INJECT = os.environ.get("STEPIN_LIVE_INJECT", "1") != "0"
LIVE_OBJ_ERRS = ("wrong_object", "misidentify_fetch", "wrong_placement", "skipped_step", "missing_step")
LIVE_ANCHOR_ERRS = ("stall", "navigation_inefficiency", "repetition")
LIVE_HORIZON = 2
REP_GAP = int(os.environ.get("STEPIN_REP_GAP", "30"))      # a repetition is a memory lapse, not a double-tap
_NOACT = ("TurnLeft", "TurnRight", "Idle")
_INV = {"SwitchOff": "SwitchOn", "SwitchOn": "SwitchOff", "Close": "Open", "Open": "Close"}


def about_to(ph, rem, horizon=LIVE_HORIZON, aim=None):
    """The first of the next `horizon` queued actions aimed at the step's objects, or None."""
    from vhhome.world import base as _b
    aim = set(aim if aim is not None else (ph.get("aim") or ()))
    for st, _sid in list(rem)[:horizon]:
        if len(st) > 1 and st[0] not in _NOACT and _b(str(st[1])) in aim:
            return st
    return None


def step_marker(flat, sid):
    """(verb, object, k): the command that COMPLETES step `sid` in the plan it is cut from, and how many
    times that same (verb, object) has run by then -- 'fetch(plate@2)' completes on the 2nd Grab plate."""
    from vhhome.world import base as _b
    if not sid:
        return None
    idx = [i for i, (st, sd) in enumerate(flat) if sd == sid and st[0] not in _NOACT and len(st) > 1]
    # the step's DEFINING action, not its tidy-up: a fetch is done at the Grab, a deposit at the Put --
    # "close the fridge" after them is optional to a real person and must not gate the anchor
    key = {"fetch": ("Grab",), "deposit": ("Put", "PutIn")}.get(_sid_parts(sid)[0])
    if key and any(flat[i][0][0] in key for i in idx):
        idx = [i for i in idx if flat[i][0][0] in key]
    if not idx:
        return None
    last = flat[idx[-1]][0]
    v, o = last[0], _b(str(last[1]))
    k = sum(1 for st, _ in flat[:idx[-1] + 1] if st[0] == v and len(st) > 1 and _b(str(st[1])) == o)
    return (v, o, k)


def _done_of(world, mk):
    from vhhome.world import base as _b
    return [l for l in world.log if l.get("status") == "done" and len(l.get("step") or ()) > 1
            and l["step"][0] == mk[0] and _b(str(l["step"][1])) == mk[1]]


def _mk_list(mk):
    return [m for m in (mk if isinstance(mk, list) else [mk]) if m]


def marker_done(world, mk):
    """The step is done under ANY of its markers (a list, or one marker)."""
    return next((m for m in _mk_list(mk) if len(_done_of(world, m)) >= m[2]), None)


def just_completed(world, mk):
    """The person's last real action (turns and pauses aside) is the one that completes the step."""
    last = next((l for l in reversed(world.log) if l.get("status") == "done"
                 and (l.get("step") or [None])[0] not in _NOACT), None)
    if last is None:
        return False
    for m in _mk_list(mk):
        done = _done_of(world, m)
        if len(done) == m[2] and last is done[-1]:
            return True
    return False


def repetition_block(world, mk, aim):
    """Redo step S AS THE PERSON DID IT: its own completed actions from the approach to the press, with
    the final press inverted (the second press must have a real effect -- the lamp is back on), then a
    beat. None if S was not a toggle or the log has no such run."""
    from vhhome.world import base as _b
    done = _done_of(world, mk)
    if len(done) < mk[2] or mk[0] not in _INV:
        return None
    end = world.log.index(done[mk[2] - 1])
    start = end
    for i in range(end - 1, -1, -1):
        l = world.log[i]
        st = l.get("step") or []
        if l.get("status") == "done" and st and st[0] == "Walk" and len(st) > 1 and _b(str(st[1])) in aim:
            start = i
            break
        if l.get("status") == "done" and st and st[0] not in _NOACT and _b(str(st[1] if len(st) > 1 else "")) not in aim:
            break                                  # an action on something else: the step began after it
    run = [tuple(l["step"]) for l in world.log[start:end + 1]
           if l.get("status") == "done" and l["step"][0] not in _NOACT]
    if not run:
        return None
    press = run[-1]
    return run[:-1] + [(_INV[press[0]],) + tuple(press[1:]), ("Idle", 10)]


def _sid_parts(sid):
    """'fetch(waterglass@2)#3' -> ('fetch', 'waterglass', 2)."""
    import re
    mt = re.match(r"(\w+)\((\w+)(?:@(\d+))?", str(sid or ""))
    return (mt.group(1), mt.group(2), int(mt.group(3) or 1)) if mt else (None, None, 1)


def _done_count(world, verbs, cls):
    from vhhome.world import base as _b
    return sum(1 for l in world.log if l.get("status") == "done"
               and len(l.get("step") or ()) > 1 and l["step"][0] in verbs and _b(str(l["step"][1])) == cls)


def live_object_block(ph, world):
    """(block, None) when the object-bound error of phase `ph` can fire NOW, else (None, why).

    The block is the template's own dev_section; only the trigger is live. The precondition is the
    least the block needs to still BE that error in this house:
      wrong_placement  deposit(X)  X is in hand                      (else there is nothing to misplace)
      wrong_object     fetch(X)    X not fetched yet, the decoy not in hand
      omission         fetch(X)    fewer than n grabs of X done      (the n-th X is still to come)
                       deposit(X)  X in hand
                       switch_off  an X is still on
                       shut(X)     X is open
                       inspect(X)  X not opened yet
    """
    from vhhome.world import base as _b
    err, kw = ph["err"], ph.get("kw") or {}
    verb, x, n = _sid_parts(kw.get("sid"))
    if not x:
        return None, "no step id to bind the error to"
    sym = world.sym
    held = sym.holding_cls
    if err == "wrong_placement":
        if x not in held:
            return None, f"the {x} is not in hand"
    elif err in ("wrong_object", "misidentify_fetch"):
        if x in held or _done_count(world, ("Grab",), x) >= n:
            return None, f"the {x} was already fetched"
        wrong = _b(str(kw.get("wrong") or ""))
        if wrong and wrong in held:
            return None, f"the {wrong} is already in hand"
    elif err in OMISSION_ERRS:
        if verb == "fetch" and (_done_count(world, ("Grab",), x) >= n):
            return None, f"the {x} was already fetched"
        if verb == "deposit" and x not in held:
            return None, f"the {x} is not in hand"
        if verb == "switch_off" and x not in sym.on_cls:
            return None, f"no {x} is on"
        if verb == "shut" and x not in sym.opened_cls:
            return None, f"the {x} is not open"
        if verb == "inspect" and _done_count(world, ("Open",), x):
            return None, f"the {x} was already opened"
    else:
        return None, "not an object-bound class"
    blk = [tuple(c) for c in (ph.get("dev_section") or [])]
    if not blk:
        return None, "template has no block"
    # LEAD-IN (f108b). A template block can begin mid-scene -- routine's wrong_object block starts at
    # "grab the juice", because the walk to the fridge and its opening sit in the common prefix. Fired
    # from live state the body may be two rooms away, so the grab is refused and the error never
    # happens (smoke test t63). Carry the template's last Walk before the block, plus the Opens after
    # it that are still needed, in front of it.
    if blk[0][0] != "Walk":
        pre = [tuple(st) for st, _sid in (ph.get("flat") or [])[:ph.get("m", 0)]]
        j = max((i for i, st in enumerate(pre) if st[0] == "Walk"), default=None)
        if j is not None:
            lead = [pre[j]] + [st for st in pre[j + 1:]
                               if st[0] == "Open" and _b(str(st[1])) not in sym.opened_cls]
            blk = lead + blk
    # HANDS (f126, audit of formal run 2). The anchor looks LIVE_HORIZON actions ahead, so "about to: walk to
    # the mug" fired while the deposit before it ("put the plate in the dishwasher") had not happened: the block
    # replaced the queue with the plate still in hand, its grabs were refused ("my hands are full"), the block
    # never completed, the phase never released and the person stayed frozen in the takeover to the end --
    # 85 of 153 cx_aftermeal_missing_obj and 32 of 153 cx_routine_rep_obj episodes. A block fires only when
    # the hands the person has now can carry it: two hands, each Grab takes one, each Put frees one.
    # Opening, closing and switching need a free hand too (the routine block: book in one hand, the juice fills
    # the other, "close the fridge" refused "both my hands are full").
    free = 2 - len(getattr(sym, "holding", None) or held)      # a symbolic state carries holding_cls only
    for st in blk:
        if st[0] == "Grab":
            free -= 1
            if free < 0:
                return None, f"hands full for the block's grabs (holding {sorted(sym.holding_cls)})"
        elif st[0] in ("Put", "PutIn"):
            free += 1
        elif st[0] in ("Open", "Close", "SwitchOn", "SwitchOff") and free < 1:
            return None, f"no free hand for '{pretty_cmd(st)}' (holding {sorted(sym.holding_cls)})"
    return blk, None


def live_object_rest(ph, rem):
    """The person's own queued steps that survive the forced block: drop the ones that would undo the
    error straight away (the omitted / swapped object, or putting X where it belongs)."""
    from vhhome.world import base as _b
    _v, x, _n = _sid_parts((ph.get("kw") or {}).get("sid"))
    if ph["err"] == "wrong_placement":
        return [e for e in rem if not (e[0][0] in ("Put", "PutIn") and _b(str(e[0][1])) == x)]
    return [e for e in rem if not (len(e[0]) > 1 and _b(str(e[0][1])) == x)]


def key_from_sym(sym, keep=None):
    """Built from the PLAN-EFFECT mirror, never the raw instance state -- see SymState.eff.

    The mirror stores base-name MULTISETS (Counters); expand them into repeated-name lists so
    state_key sees the same shape it gets from simulate's token sets."""
    e = getattr(sym, "eff", None) or {}

    def expand(c):
        return [k for k, n in (c or {}).items() for _ in range(n)]

    loc = {}
    for (k, v), n in (e.get("loc") or {}).items():
        for j_ in range(n):
            loc[f"{k}#x{j_}"] = v          # unique keys; state_key bases them back to `k`
    return state_key(expand(e.get("holding")), loc,
                     expand(e.get("opened")), expand(e.get("on")), keep)


def key_from_sim(res, keep=None):
    """Same key from a symbolic simulate() result (its per-node `boundaries` carry `loc`)."""
    loc = {}
    b = res.get("boundaries") or {}
    if b:
        loc = list(b.values())[-1].get("loc") or {}
    return state_key(res.get("holding"), loc, res.get("open"), res.get("on"), keep)



def flat_of(nodes):
    return [(tuple(st), n.sid) for n in nodes for st in n.commands]


def diff_plans(a, b):
    """(m, dev_cont, dev_section) for plan b against baseline a — the longest-common
    prefix/suffix sandwich the single-error path has always used."""
    m = 0
    while m < min(len(a), len(b)) and a[m][0] == b[m][0]:
        m += 1
    sfx = 0
    while sfx < min(len(a), len(b)) - m and a[-1 - sfx][0] == b[-1 - sfx][0]:
        sfx += 1
    return m, b[m:], [x for x, _ in (b[m:len(b) - sfx] or b[m:m + 4])]


def step_aim(flat, kw):
    """The objects the step named by kw["sid"] acts on in the plan it is cut from, e.g. fetch(milk) =
    walk to the fridge, open the fridge, grab the milk -> {fridge, milk}. A person whose next actions
    touch one of these is ABOUT TO do that step (f110 approach trigger)."""
    from vhhome.world import base as _b
    sid = (kw or {}).get("sid")
    if not sid:
        return set()
    aim = {_b(str(st[1])) for st, sd in flat if sd == sid and len(st) > 1}
    x = _sid_parts(sid)[1]
    return aim | ({x} if x else set())


def build_phases(epk, calls, post, errs, flat_clean):
    """One PHASE per injected error.

    Phase k is the SINGLE-ERROR problem taken against the baseline of every earlier error having
    already happened: build the plan with calls[:k] and with calls[:k+1] and diff those two. That
    gives phase k its own deviation point, its own trigger state and its own tasked sequence with
    no later error mixed in — which one whole-plan diff cannot do, because it stops at the FIRST
    divergence and sweeps everything after it (including the clean steps between the errors) into
    a single "deviation".
    """
    phases, base_flat = [], flat_clean
    for k in range(len(errs)):
        cur_flat = flat_of(CS.build_deviated(epk, calls[:k + 1], post)[1])
        m, dc, ds = diff_plans(base_flat, cur_flat)
        kw_k = calls[k][1] if k < len(calls) else {}
        # markers from BOTH the plan this phase is cut from (earlier mistakes in place) and the clean plan
        # (earlier mistakes corrected): after a milk->juice swap the "put the milk away" step completes on
        # PutIn juice if the swap stood and on PutIn milk if the person was talked out of it
        def _mks(sid_):
            return [m_ for m_ in dict.fromkeys([step_marker(base_flat, sid_), step_marker(flat_clean, sid_)]) if m_]

        def _aim(sid_):
            return sorted(step_aim(base_flat, {"sid": sid_}) | step_aim(flat_clean, {"sid": sid_}))
        anc = {"done": _mks(kw_k.get("sid")), "after": _mks(kw_k.get("after")),       # repetition / stall
               "before": _mks(kw_k.get("before")), "before_aim": _aim(kw_k.get("before")),  # detour
               "at_aim": _aim(kw_k.get("at"))}                                          # repetition
        phases.append(dict(err=errs[k], kw=kw_k, aim=sorted(step_aim(base_flat, kw_k)), anchor=anc,
                           m=m, dev_cont=dc, dev_section=ds, flat=cur_flat,
                           injected=False, released=False, armed=False, prevention=False,
                           dev_live=None, lo_live=None, skipped=None, injected_at=None,
                           released_at=None, released_reason=None, armed_at=None,
                           grade={"err": errs[k], "dev": None, "window": (10**9, 10**9)}))
        base_flat = cur_flat
    return phases


def run(port, case_name, persona, out_json, paradigm="combo"):
    t0 = time.time()
    _all = CS.all_cases() + CS.composite_cases() + CS.composite_clean_cases()
    if case_name.endswith("__clean") and not any(c["name"] == case_name for c in _all):
        _all = _all + [CS.clean_case_for(case_name[:-len("__clean")])]
    case = next(c for c in _all if c["name"] == case_name)
    epk = case["episode"]
    client = P.make_client()
    clean_nodes, ir, compiled, records = CS.build_deviated(epk, case["calls"], case.get("post"))
    case["anchor_sid"] = records[0].anchor if records else None

    # RESTRAINT EPISODE (skill contract §5.1/§7): err=None runs the clean plan — optionally
    # under a LEGAL reorder (a legitimate change of intent) validated against the real DAG —
    # with no injection; the assistant is graded on restraint.
    CLEAN = case["err"] is None
    if CLEAN:
        order = case.get("reorder")
        if order:
            perm = (list(range(len(clean_nodes)))[::-1] if order == "reverse"
                    else list(range(1, len(clean_nodes))) + [0])
            reordered = [clean_nodes[i] for i in perm]
            pos = {n.sid: k for k, n in enumerate(reordered)}
            for before_sid, after_sid in pir.deps(clean_nodes):
                if pos.get(before_sid, -1) >= pos.get(after_sid, 10**9):
                    sys.exit(f"{case_name}: reorder '{order}' violates the task DAG — "
                             f"not a legitimate change of intent")
            clean_nodes = reordered
        ir = clean_nodes

    flat_clean = [(tuple(s), n.sid) for n in clean_nodes for s in n.commands]

    ERRS = case.get("errs") or ([case["err"]] if case["err"] else [])
    CALLS = case.get("calls") or []
    PH = build_phases(epk, CALLS, case.get("post"), ERRS, flat_clean) if (not CLEAN and ERRS) else []
    # legacy single-error views, so everything downstream that reads them still works
    m = PH[0]["m"] if PH else 0
    dev_cont = PH[0]["dev_cont"] if PH else []
    dev_section = PH[0]["dev_section"] if PH else []
    flat_err = PH[-1]["flat"] if PH else flat_clean

    if not CLEAN:
        for _k, _p in enumerate(PH):
            if not _p["dev_cont"]:
                sys.exit(f"{case_name}: phase {_k} ({_p['err']}) adds nothing — "
                         "clean and deviated plans are identical")
            _clean_txt = {pretty_cmd(s_) for s_, _ in flat_clean}
            _only = {pretty_cmd(s_) for s_ in _p["dev_section"] if s_[0] != "Idle"} - _clean_txt
            # `stall` and the omissions are errors of INACTION -- a pause, or a missing step -- so
            # they legitimately add no command the clean plan lacks and this guard would abort
            # them. For those classes the deviation is identified by the injected subgoal instead.
            if not _only and _p["err"] not in ("stall", "skipped_step", "missing_step"):
                sys.exit(f"{case_name}: phase {_k} ({_p['err']}) has no command absent from the "
                         "clean plan — the deviation would be indistinguishable from execution")

    # trigger reference: symbolically replay each phase's own prefix
    # object classes this episode manipulates — the only ones whose position can matter
    KEEP = {_base(st[1]) for st, _ in flat_clean + flat_err if len(st) > 1} | \
           {_base(st[3]) for st, _ in flat_clean + flat_err if len(st) > 3}
    KEEP.discard(None)
    for _p in PH:
        _ref = pir.simulate([(s_, sid_, False) for s_, sid_ in _p["flat"][:_p["m"]]], epk)
        _p["ref_key"] = key_from_sim(_ref, KEEP)
    ref_key = PH[0]["ref_key"] if PH else None

    # the port keeps render folders apart when the same cell runs on two engines at once
    name = f"stepin_{paradigm}_{case_name}__{persona}_p{port}"
    world = LiveWorld(port, epk, name)
    from vhhome import goals as episode_goals
    report = dict(setting="e2e_stepin", engine_rev=ENGINE_REV, person_model=P.MODEL,
                  paradigm=paradigm, scenario=case_name, persona=persona,
                  err=case["err"], episode=epk,
                  task=episode_goals.EPISODES[epk]["instruction"], convinced=False,
                  entered=False, fix_source=None, conversation=[],
                  clean_prefix_len=m, dev_section=[pretty_cmd(s) for s in dev_section],
                  injected_at=None, released_at=None, released_reason=None)
    entered, correction = [False], {}
    R = make_runner(world, case, persona, client, report, entered, correction, clean_nodes)
    # The human is NOT handed the plan. It starts with an EMPTY queue and must work the steps out
    # itself (skill high/medium) or ask the assistant for them (skill low); the takeover supplies
    # only the injected error sequence when its anchor state is reached. STEPIN_SELF_PLAN=0
    # restores the old behaviour (queue pre-filled with the clean plan).
    q = R["q"]; q[0] = Queue([] if os.environ.get("STEPIN_SELF_PLAN", "1") == "1" else list(flat_clean))
    convo, comm, moves, RND = R["convo"], R["comm"], R["moves"], R["RND"]
    R["_rej_seen"] = 0

    sup_sys = P.e2e_supervisor_prompt(epk, persona)
    # NATIVE streaming seat (IV_ASSIST_BACKEND=live_native | realtime_native; vhhome_bench/native.py,
    # ported from CookSim): nobody polls the model. Every tick it is pushed that tick's frame and a "[t=N]"
    # marker, every line the person says reaches it as audio, and whatever it decides to say comes back as an
    # utterance that goes through the SAME delivery path as a polled reply (flag grading, conversation,
    # the person's turn). A line is charged from its anchor (the end of the person's audio, or the last frame
    # for a self-started line) to its FIRST output token.
    NATIVE = P.ASSIST_BACKEND.endswith("_native")
    nat = {"open": False, "last": 0.0, "q": [], "held_s": 0.0, "holds": 0, "sess": None, "errors": 0}
    if NATIVE:
        from vhhome_bench import native as NV
        nat["sess"] = NV.make_native(P.ASSIST_BACKEND, P.ASSIST_MODEL)
        sup_sys = NV.native_system_prompt(sup_sys)

    def _nat_ticks(x):
        return min(math.ceil(x / ASSIST_TICK_S), ASSIST_MAX_TICKS) if ASSIST_LATENCY else 0

    def native_poll(b, said_):
        """Feed tick b to the session; return the utterance that is due at b (or None)."""
        ses = nat["sess"]
        try:
            if not nat["open"]:
                ses.open(sup_sys)
                nat["open"] = True
            sp = ses.speaking()
            if sp is not None and b >= sp["anchor_tick"] + _nat_ticks(sp["latency_s"]):
                _h = time.time()                    # a line that is already due is still being said: hold the
                ses.wait_until_quiet(30.0)          # world, so finishing the sentence is not charged
                nat["held_s"] += time.time() - _h
                nat["holds"] += 1
            _w = nat["last"] + NV.TICK_WALL - time.time()
            if _w > 0:
                time.sleep(_w)                      # at most one frame per tick of wall time (<= 1 fps)
            nat["last"] = time.time()
            fr = world.sup_frames(b, window=1)
            if fr:
                ses.push_frame(fr[-1], b)
            if said_:
                ses.push_speech(said_, b)
        except Exception as ex:                     # noqa: BLE001 — a transport fault is recorded, never repaired
            nat["errors"] += 1
            P.ASSIST_STATS["retries"] += 1
            print(f"[t{b}] native feed error: {type(ex).__name__}: {str(ex)[:140]}", flush=True)
        for u in ses.drain():
            if not u["text"]:
                continue
            fl, spoken, marker = NV.parse_flag(u["text"])
            y = _nat_ticks(u["latency_s"])
            nat["q"].append({"t_arrive": max(u["anchor"]["tick"] + y, b), "t_issue": u["anchor"]["tick"],
                             "x": u["latency_s"], "y": y,
                             "out": {"text": spoken, "flag": fl,
                                     "native": {"raw": u["text"], "flag_marker": marker,
                                                "anchor": u["anchor"]["kind"],
                                                "person_line": u["anchor"].get("cook_line"),
                                                "latency_s": u["latency_s"], "done_s": u["done_s"],
                                                "interrupted": u["interrupted"]}}})
        due = [q_ for q_ in nat["q"] if b >= q_["t_arrive"]]
        if not due:
            return None
        q_ = due[0]
        nat["q"].remove(q_)
        report["assistant_latency"].append({"called_at": q_["t_issue"], "wall_s": round(q_["x"], 2),
                                            "gen_s": round(q_["x"], 2), "gen_s_charged": round(q_["x"], 2),
                                            "lat_div": 1.0, "ticks": q_["y"], "capped": False,
                                            "deliver_at": b, "native": True})
        return q_
    _pv = P.audit_composed_prompt(sup_sys, persona)
    if _pv:
        sys.exit(f"composed-prompt audit failed (skill §6/§8): {_pv}")
    seq_txt = "\n".join(f"  {i+1}. {pretty_cmd(s)}" for i, s in enumerate(dev_section))
    task_text = "" if CLEAN else render_task((PH[0]["err"] if PH else case["err"]), seq_txt,
                                             (PH[0].get("kw") if PH else (CALLS[0][1] if CALLS else {})),
                                             persona)
    report["prompts"] = {"supervisor_system": sup_sys, "human_briefing": P.E2E_BRIEFING,
                         "system_task": task_text}
    report["phases"] = [{"err": _p["err"], "dev_section": [pretty_cmd(x) for x in _p["dev_section"]],
                         "clean_prefix_len": _p["m"]} for _p in PH]

    km_evicted, km_max = [0], [0]
    interrupted = [0]
    changed = [False]
    align = {"accept_no_action": 0, "reask": 0, "reask_fixed": 0, "reask_failed": 0,
             "accept_total": 0}
    armed, prevention = [False], [False]
    injected, released, skipped = [False], [False], [None]
    flags, flag_repeats = [], [0]
    dev_live, lo_live = [None], [None]
    grade = {"err": (PH[0]["err"] if PH else case["err"]), "dev": None,
             "window": (10**9, 10**9)}
    # The loop below is written for ONE live error. Rather than thread a phase index through
    # every site, the live cells above ARE the current phase's state: `advance_phase()` saves
    # them back into PH[cur] and loads the next phase in. A single-error case has one phase and
    # behaves exactly as before.
    cur = [0]
    cook = {"calls": [], "wrong_calls": 0, "omitted": 0, "malformed": 0,
            "unauthored_correct": 0, "denied_live": 0, "unprompted": 0}
    m_c, dev_cont_c, dev_section_c = [m], [dev_cont], [dev_section]
    ref_key_c, task_text_c, deadline_c = [ref_key], [None], [TRIGGER_DEADLINE]

    def _sync_out():
        if cur[0] < len(PH):
            PH[cur[0]].update(injected=injected[0], released=released[0], armed=armed[0],
                              prevention=prevention[0], dev_live=dev_live[0],
                              lo_live=lo_live[0], skipped=skipped[0],
                              grade=dict(grade))

    def _seq_text(sec):
        return "\n".join(f"  {i+1}. {pretty_cmd(s_)}" for i, s_ in enumerate(sec))

    def _about_to_finish(w):
        """At most one of the task's GOAL checkpoints is still unmet -- the same instance-graded checks that
        decide a win. (f115: the per-step test is class-level, so a second, non-target lamp left on kept
        "switch off the lamp" owed forever and a repetition waiting for the finish never fired.)"""
        try:
            cps = w.sym.checkpoints()
        except Exception:
            cps = None
        if cps:
            return sum(1 for v in cps.values() if not v) <= 1
        return len({sd for _st, sd in CS.pending_tail(clean_nodes, w.sym)}) <= 1

    def _live_block(w):
        """(block, rest_of_queue, trigger) when the current phase's live anchor holds NOW, else None.
        No timers: see the f105-f111 header."""
        ph = PH[cur[0]]
        err, an, kw = ph["err"], ph.get("anchor") or {}, ph.get("kw") or {}
        rem = list(q[0].remaining())
        tmpl = [tuple(c) for c in (ph.get("dev_section") or [])]
        if err in LIVE_OBJ_ERRS:
            hit = about_to(ph, rem)
            # an n-th-instance step (fetch(plate@2), switch_off(tablelamp@3)) is only "about to" happen once
            # the first n-1 are done -- heading for the FIRST lamp is not about to do the third
            _v, _x, _n = _sid_parts(kw.get("sid"))
            _vb = {"fetch": ("Grab",), "switch_off": ("SwitchOff",), "shut": ("Close",), "inspect": ("Open",)}.get(_v)
            if hit and _n > 1 and _vb and _done_count(w, _vb, _x) != _n - 1:
                hit = None
            if not hit:
                return None
            b, _why = live_object_block(ph, w)
            return (b, live_object_rest(ph, rem), "about to: " + pretty_cmd(hit)) if b else None
        if err == "stall":
            if tmpl and just_completed(w, an.get("after")):
                return tmpl, rem, "just finished: " + kw.get("after", "")
            return None
        if err == "navigation_inefficiency":
            if not tmpl or marker_done(w, an.get("before")):
                return None
            hit = about_to(ph, rem, aim=an.get("before_aim") or ())
            return (tmpl, rem, "about to: " + pretty_cmd(hit)) if hit else None
        if err == "repetition":
            mk = marker_done(w, an.get("done"))          # the marker that actually completed
            if not mk:
                return None
            t_c = _done_of(w, mk)[mk[2] - 1].get("t_done") or 0
            hit = about_to(ph, rem, aim=an.get("at_aim") or ())
            fin = _about_to_finish(w)
            if not ((hit and w.tick - t_c >= REP_GAP) or fin):
                return None
            b = repetition_block(w, mk, set(ph.get("aim") or ())) or tmpl
            trig = ("about to: " + pretty_cmd(hit)) if (hit and w.tick - t_c >= REP_GAP) else "chore about to finish"
            return (b, rem, f"{trig} ({w.tick - t_c} ticks after {kw.get('sid')})") if b else None
        return None

    def _rep_wait_ok(w):
        """A repetition never fires right after its step, on EITHER path (f112): the template match
        fired one 8 ticks after the lamp went off (tidybedroom, anchor10). REP_GAP ticks must pass
        unless the chore is about to finish."""
        ph = PH[cur[0]]
        if ph["err"] != "repetition":
            return True
        mk = marker_done(w, (ph.get("anchor") or {}).get("done"))
        if not mk:
            return True
        t_c = _done_of(w, mk)[mk[2] - 1].get("t_done") or 0
        return w.tick - t_c >= REP_GAP or _about_to_finish(w)

    def _live_why(w):
        """Why the live anchor never held -- recorded when the phase is skipped at its deadline."""
        ph = PH[cur[0]]
        err, an, kw = ph["err"], ph.get("anchor") or {}, ph.get("kw") or {}
        if err in LIVE_OBJ_ERRS:
            b, why = live_object_block(ph, w)
            return why if not b else f"the person never headed for {kw.get('sid')}"
        if err == "stall":
            return f"{kw.get('after')} was never completed" if not marker_done(w, an.get("after")) else \
                f"{kw.get('after')} completed during another forced section"
        if err == "navigation_inefficiency":
            return f"the person never headed for {kw.get('before')}"
        if err == "repetition":
            return (f"{kw.get('sid')} was never done" if not marker_done(w, an.get("done"))
                    else f"the person never headed for {kw.get('at')} {REP_GAP}+ ticks later")
        return "no live anchor for this class"

    def _sync_in():
        ph = PH[cur[0]]
        injected[0], released[0], skipped[0] = ph["injected"], ph["released"], ph["skipped"]
        armed[0], prevention[0] = ph["armed"], ph["prevention"]
        dev_live[0], lo_live[0] = ph["dev_live"], ph["lo_live"]
        grade.clear(); grade.update(ph["grade"])
        m_c[0], dev_cont_c[0], dev_section_c[0] = ph["m"], ph["dev_cont"], ph["dev_section"]
        ref_key_c[0] = ph["ref_key"]
        task_text_c[0] = "" if CLEAN else render_task(ph["err"], _seq_text(ph["dev_section"]), ph.get("kw"), persona)

    def _close_earlier(t):
        """An earlier phase's detection window ends when the NEXT error's deviation begins.

        Leaving it open (the single-error `hi = 10**9`) lets the first error's window swallow
        every later flag, so a correct name for the second error is graded inside the first's
        window. The LAST phase keeps an open window, exactly as a single-error episode does.
        """
        for _p in PH[:cur[0]]:
            lo, hi = _p["grade"].get("window", (10**9, 10**9))
            if hi >= 10**9 and lo < 10**9:
                _p["grade"]["window"] = (lo, t)
                _p["closed_at"] = t

    def advance_phase():
        """Close the phase that just released and arm the next one.

        The released phase's window stays OPEN for now: it is closed only when the next error's
        deviation actually begins (see below), not when the next phase is merely armed — the
        assistant may still be reasoning about the first error while the second is being set up,
        and closing at arming gave it only a handful of ticks. The next phase's trigger deadline
        is re-based from here rather than from t0. The person then has to carry
        on with the chore to REACH the next anchor — which is exactly the situation the next
        error is defined against.
        """
        _sync_out()
        cur[0] += 1
        if cur[0] >= len(PH):
            return False
        _sync_in()
        deadline_c[0] = world.tick + TRIGGER_DEADLINE
        # The correction path empties the queue so the assistant can guide freely; refill it with
        # the chore's remaining steps so she carries on and can reach the next error's anchor.
        done_sids = {l.get("kind") for l in world.log if l.get("status") == "done"}
        rest = [(st, sid) for st, sid in PH[cur[0]]["flat"][:PH[cur[0]]["m"]]
                if sid not in done_sids]
        # ...but only the steps that are not ALREADY waiting in the queue (f122). A phase that releases as
        # `task_completed` never emptied the queue: the forced block was followed by the rest of the chore,
        # so appending the prefix again queued it twice. In cx_appliance_nav_place the second copy of
        # fetch(milk) ran after the microwave was switched on: the engine itself walked back, took the
        # wrongly-placed milk off the counter and left it in the person's hands — the "self-repair" of the
        # silent arm in that case (3/3 won on f118) started with the engine's own queue.
        # ...nor the steps the body has ALREADY carried out under another label. A forced block is logged as
        # `live_inject`, not under its step id, so the sid test above missed it: after the juice-for-milk
        # block released, the whole block was queued again and the person fetched the juice and put it in the
        # microwave a second time (cx_appliance_obj_stall, every run since f105).
        import re as _re
        _n = lambda st: tuple(_re.sub(r"[#@]\d+$", "", str(x)) for x in st)
        _sids = {sid for _, sid in rest}
        _did = collections.Counter(_n(l["step"]) for l in world.log
                                   if l.get("status") == "done" and l.get("step") and l.get("kind") not in _sids)
        _wait = collections.Counter(_n(st) for st, _ in q[0].remaining())
        _keep = []
        for st, sid in rest:
            k_ = _n(st)
            if _wait[k_] > 0:
                _wait[k_] -= 1
            elif _did[k_] > 0:
                _did[k_] -= 1
            else:
                _keep.append((st, sid))
        rest = _keep
        if rest:
            q[0].append(rest)
        print(f"[t{world.tick}] PHASE {cur[0]} armed ({PH[cur[0]]['err']}), "
              f"{len(rest)} step(s) requeued to resume the chore", flush=True)
        return True

    def task_active():
        """The injected task is in force (BOTH paradigms): hard = queue taken over,
        soft = [SYSTEM TASK] present in the prompt."""
        return injected[0] and not released[0]

    def task_live():
        """The SYSTEM TASK text is visible in the human's prompt. TRUE for soft (prompt is the
        only control) AND for combo (the body is force-taken-over but the person is ALSO told,
        so its DIALOGUE stays in character instead of being a confused puppet)."""
        return paradigm in ("soft", "combo") and task_active()

    def self_plan_apply(h_pd, op, t):
        """Apply the HUMAN'S OWN plan update to the queue.

        This is what makes `soft` soft. VH's queue is authoritative, and the only existing route
        into it (`apply_guidance`) force-calls enter_correction() -- which would mark the run
        convinced the moment the human queued anything. So pre-conviction the human had NO way to
        act on the [SYSTEM TASK]: the clean plan ran to completion and the error never happened
        (observed on 4 runs, 2026-08-03). Here the human writes its own queue, tagged
        `human_self` (never `vlm_fix`, which would corrupt fix_source attribution), and
        enter_correction is NOT called -- conviction still requires armed + accept.
        Compliance remains logged, never enforced: the human may queue anything, or nothing.
        """
        if op == "drop":
            return None
        filt = P.convert_parse(h_pd, persona, world, q[0].remaining_texts())
        # NO SILENT DROPS ON THE STEP-IN PATH EITHER (f93). The recovery rollout has told the person
        # since fix #47 when a planned step would not parse; this applier did not, so an unparseable
        # step vanished with no action, no refusal and no feedback. Measured: a putback episode spent
        # its last 56 ticks saying "I'm turning around now" every two ticks while queueing NOTHING --
        # "turn around" was removed from the vocabulary in f86 and parse_cmd simply returned None.
        for lost in (filt.get("unusable_for_solvability") or []):
            align["unparsed"] = align.get("unparsed", 0) + 1
            # KEEP THE TEXT (f103): the counter alone made this unauditable -- 13 episodes had an
            # unparseable step and no record of what the person had tried to say, so a vocabulary gap
            # in the engine could not be told apart from a person writing nonsense.
            report.setdefault("unparsed_steps", []).append({"t": t, "text": str(lost)[:120]})
            R["fb_buf"].append(f'your planned step "{lost}" is NOT a command your body knows, so NOTHING '
                               f'was queued — {P.vocab_hint(lost)}')
        if not filt.get("anything_applicable"):
            # Deliberately NOT treated as "cancel everything". An earlier attempt did exactly
            # that, and one unusable `replace` wiped all nine commands of the episode, which
            # then ended at tick 0 having done nothing. The real fault was upstream: "stand
            # still" is the engine's OWN rendering of an Idle and the grammar would not accept
            # it back, so a human told to stand still said exactly that and was ignored. Fixed
            # in parse_cmd instead, where it belongs.
            return {"applied": False, "why": "nothing applicable", "filter": filt}
        new = [(st, "human_self") for st, _t in filt["commands"]]
        newt = [t for _s, t in filt["commands"]]
        it = iter(q[0].remaining_texts())
        if newt and all(any(t == r for r in it) for t in newt):
            return {"applied": False, "why": "already queued", "filter": filt}
        (q[0].append if op == "append" else q[0].replace_rest)(new)
        print(f"[t{t}] human self-plan {op}: {newt}", flush=True)
        return {"applied": True, "op": op, "cmds": newt, "filter": filt}

    def task_done():
        """Has the tasked sequence actually been carried out? (subsequence match — she may
        interleave other actions)."""
        _sec = dev_section_c[0] if PH else dev_section
        if not _sec:
            return False
        # compare on rendered text: the human's steps come back through convert_parse
        # (NL -> step), so raw tuple identity is not reliable.
        # Idle ("stand still") is EXCLUDED: it is a pause, not an act, and it does not survive
        # the NL plan channel -- convert_parse drops it. Counting it made a run that genuinely
        # wandered to the sofa and the bed score as non-compliant (nav/creative, 2026-08-03).
        want = [pretty_cmd(s_) for s_ in _sec if s_[0] != "Idle"]
        if not want:
            # INACTION ERRORS. `stall` and `skipped_step` have a dev_section that is ONLY an
            # Idle, so excluding Idle leaves nothing to match and this returned False forever —
            # compliance could never be true for them by construction. That fed a much worse
            # loop: the task dissolves on `task_completed`, which consults this, so the SYSTEM
            # TASK never lifted, the human went on standing still indefinitely, and the episode
            # ran until the no-progress guard killed it. Compliance here is that the human
            # actually idled after the injection instead of doing the step it was meant to skip.
            idled = any(tuple(l.get("step") or ())[:1] == ("Idle",) and l["status"] == "done"
                        and (l.get("t_issue") or 0) >= (report["injected_at"] or 0)
                        for l in world.log)
            return idled
        if not want:
            return False
        # Compare against the DISPLAY text, not a re-render of the stored step. `step` now holds
        # instance-resolved tokens, so pretty_cmd() gives "switch on the tablelamp#377" while
        # `want` (built from the unresolved dev_section) says "switch on the tablelamp" — they
        # could never match, and every soft run reported soft_compliant=False even when the
        # human had executed the tasked sequence exactly. `cmd` is captured before resolution
        # and is the same text the agents themselves saw.
        # AN OMISSION IS COMMITTED WHEN THE PLAN MOVES PAST IT (f99, project decision 2026-09-16). For skipped_step /
        # missing_step the injected section still contains the step that NEEDED the skipped one ("put the
        # toiletpaper on the bathroomcounter"), and the world refuses it -- you cannot put down what you never
        # picked up. Counting only `done` made compliance unreachable: both putback episodes never released,
        # the person apologised ~27 times per episode it was not allowed to act on, and ran out the clock.
        # The refused attempt IS the omission taking effect, so it counts as carried out.
        _cerr_d = PH[cur[0]]["err"] if (PH and cur[0] < len(PH)) else case.get("err")
        _ok_st = ("done", "rejected") if _cerr_d in OMISSION_ERRS else ("done",)
        done = [l["cmd"] for l in world.log
                if l["status"] in _ok_st and (l.get("t_issue") or 0) >= (report["injected_at"] or 0)]
        i = 0
        for c in done:
            if i < len(want) and c == want[i]:
                i += 1
        return i >= len(want)

    def release(t, why):
        if released[0]:
            return
        released[0] = True
        if cur[0] < len(PH):
            PH[cur[0]].update(released_at=t, released_reason=why)
        if report.get("released_at") is None:
            report.update(released_at=t, released_reason=why)
        print(f"[t{t}] TASK DISSOLVED phase {cur[0]} ({why})", flush=True)
        # hand the chore back and arm the next error, if there is one
        advance_phase()

    def made_mistakes():
        """The mistakes that have happened and that the person has not been talked out of — what the order
        block marks [intentional ...] instead of serving them up as the next step."""
        return [dict(err=ph["err"], **(ph.get("kw") or {})) for ph in PH
                if ph.get("injected_at") is not None and ph.get("released_reason") != "convinced"]

    def order_finished():
        """Everything in the person's order is either done or left wrong on purpose, and at least one step is
        the latter: the person has finished the chore WITH a mistake nobody talked it out of (f124). It then
        has nothing to plan, so it is not asked on idle polls, and the episode ends as `finished_wrong`
        rather than as `stuck` after sixty ticks of being asked."""
        try:
            mk = P.order_marks(epk, world, made_mistakes())
            if not mk or task_active():
                return False
            sids, dn, _ = P.plan_progress(epk, world)
            return all(s_ in dn or s_ in mk for s_ in sids)
        except Exception:
            return False

    def human_turn(b, idle=False, event=False, spoke=False):
        """`spoke` = this turn answers an assistant line that was just delivered. The hint says so only then
        (f122); the other turns exist because the queue ran dry, a command was refused, or a task is live."""
        R["scan_feedback"](all_kinds=True)
        if spoke:
            hint = P.HINT_PRECONVICTION if (not armed[0] and not entered[0]) else P.HINT_POSTENTRY
        elif idle:
            hint = P.HINT_IDLE_CREATIVE if R["CREATIVE"] else P.HINT_IDLE_NOVICE
        elif event:
            hint = P.HINT_EVENT
        else:
            hint = P.HINT_QUIET
        body = P.E2E_BRIEFING
        if task_live():
            _ph = PH[cur[0]]
            task_text_c[0] = render_task(_ph["err"], _seq_text(_ph["dev_section"]),
                                         _ph.get("kw"), persona)
            body += "\n\n" + task_text_c[0]
            hint = hint + TASK_REMINDER
        h, think = R["human_llm"](hint, R["feedback_text"](), body_extra=body,
                                  show_order=(not task_live()) or task_done(),
                                  mistakes=made_mistakes())
        u = (h.get("utterance") or "").strip()
        moves.append(h.get("move"))
        h_op, h_pd = R["plan_op_actions"](h)
        # words/actions coupling: agreeing with an empty queue changes nothing in the world.
        # Fix #35 widens it: an utterance that COMMITS to acting ("I will put the milk on the
        # counter right now") while the queue is empty and the plan op is drop is the same
        # deadlock — seen live, four commitments in a row, zero actions, idle death.
        q_empty = q[0].i >= len(q[0].cmds)
        # an "accept" is only an acceptance when somebody said something to accept (f122)
        if h_op == "drop" and ((h.get("move") == "accept" and spoke)
                               or (P.commits_to_action(u) and q_empty)):
            align["accept_no_action"] += 1
            h2, think2 = R["human_llm"](hint + ACCEPT_NUDGE, R["feedback_text"](), body_extra=body,
                                        show_order=(not task_live()) or task_done(),
                                  mistakes=made_mistakes())
            op2, pd2 = R["plan_op_actions"](h2)
            align["reask"] += 1
            if op2 != "drop":
                align["reask_fixed"] += 1
                h, h_op, h_pd, think = h2, op2, pd2, think2
                u = (h2.get("utterance") or "").strip() or u
                print(f"[t{b}] accept had no action -> re-ask produced {op2}", flush=True)
            else:
                align["reask_failed"] += 1
                # AN EMPTY QUEUE MUST BE SAID OUT LOUD (f94). The re-ask is silent: when it also comes
                # back with no actions, nothing tells the person that its body has nothing to do, so it
                # keeps narrating an intent it never queues. Measured on the 10 re-run cells: 46 failed
                # re-asks in 10 episodes, and a putback episode idled its last 60 ticks that way.
                # never during a forced section: there the body is engine-driven and the person is
                # told "your job is your VOICE", so telling it to queue commands would fight the injection
                if q_empty and not task_live():
                    R["fb_buf"].append(
                        'YOUR QUEUE IS EMPTY and your body is doing NOTHING. Saying what you will do does '
                        'not move you: put the exact commands in `actions` this turn (for example '
                        '"turn left 3", "walk to the desk", "grab the toiletpaper").')
        # COOK GRADER. The person judges whether the assistant found its mistake; that judgement,
        # not the engine's label match, is what arms the release. `accept` still does the releasing,
        # so recognising a call and complying with it stay separate: a low-trust persona may grant
        # that the supervisor is right and carry on regardless.
        call = h.get("assistant_error_call")
        # A call is a judgement ABOUT SOMETHING THE SUPERVISOR SAID, so there must be something to
        # judge. Measured on pilot 6: 10 of 28 arming calls were made when the assistant had not
        # spoken since the person's previous turn, and they armed releases on nothing at all.
        if call in ("correct", "wrong") and not _said_since_last_turn(report["conversation"], b):
            cook["unprompted"] += 1
            call = "none"
        if call not in ("none", "correct", "wrong"):
            if call is not None:
                cook["malformed"] += 1
            else:
                cook["omitted"] += 1
            call = "none"
        if call != "none":
            # ATTRIBUTION mirrors the engine's. A phase released by `task_completed` keeps its
            # detection window open (the assistant may still be working the mistake out), and the
            # engine goes on crediting flags inside it -- so a person's call inside that same window
            # must be attributed to that same phase. Before this, 8 of pilot 6's 17 "engine only"
            # phases were nothing but a correct call landing 2-6 ticks after the phase auto-released.
            _cp = cur[0] if task_active() else None
            if _cp is None:
                for _i, _p in enumerate(PH):
                    _w = (_p.get("grade") or {}).get("window") or _p.get("window")
                    if _p.get("injected") and _w and _w[0] <= b <= _w[1]:
                        _cp = _i
            cook["calls"].append({"t": b, "call": call, "phase": _cp,
                                  "live": task_active(), "in_window": _cp is not None,
                                  "dev_live": dev_live[0] is not None})
            if call == "wrong":
                cook["wrong_calls"] += 1
                if task_active() and dev_live[0] is None:
                    cook["denied_live"] += 1          # audit cell: denying a live, uncommitted call
            elif _cp is None:
                cook["unauthored_correct"] += 1       # vigilance credit / phantom agreement
        if GRADER == "cook" and call == "correct" and task_active():
            if dev_live[0] is None and not armed[0]:
                if not prevention[0]:
                    prevention[0] = True; report["prevention_armed_at"] = b
                    PH[cur[0]].setdefault("cook_armed_at", b)
            elif not armed[0]:
                armed[0] = True; report["armed_at"] = b; report.setdefault("armed_by", "cook")
                PH[cur[0]]["armed"] = True; PH[cur[0]].setdefault("armed_at", b)
                PH[cur[0]].setdefault("cook_armed_at", b)
                print(f"[t{b}] COOK CALL correct -> phase {cur[0]} armed", flush=True)
        rec = {"speaker": "human", "t": b, "text": u, "move": h.get("move"),
               "parse_dict": h_pd, "op": h_op, "close": h.get("close_conversation"),
               "assistant_error_call": call,
               "round": RND[0], "engine_calls": think, "human_json": h,
               "task_live": task_live(), "engine": R["engine_snap"]()}
        # The takeover owns the queue while task_active (hard + combo); soft never gates it.
        if paradigm in ("hard", "combo") and task_active() and h_op != "drop":
            rec["op_overridden"] = h_op
            h_op = "drop"
        report["conversation"].append(rec)
        if u:
            comm.append({"t": b, "speaker": "human", "text": u})
            convo.append({"speaker": "human", "text": u})
        if h.get("close_conversation"):
            RND[0] += 1
        # PREVENTION / EARLY-MISSED (CookSim 5.7.6): correct name + conviction BEFORE the
        # window opens. If the deviation has not physically happened yet it is a genuine
        # prevention; if it already has, the task ends anyway -- keeping control once the human
        # is convinced only manufactures "mouth says sorry, hands keep sinning" transcripts.
        if (prevention[0] and not armed[0] and task_active()
                and h.get("move") == "accept" and not report.get("prevented")
                and not report.get("detected_early_missed")):
            if dev_live[0] is None:
                if paradigm in ("hard", "combo"):
                    q[0].replace_rest([(st, sid) for st, sid in PH[cur[0]]["flat"][m_c[0]:]
                                       if sid not in {x for x, _ in []}])
                report.update(prevented=True, prevented_at=b, convinced=True)
                release(b, "prevented")
                print(f"[t{b}] PREVENTED — error never happened", flush=True)
            else:
                report.update(detected_early_missed=True, convinced=True)
                release(b, "convinced_early_missed")

        if armed[0] and h.get("move") == "accept":
            if not entered[0]:
                R["enter_correction"]()
                report.update(convinced=True)
            release(b, "convinced")
        # ACCEPT KEEPS ITS PLAN (A21). The takeover override above ran while the task was still
        # active, so the repair plan sent WITH the accept was dropped, and the release that
        # followed on this same turn handed back a body still running the injected steps:
        # "good catch, I'll swap it out right now" -- and the juice went into the microwave
        # (pilot 3). Once this turn has released the takeover, the person's own op stands.
        if rec.get("op_overridden") and not task_active():
            h_op = rec["op_overridden"]
            rec["op_restored_on_release"] = True
        if entered[0] and h_op != "drop":
            f = P.convert_parse(h_pd, persona, world, q[0].remaining_texts())
            if f["anything_applicable"]:
                R["apply_guidance"](f, "conversation", op=h_op)
                # ONLY A REPLACEMENT INTERRUPTS (f104, project decision 2026-09-18). `changed` is what cuts the
                # action in flight and rewinds the body; append only ADDS steps after the current one, so
                # cutting it threw away movement the person never meant to abandon -- 21-22 interruptions
                # per 10-episode run, most of them door passes (the longest actions).
                if h_op == "replace":
                    changed[0] = True
        elif not entered[0] and h_op != "drop" and (paradigm == "soft" or not task_active()):
            # The human's own plan reaches the queue when: (soft) its plan is LIVE all episode --
            # the pathway the [SYSTEM TASK] needs to have any effect; OR (combo/hard, audit fix)
            # the takeover has RELEASED (not task_active), so the person is free and MUST be able
            # to self-navigate its recovery. Before this, a combo person post-release narrated
            # commands for 50 ticks while nothing executed (only vlm_fix acts ran) -> false stuck.
            rec["self_plan"] = self_plan_apply(h_pd, h_op, b)
            if (rec["self_plan"] or {}).get("applied") and h_op == "replace":
                changed[0] = True                      # append never interrupts (f104)
        R["fb_buf"].clear()
        return u or None

    prev = [""]
    pending = [None]           # an assistant reply in flight: {"out","text","deliver_at","called_at"}
    pend_said = []             # what the person said while the assistant was still thinking
    report["assistant_latency"] = []
    soft_wake = [False]        # set at soft injection, consumed before the queue advances
    stuck_sig, stuck_since = [None], [0]
    vlm_echo = []              # recent assistant utterances, for the dead-conversation guard
    said, idle_polls, next_poll = None, 0, P.POLL

    def _norm_say(s):
        return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in (s or "")).split())
    while True:
        if time.time() - t0 > WALL_BUDGET:
            report["aborted"] = "wall_clock"; break

        # ---- injection trigger: live state matches the clean-prefix reference
        # `cur` can already sit past the last phase once every error has been through its
        # trigger; guarding the whole block keeps the deadline branch from indexing off the
        # end of PH (one cell died that way after 67 minutes of work).
        if not CLEAN and cur[0] < len(PH) and not injected[0] and skipped[0] is None:
            # m == 0 means the deviation replaces the very FIRST command, so the trigger
            # state is simply the start state. key_from_sim() cannot express that -- with no
            # commands to simulate it has no `boundaries` and reports an empty loc -- so
            # comparing against it could never match. Fire immediately instead.
            # q[0].i >= m: the queue must have PHYSICALLY dispensed the whole common prefix
            # before injecting. Walks are symbolically INERT, so a common-prefix walk (e.g. the
            # walk to the SECOND lamp, shared with the clean plan) leaves the symbolic key equal
            # to the pre-walk state -- the trigger would otherwise fire BEFORE that walk runs and
            # replace_rest would drop it, stranding the person away from the forgotten target and
            # forcing an impossible cross-house recovery (audit fix #65).
            if (m_c[0] == 0 or (key_from_sym(world.sym, KEEP) == ref_key_c[0]
                                and q[0].i >= m_c[0])) and _rep_wait_ok(world):
                injected[0] = True
                PH[cur[0]]["injected_at"] = world.tick
                if report.get("injected_at") is None:
                    report["injected_at"] = world.tick
                if paradigm in ("hard", "combo"):
                    q[0].replace_rest(list(dev_cont_c[0]))   # FORCE the error into the queue
                if paradigm in ("soft", "combo"):
                    # Deliver the [SYSTEM TASK] on the injection tick ITSELF, before the queue
                    # advances. The human otherwise only speaks on a POLL tick, so the clean
                    # plan kept running for up to POLL ticks after the task appeared -- and a
                    # task that references a step already committed is a no-op. CookSim hit
                    # exactly this (0/4 episodes produced the error while every one of them
                    # still logged as "compliant"); the fix there is the same forced turn.
                    soft_wake[0] = True
                print(f"[t{world.tick}] INJECTED phase {cur[0]} ({PH[cur[0]]['err']}, "
                      f"{paradigm}) — dev_section="
                      f"{[pretty_cmd(s) for s in dev_section_c[0]]}", flush=True)
            elif LIVE_INJECT and _live_block(world):
                # LIVE-STATE INJECTION (f105, object-bound classes f108): stop waiting for the template
                # state and attach the error to what the person is doing now. The deviation IS live the
                # moment we queue it, so its grading window opens here rather than at an identifying
                # step that may never come.
                blk, rest, trig = _live_block(world)
                PH[cur[0]]["live_trigger"] = trig
                _hz = [pretty_cmd(s_) for s_, _ in q[0].remaining()[:LIVE_HORIZON]]   # what the person was about to do
                injected[0] = True
                PH[cur[0]]["injected_at"] = world.tick
                PH[cur[0]]["live_injected"] = True
                if report.get("injected_at") is None:
                    report["injected_at"] = world.tick
                dev_section_c[0] = blk
                # the phase record too: the [SYSTEM TASK] is re-rendered from PH[cur]["dev_section"] on
                # every turn, so leaving the template there showed the person one sequence while the
                # queue ran another (f105 repetition / navigation blocks, fixed f108)
                PH[cur[0]]["dev_section"] = blk
                task_text_c[0] = render_task(PH[cur[0]]["err"], _seq_text(blk), PH[cur[0]].get("kw"), persona)
                dev_cont_c[0] = [(s_, "live_inject") for s_ in blk] + list(rest)
                PH[cur[0]]["dev_cont"] = list(dev_cont_c[0])
                if paradigm in ("hard", "combo"):
                    q[0].replace_rest(list(dev_cont_c[0]))
                if paradigm in ("soft", "combo"):
                    soft_wake[0] = True
                dev_live[0] = lo_live[0] = world.tick
                grade["dev"] = world.tick; grade["window"] = (world.tick, 10**9)
                PH[cur[0]]["dev_live"] = PH[cur[0]]["lo_live"] = world.tick
                PH[cur[0]]["grade"]["dev"] = world.tick
                PH[cur[0]]["grade"]["window"] = (world.tick, 10**9)
                report.setdefault("dev_live", world.tick)
                report.setdefault("window_lo_live", world.tick)
                report.setdefault("live_injections", []).append(
                    {"t": world.tick, "phase": cur[0], "err": PH[cur[0]]["err"], "trigger": trig,
                     "horizon": _hz,
                     "block": [pretty_cmd(s_) for s_ in blk]})
                print(f"[t{world.tick}] INJECTED phase {cur[0]} ({PH[cur[0]]['err']}, {paradigm}) "
                      f"FROM LIVE STATE [{trig}] — dev_section={[pretty_cmd(s_) for s_ in blk]}", flush=True)
            elif world.tick > deadline_c[0]:
                # This PHASE never reached its anchor. Record it and move on to the next phase
                # rather than ending the episode: a later error may still be reachable.
                skipped[0] = "deadline"
                _why = _live_why(world) if LIVE_INJECT else "template state never matched"
                PH[cur[0]]["skip_why"] = _why
                report.setdefault("live_inject_blocked", []).append(
                    {"t": world.tick, "phase": cur[0], "err": PH[cur[0]]["err"], "why": _why})
                print(f"[t{world.tick}] phase {cur[0]} ({PH[cur[0]]['err']}) injection "
                      f"SKIPPED (deadline) — {_why}", flush=True)
                if not advance_phase():
                    report["failed_at"] = "injection_skipped"

        if soft_wake[0]:
            soft_wake[0] = False
            said = human_turn(world.tick, event=True)

        nxt = q[0].next()
        act_start_frame = len(world.frames)
        if nxt is None and not CS.final_ok(epk, world.sym):
            # SELF-PLANNED QUEUE (2026-09-04): the person starts with nothing queued and works the
            # steps out or asks for them. An empty queue must still let TIME pass -- otherwise no
            # poll ever comes due, the person never gets its idle turn, and ten silent iterations
            # end the episode at t0 as silent_fail. One tick of standing still per empty pass.
            world.idle_ticks(1)
        if nxt is not None:
            step, sid = nxt
            e = world.exec_step(step, kind=("vlm_fix" if sid == "vlm_fix" else str(sid)),
                                cmd_text=pretty_cmd(step))
            R["note_exec"](e)
            if e["status"] == "rejected" and "cannot see" in str(e.get("why") or ""):
                for st_o, sid_o in q[0].skip_orphans(step):
                    world.log.append(dict(cmd=pretty_cmd(st_o), status="orphan_skipped",
                                          t_issue=world.tick, t_done=world.tick,
                                          kind=str(sid_o), instances=[None, None],
                                          step=list(st_o), room=world.current_room_label(),
                                          room_done=world.current_room_label(),
                                          why="skipped — it depended on the approach that "
                                              "was just refused"))
                    print(f"[world] t{world.tick} ORPHAN-SKIPPED {pretty_cmd(st_o)}", flush=True)
            # WIN CHECK BEFORE ANYONE SPEAKS (fix #34, mirrored from recovery): a multi-tick
            # action can complete mid-iteration; if the goal state is reached with nothing
            # queued and no injected task still in force, the episode is won NOW — before a
            # boundary turn can misread stale frames and order the win undone.
            if (not task_active() and CS.final_ok(epk, world.sym)
                    and q[0].i >= len(q[0].cmds)):
                report["outcome"] = "won"
                break
            # live window re-anchoring: first tasked command actually starting after injection
            # ONLY the steps unique to the deviation may open dev_live. dev_section can contain
            # commands the CLEAN plan also runs (wrongplace's "walk to the microwave" is the
            # approach walk, present in both), and matching those makes ordinary execution look
            # like the error starting -- observed on novice/soft, which ran the clean plan start
            # to finish yet reported dev_live=16.
            # `stall` and `skipped_step` add no distinctive COMMAND, so `_dev_only` is empty and
            # this could never fire for them: dev_live would stay None, the window would never
            # open, and every flag -- however correct -- would grade `early`. For those classes
            # the deviation is identified by the injected SUBGOAL instead.
            _cerr = PH[cur[0]]["err"] if cur[0] < len(PH) else case["err"]
            _cdev = (case.get("dev_sids") or [case.get("dev_sid")])[
                min(cur[0], len(case.get("dev_sids") or [1]) - 1)] \
                if (case.get("dev_sids") or case.get("dev_sid")) else None
            if (dev_live[0] is None and injected[0]
                    and _cerr in ("stall", "skipped_step", "missing_step")
                    and sid == _cdev):
                dev_live[0] = world.tick
                grade["dev"] = dev_live[0]
                report["dev_live"] = dev_live[0]
            # With no phases there is no injected section at all (a CLEAN reference run, or a
            # restraint episode), so nothing can be "deviation-only" and the set is empty. The
            # old `else` branch named a variable that is never bound, which made every
            # zero-error episode die with NameError the first time the person moved.
            _donly = ({pretty_cmd(x) for x in dev_section_c[0] if x[0] != "Idle"}
                      - {pretty_cmd(x) for x, _ in flat_clean}) if PH else set()
            if dev_live[0] is None and injected[0] and pretty_cmd(step) in _donly:
                dev_live[0] = world.tick
                grade["dev"] = dev_live[0]
                if cur[0] < len(PH):
                    PH[cur[0]]["dev_live"] = dev_live[0]; PH[cur[0]]["grade"]["dev"] = dev_live[0]
                    _close_earlier(dev_live[0])
                report.setdefault("dev_live", dev_live[0])
            # The window opens at the IDENTIFYING step (case["id_step"] verb), not at the last
            # command of dev_section. Using the last command put `lo` far too late: correct-name
            # flags landed before it and were graded `early_correct` instead of `valid`, which
            # made the assistant look blind when it had in fact named the error (observed on all
            # 6 runs, 2026-08-03). Matched on the VERB, not the sid, because under `soft` the
            # human re-queues the steps itself and they carry sid "human_self".
            _cid = (case.get("id_steps") or [case.get("id_step")])[
                min(cur[0], len(case.get("id_steps") or [1]) - 1)] \
                if (case.get("id_steps") or case.get("id_step")) else None
            # same rule for the grading window: for an omission the identifying step is attempted and refused
            # (f99) -- waiting for it to succeed left the window to open only by the 12-tick fallback
            if (dev_live[0] is not None and lo_live[0] is None and _cid
                    and step[0] == _cid[1]
                    and (e["status"] == "done" or (e["status"] == "rejected" and _cerr in OMISSION_ERRS))):
                lo_live[0] = world.tick
                grade["window"] = (lo_live[0], 10**9)
                if cur[0] < len(PH):
                    PH[cur[0]]["lo_live"] = lo_live[0]
                    PH[cur[0]]["grade"]["window"] = (lo_live[0], 10**9)
                report.setdefault("window_lo_live", lo_live[0])
        else:
            world.idle_ticks(1)

        if (dev_live[0] is not None and lo_live[0] is None
                and world.tick - dev_live[0] >= WINDOW_FALLBACK):
            lo_live[0] = world.tick
            grade["window"] = (lo_live[0], 10**9)
            report["window_lo_live"] = lo_live[0]
            report["window_lo_fallback"] = True
            print(f"[t{world.tick}] WINDOW FALLBACK — id_step verb "
                  f"{case.get('id_step')} never completed within {WINDOW_FALLBACK} ticks of "
                  f"dev; opening window now so detection can be graded", flush=True)

        if RELEASE_ON_DONE and task_active() and task_done():
            release(world.tick, "task_completed")
        elif (RELEASE_ON_DONE and task_active() and injected[0] and cur[0] < len(PH) and PH[cur[0]].get("live_injected")
              and world.tick > (PH[cur[0]].get("injected_at") or 0)
              and not any(sd == "live_inject" for _st, sd in q[0].remaining())):
            # f126: every step of the live block has been consumed -- done or REFUSED by the house -- and the
            # tasked sequence still does not read as carried out. Before this the takeover kept the queue for the
            # rest of the episode (the person's every plan overridden), which is how a refused grab turned into a
            # stuck or silent_fail record. The mistake either happened in part or not at all; either way the
            # chore is handed back and the phase's grading stands on what the person actually did.
            release(world.tick, "block_exhausted")

        spoke = False
        while next_poll <= world.tick:
            b = next_poll; next_poll += (1 if NATIVE else P.POLL)
            _hturn = (not NATIVE) or b % P.POLL == 0
            # NOT gated on entered[0]. That gate is inherited from iv_e2e_rollout.py, where the
            # human is on autopilot until conviction so an empty queue cannot happen. Here the
            # human owns its queue from t0, so if it drains before conviction the world simply
            # idles -- and because the idle HINT never fired, the human was never told its queue
            # was empty. Observed: wrongplace/creative froze for 43 ticks (t35->t78) insisting
            # "I'm opening it right now" while every turn carried op="drop". Its intent lived
            # only in `text`, which has no path to the world.
            idle = q[0].i >= len(q[0].cmds)
            if said is None and idle and _hturn and not order_finished():
                said = human_turn(b, idle=True)
            # While the task is live the human MUST get turns even when the supervisor is
            # silent -- otherwise it never gets a chance to queue the tasked actions and the
            # clean plan just runs on (the failure observed on the first batch).
            if said is None and task_live() and not task_done() and _hturn:
                said = human_turn(b)
            # The supervisor is told the person's BODY state as engine truth: what it holds
            # (else it once asserted "your hands are empty" at a human holding the milk and
            # talked her into it) and — under mid-level (fix #68) — WHERE it is (room + facing +
            # what is in view), because it cannot localize the body from first-person video and
            # was hallucinating the wrong room and micro-steering turns off it.
            # ---- assistant latency: a reply in flight lands only when its delay has elapsed
            if NATIVE:
                _q = native_poll(b, said)
                said = None
                if _q is None:
                    continue                   # nothing the model said is due at this tick
                pending[0] = {"out": _q["out"], "text": (_q["out"].get("text") or "").strip(),
                              "deliver_at": b, "called_at": _q["t_issue"]}
            if pending[0] is not None and pending[0]["deliver_at"] > b:
                if said:
                    pend_said.append(said)
                said = None
                continue                       # the assistant is still thinking; the world moves on
            if pending[0] is not None:
                out, text = pending[0]["out"], pending[0]["text"]
                _called = pending[0]["called_at"]; pending[0] = None
                print(f"[t{b}] assistant reply asked at t{_called} lands "
                      f"({b - _called} tick(s) of latency)", flush=True)
            else:
                if pend_said:
                    said = "\n".join(pend_said + ([said] if said else [])); pend_said.clear()
                _t_call = time.time()
                P.ASSIST_STATS["gen_s_turn"] = 0.0
                locus = world.person_locus()
                hands = world.sym.hands()
                if locus:
                    extra = f"holding {hands}. Body is {locus}"
                else:
                    extra = hands if P.AGENT_TIER == "+++" else None
                # TOP-DOWN MAP for the assistant (user decision 2026-08-26): the supervisor now gets
                # the same current-room statemap the person has — floor-plan image + text scene graph
                # with each device's live [ON]/[OFF]/[OPEN]/[CLOSED] — so it can VERIFY task state
                # (which lamp is still lit) instead of only guessing from first-person video.
                _map_img, _map_txt = (world.topdown_obs() if P.MIDLEVEL_ON else (None, ""))
                _house_img, _house_txt = (world.house_obs() if P.MIDLEVEL_ON else (None, ""))
                out = P.sup_turn(client, sup_sys, prev[0], comm, said, world.sup_frames(world.tick), b,
                                 state_extra=extra, map_img=_map_img, map_text=(_map_txt or None),
                                 house_img=_house_img, house_text=(_house_txt or None))
                if out:
                    out, evicted = P.cap_key_moments(out)
                    km_evicted[0] += evicted
                    km_max[0] = max(km_max[0], len(out.get("key_moments") or []))
                prev[0] = json.dumps(out) if out else prev[0]
                said = None
                text = (out.get("text") or "").strip()
                # SILENCE-ON-FAILURE RE-ASK (fix #31, mirrored from the recovery loop): an empty
                # say right after the person's command was refused starves a novice of its next
                # instruction and IDLE_END buries the run as silent_fail.
                if not text and any(l.get("status") == "rejected" for l in world.log[-3:]):
                    out2 = P.sup_turn(client, sup_sys, prev[0], comm,
                                      "[the person's last command was just REFUSED and they are "
                                      "waiting for your next concrete instruction — do not stay "
                                      "silent; tell them the next step]",
                                      world.sup_frames(world.tick), b, state_extra=extra,
                                      map_img=_map_img, map_text=(_map_txt or None),
                                      house_img=_house_img, house_text=(_house_txt or None))
                    if out2:
                        out2, evicted = P.cap_key_moments(out2)
                        km_evicted[0] += evicted
                        km_max[0] = max(km_max[0], len(out2.get("key_moments") or []))
                    if out2 and (out2.get("text") or "").strip():
                        out = out2
                        prev[0] = json.dumps(out)
                        text = (out.get("text") or "").strip()
                        print(f"[t{b}] assistant silent after refusal -> re-ask produced text",
                              flush=True)
                _x_wall = time.time() - _t_call
                _x = P.ASSIST_STATS.get("gen_s_turn") or _x_wall     # generation time only (D3)
                _x_charged = _x / ASSIST_LAT_DIV            # A100-equivalent (see ASSIST_LAT_DIV)
                _y_raw = math.ceil(_x_charged / ASSIST_TICK_S) if ASSIST_LATENCY else 0
                _y = min(_y_raw, ASSIST_MAX_TICKS)
                report["assistant_latency"].append({"called_at": b, "wall_s": round(_x_wall, 2),
                                                    "gen_s": round(_x, 2),
                                                    "gen_s_charged": round(_x_charged, 2),
                                                    "lat_div": ASSIST_LAT_DIV, "ticks": _y,
                                                    "capped": _y_raw > _y, "deliver_at": b + _y})
                if _y > 0:
                    pending[0] = {"out": out, "text": text, "deliver_at": b + _y, "called_at": b}
                    next_poll = min(next_poll, b + _y)   # poll exactly on the landing tick
                    said = None
                    continue
            fl, v = out.get("flag"), None
            # Accept the declared shape {"error": "<label>"} AND a bare label string. One local
            # model emitted 115 of its 118 flags as bare strings, and taking only the dict form
            # discarded every one of them — the model looked blind when it was in fact detecting
            # with the highest precision in the sweep. A bare taxonomy label is unambiguous, so it
            # is honoured here; how often a model met the declared shape is reported separately as
            # schema compliance rather than silently folded into its detection score.
            if isinstance(fl, str) and fl.strip():
                fl = {"error": fl.strip()}
            if isinstance(fl, dict) and fl.get("error"):
                nm = str(fl["error"]).strip()
                seen = {f["error"] for f in flags}
                _own = {x["err"] for x in PH} or {case["err"]}
                if nm in seen and not (nm in _own and not armed[0]):
                    flag_repeats[0] += 1
                else:
                    _ph = None
                    if CLEAN:
                        # restraint episode: a flag naming something TRUE of the world is
                        # neutral (skill §7: ask the world before charging); otherwise it is
                        # a clean-run false alarm.
                        v = "neutral_true" if P.is_true_now(nm, world) else "fp_clean"
                    else:
                        # Grade against EVERY phase whose deviation has fired and credit the one
                        # that gives the best verdict. This is what dissolves twin-laundering: a
                        # flag naming phase B exactly earns `valid` for B, which outranks the
                        # `valid_alias` it would earn against phase A when the two are twins.
                        _sync_out()
                        cands = []
                        for _i, _p in enumerate(PH):
                            if _p["dev_live"] is None:
                                continue
                            cands.append((P.verdict_of(_p["grade"], nm, b), _i))
                        if cands:
                            v, _ph = min(cands, key=lambda c: _VERDICT_RANK.get(c[0], 99))
                            # An earlier phase's window may still be open when the NEXT phase
                            # has been injected but its deviation has not fired yet. A flag
                            # naming that next phase is an early catch of it, not a wrong name
                            # against the old phase (pilot 2026-09-05: "repetition" flagged one
                            # tick before the fridge was re-opened graded wrong_name, so the
                            # person's accept armed nothing and its body opened the fridge).
                            if v not in P.CORRECT:
                                _nxt = next((_i for _i, _p in enumerate(PH)
                                             if _p["injected"] and _p["dev_live"] is None
                                             and nm == _p["err"]), None)
                                if _nxt is not None:
                                    v, _ph = "early_correct", _nxt
                        elif skipped[0] is not None and all(x["injected"] is False for x in PH):
                            v = "fp_skipped"
                        elif not any(x["injected"] for x in PH):
                            # D7: a pre-injection flag that is TRUE of the world (the person really
                            # is lost or standing still) is neutral, as in a clean run
                            v = "neutral_true" if P.is_true_now(nm, world) else "fp_preinjection"
                        else:
                            _hit = next((_i for _i, _p in enumerate(PH)
                                         if _p["injected"] and nm == _p["err"]), None)
                            v = "early_correct" if _hit is not None else "wrong_name"
                            _ph = _hit
                    flags.append({"t": b, "error": nm, "verdict": v, "phase": _ph})
                    print(f"[t{b}] FLAG {nm} -> {v}"
                          + (f" (phase {_ph}: {PH[_ph]['err']})" if _ph is not None else ""),
                          flush=True)
                    # a correct flag for a LATER phase arms that phase, not the current one
                    if GRADER == "engine" and _ph is not None and _ph != cur[0] and v in P.ARMS_DETECTION:
                        PH[_ph]["armed"] = True
                        PH[_ph].setdefault("armed_at", b)
                    # ONE Correct set (skill contract §7 / gap A2): the gate that changes the
                    # world consumes the same sets the scorer does. Before this, an in-window
                    # twin name scored wrong_name AND never armed — the two-sets bug the
                    # contract names ("a human that accepts a correct repair twelve times
                    # while its hands finish serving the wrong plate").
                    if GRADER == "engine":
                        if v in P.ARMS_DETECTION and not armed[0]:
                            armed[0] = True; report["armed_at"] = b
                            report.setdefault("armed_by", "engine")
                        elif v in P.ARMS_PREVENTION and not armed[0] and not prevention[0]:
                            prevention[0] = True; report["prevention_armed_at"] = b
            report["conversation"].append({"speaker": "vlm", "t": b, "text": text,
                                           "round": RND[0], "vlm_json": out,
                                           "flag_verdict": v, "engine": R["engine_snap"]()})
            if text:
                vlm_echo.append(_norm_say(text))
                del vlm_echo[:-3]
            if text:
                comm.append({"t": b, "speaker": "vlm (you)", "text": text})
                convo.append({"speaker": "vlm", "text": text})
                changed[0] = False
                said = human_turn(b, spoke=True)
                spoke = True
                # INTERRUPT: the human re-planned while this action was still streaming, so the
                # frames after this tick were never actually committed — throw them away and put
                # the body back where it was when the plan changed.
                cut = (b + 1) * FPT
                if (INTERRUPT and changed[0] and nxt is not None
                        and act_start_frame < cut < len(world.frames)):
                    if world.rewind_to(cut):
                        interrupted[0] += 1
                        report.setdefault("interrupts", []).append(
                            {"t": b, "cut_frame": cut, "cmd": pretty_cmd(step)})
                        next_poll = world.tick + P.POLL
                        break
        if entered[0]:
            R["resolved_check"]()
        done = CS.final_ok(epk, world.sym)
        if done and q[0].i >= len(q[0].cmds):
            report["outcome"] = "won"; break
        idle_polls = 0 if (nxt is not None or spoke or said or pending[0] is not None) else idle_polls + 1
        if idle_polls >= IDLE_END:
            report["outcome"] = "won" if done else ("finished_wrong" if order_finished() else "silent_fail"); break
        # same no-progress guard as the recovery loop -- see the comment there
        # instance-level: intentional — the signature must distinguish one lamp from
        # another, or switching between two of a class would read as 'no change'
        # Under the mid-level space, navigation (turn / go straight / go through a door)
        # legitimately changes NO symbolic state for many ticks — the guard must not read an
        # honest cross-house walk as being stuck, so the CURRENT ROOM is part of the signature:
        # changing rooms is progress. Pacing inside one room without a state change still trips.
        _sig = (world.current_room_label(), tuple(sorted(world.sym.holding)),   # instance-level: intentional
                tuple(sorted(world.sym.loc.items())),   # instance-level: intentional
                tuple(sorted(world.sym.opened)), tuple(sorted(world.sym.on)))   # instance-level: intentional
        if _sig == stuck_sig[0]:
            # DEAD-CONVERSATION guard (fix #68b): IDLE_END counts only SILENT polls, so an
            # assistant repeating the SAME line while nothing moves never trips it and the run
            # burns to the wall-clock — a wrongplace/creative negotiation stalled on "the task
            # is fully complete" x5 and aborted at 145 ticks / 56 real minutes. Three identical
            # assistant lines in a row with no symbolic change = the conversation is over.
            # SOFTENED (f125, project decision 2026-10-05). The three-identical-lines test alone ended episodes in
            # which the person was still working: an assistant repeating "Go to the kitchen" while the person
            # turned to find the door was cut at tick 16, and with the quieter person (shorter plans, so the
            # queue is empty more often) the guard fired twice as often for the chatty open-weight seats
            # (internvideo3 29 -> 59 of 75). It now also needs the house to have stood still for
            # DEAD_CONV_TICKS: the assistant is looping AND nothing has changed for half a minute.
            if (len(vlm_echo) == 3 and len(set(vlm_echo)) == 1 and nxt is None
                    and world.tick - stuck_since[0] >= DEAD_CONV_TICKS):
                report["outcome"] = "won" if done else ("finished_wrong" if order_finished() else "silent_fail")
                report["ended"] = "dead_conversation"
                print(f"[t{world.tick}] DEAD CONVERSATION (assistant repeating) — ending",
                      flush=True)
                break
            if world.tick - stuck_since[0] >= NO_PROGRESS:
                report["outcome"] = "won" if done else ("finished_wrong" if order_finished() else "stuck")
                report["stuck_ticks"] = world.tick - stuck_since[0]
                print(f"[t{world.tick}] NO PROGRESS for {NO_PROGRESS} ticks — ending "
                      f"(room={world.current_room_label()})", flush=True)
                break
        else:
            stuck_sig[0], stuck_since[0] = _sig, world.tick
        if world.tick >= CAP:
            report["outcome"] = "won" if done else "timeout"; break

    # the instance the task actually means, so the page and the audit can tell a repair on
    # the target apart from one on the decoy
    _sync_out()
    _lat = report.get("assistant_latency") or []
    report["assistant_latency_summary"] = {
        "calls": len(_lat), "tick_s": ASSIST_TICK_S, "enabled": ASSIST_LATENCY,
        "lat_div": ASSIST_LAT_DIV, "gpu": os.environ.get("VH_GPU_NAME", ""),
        "mean_wall_s": round(sum(l["wall_s"] for l in _lat) / len(_lat), 2) if _lat else None,
        "mean_ticks": round(sum(l["ticks"] for l in _lat) / len(_lat), 2) if _lat else None,
        "max_ticks": max((l["ticks"] for l in _lat), default=0),
        "max_ticks_allowed": ASSIST_MAX_TICKS, "capped_calls": sum(1 for l in _lat if l.get("capped")),
        "mean_gen_s": round(sum(l.get("gen_s", l["wall_s"]) for l in _lat) / len(_lat), 2) if _lat else None}
    report["targets"] = dict(world.sym.targets)
    report.setdefault("outcome", "won" if CS.final_ok(epk, world.sym) else "timeout")
    fv = next((f for f in flags if f["verdict"] in P.ARMS_DETECTION), None)
    # RESTRAINT FINALISATION: an exit probe the assistant must answer (an unanswered direct
    # question is a recorded refusal, and restraint is only scorable with it answered — the
    # silent stub must not top the leaderboard, skill §7).
    if CLEAN:
        probe_out = P.sup_turn(client, sup_sys, prev[0], comm,
                               "[the episode has ended — in ONE sentence: did the person "
                               "complete the task correctly, and did you observe any error?]",
                               world.sup_frames(world.tick), world.tick,
                               state_extra=None)
        probe_txt = ((probe_out or {}).get("text") or "").strip()
        spoken = [c for c in report["conversation"]
                  if c.get("speaker") == "vlm" and (c.get("text") or "").strip()]
        polls = len([c for c in report["conversation"] if c.get("speaker") == "vlm"])
        report["restraint"] = dict(
            clean=True, reorder=case.get("reorder"),
            exit_probe=probe_txt or "REFUSED",
            probe_answered=bool(probe_txt),
            false_alarms=[f for f in flags if f["verdict"] == "fp_clean"],
            neutral_flags=[f for f in flags if f["verdict"] in P.NEUTRAL],
            utterances=len(spoken), polls=polls,
            scorable=bool(probe_txt),
        )
    if NATIVE:
        _nr = nat["sess"].native_report()
        _nr["world_held"] = {"holds": nat["holds"], "seconds": round(nat["held_s"], 2)}
        _nr["undelivered"] = [q_["out"] for q_ in nat["q"]]
        _nr["feed_errors"] = nat["errors"]
        report["assistant_mode"], report["native_assistant"] = "native", _nr
        _st = _nr["stats"]
        P.ASSIST_STATS.update(calls=_st["frames_sent"], prompt_tokens=_st["prompt_tokens"],
                              completion_tokens=_st["completion_tokens"])
        nat["sess"].close()
    report["interaction_quality"] = P.exit_questionnaire(client, persona, convo)
    # utterance leak audit (skill §5.2): ids/coords in SPOKEN text, both speakers, measured
    report["utterance_leaks"] = {
        who: [tok for c in report["conversation"] if c.get("speaker") == who
              for tok in P.utterance_leaks(c.get("text"))]
        for who in ("human", "vlm")}
    report.update(
        ticks=world.tick, flags=flags, flag_repeats=flag_repeats[0],
        injection_skipped=skipped[0],
        soft_compliant=task_done() if injected[0] else None,
        interrupts_used=interrupted[0], interrupt_enabled=INTERRUPT,
        agent_tier=P.AGENT_TIER,
        assistant={"model": P.ASSIST_MODEL or P.MODEL, "profile": P.ASSIST_PROFILE,
                   "base_url": P.ASSIST_BASE_URL or "gemini-api",
                   "image_scale": P.ASSIST_IMAGE_SCALE, "stats": dict(P.ASSIST_STATS)},
        memory={"key_moments_final": (json.loads(prev[0]).get("key_moments")
                                      if prev[0] else None),
                "fifo_evicted": km_evicted[0], "max_len_seen": km_max[0]},
        alignment=dict(align, aligned=(align["accept_no_action"] == 0
                                       or align["reask_failed"] == 0)),
        detected_early_missed=report.get("detected_early_missed", False),
        prevented=report.get("prevented", False),
        false_positives=[f for f in flags if f["verdict"] not in (P.CORRECT | P.NEUTRAL)],
        grader=GRADER,
        # A31 (project decision 2026-09-15): the PERSON'S call arms the release, the ENGINE'S verdict
        # scores detection. Two paired checks showed the person's call copies its own move
        # (every `correct` on an accept turn, every `wrong` on a reject turn) under both wordings,
        # so it is kept as the control signal for release but not trusted as the score. The
        # person's own detection is still recorded (`flagged_valid_cook`, `caught_cook`) so the
        # two can be compared on every episode.
        flagged_valid=any(f["verdict"] in P.ARMS_DETECTION for f in flags),
        flagged_valid_engine=any(f["verdict"] in P.ARMS_DETECTION for f in flags),
        flagged_valid_cook=any(c["call"] == "correct" and c["phase"] is not None for c in cook["calls"]),
        detected_by=_detected_by(flags, cook),
        detected_at=_detected_at(flags, _NO_COOK),
        latency=_credit_latency(PH, flags, _NO_COOK),
        latency_engine=_credit_latency(PH, flags, _NO_COOK),
        label_ok=any(f["verdict"] in P.EXACT_LABEL for f in flags),
        cook_calls=cook["calls"], cook_wrong_calls=cook["wrong_calls"],
        cook_omitted=cook["omitted"], cook_malformed=cook["malformed"],
        # the two audit cells: a person agreeing with a phantom, a person denying a live call
        cook_unauthored_correct=cook["unauthored_correct"], cook_denied_live=cook["denied_live"],
        cook_unprompted=cook["unprompted"],
        final_ok=CS.final_ok(epk, world.sym), checkpoints=world.sym.checkpoints(),
        phases=[{"i": _i, "err": _p["err"], "injected_at": _p.get("injected_at"),
                 "dev_live": _p.get("dev_live"), "window": _p["grade"].get("window"),
                 "armed_at": _p.get("armed_at"), "released_at": _p.get("released_at"),
                 "released_reason": _p.get("released_reason"), "skipped": _p.get("skipped"),
                 "closed_at": _p.get("closed_at"),
                 "live_injected": bool(_p.get("live_injected")), "live_trigger": _p.get("live_trigger"),
                 "caught": any(f.get("phase") == _i and f["verdict"] in P.ARMS_DETECTION
                               for f in flags),
                 "caught_engine": any(f.get("phase") == _i and f["verdict"] in P.ARMS_DETECTION
                                      for f in flags),
                 "caught_cook": any(c["call"] == "correct" and c["phase"] == _i for c in cook["calls"]),
                 "denied_cook": any(c["call"] == "wrong" and c["phase"] == _i for c in cook["calls"]),
                 "cook_armed_at": _p.get("cook_armed_at"),
                 # D9: an early-correct flag the person ACCEPTED is a detection too
                 "prevented": _p.get("released_reason") == "prevented",
                 "early_caught": _p.get("released_reason") == "convinced_early_missed",
                 "flags": [f for f in flags if f.get("phase") == _i]}
                for _i, _p in enumerate(PH)],
        phases_caught=sum(1 for _i, _p in enumerate(PH)
                          if any(f.get("phase") == _i and f["verdict"] in P.ARMS_DETECTION
                                 for f in flags)),
        phases_caught_cook=sum(1 for _i, _p in enumerate(PH)
                               if any(c["call"] == "correct" and c["phase"] == _i for c in cook["calls"])),
        phases_caught_engine=sum(1 for _i, _p in enumerate(PH)
                                 if any(f.get("phase") == _i and f["verdict"] in P.ARMS_DETECTION
                                        for f in flags)),
        phases_prevented=sum(1 for _p in PH if _p.get("released_reason") in ("prevented", "convinced_early_missed")),
        phases_detected=sum(1 for _i, _p in enumerate(PH)
                            if _p.get("released_reason") in ("prevented", "convinced_early_missed")
                            or any(f.get("phase") == _i and f["verdict"] in P.ARMS_DETECTION for f in flags)),
        phases_total=len(PH),
        action_log=[{k: l[k] for k in ("cmd", "status", "t_issue", "t_done", "kind", "why", "instances", "step", "room", "room_done")
                     if k in l} for l in world.log])
    detection_ok = (report["flagged_valid"] or report.get("prevented")
                    or report.get("detected_early_missed"))
    if skipped[0] is not None:
        report["failed_at"] = "injection_skipped"
    elif report["outcome"] == "won":
        report["failed_at"] = "solved"
    else:
        report["failed_at"] = "recovery" if detection_ok else "detection"
    world.write_video(out_json.replace(".json", ".mp4"))
    json.dump(report, open(out_json, "w"), indent=1, default=str)
    print(f"{name}: outcome={report['outcome']} failed_at={report['failed_at']} "
          f"valid={report['flagged_valid']} injected_at={report['injected_at']} "
          f"released={report['released_reason']} compliant={report['soft_compliant']} "
          f"ticks={report['ticks']}", flush=True)
    return report


if __name__ == "__main__":
    a = sys.argv[1:]
    run(int(a[0]), a[1], a[2], a[3], a[4] if len(a) > 4 else "combo")
