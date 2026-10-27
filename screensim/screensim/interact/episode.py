"""The episode loop (skill §0.2). One case, one human, one assistant.

Order per turn: assistant poll on cadence (freshest observation) → human
gesture → world tick. The takeover holds the human on the flawed plan and,
before an irreversible commit, HOLDS one extra beat so the assistant gets a
last window; it releases on a CORRECT detection that the human accepts (the
single verdict set imported from `verdicts`) or, failing that, on maximum hold
(the held commit fires and the error lands). A held plan update is reported to
the human, never dropped silently (skill §5.3).

Two modes: SCRIPTED (frame_fn=None, deterministic, no API) exercises every
mechanism; LLM (frame_fn captures PNGs) drives the real agents.
"""
from __future__ import annotations

import copy

from ..core import pacing
from ..core.engine import Device
from ..tasks import resolve_dynamic, by_id
from . import INTERFACE_VERSION, interface, readout
from .plan_ir import deep_diff
from .grade import (flag_matches_case, grade_detection, grade_restraint,
                    validity_gates)

POLL = 2
MAX_TURNS = 80
IRREV = ("bank.history", "bank.balance", "shop.order", "mail.sent")


def run_episode(case, ir, human, assistant, *, frame_fn=None,
                interface_version=INTERFACE_VERSION,
                expected_interface=None, k_frames=4, film_cap=None):
    pacing.apply_conversation_pace()
    try:
        return _run(case, ir, human, assistant, frame_fn, interface_version,
                    expected_interface or interface_version, k_frames, film_cap)
    finally:
        pacing.reset()


def _run(case, ir, human, assistant, frame_fn, interface_version,
         expected_interface, k_frames, film_cap=None):
    iface = interface(interface_version)
    task = by_id(case.task)
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)

    dev_screen = ir.nodes[case.node_idx].screen if case.node_idx >= 0 else None
    transcript, concerns, history, frames = [], [], [], []
    timeline = []                   # timed events: {t, kind, who, text}
    human_turns = []                # per-turn say-vs-do (fidelity audit)
    film = []                       # [(png_bytes, caption, tick)] for the reel
    last_line = ""

    def say(who, text):
        nonlocal last_line
        transcript.append((who, text))
        timeline.append({"t": dev.st.tick, "kind": "say", "who": who,
                         "text": text})
        last_line = f"{who}: {text}"

    def film_frame(kind):
        if film_cap is not None:
            try:
                film.append((film_cap(dev), last_line, dev.st.tick, kind))
            except Exception:
                pass

    leaks = 0
    injected_at = None
    reached_dev_screen = False      # human is on the deviation screen
    dev_observable = False          # the wrong choice is now real on screen
    accepted = False                # a correction has been accepted (latched)
    committed = False
    repaired = False
    hold_polls = 0
    turn = 0
    empty_streak = 0                # consecutive no-gesture turns (LLM human)
    repeat_gesture = 0              # same gesture emitted back-to-back
    no_progress = 0                 # gesture-turns with no world-state change
    last_gesture = None
    last_hash = dev.st.hash()
    stalled = False
    feedback = ""
    scripted = hasattr(human, "peek")
    slip_seen = scripted            # scripted humans always conduct the slip
    # ---- LLM-human takeover state (soft-hard, as in the scripted queue) ----
    # the engine FORCES the slip at the anchor and HOLDS the human on the
    # flawed plan (blocks self-repair) until a correct detection releases it
    # or the error commits. Prompt-only injection is never trusted to fire.
    released = False                # soft handoff done (a correct flag landed)
    held_blocks = 0                 # blocked self-repairs while held (give-up cap)
    _dnode = ir.nodes[case.node_idx] if (case.node_idx is not None
                                         and case.node_idx >= 0) else None
    anchor_screen = ((_dnode.from_screen if case.err == "wrong_page"
                      else _dnode.screen) if _dnode else None)
    prior_screens = ({n.screen for n in ir.nodes
                      if n.idx < case.node_idx and n.kind in ("commit", "reach")}
                     if _dnode else set())
    visited_screens = set()
    correct_gest = ((_dnode.gesture or "") if _dnode else "").lower()\
        .replace("tap ", "").strip()
    repair_labels = [g.lower().replace("tap ", "").strip()
                     for g in (case.repair or [])]

    film_frame("open")               # opening frame of the reel

    def observable():
        return dev_observable and not committed and not accepted

    def poll_assistant():
        nonlocal leaks
        rel = readout.relational(dev) if iface["relational"] else ""
        if frame_fn is not None and iface["frames"]:
            frames.append(frame_fn(dev))
            frames[:] = frames[-k_frames:]
        memory = concerns[-1][1] if concerns else ""
        try:
            reply = assistant.turn(frames=list(frames), relational_text=rel,
                                   memory=memory,
                                   deviation_observable=observable())
        except TypeError:
            reply = assistant.turn(frames=list(frames), relational_text=rel,
                                   memory=memory)
        text = (reply.get("text") or "").strip()
        concern = (reply.get("concern") or "").strip()
        if text and _has_coord(text):
            leaks += 1
        if text:
            say("assistant", text); film_frame("say")
        if concern or text:
            concerns.append((dev.st.tick, concern or text))
        # takeover release: a CORRECT detection the human accepts
        nonlocal accepted, released
        # SCRIPTED human: the forced-accept takeover (queue splice) fires here.
        if scripted and observable() and flag_matches_case(concern or text, case):
            react = human.note_correction()
            if react == "accept":
                accepted = True             # latch: the takeover is released
                say("human", "okay — let me fix that.")
            else:
                say("human", "you sure? I think it's fine.")
            return react
        # LLM human: a correct detection is the SOFT handoff — the engine stops
        # holding the human on the flawed plan and lets it repair organically
        # (whether it actually fixes it is then the model's own choice, graded).
        if (not scripted and slip_seen and not released
                and flag_matches_case(concern or text, case)):
            released = True
        return None

    while turn < MAX_TURNS and dev.st.outcome is None:
        if turn % POLL == 0:
            poll_assistant()

        peek = human.peek() if hasattr(human, "peek") else "g"
        if peek in ("HOLD", "SKIPHOLD", "LOSTHOLD"):
            # last-chance beat: before the irreversible commit, while the human
            # sits on the deviation screen with a step undone, or while it is
            # stuck on the WRONG screen after a bad turn
            if peek in ("SKIPHOLD", "LOSTHOLD"):
                dev_observable = True       # the mistake is now on screen
                if injected_at is None:
                    injected_at = dev.st.tick
            react = poll_assistant()
            hold_polls += 1
            # give a rebutting persona enough beats to come around before the
            # maximum-hold releases the commit
            budget = 1 + getattr(getattr(human, "persona", None), "rebuttals", 0)
            if react != "accept" and hold_polls > budget:
                human.clear_hold()          # maximum hold: let the error land
            turn += 1
            continue
        if peek == "END":
            # a terminal skip is observable now: the human is standing on the
            # deviation screen having left the control unset
            if case.err == "skip_step" and reached_dev_screen and not committed \
                    and not repaired:
                dev_observable = True
            if observable():
                react = poll_assistant()
                if react == "accept":
                    continue          # the human now has repair gestures queued
            break

        h = human.turn(exact_map=readout.exact_map(dev),
                       proprioception=readout.proprioception(dev),
                       history="\n".join(f"  {g}" for g in history[-6:]),
                       feedback=feedback,
                       transcript=_render_transcript(transcript))
        utter = (h.get("utterance") or "").strip()
        gesture = (h.get("gesture") or "").strip()
        if utter:
            say("human", utter)
        if not gesture:
            # an LLM human with nothing to do, three running, has finished
            empty_streak += 1
            if not scripted and empty_streak >= 3:
                break
            turn += 1
            continue
        empty_streak = 0

        # ================= LLM-human soft-hard takeover =================
        # The engine, not the prompt, makes the error happen. At the deviation
        # anchor it OVERRIDES the model's gesture with the designed slip (hard),
        # then HOLDS the human on the flawed plan — blocking any self-repair —
        # until a correct detection releases it (soft) or the error commits.
        if not scripted and case.family == "error" and anchor_screen is not None:
            visited_screens.add(dev.st.cur_screen_key())
            here = dev.st.cur_screen_key()
            gl = gesture.lower().replace("tap ", "").strip()
            at_anchor = (here == anchor_screen and prior_screens <= visited_screens)
            if not slip_seen and at_anchor:
                if case.err in ("wrong_value", "wrong_target", "wrong_page") \
                        and case.alt_gesture:
                    # force the wrong tap / wrong turn, whatever the model chose
                    gesture = case.alt_gesture
                elif case.err == "skip_step":
                    # force the OMISSION: drop the correct commit if attempted,
                    # then let the human move on none the wiser
                    if correct_gest and correct_gest in gl:
                        slip_seen = True
                        dev_observable = True
                        if injected_at is None:
                            injected_at = dev.st.tick
                        history.append(f"(skipped: {gesture})")
                        turn += 1
                        continue
            # HOLD: while the slip stands uncorrected and unreleased, the human
            # cannot undo it on their own (they are "sure it's right")
            elif slip_seen and not released and not accepted:
                is_repair = gesture != case.alt_gesture and any(
                    lbl and lbl in gl for lbl in repair_labels)
                # for wrong_page the repair starts with "press back" off the
                # wrong screen; block that too so the human stays lost
                if case.err == "wrong_page" and gesture.lower().startswith("press back"):
                    is_repair = True
                if is_repair:
                    history.append(f"(held on plan: {gesture})")
                    held_blocks += 1
                    # maximum hold: after a few blocked self-repairs with no
                    # correct detection, the human gives up on their own doubt
                    # and the error lands (the scripted LOSTHOLD analogue)
                    if held_blocks >= 3:
                        say("human", "eh, I'm sure it's fine. Moving on.")
                        break
                    turn += 1
                    continue

        # stall guards (LLM human): the same gesture repeated, or many turns
        # with no change to the world, means it is stuck in a loop — end it.
        if not scripted:
            repeat_gesture = repeat_gesture + 1 if gesture == last_gesture else 0
            last_gesture = gesture
            if repeat_gesture >= 3:
                stalled = True
                break

        pre = copy.deepcopy(dev.st.data)
        scr_before = dev.st.cur_screen_key()
        r = dev.act(resolve_dynamic(dev, gesture))
        history.append(gesture)
        timeline.append({"t": dev.st.tick, "kind": "gesture", "who": "human", "text": gesture})
        # per-turn say-vs-do record, for the fidelity audit
        human_turns.append({"t": dev.st.tick, "utterance": utter,
                            "gesture": gesture, "ok": r.ok,
                            "reason": r.reason, "screen_before": scr_before,
                            "screen_after": dev.st.cur_screen_key()})
        film_frame("gesture")
        feedback = "" if r.ok else f"That didn't work: {r.reason}"
        step = deep_diff(pre, copy.deepcopy(dev.st.data))

        # ---- did the human ACTUALLY conduct the designed slip? ----
        # scripted humans mark it with _dev; for an LLM human we detect the
        # slip GESTURE-ACCURATELY so a run that never made the mistake is not
        # scored as if it did.
        if h.get("_dev"):
            slip_seen = True
            dev_observable = True
            if injected_at is None:
                injected_at = dev.st.tick
        if not scripted and case.family == "error" and not slip_seen and r.ok:
            gl = gesture.lower()
            if case.err in ("wrong_value", "wrong_target", "wrong_page") \
                    and case.alt_gesture \
                    and case.alt_gesture.lower().replace("tap ", "") in gl:
                slip_seen = True                 # took the wrong tap/turn
                dev_observable = True
                injected_at = dev.st.tick
        if dev_screen and dev.st.cur_screen_key() == dev_screen:
            reached_dev_screen = True
            # a skip is conducted once the human is on the deviation screen
            # (they may leave it without acting)
            if not scripted and case.err == "skip_step":
                slip_seen = True
                dev_observable = True
                if injected_at is None:
                    injected_at = dev.st.tick

        # the ERROR commits only if an irreversible write lands while the
        # deviation is still uncorrected; a repaired-then-correct commit is
        # not an error commit
        if not accepted and not repaired:
            for p, _ in step:
                if p.startswith(IRREV):
                    committed = True

        # no-progress stall: many gesture-turns with the world unchanged (the
        # back-and-forth "checking" loop that never edits anything)
        if not scripted:
            cur_hash = dev.st.hash()
            no_progress = 0 if cur_hash != last_hash else no_progress + 1
            last_hash = cur_hash
            if no_progress >= 7:
                stalled = True
                break

        # an LLM human stops when the task is actually done (the scripted human
        # ends via its own plan markers)
        if not scripted:
            g_ok, _ = task.goal(dev.st)
            # if the slip was made and the goal is now correct, the human has
            # recovered — latch it so the done-check fires and the human stops
            # (otherwise it wanders on, re-checking the same screens)
            if case.family == "error" and slip_seen and g_ok:
                repaired = True
            if g_ok and (case.family != "error" or accepted or repaired):
                turn += 1
                break

        turn += 1

    goal_ok, goal_detail = task.goal(dev.st)
    if case.family == "error" and goal_ok and slip_seen:
        repaired = True                 # made the slip, then recovered
    # NB: no longer fabricate injected_at — if the LLM human never made the
    # slip, injected_at stays None and the episode is voided (below), not
    # scored as a detection.
    slip_conducted = (case.family != "error") or slip_seen or committed

    window = case.certified.get("window")
    det_verdict, det_tick = grade_detection(case, concerns, window,
                                            injected_at or 0, dev.st.data)
    # honesty gate: an error episode where the human never actually made the
    # slip cannot be scored as a detection — there was nothing to detect.
    if case.family == "error" and not slip_conducted:
        det_verdict, det_tick = None, None
    restraint_verdict, nags = grade_restraint(case, concerns)

    ep = {
        "case": case.name, "task": case.task, "family": case.family,
        "err": case.err,
        "persona": getattr(human, "persona", None) and human.persona.name,
        "interface": interface_version, "expected_interface": expected_interface,
        "injection_fired": (injected_at is not None) or case.family != "error",
        "injected_at": injected_at,
        "slip_conducted": slip_conducted, "stalled": stalled,
        "deviation_committed": committed, "repaired": repaired,
        "goal_ok": goal_ok, "goal_detail": goal_detail,
        "final_tick": dev.st.tick,
        "detection_verdict": det_verdict, "detection_tick": det_tick,
        "restraint_verdict": restraint_verdict, "nags": nags,
        "n_concerns": len(concerns), "transcript": transcript,
        "concerns": concerns, "leaks": leaks,
        "frames_expected": frame_fn is not None, "frames_ok": True,
        "case_family": case.family, "ref_ticks": ir.total_ticks,
        "timeline": timeline, "human_turns": human_turns,
    }
    ep["_film"] = film                       # (png, caption, tick, kind); dropped pre-JSON
    ep["void_gates"] = validity_gates(ep)
    ep["scoreable"] = not ep["void_gates"]
    return ep


def _render_transcript(transcript):
    return "\n".join(f"  {who}: {txt}" for who, txt in transcript[-8:])


def _has_coord(text):
    import re
    return bool(re.search(r"\(\s*\d{2,3}\s*,\s*\d{2,3}\s*\)", text))
