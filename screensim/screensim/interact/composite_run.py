"""Run a COMPOSITE (multi-error) episode with a scripted human under test by an
LLM assistant, and grade it with the CookSim-aligned metric set.

Soft-hard takeover, per beat and sequential:
  * a reversible slip LANDS when its step fires (visible on screen); the beat
    stays open until a correct detection is accepted (then repaired) or the
    episode ends.
  * a navigation slip leaves the human LOST — it holds there; a correct+accepted
    detection splices the recovery (press back + right turn) and the human
    continues, otherwise it gives up and the episode ends (that error stands).
  * before the one irreversible commit the human HOLDS a budget of polls; on
    maximum hold the commit fires with whatever errors are still open.

Grading (per the metrics page): per error — detected (a correct flag while the
error is observable and before it commits), prevented (detected + accepted +
repaired before commit); episode Recall R, Precision P, F1, Detection-timing DT,
Conversation-economy CC, in-time Success S.
"""
from __future__ import annotations

import copy
import re

from ..core import pacing
from ..core.engine import Device
from ..tasks import resolve_dynamic, by_id
from . import interface, readout
from .composite import combined_stream, _combined_plan
from .errors import dry_replay

POLL = 2
IRREV = ("bank.history", "bank.balance", "shop.order", "mail.sent")


def _words(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


_STOP = {"tap", "the", "and", "for", "with", "off", "on", "instead", "of", "to",
         "a", "an", "your", "my", "is", "it", "by", "so", "left", "unset"}


def _beat_terms(beat):
    """Words a CORRECT flag for this beat would name — the wrong control it hit,
    the correct control it should have, and the setting leaf."""
    terms = _words(beat.alt_gesture) | _words(beat.correct_gesture)
    for p in beat.paths:
        terms |= _words(p.split(".")[-1])
    return {t for t in terms if len(t) > 2 and t not in _STOP}


def _compact(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def flag_matches_beat(text, beat):
    """Does the assistant's flag name this error? Word-overlap with the beat's
    term set, plus two robustness rules the word split misses:
      * a compacted label match — "Wi-Fi" is tokenised as {wi, fi} but a
        person says "wifi"; compare hyphen/space-free forms as substrings;
      * a navigation error is correctly detected by telling the person to go
        BACK — that IS the fix — so "back" counts while that nav beat is open."""
    terms = _beat_terms(beat)
    w = _words(text)
    if terms and (terms & w):
        return True
    cf = _compact(text)
    for lab in (beat.alt_gesture, beat.correct_gesture):
        c = _compact(lab.replace("tap ", ""))
        if len(c) >= 4 and c in cf:
            return True
    if beat.mode == "nav" and "back" in w:
        return True
    return False


class CompositeOracle:
    """Ceiling: every poll, names the deviation of an open (observable,
    unrepaired) beat — proves the takeover releases and grading credits."""
    def turn(self, open_beats=(), **_):
        for b in open_beats:
            ctrl = (b.correct_gesture or b.alt_gesture).replace("tap ", "")
            return {"text": f"Wait — that's not right, you need {ctrl}.",
                    "concern": f"{b.alt_gesture} {b.correct_gesture} "
                               f"{' '.join(p.split('.')[-1] for p in b.paths)}"}
        return {"text": "", "concern": ""}


class SilentComposite:
    def turn(self, **_):
        return {"text": "", "concern": ""}


def run_composite(composite, ir, persona, assistant, *, frame_fn=None,
                  interface_version="sv1", k_frames=2, film_cap=None,
                  max_turns=90, contestant="model"):
    pacing.apply_conversation_pace()
    try:
        return _run(composite, ir, persona, assistant, frame_fn,
                    interface_version, k_frames, film_cap, max_turns, contestant)
    finally:
        pacing.reset()


def _run(composite, ir, persona, assistant, frame_fn, iface_v, k_frames,
         film_cap, max_turns, contestant="model"):
    iface = interface(iface_v)
    task = by_id(composite.task)
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    ref_rec = dry_replay(task, ir.flat())
    ref_ticks = ref_rec["end_tick"]
    ref_final = ref_rec["dev"].st.data

    stream = combined_stream(ir, composite.beats)
    beats = list(composite.beats)
    st = {id(b): {"beat": b, "injected_at": None, "observable": False,
                  "detected_tick": None, "accepted": False, "repaired": False,
                  "committed": False, "gave_up": False, "pushback": 0,
                  "screen": None, "post": []}
          for b in beats}

    frames, transcript, concerns, film = [], [], [], []
    n_polls = 0
    n_talk = 0
    false_flags = 0
    seen_fp = set()                 # deduped false-alarm term-sets (count once)
    all_beat_terms = set()
    for b in beats:
        all_beat_terms |= _beat_terms(b)
    last_line = ""

    timeline = []
    hist = []                       # every executed gesture: (g, scr_before, scr_after, changed)

    kb_hist = []                    # (gesture, keyboard_before, keyboard_after)

    def _kb():
        return bool(getattr(dev.st, "keyboard", False))

    def _exec(g):
        before_scr = dev.st.cur_screen_key()
        kb_before = _kb()
        pre_d = copy.deepcopy(dev.st.data)
        r = dev.act(resolve_dynamic(dev, g))
        hist.append((g, before_scr, dev.st.cur_screen_key(), dev.st.data != pre_d))
        kb_hist.append((g, kb_before, _kb()))
        return r

    def _route_to(target, slip_idx, step):
        """Bring the person back to `target` (the screen a slip happened on)
        from wherever they are now: find the latest screen before the slip that
        equals the current one and replay the screen-changing taps that led from
        it to the slip screen; if the current screen never occurred, press back
        until one that did appears. Scroll state persists per screen, so swipes
        are not replayed."""
        for _ in range(8):
            cur = dev.st.cur_screen_key()
            if cur == target:
                return True
            j = None
            for k in range(slip_idx - 1, -1, -1):
                if hist[k][2] == cur:
                    j = k
                    break
            if j is None and slip_idx > 0 and hist[0][1] == cur:
                j = -1
            if j is not None:
                for k in range(j + 1, slip_idx):
                    g, b4, aft, changed = hist[k]
                    if not changed and b4 != aft:
                        step(g)
                if dev.st.cur_screen_key() == target:
                    return True
                # replaying did not land there — fall through to backing out
            step("press back")
        return dev.st.cur_screen_key() == target

    def say(who, text):
        nonlocal last_line
        transcript.append((who, text))
        timeline.append({"t": dev.st.tick, "kind": "say", "who": who, "text": text})
        last_line = f"{who}: {text}"

    def film_frame(kind):
        if film_cap is not None:
            try:
                film.append((film_cap(dev), last_line, dev.st.tick, kind))
            except Exception:
                pass

    def open_beats():
        return [s["beat"] for s in st.values()
                if s["observable"] and not s["repaired"] and not s["accepted"]
                and not s.get("locked")]

    def poll():
        nonlocal n_polls, n_talk, false_flags
        n_polls += 1
        if frame_fn is not None and iface["frames"]:
            frames.append(frame_fn(dev))
            frames[:] = frames[-k_frames:]
        rel = readout.relational(dev) if iface["relational"] else ""
        memory = concerns[-1][1] if concerns else ""
        try:
            reply = assistant.turn(frames=list(frames), relational_text=rel,
                                   memory=memory, open_beats=tuple(open_beats()),
                                   deviation_observable=bool(open_beats()))
        except Exception:
            reply = {"text": "", "concern": ""}
        text = (reply.get("text") or "").strip()
        flag = (reply.get("concern") or reply.get("flag") or text or "").strip()
        if text:
            n_talk += 1
            say("assistant", text); film_frame("say")
            concerns.append((dev.st.tick, flag or text))
        # match the flag to an open beat (first correct detection wins per beat)
        matched = False
        for b in open_beats():
            s = st[id(b)]
            if flag_matches_beat(flag, b):
                matched = True
                if s["detected_tick"] is None and not s["committed"]:
                    s["detected_tick"] = dev.st.tick
                # persona acceptance: push back `rebuttals` times, then accept
                if s["pushback"] < persona.rebuttals:
                    s["pushback"] += 1
                    say("human", "you sure? I think it's fine.")
                else:
                    s["accepted"] = True
                    say("human", "okay — let me fix that.")
                    def _rep(g):
                        r = _exec(g)
                        timeline.append({"t": dev.st.tick, "kind": "gesture",
                                         "who": "human", "text": g, "slip": False,
                                         "repair": True, "ok": bool(r.ok),
                                         "beat": beats.index(b)})
                        film_frame("gesture")
                        return r.ok
                    # a late acceptance: the person has moved on since the
                    # slip — go back to the screen it happened on, fix it there,
                    # then redo the navigation-only taps taken since (the
                    # state-changing ones already happened and must not repeat)
                    kb_was_open = _kb()
                    if kb_was_open and (b.repair or [""])[0] == "press back":
                        _rep("press back")     # close the keyboard first, so the
                                               # repair's own "press back" navigates
                    if (b.mode != "nav" and s["screen"]
                            and dev.st.cur_screen_key() != s["screen"]):
                        if _route_to(s["screen"], s["slip_hist_idx"], _rep):
                            for g in (b.repair or []):
                                _rep(g)
                            # redo the screen-changing taps taken since the
                            # slip so the plan can continue where it was
                            # (every screen-changing tap is redone; a typed value
                            # or toggle is redone only if a redone tap invalidated
                            # it — e.g. re-requesting the code invalidates the
                            # code that was typed; toggles already set stay set)
                            redone = set()
                            for g, chg, scr_change, kb0, kb1 in s["post"]:
                                parents = {p.rsplit(".", 1)[0] for p in chg}
                                if scr_change:
                                    _rep(g); redone |= parents
                                elif kb0 != kb1:
                                    # a focus / dismiss: redo only when it
                                    # would have the same effect now (a
                                    # "press back" that closed the keyboard
                                    # must not navigate away instead)
                                    if _kb() == kb0:
                                        _rep(g)
                                elif parents & redone:
                                    _rep(g)
                    else:
                        for g in (b.repair or []):
                            _rep(g)
                    if kb_was_open and not _kb():
                        # put the keyboard back the way the person had it, so
                        # their next typing / dismiss step behaves as planned
                        opener = next((g for g, kb0, kb1 in reversed(kb_hist)
                                       if not kb0 and kb1 and g != "press back"), None)
                        if opener:
                            _rep(opener)
                    s["repaired"] = True
                break
        if flag and text and not matched and _is_alarm(text):
            # a spoken ALARM ("wait / that's wrong / not right ...") that names
            # no injected error and doesn't just restate a beat we already
            # matched — a false alarm. Repeats of the same alarm count once
            # (CookSim: repeated flags for one error count once).
            key = frozenset(_words(flag) - _STOP - all_beat_terms)
            if key and key not in seen_fp:
                seen_fp.add(key)
                false_flags += 1

    turn = 0
    i = 0
    while i < len(stream) and turn < max_turns and dev.st.outcome is None:
        step = stream[i]
        if turn % POLL == 0:
            poll()
        role = step["role"]
        if role == "HOLD":
            # irreversible commit imminent — give a last window, then release
            budget = 1 + persona.rebuttals + (1 if persona.confirm_irreversible else 0)
            held = 0
            while held < budget and any(not st[id(b)]["accepted"]
                                        and st[id(b)]["observable"]
                                        for b in beats):
                poll(); held += 1; turn += 1
            i += 1
            continue
        g = step["gesture"]
        scr_before = dev.st.cur_screen_key()
        kb_before_g = _kb()
        pre = copy.deepcopy(dev.st.data)
        r = _exec(g)
        if r.ok and any(p.startswith(pref) for pref in IRREV
                        for p in _diff_paths(pre, dev.st.data)):
            # an irreversible commit just fired: every error still open has
            # committed with it and can no longer be repaired — close them, so
            # a late flag is not "accepted" into a flow that no longer exists
            for sb in st.values():
                if sb["observable"] and not sb["accepted"]:
                    sb["locked"] = True
        if step["beat"] is None and r.ok:
            changed_paths = _diff_paths(pre, dev.st.data)
            # a tap that changes the screen OR focuses a field (opens the
            # keyboard) is navigation the person must redo after a detour
            scr_change = scr_before != dev.st.cur_screen_key()
            kb_after_g = _kb()
            for bb in beats:
                sb = st[id(bb)]
                if sb["observable"] and not sb["accepted"]:
                    sb["post"].append((g, changed_paths, scr_change, kb_before_g, kb_after_g))
        timeline.append({"t": dev.st.tick, "kind": "gesture", "who": "human",
                         "text": g, "slip": step["beat"] is not None,
                         "beat": (beats.index(step["beat"])
                                  if step["beat"] is not None else None)})
        film_frame("gesture")
        if step["beat"] is not None and r.ok:
            s = st[id(step["beat"])]
            s["observable"] = True
            s["screen"] = dev.st.cur_screen_key()
            s["slip_hist_idx"] = len(hist) - 1
            if s["injected_at"] is None:
                s["injected_at"] = dev.st.tick
            if step["beat"].mode == "nav":
                # LOST: hold for the assistant; recover on accept, else give up
                budget = 1 + persona.rebuttals
                held = 0
                while held < budget and not s["accepted"]:
                    poll(); held += 1; turn += 1
                if not s["accepted"]:
                    s["gave_up"] = True            # never recovered → error stands
                    break
            else:
                # a reversible slip is now visible — give the assistant an
                # immediate look (so even the LAST beat gets a detection chance)
                poll()
        i += 1
        turn += 1

    # final listening window: the person pauses before putting the phone down,
    # with the same budget as before an irreversible commit — so an error that
    # fires on the LAST step gets the same chance as any other (without this a
    # one-rebuttal persona made last-step errors unpreventable even for the
    # oracle). Not after a give-up: that episode has already ended.
    if not any(st[id(b)]["gave_up"] for b in beats):
        budget = 1 + persona.rebuttals
        held = 0
        while (held < budget and open_beats() and dev.st.outcome is None
               and turn < max_turns):
            poll(); held += 1; turn += 1

    goal_ok, _ = task.goal(dev.st)
    final = dev.st.data
    # grade per beat by COUNTERFACTUAL: replay this run's gestures WITHOUT the
    # beat's slip (and its repair); the beat COMMITTED iff the real final state
    # differs from that counterfactual — i.e. the slip left a trace in the
    # world. (Comparing the beat's own paths at the end is not enough: a form
    # field such as the payment source is reset by the commit, so final and
    # reference agree on it even when the money went out from the wrong source.)
    # PREVENTED = detected and not committed. A nav beat is prevented iff the
    # human recovered (accepted), committed iff it gave up.
    def _counterfactual(bi):
        """This run with beat `bi` never having happened: its slip is replaced
        by the correct gesture it displaced (replace beats) or dropped
        (additive beats), and its repair taps are dropped."""
        b = beats[bi]
        cf = Device(task.seed)
        if task.start_app != "home":
            cf.open_app(task.start_app)
        for e in timeline:
            if e["kind"] != "gesture":
                continue
            if e.get("beat") == bi:
                if e.get("repair") or b.mode != "replace":
                    continue
                g = b.correct_gesture
            else:
                g = e["text"]
            cf.act(resolve_dynamic(cf, g))
        return cf.st.data
    recs = []
    for bi, b in enumerate(beats):
        s = st[id(b)]
        detected = s["detected_tick"] is not None
        if b.mode == "nav":
            prevented = s["accepted"]
            committed = s["gave_up"]
        else:
            # only GOAL-RELEVANT differences count: plan-touched paths, the
            # beat's own paths, and irreversible records — never side channels
            # such as the one-time code or the text thread that carried it
            # (a repaired error legitimately re-requests a code)
            relevant = set(ir.touched_paths) | set(b.paths)
            diff = [p for p in _diff_paths(final, _counterfactual(bi))
                    if (p in relevant or any(p.startswith(pref) for pref in IRREV))
                    and not p.startswith(("bank.otp", "messages."))]
            committed = bool(diff)
            prevented = detected and not committed
        recs.append({"kind": b.kind, "detected": detected, "prevented": prevented,
                     "committed": bool(committed),
                     "injected_at": s["injected_at"],
                     "detected_tick": s["detected_tick"]})
    E = len(beats)
    D = sum(1 for r in recs if r["detected"])
    prevented = sum(1 for r in recs if r["prevented"])
    R = D / E if E else None
    TP = D
    P = TP / (TP + false_flags) if (TP + false_flags) else (1.0 if TP else None)
    F1 = (2 * P * R / (P + R)) if (P and R and (P + R)) else 0.0
    DT = sum(1.0 if r["prevented"] else 0.5 if r["detected"] else 0.0
             for r in recs) / E if E else None
    cap = 2 * ref_ticks + 8
    S = 1 if (goal_ok and dev.st.tick <= cap) else 0
    CC = 1 - min(1.0, n_talk / max(1, n_polls))
    film_frame("end")
    return {
        "episode": composite.id, "task": composite.task,
        "persona": persona.name, "n_errors": E,
        "beats": recs, "recall": R, "precision": P, "f1": round(F1, 3),
        "dt": DT, "success": S, "cc": round(CC, 3),
        "false_flags": false_flags, "committed_any": any(r["committed"] for r in recs),
        "prevented": prevented, "goal_ok": goal_ok, "final_tick": dev.st.tick,
        "n_polls": n_polls, "n_talk": n_talk, "transcript": transcript,
        "timeline": timeline, "contestant": contestant, "family": "error",
        "instruction": task.instruction, "_film": film,
    }


def _get(data, path):
    cur = data
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def _diff_paths(a, b, prefix=""):
    out = []
    for k in set(a) | set(b):
        pa = f"{prefix}.{k}" if prefix else str(k)
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _diff_paths(va, vb, pa)
        elif va != vb:
            out.append(pa)
    return out


def _is_ack(text):
    t = text.lower()
    return any(w in t for w in ("looks good", "all set", "that's right",
                                "perfect", "correct", "well done", "nicely"))


_ALARM = ("wait", "no,", "not right", "that's wrong", "isn't right", "shouldn't",
          "mistake", "wrong", "instead", "don't", "careful", "hold on",
          "actually that", "that's not")


def _is_alarm(text):
    t = " " + text.lower()
    return any(w in t for w in _ALARM)
