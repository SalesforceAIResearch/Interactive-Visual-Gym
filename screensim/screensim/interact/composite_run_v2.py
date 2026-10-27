"""Multi-error episode with the v2 (CookSim-interface) agents: an LLM human
whose hands are the engine's queue and whose voice carries moves,
close_conversation and — once released — plan_update; an assistant with a
TYPED flag, task_progress and goal_change.

Soft-hard takeover, per beat:
  * the engine fires each designed slip from the queue (hard); the human is
    told, by a system task, that its body is committed and its job is its voice;
  * a flag that names the error ARMS the beat: exact type match on the open
    beat's kind (primary), or a text match on its controls (secondary credit,
    type_wrong recorded);
  * the beat is RELEASED only when the human's move is "accept" while a beat
    is armed — the human's own judgment under its persona; its hands are then
    freed (the live schema) and its plan_update actions are executed as the
    repair;
  * a navigation slip holds for a bounded number of polls; an irreversible
    commit holds for the same budget; the final listening window is the same.

Grading is unchanged (counterfactual commit; R/P/F1/DT/S/CC), with false
alarms = flags that credit no open beat, and type accuracy recorded.
"""
from __future__ import annotations

import re

import copy

from ..core import pacing
from ..core.engine import Device
from ..tasks import resolve_dynamic, by_id
from . import interface, readout
from .composite import combined_stream
from .composite_run import (_beat_terms, _diff_paths, _get, _is_alarm, flag_matches_beat,
                            IRREV, POLL)
from .errors import dry_replay
from ..engine_rev import engine_rev as _engine_rev
_engine_rev()                      # fix the stamp to the code loaded now (see engine_rev.py)


_GENERIC = {"switch", "toggle", "button", "row", "option", "setting", "settings", "turn", "the",
            "and", "for", "you", "your", "app", "apps", "all", "tap", "off", "that", "this", "with"}
_ALARM_V2 = ("wait", "stop", "no,", "not right", "wrong", "mistake", "instead", "don't", "do not",
             "careful", "hold on", "oops", "by accident", "accidentally", "you selected", "you tapped",
             "you turned", "you picked", "you chose", "you opened", "you switched", "you just", "you've ended up",
             "wandered into", "still set", "is still",
             "switch back", "switch it back", "turn it back", "turn that back", "put it back", "set it back",
             "change it back", "go back", "back on", "back off", "should be", "needs to be", "isn't",
             "not the", "not what", "which adds", "fee")


_TOGGLE_BACK = re.compile(r"\b(turn|switch|flip|toggle|put|set|change)\w*\s+(\S+\s+){0,5}?back\s+(on|off)\b")


def _alarm_v2(text, beat=None):
    """Is the sentence a warning? "go back" warns only about a wrong page (elsewhere it is
    ordinary navigation: "next, go back to Privacy & Security"), and "back on / back off"
    only as undoing a switch ("turn it back on"), not "you're back on Tracking"."""
    t = " " + (text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
    for w in _ALARM_V2:
        if w not in t:
            continue
        if w == "go back" and beat is not None and beat.mode != "nav":
            continue
        if w in ("back on", "back off") and not _TOGGLE_BACK.search(t):
            continue
        return True
    return False


def _names_slip(text, beat):
    """Does the sentence point at THIS mistake? It names the slipped control (any distinctive
    word of the wrong row that the intended row does not share, or the compact label), or —
    for an extra write — the control that undoes it ("switch back to QuickPay balance"), or —
    for a wrong page — tells the person to go back. Merely naming the intended control is not
    enough (that is ordinary next-step advice)."""
    from .composite_run import _words, _compact, _STOP
    w = _words(text)
    if beat.mode == "nav" and "back" in w:
        return True
    alt = beat.alt_gesture.replace("tap ", "")
    right = set(_words((beat.correct_gesture or "").replace("tap ", ""))) if beat.mode in ("replace", "nav") else set()
    alt_terms = {t for t in _words(alt) if len(t) >= 3 and t not in _STOP and t not in _GENERIC} - right
    if alt_terms & w:
        return True
    c = _compact(alt)
    if len(c) >= 4 and c in _compact(text):
        return True
    if beat.mode == "add":
        for g in (beat.repair or []):
            lab = g.replace("tap ", "")
            rt = {t for t in _words(lab) if len(t) >= 3 and t not in _STOP and t not in _GENERIC}
            if rt and (rt & w):
                return True
    return False


def _credit_typed(text, etype, open_beats):
    """Which held error does a TYPED flag credit? -> (beat or None, credited_by_type).
    The type picks among held errors of that kind (the one its words name, else the first).
    Exception: when the words name none of those but do name a different held error, the
    type alone does not release an error the assistant never looked at; the flag is then
    judged like a wrongly typed one (credited only if it names that other error's slip)."""
    from .composite_run import flag_matches_beat
    same = [b for b in open_beats if b.kind == etype]
    others = [b for b in open_beats if b.kind != etype]
    if same:
        hit = next((b for b in same if flag_matches_beat(text, b)), None)
        if hit is not None or not any(flag_matches_beat(text, b) for b in others):
            return (hit or same[0]), True
    return next((b for b in others if _names_slip(text, b)), None), False


_HEALTH = ("calls", "empty", "failures", "retries", "rate_limited")


def _call_counts(assistant, human):
    return {who: {k: getattr(getattr(a, "client", None), k, 0) for k in _HEALTH}
            for who, a in (("assistant", assistant), ("human", human))}


def _call_health(assistant, human, before):
    """Model calls made during this episode, blank replies, and calls that failed after all
    retries (the runner reads both as silence). Clients are cached per thread, so this is the
    change over the episode; when both sides share one client the counts are joint."""
    now = _call_counts(assistant, human)
    out = {who: {k: now[who][k] - before[who][k] for k in _HEALTH} for who in now}
    if getattr(assistant, "client", None) is not None and getattr(assistant, "client", None) is getattr(human, "client", None):
        out["shared_client"] = True
    return out


def _set_path(data, path, value):
    cur = data
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            return
        cur = cur.setdefault(part, {})
    if isinstance(cur, dict):
        cur[parts[-1]] = value
from .prompts import TAXONOMY_DEFS

HOLD_POLLS = 2      # polls the person waits on a wrong page before giving up
HOLD_MAX = 4        # ... extended while they are still asking / conceding rather than refusing
NAV_PATIENCE = 6    # free mode: blocked attempts to leave a held wrong page before the person gives up
ENGAGED = {"ask_why", "question_correctness", "ask_for_more_guidance", "answer", "accept"}          # polls before an irreversible commit / while lost / at the end


class OracleV2:
    """Ceiling: names the open beat with the exact type, the on-screen evidence
    and the fix — once in full, then a short reminder every other poll (a real
    helper does not repeat one sentence verbatim every second)."""
    def __init__(self):
        self.said = {}
        self.nudged = {}

    _PATHS = None

    @classmethod
    def _page_paths(cls):
        """Settings page key -> the labels tapped from the Settings root to reach it (shortest route,
        screensim/apps/inventory.py's walk), and label-path -> key."""
        if cls._PATHS is None:
            import os as _os, sys as _sys
            import screensim.apps.inventory as inv
            inv.walk()
            by_key = {k: v.split(" > ")[1:] for k, v in inv.PAGE_PATH.items()}
            cls._PATHS = (by_key, {tuple(v): k for k, v in by_key.items()})
        return cls._PATHS

    @staticmethod
    def _seek(d, g, out):
        """Do gesture g on device d, scrolling first if the control is off screen; the scrolls used and
        g itself are appended to out. False if it cannot be done."""
        if d.act(resolve_dynamic(d, g)).ok:
            out.append(g)
            return True
        for direction, n in (("swipe up", 8), ("swipe down", 16)):
            for _ in range(n):
                if not d.act(resolve_dynamic(d, direction)).ok:
                    break
                out.append(direction)
                if d.act(resolve_dynamic(d, g)).ok:
                    out.append(g)
                    return True
        return False

    @classmethod
    def _route(cls, dev, node):
        """The gestures from the phone's CURRENT state that bring the person to `node` and set its
        commit, on a copy of the device: back until on a page along the target page's route from the
        Settings root, then down that route, scrolling to each control. None if it does not reach the
        commit (the caller then falls back to the plan's own words)."""
        import copy as _copy
        d = _copy.deepcopy(dev)
        out = []
        if d.st.app == "settings":
            by_key, _ = cls._page_paths()
            labels = by_key.get(node.screen)
            if labels is None:
                return None
            chain = ["settings.root"]
            for i in range(1, len(labels) + 1):
                chain.append(cls._page_paths()[1].get(tuple(labels[:i]), ""))
            for _ in range(12):
                if d.st.cur_screen_key() in chain:
                    break
                d.act(resolve_dynamic(d, "press back"))
                out.append("press back")
            cur = d.st.cur_screen_key()
            if cur not in chain:
                return None
            for lab in labels[chain.index(cur):]:
                if not cls._seek(d, f"tap {lab}", out):
                    return None
            k = len(out)
        else:
            # a small app (QuickPay): breadth-first over back and its labelled buttons/rows, on copies,
            # never through a step that changes the app's data, until the node's gestures set its commit
            from collections import deque

            def reaches(dd):
                d3 = _copy.deepcopy(dd)
                o3 = []
                return all(cls._seek(d3, g, o3) for g in (node.gestures or [])) and \
                    all(_get(d3.st.data, p) == v for p, v in (node.commit or []))

            if not reaches(d):
                seen, q, found = {(tuple(d.st.stack), d.st.app, repr(d.st.scroll), repr(d.st.keyboard))}, deque([(d, [])]), None
                while q and found is None and len(seen) < 400:
                    dd, path = q.popleft()
                    if len(path) >= 5:
                        continue
                    labels = [w.label.strip() for w in dd.laid_out_screen().widgets
                              if w.label and w.on_tap and w.kind in ("row", "account", "button", "nav")]
                    for g in ["press back", "swipe up", "swipe down"] + [f"tap {l}" for l in labels]:
                        d2 = _copy.deepcopy(dd)
                        if not d2.act(resolve_dynamic(d2, g)).ok or d2.st.data != dd.st.data:
                            continue
                        kk = (tuple(d2.st.stack), d2.st.app, repr(d2.st.scroll), repr(d2.st.keyboard))
                        if kk in seen:
                            continue
                        seen.add(kk)
                        if reaches(d2):
                            found = (d2, path + [g])
                            break
                        q.append((d2, path + [g]))
                if found is None:
                    return None
                d, out = found[0], list(found[1])
            k = len(out)
        acts = list(node.gestures or [])
        if d.st.app == "settings":
            while acts and acts[0] == "press back":      # the plan's way back from where IT was; we are already here
                acts.pop(0)
        for g in acts:
            if not cls._seek(d, g, out):
                return None
        if not all(_get(d.st.data, p) == v for p, v in (node.commit or [])):
            return None
        return out[:k], out[k:]

    def guidance(self, ir, dev):
        """Where the task continues: the first commit the world still lacks, reached from the page the
        person is ON (_route: back to the target's route, down it, scrolling to each control), every
        gesture spoken. Round 11: the old route started at the top of the plan whatever the page, and a
        novice who does exactly what it is told could not follow it."""
        if ir is None or dev is None:
            return ""
        commits = [n for n in ir.nodes if n.kind == "commit"]
        met = [all(_get(dev.st.data, p) == v for p, v in (n.commit or [])) for n in commits]
        if dev.st.app == "settings":
            todo = [n for n, m in zip(commits, met) if not m]          # independent switches: any order
        else:
            # a staged flow (QuickPay): a step's snapshot includes form fields later steps fill in, so
            # continue after the LAST step that is fully in place
            last = max((i for i, m in enumerate(met) if m), default=-1)
            todo = commits[last + 1:]
        for n in todo[:1]:
            r = self._route(dev, n)
            if r is None:                            # not a Settings page it can route to: the plan's words
                return f"Next: {', then '.join(n.gestures or [n.label])}."
            reach, act = r
            return ("From here: " if reach else "On this page, ") + ", then ".join(reach + act) + "."
        return "Everything the task asked for is done as far as I can see."

    def turn(self, open_beats=(), screen_title="", conversation="", ir=None, dev=None, **_):
        last = conversation.strip().split("\n")[-1].lower() if conversation.strip() else ""
        they_spoke_last = last.split(":", 1)[0].strip().endswith(("the person", "human", "user"))
        if not open_beats and they_spoke_last and "?" in last:
            g = self.guidance(ir, dev)
            if g:
                return {"text": g, "flag": None, "task_progress": None, "goal_change": None}
        for b in reversed(list(open_beats)):        # the newest slip first — a wrong page is urgent
            wrong = b.alt_gesture.replace("tap ", "")
            if b.kind == "extra_write":
                ctrl = (b.repair or [b.alt_gesture])[0].replace("tap ", "")
                if ctrl == wrong:
                    ctrl = "how it was"
            else:
                ctrl = (b.correct_gesture or b.alt_gesture).replace("tap ", "")
            n = self.said.get(id(b), 0)
            self.said[id(b)] = n + 1
            if n == 0:
                if b.mode == "nav":
                    text = (f"Wait — that tap on {wrong} took you to the wrong page"
                            f"{' (the title says ' + screen_title + ')' if screen_title else ''}; "
                            f"this isn't where the task continues. Go back and tap {ctrl} instead.")
                elif b.kind == "extra_write":
                    text = (f"Hold on — you just switched {wrong}; that isn't part of what you set out to do, "
                            f"and it changes a setting you didn't mean to touch. Tap {wrong} again to put it back.")
                else:
                    text = (f"Hold on — you tapped {wrong}, but the task needs {ctrl}. "
                            f"You can see {wrong} is the one that changed; undo it and tap {ctrl}.")
                return {"text": text, "flag": {"error": b.kind}, "task_progress": None, "goal_change": None}
            if n % 2 == 1 and not they_spoke_last:
                return {"text": "", "flag": None, "task_progress": None, "goal_change": None}
            if they_spoke_last and "?" in last:
                if b.mode == "nav":
                    ans = (f"Yes, I'm sure — the title at the top reads {screen_title or 'the wrong page'}, "
                           f"not where the task goes next. Back out and tap {ctrl}.")
                elif b.kind == "extra_write":
                    ans = (f"Yes — {wrong} flipped when you tapped there; it was fine as it was and nothing you're "
                           f"doing needs it changed. Set it back to {ctrl} before going on.")
                else:
                    ans = f"Yes — the screen shows {wrong} selected, and the task needs {ctrl}. Undo it and pick {ctrl}."
                return {"text": ans, "flag": {"error": b.kind}, "task_progress": None, "goal_change": None}
            if b.mode == "nav":
                nudges = [f"You're still on {wrong} — the task continues under {ctrl}. Back out and tap {ctrl}.",
                          f"Quick check: this page ({screen_title or wrong}) isn't on the way; {ctrl} is.",
                          f"I'd back out of {wrong} before going on — {ctrl} is the row you want."]
            elif b.kind == "extra_write":
                nudges = [f"{wrong} is still switched — nothing you're doing needs that; set it back to {ctrl}.",
                          f"Quick check: {wrong} got flipped by accident and should go back to {ctrl}.",
                          f"I'd put {wrong} back before going on — {ctrl}."]
            else:
                nudges = [f"{wrong} is still set instead of {ctrl} — worth going back to fix it.",
                          f"Quick check: {wrong} is what got changed; the task wants {ctrl}.",
                          f"I'd fix {wrong} before going on — {ctrl} is the right one."]
            m = self.nudged.get(id(b), 0)
            self.nudged[id(b)] = m + 1
            return {"text": nudges[m % len(nudges)], "flag": {"error": b.kind},
                    "task_progress": None, "goal_change": None}
        return {"text": "", "flag": None, "task_progress": None, "goal_change": None}


class SilentV2:
    def turn(self, **_):
        return {"text": "", "flag": None, "task_progress": None, "goal_change": None}


def run_composite_v2(composite, ir, persona, assistant, human, *, frame_fn=None,
                     interface_version="sv1", k_frames=2, film_cap=None,
                     max_turns=160, contestant="model", human_mode="free", grader="person"):
    """human_mode: "free" (default) — the person plans its own taps turn by turn and
    the engine takes over only at an error's anchor; "scripted" — the body follows
    the reference plan with the errors spliced in (deterministic validation)."""
    pacing.apply_conversation_pace()
    try:
        return _run(composite, ir, persona, assistant, human, frame_fn,
                    interface_version, k_frames, film_cap, max_turns, contestant, human_mode, grader)
    finally:
        pacing.reset()


def _run(composite, ir, persona, assistant, human, frame_fn, iface_v, k_frames,
         film_cap, max_turns, contestant, human_mode, grader="person"):
    iface = interface(iface_v)
    health0 = _call_counts(assistant, human)
    task = by_id(composite.task)
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    init_data = copy.deepcopy(dev.st.data)
    ref_rec = dry_replay(task, ir.flat())
    ref_dev = Device(task.seed)
    if task.start_app != "home":
        ref_dev.open_app(task.start_app)
    for _g in ir.flat():
        ref_dev.act(resolve_dynamic(ref_dev, _g))

    def _goal_paths(paths, slip_world):
        """G1 ruling (option B, 2026-09-11): only the slip paths that matter to the goal
        count. A path matters if the reference plan writes it, or if putting the slip's
        value onto it in the correctly finished world makes the goal fail. Side effects
        the goal does not care about are ignored."""
        touched = set(ir.touched_paths)
        out = []
        for p in paths:
            if p in touched or any(p.startswith(t + ".") or t.startswith(p + ".") for t in touched):
                out.append(p); continue
            probe = copy.deepcopy(ref_dev)
            _set_path(probe.st.data, p, copy.deepcopy(_get(slip_world, p)))
            try:
                ok = task.goal(probe.st)[0]
            except Exception:
                ok = False
            if not ok:
                out.append(p)
        return out or list(paths)            # a certified slip always breaks the goal; fall back if not
    ref_ticks = ref_rec["end_tick"]

    stream = combined_stream(ir, composite.beats)
    free = human_mode == "free"
    anchor_screen = {}
    if free:
        probe = Device(task.seed)
        if task.start_app != "home":
            probe.open_app(task.start_app)
        by_node = {b.node_idx: b for b in composite.beats}
        for n in ir.nodes:
            for g in (n.approach or []):
                probe.act(resolve_dynamic(probe, g))
            if n.idx in by_node:
                anchor_screen[id(by_node[n.idx])] = probe.st.cur_screen_key()
            if getattr(n, "gesture", None):
                probe.act(resolve_dynamic(probe, n.gesture))
    beats = list(composite.beats)
    st = {id(b): {"beat": b, "injected_at": None, "observable": False, "armed": False,
                  "engine_detected_tick": None, "person_detected_tick": None,
                  "detected_tick": None, "accepted": False, "repaired": False,
                  "gave_up": False, "locked": False, "type_ok": None,
                  "pre": None, "screen": None, "pre_screen": None, "slip_hist_idx": None,
                  "human_fix": False, "fallback": False, "post": []}
          for b in beats}

    frames, transcript, film, timeline = [], [], [], []
    conversation = []                # (tick, who, text) — the public record
    n_polls = n_talk = false_flags = 0
    typed_flags = typed_right = repeat_flags = 0
    false_flags_engine = 0                 # the engine arm's own false-alarm count, kept in parallel
    person_calls = []                      # every judgement the person made, with tick
    own_error_calls = denials = ambiguous_calls = repeat_calls = 0
    moves = []
    memory = ""
    progress_log = []
    last_line = ""
    pending = []                     # human-planned gestures (after release)
    last_credited = []               # beats the assistant's latest flag(s) named since the person last spoke
    queue = []                       # free mode: the person's OWN queue of taps
    forced = []                      # free mode: (gesture, beat idx) still to execute from a takeover
    own_hist = []                    # free mode: what the person believes it has done (slips included)
    last_g = [None]                  # free mode: the gesture just executed
    ask_pending = [False]            # free mode: the person asked something — poll the assistant next
    gave_up = [False]
    wrong_flag = [False]             # a flag/alarm since the latest slip that named no held error
    done = [False]                   # free mode: the person declared the task finished
    nav_blocked = {}                 # free mode: blocked attempts to leave a held wrong page, per beat
    hist = []                        # every executed gesture: (g, scr_before, scr_after, changed)
    kb_hist = []                     # (gesture, keyboard_before, keyboard_after)

    def say(who, text):
        nonlocal last_line
        transcript.append((who, text))
        conversation.append((dev.st.tick, who, text))
        timeline.append({"t": dev.st.tick, "kind": "say", "who": who, "text": text})
        last_line = f"{who}: {text}"

    def film_frame(kind):
        if film_cap is not None:
            try:
                film.append((film_cap(dev), last_line, dev.st.tick, kind))
            except Exception:
                pass

    def conv_for(viewer):
        out = []
        for t, who, text in conversation[-12:]:
            if viewer == "human":
                tag = "assistant" if who == "assistant" else "you"
            else:
                tag = "you (assistant)" if who == "assistant" else "the person"
            out.append(f"  t{t} {tag}: {text}")
        return "\n".join(out)

    def open_beats():
        return [s["beat"] for s in st.values()
                if s["observable"] and not s["repaired"] and not s["accepted"]
                and not s.get("locked")]

    def armed_beats():
        return [s["beat"] for s in st.values()
                if s["armed"] and not s["accepted"] and not s.get("locked")]

    def _norm(g):
        """People tap the back button by its label ("tap ‹ Send money", "tap < Settings",
        "tap Back"): that is a back press to the engine."""
        t = (g or "").strip()
        low = t.lower()
        if low.startswith("tap "):
            lab = t[4:].strip()
            if lab.startswith(("‹", "<")) or low in ("tap back", "tap go back", "tap the back button"):
                return "press back"
            try:
                from ..render.renderer import _prev_screen_title
                prev = _prev_screen_title(dev.st)
            except Exception:
                prev = None
            if prev and lab.lower() == prev.lower():
                scr = dev.laid_out_screen()
                if not any((w.label or "").lower() == lab.lower() for w in scr.widgets if w.visible):
                    return "press back"
        return g

    screens = []                     # distinct screens the phone has shown, newest last (assistant frames)

    def seen_screen():
        if frame_fn is None or not iface["frames"]:
            return
        try:
            img = frame_fn(dev)
        except Exception:
            return
        if not screens or screens[-1] != img:
            screens.append(img)
            del screens[:-8]

    def exec_gesture(g, beat_idx=None, repair=False):
        g = _norm(g)
        pre = copy.deepcopy(dev.st.data)
        pre_dev = copy.deepcopy(dev) if free else None       # the world just before (read at a commit)
        before_scr = dev.st.cur_screen_key()
        kb0 = bool(dev.st.keyboard)
        r = dev.act(resolve_dynamic(dev, g))
        seen_screen()
        hist.append((g, before_scr, dev.st.cur_screen_key(), dev.st.data != pre,
                     beat_idx is not None and not repair))
        for sb in st.values():
            if (sb["injected_at"] is not None and sb["beat"].mode == "nav" and not sb.get("nav_recovered")
                    and sb.get("target_screen") and dev.st.cur_screen_key() == sb["target_screen"]
                    and not (beat_idx is not None and not repair)):
                sb["nav_recovered"] = True
        kb_hist.append((g, kb0, bool(dev.st.keyboard)))
        if r.ok and beat_idx is None and not repair:
            # a plan step taken while an error is open: remembered so a late
            # repair can redo its EFFECT (screen change / focus / dependent value)
            chg = _diff_paths(pre, dev.st.data)
            scr_change = before_scr != dev.st.cur_screen_key()
            for sb in st.values():
                if sb["observable"] and not sb["accepted"] and not sb["repaired"]:
                    sb["post"].append((g, chg, scr_change, kb0, bool(dev.st.keyboard)))
        timeline.append({"t": dev.st.tick, "kind": "gesture", "who": "human", "text": g,
                         "slip": (beat_idx is not None and not repair), "repair": repair,
                         "beat": beat_idx, "ok": bool(r.ok)})
        film_frame("gesture")
        if r.ok and any(p.startswith(pref) for pref in IRREV for p in _diff_paths(pre, dev.st.data)):
            for sb in st.values():
                if sb["observable"] and not sb["accepted"]:
                    sb["locked"] = True
                if sb["injected_at"] is not None and sb.get("at_irrev_fixed") is None:
                    sb["at_irrev_fixed"] = _fixed_in(pre_dev, sb["beat"]) if pre_dev is not None else None
        return r

    def _fixed_in(d, b):
        """Is the slip undone on device `d`? Value beats: none of the values the
        slip wrote is still in place. Nav beats: the person is on the screen the
        correct turn leads to."""
        s = st[id(b)]
        if b.mode == "nav":
            return s.get("target_screen") is not None and d.st.cur_screen_key() == s["target_screen"]
        if not s.get("slip_paths") or s.get("post_data") is None:
            return False
        paths = s.get("goal_paths") or s["slip_paths"]     # G1 option B: goal-relevant slip paths only
        cd = s.get("correct_data")
        if b.mode == "replace" and cd is not None:
            # fixed = on every goal-relevant path where the slip's world differs from the
            # correct step's world, the world now holds the CORRECT value
            return all(_get(d.st.data, p) == _get(cd, p) for p in paths)
        # an extra tap is undone when every goal-relevant path it wrote is back to its pre-slip value
        pre = s.get("pre")
        if pre is not None:
            return all(_get(d.st.data, p) == _get(pre, p) for p in paths)
        return all(_get(d.st.data, p) != _get(s["post_data"], p) for p in paths)

    def _still_on_screen(x):
        """Is an accepted slip still what the screen shows? A wrong page: the person is still
        on that page (having moved on elsewhere is not the slip standing). A value slip: its
        goal-relevant effect has not been undone."""
        if x["beat"].mode == "nav":
            return x.get("screen") is not None and dev.st.cur_screen_key() == x["screen"]
        return not _fixed_in(dev, x["beat"])

    def _tap_retry(g, bi):
        """Execute a routing/repair gesture; a tap on a row that is scrolled out of
        view is retried after scrolling (scroll state persists per screen, so a
        page the person scrolled since is not where it was)."""
        r = exec_gesture(g, beat_idx=bi, repair=True)
        if r.ok or not g.startswith("tap "):
            return r.ok
        for sw, n in (("swipe down", 2), ("swipe up", 4)):
            for _ in range(n):
                exec_gesture(sw, beat_idx=bi, repair=True)
                if exec_gesture(g, beat_idx=bi, repair=True).ok:
                    return True
        return False

    def _route_to(target, slip_idx, bi=None):
        """Bring the person back to `target` (the screen a slip happened on) from
        wherever they are now: the shortest path over the screen transitions this
        run has actually made (never a slip's own transition), retried from one
        level up when a step fails."""
        from collections import deque
        for _ in range(4):
            cur = dev.st.cur_screen_key()
            if cur == target:
                return True
            prev = None
            for allow_changing in (False, True):
                edges = {}
                for g, b4, aft, changed, was_slip in hist:
                    if b4 != aft and not was_slip and not g.startswith("swipe") \
                            and (allow_changing or not changed):
                        edges.setdefault(b4, {}).setdefault(aft, g)
                prev = {cur: None}
                q = deque([cur])
                while q and target not in prev:
                    u = q.popleft()
                    for v, g in edges.get(u, {}).items():
                        if v not in prev:
                            prev[v] = (u, g); q.append(v)
                if target in prev:
                    break
            if target in prev:
                path = []
                v = target
                while prev[v] is not None:
                    u, g = prev[v]; path.append(g); v = u
                for g in reversed(path):
                    if not _tap_retry(g, bi):
                        break
                if dev.st.cur_screen_key() == target:
                    return True
            exec_gesture("press back", beat_idx=bi, repair=True)
        return dev.st.cur_screen_key() == target

    def _fallback_repair(b):
        """The person accepted and is convinced, but the gestures they stated did
        not undo the slip (or they named none): the body, released, undoes it the
        canonical way from the screen the slip happened on. Counted separately."""
        s = st[id(b)]
        s["fallback"] = True
        bi = beats.index(b)
        cur = dev.st.cur_screen_key()
        rep_ = list(b.repair or [])
        n_before = len(timeline)
        if b.mode == "nav":
            if cur == s["pre_screen"] and rep_ and rep_[0] == "press back":
                rep_ = rep_[1:]                  # already backed out — take the right turn
            elif cur != s["screen"]:
                _route_to(s["screen"], s["slip_hist_idx"], bi)
            for g in rep_:
                exec_gesture(g, beat_idx=bi, repair=True)
                if _fixed_in(dev, b):
                    break
            rep_ = []
        else:
            # value beat: close the keyboard if the undo starts with a back (so the
            # back navigates instead of dismissing), go back to the screen the slip
            # happened on if the person has moved on, undo it there, then redo the
            # EFFECTS of the plan steps taken since (v1 semantics)
            kb_was_open = bool(dev.st.keyboard)
            if kb_was_open and rep_ and rep_[0] == "press back":
                exec_gesture("press back", beat_idx=bi, repair=True)
            routed_ok = True
            if dev.st.cur_screen_key() != s["screen"]:
                routed_ok = _route_to(s["screen"], s["slip_hist_idx"], bi)
            for g in rep_:
                _tap_retry(g, bi)
                if _fixed_in(dev, b):
                    break
            rep_ = []
            if routed_ok and dev.st.cur_screen_key() != cur:
                # back to where the person was when they accepted, then redo any
                # value the detour invalidated (e.g. a code that was re-requested)
                d0 = copy.deepcopy(dev.st.data)
                _route_to(cur, s["slip_hist_idx"], bi)
                touched = {p.rsplit(".", 1)[0] for p in _diff_paths(d0, dev.st.data)}
                for g, chg, scr_change, kb0, kb1 in s["post"]:
                    parents = {p.rsplit(".", 1)[0] for p in chg}
                    if scr_change:
                        continue
                    if kb0 != kb1:
                        if bool(dev.st.keyboard) == kb0:      # a focus/dismiss: same effect now?
                            exec_gesture(g, beat_idx=bi, repair=True)
                    elif parents & touched:
                        exec_gesture(g, beat_idx=bi, repair=True)
            if kb_was_open and not dev.st.keyboard:
                opener = next((g for g, k0, k1 in reversed(kb_hist)
                               if not k0 and k1 and g != "press back"), None)
                if opener:
                    exec_gesture(opener, beat_idx=bi, repair=True)
        for g in rep_:
            exec_gesture(g, beat_idx=bi, repair=True)
            if _fixed_in(dev, b):
                break                            # the person may already have taken the correct step
        done_ = [e["text"] for e in timeline[n_before:] if e["kind"] == "gesture"]
        own_hist.extend(done_)
        timeline.append({"t": dev.st.tick, "kind": "fallback_repair", "who": "engine",
                         "beat": bi, "text": " ; ".join(done_)})

    # ---------------- assistant poll ----------------
    def poll():
        nonlocal n_polls, n_talk, false_flags, false_flags_engine, typed_flags, typed_right, repeat_flags, memory
        n_polls += 1
        if frame_fn is not None and iface["frames"]:
            # "the last N screenshots of their phone": the last N distinct screens the phone
            # showed, including those passed between two looks (a switch flipped and left)
            seen_screen()
            frames[:] = screens[-k_frames:]
        rel = readout.relational(dev) if iface["relational"] else ""
        try:
            reply = assistant.turn(frames=list(frames), relational_text=rel, memory=memory,
                                   conversation=conv_for("assistant"), tick=dev.st.tick,
                                   open_beats=tuple(open_beats()),
                                   screen_title=(dev.laid_out_screen().title or ""),
                                   deviation_observable=bool(open_beats()), ir=ir, dev=dev)
        except Exception:
            reply = {"text": "", "flag": None, "task_progress": None, "goal_change": None}
        text = (reply.get("text") or "").strip()
        flag = reply.get("flag") if isinstance(reply.get("flag"), dict) else None
        etype = (flag or {}).get("error")
        if reply.get("task_progress") is not None:
            progress_log.append({"t": dev.st.tick, "task_progress": reply["task_progress"],
                                 "goal_change": reply.get("goal_change")})
        memory = reply.get("note") or memory
        say_ev = None
        if text:
            n_talk += 1
            say("assistant", text); film_frame("say")
            say_ev = timeline[-1]
        # ---- credit the flag: exact type on an open beat, else its text on a control ----
        nonlocal last_credited
        credited = None
        repeat_of = None
        ob = open_beats()
        if etype:
            typed_flags += 1
            credited, by_type = _credit_typed(text, etype, ob)
            if by_type:
                typed_right += 1
            elif credited is not None:
                st[id(credited)]["type_ok"] = False
            if credited is None:
                # the same warning again, about an error the person already accepted whose slip is
                # still on screen (their fix has not run yet): a repeat, not a false alarm
                standing = [x["beat"] for x in st.values()
                            if x["accepted"] and x["injected_at"] is not None and _still_on_screen(x)]
                again = _credit_typed(text, etype, standing)[0] if standing else None
                if again is not None:
                    repeat_flags += 1
                    repeat_of = beats.index(again)
                else:
                    false_flags_engine += 1
                    if grader == "engine":
                        false_flags += 1
                    wrong_flag[0] = True
        elif text:
            # an untyped sentence is secondary credit only if it is an alarm AND names
            # the slipped control (or, for a wrong page, says to go back) — generic advice
            # that happens to mention the intended control is not a detection
            credited = next((b for b in ob if _alarm_v2(text, b) and _names_slip(text, b)), None)
            if credited is not None:
                st[id(credited)]["type_ok"] = False
            elif ob and _alarm_v2(text):
                wrong_flag[0] = True
        if etype or credited is not None:
            # log the flag with the line it came with (films and audits read it)
            fl = {"flag": etype, "credited": (beats.index(credited) if credited is not None else None)}
            if repeat_of is not None:
                fl["repeat_of"] = repeat_of
            if say_ev is not None:
                say_ev.update(fl)
            else:
                timeline.append({"t": dev.st.tick, "kind": "flag", "who": "assistant", **fl})
        if credited is not None:
            s = st[id(credited)]
            if s["type_ok"] is None:
                s["type_ok"] = True
            if s["engine_detected_tick"] is None:
                s["engine_detected_tick"] = dev.st.tick
            if grader == "engine":       # only the engine arm lets its own credit arm the release
                s["armed"] = True
                if s["detected_tick"] is None:
                    s["detected_tick"] = dev.st.tick
                if credited not in last_credited:
                    last_credited.append(credited)
        if text:
            spoke_since[0] = True
        return text

    spoke_since = [False]          # did the assistant say anything since the person's last turn?
    last_asked = [False]           # was the person's latest turn a question to the assistant?

    def _match_call(about, held):
        """Which held error is the person's call about? The only one held, else their own words."""
        from .composite_run import flag_matches_beat
        if len(held) == 1:
            return held[0]
        if about:
            hit = [b for b in held if flag_matches_beat(about, b) or _names_slip(about, b)]
            if len(hit) == 1:
                return hit[0]
        return None

    def _person_call(h, held):
        """v3: the person judges whether the assistant's words named a mistake it really made.
        Its "correct" arms the release (the accept move still does the releasing); its "wrong"
        is what confirms a false alarm; a "correct" with nothing held is a call about a mistake
        of its own, which the engine cannot see."""
        nonlocal own_error_calls, denials, ambiguous_calls, false_flags, repeat_calls
        call = (h.get("assistant_error_call") or "none").strip().lower()
        about = (h.get("call_about") or "").strip()
        if call not in ("correct", "wrong") or not spoke_since[0]:
            return
        target = _match_call(about, held) if held else None
        bi = beats.index(target) if target is not None else None
        entry = {"t": dev.st.tick, "call": call, "about": about, "beat": bi,
                 "held": [beats.index(b) for b in held], "class": None}
        person_calls.append(entry)
        timeline.append({"t": dev.st.tick, "kind": "call", "who": "human", "text": call,
                         "about": about, "beat": bi})
        if call == "correct":
            if target is not None:
                sb = st[id(target)]
                sb["armed"] = True
                if sb["person_detected_tick"] is None:
                    sb["person_detected_tick"] = dev.st.tick
                if sb["detected_tick"] is None:
                    sb["detected_tick"] = dev.st.tick
                if target not in last_credited:
                    last_credited.append(target)
                entry["class"] = "credit"
            elif held:
                ambiguous_calls += 1     # several errors held and their words fit none of them
                entry["class"] = "ambiguous"
            else:
                # nothing is held: either the person agreeing AGAIN with a warning it has just
                # accepted and not yet fixed, or a mistake of its own that the engine cannot see
                standing = [x["beat"] for x in st.values()
                            if x["accepted"] and x["injected_at"] is not None and _still_on_screen(x)]
                again = _match_call(about, standing) if standing else None
                if again is not None:
                    repeat_calls += 1
                    entry["class"] = "repeat"
                else:
                    own_error_calls += 1
                    entry["class"] = "own"
        else:
            if held:
                denials += 1             # the person denies a live error: the audit cell
                entry["class"] = "denial"
            else:
                false_flags += 1         # confirmed false alarm: nothing was wrong
                entry["class"] = "false_alarm"

    # ---------------- human turn ----------------
    def body_lines(next_gesture):
        if free:
            lines = [f"  JUST DONE (your last tap): {last_g[0] or '(nothing yet)'}"]
            lines.append("  QUEUE (your own next taps, in order): "
                         + (" → ".join(queue[:6]) if queue
                            else "(EMPTY — nothing lined up)"))
            return "\n".join(lines)
        q = [x["gesture"] for x in stream[i + 1:i + 7] if x.get("gesture")]
        if pending:
            q = list(pending) + q
        lines = [f"  NOW (your hands are executing): {next_gesture or '(nothing — you are pausing)'}"]
        lines.append("  QUEUE (lined up next): " + (" → ".join(q[:6]) if q else "(nothing)"))
        return "\n".join(lines)

    def human_turn(next_gesture):
        nonlocal pending
        kw = {}
        held = open_beats()
        status = ("correct" if any(st[id(b)]["armed"] for b in held)
                  else "incorrect" if wrong_flag[0] else "unspotted")
        mistakes = [{"kind": b.kind, "alt": b.alt_gesture, "correct": b.correct_gesture} for b in held]
        if free:
            human.released = not bool(held)              # the system task holds only while an error is held
            kw = {"history": list(own_hist[-14:]), "live": True, "engine_hint": status,
                  "queue_empty": not queue, "grader": grader, "mistakes": mistakes}
        else:
            kw = {"engine_hint": status, "grader": grader, "mistakes": mistakes}
        try:
            h = human.turn(exact_map=readout.exact_map(dev), body=body_lines(next_gesture),
                           conversation=conv_for("human"), **kw)
        except TypeError:
            h = human.turn(exact_map=readout.exact_map(dev), body=body_lines(next_gesture),
                           conversation=conv_for("human"))
        except Exception:
            return None
        mv = h.get("move") or ""
        moves.append(mv)
        u = (h.get("utterance") or "").strip()
        last_asked[0] = bool(u and "?" in u) or mv in ("ask_for_more_guidance", "ask_why")
        if u:
            say("human", u); film_frame("say")
            if "?" in u:
                ask_pending[0] = True
        want_done = bool(free and h.get("done") and not open_beats() and not forced)
        timeline.append({"t": dev.st.tick, "kind": "move", "who": "human", "text": mv,
                         "close": bool(h.get("close_conversation")),
                         "plan_update": h.get("plan_update")})
        if grader == "person":
            _person_call(h, held)                        # may arm a beat for the accept just below
        spoke_since[0] = False
        # release: accept while a beat is armed (the CookSim paradigm-end gate)
        if mv == "accept" and armed_beats():
            scope = [b for b in last_credited if b in armed_beats()]   # only what was just flagged
            if not scope:
                last_credited.clear()
                if free:
                    pu = h.get("plan_update") or {}
                    _apply_plan(pu, list(pu.get("actions") or []))
                    if want_done and not queue:
                        done[0] = True
                return h
            for b in scope:
                st[id(b)]["accepted"] = True
            last_credited.clear()
            human.released = True
            say_plan = h.get("plan_update") or {}
            acts = list(say_plan.get("actions") or [])
            if free:
                # free mode: release is ALL the engine does. The hold lifts, the person's
                # stated plan is its own queue, and undoing the slip is the person's doing —
                # no dry-run, no canonical undo. Whether it got undone is read off the final
                # world by the counterfactual grader.
                for b in [b for b in beats if st[id(b)]["accepted"] and not st[id(b)]["repaired"]]:
                    st[id(b)]["repaired"] = True          # released (the label means "no longer held")
                    timeline.append({"t": dev.st.tick, "kind": "released", "who": "engine",
                                     "beat": beats.index(b), "text": b.alt_gesture})
                forced.clear()                            # the person re-plans now; a deferred tap of theirs is theirs to restate
                _apply_plan(say_plan, acts)
                if want_done and not queue:
                    done[0] = True
                return h
            if not acts and not free:
                # the accept came under the autopilot schema — one planning turn
                # with the live schema so the person states its fix
                try:
                    h2 = human.turn(exact_map=readout.exact_map(dev),
                                    body=body_lines(next_gesture), conversation=conv_for("human"))
                except Exception:
                    h2 = {}
                u2 = (h2.get("utterance") or "").strip()
                if u2:
                    say("human", u2); film_frame("say")
                moves.append(h2.get("move") or "")
                say_plan = h2.get("plan_update") or {}
                acts = list(say_plan.get("actions") or [])
                timeline.append({"t": dev.st.tick, "kind": "plan", "who": "human",
                                 "text": h2.get("move") or "", "plan_update": say_plan})
            # execute the person's own fix now, tagged as the repair of the accepted beats
            acc = [b for b in beats if st[id(b)]["accepted"] and not st[id(b)]["repaired"]]
            # the body, released, tries the person's stated fix on a scratch copy
            # first: it is carried out for real only if it actually undoes the
            # slip; otherwise the body undoes it the canonical way (counted apart)
            fixed_by_own = set()
            prefix = 0
            if acts and acc:
                trial = copy.deepcopy(dev)
                for n_, g in enumerate(acts[:8], 1):
                    try:
                        trial.act(resolve_dynamic(trial, g))
                    except Exception:
                        break
                    fixed_by_own = {id(b) for b in acc if _fixed_in(trial, b)}
                    if fixed_by_own:
                        prefix = n_          # shortest prefix that undoes the slip
                        break
            consumed = 0
            if fixed_by_own:
                for g in acts[:prefix]:
                    exec_gesture(g, beat_idx=beats.index(acc[0]), repair=True)
                    own_hist.append(g)
                consumed = prefix
                timeline.append({"t": dev.st.tick, "kind": "human_fix", "who": "human",
                                 "beat": beats.index(acc[0]), "text": " ; ".join(acts[:prefix])})
            for b in acc:
                s_ = st[id(b)]
                if id(b) in fixed_by_own and _fixed_in(dev, b):
                    s_["human_fix"] = True
                else:
                    _fallback_repair(b)
                s_["repaired"] = True
            if (say_plan.get("op") == "replace") and acts:
                pending = []                     # their plan replaced the remaining queue
            if free:
                if consumed:
                    rest = acts[consumed:]                 # the rest of their plan after their fix
                else:
                    # the body did the undo; drop the leading part of their stated plan
                    # that was their (failed) attempt at it, keep what continues the task
                    rest, k_bad = list(acts), 0
                    if any(b.mode == "nav" for b in acc):
                        if acts and acts[0].strip().lower() == "press back":
                            k_bad = 1                       # "let me go back" — the body already did
                    else:
                        trial = copy.deepcopy(dev)
                        for n_, g in enumerate(acts[:8], 1):
                            try:
                                trial.act(resolve_dynamic(trial, g))
                            except Exception:
                                break
                            if not all(_fixed_in(trial, b) for b in acc):
                                k_bad = n_
                    rest = acts[k_bad:]
                _apply_plan(say_plan, rest)
                if want_done and not queue:
                    done[0] = True
            return h
        # not released: plan updates are recorded; in scripted mode the body stays
        # committed, in free mode the person's own queue is theirs to change
        last_credited.clear()
        if free:
            pu = h.get("plan_update") or {}
            _apply_plan(pu, list(pu.get("actions") or []))
            if want_done and not queue:
                done[0] = True                   # finished, and nothing left queued
        return h

    def _apply_plan(pu, acts):
        op = (pu or {}).get("op") or "drop"
        if acts and op in ("replace", "append"):
            done[0] = False                      # queuing new taps withdraws an earlier "done"
        if op == "replace":
            queue[:] = list(acts)
        elif op == "append":
            queue.extend(acts)

    # ---------------- firing an error / executing a step ----------------
    def fire(g, bidx):
        """Execute `g`; if it is a slip (bidx), record everything the release and
        grading machinery needs. Returns (result, beat_state or None)."""
        if bidx is not None:
            human.released = False          # a new committed sequence: the system task is back
            wrong_flag[0] = False
        pre_d = copy.deepcopy(dev.st.data) if bidx is not None else None
        target_screen = correct_data = None
        if bidx is not None and beats[bidx].mode in ("nav", "replace") and beats[bidx].correct_gesture:
            probe = copy.deepcopy(dev)          # the world had the correct step been taken instead
            try:
                probe.act(resolve_dynamic(probe, beats[bidx].correct_gesture))
                target_screen = probe.st.cur_screen_key()
                correct_data = copy.deepcopy(probe.st.data)
            except Exception:
                target_screen = correct_data = None
        r = exec_gesture(g, beat_idx=bidx)
        s = None
        if bidx is not None and r.ok:
            s = st[id(beats[bidx])]
            s["observable"] = True
            if s["injected_at"] is None:
                s["injected_at"] = dev.st.tick
                s["pre"] = pre_d
                s["post_data"] = copy.deepcopy(dev.st.data)
                s["slip_paths"] = ((_diff_paths(correct_data, dev.st.data) if correct_data is not None else [])
                                   or _diff_paths(pre_d, dev.st.data))
                s["goal_paths"] = _goal_paths(s["slip_paths"], dev.st.data)
                s["target_screen"] = target_screen
                s["correct_data"] = correct_data
                s["screen"] = dev.st.cur_screen_key()
                s["pre_screen"] = hist[-1][1]
                s["slip_hist_idx"] = len(hist) - 1
        return r, s

    def nav_hold(s):
        """The person is on a wrong page: they hold (talk) until they accept a
        correct flag or run out of patience; then they give up the task."""
        nonlocal turn
        held = 0
        while held < HOLD_MAX and not s["accepted"] and dev.st.outcome is None:
            if held >= HOLD_POLLS and not (moves and moves[-1] in ENGAGED):
                break
            if poll():
                human_turn(None)
            held += 1; turn += 1
        if not s["accepted"]:
            s["gave_up"] = True
            gave_up[0] = True
            say("human", "I can't find what I need on this page. I'm giving up on this one.")
            return False
        return True

    # ---------------- free mode: the person plans, the engine takes over at anchors ----------------
    def _effect(g):
        """Dry-run `g` on a copy: (ok, changed paths, screen after)."""
        g = _norm(g)
        d = copy.deepcopy(dev)
        try:
            r = d.act(resolve_dynamic(d, g))
        except Exception:
            return (False, [], None)
        return (bool(r.ok), _diff_paths(dev.st.data, d.st.data), d.st.cur_screen_key())

    def anchor_for(g):
        """The first unfired beat whose anchor step the person is about to take:
        same gesture, or the same effect on the world (same written paths, or the
        same destination screen for a navigation step)."""
        cur = dev.st.cur_screen_key()
        eg = None
        for b in beats:
            s = st[id(b)]
            if s["injected_at"] is not None or s.get("skipped") or not b.correct_gesture:
                continue
            if anchor_screen.get(id(b)) != cur:
                continue                     # the same label elsewhere is not this step
            a = b.correct_gesture
            if g.strip().lower() == a.strip().lower():
                return b
            ea = _effect(a)
            if not ea[0]:
                continue
            if eg is None:
                eg = _effect(g)
            if not eg[0]:
                return None
            if ea[1] and ea[1] == eg[1]:
                return b
            if not ea[1] and ea[2] != cur and ea[2] == eg[2]:
                return b
        return None

    def forced_sequence(b, g):
        bi = beats.index(b)
        if b.mode in ("replace", "nav"):
            return [(b.alt_gesture, bi)]
        if b.pre:
            return [(b.alt_gesture, bi), (g, None)]
        return [(g, None), (b.alt_gesture, bi)]

    def would_undo_open(g):
        """The hold: while an error is held (flagged or not, but not yet accepted),
        the person's own hands cannot undo it — a value must stay as the slip
        wrote it, a wrong page cannot be left."""
        held = [b for b in open_beats()]
        if not held:
            return None
        ok, chg, scr = _effect(g)
        if not ok:
            return None
        if any(p.startswith(pref) for pref in IRREV for p in chg):
            return None                  # an irreversible commit commits the error; never held back
        for b in held:
            s = st[id(b)]
            if b.mode == "nav":
                if scr != s["screen"]:
                    why[0] = f"{b.kind}: would leave {s['screen']} for {scr}"
                    return b
            else:
                d = copy.deepcopy(dev)
                try:
                    d.act(resolve_dynamic(d, g))
                except Exception:
                    continue
                bad = [p for p in (s.get("slip_paths") or [])
                       if _get(d.st.data, p) != _get(s["post_data"], p)
                       and _get(d.st.data, p) == _get(s["pre"], p)]
                # a replaced step is also "undone" by doing the intended step itself: while
                # held, the person's hands cannot put the goal-relevant value the slip left
                # untouched into its correct state (the intended toggle stays off)
                cd = s.get("correct_data")
                if not bad and b.mode == "replace" and cd is not None:
                    bad = [p for p in (s.get("goal_paths") or [])
                           if _get(s["post_data"], p) != _get(cd, p)
                           and _get(d.st.data, p) == _get(cd, p)]
                if bad:
                    why[0] = f"{b.kind}: {bad} would change"
                    return b
        return None
    why = [""]

    turn = 0
    i = 0
    if free:
        stuck = 0
        idle = 0                         # consecutive taps that changed nothing and revisited known screens
        seen_screens = {dev.st.cur_screen_key()}
        max_turns = min(max_turns, 3 * ref_ticks + 24)
        while dev.st.outcome is None and turn < max_turns and not gave_up[0] and not done[0]:
            if turn % POLL == 0 or ask_pending[0]:
                ask_pending[0] = False
                if poll():
                    human_turn(queue[0] if queue else None)
            if not queue and not forced:
                h = human_turn(None)                 # the person must decide what to do next
                if done[0]:
                    break
                if not queue and task.goal(dev.st)[0]:
                    done[0] = True                   # nothing more to do and the goal is met
                    break
                if not queue:
                    spoke = bool((h or {}).get("utterance"))
                    if ask_pending[0]:               # they asked — let the assistant answer first
                        ask_pending[0] = False
                        if poll():
                            human_turn(None)
                        turn += 1
                        if queue:
                            stuck = 0
                            continue
                    stuck += 1                       # talking is not acting
                    if spoke and stuck >= 1 and not last_asked[0]:   # asking and waiting is not idling
                        last_g[0] = ((last_g[0] or "(nothing yet)").split("   (")[0]
                                     + "   (your last reply said you would act but queued NOTHING — "
                                     "the phone only moves on plan_update actions)")
                    if stuck >= 6:
                        gave_up[0] = True
                        say("human", "That's all I'm going to do here." if task.goal(dev.st)[0] is False and own_hist
                            else "I don't know how to go on from here. I'm stopping.")
                        break
                    turn += 1
                    continue
                stuck = 0
            if forced:
                g, bidx = forced.pop(0)
            else:
                g, bidx = queue.pop(0), None
                b = anchor_for(g)
                if b is not None and not _effect(b.alt_gesture)[0]:
                    # the wrong row is scrolled out of view: the person scrolls a little
                    # (as they would to find the row) — if it still cannot be tapped,
                    # this error cannot be forced here and the beat is skipped
                    revealed = False
                    for sw in ("swipe up", "swipe up", "swipe down", "swipe down", "swipe down"):
                        if not _effect(g)[0] and not _effect(b.alt_gesture)[0]:
                            pass
                        exec_gesture(sw); own_hist.append(sw)
                        if _effect(b.alt_gesture)[0] and _effect(g)[0]:
                            revealed = True
                            break
                    if not revealed:
                        st[id(b)]["skipped"] = True
                        timeline.append({"t": dev.st.tick, "kind": "unfirable", "who": "engine",
                                         "beat": beats.index(b), "text": b.alt_gesture})
                        b = None
                if b is not None:
                    seq = forced_sequence(b, g)
                    (g, bidx), forced = seq[0], seq[1:]
                    timeline.append({"t": dev.st.tick, "kind": "takeover", "who": "engine",
                                     "beat": beats.index(b), "text": f"{b.kind}: {b.alt_gesture}"})
            if bidx is None:
                hb = would_undo_open(g)
                if hb is not None:
                    timeline.append({"t": dev.st.tick, "kind": "blocked", "who": "engine",
                                     "beat": beats.index(hb), "text": g, "why": why[0]})
                    if hb.mode == "nav":
                        nav_blocked[id(hb)] = nav_blocked.get(id(hb), 0) + 1
                        if nav_blocked[id(hb)] >= NAV_PATIENCE:
                            st[id(hb)]["gave_up"] = True
                            gave_up[0] = True
                            say("human", "I can't find what I need on this page. I'm giving up on this one.")
                            break
                    dev._tick(1)                          # a tap attempt takes time even when nothing changes
                    # a tap that did nothing interrupts the plan: the rest of the queue was planned
                    # on the assumption that it worked, so the person looks again before going on
                    queue.clear()
                    last_g[0] = f"{g}   (the phone did not respond to that tap — it stays as it is)"
                    own_hist.append(f"{g} (no effect)")
                    idle += 1
                    if idle in (4, 8):
                        last_g[0] += "  — this tap has not changed anything several times now"
                    if idle >= 12:
                        gave_up[0] = True
                        say("human", "I can't get this to work. I'm stopping here.")
                        break
                    turn += 1
                    continue
            d_before = hist[-1][3] if False else None
            r, s = fire(g, bidx)
            last_g[0] = g
            own_hist.append(g)
            changed, scr_now = hist[-1][3], dev.st.cur_screen_key()
            if changed or scr_now not in seen_screens:
                idle = 0
            else:
                idle += 1
            seen_screens.add(scr_now)
            if idle in (6, 12):
                # the body block tells the person it is going round in circles (its queue is kept)
                last_g[0] = g + "   (you have tapped around for a while without getting anywhere new)"
            if idle >= 24:
                gave_up[0] = True
                say("human", "I can't find where this is. I'm giving up on this one.")
                break
            if s is not None:
                if beats[bidx].mode == "nav":
                    nav_blocked[id(beats[bidx])] = 0     # the person gets turns; leaving is blocked until release
                    if poll():
                        human_turn(queue[0] if queue else None)
                elif poll():
                    human_turn(queue[0] if queue else None)
            turn += 1
    else:
        while i < len(stream) and turn < max_turns and dev.st.outcome is None:
            step = stream[i]
            if turn % POLL == 0:
                if poll():
                    human_turn(step.get("gesture"))
            if step["role"] == "HOLD":
                held = 0
                while held < HOLD_POLLS and open_beats() and dev.st.outcome is None:
                    if poll():
                        human_turn(None)
                    held += 1; turn += 1
                i += 1
                continue
            g = step["gesture"]
            bidx = beats.index(step["beat"]) if step["beat"] is not None else None
            r, s = fire(g, bidx)
            if s is not None:
                if step["beat"].mode == "nav":
                    if not nav_hold(s):
                        break
                elif poll():
                    human_turn(stream[i + 1]["gesture"] if i + 1 < len(stream) else None)
            i += 1
            turn += 1

    # final listening window
    if not any(st[id(b)]["gave_up"] for b in beats) and not gave_up[0]:
        held = 0
        while held < HOLD_POLLS and open_beats() and dev.st.outcome is None and turn < max_turns:
            if poll():
                human_turn(None)
            held += 1; turn += 1

    goal_ok, _ = task.goal(dev.st)
    final = dev.st.data

    def _counterfactual(bi):
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
        # authoritative detection is the UNION of the two graders (the person's call and the
        # engine's flag verdict); both are kept so either number can be re-derived offline
        ticks = [t for t in (s.get("person_detected_tick"), s.get("engine_detected_tick"),
                             s["detected_tick"]) if t is not None]
        s["detected_tick"] = min(ticks) if ticks else None
        detected = s["detected_tick"] is not None
        if b.mode == "nav" and free:
            committed = s["gave_up"] or not s.get("nav_recovered")    # never reached the right page
            prevented = detected and not committed
        elif b.mode == "nav":
            prevented = s["accepted"]
            committed = s["gave_up"]
        elif free:
            if s["injected_at"] is None:
                committed = False
            elif s.get("at_irrev_fixed") is not None:
                committed = not s["at_irrev_fixed"]
            else:
                committed = not _fixed_in(dev, b)
            prevented = detected and not committed
        else:
            relevant = set(ir.touched_paths) | set(b.paths)
            def _rel(p):
                return any(p == r_ or p.startswith(r_ + ".") for r_ in relevant)   # a dict-valued path diffs by leaf
            diff = [p for p in _diff_paths(final, _counterfactual(bi))
                    if (_rel(p) or any(p.startswith(pref) for pref in IRREV))
                    and not p.startswith(("bank.otp", "messages."))]
            committed = bool(diff)
            prevented = detected and not committed
        if free and s["accepted"]:
            s["human_fix"] = not committed
            s["fallback"] = False
        by = ("both" if (s.get("person_detected_tick") is not None and s.get("engine_detected_tick") is not None)
              else "person" if s.get("person_detected_tick") is not None
              else "engine" if s.get("engine_detected_tick") is not None else None)
        recs.append({"detected_by": by,
                     "detected_tick_person": s.get("person_detected_tick"),
                     "detected_tick_engine": s.get("engine_detected_tick"),
                     "kind": b.kind, "detected": detected, "prevented": prevented,
                     "committed": bool(committed), "accepted": s["accepted"],
                     "type_ok": s["type_ok"], "injected_at": s["injected_at"],
                     "detected_tick": s["detected_tick"], "human_fix": s["human_fix"],
                     "fallback": s["fallback"], "slip_paths": list(s.get("slip_paths") or []),
                     "goal_paths": list(s.get("goal_paths") or []),
                     "screen": s.get("screen"), "skipped": bool(s.get("skipped"))})
    fired_recs = [r for r in recs if r["injected_at"] is not None]
    E = len(fired_recs)                       # designed errors that actually fired this run
    D = sum(1 for r in fired_recs if r["detected"])
    prevented = sum(1 for r in fired_recs if r["prevented"])
    R = D / E if E else None
    P = D / (D + false_flags) if (D + false_flags) else (1.0 if D else None)
    D_eng = sum(1 for r in fired_recs if r["detected_tick_engine"] is not None)
    R_eng = D_eng / E if E else None
    P_eng = (D_eng / (D_eng + false_flags_engine) if (D_eng + false_flags_engine)
             else (1.0 if D_eng else None))
    F1 = (2 * P * R / (P + R)) if (P and R and (P + R)) else 0.0
    DT = sum(1.0 if r["prevented"] else 0.5 if r["detected"] else 0.0 for r in fired_recs) / E if E else None
    cap = 2 * ref_ticks + 8
    S_lenient = 1 if (goal_ok and dev.st.tick <= cap) else 0
    # strict success: the phone ends with NO change the owner did not ask for. Anything the
    # final state changed beyond what the reference solution changes fails the episode,
    # whoever caused it: an unrepaired slip, the person's own stray tap, or an extra step the
    # assistant told them to take.
    # text in a search box is not a setting: it is how iOS finds one. Counting it as an unwanted change
    # failed episodes whose only "extra" was a typed query (audit 2026-09-22).
    SCRATCH = ("settings._search", "settings._apps_search")
    extra_changes = sorted(p for p in (set(_diff_paths(init_data, dev.st.data))
                                       - set(_diff_paths(init_data, ref_dev.st.data)))
                           if not p.startswith(SCRATCH))
    S = 1 if (S_lenient and not extra_changes) else 0
    CC = 1 - min(1.0, n_talk / max(1, n_polls))
    film_frame("end")
    return {
        "episode": composite.id, "task": composite.task, "persona": persona.name, "n_errors": E,
        "n_designed": len(beats), "n_fired": E,
        "beats": recs, "recall": R, "precision": P, "f1": round(F1, 3), "dt": DT, "success": S,
        "success_lenient": S_lenient, "extra_changes": extra_changes,
        "human_fixes": sum(1 for r in recs if r["human_fix"]),
        "fallback_repairs": sum(1 for r in recs if r["fallback"]),
        "grader": grader, "recall_engine": R_eng, "precision_engine": P_eng,
        "false_flags_engine": false_flags_engine, "person_calls": person_calls,
        "own_error_calls": own_error_calls, "denials": denials, "ambiguous_calls": ambiguous_calls,
        "repeat_calls": repeat_calls,
        "cc": round(CC, 3), "false_flags": false_flags, "repeat_flags": repeat_flags, "typed_flags": typed_flags,
        "call_health": _call_health(assistant, human, health0),
        "typed_right": typed_right, "committed_any": any(r["committed"] for r in recs),
        "prevented": prevented, "goal_ok": goal_ok, "final_tick": dev.st.tick,
        "n_polls": n_polls, "n_talk": n_talk, "moves": moves, "transcript": transcript,
        "timeline": timeline, "progress_log": progress_log, "contestant": contestant,
        "family": "error", "instruction": task.instruction, "interface": "v2", "_film": film,
        "engine_rev": _engine_rev(), "human_mode": human_mode,
    }
