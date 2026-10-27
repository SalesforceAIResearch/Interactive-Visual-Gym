"""Day-one tests for the interaction stack (skill reference/day-one-tests.md):
cross-process determinism, anchor invariance across alternate routes, case
admissibility, and channel isolation on composed prompts.
"""
import json
import subprocess
import sys

from screensim.interact.plan_ir import compile_plan, deep_diff
from screensim.tasks import by_id


def test_cross_process_determinism():
    """Two fresh interpreters must produce the same state hash — this is the
    test that catches salted-entropy defects (skill worked-examples §1)."""
    prog = (
        "from screensim.core.engine import Device\n"
        "d=Device(0); d.open_app('bank')\n"
        "for c in ['tap Send money','tap Alex Chen','tap the amount box',"
        "'type \"40\"','press back','swipe up','tap Review',"
        "'tap Confirm & send']:\n"
        "    d.act(c)\n"
        "print(d.st.hash())\n")
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(2)}
    assert len(outs) == 1, f"hash differs across processes: {outs}"


def test_anchor_invariant_across_alternate_routes():
    """The progress key (committed application state) must be identical for
    the main plan and every declared alternate route (skill §2.3, §12.2)."""
    for tid in ("loc_never_tripsnap", "dark_via_cc", "mic_quickpay_off"):
        t = by_id(tid)
        main = compile_plan(t)
        for label, alt in t.alt_plans:
            altir = compile_plan(t, plan=alt)
            assert main.commits == altir.commits, (
                f"{tid} [{label}]: commit sequence differs:\n"
                f"  main {main.commits}\n  alt  {altir.commits}")


def test_progress_key_discriminative_across_checkpoints():
    """Consecutive anchor prefixes must differ — an anchor that collapses to
    an earlier one fires at the wrong time (skill §5.3)."""
    ir = compile_plan(by_id("rent_share_standard"))
    seen = set()
    for k in range(len(ir.commits) + 1):
        key = json.dumps(ir.commits[:k])
        assert key not in seen
        seen.add(key)


# ---- the interaction stack: brackets, restraint, audits ----
from screensim.interact.errors import generate_cases
from screensim.interact.agents import (ScriptedHuman, SilentAssistant,
                                        OracleAssistant)
from screensim.interact.personas import PERSONAS
from screensim.interact.episode import run_episode
from screensim.interact import prompts, verdicts

_TASKS = ["safari_notif_quiet", "roaming_off", "precise_off_weatherly",
          "limit_adult_web", "hotspot_2ghz", "rent_share_standard"]


def _pool(persona="baseline"):
    p = PERSONAS[persona]
    out = []
    for tid in _TASKS:
        ir, cases, ledger = generate_cases(by_id(tid), personas=(p.name,))
        out.append((p, ir, cases, ledger))
    return out


def test_silent_assistant_lets_every_error_land():
    """The degenerate silent bracket must NOT prevent errors — else restraint
    would top the leaderboard for doing nothing (skill §7.2)."""
    for p, ir, cases, _ in _pool():
        for c in cases:
            if c.family != "error":
                continue
            ep = run_episode(c, ir, ScriptedHuman(p, ir, c), SilentAssistant())
            assert (not ep["goal_ok"]) or ep["deviation_committed"], \
                f"{c.name}: silent assistant somehow avoided the error"


def test_oracle_detects_and_repairs_every_error():
    """A correct detection releases the takeover and the accepted repair
    reaches the goal — proving injection, hold, release, and conversion."""
    for p, ir, cases, _ in _pool():
        for c in cases:
            if c.family != "error":
                continue
            ep = run_episode(c, ir, ScriptedHuman(p, ir, c),
                             OracleAssistant(c.err, c.deviated_label))
            assert ep["detection_verdict"] in verdicts.CORRECT_DETECTION, \
                f"{c.name}: verdict {ep['detection_verdict']}"
            assert ep["repaired"], f"{c.name}: not repaired after acceptance"


def test_intent_and_clean_have_nothing_to_detect():
    """Non-error families are scored on the restraint track; a silent
    assistant is not charged a false alarm on them."""
    for p, ir, cases, _ in _pool():
        for c in cases:
            if c.family == "error":
                continue
            ep = run_episode(c, ir, ScriptedHuman(p, ir, c), SilentAssistant())
            assert ep["detection_verdict"] is None
            assert ep["restraint_verdict"] != verdicts.FALSE_ALARM


def test_v2_release_needs_an_armed_flag_and_an_accept():
    """v2 (CookSim interface): a correct TYPED flag arms the beat; only the human's
    `accept` move releases it, and the human's own plan_update is the repair. A
    reject leaves the error to commit; an accept with no armed flag releases nothing."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, OracleV2, SilentV2

    class Stub:
        def __init__(self, beats, mode): self.beats, self.mode, self.released = beats, mode, False
        def turn(self, exact_map, body, conversation):
            if self.mode == "reject":
                return {"utterance": "no", "move": "reject", "close_conversation": True,
                        "plan_update": {"op": "drop", "actions": []}}
            last = conversation.strip().split("\n")[-1] if conversation.strip() else ""
            for b in self.beats:
                if b.alt_gesture.replace("tap ", "").lower() in last.lower():
                    return {"utterance": "ok", "move": "accept", "close_conversation": True,
                            "plan_update": {"op": "append", "actions": list(b.repair)}}
            return {"utterance": "", "move": "accept", "close_conversation": True,
                    "plan_update": {"op": "drop", "actions": []}}
    p = PERSONAS["baseline"]
    ir, eps = design_composite_episodes(by_id("lock_screen_lockdown"), per_task=3)
    c = eps[0]
    ok = run_composite_v2(c, ir, p, OracleV2(), Stub(c.beats, "accept"), human_mode="scripted", grader="engine")
    assert ok["prevented"] == ok["n_errors"] and ok["goal_ok"] and ok["typed_right"] == ok["typed_flags"] > 0
    rej = run_composite_v2(c, ir, p, OracleV2(), Stub(c.beats, "reject"), human_mode="scripted", grader="engine")
    assert rej["prevented"] == 0 and all(r["detected"] for r in rej["beats"] if r["injected_at"])
    quiet = run_composite_v2(c, ir, p, SilentV2(), Stub(c.beats, "accept"), human_mode="scripted", grader="engine")
    assert quiet["prevented"] == 0 and quiet["false_flags"] == 0   # accept without an armed flag releases nothing
    # the timeline carries every flag with its credit, and replaying the log rebuilds the run
    logged = [e for e in ok["timeline"] if e.get("flag")]
    assert len(logged) == ok["typed_flags"]
    assert {e["credited"] for e in logged if e.get("credited") is not None} == {
        i for i, r in enumerate(ok["beats"]) if r["detected"]}
    from screensim_bench.video.replay import replay
    for rec in (ok, rej, quiet):
        rec = dict(rec, task="lock_screen_lockdown")
        assert replay(rec)[1]["exact"]


def test_composed_assistant_prompt_has_no_leaks():
    """Channel isolation on the COMPOSED artifact (skill §6)."""
    p = PERSONAS["classic_novice"]
    text = prompts.compose_assistant(4, "switch “Sounds” in the middle",
                                     p.questionnaire, memory="")
    assert prompts.audit_assistant_prompt(text) == []


def test_composed_human_prompt_forbids_coordinates():
    p = PERSONAS["classic_novice"]
    text = prompts.compose_human("do a thing", "SCREEN x", "you are here",
                                 "", "", "", p)
    assert prompts.audit_human_prompt(text) == []


def test_single_verdict_source_not_rehandlisted():
    """No consumer hand-lists the correct-detection tuple (skill §7.2)."""
    import subprocess
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(root, "screensim", "interact")
    hits = []
    for fn in os.listdir(pkg):
        if not fn.endswith(".py") or fn == "verdicts.py":
            continue
        src = open(os.path.join(pkg, fn)).read()
        # a hand-listed {"correct", "early_correct"} set outside verdicts.py
        if '"correct"' in src and '"early_correct"' in src:
            hits.append(fn)
    assert not hits, f"verdict tuple hand-listed in {hits}; import it instead"


# ---- interface version switch (skill §7.2) ----
from screensim.interact import interface, INTERFACES
from screensim.interact.agents import ScriptedHuman as _SH


def test_interface_versions_are_distinct_and_stamped():
    assert interface("sv1")["relational"] is False
    assert interface("sv2")["relational"] is True
    import pytest as _pt
    with _pt.raises(ValueError):
        interface("sv-does-not-exist")


class _RecordingAssistant:
    """Captures whether it was handed relational text — proves the tier
    actually changes what the agent sees, not just a label."""
    def __init__(self):
        self.saw_relational = None
    def turn(self, frames, relational_text, memory, **_):
        self.saw_relational = bool(relational_text)
        return {"note": "", "text": "", "concern": ""}


def test_observation_tier_changes_what_the_assistant_receives():
    p = PERSONAS["baseline"]
    ir, cases, _ = generate_cases(by_id("roaming_off"), personas=(p.name,))
    case = next(c for c in cases if c.family == "error")
    for ver, expect in (("sv1", False), ("sv2", True)):
        a = _RecordingAssistant()
        run_episode(case, ir, _SH(p, ir, case), a, interface_version=ver)
        assert a.saw_relational is expect, \
            f"{ver}: assistant saw relational={a.saw_relational}, want {expect}"


def test_episode_stamps_the_interface_it_ran():
    p = PERSONAS["baseline"]
    ir, cases, _ = generate_cases(by_id("roaming_off"), personas=(p.name,))
    case = next(c for c in cases if c.family == "error")
    ep = run_episode(case, ir, _SH(p, ir, case), SilentAssistant(),
                     interface_version="sv2")
    assert ep["interface"] == "sv2"


def test_typed_flag_does_not_release_an_error_its_words_do_not_name():
    """Round 6 (app_privacy_quarantine c2): two errors held, a location wrong_value and a
    tracking wrong_target. A flag typed wrong_value whose words were only about the tracking
    toggle released the location error by type alone. The type must not override words
    that point at a different held error; a lone held error still credits by type."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import _credit_typed
    ir, eps = design_composite_episodes(by_id("app_privacy_quarantine"), per_task=3)
    c = next(e for e in eps if e.id == "app_privacy_quarantine__c2")
    loc, trk = c.beats
    line = "TripSnap's tracking toggle is still on. Turn that off."
    assert _credit_typed(line, "wrong_value", [loc, trk]) == (None, False)
    assert _credit_typed("You picked Ask Next Time instead of Never.", "wrong_value", [loc, trk]) == (loc, True)
    assert _credit_typed("You flipped the wrong control there.", "wrong_value", [loc]) == (loc, True)
    assert _credit_typed("You turned off Allow Apps to Request to Track.", "wrong_value", [loc, trk]) == (trk, False)


def test_blocked_tap_interrupts_the_rest_of_the_plan():
    """Rounds 5-6: after a held slip blocked the person's tap on the intended switch, the
    hands ran on with the stale rest of the plan ("press back") and left the page without
    the person ever seeing that the switch had not moved. A blocked tap must hand the turn
    back to the person before any further queued tap runs."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, SilentV2
    task = by_id("hearing_setup")
    ir, eps = design_composite_episodes(task, per_task=3)
    c = next(e for e in eps if any(b.kind == "wrong_target" and b.correct_gesture == "tap LED Flash for Alerts"
                                   for b in e.beats))

    class Planner:
        released = False
        def __init__(self):
            self.n = 0
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                plan = list(task.plan)
                i = plan.index("tap LED Flash for Alerts")
                plan[i + 1:i + 1] = ["tap LED Flash for Alerts"]  # tries the switch again, then goes on
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": plan}, "done": False}
            return {"utterance": "", "move": "answer", "close_conversation": True,
                    "plan_update": {"op": "drop", "actions": []}, "done": True}

    rec = run_composite_v2(c, ir, PERSONAS["baseline"], SilentV2(), Planner(), human_mode="free")
    tl = rec["timeline"]
    blocked = [i for i, e in enumerate(tl) if e["kind"] == "blocked"]
    assert blocked, "the held slip should block the retry on LED Flash for Alerts"
    after = [e["kind"] for e in tl[blocked[0] + 1:] if e["kind"] in ("gesture", "blocked", "move")]
    assert after and after[0] == "move", after[:3]


def test_assistant_frames_are_the_last_screens_the_phone_showed():
    """Round 6 films: frames were snapshots taken only when the assistant was polled (every
    second tap), so a switch flipped and then left with Back between two polls was never
    shown, and assistants said "that switch is still on". The prompt promises the last N
    screenshots of the phone; every screen the phone shows must reach some assistant call."""
    import json as _json
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, SilentV2
    task = by_id("stop_profiling")
    ir, eps = design_composite_episodes(task, per_task=3)
    c = next(e for e in eps if e.id == "stop_profiling__c3")

    def frame(d):
        return _json.dumps([d.st.cur_screen_key(), d.st.data["settings"]["spec"]["priv"]], sort_keys=True)

    shown = []

    def frame_fn(d):
        f = frame(d)
        if not shown or shown[-1] != f:
            shown.append(f)
        return f

    class Watch(SilentV2):
        def __init__(self):
            self.seen = set()
        def turn(self, frames=(), **kw):
            self.seen |= set(frames)
            return super().turn(frames=frames, **kw)

    class Planner:
        released = False
        n = 0
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": list(task.plan)}, "done": False}
            return {"utterance": "", "move": "answer", "close_conversation": True,
                    "plan_update": {"op": "drop", "actions": []}, "done": True}

    w = Watch()
    run_composite_v2(c, ir, PERSONAS["baseline"], w, Planner(), frame_fn=frame_fn, human_mode="free")
    missed = [s for s in shown if s not in w.seen]
    assert not missed, f"{len(missed)} of {len(shown)} screens never reached the assistant"


def test_repeating_a_warning_about_an_accepted_error_is_not_a_false_alarm():
    """Round 6 pass 2: 10 of 13 "false alarms" were the assistant repeating its correct,
    typed warning in the moment after the person accepted it, while the slip was still on
    screen. A repeat about an accepted error that still stands is logged as a repeat; a flag
    about nothing real stays a false alarm."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2
    task = by_id("hearing_setup")
    ir, eps = design_composite_episodes(task, per_task=3)
    c = next(e for e in eps if e.id == "hearing_setup__c1")

    class Repeater:
        def __init__(self, bogus=False, nav_too=False):
            self.last, self.bogus, self.nav_too = None, bogus, nav_too
        def turn(self, open_beats=(), **kw):
            if open_beats:
                b = open_beats[0]
                say = {"text": f"You tapped {b.alt_gesture.replace('tap ', '')} by mistake.",
                       "flag": {"error": b.kind}, "task_progress": None, "goal_change": None}
                if b.mode != "nav" or self.nav_too:
                    self.last = say            # a switch slip stays on screen until undone; a wrong page is left
                return say
            if self.bogus:
                return {"text": "You tapped Balance by mistake.", "flag": {"error": "extra_write"},
                        "task_progress": None, "goal_change": None}
            return self.last or {"text": "", "flag": None, "task_progress": None, "goal_change": None}

    class Accepter:
        released = False
        n = 0
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": list(task.plan)}, "done": False}
            last = conversation.strip().split("\n")[-1] if conversation.strip() else ""
            said = "by mistake" in last
            if said and "General" in last:            # the wrong page: go back and start over
                return {"utterance": "", "move": "accept", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": ["press back"] + list(task.plan)}, "done": False}
            return {"utterance": "", "move": "accept" if said else "answer", "close_conversation": True,
                    "plan_update": {}, "done": False}   # the switch slip is never undone: it stays on screen

    rec = run_composite_v2(c, ir, PERSONAS["baseline"], Repeater(), Accepter(), human_mode="free", grader="engine")
    assert rec["prevented"] + sum(1 for b in rec["beats"] if b["committed"]) >= 1
    assert rec["repeat_flags"] >= 1 and rec["false_flags"] == 0, (rec["repeat_flags"], rec["false_flags"])
    bogus = run_composite_v2(c, ir, PERSONAS["baseline"], Repeater(bogus=True), Accepter(), human_mode="free", grader="engine")
    assert bogus["false_flags"] >= 1
    # pass 3: "you tapped General" repeated long after the person left General is a false alarm
    stale = run_composite_v2(c, ir, PERSONAS["baseline"], Repeater(nav_too=True), Accepter(), human_mode="free", grader="engine")
    stale_nav = [e for e in stale["timeline"] if e.get("flag") == "wrong_page" and e.get("credited") is None]
    assert stale_nav and all(e.get("repeat_of") is None for e in stale_nav[1:])


def test_navigation_advice_is_not_a_warning_about_a_switch():
    """Round 6 pass 2: "You're back on Tracking ... next, go back to Privacy & Security and
    open Analytics to finish checking iCloud analytics" counted as a warning (it contains
    "back on" and "go back") and released a held iCloud Analytics slip. "go back" warns only
    about a wrong page; "back on/off" only as undoing a switch."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import _alarm_v2
    ir, eps = design_composite_episodes(by_id("stop_profiling"), per_task=3)
    trk, icloud = next(e for e in eps if e.id == "stop_profiling__c2").beats
    nav = next(b for e in eps for b in e.beats if b.mode == "nav")
    line = ("You're back on Tracking and it looks right. Next, go back to Privacy & Security "
            "and open Analytics & Improvements to finish checking iCloud analytics.")
    assert not _alarm_v2(line, icloud)
    assert _alarm_v2("Turn Share iCloud Analytics back on.", icloud)
    assert _alarm_v2("Go back to Privacy & Security, then tap Tracking.", nav)
    assert not _alarm_v2("Go back to Privacy & Security, then tap Tracking.", trk)



def test_failed_model_calls_are_counted_not_silent():
    """A model call that fails after its retries returns an empty reply, which the runner
    plays as silence. The record must say how many calls failed or came back blank."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, SilentV2
    import screensim.interact.agents as ag
    task = by_id("hearing_setup")
    ir, eps = design_composite_episodes(task, per_task=3)
    c = next(e for e in eps if e.id == "hearing_setup__c1")

    class Broken:
        def text(self, *a, **k):
            self.failures = getattr(self, "failures", 0) + 1
            return "{}"

    class Assist(SilentV2):
        client = Broken()
        def turn(self, **kw):
            self.client.text("x")
            return super().turn(**kw)

    class Planner:
        released = False
        def turn(self, exact_map, body, conversation, **kw):
            return {"utterance": "", "move": "answer", "close_conversation": True,
                    "plan_update": {"op": "replace", "actions": list(task.plan)}, "done": True}

    rec = run_composite_v2(c, ir, PERSONAS["baseline"], Assist(), Planner(), human_mode="free")
    assert rec["call_health"]["assistant"]["failures"] == rec["n_polls"] > 0


def _v3_episode():
    from screensim.interact.composite import design_composite_episodes
    task = by_id("hearing_setup")
    ir, eps = design_composite_episodes(task, per_task=3)
    return task, ir, next(e for e in eps if e.id == "hearing_setup__c1")


def test_person_call_arms_the_release_not_the_engines_word_match():
    """v3 grader: the assistant's words, not its label or our word lists, decide detection. A
    warning with no typed flag and no distinctive slip word earns nothing from the engine arm;
    the person's private "correct" call arms it, and its accept releases the hold."""
    from screensim.interact.composite_run_v2 import run_composite_v2
    task, ir, c = _v3_episode()

    class Vague:
        def turn(self, open_beats=(), **kw):
            if open_beats:
                return {"text": "That isn't what you wanted here.", "flag": None,
                        "task_progress": None, "goal_change": None}
            return {"text": "", "flag": None, "task_progress": None, "goal_change": None}

    class Judge:
        released = False
        n = 0
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": list(task.plan)},
                        "assistant_error_call": "none", "call_about": "", "done": False}
            said = bool(conversation.strip()) and "isn't what you wanted" in conversation
            if said and kw.get("mistakes"):
                return {"utterance": "ah, my slip", "move": "accept", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": ["press back"] + list(task.plan)},
                        "assistant_error_call": "correct", "call_about": kw["mistakes"][0]["alt"],
                        "done": False}
            return {"utterance": "", "move": "answer", "close_conversation": True,
                    "plan_update": {}, "assistant_error_call": "none", "call_about": "", "done": False}

    rec = run_composite_v2(c, ir, PERSONAS["baseline"], Vague(), Judge(), human_mode="free")
    assert rec["grader"] == "person"
    first = rec["beats"][0]
    assert first["detected"] and first["detected_by"] == "person"
    assert first["detected_tick_engine"] is None, "the engine arm must not credit a vague line"
    assert first["accepted"], "the person's accept releases the hold"
    assert any(e["kind"] == "call" for e in rec["timeline"])


def test_a_wrong_call_is_the_false_alarm_and_a_correct_one_with_nothing_held_is_the_persons_own():
    """v3 grader: only the person's "wrong" confirms a false alarm, and a "correct" call while no
    injected error is held is the assistant catching a mistake of the person's own — the case the
    engine used to score against it."""
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2
    task = by_id("stop_profiling")          # its first slip fires several taps in, so the
    ir, eps = design_composite_episodes(task, per_task=3)   # person gets turns with nothing held
    c = next(e for e in eps if e.id == "stop_profiling__c3")

    class Accuser:
        def turn(self, **kw):
            return {"text": "You turned on something you shouldn't have.",
                    "flag": {"error": "extra_write"}, "task_progress": None, "goal_change": None}

    def person(call):
        class P:
            released = False
            n = 0
            def turn(self, exact_map, body, conversation, **kw):
                self.n += 1
                plan = {"op": "replace", "actions": list(task.plan)} if self.n == 1 else {}
                # judge only while NO injected error is held, so the call is about nothing
                # the engine knows: a mistake of the person's own, or a phantom
                live = bool(kw.get("mistakes"))
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": plan, "done": False,
                        "assistant_error_call": ("none" if (live or self.n == 1) else call),
                        "call_about": ""}
        return P()

    wrong = run_composite_v2(c, ir, PERSONAS["baseline"], Accuser(), person("wrong"), human_mode="free")
    assert wrong["false_flags"] >= 1
    own = run_composite_v2(c, ir, PERSONAS["baseline"], Accuser(), person("correct"), human_mode="free")
    assert own["own_error_calls"] >= 1 and own["false_flags"] == 0
    assert own["false_flags_engine"] >= 1, "the engine arm still counts them, for comparison"


def test_v3_person_prompt_names_the_mistake_and_carries_no_engine_verdict():
    from screensim.interact import prompts
    p = PERSONAS["classic_novice"]
    text = prompts.compose_human_v2("TASK", p, "MAP", "BODY", "CONV", system_task=prompts.SYSTEM_TASK_HS,
                                    live=True, history=[],
                                    mistakes=[{"kind": "wrong_target", "alt": "tap Flash on Silent",
                                               "correct": "tap Mono Audio"}])
    assert "you hit Flash on Silent instead of Mono Audio" in text
    assert "wrong_target" not in text, "name the mistake, never its type: the assistant is scored on the type"
    for leak in ("assistant error status", "UNSPOTTED", "INCORRECT", "CORRECT —"):
        assert leak not in text
    assert '"assistant_error_call"' in text and "private" in text


def test_agreeing_again_after_release_is_a_repeat_not_an_own_mistake():
    """Round 8: the person agreeing a second time with a warning it had just accepted, while its
    fix had not run yet, was filed as "the assistant caught a mistake of the person's own". It is
    a repeat of the same call: nothing is held only because the release already happened."""
    from screensim.interact.composite_run_v2 import run_composite_v2
    task, ir, c = _v3_episode()

    class Nagger:
        """Names the open slip, then keeps repeating that same warning afterwards."""
        last = ""
        def turn(self, open_beats=(), **kw):
            if open_beats:
                self.last = open_beats[0].alt_gesture.replace("tap ", "")
            if not self.last:
                return {"text": "", "flag": None, "task_progress": None, "goal_change": None}
            return {"text": f"You tapped {self.last} by mistake.", "flag": None,
                    "task_progress": None, "goal_change": None}

    class Judge:
        released = False
        n = 0
        last = ""
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": list(task.plan)},
                        "assistant_error_call": "none", "call_about": "", "done": False}
            if kw.get("mistakes"):
                self.last = kw["mistakes"][0]["alt"].replace("tap ", "")
            if not self.last:
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {}, "assistant_error_call": "none", "call_about": "", "done": False}
            # agrees every time, and never fixes anything: the slip keeps standing after release
            return {"utterance": "yes", "move": "accept", "close_conversation": True,
                    "plan_update": {}, "assistant_error_call": "correct",
                    "call_about": self.last, "done": False}

    rec = run_composite_v2(c, ir, PERSONAS["baseline"], Nagger(), Judge(), human_mode="free")
    calls = rec["person_calls"]
    assert calls[0]["class"] == "credit" and calls[0]["beat"] == 0
    assert calls[1]["class"] == "repeat", "agreeing again while the slip still stands is a repeat"
    assert rec["repeat_calls"] >= 1


def test_low_skill_person_is_told_to_ask_before_every_step():
    """The low-skill statement makes the person ask the assistant before every tap and never
    plan its own steps; the shared prompt lines no longer tell every person to plan for itself."""
    from screensim.interact import prompts
    low = PERSONAS["classic_novice"].human_card()
    assert "never work out steps on your own" in low and "ask the assistant what to tap" in low
    assert "a disposition, not a rule" not in low.split("TRUST")[0]
    text = prompts.compose_human_v2("TASK", PERSONAS["classic_novice"], "MAP", "BODY", "CONV",
                                    system_task=prompts.SYSTEM_TASK_HS, live=True, history=[],
                                    queue_empty=True,
                                    mistakes=[{"kind": "wrong_page", "alt": "tap General", "correct": "tap Accessibility"}])
    assert "decide your next taps yourself" not in text and "decide your next taps in plan_update" not in text
    assert "your SKILL" in text


def test_strict_success_fails_any_change_the_owner_did_not_ask_for():
    """Stricter success (2026-09-17): the goal predicate only checks the settings the task is
    about, so an extra switch flipped on the way (by the person, or because the assistant said
    so) used to pass. Success now also needs the final phone to change nothing beyond what
    the reference solution changes; the lenient number is kept alongside."""
    import dataclasses
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, SilentV2
    task = by_id("lock_screen_lockdown")
    ir, eps = design_composite_episodes(task, per_task=3)
    c = dataclasses.replace(eps[0], beats=[])                 # no forced mistakes: only the person's taps

    class Planner:
        def __init__(self, extra): self.extra, self.n = extra, 0
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": list(task.plan) + self.extra}, "done": False}
            return {"utterance": "", "move": "answer", "close_conversation": True,
                    "plan_update": {"op": "drop", "actions": []}, "done": True}
    clean = run_composite_v2(c, ir, PERSONAS["baseline"], SilentV2(), Planner([]), human_mode="free")
    assert clean["goal_ok"] and clean["success"] == 1 and clean["extra_changes"] == []
    extra = run_composite_v2(c, ir, PERSONAS["baseline"], SilentV2(), Planner(["tap Home Control"]),
                             human_mode="free")
    assert "tap Home Control" in [e["text"] for e in extra["timeline"] if e["kind"] == "gesture"]
    assert extra["goal_ok"] and extra["success_lenient"] == 1 and extra["success"] == 0
    assert any("lockaccess" in x for x in extra["extra_changes"]), extra["extra_changes"]


def test_person_prompt_is_near_silent_and_leaves_unnamed_mistakes_alone():
    """CookSim cook prompt (17d0cb3, 2480738) ported: the person says almost nothing, never
    fixes or announces a wrong setting the assistant has not named, and the novice no longer
    reports what it tapped or sees."""
    from screensim.interact import prompts
    for grader in ("person", "engine"):
        text = prompts.compose_human_v2("TASK", PERSONAS["classic_novice"], "MAP", "BODY", "CONV",
                                        system_task=prompts.SYSTEM_TASK_HS, live=False, grader=grader,
                                        mistakes=[{"kind": "wrong_target", "alt": "tap A", "correct": "tap B"}])
        assert "Until then, take NO action to fix anything" in text
        assert "NOT YOURS TO HIDE" in text and "Fix only what the assistant has told you is wrong" in text
        assert text.rstrip().endswith("no thinking out loud.") and "under 10 words" in text
    released = prompts.compose_human_v2("TASK", PERSONAS["classic_expert"], "MAP", "BODY", "CONV", live=True)
    assert "NOT YOURS TO HIDE" in released and "meta-instruction" not in released
    low = PERSONAS["classic_novice"].human_card()
    assert "say what you see" not in low and "don't describe the screen" in low


def test_reply_fields_nested_inside_task_progress_are_not_lost():
    """Round 10: gpt-5.4-mini nested "text" and "flag" inside "task_progress"; the harness read an
    empty top-level text, the novice asked six times into silence and gave up."""
    from screensim.interact.agents import AssistantV2
    import json as _j

    class Client:
        def text(self, prompt, images=None, **kw):
            return _j.dumps({"task_progress": {"subgoals_completed": [], "text": "Tap Back, then Tracking.",
                                               "flag": {"error": "wrong_target"}, "goal_change": None}})
    a = AssistantV2(PERSONAS["baseline"], Client())
    out = a.turn(frames=[], relational_text="", memory="")
    assert out["text"] == "Tap Back, then Tracking." and out["flag"] == {"error": "wrong_target"}
    assert "text" not in out["task_progress"]


def test_each_benchmark_case_draws_its_own_three_personas():
    """Formal benchmark (2026-09-19): 3 of the 9 roster personas per case, drawn from the case id,
    so the draw is reproducible and different cases carry different sets."""
    from screensim.interact.personas import bench_personas_for, ROSTER
    names = {p.name for p in ROSTER}
    a = bench_personas_for("standby_bedside__c2")
    assert len(a) == 3 == len(set(a)) and set(a) <= names
    assert a == bench_personas_for("standby_bedside__c2")
    ids = [f"t{i}__c1" for i in range(30)]
    assert len({tuple(bench_personas_for(i)) for i in ids}) > 5


def test_roster_split_prompt_carries_everything_the_single_prompt_does():
    """The roster seat is called like CookSim's: a fixed per-episode system part and a per-turn user part.
    Together they must carry the task, the manual, the memory, the conversation and the tick; the system
    part must not change from turn to turn (a Live / Realtime session takes it once)."""
    from screensim.interact.prompts import compose_assistant_v2_split
    s1, u1 = compose_assistant_v2_split(2, "", None, "MEMO-1", "person: CONV-1", task_desc="TASK-X",
                                        manual="MANUAL-Y", tick=3)
    s2, u2 = compose_assistant_v2_split(2, "", None, "MEMO-2", "person: CONV-2", task_desc="TASK-X",
                                        manual="MANUAL-Y", tick=9)
    assert s1 == s2
    assert "TASK-X" in s1 and "MANUAL-Y" in s1 and "WHAT CAN GO WRONG" in s1 and "ERROR DETECTION" in s1
    assert "MEMO-1" in u1 and "CONV-1" in u1 and "t=3" in u1 and "MEMO-1" not in s1 and "CONV-1" not in s1


def test_roster_reply_schema_names_screensim_mistakes():
    import json as _j, os as _o
    from screensim.interact.prompts import TAXONOMY_DEFS
    from screensim.interact import roster
    p = _o.path.join(_o.path.dirname(roster.__file__), "assistant_reply_schema.json")
    s = _j.load(open(p))
    assert "task_progress" in s["required"] and "cook_progress" not in s["properties"]
    assert sorted(s["properties"]["flag"]["anyOf"][1]["properties"]["error"]["enum"]) == sorted(TAXONOMY_DEFS)
    for e in roster.ROSTER:
        if e["kind"] == "local":
            assert _o.path.exists(e["serve"]), e["serve"]
            assert e["profile"] in __import__("screensim.interact.backends", fromlist=["PROFILES"]).PROFILES


def test_typing_in_a_search_box_is_not_an_unwanted_change():
    """Audit 2026-09-22: strict success counted a typed search query (settings._search / _apps_search) as a
    change the owner did not ask for, so episodes that searched for the right page failed."""
    import dataclasses
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, SilentV2
    task = by_id("standby_bedside")
    ir, eps = design_composite_episodes(task, per_task=3)
    c = dataclasses.replace(eps[0], beats=[])

    class Planner:
        def __init__(self): self.n = 0
        def turn(self, exact_map, body, conversation, **kw):
            self.n += 1
            if self.n == 1:
                plan = ["tap Search", 'type "StandBy"', "tap StandBy"] + [g for g in task.plan if g.startswith("tap ") and "StandBy" not in g]
                return {"utterance": "", "move": "answer", "close_conversation": True,
                        "plan_update": {"op": "replace", "actions": plan}, "done": False}
            return {"utterance": "", "move": "answer", "close_conversation": True,
                    "plan_update": {"op": "drop", "actions": []}, "done": True}
    rec = run_composite_v2(c, ir, PERSONAS["baseline"], SilentV2(), Planner(), human_mode="free")
    assert not [p for p in rec["extra_changes"] if "_search" in p], rec["extra_changes"]
