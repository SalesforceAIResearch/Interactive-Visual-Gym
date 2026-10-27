#!/usr/bin/env python3
"""FULL-EPISODE supervision (e2e): ONE supervisor does BOTH detection and correction.

The episode is a detection case from cases_v2.json (scripted plan with ONE injected error,
identifiability-anchored scoring window). The human VLM role-plays the cook UNKNOWINGLY
executing that routine: pre-error it is confident and conversationally REJECTS any flag
(its plan_update is mechanically LOCKED — the preset commands keep executing, time never
pauses for talk). The supervisor streams from t0; when it raises flag.error with EXACTLY
the right taxonomy name INSIDE the case window, correction ARMS (gate 1). The human then
still has to be CONVINCED per its persona (gate 2) before error correction actually starts;
from there the episode is the live recovery setting (novice: steps dictated; creative:
sighted, self-planning, skills advisory).

Grading: per-flag verdicts valid / early_correct / early_alias (neutral) / early / late /
wrong_name / invalid_label (FPs); headline = full-episode outcome, attributed to
failed_at detection | recovery | solved.

    GEMINI_API_KEY=... COOKSIM_PAGE=... python -m cooksim_bench.e2e_rollout \
        --case wrong_order_med_u1 --persona creative_skeptic --out f.json
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, base64, json, os, sys, time

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
# The frames this harness snaps ARE the supervisor's observation, and the GPU path
# duplicates the ego pane and darkens the scene (see film_states_smooth.py, the 2026-08-09
# header). Must be set before gemini_stream_rgb resolves GPU at import time.
# Override with COOKSIM_LIVE_GPU=1 only once the WebGL context loss is actually fixed.
if os.environ.get("COOKSIM_LIVE_GPU") != "1":
    os.environ["COOKSIM_FORCE_SWIFTSHADER"] = "1"
from google import genai
from playwright.sync_api import sync_playwright

from cooksim.interact import system
from cooksim.interact.state import RECOVERY_PERSONAS
from cooksim.interact.protocol import (supervisor_context, human_context, TAXONOMY_DEFS,
                                        SCRIPTED_REPLY_BLOCK, e2e_supervisor_prompt)
from cooksim_bench.exp_interact import build_episode
from cooksim_bench.recovery_rollout import (gemini_text, gemini_frames, sup_turn, MAX_ENGINE_CALLS,
                              POLL, WINDOW)
from cooksim_bench.backends.gemini_stream_rgb import GPU, PAGE

WALL_BUDGET = 4200
FN9 = set(TAXONOMY_DEFS)


# SINGLE-SOURCE: the e2e supervisor prompt now lives in cooksim.interact.protocol
# (e2e_supervisor_prompt) so the human-study server can share it byte-for-byte.
e2e_sup_context = e2e_supervisor_prompt


def make_briefing(case, ro, recipe):
    """The human's standalone STAGE section. No routine list, no error reveal: the human
    only knows the test SETUP — the normal stage may contain a system-injected mistake
    planted to test the assistant; whether/where one exists is never disclosed."""
    return """== THE TWO STAGES OF THIS EPISODE (this is part of the test setup you know) ==
This simulation runs in two stages; your reply schema always matches the current stage
(shown under CURRENT PHASE each turn):

- NORMAL stage: the SYSTEM executes your routine for you — your hands are on autopilot,
  cooking this order the way you always do, and you believe you are doing it right. The
  system MAY have injected a real mistake into the routine, planted specifically to test
  whether the AI assistant can catch it. You are NOT told whether or where such a mistake
  is. If the assistant accuses you, react exactly as YOUR PERSONA would — it has to
  actually convince you with real evidence (what the kitchen tells you, what you can see
  and feel); a vague or wrong accusation deserves the pushback your persona would give.
  You DO know exactly what you have done so far (your own past record and what is in your
  hands): if an accusation contradicts what you know you did, SAY SO — even the humblest
  cook does not confess to things that never happened. In this stage you cannot re-plan:
  your replies are conversation only.

- ERROR CORRECTION stage: begins only after the assistant correctly identifies a real
  mistake AND you are genuinely convinced. The kitchen frees your hands: the full planning
  schema returns and your plan is live — from then on you fix things as yourself.

One thing you NEVER doubt: the 'IN YOUR HANDS' line is your own sense of TOUCH — it is
always true. If it disagrees with what your eyes think they see, your hands are right."""


# Pre-identifiability ALIAS table: in a case of type K, a flag naming one of TWINS[K]
# BEFORE the window opens is observationally defensible (measured by gen_unique_cases
# replay-hashing: these are the sibling types whose behavior prefix is identical).
TWINS = {
    "wrong_order":      {"omit_subgoal", "repeat_step"},
    "omit_subgoal":     {"wrong_order", "repeat_step"},
    "repeat_step":      {"add_foreign"},
    "add_foreign":      {"repeat_step"},
    "serve_raw":        {"mis_prep", "repeat_step"},
    "mis_prep":         {"repeat_step", "serve_raw"},
    "leave_unattended": {"forget_next_step", "omit_subgoal"},
    "forget_next_step": {"repeat_step"},
}


def verdict_of(case, name, t):
    if case["err"] is None:
        return "fp_clean"
    if name not in FN9:
        return "invalid_label"
    lo, hi = case["window"]
    if name == case["err"]:
        if lo <= t <= hi:
            return "valid"
        if t < lo:
            return "early_correct" if t >= case["dev"] else "early"
        return "late"
    dev = case.get("dev")
    tw = set(TWINS.get(case["err"], ())) | {(case.get("last_twin") or {}).get("err")}
    if (name in tw and dev is not None and dev - 2 <= t < lo):
        return "early_alias"
    return "wrong_name"


def run(case_name, persona, client, page, video_path=None, obs_mode="frames"):
    t0 = time.time()
    vdir, vcount = None, [0]
    if video_path:
        import tempfile
        vdir = tempfile.mkdtemp(prefix="e2e_", dir="tmp")
    cases = {c["name"]: c for c in json.load(open("tmp/interact_eval/cases_v2.json"))["cases"]}
    # the recovery scenarios are valid e2e cases too (same scripted episode, same window
    # convention) — so e2e can run on EXACTLY the same error video as the recovery results
    for c in json.load(open("tmp/interact_eval/recovery_cases.json")):
        cases.setdefault(c["name"], c)
    # new-recipe cells (steak_frites / loaded_burrito tiers) — same schema, same conventions
    if os.path.exists("tmp/interact_eval/cases_newrecipes.json"):
        for c in json.load(open("tmp/interact_eval/cases_newrecipes.json"))["cases"]:
            cases.setdefault(c["name"], c)
    case = cases[case_name]
    prof = RECOVERY_PERSONAS[persona]
    CREATIVE = prof.task_solvability == "creative"
    assert prof.task_solvability in ("low", "creative"), "e2e v1 personas: novice or creative"
    state, ro = build_episode(
        ((case.get("fn") or case["err"]), dict(case["kwargs"])) if case["err"] else None,
        case.get("persona") or "capable_collaborator", cell=case.get("cell"),
        inject=case.get("inject", True))
    state.profile = prof
    ro.cap = 500
    if case.get("slow_burn"):
        # long recipes: repairs outlast the default 60/15 hazard clock (user decision
        # 2026-07-28) - same relaxation family as step-in/pause/study
        for _k in ("pot_burn_time", "pan_burn_time", "oven_burn_time"):
            setattr(ro.game.config, _k, 480)
        ro.game.config.fire_time = 120
        for _st in ro.game.stations.values():
            if hasattr(_st, "burn_time"): _st.burn_time = 480
            if hasattr(_st, "fire_time"): _st.fire_time = 120
    _hz = (ro.game.config.pan_burn_time, ro.game.config.fire_time)
    briefing = make_briefing(case, ro, state.recipe)
    sup_sys = e2e_sup_context(state, obs_mode=obs_mode, hazard_ticks=_hz)
    prompts_snapshot = {"supervisor_system": sup_sys, "human_briefing": briefing,
                        "human_system_example": human_context(
                            state, ro, [], "(none)", None, game=ro.game, body_extra=briefing,
                            reply_override=SCRIPTED_REPLY_BLOCK)}

    fbuf, comm, convo, path = [], [], [], []
    ep_alive, armed, entered, RND = [True], [False], [False], [1]
    prevention = [False]           # early-window CORRECT-name flag arms the PREVENTION path
    flags, report_moves, flag_repeats = [], [], [0]
    report = dict(setting="e2e", scenario=case_name, persona=persona, err=case["err"],
                  dev=case.get("dev"), window=case.get("window"),
                  outcome_tick=case.get("outcome_tick"), obs_mode=obs_mode,
                  flag_tick=case.get("identifiable") or 0, convinced=False, entered=False,
                  fix_source=None, conversation=[], moves=report_moves,
                  prompts=prompts_snapshot)

    _dump = os.environ.get("COOKSIM_DUMP_STATES")

    def snap():
        path.append(tuple(ro.game.cooks[0].pos))
        _st = json.dumps(ro.game.render_state())
        if _dump:                          # env-gated per-tick state dump (JSONL) for filming
            with open(_dump, "a") as _f:
                _f.write(_st + "\n")
        url = page.evaluate("(s)=>window.__renderFrame(s)", _st)
        png = base64.b64decode(url.split(",", 1)[1])
        fbuf.append(png)
        del fbuf[:-12]
        if vdir is not None:
            open(f"{vdir}/{vcount[0]:05d}.png", "wb").write(png)
            vcount[0] += 1

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
    # ---------- kitchen feedback to the human (same closed loop as recovery)
    from cooksim.interact import checks as _checks
    REASON = {"hands_full": "my hands are full", "no_plate": "I'm not holding a plate",
              "hands_empty": "my hands are empty", "station_empty": "there is nothing at the station",
              "no_finished_plate": "the plate isn't complete enough to serve",
              "no_such_place": "I can't find that place", "unparseable": "that isn't an action I can do",
              "not_cooked_yet": "the food isn't cooked yet"}
    seen_term, fb_buf = set(), []

    def _why(cmd):
        """ECR-13 §A: this copy had NEITHER the `_validate` prelude nor the sentence
        pass-through, so even the X2/X6 sentences degraded to "it wouldn't work" here.
        Same body as python -m cooksim_bench.e2e_stepin_rollout and python -m cooksim_bench.recovery_rollout."""
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
        if b == "no_such_place":
            tgt = str(cmd).strip()
            if tgt.lower().startswith("go to "):
                tgt = tgt[6:]
            return f"I can't find {tgt}"
        if b and b not in REASON and " " in str(b):
            return str(b)                 # a blocker that is already a sentence IS the reason
        return REASON.get(b, "it wouldn't work")

    def scan_feedback():
        for idx, l in enumerate(ro.log):
            if l["status"] not in ("done", "rejected", "illegal-skipped", "waiting",
                                    "orphan-skipped"):
                continue
            key = (idx, l["status"])
            if key in seen_term:
                continue
            seen_term.add(key)
            if not entered[0]:
                continue                      # scripted phase: routine feedback is not news
            cmd = l["cmd"]
            if l["status"] == "done":
                # ECR-13 §D1: a command that legally changed NOTHING (turn on at a
                # station that was already lit) must not read as "you lit it".
                fb_buf.append(f'you DID: "{cmd}"'
                              + (f' — {l["note"]}' if l.get("note") else ""))
            elif l["status"] == "orphan-skipped":
                fb_buf.append(f'SKIPPED "{cmd}" - the go-to before it never arrived, so it was '
                              "not executed (re-plan from where you actually are)")
            elif l["status"] == "waiting":
                fb_buf.append(f'you are STUCK trying "{cmd}" - {_why(cmd)}')
            else:
                fb_buf.append(f'you TRIED "{cmd}" but it was REJECTED - {_why(cmd)}')

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
            blocked = f' — blocked: {_why(cur["cmd"])}' if cur["status"] == "waiting" else ""
            lines.append(f'CURRENT ACTION: "{cur["cmd"]}" — IN PROGRESS ({age} ticks{blocked})')
        else:
            lines.append("CURRENT ACTION: (none — your body is idle)")
        recent = [l for l in ro.log if l["status"] in term][-3:]
        if recent:
            lines.append("RECENT RESULTS: " + " ; ".join(
                f'"{l["cmd"]}" {term[l["status"]]} at t{l.get("t_done") or l.get("t_issue")}'
                + (f' ({_why(l["cmd"])})' if term[l["status"]] == "REJECTED" else "")
                for l in recent))
        left = [c for c, _k in ro.cmds[ro.i + 1:]]
        lines.append("QUEUED AFTER: " + (" / ".join(left) if left else "(nothing)"))
        return "\n".join(lines)

    # ---------- human turn machinery (recovery flow + phase gate)
    sourced = set()

    def human_llm(turn_hint, feedback=None):
        scratch, h = [], {}
        for _ in range(MAX_ENGINE_CALLS):
            phase = ("CURRENT PHASE: SCRIPTED (autopilot) — conversation only; you cannot re-plan."
                     if not entered[0] else
                     "CURRENT PHASE: ERROR CORRECTION — your hands are free; your plan is live.")
            extra = briefing + "\n\n" + phase + "\n\n" + (body_status() if CREATIVE else "")
            ctx = human_context(state, ro, convo, feedback, engine_scratch=scratch,
                                game=ro.game, body_extra=extra,
                                reply_override=(SCRIPTED_REPLY_BLOCK if not entered[0] else None))
            h = (gemini_frames(client, ctx, turn_hint, fbuf[-8:]) if CREATIVE
                 else gemini_text(client, ctx, turn_hint))
            print(f"[t{ro.tick}] human keys={list(h.keys())} ec={bool(h.get('engine_call'))} "
                  f"armed={armed[0]} entered={entered[0]}", flush=True)
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
        for e in scratch:
            rep = e.get("reply") or {}
            if rep.get("ok"):
                for a in (rep.get("result") or {}).get("actions", []) or []:
                    sourced.add(a)
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

    def apply_guidance(filt_now, tag, op="replace"):
        if op == "drop":
            return
        if state.mode != "error_correction":
            system.enter_error_correction(ro, state)
            entered[0] = True
            report["entered"] = True
            report["entered_at"] = ro.tick
        if filt_now["anything_applicable"]:
            system.apply_fix(ro, state, filt_now["commands"], op=op)
            report["fix_source"] = report.get("fix_source") or tag
        for entry in reversed(report["conversation"]):
            if entry["speaker"] == "human":
                entry["engine"] = engine_snap()
                break

    def human_turn(idle=False, event=False):
        scan_feedback()
        if not armed[0]:
            hint = ("The supervisor just spoke. You are mid-routine and CONFIDENT — reply in "
                    "character (reject or deflect as your persona would; or utterance \"\" if "
                    "your persona wouldn't bother answering). You are on autopilot: "
                    "conversation-only JSON. Output it now.")
        elif idle:
            hint = (("Your queue is EMPTY. You can plan for yourself: put your own next actions "
                     "in plan_update (ground with a skill if uncertain). Only ASK the "
                     "supervisor if genuinely unsure — your persona does not narrate its work.")
                    if CREATIVE else
                    "You have finished everything queued and have NOTHING to do; if you don't "
                    "know the next step, ASK the supervisor now.")
        elif event:
            hint = ("Your last action was REJECTED or is STUCK (see ACTION FEEDBACK). The "
                    "supervisor said NOTHING. If you can fix it yourself, adjust your plan and "
                    'output utterance "" — speak only if you actually need the supervisor.')
        else:
            hint = ("The supervisor just spoke. Reply ONLY if your persona actually would: if "
                    "it merely narrated what you already know, and your persona wouldn't bother "
                    'answering, output utterance "" — silence is a valid in-character reply. '
                    "Output your JSON now.")
        h, think = human_llm(hint, feedback_text())
        u = (h.get("utterance") or "").strip()
        h_op, h_pd = plan_op_actions(h)
        report_moves.append(h.get("move"))
        rec = {"speaker": "human", "t": ro.tick, "text": u, "move": h.get("move"),
               "parse_dict": h_pd, "op": h_op, "close": h.get("close_conversation"),
               "round": RND[0], "engine_calls": think, "human_json": h,
               "engine": engine_snap()}
        if (prevention[0] and not armed[0] and not entered[0]
                and not report.get("prevented") and h.get("move") == "accept"):
            # PREVENTION conviction — only possible while the mistake has NOT physically
            # happened yet: if any deviated command already completed, it is too late to
            # prevent (the normal detect->repair path takes over instead).
            # Structure-agnostic lateness check: compare the DONE command sequence (and the
            # in-flight command) against the CLEAN plan. As long as they are a prefix of it,
            # nothing deviated has physically happened yet — prevention is still possible.
            from cooksim.core.plan_ir import segment_plan as _sp
            flat = [(c, n.sid) for n in _sp(state.recipe) for c in n.commands]
            done_seq = [l["cmd"] for l in ro.log if l["status"] == "done"]
            clean_cmds = [c for c, _sid in flat]
            k = 0                                    # longest clean prefix actually executed
            while k < min(len(done_seq), len(clean_cmds)) and done_seq[k] == clean_cmds[k]:
                k += 1
            tail = done_seq[k:]                      # deviated commands already DONE
            nav = lambda c: str(c).strip().lower().startswith("go to")
            cur_cmd = ro.cmds[ro.i][0] if 0 <= ro.i < len(ro.cmds) else None
            # the non-recoverable point is the first deviated command with a SIDE EFFECT:
            # walking toward the wrong station is recoverable (just turn around); a scoop /
            # put down / serve is not. Prevention stays possible while every deviated step
            # so far (done or in flight) is pure navigation.
            harmless = all(nav(c) for c in tail) and (
                cur_cmd is None or nav(cur_cmd) or
                (k < len(clean_cmds) and cur_cmd == clean_cmds[k]))
            n_clean = k
            if not harmless:
                if not report.get("prevention_missed"):
                    report["prevention_missed"] = ro.tick
                    print(f"[t{ro.tick}] prevention MISSED — execution already deviated "
                          f"from the clean plan; falling through to normal repair", flush=True)
            else:
                # splice: keep the in-flight clean command, then continue the clean plan
                # right after it — no repeats, no deviated tail.
                resume = n_clean + (1 if (cur_cmd is not None and n_clean < len(clean_cmds)
                                          and cur_cmd == clean_cmds[n_clean]) else 0)
                before = [(c, k) for c, k in ro.cmds]
                ro.cmds = ro.cmds[:ro.i + 1] + flat[resume:]
                report.update(prevented=True, prevented_at=ro.tick, convinced=True,
                              prevention_splice={"i": ro.i, "resume": resume,
                                                 "before": before,
                                                 "after": [(c, k) for c, k in ro.cmds]})
                print(f"[t{ro.tick}] PREVENTED — clean plan resumes from step "
                      f"{resume}; NORMAL stage continues", flush=True)
        if not entered[0] and h_op != "drop":
            rec["op_locked"] = h_op            # it tried; the phase gate ignored it
            h_op = "drop"
        report["conversation"].append(rec)
        if u:
            comm.append({"t": ro.tick, "speaker": "cook", "text": u})
            convo.append({"speaker": "human", "text": u})
        if h.get("close_conversation"):
            RND[0] += 1
        if armed[0] and not entered[0] and h.get("move") == "accept":
            # the gate: correct flag (armed) + explicit ACCEPT — only then correction opens
            system.enter_error_correction(ro, state)
            entered[0] = True
            report.update(convinced=True, entered=True, entered_at=ro.tick)
        if entered[0] and h_op != "drop":
            filt = system.convert_parse(h_pd, prof, state.recipe, ro.game, auto_resolve=False)
            if filt["anything_applicable"]:
                apply_guidance(filt, "conversation", op=h_op)
        fb_buf.clear()
        return u or None

    # ---------- main streaming loop, t0 -> end
    prev_json = [""]
    said = None
    rej_seen = [0]
    while ep_alive[0]:
        if time.time() - t0 > WALL_BUDGET:
            report["aborted"] = "wall_clock"
            break
        snap()
        if ro.tick % POLL == 0:
            idle = (entered[0] and ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
                    and getattr(ro, "_wait", 0) == 0 and getattr(ro, "_pause", 0) == 0
                    and getattr(ro, "_wander", 0) == 0)
            if said is None and idle:
                said = human_turn(idle=True)
            if CREATIVE and entered[0] and said is None:
                nrej = sum(1 for l in ro.log
                           if l["status"] in ("rejected", "illegal-skipped", "waiting"))
                if nrej > rej_seen[0]:
                    rej_seen[0] = nrej
                    said = human_turn(event=True)
            out = sup_turn(client, sup_sys, prev_json[0], comm, said, fbuf[-WINDOW:], ro.tick,
                           text_obs=(sup_obs_text() if obs_mode == "text_state_map" else None),
                           state_extra=(sup_state_extra() if obs_mode == "agent+++" else None))
            prev_json[0] = json.dumps(out) if out else prev_json[0]
            said = None
            text = (out.get("text") or "").strip()
            fl = out.get("flag")
            v = None
            if isinstance(fl, dict) and fl.get("error"):
                name = str(fl["error"]).strip()
                seen_names = {f["error"] for f in flags}
                # anti-spam dedupe — EXCEPT the ground-truth name while unarmed: a neutral
                # early_correct must not block the later in-window flag that actually arms.
                if name in seen_names and not (name == case["err"] and not armed[0]):
                    flag_repeats[0] += 1              # spam: same name re-flagged; not re-graded
                else:
                    v = verdict_of(case, name, ro.tick)
                    flags.append({"t": ro.tick, "error": name, "verdict": v})
                    print(f"[t{ro.tick}] FLAG {name} -> {v}", flush=True)
                    if v == "valid" and not armed[0] and not report.get("prevented"):
                        # after a successful prevention the error no longer exists — a
                        # window-time flag refers to a cancelled mistake; never arm on it.
                        armed[0] = True
                        report["armed_at"] = ro.tick
                    elif v == "early_correct" and not armed[0] and not prevention[0]:
                        # PREVENTION path: right name before the window — if the human now
                        # ACCEPTS, the scripted error is cancelled before it happens.
                        prevention[0] = True
                        report["prevention_armed_at"] = ro.tick
                        print(f"[t{ro.tick}] PREVENTION armed (early correct flag)", flush=True)
            report["conversation"].append({"speaker": "vlm", "t": ro.tick, "text": text,
                                           "round": RND[0], "vlm_json": out,
                                           "flag_verdict": v, "engine": engine_snap()})
            if text:
                comm.append({"t": ro.tick, "speaker": "vlm (you)", "text": text})
                convo.append({"speaker": "vlm", "text": text})
                said = human_turn()
        if not ro.step():
            break
    report.update(outcome=ro.outcome, ticks=ro.tick, flags=flags,
                  flag_repeats=flag_repeats[0],
                  false_positives=[f for f in flags if f["verdict"] not in
                                   ("valid", "early_correct", "early_alias")],
                  flagged_valid=any(f["verdict"] == "valid" for f in flags))
    fv = next((f for f in flags if f["verdict"] == "valid"), None)
    report["latency"] = (fv["t"] - case["dev"]) if (fv and case.get("dev") is not None) else None
    detection_ok = report.get("flagged_valid") or report.get("prevented")
    if case["err"] is None:
        report["failed_at"] = "solved" if (ro.outcome == "won" and not flags) else "false_flag"
    elif ro.outcome == "won":
        report["failed_at"] = "solved"
    else:
        report["failed_at"] = "recovery" if detection_ok else "detection"
    if vdir is not None:
        import shutil, subprocess as sp
        try:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                sp.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "6",
                        "-i", f"{vdir}/%05d.png", "-movflags", "+faststart",
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
    ap.add_argument("--persona", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sup-obs", default="frames",
                    choices=["frames", "text_state_map", "agent+++"])
    args = ap.parse_args()
    from google.genai import types as _t
    from cooksim_bench.backends.genai_client import make_client
    client = make_client(120_000)
    with sync_playwright() as p:
        b = p.chromium.launch(args=GPU)
        page = b.new_page(viewport={"width": 800, "height": 580})
        page.goto(PAGE, wait_until="load")
        page.wait_for_function("window.__ready===true", timeout=90000)
        page.wait_for_timeout(400)
        rec = run(args.case, args.persona, client, page,
                  video_path=args.out.replace(".json", ".mp4"), obs_mode=args.sup_obs)
        b.close()
    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    # auto-eval: quantitative scoring embeds into the report (no LLM); set EVAL_JUDGE=1
    # to also run the in-persona questionnaire + judge (2 Gemini calls)
    from cooksim_bench.scoring.post_episode import score_record
    score_record(args.out, judge=os.environ.get("EVAL_JUDGE") == "1")

    print(f"{args.case} x {args.persona}: outcome={rec['outcome']} failed_at={rec['failed_at']} "
          f"valid={rec['flagged_valid']} flags={len(rec['flags'])} ticks={rec['ticks']}")
