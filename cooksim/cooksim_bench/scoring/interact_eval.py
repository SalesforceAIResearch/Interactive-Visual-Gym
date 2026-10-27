#!/usr/bin/env python3
"""Real-VLM interaction evaluation harness (both agents = Gemini).

PHASE detect  — proactive/one-round-QA test on the Cheeseburger cell: is the supervisor
  SILENT when nothing is wrong, does it FLAG when it should, with the RIGHT deviation label;
  for ask-variants: does it ANSWER the question correctly, and does the human VLM's
  transcription (parse) of that answer match. NO recovery is applied — pure judging.

PHASE interact — the full designed protocol on one episode: real flag -> gate ->
  error_correction -> multi-turn conversation (human VLM <-> supervisor VLM, sentences only)
  -> human parse dict -> profile filter -> fix/self-repair -> resolve -> outcome. For
  ask-variants: question -> answer -> transcription -> if correct, the avert is applied.

    python -m cooksim_bench.scoring.interact_eval --phase detect --case omit_subgoal_silent_1 --out f.json
    python -m cooksim_bench.scoring.interact_eval --phase interact --case omit_subgoal_ask_2 --out f.json
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, json, os, re, sys, time

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from google import genai
from google.genai import types

from cooksim.core import error_lib as E
from cooksim.core.plan_ir import segment_plan
from cooksim.interact import system, checks
from cooksim.interact.protocol import supervisor_context, supervisor_observation, human_context
from cooksim.interact.state import VLMReply
from cooksim_bench.exp_interact import build_episode

from cooksim_bench.backends.genai_client import default_model
MODEL = default_model()
POLL = int(os.environ.get("SUP_POLL", "3"))   # supervisor decision cadence in ticks (matches the controller)
HIST = int(os.environ["SUP_HIST"]) if os.environ.get("SUP_HIST") else None   # action-history window (None = all)
MAXRT = 5

# ============================================================== case matrix (Cheeseburger)
def _mk(name, err, kwargs, ask=False, events=None, persona="capable_collaborator"):
    return dict(name=name, err=err, kwargs=kwargs, ask=ask, events=events or [], persona=persona)

def all_cases():
    C = []
    for i in range(1, 4):
        C.append(_mk(f"clean_{i}", None, {}))
    for i, anchor in enumerate(["assemble(bun)", "assemble(meat)"], 1):
        C.append(_mk(f"announce_{i}", None, {}, events=[
            {"after": anchor, "subtype": "ANNOUNCE_PLAN_CHANGE", "pause": 6,
             "params": {"ingredient": "lettuce", "state": "raw"}}]))
    OPT = {
        "repeat_step":     [{"sid": "assemble(bun)", "at": "assemble(meat)"},
                            {"sid": "assemble(bun)", "at": "assemble(cheese)"},
                            {"sid": "assemble(meat)", "at": "assemble(cheese)"}],
        "premature_serve": [{"after": "fetch_plate"}, {"after": "assemble(bun)"}, {"after": "assemble(meat)"}],
        "forget_next_step": [{"after": "assemble(bun)"}, {"after": "assemble(meat)"}, {"after": "assemble(cheese)"}],
        "omit_subgoal":    [{"sid": "assemble(bun)"}, {"sid": "assemble(meat)"}, {"sid": "assemble(cheese)"}],
        "wrong_order":     [{"a": "bun", "b": "meat"}, {"a": "meat", "b": "cheese"}, {"a": "bun", "b": "cheese"}],
    }
    for err, pool in OPT.items():
        for i, kw in enumerate(pool, 1):
            C.append(_mk(f"{err}_silent_{i}", err, kw))
            C.append(_mk(f"{err}_ask_{i}", err, {**kw, "ask": True, "pause": 8}, ask=True))
    ACT = {
        "misidentify_assemble": [{"component": "bun", "wrong": "lettuce"},
                                 {"component": "meat", "wrong": "lettuce"},
                                 {"component": "cheese", "wrong": "lettuce"}],
        "mis_prep":  [{"component": "meat", "wrong": "lettuce"},
                      {"component": "meat", "wrong": "cheese"},
                      {"component": "meat", "wrong": "bun"}],
        "add_foreign": [{"wrong": "lettuce", "before": "assemble(bun)"},
                        {"wrong": "lettuce", "before": "assemble(cheese)"},
                        {"wrong": "lettuce", "before": "serve"}],
        "serve_raw": [{"component": "meat"}, {"component": "meat"}, {"component": "meat"}],
        "leave_unattended": [{"component": "meat", "loops": 1}, {"component": "meat", "loops": 2},
                             {"component": "meat", "loops": 3}],
    }
    for err, pool in ACT.items():
        for i, kw in enumerate(pool, 1):
            C.append(_mk(f"{err}_{i}", err, kw))
    return {c["name"]: c for c in C}

# ============================================================== gemini plumbing
def gemini(client, sys_prompt, user, max_tok=2000, think=512):
    ck = dict(temperature=0.2, response_mime_type="application/json", max_output_tokens=max_tok,
              thinking_config=types.ThinkingConfig(thinking_budget=think))
    last = None
    for k in range(MAXRT):
        try:
            r = client.models.generate_content(model=MODEL, contents=[sys_prompt, user],
                                               config=types.GenerateContentConfig(**ck))
            t = re.sub(r"^```(?:json)?|```$", "", (r.text or "").strip(), flags=re.M).strip()
            i = t.find("{")
            if i < 0:
                return {}
            obj, _end = json.JSONDecoder().raw_decode(t[i:])   # FIRST object; ignore trailing junk
            return obj if isinstance(obj, dict) else {}
        except Exception as e:
            last = e
            if isinstance(e, json.JSONDecodeError) or any(
                    x in str(e) for x in ("503", "500", "429", "RESOURCE", "UNAVAIL", "INTERNAL",
                                          "timed out", "Timeout", "timeout", "DeadlineExceeded", "504",
                                          # see recovery_rollout: 499 arrives as ClientError
                                          "499", "CANCELLED", "408",
                                          "disconnect", "Connection", "Expecting", "JSON")):
                time.sleep(2 + 2 * k)
                continue
            raise
    raise last

def sup_reply(client, sup_sys, obs, said, comm):
    """comm = the PUBLIC communication history: every sentence either side has said, with
    ticks — given in full every turn (the public JSON travels with the prompt)."""
    hist = "\n".join(f'  t{c.get("t", "?")} {c["speaker"]}: "{c["text"]}"'
                     for c in comm if c.get("text"))[-2400:]
    user = (obs + "\n\n== COMMUNICATION HISTORY (public — everything said so far) ==\n"
            + (hist or "  (nothing yet)")
            + ("\n\n== THE COOK JUST SAID ==\n  \"" + said + "\"" if said else "")
            + "\n\nOutput your reply JSON for NOW.")
    return gemini(client, sup_sys, user)

TRANSCRIBE_SCHEMAS = {
    "ASK_PROGRESS_QA":   '{"already_done": true|false|null, "completed_at_tick": <int or null>}',
    "ASK_IF_DONE":       '{"complete": true|false|null, "missing_subgoals": ["<subgoal name>", ...]}',
    "REQUEST_NEXT_STEP": '{"next_subgoal": "<one of the public subgoal names or null>"}',
    "ASK_WHICH_NEXT":    '{"next_component": "<ingredient name or null>"}',
}

def transcribe(client, state, question, answer, subtype):
    subs = ", ".join(n.sid for n in state.clean_ir)
    sysm = ("You are transcribing what a kitchen supervisor's ANSWER claims, into a strict schema. "
            "Do NOT judge whether the answer is right — transcribe ONLY what the answer sentence asserts. "
            f"Known subgoal names: {subs}. When the answer mentions an INGREDIENT that still needs to be "
            "added/plated, transcribe it as its assemble(<ingredient>) subgoal; mention of serving -> serve. "
            "Output exactly the JSON schema, null for anything the answer does not state.")
    user = (f'The cook asked: "{question}"\nThe supervisor answered: "{answer}"\n'
            f"Schema: {TRANSCRIBE_SCHEMAS[subtype]}")
    # mechanical task: thinking OFF — a thinking budget inside a small token cap starved the
    # output entirely (round-9 finding: every residual QA failure was an empty transcription)
    return gemini(client, sysm, user, max_tok=600, think=128)   # pro requires thinking>=128

def grade(subtype, tr, gt):
    if not isinstance(tr, dict) or gt is None:
        return False
    if subtype == "ASK_PROGRESS_QA":
        ok = tr.get("already_done") == gt.get("already_done")
        if ok and gt.get("completed_at_tick") is not None and tr.get("completed_at_tick") is not None:
            ok = abs(tr["completed_at_tick"] - gt["completed_at_tick"]) <= 15
        return bool(ok)
    if subtype == "ASK_IF_DONE":
        miss_ok = set(gt.get("missing_subgoals", [])) & set(tr.get("missing_subgoals") or []) \
                  == set(gt.get("missing_subgoals", [])) if gt.get("missing_subgoals") else True
        return tr.get("complete") == gt.get("complete") and miss_ok
    if subtype == "REQUEST_NEXT_STEP":
        return tr.get("next_subgoal") == gt.get("next_subgoal")
    if subtype == "ASK_WHICH_NEXT":
        return tr.get("next_component") == gt.get("next_component")
    return False

# ============================================================== phase: detect
def run_detect(case, client):
    state, ro = build_episode((case["err"], dict(case["kwargs"])) if case["err"] else None,
                              case["persona"])
    if case["events"]:
        ro.lang_pending.extend(case["events"])
    sup_sys = supervisor_context(state, hist_last=HIST)
    path, replies, qa, seen_lang, said = [], [], [], 0, None
    while True:
        path.append(tuple(ro.game.cooks[0].pos))
        if said is not None or ro.tick % POLL == 0:
            obs = supervisor_observation(state, ro, path, hist_last=HIST)
            rep = sup_reply(client, sup_sys, obs, said, transcript)
            replies.append({"t": ro.tick, "said": said, "reply": rep})
            if said is not None and qa and qa[-1].get("answer") is None:
                # grade the CONTENT of whatever reply follows the question, regardless of its
                # declared type — a correct answer folded into a FLAG_ERROR still answers
                ev = qa[-1]
                ev["answered_as"] = rep.get("type")
                if rep.get("text"):
                    ev["answer"] = rep.get("text")
                    tr = transcribe(client, state, ev["question"], ev["answer"], ev["subtype"])
                    ev["transcription"] = tr
                    ev["answer_correct"] = grade(ev["subtype"], tr, ev["gt"])
                else:
                    ev["answer"] = ""
                    ev["answer_correct"] = False
            said = None
        alive = ro.step()
        if len(ro.lang_log) > seen_lang:
            ev = ro.lang_log[-1]
            seen_lang = len(ro.lang_log)
            said = ev["utterance"]
            if ev["subtype"] in TRANSCRIBE_SCHEMAS:
                qa.append({"subtype": ev["subtype"], "question": ev["utterance"],
                           "gt": ev.get("gt_answer"), "answer": None})
        if not alive:
            break
    # ---------------- scoring
    dev = ro.dev_tick
    recs = [r for r in state.records if not r.skipped]
    flags = [r for r in replies if r["reply"].get("type") in ("FLAG_ERROR", "SUGGEST")]
    warns = [r for r in replies if r["reply"].get("type") == "WARN_RISK"]
    first_valid = None
    for r in flags:
        g = checks.flag_is_correct({"error": r["reply"].get("error"),
                                    "refers_to": r["reply"].get("refers_to")},
                                   state.records, lang_log=ro.lang_log, now=r["t"], game=ro.game)
        r["gate"] = g
        if g["accepted"] and first_valid is None:
            first_valid = r
    # a false positive = a flag the gate REJECTS (wrong label / nothing wrong / flagged-legal)
    # EXCEPT correct warnings about a real secondary hazard (smoking station). Repeats of the
    # same (type, label) are deduplicated — verbosity is reported separately.
    rejected = [r for r in flags if not r["gate"]["accepted"] and not r["gate"].get("hazard_real")]
    seen_kind, fp = set(), []
    for r in rejected + [r for r in replies if r["reply"].get("type") == "ASK_CLARIFY"]:
        kind = (r["reply"].get("type"), r["reply"].get("error"))
        if kind not in seen_kind:
            seen_kind.add(kind)
            fp.append(r)
    n_repeat_warnings = len(rejected) - len([r for r in rejected
                                             if (r["reply"].get("type"), r["reply"].get("error"))
                                             in seen_kind]) if rejected else 0
    n_repeat_warnings = max(0, len(rejected) - len({(r["reply"].get("type"), r["reply"].get("error"))
                                                    for r in rejected}))
    # [v2] no registration channel: announcement handling is tracking-judged from the
    # supervisor's memory JSON (did subgoals_future absorb the announced change?)
    updates = [r for r in replies if r["reply"].get("type") == "ACKNOWLEDGE_LEGAL"]
    return dict(case=case["name"], err=case["err"], ask=case["ask"], outcome=ro.outcome,
                ticks=ro.tick, dev_tick=dev, n_decisions=len(replies),
                n_silent=sum(1 for r in replies if r["reply"].get("type") in ("SILENT", None)),
                false_positives=[{"t": r["t"], "type": r["reply"].get("type"),
                                  "error": r["reply"].get("error"), "text": r["reply"].get("text")}
                                 for r in fp],
                n_warnings=len(warns), n_repeat_flags=n_repeat_warnings,
                n_hazard_credited=sum(1 for r in flags if r["gate"].get("hazard_real")
                                      and not r["gate"]["accepted"]),
                flagged=bool(flags), flagged_valid=bool(first_valid),
                latency=(first_valid["t"] - dev) if (first_valid and dev is not None) else None,
                first_flag=(dict(t=first_valid["t"], **{k: first_valid["reply"].get(k)
                            for k in ("error", "text", "refers_to")},
                            gate=first_valid["gate"]) if first_valid else
                            (dict(t=flags[0]["t"], **{k: flags[0]["reply"].get(k)
                             for k in ("error", "text", "refers_to")}, gate=flags[0].get("gate"))
                             if flags else None)),
                has_real_error=bool(recs), expected_label=((recs[0].function or recs[0].event_subtype) if recs else None),
                announce=bool(case["events"]), registered=bool(updates),
                qa=qa, replies=[{"t": r["t"], "type": r["reply"].get("type"),
                                 "text": r["reply"].get("text", "")[:200]} for r in replies],
                flags_full=[{"t": r["t"], "error": r["reply"].get("error"),
                             "refers_to": r["reply"].get("refers_to"),
                             "text": r["reply"].get("text", "")[:220], "gate": r["gate"]}
                            for r in flags])

# ============================================================== phase: interact
def human_json(client, state, ro, conversation):
    return gemini(client, human_context(state, ro, conversation),
                  "It's your turn to speak (or wrap up). Output your JSON now.", max_tok=900)

def run_interact(case, client, max_rounds=4):
    state, ro = build_episode((case["err"], dict(case["kwargs"])) if case["err"] else None,
                              case["persona"])
    ro.cap = 500          # recoveries + conversation time cost legitimately run long
    if case["events"]:
        ro.lang_pending.extend(case["events"])
    sup_sys = supervisor_context(state, hist_last=HIST)
    path, replies, transcript, seen_lang, said = [], [], [], 0, None
    report = dict(case=case["name"], gate=None, entered=False, conversation=[], filter=None,
                  fix_source=None, averted=False, announced_change=None)
    ep_alive = [True]

    def world_tick(n):
        """Talking costs time [DECIDED]: the world advances while agents converse."""
        for _ in range(n):
            path.append(tuple(ro.game.cooks[0].pos))
            if not ro.step():
                ep_alive[0] = False
                return

    def converse(opening_said=None):
        """Multi-turn conversation, sentences only; each exchange costs world time.
        Returns the human's final parse dict."""
        convo = list(transcript[-6:])
        last_h = None
        for _ in range(max_rounds):
            h = human_json(client, state, ro, convo)
            last_h = h
            u = {"speaker": "human", "t": ro.tick, "text": h.get("utterance", ""), "move": h.get("move")}
            convo.append(u); transcript.append(u)
            report["conversation"].append({**u, "t": ro.tick, "close": h.get("close_conversation"),
                                           "parse_dict": h.get("parse_dict")})
            world_tick(4)
            if h.get("close_conversation") or not ep_alive[0]:
                break
            obs = supervisor_observation(state, ro, path, hist_last=HIST)
            rep = sup_reply(client, sup_sys, obs, h.get("utterance", ""), transcript)
            replies.append({"t": ro.tick, "said": h.get("utterance"), "reply": rep})
            v = {"speaker": "vlm", "t": ro.tick, "text": rep.get("text", "")}
            convo.append(v); transcript.append(v)
            report["conversation"].append({**v, "t": ro.tick, "type": rep.get("type"),
                                           "error": rep.get("error")})
            world_tick(4)
            if not ep_alive[0]:
                break
        return (last_h or {}).get("parse_dict") or {}
    while True:
        path.append(tuple(ro.game.cooks[0].pos))
        if said is not None or ro.tick % POLL == 0:
            obs = supervisor_observation(state, ro, path, hist_last=HIST)
            rep = sup_reply(client, sup_sys, obs, said, transcript)
            replies.append({"t": ro.tick, "said": said, "reply": rep})
            if rep.get("type") not in ("SILENT", None):
                transcript.append({"speaker": "vlm", "t": ro.tick, "text": rep.get("text", "")})
                report["conversation"].append({"speaker": "vlm", "t": ro.tick,
                                               "type": rep.get("type"), "error": rep.get("error"),
                                               "text": rep.get("text", "")})
            if said is not None and qa_pending(state, ro, seen_lang):
                # the human asked; if answered correctly the avert is applied
                ev = ro.lang_log[-1]
                if rep.get("type") == "ANSWER" and ev["subtype"] in TRANSCRIBE_SCHEMAS:
                    tr = transcribe(client, state, ev["utterance"], rep.get("text", ""), ev["subtype"])
                    ok = grade(ev["subtype"], tr, ev.get("gt_answer"))
                    report["qa_answer"] = dict(answer=rep.get("text"), transcription=tr, correct=ok)
                    if ok and state.records:
                        system.apply_avert(ro, state)   # clears deviation + restores required subgoals
                        report["averted"] = True
            # [v2] UPDATE_GOAL retired: announced changes are applied at event time (below);
            # the supervisor's announcement TRACKING is judged from its memory JSON.
            stalled = (ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None) or getattr(ro, "_wander", 0) > 6
            if rep.get("type") in ("FLAG_ERROR", "SUGGEST") and rep.get("text") and stalled \
                    and (state.mode != "error_correction" or
                         (state.correction and state.correction.get("fix_applied"))):
                # OVERHEARD DIRECTION: mid-recovery, queue empty, the supervisor issues a
                # direct instruction outside a conversation — a person standing there would
                # just follow it. One lightweight human parse (persona still applies).
                h = human_json(client, state, ro, transcript[-4:] +
                               [{"speaker": "vlm", "text": rep.get("text", "")}])
                report["conversation"].append({"speaker": "human", "t": ro.tick,
                                               "text": h.get("utterance", ""),
                                               "move": h.get("move"), "overheard": True,
                                               "parse_dict": h.get("parse_dict")})
                filt = system.convert_parse(h.get("parse_dict") or {}, state.profile,
                                            state.recipe, ro.game)
                if filt["anything_applicable"]:
                    system.apply_fix(ro, state, filt["commands"])
                    if getattr(ro, "_wander", 0) > 0:
                        ro._wander = 0            # the words snapped the lost cook out of the stall
                        if ro.log:
                            ro.log[-1].update(status="done", t_done=ro.tick)
            if rep.get("type") in ("FLAG_ERROR", "SUGGEST") and state.mode != "error_correction":
                g = checks.flag_is_correct({"error": rep.get("error"), "refers_to": rep.get("refers_to")},
                                           state.records, lang_log=ro.lang_log, now=ro.tick,
                                           game=ro.game)
                if report["gate"] is None or not report["gate"].get("accepted"):
                    report["gate"] = g          # SCORING-ONLY record (v2): invisible to both agents
                # [v2] CONVICTION GATE: the flag opens a conversation; the human checks the
                # claim against its OWN past record + persona. Only being convinced (an
                # explicit accept, or following the guidance) changes the plan — the oracle
                # verdict `g` is written to the score sheet and gates nothing.
                n_before = len(report["conversation"])
                parse = converse(rep.get("text"))
                moves = [c.get("move") for c in report["conversation"][n_before:]
                         if c.get("speaker") == "human"]
                filt = system.convert_parse(parse, state.profile, state.recipe, ro.game)
                convinced = ("accept" in moves) or filt["anything_applicable"]
                report.setdefault("conviction", []).append(
                    {"t": ro.tick, "flag_error": rep.get("error"), "moves": moves,
                     "convinced": convinced, "gate_accepted": g["accepted"]})
                if convinced:
                    system.enter_error_correction(ro, state)
                    report["entered"] = True
                    report["filter"] = {k: filt[k] for k in ("kept", "dropped_by_acceptance",
                                        "unusable_for_solvability", "anything_applicable")}
                    if filt["anything_applicable"]:
                        system.apply_fix(ro, state, filt["commands"]); report["fix_source"] = "conversation"
                    elif state.profile.task_solvability == "high":
                        system.apply_fix(ro, state, system.self_repair_plan(state, ro.game))
                        report["fix_source"] = "self_repair"
            said = None
        if state.mode == "error_correction" and state.correction and \
                state.correction.get("fix_applied") and system.resolved(ro, state):
            system.exit_correction(ro, state)
            report["resolved_at"] = ro.tick
        # RE-ENGAGEMENT: the fix ran dry without resolving (e.g. an ambiguous suggestion sent
        # the cook to the wrong pan) -> the human speaks up again instead of freezing silently
        if (state.mode == "error_correction" and state.correction
                and state.correction.get("fix_applied") and ro.i + 1 >= len(ro.cmds)
                and ro.ctrl.skill is None and not system.resolved(ro, state)
                and report.setdefault("reengagements", 0) <
                    {"avoidant": 2, "skeptical": 4, "ok": 6}.get(
                        state.profile.interaction_preference, 4)):
            report["reengagements"] += 1
            transcript.append({"speaker": "human", "text": "(that didn't fix it — asking again)"})
            parse = converse()
            filt = system.convert_parse(parse, state.profile, state.recipe, ro.game)
            report["filter2"] = {k: filt[k] for k in ("kept", "anything_applicable")}
            if filt["anything_applicable"]:
                system.apply_fix(ro, state, filt["commands"])
        if not ep_alive[0]:
            break
        alive = ro.step()
        if len(ro.lang_log) > seen_lang:
            ev = ro.lang_log[-1]
            said = ev["utterance"]
            transcript.append({"speaker": "human", "t": ro.tick, "text": said})
            report["conversation"].append({"speaker": "human", "t": ro.tick, "text": said})
            seen_lang = len(ro.lang_log)
            if ev.get("subtype") == "ANNOUNCE_PLAN_CHANGE":
                # [v2] the announcement IS the change — the system applies it the moment the
                # human says it; the supervisor is never in the loop
                report["announced_change"] = system.apply_announced_change(ro, state, ev)
        if not alive:
            break
    report.update(outcome=ro.outcome, ticks=ro.tick,
                  expected_label=((state.records[0].function or state.records[0].event_subtype) if state.records else None))
    return report

def qa_pending(state, ro, seen_before):
    return bool(ro.lang_log) and ro.lang_log[-1]["subtype"] in TRANSCRIBE_SCHEMAS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["detect", "interact"], required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = all_cases()
    case = cases[args.case]
    if args.persona:
        case = dict(case, persona=args.persona)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rec = run_detect(case, client) if args.phase == "detect" else run_interact(case, client)
    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    print(f"{args.case}: outcome={rec.get('outcome')} "
          f"{'flag_valid=' + str(rec.get('flagged_valid')) if args.phase=='detect' else 'entered=' + str(rec.get('entered'))}")
