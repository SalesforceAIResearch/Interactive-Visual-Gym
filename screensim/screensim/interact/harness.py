"""Run a sweep of cases through the interaction loop and score honestly
(skill §7). Two agent backends:

  scripted   ScriptedHuman × {SilentAssistant, OracleAssistant} — deterministic,
             no API; the leaderboard brackets (silent floor, oracle ceiling).
  llm        HumanAgent × AssistantAgent over Gemini, with real captured frames.

Reports carry provenance (code revision, interface version, pace) and the
validity gates; a voided episode is `not_measured`, never 0 (skill §7.2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from screensim.interact.manual import manual_context


from screensim.tasks import by_id, TASKS
from screensim.interact import INTERFACE_VERSION
from screensim.interact.errors import generate_cases
from screensim.interact.personas import PERSONAS
from screensim.interact.episode import run_episode
from screensim.interact.agents import (ScriptedHuman, SilentAssistant,
                                        OracleAssistant)
from screensim.interact.verdicts import (CORRECT_DETECTION, NEUTRAL_VERDICTS,
                                         FALSE_ALARM)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tmp", "interact")


def _rev():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        # no git repo — fingerprint the interact package instead
        h = hashlib.md5()
        pkg = os.path.join(ROOT, "screensim", "interact")
        for fn in sorted(os.listdir(pkg)):
            if fn.endswith(".py"):
                h.update(open(os.path.join(pkg, fn), "rb").read())
        return "pkg-" + h.hexdigest()[:10]


def _frame_fn(cap):
    return (lambda dev: cap.snap(dev)) if cap else None


def run_sweep(task_ids, personas, backend, out_path, ledger_path):
    from screensim.interact import __init__ as _i  # noqa
    episodes, ledger_all = [], []
    cap = None
    hclient = aclient = None
    if backend == "llm":
        from screensim.render.capture import Capturer
        from screensim.interact.agents import (LLMClient, HumanAgent,
                                                AssistantAgent)
        cap = Capturer(scale=1.0).__enter__()
        hclient = aclient = LLMClient()

    try:
        for tid in task_ids:
            task = by_id(tid)
            for pname in personas:
                persona = PERSONAS[pname]
                ir, cases, ledger = generate_cases(task, personas=(pname,))
                for r in ledger:
                    ledger_all.append(r)
                for case in cases:
                    case.persona = pname
                    if backend == "scripted":
                        # both brackets; the oracle only knows about injected
                        # errors, so on benign episodes it stays silent (there
                        # is nothing to detect) — an oracle that nagged on
                        # intent would just be a bad assistant
                        def _oracle(c=case):
                            return (OracleAssistant(c.err, c.deviated_label)
                                    if c.family == "error"
                                    else SilentAssistant())
                        for label, mk_assistant in (
                            ("silent", lambda: SilentAssistant()),
                            ("oracle", _oracle)):
                            h = ScriptedHuman(persona, ir, case)
                            ep = run_episode(case, ir, h, mk_assistant())
                            ep["backend"] = f"scripted:{label}"
                            _stamp(ep, case)
                            episodes.append(ep)
                    else:
                        from screensim.interact.agents import (HumanAgent,
                                                               AssistantAgent)
                        h = HumanAgent(persona, task.instruction, hclient)
                        a = AssistantAgent(persona.questionnaire, aclient,
                                           use_relational=False,
                                           task_desc=task.instruction, manual=manual_context(task.id))
                        ep = run_episode(case, ir, h, a, frame_fn=_frame_fn(cap))
                        ep["backend"] = "llm"
                        _stamp(ep, case)
                        episodes.append(ep)
    finally:
        if cap:
            cap.__exit__(None, None, None)

    report = _aggregate(episodes)
    report["provenance"] = {"code_rev": _rev(),
                            "interface": INTERFACE_VERSION,
                            "pace": "conversation", "backend": backend}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "episodes": _slim(episodes)}, f, indent=1)
    with open(ledger_path, "w") as f:
        json.dump(ledger_all, f, indent=1)
    return report


def _stamp(ep, case):
    ep["certified"] = case.certified
    ep["deviated_label"] = case.deviated_label
    ep["provenance_rev"] = _rev()


def _slim(episodes):
    keep = ("case", "task", "family", "err", "persona", "backend",
            "detection_verdict", "detection_tick", "restraint_verdict",
            "nags", "repaired", "deviation_committed", "goal_ok",
            "injected_at", "final_tick", "leaks", "scoreable", "void_gates",
            "transcript")
    return [{k: e.get(k) for k in keep} for e in episodes]


def _aggregate(episodes):
    def rate(num, den):
        return {"n": num, "of": den,
                "rate": round(num / den, 3) if den else None}

    by_backend = {}
    for e in episodes:
        b = e["backend"]
        d = by_backend.setdefault(b, {
            "errors": 0, "detected": 0, "prevented": 0, "committed": 0,
            "clean_intent": 0, "false_alarms": 0, "nags": 0,
            "void": 0, "leaks": 0})
        if e["void_gates"]:
            d["void"] += 1
            continue
        d["leaks"] += e["leaks"]
        if e["family"] == "error":
            d["errors"] += 1
            if e["detection_verdict"] in CORRECT_DETECTION:
                d["detected"] += 1
            if e["repaired"]:
                d["prevented"] += 1
            if e["deviation_committed"]:
                d["committed"] += 1
        else:
            d["clean_intent"] += 1
            d["nags"] += e["nags"]
            if e["restraint_verdict"] == FALSE_ALARM:
                d["false_alarms"] += 1

    out = {}
    for b, d in by_backend.items():
        out[b] = {
            "detection": rate(d["detected"], d["errors"]),        # track 1
            "prevention": rate(d["prevented"], d["errors"]),      # track 3
            "committed_errors": rate(d["committed"], d["errors"]),
            "false_alarm_on_benign": rate(d["false_alarms"], d["clean_intent"]),
            "nags_on_benign": d["nags"],
            "void_episodes": d["void"], "coordinate_leaks": d["leaks"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--tasks", nargs="*", default=[
        "safari_notif_quiet", "roaming_off", "precise_off_weatherly",
        "limit_adult_web", "hotspot_2ghz", "rent_share_standard"])
    ap.add_argument("--personas", nargs="*",
                    default=["guided_novice", "brisk_expert",
                             "distracted_average"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    report = run_sweep(args.tasks, args.personas, args.backend,
                       os.path.join(OUT, f"sweep_{args.backend}.json"),
                       os.path.join(OUT, f"ledger_{args.backend}.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
