"""Real assistant leaderboard.

For a leaderboard the HUMAN is scripted (deterministic) so every assistant
faces the identical world — the skill's "reproducible episodes to compare
assistants on identical worlds". Only the assistant is the model under test.
Episodes run in a thread pool, each thread owning its own headless browser.

Contestants are (name, model, interface_version). The scripted silent floor
and oracle ceiling are included for free (no API) as leaderboard anchors.

  python -m screensim.interact.leaderboard            # default roster
  python -m screensim.interact.leaderboard --tasks safari_notif_quiet roaming_off
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from screensim.interact.manual import manual_context
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screensim.tasks import by_id
from screensim.interact import INTERFACE_VERSION, MODEL, interface
from screensim.interact.errors import generate_cases, slip_text
from screensim.interact.personas import PERSONAS
from screensim.interact.episode import run_episode
from screensim.interact.agents import (ScriptedHuman, SilentAssistant,
                                        OracleAssistant, AssistantAgent,
                                        HumanAgent, LLMClient)
from screensim.interact.verdicts import CORRECT_DETECTION, FALSE_ALARM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tmp", "interact")

# The benchmark run. Both agents are gemini-3.7-flash (MODEL); the scripted
# floor/ceiling bracket the board. `llm_assistant` isolates the assistant on
# the deterministic world; `llm_pair` is the full LLM↔LLM simulation.
# roster: (label, kind, model, interface)
ROSTER = [
    ("silent floor (scripted)", "scripted", None, "sv1"),
    ("oracle ceiling (scripted)", "scripted", None, "sv1"),
    (f"{MODEL} assistant · scripted human", "llm_assistant", MODEL, "sv1"),
    (f"{MODEL} ↔ {MODEL} (full sim)", "llm_pair", MODEL, "sv1"),
]

_tl = threading.local()


def _capturer():
    """One headless browser per worker thread."""
    cap = getattr(_tl, "cap", None)
    if cap is None:
        from screensim.render.capture import Capturer
        cap = Capturer(scale=1.0).__enter__()
        _tl.cap = cap
    return cap


def _client(model):
    key = ("client", model)
    c = getattr(_tl, "clients", None)
    if c is None:
        c = _tl.clients = {}
    if model not in c:
        c[model] = LLMClient(model=model)
    return c[model]


def _one_episode(contestant, task, persona, case, ir):
    label, kind, model, iface = contestant
    cfg = interface(iface)
    if kind == "scripted":
        h = ScriptedHuman(persona, ir, case)
        if label.startswith("oracle"):
            a = (OracleAssistant(case.err, case.deviated_label)
                 if case.family == "error" else SilentAssistant())
        else:
            a = SilentAssistant()
        ep = run_episode(case, ir, h, a, interface_version=iface)
    elif kind == "llm_assistant":
        cap = _capturer()
        h = ScriptedHuman(persona, ir, case)          # deterministic world
        a = AssistantAgent(persona.questionnaire, _client(model),
                           use_relational=cfg["relational"],
                           task_desc=task.instruction, manual=manual_context(task.id))
        ep = run_episode(case, ir, h, a, frame_fn=(lambda d: cap.snap(d)),
                         interface_version=iface, k_frames=2)
    else:  # llm_pair — the full LLM↔LLM simulation (filmed)
        cap = _capturer()
        h = HumanAgent(persona, task.instruction, _client(model))
        h.set_slip(slip_text(case, ir))               # inject the designed slip
        a = AssistantAgent(persona.questionnaire, _client(model),
                           use_relational=cfg["relational"],
                           task_desc=task.instruction, manual=manual_context(task.id))
        ep = run_episode(case, ir, h, a, frame_fn=(lambda d: cap.snap(d)),
                         film_cap=(lambda d: cap.snap(d)),
                         interface_version=iface, k_frames=2)
    ep["contestant"] = label
    ep["model"] = model or "—"
    ep["persona"] = persona.name
    # film the full-sim episodes into the results reel
    if kind == "llm_pair" and ep.get("_film"):
        import os
        from tools.film_episode import film_episode
        vid = f"videos/{persona.name}__{case.name}.mp4"
        try:
            film_episode(ep, os.path.join(OUT, vid), fps=2)
            ep["video"] = vid
        except Exception as e:
            ep["video_error"] = str(e)[:120]
    ep.pop("_film", None)
    return ep


def _sample_cases(cases, per_family):
    """A family-balanced sample: keep every benign case (clean/intent) and,
    within the error family, spread the picks across distinct error KINDS so
    the filmed gallery shows diverse mistakes, not five of the same."""
    if not per_family:
        return list(cases)
    keep, by_kind = [], {}
    for c in cases:
        if c.family != "error":
            keep.append(c)                       # always keep clean + intent
        else:
            by_kind.setdefault(c.err, []).append(c)
    # round-robin across error kinds up to the budget
    kinds = list(by_kind.values())
    i = 0
    while len([c for c in keep if c.family == "error"]) < per_family and kinds:
        row = kinds[i % len(kinds)]
        if row:
            keep.append(row.pop(0))
        else:
            kinds.remove(row)
            continue
        i += 1
    return keep


def run(tasks, persona_name, workers, out_path, per_family=0, drop_clean=False):
    persona = PERSONAS[persona_name]
    # build the case pool once (shared world across contestants)
    pool = []
    for tid in tasks:
        task = by_id(tid)
        ir, cases, ledger = generate_cases(task, personas=(persona_name,))
        picked = _sample_cases(cases, per_family)
        if drop_clean:
            # a 'clean' case is the human walking the correct trajectory with
            # NO error — exclude it (announced 'intent' deviations are kept,
            # they still exercise assistant restraint)
            picked = [c for c in picked if c.family != "clean"]
        for c in picked:
            pool.append((task, ir, c))
    print(f"{len(pool)} cases × {len(ROSTER)} contestants "
          f"= {len(pool)*len(ROSTER)} episodes", flush=True)

    jobs = [(cont, task, ir, c) for cont in ROSTER for (task, ir, c) in pool]
    episodes = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_episode, cont, task, persona, c, ir):
                (cont[0], c.name) for (cont, task, ir, c) in jobs}
        for fu in as_completed(futs):
            try:
                episodes.append(fu.result())
            except Exception as e:
                lbl, cn = futs[fu]
                episodes.append({"contestant": lbl, "case": cn,
                                 "error": str(e)[:200], "void_gates": ["crash"],
                                 "family": "error"})
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    board = _leaderboard(episodes)
    report = {"persona": persona_name, "tasks": tasks,
              "roster": [{"label": r[0], "model": r[2], "interface": r[3]}
                         for r in ROSTER],
              "leaderboard": board}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "episodes": _slim(episodes)}, f, indent=1)
    _print_board(board)
    return report


def _leaderboard(episodes):
    agg = {}
    for e in episodes:
        d = agg.setdefault(e["contestant"], {
            "err": 0, "detected": 0, "prevented": 0, "committed": 0,
            "benign": 0, "false_alarm": 0, "nags": 0, "void": 0,
            "leaks": 0, "lat": []})
        if e.get("void_gates"):
            d["void"] += 1
            continue
        d["leaks"] += e.get("leaks", 0)
        if e.get("family") == "error":
            d["err"] += 1
            if e.get("detection_verdict") in CORRECT_DETECTION:
                d["detected"] += 1
                if e.get("injected_at") is not None and e.get("detection_tick"):
                    d["lat"].append(max(0, e["detection_tick"] - e["injected_at"]))
            if e.get("repaired"):
                d["prevented"] += 1
            if e.get("deviation_committed"):
                d["committed"] += 1
        else:
            d["benign"] += 1
            d["nags"] += e.get("nags", 0)
            if e.get("restraint_verdict") == FALSE_ALARM:
                d["false_alarm"] += 1
    rows = []
    for label, d in agg.items():
        det = d["detected"] / d["err"] if d["err"] else None
        prev = d["prevented"] / d["err"] if d["err"] else None
        fa = d["false_alarm"] / d["benign"] if d["benign"] else None
        lat = round(sum(d["lat"]) / len(d["lat"]), 1) if d["lat"] else None
        rows.append({
            "contestant": label, "errors": d["err"],
            "detection": _r(det), "prevention": _r(prev),
            "committed": _r(d["committed"] / d["err"] if d["err"] else None),
            "false_alarm_benign": _r(fa), "nags_benign": d["nags"],
            "mean_latency_ticks": lat, "coordinate_leaks": d["leaks"],
            "void": d["void"],
            # a single ordering score: reward detection & restraint, punish
            # false alarms and (heavily) committed irreversible errors
            "score": _r((det or 0) - (fa or 0)
                        - (d["committed"] / d["err"] if d["err"] else 0)),
        })
    rows.sort(key=lambda r: (r["score"] if r["score"] is not None else -9),
              reverse=True)
    return rows


def _r(x):
    return round(x, 3) if isinstance(x, (int, float)) else x


def _print_board(board):
    print("\n" + "=" * 92)
    print(f"{'contestant':32} {'detect':>7} {'prevent':>8} {'commit':>7} "
          f"{'falseAl':>8} {'nags':>5} {'lat':>5} {'score':>7}")
    print("-" * 92)
    for r in board:
        print(f"{r['contestant']:32} {_f(r['detection']):>7} "
              f"{_f(r['prevention']):>8} {_f(r['committed']):>7} "
              f"{_f(r['false_alarm_benign']):>8} {r['nags_benign']:>5} "
              f"{str(r['mean_latency_ticks'] or '—'):>5} {_f(r['score']):>7}")
    print("=" * 92)


def _f(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def _slim(episodes):
    keep = ("contestant", "model", "interface", "case", "task", "family",
            "err", "detection_verdict", "repaired", "deviation_committed",
            "restraint_verdict", "nags", "injected_at", "detection_tick",
            "leaks", "void_gates", "transcript", "timeline", "video",
            "goal_ok", "goal_detail", "persona", "deviated_label")
    return [{k: e.get(k) for k in keep} for e in episodes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=[
        "safari_notif_quiet", "roaming_off", "limit_adult_web"])
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--persona", default="distracted_average")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-family", type=int, default=0,
                    help="cap error cases per task (0 = all); benign kept")
    ap.add_argument("--drop-clean", action="store_true",
                    help="exclude no-error 'clean' episodes (keep error+intent)")
    args = ap.parse_args()
    import screensim.interact.episode as _EP
    _EP.MAX_TURNS = args.max_turns
    run(args.tasks, args.persona, args.workers,
        os.path.join(OUT, "leaderboard.json"), per_family=args.per_family,
        drop_clean=args.drop_clean)


if __name__ == "__main__":
    main()
