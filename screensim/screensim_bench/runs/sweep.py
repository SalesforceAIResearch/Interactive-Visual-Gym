"""Composite (multi-error) sweep: every episode × 3 personas × contestants,
scored with the CookSim-aligned metric set (R, P, F1, DT, CC, S).

Contestants: silent floor, oracle ceiling (no API), and the LLM assistant under
test (gemini-3.7-flash, frames-only, polled every 2 ticks with 2 frames).

  python screensim_bench/runs/sweep.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from screensim.tasks import by_id
from screensim.interact import MODEL, interface
from screensim.interact.composite import design_composite_episodes
from screensim.interact.composite_run import (run_composite, CompositeOracle,
                                              SilentComposite)
from screensim.interact.composite_run_v2 import run_composite_v2, OracleV2, SilentV2
from screensim.interact.personas import PERSONAS, bench_personas_for
from screensim.interact.manual import manual_context
from screensim.engine_rev import engine_rev as _engine_rev
from screensim.interact.agents import AssistantAgent, AssistantV2, PlanHuman, make_client

OUT = os.path.join(_ROOT, "tmp", "composites")
from screensim.tasks import HARD_TASKS as HARD
# the runnable roster (the two live/realtime variants are not vision-chat models):
from screensim.interact.roster import TAGS as ROSTER_TAGS
# the formal roster = CookSim's balanced-20 minus the slots that cannot run here (screensim/interact/roster.py);
# any other model name still goes through agents.make_client (the pre-roster path used by pilots)
MODELS = list(ROSTER_TAGS)
CONTESTANTS = ["silent floor", "oracle ceiling"] + MODELS

_tl = threading.local()


def _cap():
    c = getattr(_tl, "cap", None)
    if c is None:
        from screensim.render.capture import Capturer
        c = _tl.cap = Capturer(scale=1.0).__enter__()
    return c


def _client(model):
    cache = getattr(_tl, "clients", None)
    if cache is None:
        cache = _tl.clients = {}
    if model not in cache:
        cache[model] = make_client(model)
    return cache[model]


def one(contestant, task, ir, comp, persona):
    if INTERFACE == "v2":
        return one_v2(contestant, task, ir, comp, persona)
    if contestant == "silent floor":
        ep = run_composite(comp, ir, persona, SilentComposite(), frame_fn=None)
    elif contestant == "oracle ceiling":
        ep = run_composite(comp, ir, persona, CompositeOracle(), frame_fn=None)
    else:
        cap = _cap()
        a = AssistantAgent(persona.questionnaire, _client(contestant),
                           use_relational=False, task_desc=task.instruction,
                           manual=("" if NO_MANUAL else manual_context(task.id)))
        ep = run_composite(comp, ir, persona, a,
                           frame_fn=(lambda d: cap.snap(d)), k_frames=2)
    ep.pop("_film", None)
    ep["contestant"] = contestant
    return ep


NO_MANUAL = False
INTERFACE = "v2"                 # CookSim v2 protocol: typed flags, human moves, LLM human
HUMAN_MODEL = "gemini-3.7-flash"  # the person (PlanHuman) in every v2 run
ASSISTANT_CONFIG = {}           # explicit CLI overrides; published roster is never mutated


def one_v2(contestant, task, ir, comp, persona):
    """v2: the person is an LLM (PlanHuman) in EVERY run, including the floor
    and ceiling; the assistant emits typed flags graded by exact match."""
    human = PlanHuman(persona, task.instruction, _client(HUMAN_MODEL))
    if contestant == "silent floor":
        ep = run_composite_v2(comp, ir, persona, SilentV2(), human, frame_fn=None,
                              contestant=contestant)
    elif contestant == "oracle ceiling":
        ep = run_composite_v2(comp, ir, persona, OracleV2(), human, frame_fn=None,
                              contestant=contestant)
    elif contestant in ROSTER_TAGS or contestant in ASSISTANT_CONFIG:
        from screensim.interact.agents import RosterAssistant
        from screensim.interact.roster import make_backend
        cap = _cap()
        backend = (make_backend(contestant, entry=ASSISTANT_CONFIG[contestant])
                   if contestant in ASSISTANT_CONFIG else None)
        a = RosterAssistant(contestant, persona, task_desc=task.instruction,
                            manual=("" if NO_MANUAL else manual_context(task.id)), backend=backend)
        ep = run_composite_v2(comp, ir, persona, a, human,
                              frame_fn=(lambda d: cap.snap(d)), k_frames=2,
                              contestant=contestant)
        ep["assistant_stats"] = a.stats()
        ep["assistant_meta"] = a.meta_log
    else:
        cap = _cap()
        a = AssistantV2(persona, _client(contestant), use_relational=False,
                        task_desc=task.instruction,
                        manual=("" if NO_MANUAL else manual_context(task.id)))
        ep = run_composite_v2(comp, ir, persona, a, human,
                              frame_fn=(lambda d: cap.snap(d)), k_frames=2,
                              contestant=contestant)
    ep.pop("_film", None)
    ep["contestant"] = contestant
    from screensim.interact.beats_design import beats_design
    ep["beats_design"] = beats_design({"task": task.id, "episode": comp.id})
    return ep


def build_jobs(contestants=None, tasks=None, limit=0, shard=None, cells=None, episode=None, persona=None):
    """Resolve and validate the complete selection before creating clients or output files."""
    if limit < 0:
        raise ValueError("--limit must be nonnegative")
    if shard and (len(shard) != 2 or not 1 <= shard[0] <= shard[1]):
        raise ValueError("--shard must be i/n with 1 <= i <= n")
    if persona and persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    contestants = contestants or CONTESTANTS
    pool = []
    for tid in (tasks or HARD):
        task = by_id(tid)
        ir, eps = design_composite_episodes(task, per_task=3)
        for e in eps:
            if episode and e.id != episode:
                continue
            for pname in ([persona] if persona else bench_personas_for(e.id)):
                pool.append((task, ir, e, PERSONAS[pname]))
    if cells is not None:
        want = {(x["contestant"], x["episode"], x["persona"]) for x in cells}
        available = {(c, e.id, p.name) for c in contestants for t, ir, e, p in pool}
        missing = want - available
        if missing:
            raise ValueError(f"Requested cells are not in the selected tasks/personas: {sorted(missing)}")
        pool = [row for row in pool if any((c, row[2].id, row[3].name) in want for c in contestants)]
    if limit:
        pool = pool[:limit]                       # a setup check: the first N cases only
    if shard:
        i, n = shard                               # a fixed, disjoint slice: every n-th case from i (1-based)
        pool = pool[i - 1::n]
    jobs = [(c, t, ir, e, p) for c in contestants for (t, ir, e, p) in pool]
    if cells is not None:
        jobs = [j for j in jobs if (j[0], j[3].id, j[4].name) in want]
    if not jobs:
        raise ValueError("No episodes match the requested task, case, persona, and shard")
    return jobs


def run(workers, out_path, contestants=None, append=False, tasks=None, limit=0, shard=None, cells=None,
        episode=None, persona=None, dry_run=False):
    if workers < 1:
        raise ValueError("--workers must be positive")
    jobs = build_jobs(contestants, tasks, limit, shard, cells, episode, persona)
    if dry_run:
        selected = [{"contestant": c, "task": t.id, "episode": e.id, "persona": p.name}
                    for c, t, ir, e, p in jobs]
        print(json.dumps(selected, indent=2))
        return selected
    # every finished run is appended in full to <out>.records.jsonl at once (films, audits and
    # offline regrades need the timeline); a restarted run skips the cells already there
    rec_path = os.path.splitext(out_path)[0] + ".records.jsonl"
    os.makedirs(os.path.dirname(rec_path) or ".", exist_ok=True)
    have, done_elsewhere = {}, set()
    # a cell finished in ANY records file of this output directory counts as done (so a model can be
    # re-split into more slices without redoing work); only this file's own runs are carried in `results`
    import glob as _glob
    for path in sorted(_glob.glob(os.path.join(os.path.dirname(rec_path) or ".", "*.records.jsonl"))):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" in r:
                continue
            key = (r["contestant"], r["episode"], r["persona"])
            if path == rec_path:
                have[key] = r
            else:
                done_elsewhere.add(key)
    jobs = [j for j in jobs if (j[0], j[3].id, j[4].name) not in have
            and (j[0], j[3].id, j[4].name) not in done_elsewhere]
    lock = threading.Lock()
    print(f"{len(jobs)} pending runs", flush=True)

    results = list(have.values())
    if have:
        print(f"resuming: {len(have)} runs already in {rec_path}", flush=True)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, c, t, ir, e, p): (c, e.id, p.name)
                for (c, t, ir, e, p) in jobs}
        for fu in as_completed(futs):
            try:
                r = fu.result()
                r.setdefault("engine_rev", _engine_rev())
                r["human_model"] = HUMAN_MODEL
            except Exception as ex_:
                c, eid, pn = futs[fu]
                r = {"contestant": c, "episode": eid, "persona": pn, "error": str(ex_)[:200]}
            results.append(r)
            with lock, open(rec_path, "a") as fh:
                fh.write(json.dumps(r, default=str) + "\n")
            done += 1
            if done % 15 == 0:
                print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    slim = [_slim(r) for r in results]
    if append and os.path.exists(out_path):
        prev = json.load(open(out_path)).get("runs", [])
        # replace exactly the (contestant, episode, persona) cells re-run here
        done = {(r["contestant"], r["episode"], r["persona"]) for r in slim}
        prev = [r for r in prev if (r.get("contestant"), r.get("episode"), r.get("persona")) not in done]
        slim = prev + slim
    board = _aggregate([{**r} for r in slim])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump({"board": board, "runs": slim}, open(out_path, "w"), indent=1)
    _print(board)
    failures = sum("error" in r for r in results)
    if failures:
        raise RuntimeError(f"{failures} episode(s) failed; details are saved in {rec_path}. Rerun to retry failed cells.")
    return board


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 3) if xs else None


def _aggregate(results):
    agg = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["contestant"], r["persona"])
        d = agg.setdefault(key, {"R": [], "P": [], "F1": [], "DT": [], "S": [],
                                 "CC": [], "FF": [], "n": 0})
        d["R"].append(r["recall"]); d["P"].append(r["precision"])
        d["F1"].append(r["f1"]); d["DT"].append(r["dt"]); d["S"].append(r["success"])
        d["CC"].append(r["cc"]); d["FF"].append(r["false_flags"]); d["n"] += 1
    board = []
    for (c, p), d in agg.items():
        board.append({"contestant": c, "persona": p, "episodes": d["n"],
                      "recall": _mean(d["R"]), "precision": _mean(d["P"]),
                      "f1": _mean(d["F1"]), "dt": _mean(d["DT"]),
                      "success": _mean(d["S"]), "cc": _mean(d["CC"]),
                      "false_flags_per_ep": _mean(d["FF"])})
    order = {c: i for i, c in enumerate(CONTESTANTS)}
    porder = {p: i for i, p in enumerate(PERSONAS)}
    board.sort(key=lambda r: (order.get(r["contestant"], 9), porder.get(r["persona"], 9)))
    return board


def _slim(r):
    keep = ("contestant", "episode", "task", "persona", "n_errors", "recall",
            "precision", "f1", "dt", "success", "cc", "false_flags",
            "committed_any", "prevented", "goal_ok", "beats", "error",
            "typed_flags", "typed_right", "human_fixes", "fallback_repairs",
            "moves", "interface", "n_talk", "n_polls", "final_tick",
            "success_lenient", "extra_changes", "engine_rev", "grader", "human_model",
            "recall_engine", "precision_engine", "false_flags_engine")
    return {k: r.get(k) for k in keep if k in r}


def _print(board):
    print("\n" + "=" * 96)
    print(f"{'contestant':22} {'persona':20} {'R':>5} {'P':>5} {'F1':>5} "
          f"{'DT':>5} {'S':>5} {'CC':>5} {'FA/ep':>6}")
    print("-" * 96)
    for r in board:
        def f(x): return f"{x:.2f}" if isinstance(x, (int, float)) else "—"
        print(f"{r['contestant']:22} {r['persona']:20} {f(r['recall']):>5} "
              f"{f(r['precision']):>5} {f(r['f1']):>5} {f(r['dt']):>5} "
              f"{f(r['success']):>5} {f(r['cc']):>5} {f(r['false_flags_per_ep']):>6}")
    print("=" * 96)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--models", default="", help="comma list; default = full roster")
    ap.add_argument("--append", action="store_true",
                    help="merge into the results file (cell-wise)")
    ap.add_argument("--tasks", default="", help="comma list of task ids; default = all hard tasks")
    ap.add_argument("--no-manual", action="store_true", help="assistant WITHOUT the User Guide (ablation)")
    ap.add_argument("--out", default="", help="results file (default tmp/composites/composite_results_v2.json)")
    ap.add_argument("--interface", default="v2", choices=["v1", "v2"],
                    help="v2 = CookSim protocol (typed flags, human moves, LLM human); v1 = scripted human")
    ap.add_argument("--human-model", default="gemini-3.7-flash", help="the LLM that plays the person (v2)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N (case x persona) cells: a setup check")
    ap.add_argument("--shard", default="", help="i/n: run only slice i of n (1-based), e.g. 2/3")
    ap.add_argument("--cells", default="", help="json list of {contestant, episode, persona}: run only these")
    ap.add_argument("--episode", default="", help="select one designed case ID")
    ap.add_argument("--persona", default="", help="select one persona instead of the case's benchmark personas")
    ap.add_argument("--assistant-config", default="", help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help="list selected cells without calling models")
    args = ap.parse_args(argv)
    global NO_MANUAL, INTERFACE, HUMAN_MODEL, ASSISTANT_CONFIG
    NO_MANUAL = args.no_manual
    INTERFACE = args.interface
    HUMAN_MODEL = args.human_model
    ASSISTANT_CONFIG = json.loads(args.assistant_config) if args.assistant_config else {}
    contestants = args.models.split(",") if args.models else None
    tasks = args.tasks.split(",") if args.tasks else None
    default_out = "composite_results_v2.json" if INTERFACE == "v2" else "composite_results.json"
    try:
        return run(args.workers, args.out or os.path.join(OUT, default_out),
            contestants=contestants, append=args.append, tasks=tasks, limit=args.limit,
            shard=tuple(int(x) for x in args.shard.split("/")) if args.shard else None,
            cells=json.load(open(args.cells)) if args.cells else None,
            episode=args.episode or None, persona=args.persona or None, dry_run=args.dry_run)
    except (ValueError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
