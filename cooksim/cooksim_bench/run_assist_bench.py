#!/usr/bin/env python3
"""Run the composite assistant benchmark: 50 episodes x one assistant model (2026-08-19).

One process per episode (the rollout owns a playwright page); N workers in parallel.
Resume-safe: an episode whose output JSON already exists is skipped, so the same
command continues a partial sweep. Logs go to LOCAL disk (weka's wedged-dir hazard).

    $PY python -m cooksim_bench.run_assist_bench --tag gemini-3.7-flash \\
        --assist-backend gemini --assist-model gemini-3.7-flash [--workers 4]
    $PY python -m cooksim_bench.run_assist_bench --tag reference --workers 4        # shared client seat
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
def finished(path):
    """A report counts as done only when it is a complete JSON record with an outcome. A truncated or
    empty file (a job preempted mid-write, before reports were written atomically) is removed so the
    episode is re-run instead of silently skipped (2026-09-30)."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            rec = json.load(f)
        if isinstance(rec, dict) and "outcome" in rec:
            return True
    except Exception:                       # noqa: BLE001
        pass
    print(f"[resume] {path} is not a complete report -- removing it so the episode re-runs", flush=True)
    try:
        os.remove(path)
    except OSError:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--assist-backend", default=None)
    ap.add_argument("--assist-model", default=None)
    ap.add_argument("--assist-base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--assist-profile", default="default")
    ap.add_argument("--assist-max-tokens", type=int, default=4000)
    ap.add_argument("--assist-image-scale", type=float, default=1.0)
    # ---- COOK seat (cook-model ablation 2026-09-22): optional, default = shared Gemini ----
    ap.add_argument("--cook-backend", default=None)
    ap.add_argument("--cook-model", default=None)
    ap.add_argument("--cook-base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--cook-profile", default="default")
    ap.add_argument("--cook-max-tokens", type=int, default=4000)
    ap.add_argument("--sup-obs", default="frames")
    ap.add_argument("--paradigm", default="hardsoft")
    ap.add_argument("--persona-v2", nargs="+", default=["baseline"],
                    help="one or more persona names; each gets its own out dir <out-root>/<persona>/<tag>")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--cells-file", default=None,
                    help="JSON {\"cells\": [[case, persona], ...]} -- run EXACTLY these "
                         "(case, persona) cells instead of the --cases x --persona-v2 cross "
                         "product. The fixed 30-episode ablation subset is one of these files")
    ap.add_argument("--out-root", default="runs/my_run")
    ap.add_argument("--log-dir", default="/tmp/assist_bench_logs")
    ap.add_argument("--stagger", type=float, default=8.0)
    ap.add_argument("--shard", type=int, default=0,
                    help="this runner's index within --num-shards; the (persona, case) queue is split "
                         "deterministically so several jobs can work one seat without overlapping")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--reverse", action="store_true",
                    help="work the queue from the end (a second runner on the same cell)")
    args = ap.parse_args()

    pool = json.load(open(os.path.join(ROOT, os.environ.get("COOKSIM_CASES", os.path.join(ROOT, "benchmark", "cases_composite_v5.json")))))   # COOKSIM_CASES picks the suite (v3: tmp/interact_eval/cases_composite_v3.json)
    names = args.cases or [c["name"] for c in pool["cases"]]
    os.makedirs(args.log_dir, exist_ok=True)
    # the cell list: normally every (persona, episode), interleaved by episode so all personas
    # progress together; with --cells-file, exactly the cells that file names, in its order
    # (the cook-model ablation draws 30 of the 150 cells and every cook runs that same list)
    if args.cells_file:
        cf = json.load(open(args.cells_file if os.path.isabs(args.cells_file)
                            else os.path.join(ROOT, args.cells_file)))
        full = [(str(p_), str(n_)) for n_, p_ in cf["cells"]]
    else:
        full = [(p_, n_) for n_ in names for p_ in args.persona_v2]
    jobs = []
    for persona, n in full:
        outd = os.path.join(ROOT, args.out_root, persona, args.tag)
        os.makedirs(outd, exist_ok=True)
        if not finished(os.path.join(outd, n + ".json")):
            jobs.append((persona, n, outd))
    # the shard is taken over the FULL (persona, case) list, not over what is missing, so a job keeps
    # its own slice across a preemption/retry -- otherwise two shards would race for the same episode
    if args.num_shards > 1:
        mine = {full[i] for i in range(args.shard, len(full), args.num_shards)}
        jobs = [j for j in jobs if (j[0], j[1]) in mine]
    todo = list(reversed(jobs)) if args.reverse else jobs
    print(f"[{args.tag}] {len(todo)} to run of {len(full)} "
          f"(personas {args.persona_v2}"
          + (f", shard {args.shard + 1}/{args.num_shards}" if args.num_shards > 1 else "") + ")", flush=True)
    running = []
    while todo or running:
        running = [(n, p) for n, p in running if p.poll() is None]
        while todo and len(running) < args.workers:
            persona, n, outd = todo.pop(0)
            if finished(os.path.join(outd, n + ".json")):
                continue                     # another runner on this cell finished it meanwhile
            cmd = [sys.executable, "-m", "cooksim_bench.e2e_stepin_rollout",
                   "--case", n, "--persona-v2", persona,
                   "--paradigm", args.paradigm, "--sup-obs", args.sup_obs,
                   "--out", os.path.join(outd, n + ".json")]
            if args.cook_backend:
                cmd += ["--cook-backend", args.cook_backend,
                        "--cook-model", args.cook_model,
                        "--cook-base-url", args.cook_base_url,
                        "--cook-profile", args.cook_profile,
                        "--cook-max-tokens", str(args.cook_max_tokens)]
            if args.assist_backend:
                cmd += ["--assist-backend", args.assist_backend,
                        "--assist-model", args.assist_model,
                        "--assist-base-url", args.assist_base_url,
                        "--assist-profile", args.assist_profile,
                        "--assist-max-tokens", str(args.assist_max_tokens),
                        "--assist-image-scale", str(args.assist_image_scale)]
            log = open(os.path.join(args.log_dir, f"{args.tag}__{persona}__{n}.log"), "w")
            env = dict(os.environ)
            if os.environ.get("COOKSIM_DUMP_PER_EPISODE") == "1":   # suite v3 (2026-08-27): exact per-tick kitchen state for the utterance grader
                env["COOKSIM_DUMP_STATES"] = os.path.join(outd, n + ".states.jsonl")
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, env=env)
            running.append((n, p))
            print(f"[{args.tag}] launched {persona}/{n} (pid {p.pid}); "
                  f"{len(todo)} queued", flush=True)
            time.sleep(args.stagger)     # stagger playwright/photoreal startup
        time.sleep(10)
    done = sum(1 for persona, n, outd in jobs if finished(os.path.join(outd, n + ".json")))
    print(f"[{args.tag}] sweep ended: {done}/{len(jobs)} queued episodes on disk", flush=True)


if __name__ == "__main__":
    main()
