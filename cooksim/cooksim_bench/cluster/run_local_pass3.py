#!/usr/bin/env python3
"""Local-model pass@3 run (user-authorised 2026-09-15): the SAME 10 (persona, case) episodes as the v5
sample rounds (python -m cooksim_bench.cluster.run_v5_sample EPISODES), three repeats each, for one locally served assistant.

Episodes run ONE AT A TIME. The assistant's latency is charged to the game clock
(COOKSIM_ASSIST_LATENCY=measured), so two episodes sharing one GPU would each be charged the other's
compute. One lane = one model = one GPU = one episode in flight.

    $PY python -m cooksim_bench.cluster.run_local_pass3 --tag qwen3.8-27b --repeats 3 -- \
        --assist-backend openai --assist-model qwen-27b --assist-base-url http://127.0.0.1:18300/v1 ...

Everything after `--` is passed to python -m cooksim_bench.e2e_stepin_rollout unchanged. Output layout mirrors the
sweeps so the scorers work unchanged, one tag per repeat:
    <V5_SAMPLE_ROOT>/frames/<persona>/<tag>__r<k>/<case>.json
The engine commit, a dirty-tree flag and the model flags are written next to the run as run_meta.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from cooksim_bench.paths import ROOT  # the CookSim release root
from cooksim_bench.cluster.run_v5_sample import EPISODES, CASES          # noqa: E402  the one definition of the 10 episodes


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--root", default=os.environ.get("V5_SAMPLE_ROOT", "tmp/assist_bench_local_pass3"))
    ap.add_argument("--logs", default=os.path.join(ROOT, "tmp", "local_pass3_logs"))
    # API assistants only: several episodes at once. A LOCAL model must keep 1 -- its latency is charged, and
    # concurrent requests on one GPU would charge each episode the other's compute.
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    out_root = os.path.join(ROOT, a.root, "frames")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    meta_path = os.path.join(ROOT, a.root, f"run_meta_{a.tag}.json")
    meta = {"tag": a.tag, "repeats": a.repeats, "episodes": EPISODES, "cases": CASES, "rollout_args": extra,
            "engine_commit": git("rev-parse", "HEAD"), "engine_dirty": bool(git("status", "--porcelain", "--", "tools", "cooksim")),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(meta_path, "w"), indent=1)
    env = dict(os.environ)
    env.update(COOKSIM_SCHED="v5", COOKSIM_CASES=CASES, COOKSIM_PACE="tight", COOKSIM_ASSIST_LATENCY="measured",
               COOKSIM_NO_VIDEO="0", COOKSIM_DUMP_PER_EPISODE="1", GEMINI_MODEL=env.get("GEMINI_MODEL", "gemini-3.7-flash"))
    meta["workers"] = a.workers
    json.dump(meta, open(meta_path, "w"), indent=1)
    # repeats interleaved by round (r1 of all 10, then r2, then r3): a run cut short still has whole rounds
    jobs = []
    for k in range(1, a.repeats + 1):
        for persona, case in EPISODES:
            outd = os.path.join(out_root, persona, f"{a.tag}__r{k}")
            os.makedirs(outd, exist_ok=True)
            out = os.path.join(outd, case + ".json")
            if not os.path.exists(out):
                jobs.append((k, persona, case, out))

    def launch(job):
        k, persona, case, out = job
        e = dict(env)
        e["COOKSIM_DUMP_STATES"] = out.replace(".json", ".states.jsonl")
        log = os.path.join(a.logs, f"{a.tag}__r{k}__{persona}__{case}.log")
        cmd = [sys.executable, os.path.join(ROOT, "python -m cooksim_bench.e2e_stepin_rollout"), "--case", case,
               "--persona-v2", persona, "--paradigm", "hardsoft", "--sup-obs", "frames", "--out", out] + extra
        print(f"[{time.strftime('%H:%M:%S')}] {a.tag} r{k} {persona}/{case} ...", flush=True)
        return subprocess.Popen(cmd, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=e), time.time()

    running = []
    while jobs or running:
        while jobs and len(running) < max(1, a.workers):
            job = jobs.pop(0)
            p, t0 = launch(job)
            running.append((job, p, t0))
        time.sleep(2)
        for item in list(running):
            job, p, t0 = item
            if p.poll() is not None:
                running.remove(item)
                out = job[3]
                print(f"    rc={p.returncode} {time.time() - t0:.0f}s r{job[0]} {job[1]}/{job[2]} -> "
                      f"{out if os.path.exists(out) else 'NO RECORD'}", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(meta, open(meta_path, "w"), indent=1)


if __name__ == "__main__":
    main()
