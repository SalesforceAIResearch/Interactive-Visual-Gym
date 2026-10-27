#!/usr/bin/env python3
"""Submit jobs for the cells a run is still missing — and only those.

    python -m vhhome_bench.cluster.fill_missing tmp/bench_v2 [--per-job 3] [--dry-run]

Why this exists. Re-running the ordinary submitter would create a SECOND job for shards that already
have one, and two jobs sharing a cell can write the same report at the same time. This reads the run's
output root instead, writes one cells file per arm holding exactly the missing (case, persona) pairs,
and submits jobs that carry `--cells <that file>`. Running it twice is harmless: the second time there is
nothing missing, so nothing is submitted.

Local (served) arms go to the GPU cluster, API and control arms to the CPU cluster with no resource
request, exactly as in a normal run.
"""
from vhhome import paths
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)

from vhhome_bench import roster as roster_vh  # noqa: E402


def missing_cells(root, tag, cells):
    out = []
    for c in cells:
        p = os.path.join(root, tag, f"{tag}__{c['case']}__{c['persona']}.json")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="run root, e.g. tmp/bench_v2")
    ap.add_argument("--cells", default=paths.CELLS)
    ap.add_argument("--per-job", type=int, default=3,
                    help="cells per worker (smaller = more workers, cheaper preemption)")
    # JUPITER FIRST (project decision 2026-09-22): its H100s are the fastest card we have and the cluster has
    # room, so a lane lands there when it can and the run finishes sooner. The card only changes what the
    # game clock is CHARGED, never the score: the job divides its generation time by that card's measured
    # ratio (docs/GPU_LATENCY_RATIO.json) and every arm is charged A100-equivalent seconds.
    ap.add_argument("--gpu-clusters", default=os.environ.get("VHHOME_BEAKER_GPU_CLUSTERS", ""),
                    help="clusters a served model may land on, best first (all >=80GiB cards)")
    ap.add_argument("--cpu-cluster", default=os.environ.get("VHHOME_BEAKER_CPU_CLUSTER", ""))
    ap.add_argument("--only", default=None, help="comma-separated tags, for trying one arm first")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cells = json.load(open(os.path.join(ENG, a.cells)))["cells"]
    root_abs = a.root if os.path.isabs(a.root) else os.path.join(ENG, a.root)
    fills = os.path.join(root_abs, "_fill"); os.makedirs(fills, exist_ok=True)
    total, bad = 0, []
    only = set(a.only.split(",")) if a.only else None
    for e in roster_vh.runnable():
        tag = e["tag"]
        if only and tag not in only:
            continue
        miss = missing_cells(root_abs, tag, cells)
        if not miss:
            continue
        f = os.path.join(fills, f"{tag}.json")
        json.dump({"note": f"cells missing from {a.root} for {tag}", "cells": miss}, open(f, "w"), indent=1)
        n_jobs = max(1, (len(miss) + a.per_job - 1) // a.per_job)
        rel = os.path.relpath(f, ENG)
        cmd = [sys.executable, "-m", "vhhome_bench.cluster.beaker_submit", "lane", tag, "--root", a.root,
               "--cells", rel, "--shards", str(n_jobs)]
        # ONE experiment per arm, n replicas: one row in the UI, n independent resumable workers.
        if e["kind"] == "local":
            cmd += ["--replicas"]
        cmd += (["--cluster", a.gpu_clusters] if e["kind"] == "local"
                else ["--cpu-lane", "--cluster", a.cpu_cluster])
        print(f"{tag:26} missing {len(miss):>3} cells -> {n_jobs} job(s)")
        total += len(miss)
        if not a.dry_run:
            r = subprocess.run(cmd, cwd=ENG, check=False)
            if r.returncode != 0:
                bad.append(f"{tag} (submitter exit {r.returncode})")
    print(f"\n{a.root}: {total} cells missing"
          + ("" if a.dry_run else (f" — fill jobs submitted" if not bad else
                                   f" — SUBMISSION FAILED for {', '.join(bad)}; nothing will run for them")))
    if bad:
        sys.exit(7)


if __name__ == "__main__":
    main()
