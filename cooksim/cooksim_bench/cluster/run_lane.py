#!/usr/bin/env python3
"""Run one roster assistant on the standard episode set, by its roster tag (python -m cooksim_bench.roster).

    source tmp/present/env.sh; export OPENAI_API_KEY=$(cat ~/.openai_key)
    $PY python -m cooksim_bench.cluster.run_lane gemini-3.8-flash                       # 10 episodes x 3 repeats, 4 at a time
    $PY python -m cooksim_bench.cluster.run_lane qwen3.8-27b --repeats 1                # local: start serving/serve_qwen.sh first
    $PY python -m cooksim_bench.cluster.run_lane gpt-6-astra --root tmp/my_round --episode-set r5

The flags come from the roster, so a lane can never drift from the documented setup. Episodes, env and the
output layout are python -m cooksim_bench.cluster.run_local_pass3's: <root>/frames/<persona>/<tag>__r<k>/<case>.json. A LOCAL model
runs one episode at a time (its latency is charged to the game clock); an API model runs --workers at once.
"""
import argparse
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

from cooksim_bench.roster import by_tag, missing_keys      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--root", default=os.environ.get("V5_SAMPLE_ROOT", "tmp/assist_bench_lane"))
    ap.add_argument("--workers", type=int, default=4, help="API assistants only; a local model always runs 1")
    ap.add_argument("--episode-set", default=os.environ.get("V5_EPISODE_SET", "r4"), choices=["r4", "r5"])
    ap.add_argument("--no-warmup", action="store_true", help="skip warming a local server (python -m cooksim_bench.backends.warmup_server)")
    a = ap.parse_args()
    e = by_tag(a.tag)
    if not e["flags"]:
        sys.exit(f"{a.tag}: no backend yet ({e['status']})")
    if missing_keys(e):
        sys.exit(f"{a.tag}: missing {', '.join(missing_keys(e))}")
    workers = a.workers
    if e["kind"] == "local":
        workers = 1
        url = f"http://127.0.0.1:{e['port']}/v1/models"
        try:
            urllib.request.urlopen(url, timeout=3)
        except Exception:
            sys.exit(f"{a.tag}: nothing is serving on port {e['port']} -- start serving/{e['serve']} first")
        if not a.no_warmup:                      # never let a cold server's first calls land in episode 1
            subprocess.call([sys.executable, "-m", "cooksim_bench.backends.warmup_server", a.tag], cwd=os.path.dirname(HERE))
    env = dict(os.environ, V5_EPISODE_SET=a.episode_set)
    cmd = [sys.executable, "-m", "cooksim_bench.cluster.run_local_pass3", "--tag", a.tag, "--repeats", str(a.repeats),
           "--root", a.root, "--workers", str(workers), "--"] + e["flags"]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=os.path.dirname(HERE), env=env))


if __name__ == "__main__":
    main()
