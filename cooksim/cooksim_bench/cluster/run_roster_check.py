#!/usr/bin/env python3
"""End-to-end check of the roster: every assistant plays the same 3 episodes (python -m cooksim_bench.roster CHECK_EPISODES)
on the current engine, so a broken setup shows up as a broken episode rather than as a bad score later.

    source tmp/present/env.sh; export OPENAI_API_KEY=$(cat ~/.openai_key)
    $PY python -m cooksim_bench.cluster.run_roster_check                       # everything runnable (keys present)
    $PY python -m cooksim_bench.cluster.run_roster_check --tags gpt-6-astra minicpmo
    $PY python -m cooksim_bench.cluster.run_roster_check --gpus 0 1            # local models: one lane per GPU

API and native assistants run concurrently (up to --api-workers episodes at once). A LOCAL model gets its own
server, started from serving/<serve script> under the GPU's lock (/tmp/gpu<N>.lock), waited on until
/v1/models answers, used for its 3 episodes ONE AT A TIME (its latency is charged to the game clock), then
stopped. Each lane has its own renderer so Chromium never waits on another lane.

Output: <root>/frames/<persona>/<tag>/<case>.json (+ .states.jsonl, .mp4), the scorers' layout, plus
<root>/check_meta.json (engine commit, flags per tag). Summarise with python -m cooksim_bench.cluster.roster_check_report.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from cooksim_bench.paths import ROOT  # the CookSim release root
from cooksim_bench.roster import ROSTER, CHECK_EPISODES, missing_keys, by_tag      # noqa: E402

CASES = "tmp/interact_eval/cases_composite_v4.json"
PY = os.environ.get("PY", sys.executable)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def up(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def start_renderer(port, logdir):
    page = f"http://127.0.0.1:{port}/static/headless_render.html"
    if up(page):
        return page, None
    cmd = f". ./gpu_env.sh >/dev/null 2>&1; exec {PY} -m cooksim.server.app --port {port}"
    p = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, stdout=open(os.path.join(logdir, f"render_{port}.log"), "w"),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(120):
        if up(page):
            return page, p
        time.sleep(1)
    raise RuntimeError(f"renderer on {port} did not come up (see render_{port}.log)")


def episode(entry, persona, case, out_root, logdir, page):
    outd = os.path.join(out_root, "frames", persona, entry["tag"])
    os.makedirs(outd, exist_ok=True)
    out = os.path.join(outd, case + ".json")
    if os.path.exists(out):
        return out, 0
    env = dict(os.environ, COOKSIM_SCHED="v5", COOKSIM_CASES=CASES, COOKSIM_PACE="tight",
               COOKSIM_ASSIST_LATENCY="measured", COOKSIM_NO_VIDEO="0", COOKSIM_DUMP_PER_EPISODE="1",
               COOKSIM_DUMP_STATES=out.replace(".json", ".states.jsonl"), COOKSIM_PAGE=page,
               GEMINI_MODEL=os.environ.get("GEMINI_MODEL", "gemini-3.7-flash"))
    cmd = [PY, "python -m cooksim_bench.e2e_stepin_rollout", "--case", case, "--persona-v2", persona, "--paradigm", "hardsoft",
           "--sup-obs", "frames", "--out", out] + entry["flags"]
    lf = os.path.join(logdir, f"{entry['tag']}__{persona}__{case}.log")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=open(lf, "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
    log(f"{entry['tag']} {persona}/{case}: rc={rc} in {time.time() - t0:.0f}s")
    return out, rc


def api_lane(entries, out_root, logdir, page, workers):
    jobs = [(e, p, c) for e in entries for p, c in CHECK_EPISODES]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                e, p, c = jobs.pop(0)
            episode(e, p, c, out_root, logdir, page)
    ts = [threading.Thread(target=worker) for _ in range(max(1, workers))]
    [t.start() for t in ts]
    [t.join() for t in ts]


def local_lane(gpu, entries, out_root, logdir, page):
    for e in entries:
        script = os.path.join(ROOT, "serving", e["serve"])
        slog = os.path.join(logdir, f"serve_{e['tag']}.log")
        log(f"GPU {gpu}: starting {e['tag']} ({e['serve']}, port {e['port']})")
        cmd = f"exec flock -w 10800 /tmp/gpu{gpu}.lock bash {script}"
        srv = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, stdout=open(slog, "w"), stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GPU=str(gpu), PORT=str(e["port"])))
        ready = False
        for _ in range(2400):                                    # weights can take a while from network disk
            if srv.poll() is not None:
                break
            if up(f"http://127.0.0.1:{e['port']}/v1/models"):
                ready = True
                break
            time.sleep(1.5)
        if not ready:
            log(f"GPU {gpu}: {e['tag']} server did NOT come up (see {slog}); skipping its episodes")
        else:
            log(f"GPU {gpu}: {e['tag']} is serving")
            # warm the fresh server identically in every setting before any timed call (python -m cooksim_bench.backends.warmup_server)
            wlog = os.path.join(logdir, f"warmup_{e['tag']}.log")
            subprocess.call([PY, "python -m cooksim_bench.backends.warmup_server", e["tag"]], cwd=ROOT, stdout=open(wlog, "w"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            log(f"GPU {gpu}: {e['tag']} warmed ({open(wlog).read().strip().splitlines()[-1] if os.path.getsize(wlog) else 'no output'})")
            for p, c in CHECK_EPISODES:
                episode(e, p, c, out_root, logdir, page)
        try:
            os.killpg(srv.pid, signal.SIGTERM)                  # our own process group only
            srv.wait(timeout=120)
        except Exception:
            try:
                os.killpg(srv.pid, signal.SIGKILL)
            except Exception:
                pass
        log(f"GPU {gpu}: {e['tag']} stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--root", default="tmp/roster_check")
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--api-workers", type=int, default=6)
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    out_root = os.path.join(ROOT, a.root)
    logdir = os.path.join(out_root, "logs")
    os.makedirs(logdir, exist_ok=True)
    chosen = [by_tag(t) for t in a.tags] if a.tags else [r for r in ROSTER if r["status"] != "blocked"]
    runnable, skipped = [], []
    for e in chosen:
        if not e["flags"]:
            skipped.append((e["tag"], "no backend flags yet"))
        elif missing_keys(e):
            skipped.append((e["tag"], "missing " + ", ".join(missing_keys(e))))
        else:
            runnable.append(e)
    for t, why in skipped:
        log(f"skip {t}: {why}")
    api = [e for e in runnable if e["kind"] in ("api", "native")] if not a.no_api else []
    local = [e for e in runnable if e["kind"] == "local"] if not a.no_local else []
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    json.dump({"engine_commit": commit, "episodes": CHECK_EPISODES, "skipped": skipped,
               "flags": {e["tag"]: e["flags"] for e in runnable},
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(os.path.join(out_root, "check_meta.json"), "w"), indent=1)
    threads, renderers = [], []
    if api:
        page, rp = start_renderer(17774, logdir); renderers.append(rp)
        threads.append(threading.Thread(target=api_lane, args=(api, out_root, logdir, page, a.api_workers)))
    lanes = {g: [] for g in a.gpus}
    for i, e in enumerate(local):                                 # spread local models over the GPUs
        lanes[a.gpus[i % len(a.gpus)]].append(e)
    for k, (g, es) in enumerate(lanes.items()):
        if es:
            page, rp = start_renderer(17780 + k, logdir); renderers.append(rp)
            threads.append(threading.Thread(target=local_lane, args=(g, es, out_root, logdir, page)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    for rp in renderers:
        if rp is not None:
            try:
                os.killpg(rp.pid, signal.SIGTERM)
            except Exception:
                pass
    log("done; summarise with: $PY python -m cooksim_bench.cluster.roster_check_report " + a.root)


if __name__ == "__main__":
    main()
