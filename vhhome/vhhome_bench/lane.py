#!/usr/bin/env python3
"""One assistant's lane over the fixed benchmark cells.

The cells are benchmark/cells.json: 25 composite cases x 3 personas each, drawn once with a fixed
seed and shared by every arm, so the arms differ only in the assistant.

    python -m vhhome_bench.lane <tag> --root runs/mine [--engines 8080,8081] [--limit N]
                                      [--shard k/n] [--cells benchmark/cells.json]

Rules that come from the serving setup (CookSim docs/SETUP_ASSISTANTS.md §5.1):
  * a LOCAL model serves one episode at a time on its GPU -- its latency is charged to the game clock,
    so two episodes sharing the server would each be charged the other's compute. `--engines` with one
    port is the local shape; API arms may use several.
  * resume-safe: a cell whose report already exists is skipped, so a preempted job resumes.
"""
from vhhome import paths, keys
import argparse, json, os, signal, subprocess, sys, threading, time, types

from vhhome_bench import roster as roster_vh

BASE_ENV = {"VH_MIDLEVEL": "1", "STEPIN_WALL": "5400", "STEPIN_CAP": "520"}
# A STALL is the thing to kill, not a long episode. The first version of this used a 40-minute wall
# clock and killed qwen3.8-27b episodes that were still advancing (rc=-9 at exactly 2400.1 s, the log
# climbing tick by tick), losing 44 cells across two runs. What identifies a hung episode is that its log
# stops growing; what identifies a slow one is that it does not. So: kill after VH_EPISODE_STALL seconds
# of NO log growth, with a far looser absolute cap as the last resort.
EPISODE_STALL = int(os.environ.get("VH_EPISODE_STALL", "900"))        # 15 min with no new log line
EPISODE_TIMEOUT = int(os.environ.get("VH_EPISODE_TIMEOUT", "10800"))  # 3 h absolute, whatever happens
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cells(path, shard=None, limit=None):
    cells = json.load(open(path))["cells"]
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        cells = [c for i, c in enumerate(cells) if i % n == k - 1]
    return cells[:limit] if limit else cells


def key_env(env):
    """Keys the harness reads from files when the variable is absent (never printed)."""
    out = dict(env)
    keys.load_all()
    for var, path in ():
        p = os.path.expanduser(path)
        if not out.get(var) and os.path.exists(p):
            out[var] = open(p).read().strip()
    return out


def run_cell(entry, cell, port, root, log):
    name = f"{entry['tag']}__{cell['case']}__{cell['persona']}"
    out = os.path.join(root, entry["tag"], name + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip", name
    env = key_env({**BASE_ENV, **os.environ, **entry.get("env", {})})
    # unique frame directory per lane process: see iv_world.LiveWorld.__init__ (VH_RUN_TAG)
    env["VH_RUN_TAG"] = f"{entry['tag']}-{os.uname().nodename}-{os.getpid()}"
    cmd = [sys.executable, "-m", "vhhome_bench.episode", str(port), cell["case"], cell["persona"],
           out, "combo"]
    t0 = time.time()
    # A DEADLINE PER EPISODE. subprocess.run with no timeout let one wedged episode hold a lane for five
    # hours (gpt-realtime, 2026-09-22: two jobs stuck on one cell, the arm one episode short all day).
    # A normal episode is 2-15 minutes, so anything past the deadline is hung, not slow. The rollout is
    # started in its own session and the whole group is killed, because it has children (the engine
    # client, ffmpeg) that outlive a bare kill. The cell is then simply missing, which is what
    # vhhome_bench/cluster/fill_missing.py exists to pick up — a later job retries it on a fresh server.
    timed_out = False
    logp = out.replace(".json", ".log")
    with open(logp, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT_DIR, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        size, moved_at, why = -1, time.time(), ""
        while True:
            try:
                rc = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.time()
            cur = os.path.getsize(logp) if os.path.exists(logp) else -1
            if cur != size:                      # the episode is still writing: it is slow, not stuck
                size, moved_at = cur, now
            if now - moved_at > EPISODE_STALL:
                why = f"no output for {EPISODE_STALL}s"
            elif now - t0 > EPISODE_TIMEOUT:
                why = f"over the {EPISODE_TIMEOUT}s absolute cap"
            if why:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                rc = proc.wait()
                print(f"[lane] KILLED {name}: {why} — the cell stays missing for a later fill job",
                      flush=True)
                break
    r = types.SimpleNamespace(returncode=rc)
    ok = (not timed_out) and os.path.exists(out) and os.path.getsize(out) > 0
    if timed_out and os.path.exists(out) and os.path.getsize(out) == 0:
        os.remove(out)                      # never leave a stub that makes the cell look done
    # QUARANTINE (the rule tools/assist_sweep.py already uses): an assistant turn can be empty because the
    # MODEL wrote nothing parseable -- that is data -- or because the SERVER refused or never answered,
    # which measures the harness. Only the second voids a cell, and it is loud: a lane that keeps writing
    # cells whose calls were all transport failures is a lane of noise (round 1 of the Beaker checks).
    if ok:
        try:
            st = ((json.load(open(out)).get("assistant") or {}).get("stats") or {})
            calls = st.get("calls") or 0
            transport_dead = calls and (st.get("retries") or 0) and not (st.get("prompt_tokens") or 0)
            if transport_dead:
                void = os.path.join(os.path.dirname(out), "void")
                os.makedirs(void, exist_ok=True)
                os.replace(out, os.path.join(void, os.path.basename(out)))
                print(f"[lane] VOID {name}: every call was a transport failure "
                      f"(calls={calls}, retries={st.get('retries')}, prompt_tokens=0) — "
                      f"check {os.environ.get('LOGDIR', 'the serve log')}", flush=True)
                ok = False
        except Exception as e:
            print(f"[lane] could not read {name}'s assistant stats: {type(e).__name__}: {e}", flush=True)
    print(f"[{time.strftime('%H:%M')}] {'done' if ok else 'FAILED'} {name} "
          f"({time.time() - t0:.0f}s, rc={r.returncode})", flush=True)
    log.append({"cell": name, "ok": ok, "secs": round(time.time() - t0, 1), "rc": r.returncode})
    return ("done" if ok else "failed"), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--root", required=True)
    ap.add_argument("--cells", default=paths.CELLS)
    ap.add_argument("--engines", default="8080", help="comma-separated engine ports")
    ap.add_argument("--shard", default=None, help="k/n — run every n-th cell (Beaker sharding)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    entry = roster_vh.by_tag(a.tag)
    miss = roster_vh.missing_keys(entry)
    if miss:
        sys.exit(f"{a.tag}: missing {', '.join(miss)}")
    cells = load_cells(os.path.join(ROOT_DIR, a.cells), a.shard, a.limit)
    ports = [int(p) for p in a.engines.split(",")]
    if entry["kind"] == "local" and len(ports) > 1:
        sys.exit("a local model serves ONE episode at a time: give a single engine port")
    root = a.root if os.path.isabs(a.root) else os.path.join(ROOT_DIR, a.root)
    print(f"lane {a.tag}: {len(cells)} cells, engines {ports}, root {root}", flush=True)

    log, queue, lock = [], list(cells), threading.Lock()

    def worker(port):
        while True:
            with lock:
                if not queue:
                    return
                cell = queue.pop(0)
            try:
                run_cell(entry, cell, port, root, log)
            except Exception as e:                      # one bad cell must not end the lane
                print(f"[lane] {cell} raised {type(e).__name__}: {e}", flush=True)

    ts = [threading.Thread(target=worker, args=(p,), daemon=True) for p in ports]
    [t.start() for t in ts]
    [t.join() for t in ts]
    done = sum(1 for x in log if x["ok"])
    json.dump({"tag": a.tag, "cells": len(cells), "ran": len(log), "ok": done, "log": log},
              open(os.path.join(root, f"{a.tag}.lane.json"), "w"), indent=1)
    print(f"lane {a.tag} finished: {done}/{len(log)} ran ok "
          f"({len(cells) - len(log)} already present)", flush=True)
    # A LANE THAT PRODUCED NOTHING MUST NOT LOOK GREEN. The first Beaker check jobs reported SUCCEEDED
    # with 0 of 3 episodes written, because this returned 0 whatever happened.
    if log and not done:
        sys.exit(6)


if __name__ == "__main__":
    main()
