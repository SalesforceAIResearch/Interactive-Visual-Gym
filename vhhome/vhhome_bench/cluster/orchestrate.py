#!/usr/bin/env python3
"""The daemon that carries out vhhome_bench/cluster/plan.py, phase by phase.

    python -m vhhome_bench.cluster.orchestrate init                 # write the initial state (once)
    setsid nohup python -m vhhome_bench.cluster.orchestrate daemon >> tmp/bench2/daemon.out 2>&1 < /dev/null &
    python -m vhhome_bench.cluster.orchestrate status               # what is where
    python -m vhhome_bench.cluster.orchestrate approve bench-r2     # the go-ahead for a later run
    python -m vhhome_bench.cluster.orchestrate render               # rewrite + push the page now

State lives in tmp/bench2/state.json (the daemon is its only writer); every runner is resume-safe (a cell with a
record is skipped), so the daemon can be killed and restarted at any point.
"""
from vhhome import keys
import json, math, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

from vhhome_bench.cluster import plan as P  # noqa: E402
from vhhome_bench.cluster import verify_phase as V  # noqa: E402

STATE = os.path.join(P.STATE_DIR, "state.json")
LOGS = os.path.join(P.STATE_DIR, "logs")
TERMINAL = {"succeeded", "failed", "canceled", "cancelled", "stopped", "preempted"}
CELLS = json.load(open(os.path.join(P.ENG, P.CELLS)))["cells"]


def now():
    return int(time.time())


def load():
    return json.load(open(STATE))


def save(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def log(st, msg):
    line = time.strftime("%m-%d %H:%M", time.gmtime(now() - 7 * 3600)) + " PT  " + msg
    st.setdefault("log", []).append(line)
    del st["log"][:-400]
    print(line, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def count(root, tag, cells=None):
    return sum(1 for _, f in V.records(root, tag, cells or CELLS) if f)


def alive(pid, needle):
    try:
        return needle in open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except Exception:
        return False


def spawn(cmd, logname, env=None):
    os.makedirs(LOGS, exist_ok=True)
    lf = open(os.path.join(LOGS, logname), "a")
    e = dict(os.environ); e.update(env or {})
    return subprocess.Popen(cmd, cwd=P.ENG, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=e).pid


# ------------------------------------------------------------------ local engines
def engine_up(port):
    return sh(f"curl -s -m 3 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/").stdout.strip() == "501"


def engines_needed(st):
    """Local engines only serve API seats run in the session; once every such seat is done (or on Beaker) there is
    nothing for twenty Unity instances to do — and this container may have no GPU and a different CPU budget."""
    for ph in P.phases():
        if ph["kind"] != "bench":
            continue
        s = st["phases"].get(ph["id"], {})
        if s.get("status") in ("done", "blocked"):
            continue
        lane = s.get("api") or {t: {"status": "pending"} for t in ph["api"]}
        for t, x in lane.items():
            if x.get("status") not in ("done", "failed") and not on_beaker(ph, t):
                return True
    return False


def ensure_engines(st):
    if not engines_needed(st):
        return True
    down = [p for p in range(8080, 8080 + P.N_ENGINES) if not engine_up(p)]
    seen = st.setdefault("engines_down", {})
    for p in list(seen):
        if int(p) not in down:
            seen.pop(p)
    for p in down:
        k = str(p)
        seen[k] = seen.get(k, 0) + 1
        if seen[k] == 1 or seen[k] % 8 == 0:          # relaunch at once, then again every ~8 loops if still down
            spawn(["bash", "simulator/launch_simulators.sh", str(P.N_ENGINES)], f"engine_{p}.log", env={"ONLY_INDEX": str(p - 8080)})
            log(st, f"engine :{p} is down — relaunching")
    return not down


# ------------------------------------------------------------------ beaker
def beaker():
    if not os.environ.get("BEAKER_TOKEN"):
        os.environ["BEAKER_TOKEN"] = keys.get("BEAKER_TOKEN")
    from beaker import Beaker
    Beaker.TIMEOUT = 120.0
    return Beaker.from_env(default_workspace=P.BEAKER["workspace"])


def submit(mode, tag, root, replicas=None, cells=None, cluster=None, cpu_lane=False):
    b = P.BEAKER
    cmd = [P.PY, "-m", "vhhome_bench.cluster.beaker_submit", mode, tag, "--root", root, "--workspace", b["workspace"], "--budget", b["budget"],
           "--cluster", cluster or b["cluster"], "--suffix", "b2"]
    if cpu_lane:                                       # an API or silent seat: no GPU, no sized resources
        cmd.append("--cpu-lane")
    for kv in b["env"]:
        cmd += ["--env", kv]
    if replicas and replicas > 1:
        cmd += ["--shards", str(replicas), "--replicas"]
    if cells:
        cmd += ["--cells", cells]
    beaker()                                           # makes sure BEAKER_TOKEN is in the environment
    r = subprocess.run(cmd, cwd=P.ENG, capture_output=True, text=True)
    m = re.search(r"beaker\.org/ex/(\w+)", r.stdout or "")
    return (m.group(1) if m else None), ((r.stdout or "") + (r.stderr or ""))[-300:]


def _status_name(j):
    s = j.status
    ev = type(s).DESCRIPTOR.fields_by_name["status"].enum_type.values_by_number.get(s.status)
    return (ev.name if ev else str(s.status)).replace("STATUS_", "").lower()


def inspect(exp_id):
    """({status: n} over each task's LATEST job, any job still live?)"""
    try:
        b = beaker()
        wl = b.workload.get(exp_id)
        summ, live = {}, False
        for t in list(getattr(wl.experiment, "tasks", [])):
            jobs = sorted(b.job.list(task=t, limit=50), key=lambda j: getattr(j.status.created, "seconds", 0))
            if not jobs:
                live = True
                continue
            k = _status_name(jobs[-1])
            summ[k] = summ.get(k, 0) + 1
            live = live or k not in TERMINAL
        return summ, live
    except Exception as ex:                             # noqa: BLE001
        return {"inspect_error": repr(ex)[:80]}, True


def cancel(exp_id):
    try:
        b = beaker()
        b.workload.cancel(b.workload.get(exp_id))
    except Exception:
        pass


def track_exp(st, s, every=300):
    """refresh s['jobs'] / s['live'] at most every `every` seconds"""
    if s.get("exp") and now() - s.get("inspected", 0) >= every:
        s["jobs"], s["live"] = inspect(s["exp"])
        s["inspected"] = now()
        if s["live"]:
            s.pop("dead_since", None)
        else:
            s.setdefault("dead_since", now())


# ------------------------------------------------------------------ preflight
def rec_path(root, tag, case, persona):
    return os.path.join(P.ENG, root, tag, f"{tag}__{case}__{persona}.json")


def preflight(st):
    pf = st.setdefault("preflight", {"local": {}, "gpu": {}})
    ph = st["phases"].setdefault("preflight", {"status": "running", "started": now()})
    if ph["status"] in ("done", "blocked"):
        return
    # --- local: three episodes on the run's stamp
    ports = [P.SLOTS[0][0], P.SLOTS[0][1], P.SLOTS[1][0]]
    for i, (tag, case, persona) in enumerate(P.PREFLIGHT_LOCAL):
        key = f"{tag} · {case} · {persona}"
        s = pf["local"].setdefault(key, {"status": "pending"})
        if s["status"] in ("pass", "fail"):
            continue
        f = rec_path(P.PREFLIGHT_ROOT, tag, case, persona)
        if os.path.exists(f) and os.path.getsize(f) > 0:
            d = json.load(open(f))
            probs = []
            if d.get("engine_rev") != st["stamp"]:
                probs.append(f"stamp {d.get('engine_rev')}")
            phs = d.get("phases") or []
            if not any(p.get("dev_live") is not None for p in phs):
                probs.append("no designed mistake fired")
            if tag == "silent" and d.get("outcome") == "won":
                probs.append("the silent episode was WON (self-repair)")
            spoke = any(c.get("speaker") == "vlm" and (c.get("text") or "").strip() for c in d.get("conversation") or [])
            if tag != "silent" and not spoke:
                probs.append("the assistant never spoke")
            s["status"] = "fail" if probs else "pass"
            s["note"] = (f"outcome {d.get('outcome')} at t{d.get('ticks')}, "
                         f"{sum(1 for p in phs if p.get('dev_live') is not None)}/{len(phs)} mistakes fired"
                         + ("; " + "; ".join(probs) if probs else ""))
            log(st, f"preflight local {key}: {s['status']} — {s['note']}")
        elif not (s.get("pid") and alive(s["pid"], "run_lane")):
            if s.get("starts", 0) >= 3:
                s["status"], s["note"] = "fail", "no record after 3 attempts"
                continue
            if not engine_up(ports[i]):
                s["note"] = f"waiting for engine :{ports[i]}"
                continue
            cf = os.path.join(P.STATE_DIR, f"preflight_cell_{i}.json")
            json.dump({"cells": [{"case": case, "persona": persona}]}, open(cf, "w"))
            s["pid"] = spawn([P.PY, "-m", "vhhome_bench.lane", tag, "--root", P.PREFLIGHT_ROOT, "--cells", cf, "--engines", str(ports[i])],
                             f"preflight_local_{i}.log")
            s["starts"] = s.get("starts", 0) + 1
            s["status"], s["note"] = "running", f"engine :{ports[i]}"
    # --- gpu: one 3-episode check job per open-weight seat
    for tag in P.gpu_seats():
        s = pf["gpu"].setdefault(tag, {"status": "pending", "submits": 0})
        if s["status"] in ("pass", "failed"):
            continue
        cells3 = CELLS[:3]
        n = count(P.PREFLIGHT_ROOT, tag, cells3)
        if n == 3:
            v = V.check_seat(P.PREFLIGHT_ROOT, tag, cells3, st["stamp"], True)
            s["status"] = "failed" if v["problems"] else "pass"
            s["note"] = (f"3/3 records, outcomes {v['outcomes']}, assistant spoke in {v['spoke']}/3"
                         + ("; " + "; ".join(v["problems"]) if v["problems"] else "; H100, factor 1.0"))
            log(st, f"preflight gpu {tag}: {s['status']} — {s['note']}")
            continue
        if not s.get("exp"):
            s["exp"], out = submit("check", tag, P.PREFLIGHT_ROOT)
            s["submits"] += 1
            s["status"] = "submitted" if s["exp"] else "pending"
            s["note"] = "" if s["exp"] else f"submission failed: {out[-120:]}"
            log(st, f"preflight gpu {tag}: check job {s['exp'] or 'NOT created'}")
            continue
        track_exp(st, s)
        s["note"] = f"{n}/3 records"
        if not s.get("live") and now() - s.get("dead_since", now()) > 900:
            if s["submits"] >= 3:
                s["status"], s["note"] = "failed", f"{n}/3 records after 3 check jobs; last {s.get('jobs')}"
                log(st, f"preflight gpu {tag}: FAILED — {s['note']}")
            else:
                log(st, f"preflight gpu {tag}: job ended with {n}/3 records ({s.get('jobs')}) — submitting again")
                s.pop("exp", None); s.pop("dead_since", None)
    loc = [v["status"] for v in pf["local"].values()]
    gpu = [v["status"] for v in pf["gpu"].values()]
    ph["local"] = "pass" if loc and all(x == "pass" for x in loc) else ("fail" if "fail" in loc else "running")
    if ph["local"] == "fail":
        ph["status"] = "blocked"
        log(st, "PREFLIGHT BLOCKED: a local episode failed its check — nothing further is submitted")
    elif ph["local"] == "pass" and all(x in ("pass", "failed") for x in gpu):
        ph["status"], ph["done_at"] = "done", now()
        bad = [t for t, v in pf["gpu"].items() if v["status"] == "failed"]
        log(st, "preflight done" + (f" — seats that failed their check and will NOT be submitted: {bad}" if bad else " — every seat passed"))


# ------------------------------------------------------------------ f124 -> f125: re-run what the old guard ended
_SEEN = set()


def supersede_old_guard(st):
    """Move every f124 record with ended == "dead_conversation" out of its seat folder, so the cell is re-run on
    f125 (the softened guard). Records the old guard never touched play out identically and stay."""
    import glob, shutil
    moved = {}
    for ph in P.phases():
        if ph["kind"] != "bench":
            continue
        for f in glob.glob(os.path.join(P.ENG, ph["root"], "*", "*__*.json")):
            if f in _SEEN or os.path.getmtime(f) > time.time() - 90:
                continue
            try:
                d = json.load(open(f))
            except Exception:
                continue
            _SEEN.add(f)
            if d.get("ended") == "dead_conversation" and d.get("engine_rev") == "2026-10-05.f124":
                tag = os.path.basename(os.path.dirname(f))
                dst = os.path.join(P.ENG, ph["root"], "_superseded_f124", tag)
                os.makedirs(dst, exist_ok=True)
                for ext in (".json", ".log", ".mp4"):
                    if os.path.exists(f[:-5] + ext):
                        shutil.move(f[:-5] + ext, os.path.join(dst, os.path.basename(f)[:-5] + ext))
                moved[(ph["id"], tag)] = moved.get((ph["id"], tag), 0) + 1
    for (pid_, tag), n in sorted(moved.items()):
        log(st, f"{pid_} {tag}: {n} f124 record(s) ended by the old dead-conversation guard set aside for a re-run")


# ------------------------------------------------------------------ benchmark phase
def on_beaker(ph, t):
    return ph["id"] in P.API_BEAKER["phases"] and t in P.API_BEAKER["seats"]


def api_beaker_seat(st, ph, t, x):
    """An API seat that runs as Beaker CPU jobs instead of on the session's engines (P.API_BEAKER). Same life cycle
    as a GPU seat: one replica experiment, records counted on the shared filesystem, a fill for whatever is missing when it ends."""
    ab = P.API_BEAKER
    x["where"] = "beaker"
    x.setdefault("submits", 0)
    x.pop("pid", None); x.pop("ports", None)
    x["n"] = count(ph["root"], t)
    if x["status"] == "failed":
        return
    if x["n"] >= P.N:
        if x["status"] != "done":
            x["status"], x["done_at"] = "done", now()
            log(st, f"{ph['id']} api {t}: done ({x['n']}/{P.N}, Beaker)")
        return
    if x["status"] == "done":
        x["status"] = "pending"; x.pop("done_at", None)
    if not x.get("exp"):
        miss = [c for c, f in V.records(ph["root"], t, CELLS) if not f]
        cells = None
        if x["n"] > 0:                                  # a fill: only the cells still missing
            fd = os.path.join(P.ENG, ph["root"], "_fill"); os.makedirs(fd, exist_ok=True)
            cells = os.path.relpath(os.path.join(fd, f"{t}.json"), P.ENG)
            json.dump({"cells": miss}, open(os.path.join(P.ENG, cells), "w"), indent=1)
        reps = max(1, min(ab["replicas"].get(t, ab["replicas"]["default"]), math.ceil(len(miss) / 3)))
        x["exp"], out = submit("lane", t, ph["root"], replicas=reps, cells=cells, cluster=ab["cluster"], cpu_lane=True)
        x["submits"] += 1
        x["status"] = "submitted" if x["exp"] else "pending"
        x["note"] = f"Beaker CPU job, {ab['cluster']}" if x["exp"] else f"submission failed: {out[-120:]}"
        x["submitted_at"] = now()
        log(st, f"{ph['id']} api {t}: Beaker experiment {x['exp'] or 'NOT created'} ({reps} CPU replica(s) on "
                f"{ab['cluster']}, {len(miss)} cells to run)")
        return
    track_exp(st, x)
    if x.get("jobs", {}).get("running"):
        x["status"] = "running"
    if not x.get("live") and now() - x.get("dead_since", now()) > 1200:
        if x["submits"] >= 6:
            x["status"], x["note"] = "failed", f"{x['n']}/{P.N} after 6 experiments; last {x.get('jobs')}"
            log(st, f"{ph['id']} api {t}: FAILED — {x['note']}")
        else:
            log(st, f"{ph['id']} api {t}: Beaker experiment ended at {x['n']}/{P.N} ({x.get('jobs')}) — submitting a fill")
            cancel(x["exp"])
            x.setdefault("old_exps", []).append(x.pop("exp")); x.pop("dead_since", None)
            x["status"] = "pending"


def api_lane(st, ph, s):
    lane = s.setdefault("api", {t: {"status": "pending", "starts": 0} for t in ph["api"]})
    running = [t for t, x in lane.items() if x["status"] == "running"]
    for t in ph["api"]:
        x = lane[t]
        if on_beaker(ph, t):                            # never started on the session's engines
            api_beaker_seat(st, ph, t, x)
            continue
        x["n"] = count(ph["root"], t)
        if x["status"] == "done" and x["n"] < P.N:
            x["status"] = "pending"; x.pop("done_at", None)
            log(st, f"{ph['id']} api {t}: back to {x['n']}/{P.N} — re-queued")
        if x["status"] == "running":
            if x["n"] >= P.N:
                x["status"], x["done_at"] = "done", now()
                log(st, f"{ph['id']} api {t}: done ({x['n']}/{P.N})")
            elif not alive(x.get("pid", 0), "run_lane"):
                if x["starts"] >= 5:
                    x["status"], x["note"] = "failed", f"{x['n']}/{P.N} after 5 starts"
                    log(st, f"{ph['id']} api {t}: FAILED — {x['note']}")
                else:
                    x["status"] = "pending"            # restarted below, resume-safe
                    log(st, f"{ph['id']} api {t}: runner exited at {x['n']}/{P.N} — restarting")
    running = [t for t, x in lane.items() if x["status"] == "running"]
    # engines are shared by EVERY phase's API lane (runs 2 and 3 both started on engines 8080-8099 on 2026-10-05
    # 23:21 PT and drove two episodes through each Unity scene): a slot is free only if no phase is using it
    used = [tuple(y["ports"]) for p_ in st["phases"].values() for y in (p_.get("api") or {}).values()
            if y.get("status") == "running" and y.get("ports")]
    busy_total = len(used)
    for t in ph["api"]:
        x = lane[t]
        if on_beaker(ph, t) or x["status"] != "pending" or busy_total >= P.API_CONCURRENT:
            continue
        if x["n"] >= P.N:
            x["status"], x["done_at"] = "done", now()
            continue
        slot = next((sl for sl in P.SLOTS if tuple(sl) not in used), None)
        if slot is None or not all(engine_up(p) for p in slot):
            continue
        x["ports"] = slot
        x["pid"] = spawn([P.PY, "-m", "vhhome_bench.lane", t, "--root", ph["root"], "--engines", ",".join(map(str, slot))],
                         f"{ph['id']}_api_{t}.log")
        x["starts"] += 1
        x["status"] = "running"
        x.setdefault("started", now())
        running.append(t); used.append(tuple(slot)); busy_total += 1
        log(st, f"{ph['id']} api {t}: started on engines {slot[0]}-{slot[-1]} (pid {x['pid']}, {x['n']}/{P.N} already there)")
    return all(x["status"] in ("done", "failed") for x in lane.values())


def gpu_lane(st, ph, s):
    lane = s.setdefault("gpu", {t: {"status": "pending", "submits": 0} for t in ph["gpu"]})
    pf = st.get("preflight", {}).get("gpu", {})
    for t in ph["gpu"]:
        x = lane[t]
        if x["status"] == "failed":
            continue
        x["n"] = count(ph["root"], t)
        if x["status"] == "done":
            if x["n"] >= P.N:
                continue
            track_exp(st, x, every=0)
            x["status"] = "running" if x.get("live") else "pending"
            x.pop("done_at", None)
            if not x.get("live") and x.get("exp"):
                x.setdefault("old_exps", []).append(x.pop("exp")); x.pop("dead_since", None)
            x["submits"] = 0
            log(st, f"{ph['id']} gpu {t}: back to {x['n']}/{P.N} — " + ("its experiment is still live" if x.get("live") else "a fill will be submitted"))
        if x["n"] >= P.N:
            x["status"], x["done_at"] = "done", now()
            log(st, f"{ph['id']} gpu {t}: done ({x['n']}/{P.N})")
            continue
        chk = pf.get(t, {}).get("status")
        if chk == "failed":
            x["status"], x["note"] = "failed", "the seat failed its pre-flight check"
            continue
        if chk != "pass":
            x["status"], x["note"] = "waiting", "for its pre-flight check job"
            continue
        if not x.get("exp"):
            cells = None
            reps = P.BEAKER["replicas"]
            if x["n"] > 0:                              # a fill: only the cells still missing
                miss = [c for c, f in V.records(ph["root"], t, CELLS) if not f]
                fd = os.path.join(P.ENG, ph["root"], "_fill"); os.makedirs(fd, exist_ok=True)
                cells = os.path.relpath(os.path.join(fd, f"{t}.json"), P.ENG)
                json.dump({"cells": miss}, open(os.path.join(P.ENG, cells), "w"), indent=1)
                reps = max(1, min(P.BEAKER["replicas"], math.ceil(len(miss) / 5)))
            x["exp"], out = submit("lane", t, ph["root"], replicas=reps, cells=cells)
            x["submits"] += 1
            x["status"] = "submitted" if x["exp"] else "pending"
            x["note"] = "" if x["exp"] else f"submission failed: {out[-120:]}"
            x["submitted_at"] = now()
            log(st, f"{ph['id']} gpu {t}: experiment {x['exp'] or 'NOT created'} ({reps} replica(s), {P.N - x['n']} cells to run)")
            continue
        track_exp(st, x)
        if x.get("jobs", {}).get("running"):
            x["status"] = "running"
        if not x.get("live") and now() - x.get("dead_since", now()) > 1200:
            if x["submits"] >= 4:
                x["status"], x["note"] = "failed", f"{x['n']}/{P.N} after 4 experiments; last {x.get('jobs')}"
                log(st, f"{ph['id']} gpu {t}: FAILED — {x['note']}")
            else:
                log(st, f"{ph['id']} gpu {t}: experiment ended at {x['n']}/{P.N} ({x.get('jobs')}) — submitting a fill")
                cancel(x["exp"])
                x.setdefault("old_exps", []).append(x.pop("exp")); x.pop("dead_since", None)
                x["status"] = "pending"
    return all(x["status"] in ("done", "failed") for x in lane.values())


def bench(st, ph):
    s = st["phases"].setdefault(ph["id"], {"status": "pending"})
    if s["status"] in ("done", "blocked"):
        return
    full = (not ph.get("needs_approval")) or ph["id"] in st.get("approved", [])
    gpu_only = (not full) and ph["id"] in st.get("approved_gpu", [])     # the go-ahead for the the H100 cluster lane alone (2026-10-05)
    if not (full or gpu_only):
        return
    if st["phases"].get("preflight", {}).get("local") != "pass":
        return
    if s["status"] == "pending":
        s["status"], s["started"] = "running", now()
        log(st, f"{ph['id']}: started -> {ph['root']}" + (" (GPU lane only; the API lane waits for its own go)" if gpu_only else ""))
    s["gpu_only"] = gpu_only
    a_done = api_lane(st, ph, s) if full else False
    g_done = gpu_lane(st, ph, s)                  # each seat waits for its own check job, not for the whole pre-flight
    if now() - s.get("verified_at", 0) > 1800 or (a_done and g_done):
        v = V.verify(ph["root"], ph["api"], ph["gpu"], st["stamp"])
        v["at"] = now()
        s["verify"], s["verified_at"] = v, now()
    if a_done and g_done:
        v = s["verify"]
        failed = [t for lane in ("api", "gpu") for t, x in s[lane].items() if x["status"] == "failed"]
        if v["status"] == "fail" or failed:
            s["status"] = "blocked"
            log(st, f"{ph['id']}: BLOCKED — verifier {v['status']}, failed seats {failed}; see the page")
        else:
            s["status"], s["done_at"] = "done", now()
            log(st, f"{ph['id']}: DONE — verifier {v['status']}")


# ------------------------------------------------------------------ scoring phase
def score(st, ph):
    if st["phases"].get(ph["needs"], {}).get("status") != "done":
        return
    s = st["phases"].setdefault(ph["id"], {"status": "running", "started": now(), "steps": {}})
    if s["status"] in ("done", "blocked"):
        return
    for i, (name, cmd) in enumerate(P.SCORING):
        x = s["steps"].setdefault(name, {"status": "pending"})
        if x["status"] == "done":
            continue
        if x["status"] == "pending":
            c = cmd.format(PY=P.PY, root=ph["root"])
            rcf_old = f"{LOGS}/{ph['id']}_{i}.rc"                 # a stale .rc from an earlier scoring round of this phase
            if os.path.exists(rcf_old):                             # marked the step done the moment it started (2026-10-12)
                os.remove(rcf_old)
            x["pid"] = spawn(["bash", "-c", c + f"; echo EXIT=$? > {LOGS}/{ph['id']}_{i}.rc"], f"{ph['id']}_{i}.log")
            x["status"], x["cmd"] = "running", c
            log(st, f"{ph['id']}: {name} started")
            return
        rcf = f"{LOGS}/{ph['id']}_{i}.rc"
        if os.path.exists(rcf):
            rc = open(rcf).read().strip()
            x["status"] = "done" if rc == "EXIT=0" else "failed"
            x["note"] = rc
            log(st, f"{ph['id']}: {name} {x['status']} ({rc})")
            if x["status"] == "failed":
                s["status"] = "blocked"
                return
            continue
        return                                           # still running
    s["status"], s["done_at"] = "done", now()
    log(st, f"{ph['id']}: DONE")


# ------------------------------------------------------------------ page
def publish(st, force=False):
    from vhhome_bench.cluster import page as page
    if not P.PAGES_REPO:                                   # no Pages site configured: the page is written, not pushed
        os.makedirs(os.path.join(P.PAGES_DIR, P.PAGE_PATH), exist_ok=True)
        open(os.path.join(P.PAGES_DIR, P.PAGE_PATH, "index.html"), "w").write(page.render(st))
        st["rendered"] = now()
        return
    st["rendered"] = now()
    htm = page.render(st)
    tok = os.environ.get("GITHUB_TOKEN")
    url = f"https://x-access-token:{tok}@github.com:443/{P.PAGES_REPO}.git"
    if not os.path.isdir(os.path.join(P.PAGES_DIR, ".git")):
        if not tok:
            return
        r = sh(f"git -c credential.helper= clone -q --depth 1 '{url}' '{P.PAGES_DIR}' && cd '{P.PAGES_DIR}' && "
               f"git remote set-url origin https://github.com:443/{P.PAGES_REPO}.git")
        if r.returncode != 0:
            log(st, "pages clone failed: " + re.sub(r"x-access-token:[^@]*@", "***@", (r.stderr or "")[-160:]))
            return
    out = os.path.join(P.PAGES_DIR, P.PAGE_PATH, "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    strip = lambda h: re.sub(r"page written [^<]*", "", h)
    changed = (not os.path.exists(out)) or strip(open(out).read()) != strip(htm)
    # a page written inside the 10-minute push window was never pushed afterwards (the file already matched, so
    # nothing looked changed): the final run-3 page sat unpushed for seven hours on 2026-10-07. Dirty = push.
    dirty = bool(sh(f"cd '{P.PAGES_DIR}' && git status --porcelain {P.PAGE_PATH}/index.html").stdout.strip())
    if not changed and not dirty and not force:
        return
    if changed:
        open(out, "w").write(htm)
    if not tok or (now() - st.get("last_push", 0) < 600 and not force):
        return
    r = sh(f"cd '{P.PAGES_DIR}' && git add {P.PAGE_PATH}/index.html && git -c user.name='Chenhao Zheng' "
           f"-c user.email=chzheng@cs.washington.edu commit -q -m 'formal run 2: progress' && "
           f"(git -c credential.helper= pull -q --rebase '{url}' main; git -c credential.helper= push -q '{url}' HEAD:main)")
    st["last_push"] = now()
    if r.returncode != 0:
        log(st, "page push failed: " + re.sub(r"x-access-token:[^@]*@", "***@", ((r.stderr or "") + (r.stdout or ""))[-160:]))


# ------------------------------------------------------------------ commands
def init():
    os.makedirs(LOGS, exist_ok=True)
    if os.path.exists(STATE):
        sys.exit(f"{STATE} exists — the run is already initialised")

    from vhhome import world as iv_world
    commit = sh("git rev-parse --short HEAD", cwd=P.ENG).stdout.strip()
    dirty = sh("git status --porcelain --untracked-files=no", cwd=P.ENG).stdout.strip()
    if dirty:
        sys.exit("the engine checkout has uncommitted changes — commit before starting a formal run:\n" + dirty)
    st = dict(started=now(), stamp=iv_world.ENGINE_REV, commit=commit, phases={}, approved=[], log=[])
    log(st, f"initialised: engine {st['stamp']} at commit {commit}")
    save(st)


def tick(st):
    ensure_engines(st)
    if now() - st.get("superseded_at", 0) > 600:
        supersede_old_guard(st)
        st["superseded_at"] = now()
    for ph in P.phases():
        if ph["kind"] == "preflight":
            preflight(st)
        elif ph["kind"] == "bench":
            bench(st, ph)
        elif ph["kind"] == "score":
            score(st, ph)


def daemon():
    pidf = os.path.join(P.STATE_DIR, "daemon.pid")
    if os.path.exists(pidf) and alive(int(open(pidf).read().strip() or 0), "orchestrate.py"):
        sys.exit("a daemon is already running")
    open(pidf, "w").write(str(os.getpid()))
    st = load()
    log(st, f"daemon up (pid {os.getpid()})")
    while True:
        try:
            ap = os.path.join(P.STATE_DIR, "approve.txt")       # handed over by `approve`, so the daemon stays the only writer
            if os.path.exists(ap):
                for pid_ in open(ap).read().split():
                    key, pid_ = ("approved_gpu", pid_[4:]) if pid_.startswith("gpu:") else ("approved", pid_)
                    if pid_ not in st.setdefault(key, []):
                        st[key].append(pid_); log(st, f"{key.replace('_', ' ')}: {pid_}")
                os.remove(ap)
            tick(st)
            publish(st, force=os.path.exists(os.path.join(P.STATE_DIR, "push_now")))
            if os.path.exists(os.path.join(P.STATE_DIR, "push_now")):
                os.remove(os.path.join(P.STATE_DIR, "push_now"))
        except Exception as ex:                                 # noqa: BLE001
            import traceback
            log(st, f"daemon loop error: {ex!r}")
            traceback.print_exc()
        save(st)
        time.sleep(60)


def status():
    st = load()
    print(f"engine {st['stamp']} commit {st['commit']}  page {P.PAGE_URL}")
    for ph in P.phases():
        s = st["phases"].get(ph["id"], {})
        print(f"== {ph['id']:11} {s.get('status', 'pending')}")
        if ph["kind"] == "preflight":
            for k, v in st.get("preflight", {}).get("local", {}).items():
                print(f"   local {k:60} {v['status']:8} {v.get('note', '')}")
            for k, v in st.get("preflight", {}).get("gpu", {}).items():
                print(f"   gpu   {k:22} {v['status']:9} {v.get('exp', '')} {v.get('jobs', '')} {v.get('note', '')}")
        for lane in ("api", "gpu"):
            for t, x in (s.get(lane) or {}).items():
                print(f"   {lane} {t:24} {x['status']:9} {x.get('n', 0):3}/{P.N} {x.get('exp', '')} {x.get('jobs', '')} {x.get('note', '')}")
        for k, v in (s.get("steps") or {}).items():
            print(f"   {k:20} {v['status']} {v.get('note', '')}")
    print("\n".join(st.get("log", [])[-8:]))


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "status"
    if c == "init":
        init()
    elif c == "daemon":
        daemon()
    elif c == "approve":                         # approve bench-r2            (both lanes)
        open(os.path.join(P.STATE_DIR, "approve.txt"), "a").write(sys.argv[2] + "\n"); print("handed to the daemon:", sys.argv[2])
    elif c == "approve-gpu":                     # approve-gpu bench-r2        (the H100 cluster lane only)
        open(os.path.join(P.STATE_DIR, "approve.txt"), "a").write("gpu:" + sys.argv[2] + "\n"); print("handed to the daemon: gpu lane of", sys.argv[2])
    elif c == "render":
        open(os.path.join(P.STATE_DIR, "push_now"), "w").write("1"); print("the daemon will rewrite and push the page on its next loop")
    else:
        status()
