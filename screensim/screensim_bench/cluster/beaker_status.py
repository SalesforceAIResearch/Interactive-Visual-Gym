#!/usr/bin/env python3
"""Status of the ScreenSim Beaker jobs (copied from cook-bench-engine) (screensim_bench/cluster/beaker_submit.py), and their logs.

    export BEAKER_TOKEN=$(cat ~/.beaker_token_cooksim)
    python3 screensim_bench/cluster/beaker_status.py                       # recent screensim-* jobs in the workspace
    python3 screensim_bench/cluster/beaker_status.py --logs cooksim-check-molmo2-8b-0919-0532 [--tail 40]
"""
import argparse
import datetime as dt
import sys

WORKSPACE = "ai2/trajectory_video_encoder"


def fmt_ts(ts):
    if ts is None or not getattr(ts, "seconds", 0):
        return "-"
    return dt.datetime.fromtimestamp(ts.seconds, dt.timezone.utc).strftime("%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="screensim-")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--logs")
    ap.add_argument("--tail", type=int, default=40)
    a = ap.parse_args()
    from beaker import Beaker
    b = Beaker.from_env()
    ws = b.workspace.get(WORKSPACE)
    if a.logs:
        wl = b.workload.get(a.logs) if not a.logs.startswith("01") else b.workload.get(a.logs)
        j = b.workload.get_latest_job(wl)
        if j is None:
            sys.exit("no job yet")
        lines = []
        for entry in b.job.logs(j):
            msg = getattr(entry, "message", entry)
            if isinstance(msg, bytes):
                msg = msg.decode(errors="replace")
            lines.append(str(msg).rstrip("\n"))
        print("\n".join(lines[-a.tail:]))
        return
    rows = []
    for wl in b.workload.list(workspace=ws, limit=a.limit):
        name = getattr(getattr(wl, "experiment", None), "name", "") or ""
        if not name.startswith(a.prefix):
            continue
        j = b.workload.get_latest_job(wl)
        st = j.status if j is not None else None
        if st is not None:
            ev = type(st).DESCRIPTOR.fields_by_name["status"].enum_type.values_by_number.get(st.status)
            status = (ev.name if ev else str(st.status)).replace("STATUS_", "")
        else:
            status = "QUEUED"
        code = getattr(st, "exit_code", None) if st is not None else None
        node = getattr(getattr(j, "assignment_details", None), "node_id", "") if j is not None else ""
        rows.append((name, status, fmt_ts(getattr(st, "created", None)), fmt_ts(getattr(st, "started", None)),
                     fmt_ts(getattr(st, "exited", None)), code if st is not None and getattr(st, "exited", None) else ""))
    print(f"{'job':52s} {'status':14s} {'created':12s} {'started':12s} {'exited':12s} exit")
    for r in rows:
        print(f"{r[0]:52s} {r[1]:14s} {r[2]:12s} {r[3]:12s} {r[4]:12s} {r[5]}")


if __name__ == "__main__":
    main()
