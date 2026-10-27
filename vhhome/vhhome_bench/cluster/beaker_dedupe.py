#!/usr/bin/env python3
"""One arm, one run, ONE live job — across every workspace.

    python -m vhhome_bench.cluster.beaker_dedupe                 # report
    python -m vhhome_bench.cluster.beaker_dedupe --cancel        # cancel the strays

His standing rule (2026-09-22): when lanes move to another workspace, the originals must not be left
behind. A stranded job is not harmless — if its allocation frees weeks later it wakes up, replays cells
the new run already wrote, and two workers touch one episode, which is exactly what the resume design
exists to prevent.

A served (GPU) arm belongs in the GPU workspace; an API or silent arm belongs in the session workspace.
Anything live for a served arm in the session workspace is a stray.

Duplicates are counted per SUBMISSION, not per job. One submission of a CPU arm is several jobs sharing
a stamp (…-s1-0922-042852, …-s2-0922-042852) that hold DISJOINT cells, and a replica submission is one
job whatever its replica count. So a (root, arm) with two live stamps is the duplicate case — the newest
stamp is kept, a re-submission normally being the repair of a failed one — and two jobs of one stamp are
not. Calibration jobs are not lanes and are left alone.
"""
import argparse, collections, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

from vhhome_bench import roster as roster_vh  # noqa: E402
from vhhome_bench.cluster import beaker_submit as S  # noqa: E402

LIVE = ("QUEUED", "INITIALIZING", "RUNNING", "READY")


def status(b, wl):
    j = b.workload.get_latest_job(wl)
    st = j.status if j is not None else None
    if st is None:
        return "QUEUED"
    ev = type(st).DESCRIPTOR.fields_by_name["status"].enum_type.values_by_number.get(st.status)
    return (ev.name if ev else "QUEUED").replace("STATUS_", "")


def scan(b, workspace):
    out = []
    ws = b.workspace.get(workspace)
    for wl in b.workload.list(workspace=ws, limit=900):
        ex = getattr(wl, "experiment", None)
        name = getattr(ex, "name", "") or ""
        if not name.startswith("vhsim-lane-"):      # calib/check jobs are not lanes
            continue
        desc = getattr(ex, "description", "") or ""
        tag = desc.split("lane: ")[-1].split(" ->")[0] if "lane: " in desc else "?"
        root = desc.split("-> ")[-1] if "-> " in desc else "?"
        st = status(b, wl)
        if st in LIVE:
            out.append((root, tag, name, st, wl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancel", action="store_true")
    a = ap.parse_args()
    from beaker import Beaker
    b = Beaker.from_env()
    local = {e["tag"] for e in roster_vh.ROSTER if e["kind"] == "local"}
    strays = []
    for ws_name in (S.WORKSPACE, S.GPU_WORKSPACE):
        rows = scan(b, ws_name)
        here_is_gpu = ws_name == S.GPU_WORKSPACE
        by_cell = collections.defaultdict(list)
        for root, tag, name, st, wl in rows:
            if (tag in local) != here_is_gpu:
                strays.append((ws_name, root, tag, name, st, wl, "wrong workspace"))
            else:
                by_cell[(root, tag)].append((name, st, wl))
        for (root, tag), v in by_cell.items():
            stamps = sorted({n.rsplit("-", 2)[-2] + "-" + n.rsplit("-", 2)[-1] for n, _, _ in v})
            if len(stamps) < 2:
                continue
            for name, st, wl in v:
                if not name.endswith(stamps[-1]):        # everything but the newest submission
                    strays.append((ws_name, root, tag, name, st, wl, "duplicate"))
        print(f"{ws_name}: {len(rows)} live lane job(s)")
    if not strays:
        print("\nno strays: one live job per arm per run")
        return
    print(f"\n{len(strays)} stray job(s):")
    for ws_name, root, tag, name, st, wl, why in strays:
        print(f"  {why:16} {ws_name:30} {root:14} {tag:22} {name[:40]} {st}")
        if a.cancel:
            b.workload.cancel(wl)
    print("cancelled" if a.cancel else "\n(--cancel to cancel them)")


if __name__ == "__main__":
    main()
