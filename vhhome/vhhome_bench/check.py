#!/usr/bin/env python3
"""Health table for a check or lane root: one row per assistant.

    python -m vhhome_bench.check tools/tmp/beaker_check3

It is a HEALTH check, not a score. What it answers: did the arm play its episodes, did its replies reach
the harness, did it say anything, did it raise flags, and what did a call cost in game time. Outcomes are
shown because a silent or broken arm loses every episode, not because three episodes rank anyone.
"""
from vhhome import paths
import json, os, statistics, sys, glob, time


def rows(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        tag = d.rstrip("/").split("/")[-1]
        if tag in ("serve_logs",):
            continue
        reps = [f for f in sorted(glob.glob(d + "*.json")) if not f.endswith(".lane.json")]
        void = glob.glob(os.path.join(d, "void", "*.json"))
        if not reps and not void:
            continue
        calls = empty = schema = spoke = flags = 0
        lat, outcomes, revs = [], [], set()
        for f in reps:
            try:
                r = json.load(open(f))
            except Exception:
                continue
            st = ((r.get("assistant") or {}).get("stats") or {})
            calls += st.get("calls") or 0
            empty += st.get("empty") or 0
            schema += st.get("schema_fail") or 0
            if st.get("calls"):
                lat.append((st.get("latency_s") or 0) / st["calls"])
            spoke += sum(1 for c in r.get("conversation", [])
                         if c.get("speaker") == "vlm" and (c.get("text") or "").strip())
            flags += len(r.get("flags") or [])
            outcomes.append(str(r.get("outcome")))
            revs.add((r.get("engine_rev") or "")[-4:])
        out.append(dict(tag=tag, eps=len(reps), void=len(void), calls=calls, empty=empty,
                        schema_fail=schema, spoke=spoke, flags=flags,
                        lat=round(statistics.median(lat), 1) if lat else None,
                        outcomes=",".join(outcomes), rev="/".join(sorted(revs))))
    return out


def stalled(root, expect=75, quiet_min=45):
    """Arms short of their cells with nothing written for a while.

    A lane that CRASHES is loud on Beaker and silent here: the run directory just shows an arm a few
    episodes short, which looks exactly like an arm still working. Comparing the count against the last
    write is what separates the two — it is how the single-shard fill bug was caught on 2026-09-22,
    after two arms had sat short for six hours."""
    now, out = time.time(), []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        tag = d.rstrip("/").split("/")[-1]
        if tag in ("serve_logs", "_fill", "sq_v4", "sq_v4_frames"):
            continue
        fs = [f for f in glob.glob(d + "*.json") if not f.endswith(".lane.json")]
        if not fs or len(fs) >= expect:
            continue
        quiet = (now - max(os.path.getmtime(f) for f in fs)) / 60
        if quiet > quiet_min:
            out.append((tag, len(fs), quiet))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.RUNS, "check")
    rs = rows(root)
    print(f"{'assistant':24} {'eps':>4} {'void':>4} {'calls':>6} {'empty':>6} {'schema':>7} "
          f"{'spoke':>6} {'flags':>6} {'s/call':>7}  outcomes")
    for r in rs:
        print(f"{r['tag']:24} {r['eps']:>4} {r['void']:>4} {r['calls']:>6} {r['empty']:>6} "
              f"{r['schema_fail']:>7} {r['spoke']:>6} {r['flags']:>6} "
              f"{(r['lat'] if r['lat'] is not None else '-'):>7}  {r['outcomes']}")
    bad = [r for r in rs if r["void"] or (r["calls"] and r["empty"] / r["calls"] > 0.5)]
    print(f"\n{len(rs)} assistants, {sum(r['eps'] for r in rs)} episodes, "
          f"{sum(r['void'] for r in rs)} void"
          + (f" — LOOK AT: {', '.join(r['tag'] for r in bad)}" if bad else " — nothing quarantined"))
    revs = {r["rev"] for r in rs if r["rev"]}
    print("engine revs seen:", ", ".join(sorted(revs)) or "—")
    st = stalled(root)
    if st:
        print("\nSTALLED (short of 75 cells, nothing written recently — check the lane's job, it may "
              "have failed):")
        for tag, n, quiet in st:
            print(f"   {tag:24} {n:>3}/75   quiet for {quiet:.0f} min")


if __name__ == "__main__":
    main()
