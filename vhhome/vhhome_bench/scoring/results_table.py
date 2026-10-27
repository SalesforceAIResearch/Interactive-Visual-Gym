#!/usr/bin/env python3
"""The engine-scored table for one run, and the reliability table across the independent repeats.

    python -m vhhome_bench.scoring.results_table runs/mine                      # one run
    python -m vhhome_bench.scoring.results_table runs/mine tmp/bench_v2 tmp/bench_v3   # mean +- spread

DEFINITIONS, recovered from the first run's table and verified to reproduce it on all 17 arms:

  fired     phases whose forced error WENT LIVE, plus phases that were PREVENTED. A prevented error
            never goes live, but it must stay in the denominator: otherwise preventing an error would
            shrink the denominator and reward the assistant twice.
  caught    phases the graders credited as detected (phases_caught).
  prevented phases stopped before the error went live (phases_prevented). The first run's ad-hoc table
            reported this as 0 for every arm; the reports in fact carry 16, so that column was a bug in
            the table, not a gap in the engine.
  handled   caught + prevented, over fired — what fraction of forced errors the assistant did something
            about, by either route.
  won       episodes whose outcome is "won", over episodes.
  lat       median seconds of generation per assistant call (A100-equivalent from engine f119 on).

Across several runs each number is printed as mean +- half the range, and the spread is the point: it
says how much of a gap between two arms is real. The runs share one 75-cell table (same cases, same
personas, same seed), so they are paired repeats and the spread is the model's own sampling noise.
"""
import argparse, glob, json, os, statistics, sys

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_rows(root):
    rows = {}
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        tag = d.rstrip("/").split("/")[-1]
        if tag in ("serve_logs", "_fill", "sq_v4", "sq_v4_frames"):
            continue
        fs = [f for f in sorted(glob.glob(d + "*.json")) if not f.endswith(".lane.json")]
        if not fs:
            continue
        n = won = fired = caught = prevented = spoke = calls = empty = schema = 0
        lat, revs = [], set()
        for f in fs:
            try:
                r = json.load(open(f))
            except Exception:
                continue
            n += 1
            won += r.get("outcome") == "won"
            fired += sum(1 for p in (r.get("phases") or []) if p.get("dev_live") is not None)
            caught += r.get("phases_caught") or 0
            prevented += r.get("phases_prevented") or 0
            spoke += sum(1 for c in r.get("conversation") or []
                         if c.get("speaker") == "vlm" and (c.get("text") or "").strip())
            st = ((r.get("assistant") or {}).get("stats") or {})
            calls += st.get("calls") or 0
            empty += st.get("empty") or 0
            schema += st.get("schema_fail") or 0
            if st.get("calls"):
                lat.append((st.get("latency_s") or 0) / st["calls"])
            revs.add(r.get("engine_rev"))
        fired += prevented
        rows[tag] = {"tag": tag, "n": n, "won": won, "fired": fired, "caught": caught,
                     "prevented": prevented, "spoke": spoke, "calls": calls, "empty": empty,
                     "schema": schema, "lat": round(statistics.median(lat), 1) if lat else 0.0,
                     "won_rate": won / n if n else None,
                     "detect_rate": caught / fired if fired else None,
                     "handled_rate": (caught + prevented) / fired if fired else None,
                     "revs": sorted(x for x in revs if x)}
    return rows


def one(root):
    rows = sorted(run_rows(root).values(), key=lambda r: -(r["detect_rate"] or 0))
    print(f"{'arm':24} {'eps':>4} {'fired':>6} {'caught':>7} {'prev':>5} {'detect':>7} "
          f"{'handled':>8} {'won':>6} {'spoke':>6} {'s/call':>7}")
    for r in rows:
        detect = f"{100*r['detect_rate']:.0f}%" if r['detect_rate'] is not None else '—'
        handled = f"{100*r['handled_rate']:.0f}%" if r['handled_rate'] is not None else '—'
        won = f"{100*r['won_rate']:.0f}%" if r['won_rate'] is not None else '—'
        print(f"{r['tag']:24} {r['n']:>4} {r['fired']:>6} {r['caught']:>7} {r['prevented']:>5} "
              f"{detect:>7} {handled:>8} {won:>6} "
              f"{r['spoke']:>6} {r['lat']:>7}")
    return rows


def across(roots):
    per = [run_rows(r) for r in roots]
    tags = sorted(set().union(*[set(p) for p in per]),
                  key=lambda t: -statistics.fmean([p[t]["detect_rate"] or 0 for p in per if t in p]))
    print(f"{'arm':24} {'runs':>5} {'detection %':>18} {'won %':>18} {'eps':>12}")
    for t in tags:
        got = [p[t] for p in per if t in p and p[t]["n"]]
        det = [100 * (g["detect_rate"] or 0) for g in got]
        won = [100 * (g["won_rate"] or 0) for g in got]
        ns = "/".join(str(g["n"]) for g in got)
        def pm(v):
            return (f"{statistics.fmean(v):5.1f} ± {(max(v)-min(v))/2:4.1f}" if len(v) > 1
                    else f"{v[0]:5.1f}      " if v else "  —        ")
        print(f"{t:24} {len(got):>5} {pm(det):>18} {pm(won):>18} {ns:>12}")
    print("\n± is half the range over the independent repeats, on the SAME 75 cells: a gap between two "
          "arms smaller than that is not a result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    roots = [r if os.path.isabs(r) else os.path.join(ENG, r) for r in a.roots]
    if len(roots) == 1:
        rows = one(roots[0])
        if a.json:
            json.dump(rows, open(a.json, "w"), indent=1)
            print("wrote", a.json)
    else:
        across(roots)


if __name__ == "__main__":
    main()
