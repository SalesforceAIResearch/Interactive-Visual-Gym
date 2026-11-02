#!/usr/bin/env python3
"""The headline table: in-time success, efficiency, detection — one row per arm.

    python -m vhhome_bench.scoring.metrics_table runs/mine                       # one run
    python -m vhhome_bench.scoring.metrics_table runs/mine tmp/bench_v2 tmp/bench_v3   # mean +- spread
    python -m vhhome_bench.scoring.metrics_table runs/mine --json out.json --full

Every number is the CookSim v6 metrics contract as vhhome_bench/scoring/metrics.py already implements it; this
only aggregates it per arm and across the independent repeats. Three headline columns, each the mean of
its two parts, with the parts shown:

  S    in-time success      1[won] * 1[T <= C],  C = 2*(T_ref + alpha): finishing is not enough, it must
                            finish inside twice the reference plan's time plus an allowance.
  EFF  efficiency           (a*TT + c*CC)/(a+c) over SUCCESSFUL episodes only, a=2 fixed and c from the
       = action TT          persona's own interruption preference. TT is action/time efficiency, how much
       + conversation CC    of the time budget was left; CC is conversation efficiency, 1 - the share of
                            the episode the assistant spent talking.
  DET  detection            the mean of DT and F1.
       = DT (timing)        DT scores each forced error 1 if PREVENTED, 0.5 if correctly flagged after it
       + F1                 landed, 0 if missed — so prevention is worth twice a late catch.
                            F1 balances that against false alarms: an arm that flags everything scores R=1
                            and P low, and F1 refuses it the credit DT alone would give.

EFF is conditional on success, so its n is smaller than the arm's episode count and an arm that almost
never succeeds has an EFF computed on a handful of episodes. The n is printed for that reason.
"""
import argparse, functools, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)

from vhhome_bench.scoring import metrics as AM  # noqa: E402

AM.t_ref = functools.lru_cache(maxsize=None)(AM.t_ref)   # same reference plan for every episode of a family
SKIP = ("serve_logs", "_fill", "sq_v4", "sq_v4_frames", "void")


def arm_rows(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        arm = d.rstrip("/").split("/")[-1]
        if arm in SKIP:
            continue
        rows = []
        for f in sorted(glob.glob(d + "*.json")):
            if f.endswith(".lane.json"):
                continue
            try:
                rec = json.load(open(f))
            except Exception:
                continue
            rec["_arm"] = arm
            try:
                rows.append(AM.episode_metrics(rec))
            except Exception:
                continue
        if rows:
            out[arm] = rows
    return out


def summarise(rows):
    def m(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return (statistics.fmean(v) if v else None), len(v)
    S, nS = m("S")
    TT, _ = m("TT")
    CC, _ = m("CC")
    E, nE = m("E")
    DT, _ = m("DT")
    F1, _ = m("F1")
    Pr, _ = m("P")
    fired = sum(r["fired"] for r in rows)
    R = sum(r["tp"] for r in rows) / fired if fired else None
    det = statistics.fmean([x for x in (DT, F1) if x is not None]) if (DT is not None or F1 is not None) else None
    return {"n": len(rows), "S": S, "EFF": E, "n_eff": nE, "TT": TT, "CC": CC,
            "DET": det, "DT": DT, "F1": F1, "P": Pr, "R": R,
            "prevention": sum(r["prevented_n"] for r in rows) / fired if fired else None,
            "fired": sum(r["fired"] for r in rows), "fp": sum(r["fp"] for r in rows),
            "utterances": sum(r["utterances"] for r in rows)}


def pm(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.fmean(vals):.3f}±{(max(vals)-min(vals))/2:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--full", action="store_true", help="also print the parts of each score")
    a = ap.parse_args()
    roots = [r if os.path.isabs(r) else os.path.join(ENG, r) for r in a.roots]
    per = [{k: summarise(v) for k, v in arm_rows(r).items()} for r in roots]
    arms = sorted(set().union(*[set(p) for p in per]),
                  key=lambda t: -statistics.fmean([p[t]["DET"] or 0 for p in per if t in p]))
    multi = len(roots) > 1
    w = 14 if multi else 8
    print(f"{'arm':22} {'eps':>6} {'S (in-time)':>{w}} {'EFF':>{w}} {'DET':>{w}}"
          + (f" {'TT':>{w}} {'CC':>{w}} {'DT':>{w}} {'F1':>{w}} {'P':>{w}} {'R':>{w}}" if a.full else ""))
    rows_out = {}
    for t in arms:
        got = [p[t] for p in per if t in p]
        n = "/".join(str(g["n"]) for g in got)
        line = f"{t:22} {n:>6} " + " ".join(f"{pm([g[k] for g in got]):>{w}}" for k in ("S", "EFF", "DET"))
        if a.full:
            line += " " + " ".join(f"{pm([g[k] for g in got]):>{w}}" for k in ("TT", "CC", "DT", "F1", "P", "R"))
        print(line)
        rows_out[t] = got
    print(f"\nEFF is over successful episodes only (n_eff per run: "
          + "; ".join(f"{t}={'/'.join(str(g['n_eff']) for g in [p[t] for p in per if t in p])}" for t in arms[:4])
          + ", ...).")
    print("DET = mean(DT, F1). DT credits a PREVENTED error 1.0 and a correct late flag 0.5; F1 is the "
          "balance against false alarms." + ("  ± is half the range over the repeats." if multi else ""))
    if a.json:
        json.dump(rows_out, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
