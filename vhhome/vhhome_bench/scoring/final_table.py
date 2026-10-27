#!/usr/bin/env python3
"""The final table of the formal run: three independent runs pooled, one row per seat.

    python -m vhhome_bench.scoring.final_table            # writes tmp/bench2/final_results.json and results/formal_run_2.md

Objective metrics come from vhhome_bench/scoring/metrics_table.py (the CookSim v6 contract: S in-time success, EFF efficiency over
successful episodes, DET = mean of DT and F1) — the mean over the three runs, with the half-range as the spread.
Interaction quality (AQ) is the SQ v4 rubric's per-episode score, pooled over the 225 episodes of a seat, with a
2000-draw bootstrap 95% interval over episodes; the per-run means are shown beside it.
"""
import glob, json, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.abspath(os.path.join(HERE, "..", ".."))

from vhhome_bench.scoring import metrics_table as M  # noqa: E402

ROOTS = [f"runs/formal/r{k}" for k in (1, 2, 3)]   # vhhome-score passes its own
NEVER_SPEAKS = {"silent", "mmduet2"}


def boot(vals, n=2000, seed=11):
    if len(vals) < 2:
        return (None, None)
    r = random.Random(seed)
    ms = sorted(statistics.fmean(r.choices(vals, k=len(vals))) for _ in range(n))
    return (ms[int(.025 * n)], ms[int(.975 * n)])


def aq_rows():
    out = {}
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "sq_v4", "*/"))):
            if d.rstrip("/").endswith("claude_batches"):
                continue
            arm = d.rstrip("/").split("/")[-1]
            for f in glob.glob(d + "*.json"):
                aq = (json.load(open(f)).get("aggregate") or {}).get("AQ")
                if aq is not None:
                    out.setdefault(arm, {}).setdefault(root, []).append(aq)
    rows = {}
    for arm, per in out.items():
        pooled = [v for vs in per.values() for v in vs]
        lo, hi = boot(pooled)
        rows[arm] = {"n": len(pooled), "AQ": statistics.fmean(pooled), "AQ_lo": lo, "AQ_hi": hi,
                     "per_run": [statistics.fmean(per[r]) if per.get(r) else None for r in ROOTS]}
    return rows


def main(roots=None, out_dir=None):
    global ROOTS
    ROOTS = [os.path.abspath(r) for r in (roots or ROOTS)]
    out_dir = out_dir or os.path.dirname(ROOTS[0])
    per = [{k: M.summarise(v) for k, v in M.arm_rows(r).items()} for r in ROOTS]
    won = {}                                                   # plain win rate per run, the cap-free companion of S
    for r in ROOTS:
        for f in glob.glob(os.path.join(r, "*", "*__*.json")):
            if ".lane." in f or "/sq_v4/" in f or "/_fill/" in f or "_superseded" in f:
                continue
            arm = os.path.basename(os.path.dirname(f))
            try:
                oc = json.load(open(f)).get("outcome")
            except Exception:
                continue
            w = won.setdefault(arm, {}).setdefault(r, [0, 0])
            w[0] += oc == "won"; w[1] += 1
    arms = sorted(set().union(*[set(p) for p in per]))
    aq = aq_rows()
    rows = []
    for t in arms:
        got = [p[t] for p in per if t in p]
        row = {"seat": t, "episodes": sum(g["n"] for g in got)}
        for k in ("S", "EFF", "DET", "TT", "CC", "DT", "F1", "P", "R"):
            vals = [g[k] for g in got if g.get(k) is not None]
            row[k] = statistics.fmean(vals) if vals else None
            row[k + "_spread"] = (max(vals) - min(vals)) / 2 if len(vals) > 1 else None
            row[k + "_runs"] = [g.get(k) for g in got]
        wr = [won.get(t, {}).get(r) for r in ROOTS]
        row["won_runs"] = [w[0] / w[1] if w and w[1] else None for w in wr]
        wv = [x for x in row["won_runs"] if x is not None]
        row["won"] = statistics.fmean(wv) if wv else None
        row["won_spread"] = (max(wv) - min(wv)) / 2 if len(wv) > 1 else None
        a = aq.get(t)
        row.update({"AQ": a["AQ"], "AQ_lo": a["AQ_lo"], "AQ_hi": a["AQ_hi"], "AQ_runs": a["per_run"], "AQ_n": a["n"]}
                   if a else {"AQ": None, "AQ_lo": None, "AQ_hi": None, "AQ_runs": [None] * len(ROOTS), "AQ_n": 0})
        rows.append(row)
    rows.sort(key=lambda r: -(r["AQ"] if r["AQ"] is not None else -1))
    out = {"roots": ROOTS, "rows": rows,
           "notes": [f"S, EFF, DET: mean over {len(ROOTS)} supplied run(s), ± half the range when multiple runs are available (vhhome_bench/scoring/metrics_table.py).",
                     "AQ: SQ v4 per-episode interaction quality pooled over the supplied runs; AQ_n records the number of scored episodes. "
                     "Bootstrap 95% interval over episodes; per-run means follow the supplied root order.",
                     "AQ is the mean over the judges that answered on each round (vhhome-score --judges).",
                     "won: share of episodes the person finished the chore correctly, mean ± half-range over runs; "
                     "S additionally requires finishing inside the time cap C = 2(T_ref + alpha).",
                     "silent and mmduet2 never speak, so they have no AQ; silent is the floor every other number is read against."]}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "final_results.json"), "w"), indent=1)

    def f(x, d=2):
        return "—" if x is None else f"{x:.{d}f}"

    def pm(r, k):
        return "—" if r[k] is None else f"{r[k]:.2f} ± {f(r[k + '_spread'])}"
    md = [f"# VH-Home — final table ({len(ROOTS)} independent run(s), one row per seat)", "",
          "| seat | AQ [95% CI] | AQ by run | won | S (in-time success) | EFF | DET | DT | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        aqs = "—" if r["AQ"] is None else f"{r['AQ']:.3f} [{f(r['AQ_lo'])}, {f(r['AQ_hi'])}]"
        runs = " / ".join(f(x, 3) for x in r["AQ_runs"])
        md.append(f"| {r['seat']} | {aqs} | {runs} | {pm(r, 'won')} | {pm(r, 'S')} | {pm(r, 'EFF')} | {pm(r, 'DET')} | {pm(r, 'DT')} | {pm(r, 'F1')} |")
    md += [""] + [f"- {n}" for n in out["notes"]]
    open(os.path.join(out_dir, "final_results.md"), "w").write("\n".join(md) + "\n")
    print("wrote", os.path.join(out_dir, "final_results.{json,md}"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
