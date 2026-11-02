#!/usr/bin/env python3
"""The arm-level table from the per-round rubric judgements.

    python -m vhhome_bench.scoring.rubric_table runs/mine [--json out.json] [--by-category] [--by-error]

AQ is reported two ways because they answer different questions and can disagree:
  AQ_ep    mean over EPISODES of that episode's mean round score — every episode counts once, so a
           talkative episode does not outvote a terse one. This is the headline.
  AQ_round mean over ALL rounds — what a randomly drawn line scores.
A 95% interval on AQ_ep comes from a 2000-draw bootstrap over episodes (the unit that was sampled).

Also shown: rounds per episode (the assistant's talkativeness, which AQ deliberately does not reward),
the share of rounds zeroed by a must-item, and, with --by-category, where an arm actually loses its
points. Detection and outcome come from the engine's own table so quality can be read against them: an
arm can rank high on detection and low on AQ, which is the divergence the rubric exists to expose.
"""
import argparse, glob, json, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)
CATS = ["truthful", "sensible", "specific", "helpful", "well_timed", "listens", "honest", "safe",
        "consistent", "economy"]


def boot(vals, n=2000, seed=11):
    if len(vals) < 2:
        return (None, None)
    r = random.Random(seed)
    ms = sorted(statistics.fmean(r.choices(vals, k=len(vals))) for _ in range(n))
    return (ms[int(.025 * n)], ms[int(.975 * n)])


def vs_person(root):
    """Does the rubric agree with the person who was actually there?

    Every episode carries the simulated person's own end-of-episode ratings (helpful / timing / trust,
    1-5) written in persona, which no part of the rubric sees. If AQ is measuring the thing people react
    to, episodes the person rated highly should score higher. This is an external check on the
    instrument, not a metric: agreement supports the rubric, and disagreement is worth reading before
    either number is published."""
    pairs = []
    for f in glob.glob(os.path.join(root, "sq_v4", "*", "*.json")):
        try:
            j = json.load(open(f))
            rec = json.load(open(os.path.join(root, j["arm"], os.path.basename(f))))
        except Exception:
            continue
        aq = (j.get("aggregate") or {}).get("AQ")
        iq = rec.get("interaction_quality") or {}
        if aq is None or not iq:
            continue
        vals = [iq.get(k) for k in ("helpful", "timing", "trust") if isinstance(iq.get(k), (int, float))]
        if vals:
            pairs.append((aq, statistics.fmean(vals), j["arm"]))
    return pairs


def by_error(root):
    """AQ per forced-error class: the rounds that were live under each kind of mistake. It answers a
    different question from the arm table — not who talks well, but which mistake the assistants as a
    group handle badly."""
    per = {}
    for f in glob.glob(os.path.join(root, "sq_v4", "*", "*.json")):
        try:
            x = json.load(open(f))
        except Exception:
            continue
        for r in x.get("rounds") or []:
            if r.get("score") is None:
                continue
            per.setdefault(r.get("live_error") or "(no error live)", []).append(r["score"])
    return per


def arm_rows(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "sq_v4", "*/"))):
        if d.rstrip("/").endswith("claude_batches"):
            continue
        arm = d.rstrip("/").split("/")[-1]
        eps, rounds_all, cat_hits, must, n_rounds, judged = [], [], {c: [] for c in CATS}, [], 0, 0
        for f in sorted(glob.glob(d + "*.json")):
            try:
                x = json.load(open(f))
            except Exception:
                continue
            judged += 1
            from .five_categories import rounds as paper_rounds
            rs = paper_rounds(x)
            n_rounds += len(x.get("rounds") or [])
            if rs:
                eps.append(statistics.fmean(r["score"] for r in rs))
            for rr in rs:
                rounds_all.append(rr["score"])
                must.append(rr["gated_share"])
                for c, value in rr["flags"].items():
                    if c in cat_hits:
                        cat_hits[c].append(value)
        lo, hi = boot(eps)
        rows.append({"arm": arm, "episodes_judged": judged, "episodes_scored": len(eps),
                     "rounds": n_rounds, "rounds_scored": len(rounds_all),
                     "rounds_per_ep": round(n_rounds / judged, 1) if judged else None,
                     "AQ_ep": statistics.fmean(eps) if eps else None,
                     "AQ_ep_lo": lo, "AQ_ep_hi": hi,
                     "AQ_round": statistics.fmean(rounds_all) if rounds_all else None,
                     "must_fail_share": statistics.fmean(must) if must else None,
                     "by_category": {c: (statistics.fmean(v) if v else None)
                                     for c, v in cat_hits.items()}})
    return rows


def engine_numbers(root):
    p = os.path.join(root, "results_table.json")
    if not os.path.exists(p):
        return {}
    try:
        return {r["tag"]: r for r in json.load(open(p))}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    ap.add_argument("--by-category", action="store_true")
    ap.add_argument("--vs-person", action="store_true",
                    help="check AQ against the person's own end-of-episode ratings")
    ap.add_argument("--by-error", action="store_true",
                    help="AQ of the rounds live under each forced-error class")
    a = ap.parse_args()
    root = a.root if os.path.isabs(a.root) else os.path.join(ENG, a.root)
    rows = arm_rows(root)
    eng = engine_numbers(root)
    rows.sort(key=lambda r: (r["AQ_ep"] is None, -(r["AQ_ep"] or 0)))
    print(f"{'arm':24} {'eps':>5} {'rnds':>6} {'r/ep':>5} {'AQ_ep':>7} {'95% CI':>15} "
          f"{'AQ_rnd':>7} {'must0':>6}   {'caught':>7} {'won':>5}")
    for r in rows:
        e = eng.get(r["arm"], {})
        det = (f"{100*e['caught']/e['fired']:.0f}%" if e.get("fired") else "—")
        won = (f"{100*e['won']/e['n']:.0f}%" if e.get("n") else "—")
        ci = (f"[{r['AQ_ep_lo']:.2f},{r['AQ_ep_hi']:.2f}]"
              if r["AQ_ep_lo"] is not None else "—")
        aq = f"{r['AQ_ep']:.3f}" if r["AQ_ep"] is not None else "—"
        aqr = f"{r['AQ_round']:.3f}" if r["AQ_round"] is not None else "—"
        mf = f"{r['must_fail_share']:.2f}" if r["must_fail_share"] is not None else "—"
        print(f"{r['arm']:24} {r['episodes_judged']:>5} {r['rounds']:>6} {r['rounds_per_ep']:>5} "
              f"{aq:>7} {ci:>15} {aqr:>7} {mf:>6}   {det:>7} {won:>5}")
    if a.by_category:
        # One row per arm, one column per RUBRIC CATEGORY, with the overall AQ first: the interaction
        # quality table (the request 2026-09-23). A category's value is the pass rate of its items over
        # every round where they applied, so it reads directly as "how often this arm got this right".
        head = ("truthful", "sensible", "specific", "helpful", "timed", "listens", "honest", "safe",
                "consist", "economy")
        print(f"\n{'arm':22} {'rounds':>7} {'AQ':>6} " + " ".join(f"{h:>8}" for h in head))
        for r in rows:
            aq = f"{r['AQ_ep']:.3f}" if r["AQ_ep"] is not None else "—"
            print(f"{r['arm']:22} {r['rounds']:>7} {aq:>6} " + " ".join(
                f"{(f'{v:.2f}' if v is not None else '—'):>8}" for v in
                (r["by_category"][c] for c in CATS)))
        # A category that never fails anywhere is not a finding, it is an item that cannot vary in a
        # house. CookSim's metrics contract says to report such a thing as NOT MEASURED rather than as a
        # perfect score, so it is named here instead of being read as an arm's strength.
        flat = [c for c in CATS
                if all((r["by_category"][c] is None or r["by_category"][c] == 1.0) for r in rows)
                and any(r["by_category"][c] is not None for r in rows)]
        if flat:
            print("\nNOT MEASURED (never fails in any arm, so it separates nothing here): "
                  + ", ".join(flat))
    if a.by_error:
        per = by_error(root)
        print(f"\n{'live forced error':26} {'rounds':>7} {'AQ':>7}   {'95% CI':>15}")
        for k, v in sorted(per.items(), key=lambda kv: -statistics.fmean(kv[1])):
            lo, hi = boot(v)
            ci = f"[{lo:.2f},{hi:.2f}]" if lo is not None else "—"
            print(f"{k:26} {len(v):>7} {statistics.fmean(v):>7.3f}   {ci:>15}")
    if a.vs_person:
        pairs = vs_person(root)
        if len(pairs) > 2:
            xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
            r = statistics.correlation(xs, ys) if len(set(xs)) > 1 and len(set(ys)) > 1 else None
            print(f"\nAQ vs the person's own rating (helpful/timing/trust, mean), {len(pairs)} episodes:"
                  f"  Pearson r = {r:.3f}" if r is not None else "")
            print(f"{'person rated':>14} {'episodes':>9} {'mean AQ':>9}")
            for lo, hi, lab in ((1, 1.99, "1 (worst)"), (2, 2.99, "2"), (3, 3.99, "3"), (4, 5.01, "4-5 (best)")):
                v = [x for x, y, _ in pairs if lo <= y <= hi]
                if v:
                    print(f"{lab:>14} {len(v):>9} {statistics.fmean(v):>9.3f}")
    silent = [r for r in rows if r["arm"] == "silent"]
    if silent and silent[0]["rounds"] == 0:
        print("\n(the silent control has no lines, so it has no AQ — that is the correct answer, "
              "not a zero)")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
