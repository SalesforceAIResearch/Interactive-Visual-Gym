#!/usr/bin/env python3
"""Interaction-quality rubric (SQ v4) on the FIVE reported categories, for all three independent runs — ROUND-LEVEL view (rulings 2026-09-24): a line is clean in a category when none of its applicable items failed; the round score is the mean of its applicable categories, zeroed when a Factual-grounding item (T1 or T2) failed.

Run 1 was judged on the full 16-item rubric, runs 2/3 on the five-category subset (`sq_v4_score.py --categories five`).
Both are re-aggregated here from the per-round, per-judge item verdicts: per round and category, clean = no applicable
item failed (judges averaged); round score = mean over applicable categories; seat score = mean over all its rounds
(pooled); the episode-mean variant is kept in the JSON as AQ_episode_mean. Writes tmp/formal_v5/sq_v4/three_runs_five.json.

    ASSIST_ROOT=tmp/formal_v5 $PY -m cooksim_bench.scoring.sq_v4_three_runs"""
import glob, json, os, sys, collections, statistics as st
from cooksim_bench.paths import ROOT
BENCH = os.environ.get("ASSIST_ROOT", "tmp/formal_v5")
FIVE = {"T1": ("truthful", "must"), "T2": ("truthful", "must"), "S1": ("sensible", 2), "S2": ("sensible", 2),
        "H1": ("helpful", "must"), "H2": ("helpful", 2), "H3": ("helpful", 2), "L1": ("listens", 2), "L2": ("listens", "must"),
        "E1": ("economy", 1)}
CATS = ["truthful", "sensible", "helpful", "listens", "economy"]
MUST = ("T1", "T2")     # ruling 2026-09-24: only the Factual-grounding items zero a round
JUDGES = set((os.environ.get("SQ_JUDGES") or "gemini,gpt").split(","))   # ruling 2026-10-12: the two judges averaged
RULE = os.environ.get("SQ_RULE", "paper")   # paper (default, 2026-11-02): mean of per-judge gated scores | legacy: gate-mean x category-mean

if RULE not in ("paper", "legacy"):
    raise ValueError("SQ_RULE must be paper or legacy")

CAT_ITEMS = {c: [i for i, (cc, w) in FIVE.items() if cc == c] for c in CATS}


def round_clean(items):
    """Ruling 2026-09-24 (round-level view, no must-item zeroing): for each category, the line is CLEAN (1) if none
    of the category's applicable items failed, 0 if any failed, absent if no item applied. The round score is the
    mean over the categories that applied."""
    out = {}
    for c, ids in CAT_ITEMS.items():
        ps = [(items.get(i) or {}).get("pass") for i in ids]
        ps = [p for p in ps if (isinstance(p, bool) if RULE == "paper" else p is not None)]
        if ps:
            out[c] = 0.0 if any(p is False for p in ps) else 1.0
    return out


def parse_tag(model):
    for suf, run in (("__r2", 2), ("__r3", 3)):
        if model.endswith(suf):
            return model[:-4], run
    return model, 1


def main():
    per = collections.defaultdict(list)                      # (seat, run) -> episode AQ (episode = mean over its rounds)
    rounds_pool = collections.defaultdict(list)              # (seat, run) -> round scores (pooled)
    cat_pool = collections.defaultdict(lambda: collections.defaultdict(list))   # (seat, run) -> cat -> clean flags (pooled over rounds)
    files = sorted(glob.glob(os.path.join(ROOT, BENCH, "sq_v4", "frames", "*", "*", "b3_*.json")))
    # SQ_TAGS=tag1,tag2 (2026-11-01): incremental -- only these seats are re-aggregated and merged into the existing JSON
    # (a full pass reads ~5 GB of rubric files from weka; a new seat is 450 of them)
    _tags = set(filter(None, os.environ.get("SQ_TAGS", "").split(",")))
    if _tags:
        files = [f for f in files if f.split(os.sep)[-2].split("__r")[0] in _tags]
        print(f"[three_runs] incremental: {len(files)} rubric files of {sorted(_tags)}", flush=True)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        seat, run = parse_tag(d.get("model") or f.split(os.sep)[-2])
        ep_scores = []
        for r in d.get("rounds") or []:
            # the reported score uses a FIXED judge set (SQ_JUDGES, default gemini,gpt — ruling 2026-10-12) so every episode
            # is averaged over the same judges, whatever extra judges a file happens to carry
            js = [v for k, v in (r.get("judges") or {}).items() if k in JUDGES and isinstance(v, dict) and v.get("items")]
            if not js:
                continue
            # judges averaged: a category is clean with the mean of the judges' clean flags
            cl = collections.defaultdict(list)
            for v in js:
                for c, x in round_clean(v["items"]).items():
                    cl[c].append(x)
            if not cl:
                continue
            cats = {c: sum(v) / len(v) for c, v in cl.items()}
            musts = [0.0 if any((v["items"].get(i) or {}).get("pass") is False for i in MUST) else 1.0 for v in js]
            if RULE == "paper":
                # paper contract (arxiv tables/v7_results_snapshot.json provenance, 2026-10-20; applied 2026-11-02): for EACH judge
                # the five category scores and the factual-grounding gate give that judge's round score; the round is the
                # arithmetic mean of the two judge scores. (Not the same as gate-mean x category-mean, the legacy rule.)
                per_judge = []
                for v in js:
                    rc = round_clean(v["items"])
                    if not rc:
                        continue
                    per_judge.append(musts[js.index(v)] * (sum(rc.values()) / len(rc)))
                if not per_judge:
                    continue
                sc = sum(per_judge) / len(per_judge)
            else:
                # legacy (ruling 2026-09-24): must-gate averaged over judges x mean of the judge-averaged category cleans
                sc = (sum(musts) / len(musts)) * (sum(cats.values()) / len(cats))
            ep_scores.append(sc); rounds_pool[(seat, run)].append(sc)
            for c, x in cats.items():
                cat_pool[(seat, run)][c].append(x)
        if ep_scores:
            per[(seat, run)].append(sum(ep_scores) / len(ep_scores))
    seats = sorted({s for s, r in per}, key=lambda s: -st.mean(rounds_pool.get((s, 1), [0])))
    out = {}
    _outp = os.path.join(ROOT, BENCH, "sq_v4", "three_runs_five.json")
    if _tags and os.path.exists(_outp):          # keep every other seat's entry as it is
        out = {k: v for k, v in json.load(open(_outp)).items() if not k.startswith("_") and k.split("__r")[0] not in _tags}
    for s in seats:
        runs = {}
        for r in (1, 2, 3):
            rp = rounds_pool.get((s, r), []); ep = per.get((s, r), [])
            runs[r] = dict(n=len(ep), rounds=len(rp), AQ=(st.mean(rp) if rp else None), AQ_episode_mean=(st.mean(ep) if ep else None),
                           by_category={c: (st.mean(cat_pool[(s, r)][c]) if cat_pool[(s, r)][c] else None) for c in CATS},
                           applicable={c: (len(cat_pool[(s, r)][c]) / len(rp)) if rp else None for c in CATS})
        aqs = [runs[r]["AQ"] for r in (1, 2, 3) if runs[r]["AQ"] is not None]
        out[s] = dict(runs=runs, mean=(st.mean(aqs) if aqs else None), sd=(st.pstdev(aqs) if len(aqs) > 1 else 0.0))
    if _tags:                                    # merged file keeps the leaderboard order (run-1 AQ, descending)
        out = dict(sorted(((k, v) for k, v in out.items() if not k.startswith("_")), key=lambda kv: -((kv[1]["runs"].get("1") or kv[1]["runs"].get(1) or {}).get("AQ") or 0)))
    out["_rule"] = dict(rule=RULE, judges=sorted(JUDGES), note="round = mean over judges of (factual gate x mean of the judge's applicable category cleans); pooled over rounds per run; mean +- SD over runs" if RULE == "paper" else "legacy: gate-mean x category-mean")
    json.dump(out, open(os.path.join(ROOT, BENCH, "sq_v4", "three_runs_five.json"), "w"), indent=1)
    f2 = lambda x: "–" if x is None else f"{x:.2f}"
    print(f"{'seat':26s} rounds r1/r2/r3      AQ r1/r2/r3     mean±sd   " + "  ".join(f"{c[:5]:>14s}" for c in CATS))
    for s in seats:
        o = out[s]; R = o["runs"]
        cats = "  ".join("/".join(f2(R[r]["by_category"][c]) for r in (1, 2, 3)) for c in CATS)
        print(f"{s:26s} {R[1]['rounds']:5d}/{R[2]['rounds']:5d}/{R[3]['rounds']:5d}  {f2(R[1]['AQ'])}/{f2(R[2]['AQ'])}/{f2(R[3]['AQ'])}  {f2(o['mean'])}±{o['sd']:.2f}  {cats}")
    print("listens applicability (share of rounds), run 1:", {s: f2(out[s]['runs'][1]['applicable']['listens']) for s in seats[:8]})


if __name__ == "__main__":
    main()
