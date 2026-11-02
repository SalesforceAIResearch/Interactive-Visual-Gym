"""Paper interaction-quality aggregation from stored Gemini and GPT verdicts.

python -m screensim_bench.scoring.aq_contract RUN1 RUN2 RUN3 --model MODEL --json scores.json
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

from .five_categories import CATS, GATE, NAMES

from .rubric import usable

JUDGES = ("gemini", "gpt")


def judge_round(items):
    """One judge's five category flags, its gate, and its round score; None if nothing applied."""
    flags = {}
    for cat, ids in CATS.items():
        got = [items[i]["pass"] for i in ids if i in items and isinstance(items[i].get("pass"), bool)]
        if got:
            flags[cat] = 1.0 if all(got) else 0.0
    if not flags:
        return None
    gated = any((items.get(i) or {}).get("pass") is False for i in GATE)
    return {"flags": flags, "gated": gated,
            "score": 0.0 if gated else sum(flags.values()) / len(flags)}


def rounds(obj, judges=JUDGES):
    """Per-round contract scores: the mean of the judges that returned a usable verdict."""
    d = obj if isinstance(obj, dict) else json.load(open(obj))
    out = []
    for r in d.get("rounds_detail") or []:
        per = []
        for jn in judges:
            v = (r.get("judges") or {}).get(jn)
            if not usable(v):
                continue
            jr = judge_round(v.get("items") or {})
            if jr:
                per.append(jr)
        if not per:
            continue
        cats = {}
        for cat in CATS:
            vals = [p["flags"][cat] for p in per if cat in p["flags"]]
            if vals:
                cats[cat] = st.mean(vals)
        out.append({"score": st.mean([p["score"] for p in per]), "flags": cats,
                    "gated_share": st.mean([1.0 if p["gated"] else 0.0 for p in per]),
                    "judges": len(per)})
    return out


def run_scores(root, model, judges=JUDGES):
    sub = "aq_v4five" if os.path.isdir(os.path.join(root, "aq_v4five", model.replace(" ", "_"))) else "aq_v4"
    files = sorted(glob.glob(os.path.join(root, sub, model.replace(" ", "_"), "*.json")))
    rs = [x for f in files for x in rounds(f, judges)]
    if not rs:
        return None
    cats = {}
    for cat in CATS:
        vals = [r["flags"][cat] for r in rs if cat in r["flags"]]
        cats[cat] = {"score": st.mean(vals) if vals else None, "applies": len(vals) / len(rs)}
    one_judge = sum(1 for r in rs if r["judges"] < 2)
    return {"root": root, "subset": sub, "episodes": len(files), "rounds": len(rs),
            "AQ": st.mean([r["score"] for r in rs]), "by_category": cats,
            "gate_share": st.mean([r["gated_share"] for r in rs]), "rounds_one_judge": one_judge}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    per = [r for r in (run_scores(x, a.model) for x in a.roots) if r]
    if not per:
        sys.exit(f"no graded rounds for {a.model} under {a.roots}")
    out = {"model": a.model, "contract": "per-judge five categories + gate, then the mean of the "
                                         "judges; rounds pooled per run, runs averaged equally",
           "judges": list(JUDGES), "runs": per,
           "AQ": st.mean([r["AQ"] for r in per]), "AQ_sd": st.pstdev([r["AQ"] for r in per]),
           "by_category": {c: (st.mean(vals) if (vals := [r["by_category"][c]["score"] for r in per
                                       if r["by_category"][c]["score"] is not None]) else None) for c in CATS}}
    for r in per:
        print(f'{os.path.basename(r["root"]):10} {r["subset"]:10} {r["episodes"]:3} eps '
              f'{r["rounds"]:4} rounds  AQ={r["AQ"]:.3f}  gate={r["gate_share"]:.3f}'
              f'  one-judge rounds={r["rounds_one_judge"]}')
    print(f'\nAQ ({len(per)}-run mean) = {out["AQ"]:.3f}  SD {out["AQ_sd"]:.3f}')
    for c, v in out["by_category"].items():
        print(f'  {NAMES[c]:24} {v:.3f}' if v is not None else f'  {NAMES[c]:24} n/a')
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
