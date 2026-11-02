"""Paper quality reducer: per-judge factual gating, Gemini/GPT mean, pooled rounds."""
import json
import statistics as st

CATS = {"truthful": ("T1", "T2"), "sensible": ("S1", "S2"),
        "helpful": ("H1", "H2", "H3"), "listens": ("L1", "L2"), "economy": ("E1",)}
GATE = ("T1", "T2")

def usable(v):
    return isinstance(v, dict) and not v.get("error") and bool(v.get("items"))

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
    for r in d.get("rounds") or []:
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

