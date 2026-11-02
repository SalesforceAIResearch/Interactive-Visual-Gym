"""Five-category interaction quality: gate each judge, then average Gemini and GPT.

legacy_rounds retains the historical merged-item calculation for reproduction only.
"""
from __future__ import annotations

import json

CATS = {"truthful": ("T1", "T2"), "sensible": ("S1", "S2"), "helpful": ("H1", "H2", "H3"),
        "listens": ("L1", "L2"), "economy": ("E1",)}
NAMES = {"truthful": "Factual grounding", "sensible": "Situational relevance",
         "helpful": "Actionable guidance", "listens": "User intent uptake",
         "economy": "Guidance conciseness"}
GATE = ("T1", "T2")                       # Factual grounding gates the round


def legacy_rounds(path_or_obj):
    """One dict per graded line: its score, its per-category clean flags, and whether a must item failed."""
    d = path_or_obj if isinstance(path_or_obj, dict) else json.load(open(path_or_obj))
    out = []
    for r in d.get("rounds_detail") or []:
        items = r.get("items") or {}
        if r.get("error") or not items:
            continue
        flags, applicable = {}, False
        for cat, ids in CATS.items():
            got = [items[i]["pass"] for i in ids if i in items and items[i].get("pass") is not None]
            if not got:
                continue
            applicable = True
            flags[cat] = 1.0 if all(got) else 0.0
        if not applicable:
            continue
        gated = any((items.get(i) or {}).get("pass") is False for i in GATE)
        out.append({"score": 0.0 if gated else sum(flags.values()) / len(flags),
                    "flags": flags, "must_fail": gated})
    return out


def rounds(path_or_obj):
    from .aq_contract import rounds as contract_rounds
    return contract_rounds(path_or_obj)


def episode(path_or_obj):
    """Episode-level view (kept as the variant CookSim mentions): mean round score within the episode."""
    rs = rounds(path_or_obj)
    return {"AQ": (sum(r["score"] for r in rs) / len(rs)) if rs else None, "rounds": len(rs)}


def aggregate(all_rounds):
    """Seat-level numbers: AQ over rounds, per-category clean share, and how often `listens` applies."""
    n = len(all_rounds)
    if not n:
        return None
    cats = {}
    for cat in CATS:
        got = [r["flags"][cat] for r in all_rounds if cat in r["flags"]]
        cats[cat] = {"clean": (sum(got) / len(got)) if got else None, "applies": len(got) / n}
    return {"aggregation": "per-judge-gated-two-judge-mean", "AQ": sum(r["score"] for r in all_rounds) / n, "rounds": n,
            "must_fail_share": sum(r.get("gated_share", float(r.get("must_fail", False))) for r in all_rounds) / n, "by_category": cats}
