#!/usr/bin/env python3
"""The five reported categories of the AQ rubric, scored by CookSim's 2026-09-24 rules.

Experiment 2 reports five of the rubric's ten categories:

    Factual grounding     truthful  T1 T2      Actionable guidance  helpful  H1 H2 H3
    Situational relevance sensible  S1 S2      User intent uptake   listens  L1 L2
    Guidance conciseness  economy   E1

Scoring is round-level, not item-level (CookSim's rulings of 2026-09-24). For one round and one category
the line is *clean* (1) when none of that category's applicable items failed and 0 when any failed; a
category with no applicable item is absent for that round. Factual grounding is a **gate**: a line that
fails T1 (a claim about the screen or about what the person just did is false) or T2 (it names a control,
page or app that does not exist on this phone) scores 0 as a round, whatever else it got right — advice
built on a false picture of the phone is not credited for being relevant, actionable or concise. Otherwise
the round scores the mean of its applicable clean flags, with Factual grounding = 1 by construction.
H1 and L2 are no longer gates; a failure there costs its own category only. AQ is the mean over rounds,
every line weighted equally (the episode-mean variant is kept alongside). Category columns are the share
of applicable rounds clean in that category, independent of the gate.

Runs 2 and 3 were judged on these ten items directly (`aq_v4_score.py --subset five`, folder `aq_v4five`);
run 1 was judged on all sixteen and is re-aggregated here from its stored per-item verdicts.
"""
from __future__ import annotations

import json

CATS = {"truthful": ("T1", "T2"), "sensible": ("S1", "S2"), "helpful": ("H1", "H2", "H3"),
        "listens": ("L1", "L2"), "economy": ("E1",)}
NAMES = {"truthful": "Factual grounding", "sensible": "Situational relevance",
         "helpful": "Actionable guidance", "listens": "User intent uptake",
         "economy": "Guidance conciseness"}
GATE = ("T1", "T2")                       # Factual grounding gates the round


def rounds(path_or_obj):
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
    return {"AQ": sum(r["score"] for r in all_rounds) / n, "rounds": n,
            "must_fail_share": sum(1 for r in all_rounds if r["must_fail"]) / n, "by_category": cats}
