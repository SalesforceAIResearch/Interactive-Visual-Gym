#!/usr/bin/env python3
"""How well did the simulated person behave in one episode, measured against the engine.

The person is an LLM too, so an example is only worth showing if the person's own conduct holds up.
Everything here is checked against the screen the engine says was in front of them:

  refused gestures        the phone rejected the tap (a gesture the person could not actually make)
  false sight claims      the person says they cannot see a control that IS visible on that screen
  duplicate lines         the person repeats a line word for word
  longest identical run   the longest run of the same gesture back to back (a loop, if it is long)
  guidance asks           `ask_for_more_guidance` moves — by design a low-skill person asks each step
  own repairs             forced mistakes the person put right after being told

    python screensim_bench/scoring/person_quality.py <records.jsonl> --episode E --persona P --contestant C
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import re
import sys


import screensim_bench.scoring.rubric as A

BLIND = re.compile(r"\b(i (don'?t|do not|can'?t|cannot) (see|find)|where (is|do i find)|i'?m not seeing)\b", re.I)
NAME = re.compile(r"['\"‘“]?([A-Z][\w&+ -]{2,34})['\"’”]?")


def visible(dev):
    out = set()
    for w in dev.laid_out_screen().widgets:
        lab = re.sub(r"\s+", " ", str(getattr(w, "label", "") or "")).strip().lower()
        if lab and getattr(w, "visible", False):
            out.add(lab)
    return out


def audit(r):
    tl = r["timeline"]
    rep = A.Replay(r)
    ges = [e.get("text") or "" for e in tl if e["kind"] == "gesture"]
    refused = [e for e in tl if e["kind"] == "gesture" and e.get("ok") is False]
    lines = [str(e.get("text") or "") for e in tl if e["kind"] == "say" and e.get("who") == "human"]
    dup = sum(c - 1 for c in collections.Counter(lines).values() if c > 1)
    longest = max((len(list(g)) for _k, g in itertools.groupby(ges)), default=0)
    false_sight = []
    for i, e in enumerate(tl):
        if e["kind"] != "say" or e.get("who") != "human" or not BLIND.search(str(e.get("text") or "")):
            continue
        rep.advance_to(i)
        vis = visible(rep.dev)
        for m in NAME.finditer(str(e.get("text"))):
            lab = m.group(1).strip().lower()
            if len(lab) > 3 and any(lab == v or lab in v for v in vis):
                false_sight.append({"t": e["t"], "said": e.get("text"), "but_visible": lab})
    return {"episode": r["episode"], "persona": r["persona"], "assistant": r["contestant"],
            "gestures": len(ges), "refused_gestures": len(refused),
            "false_sight_claims": len(false_sight), "false_sight_detail": false_sight,
            "person_lines": len(lines), "duplicate_lines": dup,
            "longest_identical_gesture_run": longest,
            "guidance_asks": sum(1 for e in tl if e["kind"] == "move" and e.get("text") == "ask_for_more_guidance"),
            "forced_mistakes": sum(1 for b in r["beats"] if b.get("injected_at") is not None),
            "own_repairs": r.get("human_fixes"), "denials": r.get("denials"),
            "ambiguous_calls": r.get("ambiguous_calls"),
            "goal_met": bool(r.get("goal_ok")), "extra_changes": r.get("extra_changes") or [],
            "ticks": r.get("final_tick")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--contestant", required=True)
    a = ap.parse_args()
    rec = next(json.loads(l) for l in open(a.records)
               if json.loads(l).get("episode") == a.episode and json.loads(l).get("persona") == a.persona
               and json.loads(l).get("contestant") == a.contestant)
    print(json.dumps(audit(rec), indent=1))


if __name__ == "__main__":
    main()
