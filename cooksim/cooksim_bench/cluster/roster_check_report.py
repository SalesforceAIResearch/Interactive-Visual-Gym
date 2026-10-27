#!/usr/bin/env python3
"""Summarise a roster check (python -m cooksim_bench.cluster.run_roster_check): is each assistant's SETUP healthy?

    $PY python -m cooksim_bench.cluster.roster_check_report tmp/roster_check [--md out.md]

Per assistant: episodes finished (of 3), outcomes, assistant replies, how many replies the harness could
parse, how many spoke, flags raised, median reply latency, and any backend error in the run logs. These
are health signals, not scores: an assistant can be healthy and still play badly.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

from cooksim_bench.roster import ROSTER, CHECK_EPISODES      # noqa: E402

ERR = re.compile(r"(Traceback|RuntimeError|call failed|ConnectionError|Timeout|HTTP 4\d\d|HTTP 5\d\d)")


def health(root, entry):
    recs, errs = [], 0
    for persona, case in CHECK_EPISODES:
        p = os.path.join(root, "frames", persona, entry["tag"], case + ".json")
        if os.path.exists(p):
            try:
                recs.append(json.load(open(p)))
            except Exception:
                pass
        lf = os.path.join(root, "logs", f"{entry['tag']}__{persona}__{case}.log")
        if os.path.exists(lf):
            errs += len(ERR.findall(open(lf, errors="replace").read()))
    replies = [c for r in recs for c in r.get("conversation") or [] if c.get("speaker") == "vlm"]
    parsed = [c for c in replies if isinstance(c.get("vlm_json"), dict) and c["vlm_json"]]
    spoke = [c for c in replies if (c.get("text") or "").strip()]
    flags = sum(len(r.get("flags") or []) for r in recs)
    lat = [x.get("latency_s") for r in recs for x in r.get("assist_latency") or [] if x.get("latency_s") is not None]
    native = [u for r in recs for u in (r.get("native_assistant") or {}).get("utterances", [])]
    return {"done": len(recs), "outcomes": [r.get("outcome") for r in recs], "replies": len(replies),
            "parsed": len(parsed), "spoke": len(spoke), "flags": flags,
            "lat_med": st.median(lat) if lat else None, "log_errors": errs, "native_utts": len(native),
            "ticks": [r.get("ticks") for r in recs]}


def verdict(h, entry):
    if h["done"] < len(CHECK_EPISODES):
        return "INCOMPLETE"
    if entry["kind"] == "native":
        return "ok" if h["log_errors"] == 0 else "check logs"
    if h["replies"] == 0:
        return "NO REPLIES"
    if h["parsed"] / h["replies"] < 0.9:
        return "PARSE ISSUES"
    return "ok" if h["log_errors"] == 0 else "ok (errors in log)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md")
    a = ap.parse_args()
    meta = {}
    mp = os.path.join(a.root, "check_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    rows = ["| slot | assistant | episodes | outcomes | replies parsed | spoke | flags | median latency | log errors | health |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for e in ROSTER:
        # no filter on check_meta.json: parallel Beaker checks share a root and each rewrites it
        h = health(a.root, e)
        if h["done"] == 0 and h["log_errors"] == 0:
            continue
        lat = f"{h['lat_med']:.1f} s" if h["lat_med"] is not None else "—"
        rep = f"{h['parsed']}/{h['replies']}" if e["kind"] != "native" else f"{h['native_utts']} utterances"
        rows.append(f"| {e['slot']} | {e['tag']} | {h['done']}/3 | {', '.join(str(o) for o in h['outcomes'])} | {rep} | "
                    f"{h['spoke']} | {h['flags']} | {lat} | {h['log_errors']} | {verdict(h, e)} |")
    out = "\n".join(rows)
    if meta.get("skipped"):
        out += "\n\nNot run: " + "; ".join(f"{t} ({why})" for t, why in meta["skipped"])
    if meta.get("engine_commit"):
        out += f"\n\nEngine commit {meta['engine_commit'][:10]}; episodes: " + ", ".join(f"{p}/{c}" for p, c in CHECK_EPISODES)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()
