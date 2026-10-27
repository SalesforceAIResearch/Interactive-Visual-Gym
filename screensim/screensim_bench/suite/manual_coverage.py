"""Replay every `Go to Settings > …` path from the manual against the engine.

The completeness contract is: every article's procedure should be replicable
in the engine. This walks each extracted path with real gestures (tap, swipe
to reveal, ordinal fallback) and reports, per unique path: WALKABLE or the
exact segment where it breaks — ranked into a gap list that drives spec-data
edits in settings_data.py.

  python screensim_bench/suite/manual_coverage.py            # summary + top gaps
  python screensim_bench/suite/manual_coverage.py --full     # every failing path
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict


from screensim.core.engine import Device                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCS = os.path.join(ROOT, "docs", "apple_manual", "_procedures.jsonl")
OUT = os.path.join(ROOT, "docs", "apple_manual", "_coverage.json")

# placeholder segments -> concrete sim equivalents
SUBS = {
    "[ your name ]": "Chris Morgan", "[your name]": "Chris Morgan",
    "your name": "Chris Morgan",
}
APP_PLACEHOLDER = re.compile(r"\[.*app.*\]|an app$|the app$", re.I)
DYNAMIC = re.compile(r"\[|\.\.\.|…")

# paths the iPhone Settings engine legitimately cannot walk — with reasons
OUT_OF_SCOPE = {
    "Announce Messages": "CarPlay's in-car Settings app, not iPhone Settings",
    "[ your name ] > Name": "article uses the pre-iOS-18 label; iOS 18 "
                            "renamed the row 'Personal Information' (present)",
}


def scope_key(segs):
    return " > ".join(s.replace("\xa0", " ").strip() for s in segs)


def norm_segment(seg: str):
    s = seg.replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    # prose artifacts: the sentence continues past the real segment
    # ("Bluetooth and make sure Bluetooth is turned on", "Battery to see …")
    s = re.sub(r"\s+on your iPhone$", "", s, flags=re.I)
    s = re.sub(r"\s+(?:and|to|so that|then)\s+(?:make sure|see|check|turn|"
               r"review|verify)\b.*$", "", s, flags=re.I)
    s = s.replace("[ device ]", "iPhone").replace("[device]", "iPhone")
    if s in SUBS:
        return SUBS[s]
    if APP_PLACEHOLDER.search(s):
        return "Safari"                     # a stock app every list carries
    if DYNAMIC.search(s):
        return None                         # e.g. "[network name]" — dynamic
    return s


def walk(segs: list) -> tuple:
    """Try to walk one path. Returns (depth_reached, total, fail_segment)."""
    d = Device(0)
    d.open_app("settings")
    for i, raw in enumerate(segs):
        seg = norm_segment(raw)
        if seg is None:
            if i == len(segs) - 1:
                # trailing dynamic segment ("[ your profile ]") — the tree is
                # reachable; the leaf is user-created content
                return len(segs), len(segs), ""
            return i, len(segs), f"<dynamic:{raw}>"
        ok = False
        for attempt in range(6):
            r = d.act(f"tap {seg}")
            if r.ok:
                ok = True
                break
            if getattr(r, "ambiguous", False):
                r2 = d.act(f"tap the first {seg}")
                if r2.ok:
                    ok = True
                    break
            d.act("swipe up")
        if not ok:
            return i, len(segs), seg
    return len(segs), len(segs), ""


def main():
    full = "--full" in sys.argv
    paths = set()
    by_path_articles = defaultdict(list)
    for line in open(PROCS):
        rec = json.loads(line)
        for p in rec["settings_paths"]:
            t = tuple(p)
            paths.add(t)
            by_path_articles[t].append(rec["id"])

    results, gaps = [], Counter()
    gap_where = defaultdict(set)
    n_ok = n_oos = 0
    for t in sorted(paths):
        if scope_key(t) in OUT_OF_SCOPE:
            n_oos += 1
            results.append({"path": list(t), "ok": None,
                            "out_of_scope": OUT_OF_SCOPE[scope_key(t)],
                            "articles": by_path_articles[t][:6]})
            continue
        depth, total, fail = walk(list(t))
        okay = depth == total
        n_ok += okay
        results.append({"path": list(t), "ok": okay, "depth": depth,
                        "fail": fail, "articles": by_path_articles[t][:6]})
        if not okay:
            gaps[fail] += 1
            gap_where[fail].add(" > ".join(t[:depth]) or "root")

    in_scope = len(paths) - n_oos
    pct = 100.0 * n_ok / in_scope
    print(f"unique Settings paths: {len(paths)}  "
          f"({n_oos} out of scope, with reasons in the report)")
    print(f"fully walkable:        {n_ok}/{in_scope}  ({pct:.1f}%)")
    print(f"\n--- TOP GAPS (missing segment × how many paths need it) ---")
    for seg, n in gaps.most_common(25):
        wheres = sorted(gap_where[seg])[:2]
        print(f"  {n:3}×  {seg!r:44} first needed at: {wheres[0][:60]}")
    if full:
        print("\n--- ALL FAILING PATHS ---")
        for r in results:
            if not r["ok"]:
                print(f"  {' > '.join(r['path'])}   [breaks at "
                      f"{r['fail']!r}]")
    with open(OUT, "w") as f:
        json.dump({"n_paths": len(paths), "n_ok": n_ok,
                   "n_out_of_scope": n_oos, "pct": pct,
                   "results": results}, f, indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
