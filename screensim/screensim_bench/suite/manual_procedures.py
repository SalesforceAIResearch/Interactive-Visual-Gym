"""Parse every mirrored manual article into structured procedures.

Each article in docs/apple_manual/ becomes a record with its numbered step
sequences and every explicit `Go to Settings > …` path, so the manual can be
consumed programmatically: coverage checks, task generation, assistant RAG.

  python screensim_bench/suite/manual_procedures.py           # writes docs/apple_manual/_procedures.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "apple_manual")
OUT = os.path.join(SRC, "_procedures.jsonl")

# "Go to Settings > General > Date & Time." / "go to Settings > Apps > Mail,"
PATH_RE = re.compile(r"[Gg]o to Settings\s*((?:>\s*[^>.,;:()\n]+)+)")
STEP_RE = re.compile(r"^(\d+)\.\s+(.*)$")
# controls a step names: tap X / turn on X / turn off X / choose X / select X
CTRL_RE = re.compile(
    r"\b(?:[Tt]ap|[Tt]urn on|[Tt]urn off|[Cc]hoose|[Ss]elect)\s+"
    r"((?:[A-Z][\w&’'\-]*(?:\s|,|$)|\"[^\"]+\"|“[^”]+”)+)")


def parse_path(m: str) -> list:
    segs = [s.strip().rstrip(".,;:") for s in m.split(">")]
    segs = [s for s in segs if s]
    # placeholder segments like "[ your name ]" / "[the app]" stay marked
    return segs


def parse_article(path: str) -> dict:
    slug = os.path.basename(path)[:-3]
    lines = open(path).read().splitlines()
    src = lines[0].replace("source: ", "").strip() if lines else ""
    title = ""
    steps_blocks, cur = [], []
    paths = []
    for ln in lines:
        if not title and ln.startswith("# "):
            title = ln[2:].strip()
        sm = STEP_RE.match(ln.strip())
        if sm:
            n = int(sm.group(1))
            if n == 1 and cur:
                steps_blocks.append(cur)
                cur = []
            cur.append(sm.group(2).strip())
        for pm in PATH_RE.finditer(ln):
            segs = parse_path(pm.group(1))
            if segs:
                paths.append(segs)
    if cur:
        steps_blocks.append(cur)
    # controls named across all steps
    controls = []
    for block in steps_blocks:
        for st in block:
            for cm in CTRL_RE.finditer(st):
                c = cm.group(1).strip().rstrip(".,;:")
                if 2 <= len(c) <= 60:
                    controls.append(c)
    # dedupe, keep order
    seen, ctl = set(), []
    for c in controls:
        if c not in seen:
            seen.add(c)
            ctl.append(c)
    return {"id": slug, "title": title, "source": src,
            "n_step_blocks": len(steps_blocks), "steps": steps_blocks,
            "settings_paths": paths, "controls": ctl[:40]}


def main():
    arts = sorted(f for f in os.listdir(SRC)
                  if f.endswith(".md") and not f.startswith(("INDEX", "_")))
    n_steps = n_paths = 0
    with open(OUT, "w") as f:
        for a in arts:
            rec = parse_article(os.path.join(SRC, a))
            n_steps += rec["n_step_blocks"]
            n_paths += len(rec["settings_paths"])
            f.write(json.dumps(rec) + "\n")
    print(f"{len(arts)} articles -> {n_steps} step blocks, "
          f"{n_paths} settings paths\n-> {OUT}")


if __name__ == "__main__":
    main()
