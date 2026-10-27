"""Load the auto-generated manual-grounded task library.

tools/gen_tasks.py derives these from the mirrored Apple iPhone User Guide:
each record's plan was replayed at generation time and its goal is the exact
state diff the actuation produced, so every loaded Task carries verified
ground truth plus a `mirrors` citation to its source article.

    from screensim.autotasks import load_auto_tasks
    tasks = load_auto_tasks()          # [] if the corpus isn't mirrored
"""
from __future__ import annotations

import json
import os

from .tasks import Task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "apple_manual", "_auto_tasks.jsonl")


def _goal(changes):
    def g(st):
        d = st.data["settings"]
        for path, exp in changes:
            cur = d
            for part in path.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            if cur != exp:
                return False, f"{path}={cur!r} != {exp!r}"
        return True, "ok"
    return g


def load_auto_tasks() -> list:
    if not os.path.exists(SRC):
        return []
    out = []
    for line in open(SRC):
        r = json.loads(line)
        out.append(Task(
            id=r["id"], family="auto",
            instruction=r["instruction"], plan=r["plan"],
            goal=_goal([tuple(c) for c in r["changes"]]),
            start_app="settings", mirrors=r["source"]))
    return out
