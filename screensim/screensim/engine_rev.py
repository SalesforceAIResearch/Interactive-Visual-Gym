"""ENGINE REVISION STAMP — a short hash of every file that changes what an episode does.

Anything published (films, recordings, results) carries `engine_rev()`; a page whose
stamp differs from the current engine is stale and must be re-shot, not argued about.
Covers: the phone engine and apps, the renderer (what the assistant sees), the tasks,
and the interaction harness (runner, designer, agents, prompts, personas, readout, manual).
"""
from __future__ import annotations

import glob
import hashlib
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PATTERNS = ["core/*.py", "apps/*.py", "render/*.py", "tasks.py",
            "interact/composite_run_v2.py", "interact/composite.py", "interact/agents.py",
            "interact/prompts.py", "interact/personas.py", "interact/readout.py",
            "interact/manual.py", "interact/plan_ir.py", "interact/errors.py",
            "interact/composite_run.py"]


def engine_files():
    out = []
    for p in PATTERNS:
        out.extend(sorted(glob.glob(os.path.join(ROOT, p))))
    return [f for f in out if not f.endswith("_backup.py")]


_LOADED = None


def engine_rev() -> str:
    """The revision of the code THIS process loaded: hashed once, at the first call (the harness calls it
    at import time). Hashing the files on every call stamped late records of a long run with whatever
    was on disk when they finished, not the code that produced them (formal run bench_v1)."""
    global _LOADED
    if _LOADED is None:
        _LOADED = _hash_now()
    return _LOADED


def _hash_now() -> str:
    h = hashlib.sha256()
    for f in engine_files():
        h.update(os.path.relpath(f, ROOT).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:10]


if __name__ == "__main__":
    print(engine_rev())
