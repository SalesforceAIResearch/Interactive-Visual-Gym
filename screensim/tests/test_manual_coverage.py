"""Manual-completeness regression: every in-scope `Go to Settings > …` path
from the mirrored Apple iPhone User Guide must stay walkable in the engine.

This is the enforcement of "everything in the manual can be simulated": if a
spec edit breaks a path the official manual describes, this test names it.
Requires docs/apple_manual/_procedures.jsonl (built by tools/manual_procedures.py
from the mirrored corpus) — skipped if the corpus isn't present.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCS = os.path.join(ROOT, "docs", "apple_manual", "_procedures.jsonl")

pytestmark = pytest.mark.skipif(not os.path.exists(PROCS),
                                reason="manual corpus not mirrored")


def test_every_in_scope_manual_path_walks():
    import manual_coverage as MC
    paths = set()
    for line in open(PROCS):
        for p in json.loads(line)["settings_paths"]:
            paths.add(tuple(p))
    failing = []
    for t in sorted(paths):
        if MC.scope_key(t) in MC.OUT_OF_SCOPE:
            continue
        depth, total, fail = MC.walk(list(t))
        if depth != total:
            failing.append((" > ".join(t), fail))
    assert not failing, (
        f"{len(failing)} manual paths no longer walk: " +
        "; ".join(f"{p} [breaks at {f!r}]" for p, f in failing[:8]))
