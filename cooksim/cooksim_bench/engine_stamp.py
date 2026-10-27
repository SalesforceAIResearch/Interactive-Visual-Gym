#!/usr/bin/env python3
"""Which engine did this episode run under?

    python -m cooksim_bench.engine_stamp          # print the current revision

An afternoon of fixes landed WHILE batches were running, so the published pages ended up mixing
episodes from six different engines — iter1 predated the mis_prep/ignite fix, iter1-3 predated
the case-config fix, only iter5 had the stall guard and the raised trigger deadline, and NOT ONE
episode had the agent+++ readout fix. Nothing recorded that, so the only way to work it out was
to compare file mtimes against batch launch times by hand.

Every report now carries `engine_rev`, and the pages show it, so an episode that predates the
current engine says so on its own card instead of looking like a current result.

The hash covers the files that change what an episode DOES: the engine core, the interaction
rules, the harness, the observation readout, and the case definitions. It deliberately ignores
page builders and analysis tools — re-rendering a page does not invalidate an episode.
"""
from __future__ import annotations
import hashlib, json, os, sys

import os
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
TRACKED = [
    # Terrain/source/char tables. Untracked until the 2026-08-14 distractor-source
    # addition, which changes what a map can contain — that is episode-defining.
    "cooksim/core/enums.py",
    "cooksim/core/layout.py",
    "cooksim/core/plan_ir.py",
    "cooksim/core/error_lib.py",
    "cooksim/core/generator.py",
    "cooksim/core/stations.py",
    # GameConfig — cook/burn/smoke timings, capacities, horizon, penalties. stations.py was
    # tracked and the config that FEEDS it was not, so the 2026-08-14 smoke_grace change
    # (10 -> 25: whether a clean plan smokes at all) would have shipped under an unchanged
    # revision. A number that decides what an episode does belongs in the stamp.
    "cooksim/core/config.py",
    "cooksim/interact/system.py",
    "cooksim/interact/protocol.py",
    # the affordance oracle. Since ECR-13 (2026-08-13) it delegates to highlevel._validate
    # and is the single source of every refusal sentence the cook and the supervisor see —
    # a change here changes what an episode says and does, so it belongs in the stamp.
    "cooksim/interact/checks.py",
    # persona v2: behavior-defining and pending registration since it landed (see its
    # module docstring); registered in the ECR-13 window as agreed.
    "cooksim/interact/persona_v2.py",
    "cooksim/agents/highlevel.py",
    # The cook's own observation under v2. The 2026-08-15 burrito sweep caught the
    # asymmetry that earns these two entries: a pure code MOVE out of python -m cooksim_bench.agent_context
    # moved the revision, while a real change to what the cook and the observer SEE did not,
    # because neither file was listed. The state map is the artifact the v2 grounding
    # contract rests on; place phrases are how the observer is allowed to say where.
    "cooksim/agents/observe_text.py",
    "cooksim/agents/places.py",
    "cooksim_bench/e2e_stepin_rollout.py",
    # Rollout.step() lives here — the pause/hesitation/skill machinery that decides what the
    # cook physically does each tick. It was NOT tracked, which meant a change to the single
    # most behaviour-defining loop in the harness would not have bumped the revision.
    "cooksim_bench/exp_error_rollout.py",
    "cooksim_bench/exp_multi_error.py",
    "cooksim_bench/agent_context.py",
    "cooksim_bench/exp_interact.py",
]


def parts(root: str = ROOT) -> dict:
    out = {}
    for rel in TRACKED:
        p = os.path.join(root, rel)
        try:
            out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
        except OSError:
            out[rel] = None
    return out


def stamp(root: str = ROOT) -> dict:
    p = parts(root)
    blob = json.dumps(p, sort_keys=True).encode()
    return {"rev": hashlib.sha256(blob).hexdigest()[:10], "parts": p}


def rev(root: str = ROOT) -> str:
    return stamp(root)["rev"]


def differs(a: dict, b: dict):
    """Which tracked files differ between two stamps — so a stale episode can say WHY."""
    pa, pb = (a or {}).get("parts") or {}, (b or {}).get("parts") or {}
    return sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))


if __name__ == "__main__":
    s = stamp()
    print(s["rev"])
    if "-v" in sys.argv:
        for k, v in sorted(s["parts"].items()):
            print(f"  {v}  {k}")
