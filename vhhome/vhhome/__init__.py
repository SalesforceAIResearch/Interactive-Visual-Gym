"""VH-Home: the house engine.

    world          the live VirtualHome world: one command per tick, symbolic state, observations, films
    actions        the egocentric, visibility-gated action space the person moves in
    gates          the geometric perception gates behind that action space
    topdown_map    the top-down map renderer (the assistant's second picture)
    render         VirtualHome rendering corrections shared by every renderer
    tasks, goals   the household chores: scene, initial state, steps; instruction and checkpoints
    plan           the plan representation (goal -> subgoal -> step) and its symbolic simulator
    mistakes       the mistake classes as plan transforms
    paths, keys    where things are; API keys

The benchmark harness that drives this engine (the simulated person, the assistant protocol, the cases, the
scoring) is the sibling package `vhhome_bench`.
"""
import os as _os
import sys as _sys

from . import paths as _paths

# VirtualHome's python client is a plain directory, not a package: make `unity_simulator` importable.
_client = _paths.virtualhome_client()
if _os.path.isdir(_client) and _client not in _sys.path:
    _sys.path.insert(0, _client)

__version__ = "1.0.0"
