#!/usr/bin/env python3
"""Perception GATES for the mid-level action space. Pure geometry — nothing here moves the body.

Two layers, per docs/MIDLEVEL_MIGRATION.md:

  A (navigation) is the AGENT's job, made with the top-down room map it is shown. The engine
    supplies movement primitives (`turn`, `go straight`, `go through the visible door`) and this
    module supplies the REFUSALS that force the agent to use them: an agent-typed command
    against something not currently in view is turned down with a reason that says what to do
    about it.
  B (object perception) is the point of the benchmark: manipulation needs the target VISIBLE
    and within NEAR_M, and a bare reference binds to the nearest instance IN VIEW — so where you
    stand decides which milk you get.

Why this file must not execute anything
---------------------------------------
The first version planned by DOING: its scan/approach helpers called ctrl.execute() while
translating, returned the primitive strings, the caller executed them again, and exec_step then
rendered the collected script a third time. Three traversals per command, a body that ended up
nowhere near where the log said, and 198 s spent failing to find a fridge. A gate can be asked
"would this work from here?" without anything happening — that is the property the whole design
leans on, so it is enforced here by construction: no method of this module calls execute().

The scripted plan does NOT come through these gates. A scripted command is a person following
their own routine in their own house; it runs on engine pathfinding exactly as the proven
high-level pipeline always has. Only agent-typed commands (kind in AGENT_KINDS) are gated.
"""
import sys

from vhhome.actions import NEAR_M, canon  # noqa: E402


def _base(tok):
    return None if tok is None else str(tok).split("#")[0].split("@")[0]


def nearest_visible(ctrl, cls):
    """(distance_m, instance_id) of the nearest IN-VIEW instance of `cls`, else (None, None).

    any_pitch=True so a low object on a table still counts when a downward glance would show it
    — the gate should test "could you see it from this spot", not "are you looking at it this
    exact frame". Snapping the view is midlevel's documented behaviour and is what its own
    manipulation actions use.
    """
    try:
        nv = ctrl._nearest_visible(canon(_base(cls)), any_pitch=True)
    except Exception:
        return None, None
    if not nv:
        return None, None
    d, oid = float(nv[0]), int(nv[1])
    return d, oid


def _is_anchored(ctrl, oid):
    """Does this instance rest ON/IN something (i.e. have a walkable footprint)?

    A wall- or ceiling-mounted lamp is visible, often nearest, and UNREACHABLE: [Walk] to it
    fails with "my body couldn't do that here". Engine fix #12 taught the graph resolver to
    prefer anchored instances; the gate's sight binding bypassed that resolver and re-imported
    the same fault (walk to the tablelamp -> bound #376, refused twice). True on any failure —
    an anchor check must never turn a good binding into a refusal.
    """
    try:
        g = ctrl._graph()
        # A GRABBABLE THING IS NEVER WALL-MOUNTED (A33, f89). VirtualHome drops the ON edge of an
        # object put down by a script: the mug the setup placed on the nightstand is only "INSIDE
        # bedroom". Treated as a wall lamp, its binding was deferred to the stale room index and the
        # walk went to a copy in another room. Something you can pick up rests where it lies.
        n = next((m for m in g["nodes"] if m["id"] == oid), None)
        if n is not None and "GRABBABLE" in (n.get("properties") or []):
            return True
        rooms = {m["id"] for m in g["nodes"] if m.get("category") == "Rooms"}
        return any(e["from_id"] == oid and e["relation_type"] in ("ON", "INSIDE")
                   and e["to_id"] not in rooms for e in g["edges"])
    except Exception:
        return True


def gate_walk(ctrl, cls):
    """Agent-typed `walk to X`: X must be in view from where the body stands.

    Returns (ok, why, oid). The refusal names the two ways forward — turn, or change rooms —
    because under this gate those are genuinely the only ways forward, and the agent that reads
    the refusal is the one holding the map.

    If the sighted nearest is UNANCHORED (wall/ceiling-mounted), the gate still passes — the
    class is genuinely in view — but binding is deferred (oid=None) so the reachability-aware
    resolver picks the instance to walk to.
    """
    d, oid = nearest_visible(ctrl, cls)
    if oid is None:
        return False, (f"I cannot see any {_base(cls)} from here — I need to turn to look "
                       f"around, or go through a door toward it (check the map)"), None
    if not _is_anchored(ctrl, oid):
        return True, None, None
    return True, None, oid


def gate_manip(ctrl, cls):
    """Agent-typed manipulation: target visible AND within NEAR_M ({:.1f} m).

    Returns (ok, why, oid). oid is the instance the gate SAW — the caller binds the command to
    it, so what gets manipulated is exactly what passed the gate. Binding to the graph-nearest
    instead could pick an occluded instance behind a wall that happens to be closer.
    """.format(NEAR_M)
    d, oid = nearest_visible(ctrl, cls)
    if oid is None:
        return False, f"I cannot see any {_base(cls)} from where I am standing", None
    if d > NEAR_M:
        return False, (f"the {_base(cls)} is {d:.1f} m away — too far to reach; "
                       f"I need to go to it first"), None
    return True, None, oid


# verbs -> which object of the step the gate applies to (1 = step[1], 3 = step[3])
# Drink is deliberately ABSENT: its object is IN HAND (resolution binds it from the hands),
# and a sight test against the world would find the decoy on the counter — or fail outright —
# while the human is holding the very thing it wants to drink.
GATED_VERBS = {"Walk": ("walk", 1), "Open": ("manip", 1), "Close": ("manip", 1),
               "SwitchOn": ("manip", 1), "SwitchOff": ("manip", 1), "Grab": ("manip", 1),
               "Sit": ("manip", 1),
               "Put": ("manip", 3), "PutIn": ("manip", 3)}


def gate_step(ctrl, step):
    """Gate one agent-typed step. Returns (ok, why, slot, oid).

    slot/oid tell the caller which element of the step tuple the sighted instance binds to.
    Put/PutIn gate the DESTINATION (the held object is resolved from the hands, not by sight —
    you can feel what you are carrying without looking at it).
    """
    kind = GATED_VERBS.get(step[0])
    if kind is None:
        return True, None, None, None          # Idle / StandUp / MidLevel movement: no target
    which, slot = kind
    cls = step[slot] if len(step) > slot else None
    if cls is None:
        return True, None, None, None
    fn = gate_walk if which == "walk" else gate_manip
    ok, why, oid = fn(ctrl, cls)
    return ok, why, (slot if ok else None), oid
