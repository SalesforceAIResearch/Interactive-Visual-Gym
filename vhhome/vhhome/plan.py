"""VirtualHome Plan IR — goal -> subgoal -> step hierarchy over the 18 benchmark episodes.

Mirrors cooksim/core/plan_ir.py, adapted to VirtualHome. The episode GT plans in episodes.py follow a
recoverable grammar (Walk-anchored action groups), so we can segment any plan into TYPED subgoal nodes
without new authoring:

    fetch(X [, from C])   Walk X, Grab X            | Walk C, Open C, Grab X, Close C
    deposit(X, T)         Walk T, Put X T           | Walk C, Open C, PutIn X C, Close C
    switch_on/off(D)      [Walk D,] SwitchOn/Off D
    shut(C) / open(C)     [Walk C,] Close C / Open C           (state-tidy nodes)
    posture(S)            Walk S, Sit S [, StandUp]
    inspect(O)            Walk O, TurnTo O [, LookAt O]        (search episodes)
    drink(X)              Drink X
    waypoint(O)           Walk O with no action on O           (route hint)

Segmentation is LOSSLESS: compiling the nodes back yields the original step list byte-for-byte
(test T1 over all 18 episodes). Deviations are then typed transforms over subgoal ids — never raw
indices — and compose IN ORDER under the priority rule: an earlier transform owns the plan; a later
one whose anchor disappeared is skipped-with-record.

Also provides simulate(): a SYMBOLIC executor (holding / open / on / object-location state with
precondition checks) that replays a compiled plan into resolved events, grades them with the real
episode_goals checkpoints, and reports outcome {success | silent_fail | stalled} + hazard flags
(stove left ON, doors left OPEN, microwave running empty, ...) + step cost — the cheap counterpart
of a Unity rollout, used for the whole test battery and for the site's outcome tables.

Steps are episodes.py tuples: (Action, cls, room[, cls2, room2]) or (Action,). Special injected step
("Idle", n) = stand still for n render frames (the renderer freezes; the simulator counts time).
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from vhhome import tasks as episodes
from vhhome import goals as episode_goals


# --------------------------------------------------------------------------- nodes
@dataclass
class Subgoal:
    sid: str                    # unique id, e.g. "fetch(milk)", "switch_on(tv)#2"
    type: str                   # fetch | deposit | switch_on | switch_off | shut | open | posture |
                                # stand | inspect | drink | waypoint | stall | injected types
    component: Optional[str]    # the object the node is about (milk, tv, ...)
    commands: List[tuple]       # the step tuples this node owns
    target: Optional[str] = None    # container / surface / seat involved
    origin: str = "plan"        # plan | injected | repeat
    deviated: bool = False


def _uniq(sid, taken):
    if sid not in taken:
        return sid
    k = 2
    while f"{sid}#{k}" in taken:
        k += 1
    return f"{sid}#{k}"


def segment_steps(steps) -> List[Subgoal]:
    """Recover the subgoal hierarchy from a flat GT step list (the grammar above)."""
    nodes, i, pend = [], 0, None            # pend = a Walk waiting to be absorbed by its action

    def add(type_, comp, cmds, target=None):
        sid = _uniq(f"{type_}({comp})" if comp else type_, {n.sid for n in nodes})
        nodes.append(Subgoal(sid, type_, comp, list(cmds), target=target))

    while i < len(steps):
        s, a = steps[i], steps[i][0]
        if a == "Walk":
            if pend is not None:                       # two Walks in a row -> the first was a waypoint
                add("waypoint", pend[1], [pend])
            if i + 1 >= len(steps):
                add("waypoint", s[1], [s]); pend = None; i += 1; continue
            pend = s; i += 1; continue
        cmds = [pend] if pend else []
        pend = None
        if a == "Open":
            C = s[1]
            j = i + 1
            while j < len(steps) and steps[j][0] != "Walk" and not (steps[j][0] == "Close" and steps[j][1] == C):
                j += 1
            if j < len(steps) and steps[j][0] == "Close" and steps[j][1] == C:   # container window
                inner = steps[i + 1:j]
                grabs = [x for x in inner if x[0] == "Grab"]
                puts = [x for x in inner if x[0] in ("PutIn", "Put")]
                cmds += [s] + list(inner) + [steps[j]]
                if grabs and not puts:
                    add("fetch", grabs[0][1], cmds, target=C)
                elif puts:
                    add("deposit", puts[0][1], cmds, target=C)
                else:
                    add("check", C, cmds, target=C)     # open + close, nothing inside: a look-in
                i = j + 1; continue
            add("open", C, cmds + [s], target=C); i += 1; continue
        if a == "Close":
            add("shut", s[1], cmds + [s], target=s[1]); i += 1; continue
        if a in ("SwitchOn", "SwitchOff"):
            add("switch_on" if a == "SwitchOn" else "switch_off", s[1], cmds + [s], target=s[1]); i += 1; continue
        if a == "Grab":
            add("fetch", s[1], cmds + [s]); i += 1; continue
        if a in ("Put", "PutIn"):
            add("deposit", s[1], cmds + [s], target=s[3]); i += 1; continue
        if a == "Sit":
            cc = cmds + [s]
            if i + 1 < len(steps) and steps[i + 1][0] == "StandUp":
                i += 1; cc.append(steps[i])
            add("posture", s[1], cc, target=s[1]); i += 1; continue
        if a == "StandUp":
            add("stand", None, cmds + [s]); i += 1; continue
        if a in ("Read", "Watch"):
            # Two verbs this Unity build accepts that the grammar never used. They are ATTENDING
            # actions: the person settles on something and engages with it, so like `inspect`
            # they consume their Walk and take no object away. Probed against the live build
            # (tools/tmp/probe_actions.py): Read, Watch, Touch, PointAt and Run are accepted;
            # Wash, Rinse, Pour, Cut, Eat, Type, Sleep, Lie, Push and Pull are NOT in v2.3.0.
            add("read" if a == "Read" else "watch", s[1], cmds + [s], target=s[1])
            i += 1; continue
        if a in ("TurnTo", "LookAt"):
            O = s[1]; cc = cmds + [s]
            while i + 1 < len(steps) and steps[i + 1][0] in ("TurnTo", "LookAt") and steps[i + 1][1] == O:
                i += 1; cc.append(steps[i])
            add("inspect", O, cc); i += 1; continue
        if a == "Drink":
            add("drink", s[1], cmds + [s]); i += 1; continue
        add("other", s[1] if len(s) > 1 else None, cmds + [s]); i += 1
    if pend is not None:
        add("waypoint", pend[1], [pend])
    return nodes


def segment_plan(ep_key) -> List[Subgoal]:
    return segment_steps(episodes.EPISODES[ep_key]["steps"])


def compile_ir(nodes: List[Subgoal]) -> List[Tuple[tuple, str, bool]]:
    """Flatten to executor input: [(step_tuple, subgoal_id, deviated)]."""
    out = []
    for n in nodes:
        for c in n.commands:
            out.append((tuple(c), n.sid, n.deviated or n.origin != "plan"))
    return out


def flat_steps(nodes) -> List[tuple]:
    return [c for c, _, _ in compile_ir(nodes)]


# --------------------------------------------------------------------------- injected node factories
def wrong_grab_node(obj, room, container=None, croom=None):
    """Grabs a look-alike instead (MISIDENTIFY).

    When the fetch it replaces opened a CONTAINER, the wrong grab keeps that container window:
    she opens the fridge and takes the wrong carton out of it. Dropping the open/close — which
    the bare [Walk, Grab] form did — deleted the container interaction from the episode, and a
    watcher reads a missing "open the fridge" as a skipped step ON TOP of the wrong object, i.e.
    an extra unlabelled error (VLM audit of the composite suite, 2026-08-28). The look-alike must
    of course BE in that container; the episodes that use this stow it there in setup."""
    if container:
        return Subgoal(f"wrong_grab({obj})", "fetch", obj,
                       [("Walk", container, croom or room), ("Open", container, croom or room),
                        ("Grab", obj, room), ("Close", container, croom or room)],
                       target=container, origin="injected", deviated=True)
    return Subgoal(f"wrong_grab({obj})", "fetch", obj,
                   [("Walk", obj, room), ("Grab", obj, room)], origin="injected", deviated=True)

def wrong_search_node(container, room):        # opens the wrong container looking for the target
    return Subgoal(f"wrong_search({container})", "check", container,
                   [("Walk", container, room), ("Open", container, room), ("Close", container, room)],
                   target=container, origin="injected", deviated=True)

def detour_node(obj, room, idle=8):            # pointless far detour (CONFUSE_LOCATION / wander)
    return Subgoal(f"detour({obj})", "waypoint", obj,
                   [("Walk", obj, room), ("Idle", idle)], origin="injected", deviated=True)

def search_looks_node(stops, idle=6):          # hunts N wrong spots (FAIL_TO_FIND_VISIBLE)
    """idle=0 emits NO pause at all. A detour that stands still at each stop is, on camera,
    indistinguishable from a `stall` — and stall is an observation-channel TWIN of
    navigation_inefficiency, so in a multi-error episode the pause manufactures exactly the
    attribution ambiguity the composite suite exists to avoid (VLM audit, 2026-08-28). Emitting
    ("Idle", 0) was not the same as emitting nothing: the renderer still froze a frame."""
    cmds = []
    for obj, room in stops:
        cmds += [("Walk", obj, room)] + ([("Idle", idle)] if idle else [])
    return Subgoal("search", "search", None, cmds, origin="injected", deviated=True)

def stall_node(frames=60):                     # stands lost
    return Subgoal("lost", "stall", None, [("Idle", frames)], origin="injected", deviated=True)

def device_on_node(dev, room):                 # lights a device (stove!) — safety injections
    return Subgoal(f"light({dev})", "switch_on", dev,          # trailing dwell: the switch-on itself has
                   [("Walk", dev, room), ("SwitchOn", dev, room), ("Idle", 10)],  # little visual change
                   target=dev, origin="injected", deviated=True)

def put_wrong_node(obj, room, surf, sroom):    # parks the held object on the wrong surface
    return Subgoal(f"put_wrong({obj})", "deposit", obj,
                   [("Walk", surf, sroom), ("Put", obj, room, surf, sroom)],
                   target=surf, origin="injected", deviated=True)

def sit_away_node(seat="sofa", room="livingroom"):    # walks off and sits down (leaving the task)
    return Subgoal(f"sit_away({seat})", "posture", seat,
                   [("Walk", seat, room), ("Sit", seat, room)], target=seat, origin="injected", deviated=True)

def repeat_of(node: Subgoal) -> Subgoal:
    dup = copy.deepcopy(node)
    dup.sid, dup.origin, dup.deviated = node.sid, "repeat", True
    return dup


def redo_toggle_node(node: Subgoal, immediate: bool = False) -> Subgoal:
    """REPETITION with real consequence: re-presses the switch/door of a COMPLETED toggle
    subgoal — the second press INVERTS the state (the tidied lamp is back ON, the closed door
    back open). Outcome-bearing: the plan's own end-state is undone."""
    inv = {"SwitchOff": "SwitchOn", "SwitchOn": "SwitchOff", "Close": "Open", "Open": "Close"}
    last = next(c for c in reversed(node.commands) if c[0] in inv)
    tgt, room = last[1], last[2]
    # A trailing dwell, exactly as device_on_node has: the inverting press IS the entire error,
    # and a switch flipping has almost no visual change, so without a beat afterwards the one
    # frame that carries the evidence can be missed (independent audit, 2026-08-28).
    # An `immediate` redo happens while she is still standing at the target, so a leading Walk
    # would be a zero-distance move: the renderer emits no frames for it and treats a zero-frame
    # item as fatal. A `delayed` redo genuinely has to walk back. (independent audit, 2026-08-28)
    approach = [] if immediate else [("Walk", tgt, room)]
    return Subgoal(f"redo({node.sid})", node.type, node.component or tgt,
                   approach + [(inv[last[0]], tgt, room), ("Idle", 10)],
                   target=node.target, origin="injected", deviated=True)


def reload_node(node: Subgoal) -> Subgoal:
    """REPETITION, visible-benign flavor: unloads and re-loads a COMPLETED deposit — the
    object visibly leaves and re-enters its destination (a state change happens twice); the
    outcome survives but the double handling is unmistakable on camera."""
    put = next(c for c in node.commands if c[0] in ("Put", "PutIn"))
    obj, oroom, tgt, troom = put[1], put[2], put[3], put[4]
    if put[0] == "PutIn" and tgt.split("@")[0] in CONTAINERS:
        cmds = [("Walk", tgt, troom), ("Open", tgt, troom), ("Grab", obj, oroom),
                ("PutIn", obj, oroom, tgt, troom), ("Close", tgt, troom)]
    else:
        cmds = [("Walk", tgt, troom), ("Grab", obj, oroom), tuple(put)]
    return Subgoal(f"redo({node.sid})", "deposit", obj, cmds, target=tgt,
                   origin="injected", deviated=True)


# --------------------------------------------------------------------------- ground-truth record
@dataclass
class Deviation:
    transform: str
    event_subtype: str
    anchor: str
    detail: dict = field(default_factory=dict)
    language_hook: Optional[str] = None
    skipped: bool = False


def _idx(nodes, sid):
    for i, n in enumerate(nodes):
        if n.sid == sid:
            return i
    raise KeyError(f"no subgoal '{sid}' in plan IR ({[n.sid for n in nodes]})")


# --------------------------------------------------------------------------- transforms (pure)
def skip(nodes, sid):
    i = _idx(nodes, sid)
    out = nodes[:i] + nodes[i + 1:]
    return out, Deviation("skip", "OMISSION", sid, {"dropped_commands": nodes[i].commands})

def insert(nodes, node, before=None, after=None):
    assert (before is None) != (after is None), "give exactly one of before=/after="
    anchor = before or after
    i = _idx(nodes, anchor) + (0 if before else 1)
    node = copy.deepcopy(node)
    node.sid = _uniq(node.sid, {n.sid for n in nodes})
    out = nodes[:i] + [node] + nodes[i:]
    return out, Deviation("insert", "WRONG_ACTION", node.sid,
                          {"position": ("before " if before else "after ") + anchor})

def swap(nodes, sid_a, sid_b):
    ia, ib = _idx(nodes, sid_a), _idx(nodes, sid_b)
    out = list(nodes)
    out[ia], out[ib] = copy.deepcopy(out[ib]), copy.deepcopy(out[ia])
    out[ia].deviated = out[ib].deviated = True
    return out, Deviation("swap", "WRONG_ACTION", f"{sid_a}<->{sid_b}", {})

def repeat(nodes, sid, at):
    """Re-do the EARLIER subgoal `sid` after finishing `at` (strictly non-consecutive)."""
    i, j = _idx(nodes, sid), _idx(nodes, at)
    if i >= j:
        raise KeyError(f"repeat: '{sid}' must be strictly earlier than trigger '{at}'")
    dup = repeat_of(nodes[i])
    dup.sid = _uniq(dup.sid, {n.sid for n in nodes})
    out = nodes[:j + 1] + [dup] + nodes[j + 1:]
    return out, Deviation("repeat", "REPEAT_COMPLETED_STEP", dup.sid, {"of": sid, "after": at})

def replace(nodes, sid, node):
    i = _idx(nodes, sid)
    node = copy.deepcopy(node)
    node.sid = _uniq(node.sid, {n.sid for n in nodes})
    out = nodes[:i] + [node] + nodes[i + 1:]
    return out, Deviation("replace", "WRONG_ACTION", node.sid, {"replaced": sid})

def truncate_done(nodes, after, leave=True):
    """Believes the task is finished after `after`: drops the tail and (optionally) walks off
    to the sofa — the VH analogue of premature_serve."""
    i = _idx(nodes, after)
    dropped = [n.sid for n in nodes[i + 1:]]
    tail = [sit_away_node()] if leave else []
    out = nodes[:i + 1] + tail
    return out, Deviation("truncate_done", "SKIP_PREREQUISITE", after, {"dropped": dropped})

def deps(nodes):
    """The REAL dependency DAG of a plan: edges (before_sid, after_sid), derived from data
    dependencies — never from list position.

    The IR is a linear list, and everything downstream quietly treated that ORDER as required:
    the supervisor's schema example taught it to write constraints like "fetch(milk) before
    deposit(milk)", so on tasks whose subgoals are INDEPENDENT (tidy: lamp / fridge / tv) it
    hypothesised orderings that do not exist and nagged the human for tidying in a different
    order. These are the only orderings the tasks actually require:

      fetch(x)  -> any later subgoal that USES x (deposit/drink of the same component)
      deposit(x -> C) -> switch_on(C)   (the device must be loaded before it runs)
      sit(S)    -> the stand that follows it

    Everything else is legal in any order — and simulate() agrees: violating a listed edge
    fails a precondition ("I'm not holding..."), while permuting unlisted subgoals succeeds.
    """
    edges = []
    base_ = lambda t: None if t is None else str(t).split("#")[0].split("@")[0]
    for i, a_ in enumerate(nodes):
        for b_ in nodes[i + 1:]:
            if a_.type == "fetch" and b_.type in ("deposit", "drink") \
                    and base_(a_.component) == base_(b_.component):
                edges.append((a_.sid, b_.sid))
            elif a_.type == "deposit" and b_.type == "switch_on" \
                    and a_.target is not None and base_(a_.target) == base_(b_.component):
                edges.append((a_.sid, b_.sid))
            elif a_.type == "posture" and b_.type == "stand" \
                    and nodes.index(b_) == nodes.index(a_) + 1:
                edges.append((a_.sid, b_.sid))
    return edges


def constraints_text(nodes):
    """deps() rendered for a prompt. Explicit about what an EMPTY list means, because the
    absent-constraint case is exactly where the assistant used to invent orderings."""
    es = deps([n for n in nodes if getattr(n, "origin", "plan") == "plan"])
    if not es:
        return ("(none — every remaining subgoal may be done in ANY order; a different order "
                "than you expected is NOT an error)")
    return "; ".join(f"{a_} before {b_}" for a_, b_ in es) + \
           " — nothing else is ordered; any other sequence is legal"


def permute(nodes, order):
    """LEGALLY reorder independent task blocks. `order` = list of sids in the new order; they must
    be a permutation of a contiguous-free subset (caller guarantees dependency safety — verified
    by simulate() in tests)."""
    idxs = [_idx(nodes, s) for s in order]
    if sorted(idxs) != list(range(min(idxs), min(idxs) + len(idxs))):
        raise KeyError(f"permute: sids {order} are not a contiguous block")
    lo = min(idxs)
    blk = [copy.deepcopy(nodes[_idx(nodes, s)]) for s in order]
    for n in blk:
        n.deviated = True
    out = nodes[:lo] + blk + nodes[lo + len(blk):]
    return out, Deviation("permute", "REORDER_SUBGOALS", "+".join(order), {"order": order})

def forget_rest(nodes, after, stall_frames=60):
    i = _idx(nodes, after)
    dropped = [n.sid for n in nodes[i + 1:]]
    nxt = nodes[i + 1] if i + 1 < len(nodes) else None
    out = nodes[:i + 1] + [stall_node(stall_frames)]
    return out, Deviation("forget_rest", "FORGET_NEXT_STEP", after,
                          {"dropped": dropped, "restorable_tail": dropped,
                           "next_step": {"subgoal": nxt.sid, "command": nxt.commands[0]} if nxt else None})


OPS = {"skip": skip, "insert": insert, "swap": swap, "repeat": repeat, "replace": replace,
       "truncate_done": truncate_done, "permute": permute, "forget_rest": forget_rest}


def apply_pipeline(nodes, specs):
    """Apply transform specs IN ORDER (priority rule: earlier owns the plan; later spec with a
    vanished anchor is skipped-with-record)."""
    records = []
    for spec in specs:
        spec = dict(spec); op = spec.pop("op")
        try:
            nodes, rec = OPS[op](nodes, **spec)
        except KeyError as e:
            records.append(Deviation(op, "(not generated)", str(spec), {"reason": str(e)}, skipped=True))
            continue
        records.append(rec)
    return nodes, records


# --------------------------------------------------------------------------- symbolic executor
CONTAINERS = {"fridge", "microwave", "dishwasher", "cabinet", "kitchencabinet", "stove", "bathroomcabinet",
              "bookshelf", "closet", "desk", "washingmachine", "toilet", "garbagecan"}
DEVICES = {"tv", "tablelamp", "stove", "microwave", "dishwasher", "computer", "lightswitch", "washingmachine",
           "toaster", "coffeemaker", "radio", "fan"}
V2G = {"Walk": "go_to_nearest", "Grab": "pick_up_nearest", "Open": "open_nearest", "Close": "close_nearest",
       "Put": "put_on_nearest", "PutIn": "put_in_nearest", "SwitchOn": "switch_on_nearest",
       "SwitchOff": "switch_off_nearest", "Sit": "sit_nearest", "StandUp": "stand_up",
       "Drink": "drink_nearest", "TurnTo": "turn_to_nearest", "LookAt": "look_at_nearest",
       "TurnLeft": "turn_left", "TurnRight": "turn_right",
       "Read": "read_nearest", "Watch": "watch_nearest"}


def drink_node(obj, room="kitchen"):
    return Subgoal(f"drink({obj})", "drink", obj, [("Drink", obj, room)], origin="injected", deviated=True)


def simulate(compiled, ep_key, setup_extra=None, checkpoints=None):
    """Symbolically execute a compiled plan for episode `ep_key`. Returns a dict:
       outcome: success | silent_fail | stalled     (stalled = a stall/'lost' node ends the plan)
       checkpoints, refused [(k, step, why)], hazards {stove_on, left_open[], running_empty[]},
       events (resolved 5-tuples), steps (count incl. idle frames/8), holding/open/on end-state.
    Initial object placement follows the episode's setup steps (e.g. milk stowed in the fridge)."""
    ep, gdef = episodes.EPISODES[ep_key], episode_goals.EPISODES[ep_key]
    holding, opened, on, loc = set(), set(), set(), {}
    events, refused = [], []
    time = 0

    def run(step, k=None, live=True):
        nonlocal time
        a = step[0]
        if a == "Idle":
            time += step[1] / 8.0; return
        time += 3 if a == "Walk" else 1
        o = step[1] if len(step) > 1 else None
        o2 = step[3] if len(step) > 3 else None
        def ok():
            if live:      # "@i" instance selectors grade as their base class (kitchentable@2 -> kitchentable)
                events.append((V2G[a], o.split("@")[0] if o else o, 0, o2.split("@")[0] if o2 else o2, 0))
        def no(w):
            if live:
                refused.append((k, step, w))
        if a == "Grab":
            c = loc.get(o)
            if c in CONTAINERS and c not in opened: no(f"{o} is inside the CLOSED {c}"); return
            holding.add(o); loc.pop(o, None); ok(); return
        if a == "Open":
            if o in opened: no(f"{o} already open"); return
            opened.add(o); ok(); return
        if a == "Close":
            if o not in opened: no(f"{o} already closed"); return
            opened.discard(o); ok(); return
        if a == "Put":
            if o not in holding: no(f"not holding {o}"); return
            holding.discard(o); loc[o] = o2; ok(); return
        if a == "PutIn":
            if o not in holding: no(f"not holding {o}"); return
            if o2 in CONTAINERS and o2 not in opened: no(f"{o2} is closed"); return
            holding.discard(o); loc[o] = o2; ok(); return
        if a == "SwitchOn":
            if o in on: no(f"{o} already on"); return
            on.add(o); ok(); return
        if a == "SwitchOff":
            if o not in on: no(f"{o} already off"); return
            on.discard(o); ok(); return
        if a == "Drink":
            if o not in holding: no(f"not holding {o}"); return
            ok(); return
        ok()

    for s in list(ep.get("setup", [])) + list(setup_extra or []):   # stage the world (always succeeds)
        run(tuple(s), live=False)
    opened_at_start = set(opened)
    boundaries = {}                       # sid -> world snapshot right AFTER that subgoal finishes
    for k, (step, sid, dev) in enumerate(compiled):
        run(step, k)
        if k + 1 >= len(compiled) or compiled[k + 1][1] != sid:
            boundaries[sid] = dict(holding=sorted(holding), open=sorted(opened), on=sorted(on),
                                   loc=dict(loc), n_events=len(events), time=round(time, 1))

    latched = (checkpoints(events) if checkpoints else episode_goals.evaluate(gdef, events))   # goal mutation: ANNOUNCE_PLAN_CHANGE swaps these
    success = all(latched.values())
    stalled = bool(compiled) and compiled[-1][0][0] == "Idle" and not success
    inside = set(loc.values())
    # hazards are judged RELATIVE to the intended end-state: what the episode's own GT plan leaves
    # behind is not a hazard (power_hard deliberately ends with the stove ON).
    base_end = _BASE_END.get(ep_key) if _BASE_END.get(ep_key) is not None else {"open": set(), "on": set()}
    hazards = {}
    if "stove" in on and "stove" not in base_end["on"]: hazards["stove_on"] = True
    left = sorted(opened - base_end["open"])
    if left: hazards["left_open"] = left
    empty = [d for d in ("microwave", "dishwasher") if d in on and d not in inside]
    if empty: hazards["running_empty"] = empty
    return dict(outcome="success" if success else ("stalled" if stalled else "silent_fail"),
                checkpoints=latched, refused=refused, hazards=hazards, events=events,
                steps=round(time, 1), holding=sorted(holding), open=sorted(opened), on=sorted(on),
                boundaries=boundaries)


_BASE_END = {}
def _prime_base_ends():
    """Cache each episode's intended end-state (baseline plan's open/on sets) for hazard judging."""
    for k in episodes.EPISODES:
        _BASE_END[k] = {"open": set(), "on": set()}          # sentinel breaks recursion
        s = simulate(compile_ir(segment_plan(k)), k)
        _BASE_END[k] = {"open": set(s["open"]), "on": set(s["on"])}
_prime_base_ends()


def base_cost(ep_key):
    return simulate(compile_ir(segment_plan(ep_key)), ep_key)["steps"]
