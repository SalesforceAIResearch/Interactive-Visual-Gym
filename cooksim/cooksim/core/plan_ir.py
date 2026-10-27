"""Plan IR — the materialized hierarchy  GOAL (Recipe) -> SUBGOALS -> HIGH-LEVEL COMMANDS.

`ground_truth_plan()` in generator.py expands a Recipe into a flat command list by a fixed
template; the hierarchy it implies is thrown away. This module keeps it: `segment_plan(recipe)`
produces a list of typed Subgoal nodes (each carrying its command block), TRANSFORMS edit the
node list (never raw indices), and `compile_ir()` flattens back to the tagged command list the
executor consumes, alongside a ground-truth record of every deviation applied.

    ir = segment_plan(recipe)                                  # [prep(meat), fetch_plate, assemble(bun), ...]
    ir, r1 = insert(ir, foreign_scoop("lettuce"), before="assemble(cheese)")
    ir, r2 = swap(ir, "assemble(bun)", "assemble(meat)")
    cmds, records = compile_ir(ir), [r1, r2]                   # multi-mistake episode, fully ground-truthed

Transforms are pure (they return a new list) so they compose; every command in the compiled
output is tagged with the subgoal id it came from, so per-event tick windows can be recovered
from the execution log.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .enums import Terrain
from .generator import _COOK_STATION, _STATION_LABEL, chain_chop, cook_groups, cook_station_for
from .recipes import COOK, CHOP


# --------------------------------------------------------------------------- nodes
@dataclass
class Subgoal:
    sid: str                      # unique id, e.g. "prep(meat)", "assemble(bun)", "foreign_scoop(lettuce)#1"
    type: str                     # prep | fetch_plate | assemble | serve | foreign_scoop | raw_scoop |
                                  # wander | stall | premature_serve
    component: Optional[str]      # ingredient name, or None
    commands: List[str]           # the high-level command block this node expands to
    origin: str = "plan"          # "plan" (from the recipe template) | "injected" (added by a transform)
    deviated: bool = False        # touched by any transform (skip leaves no node; others mark this)


def _uniq(sid: str, taken) -> str:
    if sid not in taken:
        return sid
    i = 2
    while f"{sid}#{i}" in taken:
        i += 1
    return f"{sid}#{i}"


# --------------------------------------------------------------------------- segmentation
def _station_label(name: str) -> str:
    return _STATION_LABEL[_COOK_STATION.get(name, Terrain.POT)]


def segment_plan(recipe) -> List[Subgoal]:
    """The canonical plan as typed Subgoal nodes (the single template — generator.
    ground_truth_plan flattens this). Realistic-kitchen rules:

    * Cooked units are BATCHED per ingredient into ONE shared station
      (generator.cook_groups): 3 onions -> one pot, 2 chip portions -> one
      fryer basket; the batch is turned on once and plated with ONE scoop.
    * CHAIN ingredients (derived from the ACCEPTS matrix: the station takes
      them CHOPPED, not raw — steak strips, chips, soup veg) are chopped on the
      board first, per unit.
    * SINGLE heat group (e.g. the Cheeseburger patty): the classic block —
      load the station and turn it on immediately.
    * MULTIPLE heat groups: the LATE-IGNITE template — everything is only
      LOADED during prep, then one go-back-and-turn-on pass (in plating order)
      lights the stations right before assembly, so nothing burns while the
      solo cook preps the rest.
    """
    rid = getattr(recipe, "id", None)
    nodes: List[Subgoal] = []
    taken: set = set()

    def add(sid, typ, comp, cmds):
        sid = _uniq(sid, taken)
        taken.add(sid)
        nodes.append(Subgoal(sid, typ, comp, cmds))

    groups = cook_groups(recipe)                                # {(name, terrain): units}
    late_ignite = len(groups) >= 2
    board_users = sum(1 for n, s in recipe.contents if s == CHOP) \
        + sum(cnt for (n, _t), cnt in groups.items() if chain_chop(rid, n))
    # GROUNDING REDESIGN (2026-08-15): qualify the board UNCONDITIONALLY. The old
    # conditional keyed on how many boards the RECIPE uses, but ambiguity is a property of
    # the MAP: Caprese chops one thing, so it emitted a bare "go to the cutting board" —
    # fine while an unqualified reference silently resolved to the nearest instance, and a
    # refusal ("I need grounding — there is more than one cutting board here") the moment
    # the engine stopped guessing. A plan that says which board it means is correct on
    # every map and needs no map knowledge to write.
    from .config import is_v1 as _isv1
    board_goto = ("go to the empty cutting board"
                  if (not _isv1() or board_users > 1) else "go to the cutting board")

    loaded: dict = {}                                           # ingredient -> units already in its station
    for n, s in recipe.contents:                                # PREP/LOAD blocks (per unit)
        if s == COOK:
            st = _STATION_LABEL[cook_station_for(rid, n)]
            # first unit opens a fresh station; later units JOIN the batch
            st_goto = (f"go to the empty {st}" if not loaded.get(n)
                       else f"go to the {st} with the {n}")
            if chain_chop(rid, n):                              # chain: chop first, then the heat station
                cmds = [f"go to the {n} source", "pick up", board_goto, "put down",
                        "chop", "pick up", st_goto, "put down"]
            else:
                cmds = [f"go to the {n} source", "pick up", st_goto, "put down"]
            loaded[n] = loaded.get(n, 0) + 1
            if not late_ignite and loaded[n] == groups[(n, cook_station_for(rid, n))]:
                cmds.append("turn on")                          # single group: light it now
            add(f"prep({n})", "prep", n, cmds)
        elif s == CHOP:
            add(f"prep({n})", "prep", n,
                [f"go to the {n} source", "pick up", board_goto, "put down", "chop"])
    if late_ignite:                                             # IGNITE pass: one per group, plating order
        for (n, terr) in groups:
            st = _STATION_LABEL[terr]
            add(f"ignite({n})", "ignite", n, [f"go to the {st} with the {n}", "turn on"])
    nodes.append(Subgoal("fetch_plate", "fetch_plate", None, ["go to the plates", "pick up"]))
    plated: set = set()
    for n, s in recipe.contents:                                # ASSEMBLY blocks (order-FIXED)
        if s == COOK:
            if n in plated:
                continue                                        # whole batch landed with one scoop
            plated.add(n)
            where = f"go to the {_STATION_LABEL[cook_station_for(rid, n)]} with the {n}"
        elif s == CHOP:
            where = (f"go to the cutting board with the {n}"
                     if (not _isv1() or board_users > 1) else "go to the cutting board")
        else:
            where = f"go to the {n} source"
        add(f"assemble({n})", "assemble", n, [where, "scoop"])
    nodes.append(Subgoal("serve", "serve", None, ["go to the serving pass", "serve"]))
    return nodes


# --------------------------------------------------------------------------- node factories
def foreign_scoop(component: str) -> Subgoal:
    """Scoop an off-recipe ingredient onto the plate (wrong_ingredient)."""
    return Subgoal(f"foreign_scoop({component})", "foreign_scoop", component,
                   [f"go to the {component} source", "scoop"], origin="injected", deviated=True)


def raw_scoop(component: str) -> Subgoal:
    """Scoop a component RAW from its source, skipping its prep (raw_on_plate)."""
    return Subgoal(f"raw_scoop({component})", "raw_scoop", component,
                   [f"go to the {component} source", "scoop"], origin="injected", deviated=True)


def wander_node(think: int = 10) -> Subgoal:
    """Walk the wrong way to a far counter, think for `think` units, return (wrong_location).
    '__wander__ N' is resolved at runtime by the executor from the cook's live position."""
    return Subgoal("wander", "wander", None, [f"__wander__ {think}"], origin="injected", deviated=True)


def detour_node() -> Subgoal:
    """A pointless detour with no pause — inefficient but valid (route waste only)."""
    return Subgoal("detour", "detour", None, ["__wander__ 0"], origin="injected", deviated=True)


def busywork_node(stops: List[str], idle: int = 8) -> Subgoal:
    """Do OTHER things around the kitchen (visit real stations, linger at each) — the
    realistic 'busy elsewhere' filler used while something is left cooking."""
    cmds = []
    for stop in stops:
        cmds += [f"go to the {stop}", f"idle {idle}"]
    return Subgoal("busywork", "busywork", None, cmds, origin="injected", deviated=True)


def stall_node(ticks: int = 24) -> Subgoal:
    return Subgoal("stall", "stall", None, [f"idle {ticks}"], origin="injected", deviated=True)


def trash_node() -> Subgoal:
    """Discard whatever is held at the trash (requires a trash tile on the map)."""
    return Subgoal("trash_held", "trash_held", None, ["go to the trash", "trash"],
                   origin="injected", deviated=True)


def fetch_plate_node() -> Subgoal:
    return Subgoal("fetch_plate", "fetch_plate", None, ["go to the plates", "pick up"],
                   origin="injected", deviated=True)


def prep_of(component: str, station_label: Optional[str] = None) -> Subgoal:
    """A full prep block for `component` (used to prep a WRONG ingredient, or to re-prep
    after a repair). Station defaults to the component's canonical one; the finishing verb
    matches the station (chop at a board, turn on at a heat station)."""
    st = station_label or _station_label(component)
    finish = "chop" if "board" in st else "turn on"
    return Subgoal(f"prep({component})", "prep", component,
                   [f"go to the {component} source", "pick up", f"go to the {st}",
                    "put down", finish], origin="injected", deviated=True)


def park_plate_node(coord: str) -> Subgoal:
    """Put the held plate down on a specific counter (frees the hands mid-recovery)."""
    return Subgoal(f"park_plate({coord})", "park_plate", None,
                   [f"go to the counter ({coord})", "put down"], origin="injected", deviated=True)


def retrieve_plate_node(coord: str) -> Subgoal:
    """Pick the parked plate back up from that counter."""
    return Subgoal(f"retrieve_plate({coord})", "retrieve_plate", None,
                   [f"go to the counter ({coord})", "pick up"], origin="injected", deviated=True)


def repeat_of(node: Subgoal) -> Subgoal:
    dup = copy.deepcopy(node)
    dup.origin, dup.deviated = "injected", True
    return dup


# --------------------------------------------------------------------------- ground-truth record
@dataclass
class Deviation:
    transform: str                # skip | insert | swap | repeat | replace | truncate_and_serve | permute | forget_rest
    event_subtype: str            # the plan's behavior-event subtype this realizes
    anchor: str                   # subgoal id(s) it touched
    detail: dict = field(default_factory=dict)
    language_hook: Optional[str] = None   # the language-layer event this deviation can carry (realized later)
    skipped: bool = False         # True if a prior transform removed this one's anchor (priority rule)
    function: Optional[str] = None        # the behavior FUNCTION that generated this record (one of
                                          # the 11 catalog names) — the supervisor's label vocabulary


# --------------------------------------------------------------------------- transforms
def _idx(nodes: List[Subgoal], sid: str) -> int:
    for i, n in enumerate(nodes):
        if n.sid == sid:
            return i
    raise KeyError(f"no subgoal '{sid}' in plan IR ({[n.sid for n in nodes]})")


def skip(nodes: List[Subgoal], sid: str) -> Tuple[List[Subgoal], Deviation]:
    """Silently drop a subgoal (omission — e.g. skip(assemble(cheese)))."""
    i = _idx(nodes, sid)
    out = nodes[:i] + nodes[i + 1:]
    subtype = "SKIP_PREREQUISITE" if nodes[i].type == "prep" else "OMISSION"
    return out, Deviation("skip", subtype, sid, {"dropped_commands": nodes[i].commands})


def insert(nodes: List[Subgoal], node: Subgoal, before: Optional[str] = None,
           after: Optional[str] = None) -> Tuple[List[Subgoal], Deviation]:
    """Splice an injected node before/after an anchor subgoal."""
    assert (before is None) != (after is None), "give exactly one of before=/after="
    anchor = before or after
    i = _idx(nodes, anchor) + (0 if before else 1)
    node = copy.deepcopy(node)
    node.sid = _uniq(node.sid, {n.sid for n in nodes})
    out = nodes[:i] + [node] + nodes[i:]
    subtype = {"foreign_scoop": "MISIDENTIFY_OBJECT", "raw_scoop": "SKIP_PREREQUISITE",
               "wander": "CONFUSE_LOCATION_OR_DIRECTION", "stall": "HESITATE_OR_IDLE"}.get(node.type, "WRONG_ACTION")
    return out, Deviation("insert", subtype, node.sid, {"position": ("before " if before else "after ") + anchor})


def swap(nodes: List[Subgoal], sid_a: str, sid_b: str) -> Tuple[List[Subgoal], Deviation]:
    """Exchange two subgoals (wrong_order when applied to assembly nodes)."""
    ia, ib = _idx(nodes, sid_a), _idx(nodes, sid_b)
    out = list(nodes)
    out[ia], out[ib] = copy.deepcopy(out[ib]), copy.deepcopy(out[ia])
    out[ia].deviated = out[ib].deviated = True
    return out, Deviation("swap", "WRONG_ACTION", f"{sid_a}<->{sid_b}", {})


def repeat(nodes: List[Subgoal], sid: str, at: str) -> Tuple[List[Subgoal], Deviation]:
    """Re-do an EARLIER completed subgoal at a LATER point in the plan: after finishing `at`,
    the cook goes back and repeats `sid`. Enforced non-consecutive: `sid` must lie strictly
    BEFORE `at` in the plan (repeating the step you just finished is not a believable memory
    slip and is rejected)."""
    i, j = _idx(nodes, sid), _idx(nodes, at)
    if i >= j:
        raise KeyError(f"repeat: '{sid}' must be strictly earlier than the trigger point '{at}' "
                       f"(a consecutive/self repeat is not allowed)")
    dup = repeat_of(nodes[i])
    dup.sid = _uniq(dup.sid, {n.sid for n in nodes})
    out = nodes[:j + 1] + [dup] + nodes[j + 1:]
    return out, Deviation("repeat", "REPEAT_COMPLETED_STEP", dup.sid, {"of": sid, "after": at})


def truncate_and_serve(nodes: List[Subgoal], after: str) -> Tuple[List[Subgoal], Deviation]:
    """Drop everything after a subgoal and go serve (premature_serve)."""
    i = _idx(nodes, after)
    serve = next((n for n in nodes if n.type == "serve"), None)
    tail = copy.deepcopy(serve) if serve else Subgoal("serve", "serve", None, ["go to the serving pass", "serve"])
    tail.deviated = True
    dropped = [n.sid for n in nodes[i + 1:] if n.type != "serve"]
    out = nodes[:i + 1] + [tail]
    return out, Deviation("truncate_and_serve", "SKIP_PREREQUISITE", after, {"dropped": dropped})


def replace(nodes: List[Subgoal], sid: str, node: Subgoal) -> Tuple[List[Subgoal], Deviation]:
    """Substitute a subgoal with an injected node (e.g. assemble(meat) -> raw_scoop(meat))."""
    i = _idx(nodes, sid)
    node = copy.deepcopy(node)
    node.sid = _uniq(node.sid, {n.sid for n in nodes})
    out = nodes[:i] + [node] + nodes[i + 1:]
    return out, Deviation("replace", "WRONG_ACTION", node.sid, {"replaced": sid})


def permute_prep(nodes: List[Subgoal], order: List[str]) -> Tuple[List[Subgoal], Deviation]:
    """LEGALLY reorder the prep blocks (they are dependency-free among themselves).
    `order` = component names. This is a strategic deviation, not an error."""
    preps = [n for n in nodes if n.type == "prep"]
    if sorted(order) != sorted(n.component for n in preps):
        raise KeyError(f"permute_prep order {order} != prep components {[n.component for n in preps]}")
    by_comp = {n.component: n for n in preps}
    it = iter([copy.deepcopy(by_comp[c]) for c in order])
    out = [next(it) if n.type == "prep" else n for n in nodes]
    for n in out:
        if n.type == "prep":
            n.deviated = True
    return out, Deviation("permute", "REORDER_SUBGOALS", "prep(*)", {"order": order})


def forget_rest(nodes: List[Subgoal], after: str, stall_ticks: Optional[int] = None) -> Tuple[List[Subgoal], Deviation]:
    """The cook loses the plan: everything after `after` is dropped and it stands there,
    lost, until timeout. (The language layer later realizes this as REQUEST_NEXT_STEP,
    and a correct VLM hint would restore the dropped tail.)

    `stall_ticks` defaults to config.forget_freeze_ticks() — ONE named constant
    (FORGET_FREEZE_TICKS, env COOKSIM_FORGET_FREEZE) shared with the live harness's own
    freeze block, so the length of the stall is written down in exactly one place."""
    if stall_ticks is None:
        from .config import forget_freeze_ticks
        stall_ticks = forget_freeze_ticks()
    i = _idx(nodes, after)
    dropped = [n.sid for n in nodes[i + 1:]]
    nxt = nodes[i + 1] if i + 1 < len(nodes) else None
    lost = Subgoal("lost", "stall", None, [f"idle {stall_ticks}"], origin="injected", deviated=True)
    out = nodes[:i + 1] + [lost]
    return out, Deviation("forget_rest", "FORGET_NEXT_STEP", after,
                          {"dropped": dropped, "restorable_tail": dropped,
                           "next_step": {"subgoal": nxt.sid, "command": nxt.commands[0]} if nxt else None})


# --------------------------------------------------------------------------- ordered pipeline
OPS = {"skip": skip, "insert": insert, "swap": swap, "repeat": repeat, "replace": replace,
       "truncate_and_serve": truncate_and_serve, "permute_prep": permute_prep, "forget_rest": forget_rest}


def apply_pipeline(nodes: List[Subgoal], specs: List[dict]) -> Tuple[List[Subgoal], List[Deviation]]:
    """Apply transform specs IN ORDER. Earlier transforms own the plan: if a later spec's
    anchor no longer exists (e.g. forget_rest cleared it), that spec is SKIPPED and recorded
    as such — by design, not by error.  spec = {"op": <name>, ...kwargs}."""
    records: List[Deviation] = []
    for spec in specs:
        spec = dict(spec)
        op = spec.pop("op")
        try:
            nodes, rec = OPS[op](nodes, **spec)
        except KeyError as e:
            records.append(Deviation(op, "(not generated)", str(spec), {"reason": str(e)}, skipped=True))
            continue
        records.append(rec)
    return nodes, records


# --------------------------------------------------------------------------- compile
def compile_ir(nodes: List[Subgoal]) -> List[Tuple[str, str, bool]]:
    """Flatten to what the executor consumes: [(command, subgoal_id, deviated)]."""
    out = []
    for n in nodes:
        for c in n.commands:
            out.append((c, n.sid, n.deviated or n.origin == "injected"))
    return out


def flat_commands(nodes: List[Subgoal]) -> List[str]:
    return [c for c, _, _ in compile_ir(nodes)]
