"""Manipulation-level natural-language control for CookSim.

A streaming VLM drives the cook with small *physical* commands, never whole
sub-tasks. Two kinds of command:

  LOCOMOTION (the only mover, and the only command that names a target):
      "go to <object>"        — pathfind adjacent to an object and face it.
      Targets may carry qualifiers to disambiguate when several exist:
        content  : "go to the pot cooking rice"
        state    : "go to the done pot" / "the empty board" / "the dirty plate"
        spatial  : "the left pot" / "the nearest pot" / "the top-left pan"
        identity : "pot 2"           (stations numbered row-major within a type)
        exact    : "the pot at (2,0)"

  MANIPULATION (acts on the tile the cook FACES; never moves; one object op):
      pick up · put down · chop · turn on · scoop · serve · wash · discard · wait

No command bundles task functionality: there is no "cook the soup" or "chop the
potato" that fetches+places+dices. The VLM must compose a chain and time it
against what it sees. A raw low-level action (N/S/E/W/INTERACT/STAY, by name or
int) is also accepted for fine-grained control.

The controller is closed-loop and DEFENSIVE: every parse / grounding / execution
problem becomes a clean ``failed`` status with a human-readable reason — it never
raises and never leaves the engine in a bad state (commands only ever emit valid
low-level actions).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.enums import (SOURCE_ITEMS, Action, Direction, Terrain,
                          accepts_reason, chop_reject_reason, choppable, station_accepts)
from ..core.items import Ingredient, Plate
from ..core.stations import CookStation, CuttingBoard, Sink
from .heuristic import _adjacent_floor, _bfs_first_step, _dir_between

Pos = Tuple[int, int]

_D2A = {Direction.NORTH: Action.NORTH, Direction.SOUTH: Action.SOUTH,
        Direction.EAST: Action.EAST, Direction.WEST: Action.WEST}
_SOURCE_BY_NAME = {n: t for t, n in SOURCE_ITEMS.items()}
INGREDIENTS = sorted(_SOURCE_BY_NAME)
_ALIAS = {"patty": "meat", "beef": "steak", "sirloin": "steak", "bread": "bun",
          "salmon": "fish", "shroom": "mushroom", "spud": "potato", "egg": "egg",
          "prawn": "shrimp", "wrap": "tortilla", "pasta": "noodles"}
_STATION_TERRAIN = {
    "pot": (Terrain.POT,), "pan": (Terrain.PAN,), "oven": (Terrain.OVEN,),
    "fryer": (Terrain.FRYER,),
    "board": (Terrain.CUTTING_BOARD,), "sink": (Terrain.SINK,),
    "serving": (Terrain.SERVING,), "trash": (Terrain.TRASH,),
    "counter": (Terrain.COUNTER,), "plate": (Terrain.PLATE_SOURCE,),
}
# Public ID kinds (2026-08-15 grounding redesign): the label the cook's text state map
# prints and the grounder accepts. Derived from _STATION_TERRAIN so the two cannot drift.
_ID_KIND = {terrains[0]: name for name, terrains in _STATION_TERRAIN.items()
            if name not in ("serving", "trash", "plate")}
_ID_KIND[Terrain.SERVING] = "serving pass"
_ID_KIND[Terrain.TRASH] = "trash"
_ID_KIND[Terrain.PLATE_SOURCE] = "plate stack"

_STATION_SYN = {
    "pot": "pot", "pan": "pan", "skillet": "pan", "oven": "oven",
    "fryer": "fryer", "deep fryer": "fryer", "deep-fryer": "fryer",
    "cutting board": "board", "chopping board": "board", "board": "board",
    "sink": "sink", "serving": "serving", "pass": "serving", "window": "serving",
    "hatch": "serving", "delivery": "serving", "counter": "counter",
    "trash": "trash", "bin": "trash", "garbage": "trash",
    "plate stack": "plate", "plate dispenser": "plate", "plates": "plate", "plate": "plate",
}
_STATES = {"done": "cooked", "ready": "cooked", "cooked": "cooked", "finished": "cooked",
           "empty": "empty", "cooking": "cooking", "burning": "burnt", "burnt": "burnt",
           "raw": "raw", "dirty": "dirty", "clean": "clean", "full": "full", "chopped": "chopped"}
_SPATIAL = ["top-left", "top left", "top-right", "top right", "bottom-left", "bottom left",
            "bottom-right", "bottom right", "leftmost", "rightmost", "nearest", "closest",
            "left", "right", "top", "bottom", "upper", "lower", "far"]

_VERBS = {
    "goto":   ["go to", "walk to", "move to", "head to", "navigate to", "go back to", "approach", "go towards"],
    "wait":   ["wait", "idle", "do nothing"],
    "chop":   ["chop", "cut", "slice", "dice", "mince"],
    "start":  ["turn on", "start", "begin cooking", "switch on", "ignite"],
    "scoop":  ["scoop", "ladle", "plate up", "dish out", "spoon", "add to plate", "scoop onto"],
    "serve":  ["serve", "deliver", "hand in", "hand over", "submit"],
    "wash":   ["wash", "clean", "rinse"],
    "trash":  ["trash", "discard", "throw away", "throw out", "dump", "bin it"],
    "putdown":["put down", "set down", "put", "place", "drop", "add", "load", "insert"],
    "pickup": ["pick up", "grab", "take", "get", "collect", "lift", "fetch"],
}
_VERB_ORDER = ["goto", "wait", "chop", "start", "scoop", "serve", "wash", "trash", "putdown", "pickup"]
_LOW = {"north": Action.NORTH, "up": Action.NORTH, "south": Action.SOUTH, "down": Action.SOUTH,
        "east": Action.EAST, "right": Action.EAST, "west": Action.WEST, "left": Action.WEST,
        "interact": Action.INTERACT, "stay": Action.STAY, "wait_tick": Action.STAY}

VERB_HELP = {
    "goto": "go to <object>  — the only move; names a target (+qualifiers)",
    "pickup": "pick up  — take the item you face (into an empty hand)",
    "putdown": "put down  — drop the held item into/onto the tile you face",
    "chop": "chop  — dice the one raw item on the board you face",
    "start": "turn on  — start the filled pot/pan/oven/fryer you face",
    "scoop": "scoop  — add the faced station's food onto the plate you hold",
    "serve": "serve  — hand the finished plate to the serving pass you face",
    "wash": "wash  — run a dirty plate through the sink you face",
    "trash": "discard  — bin the item you hold at the trash you face",
    "wait": "wait [until <object> is <state>]  — idle / watch",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    verb: str
    obj: str = ""
    ingredient: str = ""
    quals: Dict[str, object] = field(default_factory=dict)
    until: Optional["Intent"] = None
    raw: str = ""


def _canon_ingredient(t: str) -> str:
    for n in INGREDIENTS:
        if re.search(rf"\b{n}\b", t):
            return n
    for a, n in _ALIAS.items():
        if re.search(rf"\b{a}\b", t):
            return n
    return ""


_COORD_RE = re.compile(r"\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?")
_ID_RE = re.compile(
    r"\b((?:\w+\s+)?source|pot|pan|oven|fryer|cutting board|board|counter|sink)"
    r"\s*#?\s*(\d+)\b")


def _parse_quals(t: str) -> Dict[str, object]:
    q: Dict[str, object] = {}
    # GROUNDING REDESIGN (user design 2026-08-15). The cook now reads a TEXT STATE MAP
    # that prints every object with an ID and a coordinate, so it can name exactly what it
    # means: `go to fryer 2`, `go to the fryer at (6,4)`. Both are parsed here.
    #
    # History this reverses: coordinates and identity numbers were deliberately UNPARSED
    # from 2026-08-03, because the ASSISTANT had started using them to point at things and
    # it read as micromanagement. That reason still holds for the assistant — and it is
    # unchanged: it still sees pixels and still speaks in spatial language. What changed is
    # that the two channels are now separate. The COOK's channel to the engine is exact
    # (id / coordinate); the CONVERSATION between assistant and cook stays human ("the
    # fryer on the far side of the island"), and translating one into the other is the
    # cook's job. So the escape hatch is now the cook's front door, and the thing that made
    # it bad — an assistant reciting coordinates at a person — is prevented in the prompt
    # contract instead of by crippling the parser.
    from ..core.config import is_v1
    if not is_v1():                       # v1 never parsed ids or coordinates
        mid = _ID_RE.search(t)
        if mid:
            q["obj_id"] = f"{_STATION_SYN.get(mid.group(1), mid.group(1))} {int(mid.group(2))}"
        mco = _COORD_RE.search(t)
        if mco:
            q["coord"] = (int(mco.group(1)), int(mco.group(2)))
    for s in _SPATIAL:
        if f" {s} " in f" {t} ":
            q["spatial"] = s.replace(" ", "-")
            break
    else:
        # W10 (user ruling 2026-08-12: NO silent drops): a positional word we do not
        # support used to vanish, so "the middle pan" walked to the NEAREST pan and
        # reported done with zero movement. Now it refuses, naming the word.
        mu = re.search(r"\b(middle|center|centre|central)\b", t)
        if mu:
            q["spatial_unknown"] = mu.group(1)
    for w, st in _STATES.items():
        if re.search(rf"\b{w}\b", t):
            q.setdefault("state", st)
    mc = re.search(r"(?:with|containing|holding|cooking|that has|that contains|of)\s+"
                   r"(?:the\s+|some\s+|a\s+)?(\w+(?:\s+\w+)?)", t)
    if mc:
        ci = _canon_ingredient(" " + mc.group(1) + " ")
        if ci:
            q["content"] = ci
        elif mc.group(1).split()[0] in ("plate", "plates"):
            q["content"] = "plate"           # "the counter with the plate"
        else:
            # W10: "the counter with the fire extinguisher" — an object that does not
            # exist — used to DROP the qualifier and resolve to the nearest counter,
            # confirming the assistant's hallucination with a `done`. Record the unknown
            # so grounding refuses with its name.
            q["content_unknown"] = mc.group(1)
    return q


def _parse_target(t: str) -> Tuple[str, str, Dict[str, object]]:
    """Return (obj_token, ingredient, quals) for a 'go to' / 'wait until' phrase."""
    quals = _parse_quals(t)
    ing = _canon_ingredient(t)
    # pick the EARLIEST-mentioned station word (the head noun usually comes first,
    # e.g. "the PLATES on the left counter" -> plate, not counter); tie-break longer.
    best = None
    for syn, canon in _STATION_SYN.items():
        m = re.search(rf"\b{re.escape(syn)}\b", t)
        if m:
            key = (m.start(), -len(syn))
            if best is None or key < best[0]:
                best = (key, canon)
    obj = best[1] if best else ("source" if ing else "")
    return obj, ing, quals


def grounding_vocabulary() -> str:
    """The CLOSED vocabulary the v1 grounder accepts, rendered for the v1 prompts.

    v2 retired this: the cook names objects by id or coordinate from its own state map,
    and the assistant is told to describe places like a person and is never shown ids or
    coordinates. Calling it under v2 means a prompt is regressing to the shared word list,
    so it raises there rather than silently re-teaching the old contract.
    """
    from ..core.config import is_v1
    if not is_v1():
        raise RuntimeError("grounding_vocabulary() is v1-only — under v2 the cook grounds "
                           "by id or coordinate and the assistant describes places in words")
    ing = ", ".join(INGREDIENTS)
    states = " · ".join(sorted(set(_STATES)))
    spatial = " · ".join(_SPATIAL)
    stations = " · ".join(sorted(set(_STATION_TERRAIN)))
    return ("  station   : " + stations + "\n"
            '  content   : "with/containing/holding the <ingredient>" — ' + ing + "\n"
            "  state     : " + states + "\n"
            "  spatial   : " + spatial + "  (TOP-DOWN KITCHEN GRID, not the cook's body)\n"
            "  Anything else does not select — say one of these or name the contents.")


# The cook's planning skills (resolve_subgoal / expand_action) answer from the recipe TEMPLATE, which
# names places generically ("go to the potato source"). On a map with several instances of that kind
# the engine refuses such a command, so the skill marks the step instead of guessing for the cook
# (user ruling 2026-09-11). A cook that pastes the marked step back gets the ordinary refusal.
GROUNDING_REMINDER = "(multiple instances in this map, resolve grounding by yourself)"


def strip_grounding_reminder(text):
    if not isinstance(text, str):
        return text
    i = text.lower().find(GROUNDING_REMINDER)
    return (text[:i] + text[i + len(GROUNDING_REMINDER):]).strip() if i >= 0 else text


def needs_grounding_reminder(cmd: str, game) -> bool:
    """True when `cmd` names a place by a kind this map has more than one of, without an id, a
    coordinate, a content qualifier ("the pan with the steak") or a side ("the left pan") — i.e. a
    step the engine would refuse, now or once the kitchen changes."""
    it = parse(cmd)
    if it is None or it.verb not in ("goto", "wait") or game is None:
        return False
    tgt = it if it.verb == "goto" else it.until
    if tgt is None:
        return False
    if any(k in tgt.quals for k in ("obj_id", "coord", "content", "spatial")):
        return False
    probe = Intent(verb=tgt.verb, obj=tgt.obj, ingredient=tgt.ingredient, quals=dict(tgt.quals),
                   until=tgt.until, raw=tgt.raw)
    try:
        ground(probe, game, game.cooks[0])
    except Exception:
        pass
    if probe.quals.get("_ambiguous"):
        return True                                    # the engine would refuse it right now
    kind = f"{tgt.ingredient} source" if (tgt.obj in ("", "source") and tgt.ingredient) else tgt.obj
    return sum(1 for v in _object_ids(game).values() if v.rsplit(" ", 1)[0] == kind) > 1


def parse(text: str) -> Optional[Intent]:
    if not isinstance(text, str):
        return None
    text = strip_grounding_reminder(text)
    t = " " + text.strip().lower() + " "
    verb = None
    for v in _VERB_ORDER:
        for ph in sorted(_VERBS[v], key=len, reverse=True):
            if f" {ph} " in t or t.strip().startswith(ph + " ") or t.strip() == ph:
                verb = v
                break
        if verb:
            break
    if verb is None:
        return None
    it = Intent(verb=verb, raw=text.strip())
    if verb == "goto":
        it.obj, it.ingredient, it.quals = _parse_target(t)
    elif verb == "wait":
        m = re.search(r"\buntil\b(.*)", t)
        if m:
            o, ing, q = _parse_target(" " + m.group(1) + " ")
            sub = Intent("goto", o, ing, q)
            it.until = sub
        nm = re.search(r"\b(\d+)\b", t)
        if nm and "id" not in it.quals:
            it.quals["ticks"] = int(nm.group(1))
    return it


# ---------------------------------------------------------------------------
# Grounding (for 'go to' / 'wait until')
# ---------------------------------------------------------------------------
_INSTANCE_STATIONS = {"pot", "pan", "oven", "fryer", "board"}   # several may hold DIFFERENT things -> instance matters


@dataclass
class Target:
    tile: Optional[Pos]
    label: str
    ambiguous: bool = False
    n: int = 1


def _tiles_of(game, terrains) -> List[Pos]:
    return [(x, y) for y, row in enumerate(game.layout.grid)
            for x, t in enumerate(row) if t in terrains]


def _state_match(obj, state) -> bool:
    if isinstance(obj, CookStation):
        return {"empty": obj.status == "empty" or not obj.contents,
                "cooked": obj.status == "cooked", "cooking": obj.status == "cooking",
                "burnt": obj.status == "burnt", "raw": obj.status == "raw",
                "full": obj.is_full}.get(state, False)
    if isinstance(obj, CuttingBoard):
        if state == "empty":
            return obj.item is None
        if state in ("chopped", "cooked"):
            return obj.item is not None and obj.item.state.value == "chopped"
        if state == "raw":
            return obj.item is not None and obj.item.state.value == "raw"
    if isinstance(obj, Sink):
        if state == "dirty":
            return obj.dirty > 0
        if state == "clean":
            return obj.clean_ready > 0
    return False


def _has_content(obj, ing) -> bool:
    if isinstance(obj, CookStation):
        return any(i.name == ing for i in obj.contents)
    if isinstance(obj, CuttingBoard):
        return obj.item is not None and obj.item.name == ing
    return False


def _pick_spatial(tiles, spatial, cook) -> Pos:
    if spatial in ("nearest", "closest"):
        return min(tiles, key=lambda p: abs(p[0] - cook.x) + abs(p[1] - cook.y))
    if spatial == "far":
        return max(tiles, key=lambda p: abs(p[0] - cook.x) + abs(p[1] - cook.y))
    keys = {"left": lambda p: (p[0], p[1]), "leftmost": lambda p: (p[0], p[1]),
            "right": lambda p: (-p[0], p[1]), "rightmost": lambda p: (-p[0], p[1]),
            "top": lambda p: (p[1], p[0]), "upper": lambda p: (p[1], p[0]),
            "bottom": lambda p: (-p[1], p[0]), "lower": lambda p: (-p[1], p[0]),
            "top-left": lambda p: (p[0] + p[1], p[1]), "top-right": lambda p: (-p[0] + p[1], p[1]),
            "bottom-left": lambda p: (p[0] - p[1], -p[1]), "bottom-right": lambda p: (-p[0] - p[1], -p[1])}
    return min(tiles, key=keys.get(spatial, lambda p: (p[1], p[0])))


def _object_ids(game) -> Dict[tuple, str]:
    """tile -> stable public ID ("pan 2", "board 1", "tomato source 1", "counter 7").

    ONE source of truth, shared by the grounder and by the cook's text state map
    (cooksim/agents/observe_text.py), so the label the cook READS is exactly the label the
    engine ACCEPTS. Row-major, 1-based, per object kind — stable for a fixed layout, and
    independent of contents so an ID never changes under the cook mid-episode.
    """
    from ..core.enums import SOURCE_ITEMS as _SI
    grid = game.layout.grid
    kinds: Dict[str, list] = {}
    for y, row in enumerate(grid):
        for x, t in enumerate(row):
            if t in _SI:
                kind = f"{_SI[t]} source"
            elif t in _ID_KIND:
                kind = _ID_KIND[t]
            else:
                continue
            kinds.setdefault(kind, []).append((x, y))
    out: Dict[tuple, str] = {}
    for kind, tiles in kinds.items():
        for i, p in enumerate(sorted(tiles, key=lambda p: (p[1], p[0])), 1):
            out[p] = f"{kind} {i}"
    return out


def ground(it: Intent, game, cook) -> Optional[Target]:
    obj = it.obj
    # W10: never resolve past an unknown qualifier — a hallucinated object or an
    # unsupported positional word must refuse (with its name, via _ground_fail_reason),
    # not fall through to "nearest". The referee never guesses.
    if "content_unknown" in it.quals or "spatial_unknown" in it.quals:
        return None
    if obj == "source" or (not obj and it.ingredient):
        if it.ingredient in _SOURCE_BY_NAME:
            tiles = _tiles_of(game, (_SOURCE_BY_NAME[it.ingredient],))
            # a crate is addressable by id like anything else — the cook's state map
            # prints "potato source 2" and its prompt uses that form
            if "obj_id" in it.quals:
                ids = _object_ids(game)
                tiles = [t for t in tiles if ids.get(t) == it.quals["obj_id"]]
            if "coord" in it.quals:
                tiles = [t for t in tiles if t == it.quals["coord"]]
            from ..core.config import is_v1 as _v1
            if (not _v1() and len(tiles) > 1
                    and not any(k in it.quals for k in ("obj_id", "coord", "spatial"))):
                ids = _object_ids(game)
                it.quals["_ambiguous"] = [ids.get(t, f"({t[0]},{t[1]})") for t in tiles]
                return None
            if tiles:
                return Target(_pick_spatial(tiles, it.quals.get("spatial", "nearest"), cook),
                              f"{it.ingredient} source")
        return None
    terrains = _STATION_TERRAIN.get(obj)
    if not terrains:
        return None
    cands = _tiles_of(game, terrains)
    if not cands:
        return None
    if obj == "counter":
        if it.quals.get("state") == "empty":
            cands = [c for c in cands if game.counter_items.get(c) is None] or cands
        if "content" in it.quals:
            want = it.quals["content"]

            def _on_counter(c):
                itm = game.counter_items.get(c)
                if itm is None:
                    return False
                if want == "plate":
                    from ..core.items import Plate
                    return isinstance(itm, Plate)
                if getattr(itm, "name", None) == want:
                    return True
                return any(getattr(i, "name", None) == want
                           for i in getattr(itm, "contents", []) or [])

            cands = [c for c in cands if _on_counter(c)]
            if not cands:
                return None
    elif "content" in it.quals:
        matched = [c for c in cands if _has_content(game.stations.get(c), it.quals["content"])]
        if not matched:
            return None                    # "the pan with the meat" and no pan has meat: no
        cands = matched                    # silent nearest-pan fallback — fail with feedback
    if "state" in it.quals and not (obj == "counter" and it.quals["state"] == "empty"):
        # W10: the old `or cands` fallback DEFEATED the state qualifier — "the empty pan"
        # silently walked to a full one. A state filter that matches nothing now refuses.
        matched = [c for c in cands if _state_match(game.stations.get(c), it.quals["state"])]
        if not matched:
            return None
        cands = matched
    # ---- EXACT GROUNDING (2026-08-15): an ID or a coordinate names ONE tile. A wrong one
    # refuses by name rather than resolving to something else (see _ground_fail_reason).
    if "obj_id" in it.quals:
        ids = _object_ids(game)
        want = it.quals["obj_id"]
        cands = [c for c in cands if ids.get(c) == want]
        if not cands:
            return None
    if "coord" in it.quals:
        cands = [c for c in cands if c == it.quals["coord"]]
        if not cands:
            return None
    if len(cands) > 1 and "spatial" not in it.quals:
        # GROUNDING REQUIREMENT (user ruling 2026-08-15, strict form): if a kind has more
        # than one instance, the cook MUST say which — always. No "these two look alike so
        # I'll guess" shortcut: a silent nearest-pick is what let "go to the middle right
        # fryer" report `done` at the wrong fryer for 200+ ticks (audit E2/E3), and the
        # nearest-pick itself ranks by Manhattan distance, which is simply wrong in a maze
        # (measured: it sends the cook to a fryer 2 ticks farther than the real nearest).
        # An engine that never guesses cannot guess wrong.
        #
        # The engine's OWN scripted plan meets the same bar rather than being exempt from
        # it: bind_scripted_command() resolves each plan command to a concrete instance and
        # rewrites it with that id before it is issued, so the plan says which one it means
        # and the log records the choice.
        from ..core.config import is_v1
        if not is_v1():
            ids = _object_ids(game)
            it.quals["_ambiguous"] = [ids.get(c, f"({c[0]},{c[1]})") for c in cands]
            return None
    amb = len(cands) > 1 and "spatial" not in it.quals
    return Target(_pick_spatial(cands, it.quals.get("spatial", "nearest"), cook), obj, ambiguous=amb, n=len(cands))


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
def _faced(game, cook):
    fx, fy = cook.facing_pos()
    if not game.layout.in_bounds(fx, fy):
        return None, None, None, None
    terrain = game.layout.grid[fy][fx]
    return (fx, fy), terrain, game.stations.get((fx, fy)), game.counter_items.get((fx, fy))


class _Skill:
    label = "skill"

    def step(self, game, cook):
        raise NotImplementedError


class GoTo(_Skill):
    def __init__(self, tile, label, max_ticks=300):
        self.tile, self.label, self.max_ticks, self.t = tile, f"go to {label}", max_ticks, 0
        self.error = None

    def step(self, game, cook):
        self.t += 1
        if self.t > self.max_ticks:
            self.error = "timeout reaching target"
            return int(Action.STAY), "failed"
        stand = _adjacent_floor(game, *self.tile)
        if not stand:
            self.error = "no floor tile next to the target"
            return int(Action.STAY), "failed"
        if cook.pos not in stand:
            a = _bfs_first_step(game, cook.pos, set(stand))
            if a is None:
                self.error = "target is unreachable"
                return int(Action.STAY), "failed"
            return int(a), "running"
        want = _dir_between(cook.pos, self.tile)
        if want is not None and cook.direction != want:
            return int(_D2A[want]), "running"
        return int(Action.STAY), "done"


# manipulation precondition + done predicates, keyed by verb -----------------
# ECR-13 (2026-08-13): `_validate` is the SINGLE ORACLE for every manipulation. It was always
# what refused the action at execution time, but `checks.is_doable` — the affordance oracle the
# cook-facing sentence and the supervisor's "blocked: ..." line are built from — carried a
# second, kitchen-wide copy of the same rules and disagreed with it in 1087 of the 2162 audited
# cells (987 of them told the cook "it wouldn't work" while the sentence it needed sat here).
# is_doable now delegates to this function, so the reason the cook hears, the reason the
# supervisor sees and the rule the body acts on are one string. Everything written below must
# therefore name the FACED tile and a way out.
MANIPULATIONS = ("pickup", "putdown", "chop", "start", "turnon", "scoop", "serve",
                 "wash", "trash")


def _burnt(st) -> str:
    """W22: a burnt station is on fire, irrecoverable — and still FULL. Every verb that lands
    on one has to say so; telling the cook it is empty (or that its food "must finish cooking")
    sends the cook back to load and light a dead station."""
    what = st.contents[0].name if st.contents else "the food"
    return (f"the {what} is BURNT — the station is on fire and cannot be restarted, emptied "
            f"or scooped; leave it and use another station")


def _validate(verb, game, cook):
    """Return (ok: bool, reason: str). Manipulations act on the FACED tile."""
    if verb == "turnon":                  # the parser only ever emits `start`; callers vary
        verb = "start"
    tile, terrain, st, ci = _faced(game, cook)
    held = cook.holding
    if verb == "pickup":
        if held is not None:
            return False, "your hands are full"
        if terrain in SOURCE_ITEMS or terrain == Terrain.PLATE_SOURCE:
            return True, ""
        if isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value != "raw":
            return True, ""
        if ci is not None:
            return True, ""
        # X2: a loaded heat station is a one-way trap — pickup is refused here, scoop wants
        # `cooked`, trash wants a held item — and the refusal used to reach the cook as
        # "it wouldn't work", burning 110-245 ticks in five episodes. Name the way out.
        if isinstance(st, CookStation) and st.contents:
            if st.status == "burnt":
                return False, _burnt(st)
            what = st.contents[0].name
            if st.status == "cooked":
                return False, (f"food can't be picked out of a station by hand — the {what} "
                               f"is DONE, hold a plate and SCOOP it")
            return False, (f"food can't be taken back out of a station — the {what} must "
                           f"finish cooking (turn the station on if it isn't), then scoop it "
                           f"onto a plate")
        return False, "nothing here to pick up — go to a source/board/counter first"
    if verb == "putdown":
        if held is None:
            return False, "your hands are empty"
        if isinstance(st, CookStation) and isinstance(held, Ingredient):
            # station-typed cooking: this station TYPE must accept the input
            if held.state.value in ("raw", "chopped") and \
                    not station_accepts(terrain, held.name, held.state):
                return False, accepts_reason(terrain, held.name, held.state)
            if st.can_add():
                return True, ""
        if isinstance(st, CuttingBoard) and st.item is None and isinstance(held, Ingredient) and held.state.value == "raw":
            if not choppable(held.name):       # e.g. a patty: the board refuses it
                return False, chop_reject_reason(held.name)
            return True, ""
        if terrain == Terrain.COUNTER and ci is None:
            return True, ""
        if isinstance(st, CookStation) and st.status == "burnt":
            return False, _burnt(st)
        return False, "can't put that down here — face an empty board/counter or an open pot/pan/oven/fryer"
    if verb == "chop":
        if held is not None:
            return False, "free your hands to chop (put the item on the board first)"
        if isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value == "raw":
            return True, ""
        return False, "you're not facing a board with a raw item — go to a board and put something on it"
    if verb == "start":
        if not isinstance(st, CookStation):
            return False, "you're not facing a pot/pan/oven/fryer"
        if st.status in ("cooking", "cooked"):
            return True, ""           # already on — a no-op success, ANNOUNCED by _noop_note
        if st.status == "burnt":
            return False, _burnt(st)
        # X6 + ECR-13 §D2: the Manipulate skill drives this with INTERACT, and INTERACT at a
        # heat station with an ACCEPTED ingredient in hand LOADS it (game._interact_cook_
        # station). A turn-on with full hands therefore performed an unasked-for put-down and
        # reported `done` — in 7 audited cells — at the very moment the doability path was
        # telling the cook its hands had to be empty. Refuse before the interact can fire.
        if held is not None:
            return False, ("your hands must be EMPTY to turn a station on — park what you're "
                           "holding on an empty counter first")
        if st.status == "raw" and st.contents:
            return True, ""
        return False, "nothing to start — the station is empty; put an ingredient in it first"
    if verb == "scoop":
        if not isinstance(held, Plate):
            return False, "hold a clean plate to scoop onto"
        if held.dirty:
            return False, ("hold a clean plate to scoop onto — the one you are carrying is "
                           "DIRTY, wash it at the sink first")
        if isinstance(st, CookStation) and st.status == "cooked":
            return True, ""
        if isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value == "chopped":
            return True, ""
        if terrain in SOURCE_ITEMS:
            return True, ""
        # Overcooked rule: a loose ingredient parked on a plain counter can be scooped
        # onto a held plate (the engine's _interact_counter already combines them —
        # this validation gate was the only thing rejecting it)
        if terrain == Terrain.COUNTER and isinstance(ci, Ingredient):
            return True, ""
        if isinstance(st, CookStation) and st.status == "burnt":
            return False, _burnt(st)
        if isinstance(st, CookStation) and st.status == "cooking":
            what = st.contents[0].name if st.contents else "the food"
            return False, (f"the {what} is still cooking — it can't be scooped until the "
                           f"station is done")
        # X2: loaded but never lit. It will not become scoopable on its own, so say that
        # instead of letting the scoop spin its 60-tick retry window in silence.
        if isinstance(st, CookStation) and st.status == "raw" and st.contents:
            return False, (f"the {st.contents[0].name} is sitting in a station that was never "
                           f"turned on — it will NEVER cook or become scoopable until the "
                           f"station is switched on (with empty hands)")
        return False, "nothing ready to scoop here — face a done station, a chopped board, a source, or a counter with food on it"
    if verb == "serve":
        # ECR-13 §B3/§B4: ONE sentence used to cover "no plate", "dirty plate" and "empty
        # plate", so a cook with EMPTY HANDS was told its plate was not complete enough (151
        # cells) and a cook holding a DIRTY plate was sent to pile more food onto a plate the
        # pass will never take (32 cells). Name the obstacle that is actually there.
        if held is None:
            return False, ("your hands are empty — hold a finished (non-empty, clean) plate "
                           "to serve: take a plate and put the cooked food on it first")
        if not isinstance(held, Plate):
            return False, (f"you are holding the {held.name}, not a plate — hold a finished "
                           f"(non-empty, clean) plate to serve; food goes on a plate first")
        if held.dirty:
            return False, ("that plate is DIRTY — the pass will not take it; wash it at the "
                           "sink before serving")
        if terrain != Terrain.SERVING:
            return False, "you're not facing the serving pass"
        # E4 (hard-map audit, user ruling 2026-08-15): serving an EMPTY plate used to be
        # refused here as undoable, so the command was illegal-skipped — the delivery
        # never reached the game, no outcome was produced, and a takeover whose queue
        # ended in that serve left the cook frozen until timeout (hm_search_expM9: 35
        # ticks idle, 18 nags charged). Handing the pass an empty plate is a real thing a
        # cook can do and a real mistake: it is now DOABLE and the pass rejects it as a
        # failed delivery (game._interact_serving), i.e. a wrong_serve outcome.
        return True, ""
    if verb == "wash":
        if not isinstance(st, Sink):
            return False, "you're not facing a sink"
        if (isinstance(held, Plate) and held.dirty) or st.dirty > 0 or st.clean_ready > 0:
            return True, ""
        return False, "nothing to wash"
    if verb == "trash":
        if held is None:
            return False, "your hands are empty"
        if terrain != Terrain.TRASH:
            return False, "you're not facing the trash"
        # ECR-13 §D4: `_interact_trash` returns without doing anything for a clean, empty
        # plate, but `_done()` calls the command complete — so the cook announced
        # "you DID: discard" while still holding the same plate. Refuse instead of lying.
        if isinstance(held, Plate) and not held.contents and not held.dirty:
            return False, ("there is nothing to discard — the plate you are holding is "
                           "already empty and clean (keep it, or put it down on a counter)")
        return True, ""
    return False, f"unknown manipulation '{verb}'"


def _noop_note(verb, game, cook):
    """A LEGAL command that will change nothing says so. `turn on` at an already-lit station
    is a success by contract (_validate treats it as a no-op), but a bare `done` told the cook
    it had just lit the station — 93 cells of the audit, ECR-13 §D1. The note rides on the log
    entry into the kitchen feedback ("you DID: ... — the station was ALREADY ON")."""
    if verb in ("start", "turnon"):
        _tile, _terrain, st, _ci = _faced(game, cook)
        if isinstance(st, CookStation) and st.status == "cooking":
            return "the station was ALREADY ON and cooking — nothing to turn on"
        if isinstance(st, CookStation) and st.status == "cooked":
            return ("the station was ALREADY ON and its food is DONE — nothing to turn on, "
                    "scoop it onto a plate")
    return None


# --------------------------------------------------------------------------- wait
# ECR-13 §D3: `wait until <station> is <state>` at a station nobody lit stood still for the
# Wait skill's entire 600-tick budget — longer than most episodes — and only then reported a
# bare `failed`. A conditional wait is admissible only if its condition can still ARRIVE from
# the live world; if it cannot, the wait is refused at ISSUE time, and a wait whose condition
# dies MID-wait (the station is emptied, or burns) ends THEN, with a sentence. Timed
# `wait N` is untouched: it always ends.
_HEAT_CONDITIONS = ("cooked", "cooking", "burnt")


def _station_wait_blocker(st, state, label):
    """None if this ONE station can still reach `state` on its own; else why it never will."""
    if _state_match(st, state):
        return None                        # already true: the wait ends on its first tick
    if st is None:
        if state not in _HEAT_CONDITIONS:
            return None                    # a plain tile, a non-heat condition: don't guess
        return (f"the {label} is not a cooking station — it can never be '{state}', so this "
                f"wait would never end; wait on the pot/pan/oven/fryer doing the work")
    if not isinstance(st, CookStation) or state not in _HEAT_CONDITIONS:
        return None                        # boards and sinks: another cook may act; no guess
    if st.status == "burnt":
        # first, and before the `cooking` tolerance below: a station that burnt while lit
        # keeps `cooking` True forever.
        return (f"the {label} is BURNT — it is on fire and will never be {state}; that "
                f"station is dead, so this wait would never end")
    if st.status == "cooked":
        if state == "burnt":
            return None                    # still on the heat: it burns if nobody scoops it
        return (f"the {label} is already DONE — it will never be '{state}' again; scoop it "
                f"onto a plate instead of waiting")
    if st.status == "cooking" or st.cooking:
        # LIT is `cooking`, not the status label: a station can carry status "raw" while the
        # heat is on (interact.system's repair blocks build exactly that), and refusing there
        # would break a wait whose condition really is on its way.
        return None
    if st.contents:
        return (f"the {label} is loaded but not turned on — this wait would never end; turn "
                f"the {label} on first (hands must be empty)")
    return (f"the {label} is empty — nothing in it can become {state}, so this wait would "
            f"never end; put food in and turn the {label} on first")


def wait_condition_blocker(until: Optional[Intent], game, cook) -> Optional[str]:
    """None if `wait until <until>` can still come true from the CURRENT world; otherwise the
    sentence saying why it never will.

    The wait ends as soon as ANY instance of the target matches (`Wait.step` re-grounds every
    tick, and `ground` filters by the very state being waited for), so the condition is
    inadmissible only when NO instance can get there. Spatial qualifiers are deliberately
    ignored: they choose among matches, they cannot make a condition reachable."""
    if until is None:
        return None
    state = until.quals.get("state", "cooked")
    label = _FRIENDLY.get(until.obj, until.obj or "target")
    terrains = _STATION_TERRAIN.get(until.obj)
    if terrains is None:
        if until.ingredient and state in _HEAT_CONDITIONS:
            # "wait until the meat is cooked" grounds to the meat SOURCE, which no amount of
            # waiting changes (the same confusion interact.system.py:336 works around).
            return (f"the {until.ingredient} source is not a cooking station — it can never "
                    f"be '{state}'; wait on the pot/pan/oven/fryer that is cooking it "
                    f"(e.g. 'wait until the pan is done')")
        return None                        # nothing we can reason about: never guess
    tiles = _tiles_of(game, terrains)
    if "content" in until.quals:
        tiles = [t for t in tiles if _has_content(game.stations.get(t), until.quals["content"])]
    if not tiles:
        probe = Intent("goto", until.obj, until.ingredient,
                       {k: v for k, v in until.quals.items() if k != "state"}, raw=until.raw)
        return _ground_fail_reason(probe, game)
    if any(_station_wait_blocker(game.stations.get(t), state, label) is None for t in tiles):
        return None
    return _station_wait_blocker(
        game.stations.get(_pick_spatial(tiles, "nearest", cook)), state, label)


def wait_admissible(it: Optional[Intent], game, cook):
    """(ok, reason) for a parsed `wait` intent — the form `checks.is_doable` consumes."""
    if it is None or it.verb != "wait" or it.until is None:
        return True, ""                    # a timed wait always ends
    b = wait_condition_blocker(it.until, game, cook)
    return (b is None), (b or "")


def _done(verb, game, cook, snap):
    tile, terrain, st, ci = _faced(game, cook)
    held = cook.holding
    if verb == "pickup":
        return held is not None
    if verb == "putdown":
        return held is None
    if verb == "chop":
        return isinstance(st, CuttingBoard) and st.item is not None and st.item.state.value == "chopped"
    if verb == "start":
        return isinstance(st, CookStation) and st.status in ("cooking", "cooked")
    if verb == "scoop":
        return isinstance(held, Plate) and len(held.contents) > snap.get("plate", 0)
    if verb == "serve":
        # E4 (2026-08-15): `not held.contents` used to count as success, because an
        # empty-plate serve was illegal and could never execute — so an empty plate could
        # only mean "the food already went to the pass". Now that handing the pass an
        # empty plate is a legal FAILED delivery, an empty clean plate is the PRE-state,
        # and treating it as success made the skill finish in 0 ticks without ever calling
        # the game (a silent no-op the fuzz matrix caught). What actually proves a serve
        # happened: the plate left the hands, or it came back dirty from the pass.
        return held is None or (isinstance(held, Plate) and held.dirty)
    if verb == "wash":
        return isinstance(held, Plate) and not held.dirty
    if verb == "trash":
        return held is None or (isinstance(held, Plate) and not held.contents and not held.dirty)
    return True


class Manipulate(_Skill):
    def __init__(self, verb, max_ticks=40):
        self.verb, self.label, self.max_ticks, self.t = verb, verb, max_ticks, 0
        self.error = None
        self.note = None                   # a legal command that changes nothing says so
        self._validated = False
        self._snap = {}

    def step(self, game, cook):
        self.t += 1
        if not self._validated:
            ok, reason = _validate(self.verb, game, cook)
            if not ok:
                self.error = reason
                return int(Action.STAY), "failed"
            self._validated = True
            self.note = _noop_note(self.verb, game, cook)
            held = cook.holding
            self._snap = {"plate": len(held.contents) if isinstance(held, Plate) else 0}
        if _done(self.verb, game, cook, self._snap):
            return int(Action.STAY), "done"
        if self.t > self.max_ticks:
            self.error = f"'{self.verb}' did not complete (stuck)"
            return int(Action.STAY), "failed"
        return int(Action.INTERACT), "running"


class Wait(_Skill):
    def __init__(self, ticks=5, until=None, max_ticks=600):
        self.label = "wait"
        self.n, self.until, self.max_ticks, self.t = ticks, until, max_ticks, 0
        self.error = None

    def step(self, game, cook):
        self.t += 1
        if self.until is not None:
            tgt = ground(self.until, game, cook)
            state = self.until.quals.get("state", "cooked")
            if tgt and tgt.tile is not None and _state_match(game.stations.get(tgt.tile), state):
                return int(Action.STAY), "done"
            # ECR-13 §D3: a condition that can no longer ARRIVE (the station was emptied, or
            # it burnt) ends the wait NOW and in words, instead of at budget expiry.
            blocker = wait_condition_blocker(self.until, game, cook)
            if blocker:
                self.error = blocker
                return int(Action.STAY), "failed"
            if self.t > self.max_ticks:
                lbl = _FRIENDLY.get(self.until.obj, self.until.obj or "target")
                self.error = (f"waited {self.max_ticks} ticks and the {lbl} never became "
                              f"{state} — stop waiting and look at the station")
                return int(Action.STAY), "failed"
            return int(Action.STAY), "running"
        if self.t > self.n:
            return int(Action.STAY), "done"
        return int(Action.STAY), "running"


class Raw(_Skill):
    def __init__(self, action):
        self.action, self.label, self._fired = int(action), f"low:{Action(int(action)).name}", False

    def step(self, game, cook):
        if self._fired:
            return int(Action.STAY), "done"
        self._fired = True
        return self.action, "running"


# ---------------------------------------------------------------------------
# Controller / driver
# ---------------------------------------------------------------------------
_FRIENDLY = {"board": "cutting board"}

def _cook_xy(game):
    try:
        c = game.cooks[0]
        return (c.x, c.y)
    except Exception:  # noqa: BLE001
        return None


def _ground_fail_reason(it, game):
    """Precise reason a 'go to' target could not be grounded (for the VLM feedback)."""
    obj = it.obj
    # ---- GROUNDING CONTRACT (user design 2026-08-15). Three answers, in the order the
    # grounder can fail: the ID does not exist, the coordinate holds no such object, or
    # the engine needs grounding the cook did not give.
    if "obj_id" in it.quals:
        ids = _object_ids(game)
        if it.quals["obj_id"] not in ids.values():
            kinds = sorted({v.rsplit(" ", 1)[0] for v in ids.values()
                            if v.rsplit(" ", 1)[0] == it.quals["obj_id"].rsplit(" ", 1)[0]})
            have = [v for v in sorted(ids.values())
                    if kinds and v.startswith(kinds[0] + " ")]
            return (f"there is no '{it.quals['obj_id']}' in this kitchen"
                    + (f" — the ones here are: {', '.join(have)}" if have else ""))
    if "coord" in it.quals:
        x, y = it.quals["coord"]
        ids = _object_ids(game)
        there = ids.get((x, y))
        if not game.layout.in_bounds(x, y):
            return f"({x},{y}) is outside the kitchen"
        what = there or (str(game.layout.grid[y][x]).split(".")[-1].replace("_", " ").lower()
                         if game.layout.in_bounds(x, y) else "off the grid")
        return f"there is no {obj or 'object'} at ({x},{y}) — that tile is {what}"
    if it.quals.get("_ambiguous"):
        # Several instances survived the qualifiers, so the engine will not guess. Name the
        # candidates — but HOW depends on how many there are. Counters are the stress case:
        # a maze has ~14 of them, and reciting fourteen ids is not guidance, it is a wall.
        # For a crowded kind the answer is the coordinate form plus the few that are
        # actually usable from here; ids stay the answer for the handful-sized kinds.
        opts = sorted(it.quals["_ambiguous"])
        kind = (f"{it.ingredient} source" if (obj in ("", "source") and it.ingredient)
                else (obj or "one"))
        said = " that matches what you said" if any(
            k in it.quals for k in ("content", "state")) else ""
        head = (f"I need grounding — there is more than one {kind}{said} and I don't know "
                f"which one you mean.")
        if len(opts) > 6:
            tiles = [t for t in _tiles_of(game, _STATION_TERRAIN.get(obj, ()))
                     if _object_ids(game).get(t) in set(opts)]
            if it.quals.get("state") == "empty" or obj == "counter":
                tiles = [t for t in tiles if game.counter_items.get(t) is None] or tiles
            near = sorted(tiles, key=lambda p: abs(p[0] - cook_pos[0]) + abs(p[1] - cook_pos[1])
                          )[:4] if (cook_pos := _cook_xy(game)) else tiles[:4]
            coords = ", ".join(f"({x},{y})" for x, y in near)
            return (f"{head} There are {len(opts)} of them — name one by COORDINATE, e.g. "
                    f'"go to the {obj} at {coords.split(", ")[0]}". Nearest usable: {coords}')
        return f"{head} Name it by id or coordinate: {', '.join(opts)}"
    # W10 (user ruling: no silent rejection — the refusal must NAME the problem)
    if "content_unknown" in it.quals:
        w = it.quals["content_unknown"]
        return (f"there is no '{w}' in this kitchen — nothing like that exists here, "
                f"so I can't go to a {obj or 'place'} with it")
    if "spatial_unknown" in it.quals:
        w = it.quals["spatial_unknown"]
        return (f"I don't know which {obj or 'one'} '{w}' means — say left/right/top/bottom, "
                f"or name it by contents ('the {obj or 'pan'} with the meat')")
    if "content" in it.quals and obj in _STATION_TERRAIN and _tiles_of(game, _STATION_TERRAIN[obj]):
        return (f"no {obj} has {it.quals['content']} in it right now — "
                f"I won't guess a different one")
    if "state" in it.quals and obj in _STATION_TERRAIN and _tiles_of(game, _STATION_TERRAIN[obj]):
        return (f"no {obj} is {it.quals['state']} right now — "
                f"I won't guess a different one")
    if obj in _STATION_TERRAIN:
        if not _tiles_of(game, _STATION_TERRAIN[obj]):
            name = _FRIENDLY.get(obj, obj)
            return f"there is no {name} in this kitchen"
        return f"could not pick which {obj} you mean"
    if it.ingredient:
        if not any(SOURCE_ITEMS.get(t) == it.ingredient for row in game.layout.grid for t in row):
            return f"there is no {it.ingredient} source in this kitchen"
        return f"could not reach the {it.ingredient} source"
    obj_txt = re.sub(r"^\s*(go|walk|move|head|navigate)\s+(to|towards)?\s*(the\s+)?", "", it.raw, flags=re.I).strip()
    return f"there is no '{obj_txt}' in this kitchen"


def _bind_ambiguous(it, game, cook):
    """Resolve an ambiguous SCRIPTED reference to one concrete instance.

    The engine demands that a kind with several instances be named exactly (user ruling
    2026-08-15). Its own reference plans are authored late-bound — "go to the empty cutting
    board" — because which instance is right depends on the run. That is a plan-authoring
    convenience, not an exemption from the rule: the plan is resolved to a concrete tile
    HERE, at issue time, so exactly one instance is ever acted on.

    Deliberately NOT done by rewriting the command string: the logged command text is
    matched verbatim in several places (intent verification, deviated-section membership,
    splice dedupe), and appending "(board 2)" to it silently broke those comparisons.
    """
    relaxed = Intent(verb=it.verb, obj=it.obj, ingredient=it.ingredient,
                     quals={k: v for k, v in it.quals.items() if k != "_ambiguous"},
                     until=it.until, raw=it.raw)
    relaxed.quals["spatial"] = "nearest"
    return ground(relaxed, game, cook)


def compile_action(nl, game, cook, strict=True, bind=False):
    """(skill, None) on success or (None, reason). Never raises."""
    try:
        if isinstance(nl, Action):
            nl = int(nl)
        if isinstance(nl, int):
            if 0 <= nl <= 5:
                return Raw(nl), None
            return None, f"invalid low-level action int: {nl}"
        if not isinstance(nl, str) or not nl.strip():
            return None, "empty command"
        low = _LOW.get(nl.strip().lower())
        if low is not None:
            return Raw(int(low)), None
        it = parse(nl)
        if it is None:
            return None, f"no recognized verb in {nl!r}"
        if it.verb == "goto":
            t = ground(it, game, cook)
            if t is None and bind and it.quals.get("_ambiguous"):
                t = _bind_ambiguous(it, game, cook)      # scripted plan: bind, don't guess
            if t is None or t.tile is None:
                return None, _ground_fail_reason(it, game)
            if strict and t.ambiguous and it.obj in _INSTANCE_STATIONS:
                return None, (f"AMBIGUOUS: {t.n} {it.obj}s exist and may hold different things — say WHICH one "
                              f"(e.g. 'the left {it.obj}' / 'the right {it.obj}' / 'the {it.obj} 2' / "
                              f"'the {it.obj} with the <ingredient>' / 'the done {it.obj}' / 'the empty {it.obj}').")
            return GoTo(t.tile, t.label), None
        if it.verb == "wait":
            blocker = wait_condition_blocker(it.until, game, cook)
            if blocker:
                return None, blocker       # a wait that could never end is refused at issue
            return Wait(ticks=int(it.quals.get("ticks", 5)), until=it.until), None
        return Manipulate(it.verb), None
    except Exception as e:                       # defensive: never crash the loop
        return None, f"internal error compiling {nl!r}: {e!r}"


class HighLevelController:
    """Holds the active command for one cook; turns it into low-level actions."""

    def __init__(self, game, cook_id: int = 0):
        self.game, self.cook_id = game, cook_id
        self.skill: Optional[_Skill] = None
        self.status, self.error = "idle", None
        self.note = None                   # set when a finished command changed nothing

    @property
    def cook(self):
        return self.game.cooks[self.cook_id]

    def issue(self, action, strict=True, bind=False) -> bool:
        self.skill, self.error = compile_action(action, self.game, self.cook,
                                                strict, bind=bind)
        self.status = "running" if self.skill is not None else "failed"
        return self.skill is not None

    def step(self) -> int:
        if self.skill is None:
            return int(Action.STAY)
        try:
            act, self.status = self.skill.step(self.game, self.cook)
        except Exception as e:                   # defensive
            self.status, self.error = "failed", f"internal error: {e!r}"
            self.skill = None
            return int(Action.STAY)
        if self.status in ("done", "failed"):
            self.error = getattr(self.skill, "error", None)
            self.note = getattr(self.skill, "note", None)
            self.skill = None
        return int(act)


def run_one(game, command, cook_id=0, others=None, max_ticks=400):
    """Execute a single high-level command, stepping the sim. Returns a dict."""
    ctrl = HighLevelController(game, cook_id)
    if not ctrl.issue(command):
        return {"command": command, "status": "failed", "ticks": 0, "error": ctrl.error}
    t = 0
    while ctrl.status == "running" and t < max_ticks:
        a = ctrl.step()
        acts = [int(Action.STAY)] * len(game.cooks)
        acts[cook_id] = a
        if others:
            for cid in range(len(game.cooks)):
                if cid != cook_id:
                    acts[cid] = int(others(game, cid))
        game.step(acts)
        t += 1
    status = ctrl.status if t < max_ticks else "timeout"
    return {"command": command, "status": status, "ticks": t, "error": ctrl.error}


def run_plan(game, plan, cook_id=0, others=None, max_ticks_per=400):
    """Execute a list of commands in order; returns a per-command log."""
    return [run_one(game, c, cook_id, others, max_ticks_per) for c in plan]
