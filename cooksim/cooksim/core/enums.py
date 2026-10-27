"""Core enumerations for CookSim."""
from __future__ import annotations

from enum import Enum, IntEnum


class Direction(IntEnum):
    """Facing / movement directions. Values index into DIRECTION_VECTORS."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


# (dx, dy) in grid coordinates, y grows downward (screen convention).
DIRECTION_VECTORS = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


class Action(IntEnum):
    """Discrete actions available to each cook every tick."""

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    INTERACT = 4
    STAY = 5


# Actions that correspond to a movement/facing direction.
MOVE_ACTIONS = {
    Action.NORTH: Direction.NORTH,
    Action.SOUTH: Direction.SOUTH,
    Action.EAST: Direction.EAST,
    Action.WEST: Direction.WEST,
}


class Terrain(str, Enum):
    """A grid cell's terrain / fixed feature.

    ``FLOOR`` is the only walkable cell. Everything else is an impassable
    feature that a cook can interact with from an adjacent floor tile.
    """

    FLOOR = "floor"
    COUNTER = "counter"        # holds at most one loose item on top
    WALL = "wall"             # purely decorative / blocking, no interaction
    # Infinite sources -------------------------------------------------
    ONION_SOURCE = "onion_source"
    TOMATO_SOURCE = "tomato_source"
    LETTUCE_SOURCE = "lettuce_source"
    MUSHROOM_SOURCE = "mushroom_source"
    MEAT_SOURCE = "meat_source"
    FISH_SOURCE = "fish_source"
    BUN_SOURCE = "bun_source"
    CHEESE_SOURCE = "cheese_source"
    RICE_SOURCE = "rice_source"
    DOUGH_SOURCE = "dough_source"
    EGG_SOURCE = "egg_source"
    POTATO_SOURCE = "potato_source"
    STEAK_SOURCE = "steak_source"
    TORTILLA_SOURCE = "tortilla_source"
    NOODLES_SOURCE = "noodles_source"
    SHRIMP_SOURCE = "shrimp_source"
    # Distractor-only ingredients (2026-08-14 hard-map v2): no recipe uses them; they
    # exist to make wrong-but-plausible fetches and add_foreign temptations real.
    LEMON_SOURCE = "lemon_source"
    PEAS_SOURCE = "peas_source"
    BUTTER_SOURCE = "butter_source"
    PEPPER_SOURCE = "pepper_source"
    AVOCADO_SOURCE = "avocado_source"
    CORN_SOURCE = "corn_source"
    PLATE_SOURCE = "plate_source"
    # Processing stations ---------------------------------------------
    POT = "pot"               # boils / cooks ingredients into soup
    PAN = "pan"               # fries ingredients (patties, fish, eggs)
    OVEN = "oven"             # bakes (pizza, bread, baked dishes)
    FRYER = "fryer"           # deep-fries (fries, battered fish, shrimp)
    CUTTING_BOARD = "cutting_board"  # chop raw ingredients
    SINK = "sink"             # wash dirty plates
    # Terminals --------------------------------------------------------
    SERVING = "serving"       # deliver finished dishes
    TRASH = "trash"           # discard anything


# Which terrain types are infinite ingredient/plate dispensers.
SOURCE_ITEMS = {
    Terrain.ONION_SOURCE: "onion",
    Terrain.TOMATO_SOURCE: "tomato",
    Terrain.LETTUCE_SOURCE: "lettuce",
    Terrain.MUSHROOM_SOURCE: "mushroom",
    Terrain.MEAT_SOURCE: "meat",
    Terrain.FISH_SOURCE: "fish",
    Terrain.BUN_SOURCE: "bun",
    Terrain.CHEESE_SOURCE: "cheese",
    Terrain.RICE_SOURCE: "rice",
    Terrain.DOUGH_SOURCE: "dough",
    Terrain.EGG_SOURCE: "egg",
    Terrain.POTATO_SOURCE: "potato",
    Terrain.STEAK_SOURCE: "steak",
    Terrain.TORTILLA_SOURCE: "tortilla",
    Terrain.NOODLES_SOURCE: "noodles",
    Terrain.SHRIMP_SOURCE: "shrimp",
    Terrain.LEMON_SOURCE: "lemon",
    Terrain.PEAS_SOURCE: "peas",
    Terrain.BUTTER_SOURCE: "butter",
    Terrain.PEPPER_SOURCE: "pepper",
    Terrain.AVOCADO_SOURCE: "avocado",
    Terrain.CORN_SOURCE: "corn",
}

# Stations that apply heat (transform RAW/CHOPPED -> COOKED over time).
COOK_STATIONS = {Terrain.POT, Terrain.PAN, Terrain.OVEN, Terrain.FRYER}

WALKABLE = {Terrain.FLOOR}

# Stations that keep their own mutable, position-keyed state.
STATEFUL_STATIONS = {Terrain.POT, Terrain.PAN, Terrain.OVEN, Terrain.FRYER,
                     Terrain.CUTTING_BOARD, Terrain.SINK}


class PrepState(str, Enum):
    """Preparation state of an ingredient."""

    RAW = "raw"
    CHOPPED = "chopped"
    COOKED = "cooked"
    BURNT = "burnt"


# Single-character codes used in textual layout files. Mirrors (and extends)
# the overcooked_ai layout grammar so existing maps are easy to port.
TERRAIN_TO_CHAR = {
    Terrain.FLOOR: " ",
    Terrain.COUNTER: "X",
    Terrain.WALL: "#",
    Terrain.ONION_SOURCE: "O",
    Terrain.TOMATO_SOURCE: "T",
    Terrain.LETTUCE_SOURCE: "L",
    Terrain.MUSHROOM_SOURCE: "M",
    Terrain.MEAT_SOURCE: "E",
    Terrain.FISH_SOURCE: "F",
    Terrain.BUN_SOURCE: "B",
    Terrain.CHEESE_SOURCE: "C",
    Terrain.RICE_SOURCE: "R",
    Terrain.DOUGH_SOURCE: "G",
    Terrain.EGG_SOURCE: "Z",
    Terrain.POTATO_SOURCE: "Y",
    Terrain.STEAK_SOURCE: "K",
    Terrain.TORTILLA_SOURCE: "I",
    Terrain.NOODLES_SOURCE: "J",
    Terrain.SHRIMP_SOURCE: "H",
    # Distractor sources use lowercase codes: every uppercase letter that reads naturally
    # was already claimed, and lowercase cannot collide with the 1-9 spawn digits.
    Terrain.LEMON_SOURCE: "l",
    Terrain.PEAS_SOURCE: "p",
    Terrain.BUTTER_SOURCE: "b",
    Terrain.PEPPER_SOURCE: "r",
    Terrain.AVOCADO_SOURCE: "a",
    Terrain.CORN_SOURCE: "c",
    Terrain.PLATE_SOURCE: "D",
    Terrain.POT: "P",
    Terrain.PAN: "A",
    Terrain.OVEN: "N",
    Terrain.FRYER: "Q",
    Terrain.CUTTING_BOARD: "/",
    Terrain.SINK: "W",
    Terrain.SERVING: "S",
    Terrain.TRASH: "U",
}
CHAR_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_CHAR.items()}
# Players are marked 1-9 on floor cells in layout files.


# ----------------------------------------------------------------------------
# Station-typed cooking: which (ingredient, prep_state) inputs each heat
# station TYPE accepts — tuned for EVERYDAY-KITCHEN realism:
#   pot   — boiling: rice/noodles go in dry/raw; soup vegetables (onion/tomato/
#           mushroom) and stew potato must be CHOPPED first (nobody boils a
#           whole onion).
#   pan   — searing: a shaped patty (`meat`), fish fillet, egg or shrimp goes
#           in as-is; a raw `steak` must be SLICED into strips first (fajita
#           strips — pan accepts steak CHOPPED only).
#   oven  — baking: raw dough only. (Component-by-component "bakes" like the
#           old pizza were nonsense; the station stays for future
#           assembled-bake mechanics.)
#   fryer — deep-frying: CHOPPED potato (chips), shrimp raw, fish raw.
#
# An ingredient whose station accepts it CHOPPED but not RAW mechanically
# REQUIRES the chop->cook chain (generator.chain_chop derives chains from
# this table — the matrix is the single source of truth).
STATION_ACCEPTS = {
    Terrain.POT: {
        ("rice", "raw"), ("noodles", "raw"),
        ("onion", "chopped"), ("tomato", "chopped"),
        ("mushroom", "chopped"), ("potato", "chopped"),
    },
    Terrain.PAN: {
        ("meat", "raw"), ("fish", "raw"), ("egg", "raw"), ("shrimp", "raw"),
        ("steak", "chopped"),
    },
    Terrain.OVEN: {
        ("dough", "raw"),
    },
    Terrain.FRYER: {
        ("potato", "chopped"), ("shrimp", "raw"), ("fish", "raw"),
    },
}

# What a cutting board will take: real prep-choppable foods only. A shaped
# patty (`meat`), a bun, an egg, dry rice/noodles, a tortilla... are NOT
# choppable — the board refuses them with a reason.
CHOPPABLE = {"onion", "tomato", "lettuce", "mushroom", "potato", "fish", "cheese", "steak"}

_UNCHOPPABLE_REASON = {
    "meat": "nobody chops a patty — it's already shaped, sear it in the pan as-is",
    "bun": "you don't chop a bun",
    "egg": "you can't chop an egg — crack it straight into the pan",
    "rice": "rice isn't chopped — it boils in the pot as-is",
    "noodles": "noodles aren't chopped — they boil in the pot as-is",
    "tortilla": "a tortilla is used whole — nothing to chop",
    "shrimp": "the shrimp are already peeled — they cook whole",
    "dough": "dough is shaped, not chopped — it goes in the oven whole",
    # distractor-only ingredients: nothing on any order needs them prepared
    "lemon": "the lemon is only a garnish — nothing on the order needs it cut",
    "peas": "peas cook whole — there is nothing to chop",
    "butter": "butter is spread or melted, never chopped",
    "pepper": "the pepper mill grinds — there is nothing to chop",
    "avocado": "the avocado is not on the order — leave it whole",
    "corn": "corn stays on the cob — there is nothing to chop",
}


def choppable(name: str) -> bool:
    return name in CHOPPABLE


def chop_reject_reason(name: str) -> str:
    return _UNCHOPPABLE_REASON.get(name, f"you don't chop {name} — it is used whole")

# friendly verb per station type, for rejection feedback
_STATION_VERB = {Terrain.POT: "boiled in the pot", Terrain.PAN: "fried in the pan",
                 Terrain.OVEN: "baked in the oven", Terrain.FRYER: "deep-fried in the fryer"}


def station_rules_text() -> str:
    """The cooking-rules sentence for PROMPTS, generated from STATION_ACCEPTS so no prompt
    can ever contradict the engine again. W1 of the 2026-08-12 audit: the hand-written rule
    'cooked items are cooked at a pot/pan/oven (put down raw, turn on, wait)' omitted the
    fryer and every chop-then-cook chain (PAN takes steak CHOPPED only; FRYER takes potato
    CHOPPED only) — the supervisor followed its briefing and deadlocked 8 of 17 audited
    episodes ordering raw steak into pans."""
    lines = []
    for terrain in (Terrain.POT, Terrain.PAN, Terrain.FRYER, Terrain.OVEN):
        keys = sorted(STATION_ACCEPTS.get(terrain, ()))
        if not keys:
            continue
        label = terrain.value if hasattr(terrain, "value") else str(terrain)
        items = ", ".join(f"{st} {n}" if st != "raw" else n for n, st in keys)
        lines.append(f"the {label} takes ONLY: {items}")
    return ("Cooking stations accept exactly these inputs and refuse everything else — "
            + "; ".join(lines)
            + ".")


def station_accepts(terrain, name: str, state) -> bool:
    """Does this heat-station TYPE accept (ingredient, prep_state) as input?"""
    if terrain not in STATION_ACCEPTS:
        return False
    st = state.value if hasattr(state, "value") else str(state)
    return (name, st) in STATION_ACCEPTS[terrain]


def accepts_reason(terrain, name: str, state) -> str:
    """Human-readable reason a station refused an input, naming the right station
    (the feedback blocker text), e.g. 'dough must be baked in the oven'."""
    st = state.value if hasattr(state, "value") else str(state)
    fits = [t for t, keys in STATION_ACCEPTS.items() if (name, st) in keys]
    label = terrain.value if hasattr(terrain, "value") else str(terrain)
    what = f"{st} {name}" if st != "raw" else name
    # this very station takes the ingredient in ANOTHER prep state -> say so
    here = sorted({k[1] for k in STATION_ACCEPTS.get(terrain, ()) if k[0] == name})
    if here:
        return (f"the {label} doesn't take {what} — {name} must be {here[0]} first, "
                f"then {_STATION_VERB[terrain]}")
    if fits:
        return f"the {label} doesn't take {what} — {name} must be {_STATION_VERB[fits[0]]}"
    other = [t for t, keys in STATION_ACCEPTS.items() if any(k[0] == name for k in keys)]
    if other:
        need = sorted({k[1] for t in other for k in STATION_ACCEPTS[t] if k[0] == name})
        return (f"the {label} doesn't take {what} — {name} must be {need[0]} first, "
                f"then {_STATION_VERB[other[0]]}")
    return f"the {label} doesn't take {what} — no station cooks {name}"
