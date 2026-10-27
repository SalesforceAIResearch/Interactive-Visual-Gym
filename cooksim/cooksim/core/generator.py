"""Compositional recipe / event generator.

This is the implementation of the generative space described on the taxonomy
page: a small **archetype grammar** that, when sampled, fills ingredient/state
slots into a concrete :class:`~cooksim.core.recipes.Recipe`, validated against
the engine's real feasibility + capacity rules, and then placed onto a kitchen
layout with a difficulty-scaled order deadline (a runnable *event*).

Pipeline (mirrors the design note in docs/AGENT_HANDOFF.md §14.1):

    sample an archetype  →  fill slots from (ingredient × plausible-state)
    →  validate (well-formed + capacity + feasible_recipes on a layout)
    →  place the required sites on a layout to a target route length
    →  set a deadline  →  an Event ready to drop into KitchenGame.

Everything here is pure-Python and deterministic given a seed, so it can be used
to procedurally generate evaluation tasks (the streaming-VLM difficulty ladder,
RL curricula, etc.).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import GameConfig
from .enums import COOK_STATIONS, SOURCE_ITEMS, Terrain, PrepState
from .layout import Layout, generate_layout
from .recipes import Recipe, feasible_recipes

RAW, CHOP, COOK = PrepState.RAW.value, PrepState.CHOPPED.value, PrepState.COOKED.value

# Ingredient -> its dispenser terrain (inverse of SOURCE_ITEMS).
ING_SOURCE: Dict[str, Terrain] = {v: k for k, v in SOURCE_ITEMS.items()}
ALL_INGREDIENTS = sorted(ING_SOURCE)

Comp = Tuple[str, str]  # (ingredient, state)


# ---------------------------------------------------------------------------
# Archetype grammar  (encodes the 6 archetypes on the taxonomy page)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Archetype:
    """A slot grammar for one family of dishes.

    ``sample`` returns ``(contents, ordered)`` where ``contents`` is the
    multiset of (ingredient, state) components — and, when ``ordered`` is True,
    also the required addition sequence (a stacked dish).
    """
    key: str
    name: str
    icon: str
    station: str            # human label of the prep it needs
    blurb: str

    def sample(self, rng: random.Random) -> Tuple[Tuple[Comp, ...], bool]:  # pragma: no cover - overridden
        raise NotImplementedError


def _soup(rng):
    base = ["onion", "tomato", "mushroom"]
    x = rng.choice(base)
    if rng.random() < 0.4:                       # 2×X + 1×Y blend
        y = rng.choice([b for b in base if b != x])
        return ((x, COOK), (x, COOK), (y, COOK)), False
    return ((x, COOK), (x, COOK), (x, COOK)), False


def _salad(rng):
    veg = ["lettuce", "tomato", "onion"]
    k = rng.randint(1, 3)
    chosen = rng.sample(veg, k)
    comps = [(v, CHOP) for v in chosen]
    if rng.random() < 0.5:                        # optional raw cheese
        comps.append(("cheese", RAW))
    rng.shuffle(comps)                            # unordered: order is irrelevant
    return tuple(comps), False


def _burger(rng):
    comps: List[Comp] = [("bun", RAW)]
    comps.append((rng.choice(["meat", "fish"]), COOK))   # the patty
    for v in ("lettuce", "tomato"):
        if rng.random() < 0.5 and len(comps) < 4:
            comps.append((v, CHOP))
    if rng.random() < 0.4 and len(comps) < 4:
        comps.append(("cheese", RAW))
    # stacked: bun -> patty -> chopped veg -> cheese (already in build order)
    return tuple(comps), True


def _fried(rng):
    fried = ["fish", "egg", "potato"]
    k = rng.randint(1, 2)
    comps = [(f, COOK) for f in rng.sample(fried, k)]
    ordered = False
    if rng.random() < 0.45:                       # optional raw cheese on top (stacked)
        comps.append(("cheese", RAW))
        ordered = True                            # "X down first, then cheese"
    return tuple(comps), ordered


def _rice(rng):
    comps: List[Comp] = [("rice", COOK)]
    pool = [("egg", COOK), ("fish", CHOP), ("onion", CHOP)]
    rng.shuffle(pool)
    for c in pool:
        if rng.random() < 0.55 and len(comps) < 4:
            comps.append(c)
    ordered = len(comps) > 1                      # a stacked bowl (rice base first)
    return tuple(comps), ordered


# (the old "baked" archetype — 3 separately-baked components — was removed in the
# realism overhaul: nobody bakes a pizza's toppings one by one. The oven remains
# for future ASSEMBLED-bake mechanics.)

_SAMPLERS = {
    "soup": _soup, "salad": _salad, "burger": _burger,
    "fried": _fried, "rice": _rice,
}

ARCHETYPES: Dict[str, Archetype] = {
    "soup":   Archetype("soup", "Soup", "🍲", "pot",
                        "3×cooked X, or 2×cooked X + 1×cooked Y; X,Y ∈ {onion,tomato,mushroom}"),
    "salad":  Archetype("salad", "Salad", "🥗", "board",
                        "nonempty subset of chopped {lettuce,tomato,onion} + optional raw cheese"),
    "burger": Archetype("burger", "Burger / sandwich", "🍔", "pan + board",
                        "bun + (cooked meat∨fish) + subset{chopped lettuce,tomato} + optional cheese, ≤4"),
    "fried":  Archetype("fried", "Fried plate", "🍤", "pan",
                        "1–2 of cooked {fish,egg,potato} + optional raw cheese"),
    "rice":   Archetype("rice", "Rice bowl", "🍚", "pot/pan + board",
                        "cooked rice + subset{cooked egg, chopped fish, chopped onion}, ≤4"),
}


# ---------------------------------------------------------------------------
# Scoring  (reward + difficulty derived from the component mix)
# ---------------------------------------------------------------------------
_STATE_VALUE = {RAW: 4.0, CHOP: 7.0, COOK: 8.0}


def _reward(contents: Tuple[Comp, ...], ordered: bool) -> int:
    base = sum(_STATE_VALUE[s] for _, s in contents)
    base *= 1.0 + 0.12 * (len(contents) - 1)     # combo bonus
    if ordered:
        base += 2.0 * len(contents)              # stacking is harder
    return int(round(base))


def _difficulty(contents: Tuple[Comp, ...], ordered: bool) -> int:
    states = {s for _, s in contents}
    ops = sum(1 for s in (CHOP, COOK) if s in states)   # distinct prep machines used
    d = 1 + ops + (1 if ordered else 0) + (1 if len(contents) >= 4 else 0)
    return max(1, min(5, d))


# ---------------------------------------------------------------------------
# Capacity / well-formedness validation
# ---------------------------------------------------------------------------
def validate_recipe(recipe: Recipe, config: Optional[GameConfig] = None,
                    layout: Optional[Layout] = None) -> Tuple[bool, str]:
    """Check a recipe is physically realisable. Returns (ok, reason)."""
    cfg = config or GameConfig()
    c = recipe.contents
    if not c:
        return False, "empty recipe"
    for name, state in c:
        if name not in ING_SOURCE:
            return False, f"unknown ingredient {name!r}"
        if state not in (RAW, CHOP, COOK):
            return False, f"bad state {state!r}"
    if len(c) > cfg.plate_capacity:
        return False, f"{len(c)} components > plate capacity {cfg.plate_capacity}"
    # A soup/bake is one batch in a single cook station -> cooked count must fit
    # the largest cook station capacity (pot=3 / oven=3 / pan=2).
    n_cooked = sum(1 for _, s in c if s == COOK)
    max_cook_cap = max(cfg.pot_capacity, cfg.pan_capacity, cfg.oven_capacity)
    if n_cooked > max_cook_cap and len({n for n, s in c if s == COOK}) == 1:
        return False, f"{n_cooked} same cooked items > max cook capacity {max_cook_cap}"
    if layout is not None and recipe not in feasible_recipes([recipe], layout):
        return False, "layout cannot source/cook/chop this recipe"
    return True, "ok"


# ---------------------------------------------------------------------------
# Recipe sampling
# ---------------------------------------------------------------------------
def sample_recipe(rng: random.Random, archetype: Optional[str] = None,
                  config: Optional[GameConfig] = None, max_tries: int = 40) -> Recipe:
    """Sample one valid Recipe from the grammar (retries until it validates)."""
    cfg = config or GameConfig()
    for i in range(max_tries):
        akey = archetype or rng.choice(list(ARCHETYPES))
        contents, ordered = _SAMPLERS[akey](rng)
        if not ordered:
            # canonical multiset order for unordered dishes (stable id)
            contents = tuple(sorted(contents))
        rid = f"{akey}_" + "_".join(f"{n}.{s[0]}" for n, s in contents)
        steps = tuple(f"{'chop' if s==CHOP else 'cook' if s==COOK else 'add'} {n}" for n, s in contents)
        rec = Recipe(id=rid, name=_pretty_name(akey, contents), contents=contents,
                     base_reward=_reward(contents, ordered),
                     color=_ARCH_COLOR[akey], difficulty=_difficulty(contents, ordered),
                     steps=steps + ("plate & serve",), ordered=ordered)
        ok, _ = validate_recipe(rec, cfg)
        if ok:
            return rec
    raise RuntimeError(f"could not sample a valid {archetype or 'recipe'} in {max_tries} tries")


_ARCH_COLOR = {"soup": "#cf7a3a", "salad": "#5fb04a", "burger": "#b5651d",
               "fried": "#e6b24a", "rice": "#dcb86a"}


def _pretty_name(akey: str, contents: Tuple[Comp, ...]) -> str:
    cnt = Counter(n for n, _ in contents)
    parts = []
    for n in dict.fromkeys(n for n, _ in contents):
        parts.append(f"{cnt[n]}× {n}" if cnt[n] > 1 else n.capitalize())
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Layout placement + route-length estimate + deadline  ->  an Event
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """A runnable generated task: a recipe placed on a layout with a deadline."""
    recipe: Recipe
    layout: Layout
    deadline_ticks: int
    route_length: int
    archetype: str
    contents_seq: Tuple[Comp, ...] = field(default_factory=tuple)
    complexity: dict = field(default_factory=dict)   # {n_ops, path_length, map_complexity, map_features}

    def summary(self) -> str:
        seq = " → ".join(f"{n}·{s}" for n, s in self.recipe.contents) if self.recipe.ordered \
            else " + ".join(f"{n}·{s}" for n, s in self.recipe.contents)
        order = " [ordered]" if self.recipe.ordered else ""
        c = self.complexity
        cx = (f"complexity(n_ops={c['n_ops']}, path={c['path_length']}, map={c['map_complexity']})"
              if c else f"route≈{self.route_length}")
        return (f"[{self.archetype}] {self.recipe.name}{order}  reward={self.recipe.base_reward} "
                f"deadline={self.deadline_ticks}t  {cx}\n        {seq}")


def _required_sites(recipe: Recipe) -> Tuple[List[Terrain], int, int, int, int, int]:
    """Stations + sources a recipe needs.
    Returns (sources, n_pots, n_pans, n_ovens, n_fryers, n_boards)."""
    names = {n for n, _ in recipe.contents}
    states = {s for _, s in recipe.contents}
    sources = [ING_SOURCE[n] for n in sorted(names)]
    cooked = {n for n, s in recipe.contents if s == COOK}
    # chains (chop->cook) also need a board even without a chopped-on-plate item
    n_boards = 1 if (CHOP in states or any(chain_chop(recipe.id, n) for n in cooked)) else 0
    n_pots = n_pans = n_ovens = n_fryers = 0
    if cooked:
        stations = {cook_station_for(recipe.id, n) for n in cooked}
        n_pots = 1 if Terrain.POT in stations else 0
        n_pans = 1 if Terrain.PAN in stations else 0
        n_ovens = 1 if Terrain.OVEN in stations else 0
        n_fryers = 1 if Terrain.FRYER in stations else 0
        if not (n_pots or n_pans or n_ovens or n_fryers):
            n_pots = 1
    return sources, n_pots, n_pans, n_ovens, n_fryers, n_boards


def _bfs_dist(layout: Layout, a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Walking distance between the floor tiles adjacent to sites a and b."""
    from collections import deque

    def adj_floor(p):
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if layout.in_bounds(nx, ny) and layout.grid[ny][nx] == Terrain.FLOOR:
                return (nx, ny)
        return None

    sa, sb = adj_floor(a), adj_floor(b)
    if sa is None or sb is None:
        return 0
    seen = {sa}
    q = deque([(sa, 0)])
    while q:
        (x, y), d = q.popleft()
        if (x, y) == sb:
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if layout.in_bounds(nx, ny) and layout.grid[ny][nx] == Terrain.FLOOR and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append(((nx, ny), d + 1))
    return 0


def _bfs_dist_floor(layout: Layout, a, b) -> int:
    """Walking distance between two FLOOR tiles (both must be floor)."""
    from collections import deque
    if a == b:
        return 0
    seen = {a}; q = deque([(a, 0)])
    while q:
        (x, y), d = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) == b:
                return d + 1
            if layout.in_bounds(nx, ny) and layout.grid[ny][nx] == Terrain.FLOOR and (nx, ny) not in seen:
                seen.add((nx, ny)); q.append(((nx, ny), d + 1))
    return 0


def _route_length(layout: Layout, recipe: Recipe) -> int:
    """Rough lower bound on the walking a single cook must do: source→prep→
    ...→plate→serving, summed greedily in component order."""
    pos = {t: [] for t in Terrain}
    for y, row in enumerate(layout.grid):
        for x, t in enumerate(row):
            pos.setdefault(t, []).append((x, y))
    def first(t):
        return pos.get(t, [None])[0] if pos.get(t) else None
    plate = first(Terrain.PLATE_SOURCE); serve = first(Terrain.SERVING)
    if not plate or not serve:
        return 0
    total = 0
    cur = plate
    for n, s in recipe.contents:
        src = first(ING_SOURCE[n])
        if src:
            total += _bfs_dist(layout, cur, src); cur = src
        if s == COOK:
            st = first(Terrain.POT) or first(Terrain.PAN) or first(Terrain.OVEN)
            if st:
                total += _bfs_dist(layout, cur, st); cur = st
        elif s == CHOP:
            st = first(Terrain.CUTTING_BOARD)
            if st:
                total += _bfs_dist(layout, cur, st); cur = st
    total += _bfs_dist(layout, cur, serve)
    return total


# ---------------------------------------------------------------------------
# Formal complexity vector:  (n_ops, path_length, map_complexity)
# ---------------------------------------------------------------------------
import math as _math
from collections import Counter as _Counter

# which cook station an ingredient is conventionally cooked in (engine is
# recipe-blind, but for plan/route estimation we pick the natural one).
_COOK_STATION = {
    "rice": Terrain.POT, "onion": Terrain.POT, "tomato": Terrain.POT, "mushroom": Terrain.POT,
    "noodles": Terrain.POT,
    "meat": Terrain.PAN, "fish": Terrain.PAN, "egg": Terrain.PAN, "shrimp": Terrain.PAN,
    "steak": Terrain.PAN,
    "potato": Terrain.FRYER,                     # potato cooked = chips (chop -> deep-fry)
    "dough": Terrain.OVEN,
}

# Station-typed cooking can make the right station RECIPE-dependent (fish is
# pan-seared as a fillet but deep-fried in a fish&chips batter). Per-recipe
# routing overrides, keyed by id:
RECIPE_STATIONS = {
    "fish_and_chips": {"fish": Terrain.FRYER},
    "fish_and_chips_platter": {"fish": Terrain.FRYER},
}


def cook_station_for(recipe_id, name: str) -> Terrain:
    """The heat station a recipe cooks `name` in (recipe override > default)."""
    ov = RECIPE_STATIONS.get(recipe_id or "", {})
    if name in ov:
        return ov[name]
    return _COOK_STATION.get(name, Terrain.POT)


def chain_chop(recipe_id, name: str) -> bool:
    """True if `name` must be CHOPPED before its heat station will take it.
    Derived from the ACCEPTS matrix (single source of truth): a chain exists
    exactly when the station refuses the RAW ingredient but takes it CHOPPED
    (steak -> strips -> pan; potato -> chips -> fryer; soup veg -> pot)."""
    from .enums import station_accepts
    st = cook_station_for(recipe_id, name)
    return (not station_accepts(st, name, "raw")) and station_accepts(st, name, "chopped")


_INTERACTIVE = (set(SOURCE_ITEMS) | {Terrain.POT, Terrain.PAN, Terrain.OVEN, Terrain.FRYER,
                Terrain.CUTTING_BOARD, Terrain.SINK, Terrain.SERVING, Terrain.PLATE_SOURCE})


def cook_groups(recipe) -> "OrderedDict":
    """Cooked units BATCHED the way a real kitchen does it: all units of the same
    ingredient share ONE station (3 onions -> one pot; 2 potato portions -> one
    fryer basket). Returns {(name, station_terrain): unit_count} in first-seen order."""
    from collections import OrderedDict
    rid = getattr(recipe, "id", None)
    g = OrderedDict()
    for nm, s in recipe.contents:
        if s == COOK:
            key = (nm, cook_station_for(rid, nm))
            g[key] = g.get(key, 0) + 1
    return g


def n_operations(recipe: Recipe, config: Optional[GameConfig] = None) -> int:
    """Minimal number of HIGH-LEVEL (VLM) commands to complete the recipe.
    Each is one command: go-to / pick up / put down / turn on / chop / scoop / serve.

    Mirrors the canonical plan exactly (len(ground_truth_plan) == n_operations):
    cooked units are BATCHED per ingredient into one shared station (cook_groups);
    chain ingredients (chop->cook, derived from the ACCEPTS matrix) add a chop
    stage per unit; recipes with >=2 heat GROUPS use the LATE-IGNITE template
    (load everything, then one go-back-and-turn-on pass in plating order, so
    nothing burns while the solo cook preps the rest); one scoop plates a whole
    batch."""
    contents = list(recipe.contents)
    rid = getattr(recipe, "id", None)
    groups = cook_groups(recipe)
    late_ignite = len(groups) >= 2
    n = 2                                            # go to plates + pick up the plate
    for (nm, _st), cnt in groups.items():
        per_unit = 8 if chain_chop(rid, nm) else 4   # (+ go-board, put, chop, pick)
        n += cnt * per_unit
        n += 2 if late_ignite else 1                 # ignite the group's station
        n += 2                                       # ONE scoop plates the whole batch
    for nm, s in contents:
        if s == CHOP:
            n += 7                                   # prep 5 + assembly scoop 2
        elif s == RAW:
            n += 2                                   # assembly scoop only
    n += 2                                           # go to serving + serve
    return n


def _site(layout, terr):
    for y, row in enumerate(layout.grid):
        for x, t in enumerate(row):
            if t == terr:
                return (x, y)
    return None


def path_length(recipe: Recipe, layout: Layout) -> int:
    """Total tiles the cook must walk along the canonical plan: spawn -> (load each
    cooked/chopped item: source [-> board for chains] then its station) -> [multi-heat:
    an ignite pass back over the loaded stations] -> plates -> (assembly: each
    component's scoop site, in recipe order) -> serving. BFS distances summed."""
    cur = layout.start_positions[0] if layout.start_positions else (0, 0)
    total = 0
    rid = getattr(recipe, "id", None)
    groups = cook_groups(recipe)
    late_ignite = len(groups) >= 2

    def hop(dst):
        nonlocal cur, total
        if dst is not None:
            total += _bfs_dist(layout, cur, dst); cur = dst

    def station_site(name):
        return _site(layout, cook_station_for(rid, name)) or _first_cook(layout)

    for name, st in recipe.contents:                 # PREP / LOAD phase (per unit)
        if st == COOK:
            hop(_site(layout, ING_SOURCE[name]))
            if chain_chop(rid, name):
                hop(_site(layout, Terrain.CUTTING_BOARD))
            hop(station_site(name))
        elif st == CHOP:
            hop(_site(layout, ING_SOURCE[name]))
            hop(_site(layout, Terrain.CUTTING_BOARD))
    if late_ignite:                                   # IGNITE pass (one per group)
        for (name, _t) in groups:
            hop(station_site(name))
    hop(_site(layout, Terrain.PLATE_SOURCE))          # fetch the plate
    seen = set()
    for name, st in recipe.contents:                  # ASSEMBLY phase (in order)
        if st == COOK:
            if name in seen:
                continue                               # the batch was scooped at once
            seen.add(name)
            hop(station_site(name))
        elif st == CHOP:
            hop(_site(layout, Terrain.CUTTING_BOARD))
        else:
            hop(_site(layout, ING_SOURCE[name]))
    hop(_site(layout, Terrain.SERVING))               # deliver
    return total


def _first_cook(layout):
    for terr in (Terrain.POT, Terrain.PAN, Terrain.OVEN, Terrain.FRYER):
        s = _site(layout, terr)
        if s:
            return s
    return None


# map_complexity = W_PATH * path_length + W_GROUND * grounding_ambiguity  (tunable)
_MAP_W = {"path": 1.0, "ground": 6.0}


def grounding_ambiguity(layout: Layout) -> int:
    """# of DUPLICATE look-alike interactive tiles to visually disambiguate."""
    counts = _Counter(t for row in layout.grid for t in row if t in _INTERACTIVE)
    return sum(max(0, c - 1) for c in counts.values())


def map_complexity(recipe: Recipe, layout: Layout):
    """The kitchen's scene difficulty for THIS recipe. Returns (score, features).

    map_complexity = W_path * path_length + W_ground * grounding_ambiguity
    — a single 'navigation + grounding' load. path_length absorbs kitchen spread,
    obstacle detours, and how far apart the needed stations sit (navigation); the
    grounding term adds the perception cost of telling duplicate stations apart.
    Scaled by obstacles / spreading stations / adding duplicates — NOT by growing
    the grid (kept <=10x10 so instances stay legible)."""
    G = grounding_ambiguity(layout)
    P = path_length(recipe, layout)
    size = sum(1 for row in layout.grid for t in row if t == Terrain.FLOOR)
    score = round(_MAP_W["path"] * P + _MAP_W["ground"] * G, 1)
    return score, dict(grounding_ambiguity=G, path_length=P, traversal_size=size,
                       w_path=_MAP_W["path"], w_ground=_MAP_W["ground"])


_STATION_LABEL = {Terrain.POT: "pot", Terrain.PAN: "pan", Terrain.OVEN: "oven",
                  Terrain.FRYER: "fryer"}


def ground_truth_plan(recipe: Recipe) -> list:
    """The canonical optimal sequence of HIGH-LEVEL (VLM) commands to complete the
    recipe. Single source of truth: the plan-IR segmentation (plan_ir.segment_plan)
    flattened — so the grid's stored plans, the IR used by the error/interaction
    harnesses and this function can never drift apart. len == n_operations when
    each prep item has its own station (which the grid builder guarantees)."""
    from .plan_ir import segment_plan, flat_commands   # lazy: plan_ir imports us
    return flat_commands(segment_plan(recipe))


def complexity(recipe: Recipe, layout: Layout, config: Optional[GameConfig] = None) -> dict:
    """Task complexity on two axes: n_ops (planning depth) and map_complexity
    (navigation + grounding = weighted path_length + grounding_ambiguity)."""
    mc, feats = map_complexity(recipe, layout)
    return {"n_ops": n_operations(recipe, config),
            "path_length": feats["path_length"],
            "map_complexity": mc, "map_features": feats}


def generate_event(rng: random.Random, archetype: Optional[str] = None,
                   config: Optional[GameConfig] = None, seed: Optional[int] = None,
                   obstacles: int = 0, style: str = "ring") -> Event:
    """Sample a recipe and place it on a freshly generated, feasible layout
    with a difficulty-scaled deadline. ``obstacles`` scatters interior walls to
    raise navigation (map) complexity; ``style`` may be 'ring' or 'divided'."""
    cfg = config or GameConfig()
    recipe = sample_recipe(rng, archetype=archetype, config=cfg)
    sources, n_pots, n_pans, n_ovens, n_fryers, n_boards = _required_sites(recipe)
    # size the kitchen by difficulty so route length scales with the dish
    extra = recipe.difficulty
    layout = generate_layout(
        width=8 + extra, height=6 + (extra // 2), n_players=1,
        seed=(seed if seed is not None else rng.randrange(1 << 30)),
        n_pots=max(1, n_pots), n_pans=n_pans, n_ovens=n_ovens, n_fryers=n_fryers,
        n_cutting_boards=n_boards, include_sink=True,
        ingredient_sources=sources, style=style, obstacles=obstacles)
    ok, why = validate_recipe(recipe, cfg, layout)
    if not ok:
        raise RuntimeError(f"generated layout invalid for {recipe.id}: {why}")
    route = _route_length(layout, recipe)
    # deadline: walking time + per-component prep time + a comfort margin
    prep = sum(20 if s == COOK else 4 if s == CHOP else 1 for _, s in recipe.contents)
    deadline = int((route * 2 + prep) * 1.6) + 30
    return Event(recipe=recipe, layout=layout, deadline_ticks=deadline,
                 route_length=route, archetype=recipe.id.split("_", 1)[0],
                 contents_seq=recipe.contents, complexity=complexity(recipe, layout, cfg))


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def _main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Sample compositional CookSim recipes / events.")
    ap.add_argument("-n", type=int, default=12, help="how many to sample")
    ap.add_argument("--archetype", choices=list(ARCHETYPES), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", action="store_true", help="also place on a layout + deadline")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    print(f"=== {a.n} sampled {'events' if a.events else 'recipes'} "
          f"({a.archetype or 'any archetype'}, seed={a.seed}) ===\n")
    for i in range(a.n):
        if a.events:
            print(generate_event(rng, archetype=a.archetype).summary())
        else:
            r = sample_recipe(rng, archetype=a.archetype)
            seq = (" → " if r.ordered else " + ").join(f"{n}·{s}" for n, s in r.contents)
            print(f"[{r.id.split('_',1)[0]:6}] {r.name:28} reward={r.base_reward:3} "
                  f"diff={r.difficulty} {'ORD ' if r.ordered else '    '} {seq}")
    print()


if __name__ == "__main__":
    _main()
