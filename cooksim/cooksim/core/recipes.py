"""Recipe definitions and matching.

A recipe is identified by the exact multiset of (ingredient name, prep state)
that must sit on a delivered plate. Cooking stations do not need to know about
recipes — they simply transform ingredient states over time — so recipe
validation happens only at the serving window. This keeps the simulation
flexible: any combination a player can physically assemble can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .enums import PrepState
from .items import Plate


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    # multiset of (name, state) required on the final plate. When ``ordered``
    # is True this tuple is also the required ADDITION SEQUENCE (the order
    # ingredients must be placed on the plate / layered), not just a multiset.
    contents: Tuple[Tuple[str, str], ...]
    base_reward: float
    color: str = "#ffcc66"        # used by the renderer for the order ticket
    difficulty: int = 1           # influences order timer / spawn weighting
    steps: Tuple[str, ...] = ()   # human-readable prep steps (for the UI)
    ordered: bool = False         # if True, the plate must be assembled in the
                                  # exact order of ``contents`` (a stacked dish)

    @property
    def content_keys(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted(self.contents))

    def accepts(self, plate) -> bool:
        """Does this plate satisfy the recipe? The caller has already confirmed
        the multiset matches; ordered recipes additionally require the plate's
        addition sequence to equal ``contents``."""
        if not self.ordered:
            return True
        return plate.content_seq == self.contents


def _c(name: str, state: PrepState) -> Tuple[str, str]:
    return (name, state.value)


# ----------------------------------------------------------------------------
# Built-in recipe catalogue. Far richer than the single onion-soup of the
# original overcooked_ai. Recipes are grouped by the kind of prep they need.
# ----------------------------------------------------------------------------
RAW = PrepState.RAW
CHOP = PrepState.CHOPPED
COOK = PrepState.COOKED

DEFAULT_RECIPES: List[Recipe] = [
    # --- Soups (chop the veg, boil together in ONE pot) ----------------------
    Recipe("onion_soup", "Onion Soup", (_c("onion", COOK),) * 3, 20, "#e8c14a", 1,
           ("Chop 3 onions", "Boil them together in one pot", "Ladle up & serve at the pass")),
    Recipe("tomato_soup", "Tomato Soup", (_c("tomato", COOK),) * 3, 20, "#d9482f", 1,
           ("Chop 3 tomatoes", "Boil them together in one pot", "Ladle up & serve")),
    Recipe("mushroom_soup", "Mushroom Soup", (_c("mushroom", COOK),) * 3, 26, "#9c8466", 2,
           ("Chop 3 mushrooms", "Boil them together in one pot", "Ladle up & serve")),
    Recipe("mixed_soup", "Mixed Soup",
           (_c("onion", COOK), _c("onion", COOK), _c("tomato", COOK)), 30, "#cf7a3a", 2,
           ("Chop 2 onions and a tomato", "Boil the onions in one pot, the tomato in a second",
            "Combine in the bowl & serve")),
    # --- Salads (chop & assemble, no heat) ----------------------------------
    Recipe("garden_salad", "Garden Salad", (_c("lettuce", CHOP), _c("tomato", CHOP)), 18, "#5fb04a", 1,
           ("Chop lettuce", "Chop tomato", "Plate both & serve")),
    Recipe("caprese", "Caprese", (_c("tomato", CHOP), _c("cheese", RAW)), 20, "#df6d52", 1,
           ("Chop tomato", "Chopped tomato first, cheese on top — in that order", "Plate & serve"),
           ordered=True),
    Recipe("greek_salad", "Greek Salad",
           (_c("lettuce", CHOP), _c("tomato", CHOP), _c("cheese", RAW)), 30, "#7fc35a", 2,
           ("Chop lettuce", "Chop tomato", "Add cheese, plate & serve")),
    # --- Burgers (fry + chop + assemble) ------------------------------------
    Recipe("burger", "Burger", (_c("bun", RAW), _c("meat", COOK)), 26, "#b5651d", 2,
           ("Bun first", "then the fried patty on top", "Plate & serve"), ordered=True),
    Recipe("cheeseburger", "Cheeseburger",
           (_c("bun", RAW), _c("meat", COOK), _c("cheese", RAW)), 34, "#c87a2a", 3,
           ("Bun", "fried patty", "cheese on top — in that order"), ordered=True),
    Recipe("deluxe_burger", "Deluxe Burger",
           (_c("bun", RAW), _c("meat", COOK), _c("lettuce", CHOP), _c("tomato", CHOP)), 46, "#a85a1c", 4,
           ("Bun → patty → lettuce → tomato", "stacked in that order", "plate & serve"), ordered=True),
    # --- Pan / fried --------------------------------------------------------
    Recipe("fried_fish", "Fried Fish", (_c("fish", COOK),), 22, "#7fa8c9", 2,
           ("Fry the fish", "Plate & serve")),
    Recipe("fried_egg", "Fried Egg", (_c("egg", COOK),), 12, "#f4d35e", 1,
           ("Crack an egg in the pan", "Fry", "Plate & serve")),
    Recipe("loaded_fries", "Loaded Fries", (_c("potato", COOK), _c("cheese", RAW)), 26, "#e6b24a", 2,
           ("Chop the potato into fries, deep-fry the basket", "Fries down first, cheese on top",
            "Plate & serve"), ordered=True),
    Recipe("fish_and_chips", "Fish & Chips", (_c("fish", COOK), _c("potato", COOK)), 38, "#c8a36a", 3,
           ("Chop the potato into chips, deep-fry the basket", "Deep-fry the fish",
            "Plate both & serve")),
    # --- Rice ---------------------------------------------------------------
    Recipe("rice_bowl", "Rice Bowl", (_c("rice", COOK),), 12, "#efe7d2", 1,
           ("Cook the rice", "Plate & serve")),
    Recipe("sushi", "Sushi", (_c("rice", COOK), _c("fish", CHOP)), 36, "#a9c4d8", 3,
           ("Rice base first", "then chopped raw fish on top", "Plate & serve"), ordered=True),
    Recipe("fried_rice", "Fried Rice",
           (_c("rice", COOK), _c("egg", COOK), _c("onion", CHOP)), 44, "#dcb86a", 4,
           ("Rice → fried egg → chopped onion", "added in that order", "plate & serve"), ordered=True),
    # (pizza & veggie_bake were REMOVED in the realism overhaul: baking a pizza's
    #  components one by one is nonsense. The oven stays — dough only — until an
    #  assembled-bake mechanic exists.)
    # --- Multi-chain (pot/pan/fryer/board combos; the harder benchmark tier) --
    Recipe("shrimp_pasta", "Shrimp Pasta",
           (_c("noodles", COOK), _c("tomato", CHOP), _c("shrimp", COOK)), 44, "#e88a6a", 4,
           ("Boil the noodles in the pot", "Chop the tomato for the fresh sauce",
            "Sear the shrimp in the pan",
            "Plate noodles → tomato → shrimp — in that order"), ordered=True),
    Recipe("burrito", "Burrito",
           (_c("tortilla", RAW), _c("rice", COOK), _c("steak", COOK)),
           46, "#d9a441", 4,
           ("Boil the rice in the pot", "Slice the steak into strips, sear them in the pan",
            "Roll: tortilla → rice → steak — in that order"), ordered=True),
    Recipe("fish_and_chips_platter", "Fish & Chips Platter",
           (_c("fish", COOK), _c("potato", COOK), _c("potato", COOK)), 52, "#c8b06a", 5,
           ("Chop both potatoes into chips, deep-fry the basket",
            "Deep-fry the fish in its own basket",
            "Plate fish first, then the two portions of chips — in that order", "Serve"),
           ordered=True),
    Recipe("loaded_burrito", "Loaded Burrito",
           (_c("tortilla", RAW), _c("rice", COOK), _c("steak", COOK),
            _c("tomato", CHOP), _c("lettuce", CHOP)), 64, "#d98f3c", 5,
           ("Rice into the pot; slice the steak into strips and load the pan (no heat yet)",
            "Mise en place the cold toppings: chop the tomato salsa, shred the lettuce",
            "Fire the pot, then the pan — the rice holds, the steak comes off sizzling",
            "Build in order: tortilla → rice → steak → salsa → lettuce",
            "Roll it up & serve"), ordered=True),
    Recipe("steak_frites", "Steak Frites",
           (_c("steak", COOK), _c("potato", COOK), _c("potato", COOK)), 56, "#b8543c", 5,
           ("Slice the steak into strips, sear them in the pan",
            "Chop both potatoes into fries, deep-fry the basket",
            "Plate steak first, then the two portions of fries — in that order", "Serve"),
           ordered=True),
]


def feasible_recipes(recipes: List[Recipe], layout) -> List[Recipe]:
    """Subset of ``recipes`` that the given layout can physically produce.

    Requires a source for every ingredient, a cook station for any cooked
    component and a cutting board for any chopped component.
    """
    from .enums import COOK_STATIONS, SOURCE_ITEMS, Terrain, station_accepts

    flat = {t for row in layout.grid for t in row}
    available = {SOURCE_ITEMS[t] for t in flat if t in SOURCE_ITEMS}
    heat = flat & COOK_STATIONS
    has_board = Terrain.CUTTING_BOARD in flat

    def cookable(name: str) -> bool:
        """Station-typed + chain-aware: some station here takes it raw, or takes
        it chopped and there's a board to chop it on (steak, chips, soup veg)."""
        return any(station_accepts(t, name, "raw") for t in heat) or (
            has_board and any(station_accepts(t, name, "chopped") for t in heat))

    out = []
    for r in recipes:
        names = {n for n, _ in r.contents}
        states = {s for _, s in r.contents}
        if not names <= available:
            continue
        if any(s == PrepState.COOKED.value and not cookable(n) for n, s in r.contents):
            continue
        if PrepState.CHOPPED.value in states and not has_board:
            continue
        out.append(r)
    return out


class RecipeBook:
    """Holds the active recipes and answers matching queries."""

    def __init__(self, recipes: Optional[List[Recipe]] = None):
        self.recipes: List[Recipe] = list(recipes if recipes is not None else DEFAULT_RECIPES)
        # One multiset can map to several recipes that differ only by required
        # assembly order, so index to a LIST of candidates.
        self._by_keys: Dict[Tuple, List[Recipe]] = {}
        for r in self.recipes:
            self._by_keys.setdefault(r.content_keys, []).append(r)
        self._by_id: Dict[str, Recipe] = {r.id: r for r in self.recipes}

    def match_plate(self, plate: Plate) -> Optional[Recipe]:
        """Return the recipe a plate exactly satisfies, or None.

        Among recipes sharing the plate's multiset, an ordered recipe matches
        only if the addition sequence is right; it is preferred over an
        unordered recipe (it is the more specific dish)."""
        if plate.dirty or not plate.contents:
            return None
        # A burnt ingredient can never satisfy a recipe.
        if any(c.state == PrepState.BURNT for c in plate.contents):
            return None
        candidates = self._by_keys.get(plate.content_keys)
        if not candidates:
            return None
        for r in candidates:               # ordered dishes are the specific match
            if r.ordered and r.accepts(plate):
                return r
        for r in candidates:               # otherwise any order-free dish
            if not r.ordered:
                return r
        return None

    def get(self, recipe_id: str) -> Optional[Recipe]:
        return self._by_id.get(recipe_id)

    def __iter__(self):
        return iter(self.recipes)

    def to_dict(self) -> List[dict]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "contents": [{"name": n, "state": s} for n, s in r.contents],
                "reward": r.base_reward,
                "color": r.color,
                "difficulty": r.difficulty,
                "steps": list(r.steps),
                "ordered": r.ordered,
            }
            for r in self.recipes
        ]
