"""NoisyHuman — a cook that executes a shared plan but deliberately deviates.

Wraps the reliable scripted GreedyChef for correct play, and injects a *seeded,
deterministic* schedule of deviations (wrong location, raw-on-plate, stall, add
ingredient, dawdle, ...). Each deviation is a small state machine that seizes
control, records ground-truth labels (start/commit tick), then hands back to the
bot. Used to test whether a pure-RGB VLM supervisor can DETECT (and later recover
from) the deviations. See docs/HUMAN_AGENT_DESIGN.md.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from ..core.enums import Action, Direction, SOURCE_ITEMS, Terrain, DIRECTION_VECTORS
from ..core.items import Plate
from ..core.stations import CookStation, CuttingBoard
from .heuristic import GreedyChef, _adjacent_floor, _bfs_first_step, _dir_between

_D2A = {Direction.NORTH: Action.NORTH, Direction.SOUTH: Action.SOUTH,
        Direction.EAST: Action.EAST, Direction.WEST: Action.WEST}


def find_tiles(game):
    """Locate interactive tiles in the current layout."""
    src = {}                                   # ingredient -> [tiles]
    plate = serving = None
    for y, row in enumerate(game.layout.grid):
        for x, t in enumerate(row):
            if t in SOURCE_ITEMS:
                src.setdefault(SOURCE_ITEMS[t], []).append((x, y))
            elif t == Terrain.PLATE_SOURCE:
                plate = (x, y)
            elif t == Terrain.SERVING:
                serving = (x, y)
    pots = [p for p, s in game.stations.items() if isinstance(s, CookStation)]
    boards = [p for p, s in game.stations.items() if isinstance(s, CuttingBoard)]
    return dict(src=src, plate=plate, serving=serving, pots=pots, boards=boards)


class _Seq:
    """Execute a list of ops: ('goto', tile) | ('act',) | ('wait', n). Commit = the first act."""
    def __init__(self, ops, label):
        self.ops, self.i, self.label = ops, 0, label
        self.committed = False
        self._wait_left = 0

    def step(self, game, cook) -> Tuple[int, bool]:
        if self._wait_left > 0:
            self._wait_left -= 1
            return int(Action.STAY), (self._wait_left <= 0 and self.i >= len(self.ops))
        if self.i >= len(self.ops):
            return int(Action.STAY), True
        op = self.ops[self.i]
        if op[0] == "goto":
            tile = op[1]
            stand = set(_adjacent_floor(game, *tile))
            if not stand:
                self.i += 1; return int(Action.STAY), False     # unreachable -> skip
            if cook.pos in stand:
                want = _dir_between(cook.pos, tile)
                if want is not None and cook.direction != want:
                    return int(_D2A[want]), False                # turn to face it
                self.i += 1; return int(Action.STAY), False      # arrived & facing -> next op
            a = _bfs_first_step(game, cook.pos, stand)
            if a is None:
                self.i += 1; return int(Action.STAY), False
            return int(a), False
        if op[0] == "wait":
            self.i += 1
            self._wait_left = op[1]
            return int(Action.STAY), False
        # act — commit stamps at the LAST manipulation in the sequence (the visible mistake);
        # earlier acts (e.g. grabbing a plate first) are legitimate setup, not the error.
        self.i += 1
        if not any(o[0] == "act" for o in self.ops[self.i:]):
            self.committed = True
        return int(Action.INTERACT), self.i >= len(self.ops)


class _Stall:
    def __init__(self, ticks, label="stall"):
        self.left, self.label, self.committed = ticks, label, False

    def step(self, game, cook):
        self.left -= 1
        if self.left <= 0:
            self.committed = True
            return int(Action.STAY), True
        return int(Action.STAY), False


def _holding_clean_plate(cook):
    return isinstance(cook.holding, Plate) and not cook.holding.dirty


class NoisyHuman:
    """schedule: list of dicts {type, at} where `at` is a trigger tick (the human
    has acted normally until then). Supported types: wrong_location, raw_on_plate,
    add_ingredient, stall, dawdle."""

    def __init__(self, game, recipe, schedule, seed=0):
        self.bot = GreedyChef(seed)
        self.recipe = recipe
        self.schedule = sorted(schedule, key=lambda d: d["at"])
        self.tick = 0
        self.fired = 0
        self.active: Optional[object] = None
        self.dawdle_left = 0
        self.events: List[dict] = []           # ground-truth: {type,start,commit,message}
        self.announced = ""                     # the human's spoken message, once an intentional change is declared
        import random as _r
        self.rng = _r.Random(seed + 1)

    def current_message(self) -> str:
        """What the human is verbally communicating right now (empty if nothing).
        A real person ANNOUNCES intentional changes/requests but never announces mistakes."""
        return self.announced

    # --- deviation builders (use live state to find tiles) ---
    def _build(self, spec, game, cook):
        T = find_tiles(game)
        typ = spec["type"]
        if typ == "stall":
            return _Stall(spec.get("ticks", 8))
        if typ == "dawdle":
            self.dawdle_left = spec.get("ticks", 12)
            return None                         # handled inline (no takeover)
        if typ == "wrong_location":
            # a location UNRELATED to the recipe: prefer stations the dish never touches (e.g. the
            # cutting board when nothing needs chopping), then off-recipe sources; pick the FARTHEST
            # one so the route visibly diverges, act there, then linger — real progress is lost.
            need = {n for n, _ in self.recipe.contents}
            states = {s for _, s in self.recipe.contents}
            unrelated = [tiles[0] for ing, tiles in T["src"].items() if ing not in need]
            if "chopped" not in states:
                unrelated += T["boards"]                  # board is untouched by this recipe
            wrong = unrelated or T["pots"][1:] or [t[0] for t in T["src"].values()]
            cx, cy = cook.pos
            tile = max(wrong, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy))
            return _Seq([("goto", tile), ("act",), ("wait", 6)], "wrong_location")
        if typ == "raw_on_plate":
            cooked = [n for n, s in self.recipe.contents if s == "cooked"]
            ing = cooked[0] if cooked else next(iter({n for n, _ in self.recipe.contents}))
            src = T["src"].get(ing, [T["plate"]])[0]
            ops = []
            if not _holding_clean_plate(cook):
                ops += [("goto", T["plate"]), ("act",)]
            ops += [("goto", src), ("act",)]    # scoop the RAW item onto the plate (skip cooking)
            return _Seq(ops, "raw_on_plate")
        if typ == "add_ingredient":
            need = {n for n, _ in self.recipe.contents}
            extra = [ing for ing in T["src"] if ing not in need] or list(T["src"])
            ing = self.rng.choice(extra)
            src = T["src"][ing][0]
            # an intentional plan change — the human SAYS so (this is the message channel)
            self.announced = f"let's also add a {ing} to the plate — adding it on purpose."
            ops = []
            if not _holding_clean_plate(cook):
                ops += [("goto", T["plate"]), ("act",)]
            ops += [("goto", src), ("act",)]    # scoop an OFF-PLAN extra ingredient
            return _Seq(ops, "add_ingredient")
        if typ == "wrong_order":
            # scoop the LAST required ingredient FIRST — right items, wrong sequence
            last = self.recipe.contents[-1][0]
            src = T["src"].get(last, [T["plate"]])[0]
            ops = []
            if not _holding_clean_plate(cook):
                ops += [("goto", T["plate"]), ("act",)]
            ops += [("goto", src), ("act",)]
            return _Seq(ops, "wrong_order")
        if typ == "wrong_ingredient":
            # plate an ingredient the dish does NOT contain (a foreign item, e.g. lettuce from a
            # decoy crate). Fallback if the map has no off-recipe source: two separate trips to the
            # first ingredient's source, scooping a visible duplicate each time.
            need = {n for n, _ in self.recipe.contents}
            foreign = [tiles[0] for ing, tiles in T["src"].items() if ing not in need]
            ops = []
            if not _holding_clean_plate(cook):
                ops += [("goto", T["plate"]), ("act",)]
            if foreign:
                cx, cy = cook.pos
                src = min(foreign, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy))
                ops += [("goto", src), ("act",)]          # scoop the foreign item onto the plate
            else:
                first = self.recipe.contents[0][0]
                src = T["src"].get(first, [T["plate"]])[0]
                ops += [("goto", src), ("act",), ("goto", T["plate"]), ("goto", src), ("act",)]
            return _Seq(ops, "wrong_ingredient")
        if typ == "premature_serve":
            # scoop ONE item, then rush to the serving pass and serve the incomplete plate
            first = self.recipe.contents[0][0]
            src = T["src"].get(first, [T["plate"]])[0]
            ops = []
            if not _holding_clean_plate(cook):
                ops += [("goto", T["plate"]), ("act",)]
            ops += [("goto", src), ("act",), ("goto", T["serving"]), ("act",)]
            return _Seq(ops, "premature_serve")
        if typ == "omission":
            # build only the RAW items (skip the ingredient that needs cooking), then serve — missing one
            raws = [n for n, s in self.recipe.contents if s != "cooked"]
            ops = []
            if not _holding_clean_plate(cook):
                ops += [("goto", T["plate"]), ("act",)]
            for ing in raws:
                src = T["src"].get(ing)
                if src:
                    ops += [("goto", src[0]), ("act",)]
            ops += [("goto", T["serving"]), ("act",)]   # serve the incomplete (no cooked item) plate
            return _Seq(ops, "omission")
        return None

    def __call__(self, game, cook_id):
        self.tick += 1
        cook = game.cooks[cook_id]
        # start a scheduled deviation
        if self.active is None and self.dawdle_left <= 0 and self.fired < len(self.schedule):
            spec = self.schedule[self.fired]
            if self.tick >= spec["at"]:
                self.fired += 1
                self.active = self._build(spec, game, cook)
                if self.active is not None:
                    self.events.append(dict(type=self.active.label, start=self.tick, commit=None,
                                            message=self.announced or ""))
                elif spec["type"] == "dawdle":
                    self.events.append(dict(type="dawdle", start=self.tick,
                                            commit=self.tick + self.dawdle_left))
        # run active takeover
        if self.active is not None:
            act, done = self.active.step(game, cook)
            if getattr(self.active, "committed", False) and self.events and self.events[-1]["commit"] is None:
                self.events[-1]["commit"] = self.tick
            if done:
                if self.events and self.events[-1]["commit"] is None:
                    self.events[-1]["commit"] = self.tick
                self.active = None
            return act
        # dawdle: occasionally idle but otherwise correct
        if self.dawdle_left > 0:
            self.dawdle_left -= 1
            if self.rng.random() < 0.6:
                return int(Action.STAY)
        return int(self.bot(game, cook_id))

    # Phase-2 hook: adopt a VLM correction (not used in detect-only)
    def receive_interrupt(self, command):
        pass
