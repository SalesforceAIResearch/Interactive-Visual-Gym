"""Kitchen layouts: textual parsing, JSON (de)serialization and procedural
generation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .enums import (
    CHAR_TO_TERRAIN,
    SOURCE_ITEMS,
    TERRAIN_TO_CHAR,
    Terrain,
)


@dataclass
class Layout:
    name: str
    grid: List[List[Terrain]]            # grid[y][x]
    start_positions: List[Tuple[int, int]]

    def __post_init__(self):
        # Enforced on EVERY Layout, not just freshly generated ones: the shipped grids in
        # tmp/grid.json were written before generate_layout sealed anything, so hand-written
        # and previously-saved maps carry the same unusable corner counters. Repairing here
        # means a map cannot enter the engine violating the invariant, whichever constructor
        # built it. See seal_unreachable() for what went wrong and why selection-time fixes
        # are the wrong layer.
        if self.grid:
            seal_unreachable(self.grid)
            ensure_trash(self.grid)

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def terrain_at(self, x: int, y: int) -> Terrain:
        return self.grid[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # -- (de)serialization -------------------------------------------------
    @classmethod
    def from_lines(cls, name: str, lines: List[str]) -> "Layout":
        """Parse a textual layout. Digits 1-9 mark cook spawn points (on floor)."""
        width = max(len(line) for line in lines)
        grid: List[List[Terrain]] = []
        starts: Dict[int, Tuple[int, int]] = {}
        for y, line in enumerate(lines):
            row: List[Terrain] = []
            for x in range(width):
                ch = line[x] if x < len(line) else " "
                if ch.isdigit() and ch != "0":
                    starts[int(ch)] = (x, y)
                    row.append(Terrain.FLOOR)
                else:
                    row.append(CHAR_TO_TERRAIN.get(ch, Terrain.FLOOR))
            grid.append(row)
        start_positions = [starts[k] for k in sorted(starts)]
        return cls(name, grid, start_positions)

    def to_lines(self) -> List[str]:
        start_lookup = {pos: i + 1 for i, pos in enumerate(self.start_positions)}
        lines = []
        for y, row in enumerate(self.grid):
            chars = []
            for x, t in enumerate(row):
                if (x, y) in start_lookup:
                    chars.append(str(start_lookup[(x, y)]))
                else:
                    chars.append(TERRAIN_TO_CHAR[t])
            lines.append("".join(chars))
        return lines

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        grid = [[Terrain(t) for t in row] for row in d["grid"]]
        starts = [tuple(p) for p in d.get("start_positions", [])]
        return cls(d.get("name", "custom"), grid, starts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "grid": [[t.value for t in row] for row in self.grid],
            "start_positions": [list(p) for p in self.start_positions],
        }

    def copy(self) -> "Layout":
        return Layout(self.name, [row[:] for row in self.grid], list(self.start_positions))

    # -- validation --------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty == valid)."""
        issues = []
        flat = [t for row in self.grid for t in row]
        if Terrain.SERVING not in flat:
            issues.append("No serving window (S).")
        if Terrain.PLATE_SOURCE not in flat:
            issues.append("No plate dispenser (D).")
        if not any(t in SOURCE_ITEMS for t in flat):
            issues.append("No ingredient sources.")
        if not self.start_positions:
            issues.append("No cook spawn points.")
        for x, y in self.start_positions:
            if not self.in_bounds(x, y) or self.grid[y][x] != Terrain.FLOOR:
                issues.append(f"Spawn point {(x, y)} is not on a floor tile.")
        return issues


# ----------------------------------------------------------------------------
# Procedural generation
# ----------------------------------------------------------------------------
def seal_unreachable(grid) -> List[Tuple[int, int]]:
    """Turn every interactive tile the cook can never stand next to into a WALL, in place.
    Returns what was sealed.

    A cook only ever interacts from an ORTHOGONALLY adjacent floor tile, so a counter or
    station whose four neighbours are all non-floor is not a surface — it is scenery that
    lies about being a surface. The border ring lays counters along all four walls, which
    puts one on each CORNER, and corners are diagonal-only by construction; the placement
    guard did not catch them because it excluded COUNTER from its reachability test.

    Measured on the shipped cheeseburger cell: 5 of 9 counters were unreachable. Nothing in
    the engine knew that, so free_counter() — which ranks by Manhattan distance and never
    tested reachability — kept handing them out as parking spots for the plate. The cook
    walked at a tile it could not occupy, stopped moving, and the episode burned its whole
    wall-clock budget standing still: 492 conversation turns, 129 of them about being stuck.

    Fixing it at selection time would only move the problem. A map should not offer a
    surface that cannot be used.

    ONLY counters are sealed. A stranded POT or plate dispenser is a different and worse
    problem — walling it over silently makes the recipe unsolvable, which an earlier version
    of this function did: it deleted the plate dispenser on every `divided` layout, and
    Layout.validate() went from clean to "No plate dispenser (D)". Stations are kept
    reachable at placement time instead (see the slot filter in generate_layout), and
    unreachable_tiles() still reports any that slip through so tests fail loudly.
    """
    h, w = len(grid), len(grid[0])
    sealed: List[Tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if grid[y][x] != Terrain.COUNTER:
                continue
            if not any(0 <= x + dx < w and 0 <= y + dy < h and grid[y + dy][x + dx] == Terrain.FLOOR
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                grid[y][x] = Terrain.WALL
                sealed.append((x, y))
    return sealed


def ensure_trash(grid) -> Optional[Tuple[int, int]]:
    """W11 (2026-08-12 audit): both prompt glossaries, the `trash` subgoal and every repair
    script assume a trash tile exists, but maps built with `inject=False` had none —
    `go to the trash` was rejected 9 times in one episode while the assistant kept
    prescribing it, and mis-plated dishes became unrepairable. Every map gets one: the
    reachable counter FARTHEST from the cook spawns becomes TRASH (farthest, so it never
    displaces a convenient parking counter). Returns the converted tile, or None if a
    trash already exists or no counter qualifies."""
    h, w = len(grid), len(grid[0])
    for row in grid:
        if Terrain.TRASH in row:
            return None
    spawns = [(x, y) for y in range(h) for x in range(w)
              if grid[y][x] == Terrain.FLOOR]
    cands = []
    for y in range(h):
        for x in range(w):
            if grid[y][x] != Terrain.COUNTER:
                continue
            if any(0 <= x + dx < w and 0 <= y + dy < h
                   and grid[y + dy][x + dx] == Terrain.FLOOR
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                d = min((abs(x - sx) + abs(y - sy)) for sx, sy in spawns) if spawns else 0
                cands.append((d, x, y))
    if not cands:
        return None
    _, x, y = max(cands)
    grid[y][x] = Terrain.TRASH
    return (x, y)


def unreachable_tiles(grid) -> List[Tuple[int, int]]:
    """Interactive tiles with no orthogonal floor neighbour — the invariant seal_unreachable
    establishes. Kept separate so tests and map audits can assert on it."""
    h, w = len(grid), len(grid[0])
    return [(x, y) for y in range(h) for x in range(w)
            if grid[y][x] not in (Terrain.FLOOR, Terrain.WALL)
            and not any(0 <= x + dx < w and 0 <= y + dy < h and grid[y + dy][x + dx] == Terrain.FLOOR
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]


def generate_layout(
    width: int = 9,
    height: int = 7,
    n_players: int = 2,
    seed: Optional[int] = None,
    n_pots: int = 2,
    n_cutting_boards: int = 1,
    include_sink: bool = True,
    n_pans: int = 0,
    n_ovens: int = 0,
    n_fryers: int = 0,
    ingredient_sources: Optional[List[Terrain]] = None,
    style: str = "ring",
    obstacles: int = 0,
    interior_stations: bool = False,
) -> Layout:
    """Generate a solvable kitchen layout.

    ``style`` is one of ``"ring"`` (counters line the walls with an open
    interior) or ``"divided"`` (a central counter wall splits two halves —
    the classic "forced coordination" topology).
    """
    rng = random.Random(seed)
    if ingredient_sources is None:
        ingredient_sources = [Terrain.ONION_SOURCE, Terrain.TOMATO_SOURCE]

    width = max(6, width)
    height = max(5, height)
    grid = [[Terrain.FLOOR for _ in range(width)] for _ in range(height)]

    # Border ring of counters.
    border: List[Tuple[int, int]] = []
    for x in range(width):
        for y in (0, height - 1):
            grid[y][x] = Terrain.COUNTER
            border.append((x, y))
    for y in range(height):
        for x in (0, width - 1):
            grid[y][x] = Terrain.COUNTER
            if (x, y) not in border:
                border.append((x, y))

    if style == "divided" and width >= 7:
        # A vertical counter wall through the middle with no gap forces the
        # two cooks (placed on opposite sides) to pass items across it.
        mid = width // 2
        for y in range(1, height - 1):
            grid[y][mid] = Terrain.COUNTER

    # Required stations to place on the border (impassable cells).
    required: List[Terrain] = [Terrain.SERVING, Terrain.PLATE_SOURCE]
    required += [Terrain.POT] * n_pots
    required += [Terrain.PAN] * n_pans
    required += [Terrain.OVEN] * n_ovens
    required += [Terrain.FRYER] * n_fryers
    required += [Terrain.CUTTING_BOARD] * n_cutting_boards
    if include_sink:
        required += [Terrain.SINK]
    required += list(ingredient_sources)

    # Station placement pool. Border edge-midpoints always; with interior_stations
    # we ALSO place stations on interior cells at the SAME probability as the border,
    # so the middle of the kitchen has functional islands (sources/pans/plates/…) that
    # are both navigation obstacles and grounding objects — not dead walls.
    corners = {(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)}
    border_slots = [p for p in border if p not in corners]
    interior_slots = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)]
    slots = border_slots + (interior_slots if interior_stations else [])

    # A station is only usable from an orthogonally adjacent floor tile, so a slot without one
    # cannot host a station. `divided` is the case that bites: the mid-column counter wall
    # takes the one floor neighbour away from the border tile directly above and below it, so
    # whatever landed there — the plate dispenser, in practice — was unusable.
    def _has_floor_neighbour(x, y):
        return any(0 <= x + dx < width and 0 <= y + dy < height
                   and grid[y + dy][x + dx] == Terrain.FLOOR
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    slots = [p for p in slots if _has_floor_neighbour(*p)]
    rng.shuffle(slots)

    def _floor_connected_and_reachable():
        floors = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == Terrain.FLOOR]
        if not floors:
            return False
        seen = {floors[0]}; st = [floors[0]]
        while st:
            x, y = st.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == Terrain.FLOOR and (nx, ny) not in seen:
                    seen.add((nx, ny)); st.append((nx, ny))
        if len(seen) != len(floors):
            return False
        # every non-floor station must have an adjacent reachable floor tile
        for y in range(height):
            for x in range(width):
                if grid[y][x] not in (Terrain.FLOOR, Terrain.WALL, Terrain.COUNTER):
                    if not any(0 <= x+dx < width and 0 <= y+dy < height and (x+dx, y+dy) in seen
                               for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                        return False
        return True

    for terrain in required:
        if not interior_stations:
            if not slots:
                break
            x, y = slots.pop()
            grid[y][x] = terrain
            continue
        for _ in range(len(slots)):        # interior placement: guard connectivity/reachability
            if not slots:
                break
            x, y = slots.pop()
            prev = grid[y][x]
            grid[y][x] = terrain
            if _floor_connected_and_reachable():
                break
            grid[y][x] = prev              # revert: this slot would block/disconnect

    # Optional interior obstacles: scatter WALL tiles in the open middle to force
    # navigation detours (raises map tortuosity), WITHOUT touching any station's
    # access tile or disconnecting the floor.
    if obstacles > 0:
        def _station_adjacent(x, y):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] not in (Terrain.FLOOR, Terrain.WALL):
                    return True
            return False

        def _floor_connected():
            floors = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == Terrain.FLOOR]
            if not floors:
                return False
            seen = {floors[0]}; stack = [floors[0]]
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and grid[ny][nx] == Terrain.FLOOR and (nx, ny) not in seen:
                        seen.add((nx, ny)); stack.append((nx, ny))
            return len(seen) == len(floors)

        cand = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)
                if grid[y][x] == Terrain.FLOOR and not _station_adjacent(x, y)]
        rng.shuffle(cand)
        placed = 0
        for (x, y) in cand:
            if placed >= obstacles:
                break
            grid[y][x] = Terrain.WALL
            if _floor_connected():
                placed += 1
            else:
                grid[y][x] = Terrain.FLOOR

    # No tile may claim to be interactive if the cook cannot stand beside it. Run this AFTER
    # obstacles: a wall dropped next to a counter can strand one that was fine when placed.
    seal_unreachable(grid)

    # Cook spawn points on interior floor.
    interior = [
        (x, y)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
        if grid[y][x] == Terrain.FLOOR
    ]
    rng.shuffle(interior)
    if style == "divided" and width >= 7:
        mid = width // 2
        left = [p for p in interior if p[0] < mid]
        right = [p for p in interior if p[0] > mid]
        starts = []
        for i in range(n_players):
            pool = left if i % 2 == 0 else right
            if pool:
                starts.append(pool.pop())
        start_positions = starts
    else:
        start_positions = interior[:n_players]

    name = f"procedural_{style}_{seed}"
    layout = Layout(name, grid, start_positions)
    return layout
