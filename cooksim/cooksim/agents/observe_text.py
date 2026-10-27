"""Text-only observation — the state map as TEXT, no pixels.

The whole kitchen is representable in text: an ASCII grid for layout + position, entity
tables for everything with state, the cook line, and two things pixels give for free that
text must carry explicitly: MOTION (a recent-path line) and last-command FEEDBACK.
Used by the text-only VLM-as-controller (tools/gemini_stream_text.py); results are directly
comparable to the RGB controller — same commands, same engine, different observation channel.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from ..core.enums import SOURCE_ITEMS, Terrain
from ..core.items import Ingredient, Plate
from .observe import _station_status
from .highlevel import _object_ids as station_ids   # ONE id source: the label the cook
# reads here is exactly the label ground() accepts (grounding redesign 2026-08-15). The
# old observe.station_ids covered stations only and numbered them differently, so a cook
# could read "pan 2" and have the grounder disagree about which pan that was.

Pos = Tuple[int, int]

_CH = {Terrain.WALL: "#", Terrain.FLOOR: ".", Terrain.COUNTER: "C", Terrain.POT: "O",
       Terrain.PAN: "P", Terrain.OVEN: "V", Terrain.FRYER: "F", Terrain.CUTTING_BOARD: "B",
       Terrain.SINK: "S", Terrain.SERVING: "X", Terrain.TRASH: "T", Terrain.PLATE_SOURCE: "D"}

LEGEND = ("#=wall .=floor C=counter O=pot P=pan V=oven F=fryer B=cutting-board S=sink "
          "X=serving-pass T=trash D=plate-stack @={who}; lowercase letter = ingredient source "
          "(its full name is in the tables below)")


def _held_str(cook) -> str:
    h = cook.holding
    if h is None:
        return "nothing"
    if isinstance(h, Plate):
        seq = getattr(h, "content_seq", None) or [(i.name, i.state.value) for i in h.contents]
        return "a plate [" + ", ".join(f"{n}({s})" for n, s in seq) + "]" if seq else "an empty plate"
    if isinstance(h, Ingredient):
        return f"{h.name} ({h.state.value})"
    return str(h)


def observe_text(game, feedback: Optional[str] = None, path: Optional[List[Pos]] = None,
                 viewpoint: str = "cook") -> str:
    """One observation = everything a watcher could read off the screen, as text.
    viewpoint="cook" addresses the acting agent (YOU/@); viewpoint="supervisor" renders the
    same world in third person (THE COOK) — the interaction VLM's default input channel."""
    you = viewpoint == "cook"
    # THE OBSERVER NEVER SEES IDS OR COORDINATES (v2, user ruling 2026-08-15). This tier
    # kept handing the assistant the cook's own addressing scheme — 30 coordinate pairs and
    # 8 object ids per render — long after the agent+++ readout had stopped, because the
    # no-coordinates rule was enforced on that readout only. Same rule, both tiers.
    from ..core.config import is_v1
    if not you and not is_v1():
        return _observe_text_places(game, feedback=feedback, path=path)
    grid = game.layout.grid
    cook = game.cooks[0]
    ids = station_ids(game)
    # ---- ascii map
    lines = ["   " + "".join(str(x % 10) for x in range(len(grid[0])))]
    for y, row in enumerate(grid):
        cs = []
        for x, t in enumerate(row):
            if (x, y) == tuple(cook.pos):
                cs.append("@")
            elif t in SOURCE_ITEMS:
                cs.append(SOURCE_ITEMS[t][0].lower())
            else:
                cs.append(_CH.get(t, "?"))
        lines.append(f"y{y % 10} " + "".join(cs))
    # ---- entity tables
    stations = []
    for (x, y), t in ((p, grid[p[1]][p[0]]) for p in sorted(game.stations.keys(), key=lambda p: (p[1], p[0]))):
        lab = ids.get((x, y), t.value)
        stations.append(f"- {lab} at ({x},{y}): {_station_status(game, (x, y))}")
    sources = [f"- {ids.get((x, y), SOURCE_ITEMS[t] + ' source')} at ({x},{y})"
               for y, row in enumerate(grid) for x, t in enumerate(row) if t in SOURCE_ITEMS]
    counters = [f"- {ids.get((x, y), 'counter')} at ({x},{y}) holds: {getattr(it, 'name', it)}"
                for (x, y), it in sorted(game.counter_items.items()) if it is not None]
    other = [f"- {ids.get((x, y), n)} at ({x},{y})" for n, terr in
             (("plate stack", Terrain.PLATE_SOURCE), ("serving pass", Terrain.SERVING), ("trash", Terrain.TRASH))
             for y, row in enumerate(grid) for x, t in enumerate(row) if t == terr]
    # ---- order
    order = ""
    if getattr(game.orders, "orders", None):
        r = game.orders.orders[0].recipe
        items = ", ".join(f"{n}({s})" for n, s in r.contents)
        order = f"{r.name}: {items}" + (" — MUST be plated in exactly this order" if r.ordered else " — any order")
    # ---- assemble
    out = ["KITCHEN MAP (text). " + LEGEND.format(who="you" if you else "the cook"), "\n".join(lines)]
    out.append("STATIONS:\n" + ("\n".join(stations) or "- none"))
    out.append("SOURCES:\n" + ("\n".join(sources) or "- none"))
    if counters:
        out.append("ON COUNTERS:\n" + "\n".join(counters))
    # FREE COUNTERS (2026-08-15): parking a plate is the most common thing a cook needs a
    # counter for, and counters are the one crowded kind — a maze has ~14. Only OCCUPIED
    # ones were listed, so an empty one had to be counted off the ascii grid before it
    # could be named. List the nearest free ones with their coordinates, which is the form
    # the engine wants for a kind this numerous.
    free = [(x, y) for y, row in enumerate(grid) for x, t in enumerate(row)
            if t == Terrain.COUNTER and game.counter_items.get((x, y)) is None]
    if free:
        cx, cy = tuple(cook.pos)
        near = sorted(free, key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))[:6]
        out.append("FREE COUNTERS (nearest first, name one by coordinate): "
                   + ", ".join(f"({x},{y})" for x, y in near))
    out.append("FIXED:\n" + "\n".join(other))
    facing = getattr(cook.direction, "name", str(cook.direction)).lower()
    who = "YOU" if you else "THE COOK"
    out.append(f"{who}: at ({cook.pos[0]},{cook.pos[1]}), facing {facing}, holding {_held_str(cook)}.")
    if path:
        lbl = "YOUR RECENT PATH" if you else "COOK'S RECENT PATH"
        out.append(f"{lbl} (oldest->newest): " + " -> ".join(f"({x},{y})" for x, y in path))
    if order:
        out.append(("ORDER TO COOK: " if you else "THE COOK'S ORDER: ") + order)
    if feedback and you:                      # engine feedback is the COOK'S private channel
        out.append("LAST COMMAND FEEDBACK: " + feedback)
    return "\n\n".join(out)


def valid_action_history(ro, last=None) -> str:
    """ALL of the cook's VALID high-level actions so far, oldest first — commands the engine
    accepted (done / still executing / legitimately waiting). Commands the engine REJECTED
    never happened as behavior and are NOT shown. Deliberately shows only the command strings
    + ticks: no subgoal tags (they would leak the private plan and name the injected error)."""
    lines = []
    for l in ro.log:
        if l["status"] in ("done", "running", "waiting", "wandering"):
            when = (f"t{l['t_issue']}-{l['t_done']}" if l.get("t_done") is not None
                    else f"t{l['t_issue']}- (in progress)")
            lines.append(f'  {when}: "{l["cmd"]}"')
    if last is not None:
        lines = lines[-last:]
        head = f"COOK'S LAST {len(lines)} VALID HIGH-LEVEL ACTIONS (oldest first; earlier actions NOT shown):"
    else:
        head = "COOK'S VALID HIGH-LEVEL ACTIONS SO FAR (oldest first):"
    return (head + "\n" + "\n".join(lines)) if lines else head + " (none yet)"


def _observe_text_places(game, feedback=None, path=None) -> str:
    """The text tier for an OBSERVER under v2: the same content as the cook's map, with
    every position expressed as a place phrase instead of a coordinate, and no ids. The
    ascii grid is omitted entirely — it is a coordinate system drawn in characters."""
    from .places import where_phrase
    from ..core.enums import SOURCE_ITEMS, Terrain
    grid = game.layout.grid
    cook = game.cooks[0]
    where = lambda x, y: where_phrase(game, x, y).split(" (")[0]
    kinds = {}
    for y, row in enumerate(grid):
        for x, t in enumerate(row):
            if t == Terrain.FLOOR or t == Terrain.WALL:
                continue
            name = (f"{SOURCE_ITEMS[t]} source" if t in SOURCE_ITEMS
                    else str(getattr(t, "value", t)).replace("_", " "))
            kinds.setdefault(name, [])
            reg = where(x, y)
            if reg not in kinds[name]:
                kinds[name].append(reg)
    out = ["KITCHEN (text). Every place is described by where it sits in the room."]
    out.append("LAYOUT:\n" + "\n".join(f"- {n}: {', '.join(v)}"
                                       for n, v in sorted(kinds.items())))
    stations = []
    for (x, y) in sorted(game.stations.keys(), key=lambda p: (p[1], p[0])):
        t = grid[y][x]
        label = str(getattr(t, "value", t)).replace("_", " ")
        stations.append(f"- the {label} {where(x, y)}: {_station_status(game, (x, y))}")
    out.append("STATIONS:\n" + ("\n".join(stations) or "- none"))
    counters = [f"- the counter {where(x, y)} holds: {getattr(it, 'name', it)}"
                for (x, y), it in sorted(game.counter_items.items()) if it is not None]
    if counters:
        out.append("ON COUNTERS:\n" + "\n".join(counters))
    facing = getattr(cook.direction, "name", str(cook.direction)).lower()
    out.append(f"THE COOK: {where(*tuple(cook.pos))}, facing {facing}, "
               f"holding {_held_str(cook)}.")
    if path:
        out.append("COOK'S RECENT PATH (oldest->newest): "
                   + " -> ".join(where(x, y) for x, y in path))
    if getattr(game.orders, "orders", None):
        r = game.orders.orders[0].recipe
        items = ", ".join(f"{n}({s})" for n, s in r.contents)
        out.append("THE COOK'S ORDER: " + f"{r.name}: {items}"
                   + (" — MUST be plated in exactly this order" if r.ordered else " — any order"))
    return "\n\n".join(out)
