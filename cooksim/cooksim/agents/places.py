"""Place phrases — how an observer is told WHERE something is, without coordinates.

ONE source, shared by the agent+++ readout (tools/agent_context.py) and the text-state-map
tier (cooksim/agents/observe_text.py). They rendered the same idea twice before, and the
text tier kept handing the assistant coordinates and object ids long after the readout had
stopped — the exact "one definition, one source" failure this project keeps re-learning.
"""
from __future__ import annotations


def spatial_phrases(game) -> dict:
    """tile -> a PLACE PHRASE the assistant can both read and say out loud.

    The assistant is deliberately never shown coordinates (user ruling 2026-08-15). It
    leaked them the moment it had them and an argument to win — "the cutting board at the
    bottom right (5,8)" — which is dictation, not help, and unusable to a cook that reads
    its own map. Removing the numbers from its input removes the temptation at the source:
    it now sees the room described the way it must describe it.

    The phrase is a 3x3 region of the room ("in the top-left", "on the right"), made unique
    per KIND: when two of a kind share a region they are separated by how they differ most
    ("the upper fryer in the top-left"). Coordinates never appear.
    """
    grid = game.layout.grid
    H, W = len(grid), len(grid[0]) if grid else 0
    def region(x, y):
        col = "left" if x < W / 3 else ("right" if x >= 2 * W / 3 else "middle")
        row = "top" if y < H / 3 else ("bottom" if y >= 2 * H / 3 else "middle")
        if row == "middle" and col == "middle":
            return "in the centre"
        if row == "middle":
            return f"on the {col}"
        if col == "middle":
            return f"at the {row}"
        return f"in the {row}-{col}"
    from cooksim.agents.highlevel import _object_ids
    ids = _object_ids(game)
    kind_of = {p: v.rsplit(" ", 1)[0] for p, v in ids.items()}
    groups: dict = {}
    for p, kind in kind_of.items():
        groups.setdefault((kind, region(*p)), []).append(p)
    out = {}
    for (kind, reg), tiles in groups.items():
        if len(tiles) == 1:
            out[tiles[0]] = reg
            continue
        xs = {t[0] for t in tiles}
        ys = {t[1] for t in tiles}
        vertical = (max(ys) - min(ys)) >= (max(xs) - min(xs))
        ordered = sorted(tiles, key=(lambda t: (t[1], t[0])) if vertical
                         else (lambda t: (t[0], t[1])))
        if vertical:
            words = ["upper", "lower"] if len(ordered) == 2 else \
                ["upper", "middle", "lower"] + [f"#{i}" for i in range(4, len(ordered) + 2)]
        else:
            words = ["left", "right"] if len(ordered) == 2 else \
                ["left", "middle", "right"] + [f"#{i}" for i in range(4, len(ordered) + 2)]
        for w, t in zip(words, ordered):
            out[t] = f"{reg} ({w} one)"
    return out


def where_phrase(game, x, y) -> str:
    """One tile's place phrase; falls back to the region for plain floor."""
    ph = spatial_phrases(game).get((x, y))
    if ph:
        return ph
    grid = game.layout.grid
    H, W = len(grid), len(grid[0]) if grid else 0
    col = "left" if x < W / 3 else ("right" if x >= 2 * W / 3 else "middle")
    row = "top" if y < H / 3 else ("bottom" if y >= 2 * H / 3 else "middle")
    return "in the centre" if row == col == "middle" else (
        f"on the {col}" if row == "middle" else
        f"at the {row}" if col == "middle" else f"in the {row}-{col}")


