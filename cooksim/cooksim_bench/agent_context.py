"""Shared context-building for the streaming-VLM ablation conditions.

Adds, on top of the base pure-RGB streaming controller:
  * a "human agent" per-step discrete action vocabulary (turn/move/stop/act/stay),
    tracked every game step together with the cook's location and plate state;
  * the 5 input CONDITIONS used in the ablation:
      basic     - current setup (rendered RGB)
      graph_state - swap the RGB image for a labelled top-down STATE MAP
      agent+    - RGB + the cook's per-step action history (ALL past steps)
      agent++   - RGB + per-step action history + per-step location
      agent+++  - RGB + per-step action history + per-step location + plate state (text)
  * the emphasis text (add-in-order is required; illegal actions are ignored) and
    the full issued-command history, shared by every model/agent for fairness.
"""
from __future__ import annotations
import io

from cooksim.core.enums import Action
from cooksim.core.items import Plate

CONDITIONS = ["basic", "graph_state", "agent", "agent+", "agent++", "agent+++"]

# Appended to every model's system prompt (sliding-window AND live).
EMPHASIS = (
    "CRITICAL — ADD IN ORDER: the dish is accepted ONLY if its components are placed on the plate in the EXACT "
    "order the recipe lists. The right ingredients in the WRONG order are REJECTED (the plate is dirtied and you "
    "lose points). Plan your scoop order to match the recipe exactly.\n"
    "CRITICAL — ILLEGAL ACTIONS ARE IGNORED: any action whose precondition does not hold RIGHT NOW does NOTHING and "
    "wastes the turn — e.g. a manipulation when you are not standing at and FACING the right tile, 'pick up' with "
    "full hands, 'scoop' without holding a clean plate, 'turn on' an empty station. Verify the precondition first."
)


# Explicit reminder used in the `basic` condition (which has NO feedback channel): tell the model that
# commands can be silently refused when their precondition is not met, and it must self-verify from the frames.
NO_FEEDBACK_NOTE = (
    "NO FEEDBACK: you get NO confirmation of whether a command worked. A command is SILENTLY REFUSED — it simply "
    "does nothing — whenever its precondition is not met RIGHT NOW: you are not standing at and FACING the right "
    "tile, you 'scoop' without holding a clean plate, you add ingredients in the wrong order, you 'serve' an "
    "incomplete/wrong dish, or you name a station/object that does not exist. Nothing tells you this happened, so "
    "assume nothing succeeded by default: read the frames to verify each command actually took effect, and if the "
    "cook did not do what you asked, work out the cause yourself and re-issue."
)


def dir_name(direction) -> str:
    return direction.name.lower()


def plate_in_order(holding) -> str:
    """Human-readable plate / hand state, components in their addition order."""
    if holding is None:
        return "empty-handed"
    if not isinstance(holding, Plate):
        st = getattr(getattr(holding, "state", None), "value", None)
        return f"carrying {getattr(holding, 'name', holding)}" + (f" ({st})" if st else "")
    if holding.dirty:
        return "a DIRTY plate (needs washing)"
    if not holding.contents:
        return "an empty clean plate"
    seq = getattr(holding, "content_seq", None)
    items = list(seq) if seq else [(i.name, getattr(i.state, "value", i.state)) for i in holding.contents]
    # W16 (2026-08-12 audit): a bare "[bun (raw)]" under IN THE COOK'S HANDS was textually
    # identical to a loose item — the supervisor argued "you are holding a raw tortilla,
    # not a plate" for ~100 ticks against a cook that was right. Say PLATE explicitly.
    return ("a PLATE holding, in order: ["
            + ", ".join(f"{n} ({s})" for n, s in items) + "]")


class History:
    """Per-step log of the human (cook) agent: discrete action, location, plate."""

    def __init__(self):
        self.steps = []  # each: {a, pos:(x,y), dir, plate, moved}

    def _was_moving(self) -> bool:
        return bool(self.steps) and self.steps[-1]["moved"]

    def record(self, low, pre_pos, pre_dir, post_pos, post_dir, holding):
        if low == int(Action.INTERACT):
            lab = "act"
        elif low == int(Action.STAY):
            lab = "stop" if self._was_moving() else "stay"
        else:                                    # a directional action (orient-first)
            if post_pos != pre_pos:
                lab = "move"
            elif post_dir != pre_dir:
                lab = "turn-" + dir_name(post_dir)
            else:
                lab = "stop" if self._was_moving() else "stay"   # blocked move
        self.steps.append(dict(a=lab, pos=tuple(post_pos), dir=dir_name(post_dir),
                               plate=plate_in_order(holding), moved=(post_pos != pre_pos)))


def input_note(condition: str) -> str:
    if condition == "graph_state":
        return ("INPUT: instead of a photo you receive a TOP-DOWN SCHEMATIC MAP of the kitchen. Every tile is "
                "labelled with its type and (for stations) its status (cooking / done / burnt) and an ID number; "
                "faint x,y coordinates are printed in each tile; the cook shows its number and a 'hold:' label of "
                "what it carries. Read the labels directly — do not infer from photo-realism.")
    return "INPUT: the rendered 3D frame(s) of the kitchen — read everything you need from the pixels."


def context_block(condition: str, hist: History, cook) -> str:
    """The condition-specific extra TEXT appended to the per-decision prompt."""
    out = []
    if condition in ("agent+", "agent++", "agent+++"):
        out.append("ACTION VOCAB of the cook's per-step log — turn-<dir>: re-faced without moving · move: walked one "
                   "tile · act: interacted with the faced tile · stop: was moving, now halted · stay: stayed idle.")
        if condition == "agent+":
            seq = ", ".join(s["a"] for s in hist.steps) or "(none yet)"
            out.append("YOUR COOK'S PER-STEP ACTIONS so far (oldest->newest, one token per game step): " + seq)
        else:  # agent++ / agent+++ : action + location each step
            seq = " | ".join(f'{s["a"]}@({s["pos"][0]},{s["pos"][1]}) face-{s["dir"]}' for s in hist.steps) or "(none yet)"
            out.append("YOUR COOK'S PER-STEP LOG so far (oldest->newest; action @ (x,y) facing-dir): " + seq)
    if condition == "agent+++":
        out.append("CURRENT PLATE STATE (components in their exact addition order): " + plate_in_order(cook.holding))
    return "\n".join(out)


def command_history_block(responses) -> str:
    """FULL list of high-level commands the VLM has issued (fairness: never truncated)."""
    cmds = [r for r in responses if r and r not in ("(empty / no-op)",)]
    if not cmds:
        return "Commands you have issued so far: (none yet)"
    return "Commands you have issued so far (oldest->newest, your full history):\n" + \
        "\n".join(f"  {i+1}. {c}" for i, c in enumerate(cmds))


def state_map_png(game) -> bytes:
    from cooksim.agents.observe import render_observation
    im = render_observation(game).convert("RGB")
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()


def _ing_text(c) -> str:
    """'chopped tomato' — prep state FIRST, so the state is never mistaken for a name."""
    return f"{c.get('state', '?')} {c.get('name', '?')}"


def _plate_text(item: dict) -> str:
    """A plate sitting on a surface, rendered from its to_dict() shape."""
    if item.get("dirty"):
        return "DIRTY PLATE (needs washing)"
    conts = item.get("contents") or []
    if not conts:
        return "empty clean PLATE"
    return "PLATE holding " + ", ".join(_ing_text(c) for c in conts)


def _surface_item_text(item) -> str:
    """One loose object on a board/counter. A cutting board's render `item` IS the
    ingredient dict ({"kind":"ingredient",...}) — NOT a container with `contents`."""
    if not isinstance(item, dict):
        return str(item)
    if item.get("kind") == "plate" or "dirty" in item:
        return _plate_text(item)
    return _ing_text(item)


from cooksim.agents.places import spatial_phrases, where_phrase  # noqa: E402,F401


def _WHERE(game, x, y):
    """How the assistant is told WHERE something is.

    v2: a place phrase, because it is never shown coordinates (it recited them at the cook
    the moment it had them). v1: the (x,y) pair every published page was run with.
    """
    from cooksim.core.config import is_v1
    return f"({x},{y})" if is_v1() else where_phrase(game, x, y)


def _stations_line(game) -> str:
    """One line of SURFACE ground truth for the aplus readout: every heat station,
    cutting board and counter that is holding anything, with contents, lit state,
    progress, smoke and — while smoking — the margin left before fire.

    F1 (2026-08-14 fix batch): this line used to enumerate heat stations ONLY and then
    close with "Any station not listed is empty". Both halves were wrong together. A
    cutting board's render `item` is the ingredient dict itself, so the `item.get(
    "contents")` probe returned nothing and EVERY board fell through the `continue` —
    a chopped tomato sitting on a board was invisible, and the closing sentence then
    asserted that invisibility was emptiness. Counters were never read at all. The
    supervisor's only ground-truth line was therefore actively false about two of the
    three surface kinds. Boards and counters are enumerated here, and the closing claim
    now scopes itself to what this line actually covers.
    """
    from cooksim.core.stations import CuttingBoard

    rs = game.render_state()
    live_stations = getattr(game, "stations", {}) or {}
    heat_rows, board_rows, counter_rows = [], [], []

    for d in (rs.get("stations") or []):
        x, y = d.get("x"), d.get("y")
        live = live_stations.get((x, y))
        item = d.get("item")
        # ---- cutting boards ------------------------------------------------
        if isinstance(live, CuttingBoard) or d.get("type") == "cutting_board":
            if item is not None:
                board_rows.append(f"the cutting board {_WHERE(game, x, y)}: [{_surface_item_text(item)}]")
            continue
        # ---- heat stations -------------------------------------------------
        conts = (item.get("contents") or []) if isinstance(item, dict) else (d.get("contents") or [])
        if not conts:
            continue
        # X3 (2026-08-13 audit): this line used to print the STATION's status in place of
        # each ingredient's PREP state, so a correctly chopped steak in a cold pan read
        # "steak, raw" — and the assistant, correctly briefed that pans take only chopped
        # steak, "fixed" a kitchen that was never broken. Say both, separately.
        names = "+".join(_ing_text(c) for c in conts)
        status = d.get("status") or ("cooking" if d.get("progress") else "raw")
        heat = {"raw": "LOADED, NOT turned on", "cooking": "cooking",
                "cooked": "DONE", "burnt": "BURNT"}.get(status, status)
        bits = f"the {d.get('type')} {_WHERE(game, x, y)}: [{names}] — {heat}"
        if d.get("progress") and status == "cooking":
            bits += f" {round(float(d['progress']) * 100)}%"
        if d.get("smoking"):
            # "SMOKING" alone never said how much room was left, so neither agent could
            # tell a station one tick into its smoke window from one tick short of fire.
            # Both numbers come off the station's own schedule.
            bits += " SMOKING" + _fire_margin(live)
        heat_rows.append(bits)

    # ---- counters ----------------------------------------------------------
    terr = rs.get("terrain") or []
    for o in (rs.get("objects") or []):
        x, y = o.get("x"), o.get("y")
        kind = "counter"
        if 0 <= (y or 0) < len(terr) and 0 <= (x or 0) < len(terr[y]):
            kind = terr[y][x]
        counter_rows.append(f"the {kind} {_WHERE(game, x, y)}: [{_surface_item_text(o.get('item'))}]")

    rows = heat_rows + board_rows + counter_rows
    if not rows:
        return ("every heat station, cutting board and counter is EMPTY — nothing is "
                "cooking or lit.")
    return ("; ".join(rows) + ". Heat stations, boards and counters not listed hold "
            "nothing. A station is only cooking if listed as cooking — "
            "LOADED-BUT-UNLIT food does not cook.")


def _fire_margin(live) -> str:
    """' for N ticks — fire after M more', read off the station's burn schedule."""
    try:
        timer = int(getattr(live, "timer"))
        smoke_at = int(getattr(live, "_smoke_at"))
        fire_at = int(getattr(live, "_fire_at"))
    except (TypeError, ValueError, AttributeError):
        return ""
    return f" for {max(0, timer - smoke_at)} ticks — fire after {max(0, fire_at - timer)} more"


def supervisor_state_extra(game):
    """SINGLE SOURCE for the `agent+++` observation tier.

    Four harnesses had grown their own copy of this readout, which is how the following
    bug survived: the supervisor was told "cook at (4,5) facing south" but never what sits
    at (4,5), where the trash is, or even which way y grows. It inferred the layout from
    eight low-res frames, inferred a MIRRORED one, and then spoke with false precision —
    "You are at the trash can!" three times while the cook stood at the pan (trash is at
    (5,2)), and "the serving pass on the right" five times when the pass is at (0,1) on
    the LEFT. The cook was correct every time and lost ~24 ticks arguing.

    Unanchored coordinates are worse than none. The kitchen layout is STATIC and any
    supervisor who watched this counter for a day would know it, so it is stated once,
    exactly, with the coordinate convention spelled out. Every harness now calls THIS —
    tests/agent_plus_qa.py fails the build if one grows a private copy again.
    """
    from cooksim.core.config import is_v1 as _iv
    _isv1_cached = [_iv()]
    rs = game.render_state()
    # NO COORDINATES ANYWHERE IN THIS READOUT (v2) (user ruling 2026-08-15). The assistant is a
    # person watching a kitchen; handing it (x,y) pairs gave it something to recite at the
    # cook the moment it wanted to win an argument ("the cutting board at the bottom right
    # (5,8)"), which is dictation rather than help. It sees the room described in the same
    # language it must speak: places.
    where = {}
    for y, row in enumerate(rs.get("terrain") or []):
        for x, t in enumerate(row):
            if t != "floor":
                # the LAYOUT line is orientation, not selection: give each kind the set of
                # places it occupies, without the "(upper one)" tie-breakers that only
                # matter when picking one out
                reg = _WHERE(game, x, y)
                if not _isv1_cached[0]:
                    reg = reg.split(" (")[0]          # v2: drop the tie-break suffix
                if reg not in where.setdefault(t, []):
                    where[t].append(reg)
    from cooksim.core.config import is_v1 as _isv1
    layout = ("; ".join(f"{t} at {', '.join(v)}" for t, v in sorted(where.items()))
              if _isv1() else
              "; ".join(f"{t.replace('_', ' ')}: {', '.join(v)}"
                        for t, v in sorted(where.items())))
    cook = game.cooks[0]
    # Naming the tile the cook OCCUPIES and the tile it FACES removes the lookup step
    # entirely. Handing over bare coordinates still let the supervisor announce "you are
    # heading to the trash!" while the cook stood on plain floor several tiles away — it
    # had to resolve (4,5) against a list and got it wrong. Stating it outright cannot be
    # misresolved.
    terr = rs.get("terrain") or []
    def at(x, y):
        if 0 <= y < len(terr) and 0 <= x < len(terr[y]):
            return terr[y][x]
        return "off-grid"
    cx, cy = tuple(cook.pos)
    dname = dir_name(cook.direction)
    dx, dy = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}.get(dname, (0, 0))
    fx, fy = cx + dx, cy + dy
    return ((f"KITCHEN LAYOUT (fixed for the whole episode; x grows RIGHT, y grows DOWN, "
             f"so y=0 is the TOP row): {layout}\n" if _isv1_cached[0] else
             f"KITCHEN LAYOUT (fixed for the whole episode): {layout}\n")
            + (f"CURRENT POSITION (same coordinates as the layout above): cook at "
               f"({cx},{cy}), standing on {at(cx, cy)}, facing {dname} toward "
               f"({fx},{fy}) which is {at(fx, fy)}.\n" if _isv1_cached[0] else
               f"WHERE THE COOK IS: {_WHERE(game, cx, cy)}, standing on "
               f"{at(cx, cy).replace('_', ' ')}, facing {dname} toward the "
               f"{at(fx, fy).replace('_', ' ')} {_WHERE(game, fx, fy)}.\n")
            # Label the HANDS, not "the plate". This line printed "CURRENT PLATE: carrying
            # tomato (raw)" while the cook held a loose ingredient and no plate at all, and the
            # supervisor read the label rather than the value: it insisted "the system state
            # explicitly confirms you are carrying a plate with a raw tomato on it" for 120
            # ticks while the cook — which was right — kept saying it held only a tomato.
            + f"IN THE COOK'S HANDS: {plate_in_order(cook.holding)}"
            + ("" if isinstance(cook.holding, Plate) else
               " (this is a LOOSE ITEM, not a plate — there is no plate in the cook's hands)")
            + ("\nCURRENT PLATE (components in the exact order added): "
               f"{plate_in_order(cook.holding)}" if isinstance(cook.holding, Plate) else "")
            # W13 (2026-08-12 audit): without station state the supervisor could neither
            # verify nor refute any stove claim — it invented fires in pots that were
            # never lit in 4 audited episodes. This line is the ground truth it lacked.
            + "\nSTATIONS RIGHT NOW: " + _stations_line(game))
