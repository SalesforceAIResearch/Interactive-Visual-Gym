"""System rules — the ONLY writer of user state (design doc §3-4).

Operates on a duck-typed rollout `ro` (MultiRollout): .cmds = [(cmd, sid)], .i = cursor,
.dev_sids = deviated subgoal ids, .log, ._idle_since, .game. All mutations of the human's
plan/actions go through these functions; the human VLM and the supervisor never touch them.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from ..core.plan_ir import (segment_plan, prep_of, trash_node, park_plate_node,
                             retrieve_plate_node, _station_label)
from .state import UserState, Profile

FIX = "vlm_fix"          # sid prefix for spliced repair/restore commands


# ===================================================================== grounding / attribution
def grounded_block(name: str, recipe) -> Optional[Tuple[str, List[str]]]:
    """subgoal name -> (sid, command block). Template subgoals resolve through segment_plan
    (the same expansion the plan itself uses); repair subgoals through the node factories."""
    name = name.strip()
    for n in segment_plan(recipe):
        if n.sid == name:
            return n.sid, list(n.commands)
    if name.startswith("trash"):        # trash / trash_held / trash(plate) / trash the plate...
        n = trash_node()
        return n.sid, list(n.commands)
    if name.startswith("prep(") and name.endswith(")"):
        comp = name[5:-1]
        n = prep_of(comp)
        return n.sid, list(n.commands)
    if name.startswith("assemble(") and name.endswith(")"):
        comp = name[9:-1]
        state = dict((c, s) for c, s in recipe.contents).get(comp)
        if state == "cooked":
            return name, [f"go to the {recipe_station_label(recipe, comp)} with the {comp}", "scoop"]
        return name, [f"go to the {comp} source", "scoop"]
    return None


def attribute_action(cmd: str, recipe) -> Optional[Tuple[str, List[str]]]:
    """One literal action -> the template subgoal that owns it (high solvability only).
    Attribution is only sound when the action is DISTINCTIVE — generic verbs like "scoop"
    appear in EVERY assemble block, and attributing them to the first match re-executes an
    arbitrary subgoal (the round-3 double-bun bug). Exactly-one-owner or nothing."""
    owners = [n for n in segment_plan(recipe) if cmd.strip() in n.commands]
    if len(owners) == 1:
        return owners[0].sid, list(owners[0].commands)
    return None


# ===================================================================== the profile filter
_MANIP_NOUN = None


_GOTO_RE = __import__("re").compile(
    r"^(?:go|walk|move|head|come|proceed|get|run)(?:\s+(?:over|back|straight|on|down|up|left|right|across|around))*"
    r"\s+to\s+(.+)$")
_MANIP_RE = __import__("re").compile(
    r"^(scoop|pick up|grab|take|put down|set down|place|drop|chop|cut|turn on|start|serve|trash|dump|put)\b\s*(.*)$")
# decorative tails a person would ignore; they break naive noun extraction
_TAIL_RE = __import__("re").compile(
    r"\s+(onto|into|in|on|to|from|off of|off|out of|inside|for|and then|then|so|to cook|to it|it|"
    r"your plate|the plate|a plate|the pan|the pot|the oven|the board|the cutting board)\b.*$")


def clean_action(a: str) -> str:
    """Normalise a human-parsed action string into something the executor can ground:
    directional 'walk/move/head ... to X' -> 'go to X'; strip decorative tails off
    manipulation verbs ('scoop the meat onto your plate' -> 'scoop the meat')."""
    import re as _re
    a = (a or "").strip().rstrip(".!").strip()
    if not a:
        return a
    low = a.lower()
    m = _GOTO_RE.match(low)
    if m:                                        # any directional go-verb -> canonical "go to"
        dest = m.group(1).strip()
        # a couple of common landmark rewrites the grounder already knows
        dest = _re.sub(r"^(a|an|the)\s+", "the ", dest)
        return "go to " + dest
    m = _MANIP_RE.match(low)
    if m:                                        # manipulation: keep verb + clean object noun
        verb, obj = m.group(1), _TAIL_RE.sub("", m.group(2)).strip()
        verb = {"cut": "chop", "start": "turn on", "set down": "put down",
                "place": "put down", "drop": "put down", "put": "put down",
                "dump": "trash"}.get(verb, verb)
        if verb in ("put down", "turn on", "chop", "trash"):
            obj = ""                              # these engine verbs take NO object phrase
        obj = _re.sub(r"^(a|an|the)\s+", "", obj).strip()
        return (verb + " " + obj).strip() if obj else verb
    return a


def _nav_prefix_for(action: str, recipe, game):
    """'scoop the meat' -> 'go to the pan with the meat' (a station holds it) or
    'go to the meat source'. Only fires for manipulation verbs that NAME an object and only
    when the object resolves; bare verbs ('scoop') stay as-is."""
    import re as _re
    if game is None:
        return None
    a = action.strip().lower()
    if a.startswith("trash"):                       # "trash the plate" -> walk to the trash first
        return "go to the trash"
    if a.startswith("put") and ("plate" in a or "counter" in a
                                 or a.strip() in ("put down", "put it down")):
        # DISTINGUISH by what's held: putting a PLATE down means "park it" -> walk to the nearest
        # empty counter. But putting an INGREDIENT down (meat into a pan, item onto a board) is a
        # placement AT the station the cook is already at -> do it directly, no counter detour
        # (routing "put the meat in the pan" to a counter stranded the meat and spun the cook).
        from ..core.items import Plate
        held = game.cooks[0].holding
        if not isinstance(held, Plate):
            return None                                 # ingredient: place it where you stand
        spot = free_counter(game)
        return f"go to the counter ({spot[0]},{spot[1]})" if spot else None
    if a.startswith("serve") and len(a) > 6:        # "serve the dish" already handled as dual
        return "go to the serving pass"
    m = _re.match(r"^(scoop|pick up|grab|take)\s+(?:the\s+)?([a-z]+)\b", a)
    if not m:
        return None
    noun = m.group(2).strip()
    if noun in ("plate", "plates", "a plate"):
        return "go to the plates"
    from ..core.stations import CookStation
    for (x, y), st in game.stations.items():
        for it in getattr(st, "contents", []) or []:
            if it.name == noun:
                lab = "pan" if isinstance(st, CookStation) else "board"
                return f"go to the {lab} with the {noun}"
        if getattr(st, "item", None) is not None and st.item.name == noun:
            return f"go to the board with the {noun}"
    from ..agents.highlevel import parse as _p, ground as _g
    it = _p(f"go to the {noun} source")
    if it is not None and _g(it, game, game.cooks[0]) is not None:
        return f"go to the {noun} source"
    return None


def free_counter(game):
    """Nearest empty counter tile — the parking spot for a held plate during a repair."""
    from ..core.enums import Terrain
    cook = game.cooks[0]
    best, bd = None, 10**9
    for y, row in enumerate(game.layout.grid):
        for x, t in enumerate(row):
            if t == Terrain.COUNTER and game.counter_items.get((x, y)) is None:
                d = abs(x - cook.pos[0]) + abs(y - cook.pos[1])
                if d < bd:
                    best, bd = (x, y), d
    return best


def _with_hands_free(cmds, sid, game):
    """Prep needs empty hands. Wrap the block in park -> do -> retrieve (one block, so template
    ordering stays intact).

    Any held item blocks a pick-up, not just a plate: the cook can be carrying a raw ingredient
    when a repair asks it to prep something else, and the bare block's "pick up" is then
    illegal-skipped with everything after it failing. The PLATE is retrieved afterwards because
    the cook needs it to plate into; a parked ingredient is left where it was put, since coming
    back for it would just re-fill the hands the next subgoal needs."""
    if game is None or game.cooks[0].holding is None:
        return cmds
    spot = free_counter(game)
    if spot is None:
        return cmds
    if not plate_held(game):
        park = park_plate_node(f"{spot[0]},{spot[1]}")
        return list(park.commands) + cmds
    park, retr = park_plate_node(f"{spot[0]},{spot[1]}"), retrieve_plate_node(f"{spot[0]},{spot[1]}")
    return list(park.commands) + cmds + list(retr.commands)


# ===================================================================== prerequisites
def required_state(recipe, comp: str) -> Optional[str]:
    """The state the recipe wants `comp` in — "raw" / "chopped" / "cooked"."""
    return dict((c, s) for c, s in recipe.contents).get(comp)


def _sname(v) -> str:
    """PrepState.COOKED / "cooked" / None -> "cooked" / ""."""
    return str(getattr(v, "value", v) or "").lower()


def _has_source(game, comp: str) -> bool:
    """Can this kitchen produce a fresh `comp` at all (is there a dispenser for it)?

    Asks the MAP, not the grounder. It used to route through ground("go to the <comp>
    source"), which conflated two different questions: "does a dispenser exist" and "can
    the cook name one right now". Once grounding turned strict (user ruling 2026-08-15) a
    kitchen with TWO potato crates answered "no potato source in this kitchen" — the
    refusal to guess between them was read as absence, and every assemble that depended on
    the ingredient became unsatisfiable.
    """
    if game is None:
        return True
    try:
        from ..core.enums import SOURCE_ITEMS
        return any(SOURCE_ITEMS.get(t) == comp
                   for row in game.layout.grid for t in row)
    except Exception:
        return False


def _named(item):
    return item.get("name") if isinstance(item, dict) else getattr(item, "name", None)


def _stated(item):
    return _sname(item.get("state") if isinstance(item, dict) else getattr(item, "state", None))


def current_goal(game, base=None):
    """THE dish the kitchen must now deliver: the open order's recipe, which carries every plan change the cook has
    announced (lang_events.extend_goal rewrites the order and the recipe book, never UserState.recipe — the public
    goal the prompts show beside `effective_goal_contents`). `base` when there is no open order.

    E6 (round-2 review, 2026-09-19): the question truths ("what is my next step?", "what still has to go on this
    plate?"), the plating anchors ("before serve"), the takeover blocks and the carried-out check all read the
    static recipe, so an announced extra cheese was missing from every truth, add_foreign(before serve) fired on the
    extra item's slot, and an order served without its announced topping was recorded as carried out. Every one of
    them now reads this function."""
    try:
        orders = list(getattr(getattr(game, "orders", None), "orders", None) or [])
        if orders and getattr(orders[0], "recipe", None) is not None:
            return orders[0].recipe
    except Exception:                                     # noqa: BLE001
        pass
    return base


def stations_holding(game, comp: str):
    """EVERY station holding a (non-burnt) `comp`, as [(station, render state)], most advanced first: a done
    heat station, then one still cooking, then a board whose chop is finished, then the rest (a cold station, a raw
    item on a board). Ties keep the map's order.

    E7 (round-2 review, 2026-09-19): the single-station version returned the FIRST station in dict order that held
    any unit, so a chopped potato waiting on a board hid the cooked one in the fryer (expert hard_5 c t141: a raw
    potato was scooped from the crate while the cooked one sat in the fryer)."""
    out = []
    for k, st in enumerate(game.stations.values()):
        d = st.render_state()
        hit = any(_named(c) == comp and _stated(c) != "burnt" for c in (d.get("contents") or []))
        it = d.get("item")
        hit = hit or (it is not None and _named(it) == comp and _stated(it) != "burnt")
        if not hit:
            continue
        status = d.get("status")
        if status == "cooked":
            rank = 0
        elif status == "cooking" or bool(getattr(st, "cooking", False)):
            rank = 1
        elif it is not None and (d.get("done") or _stated(it) == "chopped"):
            rank = 2
        else:
            rank = 3
        out.append((rank, k, st, d))
    out.sort(key=lambda x: (x[0], x[1]))
    return [(st, d) for _r, _k, st, d in out]


def station_holding(game, comp: str):
    """The station that already holds a (non-burnt) `comp`, with its render state, or None — the MOST ADVANCED
    unit when several stations hold one (see stations_holding)."""
    hs = stations_holding(game, comp)
    return hs[0] if hs else None


def recipe_station_label(recipe, comp: str, game=None) -> str:
    """The heat station label THIS recipe cooks `comp` in ("fryer" for Fish & Chips' fish), not the ingredient's
    global default (plan_ir._station_label said "pan" for fish, so the plating skill told the cook to "go to the pan
    with the fish" while the fish sat in a fryer — E12, hard review: 6/10 baseline fish left unlit 120-380 ticks).
    A station that already holds the item wins (that is where it is)."""
    if game is not None:
        try:
            from ..core.stations import CookStation
            from .. core.generator import _STATION_LABEL as _SL
            for (x, y), st in game.stations.items():
                if isinstance(st, CookStation) and any(getattr(c, "name", None) == comp for c in (st.contents or [])):
                    lab = _SL.get(game.layout.grid[y][x])
                    if lab:
                        return lab
        except Exception:                                 # noqa: BLE001
            pass
    try:
        from ..core.generator import cook_station_for, _STATION_LABEL as _SL
        return _SL[cook_station_for(getattr(recipe, "id", None), comp)]
    except Exception:                                     # noqa: BLE001
        return _station_label(comp)


def recipe_prep_block(recipe, comp: str, lit: bool = True) -> List[str]:
    """One unit of `comp` prepared the way THIS recipe's own plan does it — chopped first when the station only
    takes it chopped (steak strips, chips), loaded into the recipe's station — and, when `lit`, switched on.
    prep_of() is the recipe-blind factory (raw steak straight into "the pan": refused by the pan)."""
    node = next((n for n in segment_plan(recipe) if n.type == "prep" and n.component == comp), None) if recipe is not None else None
    if node is None:
        return list(prep_of(comp, recipe_station_label(recipe, comp)).commands)
    cmds = list(node.commands)
    if lit and _sname(required_state(recipe, comp)) == "cooked" and "turn on" not in cmds:
        cmds.append("turn on")                            # the late-ignite template loads only
    return cmds


def missing_step(game, recipe, comp: str):
    """What stands between this kitchen and a scoopable `comp`.

    Returns None when the bare assemble block will work, or the grounded commands that must run
    first. This is deliberately finer-grained than "is it there or not", because the plan shapes
    produce three genuinely different situations:

      absent            -> fetch, load, light, wait      (full prep)
      loaded, unlit     -> light it and wait             (LATE-IGNITE recipes park the ingredient
                           in a cold station between prep(x) and ignite(x); re-fetching there
                           wastes the ingredient and can overflow the station's capacity)
      lit, still going  -> wait                          (take_into_plate refuses any station
                           whose status is not "cooked", so scooping early simply fails)

    Returns None (nothing needed), a list of commands, or a STRING explaining why it cannot be
    done — never a block that will not run. Raises no opinion about RAW components: they come
    from an infinite dispenser.
    """
    want = _sname(required_state(recipe, comp))
    if want in ("", "raw"):
        return None if _has_source(game, comp) else f"there is no {comp} source in this kitchen"
    if game is None:
        return None
    label = recipe_station_label(recipe, comp, game) if want == "cooked" else "cutting board"
    wait = f"wait until the {label} with the {comp} is {want}"
    waits = [wait] if want == "cooked" else []          # a chop is done when the chop command is
    found = station_holding(game, comp)
    if found is None:
        if not _has_source(game, comp):
            return f"there is no {comp} left and no {comp} source to make one from"
        if plate_held(game) and free_counter(game) is None:
            # Prep needs empty hands and there is nowhere to put the plate down. _with_hands_free
            # silently returns the UNWRAPPED block in this case, which the cook then cannot run:
            # the pick-up is illegal-skipped and the rest fails/rejects, leaving the plate empty
            # and the subgoal still outstanding. Say it instead.
            return (f"to cook a new {comp} you need both hands, but you are holding the plate "
                    f"and every counter is occupied — clear a counter or serve first")
        return _with_hands_free(recipe_prep_block(recipe, comp) + waits, f"prep({comp})", game)
    st, d = found
    status = d.get("status")
    if status == "cooked" or (d.get("done") and want == "chopped"):
        return None
    if _stated((d.get("contents") or [None])[0] or d.get("item")) == want:
        return None
    # present but unfinished: finish it where it stands rather than starting over — at the station that HOLDS it
    if d.get("item") is not None:                       # a unit on a cutting board
        tail = [] if (d.get("done") or _stated(d.get("item")) == "chopped") else ["chop"]
        if want == "cooked":
            tail += ["pick up", f"go to the empty {label}", "put down", "turn on", wait]
        return ([f"go to the cutting board with the {comp}"] + tail) if tail else None
    if status == "cooking" or (0 < (d.get("progress") or 0) < 1):
        return [wait]
    return [f"go to the {label} with the {comp}", "turn on", wait]


def ingredient_ready(game, comp: str, want: Optional[str]) -> bool:
    """Will the BARE assemble block work right now — "go to the pan with the X / scoop"?

    Strictly: only when a `comp` is already in the wanted state. An ingredient that is still
    cooking is NOT ready, because CookStation.take_into_plate refuses any station whose status
    is not "cooked" and the scoop simply fails. missing_step() is the one that says what to do
    about it (wait / light it / start over); this only answers whether anything is needed.
    """
    from ..core.enums import PrepState
    from ..core.items import Plate
    want = _sname(want) or PrepState.RAW.value
    if want == PrepState.RAW.value:
        return _has_source(game, comp)
    if game is None:
        return True
    # E7: every station holding a unit, not the first one found
    for st, d in stations_holding(game, comp):
        if d.get("status") == "cooked" and want == "cooked":
            return True
        if want == "chopped" and (d.get("done") or _stated(d.get("item")) == "chopped"):
            return True
        for c in (d.get("contents") or []):
            if _named(c) == comp and _stated(c) == want:
                return True
    for it in list(game.counter_items.values()) + [game.cooks[0].holding]:
        if it is None or isinstance(it, Plate):
            continue
        if _named(it) == comp and _stated(it) == want:
            return True
    return False


def plate_into_hand(game):
    """Commands that put THE plate in the cook's hands, [] if it is already there, or a string
    explaining why it cannot be. Assembling scoops into a held plate — with empty hands the
    scoop just fails, and the subgoal never retires."""
    from ..core.items import Plate
    cook = game.cooks[0]
    if isinstance(cook.holding, Plate):
        return []
    pre = []
    if cook.holding is not None:                       # carrying something else: park it
        spot = free_counter(game)
        if spot is None:
            return ("your hands are full, the plate is not in them, and every counter is "
                    "occupied — use or trash what you are holding first")
        pre = [f"go to the counter ({spot[0]},{spot[1]})", "put down"]
    for (x, y), it in game.counter_items.items():      # a parked plate beats a fresh one
        if isinstance(it, Plate):
            return pre + [f"go to the counter ({x},{y})", "pick up"]
    return pre + ["go to the plates", "pick up"]


def prerequisite_block(game, recipe, comp: str) -> Optional[List[str]]:
    """Grounded commands that bring `comp` into the state the recipe wants, or None if this
    kitchen cannot produce one at all.

    Two things the bare prep template leaves out and a real cook would not: prep needs empty
    hands, so a held plate is parked and retrieved around it; and cooking takes time, so the
    block waits for the station instead of walking up to a pan that has not finished and
    scooping nothing."""
    if not _has_source(game, comp):
        return None
    want = _sname(required_state(recipe, comp))
    cmds = recipe_prep_block(recipe, comp)
    if want in ("cooked", "chopped"):
        # Target the STATION, not the ingredient. "wait until the meat is cooked" parses fine
        # but grounds to the meat SOURCE, whose status is never "cooked" — so the wait never
        # returns and the patty sitting on the heat behind it burns. highlevel._state_match
        # reads "cooked"/"chopped" off the station, so name the station.
        if want == "cooked":                            # a chop is finished by its own command
            cmds.append(f"wait until the {recipe_station_label(recipe, comp, game)} with the {comp} is {want}")
    return _with_hands_free(cmds, f"prep({comp})", game)


def convert_parse(parse: dict, profile: Profile, recipe, game=None, auto_resolve: bool = True,
                  no_prefix: bool = False) -> dict:
    """Deterministic persona/ability filter (design doc §3 + §7.4): acceptance cap in mention
    order, then solvability conversion, producing the executable command list. `game` lets the
    conversion add the hands-free choreography a capable human would do implicitly.

    auto_resolve=False [v4]: do NOT silently expand a high-ability human's words into whole
    subgoal blocks. The capable human now resolves steps ITSELF via engine_call(), so the queue
    must contain exactly what it decided — engine-injected steps it never authored were what
    drove the replace-thrash. Literal grounding (nav prefixes) still applies."""
    ALIASES = {"dump": "trash", "dump it": "trash", "throw it away": "trash",
               "throw away": "trash", "grab": "pick up", "grab it": "pick up",
               "start cooking": "turn on", "cook it": "turn on"}

    def _coerce(v):
        if isinstance(v, dict):                      # tolerate {"action": "..."} style elements
            v = v.get("action") or v.get("subgoal") or v.get("cmd") or v.get("name") or ""
        from ..agents.highlevel import strip_grounding_reminder
        v = strip_grounding_reminder(str(v).strip())   # a pasted skill step keeps no reminder text
        return ALIASES.get(v.lower(), v)   # human prompt now yields engine-valid commands; no rewrite
    elements = ([("subgoal", _coerce(s)) for s in parse.get("suggested_subgoals") or []] +
                [("action", _coerce(a)) for a in parse.get("suggested_actions") or []])
    elements = [(k, v) for k, v in elements if v]
    kept, dropped = elements, []
    if profile.information_acceptance == "low":
        kept, dropped = elements[:3], elements[3:]
    def _grounds(cmd):
        if game is None or not cmd.startswith("go to"):
            return True
        from ..agents.highlevel import parse as _p, ground as _g
        it = _p(cmd)
        return it is not None and _g(it, game, game.cooks[0]) is not None

    commands, unusable, seen_sids = [], [], set()
    solv = profile.task_solvability
    for _i, (kind, val) in enumerate(kept):
        prev_val = kept[_i - 1][1] if _i > 0 else ""    # what the human said right before this
        if kind == "subgoal":
            if val.strip().lower().startswith("trash"):
                # trashing, like serving, is universal knowledge — dual action/subgoal at
                # EVERY solvability level (a novice understands "throw it away")
                g = grounded_block("trash", recipe)
                if g and g[0] not in seen_sids:
                    seen_sids.add(g[0])
                    commands += [(c, f"{FIX}:trash_held") for c in g[1]]
                continue
            if solv == "low" or not auto_resolve:
                # low ability: can't deduce actions from a goal. auto_resolve=False: a capable
                # human must resolve the subgoal EXPLICITLY (engine_call) so the queue is its own.
                unusable.append(val)
                continue
            g = grounded_block(val, recipe)
            if g is None:
                unusable.append(val)
                continue
            sid, cmds = g
            if cmds and not _grounds(cmds[0]):
                unusable.append(val)                     # names a thing this map doesn't have
                continue
            if sid.startswith("prep("):
                cmds = _with_hands_free(cmds, sid, game)
            if sid not in seen_sids:
                seen_sids.add(sid)
                commands += [(c, f"{FIX}:{sid}") for c in cmds]
        else:                                            # literal action
            if val.strip() in ("serve", "serve it", "serve the dish"):
                # "serve" is BOTH an action and a subgoal: any human told "serve" knows to
                # walk to the pass — grounding it is public knowledge, not solvability
                g = grounded_block("serve", recipe)
                if g and g[0] not in seen_sids:
                    seen_sids.add(g[0])
                    commands += [(c, f"{FIX}:serve") for c in g[1]]
                continue
            if solv == "high" and auto_resolve:
                att = attribute_action(val, recipe)
                if att is not None:                      # deduce + complete the whole subgoal
                    sid, cmds = att
                    if sid not in seen_sids:
                        seen_sids.add(sid)
                        commands += [(c, f"{FIX}:{sid}") for c in cmds]
                    continue
            # a bare "go to the <ingredient>" while a station HOLDS that ingredient means the
            # station, not the raw source — a person checking the meat they are cooking goes
            # to the pan ("go to the meat" would otherwise ground to the meat source)
            import re as _re2
            mgo = _re2.match(r"^(?:go to|check(?: on)?)\s+(?:the\s+)?([a-z]+)\s*$", val.strip().lower())
            if mgo and game is not None:
                from ..core.stations import CookStation
                noun = mgo.group(1)
                for st in game.stations.values():
                    if any(i.name == noun for i in getattr(st, "contents", []) or []):
                        val = f"go to the pan with the {noun}" if isinstance(st, CookStation)                               else f"go to the board with the {noun}"
                        break
            # UNIVERSAL AFFORDANCE (all solvability levels): a person told "scoop the meat"
            # walks to the thing first — manipulations act on the FACED tile, so a bare
            # manipulation naming an object gets its navigation prefixed, resolved by content
            pre = None if no_prefix else _nav_prefix_for(val, recipe, game)
            # CONTEXT GUARD: a bare "put down" defaults to "park it on a counter", but if the human
            # JUST navigated somewhere ("go to the empty pan" -> "put down"), the put-down belongs
            # THERE — don't yank the cook off to a counter (that stranded the meat off the pan and
            # spun the cook for ~45 ticks). Suppress the counter-prefix when a goto precedes it.
            if (pre and pre.startswith("go to the counter")
                    and val.strip().lower() in ("put down", "put it down")
                    and prev_val.strip().lower().startswith("go to")):
                pre = None
            if pre:
                commands.append((pre, f"{FIX}:literal"))
            commands.append((val, f"{FIX}:literal"))
    return {"kept": [v for _, v in kept], "dropped_by_acceptance": [v for _, v in dropped],
            "unusable_for_solvability": unusable, "commands": commands,
            "anything_applicable": bool(commands)}


# ===================================================================== error-correction gate
def _required_sids(records) -> List[str]:
    """The subgoals that must still HAPPEN for the plan to be whole again, per record."""
    req = []
    for r in records:
        if r.skipped:
            continue
        d = r.detail or {}
        if r.transform == "skip":
            req.append(r.anchor)
        elif r.transform == "replace":
            if d.get("dropped_prep"):          # serve_raw also skipped the prep — restore it
                req.append(d["dropped_prep"])
            req.append(d.get("replaced") or r.anchor)
        elif r.transform == "truncate_and_serve":
            req += list(d.get("dropped", [])) + ["serve"]
        elif r.transform == "forget_rest":
            req += list(d.get("restorable_tail", []))
        elif r.transform == "swap":
            req += [s.strip() for s in (r.anchor or "").split("<->") if s.strip()] or []
    return req


def enter_error_correction(ro, state: UserState) -> dict:
    """[v2] The human was CONVINCED (it checked the supervisor's claim against its own past
    records and accepted — checks.flag_is_correct is scoring-only and no longer gates entry):
    clear the deviated parts of the future, stash the clean remainder, freeze. Re-entrant
    entry while already correcting is ignored (a duplicate must not wipe an in-flight
    repair). Returns the record."""
    if state.mode == "error_correction":
        return state.correction
    future = ro.cmds[ro.i + 1:]
    tail = [(c, s) for c, s in future if s not in ro.dev_sids]
    ro.cmds[ro.i + 1:] = []                    # freeze: nothing queued
    cur_dev = ro.cmds[ro.i][1] in ro.dev_sids if 0 <= ro.i < len(ro.cmds) else False
    if cur_dev:
        if ro.ctrl.skill is not None:
            ro.ctrl.skill = None               # abort a deviated command mid-execution
        if getattr(ro, "_wander", 0):
            ro._wander = 0                     # abort a deviated stall/idle (the forget freeze)
            if ro.log:
                ro.log[-1].update(status="done", t_done=ro.tick)
        if getattr(ro, "_wait", 0):
            ro._wait = 0                       # abort a deviated pending manipulation
    ro._idle_since = None
    state.pending_tail = tail
    state.mode = "error_correction"
    state.correction = {"required": _required_sids(state.records), "entered_at": ro.tick,
                        "fix_applied": False}
    return state.correction


def self_repair_plan(state: UserState, game) -> List[Tuple[str, str]]:
    """HIGH solvability: a validated bare flag is enough — the system generates the repair
    as if the user deduced it (design doc §3). Restore missing subgoals; if the plate is
    physically wrong, trash + redo from the template."""
    cmds: List[Tuple[str, str]] = []
    if not plate_ok(game, state.recipe, state.goal_contents()):
        cmds += [(c, f"{FIX}:trash_held") for c in trash_node().commands]
        import dataclasses as _dc
        from ..core.stations import CookStation
        eff = _dc.replace(state.recipe, contents=tuple(map(tuple, state.goal_contents())))
        already_cooked = {i.name for st in game.stations.values()
                          if isinstance(st, CookStation) and st.status in ("cooking", "cooked")
                          for i in st.contents}
        for n in segment_plan(eff):
            if n.sid == "fetch_plate":         # plate already in hand
                continue
            if n.sid.startswith("prep(") and n.component in already_cooked:
                continue                       # its item is already in a pan — just scoop later
            block = list(n.commands)
            if n.sid.startswith("prep("):
                block = _with_hands_free(block, n.sid, game)
            cmds += [(c, f"{FIX}:{n.sid}") for c in block]
        return cmds
    # correction may not be entered yet (the human can call full_repair_plan while still just
    # THINKING, before conviction) — fall back to computing the required sids from the records.
    required = state.correction["required"] if state.correction else _required_sids(state.records)
    for sid in required:
        g = grounded_block(sid, state.recipe)
        if g:
            cmds += [(c, f"{FIX}:{g[0]}") for c in g[1]]
    if unattended_hot_station(game):
        comp = unattended_hot_station(game)
        cmds += [(f"go to the {recipe_station_label(state.recipe, comp, game)} with the {comp}", f"{FIX}:rescue"),
                 ("scoop", f"{FIX}:rescue")] if plate_held(game) else []
    return cmds


# ============================================ engine functions the HUMAN may call (v4 design)
# "Cleverness" used to be implemented TWICE and incoherently: a persona card at the LLM layer,
# and a task_solvability flag that made the ENGINE silently expand/force-feed the plan. The
# human then saw queue steps it never authored and kept "correcting" them -> replace thrash.
# Now the resolvers are exposed as functions the HIGH-ability human CALLS ITSELF. They are its
# own skills; the queue only ever contains what the human explicitly decided. The supervisor
# has NO such channel — it never talks to the engine.
def _mark_multi_instance(actions, game):
    """Append the grounding reminder to every step that names a kind this map has several of."""
    from ..agents.highlevel import GROUNDING_REMINDER, needs_grounding_reminder
    if game is None:
        return list(actions)
    return [f"{a} {GROUNDING_REMINDER}" if isinstance(a, str) and needs_grounding_reminder(a, game) else a
            for a in actions]


def engine_call(fn: str, args: dict, state: UserState, game) -> dict:
    """Dispatch one human-issued engine call. Returns {"ok": True, "result": ...} or
    {"ok": False, "error": "<what was wrong with the input>"} — malformed input is explained
    so the human can retry. Calling these costs NO game time (it is the cook thinking)."""
    args = args if isinstance(args, dict) else {}
    if fn == "resolve_subgoal":
        name = args.get("subgoal")
        if not isinstance(name, str) or not name.strip():
            return {"ok": False, "error": 'resolve_subgoal needs args {"subgoal": "<name>"}, '
                                          'e.g. {"subgoal": "assemble(cheese)"}'}
        if name.strip().lower().replace(" ", "_") in ("park_plate", "place_plate", "put_down_plate"):
            # park_plate is grounded against the LIVE kitchen, not the recipe template: it needs
            # the nearest counter that is empty RIGHT NOW.
            if game is None:
                return {"ok": False, "error": "park_plate needs the live kitchen and it is not "
                                              "available right now."}
            from ..core.items import Plate
            if not isinstance(game.cooks[0].holding, Plate):
                return {"ok": False, "error": "you are not holding a plate — nothing to park."}
            spot = free_counter(game)
            if spot is None:
                return {"ok": False, "error": "no empty counter is free right now."}
            return {"ok": True, "result": {"subgoal": "park_plate",
                                           "actions": [f"go to the counter ({spot[0]},{spot[1]})",
                                                       "put down"]}}
        g = grounded_block(name, state.recipe)
        if g is None:
            known = [n.sid for n in segment_plan(state.recipe)] + ["park_plate"]
            return {"ok": False, "error": f"no subgoal named {name!r}. Known subgoals: {known}"}
        sid, actions = g[0], list(g[1])
        # PREREQUISITE CLOSURE. grounded_block answers from the recipe TEMPLATE, which encodes
        # the order the subgoals were meant to run in and therefore assumes every earlier one
        # already ran. That assumption breaks the moment the plan is reordered, spliced, or
        # partly undone — exactly what error injection and cook-initiated recovery do. Ground
        # the block against the LIVE kitchen instead (park_plate above already does), and when
        # what the subgoal consumes is not there, prepend the work that produces it rather than
        # handing back an unexecutable block and answering ok=True.
        inserted = None
        if game is not None and sid.startswith("assemble(") and sid.endswith(")"):
            comp = sid[len("assemble("):-1]
            hands = plate_into_hand(game)
            if isinstance(hands, str):
                return {"ok": False, "error": f"{sid} cannot be done right now: {hands}."}
            need = missing_step(game, state.recipe, comp)
            if isinstance(need, str):
                # Genuinely impossible right now: say WHY, so the cook can act on it instead of
                # re-asking. An unsatisfiable subgoal answered ok=True produced the 450-tick
                # livelock; one answered with an unexecutable block is the same bug, quieter.
                return {"ok": False, "error": f"{sid} cannot be done right now: {need}."}
            prep_first = bool(need) and any(c in ("pick up", "chop", "turn on") for c in need)
            if need:
                actions, inserted = list(need) + actions, f"prep({comp})"
            if hands:
                # scooping needs the plate in hand; without it the scoop simply fails. A prep needs EMPTY hands, so
                # the plate is fetched after it (the old order took the plate first and then "picked up" a fish
                # with the plate in hand — refused)
                if prep_first:
                    actions = list(need) + list(hands) + actions[len(need):]
                else:
                    actions = list(hands) + actions
                inserted = inserted or "fetch_plate"
        elif game is not None and sid == "serve":
            from ..core.items import Plate as _P
            if not isinstance(game.cooks[0].holding, _P):
                return {"ok": False, "error":
                        f"{sid} cannot be done right now: you are not holding the plate, so "
                        f"there is nothing to hand over. Pick the plate up first."}
        elif game is not None and sid == "fetch_plate":
            # "go to the plates / pick up" assumes empty hands. Holding the plate already makes
            # the pick-up illegal-skipped and everything after it fails; holding an INGREDIENT
            # needs it parked first. Neither was checked, so the block came back ok=True and did
            # nothing — the same silent-dead-block class as assemble() and ignite().
            from ..core.items import Plate
            held = game.cooks[0].holding
            if isinstance(held, Plate):
                return {"ok": True, "result": {"subgoal": sid, "actions": [],
                                               "already_satisfied": "you are holding the plate"}}
            if held is not None:
                spot = free_counter(game)
                if spot is None:
                    return {"ok": False, "error":
                            f"{sid} cannot be done right now: your hands are full and every "
                            f"counter is occupied — use what you are holding, or trash it."}
                actions = [f"go to the counter ({spot[0]},{spot[1]})", "put down"] + actions
                inserted = "park_held"
        elif game is not None and sid.startswith("ignite(") and sid.endswith(")"):
            # Same blindness as assemble: "go to the pot with the rice / turn on" cannot ground
            # when nothing holds rice, so the command fails and the cook is left with a subgoal
            # it can never retire. If the ingredient is missing, load it first — WITHOUT the
            # trailing "turn on" from prep_of, because igniting is this subgoal's own job.
            comp = sid[len("ignite("):-1]
            if station_holding(game, comp) is None:
                if not _has_source(game, comp):
                    return {"ok": False, "error":
                            f"{sid} cannot be done right now: there is no {comp} anywhere and no "
                            f"{comp} source to make one from."}
                if plate_held(game) and free_counter(game) is None:
                    return {"ok": False, "error":
                            f"{sid} cannot be done right now: loading a {comp} needs both hands, "
                            f"but you are holding the plate and every counter is occupied."}
                load = [c for c in prep_of(comp).commands if c != "turn on"]
                actions = _with_hands_free(load, f"prep({comp})", game) + actions
                inserted = f"prep({comp})"
        elif sid.startswith("prep(") and sid.endswith(")"):
            # E12 (round-2 review): on a LATE-IGNITE recipe (several heat groups: Fish & Chips, Steak Frites) the
            # template's prep only LOADS the station and a separate ignite(x) subgoal lights it later. A cook that
            # asked "how do I prep the fish" got a block with no "turn on" and left the fish in a cold fryer for
            # 120-380 ticks (6 of 10 baseline hard episodes). The skill now lights the station when this load
            # completes the item's batch (the fish; the SECOND potato — a lit fryer takes no more food).
            comp = sid[len("prep("):-1].split(")")[0]
            if ("turn on" not in actions and _sname(required_state(state.recipe, comp)) == "cooked"
                    and any(n.type == "ignite" and n.component == comp for n in segment_plan(state.recipe))):
                need = sum(1 for c, _s in state.recipe.contents if c == comp)
                loaded = 0
                if game is not None:
                    from ..core.stations import CookStation
                    loaded = max([sum(1 for c in (st_.contents or []) if getattr(c, "name", None) == comp)
                                  for st_ in game.stations.values()
                                  if isinstance(st_, CookStation) and st_.status in ("empty", "raw")] or [0])
                if loaded + 1 >= need:
                    actions = list(actions) + ["turn on"]
            if game is not None:
                actions = _with_hands_free(actions, sid, game)   # prep needs empty hands
        res = {"subgoal": sid, "actions": _mark_multi_instance(actions, game)}
        if inserted:
            res["prerequisite_inserted"] = inserted
        return {"ok": True, "result": res}
    if fn == "expand_action":
        a = args.get("action")
        if not isinstance(a, str) or not a.strip():
            return {"ok": False, "error": 'expand_action needs args {"action": "<one command>"}'}
        att = attribute_action(a, state.recipe)
        if att is None:
            return {"ok": False, "error": f"{a!r} does not belong to exactly one subgoal "
                                          "(generic verbs like \"scoop\" appear in several), so it "
                                          "cannot be expanded. Use resolve_subgoal instead."}
        return {"ok": True, "result": {"subgoal": att[0], "actions": _mark_multi_instance(att[1], game)}}
    if fn == "full_repair_plan":
        # DELIBERATELY UNAVAILABLE: handing the whole grounded fix would let the cook bypass the
        # supervisor entirely. The cook must be TOLD which subgoals to do, then ground them.
        return {"ok": False, "error": "full_repair_plan is not available — the supervisor decides "
                "WHAT to do. Use resolve_subgoal('<subgoal the supervisor named>') to turn each "
                "step it gives you into grounded commands."}
    return {"ok": False, "error": f"unknown function {fn!r}. Available: resolve_subgoal, expand_action"}


def _base_sid(s: str) -> str:
    return s.split(":", 1)[1] if s.startswith(FIX) else s


def merge_by_template(items: List[Tuple[str, str]], recipe) -> List[Tuple[str, str]]:
    """Order command blocks by the CLEAN plan's template order (the recipe order is public —
    the human knows bun goes before meat even when the supervisor listed steps differently).
    Unknown blocks (trash, literal rescue actions) keep their relative position and go FIRST
    (repairs precede re-plating)."""
    order = {n.sid: i for i, n in enumerate(segment_plan(recipe))}
    blocks, cur, cur_sid = [], [], None
    for cmd, sid in items:
        b = _base_sid(sid)
        if b != cur_sid and cur:
            blocks.append((cur_sid, cur))
            cur = []
        cur_sid = b
        cur.append((cmd, sid))
    if cur:
        blocks.append((cur_sid, cur))
    keyed = [(order.get(b, -1), i, cmds) for i, (b, cmds) in enumerate(blocks)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [c for _k, _i, cmds in keyed for c in cmds]


def apply_avert(ro, state: UserState):
    """A correct, timely answer to the human's own question AVERTS the error: enter
    correction (clears the deviated future) and restore the required subgoals from the
    template — the same restoration a validated bare flag triggers for a capable user."""
    enter_error_correction(ro, state)
    apply_fix(ro, state, self_repair_plan(state, ro.game))


def strip_done_task_prefix(commands, done_cmds, task_seq):
    """A SOFT system task stays in the cook's prompt until its sequence COMPLETES, so the
    cook re-announces and re-queues the WHOLE sequence every turn. Re-applying it wholesale
    superseded the in-flight tail forever: park_and_check spent 138 ticks re-parking an
    already-parked plate and the final `pick up` never ran. Drop the re-issued copies of
    tasked steps already DONE since injection, so only the remaining tail applies (and the
    apply_fix idempotency check below then absorbs a tail that is already queued)."""
    i = 0
    for c in done_cmds:
        if i < len(task_seq) and c == task_seq[i]:
            i += 1
    already = list(task_seq[:i])
    kept = []
    for cmd in commands:
        c = cmd[0] if isinstance(cmd, (list, tuple)) else cmd
        if already and c == already[0]:
            already.pop(0)
            continue
        kept.append(cmd)
    return kept


def apply_fix(ro, state: UserState, commands: List[Tuple[str, str]], op: str = None):
    """Splice usable guidance into the queue, in template order. `op` is the human's EXPLICIT
    choice of how to fold in the supervisor's suggestion:
      "append"  → add to the END of the queue, keeping the existing tail.
      "replace" → swap the remaining queue for these commands.
      None      → legacy: choose by mode (error_correction replaces, else appends) — kept for
                  the self-repair / avert callers that always rebuild the whole plan.
    A blind full-replace on EVERY parse was what caused queue thrash; letting the human pick
    append vs replace (and pick "drop" upstream, so this is never called) fixes it."""
    # IDEMPOTENCY (the real thrash killer): a step-by-step supervisor re-issues the SAME
    # multi-step plan every poll; the human faithfully "replace"s with commands it is ALREADY
    # on. If the new commands are just a SUBSEQUENCE of the plan the cook is already running
    # (current in-flight action + remaining queue), nothing has actually changed — do NOT reset
    # the queue or abort the in-flight action (that reset is what stopped "put the plate down"
    # from ever completing). Only genuine additions / reorderings fall through and apply.
    if op in ("append", "replace") and commands:
        merged_cmds = [c for c, _ in merge_by_template(commands, state.recipe)]
        # W3b (2026-08-12 audit): a MACRO command (`__wander__ 10`) re-issued verbatim
        # never matches its own expansion below, so every re-issue superseded the
        # in-flight wander and re-expanded from scratch — the cook froze for 218 ticks
        # in both macro episodes. An identical raw re-issue while the previous apply is
        # still executing is always the thrash case: make it a no-op.
        raw = [c for c, _ in commands]
        if raw == getattr(ro, "_last_applied_raw", None):
            busy = (ro.ctrl.skill is not None or ro.i + 1 < len(ro.cmds)
                    or getattr(ro, "_wander", 0) or getattr(ro, "_wait", 0))
            if busy:
                ro._idle_since = None
                if state.correction is not None:
                    state.correction["fix_applied"] = True
                return
        # E5 (hard-map audit 2026-08-15): this used to scan the ENTIRE remaining queue, so
        # an accepted repair was declared "already running" when its commands happened to
        # appear in a block dozens of steps away. hm_leave_unattended_expM9: the cook
        # accepted "go to the pan with the steak, scoop" with two stations smoking, the
        # commands matched the queue's distant assemble tail, and the merge was dropped
        # ~10 times in a row — the cook kept doing busywork saying "I'm dropping the
        # busywork" and reached the pan ~60 ticks late (nag_count 44 on a WON episode).
        # "Already doing it" can only mean the IMMINENT queue: the in-flight action plus a
        # short lookahead. Anything further away is a future intention, not current
        # behaviour, and a fresh instruction outranks it.
        LOOKAHEAD = 4
        on_now = [c for c, _ in ro.cmds[ro.i:ro.i + 1 + LOOKAHEAD]]
        it = iter(on_now)
        if all(cmd in it for cmd in merged_cmds):          # merged is a subsequence of on_now
            ro._idle_since = None
            if state.correction is not None:
                state.correction["fix_applied"] = True
            return
    # A stuck in-flight manipulation (a scoop with nothing to scoop enters a 60-tick WAIT,
    # a lost cook a wander stall) would BLOCK the whole queue behind it — the new plan could
    # never start. Following a fresh instruction means dropping whatever you were stuck on.
    if getattr(ro, "_wait", 0):
        ro._wait = 0
        if ro.log and ro.log[-1]["status"] == "waiting":
            ro.log[-1].update(status="superseded", t_done=ro.tick)
    if getattr(ro, "_wander", 0):
        ro._wander = 0
        if ro.log and ro.log[-1]["status"] in ("running", "wandering"):
            ro.log[-1].update(status="superseded", t_done=ro.tick)
    # A HEALTHY in-flight walk (ctrl.skill) is left ALONE: it finishes on its own and the new
    # queue takes over on arrival. Aborting every walk on each guidance apply caused Zeno
    # thrash — a supervisor re-guiding every 2 ticks restarted the walk endlessly and the cook
    # never arrived anywhere. Only genuinely STUCK actions (the wait/wander above) are dropped.
    merged = merge_by_template(commands, state.recipe)
    ro._last_applied_raw = [c for c, _ in commands]        # W3b: for the re-issue no-op
    if op == "append":
        ro.cmds.extend(merged)
    elif op == "replace":
        ro.cmds[ro.i + 1:] = merged
    elif state.mode == "error_correction":                  # legacy (op=None): mode decides
        ro.cmds[ro.i + 1:] = merged
    else:
        ro.cmds.extend(merged)
    ro._idle_since = None
    if state.correction is not None:
        state.correction["fix_applied"] = True


# ===================================================================== announced plan changes
def apply_announced_change(ro, state: UserState, event: dict) -> dict:
    """[v2, design doc §6b] The HUMAN's own announcement changes its plan — the supervisor is
    never in the loop (it only updates its own estimated future subgoals in memory, which is
    scored from its memory JSON, not registered here). Called by the harness AT THE MOMENT the
    ANNOUNCE_PLAN_CHANGE event fires: the system fills `modified_task` / `modified_subgoals`
    (system-private; the public goal is never overwritten) and updates the human's action
    sequence + success criterion to match what it announced."""
    from ..core.lang_events import extend_goal
    change = (event.get("gt_answer") or {}).get("expected_update")
    if change is None and (event.get("content") or {}).get("change") == "reorder_prep":
        change = {"op": "reorder_prep", "order": (event["content"] or {}).get("new_prep_order")}
    if not change:
        return {"applied": False, "reason": "announcement carries no expected_update"}
    if change["op"] == "add_ingredient":
        ing, st = change["ingredient"], change.get("state", "raw")
        extend_goal(ro.game, ing, st)                       # success criterion
        sid = f"assemble({ing})+announced"
        serve_i = next((i for i in range(ro.i + 1, len(ro.cmds))
                        if ro.cmds[i][0].startswith("go to the serving")), len(ro.cmds))
        ro.cmds[serve_i:serve_i] = [(f"go to the {ing} source", sid), ("scoop", sid)]
        state.modified_task = [list(c) for c in state.recipe.contents] + [[ing, st]]
        state.modified_subgoals = [n.sid for n in state.clean_ir[:-1]] + [sid, "serve"]
    elif change["op"] == "reorder_prep":
        state.modified_subgoals = list(change.get("order") or [])
    return {"applied": True, "change": change}


# ===================================================================== resolve predicates
def plate_held(game) -> bool:
    from ..core.items import Plate
    return isinstance(game.cooks[0].holding, Plate)


def plate_ok(game, recipe, contents=None) -> bool:
    """Plate-prefix predicate: the held plate's contents are again consistent with the goal.
    `contents` overrides the goal list (the EFFECTIVE goal after an announced change)."""
    from ..core.items import Plate
    h = game.cooks[0].holding
    if not isinstance(h, Plate):
        return True                             # nothing plated yet — nothing broken
    seq = getattr(h, "content_seq", None) or [(i.name, i.state.value) for i in h.contents]
    goal = list(contents) if contents is not None else list(recipe.contents)
    if recipe.ordered:
        return list(map(tuple, seq)) == [tuple(g) for g in goal[:len(seq)]]
    want = {tuple(g) for g in goal}
    have = [tuple(s) for s in seq]
    return all(x in want for x in have) and len(have) == len(set(have) & want)


def unattended_hot_station(game) -> Optional[str]:
    """Name of the component smoking on a station nobody is about to scoop, else None."""
    for st in game.stations.values():
        if getattr(st, "smoking", False) and getattr(st, "contents", None):
            return st.contents[0].name
    return None


def station_safe(game) -> bool:
    return not any(getattr(st, "fire_elapsed", 0) or getattr(st, "smoking", False)
                   for st in game.stations.values())


def resolved(ro, state: UserState) -> bool:
    """Per-tick judgment during error_correction (design doc §4): the plan is whole again
    (every required subgoal done or queued), the plate is consistent, no station is burning."""
    if state.correction is None:
        return True
    game = ro.game
    done = {_base_sid(l["kind"]) for l in ro.log if l["status"] == "done"}
    queued = {_base_sid(s) for _, s in ro.cmds[ro.i + 1:]}
    for sid in state.correction["required"]:
        if sid not in done and sid not in queued:
            return False
    return plate_ok(game, state.recipe, state.goal_contents()) and station_safe(game)


def _reanchor(items: List[Tuple[str, str]], recipe) -> List[Tuple[str, str]]:
    """A correction can split a template block: its 'go to' may have executed before the freeze
    while its manipulation resumes after, with the cook now somewhere else entirely. Prepend the
    nearest preceding 'go to' from the template for any block resuming mid-way."""
    tmpl = {n.sid: list(n.commands) for n in segment_plan(recipe)}
    out, seen_first = [], set()
    for cmd, sid in items:
        b = _base_sid(sid)
        if b in tmpl and b not in seen_first:
            seen_first.add(b)
            cmds = tmpl[b]
            if cmd in cmds and not cmd.startswith("go to"):
                j = cmds.index(cmd)
                back = next((cmds[k] for k in range(j - 1, -1, -1) if cmds[k].startswith("go to")), None)
                if back:
                    out.append((back, sid))
        out.append((cmd, sid))
    return out


def exit_correction(ro, state: UserState):
    """Resolved: merge the stashed non-deviated remainder back with whatever fix commands are
    still queued, in template order, deduplicating subgoals the fix already covers."""
    remaining = ro.cmds[ro.i + 1:]
    covered = {_base_sid(s) for _, s in remaining}
    tail = [(c, s) for c, s in state.pending_tail if _base_sid(s) not in covered]
    ro.cmds[ro.i + 1:] = _reanchor(merge_by_template(remaining + tail, state.recipe), state.recipe)
    state.pending_tail = []
    state.mode = "executing"
    state.correction = None
    ro._idle_since = None
