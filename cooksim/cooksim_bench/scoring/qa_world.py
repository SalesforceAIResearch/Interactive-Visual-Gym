#!/usr/bin/env python3
"""World history for question truths: what physically happened in the kitchen, tick by tick.

Why (pre-formal-run audit 2026-09-19, tmp/suite_audit/FIX_PLAN.md A1-A3): the v3 truths read the
command log and matched each command's STEP TAG ("prep(tomato)") or its WORDS ("cutting board").
The self-planning cook's commands are tagged `vlm_fix:literal` and say "go to board 1", so
  * every order question with a prep side found no prep at all (14/14 recorded asks wrong),
  * every "how many times have I walked over to a cutting board" was 0 (15/15),
  * "what is my next step?" read the cook's own queue and leaked takeover labels (17/17).
This module records the kitchen itself instead. One observer, fed one render_state() snapshot per
tick, is shared by every path that needs a truth:
  * live: exp_error_rollout.Rollout owns a WorldTrack and observes after every tick;
  * offline, with a states dump: WorldTrack.from_snapshots(<record>.states.jsonl);
  * offline, without one: python -m cooksim_bench.scoring.qa_replay re-issues the record's action log through the same
    executor and yields the same snapshots (measured exact on all 100 verify_llm_formal episodes).

Event kinds (each with the tick it is first SEEN, i.e. the snapshot index -- the state after the
step that caused it):
  chop    an item on a board turned chopped            (item, pos)
  load    an item entered a heat station               (item, pos, station)
  lit     a heat station holding the item started cooking
  cooked  ... reached cooked;   burnt  ... caught fire
  unload  an item left a heat station
  plate   an item was added to the plate being built   (item, state)   -- fullest plate, ordered diff
  plate_pickup  the cook picked up a plate;   serve  a dish was handed in (stats)
Faced tiles are kept per tick (on change) so a finished "go to ..." can be resolved to the
station the cook actually ended up facing.
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import json
import re

HEAT_TYPES = ("pot", "pan", "oven", "fryer")
# canonical visit kind per terrain value (the words the questions use)
_TERRAIN_KIND = {"cutting_board": "cutting board", "pan": "pan", "pot": "pot", "fryer": "fryer", "oven": "oven",
                 "trash": "trash", "serving": "serving pass", "counter": "counter", "plate_source": "plates",
                 "sink": "sink"}
# question words -> canonical kind (the engine's own synonym table decides; see highlevel._STATION_SYN)
_WORD_KIND = {"board": "cutting board", "cutting board": "cutting board", "chopping board": "cutting board",
              "pan": "pan", "skillet": "pan", "pot": "pot", "fryer": "fryer", "deep fryer": "fryer", "oven": "oven",
              "trash": "trash", "bin": "trash", "garbage": "trash", "serving": "serving pass", "pass": "serving pass",
              "serving pass": "serving pass", "counter": "counter", "plates": "plates", "plate": "plates",
              "plate stack": "plates", "sink": "sink"}
_DIRV = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}


def terrain_kind(value):
    v = str(value or "").split(".")[-1].lower()
    if v.endswith("_source") and v != "plate_source":
        return v[:-7] + " source"
    return _TERRAIN_KIND.get(v, v)


def visit_kind(words):
    """Canonical station kind for the words of a visit question ("cutting board", "pan", "board")."""
    w = re.sub(r"\s+", " ", str(words or "").replace("_", " ").strip().lower())
    if w in _WORD_KIND:
        return _WORD_KIND[w]
    if w.endswith(" source"):
        return w
    for k in sorted(_WORD_KIND, key=len, reverse=True):
        if re.search(r"\b" + re.escape(k) + r"\b", w):
            return _WORD_KIND[k]
    return w


def _names(lst):
    out = []
    for c in lst or []:
        out.append(c.get("name") if isinstance(c, dict) else getattr(c, "name", str(c)))
    return out


def plate_of(rs):
    """(contents [(name, state)], where) of the plate the cook is building: the FULLEST plate in hand or on a
    counter, ties to the held one -- the harness's plate_list rule (a held empty rescue plate must not hide the
    parked plate that carries the dish)."""
    cands = []
    for p in (rs.get("players") or [])[:1]:
        h = p.get("holding") or {}
        if h.get("kind") == "plate":
            cands.append(([(x.get("name"), x.get("state")) for x in h.get("contents") or []], "held"))
    for o in rs.get("objects") or []:
        it = o.get("item") or {}
        if it.get("kind") == "plate":
            cands.append(([(x.get("name"), x.get("state")) for x in it.get("contents") or []], "counter"))
    if not cands:
        return [], None
    return max(cands, key=lambda c: len(c[0]))


class WorldTrack:
    def __init__(self):
        self.events = []            # {"t", "kind", "item", "pos", "station", "state"}
        self.faced = []             # [(t, pos, dir, tile, kind)] -- appended when anything changes
        self.types = {}             # pos -> station type (from the terrain)
        self.last_t = None
        self._st = {}               # pos -> ("board", name, state) | ("heat", names tuple, status)
        self._plate = []
        self._held_plate = False
        self._served = 0

    # ------------------------------------------------------------------ feeding
    def observe_game(self, game, t=None):
        self.observe(game.render_state(), t)

    def observe(self, rs, t=None):
        t = rs.get("tick") if t is None else t
        if self.last_t is not None and t is not None and t <= self.last_t:
            return                                   # one observation per tick
        self.last_t = t
        grid = rs.get("terrain") or []
        for st in rs.get("stations") or []:
            pos = (st.get("x"), st.get("y"))
            typ = str(st.get("type") or "")
            self.types[pos] = typ
            prev = self._st.get(pos)
            if "contents" in st and typ in HEAT_TYPES:
                names = tuple(_names(st.get("contents")))
                status = st.get("status")
                cur = ("heat", names, status)
                pn = list(prev[1]) if prev else []
                ps = prev[2] if prev else "empty"
                left = list(pn)
                for n in names:
                    if n in left:
                        left.remove(n)
                    else:
                        self._ev(t, "load", n, pos, typ)
                cur_left = list(names)
                for n in pn:
                    if n in cur_left:
                        cur_left.remove(n)
                    else:
                        self._ev(t, "unload", n, pos, typ)
                if status != ps:
                    k = {"cooking": "lit", "cooked": "cooked", "burnt": "burnt"}.get(status)
                    if k:
                        for n in names:
                            self._ev(t, k, n, pos, typ)
                self._st[pos] = cur
            elif "item" in st:
                it = st.get("item") or None
                cur = ("board", it.get("name") if it else None, it.get("state") if it else None)
                if it and it.get("state") == "chopped" and prev and prev[1] == cur[1] and prev[2] != "chopped":
                    self._ev(t, "chop", cur[1], pos, typ)
                self._st[pos] = cur
        # plate being built: additions to the fullest plate, ordered diff (as the harness's track_plate)
        cur, _where = plate_of(rs)
        if cur[:len(self._plate)] == self._plate:
            for n, s in cur[len(self._plate):]:
                self._ev(t, "plate", n, None, None, state=s)
        self._plate = cur
        held = any(((p.get("holding") or {}).get("kind") == "plate") for p in rs.get("players") or [])
        if held and not self._held_plate:
            self._ev(t, "plate_pickup", None, None, None)
        self._held_plate = held
        stats = rs.get("stats") or {}
        served = int(stats.get("deliveries", 0) or 0) + int(stats.get("failed_deliveries", 0) or 0)
        if served > self._served:
            self._ev(t, "serve", None, None, None)
        self._served = served
        # the tile the cook faces
        for p in (rs.get("players") or [])[:1]:
            x, y, d = p.get("x"), p.get("y"), p.get("dir")
            dx, dy = _DIRV.get(d, (0, 0))
            fx, fy = x + dx, y + dy
            kind = None
            if 0 <= fy < len(grid) and 0 <= fx < len(grid[fy]):
                kind = terrain_kind(grid[fy][fx])
            row = (t, (x, y), d, (fx, fy), kind)
            if not self.faced or self.faced[-1][1:] != row[1:]:
                self.faced.append(row)

    def _ev(self, t, kind, item, pos, station, state=None):
        e = {"t": t, "kind": kind, "item": item}
        if pos is not None:
            e["pos"] = list(pos)
        if station:
            e["station"] = station
        if state is not None:
            e["state"] = state
        self.events.append(e)

    @classmethod
    def from_snapshots(cls, snaps):
        w = cls()
        for i, s in enumerate(snaps):
            w.observe(s, s.get("tick", i) if isinstance(s, dict) else i)
        return w

    @classmethod
    def from_dump(cls, path):
        w = cls()
        with open(path) as fh:
            for line in fh:
                try:
                    s = json.loads(line)
                except Exception:                    # noqa: BLE001
                    continue
                w.observe(s)
        return w

    # ------------------------------------------------------------------ reading
    def ticks(self, kind, item=None, upto=None):
        return [e["t"] for e in self.events if e["kind"] == kind and (item is None or e["item"] == item)
                and (upto is None or e["t"] <= upto)]

    def first(self, kind, item=None, upto=None):
        ts = self.ticks(kind, item, upto)
        return min(ts) if ts else None

    def faced_at(self, t):
        """(pos, dir, tile, kind) the cook had in the snapshot of tick t (latest change at or before t)."""
        best = None
        for row in self.faced:
            if row[0] is None or row[0] <= t:
                best = row
            else:
                break
        return best[1:] if best else None

    def station_type(self, pos):
        return self.types.get(tuple(pos)) if pos is not None else None

    def lit_before(self, pos, t):
        """Latest tick <= t at which the station at `pos` was lit."""
        ts = [e["t"] for e in self.events if e["kind"] == "lit" and e.get("pos") == list(pos) and e["t"] <= t]
        return max(ts) if ts else None


# ---------------------------------------------------------------------- visits
def visits(log, world, words, upto=None):
    """The cook's walks to a station of the asked kind, resolved against the WORLD.

    A visit is a finished "go to ..." command (the engine confirms the cook arrived and turned to
    face the target); the station visited is the one the cook FACES in the next snapshot, whatever
    the command called it ("board 1", "pan 2", "the counter (3,4)"). A "go to" that neither moved
    nor turned the cook (re-issued while already standing there) is not a new visit.
    Without a world the engine's own parser names the target kind (never a regex on the words).
    `upto`: only walks visible in the snapshot of that tick (t_done < upto)."""
    want = visit_kind(words)
    out = []
    for l in sorted((l for l in (log or []) if l.get("status") == "done" and str(l.get("cmd", "")).startswith("go to")
                     and l.get("t_done") is not None), key=lambda l: (l["t_done"], l.get("i") or 0)):
        if upto is not None and not l["t_done"] < upto:
            continue
        kind, tile = None, None
        if world is not None and world.faced:
            after = world.faced_at(l["t_done"] + 1)
            before = world.faced_at(l.get("t_issue") if l.get("t_issue") is not None else l["t_done"])
            if after is not None:
                kind, tile = after[3], after[2]
                if before is not None and before[0] == after[0] and before[2] == after[2]:
                    continue                          # already there: no walk
        if kind is None:
            try:
                from cooksim.agents.highlevel import parse
                it = parse(l["cmd"])
                obj = it.obj if it else ""
                kind = visit_kind(obj) if obj and obj != "source" else (f"{it.ingredient} source" if it and it.ingredient else obj)
            except Exception:                        # noqa: BLE001
                kind = None
        if kind == want:
            out.append({"t": l["t_done"], "cmd": l["cmd"], "tile": list(tile) if tile else None})
    return out


# ---------------------------------------------------------------------- recipe knowledge
def chopped_items(recipe):
    """Items the recipe's own plan chops (plan IR), e.g. Steak Frites -> {steak, potato}."""
    try:
        from cooksim.core.plan_ir import segment_plan
        return {n.component for n in segment_plan(recipe) if n.type == "prep" and any(c == "chop" for c in n.commands)}
    except Exception:                                 # noqa: BLE001
        return set()


def recipe_for_cell(cell):
    try:
        import os
        os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
        from cooksim_bench.exp_interact import recipe_of
        return recipe_of(cell) if cell else None
    except Exception:                                 # noqa: BLE001
        return None


def world_event_tick(world, spec, upto=None, chopped=None, first=True, log=None):
    """Tick of an event spec from the world (first occurrence if `first`, else the latest), or None.
    Specs: chop(X) heat(X) ignite(X) cook(X) prep(X) assemble(X) fetch_plate serve visit(<words>)
    prep(X) is the recipe's own processing of X: the chop when the recipe chops X, else putting X on the heat."""
    m = re.match(r"^(\w+)(?:\((.*)\))?$", spec.strip())
    if not m:
        return None
    verb, arg = m.group(1), (m.group(2) or "").strip()
    pick = min if first else max
    if verb == "visit":
        vs = visits(log, world, arg, upto=upto)
        return pick(v["t"] for v in vs) if vs else None
    if world is None:
        return None
    if verb == "prep":
        if chopped is None:                            # recipe unknown: whichever processing the item got
            ts = world.ticks("chop", arg, upto) + world.ticks("load", arg, upto)
            return pick(ts) if ts else None
        verb = "chop" if arg in chopped else "heat"
    kind = {"chop": "chop", "heat": "load", "load": "load", "ignite": "lit", "lit": "lit", "cook": "cooked",
            "cooked": "cooked", "assemble": "plate", "fetch_plate": "plate_pickup", "serve": "serve"}.get(verb)
    if kind is None:
        return None
    ts = world.ticks(kind, arg or None, upto)
    return pick(ts) if ts else None


# ---------------------------------------------------------------------- "what is my next step?"
_STAGE = {"none": 0, "held_raw": 1, "board_raw": 2, "chopped": 3, "loaded": 4, "cooking": 5, "ready": 6}


def _units(rs, item, needs_chop, final):
    """Every off-plate unit of `item` in the kitchen with its stage toward `final` (chopped/cooked/raw)."""
    out = []
    for p in (rs.get("players") or [])[:1]:
        h = p.get("holding") or {}
        if h.get("kind") != "plate" and h.get("name") == item:
            s = h.get("state")
            if final == "chopped":
                out.append("ready" if s == "chopped" else "held_raw")
            elif final == "cooked":
                out.append("ready" if s == "cooked" else ("chopped" if s == "chopped" else ("held_raw" if needs_chop else "chopped")))
    for st in rs.get("stations") or []:
        typ = str(st.get("type") or "")
        if "contents" in st and typ in HEAT_TYPES:
            for c in st.get("contents") or []:
                if (c.get("name") if isinstance(c, dict) else c) == item and final == "cooked":
                    stt = st.get("status")
                    out.append({"cooked": "ready", "cooking": "cooking", "raw": "loaded"}.get(stt, "loaded")
                               if stt != "burnt" else "none")
        elif st.get("item"):
            it = st["item"]
            if it.get("name") == item:
                if it.get("state") == "chopped":
                    out.append("ready" if final == "chopped" else "chopped")
                else:
                    out.append("board_raw")
    for o in rs.get("objects") or []:
        it = o.get("item") or {}
        if it.get("kind") != "plate" and it.get("name") == item:
            s = it.get("state")
            if final == "chopped":
                out.append("ready" if s == "chopped" else "held_raw")
            elif final == "cooked":
                out.append("ready" if s == "cooked" else ("chopped" if s == "chopped" else "held_raw"))
    return sorted(out, key=lambda s: -_STAGE[s])


def next_step_truth(rs, recipe):
    """The next thing the ORDER still needs, from the recipe and the live kitchen -- never from the
    cook's queue. Returns next_subgoal (the recipe plan's first unmet step), a plain-words answer,
    and also_acceptable: every other step the order still needs that can be done right now."""
    from cooksim.core.plan_ir import segment_plan
    contents = [(n, getattr(s, "value", s)) for n, s in recipe.contents]
    ordered = bool(getattr(recipe, "ordered", False))
    chop = chopped_items(recipe)
    plate, where = plate_of(rs)
    have_plate = where is not None
    # how much of the plate is right
    k = 0
    if ordered:
        while k < len(plate) and k < len(contents) and plate[k][0] == contents[k][0] and \
                (contents[k][1] == "raw" or plate[k][1] == contents[k][1]):
            k += 1
        wrong = plate[k:]
        remaining = contents[k:]
    else:
        left = list(contents)
        wrong = []
        for n, s in plate:
            hit = next((c for c in left if c[0] == n and (c[1] == "raw" or c[1] == s)), None)
            if hit:
                left.remove(hit)
            else:
                wrong.append((n, s))
        remaining = left
    served = int((rs.get("stats") or {}).get("deliveries", 0) or 0) + int((rs.get("stats") or {}).get("failed_deliveries", 0) or 0)
    base = {"recipe": [c[0] for c in contents], "plate": [p[0] for p in plate], "plate_where": where,
            "remaining": [r[0] for r in remaining], "source": "world"}
    if wrong:
        w = ", ".join(n for n, _ in wrong)
        return {**base, "next_subgoal": "fix_plate", "also_acceptable": [], "wrong_on_plate": [list(x) for x in wrong],
                "answer": f"start the plate over: the {w} on it is out of order or in the wrong state for this dish "
                          f"(throw the plate away and plate {', '.join(r[0] for r in contents)} in that order)"}
    if not remaining:
        return {**base, "next_subgoal": "serve", "also_acceptable": [],
                "answer": "serve the dish" if not served else "nothing — the dish has been served"}
    # stage of every remaining unit, per item (best units first)
    stage = {}
    for n in dict.fromkeys(r[0] for r in remaining):
        final = next(s for c, s in contents if c == n)
        units = _units(rs, n, n in chop, final)
        need = sum(1 for r in remaining if r[0] == n)
        stage[n] = (units + ["none"] * need)[:need]
    first_item = remaining[0][0]
    unmet = []                                         # (sid, words) in plan order
    seen = {}
    nodes = list(segment_plan(recipe))
    ignites = {n.component for n in nodes if n.type == "ignite"}
    for node in nodes:
        sid, typ, comp = node.sid, node.type, node.component
        if typ == "prep":
            if comp not in stage:
                continue                               # every unit of it is already on the plate
            idx = seen.get(("prep", comp), 0)
            seen[("prep", comp)] = idx + 1
            if idx >= len(stage[comp]):
                continue
            s = stage[comp][idx]
            final = next(st for c, st in contents if c == comp)
            # the plan either lights the station in its own ignite(X) step or inside prep(X) (Cheeseburger's meat)
            done_stage = ("loaded" if comp in ignites else "cooking") if final == "cooked" else "ready"
            if _STAGE[s] >= _STAGE[done_stage]:
                continue
            if comp in chop and _STAGE[s] < _STAGE["chopped"]:
                words = f"chop the {comp}" if s != "board_raw" else f"chop the {comp} on the board"
            elif s == "loaded":
                words = f"turn on the heat under the {comp}"
            else:
                words = f"put the {comp} on the heat"
            unmet.append((sid, words))
        elif typ == "ignite":
            if comp in stage and "loaded" in stage[comp]:
                unmet.append((sid, f"turn on the heat under the {comp}"))
        elif typ == "fetch_plate":
            if not have_plate:
                unmet.append((sid, "get a plate"))
        elif typ == "assemble":
            if comp != first_item and ordered:
                continue
            if comp not in stage or not have_plate:     # no plate yet: getting one IS the step
                continue
            s = stage[comp][0]
            if s == "ready" or next(st for c, st in contents if c == comp) == "raw":
                unmet.append((sid, f"put the {comp} on the plate"))
            elif s == "cooking":
                unmet.append((sid, f"wait for the {comp} to finish cooking, then put it on the plate"))
            else:
                unmet.append((sid, f"put the {comp} on the plate (it is not ready yet)"))
        elif typ == "serve":
            continue
    if not unmet:
        return {**base, "next_subgoal": f"assemble({first_item})", "also_acceptable": [],
                "answer": f"put the {first_item} on the plate"}
    # the canonical next step: the plan's first unmet step that can be DONE now; waiting counts only when nothing else can
    doable = [u for u in unmet if not u[1].startswith("wait for") and "not ready yet" not in u[1]]
    nxt = (doable or unmet)[0]
    also, _seen = [], {re.sub(r"#\d+$", "", nxt[0])}
    for u in unmet:
        k = re.sub(r"#\d+$", "", u[0])
        if u is not nxt and "not ready yet" not in u[1] and k not in _seen:
            _seen.add(k)
            also.append(u)
    return {**base, "next_subgoal": re.sub(r"#\d+$", "", nxt[0]), "then": (also[0][0] if also else None),
            "also_acceptable": [{"subgoal": re.sub(r"#\d+$", "", s), "words": w} for s, w in also],
            "stages": stage, "answer": nxt[1]}


# ---------------------------------------------------------------------- engine timing (Q10)
def station_timing(station_type, pace=None):
    """(cook_time, smoke_grace, burn_time) for a heat station, read from the ENGINE config and the pace
    preset -- never a hand-kept copy (the fryer burn was 45 here while the engine ran 60 since D-10)."""
    from cooksim.core.config import GameConfig, pace_preset
    cfg = GameConfig()
    typ = (station_type or "").lower()
    cook = getattr(cfg, f"{typ}_cook_time", None)
    burn = getattr(cfg, f"{typ}_burn_time", None)
    if cook is None:                                   # unknown station: the pan's numbers (the engine default)
        cook, burn = cfg.pan_cook_time, cfg.pan_burn_time
    try:
        _name, (pburn, _fire) = pace_preset(pace or "tight")
    except Exception:                                  # noqa: BLE001
        pburn = None
    if pburn is not None:
        burn = pburn
    return cook, cfg.smoke_grace, burn
