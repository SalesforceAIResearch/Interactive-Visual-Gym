#!/usr/bin/env python3
"""Question families v3 — ground truth computed from the engine trace (docs/SUITE_V3_DESIGN.md §3).

Every kind is parametrised ("base:arg") so that gen_intent_cases.ground_truth() routes it here
with (game, ro, arg): the rollout `ro` carries the action log (ro.log: cmd / kind=sid / t_done /
status), the live harness additionally attaches ro.plate_seq (plate events), ro.convo (the
conversation with ticks) and ro.qa_history (earlier questions' truths) — see e2e_stepin_rollout.

Event specs (used by order:/happened:):  prep(X) · chop(X) · heat(X) · ignite(X) · cook(X) · assemble(X)
                                          · fetch_plate · serve · visit(<words>)  e.g. visit(fryer), visit(cutting board)
  Since 2026-09-19 every event is read off the WORLD (python -m cooksim_bench.scoring.qa_world: ro.world, one kitchen snapshot per
  tick), never off a command's step tag or its words. prep(X) = the recipe's own processing of X: the chop
  when the recipe chops X, else putting X on the heat. visit(<w>) = a finished walk after which the cook
  FACES a station of that kind ("go to board 1" is a cutting-board visit).

Kinds
  order:A|B          did A finish before B?           -> before / after / neither / only_a / only_b
  happened:A         did A happen at all?             -> yes / no          (Q4 traps when "no")
  exists:<kind>      is there such a station/source?  -> yes / no
  count:visit(<w>)   how many finished "go to <w>" so far (all visits; distinct targets too)
  elapsed:X          ticks X has been on its heat, and since it was done
  longest_on:heat    which lit station has been on longest
  past_done:heat     is anything cooked and still on heat
  missing:plate      recipe items not yet on the held/any plate (multiset difference)
  empty_boards:board how many cutting boards are empty
  reask:<kind>       the same kind again + the diff against the earlier truth (ro.qa_history)
  after_ref:X        the plate item added right after X, and its state
  assistant_said:<w> the assistant's last line mentioning <w>
  next_step:plan     the next plan step not yet done
  burn_first:heat    which lit station burns first, ticks left
  ready_in:X         ticks until X is cooked
"""
import os
import re

TERM_DONE = ("done",)


# ------------------------------------------------------------------ trace helpers
def _log(ro):
    return list(getattr(ro, "log", []) or []) if ro is not None else []


def _done(ro):
    return [l for l in _log(ro) if l.get("status") in TERM_DONE]


def _sid_base(sid):
    return re.sub(r"#\d+$", "", sid or "")


def _pat(words):
    # case-insensitive since 2026-08-28: the case-sensitive form missed every line that opened
    # with the item's name ("Fish is done!"), silently dropping 1.5% of Q10 candidate lines and
    # 0.4% of Q8 topic mentions. Command strings are lower-case, so nothing else changes.
    return re.compile(r"\b" + r"[ _]".join(re.escape(w) for w in words.replace("_", " ").split()) + r"\b", re.I)


def _world(ro):
    return getattr(ro, "world", None) if ro is not None else None


def _chopped_set(ro, game=None):
    """Items this recipe's plan chops (for prep(X)); None when the recipe is unknown."""
    from cooksim_bench.scoring.qa_world import chopped_items, recipe_for_cell
    rec = getattr(ro, "recipe", None) if ro is not None else None
    if rec is None or not hasattr(rec, "contents"):
        cell = (getattr(ro, "cell", None) if ro is not None else None) or getattr(game, "cell", None)
        rec = recipe_for_cell(cell)
    return chopped_items(rec) if rec is not None else None


def _now(ro, game=None):
    t = getattr(ro, "tick", None) if ro is not None else None
    return t if t is not None else getattr(game, "tick", None)


def _event(ro, spec, first):
    """Tick of an event spec from the world history (first or latest occurrence up to now), or None.
    Without a world (a hand-built rollout) only plate history (assemble) and visits can be read; a
    prep/ignite/... then reads as not-happened and the truth says so via `source`."""
    from cooksim_bench.scoring.qa_world import world_event_tick
    m = re.match(r"^(\w+)(?:\((.*)\))?$", spec.strip())
    if not m:
        return None
    verb, arg = m.group(1), (m.group(2) or "").strip()
    w = _world(ro)
    upto = _now(ro)
    if w is None and verb == "assemble":
        ts = [e["t"] for e in (getattr(ro, "plate_seq", None) or []) if e.get("item") == arg and e.get("t") is not None
              and (upto is None or e["t"] <= upto)]
        return (min(ts) if first else max(ts)) if ts else None
    return world_event_tick(w, spec, upto=upto, chopped=_chopped_set(ro), first=first, log=_log(ro))


def event_tick(ro, spec):
    """Latest occurrence tick of an event spec up to now, or None if it has not happened."""
    return _event(ro, spec, first=False)


def _first_tick(ro, spec):
    """First occurrence tick of the spec (order questions compare FIRST occurrences)."""
    return _event(ro, spec, first=True)


def _visit_kind_of_type(station_type):
    from cooksim_bench.scoring.qa_world import terrain_kind
    return terrain_kind(station_type)


def _spec_words(spec, chopped=None, past=False):
    """Plain words for an event spec, per the item's process (chop vs put on the heat). `past` gives the past tense
    for the second clause of "Did I A before or after I B?" ("... after I chopped the potato")."""
    m = re.match(r"^(\w+)(?:\((.*)\))?$", spec.strip())
    v, a = m.group(1), (m.group(2) or "")
    if v == "prep":
        v = "chop" if (chopped is None or a in chopped) else "heat"
    base = {"chop": (f"chop the {a}", f"chopped the {a}"), "heat": (f"put the {a} on the heat", f"put the {a} on the heat"),
            "ignite": (f"turn the heat on under the {a}", f"turned the heat on under the {a}"),
            "cook": (f"cook the {a}", f"cooked the {a}"), "assemble": (f"put the {a} on the plate", f"put the {a} on the plate"),
            "fetch_plate": ("pick up the plate", "picked up the plate"),
            "serve": ("take the plate to the pass", "took the plate to the pass"),
            "visit": (f"go over to the {a}", f"went over to the {a}")}.get(v)
    return spec if base is None else base[1 if past else 0]


def _stations(game):
    """(pos, station, type) for every cook station with the engine's live timers."""
    out = []
    rs = game.render_state() if hasattr(game, "render_state") else {}
    types = {(d.get("x"), d.get("y")): d.get("type") for d in (rs.get("stations") or [])}
    for pos, st in game.stations.items():
        if hasattr(st, "cook_time"):
            out.append((tuple(pos), st, types.get(tuple(pos)) or type(st).__name__.lower()))
    return out


def _contents(st):
    return [getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else str(c)) for c in (st.contents or [])]


def _lit(st):
    return bool(getattr(st, "cooking", False)) and st.status in ("cooking", "cooked", "burnt")


def _burn_at(st):
    return st.cook_time + st.smoke_grace + st.burn_time


def _tick(ro, game):
    return getattr(ro, "tick", None) if ro is not None else getattr(game, "tick", None)


# ------------------------------------------------------------------ kinds
def gt_order(game, ro, arg):
    """Did A first happen before B? Both read off the world (qa_world); a side that has not happened is said so."""
    a, b = [x.strip() for x in arg.split("|", 1)]
    ta, tb = _first_tick(ro, a), _first_tick(ro, b)
    ch = _chopped_set(ro, game)
    if ta is None and tb is None:
        ans = "neither"
    elif tb is None:
        ans = "only_a"
    elif ta is None:
        ans = "only_b"
    else:
        ans = "before" if ta < tb else "after"
    words = {"before": f"{a} before {b}", "after": f"{a} after {b}", "neither": f"neither {a} nor {b} has happened yet",
             "only_a": f"{a} happened; {b} has not happened yet", "only_b": f"{b} happened; {a} has not happened yet"}
    return {"a": a, "b": b, "t_a": ta, "t_b": tb, "answer": ans, "answer_text": words[ans],
            "a_words": _spec_words(a, ch), "b_words": _spec_words(b, ch),
            "source": "world" if _world(ro) is not None else "no_world"}


def gt_happened(game, ro, arg):
    t = event_tick(ro, arg)
    out = {"event": arg, "happened": t is not None, "t": t, "answer": "yes" if t is not None else "no",
           "source": "world" if _world(ro) is not None else "no_world"}
    if arg.startswith("visit("):
        from cooksim_bench.scoring.qa_world import visits
        vs = visits(_log(ro), _world(ro), arg[6:-1], upto=_now(ro, game))
        out["visits"] = len(vs)
        out["first_t"] = vs[0]["t"] if vs else None
    return out


def gt_exists(game, ro, arg):
    rs = game.render_state() if hasattr(game, "render_state") else {}
    kinds = {str(d.get("type")) for d in (rs.get("stations") or [])}
    grid = game.layout.grid
    terrain = {str(t).split(".")[-1].lower() for row in grid for t in row}
    want = arg.lower()
    found = want in kinds or any(want in t for t in terrain)
    return {"kind": arg, "exists": found, "answer": "yes" if found else "no"}


def gt_count(game, ro, arg):
    """How many times the cook walked over to a station of this kind: finished walks resolved to the station the
    cook then FACED (qa_world.visits) -- "go to board 1" is a cutting-board visit; re-issuing a walk to where the
    cook already stands is not a new one."""
    from cooksim_bench.scoring.qa_world import visits, visit_kind
    m = re.match(r"^visit\((.*)\)$", arg.strip())
    words = m.group(1) if m else arg
    vs = visits(_log(ro), _world(ro), words, upto=_now(ro, game))
    tiles = {tuple(v["tile"]) if v.get("tile") else v["cmd"] for v in vs}
    return {"what": f"visits to {words}", "kind": visit_kind(words), "count": len(vs), "distinct_targets": len(tiles),
            "visit_ticks": [v["t"] for v in vs], "source": "world" if _world(ro) is not None else "parser",
            "answer": str(len(vs))}


def _station_holding(game, item):
    for pos, st, typ in _stations(game):
        if item in _contents(st):
            return pos, st, typ
    return None


def gt_elapsed(game, ro, arg):
    h = _station_holding(game, arg)
    if not h:
        return {"item": arg, "on_heat": False, "answer": f"{arg} is not on any heat"}
    pos, st, typ = h
    if not _lit(st):
        return {"item": arg, "on_heat": True, "lit": False, "station": typ, "at": list(pos), "answer": f"{arg} is in the {typ} but it is not on"}
    since_done = max(0, st.timer - st.cook_time) if st.status in ("cooked", "burnt") else 0
    return {"item": arg, "on_heat": True, "lit": True, "station": typ, "at": list(pos), "status": st.status,
            "ticks_on": st.timer, "ticks_since_done": since_done, "ticks_to_done": max(0, st.cook_time - st.timer),
            "answer": f"{st.timer} ticks on the {typ}" + (f", done for {since_done}" if since_done else f", {st.cook_time - st.timer} to go")}


def gt_longest_on(game, ro, arg):
    lit = [(st.timer, typ, list(pos), _contents(st)) for pos, st, typ in _stations(game) if _lit(st) and _contents(st)]
    if not lit:
        return {"lit": [], "answer": "nothing is on any heat"}
    lit.sort(reverse=True)
    tie = len(lit) > 1 and lit[0][0] == lit[1][0]
    return {"lit": [{"station": t, "at": p, "contents": c, "ticks_on": n} for n, t, p, c in lit],
            "answer": ("a tie: " if tie else "") + f"the {lit[0][1]} with {'/'.join(lit[0][3])} ({lit[0][0]} ticks)"}


def gt_past_done(game, ro, arg):
    rows = [{"station": typ, "at": list(pos), "contents": _contents(st), "ticks_since_done": max(0, st.timer - st.cook_time)}
            for pos, st, typ in _stations(game) if _lit(st) and st.status in ("cooked", "burnt") and _contents(st)]
    return {"done_on_heat": rows, "answer": ("yes: " + ", ".join(f"{r['contents']} in the {r['station']}" for r in rows)) if rows else "no"}


def _plate_items(game):
    """The plate the question is about. Guard added 2026-08-28 (grading audit F7): with a parked
    plate AND a fresh one in scene the old code silently read whichever it found first, so a
    repair could make the truth describe the wrong plate. The held plate wins; if there is no
    held plate and several are lying about, the ambiguity is REPORTED, not guessed."""
    rs = game.render_state() if hasattr(game, "render_state") else {}
    for p in (rs.get("players") or []):
        d = p.get("holding")
        if d and d.get("kind") == "plate":
            return [x["name"] for x in d.get("contents", [])], "held"
    loose = [o.get("item") for o in (rs.get("objects") or [])
             if (o.get("item") or {}).get("kind") == "plate"]
    if len(loose) > 1:
        return None, "ambiguous"
    if loose:
        return [x["name"] for x in loose[0].get("contents", [])], "counter"
    return None, None


def gt_missing(game, ro, arg):
    from cooksim_bench.exp_interact import recipe_of
    cell = getattr(ro, "cell", None) or getattr(game, "cell", None)
    rec = recipe_of(cell) if cell else None
    # E6 (round-2 review): the CURRENT goal — the recipe plus every announced plan change (an extra cheese is still
    # missing after the first cheese lands)
    from cooksim.interact.system import current_goal
    rec = current_goal(game, rec)
    if rec is None:
        return {"error": "no recipe on ro/game"}
    want = [n for n, _ in rec.contents]
    have, where = _plate_items(game)
    if where == "ambiguous":
        return {"recipe": want, "plate": "ambiguous", "unscorable": True,
                "answer": None, "why": "more than one plate in scene and none in hand"}
    have = list(have or [])
    left = list(want)
    for h in have:
        if h in left:
            left.remove(h)
    return {"recipe": want, "on_plate": have, "plate": where, "missing": left,
            "answer": ", ".join(left) if left else "nothing — the plate is complete"}


def gt_empty_boards(game, ro, arg):
    boards = [(tuple(pos), st) for pos, st in game.stations.items() if hasattr(st, "chop_time")]
    empty = [list(p) for p, st in boards if st.item is None]
    return {"boards": len(boards), "empty": len(empty), "empty_at": empty, "answer": str(len(empty))}


def gt_reask(game, ro, arg):
    from cooksim_bench.suite.gen_intent_cases import ground_truth
    now = ground_truth(arg, game, ro)
    hist = [h for h in (getattr(ro, "qa_history", None) or []) if h.get("kind") == arg]
    if not hist:
        return {"kind": arg, "now": now, "earlier": None, "answer": now.get("answer") if isinstance(now, dict) else now,
                "changed": "no earlier ask recorded"}
    prev = hist[-1]
    return {"kind": arg, "now": now, "earlier": prev.get("gt"), "earlier_t": prev.get("t"),
            "changed": "yes" if now != prev.get("gt") else "no",
            "answer": (now.get("answer") if isinstance(now, dict) else now)}


def gt_after_ref(game, ro, arg):
    seq = list(getattr(ro, "plate_seq", None) or [])
    if not seq:                                   # design-time replay: rebuild from the log
        for l in _done(ro):
            sid = _sid_base(l.get("kind"))
            if sid.startswith("assemble(") and l.get("cmd") in ("scoop", "put down"):
                seq.append({"item": sid[9:-1], "t": l["t_done"], "state": None})
    idx = [i for i, e in enumerate(seq) if e.get("item") == arg]
    if not idx:
        return {"ref": arg, "answer": f"{arg} is not on the plate yet, so nothing came after it"}
    i = idx[0] + 1
    if i >= len(seq):
        return {"ref": arg, "referent": None, "answer": f"nothing has been plated after the {arg} yet"}
    ref = seq[i]
    state = ref.get("state")
    if state is None:                             # read the live item state off the plate
        items, _ = _plate_items(game)
        rs = game.render_state() if hasattr(game, "render_state") else {}
        for p in (rs.get("players") or []):
            d = p.get("holding")
            if d and d.get("kind") == "plate":
                for x in d.get("contents", []):
                    if x.get("name") == ref["item"]:
                        state = x.get("state")
    return {"ref": arg, "referent": ref["item"], "referent_t": ref.get("t"), "state": state,
            "answer": f"the {ref['item']} ({state})"}


def gt_assistant_said(game, ro, arg):
    """The assistant's own last line on the topic. The line currently being graded must NOT be in
    the transcript yet, or the question grades itself (audit F9: 1 case in 136). The rollout
    computes this BEFORE appending the answer; the guard makes that ordering explicit."""
    p = _pat(arg)
    lines = [c for c in (getattr(ro, "convo", None) or []) if str(c.get("speaker", "")).startswith("vlm") and c.get("text")]
    pending = getattr(ro, "_grading_answer", None)
    if pending:
        lines = [c for c in lines if (c.get("text") or "").strip() != pending.strip()]
    hits = [c for c in lines if p.search(c["text"])]
    if not hits:
        return {"topic": arg, "said": None, "assistant_lines_so_far": len(lines), "answer": f"nothing — the assistant has not mentioned the {arg}"}
    h = hits[-1]
    return {"topic": arg, "said": h["text"], "t": h.get("t"), "answer": h["text"]}


def gt_next_step(game, ro, arg):
    """The next thing the ORDER still needs, from the recipe and the live plate/prep state (qa_world.next_step_truth).
    Never the cook's command queue: under the self-planning cook that queue held "vlm_fix:*" tags or a takeover's
    internal label ("omit_skip(tomato)", "lost"), which leaked into the gold answer (17/17 recorded asks wrong)."""
    from cooksim_bench.scoring.qa_world import next_step_truth, recipe_for_cell
    rec = getattr(ro, "recipe", None) if ro is not None else None
    if rec is None or not hasattr(rec, "contents"):
        rec = recipe_for_cell((getattr(ro, "cell", None) if ro is not None else None) or getattr(game, "cell", None))
    from cooksim.interact.system import current_goal
    rec = current_goal(game, rec)                     # E6: recipe + applied plan changes
    if rec is None:
        return {"error": "no recipe on ro/game"}
    rs = game.render_state() if hasattr(game, "render_state") else game
    return next_step_truth(rs, rec)


def gt_burn_first(game, ro, arg):
    lit = [(_burn_at(st) - st.timer, typ, list(pos), _contents(st)) for pos, st, typ in _stations(game) if _lit(st) and _contents(st)]
    if not lit:
        return {"lit": [], "answer": "nothing is on any heat"}
    lit.sort()
    return {"lit": [{"station": t, "at": p, "contents": c, "ticks_to_burn": n} for n, t, p, c in lit],
            "answer": f"the {lit[0][1]} with {'/'.join(lit[0][3])} — {lit[0][0]} ticks to burnt"}


def gt_ready_in(game, ro, arg):
    """Ticks until the item is cooked. Several loads of it (two potatoes in two fryers) are all reported; the answer is
    the load still COOKING that finishes first (audit X5/A6, 2026-09-19: the first station found used to win, so a
    potato done 30 ticks earlier made the truth "already done" while the one the cook meant was still frying)."""
    loads = []
    for pos, st, typ in _stations(game):
        if arg in _contents(st):
            loads.append({"station": typ, "at": list(pos), "lit": _lit(st), "status": st.status,
                          "ticks_left": max(0, st.cook_time - st.timer) if _lit(st) else None})
    if not loads:
        return {"item": arg, "answer": f"{arg} is not cooking anywhere"}
    cooking = sorted((l for l in loads if l["lit"] and l["status"] == "cooking"), key=lambda l: l["ticks_left"])
    lit = [l for l in loads if l["lit"]]
    if not lit:
        l = loads[0]
        return {"item": arg, "station": l["station"], "loads": loads, "answer": f"the {l['station']} with the {arg} is not on"}
    l = cooking[0] if cooking else lit[0]
    left = l["ticks_left"]
    out = {"item": arg, "station": l["station"], "at": l["at"], "ticks_left": left, "status": l["status"],
           "answer": f"{left} ticks" if left else f"already done ({l['status']})"}
    if len(loads) > 1:
        out["loads"] = loads
        out["note"] = f"{len(loads)} loads of {arg}; the answer is the one still cooking that finishes first"
    return out


# ------------------------------------------------------------------ Q10: the proactive alert
# Rebuilt 2026-08-28 after the grading audit. The old grader credited any line containing the
# item's name and any ready-word, took the FIRST such line, and ended the window when the cook
# removed the food. Measured consequences on the 2026-08 sweep: 205 of 247 "early" verdicts were
# the assistant ACKNOWLEDGING the request ("I'll let you know the second the meat is done"),
# 15% of credited alerts were lines DENYING readiness ("the steak is not done yet" scored 1.0),
# a line about one item satisfied another item's request, and a fast cook shrank the window to
# as little as 2 ticks. All four are fixed here:
#   * the window is the item's physical safe period (done -> burnt), never the cook's removal;
#   * a candidate must ASSERT that this item is ready NOW: promises and denials are excluded;
#   * the EARLIEST genuine alert is scored, not the first keyword match;
#   * a removal before any alert was possible is `pre_empted` (unscored), never `missed`.
# Timing is read from the ENGINE (qa_world.station_timing: GameConfig + the pace preset). A hand-kept copy here
# said fryer 45 for eight days after the engine moved every station to 60 (D-10, 2026-09-11), so potato and
# fish windows closed 15 ticks before the food could catch fire (audit A6, 2026-09-19).
def _timing(station_type, pace=None):
    from cooksim_bench.scoring.qa_world import station_timing
    return station_timing(station_type, pace)

# an assertion that the thing is ready NOW
ALERT_RE = re.compile(r"\b(is|are|'?s|just|now)\b[^.!?]{0,40}\b(done|ready|cooked|finished|up)\b"
                      r"|\b(done|ready|cooked|finished)\s*(!|now|already)"
                      r"|\b(scoop|take|pull|get)\b[^.!?]{0,30}\b(out|off|up|onto|on to)\b"
                      r"|\boff the (heat|pan|fryer|pot|oven|stove)\b", re.I)
# an unambiguous "it is ready RIGHT NOW" marker: it outranks a promise phrase in the same clause
NOW_RE = re.compile(r"\b(just (finished|done|cooked|came off)|is (done|ready|cooked) now|now (done|ready|cooked)"
                    r"|scoop (it|them|that)|take (it|them|that) (off|out)|pull (it|them) (off|out)"
                    r"|done!|ready!)\b", re.I)
# a promise to alert later, or a denial of readiness — never an alert
PROMISE_RE = re.compile(r"\b(i'?ll|i will|will let you know|let you know|i'?ll (tell|call|shout|ping|say)|"
                        r"as soon as|the moment|when it'?s? (done|ready)|keep an eye|watching it|monitor)\b", re.I)
DENY_RE = re.compile(r"\b(not (yet|done|ready|cooked|finished)|isn'?t (done|ready|cooked)|still (cooking|filling|going|needs|has)|"
                     r"half (done|way)|almost (done|ready)|nowhere near|no (cooked|ready)|nothing is (done|cooked|ready)|"
                     r"needs? (more|another|longer)|a (bit|little) longer|not on the heat)\b", re.I)


def ready_horizon(ev, station_type=None, pace=None):
    """The last tick at which "it is ready" is still the ASSISTANT'S news: the engine's own burn tick if the
    load burnt first, else done + smoke_grace, the tick the food starts smoking. NEVER the removal tick — the
    assistant must not be scored on how fast the cook happened to scoop.

    Ruling 2026-09-23 (user). The horizon used to run to done + grace + burn_time = done + 85 at tight pace,
    four to six times longer than cooking the food took (fryer 14, pan 15, pot 20). Measured on the formal run's
    470 credited alerts: 38% arrived after the food was already smoking and 22% arrived after the cook had
    already taken it off the heat, and both scored as successful alerts. Once the pan is smoking the kitchen has
    told the cook itself, so the window closes there. Pooled Q10 falls 0.43 -> 0.27 and the gap between the best
    seat and the rest roughly doubles; the seat ordering is unchanged (rho = 0.98).
    COOKSIM_Q10_HORIZON=burn restores the old horizon for reproducing pre-ruling numbers."""
    _cook, grace, burn = _timing(ev.get("station") or station_type, pace)
    end = ev["cooked_at"] + grace + (burn if os.environ.get("COOKSIM_Q10_HORIZON") == "burn" else 0)
    if ev.get("burnt_at") is not None:
        return min(ev["burnt_at"], end)
    return end


def bind_request(heat_events, item, asked, station_type=None, world=None, pace=None):
    """Which load of `item` a "tell me the moment the X is done" request is about, bound at the ASK.

    Two potatoes cooking in two stations were confused (audit A6, expert hard_3: the window was built on the potato
    the cook had scooped at t130, 49 ticks before the t179 ask; the potato the cook meant finished at t192). Rule:
      1. the load that was COOKING at the ask (lit, not yet done) -- the earliest to finish if several;
      2. else a load already done and still on the heat at the ask (the request is late, the alert is due now);
      3. else the next load to finish after the ask.
    The lit tick comes from the world history when available, else cooked_at - the station's cook time."""
    evs = []
    for e in heat_events or []:
        if item not in (e.get("items") or []) or e.get("cooked_at") is None:
            continue
        e = dict(e)
        typ = (world.station_type(e["pos"]) if (world is not None and e.get("pos") is not None) else None) or station_type
        e["station"] = typ
        cook, _g, _b = _timing(typ, pace)
        lit = world.lit_before(e["pos"], e["cooked_at"]) if (world is not None and e.get("pos") is not None) else None
        e["lit_at"] = lit if lit is not None else e["cooked_at"] - cook
        evs.append(e)
    cooking = [e for e in evs if e["lit_at"] <= asked < e["cooked_at"]]
    if cooking:
        return min(cooking, key=lambda e: e["cooked_at"]), "cooking at the ask"
    done_on = [e for e in evs if e["cooked_at"] <= asked and (e.get("removed_at") is None or e["removed_at"] > asked)
               and ready_horizon(e, pace=pace) > asked]
    if done_on:
        return min(done_on, key=lambda e: e["cooked_at"]), "already done and on the heat at the ask"
    later = [e for e in evs if e["cooked_at"] > asked]
    if later:
        return min(later, key=lambda e: e["cooked_at"]), "the next load to finish after the ask"
    return None, None


INGREDIENTS = ("tomato", "cheese", "bun", "meat", "fish", "potato", "steak", "rice", "tortilla",
               "lettuce", "lemon", "peas", "noodles", "shrimp", "butter", "pepper", "egg",
               "mushroom", "onion", "avocado", "corn", "dough", "chips", "fries", "patty")


def _about_this_item(seg, item):
    """Is the readiness cue in `seg` about `item`, or about a different ingredient named closer to
    it? A regex cannot resolve reference in general -- the offline judge does that -- but this
    rejects the common cross-item line ("stop checking the potato crate and scoop the fish out")."""
    mi = _pat(item).search(seg)
    ma = ALERT_RE.search(seg)
    if not mi or not ma:
        return False
    lo, hi = sorted((mi.start(), ma.start()))
    between = seg[lo:hi] + " " + ma.group(0)      # the cue's own words name the item it is about
    for other in INGREDIENTS:
        if other == item.lower() or other in item.lower() or item.lower() in other:
            continue
        if re.search(r"\b" + other + r"s?\b", between, re.I):
            return False
    return True


# STRICT alert (user, 2026-09-15: gpt-5.5 "Meat is done" 4 ticks early had scored in_window). Both the loose regex
# and the Gemini judge credited lines that never said the food was done: "If the pan's progress is full, scoop the
# cooked meat", "The moment the steak's progress bar is full, it's done", "wait for the second potato to finish",
# "Perfect—now go to the plate stack". An alert is now ONE clause that names the item (or the station's bar) and
# says, unconditionally and in the present tense, that it is done. Anything conditional, future, negated,
# imperative-only, or about "chopped" is not an alert.
# sentences and dash-separated clauses only: a ", then scoop it" clause inherits the "wait until" before it
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\s+[\u2014\u2013]\s*|\s*[\u2014\u2013]\s+|\s+-\s+", re.I)
_CLAIM_WORDS = r"(done|cooked|ready|finished|smoking)"
_BAR_FULL = re.compile(r"\b(bar|progress)\b[^.!?]{0,15}\b(is|'s|\u2019s)\s+(now\s+)?full\b", re.I)
# "so we can scoop the steak off the heat" inside a PROMISE ("I'll shout when the steak is done ... keep an ear out for
# my call") is not an alert: the scoop is embedded, not commanded (gpt-realtime-2.1 expert_medium_map_1, 2026-09-16).
# "make sure cooked fish is ready before we start assembling" asks for a state, it does not assert one; a clause that
# OPENS with then/finally/next is a step of a plan ("Once everything is cooked ... Then scoop the cooked steak onto the
# plate first"), not a call to act now (museglimmer / molmo2-8b, local pass@3, 2026-09-16).
_SEQ_OPEN = re.compile(r"^\s*(?:and\s+)?(?:then|finally|next|after that|afterwards|lastly)\b", re.I)
# "Before you start the potatoes, grab a clean plate ..., then scoop the cooked fish from the fryer" is a plan preface,
# not a call to act now -- the same message says "I don't see a cooked fish on the plate yet" (museglimmer, 2026-09-16).
_HEDGE = re.compile(r"\bbefore (?:you|we|that)\b|\b(make sure|makes sure|ensure|let'?s|let\u2019s|"
                    r"so (?:we|you|i) can|keep an ear|ear out|call out|shout|let you know|"
                    r"if|when|whenever|once|moment|instant|as soon as|wait for|wait until|wait to|until|till|will|going to|after|afterwards|later|"
                    r"about to|almost|nearly|should be|not|never|no|still|yet|soon|in a (few|bit|moment))\b|n't\b|n\u2019t\b|'ll\b|\u2019ll\b|\?", re.I)


def strict_alert(text, item):
    """True iff some clause of `text` asserts, unconditionally and now, that `item` is done."""
    if not text or not _pat(item).search(text):
        return False
    for cl in _CLAUSE_SPLIT.split(text):
        if not cl or _HEDGE.search(cl) or _SEQ_OPEN.search(cl):
            continue
        if _BAR_FULL.search(cl):
            return True
        # an unconditional order to take THIS item off the heat now ("Yes — scoop the rice onto the parked plate
        # now", "scoop the finished steak next") asserts it is done; scooping raw food from a crate does not
        for mi in _pat(item).finditer(cl):
            head, tail = cl[max(0, mi.start() - 40):mi.start()], cl[mi.end():mi.end() + 40]
            if re.match(r"^\s*(crate|source|box|bin)\b", tail, re.I) or re.search(r"\braw\b[^.!?]{0,20}$", head, re.I) or re.match(r"^\s*(s\s+)?(from|at|in)\s+the\s+[\w ]{0,15}?(crate|source|board)", tail, re.I):
                continue
            if (re.search(r"\bscoop\b(\s+\S+){0,4}\s+$", head, re.I)
                    or (re.search(r"\b(take|pull|lift)\b(\s+\S+){0,4}\s+$", head, re.I)
                        and re.match(r"^[^.!?]{0,25}\b(off|out)\b", tail, re.I))):
                return True
        for mi in _pat(item).finditer(cl):
            tail = cl[mi.end():mi.end() + 45]
            m = re.match(r"^[^.!?]*?\b(is|are|'s|\u2019s|looks|look|just|now)\b(\s+\w+){0,2}?\s+" + _CLAIM_WORDS + r"\b", tail, re.I)
            if m and not any(re.search(r"\b" + o + r"s?\b", tail[:m.end()], re.I)
                             for o in INGREDIENTS if o != item.lower() and o not in item.lower()):
                return True
    return False


def classify_alert_line(text, item):
    """Deterministic fallback classifier: alert | promise | denial | other.
    The offline re-grade (python -m cooksim_bench.scoring.regrade_proactive_v2) replaces this with a judge, which is
    what settles polarity and which item a claim is about; this keeps the live path honest."""
    if not text or not _pat(item).search(text):
        return "other"
    # the clause that mentions the item decides: a line may name two items
    clauses = re.split(r"(?<=[.!?;])\s+", text)
    mine = [c for c in clauses if _pat(item).search(c)] or [text]
    seg = " ".join(mine)
    if DENY_RE.search(seg):
        return "denial"
    if PROMISE_RE.search(seg) and not NOW_RE.search(seg):
        return "promise"
    if strict_alert(text, item):
        return "alert"
    return "other"


def grade_proactive(qa, item, comm, heat_events, end_tick, station_type=None, classify=None, world=None, pace=None):
    """Q10: was the alert spoken inside the item's ready window [done, burnt)?
    Score 1 the moment it is done, decaying linearly to 0 at the burn horizon; 0 when the
    assistant asserts readiness before the item is done, or only after the horizon, or never.
    `classify(text, item) -> alert|promise|denial|other` overrides the regex fallback.
    The load is bound at the ask (bind_request); `world` (qa_world.WorldTrack) gives each load's station and lit
    tick, `pace` the burn preset (default: the engine's tight physics)."""
    asked = qa.get("asked_at")
    if asked is None:
        return {"verdict": "not_asked", "score": None}
    item = item or ""
    classify = classify or classify_alert_line
    ev, rule = bind_request(heat_events, item, asked, station_type=station_type, world=world, pace=pace)
    if ev is None:
        return {"verdict": "no_window", "score": None,
                "why": f"{item} never reached done after the request"}
    start, end = max(ev["cooked_at"], asked), ready_horizon(ev, pace=pace)
    removed = ev.get("removed_at")
    lines = [c for c in (comm or []) if str(c.get("speaker", "")).startswith("vlm")
             and c.get("t") is not None and c["t"] >= asked and c.get("text")]
    kinds = [(c["t"], classify(c["text"], item), c["text"]) for c in lines if _pat(item).search(c["text"])]
    alerts = [(t, txt) for t, k, txt in kinds if k == "alert"]
    base = {"window": [start, end], "removed_at": removed, "bound": {"pos": ev.get("pos"), "station": ev.get("station"),
                                                                     "lit_at": ev.get("lit_at"), "cooked_at": ev["cooked_at"],
                                                                     "rule": rule},
            "candidates": len(kinds), "promises": sum(1 for _, k, _ in kinds if k == "promise"),
            "denials": sum(1 for _, k, _ in kinds if k == "denial")}
    in_win = [(t, txt) for t, txt in alerts if start <= t < end]
    early = [(t, txt) for t, txt in alerts if t < ev["cooked_at"] - 2]
    # A FALSE "it's done" before the food was done voids the request (user, 2026-09-15). The grader used to
    # look for an in-window alert first and return it, so an assistant that called the meat done 4 ticks
    # early, was contradicted by the cook, and then corrected itself on the tick the meat came off the heat
    # scored in_window 0.976 (gpt-5.5, classic_expert, b3_medium_nops_hard_map_1). A wrong early call is the
    # failure this request exists to catch; a later correction does not undo it.
    if early:
        t_a, line = min(early)
        return {"verdict": "early", "score": 0.0, "alert_t": t_a, "line": line[:200],
                **({"later_in_window_t": min(in_win)[0]} if in_win else {}), **base}
    if in_win:
        t_a, line = min(in_win)
        return {"verdict": "in_window", "score": round(1.0 - max(0, t_a - start) / max(1, end - start), 3),
                "alert_t": t_a, "line": line[:200], **base}
    late = [(t, txt) for t, txt in alerts if t >= end]
    # a cook that scooped the food before the assistant could ever have alerted pre-empts the test
    if removed is not None and removed <= ev["cooked_at"] + 2 and not in_win:
        return {"verdict": "pre_empted", "score": None,
                "why": f"{item} was taken off the heat at t{removed}, before an alert was possible", **base}
    if late:
        t_a, line = min(late)
        return {"verdict": "late", "score": 0.0, "alert_t": t_a, "line": line[:200], **base}
    return {"verdict": "missed", "score": 0.0, "why": "no line asserted the item was ready", **base}


GT_PARAM_V3 = {"ready": gt_ready_in, "order": gt_order, "happened": gt_happened, "exists": gt_exists, "count": gt_count,
               "elapsed": gt_elapsed, "longest_on": gt_longest_on, "past_done": gt_past_done,
               "missing": gt_missing, "empty_boards": gt_empty_boards, "reask": gt_reask,
               "after_ref": gt_after_ref, "assistant_said": gt_assistant_said, "next_step": gt_next_step,
               "burn_first": gt_burn_first, "ready_in": gt_ready_in}

# families (docs/SUITE_V3_DESIGN.md §3) for suite balancing and per-family judge rubrics
FAMILY = {"ready": "Q10", "order": "Q1", "count": "Q2", "elapsed": "Q3", "longest_on": "Q3", "past_done": "Q3",
          "happened": "Q4", "exists": "Q4", "missing": "Q5", "empty_boards": "Q5", "reask": "Q6",
          "after_ref": "Q7", "assistant_said": "Q8", "next_step": "Q9", "burn_first": "Q9", "ready_in": "Q9"}

# what the judge must insist on, per family (python -m cooksim_bench.scoring.regrade_qa appends this to the prompt)
RUBRIC = {
    "Q1": "Grade the ORDER word only: before / after, or an explicit statement that one or both events have not happened. The events are a_words / b_words (e.g. 'chop the tomato', 'put the fish on the heat'). A wrong relation is incorrect; asserting an order when the truth says 'neither'/'only one happened' is incorrect.",
    "Q2": "Grade the INTEGER: it must match exactly (off by one is incorrect). 'A few' or a range is incorrect.",
    "Q3": "Grade the duration/comparative: the named station must match; a stated tick/second count must be within 5 ticks (6 seconds) of the truth; 'done for a while' matches only if the truth says it is past done.",
    "Q4": "Grade the yes/no against 'answer': 'yes' means the event happened (for a visit: the cook walked over to a station of that kind at least once, 'visits' times), 'no' means it never did (or the thing does not exist). The opposite claim, a hedge ('I think so', 'I can't confirm'), or an invented detail that contradicts the truth is incorrect.",
    "Q5": "Grade the SET: every missing item must be named and nothing extra; when the truth is 'nothing', the answer must say the plate is complete / nothing is missing.",
    # Q5 covers two question kinds with different answer shapes: `missing:` asks for a SET of plate items, while
    # `empty_boards:` asks HOW MANY boards are free. Grading a count answer against the set rubric ("every missing
    # item must be named") is the wrong instruction for 446 of the 485 Q5 answers in the formal run, so the kind
    # gets its own rubric (2026-09-23).
    "Q5/empty_boards": "Grade the INTEGER of free (unused) cutting boards: it must match the truth exactly, off by one is incorrect. 'A couple', 'a few' or a range is incorrect. Naming which boards are busy is extra detail that neither helps nor hurts unless it contradicts the truth.",
    "Q6": "Grade the CURRENT answer and the change statement: if 'changed' is yes the answer must say what changed; if no, it must say nothing changed (or give the same answer as before).",
    "Q7": "The answer must resolve the referent to the right item AND give its state; naming a different item is incorrect even if the state word happens to match.",
    "Q8": "Grade against 'said': the assistant must recall its own earlier line in substance. If 'said' is null the only correct answer is that it said nothing about it; recalling a line it never said is incorrect.",
    "Q9": "Grade the predicted step / station / ticks: the station must match; ticks within 5 of the truth. For 'what is my next step': the answer is correct if it names the truth's next step ('answer' / next_subgoal) OR any step listed in also_acceptable (each is something the order still needs that can be done now); naming a step the order no longer needs, one that cannot be done yet (e.g. plating food that is still cooking when other work remains), or the wrong item is incorrect.",
}


def family_of(kind):
    return FAMILY.get(kind.split(":", 1)[0])


# ------------------------------------------------------------------ content hardness (2026-08-27 ruling)
# A hard FAMILY is not enough: the instance must be hard. Each predicate looks at the replayed trace at
# the ask tick and returns (ok, score, why). Rejected = an assistant could answer from recipe knowledge,
# from the current frame alone, or the question is about something that just happened.
# Ruling 2026-08-28: Q4 (never-happened / does-not-exist traps) is RETIRED. Measured on the suite-v3
# sweep it ran at 0.90 mean accuracy against 0.49 for every other family pooled, and three of the nine
# assistants scored >= 0.94 on it: "no, that never happened" is the safe answer to a question that
# announces its own answer. It is no longer generated (python -m cooksim_bench.suite.design_composite_v3) and no longer
# scored (every QA consumer filters RETIRED_FAMILIES); instances already asked in the 2026-08 sweep
# stay in the transcripts and are simply not counted.
# Ruling R3 (2026-09-19, pre-formal-run audit): the Q4 VISIT questions ("have I been over to the pot?") are KEPT
# and SCORED. Their truth was the defect (a regex on command words; "go to board 1" never matched), not the family;
# it is now read off the world (qa_world.visits). No family is retired; every QA consumer reads this one set.
RETIRED_FAMILIES = frozenset()


def scored_family(fam) -> bool:
    """True if this question family counts toward QA accuracy."""
    return bool(fam) and fam not in RETIRED_FAMILIES


# THE QA SCORE, one definition for every consumer (eval_v2_sweep, pass3_summary_table, build_metrics_page, paper_tables,
# paper_figures). Before 2026-09-19 they disagreed: the pages dropped Q4 while eval_v2_sweep and pass3 kept it,
# paper_tables/metrics scored `asked_too_late` as 0 while eval_v2_sweep left it unscored, and pass3 reported the
# share graded `correct` (partial = 0) instead of the published QA (partial = 0.5).
QA_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0, "evasive": 0.0, "no_answer": 0.0}
QA_UNSCORED = frozenset({"ungradable", "asked_too_late", "judge_error", None})


def qa_score(verdict, family=None):
    """Score of one graded question under the published QA metric, or None when it does not count
    (unscored verdict, or a retired family). QA = mean of the non-None scores."""
    if family is not None and family in RETIRED_FAMILIES:
        return None
    return QA_SCORE.get(verdict)


MEMORY_FAMILIES = ("Q1", "Q2", "Q4", "Q7", "Q8")
MIN_HISTORY = 25          # ticks between the referenced event and the ask ("the past should be long history")
MIN_PLAN_FRACTION = 0.35  # memory questions are asked in the middle of the plan, never at the start


def _dep(a, b):
    """True when the order of a and b is fixed by the recipe / plan structure (not a memory question)."""
    import re
    ma, mb = re.match(r"^(\w+)(?:\((.*)\))?$", a), re.match(r"^(\w+)(?:\((.*)\))?$", b)
    va, xa, vb, xb = ma.group(1), ma.group(2), mb.group(1), mb.group(2)
    if "serve" in (va, vb) or "fetch_plate" in (va, vb):
        return True                                       # plate before plating, serve last: recipe logic
    if xa == xb and {va, vb} <= {"prep", "ignite", "assemble"}:
        return True                                       # prep(X) < ignite(X) < assemble(X)
    if va == "assemble" and vb == "assemble":
        return "ordered"                                  # fixed on ordered dishes only (caller checks)
    if va == "visit" or vb == "visit":
        return True                                       # visits repeat; first-occurrence order is a riddle, not memory
    return False


def hardness(kind, gt, ro, tick, ref_ticks, ordered=False, first_error_m=None, m=None):
    base = kind.split(":", 1)[0]
    fam = FAMILY.get(base)
    if not isinstance(gt, dict) or gt.get("error"):
        return False, 0.0, "undefined"
    frac = (tick / ref_ticks) if ref_ticks else 1.0
    if fam in MEMORY_FAMILIES:
        if frac < MIN_PLAN_FRACTION:
            return False, 0.0, f"asked too early ({frac:.0%} of the plan)"
        if first_error_m is not None and m is not None and m <= first_error_m:
            return False, 0.0, "memory question before the first error"
    if base == "order":
        d = _dep(gt["a"], gt["b"])
        if d is True or (d == "ordered" and ordered):
            return False, 0.0, "order fixed by the recipe"
        # audit 2026-09-19 (T1): "did I chop the tomato before I put the cheese on?" on Caprese -- the tomato must be
        # processed and PLATED before the cheese on an ordered dish, so the answer never depends on the episode
        ma, mb = re.match(r"^(\w+)\((\w+)\)$", gt["a"]), re.match(r"^(\w+)\((\w+)\)$", gt["b"])
        if ordered and ma and mb and ma.group(1) in ("prep", "chop", "heat", "ignite", "cook") and mb.group(1) == "assemble":
            from cooksim_bench.scoring.qa_world import recipe_for_cell
            rec = recipe_for_cell(getattr(ro, "cell", None))
            items = [n for n, _ in rec.contents] if rec is not None else []
            if ma.group(2) in items and mb.group(2) in items and items.index(ma.group(2)) < items.index(mb.group(2)):
                return False, 0.0, "order fixed by the recipe (processed and plated before the other item)"
        ta, tb = gt.get("t_a"), gt.get("t_b")
        if ta is not None and tb is not None:
            gap, recency = abs(ta - tb), tick - max(ta, tb)
            if gap < 8:
                return False, 0.0, f"events {gap} ticks apart (too close to be memorable)"
            if recency < MIN_HISTORY:
                return False, 0.0, f"later event only {recency} ticks ago"
            return True, min(1.0, recency / 80) + min(0.5, gap / 60), f"both happened; {recency} ticks ago, {gap} apart"
        t1 = ta if ta is not None else tb
        if t1 is None:
            return False, 0.0, "neither happened — nothing to remember"
        if tick - t1 < MIN_HISTORY:
            return False, 0.0, f"the one event only {tick - t1} ticks ago"
        return True, 0.6 + min(0.4, (tick - t1) / 100), f"trap: one event {tick - t1} ticks ago, the other never"
    if base == "count":
        n = gt.get("count", 0)
        if n < 2:
            return False, 0.0, f"count {n} (trivial)"
        ts = list(gt.get("visit_ticks") or [])                   # world-resolved visits (gt_count)
        span, recency = (max(ts) - min(ts)) if ts else 0, (tick - max(ts)) if ts else 0
        if span < 30:
            return False, 0.0, f"visits within {span} ticks (one burst)"
        if recency < 10:
            return False, 0.0, f"last visit {recency} ticks ago (visible now)"
        return True, min(1.0, n / 5) + min(0.5, span / 120) + (0.3 if gt.get("distinct_targets", 1) > 1 else 0), f"{n} visits over {span} ticks, {gt.get('distinct_targets')} targets"
    if base == "elapsed":
        if not gt.get("lit"):
            return False, 0.0, "item not on a lit station"
        if gt.get("ticks_on", 0) < 20:
            return False, 0.0, f"on heat only {gt.get('ticks_on')} ticks"
        return True, min(1.0, gt["ticks_on"] / 60) + (0.3 if gt.get("ticks_since_done") else 0), f"{gt['ticks_on']} ticks on heat"
    if base == "longest_on":
        lit = gt.get("lit") or []
        if len(lit) < 2:
            return False, 0.0, f"{len(lit)} lit station(s) — no comparison"
        gap = lit[0]["ticks_on"] - lit[1]["ticks_on"]
        if gap < 6:
            return False, 0.0, f"stations {gap} ticks apart (a tie to the eye)"
        return True, 0.6 + min(0.6, gap / 40), f"{len(lit)} lit, {gap} ticks apart"
    if base == "past_done":
        lit = [st for _, st, _ in _stations(ro.game) if _lit(st) and _contents(st)]
        if not lit:
            return False, 0.0, "nothing on heat"
        rows = gt.get("done_on_heat") or []
        if rows:
            margin = min(r["ticks_since_done"] for r in rows)
            if margin < 8:
                return False, 0.0, f"done only {margin} ticks ago"
            return True, 0.6 + min(0.6, margin / 40), f"done {margin} ticks ago and still on"
        margin = min(st.cook_time - st.timer for st in lit if st.status == "cooking") if any(st.status == "cooking" for st in lit) else 0
        if margin < 4:
            return False, 0.0, "about to be done (ambiguous)"
        return True, 0.5 + min(0.3, margin / 40), f"no; nearest is {margin} ticks from done"
    if base == "happened":
        ev = gt.get("event", "")
        if gt.get("happened"):
            age = tick - (gt.get("t") or tick)
            if age < MIN_HISTORY:
                return False, 0.0, f"happened {age} ticks ago (too recent)"
            return True, 0.5 + min(0.6, age / 100), f"yes, {age} ticks ago"
        # a trap must be plausible: a recipe event still to come, or a station the map really has
        if ev.startswith("visit("):
            k = ev[6:-1]
            ex = gt_exists(ro.game, ro, k)
            if not ex.get("exists"):
                return False, 0.0, f"no {k} on this map (that is an exists: trap, not a memory trap)"
            if frac < 0.5:
                return False, 0.0, "visit trap asked before half the plan"
            return True, 0.7 + min(0.4, frac), f"never visited the {k}, asked at {frac:.0%}"
        done_any = [l["t_done"] for l in _done(ro) if str(l.get("kind", "")).startswith(("assemble(", "ignite("))]
        if not done_any or tick - max(done_any) < 5:
            return False, 0.0, "no comparable event yet / one just happened"
        return True, 0.7 + min(0.4, frac), f"not yet ({ev} is still to come), asked at {frac:.0%}"
    if base == "exists":
        if gt.get("exists"):
            return False, 0.0, "it exists — answerable from the frame"
        if frac < 0.3:
            return False, 0.0, "asked at the start"
        return True, 0.6 + min(0.4, frac), f"no {gt.get('kind')} on this map, asked at {frac:.0%}"
    if base == "missing":
        on, left = gt.get("on_plate") or [], gt.get("missing") or []
        if not on and left:
            return False, 0.0, "empty plate — the answer is the whole recipe"
        if left and len(left) < 2 and len(on) < 2:
            return False, 0.0, "one item on, one to go (visible at a glance)"
        return True, 0.5 + 0.15 * len(on) + 0.15 * len(left), f"{len(on)} on, {len(left)} missing"
    if base == "empty_boards":
        n, e = gt.get("boards", 0), gt.get("empty", 0)
        if n < 2 or e in (0, n):
            return False, 0.0, "all boards alike"
        return True, 0.5 + 0.2 * n, f"{e} of {n} boards empty"
    if base == "reask":
        if gt.get("changed") != "yes":
            return False, 0.0, "nothing changed since the first ask"
        return True, 1.0, "truth changed since the first ask"
    if base == "after_ref":
        if not gt.get("referent"):
            return False, 0.0, "no item after the referent yet"
        seq = list(getattr(ro, "plate_seq", None) or [])
        items = [l for l in _done(ro) if str(l.get("kind", "")).startswith("assemble(") and l.get("cmd") in ("scoop", "put down")]
        if len(items) < 3:
            return False, 0.0, f"only {len(items)} items plated (the referent is the last thing that happened)"
        age = tick - (gt.get("referent_t") or tick)
        if age < 15:
            return False, 0.0, f"referent plated {age} ticks ago"
        return True, 0.6 + min(0.6, age / 60) + 0.1 * len(items), f"{len(items)} on plate, referent {age} ticks ago"
    if base == "assistant_said":
        if frac < 0.5:
            return False, 0.0, "asked before half the plan (little to remember)"
        return True, 0.5 + frac, f"asked at {frac:.0%} of the plan (live-only truth)"
    if base == "next_step":
        nxt = gt.get("next_subgoal")
        if not nxt or nxt == "serve":
            return False, 0.0, "plan finished / only the serve left"
        return True, 0.5 + (0.3 if gt.get("then") else 0), f"next {nxt}, then {gt.get('then')}"
    if base == "burn_first":
        lit = gt.get("lit") or []
        if len(lit) < 2:
            return False, 0.0, f"{len(lit)} lit station(s)"
        gap = lit[1]["ticks_to_burn"] - lit[0]["ticks_to_burn"]
        if gap < 5:
            return False, 0.0, f"{gap} ticks apart (a coin flip)"
        return True, 0.6 + min(0.6, gap / 40), f"{len(lit)} lit, {gap} ticks apart"
    if base == "ready_in":
        left = gt.get("ticks_left")
        if left is None or left < 5:
            return False, 0.0, "already done / about to be"
        return True, 0.5 + min(0.5, left / 20), f"{left} ticks to go"
    return False, 0.0, "no hardness rule"
