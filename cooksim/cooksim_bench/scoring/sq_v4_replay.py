#!/usr/bin/env python3
"""Rebuild the per-tick kitchen state of a recorded episode by replaying its command log in the engine.

The sweep records keep the cook's every command with issue tick `t` and finish tick `t_drained`
(`kitchen_feedback`), the plate history (`plate_seq`), the heat events, the flags with verdicts, the
sub-case archive and the rounds — but not the stations' contents and timers per tick. The engine is
deterministic given the command sequence and its timing, so re-issuing each command at its recorded
tick (idling in between exactly as the cook did) reproduces the kitchen. Fidelity is checked against
the recorded plate_seq and heat_events; a mismatch is reported, never hidden.

    from cooksim_bench.scoring.sq_v4_replay import replay_episode
    rep = replay_episode(record)      # rep["states"][t] -> stamp dict; rep["fidelity"] -> {...}
"""
import copy
import json
import os
from cooksim_bench.paths import ROOT
import re
import sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.exp_interact import build_episode, recipe_of                 # noqa: E402
from cooksim.core.plan_ir import segment_plan                     # noqa: E402
from cooksim.core.config import apply_pace                        # noqa: E402
from cooksim_bench.scoring import qa_v3_lib as Q  # noqa: E402


_CASES = None


def cell_of(record):
    """The harness cell name, from the case pool (the record's `setting` is a display label)."""
    global _CASES
    if _CASES is None:
        _CASES = {}
        for pf in (os.path.join(ROOT, "benchmark", "cases_composite_v5.json"), "tmp/interact_eval/cases_composite_v3.json"):
            try:
                for c in json.load(open(pf))["cases"]:
                    _CASES[c["name"]] = (c["cell"], c.get("inject", True))
            except Exception:
                pass
    name = record.get("scenario") if isinstance(record.get("scenario"), str) else (record.get("scenario") or {}).get("name")
    name = name or record.get("case") or record.get("_name")
    if name in _CASES:
        return _CASES[name]
    m = re.match(r"^(?:bm|b3)_(.+)_\d+$", str(name or ""))
    cell = m.group(1) if m else None
    return cell, (not str(cell).endswith("hard_map"))


def stamp_of(game, tick):
    """The kitchen at one tick, in the terms the rubric uses."""
    rs = game.render_state()
    stations = []
    for pos, st in game.stations.items():
        if hasattr(st, "cook_time"):
            typ = next((d.get("type") for d in (rs.get("stations") or []) if (d.get("x"), d.get("y")) == tuple(pos)), type(st).__name__.lower())
            names = Q._contents(st)
            lit = Q._lit(st)
            stations.append({"station": typ, "at": list(pos), "contents": names, "status": st.status if names else "empty", "lit": lit,
                             "ticks_on": st.timer if lit else None,
                             "ticks_to_done": max(0, st.cook_time - st.timer) if (lit and st.status == "cooking") else (0 if names and st.status in ("cooked", "burnt") else None),
                             "ticks_to_burn": (Q._burn_at(st) - st.timer) if (lit and names) else None})
        elif hasattr(st, "chop_time"):
            it = st.item
            stations.append({"station": "cutting board", "at": list(pos), "contents": [it.name] if it else [], "status": (str(getattr(it, "state", "")).split(".")[-1].lower() if it else "empty")})
    plate, where = None, None
    for p in (rs.get("players") or []):
        d = p.get("holding")
        if d and d.get("kind") == "plate":
            plate, where = [x["name"] + f"({x.get('state')})" for x in d.get("contents", [])], "in hand"
    if plate is None:
        for o in (rs.get("objects") or []):
            d = o.get("item")
            if d and d.get("kind") == "plate":
                plate, where = [x["name"] + f"({x.get('state')})" for x in d.get("contents", [])], f"on the counter at {o.get('x')},{o.get('y')}"
    hold = game.cooks[0].holding
    hands = None
    if hold is not None:
        hd = hold.to_dict() if hasattr(hold, "to_dict") else {}
        hands = "a plate" if hd.get("kind") == "plate" else f"{hd.get('name')} ({str(hd.get('state')).split('.')[-1].lower()})"
    return {"tick": tick, "stations": stations, "plate": plate, "plate_where": where, "hands": hands or "empty",
            "cook_at": list(game.cooks[0].pos)}


def replay_episode(record, max_ticks=None):
    cell, inject = cell_of(record)
    state, ro = build_episode(None, "capable_collaborator", cell=cell, inject=inject)
    os.environ["COOKSIM_PACE"] = record.get("pace", "human")
    apply_pace(ro.game)
    ge = record.get("goal_extended")
    if ge:
        from cooksim.core.lang_events import extend_goal
        try:
            extend_goal(ro.game, ge["ingredient"], ge.get("state", "raw"))
        except Exception:
            pass
    log = sorted([l for l in (record.get("kitchen_feedback") or []) if l.get("cmd")], key=lambda l: (l.get("t") or 0))
    # commands in issue order; idle between them so each is issued at its recorded tick
    cmds, last_t = [], 0
    for l in log:
        t = l.get("t") or 0
        if t > last_t:
            cmds.append((f"idle {t - last_t}", None))
        cmds.append((l["cmd"], None))
        last_t = max(last_t, l.get("t_drained") or t)
    ro.cmds = cmds
    ro.cap = max_ticks or (record.get("ticks") or 600) + 5
    states, plate_seen, plate_prev = {}, [], []
    T = record.get("ticks") or 600
    states[0] = stamp_of(ro.game, 0)
    while ro.tick < T and (ro.i < len(ro.cmds) or ro.ctrl.skill is not None):
        if not ro.step():
            break
        states[ro.tick] = stamp_of(ro.game, ro.tick)
        cur = states[ro.tick]["plate"] or []
        names = [c.split("(")[0] for c in cur]
        if len(names) > len(plate_prev) and names[:len(plate_prev)] == plate_prev:
            for n in names[len(plate_prev):]:
                plate_seen.append({"item": n, "t": ro.tick})
        if len(names) >= len(plate_prev) or not names:
            plate_prev = names if names else plate_prev
    # fidelity against the record
    rec_plate = [(e["item"], e["t"]) for e in (record.get("plate_seq") or [])]
    got_plate = [(e["item"], e["t"]) for e in plate_seen]
    heat_rec = sorted((tuple(sorted(h["items"])), h["cooked_at"]) for h in (record.get("heat_events") or []))
    fid = {"plate_recorded": rec_plate, "plate_replayed": got_plate,
           "plate_items_match": [i for i, _ in rec_plate] == [i for i, _ in got_plate],
           "plate_tick_maxdiff": max([abs(a[1] - b[1]) for a, b in zip(rec_plate, got_plate)] or [0]) if len(rec_plate) == len(got_plate) else None,
           "ticks_replayed": ro.tick, "ticks_recorded": T, "commands": len(log)}
    return {"cell": cell, "states": states, "fidelity": fid, "log": log, "game_final": ro.game}


def state_at(rep, t):
    """The stamp at tick t (the latest tick ≤ t that was stepped)."""
    ks = [k for k in rep["states"] if k <= t]
    return rep["states"][max(ks)] if ks else rep["states"][0]


if __name__ == "__main__":
    import glob
    files = sys.argv[1:] or sorted(glob.glob("tmp/assist_bench_v3/frames/baseline/*/b3_*.json"))[:6]
    for f in files:
        d = json.load(open(f)); d["_name"] = os.path.basename(f)[:-5]
        rep = replay_episode(d)
        fd = rep["fidelity"]
        print(f.split("/")[-2], f.split("/")[-1], "| plate items match:", fd["plate_items_match"], "| tick maxdiff:", fd["plate_tick_maxdiff"],
              "| replayed", fd["ticks_replayed"], "/", fd["ticks_recorded"], "ticks,", fd["commands"], "commands")
        if not fd["plate_items_match"]:
            print("    recorded:", fd["plate_recorded"], "\n    replayed:", fd["plate_replayed"])
