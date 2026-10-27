#!/usr/bin/env python3
"""Rebuild the per-tick render states of a recorded episode by re-issuing its ACTION LOG (`actions`: cmd, t_issue, t_done)
in the engine, and write them as the COOKSIM_DUMP_STATES JSONL that film_states_smooth.py consumes -- for runs recorded
without state dumps (rerun v6). Each command is issued on its recorded issue tick, never earlier and never before the
previous one finished (the cook's executor is serial); the cook idles in between exactly as it did. Fidelity: every
command's completion tick is compared with the record's, and the recorded plate sequence must be reproduced, else the
dump is refused. (kitchen_feedback's `t` is the COMPLETION tick, which is why replaying from it drifted.)

    python -m cooksim_bench.video.replay_states_dump <record.json> <out.states.jsonl>
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.scoring import sq_v4_replay as R  # noqa: E402
from cooksim_bench.exp_interact import build_episode                            # noqa: E402
from cooksim.core.config import apply_pace                        # noqa: E402

rec_path, out_path = sys.argv[1], sys.argv[2]
record = json.load(open(rec_path))
cell, inject = R.cell_of(record)
PERSONA = os.environ.get("REPLAY_PERSONA") or "capable_collaborator"
state, ro = build_episode(None, PERSONA, cell=cell, inject=inject)
os.environ["COOKSIM_PACE"] = os.environ.get("REPLAY_PACE") or record.get("pace") or "tight"; apply_pace(ro.game)
ge = record.get("goal_extended")
if ge:
    from cooksim.core.lang_events import extend_goal
    try: extend_goal(ro.game, ge["ingredient"], ge.get("state", "raw"))
    except Exception: pass
acts = [a for a in (record.get("actions") or []) if a.get("cmd") and a.get("t_issue") is not None]
acts.sort(key=lambda a: (a["t_issue"], a.get("i", 0)))
T = record.get("ticks") or 600; ro.cmds = []; ro.cap = T + 5
def idle():
    return ro.ctrl.skill is None and getattr(ro, "_wander", 0) <= 0 and getattr(ro, "_wait", 0) <= 0 and ro.i + 1 >= len(ro.cmds)
k = 0; issued = []   # (record action, index into ro.log)
plate_seen, plate_prev = [], []
states = [json.dumps(ro.game.render_state())]
plate_tl = []
while ro.tick < T:
    if idle():
        if k < len(acts) and acts[k]["t_issue"] <= ro.tick:
            ro.cmds.append((acts[k]["cmd"], None)); issued.append((acts[k], len(ro.cmds) - 1)); k += 1
        else:
            ro.cmds.append(("idle 1", None))
    if not ro.step(): break
    states.append(json.dumps(ro.game.render_state()))
    st = R.stamp_of(ro.game, ro.tick); names = [c.split("(")[0] for c in (st["plate"] or [])]
    plate_tl.append((ro.tick, tuple(names)))
    if len(names) > len(plate_prev) and names[:len(plate_prev)] == plate_prev:
        for n in names[len(plate_prev):]: plate_seen.append({"item": n, "t": ro.tick})
    plate_prev = names
# per-command completion ticks from the rollout log (one log row per non-idle command, in order)
rows = [l for l in ro.log if not any(w in str(l.get("cmd", "")) for w in ("idle", "wander"))]
diffs = []
for (a, _), l in zip(issued, rows):
    if a.get("t_done") is not None and l.get("t_done") is not None:
        diffs.append((a["cmd"], a["t_done"], l["t_done"], l.get("status")))
bad = [d for d in diffs if abs(d[1] - d[2]) > 1]
rec_plate = [(e["item"], e["t"]) for e in (record.get("plate_seq") or [])]; got = [(e["item"], e["t"]) for e in plate_seen]
ok = [i for i, _ in rec_plate] == [i for i, _ in got]
# TICK WARP: the replay reproduces the kitchen but walks can finish a few ticks off; resample the replay states so that
# every command finishes on its RECORDED tick (piecewise-linear map recorded tick -> replay tick between anchors)
anchors = sorted({(0, 0)} | {(int(d[1]), int(d[2])) for d in diffs if d[1] is not None and d[2] is not None} | {(T, min(T, len(states) - 1))})
def warp(t):
    for (a0, b0), (a1, b1) in zip(anchors, anchors[1:]):
        if a0 <= t <= a1:
            return b0 if a1 == a0 else b0 + (b1 - b0) * (t - a0) / (a1 - a0)
    return min(t, len(states) - 1)
with open(out_path, "w") as f:
    for t in range(T + 1):
        f.write(states[max(0, min(len(states) - 1, int(round(warp(t)))))] + "\n")
print(json.dumps({"ticks_replayed": ro.tick, "ticks_recorded": T, "commands": len(acts), "issued": len(issued), "done_tick_mismatches": bad[:6],
                  "max_done_diff": max([abs(d[1]-d[2]) for d in diffs] or [0]), "plate_recorded": rec_plate, "plate_replayed": got, "plate_items_match": ok,
                  "written_states": T + 1, "warp_anchors": len(anchors)}))
if not ok or (diffs and max(abs(d[1]-d[2]) for d in diffs) > 12):
    os.remove(out_path); sys.exit("replay does not reproduce the record closely enough: dump refused")
