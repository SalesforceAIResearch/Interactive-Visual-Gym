#!/usr/bin/env python3
"""Replay an episode record's action log through the engine to recover the per-tick world.

Why (2026-09-19 question-truth audit): records written without COOKSIM_DUMP_STATES (e.g. the
real-cook verification run tmp/verify_llm_formal) keep the executed command log (`actions`: cmd,
t_issue, t_done, status) but no per-tick kitchen. Question truths that depend on what happened in
the world (which item was chopped when, which station the cook faced after each walk, what was on
each station at the ask) need that kitchen. The action log is the complete input to the cook's
body, so re-issuing each command on its recorded issue tick through the same executor
(MultiRollout on the same map, same pace) reproduces the body's trajectory. Fidelity is MEASURED,
not assumed: every replayed command's completion tick and status are compared with the recorded
ones, and `fidelity` reports the match rate and the first divergence.

Records made before 2026-09-30 carry a `smoke_hold_ticks` budget from the deleted smoke hold; it is
re-applied so their replay stays exact. Plan-change order extensions are not reproduced; they change the order, not a chop, a load, a walk or a plate.

    from cooksim_bench.scoring.qa_replay import replay_states
    snaps, fid = replay_states(record)          # snaps[t] = game.render_state() before tick t's step
"""
import os
from cooksim_bench.paths import ROOT
import sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

CANCELLED = ("superseded", "dropped", "cancelled")


_SUITE_INJECT = None


def _inject_of(r, cell):
    """The case's `inject` flag (decoy lettuce + bin on spare counters), as the live harness used it: from the suite
    file by case name, else the designer's rule (hard maps are built without decoys)."""
    global _SUITE_INJECT
    if _SUITE_INJECT is None:
        _SUITE_INJECT = {}
        import json
        for p in ("benchmark/cases_composite_v4.json",):
            try:
                d = json.load(open(p))
                for c in (d["cases"] if isinstance(d, dict) else d):
                    _SUITE_INJECT[c["name"]] = c.get("inject", True)
            except Exception:                            # noqa: BLE001
                pass
    name = r.get("scenario") or (r.get("composite") or {}).get("name")
    if name in _SUITE_INJECT:
        return _SUITE_INJECT[name]
    return not str(cell).endswith("hard_map")


def replay_states(r, max_ticks=None, with_world=False):
    """Return (snapshots, fidelity[, world]). snapshots: list of render_state dicts, index = tick.
    world: the executor's own qa_world.WorldTrack (the same observer the live rollout carries)."""
    from cooksim_bench.exp_interact import build_episode
    from cooksim.core.config import apply_pace
    cell = (r.get("composite") or {}).get("cell") or r.get("cell")
    _state, ro = build_episode(None, "capable_collaborator", cell=cell, inject=_inject_of(r, cell))
    apply_pace(ro.game, r.get("pace") or "tight")
    ro.cmds, ro.i, ro.log = [], -1, []
    ro.cap = 10 ** 6
    ro.lang_pending = []
    acts = sorted(r.get("actions") or [], key=lambda a: (a.get("t_issue") or 0, a.get("i") or 0))
    by_issue = {}
    for a in acts:
        by_issue.setdefault(a.get("t_issue"), []).append(a)
    cancels = {}
    for a in acts:
        if a.get("status") in CANCELLED and a.get("t_done") is not None:
            cancels.setdefault(a["t_done"], []).append(a)
    hold_budget = dict(r.get("smoke_hold_ticks") or {})
    end = max_ticks if max_ticks is not None else int(r.get("ticks") or 0)
    snaps = []
    fed = []                     # recorded entries in the order they were fed
    for t in range(end + 1):
        # a command the live harness cut off (queue replaced) stops on the recorded tick
        for a in cancels.get(t, []):
            if ro.log and ro.log[-1].get("status") in ("running", "waiting", "wandering"):
                ro.ctrl.skill = None
                ro._wait = ro._wander = 0
                ro.log[-1].update(status="superseded", t_done=t)
        for a in by_issue.get(t, []):
            cmd = a.get("cmd") or ""
            if cmd.startswith("wander"):
                n = (a.get("t_done") or t) - t
                cmd = f"idle {max(1, n)}"
            ro.cmds.append((cmd, a.get("kind") or "replay"))
            fed.append(a)
        ro.tick = t
        # the smoke hold, re-applied from its recorded per-item budget (the game counts held ticks itself)
        used = ro.game.stats.get("smoke_hold_ticks", {}) if hold_budget else {}
        ro.game.smoke_hold_items = {k for k, n in hold_budget.items() if used.get(k, 0) < n}
        snaps.append(ro.game.render_state())
        if t == end:
            break
        ro.step()
    # fidelity: compare replayed log entries with the recorded ones, in order
    rep = ro.log
    n = min(len(rep), len(fed))
    match = 0
    first_div = None
    for k in range(n):
        a, b = fed[k], rep[k]
        same = (a.get("t_done") == b.get("t_done") or a.get("status") in ("running",)) and \
            (a.get("status") == b.get("status") or {a.get("status"), b.get("status")} <= {"done", "running"})
        if same:
            match += 1
        elif first_div is None:
            first_div = {"k": k, "recorded": {x: a.get(x) for x in ("cmd", "t_issue", "t_done", "status")},
                         "replayed": {x: b.get(x) for x in ("cmd", "t_issue", "t_done", "status")}}
    fid = {"recorded": len(fed), "replayed": len(rep), "matched": match,
           "rate": round(match / max(1, len(fed)), 4), "first_divergence": first_div}
    if with_world:
        return snaps, fid, getattr(ro, "world", None)
    return snaps, fid


def world_for_record(r, record_path=None):
    """(world, snapshots, source) for an episode record: its states dump when one exists (the live kitchen as it
    was), else a replay of its action log (fidelity attached to the source string)."""
    from cooksim_bench.scoring.qa_world import WorldTrack
    dump = record_path[:-5] + ".states.jsonl" if record_path else None
    if dump and os.path.exists(dump):
        import json
        snaps = []
        with open(dump) as fh:
            for line in fh:
                try:
                    snaps.append(json.loads(line))
                except Exception:                        # noqa: BLE001
                    pass
        return WorldTrack.from_snapshots(snaps), snaps, "states_dump"
    snaps, fid, world = replay_states(r, with_world=True)
    return world, snaps, f"replay (action-log fidelity {fid['rate']:.2f})"


def plate_seq_from_snaps(snaps):
    """[(t, item, state)] additions to the fullest plate, diffing ordered lists (as track_plate does)."""
    out, prev = [], []
    for s in snaps:
        cur = _plate_list(s)
        if cur[:len(prev)] == prev:
            for n, st in cur[len(prev):]:
                out.append((s.get("tick"), n, st))
        prev = cur
    return out


def _plate_list(s):
    best = []
    for p in s.get("players") or []:
        h = p.get("holding") or {}
        if h.get("kind") == "plate":
            best = [(x.get("name"), x.get("state")) for x in h.get("contents") or []]
    for o in s.get("objects") or []:
        it = o.get("item") or {}
        if it.get("kind") == "plate":
            c = [(x.get("name"), x.get("state")) for x in it.get("contents") or []]
            if len(c) > len(best):
                best = c
    return best


if __name__ == "__main__":
    import json
    for f in sys.argv[1:]:
        r = json.load(open(f))
        snaps, fid = replay_states(r)
        ps = plate_seq_from_snaps(snaps)
        rec_ps = [(e["t"], e["item"], e.get("state")) for e in r.get("plate_seq") or []]
        print(f, fid["rate"], fid["recorded"], fid["replayed"], "plate", "OK" if [x[1:] for x in ps] == [x[1:] for x in rec_ps] else f"DIFF {ps} vs {rec_ps}")
        if fid["first_divergence"]:
            print("   ", fid["first_divergence"])


# ---------------------------------------------------------------------- offline truths
class _RecordRo:
    """The attributes the qa_v3_lib truth functions read off a live rollout, rebuilt from a record."""

    def __init__(self, r, world, tick):
        self.world = world
        self.log = list(r.get("actions") or [])
        self.cell = (r.get("composite") or {}).get("cell") or r.get("cell")
        self.tick = tick
        self.plate_seq = [dict(e) for e in (r.get("plate_seq") or [])]
        self.convo = [{"t": c.get("t"), "speaker": c.get("speaker"), "text": c.get("text")}
                      for c in (r.get("conversation") or []) if c.get("text")]


class _SnapGame:
    def __init__(self, snap, cell):
        self._s, self.cell = snap, cell
        self.tick = snap.get("tick") if snap else None

    def render_state(self):
        return self._s


_WORLD_CACHE = {}


def offline_truth(kind, r, record_path, tick):
    """Recompute a question truth for a recorded episode at `tick` with the live truth functions, from the
    episode's own world (states dump, else a replay of its action log). World-history kinds only: order,
    happened, count, next_step (the kinds whose live truth was wrong before 2026-09-19)."""
    from cooksim_bench.suite import gen_intent_cases as GI
    key = record_path or id(r)
    if key not in _WORLD_CACHE:
        if len(_WORLD_CACHE) > 64:
            _WORLD_CACHE.clear()
        world, snaps, src = world_for_record(r, record_path)
        _WORLD_CACHE[key] = (world, {s.get("tick", i): s for i, s in enumerate(snaps)}, src)
    world, by_t, src = _WORLD_CACHE[key]
    ro = _RecordRo(r, world, tick)
    game = _SnapGame(by_t.get(tick) or by_t.get(max(t for t in by_t if t <= tick)), ro.cell)
    gt = GI.ground_truth(kind, game, ro)
    if isinstance(gt, dict):
        gt["world_source"] = src
    return gt
