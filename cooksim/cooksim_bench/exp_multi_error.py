#!/usr/bin/env python3
"""MULTI-error rollouts from the plan IR: compose typed subgoal transforms (skip / insert /
swap / repeat / truncate) into one episode, compile to the tagged command list, execute blind,
and recover a per-deviation tick window from the log.

    python -m cooksim_bench.exp_multi_error --probe          # run the demo compositions headless
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, json, os, re, sys

from cooksim.core.recipes import Recipe
from cooksim.core.plan_ir import (segment_plan, compile_ir, foreign_scoop, raw_scoop, wander_node,
                                  stall_node, skip, insert, swap, repeat, truncate_and_serve)
from cooksim_bench.exp_error_rollout import Rollout, inject_source, baseline_time, gt_plan, CELL

from cooksim.agents.highlevel import parse as _hl_parse  # noqa: E402


class MultiRollout(Rollout):
    """Rollout over a compiled plan-IR command list. `kind` carries the SUBGOAL ID of each
    command (the base class only compares kind against 'pre', so we override the deviation
    marking); per-subgoal tick windows are recovered from the log afterwards."""

    def __init__(self, game, tagged, succeeds=False, lang_events=None, seed=0):
        super().__init__(game, [], [], [], terminal=False, succeeds=succeeds)
        self.cmds = [(c, sid) for c, sid, _dev in tagged]
        self.dev_sids = {sid for _c, sid, dev in tagged if dev}
        import random
        self.rng = random.Random(seed)
        # language events: fire at a subgoal BOUNDARY (after `after` completes); the cook
        # pauses for `pause` ticks (the no-op/utterance realization) and then resumes
        # unconditionally. Structured content + gt come from lang_events.GENERATORS.
        self.lang_pending = list(lang_events or [])    # each: {after, subtype, pause?, params?}
        self.lang_log = []

    def _issue_next(self):
        # subgoal boundary: previous command's subgoal finished -> fire any language event
        if self.i + 1 < len(self.cmds) and self.lang_pending:
            prev_sid = self.cmds[self.i][1] if self.i >= 0 else "__start__"
            next_sid = self.cmds[self.i + 1][1]
            if prev_sid != next_sid:
                for ev in list(self.lang_pending):
                    if ev["after"] == prev_sid:
                        from cooksim.core.lang_events import GENERATORS
                        rec = GENERATORS[ev["subtype"]](self.game, self, **ev.get("params", {}))
                        rec.update(after_subgoal=prev_sid, at_tick=self.tick,
                                   pause=ev.get("pause", 4))
                        from cooksim.core.utterances import realize
                        rec["utterance"] = realize(rec, self.rng)
                        self.lang_log.append(rec)
                        self.lang_pending.remove(ev)
                        self._wander = rec["pause"]        # utterance window = no-op pause
                        return
        while self.i + 1 < len(self.cmds):
            self.i += 1
            cmd, kind = self.cmds[self.i]
            if kind in self.dev_sids and self.dev_tick is None:
                self.dev_tick = self.tick
            if cmd.startswith("__wander__"):
                cook0 = self.game.cooks[0]
                from cooksim_bench.exp_error_rollout import _noun_tile, _wander_tile
                cont = next((c for c, k2 in self.cmds[self.i + 1:] if k2 not in self.dev_sids),
                            "go to the serving pass")
                it0 = _hl_parse(cont)
                ret = None
                if it0 is None or it0.verb != "goto":
                    # next correct step is a manipulation at the CURRENTLY FACED station:
                    # after the excursion the cook must walk back to it
                    from cooksim.core.enums import DIRECTION_VECTORS
                    dx, dy = DIRECTION_VECTORS[cook0.direction]
                    fx, fy = cook0.pos[0] + dx, cook0.pos[1] + dy
                    noun = self.game.layout.grid[fy][fx].value.replace("_", " ").replace(" source", "")
                    noun = (noun.replace("plate ", "plates ")).strip()
                    ret = f"go to the {noun} ({fx},{fy})"
                    nxt = (fx, fy)
                else:
                    nxt = _noun_tile(self.game, cont, cook0)
                parts = cmd.split()
                seq = []
                think = int(parts[1]) if len(parts) > 1 else 10       # __wander__: far, opposite, think N
                w = _wander_tile(self.game, cook0, nxt)
                seq.append(f"go to the counter ({w[0]},{w[1]})")
                if think > 0:
                    seq.append(f"idle {think}")
                if ret:
                    seq.append(ret)
                self.cmds[self.i:self.i + 1] = [(c, kind) for c in seq]
                cmd, kind = self.cmds[self.i]
            if cmd.startswith("idle "):
                self._wander = int(cmd.split()[1])
                self.log.append(dict(i=self.i, cmd="wander/idle (thinking)", kind=kind,
                                     t_issue=self.tick, t_done=None, status="wandering"))
                return
            it = _hl_parse(cmd)
            verb = it.verb if it else None
            if verb in {"pickup", "putdown", "chop", "turnon", "scoop", "serve", "trash"}:
                from cooksim.agents.highlevel import _validate
                ok, why = _validate(verb, self.game, self.game.cooks[0])
                if not ok:
                    from cooksim_bench.exp_error_rollout import SCOOP_WAIT, RETRY, _never_lit_faced
                    if verb == "scoop" and _never_lit_faced(self.game):
                        # E8 (round-2 review): a scoop at a loaded, never-lit station is refused at once with its
                        # reason instead of spinning the 60-tick scoop wait (see exp_error_rollout.Rollout)
                        self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick,
                                             t_done=self.tick, status="rejected", why=why))
                        return
                    self._wait = SCOOP_WAIT if verb == "scoop" else RETRY
                    self._pending = (cmd, kind, verb)
                    self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick,
                                         t_done=None, status="waiting", why=why))
                    return
            # GROUNDING (user ruling 2026-08-15): bind the late-bound scripted command to
            # a concrete instance before issuing, so the plan names which one it means —
            # same bar the cook is held to. See highlevel.bind_scripted_command.
            from cooksim_bench.exp_error_rollout import _issue_mode
            if self.ctrl.issue(cmd, **_issue_mode(kind)):
                self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick,
                                     t_done=None, status="running"))
                return
            self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick,
                                 t_done=self.tick, status="rejected"))
            # a dead 'go to X' must strand the bare manipulation verbs behind it (the
            # double-bun bug). This OVERRIDE was the one rejection path that never called
            # the base-class orphan skip — the reason the bug kept resurfacing after the
            # Rollout-side fix.
            if it and it.verb == "goto":
                self._skip_orphans()

    def windows(self):
        """subgoal id -> (first issue tick, last done tick, statuses)."""
        w = {}
        for l in self.log:
            sid = l["kind"]
            a, b, st = w.get(sid, (l["t_issue"], l["t_done"], []))
            w[sid] = (min(a, l["t_issue"]),
                      max(x for x in (b, l["t_done"], l["t_issue"]) if x is not None), st + [l["status"]])
        return w


def load_recipe(cell):
    data = json.load(open(os.environ.get("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))))
    sl = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    e = next(x for x in data["grid"] if sl(x["cell"]) == cell)
    return Recipe(id=e["recipe_id"], name=e["recipe_name"],
                  contents=tuple((n, s) for n, s in e["contents"]), base_reward=20, ordered=e["ordered"])


def run_composition(cell, name, build_fn, needs_lettuce=False):
    from cooksim_bench.backends.gemini_stream_rgb import make_gen_game
    game, _ = make_gen_game(cell)
    if needs_lettuce:
        inject_source(game, "lettuce@0,2")
    rec = load_recipe(cell)
    ir = segment_plan(rec)
    ir, records = build_fn(ir)
    tagged = compile_ir(ir)
    ro = MultiRollout(game, tagged, succeeds=all(r.transform == "insert" and r.event_subtype ==
                                                "CONFUSE_LOCATION_OR_DIRECTION" for r in records))
    while ro.step():
        pass
    base = baseline_time(cell)
    win = ro.windows()
    print(f"\n=== {name} ===")
    print(f"  plan: {[n.sid for n in ir]}")
    print(f"  outcome={ro.outcome}  total_time={ro.tick}  baseline={base}  overhead={ro.tick - base}")
    for r in records:
        w = win.get(r.anchor.split('<->')[0], None)
        print(f"  deviation: {r.transform:18} {r.event_subtype:28} anchor={r.anchor:26} window={w[:2] if w else '—'}")
    return ro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    def single_foreign(ir):
        ir, r = insert(ir, foreign_scoop("lettuce"), before="assemble(cheese)")
        return ir, [r]

    def double_wander_swap(ir):
        ir, r1 = insert(ir, wander_node(), before="prep(meat)")
        ir, r2 = swap(ir, "assemble(bun)", "assemble(meat)")
        return ir, [r1, r2]

    def triple(ir):
        ir, r1 = insert(ir, stall_node(24), after="fetch_plate")
        ir, r2 = repeat(ir, "assemble(bun)")
        ir, r3 = skip(ir, "assemble(cheese)")
        return ir, [r1, r2, r3]

    run_composition(args.cell, "SINGLE: insert(foreign_scoop(lettuce), before=assemble(cheese))",
                    single_foreign, needs_lettuce=True)
    run_composition(args.cell, "DOUBLE: insert(wander, before=prep(meat)) + swap(assemble(bun), assemble(meat))",
                    double_wander_swap)
    run_composition(args.cell, "TRIPLE: insert(stall(24), after=fetch_plate) + repeat(assemble(bun)) + skip(assemble(cheese))",
                    triple)


if __name__ == "__main__":
    main()
