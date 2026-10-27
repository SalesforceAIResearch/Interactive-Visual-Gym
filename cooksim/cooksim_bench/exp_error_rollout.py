#!/usr/bin/env python3
"""Error ROLLOUTS: blend each error type into a complete procedural episode at a chosen STATE.

The GT plan for the cell is a list of high-level commands; completing command i = "state i+1".
An error variant = (error type, phase) -> deviate after state k by transforming the REMAINING
plan (inject / substitute / skip / truncate). The cook then keeps executing BLIND (it does not
know it erred) until the episode FAILS: a wrong serve is rejected, or it times out. The result
is a realistic full-episode rollout containing one mistake that a supervisor should catch.

  probe (headless, no browser):  python -m cooksim_bench.exp_error_rollout --dev omission --phase middle --probe
  render:                        COOKSIM_PAGE=.. python -m cooksim_bench.exp_error_rollout --dev omission --phase middle \
                                     --out tmp/site_new/rollouts/omission_middle.mp4
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, asyncio, io, json, os, re, sys

from cooksim.core.enums import Action
from cooksim.agents.highlevel import HighLevelController, _validate, parse as hl_parse
from cooksim.agents.highlevel import GoTo as _GoTo

CELL = "medium_nops_medium_map"
CAP = 300                 # global tick cap = timeout
IDLE_TAIL = 30            # ticks of idling after the last command before we call it a timeout
SCOOP_WAIT = 60           # scoop may legitimately wait for cooking; others retry briefly
RETRY = 3

# ---------------------------------------------------------------------------- choreography
# The GT plan (15 commands) for Cheeseburger medium/medium:
#  0 go meat src | 1 pick up | 2 go pan | 3 put down | 4 turn on | 5 go plates | 6 pick up
#  7 go bun src | 8 scoop | 9 go pan | 10 scoop | 11 go cheese src | 12 scoop | 13 go serving | 14 serve
# A variant returns (pre, dev, cont, terminal): pre = plan[:k] executed correctly, dev = the
# deviated commands, cont = blind continuation, terminal = cook stops after dev (thinks it's done).
LETTUCE = "lettuce@0,2"

def _v(k, dev, cont, terminal=False, succeeds=False):
    return dict(k=k, dev=dev, cont=cont, terminal=terminal, succeeds=succeeds)


def variants(plan):
    LT = "go to the lettuce source"
    MS = "go to the meat source"; SV = ["go to the serving pass", "serve"]
    return {
        # walks the WRONG WAY — far, opposite the correct next target — wanders ("thinking") for
        # 10 time units, then comes back and finishes. Succeeds, but the detour costs total time.
        # NOTE: k must avoid the active-cooking window (states 5-10) or the unattended pan burns
        # and the episode can no longer succeed — so front=before fetching, middle=pan loaded but
        # not yet lit, end=dish complete, heading to serve.
        ("wrong_location", "front"):  _v(0,  ["__wander__"], plan[0:], succeeds=True),
        ("wrong_location", "middle"): _v(4,  ["__wander__"], plan[4:], succeeds=True),
        ("wrong_location", "end"):    _v(13, ["__wander__"], plan[13:], succeeds=True),
        # a foreign ingredient (decoy lettuce crate) enters the dish
        ("wrong_ingredient", "front"):  _v(0,  [LT, "pick up"], plan[2:]),              # lettuce cooked instead of meat
        ("wrong_ingredient", "middle"): _v(9,  [LT, "scoop"], plan[9:]),
        ("wrong_ingredient", "end"):    _v(13, [LT, "scoop"], plan[13:]),
        # the meat goes on the plate RAW (from the source, skipping the pan)
        ("raw_on_plate", "front"):  _v(7,  [MS, "scoop"], plan[7:]),                    # earliest feasible: needs the plate
        ("raw_on_plate", "middle"): _v(9,  [MS, "scoop"], plan[11:]),                   # replaces the pan scoop
        ("raw_on_plate", "end"):    _v(13, [MS, "scoop"], plan[13:]),
        # ordered recipe scooped out of sequence
        ("wrong_order", "front"):  _v(7,  plan[9:11], plan[7:9] + plan[11:]),           # meat before bun
        ("wrong_order", "middle"): _v(9,  plan[11:13], plan[9:11] + plan[13:]),         # cheese before meat
        ("wrong_order", "end"):    _v(10, plan[11:13], plan[9:11] + plan[13:]),         # walks off the pan mid-step
        # a required ingredient never makes it onto the plate
        ("omission", "front"):  _v(0,  [], plan[5:]),                                   # whole meat block skipped
        ("omission", "middle"): _v(9,  [], plan[11:]),                                  # meat scoop skipped
        ("omission", "end"):    _v(11, [], plan[13:]),                                  # cheese skipped
        # serves way too early (thinks the dish is done) -> the serve executes and is REJECTED.
        # NOTE: an empty-plate serve is refused by the engine and nothing happens, so the earliest
        # variant that yields a rejected serve is right after the first scoop (state 9).
        ("premature_serve", "front"):  _v(9,  SV, [], terminal=True),                   # only the bun
        ("premature_serve", "middle"): _v(11, SV, [], terminal=True),                   # bun + meat, skips cheese
        ("premature_serve", "end"):    _v(12, SV, [], terminal=True),                   # at the cheese crate, walks away
        # freezes mid-task and never resumes
        ("stall", "front"):  _v(2,  [], [], terminal=True),
        ("stall", "middle"): _v(7,  [], [], terminal=True),
        ("stall", "end"):    _v(13, [], [], terminal=True),
    }


def gt_plan(cell):
    data = json.load(open(os.environ.get("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))))
    sl = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return next(x for x in data["grid"] if sl(x["cell"]) == cell)["plan"]


def enable_burn(game, burn=None):
    """DEPRECATED no-op: smoke/fire physics is now ON by default (smoke 30 -> fire 10 -> fail)."""
    return


def inject_source(game, spec):
    """Replace a counter with another tile: 'lettuce@0,2' (ingredient source) or 'trash@5,2'."""
    from cooksim.core.enums import Terrain
    name, coord = spec.split("@")
    x, y = map(int, coord.split(","))
    t = Terrain.TRASH if name == "trash" else Terrain[f"{name.upper()}_SOURCE"]
    game.layout.grid[y][x] = t


# ---------------------------------------------------------------------------- wander helpers
def _noun_tile(game, cmd, cook):
    """Ground the NEXT correct command's target tile (nearest instance, like the executor)."""
    from cooksim.agents.noisy_human import find_tiles
    T = find_tiles(game)
    cx, cy = cook.pos
    near = lambda ts: min(ts, key=lambda t: abs(t[0] - cx) + abs(t[1] - cy)) if ts else None
    c = cmd.lower()
    for ing, tiles in T["src"].items():
        if ing in c:
            return near(tiles)
    if "pan" in c or "pot" in c or "oven" in c:
        return near(T["pots"])
    if "board" in c:
        return near(T["boards"])
    if "plate" in c:
        return T["plate"]
    if "serv" in c:
        return T["serving"]
    return T["serving"]


def _wander_tile(game, cook, nxt):
    """The wrong location: a reachable counter in the OPPOSITE half-plane from the correct next
    target, as FAR as possible (maximises both the wrong-way signal and the time cost)."""
    from cooksim.core.enums import Terrain
    from cooksim.agents.heuristic import _adjacent_floor
    cx, cy = cook.pos
    cands = []
    for y, row in enumerate(game.layout.grid):
        for x, t in enumerate(row):
            if t == Terrain.COUNTER and _adjacent_floor(game, x, y):
                cands.append((x, y))
    vx, vy = nxt[0] - cx, nxt[1] - cy
    opp = [w for w in cands if (w[0] - cx) * vx + (w[1] - cy) * vy <= 0]
    pool = opp or cands
    # time cost = walk there AND back, so cook-distance counts double; next-target distance keeps
    # the wrong-way signal strong (the farther from where it should go, the earlier a VLM can tell)
    return max(pool, key=lambda w: 2 * (abs(w[0] - cx) + abs(w[1] - cy)) + (abs(w[0] - nxt[0]) + abs(w[1] - nxt[1])))


# ---------------------------------------------------------------------------- per-tick executor

def _issue_mode(kind: str) -> dict:
    """Who authored this queue entry decides how it is grounded.

    Plan-authored commands are deliberately late-bound ("go to the empty cutting board")
    and are pinned to one instance at issue time (bind=True). Cook-authored commands --
    guidance the cook accepted and spliced in, sid prefixed FIX -- get the strict contract:
    a duplicated kind named bare is REFUSED with a reason, never resolved to nearest.
    """
    from cooksim.interact.system import FIX
    if isinstance(kind, str) and kind.startswith(FIX):
        return dict(strict=True, bind=False)
    return dict(strict=False, bind=True)


def _never_lit_faced(game):
    """The cook faces a heat station that holds food and was never switched on (E8)."""
    try:
        from cooksim.core.enums import DIRECTION_VECTORS
        from cooksim.core.stations import CookStation
        c = game.cooks[0]
        dx, dy = DIRECTION_VECTORS[c.direction]
        st = game.stations.get((c.pos[0] + dx, c.pos[1] + dy))
        return (isinstance(st, CookStation) and bool(st.contents) and st.status == "raw"
                and not getattr(st, "cooking", False))
    except Exception:                                    # noqa: BLE001
        return False


class Rollout:
    """Per-tick blind plan executor with a deviation window. Yields once per tick so a renderer
    can capture frames; records per-command completion ticks (state boundaries) + failure."""

    def __init__(self, game, pre, dev, cont, terminal, succeeds=False, cap=CAP):
        self.game = game
        self.cap = cap
        self.ctrl = HighLevelController(game, 0)
        self.cmds = [(c, "pre") for c in pre] + [(c, "dev") for c in dev] + [(c, "cont") for c in cont]
        self.terminal = terminal
        self.succeeds = succeeds
        self.i = -1               # index of the command currently executing
        # `note` (ECR-13 §D1) carries the controller's word about a command that finished
        # legally and changed NOTHING — e.g. a turn-on at a station that was already lit.
        # scan_feedback appends it to the "you DID: ..." line so `done` cannot lie.
        self.log = []             # {i, cmd, kind, t_issue, t_done, status, note}
        self.tick = 0
        self.dev_tick = None      # first tick a 'dev' command starts (or terminal stall begins)
        self.fail_tick = None
        self.outcome = None       # 'wrong_serve' | 'timeout' | 'won'
        self._wait = 0
        self._retry = 0
        self._wander = 0
        self._idle_since = None
        self._start_fail = game.stats.get("failed_deliveries", 0)
        self._start_win = game.stats.get("deliveries", 0)
        # WORLD HISTORY for question truths (2026-09-19 audit, python -m cooksim_bench.scoring.qa_world): what was chopped, loaded,
        # lit, cooked and plated when, and which station the cook faced -- read off the kitchen once per
        # tick, never off the command tags (the self-planning cook's commands carry none).
        try:
            from cooksim_bench.scoring.qa_world import WorldTrack
            self.world = WorldTrack()
            self.world.observe(game.render_state(), self.tick)
        except Exception:                                    # noqa: BLE001  (truths fall back to "unknown")
            self.world = None

    def _issue_next(self):
        while self.i + 1 < len(self.cmds):
            self.i += 1
            cmd, kind = self.cmds[self.i]
            # deviation starts at the first non-'pre' command: a 'dev' command, or (for pure-skip
            # variants like omission, which have no dev commands) the first 'cont' command.
            if kind != "pre" and self.dev_tick is None:
                self.dev_tick = self.tick
            if cmd == "__wander__":
                # resolve NOW, from the cook's live position: far counter opposite the next target
                cook0 = self.game.cooks[0]
                cont = next((c for c, k2 in self.cmds[self.i + 1:] if k2 == "cont"), "go to the serving pass")
                it0 = hl_parse(cont)
                ret = None
                if it0 is None or it0.verb != "goto":
                    # the next correct step is a manipulation at the CURRENTLY FACED station — the
                    # cook must walk back to it after wandering, or the whole plan collapses
                    from cooksim.core.enums import DIRECTION_VECTORS
                    dx, dy = DIRECTION_VECTORS[cook0.direction]
                    fx, fy = cook0.pos[0] + dx, cook0.pos[1] + dy
                    noun = self.game.layout.grid[fy][fx].value.replace("_", " ").replace(" source", "") + " "
                    noun = noun.replace("plate ", "plates ").strip()
                    ret = f"go to the {noun} ({fx},{fy})"
                    nxt = (fx, fy)
                else:
                    nxt = _noun_tile(self.game, cont, cook0)
                w = _wander_tile(self.game, cook0, nxt)
                cmd = f"go to the counter ({w[0]},{w[1]})"
                self.cmds[self.i] = (cmd, kind)
                self.cmds.insert(self.i + 1, ("idle 10", kind))       # wander/"thinking" at the wrong spot
                if ret:
                    self.cmds.insert(self.i + 2, (ret, kind))         # walk back to the abandoned station
            if cmd.startswith("idle "):
                self._wander = int(cmd.split()[1])
                self.log.append(dict(i=self.i, cmd="wander (thinking)", kind=kind, t_issue=self.tick,
                                     t_done=None, status="wandering"))
                return
            it = hl_parse(cmd); verb = it.verb if it else None
            if verb in {"pickup", "putdown", "chop", "turnon", "scoop", "serve", "trash"}:
                ok, why = _validate(verb, self.game, self.game.cooks[0])
                if not ok:
                    if verb == "scoop" and _never_lit_faced(self.game):
                        # E8 (round-2 review): a scoop at a LOADED station that was never switched on cannot become
                        # legal by waiting — nothing is cooking. It used to spin the 60-tick scoop wait in silence
                        # and read as an unlabelled forget-style freeze (5 of 20 expert episodes). Refuse it now, with
                        # the reason, as `wait until` already does.
                        self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick, t_done=self.tick,
                                             status="rejected", why=why))
                        return
                    self._wait = SCOOP_WAIT if verb == "scoop" else RETRY
                    self._pending = (cmd, kind, verb)
                    self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick, t_done=None, status="waiting",
                                         why=why))
                    return
            # GROUNDING (user ruling 2026-08-15): a kind with several instances must be
            # named exactly. The scripted plan is written late-bound ("go to the empty
            # cutting board"), so bind it to a concrete instance HERE, at issue time, and
            # log the bound form — the plan then meets the same bar as the cook, and the
            # trajectory records which instance it chose instead of a silent nearest-pick.
            #
            # BUT the two authors share this queue, and only one of them is entitled to
            # binding. Commands the COOK authored (spliced guidance, sid prefixed FIX)
            # must be issued strictly: if the cook says "the pan" on a two-pan map the
            # engine has to refuse and say so, which is the whole v2 contract. Issuing
            # everything with bind=True silently resolved the cook's ambiguous references
            # to the nearest instance — the exact guess the contract forbids — and made
            # the ambiguity refusal fire ZERO times across a nine-episode sweep.
            if self.ctrl.issue(cmd, **_issue_mode(kind)):
                self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick, t_done=None, status="running"))
                return
            self.log.append(dict(i=self.i, cmd=cmd, kind=kind, t_issue=self.tick, t_done=self.tick, status="rejected"))
            if (it and it.verb == "goto"):
                self._skip_orphans()
        # queue exhausted

    def _hesitating(self):
        """Opt-in (hesitate_p>0, set by pause-enabled harnesses): while WALKING (a goto
        command running), each tick has hesitate_p probability of standing still instead
        of stepping - human gait is not a straight-line beeline. Never fires during
        manipulation; skill state is untouched (paths recompute from the live position)."""
        hp = getattr(self, "hesitate_p", 0.0)
        if not hp or not self.log or self.log[-1].get("status") != "running":
            return False
        it = hl_parse(self.log[-1].get("cmd") or "")
        return bool(it and it.verb == "goto" and self._hes_rng.random() < hp)

    def _skip_orphans(self):
        """A failed 'go to X' strands the bare manipulation verbs queued right after it —
        they would fire at whatever tile the cook happens to face (the double-bun bug).
        Skip the contiguous bare-verb run; the human hears it in the kitchen feedback."""
        MANIP = {"pickup", "putdown", "chop", "turnon", "scoop", "serve", "trash", "wash"}
        while self.i + 1 < len(self.cmds):
            nxt, nk = self.cmds[self.i + 1]
            it2 = hl_parse(nxt)
            if it2 is None or it2.verb == "goto" or it2.verb not in MANIP:
                break
            self.i += 1
            self.log.append(dict(i=self.i, cmd=nxt, kind=nk, t_issue=self.tick,
                                 t_done=self.tick, status="orphan-skipped"))

    def step(self):
        """Advance one tick. Returns False when the episode is over."""
        g = self.game
        # deviation marker for terminal stalls (no dev commands)
        if self.terminal and not any(k == "dev" for _, k in self.cmds) and self.i + 1 >= len(self.cmds) \
                and self.dev_tick is None and (self.ctrl.skill is None):
            self.dev_tick = self.tick
        low = int(Action.STAY)
        if getattr(self, "_pause", 0) > 0:                   # ENGAGED IN CONVERSATION
            self._pause -= 1                                 # world (stations, smoke) keeps running
            # TALKING DOES NOT STOP YOUR LEGS. Freezing the cook outright for the length of its
            # own line cost a chatty persona 74-88% of every episode (measured across the rank
            # and sweep runs: novice_stepwise/frames spent 198 of 267 ticks frozen, leaving 69
            # usable ones), and the assistant then quite correctly reported "you are not moving"
            # — which drew another reply, which froze it again.
            # A walk ALREADY IN FLIGHT keeps going. Nothing new is issued and no manipulation
            # runs, so "you stop to chop while you explain" is preserved.
            if isinstance(self.ctrl.skill, _GoTo):
                low = self.ctrl.step()
                if self.ctrl.skill is None and self.log:
                    self.log[-1].update(status=self.ctrl.status, t_done=self.tick, note=self.ctrl.note)
        elif self._wander > 0:                               # wandering ("thinking") at the wrong spot
            self._wander -= 1
            if self._wander <= 0 and self.log:
                self.log[-1].update(status="done", t_done=self.tick)
        elif self._wait > 0:                                 # waiting for a manip to become legal
            cmd, kind, verb = self._pending
            ok, _ = _validate(verb, g, g.cooks[0])
            if ok:
                self._wait = 0
                if self.ctrl.issue(cmd, **_issue_mode(kind)):
                    self.log[-1]["status"] = "running"
                else:
                    self.log[-1].update(status="rejected", t_done=self.tick)
            else:
                self._wait -= 1
                # E8: the reason is taken NOW, at the refusal, against the tile the cook faces now — not later, when
                # the cook reads its feedback and may face another station (hard medium_4 B: a board refusal reached
                # the cook as a fryer refusal and trapped it for 225 ticks)
                _ok2, _why2 = _validate(verb, g, g.cooks[0])
                if _why2:
                    self.log[-1]["why"] = _why2
                if verb == "scoop" and _never_lit_faced(g):
                    self._wait = 0                           # E8: it became a never-lit station: refuse now
                    self.log[-1].update(status="rejected", t_done=self.tick)
                elif self._wait <= 0:
                    self.log[-1].update(status="illegal-skipped", t_done=self.tick)
        elif self.ctrl.skill is not None:                    # a command is executing
            if self._hesitating():
                self.hesitations.append(self.tick)           # human-like hesitation: stand
            else:                                            # still 1 tick mid-walk (opt-in)
                low = self.ctrl.step()
                if self.ctrl.skill is None and self.log:
                    self.log[-1].update(status=self.ctrl.status, t_done=self.tick, note=self.ctrl.note)
                    if self.ctrl.status in ("failed", "rejected"):
                        _it = hl_parse(self.log[-1].get("cmd") or "")
                        if _it and _it.verb == "goto":
                            self._skip_orphans()
        else:                                                # idle -> next command or tail-idle
            if self.i + 1 < len(self.cmds):
                self._issue_next()
                if self.ctrl.skill is not None:
                    low = self.ctrl.step()
                    if self.ctrl.skill is None and self.log:
                        self.log[-1].update(status=self.ctrl.status, t_done=self.tick, note=self.ctrl.note)
                        # the SAME-TICK issue+fail path (e.g. 'go to the oven with the
                        # meat' when no oven holds meat): a dead goto must strand its
                        # bare manipulation verbs here too - this was the third,
                        # unpatched entry to the double-bun bug
                        if self.ctrl.status in ("failed", "rejected"):
                            _it = hl_parse(self.log[-1].get("cmd") or "")
                            if _it and _it.verb == "goto":
                                self._skip_orphans()
            else:
                if self._idle_since is None:
                    self._idle_since = self.tick
        acts = [int(Action.STAY)] * len(g.cooks); acts[0] = int(low)
        g.step(acts)
        self.tick += 1
        if getattr(self, "world", None) is not None:
            self.world.observe(g.render_state(), self.tick)
        # episode end conditions
        if g.stats.get("kitchen_fire"):
            self.outcome, self.fail_tick = "fire", self.tick
            return False
        # W22 (user ruling 2026-08-12): once food BURNS it is irrecoverable by design
        # (scooping requires status=="cooked"), so the episode can never be completed —
        # previously it churned to the wall clock with no outcome and hazard=false.
        # It now ends IMMEDIATELY as a failed outcome. "burned" is already a member of
        # HAZARD_OUTCOMES in the scorers.
        if any(getattr(st, "status", None) == "burnt" for st in g.stations.values()):
            self.outcome, self.fail_tick = "burned", self.tick
            return False
        if g.stats.get("failed_deliveries", 0) > self._start_fail:
            self.outcome, self.fail_tick = "wrong_serve", self.tick
            return False
        if g.stats.get("deliveries", 0) > self._start_win:
            self.outcome, self.fail_tick = "won", self.tick          # accidental success (bad variant)
            return False
        idle_cap = (self.dev_tick + 38) if (self.terminal and self.dev_tick is not None) else None
        if idle_cap is not None and self.tick >= idle_cap:
            self.outcome, self.fail_tick = "timeout", self.tick
            return False
        if self._idle_since is not None and self.tick - self._idle_since >= IDLE_TAIL:
            self.outcome, self.fail_tick = "timeout", self.tick
            return False
        if self.tick >= self.cap:
            self.outcome, self.fail_tick = "timeout", self.tick
            return False
        return True


def build(cell, dev, phase):
    from cooksim_bench.backends.gemini_stream_rgb import make_gen_game
    game, _ = make_gen_game(cell)
    if dev == "wrong_ingredient":
        inject_source(game, LETTUCE)
    plan = gt_plan(cell)
    v = variants(plan)[(dev, phase)]
    return game, plan, v, Rollout(game, plan[:v["k"]], v["dev"], v["cont"], v["terminal"], v.get("succeeds", False))


def baseline_time(cell):
    """Total time (ticks; 1 low-level step = 1 unit) of the clean GT plan, no deviation."""
    from cooksim_bench.backends.gemini_stream_rgb import make_gen_game
    game, _ = make_gen_game(cell)
    plan = gt_plan(cell)
    ro = Rollout(game, plan, [], [], False)
    while ro.step():
        pass
    return ro.tick if ro.outcome == "won" else None


def probe(args):
    game, plan, v, ro = build(args.cell, args.dev, args.phase)
    while ro.step():
        pass
    states = [l for l in ro.log if l["kind"] == "pre" and l["status"] == "done"]
    base = baseline_time(args.cell)
    print(f"{args.dev}/{args.phase}: k={v['k']} dev_tick={ro.dev_tick} outcome={ro.outcome} "
          f"fail_tick={ro.fail_tick} pre_done={len(states)}/{v['k']} "
          f"total_time={ro.tick} baseline={base} overhead={ro.tick - base if base else '—'}")
    for l in ro.log:
        print(f"   [{l['kind']:4}] {l['cmd']:28} t={l['t_issue']:3}->{l['t_done']} {l['status']}")
    cook = game.cooks[0]
    h = cook.holding
    print(f"   final hold: {h if h is None else getattr(h,'contents',h)} score={int(game.score)} "
          f"failed={game.stats.get('failed_deliveries',0)}")
    return ro.outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=CELL)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--phase", required=True, choices=["front", "middle", "end"])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.probe:
        probe(args)
        return
    from cooksim_bench.video.render_rollout import render          # heavy imports only when rendering
    asyncio.run(render(args, build))


if __name__ == "__main__":
    main()
