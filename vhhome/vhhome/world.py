# libx264 spawns a thread pool at open; on a loaded shared box that allocation fails and the
# encoder never opens, losing the whole render at the final step. Cap it.
FFMPEG_THREADS = ["-threads", "2"]
"""LIVE tick-stepped VirtualHome world for the interaction experiments (recovery / e2e).

Port of the CookSim world-interface contract (docs/HANDOVER_INTERACTION_EXPERIMENTS.md §2)
onto VirtualHome: the world executes ONE closed-grammar command at a time in Unity (rendering
its frames live), keeps a SYMBOLIC state alongside (holding / opened / on / object locations —
the exact semantics of plan_ir_vh.simulate), refuses precondition-violating commands with
human-readable reasons, and exposes the supervisor observation channel (ego|map composite,
1 frame per tick, tick = 1 s = 5 raw frames) plus the creative human's own eyes (ego crop).

Time: 1 tick = 1 second of video (8 rendered frames). Commands span multiple ticks; idle
ticks freeze the last frame (VirtualHome has no autonomous dynamics — the honest analogue of
"the world keeps running").
"""
from vhhome import paths
import re, math, sys, os, io, json, glob, hashlib, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
from PIL import Image

from vhhome import tasks as episodes
from vhhome import goals as episode_goals
from vhhome import plan as pir
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import ImageDraw, ImageFont
from unity_simulator.comm_unity import UnityCommunication

# ---- rendering helpers (from render_rollout_vh.py, which can't be imported: argv at module level)
CATCOL = {"Furniture": ("#2b4a6f", "#5b9bd5"), "Appliances": ("#6b3d1f", "#e08a3c"),
          "Electronics": ("#3d2b5f", "#a97fe0"), "Food": ("#2f5a34", "#5fd38a"),
          "Lamps": ("#5a521f", "#d9c84a"), "Decor": ("#33302a", "#6b6456")}


from vhhome import render as vhfix
from vhhome.render import (EGO_PITCHES, EGO_FOV, EYE_DY, WALK_PITCH, ego_cam_name, yaw_to,
                           register_ego_ladder, aim_camera, run_setup, pitch_to)


def floorplan(g):
    SKIP = ("Rooms", "Walls", "Floor", "Ceiling", "Ceilings", "Doors", "Windows")
    rooms = [dict(cls=n["class_name"], c=n["bounding_box"]["center"], s=n["bounding_box"]["size"])
             for n in g["nodes"] if n.get("category") == "Rooms"]
    objs = [dict(cat=n.get("category"), c=[n["bounding_box"]["center"][0], n["bounding_box"]["center"][2]],
                 s=[n["bounding_box"]["size"][0], n["bounding_box"]["size"][2]])
            for n in g["nodes"] if n.get("category") not in SKIP and n["class_name"] != "character"]
    doors = [n["bounding_box"]["center"] for n in g["nodes"] if n["class_name"] == "door"]
    return rooms, objs, doors


def parse_pose(prefix, want3d=False):
    pf = glob.glob(f"{paths.UNITY_DIR}/Output/{prefix}/0/pd_*.txt")
    if not pf: return None
    rows = [ln.split() for ln in open(pf[0]).read().splitlines()[1:] if ln.strip()]
    pos, yaw, p3 = [], [], []
    for r in rows:
        v = list(map(float, r[1:]))
        hips = np.array(v[0:3]); lsh = np.array(v[11 * 3:11 * 3 + 3]); rsh = np.array(v[12 * 3:12 * 3 + 3])
        sh = rsh - lsh; facing = np.array([-sh[2], sh[0]])
        pos.append([hips[0], hips[2]]); yaw.append(np.degrees(np.arctan2(facing[0], facing[1])))
        p3.append([float(hips[0]), float(hips[1]), float(hips[2])])
    if want3d:
        return np.array(pos), np.array(yaw), p3
    return np.array(pos), np.array(yaw)


def map_frame(rooms, objs, doors, pos, yaw, k):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    xs = [r["c"][0] - r["s"][0] / 2 for r in rooms] + [r["c"][0] + r["s"][0] / 2 for r in rooms]
    zs = [r["c"][2] - r["s"][2] / 2 for r in rooms] + [r["c"][2] + r["s"][2] / 2 for r in rooms]
    _seen = {}
    for r in rooms:
        _seen[r["cls"]] = _seen.get(r["cls"], 0) + 1   # same labels as the person's map
        cx, cz, sx, sz = r["c"][0], r["c"][2], r["s"][0], r["s"][2]
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc="#161a22", ec="#39d98a", lw=1.4))
        ax.text(cx, cz - sz / 2 + .35, (r["cls"] + (f"#{_seen[r['cls']]}" if _seen[r["cls"]] > 1 else "")).upper(), ha="center", va="center",
                color="#39d98a", fontsize=8, alpha=.75, fontweight="bold")
    for o in sorted(objs, key=lambda o: -(o["s"][0] * o["s"][1])):
        cx, cz = o["c"]; sx, sz = o["s"]; fill, edge = CATCOL.get(o.get("cat"), (None, "#7a8394"))
        if fill is None or (sx < .28 and sz < .28):
            ax.plot(cx, cz, ".", color=edge, ms=2.6, alpha=.7); continue
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc=fill, ec=edge, lw=1.0, alpha=.9))
    for dc in doors:
        ax.plot(dc[0], dc[2], "s", color="#ff9ecb", ms=8, mfc="none", mew=1.6)
    kk = min(k, len(pos) - 1)
    ax.plot(pos[:kk + 1, 0], pos[:kk + 1, 1], "-", color="#4aa3ff", lw=2.3, alpha=.95)
    ax.plot(pos[0, 0], pos[0, 1], "o", color="#7ee787", ms=6)
    x, z = pos[kk]; a = yaw[kk]
    ax.arrow(x, z, .85 * np.sin(np.radians(a)), .85 * np.cos(np.radians(a)),
             head_width=.34, head_length=.32, fc="#ffd24d", ec="#0f1117", lw=1.2, zorder=8)
    ax.plot(x, z, "o", color="#ffd24d", ms=11, mec="#0f1117", mew=1.5, zorder=7)
    ax.set_xlim(min(xs) - .4, max(xs) + .4); ax.set_ylim(min(zs) - .4, max(zs) + .4)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw(); w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def _font(sz):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception: return ImageFont.load_default()


def _label(img, text, color):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 9 + 12, 24], fill=(15, 17, 23)); d.text((6, 4), text, fill=color, font=_font(16))
    return np.array(im)

FPT = 5               # frames per tick: Unity records at 5 fps, so a tick is exactly 1 s of animation (was 8 = 1.6 s, 2026-09-05)
# frames of aimed stillness held on a target before each manipulation (0 = off)
DWELL = int(os.environ.get("VH_DWELL", str(FPT)))
# EGOCENTRIC EXECUTION. With this on, a command is carried out through midlevel.py: the body
# must SEE a target before it can approach or act on it, and anything else is reached by
# turning until it comes into view. Off keeps the original high-level path, where [Walk]
# pathfinds across the house whether or not the target is visible and the room comes from the
# episode script -- i.e. the agent has oracle navigation and can never get lost.
MIDLEVEL = os.environ.get("VH_MIDLEVEL", "0") != "0"
# Bumped on every behavior-changing engine fix. Stamped into every run report so a results page
# can refuse to mix engine builds ("newest settings only" is checkable, not archaeological).
# f119 (2026-09-22): the assistant's generation time is charged to the game clock in A100-EQUIVALENT
# seconds — gen_s / ASSIST_LATENCY_DIV, the measured ratio for the card the job landed on
# (docs/GPU_LATENCY_RATIO.json). It is a no-op wherever the divisor is 1.0, which is every API arm, the
# silent control and every lane on an A100: an f118 episode with lat_div 1.0 and an f119 one are the same
# episode, so bench_v1 and the API arms of v2/v3 stand as run. Only a lane on a different card (H100,
# L40S) is charged differently, which is the point: the card must not decide the score.
# f120 (2026-10-02): the person's prompt. (1) A standing section, present every turn: a mistake is the
# assistant's to catch — say nothing about it and do not fix it yourself (iv_protocol.NO_SELF_REPAIR).
# (2) The two [SYSTEM TASK] blocks follow CookSim's shortened wording ("or say nothing", "do not fix
# anything yourself"). Both change what an uncaught mistake leads to, so f120 episodes are NOT comparable
# on task success with f118/f119 ones.
# f121 (2026-10-02): YOUR ORDER carries a status on every step and marks the steps an uncaught injected
# mistake left wrong as [intentional skip] / [intentional mistake] instead of presenting them as the next
# step (iv_protocol.order_marks). The mark lasts until the step is really done.
# f122 (2026-10-02): advance_phase no longer re-queues chore steps that are already waiting in the queue
# (a duplicate fetch undid a wrong placement on the engine's own initiative).
# f123 (2026-10-02): ...nor steps the body already carried out inside a forced block (logged `live_inject`,
# so the step-id test missed them and the juice-for-milk block ran twice); per-turn hints say the supervisor
# spoke only when it did, and the accept-nudge needs a real assistant line (both already in the f122 smoke).
# f124 (2026-10-05): a person who has finished its order with an uncaught mistake is no longer polled on
# idle ticks, and the episode ends as outcome `finished_wrong` (was: sixty ticks of idle polls, then `stuck`).
# f125 (2026-10-05): the dead-conversation guard also needs 30 ticks of standstill (VH_DEAD_CONV_TICKS) and labels
# a finished-with-mistake ending `finished_wrong`. An episode that never tripped the old guard plays out the same,
# so f124 records stand EXCEPT those with ended == "dead_conversation", which are re-run.
ENGINE_REV = "2026-10-08.f126"
CONTAINERS, DEVICES = pir.CONTAINERS, pir.DEVICES
# CLASSES UNITY ACTUALLY OPENS (A33, f89). The graph's CAN_OPEN is not a promise: an [Open] on every
# nightstand (0/11), desk (0/8), washing machine (0/5), closet, book, folder ... in all five apartments
# comes back "Script is impossible to execute", which reached the person as the opaque "my body
# couldn't do that here" (pilot 6 groceries: open the nightstand before taking the milk off it). These
# classes opened every time they were tried (tools/tmp/openable_scan.py); anything else is refused
# up front as "not something that opens".
OPENS = {"fridge", "microwave", "dishwasher", "stove", "cabinet", "kitchencabinet", "bathroomcabinet"}


def base(tok):
    """Class name of an object token: 'milk#176' / 'milk@2' -> 'milk'."""
    return None if tok is None else str(tok).split("#")[0].split("@")[0]


def _evid(tok):
    """Instance id for EVENT records. Resolved tokens carry '#id'; an explicitly indexed
    scripted token ('tablelamp@1') gets a NEGATIVE pseudo-id so two indexed twins stay
    distinguishable in offline/simulated grading too — without this, both lamps of the
    two-lamp tidy task recorded iid 0 and 'both lamps off' could never count two."""
    i = inst(tok)
    if i is not None:
        return i
    t = str(tok or "")
    if "@" in t and t.split("@", 1)[1].isdigit():
        return -int(t.split("@", 1)[1])
    return 0


def inst(tok):
    """Instance id carried by a token, or None if the token is class-level."""
    s = str(tok or "")
    return int(s.split("#", 1)[1]) if "#" in s else None


class SymState:
    # OFFLINE replays (the composite verifier) set this: a "nearest" token (cls@N) then names
    # some instance in the right state and is never refused as already off / already closed.
    # The live twin keeps the strict check -- its tokens arrive instance-resolved.
    LENIENT_NEAREST = False
    """The symbolic world — same verb semantics as plan_ir_vh.simulate().

    Object tokens are INSTANCE-UNIQUE ('milk#176'), not class names. Scene 2 ships two milks,
    nine plates and seven books, so keying this state on the class collapsed distinct objects
    onto one another: the engine would report loc={'milk': 'microwave'} while a second milk sat
    on the counter, and neither the assistant nor the human could be shown to be wrong. Tokens
    stay opaque here -- LiveWorld resolves class -> instance before calling in -- and every
    user-facing message renders through disp(), so the agents still read plain English.
    """
    def __init__(self, ep_key):
        self.ep = episodes.EPISODES[ep_key]
        self.gdef = episode_goals.EPISODES[ep_key]
        self.holding, self.opened, self.on, self.loc = set(), set(), set(), {}
        # Synthetic instance ids for "nearest" tokens (cls@N) the offline twin cannot resolve:
        # every grab / switch / open / close of such a token is its own instance, so multi-
        # instance goals (three mugs, both lamps) count correctly offline and an omitted twin
        # breaks the outcome (composite verifier, 2026-09-09).
        self._synth_n, self._synth_held = 0, {}
        # ON vs INSIDE must stay separate. `loc` answers "where is it"; `inside` answers "is it
        # shut away". Conflating them made a book resting ON a bookshelf unreachable, because
        # bookshelf is in CONTAINERS and the shelf is never "opened" -- the open-it-first rule
        # belongs only to things that are genuinely INSIDE something.
        self.inside = {}
        # Which containers actually have a DOOR. Derived from the engine's CAN_OPEN property by
        # LiveWorld; the CONTAINERS name-list is only a fallback for the offline symbolic path.
        # A bookshelf is a CONTAINER but has no CAN_OPEN and no states, so demanding it be
        # opened made every book in the house unreachable.
        self.openable = set()
        self.events = []
        self.sitting = False
        self.targets = {}          # class -> the instance the TASK means (bound after setup)
        # PLAN-EFFECT MIRROR. The step-in trigger compares the live situation against a
        # pir.simulate() replay of the clean prefix, and simulate starts from an EMPTY state
        # and records only what the plan changes. The instance-keyed state above cannot be
        # compared with that: it is seeded from the scene graph, so it also carries static
        # facts simulate never models (the microwave sits on the counter) and the decoy
        # objects (a second milk on the counter) that share a class with the target. Those
        # made the two keys permanently unequal and the injection never fired.
        # This mirror is BASE-NAME MULTISETS (collections.Counter) and starts empty, exactly
        # like simulate. Multisets, not sets: a two-lamp task legitimately holds
        # ('tablelamp','tablelamp') in `on`. Base names, not tokens: setup writes '@1'-style
        # tokens while the live plan resolves to '#256'-style, and a set keyed on either form
        # cannot be updated by the other — the trigger then never matches on any
        # multi-instance episode. Counting base names is exactly the granularity state_key()
        # compares, so it is immune to token form on both sides.
        from collections import Counter as _Ctr
        self.eff = {"holding": _Ctr(), "loc": _Ctr(), "opened": _Ctr(), "on": _Ctr()}
        for s in self.ep.get("setup", []):
            self.run(tuple(s), record=False)

    # ---- CLASS-LEVEL PROJECTIONS.
    # Instance keying is right for state and grading, but most consumers legitimately ask a
    # class-level question ("is there milk in the microwave?", "what is in hand?") and several
    # of them are load-bearing: step-in's trigger key is built from this, so leaving them to
    # slice tokens themselves silently broke injection matching. One place to project, so a
    # caller can never be half-migrated.
    @property
    def holding_cls(self):
        return {base(x) for x in self.holding}

    @property
    def opened_cls(self):
        return {base(x) for x in self.opened}

    @property
    def on_cls(self):
        return {base(x) for x in self.on}

    @property
    def loc_cls(self):
        return {base(k): base(v) for k, v in self.loc.items()}

    def _eff(self, a, o, o2):
        """Mirror one command into the plan-effect view (see eff's comment for why Counters)."""
        e, bo, bo2 = self.eff, base(o), base(o2)

        def dec(c, k):
            if c[k] > 0:
                c[k] -= 1
                if not c[k]:
                    del c[k]

        if a == "Grab":
            e["holding"][bo] += 1
            # simulate pops o's loc entry whatever its container; at base level pick the
            # first pair for this class (deterministic; our plans never track one class in
            # two containers at once)
            pair = next((k for k in sorted(e["loc"]) if k[0] == bo), None)
            if pair:
                dec(e["loc"], pair)
        elif a == "Open":
            e["opened"][bo] += 1
        elif a == "Close":
            dec(e["opened"], bo)
        elif a in ("Put", "PutIn"):
            dec(e["holding"], bo)
            e["loc"][(bo, bo2)] += 1
        elif a == "SwitchOn":
            e["on"][bo] += 1
        elif a == "SwitchOff":
            dec(e["on"], bo)

    def _shut(self, tok):
        """Is this container closed AND actually openable?"""
        if tok in self.opened:
            return False
        return tok in self.openable if self.openable else base(tok) in CONTAINERS

    @staticmethod
    def disp(tok):
        return base(tok)

    def _disp_set(self, toks):
        return ", ".join(sorted(base(t) for t in toks))

    def check(self, step):
        """Precondition dry-run -> (ok, why). why is the human-readable blocker."""
        a, o = step[0], (step[1] if len(step) > 1 else None)
        o2 = step[3] if len(step) > 3 else None
        if a == "Grab":
            c = self.inside.get(o)
            if c and self._shut(c):
                return False, f"the {base(c)} is closed — open it first"
            # grabbing what is already in hand (A33, f89): Unity refuses and the person heard "my body
            # couldn't do that here" while holding that very plate (pilot 4 aftermeal, t323)
            if o in self.holding:
                return False, f"I'm already holding the {base(o)}"
            if len(self.holding) >= 2:
                return False, "my hands are full"
        # kind capability (live-matrix fix #55): 'open the tablelamp' used to fall through to
        # the engine and come back as the generic "my body couldn't do that here"; the honest
        # reason is that the kind has no door.
        if a in ("Open", "Close") and self.openable and o not in self.openable \
                and not (base(o) in OPENS and "#" not in str(o)):
            return False, f"the {base(o)} is not something that opens"
        # a "nearest" token (cls@N) the offline twin cannot resolve names SOME instance in that
        # state -- "the nearest lamp that is on" -- so it is never refused as already off/open
        # (composite verifier 2026-09-09: shutdown's second lamp was refused as already off).
        _near = (SymState.LENIENT_NEAREST and o is not None and "@" in str(o) and _evid(o) == -1
                 and base(o) in (CONTAINERS if a in ("Open", "Close") else DEVICES))
        if a == "Open" and o in self.opened and not _near:  return False, f"the {base(o)} is already open"
        if a == "Close" and o not in self.opened and not _near: return False, f"the {base(o)} is already closed"
        # VH needs a free hand to work a door (fix #47): with both hands full the engine
        # refused with the opaque "my body couldn't do that here" — seven times in one run —
        # while the honest reason was never said.
        if a in ("Open", "Close") and len(self.holding) >= 2:
            return False, ("both my hands are full — I need to put something down before "
                           f"I can {a.lower()} the {base(o)}")
        if a in ("Put", "PutIn") and o not in self.holding:
            return False, f"I'm not holding the {base(o)} — my hands are " + \
                   (f"holding the {self._disp_set(self.holding)}" if self.holding else "empty")
        if a == "PutIn" and o2 and self._shut(o2):
            return False, f"the {base(o2)} is closed"
        if a == "SwitchOn" and o in self.on and not _near:  return False, f"the {base(o)} is already on"
        if a == "SwitchOff" and o not in self.on and not _near: return False, f"the {base(o)} is already off"
        if a == "Drink" and o not in self.holding: return False, f"I'm not holding the {base(o)}"
        if a == "Sit" and self.sitting: return False, "I'm already sitting"
        if a == "StandUp" and not self.sitting: return False, "I'm not sitting"
        return True, None

    def run(self, step, record=True):
        ok, why = self.check(step)
        if not ok:
            return False, why
        a, o = step[0], (step[1] if len(step) > 1 else None)
        o2 = step[3] if len(step) > 3 else None
        self._eff(a, o, o2)
        if a == "Grab":   self.holding.add(o); self.loc.pop(o, None); self.inside.pop(o, None)
        elif a == "Open": self.opened.add(o)
        elif a == "Close": self.opened.discard(o)
        elif a in ("Put", "PutIn"):
            self.holding.discard(o); self.loc[o] = o2
            (self.inside.__setitem__(o, o2) if a == "PutIn" else self.inside.pop(o, None))
        elif a == "SwitchOn": self.on.add(o)
        elif a == "SwitchOff": self.on.discard(o)
        elif a == "Sit": self.sitting = True
        elif a == "StandUp": self.sitting = False
        if record and a in pir.V2G:
            # (verb, cls, id, cls2, id2) -- the id slots were hardcoded to 0 since this file was
            # written, so every checkpoint graded at CLASS level and "put A plate away" scored
            # the same as "put THE plate away". They carry the real instance now.
            eid = _evid(o)
            if eid == -1 and o is not None:
                if a == "Grab":
                    self._synth_n += 1; eid = -1000 - self._synth_n; self._synth_held[o] = eid
                elif a in ("Put", "PutIn"):
                    eid = self._synth_held.pop(o, None)
                    if eid is None:
                        self._synth_n += 1; eid = -1000 - self._synth_n
                elif a in ("SwitchOn", "SwitchOff", "Open", "Close"):
                    self._synth_n += 1; eid = -1000 - self._synth_n
            self.events.append((pir.V2G[a], base(o), eid, base(o2), _evid(o2)))
        return True, None

    def semantic(self):
        """Canonical semantic state, cosmetics excluded (skill contract §1.6): what the task
        can depend on, in a deterministic serialisable shape."""
        return {
            "holding": sorted(map(str, self.holding)),
            "opened": sorted(map(str, self.opened)),
            "on": sorted(map(str, self.on)),
            "loc": {str(k): str(v) for k, v in sorted(self.loc.items(), key=lambda kv: str(kv[0]))},
            "inside": {str(k): str(v) for k, v in sorted(self.inside.items(), key=lambda kv: str(kv[0]))},
            "eff": {k: sorted((str(a), n) for a, n in v.items())
                    for k, v in sorted(self.eff.items())},
        }

    def hash(self):
        """sha256 of the canonical semantic state; equal across processes for equal replays."""
        return hashlib.sha256(
            json.dumps(self.semantic(), sort_keys=True).encode()).hexdigest()[:16]

    def checkpoints(self):
        return episode_goals.evaluate(self.gdef, self.events, self.targets)

    def hands(self):
        if not self.holding:
            return "NOTHING — your hands are empty"
        return "the " + " and the ".join(sorted(base(t) for t in self.holding))


_PASSAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_passages.json")


def scene_passages(scene):
    """Walk-verified open passages (doorways with no door node) for an apartment -- A32. They are
    doorways in every sense the agents meet: on the maps, for "go through the visible door", and for
    seeing into the next room. Empty if the scene has none or the file is missing."""
    try:
        return json.load(open(_PASSAGE_FILE)).get(str(scene), [])
    except Exception:
        return []


class LiveWorld:
    """One episode = one fresh Unity session on `port` (relaunch the container first)."""
    def __init__(self, port, ep_key, name, out_dir=None):
        self.late_notes = []    # feedback the world owes the person after the fact (a rewound move)
        self.port, self.epk, self.name = port, ep_key, name
        self.ep = episodes.EPISODES[ep_key]
        self.out = out_dir or paths.FRAMES; os.makedirs(self.out, exist_ok=True)
        # THE FRAME DIRECTORY MUST BE UNIQUE PER RUNNING PROCESS (2026-09-20). Unity writes frames to
        # unity_vol/Output/IV<tag>_*/ on a shared checkout, and each run deletes its own prefix
        # before rendering. Keyed on name+port alone, two runs of the same cell on the same port -- twelve
        # Beaker jobs playing the same check cells, or two sweeps in one session -- get the SAME tag and
        # delete each other's frames mid-episode ("FileNotFoundError: Action_0021_0_normal.png", 3/3 cells
        # of every check job). VH_RUN_TAG (set by the lane runner to arm+host+pid) keeps them apart.
        _run_tag = os.environ.get("VH_RUN_TAG", "")
        self.tag = hashlib.md5(f"{name}{port}{_run_tag}".encode()).hexdigest()[:6]
        self.container = f"vh{int(port) - 8080}"
        self.c = UnityCommunication(url="127.0.0.1", port=str(port), timeout_wait=600)
        self.sym = SymState(ep_key)
        self.frames = []      # per RAW frame: path to ego png
        self.pos, self.yaw = [], []       # smoothed, per raw frame
        self.pos3 = []                    # RAW hips (x,y,z) per frame — what move_character needs
        self._ryaw = []
        self.log = []         # command log entries {cmd, step, status, t_issue, t_done, kind, why}
        self.tick = 0
        self._last_pose3 = None
        self.item_i = 0
        self._plan_inst = {}   # class -> instance the SCRIPT means, pinned for the episode
        self._setup_scene()

    # ---------------- unity plumbing
    def _root_rm(self, path):
        vhfix.root_rm(path, self.container)

    def _dedupe_instances(self):
        """Delete surplus copies of a class named in ep['dedupe'] (e.g. keep only 1 milk), so the
        scene holds a single instance and the nearest-instance binding cannot grab a decoy during
        recovery. Prunes the extra nodes + their edges and re-applies via expand_scene (verified
        to remove nodes on this build). Runs before M/G are built and before setup, so everything
        downstream sees a scene with one milk."""
        want = self.ep.get("dedupe") or {}
        if not want:
            return
        try:
            _s, g = self.c.environment_graph()
            by_cls = {}
            for n in g["nodes"]:
                by_cls.setdefault(n["class_name"], []).append(n["id"])
            # Two spellings. `{"milk": 1}` keeps the first copy in graph order (the original).
            # `{"plate": {"room": "kitchen", "keep": 1}}` keeps `keep` copies that sit INSIDE the
            # named room and deletes every other copy in the house -- the instruction can then
            # say "the plate on the kitchen table" and there is exactly one thing it can mean.
            # Graph order is not room order, so the old spelling could keep a bathroom plate.
            rooms = {n["id"]: n["class_name"] for n in g["nodes"] if n["category"] == "Rooms"}
            inroom = {e["from_id"]: rooms[e["to_id"]] for e in g["edges"]
                      if e["relation_type"] == "INSIDE" and e["to_id"] in rooms}
            drop = set()
            for cls_, spec in want.items():
                ids = by_cls.get(cls_, [])
                if isinstance(spec, (dict, list)):
                    # a list keeps copies in SEVERAL rooms: [{"room": "bedroom", "keep": 2},
                    # {"room": "livingroom", "keep": 1}] -- for "all three books" when no
                    # single room holds three
                    keep = set()
                    for sp in (spec if isinstance(spec, list) else [spec]):
                        here = [i for i in ids if inroom.get(i) == sp.get("room")]
                        keep |= set(here[:int(sp.get("keep", 1))])
                    drop |= set(ids) - keep
                else:
                    drop |= set(ids[int(spec):])
            if not drop:
                return
            g2 = dict(g)
            g2["nodes"] = [n for n in g["nodes"] if n["id"] not in drop]
            g2["edges"] = [e for e in g["edges"]
                           if e["from_id"] not in drop and e["to_id"] not in drop]
            ok, _m = self.c.expand_scene(g2)
            print(f"[world] dedupe {want}: dropped {sorted(drop)} (ok={ok})", flush=True)
        except Exception as e:
            print(f"[world] dedupe failed: {e}", flush=True)


    # ---------------- the picture camera
    # The pitch ladder (EGO00..EGO60) was a set of cameras ADDED with add_character_camera before
    # add_character. Their parenting to the character is unreliable: in a fraction of setups an
    # added camera never attaches and renders from the WORLD ORIGIN for the whole episode -- a
    # fixed wide shot, outside the building in apartments whose footprint does not cover the
    # origin (3, 4), a plausible-looking wide interior in scene 2 where the origin is indoors.
    # Every "the person just walks between rooms" film and, very likely, every "object
    # unreachable, endless zero-frame turns" failure came from that. Rendering two cameras at
    # once, the added camera produced NO frames at all. The built-in Character_Camera_Fwd is part
    # of the character prefab and was attached in every test, and update_camera gives it the
    # same position, pitch and FOV the ladder had. So every render routes through it.
    def survey(self, oids, hold=None):
        """ESTABLISHING SHOT: from where the person stands, aim at each object in `oids` in turn
        and hold for a second, appending the frames. A reference film must SHOW where things
        start -- a toothbrush lying on a sofa that never enters the frame before it is picked up
        reads as the thing appearing from nowhere. Nothing moves; the body does not turn."""
        hold = hold or FPT
        try:
            fp = self._fp_index()
            for oid in oids:
                if oid is None:
                    continue
                p = pitch_to(self.c, oid) or WALK_PITCH
                self._aim_yaw = yaw_to(self.c, oid, limit=90.0)
                self._cam_for(ego_cam_name(p))
                ok, imgs = self.c.camera_image([fp], mode="normal", image_width=640, image_height=480)
                if not ok or not imgs:
                    continue
                d = f"{paths.UNITY_DIR}/Output/IV{self.tag}_SURVEY{oid}/0"; os.makedirs(d, exist_ok=True)
                path = f"{d}/survey_0000_normal.png"
                Image.fromarray(np.asarray(imgs[0])[..., :3].astype(np.uint8)).save(path)
                for _ in range(hold):
                    self._append_frame(path, None)
        except Exception as e:
            print(f"[world] survey skipped: {e}", flush=True)
        finally:
            self._aim_yaw = 0.0

    def _fp_index(self):
        if getattr(self, "_fp_idx", None) is None:
            try:
                for i in range(self.c.camera_count()[1]):
                    ok, d = self.c.camera_data([i])
                    if d and (d[0] or {}).get("name") == "Character_Camera_Fwd":
                        self._fp_idx = i; break
            except Exception:
                self._fp_idx = None
        return self._fp_idx

    def _cam_for(self, cam):
        """Camera-mode entry for render_script: an EGOxx ladder name becomes the built-in
        first-person camera set to that pitch; anything else passes through unchanged."""
        if isinstance(cam, str) and cam.startswith("EGO") and cam[3:].isdigit():
            fp = self._fp_index()
            if fp is not None:
                try:
                    self.c.update_camera(fp, position=[0, EYE_DY, 0.25],
                                         rotation=[int(cam[3:]), float(getattr(self, "_aim_yaw", 0.0)), 0],
                                         field_view=EGO_FOV)
                    return "FIRST_PERSON"
                except Exception:
                    pass
        return cam

    def _setup_scene(self):
        print(f"[world] reset scene {self.ep['scene']} on :{self.port}", flush=True)
        self.c.reset(self.ep["scene"])
        self._dedupe_instances()          # prune surplus copies BEFORE the character is added
        register_ego_ladder(self.c)            # must be registered BEFORE add_character
        self.c.add_character("Chars/Female1", initial_room=self.ep["init"])
        self._fp_idx = None                    # camera indices differ per apartment
        print("[world] character added", flush=True)
        s, g = self.c.environment_graph()
        rooms = {n["id"]: n["class_name"] for n in g["nodes"] if n["category"] == "Rooms"}
        objroom = {e["from_id"]: rooms[e["to_id"]] for e in g["edges"]
                   if e["relation_type"] == "INSIDE" and e["to_id"] in rooms}
        self.M, self.G = {}, {}
        for n in g["nodes"]:
            rm = objroom.get(n["id"])
            if rm:
                self.M.setdefault((n["class_name"], rm), n["id"])
                self.G.setdefault((n["class_name"], rm), []).append(n["id"])
        self.rooms_of = {}
        for (cls, rm) in self.M:
            self.rooms_of.setdefault(cls, []).append(rm)
        self.fp = floorplan(g)
        run_setup(self.c, self.ep.get("setup", []), lambda s: self._line(tuple(s)),
                  log=lambda m: print("[world]" + m, flush=True))
        # RE-HOME. A setup that carries things around leaves the body wherever its last step
        # ended -- typically right beside the first thing the task will walk to -- and a
        # zero-length first walk trips the teleport guard (three tasks died that way). `init`
        # is where the person is supposed to START, so after a moving setup walk her back to a
        # piece of furniture that anchors that room, unrecorded.
        if any(str(st[0]) == "Walk" for st in (self.ep.get("setup") or [])):
            anchor = {"bedroom": "bed", "livingroom": "sofa", "kitchen": "kitchentable",
                      "bathroom": "toilet"}.get(self.ep["init"])
            try:
                aid = self.M.get((anchor, self.ep["init"])) if anchor else None
                if aid:
                    self.c.render_script([f"<char0> [Walk] <{anchor}> ({aid})"], recording=False,
                                         find_solution=False, skip_animation=True)
                    print(f"[world]  [setup] re-homed to the {self.ep['init']} ({anchor})", flush=True)
            except Exception as e:
                print(f"[world]  [setup] re-home skipped: {e}", flush=True)
        # A setup Put can FAIL SILENTLY in Unity (a pillow "put on the kitchen table" stayed in
        # her hand: Unity reported success, the graph kept the HOLDS edge), after which every
        # later Open/Grab is refused with "my hands are full" and the reference film shows the
        # person carrying a pillow she never picked up. Say so loudly instead of letting the
        # episode proceed on a false premise.
        try:
            _s, g2 = self.c.environment_graph()
            held = [next((n["class_name"] for n in g2["nodes"] if n["id"] == e["to_id"]), "?")
                    for e in g2["edges"] if e["relation_type"] in ("HOLDS_RH", "HOLDS_LH")]
            if held:
                print(f"[world]  [setup] WARNING: setup left {held} IN HAND -- a setup Put failed "
                      f"silently; the task starts with occupied hands", flush=True)
        except Exception:
            pass
        self._sync_sym()          # symbolic state starts identical to the rendered world
        self._bind_targets()
        # camera warmup (discarded) — prevents the late-attach detached first shot
        self._root_rm(f"/unity_vol/Output/WU{self.tag}")
        try:
            self.c.render_script(["<char0> [TurnLeft]", "<char0> [TurnRight]"], recording=True,
                                 find_solution=False, frame_rate=5, output_folder="Output/",
                                 file_name_prefix=f"WU{self.tag}", image_width=64, image_height=48,
                                 camera_mode=[self._cam_for(ego_cam_name(WALK_PITCH))],
                                 save_pose_data=True)
        except Exception:
            pass
        w = parse_pose(f"WU{self.tag}")
        self.prev_rows = 0 if w is None else len(w[0])
        self._last_pose = ([0.0, 0.0], 0.0)

    def _sync_sym(self):
        """Rebuild the symbolic state from the rendered world, keyed by instance.

        SymState is constructed before M/G exist, so its own setup replay can only produce
        class-level keys ('milk'), which would then never match the instance keys ('milk#176')
        that execution produces -- a Grab would see no container and skip the open-the-fridge
        precondition. Reading the finished graph avoids the replay entirely and starts the two
        worlds in agreement.
        """
        try:
            s, g = self.c.environment_graph()
        except Exception as e:
            print(f"[world] sym sync failed ({e}); keeping replayed state", flush=True)
            return
        cls = {n["id"]: n["class_name"] for n in g["nodes"]}
        rooms = {n["id"] for n in g["nodes"] if n.get("category") == "Rooms"}
        key = lambda i: f"{cls[i]}#{i}"
        # Seed ONLY what this episode's plan touches. Seeding the whole graph looks more
        # faithful but destroys comparability with pir.simulate(), which starts empty and
        # records just the objects the plan moves: the step-in trigger compares those two
        # states, and a live state carrying 183 objects, three open doors and eight lit
        # lightswitches could never equal the reference, so the injection never fired at all.
        relevant = set()
        for st in list(self.ep.get("setup", [])) + list(self.ep["steps"]):
            if len(st) > 1: relevant.add(base(st[1]))
            if len(st) > 3: relevant.add(base(st[3]))
        relevant |= set((self.ep.get("targets") or {}).keys())
        relevant.discard(None)
        keepid = lambda i: cls.get(i) in relevant
        self.sym.loc, self.sym.inside = {}, {}
        for e in g["edges"]:
            if e["relation_type"] in ("ON", "INSIDE") and e["to_id"] not in rooms \
                    and e["from_id"] in cls and e["to_id"] in cls and keepid(e["from_id"]):
                self.sym.loc.setdefault(key(e["from_id"]), key(e["to_id"]))
                if e["relation_type"] == "INSIDE":
                    self.sym.inside.setdefault(key(e["from_id"]), key(e["to_id"]))
        st = {n["id"]: set(n.get("states") or []) for n in g["nodes"]}
        self.sym.openable = {key(n["id"]) for n in g["nodes"]
                             if "CAN_OPEN" in (n.get("properties") or []) and n["class_name"] in OPENS}
        self.sym.opened = {key(i) for i, s_ in st.items() if "OPEN" in s_ and keepid(i)}
        self.sym.on = {key(i) for i, s_ in st.items() if "ON" in s_ and keepid(i)}
        ch = next((n["id"] for n in g["nodes"] if n["class_name"] == "character"), None)
        self.sym.holding = {key(e["to_id"]) for e in g["edges"]
                            if e["from_id"] == ch and str(e["relation_type"]).startswith("HOLDS")
                            and e["to_id"] in cls}

    def _bind_targets(self):
        """Pin the instance the TASK means, e.g. 'the milk from the fridge' -> milk#176.

        Declared per episode as {class: anchor_class} and resolved once, right after setup, so
        the binding survives the object being moved. Grading consults this; command resolution
        deliberately does not (see _pick).
        """
        want = self.ep.get("targets") or {}
        anch, _, _ = self._anchor_map()
        for cls_, anchor in want.items():
            room = self.room_of(cls_)
            ids = self.G.get((cls_, room), [])
            hit = [i for i in ids if anch.get(i) == anchor]
            if len(hit) == 1:
                self.sym.targets[cls_] = hit[0]
            else:
                print(f"[world] WARNING target {cls_}@{anchor} resolved to {hit or ids} "
                      f"— grading falls back to class level", flush=True)

    def _anchor_map(self):
        """id -> class of the container/surface it is currently ON or INSIDE (rooms excluded).

        Queried live because anchors move: 'the milk in the fridge' identifies an instance only
        at the moment it is uttered, while the instance id identifies it for the whole episode.
        """
        try:
            s, g = self.c.environment_graph()
        except Exception:
            return {}, {}, None
        cls = {n["id"]: n["class_name"] for n in g["nodes"]}
        rooms = {n["id"] for n in g["nodes"] if n.get("category") == "Rooms"}
        anch = {}
        for e in g["edges"]:
            if e["relation_type"] in ("ON", "INSIDE") and e["to_id"] not in rooms:
                anch.setdefault(e["from_id"], cls.get(e["to_id"]))
        pos = {n["id"]: n["bounding_box"]["center"] for n in g["nodes"]}
        ch = next((n for n in g["nodes"] if n["class_name"] == "character"), None)
        return anch, pos, (ch["bounding_box"]["center"] if ch else None)

    def _nearest(self, ids):
        """Closest candidate, preferring ones the body can actually get to.

        Distance alone picked tablelamp#376 in tidy_hard -- a lamp that rests on nothing (its
        anchor is None; in this scene that means wall- or ceiling-mounted). It can be switched
        but never walked to, so every `walk to the tablelamp` was refused with "my body couldn't
        do that here" while the switch that followed succeeded. An object sitting ON or IN
        something has a reachable footprint; an unanchored one usually does not, so anchored
        candidates win and unanchored ones are only used when nothing else exists.
        """
        anch, pos, cp = self._anchor_map()
        if cp is None:
            return ids[0]
        cand = [(sum((pos[i][k] - cp[k]) ** 2 for k in (0, 2)), i) for i in ids if i in pos]
        if not cand:
            return ids[0]
        anchored = [c for c in cand if anch.get(c[1])]
        return min(anchored or cand)[1]

    def _instances_in_body_room(self, cls):
        """Ids of `cls` INSIDE the room the body stands in, from the live graph ([] on any failure)."""
        try:
            _s, g = self.c.environment_graph()
            here = self.current_room_label()
            labels = self.room_labels(g)
            ids = {n["id"] for n in g["nodes"] if n["class_name"] == cls}
            return [e["from_id"] for e in g["edges"] if e["relation_type"] == "INSIDE"
                    and e["from_id"] in ids and labels.get(e["to_id"]) == here]
        except Exception:
            return []

    def _walk_surrogate(self, step):
        """(script line, class) of the furniture the walk target rests ON/IN, or None."""
        tid = inst(step[1]) if len(step) > 1 else None
        if tid is None:
            return None
        try:
            _s, g = self.c.environment_graph()
        except Exception:
            return None
        byid = {n["id"]: n for n in g["nodes"]}
        rooms = {n["id"] for n in g["nodes"] if n.get("category") == "Rooms"}
        # never floor/rug/wall/ceiling: a bed or sofa rests ON the floor, and a walk to a floor node is no surrogate
        flat = {"floor", "rug", "wall", "ceiling"}
        sup = next((e["to_id"] for e in g["edges"] if e["from_id"] == tid and e["relation_type"] in ("ON", "INSIDE")
                    and e["to_id"] not in rooms and e["to_id"] in byid
                    and byid[e["to_id"]]["class_name"] not in flat
                    and byid[e["to_id"]].get("category") not in ("Floor", "Walls", "Ceiling")), None)
        if sup is None:
            return None
        return f"<char0> [Walk] <{byid[sup]['class_name']}> ({sup})", byid[sup]["class_name"]

    def _pick(self, tok, room):
        """Object token -> (instance id, class name).

        An ambiguous bare class resolves to the NEAREST candidate and is NOT nudged toward the
        task's target instance. Reaching for the decoy milk is therefore a genuine wrong_object
        error that stands in the record for the assistant to catch -- which is the whole point of
        grounding to an instance rather than to a name.
        """
        if "#" in tok:                                    # already instance-resolved
            b, i = tok.split("#")
            return int(i), b
        if "@" in tok:
            b, q = tok.split("@")
            ids = self.G.get((b, room), [])
            if q.isdigit():                               # positional, used by episode setups
                i = int(q) - 1
                return (ids[i] if 0 <= i < len(ids) else None), b
            anch, _, _ = self._anchor_map()               # relational: "the milk in the fridge"
            hit = [i for i in ids if anch.get(i) == q]
            return (hit[0] if hit else None), b
        ids = self.G.get((tok, room), [])
        if len(ids) > 1:
            return self._nearest(ids), tok
        return (ids[0] if ids else self.M.get((tok, room))), tok

    def _line(self, step):
        # Only ever called for SCRIPTED content (episode setup, and steps already resolved by
        # _resolve_step), so it pins through the same memo -- otherwise the setup could light
        # tablelamp#376 while the plan goes on to switch off tablelamp#256.
        a = step[0]
        if len(step) == 1: return "<char0> [%s]" % a
        if len(step) == 3:
            i, nm = self._resolve_tok(step[1], step[2], True)
            return None if i is None else "<char0> [%s] <%s> (%d)" % (a, nm, i)
        i, nm = self._resolve_tok(step[1], step[2], True)
        j, nm2 = self._resolve_tok(step[3], step[4], True)
        return None if (i is None or j is None) else \
            "<char0> [%s] <%s> (%d) <%s> (%d)" % (a, nm, i, nm2, j)

    def canon_class(self, cls):
        """Canonical scene class for a colloquial name, by suffix (fix #47): 'counter' ->
        'kitchencounter'. The mid-level resolver has suffix-matched since day one, but
        parse_cmd's room grounding did not — so 'put the juice on the counter' parsed to
        None and was SILENTLY dropped from the human's plan, ten times in one run, while it
        kept promising to do it. Exact names pass through; ambiguity keeps the shortest
        match (most specific suffix owner)."""
        cls = base(cls)
        if cls in self.rooms_of:
            return cls
        cands = sorted(k for k in self.rooms_of if k.endswith(cls))
        return cands[0] if cands else cls

    def _rooms_with(self, cls):
        """Rooms that hold an instance of `cls` in the LIVE scene (None if the graph read fails)."""
        try:
            _s, g = self.c.environment_graph()
            rooms = {n["id"]: n["class_name"] for n in g["nodes"] if n.get("category") == "Rooms"}
            ids = {n["id"] for n in g["nodes"] if n["class_name"] == cls}
            found = sorted({rooms[e["to_id"]] for e in g["edges"]
                            if e["relation_type"] == "INSIDE" and e["from_id"] in ids and e["to_id"] in rooms})
            return found
        except Exception:
            return None

    def room_of(self, cls):
        """Room grounding for a bare class: prefer where the symbolic loc's container sits,
        else the episode-listed room, else any room the scene knows."""
        cls = self.canon_class(cls)
        # Consult the PLAN-EFFECT mirror, not the full symbolic loc. The branch exists for
        # "the plan put X inside a container, so X is wherever that container is" -- but the
        # instance state is seeded from the scene graph, so it also knows static facts like
        # tv -> tvstand. Following those derived the wrong room and made `walk to the tv`
        # unresolvable ("I can't find that here") in tidy_hard.
        c = self.sym.eff["loc"].get(cls)
        if c is not None:
            r = self.room_of(base(c))
            if r: return r
        for s in list(self.ep.get("setup", [])) + list(self.ep["steps"]):
            if len(s) > 2 and base(s[1]) == cls: return s[2]
            if len(s) > 4 and base(s[3]) == cls: return s[4]
        rs = self.rooms_of.get(cls) or []
        for pref in ("kitchen", "livingroom", "bedroom", "bathroom"):
            if pref in rs: return pref
        return rs[0] if rs else None

    # ---------------- frame bookkeeping
    def _append_frame(self, path, pose):
        if pose is not None:
            self._last_pose = pose
        p, y = self._last_pose
        k = len(self.pos)
        if k == 0:
            self.pos.append(list(p)); self.yaw.append(y)
        else:
            sp = [0.75 * p[0] + 0.25 * self.pos[-1][0], 0.75 * p[1] + 0.25 * self.pos[-1][1]]
            self.pos.append(sp)
            a = max(0, k - 4)
            v = [sp[0] - self.pos[a][0], sp[1] - self.pos[a][1]]
            if (v[0] ** 2 + v[1] ** 2) ** 0.5 > 0.10:
                self.yaw.append(float(np.degrees(np.arctan2(v[0], v[1]))))
            else:
                self.yaw.append(self.yaw[-1])
        self.frames.append(path)
        while len(self.pos3) < len(self.frames):
            self.pos3.append(list(self._last_pose3) if self._last_pose3 else [p[0], 0.95, p[1]])
        if len(self.frames) % FPT == 0:
            self.tick += 1

    def rewind_to(self, frame_idx):
        """Discard every frame from `frame_idx` on and physically put the character back at that
        frame's pose. This is what makes an in-flight action interruptible: the clip is rendered
        once, streamed tick by tick, and if the plan changes mid-action the unwanted tail is
        thrown away rather than having been committed.

        move_character() is verified to affect LATER renders on this build (probe_move_character:
        0.04 m from the target vs 8.08 m to the walk's endpoint) — unlike
        update_character_camera(), which returns ok=True and does nothing (ENGINE_NOTES A3).
        """
        frame_idx = max(0, min(int(frame_idx), len(self.frames)))
        if frame_idx >= len(self.frames):
            return False
        dropped = len(self.frames) - frame_idx
        tgt = self.pos3[frame_idx - 1] if frame_idx else self.pos3[0]
        self.frames = self.frames[:frame_idx]
        self.pos = self.pos[:frame_idx]
        self.yaw = self.yaw[:frame_idx]
        self.pos3 = self.pos3[:frame_idx]
        self.tick = frame_idx // FPT
        # D10: the action the cut fell inside no longer ended when the log said it did.
        # Only MOVEMENT is reversed by a rewind (frames and body pose). A manipulation's state
        # change was already committed in Unity and in the symbolic state when its clip was
        # rendered, so a cut close/put/switch still happened: it stays `done`, marked as cut
        # on camera (pilot 3: a cut "close the fridge" was logged interrupted over a shut fridge).
        MOVES = ("Walk", "MidLevel", "Idle", "TurnTo", "LookAt")
        for l in self.log:
            if l.get("status") == "done" and (l.get("t_done") or 0) > self.tick \
                    and (l.get("t_issue") or 0) <= self.tick:
                if (l.get("step") or [None])[0] in MOVES:
                    l["status"] = "interrupted"; l["t_done"] = self.tick
                    l["why"] = "stopped partway — the person changed plan on the assistant's words"
                    self.late_notes.append(f'you STARTED "{l["cmd"]}" but stopped partway when '
                                           f'you changed plan — you did not finish it')
                else:
                    l["cut_on_camera"] = self.tick
        ok = self.c.move_character(0, [float(v) for v in tgt])
        print(f"[world] REWIND to frame {frame_idx} (t{self.tick}) — dropped {dropped} frames, "
              f"body -> {[round(v,2) for v in tgt]} ok={ok}", flush=True)
        return bool(ok)

    def idle_ticks(self, n):
        """n seconds of the world standing still (frozen last frame). With NO frame yet (a
        self-planned episode idles before its first action) take one live ego frame to hold,
        so time can pass from t0 and the first poll comes due."""
        if not self.frames:
            try:
                fp = self._fp_index()
                self._cam_for(ego_cam_name(WALK_PITCH))
                ok, imgs = self.c.camera_image([fp], mode="normal", image_width=640, image_height=480)
                if ok and imgs:
                    d = f"{paths.UNITY_DIR}/Output/IV{self.tag}_IDLE0/0"; os.makedirs(d, exist_ok=True)
                    path = f"{d}/idle_0000_normal.png"
                    Image.fromarray(np.asarray(imgs[0])[..., :3].astype(np.uint8)).save(path)
                    self._append_frame(path, None)
            except Exception as e:
                print(f"[world] idle seed frame failed: {e}", flush=True)
            if not self.frames:
                return
        last = self.frames[-1]
        for _ in range(n * FPT):
            self._append_frame(last, None)

    # ---------------- camera aiming
    def _aim(self, step):
        return aim_camera(self.c, step, lambda tk, rm: self._pick(tk, rm)[0])

    # ---------------- command execution
    # A SCRIPTED plan means one object by "the tablelamp" for the whole episode; an agent typing
    # "grab the milk" means whichever one it is standing next to. Those are different questions
    # and only the second one is nearest-resolved.
    AGENT_KINDS = ("human_self", "vlm_fix")

    # Verbs whose object must ALREADY BE IN HAND. You cannot put down or drink a thing you are
    # not holding, so proximity is the wrong question for them.
    HELD_VERBS = ("Put", "PutIn", "Drink", "Read")

    def _resolve_tok(self, tok, room, scripted, held=False):
        """Bind one object token to an instance.

        `held` restricts the candidates to what the agent is actually carrying. Without it a
        `put the milk down` resolved by NEARNESS, and with a decoy milk sitting on the counter
        it picked that one -- so the command was refused ("I'm not holding the milk") while the
        agent was holding the other one. In iv_wrongobj the human was carrying milk and juice,
        reasoned correctly that it had to free a hand before opening the microwave, and could
        not put either down: every attempt resolved to the copy on the counter. The episode
        deadlocked until the no-progress guard ended it.
        """
        if scripted and "#" not in tok and "@" not in tok:
            # memo per (class, room): a setup that walked to the LIVING-ROOM desk must not pin
            # the plan's later "desk in the bedroom" to it (putback film 2026-09-04)
            memo = self._plan_inst.get((tok, room))
            if memo is not None:
                return memo, tok
        if held and "#" not in tok:
            # a deposit of a class you are holding means the one in your hand -- for a bare
            # token AND for a nearest-indexed one (plate@2): the plan's deposit(plate@2) was
            # resolved by nearness to the plate on the table and refused "I'm not holding the
            # plate -- my hands are holding the plate" (pilot 2, 2026-09-09)
            mine = [k for k in self.sym.holding if base(k) == base(tok)]   # instance-level: intentional
            if mine:
                return inst(mine[0]), tok
        if not scripted and "#" not in tok and "@" not in tok:
            # THE COPY IN THIS ROOM (A33, f89). G/M are built before run_setup, so an object the
            # setup carried elsewhere is still filed under its old room, and room_of() grounds a
            # bare "mug" to the kitchen from the episode spec. A person in the bathroom next to the
            # bathroom mug said "walk to the mug" and the body walked to the KITCHEN and picked up
            # the mug already delivered there (pilot 5, 16 such walks; dishes 8/8 from the bedroom).
            # An agent's bare reference means the nearest copy in the room it is standing in, read
            # from the live graph; only when this room has none does the old grounding apply.
            here = self._instances_in_body_room(base(tok))
            if here:
                return self._nearest(here), tok
        i, nm = self._pick(tok, room)
        if scripted and i is not None and "#" not in tok and "@" not in tok:
            self._plan_inst[(tok, room)] = i
        return i, nm

    def _mlctrl(self):
        """The mid-level controller, bound to this world's Unity session and made to render
        through US: its own renderer is two-pass (decide fast, replay once at the end), which
        cannot give the assistant frames while the episode is still running."""
        if getattr(self, "_ml", None) is None:
            from vhhome.actions import MidLevelController
            self._ml = MidLevelController(self.c)
            self._ml._passages = scene_passages(self.ep["scene"])
            # EXECUTE ONCE, ANIMATED. The first hook ran skip_animation (state applied, no
            # frames) and exec_step then re-rendered the collected script — every movement was
            # executed TWICE, and the search-era path three times. Now the hook itself is the
            # one and only execution: when _ml_record is set, each script group midlevel emits
            # is rendered animated+recorded under its own sub-prefix and ingested immediately,
            # so the assistant sees the motion and the body moves exactly once.
            self._ml_record = None      # (prefix, cam) while an agent movement is executing
            self._ml_subs = [0]

            def _hook(script, fs):
                if self._ml_record is None:      # gate probes etc.: fast, stateless render
                    return self.c.render_script(list(script), recording=False,
                                                find_solution=fs, skip_animation=True)
                prefix, cam = self._ml_record
                sub = f"{prefix}S{self._ml_subs[0]:02d}"
                self._ml_subs[0] += 1
                self._root_rm(f"/unity_vol/Output/{sub}")
                try:
                    ok, msg = self.c.render_script(list(script), recording=True,
                                                   find_solution=fs, frame_rate=5,
                                                   output_folder="Output/",
                                                   file_name_prefix=sub, image_width=640,
                                                   image_height=480,
                                                   camera_mode=[self._cam_for(cam)],
                                                   save_pose_data=True)
                except Exception as e:
                    return False, str(e)
                fs_ = sorted(glob.glob(f"{paths.UNITY_DIR}/Output/{sub}/0/Action_*_normal.png"),
                             key=lambda p: int(p.split("Action_")[1].split("_")[0]))
                if fs_:
                    self._ingest_render(sub, fs_)
                return ok, str(msg)[:90]

            self._ml.render_hook = _hook
            # The controller normally resolves its first-person camera inside reset(), which also
            # resets the SCENE. This world owns the scene, so locate the camera on its own —
            # without it every observation() raises and the visibility gate cannot be evaluated.
            self._ml._locate_fp()
        return self._ml

    def _resolve_step(self, step, kind="plan"):
        """Bind a step's object tokens to concrete instances BEFORE anything else sees it.

        Resolution has to happen once, up front: the symbolic precondition check, the Unity
        script line and the recorded event must all refer to the SAME object, or the state we
        grade against drifts from the state we rendered.

        Scripted steps additionally pin the instance for the rest of the episode. Re-resolving
        per command let a plan contradict itself the moment the body moved: in tidy_hard the
        clean plan switched OFF tablelamp#256, then the injected `repetition` walked back to
        "the tablelamp" from the kitchen, resolved to #376 because that one was now nearer, and
        was refused as "already on" -- so the repetition error never happened at all while the
        run still reported itself healthy.
        """
        if len(step) < 3:
            return step
        scripted = kind not in self.AGENT_KINDS
        out = list(step)
        i, nm = self._resolve_tok(step[1], step[2], scripted, held=step[0] in self.HELD_VERBS)
        # A bound token is CLASS#ID, never CLASS@N#ID: the hands and every state set key on
        # 'plate#75', so 'plate@3#75' failed a string membership test and a deposit of the
        # plate in hand was refused "I'm not holding the plate -- my hands are holding the
        # plate" (pilot 3, three times; A20).
        out[1] = f"{base(nm)}#{i}" if i is not None else step[1]
        if len(step) > 4:
            j, nm2 = self._resolve_tok(step[3], step[4], scripted)
            out[3] = f"{base(nm2)}#{j}" if j is not None else step[3]
        return tuple(out)

    def _ingest_render(self, prefix, fs):
        """Fold one recorded render's frames+poses into the episode stream.

        Shared by the normal per-command render and the mid-level recording hook (which renders
        several small script groups under sub-prefixes within one agent command). Pose files are
        CUMULATIVE across renders (ENGINE_NOTES A5): each new pd_*.txt repeats every row since
        scene start, so the new rows are the slice past `prev_rows`.
        """
        pp = parse_pose(prefix, want3d=True)
        p3 = None
        if pp is not None:
            total = len(pp[0])
            p3 = pp[2][self.prev_rows:]
            pp = (pp[0][self.prev_rows:], pp[1][self.prev_rows:])
            self.prev_rows = total
            if not len(pp[0]):
                pp, p3 = None, None
        n = len(fs)
        for k, f in enumerate(fs):
            pose = None
            if pp is not None and len(pp[0]):
                j = min(int(round(k / max(1, n - 1) * (len(pp[0]) - 1))), len(pp[0]) - 1)
                pose = (pp[0][j].tolist(), float(pp[1][j]))
                if p3 and j < len(p3):
                    self._last_pose3 = p3[j]   # raw 3-D hips for rewind_to/move_character
            self._append_frame(f, pose)

    def exec_step(self, step, kind="plan", cmd_text=None):
        """Execute ONE engine step live. Returns the log entry (status done|rejected).

        Thin wrapper: after every execution — every exit path of the inner body — the
        frame/pose COHERENCE invariant (A) is repaired if needed: the newest frame in the
        stream must have been captured from the body's current pose. Individual causes
        (zero-frame turns #21, cinematic cameras #28, pitch ladder #36) had point fixes;
        this audit catches every cause, including ones not seen yet."""
        entry = self._exec_step_inner(step, kind, cmd_text)
        # completion-room stamp (trajectory-audit fix #57): `room` is captured at ISSUE time,
        # so a cross-room walk logged "done [room=bedroom]" — fresh readers could not
        # adjudicate stall flags against it, and neither could anyone else.
        try:
            entry["room_done"] = self.current_room_label()
        except Exception:
            pass
        if MIDLEVEL:
            self._coherence_repair()
        return entry

    def _coherence_repair(self, tol_m=0.5, tol_deg=25.0):
        try:
            true_pose = self._body_pose()
            last = getattr(self, "_last_pose", None)
            if true_pose is None or not self.frames or not last:
                return
            (lx, lz), lyaw = last
            dx = true_pose[0] - lx
            dz = true_pose[1] - lz
            # yaw sources differ in convention (camera-forward vs character-node yaw are 180
            # deg apart), so the check is 180-symmetric: only a genuine quarter-turn-or-more
            # divergence counts. Position is compared strictly.
            dyaw = abs((true_pose[2] - lyaw + 180.0) % 360.0 - 180.0)
            dyaw = min(dyaw, abs(180.0 - dyaw))
            if (dx * dx + dz * dz) ** 0.5 <= tol_m and dyaw <= tol_deg:
                return
            prefix = f"IV{self.tag}_COH{self.item_i:03d}"
            self.item_i += 1
            snap, pose = self._ego_snapshot(prefix)
            if snap is not None:
                print(f"[world] t{self.tick} COHERENCE repair (d={((dx*dx+dz*dz)**0.5):.2f}m "
                      f"dyaw={dyaw:.0f})", flush=True)
                for _ in range(FPT):
                    self._append_frame(snap, pose)
        except Exception as e:
            print(f"[world] coherence repair failed open-armed: {e}", flush=True)

    def _exec_step_inner(self, step, kind="plan", cmd_text=None):
        """Execute ONE engine step live. Returns the log entry (status done|rejected)."""
        disp = cmd_text or pretty_cmd(step)
        gate_refusal = None
        if MIDLEVEL and kind in self.AGENT_KINDS and step[0] != "MidLevel":
            # PERCEPTION GATE, before anything else. It must precede both resolution and the
            # symbolic precondition check: the gate binds the command to the instance the body
            # can SEE, and running sym.check first would judge the graph-nearest instance
            # instead (e.g. refuse "the fridge is closed" about the milk inside the fridge
            # while the human is reaching for the one on the counter in front of it).
            from vhhome.gates import gate_step
            ok_g, why_g, slot, oid = gate_step(self._mlctrl(), step)
            if not ok_g:
                gate_refusal = why_g
            elif slot is not None and oid is not None:
                s2 = list(step)
                s2[slot] = f"{base(s2[slot])}#{oid}"
                step = tuple(s2)
        step = self._resolve_step(step, kind)
        entry = dict(cmd=disp, step=list(step), kind=kind,
                     status="running", t_issue=self.tick, why=None,
                     room=self.current_room_label(),
                     instances=[inst(step[1]) if len(step) > 1 else None,
                                inst(step[3]) if len(step) > 3 else None])
        self.log.append(entry)
        if gate_refusal is not None:
            # PROGRESSIVE REFUSAL (fix #43). A small object's visibility window can be
            # narrower than the agent's scan step: one run tried 'walk to the microwave' TEN
            # times while scanning in 90-degree jumps that straddled a ~60-degree window
            # forever. Routing stays the agent's job — but after repeated failures on the
            # SAME class the refusal teaches the scan technique instead of repeating itself.
            cls_key = base(step[1]) if len(step) > 1 else "?"
            fails = getattr(self, "_gate_fails", {})
            # counters reset on ROOM CHANGE (audit fix #59): "attempt 3 for the tablelamp"
            # once counted attempts on the OTHER lamp two rooms earlier, and door counters
            # spanned different doors — a hint about a different place is misinformation.
            here_now = self.current_room_label()
            if getattr(self, "_gate_fail_room", None) != here_now:
                fails = {}
                self._door_fails = 0
            self._gate_fail_room = here_now
            fails[cls_key] = fails.get(cls_key, 0) + 1
            self._gate_fails = fails
            if fails[cls_key] >= 3:
                # room disambiguation (fix #48): 16 attempts to "see" a TV happened from the
                # WRONG ROOM. The map the agent holds shows object rooms; reading that single
                # fact into the refusal is feedback, not routing.
                here = self.current_room_label()
                live_rooms = self._room_labels_with(cls_key)
                # Name EVERY room that holds one, from the live scene. The class-level room_of()
                # answered "the mug is in the kitchen" to a person standing in the bedroom next
                # to the bedroom mug it was after (pilot 3, A20). If one is in this room, the
                # right hint is the scan hint, not a door.
                if live_rooms is None:
                    tgt_room = self.room_of(cls_key)
                    live_rooms = [tgt_room] if tgt_room else []
                if here and live_rooms and here not in live_rooms:
                    where = " and the ".join(live_rooms)
                    gate_refusal += (f" — this is attempt {fails[cls_key]}: "
                                     f"{'the' if len(live_rooms) == 1 else 'a'} {cls_key} is in the {where} "
                                     f"and you are in the {here}; get through a door toward "
                                     f"{'it' if len(live_rooms) == 1 else 'one of them'} first")
                else:
                    gate_refusal += (f" — this is attempt {fails[cls_key]} for the {cls_key}: "
                                     f"it is nearby but easy to sweep past. Check the map for "
                                     f"which way it lies from you and turn toward it — 'turn "
                                     f"right 3' is three 30-degree turns (90 degrees), 'turn left "
                                     f"6' ends facing the other way")
            entry.update(status="rejected", why=gate_refusal, t_done=self.tick)
            self.idle_ticks(1)
            return entry
        ok, why = self.sym.check(step)
        if not ok:
            entry.update(status="rejected", why=why, t_done=self.tick)
            self.idle_ticks(1)
            return entry
        if step[0] == "Idle":
            self.idle_ticks(max(1, int(step[1]) // FPT))
            self.sym.run(step)
            entry.update(status="done", t_done=self.tick)
            return entry
        # MID-LEVEL PATH. With VH_MIDLEVEL=1 the command is carried out egocentrically: the body
        # must SEE the target before it can approach or act on it, and reaching anything else
        # means turning until it comes into view. The high-level path below pathfinds across the
        # house in one call whether or not the target is visible, and takes the room from the
        # episode script -- i.e. the agent has oracle navigation and can never get lost, so
        # `navigation_inefficiency` could only ever be injected, never emerge.
        if step[0] == "MidLevel":
            # Agent movement primitive (`turn`, `look`, `go straight`, `go through the nearest
            # door`). Executed ONCE: the recording hook renders each script group midlevel
            # emits, animated, and ingests its frames on the spot. The previous shape — hook
            # executes with skip_animation, exec_step re-renders the collected script — moved
            # the body twice per command. `turn`/`look` legitimately emit zero frames
            # (ENGINE_NOTES A2 / camera-only), which is why success comes from the controller's
            # own result and never from frame count.
            ctrl = self._mlctrl()
            prefix = f"IV{self.tag}_{self.item_i:03d}"
            self.item_i += 1
            n0 = len(self.frames)
            pose0 = self._body_pose()
            self._ml_record = (prefix, ego_cam_name(WALK_PITCH))
            self._ml_subs[0] = 0
            try:
                r = ctrl.execute(step[1])
            finally:
                self._ml_record = None
            if not r.get("success", False):
                why_r = r.get("reason") or "that did not move me"
                # progressive refusal for DOOR passes too (fix #52): a run collected 12
                # identical "no door visible" refusals with no escalating help, while walk
                # gates had had #43's treatment for hours.
                if "no door or open passage in view" in why_r or "no door visible" in why_r:
                    n = getattr(self, "_door_fails", 0) + 1
                    self._door_fails = n
                    if n >= 3:
                        why_r += (f" — attempt {n}: doorways are the pink squares on the map;"
                                  f" read which way the nearest one lies from you and turn"
                                  f" toward it ('turn right 3' is three 30-degree turns), then go"
                                  f" through it")
                else:
                    self._door_fails = 0
                entry.update(status="rejected", t_done=self.tick, why=why_r)
                self.idle_ticks(1)
                return entry
            # SILENT NO-OP CHECK (fix #29). A [TurnRight] can return Success while neither the
            # body nor the camera actually rotates (seen live: t89 'turn right 3' -> done, map
            # arrow and view both unchanged). Success-without-effect poisons both agents' world
            # models — the human "knows" it turned, the assistant watches nothing happen, and
            # the two argue. Verify movement commands against the measured pose and refuse
            # honestly when the world did not change.
            noop_why = self._midlevel_noop(step[1], pose0)
            if noop_why:
                # CORNER TRAP ESCAPE (fix #33). A body jammed in a corner cannot turn AT ALL —
                # observed live: turn left/around/right all physical no-ops while go-straight
                # and door-pass were wall-blocked, so after #29 every exit was refused and the
                # run idled to death. There is no [WalkBackward] in this VH build; what DOES
                # free a jammed body is a pathfinding [Walk] to a nearby object. When a TURN
                # no-ops, the house steps the body away and says so — physics recovery is the
                # engine's job; where to go next stays the agent's.
                fixed = None
                if (str(step[1]).lower().startswith("turn")
                        and self.tick >= getattr(self, "_unstick_cooldown", 0)):
                    fixed = self._unstick(f"{prefix}U")
                    if not fixed:
                        # every candidate walk moved 0 m: pathfinding is dead from this pose.
                        # Re-probing 5 pathfinds on every turn attempt burned a minute per
                        # tick for 60 ticks — back off instead of hammering.
                        self._unstick_cooldown = self.tick + 20
                if fixed:
                    entry.update(status="rejected", t_done=self.tick,
                                 why=(f"my body was jammed in place and could not turn — I "
                                      f"stepped away toward the {fixed} to free myself. I can "
                                      f"move again now, but I did NOT make that turn; check "
                                      f"the map and reissue your move from here"))
                    while len(self.frames) % FPT:
                        self._append_frame(self.frames[-1], None)
                else:
                    entry.update(status="rejected", t_done=self.tick, why=noop_why)
                    self.idle_ticks(1)
                return entry
            print(f"[world] t{self.tick} exec {entry['cmd']} -> {len(self.frames) - n0} frames",
                  flush=True)
            if self.frames:
                if len(self.frames) == n0:
                    # zero-frame turn/look: capture the NEW view, or both agents stay blind.
                    # TURNS are ANIMATED (sweep the camera through the arc) so the first-person
                    # view does not flip instantly; look up/down stays a single fresh frame.
                    seq = (self._animate_turn(prefix, pose0, str(step[1]))
                           if "turn" in str(step[1]).lower() else [])
                    if seq:
                        for _pth, _po in seq:
                            self._append_frame(_pth, _po)
                        while len(self.frames) % FPT:
                            self._append_frame(self.frames[-1], None)
                    else:
                        snap, pose = self._ego_snapshot(prefix)
                        if snap is not None:
                            for _ in range(FPT):
                                self._append_frame(snap, pose)
                        else:
                            self.idle_ticks(1)
                else:
                    while len(self.frames) % FPT:
                        self._append_frame(self.frames[-1], None)
            self.sym.run(step)
            # WEDGE CONTAINMENT AT FIRST CONTACT (fix #39). Door passes are not the only way
            # into the incurable corner trap: repeated go-straight into a corner (plus turns
            # grinding the capsule into geometry) wedge the body PROGRESSIVELY — turns still
            # worked two ticks after the first blocked move and were dead four ticks later.
            # At first contact pathfinding is still alive, so probe rotation right when a
            # forward move reports blocked and unstick immediately if it is already gone.
            why_txt = str(r.get("reason") or "")
            if (str(step[1]).lower().startswith("go straight")
                    and ("blocked" in why_txt or "no space" in why_txt)
                    and self.tick >= getattr(self, "_unstick_cooldown", 0)):
                pa = self._body_pose()
                ctrl._do("[TurnRight]", False)
                pb = self._body_pose()
                if pa and pb and abs((pb[2] - pa[2] + 180.0) % 360.0 - 180.0) < 10.0:
                    fixed = self._unstick(f"{prefix}W")
                    if fixed:
                        why_txt += (f" — and I found myself briefly wedged; I stepped away "
                                    f"toward the {fixed} to stay free")
                    else:
                        self._unstick_cooldown = self.tick + 20
                        why_txt += " — and I am wedged tight here; turning is blocked too"
                else:
                    ctrl._do("[TurnLeft]", False)      # undo the probe rotation
            # honest status for zero-effect forwards (audit fix #61): "no space to move
            # forward" means the body moved ZERO metres; reporting that as done taught
            # humans they had advanced when they had not. Partial moves ("walked 2m, then
            # blocked") stay done — ground was gained.
            if "no space to move forward" in (why_txt or ""):
                entry.update(status="rejected", t_done=self.tick, why=why_txt)
                self.idle_ticks(1)
                return entry
            entry.update(status="done", t_done=self.tick, why=why_txt or r.get("reason"))
            return entry
        sl = self._line(step)
        if sl is None:
            entry.update(status="rejected", why="I can't find that here", t_done=self.tick)
            self.idle_ticks(1)
            return entry
        sl = [sl]
        # CAMERA (fix #28). aim_camera frames the TARGET OBJECT — a cinematic. Under the
        # mid-level action space the frame stream is served to the creative human as ITS OWN
        # EYES (ego_frames) and the gates test visibility from the body's true pose, so a
        # cinematic in that stream makes the two disagree by construction: the human argued
        # "I'm looking at the kitchen tables" from an item-cam angle while standing in the
        # bedroom, and the assistant told a human facing a lamp that it wasn't. Egocentric
        # setting => every frame is the body's view, same as the VLM-as-controller statemap
        # runs. Manipulations pitch the ego ladder at the target (the documented manipulation
        # glance) so the act is in frame without leaving the body's pose; a straight-ahead
        # ego render at NEAR range showed only wall and the assistant could not see the very
        # act it must judge. The high-level (non-midlevel) pipeline keeps its cinematics.
        if MIDLEVEL:
            p = WALK_PITCH
            if step[0] != "Walk":
                # StandUp is the one verb with NO object: its step is the 1-tuple
                # ("StandUp",). It only ever reached here inside a `posture` node, where
                # segment_steps absorbs it after the Sit, so the bare form was never executed
                # and step[1] raised IndexError. A standalone StandUp is legal grammar
                # (`stand`), so it must be handled rather than assumed away.
                tgt = (None if len(step) < 2 else
                       step[3] if step[0] in ("Put", "PutIn") and len(step) > 3 else step[1])
                # An objectless verb (StandUp) has nothing to aim the camera at, so it keeps the
                # walking pitch. Passing None through pitch_to() reaches arithmetic on a missing
                # instance and raises TypeError, which the except below would NOT catch on some
                # paths because the failure happens before the call.
                self._aim_yaw = 0.0
                if tgt is None:
                    p = WALK_PITCH
                else:
                    try:
                        p = pitch_to(self.c, inst(tgt))
                        self._aim_yaw = yaw_to(self.c, inst(tgt))   # bounded glance, see yaw_to
                    except Exception:
                        p = WALK_PITCH
            item_cam = ego_cam_name(p)
        else:
            item_cam = self._aim(step)
        prefix = f"IV{self.tag}_{self.item_i:03d}"
        self.item_i += 1
        # LEGIBILITY DWELL. A Grab is 7-8 frames -- under a second -- of a small object on a
        # cluttered surface, and a reviewer (human or model) reads it as the thing "appearing"
        # in the hand. Before every manipulation, hold the aimed view on the target for one
        # second so the object is SEEN before it is acted on. This is the camera already pitched
        # at the target (item_cam); it moves nothing and changes no state.
        if DWELL and step[0] in ("Grab", "Put", "PutIn", "Open", "Close", "SwitchOn", "SwitchOff"):
            try:
                fp = self._fp_index()
                self._cam_for(item_cam)
                ok_, imgs_ = self.c.camera_image([fp], mode="normal",
                                                 image_width=640, image_height=480)
                if ok_ and imgs_:
                    d_ = f"{paths.UNITY_DIR}/Output/{prefix}D/0"; os.makedirs(d_, exist_ok=True)
                    path_ = f"{d_}/dwell_0000_normal.png"
                    Image.fromarray(np.asarray(imgs_[0])[..., :3].astype(np.uint8)).save(path_)
                    for _ in range(DWELL):
                        self._append_frame(path_, None)
            except Exception as e_:
                print(f"[world] dwell skipped: {e_}", flush=True)
        if step[0] in ("TurnTo", "LookAt"):
            # LOOK. Unity refuses [TurnTo]/[LookAt] outright (every reference film: "my body
            # couldn't do that here"), so an inspection is rendered here as one second of the
            # aimed view: the eyes turn to the target and hold. Pilot 2026-09-05: the refusal
            # left the tasked sequence incomplete, the takeover never released, and every human
            # plan was silently dropped -- all three search episodes froze from ~t30 to the end.
            try:
                fp = self._fp_index()
                self._cam_for(item_cam)
                ok_, imgs_ = self.c.camera_image([fp], mode="normal",
                                                 image_width=640, image_height=480)
                if ok_ and imgs_:
                    d_ = f"{paths.UNITY_DIR}/Output/{prefix}L/0"; os.makedirs(d_, exist_ok=True)
                    path_ = f"{d_}/look_0000_normal.png"
                    Image.fromarray(np.asarray(imgs_[0])[..., :3].astype(np.uint8)).save(path_)
                    for _ in range(FPT):
                        self._append_frame(path_, None)
                    self._aim_yaw = 0.0
                    entry.update(status="done", t_done=self.tick, why="looked at it")
                    self.sym.run(step)
                    return entry
            except Exception as e_:
                print(f"[world] look fell through to Unity: {e_}", flush=True)
            self._aim_yaw = 0.0
        fs, uok = [], False
        for attempt in range(2):
            self._root_rm(f"/unity_vol/Output/{prefix}")
            try:
                uok, msg = self.c.render_script(sl, recording=True, find_solution=False,
                                                frame_rate=5, output_folder="Output/",
                                                file_name_prefix=prefix, image_width=640,
                                                image_height=480,
                                                camera_mode=[self._cam_for(item_cam)],
                                                save_pose_data=True)
            except Exception as e:
                uok, msg = False, str(e)
            self._aim_yaw = 0.0
            fs = sorted(glob.glob(f"{paths.UNITY_DIR}/Output/{prefix}/0/Action_*_normal.png"),
                        key=lambda p: int(p.split("Action_")[1].split("_")[0]))
            if uok and fs:
                break
        if not (uok and fs) and step[0] == "Walk":
            # A RECORDED WALK THAT GOES NOWHERE (A33, f89). A recorded [Walk] to either apartment-1
            # bedroom table lamp (and two apartment-3 kitchen books) returns Success with ZERO frames
            # and the body does not move, while the same walk unrecorded moves 2.6 m -- so every
            # "walk to the tablelamp" was refused and people blamed their own wording. Walking up to
            # the furniture the object rests on puts the body at the object; use it when the direct
            # walk rendered nothing.
            sur = self._walk_surrogate(step)
            if sur is not None:
                self._root_rm(f"/unity_vol/Output/{prefix}")
                try:
                    uok, msg = self.c.render_script([sur[0]], recording=True, find_solution=False,
                                                    frame_rate=5, output_folder="Output/",
                                                    file_name_prefix=prefix, image_width=640,
                                                    image_height=480,
                                                    camera_mode=[self._cam_for(item_cam)],
                                                    save_pose_data=True)
                except Exception as e:
                    uok, msg = False, str(e)
                self._aim_yaw = 0.0
                fs = sorted(glob.glob(f"{paths.UNITY_DIR}/Output/{prefix}/0/Action_*_normal.png"),
                            key=lambda p: int(p.split("Action_")[1].split("_")[0]))
        if not (uok and fs):
            # ALREADY THERE. VirtualHome refuses a [Walk] to an object the character is already
            # standing at, and the harness reported that as a failure the agent then had to
            # react to -- in one run the human retried the same walk three times after having
            # just opened that very microwave. "Walk to X" asks for a goal state, and that goal
            # already holds, so it is satisfied, not refused. Only reclassify when the body is
            # genuinely next to the target; a real pathing failure must still surface.
            if step[0] == "Walk":
                tid = inst(step[1])
                if tid is None:                      # an unresolved token: use the instance the step bound
                    tid = (entry.get("instances") or [None])[0]
                _a, pos, cp = self._anchor_map()
                if tid is not None and cp is not None and tid not in pos:
                    # not an anchor (a lamp on a desk, a mug on a counter): read its position
                    # from the scene graph instead, so "walk to the lamp" from arm's length
                    # is satisfied rather than refused (every tidybedroom film, 2026-09-09)
                    try:
                        _s, _g = self.c.environment_graph()
                        _n = next((n for n in _g["nodes"] if n["id"] == tid), None)
                        if _n is not None:
                            pos = dict(pos); pos[tid] = _n["obj_transform"]["position"]
                    except Exception:
                        pass
                if tid is not None and cp is not None and tid in pos:
                    d2 = sum((pos[tid][k] - cp[k]) ** 2 for k in (0, 2))
                    if d2 <= 2.25:                       # within 1.5 m
                        entry.update(status="done", t_done=self.tick,
                                     why="already standing there")
                        self.sym.run(step)
                        self.idle_ticks(1)
                        return entry
            # verb-specific translation of the engine's opaque refusal (audit fix #62): a
            # placement the world refuses almost always means no free spot on that surface
            # within reach — say so, so the agent moves or picks another surface instead of
            # retrying the same spot (three identical refusals in one run).
            if step[0] in ("Put", "PutIn") and len(step) > 3:
                why_w = (f"I couldn't place the {base(step[1])} there — no free spot on the "
                         f"{base(step[3])} within my reach; move closer or use another surface")
            elif step[0] == "Walk":
                why_w = (f"I couldn't find a way to walk up to the {base(step[1])} from here — "
                         f"try walking to the furniture next to it")
            else:
                why_w = "my body couldn't do that here"
            entry.update(status="rejected", why=why_w, t_done=self.tick)
            self.idle_ticks(1)
            return entry
        # black-ego gate on the very first rendered frame of the episode
        if not self.frames:
            if float(np.asarray(Image.open(fs[0]).convert("L")).mean()) < 5.0:
                print("DEGRADED: ego camera renders BLACK", flush=True)
                sys.exit(3)
            if step[0] == "Walk" and len(fs) < 6:
                print("DEGRADED: teleport-walk on first item", flush=True)
                sys.exit(3)
        print(f"[world] t{self.tick} exec {entry['cmd']} -> {len(fs)} frames", flush=True)
        self._ingest_render(prefix, fs)
        # pad to a whole tick so ticks stay = frames/8
        while len(self.frames) % FPT:
            self._append_frame(fs[-1], None)
        # MANIPULATION POSTCONDITION (invariant B, consolidation rev). Unity can play an
        # animation without the state flipping; sym.run trusting the render would silently
        # fork the symbolic world from the graph and poison every downstream judgment
        # (checkpoints, final_ok, engine snaps) — the #34 disaster shape, self-inflicted.
        # Verify the expected graph delta; refuse honestly when it did not happen.
        ok_post, why_post = self._verify_postcondition(step)
        if not ok_post:
            entry.update(status="rejected", t_done=self.tick, why=why_post)
            self.idle_ticks(1)
            return entry
        self.sym.run(step)
        if len(step) > 1:      # a success resets the progressive-refusal counter (fix #43)
            getattr(self, "_gate_fails", {}).pop(base(step[1]), None)
        entry.update(status="done", t_done=self.tick)
        return entry

    # postcondition table: verb -> (checker(graph_helpers, iid, dest_iid), description).
    # Walk is covered by arrival-distance reclassification; Drink has no reliable graph
    # delta in VH (the held item is consumed visually only) — both exempt by design.
    # Read/Watch change no graph state (the person just engages with the thing), so like Drink
    # they have no postcondition to verify.
    _POST_EXEMPT = ("Walk", "Drink", "Sit", "StandUp", "MidLevel", "Idle", "Read", "Watch")

    def _verify_postcondition(self, step):
        """(ok, why). Checks the graph reflects the verb's effect for resolved instances.
        Class-level steps (no '#id' by exec time) skip with a log — nothing to verify against."""
        verb = step[0]
        if verb in self._POST_EXEMPT:
            return True, None
        oid = inst(step[1]) if len(step) > 1 else None
        dest = inst(step[3]) if len(step) > 3 else None
        if oid is None:
            print(f"[world] t{self.tick} postcondition skipped (class-level): {step}", flush=True)
            return True, None
        try:
            ok, g = self.c.environment_graph()
            node = next((n for n in g["nodes"] if n["id"] == oid), None)
            if node is None:
                return True, None                      # object left the graph; nothing to say
            states = set(node.get("states") or ())
            ch = next(n["id"] for n in g["nodes"] if n["class_name"] == "character")
            def edge(frm, rel, to):
                return any(e["from_id"] == frm and e["to_id"] == to
                           and e["relation_type"].startswith(rel) for e in g["edges"])
            o = base(step[1])
            if verb == "Open" and "OPEN" not in states:
                return False, f"I pulled at the {o} but it did not actually open"
            if verb == "Close" and "OPEN" in states:
                return False, f"I pushed the {o} but it did not actually close"
            if verb == "SwitchOn" and "ON" not in states:
                return False, f"I pressed the switch but the {o} did not come on"
            if verb == "SwitchOff" and "ON" in states:
                return False, f"I pressed the switch but the {o} is still on"
            if verb == "Grab" and not edge(ch, "HOLDS", oid):
                return False, f"I reached for the {o} but it is not in my hand"
            if verb in ("Put", "PutIn") and dest is not None:
                rel = "INSIDE" if verb == "PutIn" else "ON"
                if not edge(oid, rel, dest):
                    return False, (f"I let go of the {o} but it did not end up "
                                   f"{'inside' if verb == 'PutIn' else 'on'} the {base(step[3])}")
            return True, None
        except Exception as e:
            print(f"[world] postcondition check failed open-armed: {e}", flush=True)
            return True, None                          # never turn a checker fault into a refusal

    # ---------------- observation channels
    def comp(self, k):
        """Composite ego|map frame k as np array (same layout as the detection videos)."""
        k = max(0, min(k, len(self.frames) - 1))
        H = 480
        eg = np.array(Image.open(self.frames[k]).convert("RGB").resize((640, H)))
        rooms, objs, doors = self.fp
        mp = np.array(Image.fromarray(
            map_frame(rooms, objs, doors, np.array(self.pos[:k + 1], float),
                      np.array(self.yaw[:k + 1], float), k)).resize((H, H)))
        eg = _label(eg, "first-person", (199, 146, 234))
        mp = _label(mp, "top-down map", (255, 210, 74))
        return np.concatenate([eg, mp], axis=1)

    @staticmethod
    def _commanded_turn_deg(text):
        """The rotation a turn command ASKED for, right-positive degrees, or None. "turn left 6"
        ends at +-180 whichever way the body went, and "turn right 8" (240) wraps to -120, so the
        measured end pose alone cannot say which way the view should sweep (A29)."""
        t = str(text or "").strip().lower()
        m = re.match(r"^turn\s+(left|right)(?:\s+(\d+))?$", t)
        if m:
            k = int(m.group(2) or 1)
            return 30.0 * k * (1 if m.group(1) == "right" else -1)
        return None

    def _right_yaw_sign(self):
        """+1 if [TurnRight] increases yaw on this build, else -1 -- measured once, not assumed."""
        if getattr(self, "_rys", None) is None:
            ctrl = self._mlctrl()
            a = self._body_pose()
            ctrl._do("[TurnRight]"); b = self._body_pose(); ctrl._do("[TurnLeft]")
            self._rys = 1.0 if ((b[2] - a[2] + 180.0) % 360.0 - 180.0) > 0 else -1.0
        return self._rys

    def _animate_turn(self, prefix, pose0, cmd_text=None):
        """Smooth a zero-frame turn for video (audit: user saw a 'sudden jump'). VH renders NO
        frames for [TurnLeft]/[TurnRight], so a single end-orientation snapshot flips the
        first-person view instantly. Here the CAMERA is swept through the rotation arc (the body
        is already at its final orientation; only the capture camera is rotated, then restored),
        snapshotting intermediate angles so the turn animates. Returns [(path,pose),...] or []."""
        try:
            ctrl = self._mlctrl()
            p1 = self._body_pose()
            if pose0 is None or p1 is None:
                return []
            delta = (p1[2] - pose0[2] + 180.0) % 360.0 - 180.0   # signed turn amount (deg)
            if abs(delta) < 8.0:
                return []
            # sweep the arc that was COMMANDED whenever it explains the measured end pose, so an
            # 'turn left 6' sweeps the way the body turned and a 240-degree turn is not shown as 120
            want = self._commanded_turn_deg(cmd_text)
            if want:
                cand = want * self._right_yaw_sign()
                if abs((cand - delta + 180.0) % 360.0 - 180.0) < 12.0:
                    delta = cand
            n = max(3, min(14, int(round(abs(delta) / 24.0))))
            d = f"{paths.UNITY_DIR}/Output/{prefix}/0"; os.makedirs(d, exist_ok=True)
            out = []
            for k in range(1, n + 1):
                yaw_off = delta * (k / n - 1.0)   # -delta (pre-turn view) -> 0 (final)
                self.c.update_camera(ctrl._fp, position=ctrl._FP_POS,
                                     rotation=[WALK_PITCH, yaw_off, 0], field_view=ctrl._FP_FOV)
                ok, imgs = self.c.camera_image([ctrl._fp], mode="normal",
                                               image_width=640, image_height=480)
                if not ok or not imgs:
                    continue
                img = np.asarray(imgs[0])[..., :3].astype(np.uint8)
                path = f"{d}/turn_{k:03d}_normal.png"
                Image.fromarray(img).save(path)
                yaw_i = pose0[2] + delta * (k / n)   # map arrow rotates through the arc too
                out.append((path, ([p1[0], p1[1]], yaw_i)))
            self.c.update_camera(ctrl._fp, position=ctrl._FP_POS,
                                 rotation=[WALK_PITCH, 0, 0], field_view=ctrl._FP_FOV)
            return out
        except Exception as e:
            print(f"[world] animate_turn failed: {e}", flush=True)
            return []

    def _ego_snapshot(self, prefix):
        """One FRESH ego frame from the live camera, as a file _append_frame can take.

        Zero-frame actions ([TurnLeft]/[TurnRight], look up/down — ENGINE_NOTES A2) change the
        camera but render nothing, and idle_ticks repeats the LAST frame. So after a turn BOTH
        agents kept seeing the pre-turn view: the human said "still can't see it, I'll keep
        turning" about a microwave its live camera may well have been facing, and the assistant
        said "you are not turning — you are completely still", which was a faithful report of a
        stale picture. Everyone was blind to every turn. This snapshot is the fix: the moment a
        zero-frame action succeeds, the new orientation becomes what everyone sees.
        Returns (path, pose) or (None, None).
        """
        try:
            ctrl = self._mlctrl()
            ok, imgs = self.c.camera_image([ctrl._fp], mode="normal",
                                           image_width=640, image_height=480)
            if not ok or not imgs:
                return None, None
            img = np.asarray(imgs[0])[..., :3].astype(np.uint8)
            d = f"{paths.UNITY_DIR}/Output/{prefix}/0"
            os.makedirs(d, exist_ok=True)
            path = f"{d}/snap_0000_normal.png"
            Image.fromarray(img).save(path)
            _P, V, cam = ctrl._cam_matrices()
            R = np.linalg.inv(V)[:3, :3]
            fwd = R @ np.array([0.0, 0.0, 1.0])
            yaw = float(np.degrees(np.arctan2(fwd[0], fwd[2])))
            return path, ([float(cam[0]), float(cam[2])], yaw)
        except Exception as e:
            print(f"[world] ego snapshot failed: {e}", flush=True)
            return None, None

    def _body_pose(self):
        """(x, z, yaw_deg) of the body from the live graph, or None on any failure."""
        try:
            ctrl = self._mlctrl()
            p, yaw = ctrl._char(ctrl._graph())
            return float(p[0]), float(p[2]), float(yaw)
        except Exception:
            return None

    def _midlevel_noop(self, text, pose0):
        """Fix #29: refusal text if a movement command reported Success but measurably did
        nothing — else None. Only movement commands are checked ('look up/down' changes the
        camera, 'stand still' changes nothing, both legitimately leave the pose alone). A turn
        must change yaw; a door pass must move the body. 'go straight' already reports blocked
        honestly, so it is left to its own reason."""
        pose1 = self._body_pose()
        if pose0 is None or pose1 is None:
            return None
        t = str(text).lower()
        dyaw = abs((pose1[2] - pose0[2] + 180.0) % 360.0 - 180.0)
        dpos = ((pose1[0] - pose0[0]) ** 2 + (pose1[1] - pose0[1]) ** 2) ** 0.5
        if t.startswith("turn") and dyaw < 10.0 and dpos < 0.2:
            return ("my body did not actually turn — something is blocking me here; "
                    "try 'go straight' toward open space, or 'walk to' something in view")
        if "door" in t and dpos < 0.5:
            return ("I did not actually move through the door — I seem to be stuck; "
                    "turn to face the doorway squarely, or go straight in an open "
                    "direction first")
        if "door" in t:
            # a door pass that walked somewhere but left the body in the SAME room is not a pass
            # (A22: the stepping fallback reported "passed through" after a metre inside the room)
            def _room_at(x, z):
                for r in (getattr(self, "fp", None) or [[]])[0]:
                    if abs(x - r["c"][0]) <= r["s"][0] / 2 and abs(z - r["c"][2]) <= r["s"][2] / 2:
                        return r.get("cls") or r.get("lab")
                return None
            r0, r1 = _room_at(pose0[0], pose0[1]), _room_at(pose1[0], pose1[1])
            if r0 is not None and r0 == r1:
                return (f"I walked toward a door but I am still in the {r0} — I did not get "
                        f"through; turn until a doorway is straight ahead and try again")
        return None

    def _unstick(self, prefix):
        """Fix #33: free a body jammed in scene geometry by pathfind-walking to the nearest
        sizeable object in the current room. Returns that object's class name, or None if no
        candidate exists or the body did not actually move. Renders through the recording hook
        so both agents SEE the escape happen."""
        try:
            ctrl = self._mlctrl()
            g = ctrl._graph()
            p, _ = ctrl._char(g)
            rooms = {n["id"]: n for n in g["nodes"] if n.get("category") == "Rooms"}
            room = None
            for rid, rn in rooms.items():
                c, s = rn["bounding_box"]["center"], rn["bounding_box"]["size"]
                if abs(p[0] - c[0]) <= s[0] / 2 and abs(p[2] - c[2]) <= s[2] / 2:
                    room = rid
                    break
            inroom = {e["from_id"] for e in g["edges"]
                      if e["relation_type"] == "INSIDE" and e["to_id"] == room}
            cands = []
            for n in g["nodes"]:
                if (n["id"] not in inroom
                        or n.get("category") not in ("Furniture", "Appliances", "Electronics")
                        or max(n["bounding_box"]["size"]) < 0.6):
                    continue
                q = n["obj_transform"]["position"]
                d = ((q[0] - p[0]) ** 2 + (q[2] - p[2]) ** 2) ** 0.5
                if d < 0.8:      # the thing we are jammed against cannot pull us off itself
                    continue
                cands.append((d, n["id"], n["class_name"]))
            # DOOR FALLBACK: seen live — every nearby furniture walk moved the body <0.3 m and
            # the trap held. Doors are engine-pathfinding targets too and always have walkable
            # clearance in front of them, so they go in as the second wave.
            doors = []
            for n in g["nodes"]:
                if n["class_name"] != "door":
                    continue
                q = n["obj_transform"]["position"]
                d = ((q[0] - p[0]) ** 2 + (q[2] - p[2]) ** 2) ** 0.5
                if d >= 0.8:
                    doors.append((d, n["id"], "door"))
            # nearest first; a walk that barely moves the body did not free it — try the next
            for d, oid, cls in sorted(cands)[:3] + sorted(doors)[:2]:
                self._ml_record = (prefix, ego_cam_name(WALK_PITCH))
                try:
                    ctrl._do(f"[Walk] <{cls}> ({oid})", True)
                finally:
                    self._ml_record = None
                p2, _ = ctrl._char(ctrl._graph())
                moved = ((p2[0] - p[0]) ** 2 + (p2[2] - p[2]) ** 2) ** 0.5
                print(f"[world] t{self.tick} unstick try {cls}#{oid}: moved {moved:.2f} m",
                      flush=True)
                if moved >= 0.3:
                    print(f"[world] t{self.tick} UNSTUCK via walk to {cls}#{oid}", flush=True)
                    return cls
            return None
        except Exception as e:
            print(f"[world] unstick failed: {e}", flush=True)
            return None

    def current_room(self):
        """Which room the body is in now. Uses the TRUE body pose from the live graph
        (trajectory-audit fix #58): the old source was self.pos[-1] — the SMOOTHED display
        trail (75/25 blended for the map) — which jitters across a doorway threshold, so
        turn-only acts flipped the room tag and the #48 refusal hints sometimes named the
        wrong room. Falls back to the trail only when the graph is unreadable."""
        p = self._body_pose()
        if p is not None:
            x, z = p[0], p[1]
        elif len(self.pos):
            x, z = self.pos[-1]
        else:
            return None
        for r in self.fp[0]:
            if (abs(x - r["c"][0]) <= r["s"][0] / 2 and abs(z - r["c"][2]) <= r["s"][2] / 2):
                return r["cls"]
        return None

    def room_labels(self, g=None):
        """{room_id: label} exactly as the maps label rooms (topdown_map._rooms_labeled): the class
        name, and #2, #3 ... for a second or third room of the same class, in scene-graph order.
        A scene with two bedrooms showed "bedroom#2" on the map while every refusal hint and the
        action log said "bedroom" for both (pilot 6, groceries), so the person could not tell
        which one a hint meant."""
        try:
            if g is None:
                _s, g = self.c.environment_graph()
            seen, out = {}, {}
            for n in g["nodes"]:
                if n.get("category") != "Rooms":
                    continue
                c = n["class_name"]; seen[c] = seen.get(c, 0) + 1
                out[n["id"]] = c + (f"#{seen[c]}" if seen[c] > 1 else "")
            return out
        except Exception:
            return {}

    def current_room_label(self):
        """The body's room under the map's own label ("bedroom#2"), for anything an agent reads.
        Internal grounding keeps using current_room() (the bare class), which symbolic plans and
        room_of() compare against."""
        try:
            _s, g = self.c.environment_graph()
            p = self._body_pose()
            if p is None:
                return self.current_room()
            labels = self.room_labels(g)
            for n in g["nodes"]:          # first containing room, in graph order -- as the map does
                if n.get("category") != "Rooms":
                    continue
                c, sz = n["bounding_box"]["center"], n["bounding_box"]["size"]
                if abs(p[0] - c[0]) <= sz[0] / 2 and abs(p[1] - c[2]) <= sz[2] / 2:
                    return labels.get(n["id"], n["class_name"])
        except Exception:
            pass
        return self.current_room()

    def _room_labels_with(self, cls):
        """Labelled rooms that hold an instance of `cls` in the live scene (None if unreadable)."""
        try:
            _s, g = self.c.environment_graph()
            labels = self.room_labels(g)
            ids = {n["id"] for n in g["nodes"] if n["class_name"] == cls}
            return sorted({labels[e["to_id"]] for e in g["edges"]
                           if e["relation_type"] == "INSIDE" and e["from_id"] in ids and e["to_id"] in labels})
        except Exception:
            return None

    def person_locus(self):
        """Compact engine-truth of WHERE the person's body is, for the SUPERVISOR (mid-level).

        The supervisor watches first-person video and cannot reliably localize the body from it
        — trajectory audit (fix #68) saw it assert the WRONG ROOM a dozen times ("you're in the
        bathroom" at a person in the bedroom, "that's a mirror") and micro-steer turns off that
        hallucination for ~60 ticks. The benchmark measures ERROR DETECTION, not the
        supervisor's localization, so grounding room + facing + what is currently in view
        removes scaffolding noise without touching the detection challenge. The person still
        navigates itself; this only stops the supervisor misdirecting it. Returns "" outside
        mid-level or on any failure."""
        if not MIDLEVEL:
            return ""
        try:
            room = self.current_room_label() or "an unlabelled area"
            pose = self._body_pose()
            DIRS = ["north", "north-east", "east", "south-east",
                    "south", "south-west", "west", "north-west"]
            face = DIRS[int(((pose[2] % 360) + 22.5) // 45) % 8] if pose else "?"
            obs = self._mlctrl().observation()
            vis = sorted({v.split("#")[0] for v in obs.get("visible", [])})
            vis_s = ", ".join(vis[:10]) if vis else "nothing (facing a wall or empty space)"
            return f"in the {room}, facing {face}; currently IN VIEW: {vis_s}"
        except Exception as e:
            print(f"[world] person_locus failed: {e}", flush=True)
            return ""

    def topdown_obs(self):
        """(map_png, scene_text) for the CURRENT room — the `statemap` observation.

        This is what makes A (navigation) the agent's decision rather than the engine's: the
        previous VLM-as-controller setting fed exactly this pair (topdown_map.render_room +
        room_text) and told the model to cross-reference them and route itself through labelled
        doors. Asking an agent to navigate a visibility-gated house WITHOUT it is asking it to
        navigate blind, which is why "even the most creative agent could not run it".

        The floorplan is snapshotted from the LIVE graph on every call, exactly like
        gemini_stream_vh.snapshot_floorplan: topdown_map needs fields self.fp never carried
        (room `id` for door labelling, object `cls`/`st`), and object states drawn from a
        setup-time snapshot would be stale the moment a door opened. Returns (None, "") if
        there is no pose yet.
        """
        from vhhome import topdown_map as topdown_map
        if not len(self.pos):
            return None, ""
        try:
            s_, g = self.c.environment_graph()
            SKIP = ("Rooms", "Walls", "Floor", "Ceiling", "Ceilings", "Doors", "Windows")
            rooms = [dict(id=n["id"], cls=n["class_name"], c=n["bounding_box"]["center"],
                          s=n["bounding_box"]["size"])
                     for n in g["nodes"] if n.get("category") == "Rooms"]
            objs = [dict(cls=n["class_name"], cat=n.get("category"),
                         c=[n["bounding_box"]["center"][0], n["bounding_box"]["center"][2]],
                         s=[n["bounding_box"]["size"][0], n["bounding_box"]["size"][2]],
                         st=[x for x in (n.get("states") or [])
                             if x in ("OPEN", "CLOSED", "ON", "OFF")])
                    for n in g["nodes"]
                    if n.get("category") not in SKIP and n["class_name"] != "character"]
            doors = [n["bounding_box"]["center"] for n in g["nodes"]
                     if n["class_name"] == "door"]
            doors += [[ps["center"][0], 1.0, ps["center"][1]] for ps in scene_passages(self.ep["scene"])]
            fpd = dict(rooms=rooms, objs=objs, doors=doors)
            pos = [float(self.pos[-1][0]), float(self.pos[-1][1])]
            yaw = float(self.yaw[-1])
            return topdown_map.render_room(fpd, pos, yaw), topdown_map.room_text(fpd, pos, yaw)
        except Exception as e:
            print(f"[world] statemap failed: {e}", flush=True)
            return None, ""

    def house_obs(self):
        """(house_map_png, house_text) — the WHOLE-HOUSE overview: room layout, doors, the
        person's position + trail, and any device left ON in ANY room. Given to BOTH agents
        alongside the current-room statemap so cross-house routing works and a forgotten far-room
        device is legible (user decision 2026-08-26). Returns (None, "") if there is no pose."""
        from vhhome import topdown_map as topdown_map
        if not len(self.pos):
            return None, ""
        try:
            _s, g = self.c.environment_graph()
            SKIP = ("Rooms", "Walls", "Floor", "Ceiling", "Ceilings", "Doors", "Windows")
            rooms = [dict(id=n["id"], cls=n["class_name"], c=n["bounding_box"]["center"],
                          s=n["bounding_box"]["size"])
                     for n in g["nodes"] if n.get("category") == "Rooms"]
            objs = [dict(cls=n["class_name"], cat=n.get("category"),
                         c=[n["bounding_box"]["center"][0], n["bounding_box"]["center"][2]],
                         s=[n["bounding_box"]["size"][0], n["bounding_box"]["size"][2]],
                         st=[x for x in (n.get("states") or [])
                             if x in ("OPEN", "CLOSED", "ON", "OFF")])
                    for n in g["nodes"]
                    if n.get("category") not in SKIP and n["class_name"] != "character"]
            doors = [n["bounding_box"]["center"] for n in g["nodes"]
                     if n["class_name"] == "door"]
            doors += [[ps["center"][0], 1.0, ps["center"][1]] for ps in scene_passages(self.ep["scene"])]
            fpd = dict(rooms=rooms, objs=objs, doors=doors)
            trail = [[float(p[0]), float(p[1])] for p in self.pos]
            yaws = [float(y) for y in self.yaw]
            return (topdown_map.render_house(fpd, trail, yaws),
                    topdown_map.house_text(fpd, trail[-1], yaws[-1]))
        except Exception as e:
            print(f"[world] house map failed: {e}", flush=True)
            return None, ""

    def sup_frames(self, now_tick, window=8):
        """Last `window` tick-frames (1/s) up to now_tick, as JPEG bytes oldest->newest."""
        out = []
        for t in range(max(0, now_tick - window + 1), now_tick + 1):
            k = min(t * FPT + FPT - 1, len(self.frames) - 1)
            if k < 0: continue
            im = Image.fromarray(self.comp(k)).resize((1000, 430))
            b = io.BytesIO(); im.save(b, "JPEG", quality=82)
            out.append(b.getvalue())
        return out

    def ego_frames(self, now_tick, window=8):
        """The creative human's own eyes: ego-only, last `window` ticks."""
        out = []
        for t in range(max(0, now_tick - window + 1), now_tick + 1):
            k = min(t * FPT + FPT - 1, len(self.frames) - 1)
            if k < 0: continue
            im = Image.open(self.frames[k]).convert("RGB").resize((512, 384))
            b = io.BytesIO(); im.save(b, "JPEG", quality=82)
            out.append(b.getvalue())
        return out

    # ---------------- final video
    def write_video(self, path):
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=FPT, macro_block_size=None, output_params=FFMPEG_THREADS) as w:
            for k in range(len(self.frames)):
                w.append_data(self.comp(k))
        self._root_rm(f"/unity_vol/Output/IV{self.tag}_*")
        self._root_rm(f"/unity_vol/Output/WU{self.tag}")


def pretty_cmd(step):
    a = step[0]
    if a == "Idle":
        return "stand still"
    # strip BOTH token forms: 'lamp@1' (plan) and 'fridge#157' (resolved instance, e.g. a repetition block
    # replayed from the log) -- an instance id must never reach the person's [SYSTEM TASK] text (f113)
    o = re.split(r"[@#]", str(step[1]))[0] if len(step) > 1 else None
    o2 = re.split(r"[@#]", str(step[3]))[0] if len(step) > 3 else None
    return {"Walk": f"walk to the {o}", "Grab": f"grab the {o}", "Open": f"open the {o}",
            "Close": f"close the {o}", "Put": f"put the {o} on the {o2}",
            "PutIn": f"put the {o} in the {o2}", "SwitchOn": f"switch on the {o}",
            "SwitchOff": f"switch off the {o}", "Sit": f"sit on the {o}",
            "StandUp": "stand up", "Drink": f"drink the {o}",
            "Idle": "stand still", "MidLevel": str(o), "TurnTo": f"turn to the {o}",
            "LookAt": f"look at the {o}", "Read": f"read the {o}",
            "Watch": f"watch the {o}"}.get(a, a.lower())
