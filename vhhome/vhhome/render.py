"""Shared VirtualHome rendering corrections — import this from EVERY renderer.

Two defects made rendered footage disagree with what the script actually did. Both were found on the
detection videos (2026-07-31) but they affect every episode we render, so the fixes live here rather
than being copy-pasted into each renderer.

1. REDUNDANT SETUP INVERTS DEVICE APPEARANCE.
   VirtualHome sets an object's *graph* state absolutely but *toggles* its visual. Issuing a command
   for a state the object already holds therefore flips its appearance permanently: the graph keeps
   reading ON while the pixels show OFF. Scene 2 loads `tablelamp` already ON, and both
   `routine_hard` and `tidy_hard` opened with `SwitchOn(tablelamp)` — so every later `switch_off` in
   those episodes LOOKED like someone switching the lamp on. The renderer itself is correct; a
   controlled repeated-toggle probe (probe_lamp_semantics.py) gives ON=159.6 bright / OFF=111.3 dark.
   FIX: `run_setup()` skips any setup command whose target already satisfies it.

2. THE EGO CAMERA DID NOT LOOK AT WHAT THE CHARACTER WAS DOING.
   The built-in FIRST_PERSON camera is level at head height, so objects on tables, nightstands and
   counters fall to the bottom edge and the manipulation is barely visible. `TurnTo`/`LookAt` cannot
   fix this — they return ok=True but record zero frames. `update_character_camera()` also returns
   ok=True yet does NOT affect frames recorded later in a render loop (it works in isolation, so the
   failure is loop-specific — do not re-litigate this).
   FIX: register a LADDER of character-attached cameras (one per pitch) BEFORE `add_character`, then
   SELECT the right one per action via `camera_mode`, which demonstrably works.

Aiming rule: for Put/PutIn the camera looks at the DESTINATION (you look at the table you are setting
the book on), not at the carried object; for Grab/Open/Close/SwitchOn/SwitchOff/Drink it looks at the
object itself; Walk keeps a mild forward tilt because you look where you are going.
"""
import math, os, shutil

# Pitch ladder. 10-degree steps are finer than the aiming error from bounding-box centres, and seven
# cameras register in well under a second, so there is no reason to be coarser.
EGO_PITCHES = [0, 10, 20, 30, 40, 50, 60]
EGO_FOV = 80          # wider than default so the object AND the hands fit
EYE_DY = 1.45         # camera height above the character origin
WALK_PITCH = 10       # mild downward tilt while walking
MAX_PITCH = 40        # measured 2026-09-03: at 60 the built-in camera shows the character's
                      # own hair and body; 40 keeps a low coffee table in frame without them
MIN_PITCH = -15
REACH_STANDOFF = 0.75  # see pitch_to()

# State a setup command is asking for, used to detect redundant ones.
SETUP_WANT = {"SwitchOn": "ON", "SwitchOff": "OFF", "Open": "OPEN", "Close": "CLOSED"}


# Unity writes Output/ as root, and a stale frame directory that survives into the next glob is the
# "teleporting agent" bug (ENGINE_NOTES B3). Cleaning it used to be `docker exec vhN rm -rf`, which
# is a SILENT no-op on a host with no docker — exactly the failure mode the gate exists to prevent.
# On the docker-free launcher (tools/launch_native_hostgl.sh) Unity runs on the host as root, so the
# in-container path /unity_vol/... is really <repo>/unity_vol/... and a plain rm works.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAVE_DOCKER = shutil.which("docker") is not None


# ---------------------------------------------------------------- Unity script time limit
# comm_unity.render_script defaults processing_time_limit=10 SECONDS, and nothing in this repo
# overrides it. On software GL under load a single cross-room walk in the SETUP phase exceeds
# that and Unity answers HTTP 408 {'message': 'Timeout'} -- which killed three consecutive
# routine_hard renders while the same episode had rendered fine earlier the same day under a
# lighter box. Raising it is a timeout, NOT one of the renderer's integrity gates (black frame /
# teleport walk / zero-frame item / degraded length); those are untouched.
PROC_LIMIT = int(os.environ.get("VH_PROC_LIMIT", "300"))


def _install_proc_limit():
    """Give every render_script call in every tool the raised limit, once, at import."""
    try:
        from unity_simulator.comm_unity import UnityCommunication
    except Exception:
        return
    if getattr(UnityCommunication.render_script, "_proc_limit_wrapped", False):
        return
    _orig = UnityCommunication.render_script

    def render_script(self, *a, **kw):
        kw.setdefault("processing_time_limit", PROC_LIMIT)
        return _orig(self, *a, **kw)

    render_script._proc_limit_wrapped = True
    UnityCommunication.render_script = render_script


_install_proc_limit()


def root_rm(path, container=None):
    """Remove Unity's root-owned output `path` (an in-container absolute path, globs allowed).

    Returns True when the cleanup ran. Raises if neither route is available, rather than silently
    leaving stale frames behind.
    """
    if _HAVE_DOCKER and container:
        os.system(f"docker exec {container} bash -c 'rm -rf {path}' 2>/dev/null")
        return True
    if path.startswith("/unity_vol/"):
        host = os.path.join(_REPO, path.lstrip("/"))
        os.system(f"rm -rf {host}")
        return True
    raise RuntimeError(f"root_rm: no docker and {path!r} is not under /unity_vol — cannot clean")


def ego_cam_name(pitch):
    """Name of the ladder camera closest to `pitch`."""
    return f"EGO{min(EGO_PITCHES, key=lambda q: abs(q - pitch)):02d}"


def register_ego_ladder(comm, position=None, fov=EGO_FOV):
    """Register the whole pitch ladder. MUST be called after reset() and BEFORE add_character()."""
    pos = position or [0, EYE_DY, 0.25]
    for p in EGO_PITCHES:
        comm.add_character_camera(position=pos, rotation=[p, 0, 0], field_view=fov,
                                  name=f"EGO{p:02d}")
    return [f"EGO{p:02d}" for p in EGO_PITCHES]


def aim_target(step):
    """What the camera should look AT for this step, as (token, room), or None."""
    if len(step) > 4:
        return step[3], step[4]      # Put/PutIn -> the destination surface/container
    if len(step) > 2:
        return step[1], step[2]      # Grab/Open/Close/SwitchOn/SwitchOff/Drink -> the object
    return None


def pitch_to(comm, oid):
    """Downward pitch (deg) from the character's eyes to object `oid`'s centre, right now."""
    if oid is None:
        return None
    try:
        _ok, g = comm.environment_graph()
        tgt = next((n["bounding_box"]["center"] for n in g["nodes"] if n["id"] == oid), None)
        chn = next((n for n in g["nodes"] if n["class_name"] == "character"), None)
        if tgt is None or chn is None:
            return None
        ch = chn["bounding_box"]["center"]
        # The bbox CENTRE is mid-body (y~0.90 of a 1.82 m character) — the eyes are near the TOP of
        # the box. centre+1.45 lands a metre above her head and pegs every pitch at the clamp.
        eye_y = ch[1] + chn["bounding_box"]["size"][1] / 2.0 - 0.12
        d = math.hypot(tgt[0] - ch[0], tgt[2] - ch[2])
        # [Walk] parks the character essentially ON the target, so the raw horizontal distance
        # collapses to ~0 and atan2 saturates at the clamp — pointing the camera at her own chest.
        # Model the standoff you would actually have while reaching for something instead.
        return max(MIN_PITCH, min(MAX_PITCH,
                                  math.degrees(math.atan2(eye_y - tgt[1], max(d, REACH_STANDOFF)))))
    except Exception:
        return None


def yaw_to(comm, oid, limit=35.0):
    """Local yaw (deg, + = right) from the character's facing to object `oid`, clamped to ±limit.

    A manipulation is rendered with the camera pitched at the target but pointed straight ahead,
    and the body often parks beside the object rather than square to it: at a coffee table that
    stands against an exterior wall the straight-ahead view was the wall, near-plane clipped to
    open sky, while the plate went down out of frame. A bounded glance toward the target keeps
    the act in the picture without pretending the body turned."""
    if oid is None:
        return 0.0
    try:
        _ok, g = comm.environment_graph()
        tgt = next((n["bounding_box"]["center"] for n in g["nodes"] if n["id"] == oid), None)
        chn = next((n for n in g["nodes"] if n["class_name"] == "character"), None)
        if tgt is None or chn is None:
            return 0.0
        p = chn["obj_transform"]["position"]
        x, y, z, w = chn["obj_transform"]["rotation"]
        body = math.degrees(math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y)))
        bearing = math.degrees(math.atan2(tgt[0] - p[0], tgt[2] - p[2]))
        off = (bearing - body + 180.0) % 360.0 - 180.0
        return max(-limit, min(limit, off))
    except Exception:
        return 0.0


def aim_camera(comm, step, resolve, log=None):
    """Camera name to render `step` with. `resolve(token, room) -> object id` (or (id, name))."""
    pitch = WALK_PITCH
    tgt = aim_target(step)
    if step[0] != "Walk" and tgt is not None:
        try:
            oid = resolve(*tgt)
            if isinstance(oid, (tuple, list)):
                oid = oid[0]
            got = pitch_to(comm, oid)
            if got is not None:
                pitch = got
        except Exception:
            pass
    name = ego_cam_name(pitch)
    if log:
        log(f"  [aim] {step[0]:9} -> {(tgt or ('-', '-'))[0]:14} pitch={pitch:5.1f} cam={name}")
    return name


def is_redundant_setup(comm, step, oid):
    """True when this setup command asks for a state the object already holds (see module docstring)."""
    if step[0] not in SETUP_WANT or len(step) <= 2 or oid is None:
        return False, None
    try:
        _ok, g = comm.environment_graph()
        cur = next(([x.upper() for x in (n.get("states") or [])]
                    for n in g["nodes"] if n["id"] == oid), [])
        return (SETUP_WANT[step[0]] in cur), cur
    except Exception:
        return False, None


def run_setup(comm, setup, line_of, log=None, skip_animation=True):
    """Apply an episode's setup, skipping commands that would invert an already-correct state.

    `line_of(step)` renders a step tuple to a VH script line (or None if unresolvable). Returns the
    list of steps that were skipped, so callers can report them.
    """
    skipped = []
    for s in setup or []:
        s = tuple(s)
        sl = line_of(s)
        if not sl:
            continue
        oid = None
        try:
            oid = int(sl.split("(")[-1].split(")")[0])
        except Exception:
            pass
        redundant, cur = is_redundant_setup(comm, s, oid)
        if redundant:
            skipped.append(s)
            if log:
                log(f"  [setup] SKIP redundant {s[0]}({s[1]}) — already {cur}")
            continue
        comm.render_script([sl], recording=False, find_solution=False,
                           skip_animation=skip_animation)
    return skipped


def run_setup_lines(comm, lines, log=None, skip_animation=True):
    """Same guard as run_setup(), for callers that already rendered setup to VH script lines
    (e.g. `<char0> [SwitchOn] <tablelamp> (123)`). Returns the lines that were skipped."""
    skipped = []
    for sl in lines or []:
        if not sl:
            continue
        verb = sl.split("[", 1)[1].split("]", 1)[0] if "[" in sl else ""
        oid = None
        try:
            oid = int(sl.rsplit("(", 1)[1].split(")")[0])
        except Exception:
            pass
        redundant, cur = is_redundant_setup(comm, (verb, None, None), oid)
        if redundant:
            skipped.append(sl)
            if log:
                log(f"  [setup] SKIP redundant {sl} — already {cur}")
            continue
        comm.render_script([sl], recording=False, find_solution=False,
                           skip_animation=skip_animation)
    return skipped
