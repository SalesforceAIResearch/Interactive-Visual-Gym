import math
"""Mid-level action controller for VirtualHome.

Sits between the HIGH-level action scene graph ([Walk] <obj> (id) — teleports/pathfinds anywhere)
and the LOW-level scene graph ([WalkForward]/[TurnLeft] — single motor steps).

Design: every action is EGOCENTRIC, VISIBILITY-GATED and LOCALLY-BOUNDED. The agent can only act on
what is currently in its first-person view (via get_visible_objects), and object references always
resolve to the NEAREST visible instance — so to reach a distant object the controller (driven by a
streaming VLM) must actively turn to perceive it, then approach it, then manipulate it. The exact
footsteps are owned by the engine (not step-level), but the "where to look / which way to go"
decisions are the caller's job — the human-assistant role.

Two interfaces over the SAME logic:
  - programmatic: ctrl.go_to_nearest("door"), ctrl.pick_up_nearest("milk"), ctrl.turn_left(), ...
  - natural language: ctrl.execute("go through the visible door") -> parsed -> the same programmatic call.

Every action returns a structured Result so a streaming loop can feed observation+outcome back.
"""
import json, math, re
import numpy as np

TURN_DEG = 30.0          # one TurnLeft/TurnRight increment (measured)
# TURNING IS ONLY "turn left/right [N]" (A30, project decision 2026-09-15). Clock turns and the about-face
# phrase were removed from the action space; they are still RECOGNISED here only so an agent that
# says one is refused with the allowed form, instead of having its words silently dropped.
REMOVED_TURN_RE = re.compile(r"^(?:(?:turn\s+to|face)\s+(?:your\s+)?\d{1,2}\s*o'?\s*clock|turn\s+(?:around|back))$")
STEP_M = 1.0             # one WalkForward / WalkTowards advance (measured)
NEAR_M = 1.5             # manipulation requires the object visible AND within this distance.
                         # Anti-cheat: the [Walk] inside a manipulation is then only a <=1.5m final
                         # nudge, never a long walk that finds/reaches the object for you. (1.5m not
                         # 1.0m because VirtualHome's first-person camera drops objects below the FOV
                         # under ~1-2m, so "visible AND <=1m" is nearly unsatisfiable in practice.)
ARRIVE_M = 1.5           # go_to stops at NEAR_M — close enough to manipulate, far enough to stay visible
MAX_APPROACH_STEPS = 12  # bound on a single go_to / go_through
MAX_STRAIGHT_STEPS = 12  # bound on go_straight
# Egocentric camera PITCH levels (deg down from horizontal). VirtualHome's built-in FIRST_PERSON is fixed
# level + narrow, so a low object on a table drops below view at <1.5m and the visible-gate refuses the
# grab. We instead use our own wide-FOV character camera and let the VLM pitch it: level (navigation),
# 14deg (table/counter height — default), 28deg (floor/very low). 14deg is verified to bring a close
# plate back into view where level does not; 28deg+ overshoots table height.
EGO_PITCHES = [0, 14, 28]
EGO_DEFAULT_LVL = 0      # default to a LEVEL view so tall objects (fridge/microwave/tv/doors) are visible
                         # while navigating — a down-tilt default hides them and the agent can't find them.
                         # Low-object manipulation is covered by the multi-pitch gate (auto-glances down),
                         # and the VLM can "look down" to visually inspect a table/counter when it wants.
EGO_FOV = 80             # MUST equal vh_render_fix.EGO_FOV: the gate camera and the rendered
                         # frames are the same eye (fix #44) — a 60/80 split let the VLM see
                         # objects the gate then refused, and the two argued about it.
GAZE_CONE = 35.0         # (legacy) deg half-angle once used by the manip gate; gate is now near-only
GEOM_RANGE = 8.0         # m: VESTIGIAL for live visibility (no distance cap anymore — sight is limited by
                         # walls/doorways + frame + occlusion only); still imported by offline dataset tools.
OCCLUDE_MIN = 2.5        # only apply wall-ray occlusion beyond this distance (close objects are never
                         # falsely hidden by their own backing wall; same-room handles cross-room walls)
# field_view=65 is the VERTICAL fov; the rendered image is 480x320 (aspect 1.5), so the HORIZONTAL
# half-fov is atan(tan(32.5deg)*1.5) ~= 44deg. Use that (a touch generous) so the controller's notion of
# "visible" matches what the VLM actually sees in the wide frame — else 'go to nearest X' is refused for
# an X plainly in view at the frame edge.
VIS_HALF_FOV = 48.0      # TRUE horizontal half-FOV of the 80-deg-vertical 4:3 camera:
                         # atan(tan(40deg)*4/3) = 48.2deg. The old 66 OVERSTATED the window
                         # ~1.4x, so agents believed a 30-deg turn kept things in view when it
                         # did not. (Historic note below kept for the calibration context.)
                         # horizontal half-FOV for "visible". Calibrated against Gemini-as-visual-oracle on
                         # 1072 labelled (frame,object) samples: Gemini recognises objects out to ~71deg
                         # bearing (the rendered frame is wide), so 46deg was far too narrow and wrongly
                         # refused 'go to X' for things plainly in view. 66deg captures ~88% of what Gemini
                         # sees. (Same-room filtering keeps the wider cone from leaking into other rooms.)
VIS_FAR_M = 4.0          # beyond this, an object must have enough angular size to be recognisable...
VIS_MIN_ANG = 0.06       # ...(bbox_width / distance) — matches Gemini not "seeing" a fork 5m away.


class Result(dict):
    def __init__(self, action, target=None, success=False, reason="", **kw):
        super().__init__(action=action, target=target, success=success, reason=reason, **kw)
    def __repr__(self):
        t = f" {self['target']}" if self.get('target') else ""
        return f"[{'OK ' if self['success'] else 'FAIL'}] {self['action']}{t}: {self['reason']}"


# common VLM vocabulary -> VirtualHome class names (so "go to nearest television" resolves to <tv>).
# applied after spaces are stripped, so "coffee table"->"coffeetable" etc. are already handled.
SYN = {
    "television": "tv", "telly": "tv", "screen": "tv",
    "couch": "sofa", "settee": "sofa",
    "lamp": "tablelamp", "desklamp": "tablelamp", "nightlight": "tablelamp",
    "fork": "cutleryfork", "knife": "cutleryknife",
    "refrigerator": "fridge", "icebox": "fridge",
    "trash": "garbagecan", "trashcan": "garbagecan", "garbage": "garbagecan",
    "bin": "garbagecan", "dustbin": "garbagecan",
    "pc": "computer", "desktop": "computer", "monitor": "computer",
    "cupboard": "kitchencabinet",
}
def canon(word):
    w = word.lower().replace(" ", "")
    return SYN.get(w, w)


class MidLevelController:
    def __init__(self, comm, char_resource="Chars/Female1"):
        self.c = comm
        self.char = char_resource
        self._fp = None
        self._scene = None
        self._room = None
        self._log = []        # low-level (script_lines, find_solution) emitted — the trajectory, for video
        self._replaying = False

    def _render(self, script, find_solution):
        """Central render for the LIVE decision loop: fast, non-recording (skip_animation=True). Each
        emitted low-level group is logged so the whole rollout can be replayed into a video afterwards
        (single-action recorded renders emit no frames in VH; one big replayed render does).

        `render_hook` lets a caller take the render over. The interaction experiments need frames
        PER COMMAND — the assistant watches live and the human may interrupt mid-action — which
        the two-pass replay above cannot provide. Note the parenthetical is about skip_animation,
        not about recording: a recorded render DOES emit frames when it actually animates, which
        is what the hook does.
        """
        hook = getattr(self, "render_hook", None)
        if hook is not None:
            return hook(script, find_solution)
        ok, msg = self.c.render_script(script, recording=False,
                                       find_solution=find_solution, skip_animation=True)
        if ok and not self._replaying:
            self._log.append((list(script), find_solution))
        return ok, str(msg)[:90]

    # ---------- video: two-pass (decide fast, then replay the trajectory as one recorded render) ----------
    @staticmethod
    def _as_walk(line):
        """'<char0> [Open] <fridge> (12)' -> '<char0> [Walk] <fridge> (12)' (target part preserved)."""
        head, _, rest = line.partition("]")
        return head.split("[")[0] + "[Walk]" + rest if "]" in line else line

    def render_trajectory(self, folder, camera_mode=("FIRST_PERSON",), width=480, height=320):
        """Reset to the episode's initial state and replay EVERY low-level action the VLM's commands
        produced, as ONE recorded render -> a continuous egocentric video of the whole rollout. A
        find_solution=True manip (Open/Close/Switch/Drink) is replayed as the [Walk]+[action] pair so it
        still commits and contributes frames under the single find_solution=False render. Returns
        (ok, msg, n_frames_actions, frame_dir)."""
        log = list(self._log)
        self.c.reset(self._scene)
        self.c.add_character(self.char, initial_room=self._room)
        flat = []
        for lines, fs in log:
            if fs and len(lines) == 1:               # single manip that needed facing -> prepend a Walk
                flat.append(self._as_walk(lines[0]))
            flat.extend(lines)
        ok, msg = self.c.render_script(flat, recording=True, find_solution=False, skip_animation=False,
                                       output_folder=folder.rstrip("/") + "/", file_name_prefix="rollout",
                                       frame_rate=5, image_width=width, image_height=height,
                                       camera_mode=list(camera_mode))
        return ok, str(msg)[:120], len(flat), folder.rstrip("/") + "/rollout/0"

    # ---------- setup ----------
    def _locate_fp(self):
        # THE agent's eye = the built-in forward first-person camera ("Character_Camera_Fwd"). Resolve it
        # by NAME via camera_data (the character_cameras() list order does NOT match global camera indices
        # — a bug that had us rendering a side-facing camera). This camera is used for BOTH the rendered
        # image the VLM sees AND (via its real projection/view matrices) the geometric visibility.
        self._fp = None
        for i in range(self.c.camera_count()[1]):
            ok, d = self.c.camera_data([i])
            if d and d[0].get("name") == "Character_Camera_Fwd":
                self._fp = i; break
        if self._fp is None:                                   # fallback: FIRST_PERSON via the name list
            ncam = self.c.camera_count()[1]; ok, cc = self.c.character_cameras()
            cc = json.loads(cc) if isinstance(cc, str) else cc
            self._fp = ncam - len(cc) + cc.index("FIRST_PERSON")
        self._lvl = 0
        self._seen = set()
        return self._fp

    def _third_person_cam(self):
        # Ground-truth over-the-shoulder camera ("Character_Camera_FBack"): looks at the agent from behind,
        # so a viewer sees the agent AND the scene ahead. Resolved by name (indices are not stable).
        for i in range(self.c.camera_count()[1]):
            ok, d = self.c.camera_data([i])
            if d and d[0].get("name") == "Character_Camera_FBack":
                return i
        return None

    def _cam_matrices(self):
        """(projection 4x4, view 4x4, camera world position) for the forward camera, NOW. Column-major in
        the JSON -> transpose. clip = P @ (V @ [x,y,z,1]); in front iff clip.w>0; in frame iff |ndc|<=1."""
        ok, d = self.c.camera_data([self._fp]); d = d[0]
        P = np.array(d["projection_matrix"], float).reshape(4, 4).T
        V = np.array(d["world_to_camera_matrix"], float).reshape(4, 4).T
        cam = np.linalg.inv(V)[:3, 3]
        return P, V, cam

    @staticmethod
    def _bbox_corners(bb):
        ctr = np.array(bb["center"], float); s = np.array(bb["size"], float) / 2.0
        return [ctr + s * np.array([dx, dy, dz]) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]

    @staticmethod
    def _ray_hits_aabb(o, dirn, tmax, lo, hi):
        tmin, tM = 0.02, tmax - 0.08                            # ignore hits at the very ends (grazing)
        for i in range(3):
            if abs(dirn[i]) < 1e-9:
                if o[i] < lo[i] or o[i] > hi[i]: return False
            else:
                t1 = (lo[i] - o[i]) / dirn[i]; t2 = (hi[i] - o[i]) / dirn[i]
                if t1 > t2: t1, t2 = t2, t1
                tmin = max(tmin, t1); tM = min(tM, t2)
                if tmin > tM: return False
        return True

    # Head-pitch control via update_camera on the built-in forward camera (default = 30deg down; verified
    # that reset+add_character restores the default, so each episode starts from the same baseline).
    _FP_POS = [0, 1.8, 0.06]           # camera local position (matches the default eye height ~1.79 world)
    _FP_FOV = 80          # = EGO_FOV: gate eye == picture eye (fix #44)
    _NAV_PITCH = 30                    # the camera's default navigation pitch (deg down)
    _TILTS = (30, 50, 10, 65)          # auto-tilt candidates: nav, low/close, high/tall, floor-close

    def _set_gaze(self, pitch, yaw=0):
        """Set the HEAD camera orientation relative to the body: pitch (deg down) + local yaw (deg right).
        The head can look at a target even while the body walks a path that points elsewhere."""
        pitch, yaw = int(round(pitch)), int(round(yaw))
        if getattr(self, "_pitch", None) == pitch and getattr(self, "_gyaw", 0) == yaw:
            return
        self.c.update_camera(self._fp, position=self._FP_POS, rotation=[pitch, yaw, 0], field_view=self._FP_FOV)
        self._pitch, self._gyaw = pitch, yaw

    def _set_pitch(self, pitch):
        self._set_gaze(pitch, 0)

    def _ensure_in_view(self, oid, g=None):
        """AUTO HEAD-TILT: pitch the head so @oid sits in the CENTRE of the view (not merely inside the
        frame). Computes the exact down-pitch that points the camera at the object's bbox centre
        (atan of height drop over ground distance) and sets it continuously; falls back to the discrete
        tilt ladder if the centred pitch is still blocked (occlusion). Used during 'go to X' so the target
        stays centred as the agent closes in. Returns True if the target ends up visible."""
        g = g or self._graph()
        tgt = next((n for n in g["nodes"] if n["id"] == oid), None)
        if tgt is not None:
            oc = np.array(tgt["bounding_box"]["center"], float)
            # first: analytic aim (gets close even if the target is currently behind the view)
            _, _, cam = self._cam_matrices()
            ground = float(np.hypot(oc[0] - cam[0], oc[2] - cam[2]))
            pitch = max(-10.0, min(70.0, math.degrees(math.atan2(cam[1] - oc[1], max(ground, 0.1)))))
            _p, byaw = self._char(g)
            dyaw = (math.degrees(math.atan2(oc[0] - cam[0], oc[2] - cam[2])) - byaw + 180) % 360 - 180
            self._set_gaze(pitch, max(-80.0, min(80.0, dyaw)))
            # then: residual-driven refinement measured in the ACTUAL camera frame (robust to conventions)
            for _pass in range(3):
                P, V, _ = self._cam_matrices()
                cc = P @ (V @ np.array([oc[0], oc[1], oc[2], 1.0]))
                if cc[3] <= 0.05:
                    break
                ndx, ndy = float(cc[0] / cc[3]), float(cc[1] / cc[3])
                if abs(ndx) < 0.05 and abs(ndy) < 0.05:
                    break                                            # centred on both axes
                doff = math.degrees(math.atan(ndy * math.tan(math.radians(self._FP_FOV / 2))))
                yoff = math.degrees(math.atan(ndx * math.tan(math.radians(self._FP_FOV / 2)) * 4 / 3))
                self._set_gaze(max(-10.0, min(70.0, self._pitch - doff)),
                               max(-80.0, min(80.0, self._gyaw + yoff)))
            if oid in self.visible_ids(g=g):
                return True
        for pitch in self._TILTS:                          # fallback: any tilt that sees it at all
            self._set_pitch(pitch)
            if oid in self.visible_ids(g=g):
                return True
        self._set_pitch(self._NAV_PITCH)
        return False

    def _set_level(self, lvl):     # legacy level API -> mapped onto real head pitches
        self._lvl = max(0, min(len(self._TILTS) - 1, int(lvl)))
        self._set_pitch(self._TILTS[self._lvl])
        return self._TILTS[self._lvl]

    def _pitch_step(self, n):
        """Move n steps through the PITCH-SORTED tilt ladder (negative = up, toward 10deg).

        _TILTS is ordered for the auto-tilt SCAN (nav first, then close-low, high, floor) and
        was never sorted by pitch — but look_up/look_down used to step through its INDEX, so
        from the default (index 0 = 30deg) 'look up' clamped at index 0 and reported "already
        looking level (30deg)" while pointing at the floor. A human spent 14 ticks "looking
        up" at a supervisor's request with the command physically unable to do anything.
        """
        order = sorted(self._TILTS)                     # (10, 30, 50, 65): up ... down
        cur = order.index(self._TILTS[self._lvl])
        tgt = max(0, min(len(order) - 1, cur + int(n)))
        self._set_level(self._TILTS.index(order[tgt]))
        return order[tgt], tgt != cur

    def look_down(self, n=1):
        p, moved = self._pitch_step(int(n))
        msg = (f"looking down ({p}deg) — low objects on tables/the floor are now in view"
               if moved else f"already looking as far down as possible ({p}deg)")
        return Result("look_down", f"{p}deg", True, msg, **self.observation())

    def look_up(self, n=1):
        p, moved = self._pitch_step(-int(n))
        msg = (f"looking up ({p}deg) — better view ahead for walking/finding things"
               if moved else f"already looking as far up as possible ({p}deg)")
        return Result("look_up", f"{p}deg", True, msg, **self.observation())

    def reset(self, scene, room="kitchen"):
        self._scene, self._room = scene, room
        self._log = []
        self.c.reset(scene)
        self.c.add_character(self.char, initial_room=room)
        return self._locate_fp()

    def attach(self, scene, room="kitchen"):
        """Use when the caller has ALREADY reset the scene + added the character (e.g. after running an
        episode's off-camera setup): just locate the fp camera + reset bookkeeping, don't touch the scene."""
        self._scene, self._room = scene, room
        self._log = []
        fp = self._locate_fp()
        self._pitch = None
        self._set_pitch(self._NAV_PITCH)   # known baseline (= the camera's default 30deg-down nav view)
        return fp

    # ---------- perception ----------
    def _graph(self):
        s, g = self.c.environment_graph()
        return g

    def _char(self, g=None):
        g = g or self._graph()
        ch = [n for n in g["nodes"] if n["class_name"] == "character"][0]
        p = np.array(ch["obj_transform"]["position"], dtype=float)
        x, y, z, w = ch["obj_transform"]["rotation"]
        yaw = math.degrees(math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y)))
        return p, yaw

    _STRUCT = {"wall", "ceiling", "floor", "doorjamb", "window", "slab", "curtain", "rug", "ceilinglamp",
               "wallshelf", "wallpictureframe", "powersocket", "lightswitch", "orchid", "candle"}
    def scene_object_classes(self, rooms=None):
        """Sorted distinct object class names that actually sit in the given rooms (default: every room
        EXCEPT the bathroom, since the benchmark tasks only span kitchen/bedroom/livingroom) — minus
        structural clutter. The exact names the controller understands, to ground the VLM's references."""
        g = self._graph()
        roomname = {n["id"]: n["class_name"] for n in g["nodes"] if n.get("category") == "Rooms"}
        want = set(rooms) if rooms else {r for r in roomname.values() if "bathroom" not in r}
        objroom = {e["from_id"]: roomname[e["to_id"]] for e in g["edges"]
                   if e["relation_type"] == "INSIDE" and e["to_id"] in roomname}
        return sorted({n["class_name"] for n in g["nodes"]
                       if n.get("category") != "Rooms" and n["class_name"] != "character"
                       and n["class_name"] not in self._STRUCT and objroom.get(n["id"]) in want})

    @staticmethod
    def _room_of(px, pz, rooms):
        """Which room's bounding box (XZ) contains the point — robust room assignment by POSITION (object
        INSIDE-edges point to containers/surfaces, not rooms, so they can't be used). Falls back to the
        nearest room center if the point is outside every box."""
        inside = [(abs(px - bb["center"][0]) + abs(pz - bb["center"][2]), rid)
                  for rid, bb in rooms
                  if abs(px - bb["center"][0]) <= bb["size"][0] / 2 and abs(pz - bb["center"][2]) <= bb["size"][2] / 2]
        if inside:
            return min(inside)[1]
        return min(rooms, key=lambda r: (px - r[1]["center"][0]) ** 2 + (pz - r[1]["center"][2]) ** 2)[1]

    @staticmethod
    def _seg_hits_box(ax, az, bx, bz, cx, cz, hx, hz):
        """2D Liang-Barsky: does segment (a->b) cross the axis-aligned box centred (cx,cz) half (hx,hz)?"""
        dx, dz = bx - ax, bz - az
        tmin, tmax = 0.0, 1.0
        for p, q in ((-dx, ax - (cx - hx)), (dx, (cx + hx) - ax), (-dz, az - (cz - hz)), (dz, (cz + hz) - az)):
            if abs(p) < 1e-9:
                if q < 0: return False
            else:
                t = q / p
                if p < 0: tmin = max(tmin, t)
                else:     tmax = min(tmax, t)
                if tmin > tmax: return False
        return True

    def _ray_blocked(self, ax, az, bx, bz, walls):
        """True if a WALL crosses the segment a->b. Endpoint pulled 0.4m back so the wall a fridge/tv is
        mounted against doesn't count."""
        dx, dz = bx - ax, bz - az
        L = math.hypot(dx, dz)
        if L > 0.4:
            bx, bz = ax + dx * (L - 0.4) / L, az + dz * (L - 0.4) / L
        return any(self._seg_hits_box(ax, az, bx, bz, cx, cz, hx, hz) for cx, cz, hx, hz in walls)

    def _wall_visible(self, ax, az, node, walls):
        """Visible if a wall-free line reaches ANY sampled point of the target's footprint (centre + the
        four XZ bbox corners) — so a thin/small occluder grazing the centre line can't hide a large object
        whose edges are in plain view, and a wall only blocks when it covers the WHOLE object."""
        c = node["bounding_box"]["center"]; s = node["bounding_box"]["size"]
        hx, hz = s[0] / 2.0, s[2] / 2.0
        pts = [(c[0], c[2]), (c[0] - hx, c[2] - hz), (c[0] + hx, c[2] - hz),
               (c[0] - hx, c[2] + hz), (c[0] + hx, c[2] + hz)]
        return any(not self._ray_blocked(ax, az, tx, tz, walls) for tx, tz in pts)

    @staticmethod
    def _bigger_all(a, b):
        """is bbox-size a bigger than b in ALL three dimensions (orientation-independent: compare sorted)?"""
        a, b = sorted(a), sorted(b)
        return a[0] > b[0] + 0.05 and a[1] > b[1] + 0.05 and a[2] > b[2] + 0.05

    def _occluded_by_bigger(self, px, pz, n, ndist, occ):
        """n is occluded iff some object that is CLOSER and BIGGER-in-all-dims lies on the agent->n line
        (its bbox crosses the ray). 'bigger in all dims' = a definite blocker, so a small object can never
        occlude a large one (per the lightweight rule) and a thin wall (small in one dim) never counts."""
        nc = n["bounding_box"]["center"]; nsz = n["bounding_box"]["size"]
        for md, msz, mc, mid in occ:
            if mid == n["id"] or md >= ndist - 0.3:
                continue
            if not self._bigger_all(msz, nsz):
                continue
            if self._seg_hits_box(px, pz, nc[0], nc[2], mc[0], mc[2], msz[0] / 2, msz[2] / 2):
                return True
        return False

    # SEEING OVER OPEN-TOPPED FURNITURE (A26). A bounding box is not a solid -- A23 fixed the case
    # where the box CONTAINS the target; this is the case where it sits BETWEEN eye and target. A
    # sofa's box is filled solid to the top of its backrest (0.97 m in apt 1/3), but the thing at
    # that height is a 0.25 m deep backrest, and the rest is air over a 0.45 m seat. Measured in
    # pilot 5: looking from the bookshelf at a plate on the coffee table, the eye->plate line enters
    # the sofa footprint at 1.14 m (clear) and leaves it at 0.88 m -- open air above the cushions --
    # and the box test called it blocked. The plate was culled at all 12 headings while the 1.28 m
    # coffee table beside it survived at 4 (its corners peek past the sofa's end), i.e. the rule was
    # deciding by object size. For these classes the occluder is only its seat: cap the box there.
    SEE_OVER_H = {"sofa": 0.55, "sofachair": 0.55, "loveseat": 0.55, "armchair": 0.55,
                  "chair": 0.55, "bench": 0.50, "stool": 0.50}

    @staticmethod
    def _see_over_box(cls_name, ctr, sz):
        """(centre, size) of the part that actually blocks a low sight line: for open-topped
        seating the seat, for everything else the whole box."""
        cap = MidLevelController.SEE_OVER_H.get(cls_name)
        if cap is None or sz[1] <= cap:
            return ctr, sz
        lo_y = ctr[1] - sz[1] / 2
        ctr = np.array([ctr[0], lo_y + cap / 2, ctr[2]], float)
        sz = np.array([sz[0], cap, sz[2]], float)
        return ctr, sz

    def _occluded(self, cam, tgt, nctr, nsz, nid, occ):
        d = tgt - cam; dist = float(np.linalg.norm(d))
        if dist < 1e-3:
            return False
        dirn = d / dist
        for mc, ms, mid in occ:
            if mid == nid or np.linalg.norm(mc - nctr) < 0.8:               # skip self + adjacent support
                continue
            sm, sn = sorted(ms), sorted(nsz)
            if not (sm[0] > sn[0] and sm[1] > sn[1] and sm[2] > sn[2]):     # occluder bigger in ALL dims
                continue
            # CONCAVE FURNITURE (A23). A bounding box is not a solid: an L-shaped sectional's box
            # (3.3 x 3.9 m in apt 1) CONTAINS the coffee table and the plate it wraps around, so every
            # sight line to them "hit the sofa" and the plate was invisible from the whole room. A
            # box that contains the target's centre, or the eye itself, is not a wall between them.
            lo_, hi_ = mc - ms / 2, mc + ms / 2
            if (np.all(nctr >= lo_) and np.all(nctr <= hi_)
                    and (nid, mid) not in getattr(self, "_closed_inside", ())):   # a CLOSED container still hides its contents
                continue
            if np.all(cam >= lo_) and np.all(cam <= hi_):
                continue
            if self._ray_hits_aabb(cam, dirn, dist, mc - ms / 2, mc + ms / 2):
                return True
        return False

    def _proj_visible(self, P, V, cam, n, occ):
        """Visible iff the object's projected bounding box OVERLAPS the viewport [-1,1]^2 (so a large object
        that spans past the frame edges, with no single corner inside, still counts) AND a ray from the
        camera to an in-view point on it isn't blocked by a closer, bigger object."""
        nsz = np.array(n["bounding_box"]["size"], float); nctr = np.array(n["bounding_box"]["center"], float)
        front = []
        for pc in self._bbox_corners(n["bounding_box"]):
            clip = P @ (V @ np.array([pc[0], pc[1], pc[2], 1.0]))
            if clip[3] > 0.05:
                ndc = clip[:3] / clip[3]; front.append((ndc[0], ndc[1], pc))
        if not front:                                                       # entirely behind the camera
            return False
        xs = [f[0] for f in front]; ys = [f[1] for f in front]
        if max(xs) < -1 or min(xs) > 1 or max(ys) < -1 or min(ys) > 1:      # projected box misses viewport
            return False
        # occlusion: try the center, then any corner whose projection lands inside the viewport
        cc = P @ (V @ np.array([nctr[0], nctr[1], nctr[2], 1.0]))
        pts = []
        if cc[3] > 0.05 and abs(cc[0] / cc[3]) <= 1 and abs(cc[1] / cc[3]) <= 1:
            pts.append(nctr)
        pts += [f[2] for f in front if abs(f[0]) <= 1 and abs(f[1]) <= 1]
        if not pts:                                                         # spans frame but no sample inside
            pts = [nctr]
        return any(not self._occluded(cam, t, nctr, nsz, n["id"], occ) for t in pts)

    # WITHIN ARM'S REACH THE ROOM LABELS DO NOT DECIDE (A24). The wall test culls a sight line that
    # crosses a room boundary away from a door, but room boxes overlap by ~0.5 m everywhere and VH's
    # wall nodes are 2.5 m cubes, so an object standing right AT a boundary -- apartment 4's desk lamp,
    # 0.9 m from the person, unobstructed and squarely in frame -- was culled as "through a wall" and
    # every walk and switch on it was refused. Nothing real can hide behind a wall this close: below
    # this distance only the occlusion test decides.
    SHORT_SIGHT_M = 2.0

    DOOR_APERTURE = 0.85     # m: a room-to-room sight-line crossing counts as "through the doorway" if the
                             # crossing point is within this radius of a door centre (door ~0.9m wide + slack)

    def _sightline_clear(self, p2, q2, rooms, doors, my_room, walls=()):
        """True if the straight ground-plane sight-line p2 -> q2 does not pass through a WALL. Same room is
        trivially clear; every room-boundary crossing along the line must happen inside a doorway aperture
        (so an object in the next room IS visible through an open door, and an object two rooms away is
        visible when the line threads both doorways). Replaces the old binary same-room cull."""
        if self._room_of(q2[0], q2[1], rooms) == my_room:
            return True
        dx, dz = q2[0] - p2[0], q2[1] - p2[1]
        L = float(np.hypot(dx, dz))
        n = max(2, int(L / 0.15))
        cur = my_room
        for i in range(1, n + 1):
            x, z = p2[0] + dx * i / n, p2[1] + dz * i / n
            r = self._room_of(x, z, rooms)
            if r != cur:                                   # boundary crossing: must be at a doorway
                if not any((x - dc[0]) ** 2 + (z - dc[2]) ** 2 <= self.DOOR_APERTURE ** 2 for dc in doors):
                    return False
                cur = r
        return True

    # Pitches a gate may test (deg down): what the rendered picture can actually show -- walking tilt
    # (vh_render_fix.WALK_PITCH = 10) up to the render cap (MAX_PITCH = 40; beyond it the camera
    # shows the character's own body).
    GATE_PITCHES = (10, 25, 40)

    def visible_ids(self, level=None, g=None):
        """`level` indexes GATE_PITCHES: the camera is set to that pitch for the test (A23 -- the
        argument used to be ignored, so an "any pitch" sweep tested one view three times)."""
        if level is None:
            return self._visible_ids_now(g=g)
        pitch = self.GATE_PITCHES[max(0, min(len(self.GATE_PITCHES) - 1, int(level)))]
        self._pitch = None                     # the world re-aims this camera for every render:
        self._set_gaze(pitch, 0)               # never trust the cached value, force the pitch
        return self._visible_ids_now(g=g)

    def _visible_ids_now(self, g=None):
        """Objects the agent can SEE — computed from the forward camera's REAL projection + view matrices
        and 3D ray/AABB occlusion (not the engine's flaky get_visible_objects). Visible iff a bbox corner
        projects into the frame AND its ray from the camera isn't blocked by a closer, bigger object AND the
        sight-line doesn't pass through a wall (door-aware: seeing into the next room THROUGH a doorway is
        allowed). No distance cap. Matches the rendered frame, so 'go to nearest X' works."""
        _g = g or self._graph()
        _closed = {n["id"] for n in _g["nodes"] if "CLOSED" in (n.get("states") or [])}
        self._closed_inside = {(e["from_id"], e["to_id"]) for e in _g["edges"]
                               if e["relation_type"] == "INSIDE" and e["to_id"] in _closed}
        g = _g
        g = g or self._graph()
        p, _ = self._char(g)
        rooms = [(n["id"], n["bounding_box"]) for n in g["nodes"] if n.get("category") == "Rooms"]
        doors = [n["bounding_box"]["center"] for n in g["nodes"] if n["class_name"] == "door"]
        for ps in self._passages:                      # an open passage is a doorway for sight too (A32)
            for q in (ps.get("points") or [ps["center"]]):
                doors.append([q[0], 1.0, q[1]])

        my_room = self._room_of(p[0], p[2], rooms) if rooms else None
        P, V, cam = self._cam_matrices()
        occ = [self._see_over_box(n["class_name"],
                                  np.array(n["bounding_box"]["center"], float),
                                  np.array(n["bounding_box"]["size"], float)) + (n["id"],)
               for n in g["nodes"] if n.get("category") not in ("Rooms", "Floor", "Ceiling", "Walls")
               and n["class_name"] != "character" and max(n["bounding_box"]["size"]) > 0.5]
        ids = set()
        for n in g["nodes"]:
            if n.get("category") in ("Rooms", "Walls", "Floor", "Ceiling") or n["class_name"] == "character":
                continue
            oc = n["bounding_box"]["center"]
            # Doors ARE the apertures — exempt from the wall test (their centre sits on the boundary).
            if (my_room is not None and n.get("category") != "Doors"
                    and float(np.hypot(oc[0] - cam[0], oc[2] - cam[2])) > self.SHORT_SIGHT_M
                    and not self._sightline_clear((cam[0], cam[2]), (oc[0], oc[2]), rooms, doors, my_room)):
                continue
            if self._proj_visible(P, V, cam, n, occ):
                ids.add(n["id"])
        if getattr(self, "_seen", None) is not None:
            self._seen |= ids
        return ids

    def _visible_any(self):
        """Map id -> the ego pitch LEVEL (nearest to the current one) at which it is visible. Lets a
        manipulation 'glance' down/up to find an object that dropped out of the level view at close
        range — a low plate is seen at the 14deg level, a tall fridge at the level (0deg) view."""
        cur = getattr(self, "_pitch", None)
        cur = self._NAV_PITCH if cur is None else cur
        order = sorted(range(len(self.GATE_PITCHES)), key=lambda l: abs(self.GATE_PITCHES[l] - cur))
        seen = {}
        for lvl in order:
            for i in self.visible_ids(level=lvl):
                seen.setdefault(i, lvl)
        return seen

    def observation(self, g=None):
        """What a streaming VLM would get back: visible object classes + a few nearest, by distance."""
        g = g or self._graph()
        byid = {n["id"]: n for n in g["nodes"]}
        p, yaw = self._char(g)
        vis = self.visible_ids()
        items = []
        for i in vis:
            n = byid.get(i)
            if not n or n["class_name"] in ("character", "kitchen", "livingroom", "bedroom", "bathroom"):
                continue
            d = float(np.linalg.norm((np.array(n["obj_transform"]["position"], float) - p)[[0, 2]]))
            items.append((round(d, 2), n["class_name"], i))
        items.sort()
        return {"yaw": round(yaw, 1), "pos": [round(float(x), 2) for x in p[[0, 2]]],
                "visible": [f"{cl}#{i}({d}m)" for d, cl, i in items[:12]]}

    def _nearest_visible(self, query, g=None, require_visible=True, any_pitch=False):
        """Nearest VISIBLE instance whose class matches @query, by ground-plane distance. Prefers an
        EXACT class match (so 'bed' != 'bedroom', 'tv' != 'tvstand') and falls back to a suffix match
        only if there is no exact one. Rooms and the character are never matched. With any_pitch=True the
        visibility check spans all ego pitch levels and the view SNAPS to the level where the chosen
        object is seen (so a close low object the level view missed still counts as visible)."""
        g = g or self._graph()
        p, _ = self._char(g)
        vmap = self._visible_any() if any_pitch else None
        vis = set(vmap) if any_pitch else self.visible_ids()
        q = canon(query)                       # normalize synonyms: television->tv, couch->sofa, ...
        exact, partial = [], []
        for n in g["nodes"]:
            if n.get("category") == "Rooms" or n["class_name"] == "character":
                continue
            if require_visible and n["id"] not in vis:
                continue
            cl = n["class_name"].lower()
            # exact, else SUFFIX match only ('table' -> 'kitchentable', 'cabinet' -> 'kitchencabinet')
            # — a suffix, NOT any substring, so 'book' does NOT match 'bookshelf' nor 'tv' 'tvstand'.
            if cl == q:
                exact.append((float(np.linalg.norm((np.array(n["obj_transform"]["position"], float) - p)[[0, 2]])), n["id"], n["class_name"]))
            elif cl.endswith(q):
                partial.append((float(np.linalg.norm((np.array(n["obj_transform"]["position"], float) - p)[[0, 2]])), n["id"], n["class_name"]))
        pool = exact if exact else partial
        pool.sort()
        if not pool:
            return None
        if any_pitch and vmap is not None:
            self._pitch = None
            self._set_pitch(self.GATE_PITCHES[vmap[pool[0][1]]])   # snap the view to where it is seen
        return pool[0]  # (dist, id, class)

    # ---------- low-level emitters ----------
    def _do(self, script_line, find_solution=False):
        # locomotion uses find_solution=False (pure low-level steps). Manipulation uses
        # find_solution=True so VirtualHome establishes the required CLOSE/facing relation for the
        # final ~1m before Grab/Open/etc. — this is bounded because the action is only attempted
        # once the target is already gated to be visible AND within REACH_M.
        return self._render([f"<char0> {script_line}"], find_solution)

    # ---------- mid-level actions (programmatic) ----------
    def _face_heading(self, tx, tz):
        """D8 (pilot 2026-09-05): turn until the body faces the world point (tx, tz) as closely
        as 30-degree turns allow. The sign of a turn is measured from the first one rather than
        assumed, so this holds whichever way the engine spins."""
        def yaw_to_pt(p):
            return math.degrees(math.atan2(tx - p[0], tz - p[2]))
        def wrap(d):
            return (d + 180.0) % 360.0 - 180.0
        p, y = self._char()
        delta = wrap(yaw_to_pt(p) - y)
        if abs(delta) < 15.0:
            return 0
        self._do("[TurnRight]")
        p1, y1 = self._char()
        right_sign = 1.0 if wrap(y1 - y) > 0 else -1.0     # +: TurnRight increases yaw
        delta = wrap(yaw_to_pt(p1) - y1)
        n = int(round(abs(delta) / TURN_DEG))
        cmd = "[TurnRight]" if (delta * right_sign) > 0 else "[TurnLeft]"
        for _ in range(min(n, 6)):
            self._do(cmd)
        return n + 1

    def _door_crossing(self, oid, g=None):
        """(door_centre, unit crossing vector, here_room_id, target_room_id) or None.

        VirtualHome walls are axis-aligned, so a doorway is crossed along x or z. Of the four
        cardinal directions, the crossing direction is the one whose point 1 m ahead of the door
        lies inside a room other than the body's while the point 1 m behind lies inside the
        body's room. This reads the ROOMS, not the door leaf: a swung-open leaf's own bounding
        box runs along the wall (A19), and the nearest neighbouring room centre can belong to a
        third room when three rooms meet."""
        g = g or self._graph()
        d = [n for n in g["nodes"] if n["id"] == oid][0]
        return self._crossing_at(np.array(d["bounding_box"]["center"], float), g)

    def _crossing_at(self, c, g=None):
        """_door_crossing for any doorway POINT -- a door node's centre or an open passage's."""
        g = g or self._graph()
        c = np.array(c, float)
        p, _ = self._char(g)
        rooms = [(n["id"], n["bounding_box"]) for n in g["nodes"] if n.get("category") == "Rooms"]
        def rooms_at(q):
            return [rid for rid, bb in rooms
                    if abs(q[0] - bb["center"][0]) <= bb["size"][0] / 2
                    and abs(q[2] - bb["center"][2]) <= bb["size"][2] / 2]
        here = self._room_of(p[0], p[2], rooms)
        best = None
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            for sgn in (1.0, -1.0):
                v = axis * sgn
                ahead = [r for r in rooms_at(c + v * 1.0) if r != here]
                behind = rooms_at(c - v * 1.0)
                if not ahead:
                    continue
                score = (2 if here in behind else 0) + (1 if float(np.dot(c - p, v)) > 0 else 0)
                if best is None or score > best[0]:
                    best = (score, v, ahead[0])
        if best is None:
            return None
        return c, best[1], here, best[2]

    def _char_room_id(self, g):
        """Room id the character is INSIDE per the scene graph edges (None if absent)."""
        rooms = {n["id"] for n in g["nodes"] if n.get("category") == "Rooms"}
        ch = [n["id"] for n in g["nodes"] if n["class_name"] == "character"]
        if not ch:
            return None
        for e in g["edges"]:
            if e["from_id"] == ch[0] and e["relation_type"] == "INSIDE" and e["to_id"] in rooms:
                return e["to_id"]
        return None

    # OPEN PASSAGES ARE DOORWAYS (A32, project decision 2026-09-15). The world hands the controller the
    # scene's walk-verified passages; each is {"room_ids": (a, b), "center": [x, z], "points": [...]}.
    _passages = ()

    def _visible_portals(self, g, ignore_frame=False):
        """Doors and open passages the body can go through FROM HERE, nearest first:
        [(dist, "door"|"passage", door_id|None, centre xyz, room ids)].

        A portal counts only if it is on the body's OWN room boundary (a door two rooms away is
        not a way out of this room), within 0.3-8 m, inside the camera's horizontal field of view
        (+-VIS_HALF_FOV, the picture's own), and not hidden behind furniture. Walls are not tested: the sight line to a
        portal on your own room's boundary stays inside that (rectangular) room, and VH's 2.5 m wall
        cubes near a corner would otherwise hide a doorway that is plainly in frame (A22)."""
        p, yaw = self._char(g)
        rooms = [(n["id"], n["bounding_box"]) for n in g["nodes"] if n.get("category") == "Rooms"]
        here = self._char_room_id(g)
        if here is None and rooms:
            here = self._room_of(p[0], p[2], rooms)
        def rooms_at_door(c, pad=0.6):
            # A DOOR BELONGS TO BOTH ROOMS IT JOINS, decided by GEOMETRY. The scene graph puts a door node
            # INSIDE only one of them (7 of the 14 doors over the five apartments), so from the other side
            # an edge-based test never counted it as "this room's door": facing it was refused, or the
            # pass took a different door in view (pilot probe, f88 draft).
            return {rid for rid, bb in rooms
                    if abs(c[0] - bb["center"][0]) <= bb["size"][0] / 2 + pad
                    and abs(c[2] - bb["center"][2]) <= bb["size"][2] / 2 + pad}
        fwd = np.array([math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])
        try:
            _P, _V, cam = self._cam_matrices()
        except Exception:
            _P = _V = None
            cam = np.array([p[0], 1.6, p[2]], float)
        occ = [self._see_over_box(n["class_name"], np.array(n["bounding_box"]["center"], float),
                                  np.array(n["bounding_box"]["size"], float)) + (n["id"],)
               for n in g["nodes"]
               if n.get("category") not in ("Rooms", "Floor", "Ceiling", "Walls", "Doors", "Windows")
               and n["class_name"] not in ("character", "door", "doorjamb", "window", "wall")
               and max(n["bounding_box"]["size"]) > 0.5]

        def off_centre(c):
            d = np.array([c[0] - p[0], c[2] - p[2]]); L = float(np.linalg.norm(d))
            return 0.0 if L <= 0.3 else math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d / L, fwd))))))

        def occ_near(c):
            # the doorway's own frame and whatever stands right beside it never hide the doorway itself
            return [(mc, ms, mid) for mc, ms, mid in occ if float(np.hypot(mc[0] - c[0], mc[2] - c[2])) >= 1.0]

        def in_frame(q):
            clip = _P @ (_V @ np.array([q[0], q[1], q[2], 1.0]))
            return clip[3] > 0.05 and abs(clip[0] / clip[3]) <= 1.0 and abs(clip[1] / clip[3]) <= 1.0

        def in_view(c):
            d = np.array([c[0] - p[0], c[2] - p[2]]); L = float(np.linalg.norm(d))
            if L >= 8.0:
                return None
            if L <= 0.3:
                return L                                   # standing in the doorway counts, whichever way you face
            # A DOORWAY IS AN OPENING, NOT A POINT. Sample it across its width (perpendicular to the line of
            # sight) at three heights. Each sample is a POINT with a point-sized target for _occluded, or its
            # "an occluder must be bigger in every dimension" rule lets a 0.4 m-deep bookshelf hide nothing.
            side = np.array([-d[1], 0.0, d[0]]) / max(L, 1e-6)
            samples = [np.array([c[0], h, c[2]], float) + side * off
                       for h in (0.6, 1.2, 1.8) for off in (-0.4, 0.0, 0.4)]
            # IN THE PICTURE BY PROJECTION (f97). The old test was a flat +-VIS_HALF_FOV cone on the doorway
            # CENTRE from the body heading. A doorway is ~1 m wide and the camera looks 10 deg down, so one whose
            # centre sat 48-57 deg off-centre still showed its near edge in the frame: Gemini, shown the exact
            # frames, saw 29 of 40 such "refused" doorways (72%, vs 45% for doorways far outside), and 6 of 8
            # checked by eye were plainly there. Now a doorway is in view iff some sample PROJECTS INTO the
            # frame through the same camera matrices the object gate uses. Without matrices (offline) the
            # centre cone remains the fallback.
            if _P is None or _V is None:
                if float(np.dot(d / L, fwd)) < math.cos(math.radians(VIS_HALF_FOV)):
                    return None
                visible_samples = samples
            else:
                visible_samples = samples if ignore_frame else [q for q in samples if in_frame(q)]   # ignore_frame: probes only
                if not visible_samples:
                    return None
            if L <= 1.0:
                return L
            nctr = np.array([c[0], 1.0, c[2]], float); nsz = np.array([0.1, 0.1, 0.1], float)
            blockers = occ_near(c)
            for q in visible_samples:
                if not self._occluded(cam, q, nctr, nsz, -1, blockers):
                    return L
            return None

        out = []
        for n in g["nodes"]:
            if n["class_name"] != "door":
                continue
            c = n["bounding_box"]["center"]
            rooms_of = rooms_at_door(c)
            if here is None or here not in rooms_of or len(rooms_of) < 2:
                continue
            L = in_view(c)
            if L is not None:
                out.append((L, "door", n["id"], np.array(c, float), rooms_of))
        for ps in self._passages:
            ids = set(ps["room_ids"])
            if here not in ids:
                continue
            c = np.array([ps["center"][0], 1.0, ps["center"][1]], float)
            L = in_view(c)
            if L is not None:
                out.append((L, "passage", None, c, ids))
        # "the visible door" is the one the body FACES: most centred first, nearer breaking a tie
        # (nearest-first sent a body facing the bathroom door through the bedroom door at the frame edge)
        return sorted(out, key=lambda t: (round(off_centre(t[3]) / 10.0), t[0]))

    def _portal_target(self, g, kind, oid, pc, ids):
        """(target_room_id, unit crossing vector along the boundary normal) for a portal seen from here.

        The target comes from the PORTAL'S OWN ROOMS, never from probing which box a point beyond the
        doorway falls in: scene 5's bathroom box overlaps its neighbours, so from the bedroom, door#181
        (bedroom<->kitchen) resolved to the BATHROOM and the body was walked there (probe, f88 draft)."""
        rooms = {n["id"]: n["bounding_box"] for n in g["nodes"] if n.get("category") == "Rooms"}
        p, _ = self._char(g)
        here = self._char_room_id(g)
        if here is None:
            here = self._room_of(p[0], p[2], list(rooms.items()))
        others = [r for r in ids if r != here and r in rooms]
        if not others:
            return None
        hb = rooms[here]
        dx, dz = pc[0] - hb["center"][0], pc[2] - hb["center"][2]
        # the boundary the portal sits on is the one it is nearest to, relative to the room's half-size
        if abs(dx) / max(hb["size"][0] / 2, 1e-6) >= abs(dz) / max(hb["size"][2] / 2, 1e-6):
            v = np.array([1.0 if dx > 0 else -1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 0.0, 1.0 if dz > 0 else -1.0])
        if len(others) == 1:
            return others[0], v
        ahead = pc + v * 1.0
        inside = [r for r in others if abs(ahead[0] - rooms[r]["center"][0]) <= rooms[r]["size"][0] / 2
                  and abs(ahead[2] - rooms[r]["center"][2]) <= rooms[r]["size"][2] / 2]
        return (inside or others)[0], v

    def _pass_through_portal(self, portal, g):
        """Carry the body through ONE portal -- the one the person faced -- or report that it could not.
        Walks (engine pathfinding) to a landmark just beyond the doorway in the target room: up to three,
        ranked by how straight through and how near they are, over a window wide enough for real rooms
        (the old 0.8-3.5 m x 2.0 m window admitted NO landmark at 4 of 6 failing doorways). Success only
        when the body is in the target room and past the doorway plane. Never tries another portal:
        falling through to the next one sent a person facing the bathroom door into the kitchen."""
        _d, kind, oid, pc, ids = portal
        tg = self._portal_target(g, kind, oid, pc, ids)
        if tg is None:
            return None
        tgt_room, v = tg
        roomcls = {n["id"]: n["class_name"] for n in g["nodes"] if n.get("category") == "Rooms"}
        tb = next(n["bounding_box"] for n in g["nodes"] if n["id"] == tgt_room)
        inroom = {e["from_id"]: e["to_id"] for e in g["edges"]
                  if e["relation_type"] == "INSIDE" and e["to_id"] in roomcls}
        cands = []
        for n in g["nodes"]:
            if inroom.get(n["id"]) != tgt_room or n.get("category") in ("Rooms", "Walls", "Floor", "Ceiling", "Doors", "Windows"):
                continue
            if n["class_name"] in self._STRUCT or n["class_name"] == "character":
                continue
            sz = n["bounding_box"]["size"]
            if max(sz) < 0.2 or max(sz) > 2.5:
                continue
            q = np.array(n["bounding_box"]["center"], float)
            depth = float(np.dot(q - pc, v)); lateral = float(np.linalg.norm(((q - pc) - v * depth)[[0, 2]]))
            if 0.5 <= depth <= 4.5 and lateral <= 2.5:
                cands.append((abs(depth - 1.5) + 1.5 * lateral, n["id"], n["class_name"]))
        name = "door" if kind == "door" else "open passage"
        for _score, lid, lcls in sorted(cands)[:3]:
            try:
                self._do(f"[Walk] <{lcls}> ({lid})", find_solution=True)
            except Exception:
                continue
            try:
                g1 = self._graph(); p1, _ = self._char(g1)
            except Exception:
                continue
            in_box = (abs(p1[0] - tb["center"][0]) <= tb["size"][0] / 2 and abs(p1[2] - tb["center"][2]) <= tb["size"][2] / 2)
            if self._char_room_id(g1) == tgt_room and in_box and float(np.dot(p1 - pc, v)) >= 0.3:
                return Result("go_through_nearest_door", f"door#{oid}" if kind == "door" else "passage", True,
                              f"passed through the {name} into the {roomcls.get(tgt_room, 'next room')}",
                              **self.observation())
        return None

    def _cross_by_pathfinding(self, portals=None):
        """Walk through the nearest VISIBLE door that connects this room to another, by pathfinding
        to the nearest object that lies 0.8-3.5 m beyond the door plane inside the room on the far
        side. Returns a Result on success or on a definite refusal, or None to fall back to the
        stepping pass (no connecting door visible / no landmark beyond it)."""
        try:
            g = self._graph()
        except Exception:
            return None
        p, yaw = self._char(g)
        vis = self.visible_ids(g=g)
        roomcls = {n["id"]: n["class_name"] for n in g["nodes"] if n.get("category") == "Rooms"}
        here0 = self._char_room_id(g)
        inroom = {e["from_id"]: e["to_id"] for e in g["edges"]
                  if e["relation_type"] == "INSIDE" and e["to_id"] in roomcls}
        # Which doors count is decided in ONE place, _visible_portals (A32).
        if portals is None:
            portals = self._visible_portals(g)
        if not portals:
            return None
        for _dist, kind, oid, pc, _ids in portals[:3]:
            try:
                xing = self._door_crossing(oid, g) if kind == "door" else self._crossing_at(pc, g)
            except Exception:
                xing = None
            if xing is None:
                continue                                  # a door that leads to no room
            c, v, _here, tgt_room = xing
            if here0 is not None and tgt_room == here0:
                continue
            best = None
            for n in g["nodes"]:
                if inroom.get(n["id"]) != tgt_room or n.get("category") in ("Rooms", "Walls", "Floor", "Ceiling", "Doors", "Windows"):
                    continue
                if n["class_name"] in self._STRUCT or n["class_name"] == "character":
                    continue
                sz = n["bounding_box"]["size"]
                if max(sz) < 0.2 or max(sz) > 2.5:
                    continue
                q = np.array(n["bounding_box"]["center"], float)
                depth = float(np.dot(q - c, v)); lateral = float(np.linalg.norm((q - c) - v * depth))
                if 0.8 <= depth <= 3.5 and lateral <= 2.0:
                    score = depth + 1.5 * lateral
                    if best is None or score < best[0]:
                        best = (score, n["id"], n["class_name"])
            if best is None:
                continue
            _s, lid, lcls = best
            try:
                r = self._do(f"[Walk] <{lcls}> ({lid})", find_solution=True)
            except Exception:
                r = None
            try:
                g1 = self._graph()
            except Exception:
                return None
            if self._char_room_id(g1) == tgt_room:
                return Result("go_through_nearest_door", f"door#{oid}" if kind == "door" else "passage", True,
                              f"passed through the {'door' if kind == 'door' else 'open passage'} into the "
                              f"{roomcls.get(tgt_room, 'next room')}", **self.observation())
            return None                                     # did not arrive: let the stepping pass try
        return None

    def _square_to_door(self, oid):
        """Face straight through the doorway along its crossing axis (see _door_crossing)."""
        try:
            x = self._door_crossing(oid)
            if x is None:
                return -1
            c, v, _here, _tgt = x
            tgt = c + v * 1.5
            return self._face_heading(float(tgt[0]), float(tgt[2]))
        except Exception:
            return -1

    def turn_left(self, n=1):
        for _ in range(int(n)):
            self._do("[TurnLeft]")
        return Result("turn_left", f"{int(n)*TURN_DEG:.0f}deg", True, "turned", **self.observation())

    def turn_right(self, n=1):
        for _ in range(int(n)):
            self._do("[TurnRight]")
        return Result("turn_right", f"{int(n)*TURN_DEG:.0f}deg", True, "turned", **self.observation())

    def turn_to_nearest(self, cls):
        nv = self._nearest_visible(cls)
        if not nv:
            return Result("turn_to_nearest", cls, False, f"no visible {cls} to turn to — scan with turn_left/right",
                          **self.observation())
        d, oid, c = nv
        ok, m = self._do(f"[TurnTo] <{c}> ({oid})", find_solution=True)
        return Result("turn_to_nearest", f"{c}#{oid}", ok, f"faced the {c}" if ok else m,
                      **self.observation())

    def go_to_nearest(self, cls):
        nv = self._nearest_visible(cls)
        if not nv:
            return Result("go_to_nearest", cls, False, f"no visible {cls} — turn to find one first",
                          **self.observation())
        d0, oid, c = nv
        self._do(f"[TurnTo] <{c}> ({oid})")
        steps = 0
        d = d0
        while steps < MAX_APPROACH_STEPS:
            g = self._graph()
            p, _ = self._char(g)
            tgt = [n for n in g["nodes"] if n["id"] == oid][0]
            d = float(np.linalg.norm((np.array(tgt["obj_transform"]["position"], float) - p)[[0, 2]]))
            if d <= ARRIVE_M:
                break
            p0, _ = self._char(g)
            ok, m = self._do(f"[WalkTowards] <{c}> ({oid})")
            steps += 1
            p1, _ = self._char()
            if float(np.linalg.norm((p1 - p0)[[0, 2]])) < 0.05:   # didn't move -> blocked
                return Result("go_to_nearest", f"{c}#{oid}", False,
                              f"can't get closer to the {c} — something is blocking the way. "
                              f"Turn left or right and try 'go to nearest {cls}' from another angle.",
                              steps=steps, **self.observation())
        if d <= NEAR_M:
            return Result("go_to_nearest", f"{c}#{oid}", True,
                          f"arrived at the {c}.", steps=steps, **self.observation())
        return Result("go_to_nearest", f"{c}#{oid}", False,
                      f"not close enough to the {c} yet. Issue 'go to nearest {cls}' again to keep approaching.",
                      steps=steps, **self.observation())

    def go_straight(self, max_steps=MAX_STRAIGHT_STEPS):
        moved = 0
        for _ in range(max_steps):
            p0, _ = self._char()
            ok, m = self._do("[WalkForward]")
            p1, _ = self._char()
            if float(np.linalg.norm((p1 - p0)[[0, 2]])) < 0.05:
                msg = (f"no space to move forward — a wall or object is right in front of you. Turn left or "
                       f"right to face an open direction, then move." if moved == 0
                       else f"walked forward {moved}m, then a wall/object blocked the way — turn to go around it.")
                return Result("go_straight", None, True, msg, steps=moved, **self.observation())
            moved += 1
        return Result("go_straight", None, True, f"walked straight {moved}m (open space ahead).",
                      steps=moved, **self.observation())

    def go_through_nearest_door(self):
        """Pass through the nearest visible door. Two hard lessons are baked in:

        (1) step until the ROOM CHANGES, never a blind fixed count — the fixed 2×[WalkForward]
            could carry the body past the doorway into the far wall at an angle;
        (2) after the pass, PROBE that the body can still rotate. A pass that ends wedged in
            the door frame is a soft-lock the rest of the stack cannot fix (turns no-op, walks
            and even engine pathfinding refuse — observed live for 60 straight ticks). At the
            threshold pathfinding still works, so back out to the door and report failure
            honestly instead of stranding the body."""
        # v4 (A22): a pass through a CONNECTING door, carried by the engine's pathfinder to the
        # nearest thing just beyond it. The honest probe (door_probe_world.py) measured the
        # threshold-shove passes at 7-8 real crossings in 40, whatever the squaring: the nearest
        # "door" was often a door leading nowhere, a door the body already stood in could not be
        # approached, and a forward shove from the frame did not move the body.
        # A doorway counts if it was IN VIEW WHEN THE COMMAND WAS ISSUED (A32). Re-checking after the
        # pathfinding stage walked the body up to the threshold refused a door that was 1 m ahead and
        # 23 degrees off centre at command time: from the doorway itself it no longer "looked" visible.
        g_cmd = self._graph()
        portals = self._visible_portals(g_cmd)
        if portals:
            crossed_by_walk = self._pass_through_portal(portals[0], g_cmd)   # the doorway the person faces, only
            if crossed_by_walk is not None:
                return crossed_by_walk
        if not portals:
            return Result("go_through_nearest_door", "door", False,
                          "no door or open passage in view — turn to find one", **self.observation())
        d, kind, oid, pc, _ids = portals[0]
        c = "door"
        if kind == "passage":
            return self._step_through_passage(pc)
        # approach by PATHFINDING, not straight-line increments: [WalkTowards] jammed against
        # door frames on diagonal approaches and the pass then honestly-but-uselessly refused
        # ("could not get through") from the same bad angle forever. [Walk] positions the body
        # squarely at the door; the straight-line loop stays as fallback.
        ok_walk = False
        try:
            r = self._do(f"[Walk] <{c}> ({oid})", find_solution=True)
            ok_walk = bool(r) and (r[0] if isinstance(r, tuple) else True)
        except Exception:
            ok_walk = False
        if not ok_walk:
            self._do(f"[TurnTo] <{c}> ({oid})")
            for _ in range(MAX_APPROACH_STEPS):
                self._do(f"[WalkTowards] <{c}> ({oid})")
                try:
                    g = self._graph(); p, _ = self._char(g)
                    tgt = [n for n in g["nodes"] if n["id"] == oid][0]
                except Exception:
                    continue
                if float(np.linalg.norm((np.array(tgt["obj_transform"]["position"], float) - p)[[0, 2]])) <= 1.0:
                    break
            self._do(f"[TurnTo] <{c}> ({oid})")
        # D8: square up to the doorway before shoving forward. Three refusals in a row from a
        # slightly bad angle was the commonest stuck pattern in the 2026-09-05 pilot.
        self._square_to_door(oid)
        g0 = self._graph()
        rooms0 = [(n["id"], n["bounding_box"]) for n in g0["nodes"] if n.get("category") == "Rooms"]
        p0, _ = self._char(g0)
        start_room = self._room_of(p0[0], p0[2], rooms0)
        try:
            xing = self._door_crossing(oid, g0)
        except Exception:
            xing = None
        crossed = False
        resquared = False
        p_prev = p0
        for _ in range(6):
            self._do("[WalkForward]")
            try:                     # a transient 408 on the graph read must cost one step,
                p1, _ = self._char()  # not the whole episode (KeyError:'nodes' killed 16 runs)
            except Exception:
                continue
            if xing is not None:
                # A REAL crossing (A20): 0.6 m past the door plane along the crossing axis and
                # inside the target room. Room boxes overlap at a doorway, so "the room lookup
                # changed" alone was satisfied by a half step on the threshold -- the world then
                # measured < 0.5 m and refused the pass as a no-op (122 times in pilot 3).
                dc, v, _h, tgt_room = xing
                depth = float(np.dot(p1 - dc, v))
                bb = next((b_ for r_, b_ in rooms0 if r_ == tgt_room), None)
                in_tgt = bb is not None and (abs(p1[0] - bb["center"][0]) <= bb["size"][0] / 2
                                             and abs(p1[2] - bb["center"][2]) <= bb["size"][2] / 2)
                if depth >= 0.6 and in_tgt:
                    crossed = True
                    break
                if float(np.linalg.norm((p1 - p_prev)[[0, 2]])) < 0.1 and not resquared:
                    self._square_to_door(oid); resquared = True      # a step that went nowhere
                p_prev = p1
            elif self._room_of(p1[0], p1[2], rooms0) != start_room:
                crossed = True
                break
        if not crossed:
            # BACK OUT before giving up (fix #49). The failed forward shoves leave the body
            # pressed into the door frame — the exact half-wedged pose the crossing path's
            # probe exists to prevent. One run was abandoned here at t60 and by t83 was in
            # the incurable pathfind-dead pocket. Walk back to the door while pathfinding
            # still works; only then refuse.
            try:
                self._do(f"[Walk] <{c}> ({oid})", find_solution=True)
            except Exception:
                pass
            return Result("go_through_nearest_door", f"{c}#{oid}", False,
                          "could not get through — the doorway seems blocked from this angle; "
                          "I stepped back to the door. Try approaching it more squarely",
                          **self.observation())
        # wedge probe: a body that cannot rotate here is soft-locked — back out NOW.
        # Guarded: a failed graph read must never turn a good pass into a back-out.
        try:
            _, y0 = self._char()
            self._do("[TurnRight]")
            _, y1 = self._char()
        except Exception:
            return Result("go_through_nearest_door", f"{c}#{oid}", True,
                          "passed through the doorway", **self.observation())
        if abs((y1 - y0 + 180.0) % 360.0 - 180.0) < 10.0:
            self._do(f"[Walk] <{c}> ({oid})", find_solution=True)
            return Result("go_through_nearest_door", f"{c}#{oid}", False,
                          "the doorway was too tight to pass cleanly — I got briefly stuck and "
                          "stepped back to the door; try again, or take another route",
                          **self.observation())
        self._do("[TurnLeft]")           # undo the probe turn
        return Result("go_through_nearest_door", f"{c}#{oid}", True, "passed through the doorway",
                      **self.observation())

    def _step_through_passage(self, pc):
        """Stepping pass for an OPEN passage (no door node to [Walk] to): face through it along its
        crossing axis and step forward until the body is 0.6 m into the room on the far side."""
        g0 = self._graph()
        rooms0 = [(n["id"], n["bounding_box"]) for n in g0["nodes"] if n.get("category") == "Rooms"]
        try:
            xing = self._crossing_at(pc, g0)
        except Exception:
            xing = None
        if xing is None:
            return Result("go_through_nearest_door", "passage", False,
                          "that open passage does not lead anywhere from here", **self.observation())
        dc, v, _h, tgt_room = xing
        p0, _ = self._char(g0)
        self._face_heading(float(dc[0]), float(dc[2]))        # walk up to the passage
        for _ in range(8):
            p1, _ = self._char()
            if float(np.linalg.norm((p1 - dc)[[0, 2]])) <= 0.8:
                break
            self._do("[WalkForward]")
        tgt = dc + v * 1.5
        self._face_heading(float(tgt[0]), float(tgt[2]))      # square to the crossing axis
        for _ in range(6):
            self._do("[WalkForward]")
            p1, _ = self._char()
            bb = next((b_ for r_, b_ in rooms0 if r_ == tgt_room), None)
            in_tgt = bb is not None and (abs(p1[0] - bb["center"][0]) <= bb["size"][0] / 2
                                         and abs(p1[2] - bb["center"][2]) <= bb["size"][2] / 2)
            if float(np.dot(p1 - dc, v)) >= 0.6 and in_tgt:
                return Result("go_through_nearest_door", "passage", True,
                              "passed through the open passage", **self.observation())
        return Result("go_through_nearest_door", "passage", False,
                      "could not get through the open passage from this angle — turn to face it squarely "
                      "and try again", **self.observation())

    def can_manipulate(self, cls, g=None):
        """Side-effect-free probe: would a manipulation on the nearest <cls> pass the visible+near
        gate right now? Returns (allowed: bool, reason, dist|None). Lets a caller check before acting."""
        act = self._nearest_actionable(cls, g=g)
        if not act:
            return False, f"no perceived {cls} within {NEAR_M:.1f}m", None
        d, oid, c, _ = act
        return True, f"nearest {c} actionable at {d:.1f}m", d

    def _held_of(self, cls, g=None):
        """id of a HELD instance of class @cls (substring match), else None."""
        g = g or self._graph()
        ch = [n for n in g["nodes"] if n["class_name"] == "character"][0]["id"]
        bid = {n["id"]: n for n in g["nodes"]}
        q = cls.lower().replace(" ", "")
        for e in g["edges"]:
            if e["from_id"] == ch and e["relation_type"].startswith("HOLDS"):
                if q in bid.get(e["to_id"], {}).get("class_name", "").lower():
                    return e["to_id"], bid[e["to_id"]]["class_name"]
        return None

    def _rel_bearing(self, node, p, yaw):
        """Signed angle (deg) of @node relative to the agent's facing — 0 = dead ahead."""
        d = np.array(node["obj_transform"]["position"], float) - p
        return ((math.degrees(math.atan2(d[0], d[2])) - yaw + 180) % 360) - 180

    def _nearest_actionable(self, cls, g=None, face=False):
        """Nearest instance of @cls the agent may manipulate: within NEAR_M (the anti-cheat proximity bound;
        the manip's own [Walk]/find_solution then faces it). @face=True measures distance to the object's
        nearest BOUNDING-BOX FACE instead of its centre — needed for large furniture (a 2.4 m sofa/bed):
        standing at its edge you're touching it but >NEAR_M from the centre. Used for 'sit on', which you can
        do by proximity alone (back into the seat) without the object being centred in view. Returns
        (dist, id, class, True)|None."""
        g = g or self._graph()
        p, yaw = self._char(g)
        q = canon(cls)
        best = None
        for n in g["nodes"]:
            if n.get("category") == "Rooms" or n["class_name"] == "character":
                continue
            cl = n["class_name"].lower()
            if not (cl == q or cl.endswith(q)):
                continue
            if face:
                bb = n["bounding_box"]; c = np.array(bb["center"], float); s = np.array(bb["size"], float)
                np_ = np.clip(p, c - s / 2, c + s / 2)          # nearest point on the object's AABB
                d = float(np.linalg.norm((p - np_)[[0, 2]]))     # ground-plane distance to that face
            else:
                d = float(np.linalg.norm((np.array(n["obj_transform"]["position"], float) - p)[[0, 2]]))
            if d <= NEAR_M:
                if best is None or d < best[0]:
                    best = (d, n["id"], n["class_name"], True)
        return best

    # actions whose SOURCE object must be in the agent's hand (not a world object)
    _HELD_SRC = {"Put", "PutIn", "Drink"}

    def _manip(self, action_name, vh_action, cls, second=None):
        # ANTI-CHEAT GATE: a world object must be currently VISIBLE and within NEAR_M (1m) before any
        # manipulation runs. This stops the agent from manipulating something it never looked at, or
        # from acting from across the room and letting the action's [Walk] take a long walk to reach
        # it. To satisfy the gate the agent must first turn to see the object and go_to_nearest it.
        g = self._graph()
        vis = self.visible_ids()
        verb = action_name.replace("_nearest", "").replace("_", " ")   # 'pick up', 'open', 'drink', ...
        if verb == "sit": verb = "sit on"
        if vh_action in self._HELD_SRC:
            # Put / PutIn / Drink act on the object IN HAND — exempt from the visible+near gate.
            h = self._held_of(cls, g)
            if h is None:
                return Result(action_name, cls, False, f"you are not holding a {cls} — pick one up first",
                              **self.observation(g))
            oid, c = h
            d = 0.0
        else:
            # 'sit on' is proximity-only via nearest bbox FACE (you back into a seat you're beside); other
            # manipulations use centre distance (you go up to and face them).
            act = self._nearest_actionable(cls, g, face=(vh_action == "Sit"))
            if not act:
                # not actionable — explain why using ONLY what the agent can currently SEE (never reference an
                # off-camera / other-room instance: that would leak the map and suggest a 'go to' that would
                # itself be refused as not-visible). A VISIBLE instance beyond reach -> walk to it; nothing
                # visible -> scan.
                seen = self._nearest_visible(cls, g=g, require_visible=True)
                if seen:
                    return Result(action_name, f"{cls}", False,
                                  f"you're too far from the {cls} to {verb} it. "
                                  f"'go to nearest {cls}' to walk up to it first.", **self.observation(g))
                return Result(action_name, cls, False,
                              f"no {cls} is in view — turn left or right to scan the room until a {cls} "
                              f"appears, then 'go to nearest {cls}' and try '{verb}' again.",
                              **self.observation(g))
            d, oid, c, _ = act     # within NEAR_M and perceived — gate satisfied
        # Emit the action as a combined [Walk <anchor>][<action>] script: the leading high-level Walk
        # establishes VirtualHome's CLOSE/facing relation (and fully transfers a grabbed object off
        # its surface) which the manipulation requires to commit. This is bounded because the gate
        # above already ensured the object is visible+within reach, so Walk only does the final step.
        # Grab/Put/Sit need a leading high-level [Walk] to establish VH's CLOSE relation and to fully
        # transfer a grabbed object off its surface. Open/Close/Switch/Drink commit reliably as a
        # single action once the agent is near (find_solution handles the final cm).
        needs_walk = vh_action in ("Grab", "Put", "PutIn", "Sit")
        if second is not None:  # put X on/in nearest Y — destination must ALSO be perceived + near
            act2 = self._nearest_actionable(second, g)
            if not act2:
                return Result(action_name, f"{c}#{oid}", False,
                              f"you can't place it yet — get next to the {second} ('go to nearest {second}') "
                              f"so you're looking at it, then put the {cls} {'in' if vh_action=='PutIn' else 'on'} it.",
                              **self.observation(g))
            d2, oid2, c2, _ = act2     # destination perceived + within reach
            script = [f"[Walk] <{c2}> ({oid2})", f"[{vh_action}] <{c}> ({oid}) <{c2}> ({oid2})"]
            tgt = f"{c}#{oid}->{c2}#{oid2}"
            ok, m = self._do_multi(script)
        elif needs_walk:
            ok, m = self._do_multi([f"[Walk] <{c}> ({oid})", f"[{vh_action}] <{c}> ({oid})"])
            tgt = f"{c}#{oid}"
        else:  # Open / Close / SwitchOn / SwitchOff / Drink
            ok, m = self._do(f"[{vh_action}] <{c}> ({oid})", find_solution=True)
            tgt = f"{c}#{oid}"
        return Result(action_name, tgt, ok, f"{verb} done" if ok else m, **self.observation())

    def _do_multi(self, lines, find_solution=False):
        script = [f"<char0> {l}" for l in lines]
        return self._render(script, find_solution)

    def pick_up_nearest(self, cls):      return self._manip("pick_up_nearest", "Grab", cls)
    def open_nearest(self, cls):         return self._manip("open_nearest", "Open", cls)
    def close_nearest(self, cls):        return self._manip("close_nearest", "Close", cls)
    def switch_on_nearest(self, cls):    return self._manip("switch_on_nearest", "SwitchOn", cls)
    def switch_off_nearest(self, cls):   return self._manip("switch_off_nearest", "SwitchOff", cls)
    def drink_nearest(self, cls):        return self._manip("drink_nearest", "Drink", cls)
    def sit_nearest(self, cls):          return self._manip("sit_nearest", "Sit", cls)
    def read_nearest(self, cls):         return self._manip("read_nearest", "Read", cls)
    def watch_nearest(self, cls):        return self._manip("watch_nearest", "Watch", cls)
    def stand_up(self):
        ok, m = self._do("[StandUp]")
        return Result("stand_up", None, ok, "stood up" if ok else m, **self.observation())
    def put_on_nearest(self, cls, target): return self._manip("put_on_nearest", "Put", cls, second=target)
    def put_in_nearest(self, cls, target): return self._manip("put_in_nearest", "PutIn", cls, second=target)

    # ---------- natural-language interface ----------
    # NL commands map deterministically onto the programmatic API above.
    def execute(self, command):
        s = command.strip().lower().rstrip(".")
        m = re.match(r"turn (left|right)(?:\s+(\d+))?$", s)
        if m:
            n = int(m.group(2)) if m.group(2) else 1
            return self.turn_left(n) if m.group(1) == "left" else self.turn_right(n)
        if REMOVED_TURN_RE.match(s):   # before "turn to <object>", which would swallow a clock phrase
            return Result("turn", s, False, "this body only turns with 'turn left' or 'turn right', optionally with a count of 30-degree steps — 'turn right 3' is three 30-degree turns, 'turn left 6' ends facing the other way", **self.observation())
        if s in ("stand up", "stand", "get up", "stand up from chair", "stand up from sofa"):
            return self.stand_up()
        m = re.match(r"look (down|up)(?:\s+(\d+))?$", s)
        if m:
            n = int(m.group(2)) if m.group(2) else 1
            return self.look_down(n) if m.group(1) == "down" else self.look_up(n)
        if s in ("tilt down", "look at the floor", "look at floor"): return self.look_down()
        if s in ("tilt up", "look ahead", "look forward"): return self.look_up()
        if re.match(r"(go|walk) (straight|forward)", s):
            return self.go_straight()
        if re.match(r"^(?:go|walk) through (?:the )?visible (?:door|doorway)$", s):
            return self.go_through_nearest_door()
        if re.match(r"^(?:go|walk) through (?:the )?(?:nearest |closest )?(?:door|doorway)$", s):
            return Result("go_through_nearest_door", "door", False, "the door command is 'go through the visible door' — it passes through a doorway or open passage you can see right now", **self.observation())
        N = r"(?:the )?(?:nearest |closest )?"   # 'nearest' is optional (reference is always nearest)
        m = re.match(rf"turn to {N}(.+)$", s)
        if m:
            return self.turn_to_nearest(m.group(1))
        m = re.match(rf"(?:go|walk) to {N}(.+)$", s)
        if m:
            return self.go_to_nearest(m.group(1))
        m = re.match(rf"put {N}(.+?) (on|in|into|inside|onto) {N}(.+)$", s)
        if m:
            fn = self.put_in_nearest if m.group(2) in ("in", "into", "inside") else self.put_on_nearest
            return fn(m.group(1), m.group(3))
        verbs = [(r"pick up|pickup|grab|take", self.pick_up_nearest), (r"open", self.open_nearest),
                 (r"close|shut", self.close_nearest), (r"switch on|turn on|toggle on", self.switch_on_nearest),
                 (r"switch off|turn off|toggle off", self.switch_off_nearest), (r"drink", self.drink_nearest),
                 (r"sit on|sit in|sit", self.sit_nearest),
                 (r"read", self.read_nearest), (r"watch", self.watch_nearest)]
        for pat, fn in verbs:
            m = re.match(rf"(?:{pat}) {N}(.+)$", s)
            if m:
                return fn(m.group(1))
        return Result("unknown", None, False, f"could not parse command: '{command}'")

    # ---------- STEPPABLE / interruptible execution (for frame-by-frame streaming) ----------
    def begin(self, command):
        """Steppable form of execute(): returns a GENERATOR that performs @command ONE low-level primitive
        at a time (1 m walk / 30 deg turn / one manipulation), yielding a status dict AFTER each so a
        streaming caller can render a fresh egocentric frame and re-poll the model BETWEEN primitives.
        This makes a multi-step action (go to X / go through door / go straight) watchable frame-by-frame
        and interruptible: advance it with next() while the model says 'continue', or drop it and begin() a
        new command to take over mid-action. Instant actions (turn / turn-to / manipulation) yield once.
        Status dict: {phase: progress|done|blocked|refused, msg, success, dist?}. 'progress' means more to
        come; anything else is terminal (the next next() would StopIteration)."""
        self._set_pitch(self._NAV_PITCH)   # every new command starts from the nav view; go-to auto-tilts
        s = command.strip().lower().rstrip(".")
        if re.match(r"(go|walk) (straight|forward)", s):
            return self._gen_go_straight()
        if re.match(r"(?:go|walk) through (?:the )?visible (?:door|doorway)", s):
            return self._gen_approach("door", through=True)
        N = r"(?:the )?(?:nearest |closest )?"
        m = re.match(rf"(?:go|walk) to {N}(.+)$", s)
        if m:
            return self._gen_approach(canon(m.group(1)))
        return self._gen_instant(command)

    def _gen_instant(self, command):
        r = self.execute(command)
        yield {"phase": "done" if r["success"] else "refused", "msg": r["reason"], "success": r["success"],
               "verb": r["action"], "target": r.get("target")}   # RESOLVED action + "cls#id[->cls2#id2]"

    def _gen_go_straight(self):
        moved = 0
        for _ in range(MAX_STRAIGHT_STEPS):
            p0, _ = self._char()
            self._do("[WalkForward]")
            p1, _ = self._char()
            if float(np.linalg.norm((p1 - p0)[[0, 2]])) < 0.05:
                msg = ("no space to move forward — a wall/object is right in front; turn to an open direction."
                       if moved == 0 else f"walked {moved}m, then blocked — turn to go around it.")
                yield {"phase": "blocked", "msg": msg, "success": True, "steps": moved}; return
            moved += 1
            yield {"phase": "progress", "msg": f"walked forward {moved}m", "success": True, "steps": moved}
        yield {"phase": "done", "msg": f"walked straight {moved}m (open space ahead).", "success": True, "steps": moved}

    def _gen_approach(self, cls, through=False):
        """Walk to the nearest visible <cls> 1 m at a time. through=True => a door: stop ~1 m short then
        step through the doorway. Yields once per primitive so the caller can interrupt each metre."""
        nv = self._nearest_visible(cls)
        if not nv:
            yield {"phase": "refused", "msg": f"no visible {cls} — turn to scan for one first", "success": False}; return
        _d0, oid, c = nv
        self._do(f"[TurnTo] <{c}> ({oid})")
        self._ensure_in_view(oid)          # centred from the very first frame of the approach
        yield {"phase": "progress", "msg": f"faced the {c}, heading over", "success": True}
        arrive = 1.0 if through else ARRIVE_M
        d = _d0
        for _ in range(MAX_APPROACH_STEPS):
            g = self._graph(); p, _ = self._char(g)
            tn = [n for n in g["nodes"] if n["id"] == oid]
            if not tn:
                break
            d = float(np.linalg.norm((np.array(tn[0]["obj_transform"]["position"], float) - p)[[0, 2]]))
            if d <= arrive:
                break
            p0, _ = self._char(g)
            self._do(f"[WalkTowards] <{c}> ({oid})")
            p1, _ = self._char()
            if float(np.linalg.norm((p1 - p0)[[0, 2]])) < 0.05:
                yield {"phase": "blocked", "msg": f"blocked before reaching the {c} — turn and approach from "
                       f"another angle", "success": False, "dist": d}; return
            # AUTO HEAD-TILT: pitch the head so the target stays in the camera frame as we close in
            self._ensure_in_view(oid)      # auto head-tilt keeps the target in frame; silent
            yield {"phase": "progress", "msg": f"walking toward the {c}", "success": True, "dist": d}
        if through:
            # walk forward until the ROOM actually changes (not a fixed 2 steps — that can strand the agent
            # in the doorway boundary zone where room detection, and thus the room map, doesn't flip).
            g0 = self._graph()
            rooms0 = [(n["id"], n["bounding_box"]) for n in g0["nodes"] if n.get("category") == "Rooms"]
            p0, _ = self._char(g0)
            start_room = self._room_of(p0[0], p0[2], rooms0)
            for _ in range(4):
                self._do("[WalkForward]")
                p1, _ = self._char()
                if self._room_of(p1[0], p1[2], rooms0) != start_room:
                    break
                yield {"phase": "progress", "msg": "stepping through the doorway", "success": True}
            yield {"phase": "done", "msg": "passed through the doorway into the next room", "success": True,
                   "verb": "go_through_nearest_door", "target": f"{c}#{oid}"}; return
        if d <= NEAR_M:
            self._ensure_in_view(oid)      # arrive with the target in frame for the VLM's next look
            yield {"phase": "done", "msg": f"arrived at the {c}", "success": True, "dist": d,
                   "verb": "go_to_nearest", "target": f"{c}#{oid}"}   # RESOLVED, for the 'reached' checkpoint
        else:
            yield {"phase": "blocked", "msg": f"not close enough to the {c} yet — 'continue' to keep approaching",
                   "success": False, "dist": d}


# ---------------------------------------------------------------- prompt-facing description
def midlevel_vocabulary():
    """The mid-level action space, rendered for BOTH agents' prompts.

    GENERATED from this module's own constants, never hand-listed. CookSim learned this the
    hard way (cooksim/agents/highlevel.py::grounding_vocabulary): a prompt that copies a
    parser's vocabulary is a copy waiting to go stale, and the failure is silent — an
    unrecognised phrasing is dropped and the nearest candidate is chosen, so the agent gets a
    confidently wrong object rather than an error. Both sides must be shown what the parser
    actually accepts, from the parser itself.
    """
    return f"""  MOVING / LOOKING (no target needs to be visible)
    "turn left" · "turn right"            — {TURN_DEG:.0f}deg each; add a count for several turns in a row:
                                            "turn right 3" = three turns (90deg), one look after each
    "go straight"                         — advance ~{STEP_M:.0f} m (max {MAX_STRAIGHT_STEPS} in one command)
    "look down" · "look up"               — pitch the view between {EGO_PITCHES} deg below level
    "go through the visible door"         — step through a doorway or open passage you can see now

  ACTING ON SOMETHING (the target MUST be visible first)
    "turn to <object>"                    — face it
    "go to <object>"                      — approach until ~{ARRIVE_M:.1f} m away
    "pick up <object>" · "open <object>" · "close <object>"
    "switch on <object>" · "switch off <object>" · "drink <object>" · "sit on <object>"
    "put <object> on <surface>" · "put <object> in <container>"
    "stand up"

  THE TWO RULES THAT DECIDE WHETHER A COMMAND WORKS
    VISIBILITY GATE — you can only act on what is in your FIRST-PERSON VIEW right now
      (about {VIS_HALF_FOV:.0f}deg either side of where you face). An object in another room, behind
      you, or out of frame does not exist as far as your body is concerned. There is NO
      command that walks you to something you cannot see: to reach it you must TURN until it
      comes into view, then "go to" it.
    DISTANCE GATE — manipulating needs the object visible AND within {NEAR_M:.1f} m. "go to" stops
      at about that distance, which is close enough to act and far enough to keep it in view.

  WHICH ONE YOU GET
    Every reference resolves to the NEAREST instance THAT YOU CAN SEE. Saying "nearest" is
    optional — "go to the fridge" and "go to the nearest fridge" mean the same thing. So WHERE
    YOU STAND AND WHICH WAY YOU FACE decides which object you act on."""
