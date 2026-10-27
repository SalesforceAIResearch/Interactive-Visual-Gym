"""Reusable top-down semantic map renderer. Given a floor-plan snapshot (rooms + object footprints + doors)
and the agent's trajectory so far, draw the map (room plan, object footprints colour-coded by category,
doorways, past path, current position + heading) and return PNG bytes. Used live by the streaming runner
(statemap condition) and offline by build_topdown.py."""
import io, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

CATCOL = {"Furniture": ("#2b4a6f", "#5b9bd5"), "Appliances": ("#6b3d1f", "#e08a3c"),
          "Electronics": ("#3d2b5f", "#a97fe0"), "Food": ("#2f5a34", "#5fd38a"),
          "Lamps": ("#5a521f", "#d9c84a"), "Decor": ("#33302a", "#6b6456")}


def render(fp, pos, yaw, size_px=520):
    """fp: {rooms:[{c,s,..}], objs:[{cat,c:[x,z],s:[sx,sz]}], doors:[[x,y,z]]}; pos: Nx2 [x,z]; yaw: N."""
    rooms, objs, doors = fp["rooms"], fp["objs"], fp["doors"]
    xs = [r["c"][0] - r["s"][0] / 2 for r in rooms] + [r["c"][0] + r["s"][0] / 2 for r in rooms]
    zs = [r["c"][2] - r["s"][2] / 2 for r in rooms] + [r["c"][2] + r["s"][2] / 2 for r in rooms]
    xlo, xhi, zlo, zhi = min(xs) - .4, max(xs) + .4, min(zs) - .4, max(zs) + .4
    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    for r in rooms:
        cx, cz, sx, sz = r["c"][0], r["c"][2], r["s"][0], r["s"][2]
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc="#161a22", ec="#39d98a", lw=1.4))
        ax.text(cx, cz - sz / 2 + .28, r["cls"].upper(), ha="center", va="center",
                color="#39d98a", fontsize=7, alpha=.5, fontweight="bold")
    for o in sorted(objs, key=lambda o: -(o["s"][0] * o["s"][1])):
        cx, cz = o["c"]; sx, sz = o["s"]; fill, edge = CATCOL.get(o.get("cat"), (None, "#7a8394"))
        if fill is None or (sx < .28 and sz < .28):
            ax.plot(cx, cz, ".", color=edge, ms=2.6, alpha=.7); continue
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc=fill, ec=edge, lw=1.0, alpha=.9))
    for dc in doors:
        ax.plot(dc[0], dc[2], "s", color="#ff9ecb", ms=8, mfc="none", mew=1.6)
    pos = np.asarray(pos, float)
    if len(pos):
        ax.plot(pos[:, 0], pos[:, 1], "-", color="#4aa3ff", lw=2.3, alpha=.95)
        ax.plot(pos[0, 0], pos[0, 1], "o", color="#7ee787", ms=6)
        x, z = pos[-1]; a = float(yaw[-1])
        ax.arrow(x, z, .85 * np.sin(np.radians(a)), .85 * np.cos(np.radians(a)),
                 head_width=.34, head_length=.32, fc="#ffd24d", ec="#0f1117", lw=1.2, zorder=8)
        ax.plot(x, z, "o", color="#ffd24d", ms=11, mec="#0f1117", mew=1.5, zorder=7)
    ax.set_xlim(xlo, xhi); ax.set_ylim(zlo, zhi); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    fig.tight_layout(pad=0.2)
    b = io.BytesIO(); fig.savefig(b, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return b.getvalue()


def _house_adjacency(fp, rooms):
    """Pairs of room labels joined by a door — the routing graph for the whole-house text."""
    adj = set()
    for dc in fp["doors"]:
        near = []
        for ddx, ddz in [(1, 0), (-1, 0), (0, 1), (0, -1), (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)]:
            near.append(_room_of(rooms, dc[0] + ddx * 0.9, dc[2] + ddz * 0.9)["id"])
        uniq = []
        for i in near:
            if i not in uniq:
                uniq.append(i)
        if len(uniq) >= 2:
            a = next(r for r in rooms if r["id"] == uniq[0])["lab"]
            b = next(r for r in rooms if r["id"] == uniq[1])["lab"]
            adj.add(tuple(sorted((a, b))))
    return sorted(adj)


def render_house(fp, pos, yaw, size_px=560):
    """WHOLE-HOUSE map: every room labelled, doors marked 'to <room>', the person's position +
    heading + trail, and any device that is ON drawn as a bright star — so a lamp left lit in a
    room the person is NOT in is still visible for routing AND for catching a forgotten device.
    Same frame as render(); this is the routing/overview companion to the current-room submap."""
    rooms_l = _rooms_labeled(fp)
    xs = [r["c"][0] - r["s"][0] / 2 for r in rooms_l] + [r["c"][0] + r["s"][0] / 2 for r in rooms_l]
    zs = [r["c"][2] - r["s"][2] / 2 for r in rooms_l] + [r["c"][2] + r["s"][2] / 2 for r in rooms_l]
    xlo, xhi, zlo, zhi = min(xs) - .4, max(xs) + .4, min(zs) - .4, max(zs) + .4
    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    for r in rooms_l:
        cx, cz, sx, sz = r["c"][0], r["c"][2], r["s"][0], r["s"][2]
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc="#161a22", ec="#39d98a", lw=1.4))
        ax.text(cx, cz - sz / 2 + .3, r["lab"].upper(), ha="center", va="center",
                color="#39d98a", fontsize=8, alpha=.65, fontweight="bold")
    for o in sorted(fp["objs"], key=lambda o: -(o["s"][0] * o["s"][1])):
        if o["cls"] == "lightswitch":
            continue
        cx, cz = o["c"]; sx, sz = o["s"]
        if "ON" in (o.get("st") or []):
            ax.plot(cx, cz, "*", color="#ffd23f", ms=15, mec="#8a6d00", mew=.8, zorder=6)
            ax.annotate(f"{o['cls']} ON", (cx, cz), textcoords="offset points", xytext=(0, -12),
                        ha="center", color="#ffd23f", fontsize=6.5, fontweight="bold", zorder=6)
            continue
        fill, edge = CATCOL.get(o.get("cat"), (None, "#7a8394"))
        if fill is None or (sx < .28 and sz < .28):
            ax.plot(cx, cz, ".", color=edge, ms=2.6, alpha=.6); continue
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc=fill, ec=edge, lw=1.0, alpha=.85))
    for dc in fp["doors"]:
        ax.plot(dc[0], dc[2], "s", color="#ff9ecb", ms=9, mfc="none", mew=1.7)
    pos = np.asarray(pos, float)
    if len(pos):
        ax.plot(pos[:, 0], pos[:, 1], "-", color="#4aa3ff", lw=2.2, alpha=.9)
        x, z = pos[-1]; a = float(yaw[-1])
        ax.arrow(x, z, .85 * np.sin(np.radians(a)), .85 * np.cos(np.radians(a)),
                 head_width=.34, head_length=.32, fc="#ffd24d", ec="#0f1117", lw=1.2, zorder=8)
        ax.plot(x, z, "o", color="#ffd24d", ms=12, mec="#0f1117", mew=1.5, zorder=7)
        ax.annotate("YOU", (x, z), textcoords="offset points", xytext=(0, 11), ha="center",
                    color="#ffd24d", fontsize=8, fontweight="bold", zorder=8)
    ax.set_xlim(xlo, xhi); ax.set_ylim(zlo, zhi); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    fig.tight_layout(pad=0.2)
    b = io.BytesIO(); fig.savefig(b, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return b.getvalue()


def house_text(fp, pos, yaw_deg):
    """Whole-house routing text: which room you are in, the room-to-room door graph, and any
    room that currently has a lit device (so a forgotten lamp is nameable). Companion to the
    current-room scene graph, which carries the full per-object detail of the room you are in."""
    rooms = _rooms_labeled(fp)
    cur = _room_of(rooms, pos[0], pos[1])
    lines = [f"WHOLE-HOUSE MAP — you are in the {cur['lab']}.",
             "rooms: " + ", ".join(r["lab"] for r in rooms)]
    lit = []
    for r in rooms:
        RX0, RZ0 = r["c"][0] - r["s"][0] / 2, r["c"][2] - r["s"][2] / 2
        devs = sorted({o["cls"] for o in fp["objs"]
                       if "ON" in (o.get("st") or []) and o["cls"] != "lightswitch"
                       and 0 <= o["c"][0] - RX0 <= r["s"][0] and 0 <= o["c"][1] - RZ0 <= r["s"][2]})
        if devs:
            lit.append(f"{r['lab']} ({', '.join(devs)})")
    lines.append("rooms with a device still ON: " + (" · ".join(lit) if lit else "none"))
    # ...and the same for containers left OPEN. Reporting only ON made a forgotten fridge door
    # unobservable on every channel at once — the person never returns to it (omissions do not
    # navigate), the ego frames never show it, and this map stayed silent (independent audit,
    # 2026-08-28).
    opened = []
    for r in rooms:
        RX0, RZ0 = r["c"][0] - r["s"][0] / 2, r["c"][2] - r["s"][2] / 2
        cs = sorted({o["cls"] for o in fp["objs"]
                     if "OPEN" in (o.get("st") or []) and o["cls"] != "lightswitch"
                     and min(o["s"]) >= 0.3
                     and 0 <= o["c"][0] - RX0 <= r["s"][0] and 0 <= o["c"][1] - RZ0 <= r["s"][2]})
        if cs:
            opened.append(f"{r['lab']} ({', '.join(cs)})")
    lines.append("rooms with something still OPEN: " + (" · ".join(opened) if opened else "none"))
    adj = _house_adjacency(fp, rooms)
    lines.append("doors connect: " + (" · ".join(f"{a}↔{b}" for a, b in adj) or "n/a"))
    return "\n".join(lines)


# ---------------- statemap-v5: CURRENT-ROOM submap + room-local text scene graph ----------------
LANDMARK_CATS = ("Furniture", "Appliances", "Electronics")
_DIRS8 = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def _rooms_labeled(fp):
    """Rooms with display labels; duplicate class names get #2, #3 suffixes."""
    seen, out = {}, []
    for r in fp["rooms"]:
        seen[r["cls"]] = seen.get(r["cls"], 0) + 1
        lab = r["cls"] + (f"#{seen[r['cls']]}" if seen[r["cls"]] > 1 else "")
        out.append({**r, "lab": lab})
    return out


def _room_of(rooms, x, z):
    for r in rooms:
        if abs(x - r["c"][0]) <= r["s"][0] / 2 and abs(z - r["c"][2]) <= r["s"][2] / 2:
            return r
    return min(rooms, key=lambda r: (x - r["c"][0]) ** 2 + (z - r["c"][2]) ** 2)


def _room_doors(fp, rooms, cur):
    """Doors on this room's boundary, with the DESTINATION room label."""
    RX0, RZ0 = cur["c"][0] - cur["s"][0] / 2, cur["c"][2] - cur["s"][2] / 2
    RW, RD = cur["s"][0], cur["s"][2]
    out = []
    for dc in fp["doors"]:
        dxl, dyl = dc[0] - RX0, dc[2] - RZ0
        if -0.6 <= dxl <= RW + 0.6 and -0.6 <= dyl <= RD + 0.6:
            dest = None
            for ddx, ddz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = _room_of(rooms, dc[0] + ddx * 0.9, dc[2] + ddz * 0.9)
                if rr["id"] != cur["id"]:
                    dest = rr["lab"]; break
            out.append((max(0, min(RW, dxl)), max(0, min(RD, dyl)), dest or "?"))
    return out


def render_room(fp, pos, yaw_deg):
    """CURRENT-ROOM submap PNG: object footprints (colour by category, NO text labels), doors labelled
    'to <room>', agent = YOU marker + view cone. North-up, y up; (0,0) = the room's SW corner — the same
    frame as room_text(), so image and text cross-reference exactly."""
    rooms = _rooms_labeled(fp)
    cur = _room_of(rooms, pos[0], pos[1])
    RX0, RZ0 = cur["c"][0] - cur["s"][0] / 2, cur["c"][2] - cur["s"][2] / 2
    RW, RD = cur["s"][0], cur["s"][2]
    lx, ly = pos[0] - RX0, pos[1] - RZ0
    fig, ax = plt.subplots(figsize=(5.6, max(3.8, 5.6 * RD / RW)) if RW >= RD else
                                    (max(3.8, 5.6 * RW / RD), 5.6), dpi=100)
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    ax.add_patch(Rectangle((0, 0), RW, RD, fc="#161a22", ec="#39d98a", lw=2.2))
    for o in fp["objs"]:
        if o["cls"] == "lightswitch":          # non-actionable no-op; keep image and text in sync
            continue
        cx, cz = o["c"][0] - RX0, o["c"][1] - RZ0
        sx, sz = o["s"]
        if cx < -0.3 or cx > RW + 0.3 or cz < -0.3 or cz > RD + 0.3:
            continue
        # A device that is ON reads as a bright lit star so the assistant can SEE a still-lit
        # lamp on the map (statemap given to the supervisor, 2026-08-26). OFF devices render
        # normally — highlighting only what is lit keeps the signal clean.
        if "ON" in (o.get("st") or []):
            ax.plot(cx, cz, "*", color="#ffd23f", ms=16, mec="#8a6d00", mew=.8, zorder=5)
            ax.annotate("ON", (cx, cz), textcoords="offset points", xytext=(0, -13),
                        ha="center", color="#ffd23f", fontsize=7, fontweight="bold", zorder=5)
            continue
        fill, edge = CATCOL.get(o.get("cat"), (None, "#7a8394"))
        if fill is None or (sx < .28 and sz < .28):
            ax.plot(cx, cz, ".", color=edge, ms=4, alpha=.75); continue
        ax.add_patch(Rectangle((cx - sx / 2, cz - sz / 2), sx, sz, fc=fill, ec=edge, lw=1.3, alpha=.92))
    for dxl, dyl, dest in _room_doors(fp, rooms, cur):
        ax.plot(dxl, dyl, "s", color="#ff9ecb", ms=13, mfc="none", mew=2.6)
        ax.annotate(f"to {dest}", (dxl, dyl), textcoords="offset points", xytext=(0, 12),
                    ha="center", color="#ff9ecb", fontsize=11, fontweight="bold")
    a0 = 90 - yaw_deg
    ax.add_patch(Wedge((lx, ly), 1.9, a0 - 30, a0 + 30, fc="#ffd24d", ec="none", alpha=0.22))
    ax.add_patch(Wedge((lx, ly), 1.9, a0 - 30, a0 + 30, fc="none", ec="#ffd24d", lw=1.3, alpha=.9))
    ax.plot(lx, ly, "o", color="#ffd24d", ms=15, mec="white", mew=2.3, zorder=10)
    ax.annotate("YOU", (lx, ly), textcoords="offset points", xytext=(0, -17),
                ha="center", color="#ffd24d", fontsize=11, fontweight="bold", zorder=10)
    ax.set_xlim(-0.4, RW + 0.4); ax.set_ylim(-0.4, RD + 0.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"CURRENT ROOM: {cur['lab']}   (north-up; (0,0) = bottom-left corner)",
                 color="#d7dce5", fontsize=10)
    fig.tight_layout(pad=0.3)
    b = io.BytesIO(); fig.savefig(b, format="png", facecolor=fig.get_facecolor()); plt.close(fig)
    return b.getvalue()


def room_text(fp, pos, yaw_deg):
    """Room-local TEXT scene graph in real-world meters: room span x[0..W], y[0..D] from the SW corner,
    explicit AGENT LOCATION / ORIENTATION, landmark objects (name#i: (x,y) size WxD [states]), labelled
    doors, and the other-room list. Matches render_room()'s frame exactly."""
    rooms = _rooms_labeled(fp)
    cur = _room_of(rooms, pos[0], pos[1])
    RX0, RZ0 = cur["c"][0] - cur["s"][0] / 2, cur["c"][2] - cur["s"][2] / 2
    RW, RD = cur["s"][0], cur["s"][2]
    lx, ly = pos[0] - RX0, pos[1] - RZ0
    fac = _DIRS8[int(((yaw_deg % 360) + 22.5) // 45) % 8]
    lines = [f"CURRENT ROOM MAP — {cur['lab']}. Meters; room spans x[0..{RW:.1f}], y[0..{RD:.1f}]; "
             f"(0,0)=bottom-left corner of the map image, +x=east, +y=north.",
             f"AGENT LOCATION: room={cur['lab']}, position=({lx:.1f}, {ly:.1f})",
             f"AGENT ORIENTATION: facing {fac} (heading {yaw_deg % 360:.0f} deg; 0=north, clockwise)"]
    counts, items = {}, []
    for o in fp["objs"]:
        # Keep an object if it is a sizeable landmark OR it is a switchable DEVICE (ON/OFF). A
        # tablelamp is tiny and not a landmark category, so the size/category filter used to drop
        # it — but its ON/OFF state is exactly what the assistant needs to read off the map to
        # catch a still-lit lamp (statemap given to the supervisor, 2026-08-26). Only ON/OFF
        # forces inclusion; small OPEN/CLOSED things (curtains, folders) stay culled as noise,
        # while openable containers that matter (fridge, microwave, desk) are landmark-sized and
        # come through the normal path anyway.
        # a wall lightswitch reports ON but is a non-actionable no-op in these tasks — showing
        # "lightswitch [ON]" would read as "the light is handled" and mislead. Cull it.
        if o["cls"] == "lightswitch":
            continue
        device = any(s in ("ON", "OFF") for s in (o.get("st") or []))
        if not device and (o.get("cat") not in LANDMARK_CATS or min(o["s"]) < 0.3):
            continue
        cx, cz = o["c"][0] - RX0, o["c"][1] - RZ0
        if not (0 <= cx <= RW and 0 <= cz <= RD):
            continue
        counts[o["cls"]] = counts.get(o["cls"], 0) + 1
        lab = o["cls"] + (f"#{counts[o['cls']]}" if counts[o["cls"]] > 1 else "")
        st = f" [{', '.join(o['st'])}]" if o.get("st") else ""
        items.append((lab, f"{lab}: ({cx:.1f}, {cz:.1f}) size {o['s'][0]:.1f}x{o['s'][1]:.1f}{st}"))
    lines += [t for _, t in sorted(items)]
    lines.append("doors: " + (" · ".join(f"to {dest} at ({dxl:.1f}, {dyl:.1f})"
                                          for dxl, dyl, dest in _room_doors(fp, rooms, cur)) or "none"))
    lines.append("other rooms (not on this map): " +
                 ", ".join(sorted(r["lab"] for r in rooms if r["id"] != cur["id"])))
    return "\n".join(lines)
