"""Auto-generate validated tasks from the mirrored Apple manual.

For every walkable `Go to Settings > …` path: replay it with real gestures,
find the actionable controls on the destination screen (toggles and choice
rows — preferring ones the article names in its steps), actuate each, and
deep-diff the engine state to learn exactly what changed. Each actuation
becomes a task record:

  {id, instruction, plan, changes: [(state_path, new_value)], mirrors, source}

The goal predicate IS the recorded diff, so every task ships with verified
ground truth. Every record is re-validated with a fresh replay before it is
written. Output: docs/apple_manual/_auto_tasks.jsonl (loaded by
screensim.autotasks).

  python screensim_bench/suite/gen_auto_tasks.py
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from collections import defaultdict


from screensim.core.engine import Device                     # noqa: E402
import screensim_bench.suite.manual_coverage as MC                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCS = os.path.join(ROOT, "docs", "apple_manual", "_procedures.jsonl")
OUT = os.path.join(ROOT, "docs", "apple_manual", "_auto_tasks.jsonl")

MAX_PER_SCREEN = 4          # keep the library broad, not repetitive


def deep_diff(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b), key=str):
        pa = f"{prefix}.{k}" if prefix else str(k)
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += deep_diff(va, vb, pa)
        elif va != vb:
            out.append((pa, vb))
    return out


def walk_recording(segs):
    """Walk a path like the coverage walker, but record the gesture plan."""
    d = Device(0)
    d.open_app("settings")
    plan = []
    for i, raw in enumerate(segs):
        seg = MC.norm_segment(raw)
        if seg is None:
            return None, None
        ok = False
        for _ in range(6):
            r = d.act(f"tap {seg}")
            if r.ok:
                plan.append(f"tap {seg}")
                ok = True
                break
            if getattr(r, "ambiguous", False):
                r2 = d.act(f"tap the first {seg}")
                if r2.ok:
                    plan.append(f"tap the first {seg}")
                    ok = True
                    break
            d.act("swipe up")
            plan.append("swipe up")
        if not ok:
            return None, None
    return d, plan


def find_control(d, label, plan):
    """Bring a labeled widget into view (recording swipes); return it."""
    for _ in range(8):
        scr = d.laid_out_screen()
        for w in scr.widgets:
            if w.visible and w.label.lower() == label.lower() \
                    and w.interactive():
                return w
        r = d.act("swipe up")
        if r.ok:
            plan.append("swipe up")
    return None


def snap(d):
    return copy.deepcopy(d.st.data["settings"])


_IGNORE = re.compile(r"^(wifi\.status|wifi\.joining|wifi\.join_pw)")


def actuate_toggle(d, w, plan):
    before = snap(d)
    r = d.act(f"tap {w.label}")
    if not r.ok:
        return None
    plan.append(f"tap {w.label}")
    return [c for c in deep_diff(before, snap(d)) if not _IGNORE.match(c[0])]


def actuate_choice(d, w, plan):
    """Row that opens a radio list: enter, pick a different option."""
    before = snap(d)
    r = d.act(f"tap {w.label}")
    if not r.ok:
        return None
    plan.append(f"tap {w.label}")
    scr = d.laid_out_screen()
    radios = [x for x in scr.widgets if x.kind == "radio"]
    pick = next((x for x in radios if not x.value), None)
    if pick is None:
        return None
    r2 = d.act(f"tap {pick.label}")
    if not r2.ok:
        return None
    plan.append(f"tap {pick.label}")
    changes = [c for c in deep_diff(before, snap(d)) if not _IGNORE.match(c[0])]
    return changes, pick.label


def instruction_for(kind, label, screen_title, extra=""):
    if kind == "on":
        return f"Turn on {label} in {screen_title}."
    if kind == "off":
        return f"Turn off {label} in {screen_title}."
    return f"Set {label} to {extra} in {screen_title}."


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def validate(rec):
    d = Device(0)
    d.open_app("settings")
    for c in rec["plan"]:
        d.act(c)
    st = d.st.data["settings"]
    for path, exp in rec["changes"]:
        cur = st
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur != exp:
            return False
    return True


def main():
    # unique walkable paths + the article controls that mention them
    paths = {}
    for line in open(PROCS):
        rec = json.loads(line)
        for p in rec["settings_paths"]:
            t = tuple(p)
            if MC.scope_key(t) in MC.OUT_OF_SCOPE:
                continue
            paths.setdefault(t, {"articles": [], "controls": set()})
            paths[t]["articles"].append(rec["id"])
            paths[t]["controls"] |= {c.lower() for c in rec["controls"]}

    tasks, seen_ids, n_invalid = [], set(), 0
    per_screen = defaultdict(int)
    for t in sorted(paths):
        d, nav_plan = walk_recording(list(t))
        if d is None:
            continue
        screen = d.laid_out_screen()
        skey, stitle = screen.key, screen.title or "Settings"
        mentioned = paths[t]["controls"]
        art = paths[t]["articles"][0]

        # candidate controls on this screen: toggles + choice rows,
        # article-mentioned ones first
        cands = [w for w in screen.widgets
                 if w.kind == "toggle"
                 or (w.kind == "row" and w.on_tap and w.on_tap[0] == "nav"
                     and "__choice__" in str(w.on_tap[1]))]
        cands.sort(key=lambda w: (w.label.lower() not in mentioned, w.row))

        for w in cands:
            if per_screen[(skey, w.label)] or per_screen[skey] >= MAX_PER_SCREEN:
                continue
            # fresh device per actuation so diffs are isolated
            d2, plan = walk_recording(list(t))
            if d2 is None:
                continue
            w2 = find_control(d2, w.label, plan)
            if w2 is None:
                continue
            if w2.kind == "toggle":
                changes = actuate_toggle(d2, w2, plan)
                if not changes:
                    continue
                kind = "off" if w2.value else "on"
                instr = instruction_for(kind, w2.label, stitle)
                tid = slug(f"auto_{stitle}_{w2.label}_{kind}")
            else:
                res = actuate_choice(d2, w2, plan)
                if not res or not res[0]:
                    continue
                changes, opt = res
                instr = instruction_for("choice", w2.label, stitle, opt)
                tid = slug(f"auto_{stitle}_{w2.label}_{slug(opt)}")
            if tid in seen_ids:
                continue
            rec = {"id": tid, "instruction": instr, "plan": plan,
                   "changes": changes, "screen": skey,
                   "mentioned": w.label.lower() in mentioned,
                   "mirrors": art,
                   "source": f"docs/apple_manual/{art}.md"}
            if not validate(rec):
                n_invalid += 1
                continue
            seen_ids.add(tid)
            tasks.append(rec)
            per_screen[skey] += 1
            per_screen[(skey, w.label)] = 1

    with open(OUT, "w") as f:
        for rec in tasks:
            f.write(json.dumps(rec) + "\n")
    n_ment = sum(1 for r in tasks if r["mentioned"])
    print(f"auto-generated {len(tasks)} validated tasks "
          f"({n_ment} actuate a control the article names; "
          f"{n_invalid} rejected by validation)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
