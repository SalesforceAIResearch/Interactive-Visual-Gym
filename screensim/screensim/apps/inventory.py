"""Every settings control that exists in the simulated phone, with its page path and value.

Used to verify task instructions: given a request, which controls could possibly satisfy it?
Walks every screen reachable from the Settings root (choice pages included) breadth-first, so
each page is listed under the shortest route that reaches it.

  python screensim/apps/inventory.py > tmp/settings_inventory.txt
"""
from __future__ import annotations

import os
import random
import sys
from collections import deque


from screensim.apps.settings_app import SettingsApp
from screensim.core.state import DeviceState


def _fresh():
    app = SettingsApp()
    st = DeviceState()
    st.app = "settings"
    st.data = {"settings": app.initial_data(random.Random(0))}
    st.stack = ["settings.root"]
    return app, st


def _value(w):
    v = w.value
    if isinstance(v, bool):
        return "on" if v else "off"
    return "" if v is None else str(v)


def _path(w):
    tap = w.on_tap
    if isinstance(w.on_swipe, dict) and w.on_swipe:
        return list(w.on_swipe.values())[0][1]
    if tap and len(tap) > 1 and isinstance(tap[1], str) and tap[0] != "nav":
        return " ".join(str(x) for x in tap)
    return ""


# hand-built controls whose widget ids differ but which drive one setting
def canonical(w):
    tap = w.on_tap or ()
    if isinstance(w.on_swipe, dict) and w.on_swipe:
        return "slider:" + list(w.on_swipe.values())[0][1]
    if not tap:
        return ""
    k = tap[0]
    if k in ("sset_toggle", "sslider") and len(tap) > 1:
        return f"{k}:{tap[1]}"
    if k == "schoice" and len(tap) > 2:
        return f"schoice:{tap[1]}={tap[2]}"
    if k == "loc_mode_set" and len(tap) > 2:
        return f"loc:{tap[1]}={tap[2]}"
    if k == "toggle" and len(tap) > 1:
        wid = str(tap[1])
        if wid == "bold_on":
            return "sset_toggle:spec.accessibility.bold"
        for pre, field in (("bgapp2_", "bg"), ("bgapp_", "bg"), ("cellapp2_", "cellular"), ("cellapp_", "cellular")):
            if wid.startswith(pre):
                return f"app:{wid[len(pre):]}.{field}"
        return f"toggle:{wid}"
    return ""


def walk(with_paths=False):
    app, st = _fresh()
    seen, pages = {"settings.root"}, []
    PAGE_PATH.clear()
    PAGE_PATH["settings.root"] = "Settings"
    q = deque([("settings.root", ["Settings"])])
    while q:
        key, path = q.popleft()
        st.stack = [key]
        try:
            scr = app.view(st)
        except Exception:
            continue
        if key != "settings.root" and scr.key == "settings.root":
            continue                                  # a dead link that falls back to the root
        controls = []
        for w in scr.widgets:
            label = (w.label or "").strip()
            if not label:
                continue
            sp = f"      <{_path(w)}>" if with_paths else ""
            if w.kind == "toggle":
                controls.append(f"switch  {label} = {_value(w)}{sp}")
            elif w.kind == "slider":
                controls.append(f"slider  {label} = {_value(w)}{sp}")
            elif w.kind == "radio":
                controls.append(f"option  {label}{'  (selected)' if w.value else ''}{sp}")
            elif w.kind == "button" or (w.kind in ("row", "account") and w.on_tap and w.on_tap[0] != "nav"):
                controls.append(f"button  {label}{sp}")
            elif w.kind in ("row", "account") and w.on_tap and w.on_tap[0] == "nav":
                shown = (getattr(w, "sub", "") or "").strip()
                if shown and len(shown) <= 24:
                    controls.append(f"page    {label}  (the row shows: {shown})\x01{w.on_tap[1]}{sp}")
            if w.kind in ("toggle", "slider", "radio"):
                ck = canonical(w)
                if ck:
                    CANON.setdefault(ck, []).append((" > ".join(path), label))
                    controls[-1] = controls[-1] + f"\x00{ck}"
            tap = w.on_tap
            if tap and tap[0] == "nav" and isinstance(tap[1], str) and tap[1].startswith("settings.") \
                    and tap[1] not in seen:
                seen.add(tap[1])
                PAGE_PATH[tap[1]] = " > ".join(path + [label])
                q.append((tap[1], path + [label]))
        pages.append((" > ".join(path), controls))
    return pages


CANON = {}
PAGE_PATH = {}


def render(pages):
    """Text inventory; a control shown on several pages carries a note naming the others."""
    out = []
    for path, controls in pages:
        if not controls:
            continue
        out.append(f"[{path}]")
        for c in controls:
            if "\x01" in c:
                head, _, tail = c.partition("\x01")
                target, _, rest = tail.partition("      <")
                where = PAGE_PATH.get(target)
                c = head + (f" — opens the page listed as [{where}]" if where and where != f"{path} > " + head.split("  (")[0].split("    ", 1)[-1].strip() else "")
                c += ("      <" + rest) if rest else ""
            text, _, ck = c.partition("\x00")
            others = [p for p, _l in CANON.get(ck, []) if p != path] if ck else []
            if others:
                text += "   (the same setting also appears under: " + "; ".join(sorted(set(others))) + ")"
            out.append(f"  {text}")
    return "\n".join(out)


def main():
    pages = walk(with_paths="--paths" in sys.argv)
    print(render(pages))


if __name__ == "__main__":
    main()
