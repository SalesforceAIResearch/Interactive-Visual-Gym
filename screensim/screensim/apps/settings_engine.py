"""Declarative Settings interpreter.

The whole Settings tree is DATA (see settings_data.SCREENS): each screen is a
list of sections, each section a list of typed rows. This module turns that
data + the live state into engine Screens, and applies taps generically, so
adding anything the iPhone manual lists is a data edit — no new code.

Row types (the `t` field):
  nav     -> pushes another screen (target)               keys: label, target, icon, sub/subval
  toggle  -> on/off switch bound to a state path          keys: label, skey, icon, sub, disable_if
  choice  -> opens a generated radio list, sets a path     keys: label, skey, options, icon
  slider  -> 0-100 slider bound to a state path            keys: label, skey
  field   -> text entry bound to a state path              keys: label, skey, kb, mask, hint
  info    -> read-only value                               keys: label, val/valkey
  action  -> fires an event (custom handler)               keys: label, event, style, confirm
  text    -> explanatory paragraph                         keys: text
  link    -> pushes into another app                       keys: label, target(app.screen), icon

Dynamic screens (Wi-Fi list, Location per app, Storage…) are registered as
BUILDERS: builder(state, arg) -> a screen spec dict (same shape), so they get
the same rendering + a bit of custom logic.
"""
from __future__ import annotations

import re

from ..core.model import Screen, Widget, w_field, w_row, w_section, w_text, w_toggle


# ---- state path helpers ----
def sget(state, path, default=None):
    cur = state
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def sset(state, path, value):
    parts = path.split(".")
    cur = state
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _fmt(state, spec_val, subkey=None):
    """Resolve a subtitle: a static string, a state value, an On/Off flag, or a
    formatter."""
    if callable(spec_val):
        return spec_val(state)
    if subkey:
        if subkey.startswith("_onoff:"):
            return "On" if sget(state, subkey[7:]) else "Off"
        v = sget(state, subkey)
        return "" if v is None else str(v)
    return spec_val or ""


# choice screens are generated on the fly with a synthetic key
CHOICE_PREFIX = "settings.__choice__"


def choice_key(skey):
    return f"{CHOICE_PREFIX}:{skey}"


# ---- build an engine Screen from a spec screen ----
STUB_LABELS = {}          # slug -> row label, for placeholder pages


def stub_target(target, label):
    """A row that leads nowhere in this manual ("stub_none") used to open one
    shared blank page titled "None" — a wrong turn onto it looked like an
    engine fault. Give every such row its own placeholder page, titled by the
    row, so a wrong turn reads as a real (empty) settings page."""
    if target.endswith("stub_none"):
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]
        STUB_LABELS[slug] = label
        try:
            from .settings_gen import GEN_SCREENS
            if slug in GEN_SCREENS:               # a synthesized realistic page
                return target[:-len("stub_none")] + "gen_" + slug
        except ImportError:
            pass
        return target[:-len("stub_none")] + "stub_none__" + slug
    return target


def build_screen(app, state, key, spec):
    """spec: {title, back?, sections:[{header, rows:[...]}]} -> Screen."""
    ws = []
    for sec in spec.get("sections", []):
        header = sec.get("header", "")
        # a section header (even empty, for spacing groups)
        ws.append(w_section(header))
        for r in sec.get("rows", []):
            ws.extend(_row_widgets(app, state, key, r))
    scr = Screen(key, spec.get("title", ""), ws)
    if "back" in spec and spec["back"]:
        scr.back_label = spec["back"]
    if spec.get("appbar_actions"):
        scr.appbar_actions = spec["appbar_actions"]
    return scr


def _row_widgets(app, state, screen_key, r):
    t = r["t"]
    icon = r.get("icon", "")
    if t == "nav" or t == "link":
        sub = _fmt(state, r.get("sub"), r.get("subval"))
        target = stub_target(r["target"], r["label"])
        return [w_row(_rid(screen_key, r), r["label"], sub=sub, icon=icon,
                      on_tap=("nav", target))]
    if t == "toggle":
        val = bool(sget(state, r["skey"]))
        disabled = bool(r.get("disable_if") and sget(state, r["disable_if"]))
        sub = _fmt(state, r.get("sub"), r.get("subval"))
        w = w_toggle(_rid(screen_key, r), r["label"], val, sub=sub, icon=icon,
                     event=("sset_toggle", r["skey"]), enabled=not disabled)
        return [w]
    if t == "choice":
        cur = sget(state, r["skey"])
        return [w_row(_rid(screen_key, r), r["label"], sub=str(cur) if cur is not None else "",
                      icon=icon, on_tap=("nav", choice_key(r["skey"])))]
    if t == "slider":
        # a slider moves: swipe right on it = one step up, left = one step down (0-100)
        return [Widget("slider", _rid(screen_key, r), label=r["label"],
                       value=int(sget(state, r["skey"], 0)), h=r.get("h", 56),
                       on_swipe={"right": ("sslider", r["skey"], +25),
                                 "left": ("sslider", r["skey"], -25)})]
    if t == "field":
        # id encodes the state path so generic typing routes back to it
        return [w_field(f"spec::{r['skey']}", r["label"],
                        value=str(sget(state, r["skey"], "")),
                        hint=r.get("hint", ""), mask=r.get("mask", False),
                        kb=r.get("kb", "text"))]
    if t == "info":
        val = _fmt(state, r.get("val"), r.get("valkey"))
        w = w_row(_rid(screen_key, r), r["label"], sub=val, icon=icon)
        w.on_tap = None
        return [w]
    if t == "action":
        style = r.get("style", "text")
        return [Widget("button", _rid(screen_key, r), label=r["label"],
                       on_tap=("saction", r["event"]), h=50,
                       meta={"style": style})]
    if t == "text":
        return [w_text(_rid(screen_key, r), r["text"], h=r.get("h"))]
    if t == "custom":
        # a raw widget factory: r["make"](state) -> Widget or [Widget]
        out = r["make"](state)
        return out if isinstance(out, list) else [out]
    return []


def _rid(screen_key, r):
    base = r.get("id") or r.get("skey") or r.get("label", "row")
    return f"{screen_key.split('.')[-1]}_{base}".replace(" ", "_").replace(".", "_")[:48]


def build_choice_screen(state, skey, opts_spec, title):
    """A radio list that sets `skey` to the chosen option and pops."""
    ws = []
    cur = sget(state, skey)
    footer = opts_spec.get("footer")
    for opt in opts_spec["options"]:
        val = opt if isinstance(opt, str) else opt[0]
        label = opt if isinstance(opt, str) else opt[1]
        ws.append(Widget("radio", f"ch_{val}"[:40], label=label,
                         value=(cur == val), h=48,
                         on_tap=("schoice", skey, val)))
    if footer:
        ws.append(w_text("ch_footer", footer, h=64))
    scr = Screen(choice_key(skey), opts_spec.get("title", title), ws)
    return scr
