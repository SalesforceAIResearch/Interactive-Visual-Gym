"""OS shell: home launcher, app switching, navigation stack, notification
shade, quick settings, system dialogs, share sheet.

The shell owns screen keys with prefixes: home, shade, quicksettings, and
routes `nav`/system events. Apps own everything else.
"""
from __future__ import annotations

from typing import Optional

from . import app as app_mod
from .model import (SCREEN_W, Screen, Widget, w_banner, w_row, w_section,
                    w_text)
from .state import DeviceState

SHELL_EVENTS = {"nav", "back", "home", "open_app", "open_stub_app",
                "open_shade", "close_shade",
                "qs_toggle", "cc_toggle", "dismiss_notif", "open_notif", "toast",
                "dialog", "close_dialog", "share_to"}


def init(st: DeviceState):
    st.app = "home"
    st.stack = ["home"]


# ---- home launcher (wallpaper, full grid, search pill, dock) ----
_STOCK_HOME = [("Safari", "wifi", "#1f7ae0"), ("Maps", "pin", "#34c759"),
               ("Music", "vol", "#ff2d55"), ("Notes", "file", "#e0b23e"),
               ("Calendar", "bell", "#ff3b30"), ("Health", "batt", "#ff2d55"),
               ("FaceTime", "camera", "#34c759"), ("Translate", "chat", "#5a8cf0"),
               ("Wallet", "pay", "#1c1c1e")]
_DOCK = [("phone", None), ("Safari", "stub"), ("messages", None), ("mail", None)]


def _home_screen(st: DeviceState) -> Screen:
    tiles = []
    dock_ids = {d[0] for d in _DOCK}
    for a in app_mod.all_apps():
        if a.id in dock_ids:
            continue
        badge = _badge_count(st, a.id)
        tiles.append(Widget("tile", f"app_{a.id}", label=a.name, icon=a.icon,
                            h=96, on_tap=("open_app", a.id),
                            meta={"color": a.color, "badge": badge}))
    for name, icon, color in _STOCK_HOME:
        if name in dock_ids:
            continue
        tiles.append(Widget("tile", f"stock_{name}", label=name, icon=icon,
                            h=96, on_tap=("open_stub_app", name),
                            meta={"color": color}))
    # search pill + dock (absolute, like iOS)
    tiles.append(Widget("homesearch", "home_search", label="Search", h=0,
                        on_tap=("noop",), meta={"abs": (146, 736, 120, 36)}))
    tiles.append(Widget("dockbar", "dock_bar", h=0,
                        meta={"abs": (10, 780, 392, 82)}))
    dx = 32
    for entry, kind in _DOCK:
        if kind is None:
            a = app_mod.get(entry)
            tiles.append(Widget("tile", f"dock_{a.id}", label="", icon=a.icon,
                                h=64, on_tap=("open_app", a.id),
                                meta={"color": a.color, "nolabel": True,
                                      "abs": (dx, 789, 64, 64),
                                      "badge": _badge_count(st, a.id)}))
        else:
            color = dict((n, c) for n, i, c in _STOCK_HOME)[entry]
            icon = dict((n, i) for n, i, c in _STOCK_HOME)[entry]
            tiles.append(Widget("tile", f"dock_{entry}", label="", icon=icon,
                                h=64, on_tap=("open_stub_app", entry),
                                meta={"color": color, "nolabel": True,
                                      "abs": (dx, 789, 64, 64)}))
        dx += 88
    return Screen("home", None, tiles, grid=4, show_back=False, scrollable=False)


def _stub_app_screen(st: DeviceState) -> Screen:
    name = st.data.get("_stub_app", "App")
    ws = [Widget("apphdr", "stub_hdr", label=name, sub="", icon="apps", h=200,
                 meta={"color": "#8e8e93"}),
          Widget("text", "stub_txt", label=f"{name} has nothing to do for this "
                 "task. Go Home to get back on track.", h=64)]
    return Screen("stubapp", name, ws)


def _badge_count(st, app_id) -> int:
    return sum(1 for n in st.notifications if n.app == app_id)


# ---- notification shade ----
def _shade_screen(st: DeviceState) -> Screen:
    ws = [w_section("Quick settings")]
    qs = st.data.get("_qs", {})
    tiles = []
    for key, (label, icon) in _QS_TILES.items():
        tiles.append(Widget("qs_tile", f"qs_{key}", label=label, icon=icon,
                            value=qs.get(key, False), h=72,
                            on_tap=("qs_toggle", key)))
    ws.extend(tiles)
    ws.append(w_section("Notifications"))
    if not st.notifications:
        ws.append(w_text("no_notif", "No new notifications", h=48))
    for n in st.notifications:
        ws.append(Widget("notif", f"notif_{n.id}", label=n.title, sub=n.text,
                         h=76, on_tap=("open_notif", n.id),
                         on_swipe={"left": ("dismiss_notif", n.id),
                                   "right": ("dismiss_notif", n.id)},
                         meta={"app": n.app}))
    return Screen("shade", "Notifications", ws, grid=1, show_back=True)


_QS_TILES = {
    "wifi": ("Wi-Fi", "wifi"),
    "bluetooth": ("Bluetooth", "bt"),
    "hotspot": ("Hotspot", "hotspot"),
    "airplane": ("Airplane", "airplane"),
    "flashlight": ("Flashlight", "flash"),
    "dnd": ("Do Not Disturb", "dnd"),
}


def current_screen(st: DeviceState) -> Screen:
    if st.dialog:
        return _dialog_screen(st)
    if getattr(st, "cc_open", False):
        return _cc_screen(st)
    if st.shade_open:
        return _shade_screen(st)
    if st.app == "home":
        if st.stack and st.stack[-1] == "stubapp":
            return _stub_app_screen(st)
        return _home_screen(st)
    owner = app_mod.get(st.app)
    return owner.view(st)


# ---- Control Center (iOS 18: circular-slot grid, connectivity cluster
#      top-left, vertical brightness/volume sliders, page dots) ----
def _cc_screen(st: DeviceState) -> Screen:
    s = st.data.get("settings", {})
    wifi_on = s.get("wifi", {}).get("on", True)
    bt_on = s.get("bluetooth", {}).get("on", False)
    hot_on = s.get("hotspot", {}).get("on", False)
    fl_on = st.data.get("_qs", {}).get("flashlight", False)

    def rnd(id, label, icon, value, x, y, ev, d=64):
        return Widget("cc_round", id, label=label, icon=icon, value=value,
                      on_tap=ev, meta={"abs": (x, y, d, d)})

    ws = [
        # connectivity cluster (2x2 module, top-left) — Airplane/Cellular/Wi-Fi/BT
        Widget("cc_module", "cc_cluster", label="", h=0,
               meta={"abs": (20, 120, 184, 184)}),
        rnd("cc_airplane", "Airplane Mode", "airplane",
            s.get("airplane", False), 38, 138, ("cc_toggle", "airplane")),
        rnd("cc_cell", "Cellular Data", "cell",
            s.get("cellular", {}).get("data", True), 122, 138,
            ("cc_toggle", "cellular")),
        rnd("cc_wifi", "Wi-Fi", "wifi", wifi_on, 38, 222, ("cc_toggle", "wifi")),
        rnd("cc_bt", "Bluetooth", "bt", bt_on, 122, 222, ("cc_toggle", "bluetooth")),
        # now-playing module (top-right)
        Widget("cc_media", "cc_media", label="Not Playing", sub="Music",
               h=0, meta={"abs": (220, 120, 172, 184)}),
        # vertical sliders
        Widget("cc_vslider", "cc_bright", label="Brightness", icon="sun",
               value=s.get("brightness", 60),
               meta={"abs": (220, 322, 78, 196)}),
        Widget("cc_vslider", "cc_vol", label="Volume", icon="vol",
               value=s.get("sound", {}).get("ring", 60),
               meta={"abs": (314, 322, 78, 196)}),
        # circular controls, left block
        rnd("cc_flash", "Flashlight", "flash", fl_on, 38, 340,
            ("cc_toggle", "flashlight")),
        rnd("cc_timer", "Timer", "bell", False, 122, 340, ("noop",)),
        rnd("cc_dark", "Dark Mode", "darkmode", s.get("dark", False), 38, 424,
            ("cc_toggle", "dark")),
        rnd("cc_lowpower", "Low Power Mode", "batt", s.get("low_power", False),
            122, 424, ("cc_toggle", "low_power")),
        # 2x1 rectangular controls (these keep labels on real iOS)
        Widget("cc_tile", "cc_hotspot", label="Personal Hotspot", icon="hotspot",
               value=hot_on, on_tap=("cc_toggle", "hotspot"),
               meta={"abs": (20, 540, 184, 64)}),
        Widget("cc_tile", "cc_mirror", label="Screen Mirroring", icon="share",
               value=False, on_tap=("noop",),
               meta={"abs": (220, 540, 172, 64)}),
        Widget("cc_tile", "cc_focus", label="Focus", icon="moon",
               value=False, on_tap=("noop",),
               meta={"abs": (20, 622, 184, 64)}),
        Widget("cc_tile", "cc_airdrop", label="AirDrop", icon="share",
               value=False, on_tap=("noop",),
               meta={"abs": (220, 622, 172, 64)}),
    ]
    scr = Screen("ccenter", None, ws, show_back=False, scrollable=False)
    return scr


# ---- dialogs ----
def _dialog_screen(st: DeviceState) -> Screen:
    d = st.dialog
    ws = [w_text("dlg_title", d["title"], h=40)]
    if d.get("body"):
        ws.append(w_text("dlg_body", d["body"], h=64))
    for (label, event, style) in d["buttons"]:
        ws.append(Widget("button", f"dlg_{label}", label=label, on_tap=event,
                        h=48, meta={"style": style}))
    s = Screen(d["screen"], None, ws, grid=1, show_back=False,
               scrollable=False, is_dialog=True)
    s.sheet = bool(d.get("sheet"))
    return s


# ---- event handling ----
def press_key(st: DeviceState, key: str):
    if key == "home":
        st.shade_open = False
        st.dialog = None
        st.goto_app("home", "home")
        return ("home",)
    if key == "back":
        return _back(st)
    if key == "recents":
        return ("recents",)
    return None


def _back(st: DeviceState):
    if st.dialog:
        if st.dialog.get("dismissable", True):
            st.dialog = None
        return ("back",)
    if getattr(st, "cc_open", False):
        st.cc_open = False
        return ("back",)
    if st.shade_open:
        st.shade_open = False
        return ("back",)
    if st.keyboard:
        st.keyboard = False
        st.focus = None
        return ("back",)
    # ask the app if leaving this screen needs a guard
    owner = app_mod.owner_of(st.cur_screen_key())
    if owner:
        note = owner.handle(st, ("back",), None)
        if st.dialog:      # app raised a confirm dialog
            return ("back",)
    if len(st.stack) > 1:
        st.pop()
    else:
        st.goto_app("home", "home")
    return ("back",)


def navigate(st: DeviceState, key: str):
    st.shade_open = False
    st.keyboard = False
    st.focus = None
    st.push(key)
    return None


def handle(st: DeviceState, event: tuple):
    kind = event[0]
    if kind == "open_stub_app":
        st.data["_stub_app"] = event[1]
        st.push("stubapp")
        return None
    if kind == "open_app":
        a = app_mod.get(event[1])
        st.shade_open = False
        st.dialog = None
        st.goto_app(a.id, a.root_key)
        return None
    if kind == "nav":
        return navigate(st, event[1])
    if kind == "back":
        return _back(st)
    if kind == "home":
        return press_key(st, "home")
    if kind == "open_shade":
        st.shade_open = True
        return None
    if kind == "close_shade":
        st.shade_open = False
        return None
    if kind == "qs_toggle":
        qs = st.data.setdefault("_qs", {})
        key = event[1]
        qs[key] = not qs.get(key, False)
        # mirror QS toggles into the real app state where relevant
        _mirror_qs(st, key, qs[key])
        return None
    if kind == "cc_toggle":
        return _cc_toggle(st, event[1])
    if kind == "dismiss_notif":
        st.notifications = [n for n in st.notifications if n.id != event[1]]
        return None
    if kind == "open_notif":
        n = next((x for x in st.notifications if x.id == event[1]), None)
        if n and n.on_tap:
            st.shade_open = False
            return handle(st, n.on_tap) if n.on_tap[0] in SHELL_EVENTS \
                else _app_route(st, n.on_tap)
        return None
    if kind == "toast":
        st.toast = {"text": event[1], "until": st.tick + event[2] if len(event) > 2 else st.tick + 3}
        return None
    if kind == "dialog":
        st.dialog = event[1]
        return None
    if kind == "close_dialog":
        st.dialog = None
        return None
    if kind == "share_to":
        owner = app_mod.get(event[1])
        st.dialog = None
        st.goto_app(owner.id, owner.root_key)
        owner.handle(st, ("receive_share", event[2] if len(event) > 2 else None), None)
        return None
    return None


def _app_route(st, event):
    owner = app_mod.owner_of(event[1]) if len(event) > 1 and isinstance(event[1], str) else None
    if event[0] == "nav":
        # jump into the target app then navigate
        target_app = event[1].split(".")[0]
        a = app_mod.get(target_app)
        st.goto_app(a.id, a.root_key)
        if event[1] != a.root_key:
            st.push(event[1])
    return None


def _cc_toggle(st: DeviceState, key: str):
    """Control Center toggles mirror straight into Settings state — a second
    legitimate route to several task goals, exactly like the real iPhone."""
    s = st.data.get("settings", {})
    if key == "airplane":
        s["airplane"] = not s.get("airplane", False)
        if s["airplane"]:
            s.setdefault("wifi", {})["on"] = False
            st.net = "none"
    elif key == "wifi":
        w = s.setdefault("wifi", {})
        w["on"] = not w.get("on", True)
        st.net = "wifi" if w["on"] else "cellular"
    elif key == "bluetooth":
        b = s.setdefault("bluetooth", {})
        b["on"] = not b.get("on", False)
    elif key == "cellular":
        c = s.setdefault("cellular", {})
        c["data"] = not c.get("data", True)
    elif key == "hotspot":
        h = s.setdefault("hotspot", {})
        if not h.get("password"):
            st.dialog = {
                "screen": "settings.dlg_hs",
                "title": "Set a Password First",
                "body": "Set a Wi-Fi password in Settings › Personal Hotspot "
                        "before others can join.",
                "buttons": [("OK", ("close_dialog",), "filled")],
                "dismissable": True,
            }
            return None
        h["on"] = not h.get("on", False)
    elif key == "dark":
        s["dark"] = not s.get("dark", False)
    elif key == "low_power":
        s["low_power"] = not s.get("low_power", False)
    elif key == "flashlight":
        qs = st.data.setdefault("_qs", {})
        qs["flashlight"] = not qs.get("flashlight", False)
    return None


def _mirror_qs(st, key, value):
    if key == "wifi":
        st.net = "wifi" if value else "cellular"
        st.data.get("settings", {}).setdefault("wifi", {})["on"] = value
    if key == "hotspot" and "settings" in st.data:
        st.data["settings"].setdefault("hotspot", {})["on"] = value
    if key == "bluetooth" and "settings" in st.data:
        st.data["settings"].setdefault("bluetooth", {})["on"] = value
