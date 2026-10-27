"""Settings — full iOS 18 density.

The root has the real row set in iOS groups; subscreens carry the options a
real iPhone shows (Wi-Fi with Ask-to-Join and a password join sheet, Personal
Hotspot with Maximize Compatibility, General with Background App Refresh and
Auto-Lock choice lists, Privacy & Security with Location Services modes and
per-category app lists). Distractor rows open plausible stub screens so a
wrong turn looks — and costs — like the real thing. Many goals are reachable
by more than one route (per-app page vs Privacy category), exactly like iOS.
"""
from __future__ import annotations

from ..core.app import App, register
from ..core.model import (Screen, Widget, w_banner, w_button, w_field, w_row,
                          w_section, w_text, w_toggle)
from ..core.state import DeviceState, Timer

OUR_APPS = ["TripSnap", "QuickPay", "Weatherly"]
STOCK_APPS = ["Safari", "Maps", "Music", "Notes", "Calendar", "Health",
              "FaceTime", "Translate", "Shortcuts",
              # the manual's per-app articles reference all of these
              "Mail", "Phone", "Messages", "Photos", "Contacts",
              "Voice Memos", "Journal", "Wallet", "Reminders", "Podcasts",
              "App Store", "Books", "Freeform", "News", "Passwords",
              "TV", "Find My"]
APP_ICON = {"TripSnap": "camera", "QuickPay": "pay", "Weatherly": "sun",
            "Safari": "wifi", "Maps": "pin", "Music": "vol", "Notes": "file",
            "Calendar": "bell", "Health": "batt", "FaceTime": "camera",
            "Translate": "chat", "Shortcuts": "apps"}
APP_COLOR = {"TripSnap": "#e0663e", "QuickPay": "#1f7a5a", "Weatherly": "#2f6bd4",
             "Safari": "#1f7ae0", "Maps": "#34c759", "Music": "#ff2d55",
             "Notes": "#e0b23e", "Calendar": "#ff3b30", "Health": "#ff2d55",
             "FaceTime": "#34c759", "Translate": "#5a8cf0", "Shortcuts": "#af52de"}
LOC_MODES = ["Never", "Ask Next Time Or When I Share", "While Using the App", "Always"]


def _deep_merge(dst, src):
    """Recursively fill dst with src's keys without overwriting existing ones."""
    for k, v in src.items():
        if isinstance(v, dict):
            _deep_merge(dst.setdefault(k, {}), v)
        else:
            dst.setdefault(k, v)



def _stub_nav(label):
    """on_tap for a row that leads to an informational placeholder page."""
    from . import settings_engine as SE
    return ("nav", SE.stub_target("settings.stub_none", label))


class SettingsApp(App):
    id = "settings"
    name = "Settings"
    icon = "gear"
    color = "#8e8e93"
    root_key = "settings.root"

    # ------------------------------------------------------------------ data
    def initial_data(self, rng) -> dict:
        apps = {}
        for n in OUR_APPS:
            apps[n] = {
                "perms": {"Camera": n != "Weatherly", "Microphone": n == "QuickPay",
                          "Contacts": True},
                "loc_mode": "While Using the App", "precise": True,
                "cellular": True, "bg": True, "track": True,
                "notif": {"allow": True, "sounds": True, "badges": True,
                          "time_sensitive": False},
            }
        for n in STOCK_APPS:
            apps[n] = {"perms": {}, "loc_mode": "While Using the App",
                       "precise": True, "cellular": True, "bg": n != "Notes",
                       "notif": {"allow": n not in ("Translate", "Shortcuts"),
                                 "sounds": True, "badges": True,
                                 "time_sensitive": False}}
        base = {
            "wifi": {"on": True, "network": "CoffeeBar_Guest", "status": "connected",
                     "known": ["Home_5G", "CoffeeBar_Guest"],
                     "ask_join": "Notify", "auto_hotspot": "Ask to Join",
                     "joining": None, "join_pw": ""},
            "bluetooth": {"on": False, "devices": ["Pixel Buds", "Car-4821"]},
            "cellular": {"data": True, "roaming": True, "mode": "Standard"},
            "hotspot": {"on": False, "name": "iPhone (Chris)", "password": "",
                        "maxcompat": False},
            "airplane": False,
            "brightness": 62, "dark": False, "auto_appearance": False,
            "true_tone": True, "night_shift": False, "bold": False,
            "auto_lock": "30 Seconds", "raise_wake": True,
            "low_power": False, "batt_health": 89, "optimized": True,
            "bg_global": "Wi-Fi & Cellular",
            "sound": {"ring": 70, "vibrate_ring": True, "vibrate_silent": False,
                      "keyboard_clicks": True},
            "storage": {"used_gb": 118, "total_gb": 128, "photos_gb": 61,
                        "apps_gb": 34, "system_gb": 14, "offloadable_gb": 9},
            "apps": apps,
        }
        from . import settings_data as SD
        for k, v in SD.SPEC_DEFAULTS.items():
            base.setdefault(k, {})
            _deep_merge(base[k], v)
        try:                                        # synthesized pages' state
            from . import settings_gen as SG
            base.setdefault("spec", {}).setdefault("gen", {})
            _deep_merge(base["spec"]["gen"], SG.GEN_DEFAULTS)
            for k, v in SG.GEN_CHOICES.items():
                SD.CHOICES.setdefault(k, v)
        except ImportError:
            pass
        return base

    # ------------------------------------------------------------------ view
    def view(self, st: DeviceState) -> Screen:
        full = st.cur_screen_key()
        key = full.split(".", 1)[1]
        d = st.data["settings"]
        # generated choice screens (declarative)
        from . import settings_engine as SE
        from . import settings_data as SD
        if full.startswith(SE.CHOICE_PREFIX):
            skey = full.split(":", 1)[1]
            return SE.build_choice_screen(d, skey, SD.CHOICES[skey],
                                          SD.CHOICE_TITLES.get(skey, ""))
        if key.startswith("app_"):
            return self._scr_app(st, d, key[4:])
        if key.startswith("loc_app_"):
            return self._scr_loc_app(st, d, key[8:])
        if key.startswith("notif_app_"):
            return self._scr_notif_app(st, d, key[10:])
        if key.startswith("gen_"):                 # a synthesized realistic page
            from . import settings_gen as SG
            spec = SG.GEN_SCREENS.get(key[4:])
            if spec:
                return SE.build_screen(self, d, full, spec)
        if key.startswith("stub_"):
            return self._scr_stub(st, d, key[5:])
        fn = getattr(self, "_scr_" + key, None)
        if fn:
            return fn(st, d)
        # declarative spec fallback — the whole manual lives here as data
        if key in SD.BUILDERS:
            spec = SD.BUILDERS[key](d)
            return SE.build_screen(self, d, full, spec)
        if key in SD.SCREENS:
            return SE.build_screen(self, d, full, SD.SCREENS[key])
        return self._scr_root(st, d)

    _SEARCH_INDEX = None

    @classmethod
    def _search_index(cls):
        """label -> screen key, for every Settings page reachable from the root (built once)."""
        if cls._SEARCH_INDEX is None:
            import os, sys
            import screensim.apps.inventory as inv
            inv.walk()
            idx = {}
            for key, path in inv.PAGE_PATH.items():
                label = path.split(" > ")[-1]
                idx.setdefault(label, key)
            cls._SEARCH_INDEX = idx
        return cls._SEARCH_INDEX

    def _scr_search(self, st, d, q):
        """What the Settings search shows for a query: the pages whose name matches."""
        ql = q.lower().replace(" ", "")
        hits = [(label, key) for label, key in self._search_index().items()
                if ql in label.lower().replace(" ", "")]
        hits.sort(key=lambda lk: (len(lk[0]), lk[0]))
        ws = [w_field("s_searchbar", "Search", value=q, hint="Search"), w_section("")]
        for label, key in hits[:8]:
            ws.append(w_row(f"sr_{key}", label, on_tap=("nav", key)))
        if not hits:
            ws.append(Widget("text", "sr_none", label="No Results", h=44))
        return Screen("settings.root", "Settings", ws)

    # ---- root (iOS 18 grouping) ----
    def _scr_root(self, st, d):
        wifi_sub = d["wifi"]["network"] if d["wifi"]["on"] else "Off"
        q = (d.get("_search") or "").strip()
        if q:
            return self._scr_search(st, d, q)
        ws = [
            w_field("s_searchbar", "Search", hint="Search"),
            Widget("account", "s_account", label="Chris Morgan",
                   sub="Apple Account, iCloud, and more", h=88,
                   on_tap=("nav", "settings.account")),
            # group: radios
            w_toggle("airplane_on", "Airplane Mode", d["airplane"], icon="airplane"),
            w_row("s_wifi", "Wi-Fi", sub=wifi_sub, icon="wifi",
                  on_tap=("nav", "settings.wifi_list")),
            w_row("s_bt", "Bluetooth", sub="On" if d["bluetooth"]["on"] else "Off",
                  icon="bt", on_tap=("nav", "settings.bluetooth")),
            w_row("s_cell", "Cellular", icon="cell",
                  on_tap=("nav", "settings.cellular")),
            w_row("s_hotspot", "Personal Hotspot",
                  sub="On" if d["hotspot"]["on"] else "Off", icon="hotspot",
                  on_tap=("nav", "settings.hotspot")),
            w_row("s_batt", "Battery", icon="batt",
                  on_tap=("nav", "settings.battery")),
            # group: system (General first, then alphabetical — iOS 18)
            w_row("s_general", "General", icon="gear",
                  on_tap=("nav", "settings.general")),
            w_row("s_access", "Accessibility", icon="person",
                  on_tap=("nav", "settings.stub_accessibility")),
            w_row("s_action", "Action Button", icon="apps",
                  on_tap=("nav", "settings.stub_actionbutton")),
            w_row("s_camera", "Camera", icon="camera",
                  on_tap=("nav", "settings.stub_camera")),
            w_row("s_cc", "Control Center", icon="apps",
                  on_tap=("nav", "settings.stub_controlcenter")),
            w_row("s_disp", "Display & Brightness", icon="sun",
                  on_tap=("nav", "settings.display")),
            w_row("s_home", "Home Screen & App Library", icon="apps",
                  on_tap=("nav", "settings.stub_homescreen")),
            w_row("s_search", "Search", icon="apps",
                  on_tap=("nav", "settings.stub_searchset")),
            w_row("s_siri", "Siri", icon="mic",
                  on_tap=("nav", "settings.stub_sirimain")),
            w_row("s_standby", "StandBy", icon="sun",
                  on_tap=("nav", "settings.stub_standby")),
            w_row("s_wall", "Wallpaper", icon="photo",
                  on_tap=("nav", "settings.stub_wallpaper")),
            # group: attention
            w_row("s_notif", "Notifications", icon="bell",
                  on_tap=("nav", "settings.notifications")),
            w_row("s_sound", "Sounds & Haptics", icon="vol",
                  on_tap=("nav", "settings.sounds")),
            w_row("s_focus", "Focus", icon="moon",
                  on_tap=("nav", "settings.stub_focus")),
            w_row("s_screentime", "Screen Time", icon="disk",
                  on_tap=("nav", "settings.stub_screentime")),
            # group: security
            w_row("s_faceid", "Face ID & Passcode", icon="lock",
                  on_tap=("nav", "settings.stub_faceid")),
            w_row("s_sos", "Emergency SOS", icon="bell",
                  on_tap=("nav", "settings.stub_sos")),
            w_row("s_priv", "Privacy & Security", icon="lock",
                  on_tap=("nav", "settings.privacy")),
            # group: services
            w_row("s_family", "Family", icon="person",
                  on_tap=("nav", "settings.family")),
            w_row("s_gamecenter", "Game Center", icon="apps",
                  on_tap=("nav", "settings.gamecenter")),
            w_row("s_wallet", "Wallet & Apple Pay", icon="pay",
                  on_tap=("nav", "settings.wallet")),
            # group: apps
            w_row("s_apps", "Apps", icon="apps", on_tap=("nav", "settings.apps")),
        ]
        return Screen("settings.root", "Settings", ws)

    # ---- Wi-Fi ----
    def _scr_wifi_list(self, st, d):
        w = d["wifi"]
        ws = [w_toggle("wifi_on", "Wi-Fi", w["on"], icon="wifi")]
        status = w.get("status", "connected")
        if w["on"] and w["network"]:
            sel = status != "connecting"
            ws.append(w_section(""))
            ws.append(w_row("net_current", w["network"],
                            sub="Connecting…" if status == "connecting" else "",
                            on_tap=("noop",), info=True,
                            **({"selected": True} if sel else {})))
        ws.append(w_section("My Networks"))
        for name in w["known"]:
            if name == w["network"] and w["on"]:
                continue
            ws.append(w_row(f"net_{name}", name, on_tap=("wifi_connect", name),
                            info=True, lock=True))
        ws.append(w_section("Other Networks"))
        for name, secured in [("Library_Public", False), ("SkyNet_2.4", True),
                              ("Verizon_5G_Home", True), ("HP-Print-a2", False)]:
            if name == w["network"] and w["on"]:
                continue
            ws.append(w_row(f"net_{name}", name, on_tap=("wifi_connect", name),
                            info=True, lock=secured))
        ws.append(w_section(""))
        ws.append(w_row("wifi_askjoin", "Ask to Join Networks", sub=w["ask_join"],
                        on_tap=("nav", "settings.stub_askjoin")))
        ws.append(w_row("wifi_autohot", "Auto-Join Hotspot", sub=w["auto_hotspot"],
                        on_tap=("nav", "settings.stub_autohotspot")))
        return Screen("settings.wifi_list", "Wi-Fi", ws)

    def _scr_wifi_join(self, st, d):
        j = d["wifi"].get("joining") or ""
        ws = [
            w_text("wj_head", f'Enter the password for “{j}”', h=48),
            w_field("wj_pw", "Password", value=d["wifi"].get("join_pw", ""),
                    hint="Password", mask=False),
            w_text("wj_note", "You can also access this Wi-Fi network by "
                   "bringing your iPhone near any iPhone, iPad, or Mac that "
                   "has connected to this network and has you in their "
                   "contacts.", h=88),
        ]
        scr = Screen("settings.wifi_join", "Enter Password", ws)
        scr.back_label = "Cancel"
        scr.appbar_actions = [Widget("icon_button", "wj_join", label="Join",
                                     on_tap=("wifi_join_go",), w=52,
                                     enabled=len(d["wifi"].get("join_pw", "")) >= 8)]
        return scr

    # ---- Bluetooth / Cellular ----
    def _scr_bluetooth(self, st, d):
        b = d["bluetooth"]
        ws = [w_toggle("bt_on", "Bluetooth", b["on"], icon="bt"),
              w_text("bt_note", "This iPhone is discoverable as “iPhone (Chris)” "
                     "while Bluetooth Settings is open.", h=56)]
        if b["on"]:
            ws.append(w_section("My Devices"))
            for dev in b["devices"]:
                ws.append(w_row(f"bt_{dev}", dev, sub="Not Connected",
                                on_tap=("noop",), info=True))
            ws.append(w_section("Other Devices"))
            ws.append(w_row("bt_scan", "Searching…", on_tap=("noop",)))
        return Screen("settings.bluetooth", "Bluetooth", ws)

    def _scr_cellular(self, st, d):
        c = d["cellular"]
        ws = [
            w_toggle("cell_data", "Cellular Data", c["data"]),
            w_row("cell_opts", "Cellular Data Options",
                  sub="Roaming " + ("On" if c["roaming"] else "Off"),
                  on_tap=("nav", "settings.cellular_options")),
            w_row("cell_hotspot", "Personal Hotspot",
                  sub="On" if d["hotspot"]["on"] else "Off",
                  on_tap=("nav", "settings.hotspot")),
            w_section("SIMs"),
            w_row("sim_primary", "Primary", sub="+1 (415) 555-0136 · On",
                  on_tap=("noop",)),
            w_row("sim_add", "Add eSIM", on_tap=("nav", "settings.stub_esim")),
            w_section("Cellular Data"),
            w_row("cell_usage", "Current Period", sub="14.2 GB", on_tap=("noop",)),
        ]
        for n in (OUR_APPS + STOCK_APPS)[:6]:
            a = d["apps"][n]
            ws.append(w_toggle(f"cellapp_{n}", n, a["cellular"]))
        return Screen("settings.cellular", "Cellular", ws)

    def _scr_cellular_options(self, st, d):
        c = d["cellular"]
        ws = [
            w_toggle("roaming_on", "Data Roaming", c["roaming"]),
            w_row("cell_mode", "Data Mode", sub=c["mode"],
                  on_tap=("nav", "settings.stub_datamode")),
            w_row("cell_voice", "Voice & Data", sub="5G Auto",
                  on_tap=("nav", "settings.stub_voicedata")),
            w_text("roam_note", "Turn off data roaming when traveling to avoid "
                   "charges when web browsing and using email and other data "
                   "services.", h=72),
        ]
        return Screen("settings.cellular_options", "Cellular Data Options", ws)

    # ---- Personal Hotspot ----
    def _scr_hotspot(self, st, d):
        h = d["hotspot"]
        ws = [
            w_text("hs_head", "Personal Hotspot on your iPhone can provide "
                   "internet access to other devices signed in to your Apple "
                   "Account without requiring you to enter the password.", h=76),
            w_toggle("hotspot_on", "Allow Others to Join", h["on"]),
            w_row("hs_pw_row", "Wi-Fi Password",
                  sub=("•" * len(h["password"])) if h["password"] else "Required",
                  on_tap=("nav", "settings.hotspot_pw")),
            w_text("hs_note", 'Other users can join your shared Wi-Fi network '
                   f'“{h["name"]}” using the password shown above.', h=56),
            w_toggle("maxcompat_on", "Maximize Compatibility", h["maxcompat"],
                     sub="Uses 2.4 GHz only. Internet performance may be reduced"),
            w_section("Family Sharing"),
            w_row("hs_family", "Family Sharing", sub="Off",
                  on_tap=("nav", "settings.stub_family")),
        ]
        return Screen("settings.hotspot", "Personal Hotspot", ws)

    # ---- Battery ----
    def _scr_hotspot_pw(self, st, d):
        ws = [w_field("hs_pw", "Wi-Fi Password",
                      value=d["hotspot"]["password"], hint="Password"),
              w_text("hspw_note", "The Wi-Fi password must be at least 8 "
                     "characters using ASCII characters.", h=52)]
        scr = Screen("settings.hotspot_pw", "Wi-Fi Password", ws)
        scr.back_label = "Cancel"
        scr.appbar_actions = [Widget("icon_button", "hspw_done", label="Done",
                                     on_tap=("save_hs_pw",), w=52,
                                     enabled=len(d["hotspot"]["password"]) >= 8)]
        return scr

    def _scr_battery(self, st, d):
        ws = [
            w_toggle("lowpower_on", "Low Power Mode", d["low_power"],
                     sub="Temporarily reduces background activity"),
            w_row("batt_health", "Battery Health & Charging",
                  sub=f"{d['batt_health']}% Capacity",
                  on_tap=("nav", "settings.batt_health")),
            w_section("Battery Level"),
            w_row("batt_now", f"{st.battery}%", sub="Last charged to 100%, 2 hr ago",
                  on_tap=("noop",)),
            w_section("Battery Usage by App"),
            w_row("bu_1", "Marketplace", sub="34%", on_tap=("noop",)),
            w_row("bu_2", "Safari", sub="21%", on_tap=("noop",)),
            w_row("bu_3", "Messages", sub="12%", on_tap=("noop",)),
        ]
        return Screen("settings.battery", "Battery", ws)

    def _scr_batt_health(self, st, d):
        ws = [
            w_row("bh_cap", "Maximum Capacity", sub=f"{d['batt_health']}%",
                  on_tap=("noop",)),
            w_toggle("optimized_on", "Optimized Battery Charging", d["optimized"],
                     sub="Reduces battery aging by learning your routine"),
            w_text("bh_note", "Your battery is currently supporting normal peak "
                   "performance.", h=52),
        ]
        return Screen("settings.batt_health", "Battery Health & Charging", ws)

    # ---- General ----
    def _scr_general(self, st, d):
        s = d["storage"]
        ws = [
            w_row("gen_about", "About", on_tap=("nav", "settings.stub_about")),
            w_row("gen_update", "Software Update",
                  on_tap=("nav", "settings.stub_update")),
            w_section(""),
            w_row("gen_airdrop", "AirDrop", on_tap=("nav", "settings.stub_airdrop")),
            w_row("gen_airplay", "AirPlay & Continuity",
                  on_tap=("nav", "settings.stub_airplay")),
            w_row("gen_pip", "Picture in Picture",
                  on_tap=("nav", "settings.stub_pip")),
            w_row("gen_carplay", "CarPlay", on_tap=("nav", "settings.stub_carplay")),
            w_section(""),
            w_row("gen_storage", "iPhone Storage",
                  sub=f"{s['used_gb']} GB of {s['total_gb']} GB Used",
                  on_tap=("nav", "settings.storage")),
            w_row("gen_bg", "Background App Refresh",
                  sub=d["bg_global"], on_tap=("nav", "settings.bg_refresh")),
            w_section(""),
            w_row("gen_date", "Date & Time", on_tap=("nav", "settings.stub_date")),
            w_row("gen_kbd", "Keyboard", on_tap=("nav", "settings.stub_keyboard")),
            w_row("gen_fonts", "Fonts", on_tap=("nav", "settings.stub_fonts")),
            w_row("gen_lang", "Language & Region",
                  on_tap=("nav", "settings.stub_language")),
            w_row("gen_dict", "Dictionary", on_tap=("nav", "settings.stub_dict")),
            w_row("gen_tvprov", "TV Provider",
                  on_tap=_stub_nav("TV Provider")),
            w_row("gen_gamectl", "Game Controller",
                  on_tap=("nav", "settings.game_controller")),
            w_row("gen_autofill", "AutoFill & Passwords",
                  on_tap=("nav", "settings.autofill")),
            w_section(""),
            w_row("gen_vpn", "VPN & Device Management",
                  on_tap=("nav", "settings.stub_vpn")),
            w_section(""),
            w_row("gen_legal", "Legal & Regulatory",
                  on_tap=("nav", "settings.stub_legal")),
            w_row("gen_reset", "Transfer or Reset iPhone",
                  on_tap=("nav", "settings.stub_reset")),
            w_row("gen_shutdown", "Shut Down", on_tap=("noop",)),
        ]
        return Screen("settings.general", "General", ws)

    def _scr_bg_refresh(self, st, d):
        ws = [w_row("bgg", "Background App Refresh", sub=d["bg_global"],
                    on_tap=("nav", "settings.bg_global")),
              w_text("bg_note", "Allow apps to refresh their content when on "
                     "Wi-Fi or cellular in the background. Turning off apps may "
                     "help preserve battery life.", h=72),
              w_section("")]
        for n in OUR_APPS + STOCK_APPS[:5]:
            a = d["apps"][n]
            ws.append(w_toggle(f"bgapp_{n}", n, a["bg"], icon=APP_ICON.get(n, "apps"),
                               enabled=d["bg_global"] != "Off"))
        return Screen("settings.bg_refresh", "Background App Refresh", ws)

    def _scr_bg_global(self, st, d):
        ws = []
        for opt in ["Off", "Wi-Fi", "Wi-Fi & Cellular"]:
            ws.append(Widget("radio", f"bgg_{opt}", label=opt,
                             value=d["bg_global"] == opt, h=48,
                             on_tap=("bg_global_set", opt)))
        return Screen("settings.bg_global", "Background App Refresh", ws)

    # ---- iPhone Storage ----
    def _scr_storage(self, st, d):
        s = d["storage"]
        segs = [("Apps", s["apps_gb"], "#5b8def"),
                ("Photos", s["photos_gb"], "#ffcc00"),
                ("System Data", s["system_gb"], "#8e8e93")]
        ws = [
            Widget("usagebar", "store_bar", h=96,
                   label=f"{s['used_gb']} GB used", sub=f"of {s['total_gb']} GB",
                   meta={"segments": segs, "total": s["total_gb"]}),
            w_section("Recommendations"),
            w_row("store_offload", "Offload Unused Apps",
                  sub=f"Save {s['offloadable_gb']} GB", on_tap=("offload_apps",)),
            w_row("store_attach", "Review Large Attachments", sub="Save 1.9 GB",
                  on_tap=("nav", "settings.stub_attachments")),
            w_row("store_dupes", "Review Duplicates", sub="Save 1.1 GB",
                  on_tap=("nav", "settings.stub_dupes")),
            w_section(""),
            w_row("st_1", "Photos", sub=f"{s['photos_gb']} GB",
                  on_tap=("nav", "photos.root")),
            w_row("st_2", "Marketplace", sub="8.4 GB", on_tap=("noop",)),
            w_row("st_3", "Music", sub="6.1 GB", on_tap=("noop",)),
            w_row("st_4", "Safari", sub="2.3 GB", on_tap=("noop",)),
        ]
        return Screen("settings.storage", "iPhone Storage", ws)

    # ---- Display & Brightness ----
    def _scr_display(self, st, d):
        ws = [
            w_section("Appearance"),
            Widget("radio", "appear_light", label="Light", value=not d["dark"],
                   h=48, on_tap=("set_appearance", "light")),
            Widget("radio", "appear_dark", label="Dark", value=d["dark"],
                   h=48, on_tap=("set_appearance", "dark")),
            w_toggle("auto_appear_on", "Automatic", d["auto_appearance"]),
            w_section("Text"),
            w_row("disp_textsize", "Text Size",
                  on_tap=("nav", "settings.stub_textsize")),
            # the same setting as Accessibility > Display & Text Size > Bold Text, as on iOS
            w_toggle("bold_on", "Bold Text", d["spec"]["accessibility"]["bold"]),
            w_section("Brightness"),
            Widget("slider", "brightness", label="Brightness", value=d["brightness"], h=52,
                   on_swipe={"right": ("slider_step", "brightness", +25),
                             "left": ("slider_step", "brightness", -25)}),
            w_toggle("truetone_on", "True Tone", d["true_tone"],
                     sub="Automatically adapt display based on lighting"),
            w_section(""),
            w_row("disp_nightshift", "Night Shift", sub="Off",
                  on_tap=("nav", "settings.stub_nightshift")),
            w_row("disp_autolock", "Auto-Lock", sub=d["auto_lock"],
                  on_tap=("nav", "settings.autolock")),
            w_toggle("raisewake_on", "Raise to Wake", d["raise_wake"]),
            w_row("disp_zoom", "Display Zoom", sub=d["spec"]["disp"].get("display_zoom", "Default"),
                  on_tap=("nav", "settings.__choice__:spec.disp.display_zoom")),
        ]
        return Screen("settings.display", "Display & Brightness", ws)

    def _scr_autolock(self, st, d):
        ws = []
        for opt in ["30 Seconds", "1 Minute", "2 Minutes", "3 Minutes",
                    "5 Minutes", "Never"]:
            ws.append(Widget("radio", f"al_{opt}", label=opt,
                             value=d["auto_lock"] == opt, h=48,
                             on_tap=("autolock_set", opt)))
        return Screen("settings.autolock", "Auto-Lock", ws)

    # ---- Sounds & Haptics ----
    def _scr_sounds(self, st, d):
        s = d["sound"]
        ws = [
            w_section("Vibrate"),
            w_toggle("vibring_on", "Vibrate on Ring", s["vibrate_ring"]),
            w_toggle("vibsilent_on", "Vibrate on Silent", s["vibrate_silent"]),
            w_section("Ringtone and Alert Volume"),
            Widget("slider", "ring_vol", label="Ringtone and Alert Volume", value=s["ring"], h=52,
                   on_swipe={"right": ("slider_step", "sound.ring", +25),
                             "left": ("slider_step", "sound.ring", -25)}),
            w_section("Sounds"),
            w_row("snd_ringtone", "Ringtone", sub="Reflection",
                  on_tap=("nav", "settings.stub_ringtone")),
            w_row("snd_text", "Text Tone", sub="Note",
                  on_tap=("nav", "settings.stub_texttone")),
            w_toggle("kbd_clicks_on", "Keyboard Clicks", s["keyboard_clicks"]),
            w_section(""),
            w_row("snd_hpsafety", "Headphone Safety",
                  on_tap=("nav", "settings.headphone_safety")),
            w_row("snd_vollimit", "Volume Limit", sub="Off",
                  on_tap=_stub_nav("Volume Limit")),
        ]
        return Screen("settings.sounds", "Sounds & Haptics", ws)

    # ---- Notifications ----
    def _scr_notifications(self, st, d):
        ws = [w_row("no_style", "Display As", sub="Stack",
                    on_tap=("nav", "settings.stub_notifstyle")),
              w_row("no_sched", "Scheduled Summary", sub="Off",
                    on_tap=("nav", "settings.stub_summary")),
              w_row("no_journal", "Journaling Suggestions",
                    on_tap=("nav", "settings.journaling")),
              w_section("Notification Style")]
        for n in OUR_APPS + STOCK_APPS[:5] + ["Mail", "Messages"]:
            a = d["apps"][n]["notif"]
            sub = ("Banners, Sounds, Badges" if a["allow"] and a["sounds"] and a["badges"]
                   else "Banners, Badges" if a["allow"] and a["badges"]
                   else "Banners" if a["allow"] else "Off")
            ws.append(w_row(f"no_{n}", n, sub=sub,
                            on_tap=("nav", f"settings.notif_app_{n}"),
                            icon=APP_ICON.get(n, "apps")))
        return Screen("settings.notifications", "Notifications", ws)

    def _scr_notif_app(self, st, d, name):
        a = d["apps"][name]["notif"]
        ws = [w_toggle(f"nallow_{name}_on", "Allow Notifications", a["allow"])]
        if a["allow"]:
            ws += [
                w_toggle(f"nts_{name}_on", "Time-Sensitive Notifications",
                         a["time_sensitive"]),
                w_section("Alerts"),
                Widget("alerttiles", f"na_alerts_{name}", h=112,
                       meta={"lock": True, "nc": True, "banner": True}),
                w_row("na_style", "Banner Style", sub="Temporary",
                      on_tap=("nav", "settings.stub_bannerstyle")),
                w_toggle(f"nsounds_{name}_on", "Sounds", a["sounds"]),
                w_toggle(f"nbadges_{name}_on", "Badges", a["badges"]),
                w_section("Options"),
                w_row("na_preview", "Show Previews", sub="When Unlocked",
                      on_tap=("nav", "settings.stub_previews")),
                w_row("na_group", "Notification Grouping", sub="Automatic",
                      on_tap=("nav", "settings.stub_grouping")),
                w_row("na_cust", "Customize Notifications", on_tap=("noop",)),
            ]
        return Screen(f"settings.notif_app_{name}", name, ws)

    # ---- Privacy & Security ----
    def _scr_privacy(self, st, d):
        ws = [
            w_section(""),
            w_row("pv_loc", "Location Services", sub="On", icon="pin",
                  on_tap=("nav", "settings.loc_services")),
            w_row("pv_track", "Tracking", sub="",
                  on_tap=("nav", "settings.stub_tracking")),
            w_section(""),
            w_row("pv_contacts", "Contacts", icon="person",
                  on_tap=("nav", "settings.stub_pv_contacts")),
            w_row("pv_cal", "Calendars", icon="bell",
                  on_tap=("nav", "settings.stub_pv_cal")),
            w_row("pv_photos", "Photos", icon="photo",
                  on_tap=("nav", "settings.stub_pv_photos")),
            w_row("pv_mic", "Microphone", icon="mic",
                  on_tap=("nav", "settings.priv_mic")),
            w_row("pv_cam", "Camera", icon="camera",
                  on_tap=("nav", "settings.priv_cam")),
            w_row("pv_health", "Health", icon="batt",
                  on_tap=("nav", "settings.stub_pv_health")),
            w_row("pv_bt", "Bluetooth", icon="bt",
                  on_tap=("nav", "settings.stub_pv_bt")),
            w_row("pv_localnet", "Local Network",
                  on_tap=("nav", "settings.stub_pv_localnet")),
            w_section("Security"),
            w_row("pv_appreport", "App Privacy Report",
                  on_tap=("nav", "settings.stub_appreport")),
            w_row("pv_analytics", "Analytics & Improvements",
                  on_tap=("nav", "settings.analytics")),
            w_row("pv_ads", "Apple Advertising",
                  on_tap=("nav", "settings.apple_ads")),
            w_row("pv_safety", "Safety Check",
                  on_tap=_stub_nav("Safety Check")),
            w_row("pv_lockdown", "Lockdown Mode",
                  on_tap=("nav", "settings.lockdown")),
            w_row("pv_journal", "Journaling Suggestions",
                  on_tap=("nav", "settings.journaling")),
        ]
        return Screen("settings.privacy", "Privacy & Security", ws)

    def _scr_loc_services(self, st, d):
        ws = [w_toggle("locsvc_on", "Location Services",
                       d.get("loc_master", True)),
              w_text("loc_note", "Location Services uses GPS, Bluetooth and "
                     "crowd-sourced Wi-Fi hotspot and cell tower locations to "
                     "determine your approximate location.", h=72),
              w_section("")]
        for n in OUR_APPS + ["Maps", "Safari", "Find My"]:
            a = d["apps"][n]
            mode = a["loc_mode"]
            short = {"Never": "Never", "Ask Next Time Or When I Share": "Ask",
                     "While Using the App": "While Using", "Always": "Always"}[mode]
            ws.append(w_row(f"ls_{n}", n, sub=short, icon=APP_ICON.get(n, "apps"),
                            on_tap=("nav", f"settings.loc_app_{n}")))
        return Screen("settings.loc_services", "Location Services", ws)

    def _scr_loc_app(self, st, d, name):
        a = d["apps"][name]
        ws = [w_section("Allow Location Access")]
        for opt in LOC_MODES:
            ws.append(Widget("radio", f"lm_{name}_{opt[:6]}", label=opt,
                             value=a["loc_mode"] == opt, h=48,
                             on_tap=("loc_mode_set", name, opt)))
        ws.append(w_text("lm_note", f'App explanation: “{name} uses your location '
                         'to tag photos and improve suggestions.”', h=56))
        ws.append(w_toggle(f"precise_{name}_on", "Precise Location", a["precise"],
                           sub="Allows the app to use your specific location"))
        return Screen(f"settings.loc_app_{name}", name, ws)

    def _scr_priv_mic(self, st, d):
        ws = [w_text("pm_note", "Applications that have requested access to the "
                     "microphone will appear here.", h=52)]
        for n in OUR_APPS:
            a = d["apps"][n]
            if "Microphone" in a["perms"]:
                ws.append(w_toggle(f"privmic_{n}_on", n, a["perms"]["Microphone"],
                                   icon=APP_ICON.get(n, "apps")))
        return Screen("settings.priv_mic", "Microphone", ws)

    def _scr_priv_cam(self, st, d):
        ws = [w_text("pc_note", "Applications that have requested access to the "
                     "camera will appear here.", h=52)]
        for n in OUR_APPS:
            a = d["apps"][n]
            if "Camera" in a["perms"]:
                ws.append(w_toggle(f"privcam_{n}_on", n, a["perms"]["Camera"],
                                   icon=APP_ICON.get(n, "apps")))
        return Screen("settings.priv_cam", "Camera", ws)

    # ---- Apps ----
    def _scr_apps(self, st, d):
        q = (d.get("_apps_search") or "").strip().lower()
        ws = [w_field("apps_search", "Search Apps", value=d.get("_apps_search", ""), hint="Search Apps"),
              w_section("")]
        names = [n for n in sorted(d["apps"].keys()) if not q or q in n.lower()]
        if not names:
            ws.append(w_text("apps_none", "No Results", h=48))
        for n in names:
            ws.append(w_row(f"app_{n}", n, icon=APP_ICON.get(n, "apps"),
                            on_tap=("nav", f"settings.app_{n}")))
        return Screen("settings.apps", "Apps", ws)

    def _scr_app(self, st, d, name):
        a = d["apps"][name]
        picon = {"Camera": "camera", "Microphone": "mic", "Contacts": "person"}
        loc_short = {"Never": "Never", "Ask Next Time Or When I Share": "Ask",
                     "While Using the App": "While Using", "Always": "Always"}
        ws = [
            Widget("apphdr", f"hdr_{name}", label=name, sub="Version 3.2",
                   icon=APP_ICON.get(name, "apps"), h=104,
                   meta={"color": APP_COLOR.get(name, "#5b6b8c")}),
            w_section(f"Allow {name} to Access"),
            w_row(f"aloc_{name}", "Location", sub=loc_short[a["loc_mode"]],
                  icon="pin", on_tap=("nav", f"settings.loc_app_{name}")),
        ]
        for p, on in a["perms"].items():
            ws.append(w_toggle(f"perm_{name}_{p}", p, on, icon=picon.get(p, "")))
        ws.append(w_toggle(f"bgapp2_{name}", "Background App Refresh", a["bg"]))
        ws.append(w_toggle(f"cellapp2_{name}", "Cellular Data", a["cellular"]))
        ws.append(w_section(""))
        ws.append(w_row("app_siri", "Siri & Search",
                        on_tap=("nav", "settings.stub_siri")))
        ws.append(w_row(f"anotif_{name}", "Notifications",
                        sub="Banners, Sounds, Badges" if a["notif"]["allow"] else "Off",
                        on_tap=("nav", f"settings.notif_app_{name}")))
        # the app's OWN settings below the permission block, spec-driven —
        # real iOS per-app pages carry these (Mail's Preview, Music's EQ, …)
        from . import settings_data as SD
        from . import settings_engine as SE
        for sec_spec in SD.APP_EXTRAS.get(name, []):
            ws.append(w_section(sec_spec.get("header", "")))
            for r in sec_spec.get("rows", []):
                ws.extend(SE._row_widgets(self, d, f"settings.app_{name}", r))
        return Screen(f"settings.app_{name}", name, ws)

    # ---- stubs (plausible dead-end screens) ----
    STUBS = {
        "vpn": ("VPN & Device Management",
                [("VPN", "Not Connected"), ("Sign In to Work or School Account…", "")]),
        "accessibility": ("Accessibility",
                          [("VoiceOver", "Off"), ("Zoom", "Off"), ("Display & Text Size", ""),
                           ("Motion", ""), ("Touch", ""), ("Face ID & Attention", "")]),
        "camera": ("Camera", [("Formats", ""), ("Record Video", "1080p at 30 fps"),
                              ("Grid", "Off"), ("Mirror Front Camera", "Off")]),
        "controlcenter": ("Control Center",
                          [("Show When Locked", "On"), ("Access Within Apps", "On")]),
        "homescreen": ("Home Screen & App Library",
                       [("Newly Downloaded Apps", "Add to Home Screen"),
                        ("Show in App Library", "On")]),
        "wallpaper": ("Wallpaper", [("Add New Wallpaper", "")]),
        "focus": ("Focus", [("Do Not Disturb", ""), ("Sleep", ""), ("Work", ""),
                            ("Share Across Devices", "On")]),
        "screentime": ("Screen Time", [("Daily Average", "4 h 12 min"),
                                       ("App Limits", ""), ("Downtime", "")]),
        "faceid": ("Face ID & Passcode", [("iPhone Unlock", "On"),
                                          ("Wallet & Apple Pay", "On"),
                                          ("Require Attention for Face ID", "On")]),
        "sos": ("Emergency SOS", [("Call with Hold and Release", "On"),
                                  ("Call Quietly", "Off")]),
        "about": ("About", [("Name", "iPhone (Chris)"), ("iOS Version", "18.6"),
                            ("Model Name", "iPhone 16 Pro"), ("Serial Number", "F2LX…"),
                            ("Coverage", "Expires 3/12/27")]),
        "update": ("Software Update", [("iOS 18.6", "Your iPhone is up to date"),
                                       ("Automatic Updates", "On")]),
        "airdrop": ("AirDrop", [("Receiving Off", ""), ("Contacts Only", "✓"),
                                ("Everyone for 10 Minutes", "")]),
        "airplay": ("AirPlay & Continuity", [("Automatically AirPlay", "Ask"),
                                             ("Handoff", "On")]),
        "pip": ("Picture in Picture", [("Start PiP Automatically", "On")]),
        "carplay": ("CarPlay", [("Available Cars", "")]),
        "date": ("Date & Time", [("24-Hour Time", "Off"), ("Set Automatically", "On"),
                                 ("Time Zone", "Cupertino")]),
        "keyboard": ("Keyboard", [("Keyboards", "2"), ("Text Replacement", ""),
                                  ("Auto-Correction", "On"), ("Enable Dictation", "On")]),
        "fonts": ("Fonts", [("No Fonts Installed", "")]),
        "language": ("Language & Region", [("iPhone Language", "English (US)"),
                                           ("Temperature", "°F")]),
        "dict": ("Dictionary", [("English (US)", "✓")]),
        "legal": ("Legal & Regulatory", [("Legal Notices", ""), ("License", "")]),
        "reset": ("Transfer or Reset iPhone",
                  [("Prepare for New iPhone", ""), ("Reset", ""),
                   ("Erase All Content and Settings", "")]),
        "askjoin": ("Ask to Join Networks", [("Off", ""), ("Notify", "✓"), ("Ask", "")]),
        "autohotspot": ("Auto-Join Hotspot", [("Never", ""), ("Ask to Join", "✓"),
                                              ("Automatic", "")]),
        "esim": ("Add eSIM", [("Use QR Code", ""), ("Transfer From Nearby iPhone", "")]),
        "datamode": ("Data Mode", [("Allow More Data on 5G", ""), ("Standard", "✓"),
                                   ("Low Data Mode", "")]),
        "voicedata": ("Voice & Data", [("5G Auto", "✓"), ("5G On", ""), ("LTE", "")]),
        "family": ("Family Sharing", [("Set Up Family Sharing", "")]),
        "textsize": ("Text Size", [("Drag the slider below", "")]),
        "nightshift": ("Night Shift", [("Scheduled", "Off"),
                                       ("Manually Enable Until Tomorrow", "Off")]),
        "ringtone": ("Ringtone", [("Reflection", "✓"), ("Radar", ""), ("Chimes", "")]),
        "texttone": ("Text Tone", [("Note", "✓"), ("Aurora", ""), ("Chord", "")]),
        "notifstyle": ("Display As", [("Count", ""), ("Stack", "✓"), ("List", "")]),
        "summary": ("Scheduled Summary", [("Scheduled Summary", "Off")]),
        "bannerstyle": ("Banner Style", [("Temporary", "✓"), ("Persistent", "")]),
        "previews": ("Show Previews", [("Always", ""), ("When Unlocked", "✓"),
                                       ("Never", "")]),
        "tracking": ("Tracking", [("Allow Apps to Request to Track", "Off")]),
        "pv_contacts": ("Contacts", [("Safari", "On"), ("QuickPay", "On")]),
        "pv_cal": ("Calendars", [("No apps have requested access", "")]),
        "pv_photos": ("Photos", [("Marketplace", "Limited"), ("Messages", "Full Access")]),
        "pv_health": ("Health", [("No apps have requested access", "")]),
        "pv_bt": ("Bluetooth", [("Music", "On")]),
        "pv_localnet": ("Local Network", [("HP Smart", "On")]),
        "appreport": ("App Privacy Report", [("Turn On App Privacy Report", "")]),
        "attachments": ("Large Attachments", [("IMG_2041.MOV", "812 MB"), ("Keynote.key", "214 MB")]),
        "analytics": ("Analytics & Improvements", [("Share iPhone Analytics", "Off")]),
        "offload": ("Offload Unused Apps", [("Enable", "Automatically offload unused apps")]),
        "dupes": ("Review Duplicates", [("2 duplicates", "1.1 GB")]),
        "actionbutton": ("Action Button", [("Choose an action", "Flashlight")]),
        "searchset": ("Search", [("Show Recent Searches", "On"),
                                 ("Show Related Content", "On")]),
        "sirimain": ("Siri", [("Talk to Siri", "\u201cHey Siri\u201d"),
                              ("Siri Responses", ""), ("Announce Calls", "Never")]),
        "standby": ("StandBy", [("StandBy", "On"), ("Night Mode", "On")]),
        "siri": ("Siri & Search", [("Learn from this App", "On"),
                                   ("Show App in Search", "On"),
                                   ("Suggest App", "On")]),
    }

    def _scr_stub(self, st, d, key):
        # a "stub" target may now resolve to a full declarative screen from the
        # manual — prefer that over the flat placeholder table.
        from . import settings_engine as SE
        from . import settings_data as SD
        skey = SD.STUB_ALIAS.get(key, key)
        if skey in SD.SCREENS:
            return SE.build_screen(self, d, f"settings.{key}", SD.SCREENS[skey])
        if key.startswith("none__"):            # a per-row placeholder page
            label = SE.STUB_LABELS.get(key[6:], key[6:].replace("_", " ").title())
            return Screen(f"settings.stub_{key}", label,
                          [w_text(f"stub_{key[6:]}_note",
                                  "There are no options to change here.",
                                  sub="This page is informational on this phone.")])
        title, rows = self.STUBS.get(key, (key.title(), []))
        ws = []
        for label, sub in rows:
            ws.append(w_row(f"stub_{key}_{label[:12]}", label, sub=sub,
                            on_tap=("noop",)))
        return Screen(f"settings.stub_{key}", title, ws)

    # ---------------------------------------------------------------- events
    def handle(self, st, event, gesture):
        d = st.data["settings"]
        k = event[0]
        # ---- generic declarative handlers (state paths) ----
        from . import settings_engine as SE
        if k == "sslider":
            cur = int(SE.sget(d, event[1], 0) or 0)
            SE.sset(d, event[1], max(0, min(100, cur + int(event[2]))))
            self._spec_side_effects(st, d, event[1])
            return None
        if k == "slider_step":                  # hand-built sliders keyed by a top-level path
            cur = int(SE.sget(d, event[1], 0) or 0)
            SE.sset(d, event[1], max(0, min(100, cur + int(event[2]))))
            return None
        if k == "sset_toggle":
            SE.sset(d, event[1], not SE.sget(d, event[1]))
            self._spec_side_effects(st, d, event[1])
            return None
        if k == "schoice":
            # like real iOS: selecting a radio checkmarks it and stays put —
            # the user taps Back themselves
            SE.sset(d, event[1], event[2])
            self._spec_side_effects(st, d, event[1])
            return None
        if k == "sset_field":
            SE.sset(d, event[1], event[2])
            return None
        if k == "saction":
            return self._spec_action(st, d, event[1])
        if k == "toggle":
            return self._toggle(st, d, event[1])
        if k == "set_field":
            return self._set_field(st, d, event[1], event[2])
        if k == "set_appearance":
            d["dark"] = event[1] == "dark"
            return None
        if k == "loc_mode_set":
            d["apps"][event[1]]["loc_mode"] = event[2]
            return None
        if k == "autolock_set":
            d["auto_lock"] = event[1]
            return None
        if k == "bg_global_set":
            d["bg_global"] = event[1]
            return None
        if k == "wifi_connect":
            name = event[1]
            secured = name in ("SkyNet_2.4", "Verizon_5G_Home")
            known = name in d["wifi"]["known"]
            if secured and not known:
                d["wifi"]["joining"] = name
                d["wifi"]["join_pw"] = ""
                st.push("settings.wifi_join")
                return None
            return self._connect(st, d, name)
        if k == "wifi_join_go":
            pw = d["wifi"].get("join_pw", "")
            if len(pw) < 8:
                return "Wi-Fi passwords are at least 8 characters"
            name = d["wifi"].get("joining")
            st.pop()
            return self._connect(st, d, name)
        if k == "wifi_connected":
            d["wifi"]["status"] = "connected"
            return None
        if k == "nav_back":
            st.pop()
            return None
        if k == "save_hs_pw":
            if len(d["hotspot"]["password"]) >= 8:
                st.pop()
            return None
        if k == "offload_apps":
            st.dialog = {
                "screen": "settings.dlg_offload",
                "title": "Offload Unused Apps?",
                "body": "Automatically offload unused apps when you're low on "
                        "storage. Your documents and data will be saved.",
                "buttons": [("Enable", ("do_offload",), "filled"),
                            ("Cancel", ("close_dialog",), "text")],
                "dismissable": True,
            }
            return None
        if k == "do_offload":
            freed = d["storage"]["offloadable_gb"]
            d["storage"]["offloadable_gb"] = 0
            d["storage"]["apps_gb"] -= freed
            d["storage"]["used_gb"] -= freed
            d["storage"]["offload_on"] = True
            st.dialog = None
            return None
        return None

    def _connect(self, st, d, name):
        d["wifi"]["on"] = True
        d["wifi"]["network"] = name
        d["wifi"]["status"] = "connecting"
        if name not in d["wifi"]["known"]:
            d["wifi"]["known"].append(name)
        st.net = "wifi"
        st.timers = [t for t in st.timers if t.id != "wifi_connect"]
        st.timers.append(Timer("wifi_connect", st.tick + 3,
                               on_expire=("wifi_connected",), label="",
                               visible=False))
        return None

    def _toggle(self, st, d, wid):
        a = d["apps"]
        if wid == "locsvc_on":
            d["loc_master"] = not d.get("loc_master", True)
        elif wid in ("wifi_on", "wifi_on2"):
            d["wifi"]["on"] = not d["wifi"]["on"]
            st.net = "wifi" if d["wifi"]["on"] else "cellular"
        elif wid == "airplane_on":
            d["airplane"] = not d["airplane"]
            if d["airplane"]:
                d["wifi"]["on"] = False
                st.net = "none"
        elif wid == "hotspot_on":
            if not d["hotspot"]["password"]:
                st.dialog = {
                    "screen": "settings.dlg_hs",
                    "title": "Set a Password First",
                    "body": "Your Personal Hotspot needs a Wi-Fi password before "
                            "others can join.",
                    "buttons": [("OK", ("close_dialog",), "filled")],
                    "dismissable": True,
                }
                return None
            d["hotspot"]["on"] = not d["hotspot"]["on"]
        elif wid == "maxcompat_on":
            d["hotspot"]["maxcompat"] = not d["hotspot"]["maxcompat"]
        elif wid == "bt_on":
            d["bluetooth"]["on"] = not d["bluetooth"]["on"]
        elif wid == "cell_data":
            d["cellular"]["data"] = not d["cellular"]["data"]
        elif wid == "roaming_on":
            d["cellular"]["roaming"] = not d["cellular"]["roaming"]
        elif wid == "lowpower_on":
            d["low_power"] = not d["low_power"]
        elif wid == "optimized_on":
            d["optimized"] = not d["optimized"]
        elif wid == "auto_appear_on":
            d["auto_appearance"] = not d["auto_appearance"]
        elif wid == "truetone_on":
            d["true_tone"] = not d["true_tone"]
        elif wid == "bold_on":
            a = d["spec"]["accessibility"]
            a["bold"] = not a["bold"]
        elif wid == "raisewake_on":
            d["raise_wake"] = not d["raise_wake"]
        elif wid == "vibring_on":
            d["sound"]["vibrate_ring"] = not d["sound"]["vibrate_ring"]
        elif wid == "vibsilent_on":
            d["sound"]["vibrate_silent"] = not d["sound"]["vibrate_silent"]
        elif wid == "kbd_clicks_on":
            d["sound"]["keyboard_clicks"] = not d["sound"]["keyboard_clicks"]
        elif wid.startswith("bgapp_"):
            n = wid[len("bgapp_"):]
            a[n]["bg"] = not a[n]["bg"]
        elif wid.startswith("bgapp2_"):
            n = wid[len("bgapp2_"):]
            a[n]["bg"] = not a[n]["bg"]
        elif wid.startswith("cellapp2_"):
            n = wid[len("cellapp2_"):]
            a[n]["cellular"] = not a[n]["cellular"]
        elif wid.startswith("cellapp_"):
            n = wid[len("cellapp_"):]
            a[n]["cellular"] = not a[n]["cellular"]
        elif wid.startswith("perm_"):
            _, name, perm = wid.split("_", 2)
            a[name]["perms"][perm] = not a[name]["perms"][perm]
        elif wid.startswith("privmic_"):
            n = wid[len("privmic_"):-3]
            a[n]["perms"]["Microphone"] = not a[n]["perms"]["Microphone"]
        elif wid.startswith("privcam_"):
            n = wid[len("privcam_"):-3]
            a[n]["perms"]["Camera"] = not a[n]["perms"]["Camera"]
        elif wid.startswith("precise_"):
            n = wid[len("precise_"):-3]
            a[n]["precise"] = not a[n]["precise"]
        elif wid.startswith("nallow_"):
            n = wid[len("nallow_"):-3]
            a[n]["notif"]["allow"] = not a[n]["notif"]["allow"]
        elif wid.startswith("nsounds_"):
            n = wid[len("nsounds_"):-3]
            a[n]["notif"]["sounds"] = not a[n]["notif"]["sounds"]
        elif wid.startswith("nbadges_"):
            n = wid[len("nbadges_"):-3]
            a[n]["notif"]["badges"] = not a[n]["notif"]["badges"]
        elif wid.startswith("nts_"):
            n = wid[len("nts_"):-3]
            a[n]["notif"]["time_sensitive"] = not a[n]["notif"]["time_sensitive"]
        elif wid == "locsvc_on":
            pass  # global location toggle guarded (would break every app)
        return None

    def _set_field(self, st, d, fid, text):
        if fid.startswith("spec::"):
            from . import settings_engine as SE
            SE.sset(d, fid[6:], text)
            return None
        if fid == "apps_search":
            d["_apps_search"] = text            # the Apps list filters on it
        elif fid == "s_searchbar":
            d["_search"] = text                 # the Settings root lists the pages that match
        elif fid == "hs_pw":
            d["hotspot"]["password"] = text
        elif fid == "wj_pw":
            d["wifi"]["join_pw"] = text
        return None

    # ---- spec hooks ----
    def _spec_side_effects(self, st, d, skey):
        """A few declarative toggles have real consequences (mirroring iOS)."""
        if skey == "airplane" and d.get("airplane"):
            d.setdefault("wifi", {})["on"] = False
            st.net = "none"
        return None

    def _spec_action(self, st, d, event):
        from . import settings_data as SD
        fn = SD.ACTIONS.get(event[0] if isinstance(event, (list, tuple)) else event)
        if fn:
            return fn(st, d, event)
        return None


register(SettingsApp())
