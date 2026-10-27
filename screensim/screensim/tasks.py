"""Task library v2 — long-horizon, choice-heavy, realistic iOS tasks.

Each task mirrors a real tutorial-class problem ("things people search for"),
runs 5-15 gestures deep through dense screens where the right row is one of
many, and often has a TRAP (a plausible-looking wrong option that breaks the
goal) plus an ALTERNATE legitimate route (like the real iPhone). `alt_plans`
lists other valid routes; goal predicates check the trap didn't fire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Task:
    id: str
    family: str
    instruction: str
    plan: list
    goal: Callable
    instruction_label: str = ""      # the older label-level wording, kept for ablations
    start_app: str = "home"
    seed: int = 0
    urgency: str = ""
    hazard: str = ""
    mirrors: str = ""
    trap: str = ""                      # the plausible wrong turn
    band: int = 0                       # complexity band for the RL library (0 = the benchmark set)
    alt_plans: list = field(default_factory=list)   # [(label, plan)]


TASKS: list = []


def task(**kw):
    t = Task(**kw)
    TASKS.append(t)
    return t


def S(st):
    return st.data["settings"]


def g_paths(required):
    """Goal = every one of these state paths holds this value. The paths and values come from a
    verified replay of the task's own plan (tools/build_rl_tasks.py), so the predicate cannot drift
    from what the reference solution actually does."""
    def f(st):
        bad = []
        for path, want in required.items():
            cur = st.data
            for k in path.split("."):
                cur = cur[k] if isinstance(cur, dict) and k in cur else "<missing>"
            if cur != want:
                bad.append(f"{path}={cur!r} wanted {want!r}")
        return (not bad), ("; ".join(bad) if bad else " ".join(f"{k}={v!r}" for k, v in required.items()))
    return f


# ---------------------------------------------------------------- goals
def g_hotspot_ready(st):
    h = S(st)["hotspot"]
    ok = h["on"] and h["password"] == "trailmix88" and h["maxcompat"]
    return ok, f"on={h['on']} pw={h['password']!r} maxcompat={h['maxcompat']}"


def g_loc_never(app):
    def f(st):
        m = S(st)["apps"][app]["loc_mode"]
        return m == "Never", f"{app}.loc_mode={m}"
    return f


def g_precise_off_keep_mode(app):
    def f(st):
        a = S(st)["apps"][app]
        ok = (not a["precise"]) and a["loc_mode"] == "While Using the App"
        return ok, f"precise={a['precise']} mode={a['loc_mode']}"
    return f


def g_mic_off(app):
    def f(st):
        v = S(st)["apps"][app]["perms"].get("Microphone")
        return v is False, f"{app}.Microphone={v}"
    return f


def g_bg_off_only(app):
    def f(st):
        d = S(st)
        others = [n for n in d["apps"] if n != app and "bg" in d["apps"][n]]
        ok = (not d["apps"][app]["bg"]) and d["bg_global"] != "Off" \
            and all(d["apps"][n]["bg"] or n in ("Notes",) for n in others)
        return ok, f"{app}.bg={d['apps'][app]['bg']} global={d['bg_global']}"
    return f


def g_autolock(val):
    def f(st):
        return S(st)["auto_lock"] == val, f"auto_lock={S(st)['auto_lock']}"
    return f


def g_notif_sounds_off_keep_allow(app):
    def f(st):
        n = S(st)["apps"][app]["notif"]
        return n["allow"] and not n["sounds"], \
            f"allow={n['allow']} sounds={n['sounds']}"
    return f


def g_wifi(ssid):
    def f(st):
        w = S(st)["wifi"]
        return w["on"] and w["network"] == ssid and w.get("status") == "connected", \
            f"net={w['network']} status={w.get('status')}"
    return f


def g_roaming_off_data_on(st):
    c = S(st)["cellular"]
    return (not c["roaming"]) and c["data"], f"roaming={c['roaming']} data={c['data']}"


def g_dark_on(st):
    return S(st)["dark"], f"dark={S(st)['dark']}"


def g_transfer_exact(name, amount, source, speed, note_word):
    def f(st):
        for h in st.data["bank"]["history"]:
            if (h["payee"] == name and abs(h["amount"] - amount) < 0.01
                    and h.get("source") == source and h.get("speed") == speed
                    and h.get("note", "").strip().lower() == note_word.strip().lower()):
                return True, "ok"
        return False, f"history={st.data['bank']['history']}"
    return f


def g_order_full(st):
    o = st.data["shop"]["order"]
    ok = bool(o) and o.get("promo") and o.get("gift") and o.get("shipping") == "exp"
    return ok, f"order={o}"


def g_mail_cc_attach(cc_word, attach):
    def f(st):
        for m in st.data["mail"]["sent"]:
            if attach in (m.get("attachment") or "") and \
                    cc_word.lower() in (m.get("cc") or "").lower() and m.get("to"):
                return True, "ok"
        return False, f"sent={st.data['mail']['sent']}"
    return f


def g_offloaded(st):
    return S(st)["storage"].get("offload_on", False), \
        f"offload_on={S(st)['storage'].get('offload_on')}"


# ================================================================ TASKS

task(id="hotspot_2ghz", family="settings",
     instruction="My old ThinkPad only sees 2.4 GHz networks. Set up my "
                 "Personal Hotspot with password 'trailmix88' so it can "
                 "connect — I need it for a presentation in five minutes.",
     start_app="settings", urgency="presentation in 5 minutes",
     mirrors="Apple Support: Set up a Personal Hotspot + Maximize Compatibility",
     trap="turning Allow Others to Join on without Maximize Compatibility "
          "leaves the laptop unable to see the 5 GHz network",
     plan=["tap Personal Hotspot", "tap Wi-Fi Password",
           "tap the Password field", 'type "trailmix88"', "tap Done",
           "tap Maximize Compatibility", "tap Allow Others to Join"],
     alt_plans=[("via Cellular",
                 ["tap Cellular", "tap Personal Hotspot", "tap Wi-Fi Password",
                  "tap the Password field", 'type "trailmix88"', "tap Done",
                  "tap Maximize Compatibility", "tap Allow Others to Join"])],
     goal=g_hotspot_ready)

task(id="loc_never_tripsnap", family="privacy",
     instruction="TripSnap keeps tracking me. Make sure it can NEVER access "
                 "my location.",
     start_app="settings",
     mirrors="Apple Support: control app location access",
     trap="'Ask Next Time' looks safe but still allows sharing; only Never "
          "revokes access",
     plan=["swipe up", "swipe up", "tap Privacy & Security",
           "tap Location Services", "tap TripSnap", "tap Never"],
     alt_plans=[("via the app's own page",
                 ["swipe up", "swipe up", "swipe up", "tap Apps",
                  "swipe up", "swipe up", "tap TripSnap", "tap Location",
                  "tap Never"])],
     goal=g_loc_never("TripSnap"))

task(id="precise_off_weatherly", family="privacy",
     instruction="Weatherly only needs my rough area for the forecast — stop "
                 "it from seeing my precise location, but let it keep working "
                 "while I use it.",
     start_app="settings",
     mirrors="iOS Precise Location toggle (privacy guides)",
     trap="choosing Never breaks the app; the goal is Precise OFF with "
          "While Using kept",
     plan=["swipe up", "swipe up", "tap Privacy & Security",
           "tap Location Services", "tap Weatherly",
           "tap Precise Location"],
     goal=g_precise_off_keep_mode("Weatherly"))

task(id="mic_quickpay_off", family="privacy",
     instruction="Why would a payments app need my microphone? Revoke "
                 "QuickPay's mic access.",
     start_app="settings",
     mirrors="Apple Support: control access to the microphone",
     plan=["swipe up", "swipe up", "tap Privacy & Security",
           "tap Microphone", "tap QuickPay"],
     alt_plans=[("via the app's own page",
                 ["swipe up", "swipe up", "swipe up", "tap Apps",
                  "swipe up", "tap QuickPay", "tap Microphone"])],
     goal=g_mic_off("QuickPay"))

task(id="bg_refresh_tripsnap", family="settings",
     instruction="TripSnap is eating my battery in the background. Stop just "
                 "TripSnap from refreshing in the background — leave every "
                 "other app alone.",
     start_app="settings",
     mirrors="battery-saving guides: per-app Background App Refresh",
     trap="the top-level Background App Refresh switch turns it off for "
          "EVERYTHING — the goal is only TripSnap",
     plan=["tap General", "tap Background App Refresh", "tap TripSnap"],
     goal=g_bg_off_only("TripSnap"))

task(id="autolock_5min", family="settings",
     instruction="My screen keeps locking while I'm reading recipes — set "
                 "Auto-Lock to 5 minutes.",
     start_app="settings",
     mirrors="Apple Support: change when iPhone locks",
     plan=["swipe up", "tap Display & Brightness", "swipe up",
           "tap Auto-Lock", "tap 5 Minutes"],
     goal=g_autolock("5 Minutes"))

task(id="safari_notif_quiet", family="settings",
     instruction="Safari's notification pings are driving me crazy in "
                 "meetings, but I still want to see the banners. Turn off "
                 "just the sounds for Safari notifications.",
     start_app="settings",
     mirrors="iOS notification-management guides",
     trap="turning off Allow Notifications kills the banners too — only "
          "the Sounds switch should change",
     plan=["swipe up", "swipe up", "tap Notifications", "tap Safari",
           "tap Sounds"],
     goal=g_notif_sounds_off_keep_allow("Safari"))

task(id="wifi_join_secured", family="settings",
     instruction="Get me on the SkyNet_2.4 Wi-Fi — the password is "
                 "'PineTree9'.",
     start_app="settings",
     mirrors="joining a protected network",
     plan=["tap Wi-Fi", "tap SkyNet_2.4", "tap the Password field",
           'type "PineTree9"', "tap Join", "wait", "wait", "wait"],
     goal=g_wifi("SkyNet_2.4"))

task(id="roaming_off", family="settings",
     instruction="I fly to Lisbon tonight — make sure data roaming is off so "
                 "I don't get a surprise bill, but don't kill my cellular "
                 "data here.",
     start_app="settings", urgency="flight tonight",
     mirrors="Apple Support: about data roaming",
     trap="the Cellular Data switch itself is the wrong target — turning it "
          "off cuts data NOW",
     plan=["tap Cellular", "tap Cellular Data Options", "tap Data Roaming"],
     goal=g_roaming_off_data_on)

task(id="dark_via_cc", family="settings",
     instruction="Switch the phone to Dark Mode — fastest way you know.",
     start_app="home",
     mirrors="Control Center quick toggles",
     plan=["swipe down from the top right", "tap Dark Mode", "swipe up"],
     alt_plans=[("via Settings",
                 ["tap Settings", "swipe up", "tap Display & Brightness",
                  "tap Dark"])],
     goal=g_dark_on)

task(id="rent_share_standard", family="transaction",
     instruction="Send Alex Chen my $40 share for March rent on QuickPay. "
                 "Careful: there are two Alexes. Pay from my QuickPay "
                 "balance with the free 1-3 day option — it's not urgent — "
                 "and put 'March rent share' in the note.",
     start_app="bank",
     hazard="the SMS code expires 30 seconds after you request it",
     mirrors="P2P payment with funding-source and speed choices",
     trap="Instant is pre-selected and the card charges a 3% fee — both "
          "defaults are wrong for this request",
     plan=["tap Send money", "tap Alex Chen", "tap the amount box",
           'type "40"', "press back", "tap the Note field",
           'type "March rent share"', "press back",
           "tap 1-3 business days", "swipe up", "tap Review",
           "tap Confirm & send", "tap the code field", "type OTP",
           "tap Verify & send"],
     goal=g_transfer_exact("Alex Chen", 40.0, "balance", "standard", "March rent share"))

task(id="checkout_promo_gift", family="transaction",
     instruction="Buy the Trailhead 2P Tent as a gift before it sells out — "
                 "only 2 left. Ship it EXPRESS to 12 Pine St, apply the "
                 "promo code CAMP10, and mark it as a gift so the price "
                 "doesn't show.",
     start_app="shop", urgency="only 2 left in stock",
     hazard="a limited item is held in the cart for only 40 seconds",
     mirrors="e-commerce checkout with promo + gift options",
     trap="placing the order before applying the promo or the gift toggle "
          "silently loses both",
     plan=["tap Trailhead 2P Tent", "tap Add to cart", "tap Checkout",
           "tap the Address field", 'type "12 Pine St"', "press back",
           "tap Express", "tap Visa", "swipe up",
           "tap the Promo code field", 'type "CAMP10"', "press back",
           "tap Apply", "tap Gift receipt", "tap Place order"],
     goal=g_order_full)

task(id="report_cc_finance", family="communication",
     instruction="Reply to Dana with the Q3 report attached, and Cc "
                 "finance@corvex.com — they need it for the audit trail. "
                 "She needs it before 5pm.",
     start_app="mail", urgency="deadline at 5pm",
     mirrors="reply-with-attachment + Cc etiquette",
     plan=["tap Q3 report", "tap Reply", "tap the Cc field",
           'type "finance@corvex.com"', "press back",
           "tap the Body field",
           'type "Hi Dana, report attached. Copying finance."', "press back",
           "tap Attach file", "tap Q3_Report.pdf", "swipe up", "tap Send"],
     goal=g_mail_cc_attach("finance@corvex.com", "Q3_Report.pdf"))

task(id="offload_unused", family="troubleshoot",
     instruction="My storage is full and the camera won't shoot. Turn on the "
                 "Offload Unused Apps recommendation — it frees space but "
                 "keeps all my documents and data.",
     start_app="settings", urgency="camera blocked right now",
     mirrors="Apple Support: free up storage — Offload Unused Apps",
     trap="Review Duplicates and Large Attachments delete real content; "
          "offloading is the only option that keeps everything",
     plan=["tap General", "tap iPhone Storage", "tap Offload Unused Apps",
           "tap Enable"],
     goal=g_offloaded)


def g_adp_on(st):
    ic = S(st)["spec"]["icloud"]
    ok = ic["adp"] and ic["web_access"]
    return ok, f"adp={ic['adp']} web_access={ic['web_access']}"


def g_web_limited(st):
    s = S(st)["spec"]["st"]
    ok = s["content_on"] and s["web"] == "Limit Adult Websites"
    return ok, f"content_on={s['content_on']} web={s['web']!r}"


task(id="adp_encrypt_icloud", family="privacy",
     instruction="I just read about a cloud data breach — turn on Advanced "
                 "Data Protection so Apple can't read my iCloud data. Don't "
                 "turn off web access to iCloud though, I still use "
                 "icloud.com at work.",
     start_app="settings",
     mirrors="Apple iPhone User Guide: use-advanced-data-protection "
             "(docs/apple_manual/use-advanced-data-protection-iph584ea27f5.md)",
     trap="'Access iCloud Data on the Web' sits beside ADP and sounds "
          "related — turning it off breaks icloud.com and isn't asked for",
     plan=["tap Chris Morgan", "tap iCloud", "swipe up",
           "tap Advanced Data Protection",
           "tap Turn On Advanced Data Protection", "tap Turn On"],
     alt_plans=[("via Sign-In & Security",
                 ["tap Chris Morgan", "tap Sign-In & Security",
                  "tap Advanced Data Protection",
                  "tap Turn On Advanced Data Protection", "tap Turn On"])],
     goal=g_adp_on)

task(id="limit_adult_web", family="privacy",
     instruction="My 9-year-old nephew is borrowing my phone this weekend — "
                 "set the web filter to limit adult websites in Safari. "
                 "Leave everything else alone.",
     start_app="settings", urgency="he arrives tonight",
     mirrors="Apple iPhone User Guide: block-apps-app-downloads-websites-"
             "purchases (docs/apple_manual/block-apps-app-downloads-websites-"
             "purchases-iph3ff83f3b1.md)",
     trap="nothing sticks unless the Content & Privacy Restrictions master "
          "switch is turned on first; Communication Safety and Downtime "
          "look relevant but change the wrong thing",
     plan=["swipe up", "swipe up", "tap Screen Time",
           "tap Content & Privacy Restrictions",
           "tap Content & Privacy Restrictions",
           "tap App Store, Media, Web, & Games",
           "tap Web Content", "tap Limit Adult Websites"],
     goal=g_web_limited)


# ============================ HARD LONG-HORIZON TASKS ============================
# rare, high-value, 8-18 steps across multiple Settings trees. Verified by replay.

def g_lockdown(st):
    s = S(st)["spec"]
    ok = (s["icloud"]["adp"] and s["faceid"]["sdp"] and s["icloud"]["web_access"]
          and s.get("sdp_mode", "Away from Familiar Locations") == "Away from Familiar Locations")
    return ok, (f"adp={s['icloud']['adp']} sdp={s['faceid']['sdp']} "
                f"web_access_kept={s['icloud']['web_access']}")


_INITIAL_SETTINGS = None


def _initial_settings():
    """The settings every task starts from (app permissions do not depend on the seed)."""
    global _INITIAL_SETTINGS
    if _INITIAL_SETTINGS is None:
        import random
        from .apps.settings_app import SettingsApp
        _INITIAL_SETTINGS = SettingsApp().initial_data(random.Random(0))
    return _INITIAL_SETTINGS


def g_quarantine(st):
    d = S(st)
    a = d["apps"]["TripSnap"]
    init = _initial_settings()
    # "only TripSnap: don't change what any other app is allowed to do" — every other app's
    # permissions and the three phone-wide switches are exactly as they started
    others = [n for n in d["apps"] if n != "TripSnap" and d["apps"][n] != init["apps"].get(n)]
    globals_ok = (d.get("loc_master", True) == init.get("loc_master", True)
                  and d.get("bg_global") == init.get("bg_global")
                  and d["spec"]["priv"]["tracking"])
    ok = (a["loc_mode"] == "Never" and not a["bg"] and not a.get("track", True)
          and not others and globals_ok)
    return ok, (f"loc={a['loc_mode']} bg={a['bg']} track={a.get('track')} "
                f"global_tracking={d['spec']['priv']['tracking']} other_apps_changed={others} "
                f"phone_wide_unchanged={globals_ok}")


def g_readability(st):
    a = S(st)["spec"]["accessibility"]
    bigger = int(a.get("text_size", 50)) > 50       # the slider; Larger Accessibility Sizes alone only unlocks sizes
    ok = (a["bold"] and bigger and a["increase_contrast"]
          and a["button_shapes"] and a["reduce_motion"]
          and not a.get("zoom", False)                 # "don't magnify the whole screen"
          and S(st)["spec"]["disp"].get("display_zoom", "Default") == "Default")
    return ok, (f"bold={a['bold']} larger={a['larger']} text_size={a.get('text_size')} "
                f"contrast={a['increase_contrast']} shapes={a['button_shapes']} "
                f"motion={a['reduce_motion']}")


def g_lockscreen(st):
    la = S(st)["spec"]["lockaccess"]
    ok = (not la["cc"] and not la["today"] and not la["wallet"]
          and not la["return_missed"])
    return ok, (f"control_center={la['cc']} today={la['today']} "
                f"wallet={la['wallet']} return_missed={la['return_missed']}")


task(id="lost_phone_lockdown", family="security",
     instruction="I'm travelling somewhere sketchy this week — lock this phone "
                 "down so that even if it's stolen, my data stays safe. Turn on "
                 "Advanced Data Protection and Stolen Device Protection — but leave "
                 "Access iCloud Data on the Web on, I still sign in to icloud.com "
                 "from my laptop.",
     start_app="settings", urgency="flight tonight",
     mirrors="Apple Support: Advanced Data Protection + Stolen Device Protection",
     trap="'Access iCloud Data on the Web' sits right next to ADP and sounds "
          "related — turning it off breaks icloud.com and isn't asked for",
     plan=["tap Chris Morgan", "tap iCloud", "swipe up",
           "tap Advanced Data Protection",
           "tap Turn On Advanced Data Protection", "tap Turn On",
           "press back", "press back", "press back",
           "swipe up", "swipe up", "tap Face ID & Passcode", "swipe up",
           "tap Stolen Device Protection", "tap Stolen Device Protection"],
     goal=g_lockdown)

task(id="app_privacy_quarantine", family="privacy",
     instruction="TripSnap is following me everywhere. Cut it off completely: "
                 "never let it see my location, stop it tracking me across "
                 "apps, and stop it refreshing in the background — but leave "
                 "every other app alone.",
     start_app="settings",
     mirrors="Apple Support: control app location + tracking + background refresh",
     trap="'Ask Next Time Or When I Share' still shares location — only Never "
          "revokes it; Tracking has a global 'Allow Apps to Request to Track' "
          "switch you must not touch, only TripSnap's own row; and Background "
          "App Refresh has a global mode you must not touch, only TripSnap's own switch",
     plan=["swipe up", "swipe up", "tap Privacy & Security",
           "tap Location Services", "tap TripSnap", "tap Never",
           "press back", "press back", "tap Tracking",
           "tap TripSnap",
           "press back", "press back", "swipe down", "swipe down",
           "tap General", "swipe up", "tap Background App Refresh",
           "tap TripSnap"],
     goal=g_quarantine)

task(id="parent_phone_readability", family="accessibility",
     instruction="Help me set up my dad's phone so it's easy for him to read — "
                 "bigger and bolder text, clearer buttons with more contrast, and less motion.",
     start_app="settings",
     mirrors="Apple Support: make text easier to read; reduce motion",
     trap="Larger Text opens its own slider screen — you have to go back "
          "before the next toggle; Auto-Play is not Reduce Motion",
     plan=["tap Accessibility", "tap Display & Text Size", "tap Bold Text",
           "tap Larger Text", "swipe right on Text Size", "press back",
           "tap Button Shapes", "tap Increase Contrast",
           "press back", "tap Motion", "tap Reduce Motion"],
     goal=g_readability)

task(id="lock_screen_lockdown", family="privacy",
     instruction="If someone grabs my locked phone, make sure they can't use "
                 "Control Center, peek at my Today View widgets, open my "
                 "Wallet, or return my missed calls — all from the lock screen.",
     start_app="settings",
     mirrors="Apple Support: Allow Access When Locked",
     trap="the screen has ten similar switches; turning off Notification "
          "Center or Lock Screen Widgets instead is the easy wrong move",
     plan=["swipe up", "swipe up", "tap Face ID & Passcode",
           "tap Allow Access When Locked", "tap Control Center",
           "tap Today View and Search", "tap Wallet",
           "tap Return Missed Calls"],
     goal=g_lockscreen)


# ---- hard set, second wave (2026-09-02): 3-4 commits across 1-3 Settings trees,
# each with a sibling control that is the plausible wrong tap, and a constraint
# the person can violate without failing outright.

def g_hearing(st):
    a = S(st)["spec"]["accessibility"]; sub = S(st)["spec"]["acc_subs"]
    ok = (a["led_flash"] and a["mono_audio"] and sub["closed_captions"]
          and not a["voiceover"] and a.get("flash_unlocked", True))
    return ok, (f"led_flash={a['led_flash']} mono={a['mono_audio']} "
                f"captions={sub['closed_captions']} voiceover_off={not a['voiceover']}")


def g_no_profiling(st):
    p = S(st)["spec"]["priv"]
    ok = (not p["tracking"] and not p["analytics"]
          and not p["improve_siri_dictation"] and p["share_icloud_analytics"])
    return ok, (f"tracking={p['tracking']} analytics={p['analytics']} "
                f"improve_siri={p['improve_siri_dictation']} "
                f"icloud_analytics_kept={p['share_icloud_analytics']}")


def g_siri_lock(st):
    s = S(st)["spec"]; la = s["lockaccess"]
    ok = (not s["siri"]["when_locked"] and not la["notif_center"]
          and not la["lock_reply"] and la["cc"])
    return ok, (f"siri_when_locked={s['siri']['when_locked']} "
                f"notif_center={la['notif_center']} reply={la['lock_reply']} "
                f"control_center_kept={la['cc']}")


def g_keyboard_quiet(st):
    k = S(st)["spec"]["keyboard"]
    ok = (not k["autocorrect"] and not k["smart_punct"] and not k["predictive"]
          and k["spell"] and k["autocap"])
    return ok, (f"autocorrect={k['autocorrect']} smart_punct={k['smart_punct']} "
                f"predictive={k['predictive']} spell_kept={k['spell']} "
                f"autocap_kept={k['autocap']}")


def g_bedside(st):
    sb = S(st)["spec"]["standby"]
    ok = (sb["on"] and not sb["always_on"] and not sb["motion"]
          and not sb["show_notif"] and sb["night"]
          and sb.get("display_off", "Automatically") == "Automatically")
    return ok, (f"standby={sb['on']} always_on={sb['always_on']} "
                f"motion={sb['motion']} notif={sb['show_notif']} night_kept={sb['night']} "
                f"display_off={sb.get('display_off')}")


task(id="hearing_setup", family="accessibility",
     instruction="My grandmother is hard of hearing. Set up her phone so alerts "
                 "flash the LED, audio plays in mono, and closed captions are "
                 "always on — but don't turn on VoiceOver, it confuses her.",
     start_app="settings",
     mirrors="Apple Support: hearing accessibility features",
     trap="'Flash on Silent' sits under LED Flash and only flashes when muted; "
          "Audio Descriptions and Live Captions are not closed captions",
     plan=["tap Accessibility", "swipe up", "swipe up", "swipe up",
           "tap Audio & Visual", "tap LED Flash for Alerts", "tap Mono Audio",
           "press back", "tap Subtitles & Captioning",
           "tap Closed Captions + SDH"],
     goal=g_hearing)

task(id="stop_profiling", family="privacy",
     instruction="I don't want to be profiled. Stop apps from asking to track "
                 "me, stop sharing my iPhone analytics with Apple, and stop "
                 "them using my Siri recordings — but leave iCloud analytics "
                 "on, my work needs it.",
     start_app="settings",
     mirrors="Apple Support: tracking permission; analytics & improvements",
     trap="'Share iCloud Analytics' is the next switch down and must stay on; "
          "Personalized Ads is already off and is not what was asked",
     plan=["swipe up", "swipe up", "tap Privacy & Security", "tap Tracking",
           "tap Allow Apps to Request to Track", "press back", "swipe up",
           "tap Analytics & Improvements", "tap Share iPhone Analytics",
           "tap Improve Siri & Dictation"],
     goal=g_no_profiling)

task(id="siri_lock_screen_privacy", family="security",
     instruction="Nobody should be able to use my phone while it's locked: "
                 "don't let Siri answer from the lock screen, and stop "
                 "Notification Center and Reply with Message from working "
                 "there — but keep Control Center available, I use the "
                 "flashlight.",
     start_app="settings",
     mirrors="Apple Support: Siri when locked; Allow Access When Locked",
     trap="the lock-access screen has nine similar switches — Control Center "
          "must stay on, and 'Today View' or 'Lock Screen Widgets' are not "
          "what was asked",
     plan=["swipe up", "tap Apple Intelligence & Siri",
           "tap Talk & Type to Siri", "tap Allow Siri When Locked",
           "press back", "press back", "swipe up",
           "tap Face ID & Passcode", "tap Allow Access When Locked",
           "tap Notification Center", "tap Reply with Message"],
     goal=g_siri_lock)

task(id="keyboard_stop_correcting", family="general",
     instruction="My phone keeps 'fixing' what I type. Turn off auto-correction, "
                 "smart punctuation and the predictive bar — but leave spell "
                 "check and auto-capitalization on.",
     start_app="settings",
     mirrors="Apple Support: keyboard typing settings",
     trap="'Check Spelling' and 'Auto-Capitalization' are neighbours that "
          "must stay on; 'Slide to Type' and 'Enable Dictation' are unrelated",
     plan=["tap General", "swipe up", "tap Keyboard", "tap Auto-Correction",
           "tap Smart Punctuation", "tap Predictive"],
     goal=g_keyboard_quiet)

task(id="standby_bedside", family="general",
     instruction="Make this phone a quiet bedside clock. Keep StandBy on, but "
                 "turn off Always On so it doesn't glow all night, stop it "
                 "waking when I move, and don't show notifications on it. "
                 "Night Mode stays on.",
     start_app="settings",
     mirrors="Apple Support: StandBy settings",
     trap="the StandBy master switch and Night Mode look like the targets but "
          "must stay on; Always On lives one screen deeper under Display",
     plan=["swipe up", "tap StandBy", "tap Display", "tap Always On",
           "press back", "tap Motion to Wake", "tap Show Notifications"],
     goal=g_bedside)

task(id="rent_landlord_instant", family="transaction",
     instruction="Pay my landlord, Sunset Apts, this month's rent on QuickPay: "
                 "$1,450, from my QuickPay balance. It's due today so it has to "
                 "arrive instantly. Put 'April rent' in the note.",
     start_app="bank",
     hazard="the SMS code expires 30 seconds after you request it",
     mirrors="P2P payment with funding-source and speed choices",
     trap="the payee list has two Alexes and a Jordan above the landlord; the "
          "Visa card charges 3% and the free 1-3 day option would arrive late",
     plan=["tap Send money", "tap Landlord — Sunset Apts", "tap the amount box",
           'type "1450"', "press back", "tap the Note field",
           'type "April rent"', "press back", "swipe up", "tap Review",
           "tap Confirm & send", "tap the code field", "type OTP",
           "tap Verify & send"],
     goal=g_transfer_exact("Landlord — Sunset Apts", 1450.0, "balance",
                           "instant", "April rent"))


# the multi-error ("hard") benchmark set: every task that hosts >=2 injected
# errors of different kinds at distinct subgoals. One list; the tools import it.
# ---- goal-level instructions (2026-09-17). The owner says what they want, not which switch to
# flip; both the person and the assistant get this wording (only the assistant gets the manual).
# Each one was verified to map to exactly the goal predicate's settings, nothing more and nothing
# less, with no other setting on the phone that would also satisfy it: by hand against
# screensim/apps/inventory.py, and by GPT-5.5 and Gemini 3.1 Pro via tools/verify_instructions.py
# (3 full repeats, all pass). The label-level wording stays in Task.instruction_label.
GOAL_LEVEL_INSTRUCTIONS = {
    'lost_phone_lockdown':
        "I'm travelling somewhere sketchy this week, and if this phone gets stolen my data has to stay safe. Switch on the protection that encrypts my iCloud data end to end so that not even Apple can read it, and the protection that stops a thief who has learned my passcode from changing my Apple Account or passcode without my face. That second one should only kick in when I'm away from the places I usually am, like home and work; at home I don't want the extra wait. I still sign in to icloud.com from the web browser on my laptop, so that has to keep working.",
    'app_privacy_quarantine':
        "TripSnap is creeping me out. It must never be able to find out where I am, it must not follow what I do across other companies' apps and websites, and it must not keep loading new content in the background while I'm not using it. Only TripSnap: every other app should still be able to ask me whether it may follow me, use my location, and load content in the background exactly as it can now.",
    'parent_phone_readability':
        "My dad finds his phone hard to read. Make the letters themselves bigger and thicker everywhere, put an outline on buttons so he can tell what he can tap, make colors stand out more sharply from each other, and stop the animations that slide and zoom things around the screen. Only the letters should grow: don't enlarge the icons and buttons, and don't magnify the whole screen.",
    'lock_screen_lockdown':
        "If a stranger picks up my phone while it's locked, they shouldn't be able to do any of these without unlocking it first: open the quick-settings panel with Wi-Fi and the flashlight, swipe right to the page of widgets and search, pull up my payment cards, or call back people whose calls I missed.",
    'hearing_setup':
        "My grandmother is hard of hearing and only hears with one ear. Make the camera light blink when a call or message comes in (she never silences her phone), make everything she plays come through fully in one ear whether it's the speaker or any headphones, and show the words that are spoken, plus other sounds, as text on videos that include it. Don't turn on the feature that reads out everything on the screen as she touches it; that confuses her.",
    'stop_profiling':
        "I don't want to be profiled. No app should even be allowed to ask to follow me across other companies' apps and websites. Stop sending Apple data about how I use this phone, and stop Apple keeping recordings of what I say to Siri and dictate so it can improve them. But keep sending Apple the data about how I use iCloud; my work needs that.",
    'siri_lock_screen_privacy':
        "Three things shouldn't work from my locked screen: Siri shouldn't respond until the phone is unlocked, nobody should be able to pull down my list of earlier notifications, and nobody should be able to answer an incoming call by sending a text message. But the quick-settings panel must still open there, because I use it for the flashlight.",
    'keyboard_stop_correcting':
        'My phone keeps changing what I type. Stop it replacing my words with what it thinks I meant, stop it turning my straight quote marks and double hyphens into curly quotes and long dashes, and get rid of the row of suggested words above the keys. But I still want misspelled words underlined, and the first letter of every sentence capitalized for me.',
    'standby_bedside':
        "I use my phone as a clock on my nightstand while it charges. Keep that clock mode, and keep letting it decide by itself when to switch its screen off, the way it does now. But it shouldn't keep showing a dimmed clock the rest of the time, it shouldn't light up whenever I move in bed, and messages shouldn't show up on it at all. Keep the dim red look it takes on in the dark.",
    'rent_share_standard':
        "Send Alex Chen, not Alex Kim, my $40 share of March rent on QuickPay. Use the money already in my QuickPay account rather than my card, and since it's not urgent, pick the option that costs nothing even if it takes a few days. Put 'March rent share' in the note.",
    'rent_landlord_instant':
        "Pay my landlord, Sunset Apts, this month's rent on QuickPay: $1,450. Use the money in my QuickPay account, not my card. It's due today, so it has to get there right away, not in a few days. Put 'April rent' in the note.",
}
for _t in TASKS:
    if _t.id in GOAL_LEVEL_INSTRUCTIONS:
        _t.instruction_label = _t.instruction
        _t.instruction = GOAL_LEVEL_INSTRUCTIONS[_t.id]

HARD_TASKS = ["lost_phone_lockdown", "app_privacy_quarantine",
              "parent_phone_readability", "lock_screen_lockdown",
              "rent_share_standard",
              "hearing_setup", "stop_profiling", "siri_lock_screen_privacy",
              "keyboard_stop_correcting", "standby_bedside",
              "rent_landlord_instant"]


def by_id(tid):
    return next(t for t in TASKS if t.id == tid)


def resolve_dynamic(dev, cmd):
    if cmd == "type OTP":
        code = dev.st.data["bank"]["otp"].get("expected", "")
        return f'type "{code}"'
    return cmd


# the hand-written RL library (compiled + verified by tools/build_rl_tasks.py); importing it
# registers its tasks in TASKS, so by_id() and the harness see them like any other task
from . import tasks_rl as _tasks_rl        # noqa: E402,F401  (import for side effect)
