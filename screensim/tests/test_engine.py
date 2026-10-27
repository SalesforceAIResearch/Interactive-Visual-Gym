"""Engine mechanics: determinism, navigation stack, keyboard/typing, toggles,
timers/hazards, outcomes, and defensiveness (never crash on junk)."""
import pytest
from screensim.core.engine import Device


# ---- determinism ----
def test_double_replay_identical_hash():
    plan = ["tap Send money", "tap Jordan Diaz", "tap the amount box",
            'type "15"', "tap Review", "tap Confirm & send"]

    def run():
        d = Device(7); d.open_app("bank")
        return [(_ := d.act(c)) and d.st.hash() for c in plan]
    assert run() == run()


def test_seed_changes_nothing_structural_but_state_stable():
    for seed in (0, 1, 2, 3):
        a = Device(seed); a.open_app("settings")
        b = Device(seed); b.open_app("settings")
        assert a.st.hash() == b.st.hash()


# ---- navigation ----
def test_back_pops_stack():
    d = Device(0); d.open_app("settings")
    d.act("tap Wi-Fi")
    assert d.st.cur_screen_key() == "settings.wifi_list"
    d.act("press back")
    assert d.st.cur_screen_key() == "settings.root"


def test_home_resets_to_launcher():
    d = Device(0); d.open_app("settings")
    d.act("tap Wi-Fi")
    d.act("press home")
    assert d.st.app == "home"


def test_back_from_root_goes_home():
    d = Device(0); d.open_app("settings")
    d.act("press back")
    assert d.st.app == "home"


def test_navigation_dismisses_keyboard():
    d = Device(0); d.open_app("bank")
    d.act("tap Send money"); d.act("tap Alex Chen")
    d.act("tap the amount box"); d.act('type "40"')
    assert d.st.keyboard
    d.act("press back")          # dismiss keyboard (iOS: fields keep value)
    assert not d.st.keyboard and d.st.focus is None
    d.act("swipe up")
    d.act("tap Review")          # now below-the-fold button is visible
    assert d.st.cur_screen_key() == "bank.review"


# ---- typing ----
def test_type_requires_focus():
    d = Device(0); d.open_app("bank"); d.act("tap Send money"); d.act("tap Alex Chen")
    r = d.act('type "40"')  # no focus yet
    assert not r.ok and "field" in r.reason.lower()


def test_numeric_field_rejects_letters():
    d = Device(0); d.open_app("bank"); d.act("tap Send money"); d.act("tap Alex Chen")
    d.act("tap the amount box")
    r = d.act('type "abc"')
    assert not r.ok and "number" in r.reason.lower()


def test_clear_field():
    d = Device(0); d.open_app("settings")
    d.act("tap Personal Hotspot"); d.act("tap Wi-Fi Password")
    d.act("tap the Password field"); d.act('type "secret12"')
    assert d.st.data["settings"]["hotspot"]["password"] == "secret12"
    d.act("clear")
    assert d.st.data["settings"]["hotspot"]["password"] == ""


# ---- toggles ----
def test_toggle_flips_value():
    d = Device(0); d.open_app("settings")
    d.act("swipe up"); d.act("tap Display & Brightness")
    before = d.st.data["settings"]["dark"]
    d.act("tap Dark")
    assert d.st.data["settings"]["dark"] != before


def test_hotspot_requires_password_first():
    d = Device(0); d.open_app("settings")
    d.act("tap Personal Hotspot")
    d.act("tap Allow Others to Join")  # no password
    assert d.st.dialog is not None
    assert not d.st.data["settings"]["hotspot"]["on"]


# ---- timers & hazards ----
def test_otp_expiry_locks_out_the_code():
    d = Device(1); d.open_app("bank")
    for c in ["tap Send money", "tap Jordan Diaz", "tap the amount box",
              'type "15"', "press back", "swipe up", "tap Review", "tap Confirm & send"]:
        d.act(c)
    exp = next(t.expires_at for t in d.st.timers if t.id == "otp_expiry")
    while d.st.tick <= exp + 1:
        d.act("wait")
    assert next(t.fired for t in d.st.timers if t.id == "otp_expiry")
    code = d.st.data["bank"]["otp"]["expected"]
    d.act("tap the code field"); d.act(f'type "{code}"')
    d.act("tap Verify & send")
    assert d.st.cur_screen_key() == "bank.otp"  # expired -> not sent


def test_three_wrong_pins_locks_account():
    d = Device(2); d.open_app("bank")
    for c in ["tap Send money", "tap Alex Chen", "tap the amount box",
              'type "20"', "press back", "swipe up", "tap Review", "tap Confirm & send"]:
        d.act(c)
    for _ in range(3):
        d.act("tap the code field"); d.act('type "000000"'); d.act("tap Verify & send")
    assert d.st.cur_screen_key() == "bank.locked"


def test_cart_hold_releases_limited_item():
    d = Device(0); d.open_app("shop")
    d.act("tap Trailhead 2P Tent"); d.act("tap Add to cart")
    exp = next(t.expires_at for t in d.st.timers if t.id == "cart_hold")
    while d.st.tick <= exp + 1:
        d.act("wait")
    assert not any(c.get("limited") for c in d.st.data["shop"]["cart"])


def test_tick_cap_timeout():
    d = Device(0, tick_cap=10); d.open_app("settings")
    for _ in range(20):
        d.act("wait")
    assert d.st.outcome == "timeout"


# ---- defensiveness ----
@pytest.mark.parametrize("junk", [
    "asdf qwer", "tap the flibbertigibbet", "", "swipe", "type",
    "tap tap tap", "1234", "tap at (999, 999)", "long press nothing here",
])
def test_junk_never_crashes(junk):
    d = Device(0); d.open_app("settings")
    r = d.act(junk)  # must return, never raise
    assert r is not None
    assert d.st.outcome in (None, "timeout")


def test_fuzz_never_corrupts():
    import random
    verbs = ["tap", "swipe up", "swipe left", "press back", "type \"x\"",
             "tap Send", "tap the second one", "wait", "tap Continue"]
    for seed in range(5):
        d = Device(seed); d.open_app(["settings", "bank", "shop", "mail"][seed % 4])
        rng = random.Random(seed)
        for _ in range(40):
            d.act(rng.choice(verbs))
        # engine still answers and state hashes cleanly
        assert isinstance(d.st.hash(), str)
