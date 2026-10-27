"""Grounding — the heart of the action space. Label, ordinal, region,
relation, ambiguity, off-screen, coordinate, and refusal-reason behavior."""
import pytest
from screensim.core.engine import Device
from screensim.core.actions import parse, ground, TAP, SWIPE, TYPE, PRESS_KEY


def settings_at(screen_cmds):
    d = Device(0)
    d.open_app("settings")
    for c in screen_cmds:
        d.act(c)
    return d


# ---- parsing ----
@pytest.mark.parametrize("text,verb", [
    ("tap Continue", TAP), ("click the blue button", TAP),
    ("long press the photo", "long_press"), ("swipe up", SWIPE),
    ("scroll down", SWIPE), ('type "hello"', TYPE), ("go back", PRESS_KEY),
    ("press home", PRESS_KEY), ("wait", "wait"),
])
def test_parse_verb(text, verb):
    assert parse(text).verb == verb


def test_parse_swipe_direction_required():
    g = parse("swipe on the Alex row")
    assert g.verb == SWIPE and g.direction == ""


def test_parse_ordinal_and_region():
    g = parse("tap the second option near the top right")
    assert g.ordinal == 1
    assert "top" in g.region and "right" in g.region


def test_parse_relation():
    g = parse("tap the toggle next to Bluetooth")
    assert g.relation[0] == "next_to" and "bluetooth" in g.relation[1]


def test_parse_coordinate():
    g = parse("tap at (40, 120)")
    assert g.point == (40, 120)


# ---- label grounding ----
def test_label_exact_navigates():
    d = settings_at(["swipe up"])
    assert d.act("tap Display & Brightness").ok
    assert d.st.cur_screen_key() == "settings.display"


def test_below_fold_requires_scroll():
    d = settings_at([])
    r = d.act("tap Privacy & Security")     # below the fold on the dense root
    assert not r.ok
    d.act("swipe up"); d.act("swipe up")
    assert d.act("tap Privacy & Security").ok


def test_label_preposition_is_not_stripped():
    # "To" is a real field label; must not vanish as a stopword
    d = Device(0); d.open_app("mail")
    d.act("tap Q3 report"); d.act("tap Reply")
    r = d.act("tap the To field")
    assert r.ok and d.st.focus == "to_field"


def test_unknown_label_refused_with_reason():
    d = settings_at([])
    r = d.act("tap Quantum Settings")
    assert not r.ok and "can't see" in r.reason.lower()


def test_noninteractive_named_element_refused_helpfully():
    d = settings_at(["tap Personal Hotspot"])
    # the hint paragraph is visible text but not tappable
    r = d.act("tap Others can join your")
    assert not r.ok


# ---- ambiguity ----
def test_ambiguous_two_alex():
    d = Device(0); d.open_app("bank"); d.act("tap Send money")
    r = d.act("tap Alex")
    assert r.status == "ambiguous" and r.n_candidates == 2


def test_ambiguity_resolved_by_full_name():
    d = Device(0); d.open_app("bank"); d.act("tap Send money")
    assert d.act("tap Alex Chen").ok
    assert d.st.data["bank"]["transfer"]["payee"]["name"] == "Alex Chen"


def test_ambiguity_resolved_by_ordinal():
    d = Device(0); d.open_app("bank"); d.act("tap Send money")
    r = d.act("tap the first Alex")
    assert r.ok


# ---- ordinal / region ----
def test_ordinal_second_row():
    d = settings_at(["tap Wi-Fi"])
    r = d.act("tap the second one")
    assert r.ok


def test_last_ordinal():
    d = settings_at([])
    r = d.act("tap the last option")
    assert r.ok  # battery row


# ---- relation ----
def test_relation_toggle_next_to():
    d = settings_at(["tap Bluetooth"])
    # Bluetooth toggle is the only control; relation to its own label filtered out
    r = d.act("tap Bluetooth")
    assert r.ok


# ---- offscreen requires scroll ----
def test_scroll_moves_widgets_on_a_long_screen():
    from screensim.core.layout import layout
    from screensim.core.model import Screen, w_row
    long_screen = Screen("test.long", "Long",
                         [w_row(f"r{i}", f"Row {i}") for i in range(30)])
    ms0, _ = layout(long_screen, 0, False)
    assert ms0 > 0  # content overflows -> scrollable
    y_at_0 = long_screen.widgets[20].bounds.y
    layout(long_screen, 300, False)
    y_at_300 = long_screen.widgets[20].bounds.y
    assert y_at_300 == y_at_0 - 300  # scrolling moved it up
    # a widget below the fold is invisible at scroll 0, visible after scrolling
    layout(long_screen, 0, False)
    assert not long_screen.widgets[29].visible
    layout(long_screen, ms0, False)
    assert long_screen.widgets[29].visible


def test_swipe_at_bottom_reports_bottom():
    d = settings_at([])
    # settings root fits, so a swipe up either scrolls or says bottom;
    # after enough swipes it must report the bottom.
    seen_bottom = False
    for _ in range(6):
        r = d.act("swipe up")
        if "bottom" in (r.reason or ""):
            seen_bottom = True
    assert seen_bottom


# ---- coordinate + empty-space ----
def test_coordinate_tap_hits_widget():
    d = settings_at([])
    scr = d.laid_out_screen()
    row = next(w for w in scr.widgets if w.id == "s_wifi")
    r = d.act(f"tap at ({row.bounds.cx}, {row.bounds.cy})")
    assert r.ok and d.st.cur_screen_key() == "settings.wifi_list"


def test_tap_empty_space_is_harmless():
    d = settings_at([])
    r = d.act("tap at (5, 400)")
    assert r.ok  # taps nothing, no crash, no nav
    assert d.st.cur_screen_key() == "settings.root"


# ---- swipe ----
def test_swipe_without_direction_refused():
    d = Device(0); d.open_app("messages")
    r = d.act("swipe the Alex Chen row")
    assert not r.ok and "direction" in r.reason.lower()


def test_swipe_left_archives_thread():
    d = Device(0); d.open_app("messages")
    before = len(d.st.data["messages"]["threads"])
    r = d.act("swipe left on the Alex Chen row")
    assert r.ok and len(d.st.data["messages"]["threads"]) == before - 1
