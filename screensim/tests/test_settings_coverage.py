"""Manual-completeness coverage: every nav row reachable from the Settings root
must open a REAL screen, never silently fall back to the root. This is what
lets us claim "everything in the manual can be simulated" — if a row is added
that points nowhere, this test fails.
"""
import random

from screensim.apps.settings_app import SettingsApp
from screensim.apps import settings_engine as SE
from screensim.core.state import DeviceState


def _fresh_state():
    app = SettingsApp()
    d = app.initial_data(random.Random(0))
    st = DeviceState()
    st.app = "settings"
    st.data = {"settings": d}
    st.stack = ["settings.root"]
    return app, st


def _nav_targets(screen):
    """Collect settings.* nav targets from a built screen's widgets."""
    out = []
    for w in screen.widgets:
        tap = w.on_tap
        if tap and tap[0] == "nav" and isinstance(tap[1], str) \
                and tap[1].startswith("settings."):
            out.append(tap[1])
    return out


def test_every_nav_row_opens_a_real_screen():
    app, st = _fresh_state()
    seen = set()
    frontier = ["settings.root"]
    unresolved = []

    while frontier:
        key = frontier.pop()
        if key in seen:
            continue
        seen.add(key)
        # choice screens are generated separately; skip in the nav walk
        if key.startswith(SE.CHOICE_PREFIX):
            continue
        st.stack = [key]
        scr = app.view(st)
        # a non-root target that resolves to the root screen is a dead link
        if key != "settings.root" and scr.key == "settings.root":
            unresolved.append(key)
            continue
        for t in _nav_targets(scr):
            if t not in seen:
                frontier.append(t)

    assert not unresolved, f"nav rows dead-end on root fallback: {unresolved}"
    # sanity: we actually walked a broad tree, not just a handful of screens
    assert len(seen) >= 60, f"only reached {len(seen)} screens"


def test_choice_rows_have_option_specs():
    """Every `choice` row in the spec references a CHOICES entry, so tapping it
    opens a real radio list instead of crashing."""
    from screensim.apps import settings_data as SD
    missing = []
    for key, spec in SD.SCREENS.items():
        for sec in spec.get("sections", []):
            for r in sec.get("rows", []):
                if r.get("t") == "choice" and r["skey"] not in SD.CHOICES:
                    missing.append((key, r["skey"]))
    assert not missing, f"choice rows without option specs: {missing}"


def test_every_settings_icon_draws_a_glyph():
    """A widget icon name with no glyph used to render as its literal name ("cloud"
    printed before "Weatherly"), which the assistant then read as part of the label.
    Walk every reachable screen; every icon must resolve to an SVG glyph or a fallback."""
    from screensim.render import icons
    from screensim.render.renderer import ICON
    app, st = _fresh_state()
    seen, frontier, missing = set(), ["settings.root"], {}
    while frontier:
        key = frontier.pop()
        if key in seen or key.startswith(SE.CHOICE_PREFIX):
            continue
        seen.add(key)
        st.stack = [key]
        scr = app.view(st)
        for w in scr.widgets:
            if w.icon and not icons.has(w.icon) and w.icon not in ICON:
                missing.setdefault(w.icon, key)
        frontier += [t for t in _nav_targets(scr) if t not in seen]
    assert not missing, f"icons with no glyph: {missing}"


def test_no_two_different_controls_share_a_state_path():
    """Settings twins, extended to sliders (round 6): once sliders took swipes, sliders
    that shared a path with an unrelated setting became visible. Balance was drawn from
    Speaking Rate's value (knob at the far left), and two Volume sliders wrote the battery
    charge limit. Every toggle and slider label reachable from the root owns its path."""
    app, st = _fresh_state()
    seen, frontier, paths = set(), ["settings.root"], {}
    while frontier:
        key = frontier.pop()
        if key in seen or key.startswith(SE.CHOICE_PREFIX):
            continue
        seen.add(key)
        st.stack = [key]
        scr = app.view(st)
        for w in scr.widgets:
            p = None
            if w.kind == "toggle" and w.on_tap and len(w.on_tap) > 1:
                p = w.on_tap[1]
            if w.kind == "slider" and isinstance(w.on_swipe, dict) and w.on_swipe:
                p = list(w.on_swipe.values())[0][1]
            if isinstance(p, str):
                paths.setdefault(p, set()).add((w.label or "").strip())
        frontier += [t for t in _nav_targets(scr) if t not in seen]
    shared = {p: sorted(l) for p, l in paths.items() if len(l) > 1}
    assert not shared, f"controls sharing one state path: {shared}"


def test_bold_text_is_one_setting_in_both_places():
    """iOS shows Bold Text under Display & Brightness and under Accessibility > Display & Text
    Size; it is one setting. They were two separate states, so turning it on under Display &
    Brightness failed a task that asks for bolder text."""
    from screensim.core.engine import Device
    d = Device(0)
    d.open_app("settings")
    for _ in range(4):
        if d.act("tap Display & Brightness").ok:
            break
        d.act("swipe up")
    for _ in range(4):
        if d.act("tap Bold Text").ok:
            break
        d.act("swipe up")
    assert d.st.data["settings"]["spec"]["accessibility"]["bold"] is True


def test_text_size_under_display_is_the_same_setting():
    """Display & Brightness > Text Size was a placeholder page with no slider, although it is the
    usual route to bigger text on iOS and the same setting as Accessibility > Larger Text."""
    from screensim.core.engine import Device
    d = Device(0)
    d.open_app("settings")
    for _ in range(4):
        if d.act("tap Display & Brightness").ok:
            break
        d.act("swipe up")
    for _ in range(4):
        if d.act("tap Text Size").ok:
            break
        d.act("swipe up")
    before = d.st.data["settings"]["spec"]["accessibility"]["text_size"]
    assert d.act("swipe right on Text Size").ok
    assert d.st.data["settings"]["spec"]["accessibility"]["text_size"] > before


def test_manual_described_controls_exist_as_real_controls():
    # the assistant's manual describes these; a text label or a missing row invites a phantom alternative
    import screensim.apps.inventory as inv
    text = inv.render(inv.walk())
    for line in ("page    Turn Display Off  (the row shows: Automatically)",
                 "option  After 20 Seconds",
                 "page    Display Zoom  (the row shows: Default)",
                 "switch  Flash While Unlocked = on",
                 "switch  Dim Flashing Lights = off",
                 "switch  Prefer Non-Blinking Cursor = off",
                 "switch  Auto-Play Animated Images = on"):
        assert line in text, line


def test_settings_search_finds_and_opens_a_page():
    """Audit 2026-09-22: the root Search field accepted typed text and dropped it — the person saw an empty
    box and no results, and the assistant that sent them there was charged for the dead end."""
    from screensim.core.engine import Device
    from screensim.tasks import by_id, resolve_dynamic
    t = by_id("standby_bedside")
    d = Device(t.seed)
    d.open_app("settings")
    assert d.act(resolve_dynamic(d, "tap Search")).ok
    assert d.act(resolve_dynamic(d, 'type "StandBy"')).ok
    labels = [w.label for w in d.laid_out_screen().widgets if w.label]
    assert "StandBy" in labels, labels
    assert d.act(resolve_dynamic(d, "tap StandBy")).ok
    assert d.laid_out_screen().title == "StandBy"
