"""Every task reaches its goal from its clean plan; layout invariants; and a
render smoke test (HTML is well-formed and mentions on-screen content)."""
import pytest
from screensim.core.engine import Device
from screensim.core.layout import layout
from screensim.tasks import TASKS
from screensim.runner import run_task
from screensim.render.renderer import render


# ---- task correctness ----
@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_task_reaches_goal(task):
    ro = run_task(task)
    assert ro.won, f"{task.id}: {ro.detail}\n" + "\n".join(
        f"  [{s.idx}] {s.cmd!r} {s.status} {s.reason}" for s in ro.steps if not s.ok)


def test_all_families_covered():
    fams = {t.family for t in TASKS}
    for expected in ("settings", "privacy", "transaction", "communication",
                     "troubleshoot"):
        assert expected in fams


def test_alt_plans_reach_goal():
    from screensim.tasks import resolve_dynamic
    for t in TASKS:
        for label, plan in t.alt_plans:
            d = Device(t.seed)
            if t.start_app != "home":
                d.open_app(t.start_app)
            for cmd in plan:
                d.act(resolve_dynamic(d, cmd))
            won, detail = t.goal(d.st)
            assert won, f"{t.id} alt '{label}': {detail}"


def test_hazard_tasks_have_hazard_note():
    for t in TASKS:
        if t.family == "transaction":
            assert t.hazard, f"{t.id} should document its hazard clock"


# ---- layout invariants ----
def test_widgets_have_bounds_and_visibility():
    d = Device(0); d.open_app("settings")
    scr = d.laid_out_screen()
    for w in scr.widgets:
        assert w.bounds is not None
        assert isinstance(w.visible, bool)


def test_visible_widgets_are_within_screen():
    from screensim.core.model import SCREEN_W
    d = Device(0); d.open_app("bank")
    d.act("tap Send money")
    for w in d.laid_out_screen().widgets:
        if w.visible:
            assert 0 <= w.bounds.x and w.bounds.x2 <= SCREEN_W + 1


def test_grid_layout_packs_tiles():
    d = Device(0)  # home = 4-col grid
    scr = d.laid_out_screen()
    tiles = [w for w in scr.widgets if w.kind == "tile"]
    rows = {}
    for t in tiles:
        rows.setdefault(t.bounds.y, []).append(t)
    assert any(len(r) > 1 for r in rows.values())  # multiple per row


# ---- render smoke ----
def test_render_is_wellformed_html():
    d = Device(0); d.open_app("settings")
    html = render(d)
    assert html.count("<div") > 5
    assert html.count("<div") >= html.count("</div>")  # balanced-ish
    assert "Wi-Fi" in html


def test_render_reflects_state_change():
    d = Device(0); d.open_app("settings")
    d.act("swipe up"); d.act("tap Display & Brightness")
    light = render(d)
    d.act("tap Dark")
    dark = render(d)
    assert light != dark
    assert "phone dark" in dark  # dark class applied


def test_render_shows_focused_keyboard():
    d = Device(0); d.open_app("bank")
    d.act("tap Send money"); d.act("tap Alex Chen"); d.act("tap the amount box")
    html = render(d)
    assert "keyboard" in html and "caret" in html


def test_gesture_markers_only_mark_what_was_touched():
    """Rounds 5-6: synthetic touch markers were read as taps. A tap that opened a new screen
    left its ring on the new screen, and a scroll left an arrow mid-screen on some row
    ("You tapped Stolen Device Protection by mistake"). A marker may stay only on the control
    that was operated in place."""
    from screensim.core.engine import Device
    d = Device(0)
    d.open_app("settings")
    d.act("tap General")
    assert d.st.fx == [], "navigation leaves no ring on the new screen"
    d.act("swipe up")
    assert d.st.fx == [], "a scroll leaves no arrow"
    d.act("press back")
    for _ in range(4):
        if d.act("tap Airplane Mode").ok:
            break
        d.act("swipe down")
    fx = d.st.fx
    assert fx and fx[0]["type"] == "tap" and fx[0]["point"][0] > 300, "a toggle's ring sits on its switch"
