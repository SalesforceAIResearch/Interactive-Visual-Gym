"""Deterministic layout: widget list -> bounds, with a scroll viewport.

Vertical flow with fixed metrics; `grid` packs N widgets per row. The
content area sits between the app bar and the nav bar (or keyboard). A
widget is `visible` when its bounds intersect the viewport after scroll.
"""
from __future__ import annotations

from .model import (APPBAR_H, KEYBOARD_H, NAVBAR_H, Rect, SCREEN_H, SCREEN_W,
                    STATUS_H, Screen, Widget)

MARGIN = 16
GUTTER = 10
GROUP_GAP = 22          # space between grouped-list blocks / other widgets

# widget kinds that merge into an iOS inset-grouped list (flush, hairline rules)
GROUPABLE = {"row", "toggle", "radio", "checkbox", "listitem"}


def _groupable(wd):
    return (wd.kind in GROUPABLE or wd.meta.get("mailrow")) \
        and not wd.meta.get("bubble")


def content_top(has_appbar: bool = True) -> int:
    return STATUS_H + (APPBAR_H if has_appbar else 0)


def content_bottom(keyboard_open: bool) -> int:
    return SCREEN_H - (KEYBOARD_H if keyboard_open else NAVBAR_H)


def layout(screen: Screen, scroll: int, keyboard_open: bool,
           has_appbar: bool = True) -> tuple:
    """Assign bounds/visible to screen.widgets. Returns (max_scroll, viewport)."""
    top = content_top(has_appbar)
    bottom = content_bottom(keyboard_open)
    viewport = Rect(0, top, SCREEN_W, bottom - top)

    # absolutely-positioned widgets (Control Center modules, nav-bar actions)
    # take their bounds from meta["abs"] and never join the flow
    flow = []
    for wd in screen.widgets:
        ab = wd.meta.get("abs")
        if ab:
            wd.bounds = Rect(*ab)
            wd.visible = True
        else:
            flow.append(wd)
    # nav-bar action buttons: right-aligned in the small top bar
    ax = SCREEN_W - 16
    for a in reversed(screen.appbar_actions):
        aw = a.w or 40
        a.bounds = Rect(ax - aw, STATUS_H + 4, aw, 36)
        a.visible = True
        ax -= aw + 8
        if a not in screen.widgets:
            screen.widgets.append(a)
    # bottom toolbar buttons (iOS Mail-style): evenly spread; sits above the
    # keyboard when it is open, else above the home indicator.
    if screen.bottom_nav:
        nb = screen.bottom_nav
        cell = SCREEN_W // len(nb)
        toolbar_y = (bottom - 52) if keyboard_open else (SCREEN_H - NAVBAR_H - 52)
        for k, b in enumerate(nb):
            b.bounds = Rect(k * cell, toolbar_y, cell, 52)
            b.visible = True
            if b not in screen.widgets:
                screen.widgets.append(b)

    y = top + GUTTER - scroll
    i = 0
    row_idx = 0
    widgets = flow
    n = len(widgets)
    while i < n:
        if screen.grid > 1 and widgets[i].kind in ("tile", "keycap", "cc_tile"):
            # pack a grid row of same-kind widgets
            j = i
            while j < n and widgets[j].kind == widgets[i].kind and j - i < screen.grid:
                j += 1
            count = j - i
            cell_w = (SCREEN_W - 2 * MARGIN - (screen.grid - 1) * GUTTER) // screen.grid
            for k in range(count):
                wd = widgets[i + k]
                wd.bounds = Rect(MARGIN + k * (cell_w + GUTTER), y, cell_w, wd.h)
                wd.row = row_idx
            y += widgets[i].h + GUTTER
            i = j
        elif _groupable(widgets[i]):
            # an iOS inset-grouped block: consecutive groupable rows, flush,
            # tagged first/mid/last so the renderer draws one rounded card with
            # hairline separators.
            j = i
            while j < n and _groupable(widgets[j]):
                j += 1
            run = widgets[i:j]
            for idx, wd in enumerate(run):
                ww = SCREEN_W - 2 * MARGIN
                wd.bounds = Rect(MARGIN, y, ww, wd.h)
                wd.row = row_idx + idx
                pos = ("single" if len(run) == 1 else
                       "first" if idx == 0 else
                       "last" if idx == len(run) - 1 else "mid")
                wd.meta["grp"] = pos
                y += wd.h        # flush, no gap within the group
            y += GROUP_GAP
            row_idx += len(run)
            i = j
            continue
        else:
            wd = widgets[i]
            bubble = wd.meta.get("bubble")
            if bubble:
                ww = wd.w or min(SCREEN_W - 2 * MARGIN,
                                 max(90, 24 + len(wd.label) * 9))
                x = (SCREEN_W - MARGIN - ww) if bubble == "me" else MARGIN
            else:
                ww = wd.w or (SCREEN_W - 2 * MARGIN)
                x = (SCREEN_W - ww) // 2 if wd.meta.get("center") else MARGIN
            wd.bounds = Rect(x, y, ww, wd.h)
            wd.row = row_idx
            gap = 4 if wd.kind == "section" else GUTTER
            y += wd.h + gap
            i += 1
        row_idx += 1

    content_h = y + scroll - top
    max_scroll = max(0, content_h - viewport.h + GUTTER)
    for wd in widgets:
        wd.visible = wd.bounds is not None and wd.bounds.intersects(viewport)
    return max_scroll, viewport


def layout_sheet(screen: Screen) -> Rect:
    """iOS action sheet: bottom-anchored, full-width stacked buttons, a Cancel
    button separated below. Assigns button/title bounds and returns the sheet
    Rect (unused by the renderer but kept for parity)."""
    side = 10
    w = SCREEN_W - 2 * side
    x = side
    buttons = [b for b in screen.widgets if b.kind == "button"]
    titles = [t for t in screen.widgets if t.kind == "text" and t.id.endswith("title")]
    title_h = 54 if titles else 0
    btn_h = 58
    cancel = [b for b in buttons if b.meta.get("style") == "cancel"]
    mains = [b for b in buttons if b.meta.get("style") != "cancel"]
    gap = 8
    cancel_block = (btn_h + gap) if cancel else 0
    total_h = title_h + btn_h * len(mains) + cancel_block
    y = SCREEN_H - NAVBAR_H - 10 - total_h
    top = y
    for t in titles:
        t.bounds = Rect(x, y, w, title_h)
        t.visible = True
    y += title_h
    for b in mains:
        b.bounds = Rect(x, y, w, btn_h)
        b.visible = True
        y += btn_h
    y += gap
    for b in cancel:
        b.bounds = Rect(x, y, w, btn_h)
        b.visible = True
    return Rect(x, top, w, total_h)


def layout_dialog(screen: Screen) -> Rect:
    if getattr(screen, "sheet", False):
        return layout_sheet(screen)
    """Lay out a dialog: a centered card whose buttons get real bounds in a
    right-aligned action row, so grounding, the tap ripple and the rendered
    pixels all agree. Returns the card Rect (the renderer draws the card behind
    the button bounds). Title/body are non-interactive and get card-relative
    bounds too."""
    # iOS alert: 270 wide, centered; title + wrapped body; button row(s)
    # pinned to the card bottom (two buttons side-by-side, Cancel on the
    # left; 3+ stack). Height grows with the body so buttons never clip.
    card_w = 270
    card_x = (SCREEN_W - card_w) // 2
    titles = [w for w in screen.widgets if w.kind == "text" and w.id.endswith("title")]
    bodies = [w for w in screen.widgets if w.kind == "text" and not w.id.endswith("title")]
    buttons = [w for w in screen.widgets if w.kind == "button"]
    # iOS: for a two-button alert Cancel sits on the left
    if len(buttons) == 2 and buttons[0].label.lower() != "cancel" \
            and buttons[1].label.lower() == "cancel":
        buttons = [buttons[1], buttons[0]]

    inner_w = card_w - 32
    title_h = sum(22 * max(1, -(-len(w.label) // 24)) for w in titles)
    body_h = sum(18 * max(1, -(-len(w.label) // 35)) for w in bodies)
    btn_h = 48
    two = len(buttons) == 2
    act_h = btn_h if two else btn_h * len(buttons)
    head_h = 18 + title_h + (4 + body_h if bodies else 0) + 14
    card_h = head_h + act_h
    card_y = (SCREEN_H - card_h) // 2

    y = card_y + 18
    for w in titles:
        w.bounds = Rect(card_x + 16, y, inner_w, title_h)
        w.visible = True
    y += title_h + 4
    for w in bodies:
        w.bounds = Rect(card_x + 16, y, inner_w, body_h)
        w.visible = True
    ay = card_y + head_h
    if two:
        half = card_w // 2
        buttons[0].bounds = Rect(card_x, ay, half, btn_h)
        buttons[1].bounds = Rect(card_x + half, ay, card_w - half, btn_h)
    else:
        for i, w in enumerate(buttons):
            w.bounds = Rect(card_x, ay + i * btn_h, card_w, btn_h)
    for w in buttons:
        w.visible = True
        w.enabled = True
    return Rect(card_x, card_y, card_w, card_h)


def region_of(r: Rect) -> set:
    """Named screen regions a widget belongs to, for spatial grounding."""
    regions = set()
    if r.cy < STATUS_H + APPBAR_H + 40:
        regions.add("top")
    if r.cy > SCREEN_H * 2 // 3:
        regions.add("bottom")
    if r.cx < SCREEN_W // 3:
        regions.add("left")
    if r.cx > SCREEN_W * 2 // 3:
        regions.add("right")
    if not regions:
        regions.add("middle")
    return regions
