"""Widget model and geometry.

Widgets are rebuilt every tick from persistent app state by each app's view
function; they are ephemeral and may hold event names, never state. All
geometry lives here in Python — the HTML renderer positions elements from
these exact bounds, so grounding, visibility and pixels can never diverge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Screen metrics (iPhone-class portrait). The renderer mirrors these exactly.
SCREEN_W = 412
SCREEN_H = 892
STATUS_H = 54          # taller safe area (Dynamic Island)
APPBAR_H = 96          # nav row + large title
NAVBAR_H = 34          # home-indicator area
KEYBOARD_H = 300


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    def intersects(self, other: "Rect") -> bool:
        return not (self.x2 <= other.x or other.x2 <= self.x
                    or self.y2 <= other.y or other.y2 <= self.y)

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x2 and self.y <= y < self.y2


# Widget kinds understood by layout, grounding and the renderer.
KINDS = {
    "row",          # list row: icon, label, sub, optional trailing widget ref
    "toggle",       # a row whose trailing control is a switch; value: bool
    "button",       # filled/tonal button; meta.style: filled|tonal|text|danger
    "icon_button",  # small square icon button (app bar actions, etc.)
    "field",        # text input; value: current text; meta.hint, meta.mask
    "radio",        # single-select row; value: bool (selected)
    "checkbox",     # multi-select row; value: bool
    "slider",       # value: 0-100
    "section",      # non-interactive section header
    "text",         # non-interactive paragraph / caption
    "banner",       # alert/countdown banner; meta.tone: info|warn|alert
    "tile",         # grid tile (photo, product, app icon); meta.color/img
    "chip",         # small pill (filters, suggestions)
    "tab",          # bottom-nav or top tab
    "notif",        # notification card (in shade)
    "qs_tile",      # quick-settings tile; value: bool
    "keycap",       # keyboard/pin-pad key
    "listitem",     # message/mail row with swipe actions
    "image",        # decorative image block
    "spacer",
}


@dataclass
class Widget:
    kind: str
    id: str
    label: str = ""
    sub: str = ""
    value: Any = None
    icon: str = ""
    enabled: bool = True
    on_tap: Optional[tuple] = None          # event tuple, e.g. ("nav", "settings.network")
    on_long: Optional[tuple] = None
    on_swipe: dict = field(default_factory=dict)   # {"left": event, "right": event}
    h: int = 56
    w: Optional[int] = None                 # None = full row width (minus margins)
    meta: dict = field(default_factory=dict)
    # Assigned by layout:
    bounds: Optional[Rect] = None
    visible: bool = False
    row: int = -1                           # index in the laid-out sequence

    def interactive(self) -> bool:
        return bool(self.on_tap or self.on_long or self.on_swipe
                    or self.kind in ("field", "toggle", "slider", "checkbox", "radio", "qs_tile"))


@dataclass
class Screen:
    """One screen of an app: an app-bar title plus a widget list.

    `grid` lays widgets out N per row (photo grids, keypads, app grids).
    `scrollable=False` pins content (dialogs, call screen).
    """
    key: str                    # e.g. "settings.hotspot"
    title: str
    widgets: list
    grid: int = 1
    scrollable: bool = True
    show_back: bool = True
    fab: Optional[Widget] = None     # floating action button
    appbar_actions: list = field(default_factory=list)   # icon_buttons
    is_dialog: bool = False
    sheet: bool = False                               # bottom action-sheet dialog
    bottom_nav: list = field(default_factory=list)   # tabs pinned to the bottom
    card_rect: object = None                          # dialog card bounds
    back_label: str = ""                              # override for the ‹ Back label


def w_row(id, label, sub="", icon="", on_tap=None, **meta) -> Widget:
    return Widget("row", id, label=label, sub=sub, icon=icon, on_tap=on_tap,
                  h=62 if sub else 48, meta=meta)


def w_toggle(id, label, value, sub="", icon="", event=None, enabled=True) -> Widget:
    return Widget("toggle", id, label=label, sub=sub, icon=icon, value=bool(value),
                  enabled=enabled, on_tap=event or ("toggle", id),
                  h=62 if sub else 48)


def w_button(id, label, event, style="filled", enabled=True, w=None) -> Widget:
    return Widget("button", id, label=label, on_tap=event, enabled=enabled,
                  h=50, w=w, meta={"style": style})


def w_field(id, label, value="", hint="", mask=False, kb="text", enabled=True) -> Widget:
    return Widget("field", id, label=label, value=value, enabled=enabled, h=52,
                  on_tap=("focus", id), meta={"hint": hint, "mask": mask, "kb": kb})


def w_section(label) -> Widget:
    return Widget("section", f"sec_{label}", label=label, h=44)


def w_text(id, label, sub="", h=None) -> Widget:
    lines = max(1, len(label) // 46 + 1) + (1 if sub else 0)
    return Widget("text", id, label=label, sub=sub, h=h or (20 + 22 * lines))


def w_banner(id, label, tone="info", sub="") -> Widget:
    return Widget("banner", id, label=label, sub=sub, h=64 if sub else 48,
                  meta={"tone": tone})
