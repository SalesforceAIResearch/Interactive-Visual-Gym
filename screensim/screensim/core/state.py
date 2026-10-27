"""DeviceState — the single JSON scene graph, and its deterministic hash.

Persistent state only lives here: per-app data, the navigation stack, the
notification queue, active timers, the keyboard/focus, and any transient
touch-feedback marks for the current frame. Screens (widgets, bounds) are
rebuilt from this every tick and are NOT stored here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Timer:
    id: str
    expires_at: int          # tick
    on_expire: tuple         # event applied when it fires
    label: str = ""
    visible: bool = True
    tone_warn_at: Optional[int] = None   # tick to turn amber
    fired: bool = False


@dataclass
class Notification:
    id: str
    app: str
    title: str
    text: str
    arrived_at: int
    on_tap: Optional[tuple] = None
    sticky: bool = False


@dataclass
class DeviceState:
    tick: int = 0
    # navigation
    app: str = "home"                     # current foreground app id
    stack: list = field(default_factory=lambda: ["home"])   # screen keys
    # per-app persistent data (dicts owned by each app module)
    data: dict = field(default_factory=dict)
    scroll: dict = field(default_factory=dict)     # screen_key -> scroll px
    # input
    focus: Optional[str] = None           # focused field widget id
    keyboard: bool = False
    # system surfaces
    shade_open: bool = False
    cc_open: bool = False              # Control Center overlay
    dialog: Optional[dict] = None         # {"screen": key, "data": {...}}
    toast: Optional[dict] = None          # {"text":..., "until": tick}
    notifications: list = field(default_factory=list)
    timers: list = field(default_factory=list)
    # status bar
    clock: str = "9:41"
    battery: int = 82
    net: str = "wifi"                     # wifi | cellular | none
    # outcome
    outcome: Optional[str] = None         # won | wrong_commit | timeout | locked_out
    outcome_detail: str = ""
    # transient (per-tick) feedback marks, excluded from the semantic hash
    fx: list = field(default_factory=list)

    def cur_screen_key(self) -> str:
        if self.dialog:
            return self.dialog["screen"]
        return self.stack[-1] if self.stack else "home"

    def push(self, key: str):
        self.stack.append(key)
        self._on_nav()

    def pop(self):
        if len(self.stack) > 1:
            self.stack.pop()
        self._on_nav()

    def replace(self, key: str):
        if self.stack:
            self.stack[-1] = key
        else:
            self.stack = [key]
        self._on_nav()

    def _on_nav(self):
        # changing screens dismisses the soft keyboard and clears field focus
        self.focus = None
        self.keyboard = False

    def goto_app(self, app: str, root_key: str):
        self.app = app
        self.stack = [root_key]
        self.focus = None
        self.keyboard = False

    def clone(self) -> "DeviceState":
        return copy.deepcopy(self)

    # --- serialization ---
    def semantic_dict(self) -> dict:
        """State that defines the world, excluding transient fx and cosmetic
        clock. This is what the replay hash and injection keys read."""
        return {
            "app": self.app,
            "stack": list(self.stack),
            "data": self.data,
            "focus": self.focus,
            "keyboard": self.keyboard,
            "shade": self.shade_open,
            "cc": self.cc_open,
            "dialog": self.dialog,
            "notifs": [n.id for n in self.notifications],
            "timers": sorted((t.id, t.expires_at, t.fired) for t in self.timers),
            "net": self.net,
            "outcome": self.outcome,
        }

    def hash(self) -> str:
        blob = json.dumps(self.semantic_dict(), sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:16]

    def render_dict(self) -> dict:
        """Everything the renderer / observation text needs (adds status bar,
        toast, fx). Superset of semantic_dict."""
        d = self.semantic_dict()
        d.update(tick=self.tick, clock=self.clock, battery=self.battery,
                 toast=self.toast, fx=list(self.fx),
                 notifications=[vars(n) for n in self.notifications],
                 timers=[vars(t) for t in self.timers])
        return d
