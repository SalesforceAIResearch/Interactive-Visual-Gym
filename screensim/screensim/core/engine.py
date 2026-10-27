"""ScreenSim engine — the deterministic step loop.

    dev = Device(seed=0)
    dev.open_app("settings")
    res = dev.act("tap Network & internet")   # a gesture string or Gesture
    ...

One `act()` is one high-level gesture; it advances the clock by the gesture's
tick cost, resolves timers/hazards each tick, and returns an ActionResult with
the spoken feedback the human/assistant loops consume. The engine never raises
on bad input — a bad gesture is a clean refusal, exactly like CookSim.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Union

from . import app as app_mod
from . import shell
from .actions import (CLEAR, Gesture, Ground, LONG_PRESS, PRESS_KEY, SWIPE,
                      TAP, TYPE, WAIT, ground, parse)
from .layout import layout
from .model import Screen, Widget
from .state import DeviceState, Timer


@dataclass
class ActionResult:
    ok: bool
    status: str                 # done|refused|ambiguous|no_target
    reason: str = ""
    gesture: Optional[Gesture] = None
    widget_id: str = ""
    event: tuple = ()
    ticks: int = 1
    ambiguous: bool = False
    n_candidates: int = 0
    outcome: Optional[str] = None

    def __bool__(self):
        return self.ok


TICK_COSTS = {TAP: 1, LONG_PRESS: 2, SWIPE: 1, CLEAR: 1, PRESS_KEY: 1, WAIT: 1}


class Device:
    def __init__(self, seed: int = 0, tick_cap: int = 600):
        self.rng = random.Random(seed)
        self.seed = seed
        self.tick_cap = tick_cap
        self.st = DeviceState()
        self.log: list = []
        # initialize every registered app's persistent data
        for a in app_mod.all_apps():
            self.st.data[a.id] = a.initial_data(
                random.Random(seed ^ int.from_bytes(a.id.encode()[:4].ljust(4, b'_'), 'big')))
        shell.init(self.st)

    # ---- view ----
    def current_screen(self) -> Screen:
        return shell.current_screen(self.st)

    def laid_out_screen(self) -> Screen:
        scr = self.current_screen()
        if scr.is_dialog:
            from .layout import layout_dialog
            scr.card_rect = layout_dialog(scr)
            return scr
        scroll = 0 if not scr.scrollable else self.st.scroll.get(scr.key, 0)
        layout(scr, scroll, self.st.keyboard, has_appbar=scr.title is not None)
        return scr

    # ---- navigation helpers ----
    def open_app(self, app_id: str):
        a = app_mod.get(app_id)
        self.st.shade_open = False
        self.st.dialog = None
        self.st.goto_app(app_id, a.root_key)

    # ---- the step ----
    def act(self, cmd: Union[str, Gesture]) -> ActionResult:
        if self.st.outcome:
            return ActionResult(False, "done", reason="the task is already over",
                                outcome=self.st.outcome)
        g = parse(cmd) if isinstance(cmd, str) else cmd
        if g is None:
            self._tick(1)
            return ActionResult(False, "no_target", reason="I didn't catch a gesture there")

        scr = self.laid_out_screen()
        gr = ground(g, scr.widgets)
        cost = TICK_COSTS.get(g.verb, 1)

        if not gr.ok:
            self.st.fx = [{"type": "miss"}]
            self._tick(cost)
            res = ActionResult(False,
                               "ambiguous" if gr.ambiguous else ("refused" if gr.reason else "no_target"),
                               reason=gr.reason, gesture=g,
                               ambiguous=gr.ambiguous, n_candidates=gr.n_candidates,
                               ticks=cost, outcome=self.st.outcome)
            self.log.append(self._log_entry(g, res))
            return res

        where = (self.st.app, self.st.cur_screen_key())
        event, note, ok = self._apply(g, gr, scr)
        if (self.st.fx and self.st.fx[0].get("type") in ("tap", "long", "swipe")
                and (self.st.app, self.st.cur_screen_key()) != where):
            # the tap opened another screen: its touch ring belongs to the screen it left, and
            # drawn on the new one it lands on whatever happens to sit at that spot
            self.st.fx = []
        self._tick(cost)
        res = ActionResult(ok, "done" if ok else "refused", reason=note or "",
                           gesture=g,
                           widget_id=gr.widget.id if gr.widget else "",
                           event=event or (), ticks=cost, outcome=self.st.outcome)
        self.log.append(self._log_entry(g, res))
        return res

    def run(self, plan: list) -> list:
        out = []
        for cmd in plan:
            r = self.act(cmd)
            out.append(r)
            if self.st.outcome:
                break
        return out

    # ---- gesture application ----
    # each branch returns (event, note, ok)
    def _apply(self, g: Gesture, gr: Ground, scr: Screen):
        st = self.st
        st.fx = []

        if g.verb == WAIT:
            return None, None, True

        if g.verb == PRESS_KEY:
            st.fx = [{"type": "navpress", "key": g.key}]
            return shell.press_key(st, g.key), None, True

        if g.verb == SWIPE:
            return self._swipe(g, gr, scr)

        if g.verb == CLEAR:
            if st.focus:
                self._set_field(st, st.focus, "")
                st.fx = [{"type": "type", "id": st.focus}]
                return ("clear", st.focus), None, True
            st.fx = [{"type": "miss"}]
            return None, "no field is focused", False

        if g.verb == TYPE:
            if not st.focus or not st.keyboard:
                st.fx = [{"type": "miss"}]
                return None, "no field is focused — tap the box first", False
            fw = self._find_widget(scr, st.focus)
            if fw and fw.meta.get("kb") == "number" and not _is_number(g.text):
                st.fx = [{"type": "miss"}]
                return None, "that field only takes numbers", False
            self._set_field(st, st.focus, g.text)
            st.fx = [{"type": "type", "id": st.focus}]
            return ("type", st.focus, g.text), None, True

        # tap / long_press
        w = gr.widget
        if w is None:
            st.fx = [{"type": "tap", "point": gr.point}]
            return None, None, True   # tapped empty space: harmless
        if not w.enabled:
            st.fx = [{"type": "tap", "point": gr.point, "disabled": True}]
            return None, self._disabled_reason(w), False

        # a switch row is operated on its switch (right edge), as on iOS; a ring in the middle of
        # the row reads as a tap that missed the switch
        px = (w.bounds.x2 - 41) if w.kind == "toggle" else w.bounds.cx
        st.fx = [{"type": ("long" if g.verb == LONG_PRESS else "tap"),
                  "point": (px, w.bounds.cy), "id": w.id}]

        event = w.on_long if g.verb == LONG_PRESS else w.on_tap
        if event is None and w.kind == "slider":
            lab = w.label or "the slider"
            return None, (f"tapping a slider doesn't move it — swipe right on {lab} to raise it, "
                          f"swipe left on {lab} to lower it"), False
        if event is None and w.kind == "toggle":
            event = ("toggle", w.id)
        if event is None:
            return None, None, True

        note = self._dispatch(st, event, w, g)
        return event, note, True

    def _swipe(self, g: Gesture, gr: Ground, scr: Screen):
        st = self.st
        if g.anchor_text and gr.widget is not None:
            w = gr.widget
            ev = w.on_swipe.get(g.direction) if w.on_swipe else None
            st.fx = [{"type": "swipe", "dir": g.direction,
                      "point": (w.bounds.cx, w.bounds.cy), "id": w.id}]
            if ev is None:
                return None, f"that row has no {g.direction} swipe action", False
            note = self._dispatch(st, ev, w, g)
            return ev, note, True
        # edge swipes: Control Center (top-right) / Notification Center (top)
        if g.direction == "down" and g.region == "top-right" and not st.shade_open \
                and not getattr(st, "cc_open", False):
            st.fx = [{"type": "swipe", "dir": "down", "point": (370, 30)}]
            st.cc_open = True
            return ("open_cc",), None, True
        if g.direction == "down" and g.region == "top" and not st.shade_open \
                and not getattr(st, "cc_open", False):
            st.fx = [{"type": "swipe", "dir": "down", "point": (150, 30)}]
            return shell.handle(st, ("open_shade",)), None, True
        if g.direction == "up" and getattr(st, "cc_open", False):
            st.cc_open = False
            st.fx = []
            return ("close_cc",), None, True
        # whole-screen scroll: map swipe direction to content movement. No marker: the content
        # has moved under the finger, and an arrow left mid-screen sits on some row it never
        # touched (assistants read it as a tap on that row)
        st.fx = []
        if g.direction in ("up", "down"):
            key = scr.key
            cur = st.scroll.get(key, 0)
            delta = 360 if g.direction == "up" else -360   # swipe up -> content up
            ms, _ = layout(scr, cur, st.keyboard, has_appbar=scr.title is not None)
            new = max(0, min(ms, cur + delta))
            st.scroll[key] = new
            if new == cur:
                return (("scroll", key, new),
                        ("already at the bottom" if g.direction == "up"
                         else "already at the top"), True)
            return ("scroll", key, new), None, True
        # horizontal screen swipe right: a back gesture
        if g.direction == "right":
            return shell.press_key(st, "back"), None, True
        return None, None, True

    def _dispatch(self, st, event, w, g):
        """Route an event: shell handles system events; else the owning app."""
        kind = event[0]
        if kind == "noop":
            # a placeholder control: it is a LEGAL action but has no effect in
            # this build. Announce it (never a silent no-op) so the affordance
            # contract holds — every legal action either changes the world or
            # says why it did not.
            label = (w.label or "This").strip() if w else "This"
            st.toast = {"text": f"“{label}” isn’t available in this demo.",
                        "until": st.tick + 3}
            return None
        if kind == "focus":
            st.focus = event[1]
            st.keyboard = True
            return None
        if kind == "toggle":
            return self._do_toggle(st, w, event[1])
        if kind in shell.SHELL_EVENTS:
            return shell.handle(st, event)
        # app event
        owner = app_mod.owner_of(st.cur_screen_key()) or app_mod.get_opt(st.app)
        if kind == "nav":
            # navigation is a shell concern but target belongs to an app
            return shell.navigate(st, event[1])
        note = owner.handle(st, event, g) if owner else None
        return note

    def _do_toggle(self, st, w, wid):
        owner = app_mod.owner_of(st.cur_screen_key()) or app_mod.get_opt(st.app)
        # let the app veto / react; default flips the stored bool
        note = owner.handle(st, ("toggle", wid), None) if owner else None
        return note

    # ---- field helpers ----
    def _set_field(self, st, field_id, text):
        owner = app_mod.owner_of(st.cur_screen_key()) or app_mod.get_opt(st.app)
        if owner:
            owner.handle(st, ("set_field", field_id, text), None)

    def _find_widget(self, scr, wid):
        for w in scr.widgets:
            if w.id == wid:
                return w
        return None

    def _disabled_reason(self, w: Widget) -> str:
        return w.meta.get("disabled_reason",
                          f"{w.label or 'that'} is greyed out right now")

    # ---- clock & hazards ----
    def _tick(self, n: int):
        for _ in range(n):
            if self.st.outcome:
                return
            self.st.tick += 1
            # toast expiry
            if self.st.toast and self.st.tick >= self.st.toast.get("until", 0):
                self.st.toast = None
            # timers
            for t in list(self.st.timers):
                if not t.fired and self.st.tick >= t.expires_at:
                    t.fired = True
                    note = self._dispatch_timer(t)
            if self.st.tick >= self.tick_cap and not self.st.outcome:
                self.st.outcome = "timeout"
                self.st.outcome_detail = "ran out of time"

    def _dispatch_timer(self, t: Timer):
        ev = t.on_expire
        if not ev:
            return
        if ev[0] == "outcome":
            self.st.outcome = ev[1]
            self.st.outcome_detail = t.label
            return
        if ev[0] in shell.SHELL_EVENTS:
            return shell.handle(self.st, ev)
        owner = app_mod.owner_of(ev[1]) if len(ev) > 1 and isinstance(ev[1], str) else None
        owner = owner or app_mod.get_opt(self.st.app)   # None on the home shell
        if owner:
            owner.handle(self.st, ev, None)

    # ---- logging ----
    def _log_entry(self, g, res):
        return {"t": self.st.tick, "gesture": g.describe(), "verb": g.verb,
                "status": res.status, "reason": res.reason,
                "widget": res.widget_id, "event": list(res.event),
                "screen": self.st.cur_screen_key(), "outcome": self.st.outcome}


def _is_number(s: str) -> bool:
    return bool(s) and all(c.isdigit() or c in " -" for c in s)
