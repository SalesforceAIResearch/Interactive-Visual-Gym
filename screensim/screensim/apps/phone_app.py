"""Phone — incoming call as a hazard clock; dialer; contacts."""
from __future__ import annotations

from ..core.app import App, register
from ..core.model import (Screen, Widget, w_button, w_row, w_section, w_text)
from ..core.state import DeviceState


class PhoneApp(App):
    id = "phone"
    name = "Phone"
    icon = "phone"
    color = "#2e8b57"
    root_key = "phone.root"

    def initial_data(self, rng) -> dict:
        return {
            "contacts": [
                {"id": "mom", "name": "Mom", "num": "+1 555 0100"},
                {"id": "clinic", "name": "Downtown Clinic", "num": "+1 555 0777"},
                {"id": "alex", "name": "Alex Chen", "num": "+1 555 0142"},
            ],
            "incoming": None,     # {"name":..., "num":...} when ringing
            "in_call": None,
            "mic_muted": False,
            "speaker": False,
        }

    def view(self, st) -> Screen:
        key = st.cur_screen_key().split(".", 1)[1]
        d = st.data["phone"]
        if key == "incall":
            return self._incall(st, d)
        if key == "ringing":
            return self._ringing(st, d)
        return self._root(st, d)

    def _root(self, st, d):
        ws = [w_section("Contacts")]
        for c in d["contacts"]:
            ws.append(w_row(f"c_{c['id']}", c["name"], sub=c["num"],
                            on_tap=("call", c["id"]), avatar=True))
        return Screen("phone.root", "Phone", ws)

    def _ringing(self, st, d):
        inc = d["incoming"] or {}
        ws = [
            w_text("ring_name", inc.get("name", "Unknown"), sub=inc.get("num", ""), h=80),
            w_text("ring_lbl", "Incoming call…", h=40),
            w_button("answer", "Answer", ("answer",), style="filled"),
            w_button("decline", "Decline", ("decline",), style="danger"),
        ]
        return Screen("phone.ringing", None, ws, show_back=False, scrollable=False)

    def _incall(self, st, d):
        ic = d["in_call"] or {}
        ws = [
            w_text("call_name", ic.get("name", ""), sub="00:04", h=72),
            Widget("toggle", "mute", label="Mute", value=d["mic_muted"], h=56,
                   on_tap=("toggle_mute",), icon="mic"),
            Widget("toggle", "spk", label="Speaker", value=d["speaker"], h=56,
                   on_tap=("toggle_speaker",), icon="vol"),
            w_button("hang", "End call", ("hangup",), style="danger"),
        ]
        return Screen("phone.incall", None, ws, show_back=False, scrollable=False)

    def handle(self, st, event, gesture):
        d = st.data["phone"]
        k = event[0]
        if k == "call":
            c = next(x for x in d["contacts"] if x["id"] == event[1])
            d["in_call"] = {"name": c["name"], "num": c["num"]}
            st.push("phone.incall")
            return None
        if k == "ring":  # triggered by a task/timer
            d["incoming"] = event[1]
            st.goto_app("phone", "phone.ringing")
            return None
        if k == "answer":
            d["in_call"] = d["incoming"]
            d["incoming"] = None
            st.replace("phone.incall")
            return None
        if k == "decline":
            d["incoming"] = None
            st.goto_app("home", "home")
            return None
        if k == "toggle_mute":
            d["mic_muted"] = not d["mic_muted"]
            return None
        if k == "toggle_speaker":
            d["speaker"] = not d["speaker"]
            return None
        if k == "hangup":
            d["in_call"] = None
            st.goto_app("home", "home")
            return None
        return None


register(PhoneApp())
