"""Photos — grid, select, share (cross-app), delete (delete-wrong-thing error)."""
from __future__ import annotations

from ..core.app import App, register
from ..core.model import (Screen, Widget, w_button, w_row, w_section, w_text)
from ..core.state import DeviceState


# each photo gets a CSS gradient + a scene glyph so the grid reads as real
# photographs, not flat color swatches.
_SCENES = [
    ("Beach sunset",   "linear-gradient(160deg,#ff9a5a,#ffd27f 45%,#5aa9d6)", "🌅"),
    ("Receipt — lunch","linear-gradient(160deg,#f5f2e8,#dcd6c3)",             "🧾"),
    ("Whiteboard",     "linear-gradient(160deg,#f7f8fa,#e3e6ec)",             "📋"),
    ("Dog park",       "linear-gradient(160deg,#8fce6b,#5a8f3c 70%,#c8b88a)", "🐕"),
    ("Passport scan",  "linear-gradient(160deg,#3a5a8c,#22345a)",             "🛂"),
    ("Concert",        "linear-gradient(160deg,#3a1f5a,#a03ad0 60%,#ff5aa0)", "🎤"),
    ("Recipe card",    "linear-gradient(160deg,#fff4e0,#f0d9b0)",             "🍳"),
    ("Sunset hike",    "linear-gradient(160deg,#ff7e5a,#feb47b 50%,#7a5a8c)", "🏔"),
]


class PhotosApp(App):
    id = "photos"
    name = "Photos"
    icon = "photo"
    color = "#7b5bd4"
    root_key = "photos.root"

    def initial_data(self, rng) -> dict:
        photos = []
        for i, (nm, grad, glyph) in enumerate(_SCENES):
            photos.append({"id": f"ph{i}", "name": nm, "color": grad,
                           "glyph": glyph, "fav": i in (0, 3)})
        return {"photos": photos, "selected": [], "trash": []}

    def view(self, st) -> Screen:
        key = st.cur_screen_key().split(".", 1)[1]
        d = st.data["photos"]
        if key.startswith("view_"):
            return self._view_one(st, d, key[len("view_"):])
        return self._grid(st, d)

    def _grid(self, st, d):
        ws = [w_section("Recent")]
        sel = d["selected"]
        for p in d["photos"]:
            checked = p["id"] in sel
            ws.append(Widget("tile", f"ph_{p['id']}", label=p["name"], h=120,
                             on_tap=("open_photo", p["id"]),
                             on_long=("select_photo", p["id"]),
                             meta={"color": p["color"], "selected": checked,
                                   "fav": p["fav"], "glyph": p.get("glyph", ""),
                                   "photo": True}))
        s = Screen("photos.root", "Photos", ws, grid=3)
        if sel:
            s.title = f"{len(sel)} selected"
            s.appbar_actions = [
                Widget("icon_button", "act_share", icon="share", h=40,
                       on_tap=("share_selected",)),
                Widget("icon_button", "act_delete", icon="trash", h=40,
                       on_tap=("delete_selected",)),
            ]
        return s

    def _view_one(self, st, d, pid):
        p = next((x for x in d["photos"] if x["id"] == pid), None)
        if not p:
            return self._grid(st, d)
        ws = [
            Widget("image", f"big_{pid}", label=p["name"], h=320,
                   meta={"color": p["color"], "glyph": p.get("glyph", "")}),
            w_text("pv_name", p["name"], h=40),
            Widget("button", "pv_share", label="Share", on_tap=("share_one", pid),
                   h=48, w=170, meta={"style": "tonal"}),
            Widget("button", "pv_delete", label="Delete", on_tap=("delete_one", pid),
                   h=48, w=170, meta={"style": "danger"}),
        ]
        return Screen(f"photos.view_{pid}", p["name"], ws)

    def handle(self, st, event, gesture):
        d = st.data["photos"]
        k = event[0]
        if k == "open_photo":
            st.push(f"photos.view_{event[1]}")
            return None
        if k == "select_photo":
            if event[1] in d["selected"]:
                d["selected"].remove(event[1])
            else:
                d["selected"].append(event[1])
            return None
        if k == "delete_one":
            self._delete(st, d, [event[1]])
            st.pop()
            return None
        if k == "delete_selected":
            st.dialog = {
                "screen": "photos.dlg_del",
                "title": f"Delete {len(d['selected'])} item(s)?",
                "body": "They'll move to Trash for 30 days.",
                "buttons": [("Delete", ("confirm_delete",), "danger"),
                            ("Cancel", ("close_dialog",), "text")],
                "dismissable": True,
            }
            return None
        if k == "confirm_delete":
            self._delete(st, d, list(d["selected"]))
            d["selected"] = []
            st.dialog = None
            return None
        if k == "share_one":
            return self._open_share(st, [event[1]])
        if k == "share_selected":
            return self._open_share(st, list(d["selected"]))
        return None

    def _delete(self, st, d, ids):
        d["trash"].extend(ids)
        d["photos"] = [p for p in d["photos"] if p["id"] not in ids]
        st.toast = {"text": f"Deleted {len(ids)} item(s)", "until": st.tick + 3}

    def _open_share(self, st, ids):
        st.dialog = {
            "screen": "photos.share_sheet",
            "title": "1 Photo Selected",
            "body": "",
            "sheet": True,
            "buttons": [("Messages", ("share_to", "messages", ids), "tonal"),
                        ("Mail", ("share_to", "mail", ids), "tonal"),
                        ("Save to Files", ("close_dialog",), "tonal"),
                        ("Cancel", ("close_dialog",), "cancel")],
            "dismissable": True,
        }
        return None


register(PhotosApp())
