"""Messages — thread list + conversation; OTP delivery channel; send-to-wrong."""
from __future__ import annotations

from ..core.app import App, register
from ..core.model import Screen, Widget, w_field, w_row, w_section, w_text
from ..core.state import DeviceState


class MessagesApp(App):
    id = "messages"
    name = "Messages"
    icon = "chat"
    color = "#2f6bd4"
    root_key = "messages.root"

    def initial_data(self, rng) -> dict:
        return {"threads": [
            {"id": "mom", "name": "Mom", "handle": "+1 555 0100", "time": "9:02",
             "msgs": [{"from": "them", "text": "Are you coming Sunday?"}], "unread": False},
            {"id": "alex", "name": "Alex Chen", "handle": "@alexc", "time": "9:31",
             "msgs": [{"from": "them", "text": "did you send the $40 yet?"}], "unread": True},
        ], "_open": None, "_draft": ""}

    def view(self, st) -> Screen:
        key = st.cur_screen_key().split(".", 1)[1]
        d = st.data["messages"]
        if key == "share":
            return self._share(st, d)
        if key.startswith("thread_"):
            return self._thread(st, d, key[len("thread_"):])
        return self._root(st, d)

    def _share(self, st, d):
        sh = d.get("_sharing", {})
        ws = [w_section("Send photo to"),
              Widget("image", "share_prev", label=sh.get("caption", "Photo"), h=140,
                     meta={"color": sh.get("grad", "#8aa1c1"),
                           "glyph": sh.get("glyph", "🖼")})]
        ws.append(w_section("Recipients"))
        for t in d["threads"]:
            picked = sh.get("to") == t["id"]
            ws.append(w_row(f"sh_{t['id']}", t["name"], sub=t["handle"],
                            on_tap=("share_pick", t["id"]), avatar=True,
                            **({"selected": True} if picked else {})))
        ws.append(Widget("button", "share_send", label="Send",
                         on_tap=("share_send",), h=48,
                         enabled=bool(sh.get("to")),
                         meta={"style": "filled",
                               "disabled_reason": "pick a recipient first"}))
        return Screen("messages.share", "Share", ws)

    def _root(self, st, d):
        ws = [w_field("msg_search", "Search", hint="Search messages"),
              w_section("Conversations")]
        for t in d["threads"]:
            last = t["msgs"][-1]["text"] if t["msgs"] else ""
            ws.append(Widget("listitem", f"th_{t['id']}", label=t["name"], sub=last,
                             h=72,
                             on_tap=("open_thread", t["id"]),
                             on_swipe={"left": ("archive_thread", t["id"])},
                             meta={"unread": t["unread"], "avatar": True,
                                   "time": t.get("time", "")}))
        return Screen("messages.root", "Messages", ws)

    def _thread(self, st, d, tid):
        t = next((x for x in d["threads"] if x["id"] == tid), None)
        if not t:
            return self._root(st, d)
        ws = [w_section(t["name"])]
        for m in t["msgs"]:
            if m.get("photo") or m["text"].startswith("📷"):
                wid = Widget("image", f"m_{id(m)}", label="", h=140, w=180,
                             meta={"color": m.get("grad", "linear-gradient(160deg,#ff9a5a,#ffd27f,#5aa9d6)"),
                                   "glyph": m.get("glyph", "🖼"), "bubble": m["from"]})
                ws.append(wid)
            else:
                ws.append(w_text(f"m_{id(m)}", m["text"], h=44))
                ws[-1].meta["bubble"] = m["from"]
        ws.append(w_field("compose", "Message", value=d.get("_draft", ""),
                          hint="Text message"))
        ws.append(Widget("button", "send_msg", label="Send", on_tap=("send_msg", tid),
                         h=48, meta={"style": "filled"}))
        return Screen(f"messages.thread_{tid}", t["name"], ws)

    def handle(self, st, event, gesture):
        d = st.data["messages"]
        k = event[0]
        if k == "open_thread":
            t = next(x for x in d["threads"] if x["id"] == event[1])
            t["unread"] = False
            st.push(f"messages.thread_{event[1]}")
            return None
        if k == "set_field":
            if event[1] == "compose":
                d["_draft"] = event[2]
            return None
        if k == "send_msg":
            t = next(x for x in d["threads"] if x["id"] == event[1])
            if d.get("_draft"):
                t["msgs"].append({"from": "me", "text": d["_draft"]})
                d["_draft"] = ""
                st.toast = {"text": "Sent", "until": st.tick + 2}
            return None
        if k == "archive_thread":
            d["threads"] = [x for x in d["threads"] if x["id"] != event[1]]
            return None
        if k == "receive_share":
            ids = event[1] or []
            pd = st.data.get("photos", {}).get("photos", [])
            first = next((p for p in pd if p["id"] in ids), None)
            d["_sharing"] = {
                "photos": ids, "to": None,
                "caption": first["name"] if first else "Photo",
                "glyph": first.get("glyph", "🖼") if first else "🖼",
                "grad": first.get("color", "#8aa1c1") if first else "#8aa1c1"}
            st.goto_app("messages", "messages.root")
            st.push("messages.share")
            return None
        if k == "share_pick":
            d.setdefault("_sharing", {})["to"] = event[1]
            return None
        if k == "share_send":
            sh = d.get("_sharing", {})
            if not sh.get("to"):
                st.toast = {"text": "Pick a recipient first", "until": st.tick + 3}
                return None
            t = next(x for x in d["threads"] if x["id"] == sh["to"])
            t["msgs"].append({"from": "me", "text": "📷 Photo", "photo": True,
                              "glyph": sh.get("glyph", "🧾"),
                              "grad": sh.get("grad", "linear-gradient(160deg,#f5f2e8,#dcd6c3)")})
            d["_shared"] = True
            st.toast = {"text": "Photo sent", "until": st.tick + 3}
            st.replace(f"messages.thread_{sh['to']}")
            return None
        return None


register(MessagesApp())
