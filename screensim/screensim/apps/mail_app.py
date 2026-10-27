"""Mail — compose with attachment (the omit-attachment error), inbox, filters."""
from __future__ import annotations

from ..core.app import App, register
from ..core.model import (Screen, Widget, w_button, w_field, w_row, w_section,
                          w_text)
from ..core.state import DeviceState


class MailApp(App):
    id = "mail"
    name = "Mail"
    icon = "mail"
    color = "#c1432f"
    root_key = "mail.root"

    def initial_data(self, rng) -> dict:
        return {
            "inbox": [
                {"id": "e1", "from": "Dana (Manager)", "subj": "Q3 report — need today",
                 "body": "Can you send the finished report before 5pm?",
                 "unread": True, "time": "4:12 PM"},
                {"id": "e2", "from": "Payroll", "subj": "Your payslip", "body": "Attached.",
                 "unread": False, "time": "Mon"},
            ],
            "compose": {"to": "", "cc": "", "subj": "", "body": "", "attachment": None},
            "attachments_available": ["Q3_Report.pdf", "budget.xlsx", "photo.jpg"],
            "sent": [],
        }

    def view(self, st) -> Screen:
        key = st.cur_screen_key().split(".", 1)[1]
        d = st.data["mail"]
        if key == "compose":
            return self._compose(st, d)
        if key == "attach":
            return self._attach(st, d)
        if key.startswith("read_"):
            return self._read(st, d, key[len("read_"):])
        return self._root(st, d)

    def _root(self, st, d):
        ws = [w_section("Inbox")]
        for e in d["inbox"]:
            ws.append(Widget("listitem", f"e_{e['id']}", label=e["from"], sub=e["subj"],
                             h=76, on_tap=("open_email", e['id']),
                             on_swipe={"left": ("archive", e["id"])},
                             meta={"unread": e["unread"], "avatar": True,
                                   "time": e.get("time", "")}))
        scr = Screen("mail.root", "Mail", ws)
        scr.bottom_nav = [
            Widget("toolbtn", "tb_filter", label="", icon="flag", on_tap=("noop",)),
            Widget("text", "tb_count", label=f"{len(d['inbox'])} Unread", h=20,
                   meta={"toolstatus": True}),
            Widget("toolbtn", "compose_btn", label="", icon="sendarrow",
                   on_tap=("nav", "mail.compose")),
        ]
        return scr

    def _read(self, st, d, eid):
        # pure: no writes here — marking read / setting the reply target is an
        # EVENT (open_email), so re-laying-out the screen never mutates state
        e = next((x for x in d["inbox"] if x["id"] == eid), None)
        if not e:
            return self._root(st, d)
        ws = [w_row("r_hdr", e["from"], sub="To: me", avatar=True, on_tap=("noop",)),
              w_text("r_subj", e["subj"], h=36),
              w_text("r_body", e["body"], h=120)]
        scr = Screen(f"mail.read_{eid}", "", ws)
        scr.bottom_nav = [
            Widget("toolbtn", "reply", label="Reply", icon="replyarrow",
                   on_tap=("reply_to", eid)),
            Widget("toolbtn", "flag", label="Flag", icon="flag", on_tap=("noop",)),
            Widget("toolbtn", "archive2", label="Archive", icon="trash",
                   on_tap=("noop",)),
        ]
        return scr

    def _compose(self, st, d):
        c = d["compose"]
        def mrow(fid, label, val, hint, h=44):
            w = w_field(fid, label, value=val, hint=hint)
            w.h = h
            w.meta["mailrow"] = True
            if label == "Body":
                w.meta["hidelabel"] = True
            return w
        ws = [
            mrow("to_field", "To:", c["to"], ""),
            mrow("cc_field", "Cc/Bcc:", c.get("cc", ""), ""),
            mrow("subj_field", "Subject:", c["subj"], ""),
            mrow("body_field", "Body", c["body"], "", h=150),
        ]
        if c["attachment"]:
            ws.append(w_row("att_chip", c["attachment"], sub="PDF Document · 1.2 MB",
                            icon="file", on_tap=("detach",)))
        ws.append(w_text("sig", "Sent from my iPhone", h=40))
        scr = Screen("mail.compose", "New Message", ws)
        scr.back_label = "Cancel"
        scr.appbar_actions = [Widget("icon_button", "send_mail", label="Send",
                                     icon="sendarrow", on_tap=("send_mail",),
                                     w=44, enabled=bool(c["to"]))]
        # iOS attaches via a formatting/attachment bar above the keyboard
        scr.bottom_nav = [
            Widget("toolbtn", "attach_cam", label="", icon="camera", on_tap=("noop",)),
            Widget("toolbtn", "attach_photo", label="", icon="photo", on_tap=("noop",)),
            Widget("toolbtn", "attach_btn", label="Attach file", icon="file",
                   on_tap=("nav", "mail.attach")),
        ]
        return scr

    def _attach(self, st, d):
        sizes = {"Q3_Report.pdf": "PDF Document · 1.2 MB",
                 "budget.xlsx": "Spreadsheet · 348 KB",
                 "photo.jpg": "Image · 2.4 MB"}
        ws = [w_section("iCloud Drive")]
        for f in d["attachments_available"]:
            ws.append(w_row(f"att_{f}", f, sub=sizes.get(f, ""), icon="file",
                            on_tap=("attach_file", f)))
        ws.append(w_section("Recents"))
        ws.append(w_row("att_recent", "Scanned Document.pdf",
                        sub="PDF Document · 3.1 MB", icon="file", on_tap=("noop",)))
        scr = Screen("mail.attach", "Choose a File", ws)
        scr.back_label = "Cancel"
        return scr

    def handle(self, st, event, gesture):
        d = st.data["mail"]
        c = d["compose"]
        k = event[0]
        if k == "open_email":
            eid = event[1]
            e = next((x for x in d["inbox"] if x["id"] == eid), None)
            if e:
                e["unread"] = False
                d["_reply_to"] = eid
            st.push(f"mail.read_{eid}")
            return None
        if k == "reply_to":
            e = next((x for x in d["inbox"] if x["id"] == event[1]), None)
            if e:
                sender = e["from"].split(" (")[0]
                d["compose"] = {"to": sender, "cc": "", "subj": "Re: " + e["subj"],
                                "body": "", "attachment": None}
            st.push("mail.compose")
            return None
        if k == "set_field":
            fid, text = event[1], event[2]
            if fid == "to_field":
                c["to"] = text
            elif fid == "cc_field":
                c["cc"] = text
            elif fid == "subj_field":
                c["subj"] = text
            elif fid == "body_field":
                c["body"] = text
            return None
        if k == "attach_file":
            c["attachment"] = event[1]
            st.pop()
            return None
        if k == "detach":
            c["attachment"] = None
            return None
        if k == "send_mail":
            if not c["to"]:
                st.toast = {"text": "Add a recipient first", "until": st.tick + 3}
                return None
            d["sent"].append(dict(c))
            d["compose"] = {"to": "", "cc": "", "subj": "", "body": "", "attachment": None}
            st.toast = {"text": "Message sent", "until": st.tick + 3}
            st.goto_app("mail", "mail.root")
            return None
        if k == "archive":
            d["inbox"] = [x for x in d["inbox"] if x["id"] != event[1]]
            return None
        return None


register(MailApp())
