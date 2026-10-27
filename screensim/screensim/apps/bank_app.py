"""QuickPay — a bank/payment app. The irreversibility flagship.

A transfer walks through: pick payee -> enter amount -> review -> OTP ->
confirm. The confirm is the irreversible commit (analog of serving the dish).
The OTP is delivered as a Messages notification and expires on a visible
countdown (the smoke->fire hazard). Wrong PIN three times locks the account.
"""
from __future__ import annotations

from ..core import pacing
from ..core.app import App, get, register
from ..core.model import (Screen, Widget, w_banner, w_button, w_field, w_row,
                          w_section, w_text)
from ..core.state import DeviceState, Notification, Timer


class BankApp(App):
    id = "bank"
    name = "QuickPay"
    icon = "pay"
    color = "#1f7a5a"
    root_key = "bank.home"

    def initial_data(self, rng) -> dict:
        return {
            "balance": 5284.55,
            "payees": [
                {"name": "Alex Chen", "handle": "@alexc", "last": "$25 · Mar 2"},
                {"name": "Alex Kim", "handle": "@akim", "last": "$60 · Feb 18"},
                {"name": "Jordan Diaz", "handle": "@jdiaz", "last": "$12 · Jan 9"},
                {"name": "Landlord — Sunset Apts", "handle": "@sunsetapts", "last": "$1450 · Mar 1"},
            ],
            "transfer": {"payee": None, "amount": "", "stage": "idle"},
            "otp": {"code": "", "attempts": 0, "verified": False},
            "authed": True,
            "history": [],
        }

    def view(self, st) -> Screen:
        key = st.cur_screen_key().split(".", 1)[1]
        return getattr(self, "_scr_" + key, self._scr_home)(st, st.data["bank"])

    def _scr_home(self, st, d):
        ws = [
            w_banner("bal", f"Balance  ${d['balance']:,.2f}", tone="info"),
            w_button("send", "Send money", ("nav", "bank.pick"), style="filled"),
            w_button("request", "Request", ("noop",), style="tonal"),
            w_section("Recent activity"),
        ]
        if not d["history"]:
            ws.append(w_text("noact", "No recent transfers.", h=44))
        for h in d["history"][-5:]:
            ws.append(w_row(f"h_{h['id']}", f"Sent ${h['amount']:.2f}",
                            sub=f"to {h['payee']}", icon="pay"))
        return Screen("bank.home", "QuickPay", ws)

    def _scr_pick(self, st, d):
        ws = [w_field("payee_search", "To", hint="Search name or @handle"),
              w_section("People")]
        q = st.data["bank"].get("_pick_q", "").lower()
        for p in d["payees"]:
            if q and q not in p["name"].lower() and q not in p["handle"].lower():
                continue
            ws.append(w_row(f"pay_{p['handle']}", p["name"],
                            sub=f"{p['handle']} · {p['last']}",
                            on_tap=("pick_payee", p["handle"]), avatar=True))
        return Screen("bank.pick", "Send money", ws)

    def _scr_amount(self, st, d):
        if not (d.get('transfer') or {}).get('payee'):
            return self._scr_home(st, d)   # no transfer in progress
        p = d["transfer"]["payee"]
        src = d["transfer"].get("source", "balance")
        ws = [
            w_text("to_line", f"To  {p['name']}", sub=p["handle"], h=56),
            w_field("amount_field", "Amount", value=d["transfer"]["amount"],
                    hint="$0.00", kb="number"),
            w_field("note_field", "Note (optional)", value=d["transfer"].get("note", ""),
                    hint="What's it for?"),
            w_section("Pay With"),
            Widget("radio", "src_balance",
                   label=f"QuickPay balance · ${d['balance']:,.2f}",
                   value=src == "balance", h=48, on_tap=("set_source", "balance")),
            Widget("radio", "src_card", label="Visa ···· 4821 · 3% fee",
                   value=src == "card", h=48, on_tap=("set_source", "card")),
            w_section("Speed"),
            Widget("radio", "spd_instant", label="Instant · arrives in seconds",
                   value=d["transfer"].get("speed", "instant") == "instant",
                   h=48, on_tap=("set_speed", "instant")),
            Widget("radio", "spd_standard", label="1-3 business days · no fee",
                   value=d["transfer"].get("speed") == "standard",
                   h=48, on_tap=("set_speed", "standard")),
            w_button("to_review", "Review", ("to_review",), style="filled"),
        ]
        return Screen("bank.amount", "Amount", ws)

    def _scr_review(self, st, d):
        if not (d.get('transfer') or {}).get('payee'):
            return self._scr_home(st, d)   # no transfer in progress (already sent)
        t = d["transfer"]
        p = t["payee"]
        src = ("QuickPay balance" if t.get("source", "balance") == "balance"
               else "Visa ···· 4821 (3% fee)")
        spd = ("Instant" if t.get("speed", "instant") == "instant"
               else "1-3 business days")
        note = t.get("note", "")
        ws = [
            w_section("Review transfer"),
            w_text("rv_to", f"To    {p['name']}", sub=p["handle"], h=52),
            w_text("rv_amt", f"Amount    ${_amt(t['amount']):,.2f}", h=40),
            w_text("rv_from", f"From    {src}", h=40),
            w_text("rv_spd", f"Speed    {spd}", h=40),
        ]
        if note:
            ws.append(w_text("rv_note", f"Note    {note}", h=40))
        ws += [
            w_banner("rv_warn", "This can't be undone once sent.", tone="warn"),
            w_button("confirm_send", "Confirm & send", ("request_otp",), style="filled"),
            w_button("rv_cancel", "Cancel", ("nav_back",), style="text"),
        ]
        return Screen("bank.review", "Review", ws)

    def _scr_otp(self, st, d):
        if not (d.get('transfer') or {}).get('payee'):
            return self._scr_home(st, d)   # no transfer in progress
        o = d["otp"]
        left = self._otp_left(st)
        tone = "alert" if left is not None and left <= 8 else "info"
        ws = [
            w_text("otp_hd", "Enter the 6-digit code",
                   sub="We texted a code to your phone", h=56),
            w_banner("otp_timer",
                     f"Code expires in {left}s" if left is not None else "Code expired",
                     tone=tone if left else "alert"),
            w_field("otp_field", "6-digit code", value=o["code"], hint="——————",
                    kb="number"),
            w_button("otp_submit", "Verify & send", ("verify_otp",), style="filled"),
        ]
        # iOS surfaces the texted code as a one-tap AutoFill chip above the keys
        if o.get("expected") and left:
            ws[2].meta["autofill"] = o["expected"]
        ws += [
            w_text("otp_note", f"Attempts: {o['attempts']}/3" if o["attempts"] else "", h=32),
        ]
        return Screen("bank.otp", "Verify", ws)

    def _scr_done(self, st, d):
        t = d.get("_last", {})
        ws = [
            w_banner("done_ok", "Payment sent", tone="info"),
            w_text("done_l", f"${t.get('amount',0):,.2f} to {t.get('payee','')}", h=44),
            w_button("done_home", "Done", ("nav", "bank.home"), style="filled"),
        ]
        return Screen("bank.done", "Sent", ws)

    def _scr_locked(self, st, d):
        ws = [w_banner("lk", "Account locked", tone="alert",
                       sub="Too many wrong codes. Try again later."),
              w_button("lk_ok", "OK", ("nav", "bank.home"), style="tonal")]
        return Screen("bank.locked", "Locked", ws)

    # --- helpers ---
    def _otp_left(self, st):
        for t in st.timers:
            if t.id == "otp_expiry":
                return None if t.fired else max(0, t.expires_at - st.tick)
        return None

    # --- events ---
    def handle(self, st, event, gesture):
        d = st.data["bank"]
        k = event[0]
        if k == "set_field":
            fid, text = event[1], event[2]
            if fid == "amount_field":
                d["transfer"]["amount"] = text
            elif fid == "payee_search":
                d["_pick_q"] = text
            elif fid == "otp_field":
                d["otp"]["code"] = text
            elif fid == "note_field":
                d["transfer"]["note"] = text
            return None
        if k == "pick_payee":
            p = next(x for x in d["payees"] if x["handle"] == event[1])
            prev = d.get("transfer") or {}
            if prev.get("stage") == "amount":
                # re-picking the payee keeps what was already filled in (as the
                # real app does) — only the payee changes
                d["transfer"] = {**prev, "payee": p}
            else:
                d["transfer"] = {"payee": p, "amount": "", "stage": "amount",
                                 "source": "balance", "speed": "instant", "note": ""}
            st.push("bank.amount")
            return None
        if k == "set_source":
            d["transfer"]["source"] = event[1]
            return None
        if k == "set_speed":
            d["transfer"]["speed"] = event[1]
            return None
        if k == "to_review":
            if _amt(d["transfer"]["amount"]) <= 0:
                st.toast = {"text": "Enter an amount first", "until": st.tick + 3}
                return None
            st.push("bank.review")
            return None
        if k == "nav_back":
            st.pop()
            return None
        if k == "request_otp":
            return self._request_otp(st, d)
        if k == "verify_otp":
            return self._verify_otp(st, d)
        if k == "back":
            return None
        return None

    def _request_otp(self, st, d):
        # generate a stable code from seed+tick-free source: use payee+amount
        import hashlib
        seed_s = f"{d['transfer']['payee']['handle']}|{d['transfer']['amount']}"
        h = int(hashlib.md5(seed_s.encode()).hexdigest(), 16)
        code = f"{h % 900000 + 100000}"
        d["otp"] = {"code": "", "attempts": 0, "verified": False, "expected": code}
        # deliver as a Messages notification
        st.notifications.insert(0, Notification(
            id="sms_otp", app="messages", title="QuickPay",
            text=f"Your QuickPay code is {code}. Expires in {pacing.OTP_TICKS}s.",
            arrived_at=st.tick, on_tap=None))
        # hazard clock: 30 ticks to enter it
        st.timers = [t for t in st.timers if t.id != "otp_expiry"]
        st.timers.append(Timer(id="otp_expiry", expires_at=st.tick + pacing.OTP_TICKS,
                               on_expire=("otp_timeout",), label="the code expired",
                               tone_warn_at=st.tick + pacing.OTP_WARN, visible=True))
        # also make the code visible to Messages app
        st.data["messages"]["threads"].insert(0, {
            "id": "qp", "name": "QuickPay", "handle": "22395",
            "msgs": [{"from": "them", "text": f"Your QuickPay code is {code}. Expires in 30s."}],
            "unread": True})
        st.push("bank.otp")
        return None

    def _verify_otp(self, st, d):
        o = d["otp"]
        left = self._otp_left(st)
        if left == 0 or (left is None and o.get("expected")):
            st.replace("bank.locked") if False else None
            st.toast = {"text": "That code has expired", "until": st.tick + 3}
            return None
        if o["code"] == o.get("expected"):
            o["verified"] = True
            self._commit(st, d)
            return None
        o["attempts"] += 1
        o["code"] = ""
        d["otp"] = o
        if o["attempts"] >= 3:
            st.timers = [t for t in st.timers if t.id != "otp_expiry"]
            st.replace("bank.locked")
            st.outcome = st.outcome or None
            return None
        st.toast = {"text": "Wrong code, try again", "until": st.tick + 3}
        return None

    def _commit(self, st, d):
        t = d["transfer"]
        amt = _amt(t["amount"])
        d["balance"] -= amt
        rec = {"id": len(d["history"]) + 1, "amount": amt,
               "payee": t["payee"]["name"], "handle": t["payee"]["handle"],
               "source": t.get("source", "balance"),
               "speed": t.get("speed", "instant"), "note": t.get("note", "")}
        d["history"].append(rec)
        d["_last"] = rec
        st.timers = [x for x in st.timers if x.id != "otp_expiry"]
        st.notifications = [n for n in st.notifications if n.id != "sms_otp"]
        d["transfer"] = {"payee": None, "amount": "", "stage": "idle"}
        st.replace("bank.done")
        return None

    # timer callbacks route here via ("otp_timeout",)
    def _timer(self, st, name):
        pass


def _amt(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


# route the otp_timeout timer event through handle()
_orig = BankApp.handle
def _handle(self, st, event, gesture):
    if event and event[0] == "otp_timeout":
        st.toast = {"text": "Your code expired", "until": st.tick + 4}
        return None
    return _orig(self, st, event, gesture)
BankApp.handle = _handle

register(BankApp())
