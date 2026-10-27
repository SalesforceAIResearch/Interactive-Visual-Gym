"""Shop — a browser-style store with a checkout that has a hold countdown.

Flow: browse -> product -> add to cart -> cart -> checkout (address, shipping,
payment) -> place order. A limited-stock item holds the cart for N ticks (the
hazard); if it expires the item is released and checkout resets.
"""
from __future__ import annotations

from ..core import pacing
from ..core.app import App, register
from ..core.model import (Screen, Widget, w_banner, w_button, w_field, w_row,
                          w_section, w_text)
from ..core.state import DeviceState, Timer


class ShopApp(App):
    id = "shop"
    name = "Marketplace"
    icon = "cart"
    color = "#d47b1f"
    root_key = "shop.home"

    def initial_data(self, rng) -> dict:
        return {
            "products": [
                {"id": "tent", "name": "Trailhead 2P Tent", "price": 129.00,
                 "stock": "Only 2 left", "limited": True},
                {"id": "lamp", "name": "Camp Lantern", "price": 24.50, "stock": "In stock"},
                {"id": "bag", "name": "Down Sleeping Bag", "price": 89.00, "stock": "In stock"},
                {"id": "stove", "name": "Pocket Stove", "price": 39.99, "stock": "In stock"},
            ],
            "cart": [],
            "checkout": {"address": "", "shipping": None, "payment": None,
                         "promo": "", "promo_applied": False, "gift": False,
                         "placed": False},
            "order": None,
        }

    def view(self, st) -> Screen:
        key = st.cur_screen_key().split(".", 1)[1]
        d = st.data["shop"]
        if key.startswith("product_"):
            return self._product(st, d, key[len("product_"):])
        return getattr(self, "_scr_" + key, self._scr_home)(st, d)

    def _scr_home(self, st, d):
        ws = [w_field("shop_search", "Search", hint="Search products"),
              w_section("Camping")]
        for p in d["products"]:
            ws.append(w_row(f"p_{p['id']}", p["name"],
                            sub=f"${p['price']:.2f} · {p['stock']}", icon="box",
                            on_tap=("nav", f"shop.product_{p['id']}")))
        cart_n = len(d["cart"])
        ws.append(w_button("go_cart", f"View cart ({cart_n})", ("nav", "shop.cart"),
                           style="tonal" if cart_n else "text"))
        return Screen("shop.home", "Marketplace", ws)

    def _product(self, st, d, pid):
        p = next(x for x in d["products"] if x["id"] == pid)
        ws = [
            Widget("image", f"img_{pid}", label=p["name"], h=200,
                   meta={"color": "#cbb79a"}),
            w_text("p_name", p["name"], sub=f"${p['price']:.2f}", h=56),
            w_banner("p_stock", p["stock"],
                     tone="warn" if p.get("limited") else "info"),
            w_button("add_cart", "Add to cart", ("add_cart", pid), style="filled"),
        ]
        return Screen(f"shop.product_{pid}", p["name"], ws)

    def _scr_cart(self, st, d):
        ws = [w_section("Your cart")]
        if not d["cart"]:
            ws.append(w_text("empty", "Your cart is empty.", h=44))
        total = 0.0
        for it in d["cart"]:
            total += it["price"]
            hold = self._hold_left(st)
            sub = f"${it['price']:.2f}"
            if it.get("limited") and hold is not None:
                sub += f" · held {hold}s"
            ws.append(Widget("listitem", f"c_{it['id']}", label=it["name"], sub=sub,
                             h=64, on_swipe={"left": ("remove_cart", it["id"])},
                             icon="box"))
        if d["cart"]:
            hold = self._hold_left(st)
            if hold is not None:
                ws.append(w_banner("hold", f"Limited item held for {hold}s",
                                   tone="alert" if hold <= 8 else "warn"))
            ws.append(w_text("total", f"Total  ${total:.2f}", h=44))
            ws.append(w_button("checkout", "Checkout", ("nav", "shop.checkout"),
                               style="filled"))
        return Screen("shop.cart", "Cart", ws)

    def _scr_checkout(self, st, d):
        c = d["checkout"]
        ws = [
            w_section("Shipping address"),
            w_field("addr_field", "Address", value=c["address"], hint="Street, city, ZIP"),
            w_section("Delivery"),
            Widget("radio", "ship_std", label="Standard (5–7 days) · Free",
                   value=c["shipping"] == "std", h=56, on_tap=("ship", "std")),
            Widget("radio", "ship_exp", label="Express (2 days) · $12",
                   value=c["shipping"] == "exp", h=56, on_tap=("ship", "exp")),
            w_section("Payment"),
            Widget("radio", "pay_card", label="Visa ···· 4821",
                   value=c["payment"] == "card", h=56, on_tap=("pay", "card")),
            Widget("radio", "pay_gift", label="Gift card balance",
                   value=c["payment"] == "gift", h=56, on_tap=("pay", "gift")),
            w_section("Promo & extras"),
            w_field("promo_field", "Promo code",
                    value="CAMP10 ✓ (-10%)" if c["promo_applied"] else c["promo"],
                    hint="Enter promo code"),
            Widget("button", "apply_promo", label="Apply", on_tap=("apply_promo",),
                   h=40, w=120, meta={"style": "tonal"}),
            Widget("toggle", "gift_receipt", label="Gift receipt",
                   value=c["gift"], h=48, on_tap=("toggle_gift",)),
            w_button("place", "Place order", ("place_order",), style="filled",
                     enabled=bool(c["address"] and c["shipping"] and c["payment"])),
        ]
        if not (c["address"] and c["shipping"] and c["payment"]):
            ws[-1].meta["disabled_reason"] = "fill address, delivery and payment first"
        return Screen("shop.checkout", "Checkout", ws)

    def _scr_order(self, st, d):
        o = d["order"] or {}
        ws = [w_banner("ord_ok", "Order placed", tone="info"),
              w_text("ord_l", o.get("summary", ""), h=44),
              w_button("ord_home", "Continue shopping", ("nav", "shop.home"), style="tonal")]
        return Screen("shop.order", "Order confirmed", ws)

    def _hold_left(self, st):
        for t in st.timers:
            if t.id == "cart_hold":
                return None if t.fired else max(0, t.expires_at - st.tick)
        return None

    def handle(self, st, event, gesture):
        d = st.data["shop"]
        k = event[0]
        if k == "set_field":
            if event[1] == "addr_field":
                d["checkout"]["address"] = event[2]
            elif event[1] == "promo_field":
                d["checkout"]["promo"] = event[2]
                d["checkout"]["promo_applied"] = False
            elif event[1] == "shop_search":
                d["_q"] = event[2]
            return None
        if k == "apply_promo":
            c = d["checkout"]
            if c["promo"].strip().upper() == "CAMP10":
                c["promo_applied"] = True
                return None
            return "that promo code isn't valid"
        if k == "toggle_gift":
            d["checkout"]["gift"] = not d["checkout"]["gift"]
            return None
        if k == "add_cart":
            p = next(x for x in d["products"] if x["id"] == event[1])
            if p["id"] not in [c["id"] for c in d["cart"]]:
                d["cart"].append(dict(p))
                if p.get("limited"):
                    st.timers = [t for t in st.timers if t.id != "cart_hold"]
                    st.timers.append(Timer("cart_hold", st.tick + pacing.CART_TICKS,
                                           on_expire=("cart_expire",),
                                           label="the held item was released",
                                           tone_warn_at=st.tick + 28))
            st.toast = {"text": "Added to cart", "until": st.tick + 2}
            st.push("shop.cart")
            return None
        if k == "remove_cart":
            d["cart"] = [c for c in d["cart"] if c["id"] != event[1]]
            return None
        if k == "ship":
            d["checkout"]["shipping"] = event[1]
            return None
        if k == "pay":
            d["checkout"]["payment"] = event[1]
            return None
        if k == "place_order":
            c = d["checkout"]
            if not (c["address"] and c["shipping"] and c["payment"]):
                st.toast = {"text": "Complete all fields first", "until": st.tick + 3}
                return None
            total = sum(x["price"] for x in d["cart"]) + (12 if c["shipping"] == "exp" else 0)
            if c["promo_applied"]:
                total *= 0.9
            d["order"] = {"summary": f"{len(d['cart'])} item(s) · ${total:.2f} · {c['shipping']}",
                          "promo": c["promo_applied"], "gift": c["gift"],
                          "shipping": c["shipping"]}
            c["placed"] = True
            st.timers = [t for t in st.timers if t.id != "cart_hold"]
            st.replace("shop.order")
            return None
        if k == "cart_expire":
            d["cart"] = [c for c in d["cart"] if not c.get("limited")]
            st.toast = {"text": "Held item was released", "until": st.tick + 4}
            return None
        return None


register(ShopApp())
