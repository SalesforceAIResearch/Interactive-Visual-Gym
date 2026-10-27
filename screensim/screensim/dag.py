"""Task DAGs v2 — multi-path union graphs with choice difficulty.

A task's DAG is the union of its main route and every alternate route: nodes
are screens (merged across routes, so shared screens become branch/merge
points), each node carries the gestures done there and HOW MANY interactive
options were on screen at that moment — the visual measure of "the right
choice is one of N". Hazards attach as clock nodes; traps as warning nodes.
Emitted as mermaid for the website.
"""
from __future__ import annotations

import re

from .core.engine import Device
from .tasks import resolve_dynamic


NAMES = {
    "home": "Home Screen", "ccenter": "Control Center", "shade": "Notifications",
    "settings.root": "Settings", "settings.wifi_list": "Wi-Fi",
    "settings.wifi_join": "Enter Password", "settings.hotspot": "Personal Hotspot",
    "settings.display": "Display & Brightness", "settings.general": "General",
    "settings.storage": "iPhone Storage", "settings.autolock": "Auto-Lock",
    "settings.bg_refresh": "Background App Refresh",
    "settings.privacy": "Privacy & Security",
    "settings.loc_services": "Location Services",
    "settings.priv_mic": "Microphone", "settings.priv_cam": "Camera",
    "settings.notifications": "Notifications", "settings.apps": "Apps",
    "settings.cellular": "Cellular",
    "settings.cellular_options": "Cellular Data Options",
    "settings.bluetooth": "Bluetooth", "settings.battery": "Battery",
    "bank.home": "QuickPay", "bank.pick": "Choose Payee",
    "bank.amount": "Amount & Options", "bank.review": "Review Transfer",
    "bank.otp": "Verify Code", "bank.done": "Payment Sent",
    "shop.home": "Marketplace", "shop.cart": "Cart",
    "shop.checkout": "Checkout", "shop.order": "Order Placed",
    "mail.root": "Mailbox", "mail.compose": "Compose", "mail.attach": "Attach",
    "photos.root": "Photos", "messages.share": "Share To…",
}


def _pretty(key: str) -> str:
    if key in NAMES:
        return NAMES[key]
    if key.startswith("settings.loc_app_"):
        return key.split("loc_app_")[1] + " Location"
    if key.startswith("settings.notif_app_"):
        return key.split("notif_app_")[1] + " Notifications"
    if key.startswith("settings.app_"):
        return key.split("app_")[1]
    if key.startswith("settings.stub_"):
        return key.split("stub_")[1].replace("_", " ").title()
    if key.startswith("shop.product_"):
        return "Product Page"
    if key.startswith("mail.read_"):
        return "Message"
    if key.startswith("photos.view_"):
        return "Photo"
    if key.startswith("settings.dlg") or ".dlg" in key:
        return "Confirm"
    return key.split(".")[-1].replace("_", " ").title()


def _short(cmd: str) -> str:
    c = cmd.strip()
    m = re.match(r'type "(.+)"', c)
    if m:
        v = m.group(1)
        return f'type “{v[:14]}{"…" if len(v) > 14 else ""}”'
    if c == "wait":
        return ""
    if c == "press back":
        return "back"
    if c.startswith("swipe down from the top right"):
        return "open Control Center"
    if c.startswith("swipe"):
        return c
    c = re.sub(r"^tap (the )?", "", c)
    return c


def _route_nodes(task, plan):
    """Run one plan; return the screen-node sequence with acts + option counts."""
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    seq = []
    cur = None
    for raw in plan:
        cmd = resolve_dynamic(dev, raw)
        pre = dev.laid_out_screen()
        n_opts = sum(1 for w in pre.widgets if w.visible and w.interactive())
        key = dev.st.cur_screen_key()
        if dev.st.cc_open:
            key = "ccenter"
        if cur is None or cur["key"] != key:
            cur = {"key": key, "acts": [], "opts": n_opts}
            seq.append(cur)
        cur["opts"] = max(cur["opts"], n_opts)
        s = _short(cmd)
        if s and not s.startswith("swipe up") and s != "back":
            cur["acts"].append(s)
        dev.act(cmd)
    won, _ = task.goal(dev.st)
    return seq, won


def task_graph(task):
    """Union of the main route + alternates: nodes by screen, edges labeled by
    route. Returns {nodes:{key:{...}}, edges:set, routes:[(label, [keys])],
    won, hazard, trap}."""
    routes = [("main", task.plan)] + [(lbl, p) for lbl, p in task.alt_plans]
    nodes = {}
    edges = []
    route_keys = []
    won_main = True
    for r_i, (label, plan) in enumerate(routes):
        seq, won = _route_nodes(task, plan)
        if r_i == 0:
            won_main = won
        keys = ["__start__"] + [n["key"] for n in seq] + ["__done__"]
        route_keys.append((label, keys))
        for n in seq:
            if n["key"] not in nodes:
                nodes[n["key"]] = {"acts": list(n["acts"]), "opts": n["opts"]}
            else:
                nodes[n["key"]]["opts"] = max(nodes[n["key"]]["opts"], n["opts"])
                for a in n["acts"]:
                    if a not in nodes[n["key"]]["acts"]:
                        nodes[n["key"]]["acts"].append(a)
        for a, b in zip(keys, keys[1:]):
            lbl = label if len(routes) > 1 and r_i > 0 else ""
            e = (a, b, lbl)
            # keep unlabeled edge if same pair exists on main
            if (a, b, "") in [(x, y, l) for x, y, l in edges] and lbl:
                continue
            if e not in edges:
                edges.append(e)
    return {"nodes": nodes, "edges": edges, "routes": route_keys,
            "won": won_main, "hazard": task.hazard, "trap": task.trap,
            "start_label": (task.start_app.title() if task.start_app != "home"
                            else "Home Screen")}


def to_mermaid(task) -> str:
    g = task_graph(task)
    ids = {k: f"n{i}" for i, k in enumerate(g["nodes"])}
    ids["__start__"] = "start"
    ids["__done__"] = "done"
    lines = ["flowchart TD"]
    lines.append(f'  start(["📱 {_esc(g["start_label"])}"]):::term')
    densest = max(g["nodes"].items(), key=lambda kv: kv[1]["opts"])[0] \
        if g["nodes"] else None
    for k, n in g["nodes"].items():
        acts = "<br/>".join(_esc(a) for a in n["acts"][:3])
        opts = f'{n["opts"]} options on screen'
        label = f'{_esc(_pretty(k))}<br/><i>{opts}</i>'
        if acts:
            label += f'<br/>{acts}'
        lines.append(f'  {ids[k]}["{label}"]')
    cls = "good" if g["won"] else "bad"
    lines.append(f'  done(["✓ Goal reached"]):::{cls}')
    seen = set()
    for a, b, lbl in g["edges"]:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        arrow = f'-- {_esc(lbl)} -->' if lbl else "-->"
        lines.append(f'  {ids[a]} {arrow} {ids[b]}')
    # anchor hazard/trap on the DECISION screen: the last real node on the
    # main route (where the commit/choice happens)
    main_keys = [k for k in g["routes"][0][1] if k in ids and not k.startswith("__")]
    anchor = main_keys[-1] if main_keys else densest
    if g["hazard"]:
        lines.append(f'  haz[/"⏱ {_esc(g["hazard"])}"/]:::haz')
        lines.append(f'  {ids[anchor]} -.-> haz')
    if g["trap"]:
        lines.append(f'  trap[/"⚠ trap: {_esc(g["trap"][:70])}"/]:::trapc')
        if anchor:
            lines.append(f'  {ids[anchor]} -.-> trap')
    lines.append("  classDef term fill:#e8eefb,stroke:#2f63e0,color:#123;")
    lines.append("  classDef good fill:#e3f3ea,stroke:#1d7a4f,color:#0a3;")
    lines.append("  classDef bad fill:#f9e7e7,stroke:#b03535,color:#900;")
    lines.append("  classDef haz fill:#f8efdb,stroke:#a26a00,color:#7a5000;")
    lines.append("  classDef trapc fill:#fbe3e1,stroke:#a3241a,color:#7a1a12;")
    return "\n".join(lines)


def _esc(s: str) -> str:
    return str(s).replace('"', "'").replace("[", "(").replace("]", ")")
