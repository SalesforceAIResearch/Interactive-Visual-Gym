"""iOS renderer: a laid-out Screen + DeviceState -> a self-contained HTML phone
screen styled to look like a current iPhone (iOS 18-era).

The visual language is iOS: SF-like type (Inter), inset grouped lists with
hairline separators, a Dynamic Island, green switches, the iOS keyboard, iOS
alerts, and a home-indicator pill. Widgets are absolutely positioned at the
EXACT bounds the engine laid out, so what the model sees is what grounding used.
"""
from __future__ import annotations

import html as _html

from ..core.layout import content_top, layout, MARGIN
from ..core.model import (APPBAR_H, KEYBOARD_H, NAVBAR_H, SCREEN_H, SCREEN_W,
                          STATUS_H)
from . import icons

# ---- palette (iOS) --------------------------------------------------------
LIGHT = dict(
    bg="#f2f2f7", group="#ffffff", ink="#000000", sub="#8a8a8e",
    blue="#007aff", green="#34c759", red="#ff3b30", orange="#ff9500",
    sep="#c6c6c8", field="#e4e4ea", fieldink="#000", statusink="#000",
    nav="#f2f2f7", chip="#e9e9ef", secink="#6d6d72", islandbg="#000",
    kb_bg="#d1d5db", key="#ffffff", keyfn="#adb2bd", keyink="#000",
    warn_bg="#fff4e0", warn_ink="#b25f00", info_bg="#e8f0fe", info_ink="#0a52c4",
    alert_bg="#f7f7f7cc",
)
DARK = dict(
    bg="#000000", group="#1c1c1e", ink="#ffffff", sub="#8d8d93",
    blue="#0a84ff", green="#30d158", red="#ff453a", orange="#ff9f0a",
    sep="#38383a", field="#1c1c1e", fieldink="#fff", statusink="#fff",
    nav="#000000", chip="#2c2c2e", secink="#8d8d93", islandbg="#000",
    kb_bg="#2c2c2e", key="#6b6b6e", keyfn="#48484a", keyink="#fff",
    warn_bg="#3a2f12", warn_ink="#ffd479", info_bg="#0a2647", info_ink="#7fb0ff",
    alert_bg="#2c2c2ecc",
)

ICON = {  # plain emoji fallbacks for anything not in the SVG icon set
    "box": "📦", "file": "📄", "flash": "🔦", "dnd": "🌙", "trash": "🗑",
    "share": "􀈂",
}
APP_TINT = {"settings": "#8e8e93", "bank": "#34c759", "messages": "#34c759",
            "shop": "#ff9500", "mail": "#007aff", "phone": "#34c759",
            "photos": "#ff2d55"}

# Inter (installed system-wide) stands in for SF Pro.
FONT_STACK = "'Inter Variable','Inter',-apple-system,'SF Pro Text',system-ui,sans-serif"


def esc(s):
    return _html.escape(str(s))


_AV_COLORS = ["#5b8def", "#ff9500", "#34c759", "#af52de", "#ffcc00",
              "#5ac8fa", "#ff2d55", "#00c7be"]


def _avatar(name):
    initials = "".join(w[0] for w in name.split()[:2] if w and w[0].isalnum()).upper() or "?"
    color = _AV_COLORS[sum(ord(c) for c in name) % len(_AV_COLORS)]
    return f'<span class="avatar" style="background:{color}">{esc(initials)}</span>'


def _leading(w):
    if w.meta.get("avatar"):
        return _avatar(w.label)
    if not w.icon:
        return ""
    if icons.has(w.icon):
        tint = icons.TINT.get(w.icon, "#8e8e93")
        return (f'<span class="ic sq" style="background:{tint}">'
                f'{icons.svg_icon(w.icon)}</span>')
    return f'<span class="ic">{ICON.get(w.icon, "")}</span>'   # never print an icon NAME as text


def render(dev, fx_override=None) -> str:
    st = dev.st
    dark = _is_dark(st)
    C = DARK if dark else LIGHT
    fx = fx_override if fx_override is not None else st.fx

    dialog = st.dialog
    if dialog:
        st.dialog = None
        base = dev.laid_out_screen()
        st.dialog = dialog
    else:
        base = dev.laid_out_screen()

    parts = [_css(C, dark)]
    parts.append(f'<div class="phone {"dark" if dark else ""}">')
    if getattr(st, "cc_open", False):
        parts.append('<div class="ccback"></div>')
    elif st.app == "home" and base.key == "home":
        parts.append('<div class="wallpaper"></div>')
    parts.append(_status_bar(st, C, base))
    if base.title is not None:
        parts.append(_nav_bar(base, C, st, fx))
    parts.append(_content(base, st, C))
    if base.fab and not dialog:
        parts.append(_fab(base.fab, C))
    if st.keyboard and not dialog:
        parts.append(_keyboard(st, C, base, fx))
    else:
        parts.append(_home_indicator(C))
    if dialog:
        from ..core.shell import current_screen as _cur
        from ..core.layout import layout_dialog
        dscr = _cur(st)
        card = layout_dialog(dscr)
        parts.append(_alert(dscr, card, C))
    scr = base
    # (iOS has no Android-style toasts; state changes are their own feedback)
    heads = [n for n in st.notifications
             if getattr(n, "arrived_at", -99) >= st.tick - 3 and not scr.is_dialog]
    if heads and not st.shade_open:
        parts.append(_heads_up(heads[0], C))
    parts.append(_fx_layer(fx, C))
    parts.append("</div>")
    return "".join(parts)


def _is_dark(st):
    return bool(st.data.get("settings", {}).get("dark"))


# ---- chrome ---------------------------------------------------------------
def _status_bar(st, C, scr):
    sig = icons.status_signal(C["statusink"])
    wifi = icons.status_wifi(C["statusink"])
    return (f'<div class="statusbar">'
            f'<span class="sb-time">{esc(st.clock)}</span>'
            f'<div class="island"></div>'
            f'<span class="sb-right">{sig}{wifi}'
            f'<span class="bat-shell"><span class="bat-fill" '
            f'style="width:{max(8,min(100,st.battery))}%"></span></span>'
            f'</span></div>')



def _prev_screen_title(st):
    """The real title of the screen below the top of the navigation stack (what
    iOS prints after the chevron), built by its owning app; never a key."""
    try:
        from ..core import app as app_mod
        key = st.stack[-2]
        owner = app_mod.owner_of(key)
        if owner is None:
            return ""
        saved = st.stack
        st.stack = list(saved[:-1])
        try:
            return (owner.view(st).title or "")
        finally:
            st.stack = saved
    except Exception:
        return ""


def _nav_bar(scr, C, st, fx=None):
    # small top bar (back button) + a large title, the modern iOS look.
    # iOS labels Back with the PREVIOUS screen's title, and the root of a
    # navigation stack has no back button at all.
    press = _navpress(fx)
    backcls = " navpressed" if press == "back" else ""
    show = scr.show_back and len(st.stack) > 1
    prev_label = "Back"
    if show:
        if getattr(scr, "back_label", ""):
            prev_label = scr.back_label
        else:
            prev_label = _prev_screen_title(st) or "Back"
            if len(prev_label) > 14:
                prev_label = "Back"
    back = (f'<span class="nav-back{backcls}">‹ {esc(prev_label)}</span>' if show
            else '<span class="nav-back" style="visibility:hidden">‹</span>')
    actions = ""
    for a in scr.appbar_actions:
        if a.icon and icons.has(a.icon):
            gl = icons.svg_icon(a.icon, 22)
        elif a.icon:
            gl = ICON.get(a.icon, a.icon)
        else:
            gl = esc(a.label)               # text action e.g. Done / Join / Send
        dim = " navdim" if not a.enabled else ""
        actions += f'<span class="nav-act{dim}" data-id="{a.id}">{gl}</span>'
    return (f'<div class="navbar"><div class="nav-top">{back}'
            f'<span class="nav-actions">{actions}</span></div>'
            f'<div class="nav-large">{esc(scr.title)}</div></div>')


def _navpress(fx):
    for f in (fx or []):
        if f.get("type") == "navpress":
            return f.get("key", "")
    return ""


def _home_indicator(C):
    return '<div class="homebar"><div class="homepill"></div></div>'


def _fab(w, C):
    # iOS Mail's compose button: a filled rounded square with a pencil glyph,
    # sitting just above the home indicator on the right.
    return (f'<div class="composebtn" style="left:{SCREEN_W-74}px;'
            f'top:{SCREEN_H-92}px">✎</div>')


# ---- content --------------------------------------------------------------
def _content(scr, st, C):
    out = ['<div class="content" style="top:0">']
    # group backgrounds first (behind the rows)
    out.append(_group_cards(scr, C))
    for w in scr.widgets:
        if not w.bounds:
            continue
        out.append(_widget(w, st, C))
    out.append("</div>")
    return "".join(out)


def _group_cards(scr, C):
    """Draw one rounded card behind each run of grouped rows."""
    out = []
    run = []
    for w in scr.widgets:
        if w.bounds and w.meta.get("grp"):
            run.append(w)
            if w.meta["grp"] in ("single", "last"):
                top = run[0].bounds.y
                bot = run[-1].bounds.y2
                x = MARGIN
                ww = SCREEN_W - 2 * MARGIN
                out.append(f'<div class="groupcard" style="left:{x}px;top:{top}px;'
                           f'width:{ww}px;height:{bot-top}px"></div>')
                run = []
        else:
            run = []
    return "".join(out)


def _pos(b):
    return f"left:{b.x}px;top:{b.y}px;width:{b.w}px;height:{b.h}px"


def _sep(w):
    """A hairline separator under a grouped row unless it's the last."""
    if w.meta.get("grp") in ("first", "mid"):
        return '<span class="hair"></span>'
    return ""


def _widget(w, st, C):
    b = w.bounds
    k = w.kind
    grp = w.meta.get("grp", "")
    gcls = f" g-{grp}" if grp else ""
    dis = "" if w.enabled else " dis"
    base = f'<div class="wg {k}{gcls}{dis}" style="{_pos(b)}" data-id="{w.id}">'

    if k == "section":
        return f'<div class="wg section" style="{_pos(b)}">{esc(w.label)}</div>'

    if k == "row":
        lead = _leading(w)
        sub = f'<div class="sub">{esc(w.sub)}</div>' if w.sub else ""
        chev = '<span class="chev">›</span>' if w.on_tap and w.on_tap[0] == "nav" else ""
        ts = f'<span class="ts">{esc(w.meta["time"])}</span>' if w.meta.get("time") else ""
        # iOS Wi-Fi-style furniture: leading checkmark for the selected network,
        # trailing lock + info glyphs
        lcheck = '<span class="lcheck">✓</span>' if w.meta.get("selected") else ""
        lock = '<span class="rlock">🔒</span>' if w.meta.get("lock") else ""
        info = '<span class="rinfo">ⓘ</span>' if w.meta.get("info") else ""
        return (f'{base}{lead}{lcheck}'
                f'<div class="rl"><div class="lb">{esc(w.label)}</div>{sub}</div>'
                f'{ts}{lock}{info}{chev}{_sep(w)}</div>')

    if k == "toggle":
        lead = _leading(w)
        sub = f'<div class="sub">{esc(w.sub)}</div>' if w.sub else ""
        on = "on" if w.value else "off"
        sw = f'<div class="switch {on}"><div class="knob"></div></div>'
        return (f'{base}{lead}<div class="rl"><div class="lb">{esc(w.label)}</div>{sub}</div>'
                f'{sw}{_sep(w)}</div>')

    if k in ("radio", "checkbox"):
        lead = _leading(w)
        mark = ('<span class="iocheck">✓</span>' if w.value else '')
        sel = " rowsel" if w.value else ""
        return (f'{base.replace("wg "+k,"wg "+k+sel)}{lead}'
                f'<div class="rl"><div class="lb">{esc(w.label)}</div></div>'
                f'{mark}{_sep(w)}</div>')

    if k == "listitem":
        lead = _leading(w)
        unread = ' unread' if w.meta.get("unread") else ""
        sub = f'<div class="sub">{esc(w.sub)}</div>' if w.sub else ""
        ts = f'<span class="ts">{esc(w.meta["time"])}</span>' if w.meta.get("time") else ""
        dot = '<span class="unread-dot"></span>' if w.meta.get("unread") else ""
        chev = '<span class="chev">›</span>'
        return (f'{base.replace("wg listitem","wg listitem"+unread)}{lead}'
                f'<div class="rl"><div class="lb">{esc(w.label)}</div>{sub}</div>'
                f'<div class="ritem">{ts}{dot}</div>{chev}{_sep(w)}</div>')

    if k == "button":
        style = w.meta.get("style", "filled")
        return f'{base.replace("wg button","wg button "+style)}<span>{esc(w.label)}</span></div>'

    if k == "toolbtn":
        gl = icons.svg_icon(w.icon, 24) if icons.has(w.icon) else ICON.get(w.icon, "")
        return (f'{base}<div class="tb-in"><span class="tb-ic">{gl}</span>'
                f'<span class="tb-lb">{esc(w.label)}</span></div></div>')

    if k == "icon_button":
        gl = icons.svg_icon(w.icon, 22) if icons.has(w.icon) else ICON.get(w.icon, "")
        return f'{base}<span>{gl}</span></div>'

    if k == "field":
        if w.meta.get("mailrow"):
            focused = " focus" if st.focus == w.id else ""
            val = w.value or ""
            cursor = '<span class="caret"></span>' if st.focus == w.id else ""
            hide = w.meta.get("hidelabel")
            lab = f'<span class="mr-label">{esc(w.label)}</span>' if (w.label and not hide) else ""
            sep = '<span class="hair" style="left:16px"></span>' if w.meta.get("grp") in ("first", "mid") else ""
            return (f'{base.replace("wg field","wg field mailrow"+focused)}'
                    f'{lab}<span class="mr-val">{esc(val)}{cursor}</span>{sep}</div>')
        focused = " focus" if st.focus == w.id else ""
        val = w.value or ""
        if w.meta.get("mask") and val:
            val = "•" * len(str(val))
        is_search = str(w.meta.get("hint", "")).lower().startswith("search")
        show = esc(val) if val else f'<span class="hint">{esc(w.meta.get("hint",""))}</span>'
        cursor = '<span class="caret"></span>' if st.focus == w.id else ""
        mag = f'<span class="mag">{icons.search_glyph(C["sub"])}</span>' if is_search else ""
        sb = " searchbox" if is_search else ""
        lbl = ""
        if w.label and not (w.label == w.meta.get("hint")) and not is_search:
            lbl = f'<div class="flabel">{esc(w.label)}</div>'
        return (f'{base.replace("wg field","wg field"+focused+sb)}{lbl}'
                f'<div class="fbox{sb}">{mag}{show}{cursor}</div></div>')

    if k == "slider":
        pct = int(w.value or 0)
        return (f'{base}<div class="slabel">{esc(w.label)}</div>'
                f'<div class="strack"><div class="sfill" style="width:{pct}%"></div>'
                f'<div class="sthumb" style="left:{pct}%"></div></div></div>')

    if k == "banner":
        tone = w.meta.get("tone", "info")
        sub = f'<div class="bsub">{esc(w.sub)}</div>' if w.sub else ""
        return f'{base.replace("wg banner","wg banner "+tone)}<div class="btext">{esc(w.label)}</div>{sub}</div>'

    if k == "account":
        av = _avatar(w.label)
        return (f'{base}<span class="acct-av">{av}</span>'
                f'<div class="rl"><div class="acct-name">{esc(w.label)}</div>'
                f'<div class="acct-sub">{esc(w.sub)}</div></div>'
                f'<span class="chev">›</span></div>')

    if k == "apphdr":
        color = w.meta.get("color", "#8e8e93")
        ic = icons.svg_icon(w.icon, 32) if icons.has(w.icon) else ICON.get(w.icon, w.icon)
        return (f'{base}<div class="apphdr-in">'
                f'<div class="apphdr-ic" style="background:{color}">{ic}</div>'
                f'<div class="apphdr-name">{esc(w.label)}</div>'
                f'<div class="apphdr-sub">{esc(w.sub)}</div></div></div>')

    if k == "usagebar":
        segs = w.meta.get("segments", [])
        total = max(1, w.meta.get("total", 1))
        bar = '<div class="usebar">'
        for name, val, color in segs:
            bar += f'<div class="useseg" style="width:{100*val/total:.1f}%;background:{color}"></div>'
        bar += '</div>'
        legend = '<div class="uselegend">'
        for name, val, color in segs:
            legend += (f'<span class="uleg"><span class="udot" style="background:{color}">'
                       f'</span>{esc(name)} {val} GB</span>')
        legend += '</div>'
        return (f'{base}<div class="usewrap"><div class="usetop">'
                f'<span class="usebig">{esc(w.label)}</span>'
                f'<span class="usesub">{esc(w.sub)}</span></div>{bar}{legend}</div></div>')

    if k == "text":
        sub = f'<div class="tsub">{esc(w.sub)}</div>' if w.sub else ""
        bubble = w.meta.get("bubble")
        cls = f" bubble {bubble}" if bubble else ""
        return f'{base.replace("wg text","wg text"+cls)}<div class="tmain">{esc(w.label)}</div>{sub}</div>'

    if k == "tile":
        color = w.meta.get("color", "#889")
        badge = w.meta.get("badge", 0)
        sel = " selected" if w.meta.get("selected") else ""
        bdg = f'<div class="badge">{badge}</div>' if badge else ""
        fav = '<div class="fav">♥</div>' if w.meta.get("fav") else ""
        if w.meta.get("photo"):
            glyph = w.meta.get("glyph", "")
            inner = f'<div class="tile-img" style="background:{color}"><span class="tile-glyph">{glyph}</span></div>'
            return f'{base.replace("wg tile","wg tile photo"+sel)}{inner}{fav}{bdg}</div>'
        if w.icon and icons.has(w.icon):
            inner = (f'<div class="tile-ic" style="background:{color}">'
                     f'{icons.svg_icon(w.icon, 30)}</div>')
        elif w.icon:
            inner = f'<div class="tile-ic" style="background:{color}">{ICON.get(w.icon, w.icon)}</div>'
        else:
            inner = f'<div class="tile-img" style="background:{color}"></div>'
        lbl = "" if w.meta.get("nolabel") else f'<div class="tlabel">{esc(w.label)}</div>'
        return (f'{base.replace("wg tile","wg tile"+sel)}{inner}{fav}{bdg}{lbl}</div>')

    if k == "cc_tile":
        on = "on" if w.value else "off"
        gl = icons.svg_icon(w.icon, 20) if icons.has(w.icon) else ICON.get(w.icon, w.icon)
        return (f'{base.replace("wg cc_tile","wg cc_tile "+on)}'
                f'<span class="cc-ic">{gl}</span>'
                f'<div class="cc-tx"><div class="cc-lb">{esc(w.label)}</div>'
                f'<div class="cc-st">{"On" if w.value else "Off"}</div></div></div>')

    if k == "cc_module":
        return f'{base}</div>'

    if k == "alerttiles":
        tiles = [("Lock Screen", "lock"), ("Notification Center", "bell"),
                 ("Banners", "bell")]
        cells = ""
        for lbl, ic in tiles:
            gl = icons.svg_icon(ic, 26) if icons.has(ic) else "▭"
            cells += (f'<div class="at-cell"><span class="at-check">✓</span>'
                      f'<div class="at-icon">{gl}</div>'
                      f'<div class="at-lbl">{esc(lbl)}</div></div>')
        return f'{base}{cells}</div>'

    if k == "cc_round":
        on = "on" if w.value else "off"
        gl = icons.svg_icon(w.icon, 26) if icons.has(w.icon) else ICON.get(w.icon, w.icon)
        return (f'{base.replace("wg cc_round","wg cc_round "+on)}'
                f'<span class="ccr-ic">{gl}</span></div>')

    if k == "cc_media":
        return (f'{base}<div class="ccm"><span class="ccm-note">♫</span>'
                f'<div class="ccm-title">{esc(w.label)}</div>'
                f'<div class="ccm-sub">{esc(w.sub)}</div></div></div>')

    if k == "cc_vslider":
        pct = int(w.value or 0)
        gl = icons.svg_icon(w.icon, 20) if icons.has(w.icon) else ICON.get(w.icon, w.icon)
        return (f'{base}<div class="ccv-track">'
                f'<div class="ccv-fill" style="height:{pct}%"></div>'
                f'<span class="ccv-ic">{gl}</span></div></div>')

    if k == "image":
        glyph = w.meta.get("glyph", "")
        gl = f'<span class="img-glyph">{glyph}</span>' if glyph else esc(w.label)
        bub = w.meta.get("bubble")
        cls = f" imgbubble {bub}" if bub else ""
        return (f'{base.replace("wg image","wg image"+cls)}<div class="imgbox" '
                f'style="background:{w.meta.get("color","#889")}">{gl}</div></div>')

    return f'{base}{esc(w.label)}</div>'


# ---- overlays -------------------------------------------------------------
def _action_sheet(dscr, C):
    """An iOS action sheet: bottom-anchored blurred cards with stacked buttons,
    Cancel separated below. Buttons render at their engine-laid-out bounds."""
    out = ['<div class="scrim"></div>']
    buttons = [w for w in dscr.widgets if w.kind == "button"]
    titles = [w for w in dscr.widgets if w.kind == "text" and w.id.endswith("title")]
    mains = [w for w in buttons if w.meta.get("style") != "cancel"]
    cancels = [w for w in buttons if w.meta.get("style") == "cancel"]
    # main card wraps title + main buttons
    if mains or titles:
        top = (titles[0].bounds.y if titles else mains[0].bounds.y)
        bot = mains[-1].bounds.y2 if mains else titles[0].bounds.y2
        from ..core.layout import MARGIN as _m
        out.append(f'<div class="sheet-card" style="left:10px;top:{top}px;'
                   f'width:{SCREEN_W-20}px;height:{bot-top}px"></div>')
    for t in titles:
        out.append(f'<div class="sheet-title" style="{_pos(t.bounds)}">{esc(t.label)}</div>')
    for i, w in enumerate(mains):
        hair = '<span class="sheethair"></span>' if i < len(mains) - 1 else ""
        out.append(f'<div class="sheet-btn" style="{_pos(w.bounds)}" data-id="{w.id}">'
                   f'{esc(w.label)}{hair}</div>')
    for w in cancels:
        out.append(f'<div class="sheet-btn cancel" style="{_pos(w.bounds)}" '
                   f'data-id="{w.id}">{esc(w.label)}</div>')
    return "".join(out)


def _alert(dscr, card, C):
    if getattr(dscr, "sheet", False):
        return _action_sheet(dscr, C)
    # geometry comes from layout_dialog (the card rect + button bounds), so
    # rendered pixels and engine grounding always agree
    buttons = [w for w in dscr.widgets if w.kind == "button"]
    titles = [w for w in dscr.widgets if w.kind == "text" and w.id.endswith("title")]
    bodies = [w for w in dscr.widgets if w.kind == "text" and not w.id.endswith("title")]
    two = len(buttons) == 2
    if two and buttons[0].label.lower() != "cancel" \
            and buttons[1].label.lower() == "cancel":
        buttons = [buttons[1], buttons[0]]   # same order as layout_dialog
    out = [f'<div class="scrim blur"></div>',
           f'<div class="alert" style="left:{card.x}px;top:{card.y}px;'
           f'width:{card.w}px;height:{card.h}px">']
    out.append('<div class="alert-head">')
    for t in titles:
        out.append(f'<div class="alert-title">{esc(t.label)}</div>')
    for bd in bodies:
        out.append(f'<div class="alert-body">{esc(bd.label)}</div>')
    out.append('</div>')
    out.append(f'<div class="alert-acts {"two" if two else "stack"}">')
    for idx, w in enumerate(buttons):
        style = w.meta.get("style", "text")
        strong = " strong" if style in ("filled",) else ""
        danger = " danger" if style == "danger" else ""
        out.append(f'<div class="alert-btn{strong}{danger}" data-id="{w.id}">{esc(w.label)}</div>')
    out.append('</div></div>')
    # ground the tap: overlay invisible hit targets at the engine bounds
    for w in buttons:
        if w.bounds:
            out.append(f'<div class="hit" style="{_pos(w.bounds)}" data-id="{w.id}"></div>')
    return "".join(out)


def _toast(st, C):
    return f'<div class="toast">{esc(st.toast["text"])}</div>'


def _heads_up(n, C):
    tint = APP_TINT.get(n.app, "#8e8e93")
    glyph = {"messages": "chat", "bank": "pay"}.get(n.app, "bell")
    return (f'<div class="headsup"><div class="hu-ic" style="background:{tint}">'
            f'{icons.svg_icon(glyph, 22)}</div>'
            f'<div class="hu-body"><div class="hu-title">{esc(n.title)}</div>'
            f'<div class="hu-text">{esc(n.text)}</div></div></div>')


def _keyboard(st, C, scr, fx=None):
    fw = next((w for w in scr.widgets if w.id == st.focus), None)
    numeric = fw and fw.meta.get("kb") == "number"
    pressed = ""
    for f in (fx or []):
        if f.get("type") == "key":
            pressed = str(f.get("char", "")).lower()

    def kc(ch):
        return " pressed" if pressed and ch.lower() == pressed else ""

    autofill = fw.meta.get("autofill") if fw else None
    kbcls = "keyboard afk" if (numeric and autofill) else "keyboard"

    out = [f'<div class="{kbcls}" style="height:{KEYBOARD_H}px">']

    if numeric:
        if autofill:
            out.append('<div class="ksugg autofill">'
                       f'<span class="afchip"><span class="afcode">{esc(str(autofill))}</span>'
                       '<span class="aflabel">From Messages</span></span></div>')
        subs = {"2": "A B C", "3": "D E F", "4": "G H I", "5": "J K L",
                "6": "M N O", "7": "P Q R S", "8": "T U V", "9": "W X Y Z"}
        pad = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["", "0", "⌫"]]
        for r in pad:
            out.append('<div class="krow numrow">')
            for ch in r:
                if ch == "":
                    out.append('<div class="key ghost"></div>')
                elif ch == "⌫":
                    out.append('<div class="key numkey del">⌫</div>')
                else:
                    sub = f'<span class="numsub">{subs.get(ch,"")}</span>' if subs.get(ch) else ""
                    out.append(f'<div class="key numkey{kc(ch)}"><span class="numbig">{ch}</span>{sub}</div>')
            out.append("</div>")
        out.append('<div class="kbhome"><div class="homepill"></div></div>')
        out.append("</div>")
        return "".join(out)

    out.append('<div class="ksugg"><span class="sugg">“I”</span>'
               '<span class="sugg mid">I</span><span class="sugg">I\'m</span></div>')
    for row, cls in (("qwertyuiop", ""), ("asdfghjkl", " indent")):
        out.append(f'<div class="krow{cls}">')
        for ch in row:
            out.append(f'<div class="key{kc(ch)}">{ch}</div>')
        out.append("</div>")
    out.append('<div class="krow"><div class="key fnkey shift">⇧</div>')
    for ch in "zxcvbnm":
        out.append(f'<div class="key{kc(ch)}">{ch}</div>')
    out.append('<div class="key fnkey del">⌫</div></div>')
    out.append('<div class="krow bottomrow"><div class="key fnkey num">123</div>'
               '<div class="key fnkey globe">🌐</div>'
               f'<div class="key spacebar{kc(" ")}">space</div>'
               '<div class="key fnkey mic">🎤</div>'
               '<div class="key returnkey">return</div></div>')
    out.append('<div class="kbhome"><div class="homepill"></div></div>')
    out.append("</div>")
    return "".join(out)


def _fx_layer(fx, C):
    if not fx:
        return ""
    out = ['<div class="fxlayer">']
    for f in fx:
        t = f.get("type")
        pt = f.get("point")
        if t in ("tap", "long") and pt:
            cls = "ripple long" if t == "long" else "ripple"
            dis = " disabled" if f.get("disabled") else ""
            out.append(f'<div class="{cls}{dis}" style="left:{pt[0]}px;top:{pt[1]}px"></div>')
        elif t == "swipe" and pt:
            d = f.get("dir", "up")
            arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}.get(d, "↑")
            out.append(f'<div class="swipefx" style="left:{pt[0]}px;top:{pt[1]}px">{arrow}</div>')
        elif t == "miss":
            out.append('<div class="missfx">✕</div>')
    out.append("</div>")
    return "".join(out)


# ---- CSS ------------------------------------------------------------------
def _css(C, dark):
    return f"""<style>
*{{margin:0;padding:0;box-sizing:border-box;-webkit-font-smoothing:antialiased}}
html,body{{background:#000}}
.phone{{position:relative;width:{SCREEN_W}px;height:{SCREEN_H}px;overflow:hidden;
  background:{C['bg']};color:{C['ink']};border-radius:52px;
  font-family:{FONT_STACK};font-size:17px;letter-spacing:-.01em}}
/* status bar + dynamic island */
.statusbar{{position:absolute;top:0;left:0;width:100%;height:{STATUS_H}px;
  display:flex;align-items:flex-start;justify-content:space-between;
  padding:16px 30px 0 36px;z-index:60;color:{C['statusink']};
  background:{C['bg']}}}
.sb-time{{font-size:16px;font-weight:640;letter-spacing:.01em}}
.sb-right{{display:flex;align-items:center;gap:6px;font-size:15px}}
.bat-shell{{display:inline-block;width:26px;height:13px;border:1.4px solid {C['statusink']};
  border-radius:4px;position:relative;opacity:.9}}
.bat-fill{{position:absolute;top:1.4px;left:1.4px;bottom:1.4px;background:{C['statusink']};border-radius:2px}}
.island{{position:absolute;left:50%;top:8px;transform:translateX(-50%);
  width:126px;height:36px;background:#000;border-radius:20px}}
/* nav bar with large title */
.navbar{{position:absolute;top:{STATUS_H}px;left:0;width:100%;height:{APPBAR_H}px;
  z-index:40;background:{C['nav']}}}
.nav-top{{height:44px;display:flex;align-items:center;justify-content:space-between;padding:0 8px}}
.nav-back{{color:{C['blue']};font-size:17px;font-weight:400;padding:0 6px}}
.nav-back.navpressed{{opacity:.4}}
.nav-actions{{display:flex;gap:12px;padding-right:8px;color:{C['blue']};font-size:17px;font-weight:600}}
.nav-act.navdim{{opacity:.35}}
.nav-large{{font-size:34px;font-weight:700;letter-spacing:-.02em;padding:0 16px 6px;line-height:1.05}}
.content{{position:absolute;left:0;width:100%;bottom:0;overflow:hidden}}
/* grouped list card + separators */
.groupcard{{position:absolute;background:{C['group']};border-radius:11px;z-index:1}}
.wg{{position:absolute;display:flex;align-items:center;z-index:2}}
.hair{{position:absolute;left:56px;right:0;bottom:0;height:1px;background:{C['sep']};opacity:.6}}
.row .hair,.toggle .hair,.listitem .hair{{left:56px}}
.section{{color:{C['secink']};font-size:13px;font-weight:400;letter-spacing:.02em;
  align-items:flex-end;padding:0 20px 6px;text-transform:uppercase}}
.row,.toggle,.listitem,.radio,.checkbox{{padding:0 16px;gap:12px;background:transparent}}
.row .ic,.toggle .ic,.listitem .ic,.radio .ic,.checkbox .ic{{font-size:20px;width:29px;
  height:29px;display:flex;align-items:center;justify-content:center;flex:none}}
.ic.sq{{border-radius:7px}}
.ic.sq svg{{display:block}}
.statusbar svg{{display:block}}
.sb-right svg{{margin-bottom:1px}}
.rl{{flex:1;min-width:0}}
.lb{{font-size:17px;font-weight:400;color:{C['ink']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.sub{{font-size:14px;color:{C['sub']};margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.chev{{color:{C['sub']};font-size:20px;opacity:.5;font-weight:600}}
.rowsel{{}}
.rowcheck,.iocheck{{color:{C['blue']};font-size:19px;font-weight:700;flex:none}}
.lcheck{{color:{C['blue']};font-size:17px;font-weight:700;flex:none;width:18px}}
.rlock{{font-size:13px;flex:none;opacity:.75}}
.rinfo{{color:{C['blue']};font-size:18px;flex:none;font-weight:500}}
.ts{{font-size:14px;color:{C['sub']};flex:none}}
.ritem{{display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex:none}}
.avatar{{width:40px;height:40px;border-radius:50%;flex:none;color:#fff;font-size:15px;
  font-weight:600;display:flex;align-items:center;justify-content:center}}
/* iOS switch */
.switch{{width:51px;height:31px;border-radius:16px;position:relative;flex:none}}
.switch.on{{background:{C['green']}}}
.switch.off{{background:{'#e9e9ea' if not dark else '#39393d'}}}
.knob{{position:absolute;top:2px;width:27px;height:27px;border-radius:50%;background:#fff;
  box-shadow:0 2px 4px rgba(0,0,0,.2)}}
.switch.on .knob{{left:22px}}
.switch.off .knob{{left:2px}}
/* buttons */
.button{{border-radius:13px;justify-content:center;font-size:17px;font-weight:600}}
.button.filled{{background:{C['blue']};color:#fff}}
.button.tonal{{background:{C['chip']};color:{C['blue']}}}
.button.text{{background:transparent;color:{C['blue']}}}
.button.danger{{background:{C['red']};color:#fff}}
.button.dis{{opacity:.4}}
.icon_button{{justify-content:center;font-size:20px;color:{C['blue']}}}
/* fields */
.field{{flex-direction:column;align-items:stretch;justify-content:center;gap:5px;
  padding:0 16px;background:transparent}}
.flabel{{font-size:13px;color:{C['sub']};padding-left:2px}}
.fbox{{background:{C['field']};border-radius:10px;height:38px;display:flex;align-items:center;
  padding:0 12px;font-size:17px;color:{C['fieldink']}}}
.field.focus .fbox{{outline:2px solid {C['blue']}}}
.fbox.searchbox{{border-radius:11px;justify-content:flex-start}}
.hint{{color:{C['sub']}}}
.caret{{width:2px;height:20px;background:{C['blue']};margin-left:1px;display:inline-block}}
.mag{{font-size:15px;margin-right:7px;color:{C['sub']}}}
.eye{{margin-left:auto;font-size:15px;opacity:.5}}
.field.mailrow{{flex-direction:row;align-items:center;gap:8px;padding:0 16px;
  background:{C['group']}}}
.mr-label{{font-size:16px;color:{C['sub']};flex:none}}
.mr-val{{font-size:16px;color:{C['ink']};min-width:2px}}
.field.mailrow .hair{{left:0}}
/* radios / checkboxes as grouped rows (checkmark on right, iOS style) */
.slider{{flex-direction:column;align-items:stretch;justify-content:center;gap:8px;padding:0 16px}}
.slabel{{font-size:15px;color:{C['ink']}}}
.strack{{position:relative;height:4px;background:{C['sep']};border-radius:2px}}
.sfill{{position:absolute;height:4px;background:{C['blue']};border-radius:2px}}
.sthumb{{position:absolute;top:-11px;width:26px;height:26px;border-radius:50%;background:#fff;
  margin-left:-13px;box-shadow:0 1px 4px rgba(0,0,0,.3)}}
/* banners */
.banner{{border-radius:12px;flex-direction:column;align-items:flex-start;justify-content:center;padding:0 16px;gap:2px}}
.banner .btext{{font-size:16px;font-weight:600}}
.banner .bsub{{font-size:13px;opacity:.85}}
.banner.info{{background:{C['info_bg']};color:{C['info_ink']}}}
.banner.warn{{background:{C['warn_bg']};color:{C['warn_ink']}}}
.banner.alert{{background:{C['warn_bg']};color:{C['red']};font-weight:700}}
/* app header */
.account{{background:{C['group']};border-radius:11px;padding:0 14px;gap:14px}}
.acct-av .avatar{{width:58px;height:58px;font-size:22px}}
.acct-name{{font-size:22px;font-weight:600;color:{C['ink']}}}
.acct-sub{{font-size:13px;color:{C['sub']};margin-top:2px}}
.apphdr{{justify-content:center}}
.apphdr-in{{display:flex;flex-direction:column;align-items:center;gap:7px;width:100%}}
.apphdr-ic{{width:62px;height:62px;border-radius:15px;display:flex;align-items:center;
  justify-content:center;font-size:30px;color:#fff}}
.apphdr-name{{font-size:22px;font-weight:600;color:{C['ink']}}}
.apphdr-sub{{font-size:13px;color:{C['sub']}}}
/* usage bar */
.usagebar{{background:{C['group']};border-radius:11px;padding:0 16px}}
.usewrap{{width:100%;display:flex;flex-direction:column;gap:8px}}
.usetop{{display:flex;align-items:baseline;gap:8px}}
.usebig{{font-size:19px;font-weight:700;color:{C['ink']}}}
.usesub{{font-size:13px;color:{C['sub']}}}
.usebar{{display:flex;height:12px;border-radius:6px;overflow:hidden;background:{C['field']};gap:2px}}
.useseg{{height:12px}}
.uselegend{{display:flex;gap:14px;flex-wrap:wrap}}
.uleg{{font-size:12px;color:{C['sub']};display:flex;align-items:center;gap:5px}}
.udot{{width:9px;height:9px;border-radius:3px;display:inline-block}}
/* text + bubbles */
.text{{flex-direction:column;align-items:flex-start;justify-content:center;padding:0 8px}}
.tmain{{font-size:16px;line-height:1.35;color:{C['ink']}}}
.tsub{{font-size:13px;color:{C['sub']};margin-top:3px}}
.bubble{{border-radius:19px;padding:9px 14px;max-width:74%;font-size:16px}}
.bubble.them{{background:{C['chip']};color:{C['ink']}}}
.bubble.me{{background:{C['blue']};color:#fff}}
/* tiles / photos */
.tile{{flex-direction:column;align-items:center;justify-content:flex-start;gap:6px}}
.tile-ic{{width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;font-size:28px;color:#fff}}
.tile-img{{width:100%;height:76px;border-radius:6px;display:flex;align-items:center;justify-content:center}}
.tile.photo .tile-img{{height:100%}}
.tile.photo{{border-radius:2px;overflow:hidden}}
.tile-glyph{{font-size:34px;filter:drop-shadow(0 1px 3px rgba(0,0,0,.35))}}
.img-glyph{{font-size:92px;filter:drop-shadow(0 2px 6px rgba(0,0,0,.35))}}
.tile.selected{{outline:3px solid {C['blue']};border-radius:4px}}
.tlabel{{font-size:12px;text-align:center;color:{C['ink']};line-height:1.15;overflow:hidden;max-height:28px}}
.wallpaper~.content .tile .tlabel{{color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.4)}}
.badge{{position:absolute;top:-2px;right:14px;background:{C['red']};color:#fff;min-width:19px;height:19px;
  border-radius:10px;font-size:12px;display:flex;align-items:center;justify-content:center;font-weight:700}}
.fav{{position:absolute;top:6px;right:8px;color:#fff;font-size:14px;text-shadow:0 1px 2px rgba(0,0,0,.4)}}
.imgbox{{width:100%;height:100%;border-radius:14px;display:flex;align-items:center;justify-content:center;
  color:rgba(255,255,255,.9);font-size:15px;font-weight:600;overflow:hidden}}
.image.imgbubble{{border-radius:18px;overflow:hidden}}
.listitem.unread .lb{{font-weight:700}}
.unread-dot{{width:10px;height:10px;border-radius:50%;background:{C['blue']};flex:none}}
.wallpaper{{position:absolute;inset:0;z-index:0;
  background:{'radial-gradient(120% 90% at 30% 10%,#7fb2f0 0%,#5a8cd8 34%,#8a6fd0 68%,#c76fae 100%)' if not dark else 'radial-gradient(120% 90% at 30% 10%,#263550 0%,#1a2438 40%,#331f42 75%,#40204a 100%)'}}}
.dockbar{{border-radius:30px;background:rgba(255,255,255,.32);backdrop-filter:blur(24px)}}
.homesearch{{border-radius:18px;background:rgba(255,255,255,.3);backdrop-filter:blur(14px);
  justify-content:center;gap:6px}}
.hs-lb{{font-size:14px;color:#fff;font-weight:500}}
.ccback{{position:absolute;inset:0;z-index:0;
  background:{'linear-gradient(160deg,#c9d4e4,#a8b4c8)' if not dark else 'linear-gradient(160deg,#20242e,#101318)'}}}
/* Control Center modules */
.cc_tile{{border-radius:18px;padding:0 12px;gap:10px;
  background:{'rgba(255,255,255,.55)' if not dark else 'rgba(60,60,64,.6)'};
  backdrop-filter:blur(18px)}}
.cc-ic{{width:36px;height:36px;border-radius:50%;flex:none;display:flex;
  align-items:center;justify-content:center;background:{'#b9bec9' if not dark else '#5a5a5e'}}}
.cc_tile.on .cc-ic{{background:{C['blue']}}}
.cc_tile.on .cc-ic svg{{filter:none}}
.alerttiles{{gap:10px;justify-content:space-between}}
.at-cell{{flex:1;height:100%;border-radius:12px;background:{C['group']};border:1px solid {C['sep']};
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;position:relative}}
.at-icon{{width:40px;height:40px;border-radius:9px;background:{C['field']};display:flex;
  align-items:center;justify-content:center}}
.at-icon svg [fill="#fff"]{{fill:{C['sub']}}}
.at-icon svg [stroke="#fff"]{{stroke:{C['sub']}}}
.at-lbl{{font-size:11px;color:{C['ink']};text-align:center;line-height:1.1;padding:0 4px}}
.at-check{{position:absolute;top:6px;right:8px;color:{C['blue']};font-size:14px;font-weight:700}}
.cc-lb{{font-size:13.5px;font-weight:600;color:{C['ink']};line-height:1.15}}
.cc-st{{font-size:12px;color:{C['sub']}}}
.cc_module{{border-radius:38px;background:{'rgba(255,255,255,.5)' if not dark else 'rgba(70,70,74,.55)'};
  backdrop-filter:blur(18px)}}
.cc_round{{border-radius:50%;justify-content:center;
  background:{'rgba(255,255,255,.72)' if not dark else 'rgba(90,90,96,.62)'}}}
.cc_round.on{{background:{C['blue']}}}
.ccr-ic{{display:flex;align-items:center;justify-content:center}}
.cc_round.off .ccr-ic svg [fill="#fff"]{{fill:{'#3a3a3e' if not dark else '#fff'}}}
.cc_round.off .ccr-ic svg [stroke="#fff"]{{stroke:{'#3a3a3e' if not dark else '#fff'}}}
.cc_media{{border-radius:38px;background:{'rgba(255,255,255,.5)' if not dark else 'rgba(70,70,74,.55)'};
  backdrop-filter:blur(18px);justify-content:center}}
.ccm{{display:flex;flex-direction:column;align-items:center;gap:6px}}
.ccm-note{{font-size:34px;opacity:.5}}
.ccm-title{{font-size:14px;font-weight:600;color:{C['ink']}}}
.ccm-sub{{font-size:12px;color:{C['sub']}}}
.cc_vslider{{justify-content:center}}
.ccv-track{{position:relative;width:100%;height:100%;border-radius:26px;overflow:hidden;
  background:{'rgba(255,255,255,.45)' if not dark else 'rgba(70,70,74,.5)'}}}
.ccv-fill{{position:absolute;left:0;right:0;bottom:0;background:{'#fff' if not dark else '#e8e8ec'}}}
.ccv-ic{{position:absolute;left:50%;bottom:14px;transform:translateX(-50%)}}
.ccv-ic svg [fill="#fff"]{{fill:#5a5a5e}}
.ccv-ic svg [stroke="#fff"]{{stroke:#5a5a5e}}
/* fab */
.fab{{position:absolute;width:56px;height:56px;border-radius:50%;background:{C['blue']};color:#fff;
  display:flex;align-items:center;justify-content:center;font-size:28px;box-shadow:0 4px 12px rgba(0,0,0,.3);z-index:35}}
.composebtn{{position:absolute;width:52px;height:52px;border-radius:16px;background:{C['blue']};color:#fff;
  display:flex;align-items:center;justify-content:center;font-size:24px;
  box-shadow:0 4px 14px rgba(0,122,255,.4);z-index:35}}
/* home indicator */
.toolbtn{{justify-content:center;z-index:20}}
.tb-in{{display:flex;flex-direction:column;align-items:center;gap:2px}}
.tb-ic{{display:flex;align-items:center;justify-content:center}}
.tb-lb{{font-size:10px;color:{C['blue']}}}
.homebar{{position:absolute;bottom:0;left:0;width:100%;height:{NAVBAR_H}px;display:flex;
  align-items:center;justify-content:center;z-index:30}}
.homepill{{width:140px;height:5px;border-radius:3px;background:{C['ink']};opacity:.85}}
/* iOS keyboard */
.keyboard{{position:absolute;bottom:0;left:0;width:100%;background:{C['kb_bg']};
  padding:6px 3px 0;display:flex;flex-direction:column;gap:11px;z-index:45}}
.ksugg{{display:flex;justify-content:space-around;align-items:center;height:40px}}
.sugg{{font-size:17px;color:{C['ink']};flex:1;text-align:center;opacity:.9}}
.sugg.mid{{font-weight:600}}
.ksugg.autofill{{height:38px;justify-content:center}}
.afchip{{display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:{C['key']};border-radius:8px;padding:3px 16px;box-shadow:0 1px 0 rgba(0,0,0,.2)}}
.afcode{{font-size:16px;font-weight:600;color:{C['keyink']};letter-spacing:2px;line-height:1.1}}
.aflabel{{font-size:9px;color:{C['sub']};line-height:1.1}}
.keyboard.afk{{gap:8px}}
.keyboard.afk .key.numkey{{height:46px}}
.krow{{display:flex;justify-content:center;gap:6px;padding:0 3px}}
.krow.indent{{padding:0 20px}}
.key{{background:{C['key']};border-radius:6px;height:44px;display:flex;align-items:center;
  justify-content:center;font-size:19px;flex:1;box-shadow:0 1px 0 rgba(0,0,0,.28);color:{C['keyink']}}}
.key.fnkey{{background:{C['keyfn']};font-size:16px;flex:1.35;color:{C['keyink']}}}
.key.shift,.key.del{{flex:1.35}}
.key.spacebar{{flex:5;font-size:16px}}
.key.returnkey{{background:{C['keyfn']};flex:2;font-size:15px;color:{C['keyink']}}}
.krow.bottomrow .key{{height:44px}}
.key.pressed{{background:{C['blue']};color:#fff;transform:translateY(1px)}}
.key.pressed .numbig,.key.pressed .numsub{{color:#fff}}
.key.ghost{{background:transparent;box-shadow:none}}
.krow.numrow{{padding:0 4px;gap:6px}}
.key.numkey{{flex:1;height:50px;flex-direction:column;gap:0;background:{C['key']}}}
.numbig{{font-size:24px;line-height:1;color:{C['keyink']}}}
.numsub{{font-size:9px;letter-spacing:1px;color:{C['sub']};margin-top:1px}}
.key.numkey.del{{background:transparent;box-shadow:none;font-size:22px}}
.kbhome{{height:{NAVBAR_H}px;display:flex;align-items:center;justify-content:center;margin-top:2px}}
.kbhome .homepill{{background:{C['ink']}}}
/* alert (iOS) */
.scrim{{position:absolute;inset:0;background:rgba(0,0,0,.2);z-index:50}}
.scrim.blur{{backdrop-filter:blur(2px)}}
.alert{{position:absolute;background:{C['alert_bg']};border-radius:14px;z-index:51;overflow:hidden;
  backdrop-filter:blur(20px);box-shadow:0 10px 40px rgba(0,0,0,.3);display:flex;flex-direction:column}}
.alert-head{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;padding:18px 16px 14px}}
.alert-title{{font-size:17px;font-weight:600;color:{C['ink']}}}
.alert-body{{font-size:13px;color:{C['ink']};margin-top:4px;line-height:1.35;opacity:.9}}
.alert-acts{{border-top:.5px solid {C['sep']};display:flex}}
.alert-acts.two{{flex-direction:row}}
.alert-acts.two .alert-btn+.alert-btn{{border-left:.5px solid {C['sep']}}}
.alert-acts.stack{{flex-direction:column}}
.alert-acts.stack .alert-btn+.alert-btn{{border-top:.5px solid {C['sep']}}}
.alert-btn{{flex:1;height:48px;display:flex;align-items:center;justify-content:center;
  font-size:17px;color:{C['blue']}}}
.alert-btn.strong{{font-weight:600}}
.alert-btn.danger{{color:{C['red']};font-weight:600}}
.hit{{position:absolute;z-index:52;background:transparent}}
/* action sheet */
.sheet-card{{position:absolute;background:{C['alert_bg']};border-radius:14px;z-index:51;
  backdrop-filter:blur(20px);box-shadow:0 8px 30px rgba(0,0,0,.25)}}
.sheet-title{{position:absolute;z-index:52;display:flex;align-items:center;justify-content:center;
  text-align:center;font-size:13px;color:{C['sub']};padding:0 16px}}
.sheet-btn{{position:absolute;z-index:52;display:flex;align-items:center;justify-content:center;
  font-size:20px;color:{C['blue']}}}
.sheet-btn .sheethair{{position:absolute;left:0;right:0;bottom:0;height:.5px;background:{C['sep']}}}
.sheet-btn.cancel{{background:{C['group']};border-radius:14px;font-weight:600;
  backdrop-filter:blur(20px)}}
/* toast */
.toast{{position:absolute;left:40px;right:40px;bottom:{NAVBAR_H+18}px;background:rgba(0,0,0,.82);
  color:#fff;padding:12px 18px;border-radius:20px;font-size:14px;z-index:56;text-align:center}}
/* heads-up notification */
.headsup{{position:absolute;top:{STATUS_H+8}px;left:10px;right:10px;background:{C['alert_bg']};
  border-radius:22px;padding:12px 14px;z-index:55;display:flex;gap:12px;align-items:center;
  backdrop-filter:blur(20px);box-shadow:0 6px 24px rgba(0,0,0,.22)}}
.hu-ic{{font-size:20px;width:38px;height:38px;border-radius:9px;display:flex;align-items:center;
  justify-content:center;color:#fff}}
.hu-title{{font-size:14px;font-weight:600;color:{C['ink']}}}
.hu-text{{font-size:14px;color:{C['ink']};margin-top:1px;opacity:.9}}
/* fx */
.fxlayer{{position:absolute;inset:0;pointer-events:none;z-index:70}}
.ripple{{position:absolute;width:56px;height:56px;margin:-28px;border-radius:50%;
  background:rgba(0,122,255,.06);border:2.5px solid rgba(0,122,255,.75)}}
.ripple.long{{width:72px;height:72px;margin:-36px;background:rgba(0,122,255,.05)}}
.ripple.disabled{{background:rgba(150,150,150,.28);border-color:rgba(140,140,140,.6)}}
.swipefx{{position:absolute;font-size:34px;color:rgba(0,122,255,.85);margin:-20px;font-weight:700}}
.missfx{{position:absolute;top:50%;left:50%;font-size:40px;color:rgba(255,59,48,.6);margin:-20px}}
</style>"""
