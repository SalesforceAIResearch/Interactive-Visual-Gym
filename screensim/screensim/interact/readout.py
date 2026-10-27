"""The two observation channels, from one geometry (skill §1.6, §5.2):

  exact_map(dev)  — the HUMAN's channel: every visible interactive widget with
                    its kind, value, and (x, y) centre. This is what the human
                    reads to drive its own body; it may name widgets to the
                    ENGINE by label/ordinal, but never to the assistant by
                    coordinate.
  relational(dev) — the ASSISTANT's optional text ablation: the same widgets
                    by label, kind, value, and screen REGION — no coordinates,
                    no ordinals-as-ids. On a screen the visible labels are the
                    shared vocabulary; that is still "never coordinates".

The assistant's primary channel is rendered frames (render/capture). Both text
readouts come from the laid-out scene, so they never assert anything the
renderer did not draw.
"""
from __future__ import annotations

from ..core.layout import region_of


def _widgets(dev):
    scr = dev.laid_out_screen()
    return scr, [w for w in scr.widgets if w.visible]


def exact_map(dev) -> str:
    scr, ws = _widgets(dev)
    lines = [f"SCREEN {scr.title or scr.key}"]
    # the back affordance in the app bar, as the renderer draws it
    try:
        from ..render.renderer import _prev_screen_title
        prev = _prev_screen_title(dev.st) if scr.title is not None else None
    except Exception:
        prev = None
    if prev:
        lines.append(f"  (top left) back     ‹ {prev}   (the gesture for it is: press back)")
    for w in ws:
        b = w.bounds
        if not w.interactive():
            # static text the person can read (headers such as "To Alex Kim", notes,
            # section titles) — it carries no tap, but it IS on the screen
            if w.kind in ("text", "section", "apphdr") and (w.label or getattr(w, "sub", None)):
                sub = f" · {w.sub}" if getattr(w, "sub", None) else ""
                lines.append(f"  ({b.cx:>3},{b.cy:>3}) {w.kind:8} {(w.label or '')[:40]}{sub[:30]}")
            continue
        val = "" if w.value is None else f" [{_val(w.value)}]"
        lines.append(f"  ({b.cx:>3},{b.cy:>3}) {w.kind:8} "
                     f"{(w.label or '—')[:34]}{val}")
    if dev.st.keyboard:
        lines.append("  <keyboard open>")
    if dev.st.dialog:
        lines.append(f"  <dialog: {dev.st.dialog.get('title','')}>")
    # a heads-up banner at the top of the screen (what the renderer shows for
    # the last 3 ticks after a notification arrives), and what is waiting in
    # Notification Center after that — a person can read both
    st = dev.st
    heads = [n for n in st.notifications
             if getattr(n, "arrived_at", -99) >= st.tick - 3 and not scr.is_dialog]
    if heads and not getattr(st, "shade_open", False):
        n = heads[0]
        lines.append(f"  <banner at the top — {n.title}: {n.text}>")
    elif st.notifications and not getattr(st, "shade_open", False):
        lines.append(f"  <{len(st.notifications)} notification(s) waiting — swipe down from the top edge to read them>")
    return "\n".join(lines)


def relational(dev) -> str:
    scr, ws = _widgets(dev)
    lines = [f"SCREEN: {scr.title or scr.key}"]
    for w in ws:
        if not (w.interactive() and w.label):
            continue
        reg = "-".join(sorted(region_of(w.bounds)))
        val = "" if w.value is None else f" (currently {_val(w.value)})"
        lines.append(f"  {w.kind} “{w.label[:36]}” in the {reg}{val}")
    return "\n".join(lines)


def proprioception(dev) -> str:
    """The human's own-state readout, every turn (skill §5.4)."""
    st = dev.st
    parts = [f"app={st.app}", f"screen={st.cur_screen_key()}"]
    if st.focus:
        parts.append(f"focused-field={st.focus}")
    parts.append("keyboard=open" if st.keyboard else "keyboard=closed")
    return "you are here: " + ", ".join(parts)


def _val(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)[:24]
