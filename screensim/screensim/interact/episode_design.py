"""Multi-event episode design (the CookSim composite-episode analogue).

An episode is an ordered sequence of events fired at anchors along a task's
reference plan:

  E  error       — an injected slip (from the certified error pool); the
                   assistant must catch it.
  D  deviation   — a legitimate, announced change of plan; the assistant must
                   NOT flag it as an error (restraint), only acknowledge.
  Q  question    — a spot-check the human asks the assistant; the engine
                   computes the ground-truth answer from the screen at that
                   anchor, and the assistant is scored on answering it.

`design_episodes(task)` returns three diverse episodes per task, each a mix of
the three, with the firing pattern (e.g. "QED"). Questions and deviations are
validated against the engine; errors come from the admissible pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.engine import Device
from ..tasks import by_id, resolve_dynamic
from .plan_ir import compile_plan
from .errors import generate_cases


# ---- authored, in-character legitimate deviations (per task) ----
DEVIATIONS = {
    "lost_phone_lockdown": (
        "Actually, hold off on Stolen Device Protection for now — the iCloud "
        "encryption is enough for this trip.", "sdp"),
    "app_privacy_quarantine": (
        "Actually, let TripSnap use my location While Using the App — I do "
        "check its map sometimes, I just don't want it in the background.",
        "loc_while"),
    "parent_phone_readability": (
        "Actually skip the button shapes — my dad finds them cluttered.",
        "no_button_shapes"),
    "lock_screen_lockdown": (
        "Actually keep Wallet on the lock screen — he taps it for the bus.",
        "keep_wallet"),
    "rent_share_standard": (
        "Actually make it Instant — he needs it today and the fee's fine.",
        "instant"),
}


@dataclass
class Event:
    kind: str            # "E" | "D" | "Q"
    node_idx: int        # anchor (which commit node it fires at)
    order: int           # firing order within the episode
    label: str = ""      # short description
    detail: str = ""     # full "what happens"
    ground_truth: str = ""   # for Q
    q_kind: str = ""     # for Q
    err: str = ""        # for E


@dataclass
class EpisodeSpec:
    id: str
    task: str
    events: list
    pattern: str
    n_nodes: int


def _screen_at(task, ir, node_idx):
    """Replay the plan up to (but not committing) node_idx; return the Device
    positioned on that node's screen."""
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    for nd in ir.nodes:
        if nd.idx == node_idx:
            for g in nd.approach:
                dev.act(resolve_dynamic(dev, g))
            return dev
        for g in nd.gestures:
            dev.act(resolve_dynamic(dev, g))
    return dev


def make_question(task, ir, node_idx):
    """A spot-check about a control visible on this node's screen, with the
    engine's ground-truth answer."""
    dev = _screen_at(task, ir, node_idx)
    scr = dev.laid_out_screen()
    vis = [w for w in scr.widgets if w.visible and w.interactive() and w.label]
    toggles = [w for w in vis if w.kind == "toggle"]
    radios = [w for w in vis if w.kind == "radio"]
    if radios:
        sel = next((w for w in radios if w.value), None)
        opts = ", ".join(f"“{w.label}”" for w in radios[:4])
        return Event("Q", node_idx, 0, q_kind="selected_option",
                     label="which option is selected",
                     detail=f"“On this screen — {opts} — which one is picked "
                            f"right now?”",
                     ground_truth=(sel.label if sel else "none selected"))
    if toggles:
        w = toggles[0]
        return Event("Q", node_idx, 0, q_kind="toggle_state",
                     label=f"is “{w.label}” on?",
                     detail=f"“Is the “{w.label}” switch on or off right now?”",
                     ground_truth=("on" if w.value else "off"))
    # fallback: how many controls are on this screen
    where = f"{scr.title} screen" if scr.title else "screen"
    return Event("Q", node_idx, 0, q_kind="which_control",
                 label="what can I do here",
                 detail=f"“What are my options on this {where}?”",
                 ground_truth=f"{len(vis)} controls incl. "
                              f"{', '.join(w.label for w in vis[:3])}")


def _err_event(case, node_idx, order):
    right = case.deviated_label or "the right control"
    if case.err == "wrong_page":
        dest = (case.certified.get("wrong_dest", "") or "").split(".")[-1]
        d = (f"the human taps “{case.alt_gesture.replace('tap ','')}” and lands "
             f"on the wrong screen ({dest}) — lost, needs steering back to "
             f"“{right}”.")
    elif case.err == "skip_step":
        d = f"the human moves on WITHOUT doing “{right}”."
    elif case.alt_gesture:
        d = (f"the human taps “{case.alt_gesture.replace('tap ','')}” instead "
             f"of “{right}”.")
    else:
        d = "the human commits early, before finishing the setup."
    return Event("E", node_idx, order, label=case.err, detail=d, err=case.err)


def _dev_event(task_id, node_idx, order):
    text, tag = DEVIATIONS.get(task_id, ("Actually, small change of plan.", ""))
    return Event("D", node_idx, order, label="intent change", detail=text)


def design_episodes(task):
    """Three diverse multi-event episodes for one task, now spanning the
    denser subgoal axis (navigation reach-points + state commits) and using
    the wrong_page navigation error where one exists."""
    ir = compile_plan(task)
    _, cases, _ = generate_cases(task)
    errs = [c for c in cases if c.family == "error"]
    axis_nodes = [n for n in ir.nodes if n.kind in ("reach", "commit")]
    node_ids = [n.idx for n in axis_nodes]
    if not node_ids:
        return []
    lo, mid, hi = node_ids[0], node_ids[len(node_ids) // 2], node_ids[-1]
    # error pools by kind
    wrong_page = [c for c in errs if c.err == "wrong_page"]
    control_err = [c for c in errs if c.err != "wrong_page"]
    by_node = {}
    for c in control_err:
        by_node.setdefault(c.node_idx, c)
    ctrl_at = lambda ni: by_node.get(ni) or (control_err[0] if control_err else None)
    dev_node = mid if len(node_ids) > 2 else hi
    episodes = []

    # Episode 1 — a NAVIGATION error, caught mid-flow: QUESTION then wrong_page.
    e1 = []
    e1.append(make_question(task, ir, lo)); e1[-1].order = 1
    wp = wrong_page[0] if wrong_page else None
    if wp:
        e1.append(_err_event(wp, wp.node_idx, 2))
    elif ctrl_at(mid):
        e1.append(_err_event(ctrl_at(mid), mid, 2))
    episodes.append(_mk(task, 1, e1, len(node_ids)))

    # Episode 2 — a control ERROR then a legitimate DEVIATION (stay quiet).
    e2 = []
    ce = ctrl_at(lo) or ctrl_at(mid)
    if ce:
        e2.append(_err_event(ce, ce.node_idx, 1))
    e2.append(_dev_event(task.id, dev_node, 2))
    episodes.append(_mk(task, 2, e2, len(node_ids)))

    # Episode 3 — a dense run: wrong_page (or control err) · QUESTION · control err.
    e3 = []
    order = 1
    seen = set()
    first = (wrong_page[1] if len(wrong_page) > 1 else
             (wrong_page[0] if wrong_page else ctrl_at(lo)))
    if first:
        e3.append(_err_event(first, first.node_idx, order)); order += 1
        seen.add(first.node_idx)
    e3.append(make_question(task, ir, mid)); e3[-1].order = order; order += 1
    ce = ctrl_at(hi)
    if ce and ce.node_idx not in seen:
        e3.append(_err_event(ce, ce.node_idx, order)); order += 1
    else:
        e3.append(_dev_event(task.id, hi, order))
    e3.sort(key=lambda e: (e.node_idx, e.order))
    for i, e in enumerate(e3):
        e.order = i + 1
    episodes.append(_mk(task, 3, e3, len(node_ids)))
    return episodes


def _mk(task, n, events, nnodes):
    events = sorted(events, key=lambda e: (e.order,))
    pattern = "".join(e.kind for e in events)
    return EpisodeSpec(id=f"{task.id}__ep{n}", task=task.id, events=events,
                       pattern=pattern, n_nodes=nnodes)


def all_designs():
    from ..tasks import TASKS
    hard = ["lost_phone_lockdown", "app_privacy_quarantine",
            "parent_phone_readability", "lock_screen_lockdown",
            "rent_share_standard"]
    out = []
    for tid in hard:
        out += design_episodes(by_id(tid))
    return out
