"""Plan IR — lift a task's flat gesture list into named subgoal nodes.

A node is "navigate somewhere and commit one piece of application state":
its `approach` gestures produce no durable state change, its final `gesture`
does, and `commit` records the exact deep-diff it produced. The ordered
commit sequence, restricted to the paths the reference plan touches, is the
task's PROGRESS KEY — committed application state, never screen state, never
the widget tree (skill §12.2): both legal routes to an outcome produce the
same commits, and scratch writers never enter it.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from ..core.engine import Device
from ..tasks import Task, resolve_dynamic

# durable-commit filter: scratch keys are underscore-prefixed by convention;
# wifi join plumbing is transient UI state, not progress
_VOLATILE = ("wifi.status", "wifi.joining", "wifi.join_pw")

# paths whose commit cannot be undone in-world
IRREVERSIBLE_PREFIXES = ("bank.history", "bank.balance", "shop.order",
                         "mail.sent")


def _volatile(path: str) -> bool:
    if any(seg.startswith("_") for seg in path.split(".")):
        return True
    return any(path == v or path.startswith(v + ".") for v in _VOLATILE)


def deep_diff(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b), key=str):
        pa = f"{prefix}.{k}" if prefix else str(k)
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += deep_diff(va, vb, pa)
        elif va != vb:
            out.append((pa, vb))
    return [(p, v) for p, v in out if not _volatile(p)]


@dataclass
class Node:
    idx: int
    kind: str                    # "reach" | "commit" | "navigate"
    gestures: list               # approach + the final (nav or commit) gesture
    commit: list = field(default_factory=list)   # [(path, new_value)] for commit
    screen: str = ""             # screen reached / committed on
    from_screen: str = ""        # (reach) the screen the nav was made FROM
    irreversible: bool = False
    label: str = ""

    @property
    def approach(self):
        # commit and reach nodes both END in the acting gesture; a bare
        # navigate (trailing) node is all approach
        return (self.gestures[:-1] if self.kind in ("commit", "reach")
                else list(self.gestures))

    @property
    def gesture(self):
        return (self.gestures[-1] if self.kind in ("commit", "reach")
                else None)


@dataclass
class PlanIR:
    task_id: str
    nodes: list
    commits: list                # ordered [(path, value)] across the whole plan
    touched_paths: list          # sorted unique committed paths
    total_ticks: int = 0

    def flat(self):
        out = []
        for n in self.nodes:
            out += n.gestures
        return out


def _snap(dev):
    return copy.deepcopy(dev.st.data)


def compile_plan(task: Task, plan=None) -> PlanIR:
    """Replay the plan headlessly and segment into subgoals. A subgoal ends
    either at a durable-state COMMIT or at a forward NAVIGATION to a required
    new screen (a 'reach' subgoal — the pages the task must pass through).
    Navigation subgoals densify the axis and are the anchors for wrong-page
    errors; the path-invariant progress key is still the commit sequence only.
    """
    plan = list(plan if plan is not None else task.plan)
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    nodes, commits, acc = [], [], []
    before = _snap(dev)
    prev_screen = dev.st.cur_screen_key()
    prev_depth = len(dev.st.stack)
    for raw in plan:
        cmd = resolve_dynamic(dev, raw)
        dev.act(cmd)
        after = _snap(dev)
        diff = deep_diff(before, after)
        acc.append(raw)                       # keep the raw form ("type OTP")
        cur_screen = dev.st.cur_screen_key()
        cur_depth = len(dev.st.stack)
        forward_nav = (cur_screen != prev_screen and cur_depth > prev_depth)
        if diff:
            irr = any(p.startswith(IRREVERSIBLE_PREFIXES) for p, _ in diff)
            top = diff[0]
            nodes.append(Node(idx=len(nodes), kind="commit",
                              gestures=list(acc), commit=diff,
                              screen=cur_screen, from_screen=prev_screen,
                              irreversible=irr,
                              label=f"{raw} -> {top[0]}={json.dumps(top[1])[:40]}"))
            commits += diff
            acc = []
        elif forward_nav:
            # a required forward navigation to a new screen — a reach subgoal
            nodes.append(Node(idx=len(nodes), kind="reach",
                              gestures=list(acc), screen=cur_screen,
                              from_screen=prev_screen,
                              label=f"{raw} -> {_pretty(cur_screen)}"))
            acc = []
        before = after
        prev_screen, prev_depth = cur_screen, cur_depth
    if acc:
        nodes.append(Node(idx=len(nodes), kind="navigate", gestures=acc,
                          screen=dev.st.cur_screen_key(), label="finish"))
    touched = sorted({p for p, _ in commits})
    return PlanIR(task_id=task.id, nodes=nodes, commits=commits,
                  touched_paths=touched, total_ticks=dev.st.tick)


def _pretty(screen_key):
    return screen_key.split(".", 1)[-1] if "." in screen_key else screen_key


def commits_between(prev_data, cur_data, touched_paths):
    """The commits one world step produced, restricted to plan-touched paths —
    the live loop's incremental progress reader."""
    return [(p, v) for p, v in deep_diff(prev_data, cur_data)
            if p in touched_paths]


def progress_prefix(ir: PlanIR, k: int):
    """The anchor value: the first k commits of the reference sequence."""
    return [list(c) for c in ir.commits[:k]]
