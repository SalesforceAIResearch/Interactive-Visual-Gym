"""Execute a task's clean plan through the engine, returning a trace of
(gesture, ActionResult, state-hash) per step plus the final goal verdict.

This is the shared source for: the task-correctness tests, the video pipeline,
and later the error-injection harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core.engine import Device
from .tasks import Task, resolve_dynamic


@dataclass
class Step:
    idx: int
    cmd: str
    ok: bool
    status: str
    reason: str
    screen: str
    tick: int
    hash: str
    n_options: int = 0      # visible interactive widgets BEFORE this gesture
    pre_screen: str = ""    # screen the gesture was made on


@dataclass
class Rollout:
    task_id: str
    steps: list
    won: bool
    detail: str
    outcome: str
    final_tick: int


def run_task(task: Task, seed: int = None, verbose: bool = False,
             on_frame=None) -> Rollout:
    dev = Device(seed=seed if seed is not None else task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    if on_frame:
        on_frame(dev, -1, None)      # initial frame

    steps = []
    for i, raw in enumerate(task.plan):
        cmd = resolve_dynamic(dev, raw)
        pre = dev.laid_out_screen()
        n_opts = sum(1 for w in pre.widgets if w.visible and w.interactive())
        pre_key = dev.st.cur_screen_key()
        r = dev.act(cmd)
        steps.append(Step(i, cmd, r.ok, r.status, r.reason,
                          dev.st.cur_screen_key(), dev.st.tick, dev.st.hash(),
                          n_options=n_opts, pre_screen=pre_key))
        if verbose:
            print(f"  [{i}] {cmd!r:34} ok={r.ok} {r.status:9} "
                  f"screen={dev.st.cur_screen_key()} {r.reason}")
        if on_frame:
            on_frame(dev, i, r)
        if dev.st.outcome:
            break

    won, detail = task.goal(dev.st)
    return Rollout(task.id, steps, won, detail,
                   dev.st.outcome or ("won" if won else "incomplete"),
                   dev.st.tick)
