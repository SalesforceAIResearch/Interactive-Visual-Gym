#!/usr/bin/env python3
"""Compile hand-written task specs into verified Task definitions.

Each spec is written by hand: the situation, the wording, the trap, and WHICH controls the owner
wants. Everything mechanical is derived from the engine — the route to each page (with the scrolls),
the gesture plan, and the goal predicate (the state the plan actually produces). Every spec is then
replayed on a fresh phone and rejected unless the plan lands and the goal holds.

    python screensim_bench/suite/build_tasks.py            # verify + write screensim/tasks_rl.py
    python screensim_bench/suite/build_tasks.py --check    # verify only
"""
from __future__ import annotations

import copy
import os
import sys


from screensim.core.engine import Device
from screensim.interact.composite_run_v2 import OracleV2
from screensim.tasks import resolve_dynamic
from screensim_bench.suite.task_specs import SPECS


def _diff(a, b, pre=""):
    out = []
    if isinstance(a, dict):
        for k in a:
            out += _diff(a[k], (b or {}).get(k), f"{pre}.{k}" if pre else k)
    elif a != b:
        out.append((pre, b))
    return out


def goto(dev, page_key, out):
    """Back out to the deepest shared ancestor of the target, then walk down to it, scrolling."""
    by_key, _ = OracleV2._page_paths()
    labels = by_key.get(page_key)
    if labels is None:
        return False
    chain = ["settings.root"]
    for i in range(1, len(labels) + 1):
        chain.append(OracleV2._page_paths()[1].get(tuple(labels[:i]), ""))
    for _ in range(12):
        if dev.st.cur_screen_key() in chain:
            break
        dev.act("press back")
        out.append("press back")
    cur = dev.st.cur_screen_key()
    if cur not in chain:
        return False
    for lab in labels[chain.index(cur):]:
        if not OracleV2._seek(dev, f"tap {lab}", out):
            return False
    return True


def build(spec):
    """(plan, required) or (None, reason). `required` is the state the plan produces.

    A step is either ("<settings page key>", [actions]) — the compiler routes there and scrolls to
    each control — or ("raw", [gestures]) — the gestures are performed exactly as written from
    wherever the phone is. Raw steps are how dialogs, sliders, typing and the non-Settings apps are
    written: hand-written gesture sequences, still rejected unless they replay.
    """
    start = spec.get("start_app", "settings")
    d = Device(spec.get("seed", 0))
    if start != "home":
        d.open_app(start)
    before = copy.deepcopy(d.st.data)
    plan = []
    for page_key, actions in spec["steps"]:
        if page_key == "raw":
            for a in actions:
                r = d.act(resolve_dynamic(d, a))
                if not getattr(r, "ok", True):
                    return None, f"gesture refused: {a} ({getattr(r, 'reason', '')})"
                plan.append(a)
            continue
        if not goto(d, page_key, plan):
            return None, f"cannot route to {page_key}"
        for a in actions:
            if not OracleV2._seek(d, a, plan):
                return None, f"action did not land: {a} (on {page_key})"
    changes = _diff(before, d.st.data)
    changes = [(p, v) for p, v in changes if not p.startswith(("settings._search", "settings._apps_search"))]
    if len(changes) < spec.get("min_changes", 2):
        return None, f"only {len(changes)} state changes: {changes}"
    return plan, dict(changes)


def verify(spec, plan, required):
    """Replay the plan on a fresh phone: the goal must hold and nothing beyond it may move."""
    d = Device(spec.get("seed", 0))
    if spec.get("start_app", "settings") != "home":
        d.open_app(spec.get("start_app", "settings"))
    before = copy.deepcopy(d.st.data)
    for g in plan:
        d.act(resolve_dynamic(d, g))
    got = dict((p, v) for p, v in _diff(before, d.st.data)
               if not p.startswith(("settings._search", "settings._apps_search")))
    if got != required:
        return f"replay diverged: {sorted(set(got.items()) ^ set(required.items()))}"
    for path, val in required.items():
        cur = d.st.data
        for k in path.split("."):
            cur = cur[k]
        if cur != val:
            return f"{path} is {cur!r}, wanted {val!r}"
    return None


HEAD = '''"""Hand-written tasks for RL, compiled by screensim_bench/suite/build_tasks.py — do not edit by hand.

The wording, the situation and the choice of controls are hand-written (screensim_bench/suite/task_specs.py);
the plans and the goal states below are what the engine produced and verified by replay. `band` is
the complexity band: 1 = one page, a couple of controls; 2 = one page with a leave-alone
constraint; 3 = benchmark parity (several pages, a trap); 4 = above benchmark.
"""
from .tasks import g_paths, task

'''


def main():
    ok, bad = [], []
    for s in SPECS:
        plan, required = build(s)
        if plan is None:
            bad.append((s["id"], required))
            continue
        why = verify(s, plan, required)
        if why:
            bad.append((s["id"], why))
            continue
        ok.append((s, plan, required))
    for i, r in bad:
        print(f"  REJECTED {i}: {r}")
    print(f"{len(ok)} verified, {len(bad)} rejected")
    by_band = {}
    for s, _p, _r in ok:
        by_band[s["band"]] = by_band.get(s["band"], 0) + 1
    print("by band:", dict(sorted(by_band.items())))
    if "--check" in sys.argv:
        return
    lines = [HEAD]
    for s, plan, required in sorted(ok, key=lambda x: (x[0]["band"], x[0]["id"])):
        lines.append(f'''task(id={s["id"]!r}, family={s["family"]!r}, band={s["band"]},
     instruction={s["instruction"]!r},
     start_app={s.get("start_app", "settings")!r}, seed={s.get("seed", 0)!r},
     mirrors={s.get("mirrors", "")!r},
     trap={s.get("trap", "")!r},
     plan={plan!r},
     goal=g_paths({required!r}))
''')
    open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "screensim", "tasks_rl.py"), "w").write("\n".join(lines))
    print("wrote screensim/tasks_rl.py")


if __name__ == "__main__":
    main()
