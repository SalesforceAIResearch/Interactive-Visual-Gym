#!/usr/bin/env python3
"""Authoring aid for hand-written tasks: find a page, see its real controls, get an executable plan.

The wording, the goal and the trap are written by hand. The ROUTE is derived from the engine, so a
task's plan is executable by construction rather than by hope.

    python screensim_bench/suite/author.py card "lock screen"          # pages + their controls and state paths
    python screensim_bench/suite/author.py plan settings.lockaccess "tap Notification Center" "tap Wallet"
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys


from screensim.core.engine import Device
from screensim.interact.composite_run_v2 import OracleV2
from screensim.tasks import resolve_dynamic
from screensim.apps.inventory import PAGE_PATH, walk


def fresh(start="settings"):
    d = Device(0)
    if start != "home":
        d.open_app(start)
    return d


def _diff(a, b, pre=""):
    out = []
    if isinstance(a, dict):
        for k in a:
            out += _diff(a[k], (b or {}).get(k), f"{pre}.{k}" if pre else k)
    elif a != b:
        out.append((pre, b))
    return out


def route(page_key, dev=None):
    """The gestures from wherever the phone is to `page_key` (scrolling included), or None."""
    d = dev or fresh()
    by_key, _ = OracleV2._page_paths()
    labels = by_key.get(page_key)
    if labels is None:
        return None
    out = []
    for lab in labels:
        if not OracleV2._seek(d, f"tap {lab}", out):
            return None
    return out


def plan_for(page_key, actions):
    """route + the given actions, replayed on a fresh phone; returns (plan, state diff, ok)."""
    d = fresh()
    before = copy.deepcopy(d.st.data)
    out = route(page_key, d)
    if out is None:
        return None, [], False
    ok = True
    for a in actions:
        if not OracleV2._seek(d, a, out):
            ok = False
            break
    return out, _diff(before, d.st.data), ok


def cards(query):
    q = query.lower()
    hits = []
    for page, items in walk(with_paths=True):
        key = None
        for k, v in PAGE_PATH.items():
            if v.split(" > ")[-1].lower() == page.lower():
                key = k
                break
        if q in page.lower() or (key and q in key.lower()):
            hits.append((page, key, items))
    for page, key, items in hits[:12]:
        print("=" * 90)
        print(f"PAGE  {page}    key={key}    route={PAGE_PATH.get(key, '?')}")
        r = route(key) if key else None
        print(f"  gestures to get here: {r}")
        for it in items:
            head = it.split("\x00")[0].split("\x01")[0]
            path = it.split("\x00")[1] if "\x00" in it else ""
            print(f"   {head.strip()[:78]:78} {path}")


if __name__ == "__main__":
    if sys.argv[1] == "card":
        cards(sys.argv[2])
    else:
        p, diff, ok = plan_for(sys.argv[2], sys.argv[3:])
        print(json.dumps({"plan": p, "diff": diff, "all_actions_landed": ok}, indent=1))
