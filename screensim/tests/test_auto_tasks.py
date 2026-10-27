"""Every auto-generated manual task must replay to a win.

The library is derived from the mirrored Apple manual with recorded state-diff
goals; if an engine or spec change breaks any of them, this names the task.
Skipped when the corpus isn't mirrored.
"""
import os

import pytest

from screensim.autotasks import SRC, load_auto_tasks
from screensim.runner import run_task

pytestmark = pytest.mark.skipif(not os.path.exists(SRC),
                                reason="auto-task library not generated")


def test_auto_library_nonempty_and_cited():
    tasks = load_auto_tasks()
    assert len(tasks) >= 150
    assert all(t.mirrors.startswith("docs/apple_manual/") for t in tasks)


def test_every_auto_task_replays_to_won():
    failing = []
    for t in load_auto_tasks():
        r = run_task(t)
        if not r.won:
            failing.append((t.id, r.detail))
    assert not failing, (
        f"{len(failing)} auto tasks broke: " +
        "; ".join(f"{i} ({d})" for i, d in failing[:6]))
