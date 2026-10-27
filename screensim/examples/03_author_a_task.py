"""Author a new task: hand-written intent, engine-derived route and goal.

Add a spec to screensim_bench/suite/task_specs.py, then:

    python -m screensim_bench.suite.build_tasks --check   # every spec must replay
    python -m screensim_bench.suite.build_tasks           # compile into screensim/tasks_rl.py
    python -m screensim_bench.suite.split                 # re-check training/benchmark separation

This script shows the authoring aid that finds a page and its real controls.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "screensim_bench.suite.author", "card", "lock screen"], check=False)
