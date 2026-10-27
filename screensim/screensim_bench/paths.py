"""Where the ScreenSim release keeps its data.

The release root is the directory holding ``benchmark/`` (the mirrored User Guide derivatives the
engine reads) and the two packages. Every tool imports ``ROOT`` from here rather than guessing from
its own location, so a tool works the same whatever directory it is run from.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BENCHMARK = os.path.join(ROOT, "benchmark")
AUTO_TASKS = os.path.join(BENCHMARK, "auto_tasks.jsonl")
MANUAL_PROCEDURES = os.path.join(BENCHMARK, "manual_procedures.jsonl")
RESULTS = os.environ.get("SCREENSIM_RESULTS") or os.path.join(ROOT, "results")
