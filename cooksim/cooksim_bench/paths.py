"""Where the CookSim release keeps its data.

The release root is the directory that holds ``benchmark/`` (the suite, kitchen maps and reference ticks),
``web/`` (the renderer) and ``serving/`` (local model servers). Every harness module imports ``ROOT`` from
here instead of guessing from its own location.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BENCHMARK = os.path.join(ROOT, "benchmark")
GRID = os.path.join(BENCHMARK, "grid.json")
SUITE = os.path.join(BENCHMARK, "cases_composite_v5.json")
COOK_ABLATION_CELLS = os.path.join(BENCHMARK, "cook_ablation_cells.json")
WEB = os.path.join(ROOT, "web")
