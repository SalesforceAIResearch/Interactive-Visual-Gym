"""screensim-score: the headline table for a results file — success, efficiency, detection.

    screensim-score --results runs/gpt

Strict success means the owner's goal is met AND nothing else on the phone changed. Efficiency is
reported over all episodes and over the ones the seat actually won; detection is prevention timing and
F1. Interaction quality comes from screensim-grade and is reported separately.
"""
import argparse
import runpy
import sys
from pathlib import Path
from screensim_bench.cli._common import record_roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, nargs="+", help="one or more run directories")
    ap.add_argument("--out", help="output JSON; defaults to the first run directory")
    ap.add_argument("--three-runs", action="store_true",
                    help="aggregate three independent runs: mean, SD, pass^3 and flip rate")
    a = ap.parse_args(argv)
    try:
        roots = record_roots(a.results)
    except ValueError as exc:
        ap.error(str(exc))
    if a.three_runs and len(roots) != 3:
        ap.error("--three-runs requires three distinct run directories after --results")
    mod = "screensim_bench.scoring.three_runs" if a.three_runs else "screensim_bench.scoring.scores"
    output = a.out or str(Path(roots[0]) / ("three_runs.json" if a.three_runs else "objective_scores.json"))
    previous = sys.argv
    try:
        sys.argv = ["screensim-score", *roots, "--out", output]
        runpy.run_module(mod, run_name="__main__")
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
