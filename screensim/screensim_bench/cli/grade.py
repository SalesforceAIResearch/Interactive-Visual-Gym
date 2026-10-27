"""screensim-grade: judge interaction quality, round by round.

    screensim-grade --results results/gpt

Every assistant line is one round, judged against the exact screen the person was looking at. Two
judges (gemini-3.7-flash and gpt-5.4-mini) score each round and the score is their mean; resume is per
judge. See docs/INTERACTION_QUALITY_GRADING.md.
"""
import argparse
import runpy
import sys
from screensim_bench.cli._common import record_roots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="a run directory")
    ap.add_argument("--judge", default="gemini,gpt", help="gemini,gpt (default) | all | one name")
    ap.add_argument("--subset", default="five", choices=["all", "five"],
                    help="'five' grades only the ten items of the five reported categories")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default="", help="comma list of seats; default every seat in the run")
    a = ap.parse_args(argv)
    try:
        root = record_roots([a.results])[0]
    except ValueError as exc:
        ap.error(str(exc))
    if a.workers < 1:
        ap.error("--workers must be positive")
    previous = sys.argv
    try:
        sys.argv = ["screensim-grade", root, "--judge", a.judge, "--subset", a.subset,
                    "--workers", str(a.workers)] + (["--models", a.models] if a.models else [])
        runpy.run_module("screensim_bench.scoring.rubric", run_name="__main__")
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
