"""cooksim-score: grade a run -- task outcome and detection, question answering, proactive alerts, and the
interaction-quality rubric -- then print the leaderboard rows.

    cooksim-score --run runs/gemini                 # everything
    cooksim-score --run runs/gemini --only objective  # one step

Steps (all resume-safe; re-running grades only what is missing):
  objective   outcome, in-time success, efficiency, prevention timing, detection F1   (no API calls)
  qa          question answers graded by the LLM judge                               (OPENAI_API_KEY)
  proactive   "tell me when X is ready" alerts graded                                (OPENAI_API_KEY)
  rubric      interaction quality: every assistant line graded by independent judges (GEMINI_API_KEY, OPENAI_API_KEY)
  table       paper objective metrics and five-category quality over runs 1-3
"""
import argparse
import os
import runpy
import subprocess
import sys

from cooksim_bench.cli._common import apply_bench_env

STEPS = ["objective", "qa", "proactive", "rubric", "table"]
MODULES = {
    "objective": ("cooksim_bench.scoring.eval_v2_sweep", []),
    "qa": ("cooksim_bench.scoring.regrade_qa", ["--workers", "8"]),
    "proactive": ("cooksim_bench.scoring.regrade_proactive_v2", []),
    "rubric": ("cooksim_bench.scoring.sq_v4_score", ["--judge", "gemini,gpt", "--workers", "64"]),
    "table": ("cooksim_bench.scoring.sq_v4_three_runs", []),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cooksim-score", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run root given to cooksim-benchmark / cooksim-episode")
    ap.add_argument("--only", nargs="*", choices=STEPS, default=None, help="run only these steps")
    ap.add_argument("--judges", default="gemini,gpt", help="rubric judges: gemini,gpt (the reported setting) | all (adds claude)")
    ap.add_argument("--workers", type=int, default=64, help="parallel judge calls for the rubric")
    args, extra = ap.parse_known_args(argv)
    apply_bench_env()
    run = os.path.abspath(args.run)
    env = dict(os.environ, ASSIST_ROOT=run)
    steps = args.only or STEPS
    for step in steps:
        mod, flags = MODULES[step]
        if step == "rubric":
            flags = ["--judge", args.judges, "--workers", str(args.workers)]
        print(f"== {step}: python -m {mod} {' '.join(flags)}", flush=True)
        rc = subprocess.call([sys.executable, "-m", mod, *flags, *extra], env=env)
        if rc == 0 and step == "table":
            rc = subprocess.call([sys.executable, "-m", "cooksim_bench.scoring.paper_table"], env=env)
        if rc != 0:
            sys.exit(f"step {step} failed (exit {rc}); fix the cause and re-run -- every step resumes")
    print(f"done. objective: {run}/eval_v2.json  qa: {run}/qa_regrade.json  proactive: {run}/proactive_v2.json  "
          f"rubric: {run}/sq_v4/  table: {run}/sq_v4/three_runs_five.json")


if __name__ == "__main__":
    main()
