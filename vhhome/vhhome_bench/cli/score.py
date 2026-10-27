"""vhhome-score: grade a run root (or several independent runs) and print the table.

Steps, in order:
  objective  task outcome, in-time success S, efficiency EFF, detection DET (no API calls); writes <run>/metrics_table.json
  rubric     the per-round interaction-quality rubric (AQ) with the judges named by --judges; resumable per judge;
             writes <run>/sq_v4/<tag>/*.json and prints the arm table
  table      the pooled final table over all --run roots (mean +- half-range over runs; AQ with a bootstrap interval);
             writes final_results.{json,md} next to the first run root (or --out)
"""
import argparse, os, subprocess, sys

STEPS = ["objective", "rubric", "table"]


def sh(args):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(args)
    if r.returncode:
        sys.exit(f"step failed (rc={r.returncode})")


def main():
    ap = argparse.ArgumentParser(prog="vhhome-score", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run root(s) given to vhhome-benchmark / vhhome-episode")
    ap.add_argument("--only", nargs="*", choices=STEPS, default=None, help="run only these steps")
    ap.add_argument("--judges", default="gemini,gpt",
                    help="rubric judges: gemini,gpt (the reported setting; GEMINI_API_KEY + OPENAI_API_KEY) | all (adds claude)")
    ap.add_argument("--workers", type=int, default=32, help="parallel judge calls")
    ap.add_argument("--out", default=None, help="where the pooled table is written (default: beside the first run root)")
    a = ap.parse_args()
    steps = a.only or STEPS
    runs = [os.path.abspath(r) for r in a.run]
    py = sys.executable
    from vhhome import keys
    keys.load_all()
    for root in runs:
        if "objective" in steps:
            sh([py, "-m", "vhhome_bench.scoring.results_table", root])
            sh([py, "-m", "vhhome_bench.scoring.metrics_table", root, "--json", os.path.join(root, "metrics_table.json")])
        if "rubric" in steps:
            sh([py, "-m", "vhhome_bench.scoring.rubric", root, "--subset", "--judge", a.judges, "--workers", str(a.workers)])
            sh([py, "-m", "vhhome_bench.scoring.rubric_table", root])
    if "table" in steps:
        from vhhome_bench.scoring import final_table
        final_table.main(roots=runs, out_dir=a.out or os.path.dirname(runs[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
