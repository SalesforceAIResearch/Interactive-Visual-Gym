"""cooksim-benchmark: run an assistant over the whole suite (50 cases x 3 personas = 150 episodes), resumable.

    cooksim-benchmark --assistant gemini:gemini-3.8-flash --out runs/gemini --workers 4

Existing complete records under <out>/frames are skipped, so a killed run resumes where it stopped. Repeat the
command with --repeat 2 / --repeat 3 for the independent runs the leaderboard averages (records are tagged
<tag>__r2, <tag>__r3). Score the run with `cooksim-score --run runs/gemini`.
"""
import argparse
import os
import runpy
import sys

from cooksim_bench.cli._common import apply_bench_env, require_renderer
from cooksim_bench.cli.episode import assistant_flags


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cooksim-benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assistant", required=True, help="<backend>:<model> or a roster tag")
    ap.add_argument("--base-url", default=None); ap.add_argument("--profile", default=None)
    ap.add_argument("--out", required=True, help="run root (records under <out>/frames/<persona>/<tag>/)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--personas", nargs="+", default=["baseline", "classic_novice", "classic_expert"])
    ap.add_argument("--cases", nargs="*", default=None, help="a subset of case names (default: the whole suite)")
    ap.add_argument("--suite", default=None, help="a case file instead of the shipped suite")
    ap.add_argument("--workers", type=int, default=4, help="episodes run at once (API rate limits: 3-4; a local model: 2-3)")
    ap.add_argument("--repeat", type=int, default=1, help="independent run number (1, 2, 3); tags the records")
    ap.add_argument("--latency", default=None, help="measured (default) | fixed ticks per reply, e.g. 2")
    ap.add_argument("--dump-states", action="store_true")
    args, extra = ap.parse_known_args(argv)

    apply_bench_env({"COOKSIM_CASES": os.path.abspath(args.suite) if args.suite else None,
                     "COOKSIM_ASSIST_LATENCY": args.latency,
                     "COOKSIM_DUMP_PER_EPISODE": "1" if args.dump_states else None})
    require_renderer()
    flags, tag = assistant_flags(args.assistant, args.base_url, args.profile)
    tag = args.tag or tag
    if args.repeat > 1:
        tag = f"{tag}__r{args.repeat}"
    sys.argv = ["run_assist_bench", "--tag", tag, "--sup-obs", "frames", "--paradigm", "hardsoft",
                "--persona-v2", *args.personas, "--workers", str(args.workers),
                "--out-root", os.path.join(args.out, "frames"), "--log-dir", os.path.join(args.out, "logs")] + flags + extra
    if args.cases:
        sys.argv += ["--cases", *args.cases]
    runpy.run_module("cooksim_bench.run_assist_bench", run_name="__main__")


if __name__ == "__main__":
    main()
