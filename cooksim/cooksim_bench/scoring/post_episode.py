"""Post-rollout scoring through installed modules; failures must reach the CLI."""
import subprocess
import sys


def score_record(path, judge=False):
    modules = ["eval_assistant"]
    if judge:
        modules += ["eval_judge", "eval_assistant"]
    for name in modules:
        subprocess.run([sys.executable, "-m", f"cooksim_bench.scoring.{name}", str(path), "--write"], check=True)
