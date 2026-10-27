"""Run one episode and read the record back.

    OPENAI_API_KEY=... GEMINI_API_KEY=... python examples/01_one_episode.py
"""
import json
import subprocess
import sys

OUT = "runs/example"
subprocess.run([sys.executable, "-m", "screensim_bench.cli.episode",
                "--task", "hearing_setup", "--persona", "classic_novice",
                "--assistant", "openai:gpt-5.4-mini", "--out", OUT], check=True)

with open(f"{OUT}/composite_results_v2.records.jsonl") as fh:
    rec = next(json.loads(line) for line in fh if line.strip())
print(rec.get("episode"), rec.get("success"), rec.get("final_tick"))
