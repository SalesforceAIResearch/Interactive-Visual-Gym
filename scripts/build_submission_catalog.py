"""Freeze membership from executable suite definitions; no model or judge calls.

Run from the repository: PYTHONPATH=screensim python scripts/build_submission_catalog.py
The ScreenSim executable task designer is intentionally used instead of an old paper table.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    from screensim.tasks import HARD_TASKS, by_id
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.personas import bench_personas_for

    cook = json.loads((ROOT / "cooksim/benchmark/cases_composite_v5.json").read_text())
    vh = json.loads((ROOT / "vhhome/benchmark/cells.json").read_text())
    cells = {
        "cooksim": [
            {"case": c["name"], "persona": p}
            for c in cook["cases"]
            for p in ("baseline", "classic_novice", "classic_expert")
        ],
        "vhhome": [{"case": c["case"], "persona": c["persona"]} for c in vh["cells"]],
        "screensim": [
            {"case": e.id, "persona": p}
            for t in HARD_TASKS
            for e in design_composite_episodes(by_id(t), per_task=3)[1]
            for p in bench_personas_for(e.id)
        ],
    }
    for engine, n in (("cooksim", 150), ("vhhome", 75), ("screensim", 90)):
        assert (
            len(cells[engine])
            == len({(r["case"], r["persona"]) for r in cells[engine]})
            == n
        )
        cells[engine].sort(key=lambda r: (r["case"], r["persona"]))
    # Freeze engine/harness/scoring sources and suite assets, including all user prompts.
    paths = set()
    for engine in cells:
        for package in (
            engine,
            "cooksim_bench" if engine == "cooksim" else engine + "_bench",
        ):
            paths.update(
                str(p.relative_to(ROOT))
                for p in (ROOT / engine / package).rglob("*")
                if p.is_file() and p.suffix in (".py", ".json", ".jsonl", ".txt")
            )
        paths.update(
            str(p.relative_to(ROOT))
            for p in (ROOT / engine / "benchmark").rglob("*")
            if p.is_file() and p.suffix in (".json", ".jsonl")
        )
    paths = sorted(paths)
    result = {
        "version": "ivg-bench-submissions-v4",
        "runs": [1, 2, 3],
        "engines": cells,
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
        },
        "settings": {
            "cooksim": {
                "delay": "fixed-2-ticks",
                "interface": "pulled",
                "human_model": "gemini-3.7-flash",
            },
            "vhhome": {
                "delay": "fixed-2-ticks",
                "interface": "pulled",
                "human_model": "gemini-3.1-pro-preview",
            },
            "screensim": {
                "delay": "interaction-steps",
                "interface": "pulled",
                "human_model": "gemini-3.7-flash",
            },
        },
        "judges": {"gemini": "gemini-3.7-flash", "gpt": "gpt-5.4-mini"},
    }
    return result


if __name__ == "__main__":
    result = build()
    target = ROOT / "ivg_submission/catalog.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {target.name}: 315 case-persona pairs, three runs (945 episodes).")
