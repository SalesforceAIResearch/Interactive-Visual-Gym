#!/usr/bin/env python3
"""v5 sample run (2026-09-09, user-authorised): the same 10 (persona, case) episodes for each of
five API assistants = 50 episodes under scheduler v5 on the curated suite, frames setting,
tight pace, latency charged. Output layout mirrors the sweeps so the scorers work unchanged:
    tmp/assist_bench_v5_sample/frames/<persona>/<tag>/<case>.json
"""
import json, os, subprocess, sys, time
import os
from cooksim_bench.paths import ROOT  # the CookSim release root
OUT = os.path.join(ROOT, os.environ.get("V5_SAMPLE_ROOT", "tmp/assist_bench_v5_sample"), "frames")
LOGS = "/tmp/assist_bench_logs"
TAG = os.environ.get("V5_SAMPLE_TAG", "v5s")             # log-file prefix, one per sample round
CASES = "tmp/interact_eval/cases_composite_v4.json"
# one case per cell, personas rotated so every persona meets every tier
EPISODES_R4 = [("baseline", "b3_easy_nops_medium_map_2"), ("classic_novice", "b3_easy_nops_hard_map_3"),
               ("classic_expert", "b3_medium_nops_medium_map_1"), ("baseline", "b3_medium_nops_hard_map_2"),
               ("classic_novice", "b3_hard_nops_medium_map_3"), ("classic_expert", "b3_hard_nops_hard_map_4"),
               ("baseline", "b3_expert_nops_medium_map_1"), ("classic_novice", "b3_expert_nops_hard_map_2"),
               ("classic_expert", "b3_master_nops_medium_map_3"), ("baseline", "b3_master_nops_hard_map_3")]

# Round 5 (2026-09-16, user ruling "choose another ten episodes"): a FRESH draw for the re-run on the
# no-self-repair cook prompt. Same 10 cells (5 recipe tiers x medium/hard map), a different map instance
# in every cell, and the persona rotation shifted one step, so no (persona, case) pair repeats round 4.
# Composition parity with R4 (R4 in brackets): 30 errors (30), 20 qa_probe (20), 6 qa_proactive (7),
# 2 plan_change (1), 22 ruinous (20); no two cases share an error set, and every error family appears.
EPISODES_R5 = [("classic_novice", "b3_easy_nops_medium_map_3"), ("classic_expert", "b3_easy_nops_hard_map_4"),
               ("baseline", "b3_medium_nops_medium_map_2"), ("classic_novice", "b3_medium_nops_hard_map_3"),
               ("classic_expert", "b3_hard_nops_medium_map_4"), ("baseline", "b3_hard_nops_hard_map_1"),
               ("classic_novice", "b3_expert_nops_medium_map_2"), ("classic_expert", "b3_expert_nops_hard_map_3"),
               ("baseline", "b3_master_nops_medium_map_4"), ("classic_novice", "b3_master_nops_hard_map_4")]

# V5_EPISODE_SET picks the draw; default stays r4 so every earlier run reproduces unchanged.
EPISODES = EPISODES_R5 if os.environ.get("V5_EPISODE_SET", "r4").lower() == "r5" else EPISODES_R4
ASSISTANTS = [("gemini-3.1-pro-preview", ["--assist-backend", "gemini", "--assist-model", "gemini-3.1-pro-preview"]),
              ("gemini-3.7-flash", ["--assist-backend", "gemini", "--assist-model", "gemini-3.7-flash"]),
              ("gpt-5.5", ["--assist-backend", "openai", "--assist-model", "gpt-5.5", "--assist-base-url", "https://api.openai.com/v1", "--assist-profile", "gpt5"]),
              ("gpt-5.4-mini", ["--assist-backend", "openai", "--assist-model", "gpt-5.4-mini", "--assist-base-url", "https://api.openai.com/v1", "--assist-profile", "gpt5"]),
              ("gpt-realtime-2.1", ["--assist-backend", "realtime", "--assist-model", "gpt-realtime-2.1"])]
WORKERS = int(os.environ.get("V5_SAMPLE_WORKERS", 25))

def main():
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ)
    env.update(COOKSIM_SCHED="v5", COOKSIM_CASES=CASES, COOKSIM_PACE="tight", COOKSIM_ASSIST_LATENCY="measured",
               COOKSIM_NO_VIDEO="0", COOKSIM_DUMP_PER_EPISODE="1", GEMINI_MODEL="gemini-3.7-flash")
    jobs = []
    for tag, extra in ASSISTANTS:
        for persona, case in EPISODES:
            outd = os.path.join(OUT, persona, tag); os.makedirs(outd, exist_ok=True)
            out = os.path.join(outd, case + ".json")
            if os.path.exists(out):
                continue
            jobs.append((tag, extra, persona, case, out))
    print(f"{len(jobs)} episodes to run", flush=True)
    running = []
    while jobs or running:
        running = [(j, p) for j, p in running if p.poll() is None]
        while jobs and len(running) < WORKERS:
            tag, extra, persona, case, out = jobs.pop(0)
            e = dict(env); e["COOKSIM_DUMP_STATES"] = out.replace(".json", ".states.jsonl")
            log = open(os.path.join(LOGS, f"{TAG}_{tag}__{persona}__{case}.log"), "w")
            cmd = [sys.executable, os.path.join(ROOT, "python -m cooksim_bench.e2e_stepin_rollout"), "--case", case, "--persona-v2", persona,
                   "--paradigm", "hardsoft", "--sup-obs", "frames", "--out", out] + extra
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=e)
            running.append(((tag, persona, case), p))
            print(f"launched {tag}/{persona}/{case} (pid {p.pid}); {len(jobs)} queued", flush=True)
            time.sleep(4)
        time.sleep(10)
    print("sample run ended", flush=True)

if __name__ == "__main__":
    main()
