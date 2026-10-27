"""Formal benchmark, second edition (project decision 2026-10-05): the 75-cell table on the no-self-repair person
(engine f124), run phase by phase, each phase verified before it is marked done.

Two lanes per benchmark phase, independent of each other:
  API lane  three polled API assistants, the two streaming APIs in their native setting, and the silent control, run HERE on local engines, API_CONCURRENT seats at a time
  GPU lane  the eleven open-weight assistants, one Beaker experiment per seat on the H100 cluster (H100), REPLICAS replicas
            sharing the 75 cells. Generation time is charged as measured on the H100 (VH_LAT_NORMALISE=0): every
            served seat runs on the same card, so there is no A100-equivalent factor in this run.
Run 1 first. Runs 2 and 3 wait for the go-ahead (`orchestrate.py approve bench-r2`).
"""
from vhhome import paths
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.abspath(os.path.join(HERE, "..", ".."))  # the checkout (this file is vhhome_bench/cluster/plan.py)

PY = os.environ.get("VHHOME_PY") or sys.executable   # the interpreter with beaker-py and the package installed
STATE_DIR = os.path.join(paths.RUNS, "formal", "state")
CELLS = paths.CELLS                # the same 75 cells as runs v1-v3
N = 75
PAGES_REPO = os.environ.get("VHHOME_PAGES_REPO", "")           # "<owner>/<repo>" of a GitHub Pages site, or empty for no page
PAGES_DIR = os.environ.get("VHHOME_PAGES_DIR") or os.path.join(paths.RUNS, "formal", "pages")
PAGE_PATH = "formal-run-2"
PAGE_URL = os.environ.get("VHHOME_PAGE_URL", "")

API_CONCURRENT = 4                                 # project decision 2026-10-05 10:25 PT: four at a time (was two)
ENGINES_PER_SEAT = 5
SLOTS = [list(range(8080 + k * ENGINES_PER_SEAT, 8080 + (k + 1) * ENGINES_PER_SEAT)) for k in range(API_CONCURRENT)]
N_ENGINES = API_CONCURRENT * ENGINES_PER_SEAT
# order: the control first (it is the floor every other number is read against, and the fastest)
# project decision 2026-10-04 20:50 PT: the two streaming APIs are evaluated in their NATIVE setting only (a live session that
# is pushed frames and the person's speech and speaks when it chooses), never the polled one. They join the lane as
# `__native` seats once that backend is ported to this engine; until then they are not in the plan at all.
NATIVE_SEATS = ["gemini-3.8-live__native", "gpt-realtime-2.1__native"]      # ported 2026-10-04 (vhhome_bench/native.py)
# project decision 2026-10-12: the two streaming APIs ALSO run in the polled setting (one call per poll, like the other API
# seats), beside their native seats, in all three runs — as Beaker CPU jobs like every other API fill.
POLLED_STREAMING_SEATS = ["gemini-3.8-live", "gpt-realtime-2.1"]
API_SEATS = ["silent", "gemini-3.8-flash", "gpt-6-astra", "gemini-3.1-pro-preview"] + NATIVE_SEATS + POLLED_STREAMING_SEATS
NEVER_SPEAKS = {"silent", "mmduet2"}

BEAKER = dict(cluster=os.environ.get("VHHOME_BEAKER_CLUSTER", ""), workspace=os.environ.get("VHHOME_BEAKER_WORKSPACE", ""), budget=os.environ.get("VHHOME_BEAKER_BUDGET", ""), replicas=5,
              env=["VH_LAT_NORMALISE=0"])
# API SEATS AS BEAKER JOBS (project decision 2026-10-06 12:00 PT: "API lanes lag behind ... submit run 3 api lanes as gpu jobs as
# well ... the A100 cluster and the CPU cluster ... you only need renderer"). The renderer is Unity on software GL, i.e. CPU only, so the
# job asks for NO GPU and no sized resources (beaker_submit_vh.py --cpu-lane, the 2026-09-21 recipe that ran bench_v1's
# API arms): it starts on whichever of the two clusters has CPU to spare and holds no card. The API's own generation
# time is what is charged to the game clock, so the node it runs on does not enter the score.
# `silent` of run 3 is not listed: it was already running on local engines when the ruling came and stays there.
# The two native seats get fewer replicas: each replica is one live streaming session, and a refused session is
# recorded as a mute episode (feed_errors in the record), never retried.
# 2026-10-08: the f126 re-run of the frozen-takeover cells (two cases, every seat, all three runs) sends every API seat's
# fill to Beaker the same way, runs 1 and 2 included; the session's engines serve only the smoke tests now.
API_BEAKER = dict(phases=["bench-r1", "bench-r2", "bench-r3"],
                  seats=API_SEATS,
                  cluster=os.environ.get("VHHOME_BEAKER_CPU_CLUSTERS", ""),
                  replicas={"default": 25, "gemini-3.8-live__native": 10, "gpt-realtime-2.1__native": 10})
PREFLIGHT_ROOT = "runs/formal/preflight"
PREFLIGHT_LOCAL = [("silent", "cx_appliance_obj_stall", "baseline"), ("silent", "cx_shutdown_skip_stall", "baseline"),
                   ("gemini-3.8-flash", "cx_appliance_obj_stall", "baseline")]


def gpu_seats():
    from vhhome_bench import roster as roster_vh
    return [e["tag"] for e in roster_vh.ROSTER if e["kind"] == "local" and e["status"] == "ready"]


def phases():
    g = gpu_seats()
    out = [dict(id="preflight", kind="preflight", title="Pre-flight: the new engine, locally and on the H100",
                note="three local episodes on the run's engine stamp (two silent, one API) and a 3-episode check job per "
                     "open-weight seat on the H100 cluster: records present, stamp right, card is an H100, no latency factor")]
    for k in (1, 2, 3):
        out.append(dict(id=f"bench-r{k}", kind="bench", run=k, root=f"tmp/bench2_r{k}", api=list(API_SEATS), gpu=list(g),
                        needs_approval=(k > 1), title=f"Benchmark, independent run {k}",
                        note=f"19 seats x 75 cells = 1,425 episodes (the two streaming APIs in both their native and polled settings); API lane local ({API_CONCURRENT} seats at a time, "
                             f"{ENGINES_PER_SEAT} engines each), GPU lane on the H100 cluster H100 ({BEAKER['replicas']} replicas per seat)"))
        out.append(dict(id=f"score-r{k}", kind="score", run=k, root=f"tmp/bench2_r{k}", needs=f"bench-r{k}",
                        title=f"Scoring of run {k}",
                        note="objective metrics (in-time success, efficiency, detection) and the interaction-quality "
                             "rubric (GPT-5.4 mini, per conversation round, five categories)"))
    return out


SCORING = [("results table", "{PY} -m vhhome_bench.scoring.results_table {root}"),
           ("objective metrics", "{PY} -m vhhome_bench.scoring.metrics_table {root} --json {root}/metrics_table.json"),
           # project decision 2026-10-12 10:30 PT: the Anthropic balance ran out with ~22k rounds unjudged by Claude; a round is
           # scored by the judges that answered (gemini + gpt where Claude is missing). The step completes those two.
           ("rubric judge", "{PY} -m vhhome_bench.scoring.rubric {root} --subset --judge gemini,gpt --workers 96"),
           ("rubric table", "{PY} -m vhhome_bench.scoring.rubric_table {root}")]
