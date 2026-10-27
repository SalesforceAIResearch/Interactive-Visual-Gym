#!/usr/bin/env python3
"""Submit ScreenSim roster assistants as Beaker batch jobs -- one job, one A100, one model.
Same workspace, budget, cluster, image, mounts and HOME as CookSim's submitter
(cook-bench-engine/screensim_bench/cluster/beaker_submit.py, docs/BEAKER_JOBS.md), so the models run on the setup CookSim audited.

    export BEAKER_TOKEN=$(cat ~/.beaker_token_cooksim)
    python3 screensim_bench/cluster/beaker_submit.py check --all-local --out tmp/beaker_check          # 2 cases per model
    python3 screensim_bench/cluster/beaker_submit.py lane  qwen3.8-27b --out tmp/bench_v1                 # all 90 cases
    python3 screensim_bench/cluster/beaker_submit.py lane  minicpmo --out tmp/bench_v1 --dry-run

Needs the system Python's beaker-py (python3) and a Beaker token in BEAKER_TOKEN.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)
from screensim.interact.roster import ROSTER, by_tag      # noqa: E402

WORKSPACE = "ai2/trajectory_video_encoder"
BUDGET = "ai2/oe-omai"
CLUSTER = "ai2/saturn"               # A100-80GB
CLUSTER_48G = "ai2/neptune"          # A40-48GB: enough for every served model except the two below
BIG_GPU = {"qwen3.8-27b", "qwen3-omni-30b-a3b"}      # bf16 weights ~54 GB and ~60 GB
IMAGE = "sanghol/molmo3-torch2.9.1-cuda128-fa4"
HOME = "/weka/oe-training-default/jasonr/chenhao_prior/home"
WEKA = [("oe-training-default", "/weka/oe-training-default"), ("oe-mm-default", "/weka/oe-mm")]


def spec_for(mode, tag, out, ncases, priority, preemptible, retries, cpus, memory, gpus, budget=BUDGET, shard="",
             manual="manual", cluster=CLUSTER, cells=""):
    from beaker import (BeakerExperimentSpec, BeakerTaskSpec, BeakerImageSource, BeakerTaskContext,
                        BeakerConstraints, BeakerTaskResources, BeakerEnvVar, BeakerDataMount, BeakerJobPriority,
                        BeakerRetrySpec, BeakerResultSpec)
    cmd = f"bash {ENG}/tools/beaker_job.sh {mode} {tag} {out} {ncases} {shard or 0} {manual} {cells or 0}"
    task = BeakerTaskSpec(
        name=f"{mode}-{tag}{('-s' + shard.replace('/', 'of')) if shard else ''}"[:128],
        image=BeakerImageSource(beaker=IMAGE),
        command=["bash", "-c", cmd],
        env_vars=[BeakerEnvVar(name="HOME", value=HOME)],
        datasets=[BeakerDataMount.new(mount_path=p, weka=w) for w, p in WEKA],
        resources=BeakerTaskResources(gpu_count=gpus, cpu_count=cpus, memory=memory, shared_memory="16GiB"),
        constraints=BeakerConstraints(cluster=[cluster]),
        context=BeakerTaskContext(priority=BeakerJobPriority[priority], preemptible=preemptible),
        result=BeakerResultSpec(path="/results"),
    )
    return BeakerExperimentSpec(description=f"ScreenSim {mode}: {tag} -> {out}", tasks=[task], budget=budget,
                                retry=BeakerRetrySpec(allowed_task_retries=retries) if retries else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "lane"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--all-local", action="store_true")
    ap.add_argument("--all", action="store_true", help="same as --all-local: Beaker runs GPU jobs only")
    ap.add_argument("--shards", type=int, default=1, help="lane mode: split each LOCAL model's cases over n jobs")
    ap.add_argument("--no-manual", action="store_true", help="the assistant gets no User Guide text (ablation arm)")
    ap.add_argument("--cluster", default="", help="override the cluster; default per model (A40 where the weights fit)")
    ap.add_argument("--cells", default="", help="json file of cells to re-run (engine fix, lost job)")
    ap.add_argument("--out", required=True, help="output dir, relative to the engine")
    ap.add_argument("--cases", type=int, default=2, help="check mode: how many cases")
    ap.add_argument("--priority", default="high", choices=["low", "normal", "high", "urgent"])
    ap.add_argument("--not-preemptible", action="store_true")
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--cpus", type=float, default=24)
    ap.add_argument("--memory", default="110GiB")
    ap.add_argument("--budget", default=BUDGET)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tags = [e["tag"] for e in ROSTER if e["kind"] == "local"] if (a.all or a.all_local) else a.tags
    # Beaker is for GPU work only (user ruling 2026-09-19): API lanes and the floor/ceiling run in the
    # interactive session with screensim_bench/runs/sweep.py
    cpu = [t for t in tags if t == "floor-ceiling" or by_tag(t)["kind"] != "local"]
    if cpu:
        sys.exit(f"not submitting CPU jobs {cpu}: run API lanes and floor-ceiling in the session")
    if not tags:
        sys.exit("no tags")
    if not a.dry_run:
        from beaker import Beaker
        b = Beaker.from_env()
        ws = b.workspace.get(WORKSPACE)
    stamp = time.strftime("%m%d-%H%M")
    for t in tags:
        local = t != "floor-ceiling" and by_tag(t)["kind"] == "local"
        gpus = 1 if local else 0                         # API lanes and floor/ceiling need no GPU
        n = a.shards if (local and a.mode == "lane") else 1
        for i in range(1, n + 1):
            shard = f"{i}/{n}" if n > 1 else ""
            spec = spec_for(a.mode, t, a.out, a.cases, a.priority, not a.not_preemptible, a.retries,
                            a.cpus if gpus else 8, a.memory if gpus else "32GiB", gpus, a.budget, shard,
                            "nomanual" if a.no_manual else "manual",
                            a.cluster or (CLUSTER if t in BIG_GPU else CLUSTER_48G), a.cells)
            run = os.path.basename(a.out.rstrip("/"))          # repeats differ only by --out: keep names apart
            name = (f"screensim-{a.mode}-{run}-{t}" + (f"-s{i}of{n}" if n > 1 else "")
                    + f"-{stamp}").replace(".", "-")[:120]
            if a.dry_run:
                print(name, "|", spec.tasks[0].command[-1], "| gpus", gpus); continue
            wl = b.experiment.create(spec=spec, name=name, workspace=ws)
            print(f"submitted {name}: https://beaker.org/ex/{wl.experiment.id}", flush=True)


if __name__ == "__main__":
    main()
