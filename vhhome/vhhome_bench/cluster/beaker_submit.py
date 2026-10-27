#!/usr/bin/env python3
"""Submit VH benchmark lanes as Beaker batch jobs — one job, one A100, one local model.

    export BEAKER_TOKEN=...
    python -m vhhome_bench.cluster.beaker_submit check minicpmo --root tools/tmp/beaker_check
    python -m vhhome_bench.cluster.beaker_submit lane --all-local --root runs/mine --shards 3
    python -m vhhome_bench.cluster.beaker_submit lane qwen3.8-27b --root runs/mine --dry-run

Same shape as CookSim's tools/beaker_submit.py (same workspace, budget, cluster, image, shared
mounts and HOME), because the point of a job is to be indistinguishable from the interactive session:
no git clone, the job runs the checkout on the shared filesystem and writes its reports back to the shared filesystem.

Two VH differences:
  * the job also starts a VirtualHome engine (Unity, software GL on the job's CPUs) — see
    tools/beaker_job_vh.sh. The GPU stays with the model.
  * GPU JOBS ONLY (project decision 2026-09-19, carried over from ScreenSim): API and silent arms run in the
    interactive session with vhhome_bench/lane.py, never here.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)

from vhhome_bench import roster as roster_vh  # noqa: E402

WORKSPACE = os.environ.get("VHHOME_BEAKER_WORKSPACE", "")        # CPU lanes (API and silent arms) keep running here
BUDGET = os.environ.get("VHHOME_BEAKER_BUDGET", "")
# GPU LANES GO SOMEWHERE ELSE (project decision 2026-09-22). The 219 served-model lanes of runs v2 and v3 sat
# QUEUED for six hours with nothing running: it was not cluster capacity but this workspace's GPU
# allocation, already full. A GPU lane is therefore submitted to a workspace with room, on its own budget.
GPU_WORKSPACE = os.environ.get("VHHOME_BEAKER_GPU_WORKSPACE", WORKSPACE)
GPU_BUDGET = os.environ.get("VHHOME_BEAKER_GPU_BUDGET", BUDGET)
CLUSTER = os.environ.get("VHHOME_BEAKER_CLUSTER", "")                                       # 8x A100-80GB
IMAGE = os.environ.get("VHHOME_BEAKER_IMAGE", "")              # the interactive session's image
# SITE SETTINGS. The published runs used the Ai2 Beaker cluster; set these for yours. HOME is the home directory the
# job should see (the checkout, keys file and model root must be visible under it on the job's mounts); MOUNTS lists
# "<dataset or the shared filesystem name>:<mount path>" pairs, comma-separated.
HOME = os.environ.get("VHHOME_JOB_HOME") or os.path.expanduser("~")
WEKA = [tuple(x.split(":", 1)) for x in os.environ.get("VHHOME_JOB_MOUNTS", "").split(",") if ":" in x]


def spec_for(mode, tag, root, shard, priority, preemptible, retries, cpus, memory, budget,
             cluster=CLUSTER, cpu_lane=False, cells=None, replicas=None, extra_env=()):
    from beaker import (BeakerExperimentSpec, BeakerTaskSpec, BeakerImageSource, BeakerTaskContext,
                        BeakerConstraints, BeakerTaskResources, BeakerEnvVar, BeakerDataMount,
                        BeakerJobPriority, BeakerRetrySpec, BeakerResultSpec)
    # A "-" PLACEHOLDER, never an empty word: with no shard the command line collapsed its double space
    # and bash handed the CELLS file to the job as its shard, which died on int("tmp") — every
    # single-shard fill job failed that way on 2026-09-22 and left its cells unwritten.
    cmd = (f"bash {ENG}/vhhome_bench/cluster/beaker_job.sh {mode} {tag} {root} "
           f"{shard or '-'} {cells or '-'}")
    # A CPU LANE (API and silent arms) asks for NO resources at all — project decision 2026-09-21. Such a job
    # needs only the CPU its Unity engine runs on, and an unsized job does not draw on the workspace's
    # GPU allocation, so these can run on a general cluster (the CPU cluster) beside the GPU lanes on the A100 cluster.
    res = (BeakerTaskResources(shared_memory="4GiB") if cpu_lane else
           BeakerTaskResources(gpu_count=1, cpu_count=cpus, memory=memory, shared_memory="16GiB"))
    env = [BeakerEnvVar(name="HOME", value=HOME),
           # Unity's software GL gets one slot's worth of CPU, not the whole node: a job that asks for
           # more than its slot waits for CPU that a busy node never frees.
           BeakerEnvVar(name="UNITY_JOB_WORKERS", value="8")]
    for kv in extra_env:                      # e.g. STEPIN_SELF_PLAN=0 for the clean-plan reference run
        k, _, v = kv.partition("=")
        env.append(BeakerEnvVar(name=k, value=v))
    task = BeakerTaskSpec(
        name=f"{mode}-{tag}"[:128],
        image=BeakerImageSource(beaker=IMAGE),
        command=["bash", "-c", cmd],
        env_vars=env,
        datasets=[BeakerDataMount.new(mount_path=p, weka=w) for w, p in WEKA],
        resources=res,
        constraints=BeakerConstraints(cluster=(cluster if isinstance(cluster, list) else [cluster])),
        context=BeakerTaskContext(priority=BeakerJobPriority[priority], preemptible=preemptible),
        result=BeakerResultSpec(path="/results"),
        # REPLICAS, not n experiments (project decision 2026-09-22): an arm is ONE row in the Beaker UI whose
        # replicas each resolve their own shard from BEAKER_REPLICA_RANK (see tools/beaker_job_vh.sh).
        # They are independent workers, so nothing is synchronised and no replica's death touches the
        # others: a preempted replica retries alone and resumes from the cells its shard is still missing.
        **({"replicas": replicas, "leader_selection": False, "propagate_failure": False,
            "propagate_preemption": False} if replicas and replicas > 1 else {}),
    )
    return BeakerExperimentSpec(description=f"VHSim {mode}: {tag} -> {root}", tasks=[task], budget=budget,
                                retry=BeakerRetrySpec(allowed_task_retries=retries) if retries else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "lane", "calib"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--all-local", action="store_true")
    ap.add_argument("--root", required=True, help="output root, relative to the engine")
    ap.add_argument("--shards", type=int, default=1, help="split each lane into n shards")
    ap.add_argument("--replicas", action="store_true",
                    help="one experiment with --shards replicas instead of one experiment per shard")
    ap.add_argument("--suffix", default="", help="extra text in the job name (e.g. the GPU kind)")
    ap.add_argument("--workspace", default=None, help="default: the GPU workspace for served models")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the job, repeatable")
    ap.add_argument("--priority", default="high", choices=["low", "normal", "high", "urgent"])
    ap.add_argument("--not-preemptible", action="store_true")
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--cpus", type=float, default=10)
    ap.add_argument("--memory", default="100GiB")   # a Saturn slot is ~116 GiB; more silently takes a 2nd GPU
    ap.add_argument("--budget", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cluster", default=CLUSTER,
                    help="one cluster, or several comma-separated: the scheduler takes whichever has room "
                         "first. GPU lanes want >=80GiB cards (the A100 cluster A100, the H100 clusters H100, another cluster "
                         "A100); the A100 cluster alone left 219 jobs queued for six hours on 2026-09-22.")
    ap.add_argument("--cells", default=None, help="a cells file for the job (e.g. only the missing cells)")
    ap.add_argument("--cpu-lane", action="store_true",
                    help="API/silent arm: ask for no resources, run on a general cluster")
    a = ap.parse_args()
    clusters = [c.strip() for c in a.cluster.split(",") if c.strip()]

    tags = ([e["tag"] for e in roster_vh.ROSTER if e["kind"] == "local" and e["status"] == "ready"]
            if a.all_local else a.tags)
    if not tags:
        sys.exit("no tags")
    for t in tags:
        e = roster_vh.by_tag(t)
        if e["kind"] == "local" and a.cpu_lane:
            sys.exit(f"{t} is a served model: it needs its GPU, so it cannot run as a CPU lane")
        if e["kind"] != "local" and not a.cpu_lane:
            sys.exit(f"{t} is not a local model: run it in the session, or as a CPU lane "
                     f"(--cpu-lane --cluster <cpu cluster>)")
    # A served model needs a GPU, so it goes to the GPU workspace and budget unless told otherwise; an
    # API or silent arm asks for no resources and stays where it already runs.
    gpu_side = any(roster_vh.by_tag(t)["kind"] == "local" for t in tags) and not a.cpu_lane
    workspace = a.workspace or (GPU_WORKSPACE if gpu_side else WORKSPACE)
    budget = a.budget or (GPU_BUDGET if gpu_side else BUDGET)
    print(f"[submit] {len(tags)} tag(s) -> workspace={workspace} budget={budget} "
          f"cluster={','.join(clusters)}"
          + (f" replicas={a.shards}" if a.replicas else f" shards={a.shards}"), flush=True)
    if not a.dry_run:
        from beaker import Beaker
        b = Beaker.from_env()
        ws = b.workspace.get(workspace)
    created, failed = 0, []
    stamp = time.strftime("%m%d-%H%M%S")     # seconds too: two submissions in the same minute collide
    n = a.shards or 1
    if n <= 1 or a.mode in ("check", "calib"):
        shards = [None]
    elif a.replicas:
        shards = [f"auto/{n}"]          # ONE experiment; each replica resolves its own k from its rank
    else:
        shards = [f"{k}/{n}" for k in range(1, n + 1)]
    for t in tags:
        for sh in shards:
            spec = spec_for(a.mode, t, a.root, sh, a.priority, not a.not_preemptible, a.retries,
                            a.cpus, a.memory, budget, cluster=clusters, cpu_lane=a.cpu_lane,
                            cells=a.cells, replicas=(n if a.replicas else None), extra_env=a.env)
            suffix = (f"-{a.suffix}" if a.suffix else "") + \
                     ("" if not sh else (f"-x{n}" if a.replicas else f"-s{sh.split('/')[0]}"))
            name = f"vhsim-{a.mode}-{t}{suffix}-{stamp}".replace(".", "-")[:120]
            if a.dry_run:
                print(name, "->", f"bash tools/beaker_job_vh.sh {a.mode} {t} {a.root} {sh or ''}".strip())
                continue
            # A CONFLICT IS RETRIED UNDER A NEW NAME, never shrugged off. Beaker answers 409 for reasons
            # that have nothing to do with the name (seen on fresh unique names, 2026-09-23), and the old
            # code printed a line and continued — so a whole refill reported "fill jobs submitted" while
            # creating NOTHING, and qwen3.8-27b sat 44 cells short for a day (2026-09-24).
            wl = None
            for attempt in range(4):
                try:
                    nm = name if attempt == 0 else f"{name}-r{attempt}"
                    wl = b.experiment.create(spec=spec, name=nm, workspace=ws)
                    print(f"submitted {nm}: https://beaker.org/ex/{wl.experiment.id}", flush=True)
                    created += 1
                    break
                except Exception as ex:
                    if "Conflict" in type(ex).__name__ or "409" in str(ex):
                        print(f"conflict on {nm} (attempt {attempt + 1}/4) — retrying under a new name",
                              flush=True)
                        time.sleep(2 + 3 * attempt)
                        continue
                    raise
            if wl is None:
                failed.append(name)


    if a.dry_run:
        return
    print(f"[submit] {created} job(s) created" + (f", {len(failed)} FAILED: {', '.join(failed)}"
                                                 if failed else ""), flush=True)
    if failed or not created:
        sys.exit(7)              # a caller must not read "submitted" from a submission that created none


if __name__ == "__main__":
    main()
