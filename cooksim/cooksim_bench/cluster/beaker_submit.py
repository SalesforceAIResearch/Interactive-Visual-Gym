#!/usr/bin/env python3
"""Submit roster assistants as Beaker batch jobs -- one job, one A100, one model.

    export BEAKER_TOKEN=$(cat ~/.beaker_token_cooksim)
    python3 python -m cooksim_bench.cluster.beaker_submit check glm-4.6v-flash qwen3-omni-30b-a3b --root tmp/beaker_check
    python3 python -m cooksim_bench.cluster.beaker_submit lane  --all-local --root tmp/round6 --repeats 3
    python3 python -m cooksim_bench.cluster.beaker_submit lane  minicpmo --dry-run          # print the spec only

Every job mirrors the interactive session it is launched from, so results stay comparable with local runs:
the same image, the same weka mounts (the engine, the envs, the weights and the outputs are all on
oe-training-default), the same HOME, one A100-80GB on ai2/saturn. There is no git clone: the job cds into
the engine checkout on weka (cooksim_bench/cluster/beaker_job.sh). A lane can be resumed -- existing reports are skipped --
so jobs are preemptible (they run on unallocated capacity) with automatic retries.

Needs the system Python's beaker-py (python3, not the cooksim env) and a Beaker token in BEAKER_TOKEN.
"""
import argparse
import re
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
from cooksim_bench.paths import ROOT as ENG
from cooksim_bench.roster import ROSTER, by_tag      # noqa: E402
try:                                   # the cook roster lives on the ablation branch only
    from cooksim_bench.roster import cook_by_tag     # noqa: E402
except ImportError:
    cook_by_tag = None

WORKSPACE = "ai2/trajectory_video_encoder"
BUDGET = "ai2/oe-omai"                 # the workspace's own default budget, made explicit
CLUSTER = "ai2/saturn"                  # 8x A100-80GB nodes, the GPU every local number was measured on
IMAGE = "sanghol/molmo3-torch2.9.1-cuda128-fa4"   # the image of the interactive session
HOME = "/weka/oe-training-default/jasonr/chenhao_prior/home"
WEKA = [("oe-training-default", "/weka/oe-training-default"), ("oe-mm-default", "/weka/oe-mm")]


def spec_for(mode, tag, root, repeats, epset, priority, preemptible, retries, cpus, memory, budget=BUDGET,
             cluster=CLUSTER, extra_env=(), replicas=0, gpus=1):
    from beaker import (BeakerExperimentSpec, BeakerTaskSpec, BeakerImageSource, BeakerTaskContext,
                        BeakerConstraints, BeakerTaskResources, BeakerEnvVar, BeakerDataMount, BeakerJobPriority,
                        BeakerRetrySpec, BeakerResultSpec)
    cmd = f"bash {ENG}/cooksim_bench/cluster/beaker_job.sh {mode} {tag} {root} {repeats} {epset}"
    task = BeakerTaskSpec(
        # Beaker task names allow only letters, numbers and -_. : a manifest "tag" is a path
        name=re.sub(r"[^A-Za-z0-9._-]", "-", f"{mode}-{os.path.basename(tag)}")[:128].lstrip("-"),
        image=BeakerImageSource(beaker=IMAGE),
        command=["bash", "-c", cmd],
        env_vars=[BeakerEnvVar(name="HOME", value=HOME)]
                 + [BeakerEnvVar(name=k, value=v) for k, v in (e.split("=", 1) for e in extra_env)],
        datasets=[BeakerDataMount.new(mount_path=p, weka=w) for w, p in WEKA],
        replicas=replicas or None,
        resources=BeakerTaskResources(gpu_count=gpus, cpu_count=cpus, memory=memory, shared_memory="16GiB"),
        constraints=BeakerConstraints(cluster=[cluster]),
        context=BeakerTaskContext(priority=BeakerJobPriority[priority], preemptible=preemptible),
        result=BeakerResultSpec(path="/results"),
    )
    return BeakerExperimentSpec(description=f"CookSim {mode}: {tag} -> {root}", tasks=[task], budget=budget,
                                retry=BeakerRetrySpec(allowed_task_retries=retries) if retries else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "lane", "bench", "cookabl", "manifest"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--all-local", action="store_true", help="every local roster entry that is ready")
    ap.add_argument("--root", required=True, help="output root, relative to the engine")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeat", type=int, nargs="+", default=[1],
                    help="bench mode: which independent repeat(s) to run. Repeat k>1 writes <tag>__r<k>, "
                         "so the three runs are independent samples of the same seat")
    ap.add_argument("--name", default=None,
                    help="experiment name (manifest mode); default is derived from the mode and tag")
    ap.add_argument("--replicas", type=int, default=0,
                    help="run the seat's shards as REPLICAS of one experiment (one row in the Beaker UI) "
                         "instead of separate jobs; each replica takes its slice from BEAKER_REPLICA_RANK")
    ap.add_argument("--shards", type=int, default=1,
                    help="bench mode: split one seat's 150 episodes over this many jobs. Each job owns a "
                         "fixed slice of the (persona, case) list, so shards never race and a preempted "
                         "job resumes its own slice")
    ap.add_argument("--api", action="store_true",
                    help="bench mode: allow API/native seats (they need no model server, only the renderer)")
    ap.add_argument("--episode-set", default="r4", choices=["r4", "r5"])
    ap.add_argument("--priority", default="high", choices=["low", "normal", "high", "urgent"])
    ap.add_argument("--not-preemptible", action="store_true")
    ap.add_argument("--retries", type=int, default=10)   # preemptible jobs do get preempted; a retry resumes the lane
    ap.add_argument("--cpus", type=float, default=24)
    ap.add_argument("--memory", default="110GiB")   # one Saturn slot is ~128 GiB; more takes a second GPU
    ap.add_argument("--budget", default=BUDGET)
    ap.add_argument("--workspace", default=WORKSPACE, help="e.g. ai2/vsg")
    ap.add_argument("--cluster", default=CLUSTER, help="e.g. ai2/neptune")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs per job; 0 for API seats (ruling 2026-10-09: API seats may run on Beaker, not on jupiter)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra environment for the job, repeatable (e.g. --env COOKSIM_ASSIST_LATENCY=off)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tags = [e["tag"] for e in ROSTER if e["kind"] == "local" and e["status"] == "ready"] if a.all_local else a.tags
    if not tags:
        sys.exit("no tags")
    if a.mode == "manifest":
        # ONE experiment, one replica per manifest item: the replica reads its own line by BEAKER_REPLICA_RANK
        import json as _json
        n_items = len(_json.load(open(os.path.join(ENG, tags[0])))["items"])
        a.replicas = a.replicas or n_items
        if a.replicas != n_items:
            sys.exit(f"manifest has {n_items} items but --replicas is {a.replicas}")
        print(f"manifest {tags[0]}: {n_items} items -> one experiment with {n_items} replicas")
    for t in (tags if a.mode != "manifest" else []):       # manifest mode: the "tag" is the manifest path
        if a.mode == "cookabl":
            if cook_by_tag is None:
                sys.exit("cookabl needs the cook roster (branch cook-model-ablation)")
            cook_by_tag(t)          # the COOK roster; API cooks are fine, they just serve nothing
            continue
        e = by_tag(t)
        if e["kind"] != "local" and not (a.mode == "bench" and a.api):
            sys.exit(f"{t} is not a local model; API lanes do not need a GPU job (bench mode: pass --api)")
    if not a.dry_run:
        # the token lives outside every repo (never committed); read it here so the submit command needs
        # no shell prelude -- that also keeps it matchable by a narrow Bash permission rule
        tok = os.path.expanduser("~/.beaker_token_cooksim")
        if not os.environ.get("BEAKER_TOKEN") and os.path.exists(tok):
            os.environ["BEAKER_TOKEN"] = open(tok).read().strip()
        from beaker import Beaker
        Beaker.TIMEOUT = 120.0          # a many-replica spec takes well over the 5 s default to create
        b = Beaker.from_env()
        ws = b.workspace.get(a.workspace)
        # 2026-10-12: Beaker now requires the budget of the workspace's group ("provided budget must match the workspace
        # budget <id>"); take it from the workspace itself rather than the hard-coded name
        if getattr(ws, "budget_id", None):
            a.budget = ws.budget_id
    stamp = time.strftime("%m%d-%H%M")
    reps = a.repeat if a.mode in ("bench", "cookabl") else [1]
    shardlike = a.mode in ("bench", "cookabl", "manifest")
    if a.replicas:
        a.shards = 1                     # the replicas ARE the shards; one experiment covers the seat
    for t in tags:
        for rep in reps:
            for sh in range(1, a.shards + 1):
                # bench mode reuses the two trailing positional slots as (repeat, shard/nshards)
                p4 = rep if shardlike else a.repeats
                p5 = (f"auto/{a.replicas}" if a.replicas
                      else (f"{sh}/{a.shards}" if shardlike else a.episode_set))
                spec = spec_for(a.mode, t, a.root, p4, p5, a.priority, not a.not_preemptible,
                                a.retries, a.cpus, a.memory, a.budget, a.cluster, a.env, a.replicas, a.gpus)
                nm = f"cooksim-{a.mode}-{t}-r{rep}"
                if a.shards > 1:
                    nm += f"-s{sh}"
                name = (a.name or f"{nm}-{stamp}").replace(".", "-")[:120]
                if a.dry_run:
                    print(name); print(spec.to_json() if hasattr(spec, "to_json") else spec); continue
                try:
                    wl = b.experiment.create(spec=spec, name=name, workspace=ws)
                except Exception as ex:                       # a name that already exists must not abort the batch
                    if type(ex).__name__ == "BeakerExperimentConflict":
                        # 409 is also what the server returns for a spec it will not take -- an experiment
                        # with too many replicas, for instance (44 was refused, 24 accepted, 2026-09-22)
                        print(f"skipped {name}: server returned conflict (name in use, or the spec was "
                              f"refused -- try fewer replicas: this one had {a.replicas or 1})", flush=True)
                        continue
                    raise
                print(f"submitted {name}: https://beaker.org/ex/{wl.experiment.id}", flush=True)


if __name__ == "__main__":
    main()
