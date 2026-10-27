#!/usr/bin/env python3
"""Build the work list for ONE Beaker experiment whose replicas cover every remaining run.

    python -m cooksim_bench.cluster.make_replica_manifest --out tmp/replica_manifest.json

Each replica of that experiment reads its own line of this manifest by BEAKER_REPLICA_RANK, so a single
row in the Beaker UI covers every seat, shard and ablation. Item fields:
    kind      bench (an assistant seat on the suite) | cookabl (a cook-model seat on the 30-cell draw)
    tag       roster tag of the seat
    root      output root, relative to the engine
    repeat    independent repeat index (bench: 1,2,3 -> <tag>__r<k>)
    shard     1-based slice of that seat's episode list, of nshards
    env       extra environment for this item (e.g. the zero-latency ablation)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

from cooksim_bench.roster import ROSTER                                    # noqa: E402

API_SEATS = ["gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.8-live",
             "gemini-3.8-live__native", "gpt-realtime-2.1", "gpt-realtime-2.1__native"]
COOKS = ["gpt-6-astra", "gpt-realtime-2.1", "qwen3.8-27b", "museglimmer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/replica_manifest.json")
    ap.add_argument("--nolat-root", default="tmp/formal_v5_nolat")
    ap.add_argument("--cook-root", default="tmp/cook_model_ablation")
    ap.add_argument("--local-shards", type=int, default=3)
    ap.add_argument("--api-shards", type=int, default=2)
    ap.add_argument("--cook-shards", type=int, default=3)
    a = ap.parse_args()
    local = [e["tag"] for e in ROSTER if e["kind"] == "local" and e["status"] == "ready"]
    items = []
    for tag in local:                       # the zero-latency round: every seat, one repeat
        for s in range(1, a.local_shards + 1):
            items.append(dict(kind="bench", tag=tag, root=a.nolat_root, repeat=1,
                              shard=s, nshards=a.local_shards, env={"COOKSIM_ASSIST_LATENCY": "off"}))
    for tag in API_SEATS:
        for s in range(1, a.api_shards + 1):
            items.append(dict(kind="bench", tag=tag, root=a.nolat_root, repeat=1,
                              shard=s, nshards=a.api_shards, env={"COOKSIM_ASSIST_LATENCY": "off"}))
    for tag in COOKS:                       # the cook-model ablation: the same 30 cells per cook
        for s in range(1, a.cook_shards + 1):
            items.append(dict(kind="cookabl", tag=tag, root=a.cook_root, repeat=1,
                              shard=s, nshards=a.cook_shards, env={}))
    out = os.path.join(os.path.dirname(HERE), a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"items": items}, open(out, "w"), indent=1)
    print(f"{len(items)} replicas -> {a.out}")
    for k in ("bench", "cookabl"):
        print(f"   {k}: {sum(1 for i in items if i['kind'] == k)}")


if __name__ == "__main__":
    main()
