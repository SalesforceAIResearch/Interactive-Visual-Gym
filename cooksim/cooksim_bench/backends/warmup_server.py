#!/usr/bin/env python3
"""Warm a freshly started local assistant server before any timed episode.

    python -m cooksim_bench.backends.warmup_server <roster tag> [--n 4]

A new server pays one-time costs on its first requests -- the reply schema compiled into a grammar, CUDA
kernels selected or JIT-compiled, weights and processor files paged in from network disk. Those costs must
not land in the first episode's measured latency, and they must be paid identically wherever the model
runs (an interactive session or a Beaker job), so every driver calls this between "server is up" and the
first episode: python -m cooksim_bench.cluster.run_roster_check, python -m cooksim_bench.cluster.run_lane and cooksim_bench/beaker_job.sh.

It replays the first --n genuine assistant calls (bench/assistant_calls_24.jsonl.gz) through the benchmark's
own request code with the model's own roster flags (same profile, schema, image scale), untimed as far as the
benchmark is concerned, and prints each call's latency so the log shows the server settling. It never calls
/tokenize (on vLLM 0.19.1 that corrupts the image cache).
"""
import argparse
import base64
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

from cooksim_bench.roster import by_tag      # noqa: E402

CALLS = os.path.join(os.path.dirname(HERE), "bench", "assistant_calls_24.jsonl.gz")


def flag(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    e = by_tag(a.tag)
    if e["kind"] != "local":
        print(f"[warmup] {a.tag} is not a local model; nothing to warm")
        return
    from cooksim_bench.backends import controller_eval as CE
    f = e["flags"]
    be = CE.OpenAIBackend(flag(f, "--assist-model"), flag(f, "--assist-base-url"),
                          profile=flag(f, "--assist-profile", "default"),
                          max_tokens=int(flag(f, "--assist-max-tokens", 4000)),
                          image_scale=float(flag(f, "--assist-image-scale", 1.0)))
    calls = []
    with gzip.open(CALLS, "rt") as fh:
        for line in fh:
            calls.append(json.loads(line))
            if len(calls) >= a.n:
                break
    lat = []
    for i, c in enumerate(calls):
        imgs = [base64.b64decode(x) for x in c["frames_b64"]]
        t0 = time.monotonic()
        try:
            be.call(c["system"], c["user"], imgs)
            ok = "ok"
        except Exception as ex:                                   # noqa: BLE001
            ok = f"error {type(ex).__name__}: {str(ex)[:120]}"
        lat.append(round(time.monotonic() - t0, 2))
        print(f"[warmup] {a.tag} call {i + 1}/{len(calls)}: {lat[-1]:.2f} s {ok}", flush=True)
    print(f"[warmup] {a.tag} done: {lat}", flush=True)


if __name__ == "__main__":
    main()
