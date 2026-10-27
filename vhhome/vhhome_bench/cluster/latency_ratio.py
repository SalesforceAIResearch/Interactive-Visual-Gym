#!/usr/bin/env python3
"""How much slower is this card than an A100, for THIS model — measured, then charged.

    python -m vhhome_bench.cluster.latency_ratio --submit --root tmp/gpu_calib      # time every served model on
                                                                         # each GPU kind we may land on
    python -m vhhome_bench.cluster.latency_ratio --build  --root tmp/gpu_calib      # -> docs/GPU_LATENCY_RATIO.json
    python -m vhhome_bench.cluster.latency_ratio --lookup molmo2-8b --gpu "NVIDIA L40S"   # prints 2.05

Why it exists. A served model's generation time is charged to the game clock (ASSIST_TICK_S), so the
person keeps moving while the assistant thinks. bench_v1 ran its served arms on A100s. When the workspace's
A100 allocation is full and the same arm runs on an L40S instead, the identical model answers roughly twice
as slowly, the person walks twice as far mid-reply, and the arm scores worse for a reason that has nothing
to do with the model. The fix is to charge A100-EQUIVALENT seconds: measure ratio = (this card's median
generation time) / (an A100's median) on the same synthetic call, and let the rollout divide by it.

The measurement is deliberately the preflight call — 8 frames + 2 maps at episode size, through the
harness's own protocol path — because that is the shape whose latency is being charged.

A100 rows are the baseline and always ratio 1.0. A model with no row for a card falls back to 1.0, i.e.
it is charged what it measured: never silently scaled by another model's number.
"""
from vhhome import paths
import argparse, glob, json, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)

from vhhome_bench import roster as roster_vh  # noqa: E402

TABLE = paths.GPU_LATENCY_RATIO
BASE = "A100"                       # the card bench_v1 was charged on
# Where a served model can physically run: an L40S has 48 GiB, so the two biggest weights need an 80 GiB
# card (bf16 27B and 30B-A3B do not fit, and TP=2 on L40S is a different serving config, not a re-run).
BIG = {"qwen3.8-27b", "qwen3-omni-30b-a3b"}
# One calibration job per GPU KIND, not per cluster: what changes the generation time is the card.
# H100 is FASTER than the A100 baseline, so its ratio is below 1 and gen_s / ratio scales the charge UP —
# the arithmetic is the same in both directions (project note 2026-09-22).
KINDS = {"A100": os.environ.get("VHHOME_CLUSTERS_A100", ""),
         "L40S": os.environ.get("VHHOME_CLUSTERS_L40S", ""),   # another cluster is L40S too but "doesn't allow tasks"
         "H100": os.environ.get("VHHOME_CLUSTERS_H100", "")}
ROOMY = ("A100", "H100")            # 80 GiB: the only cards bf16 27B / 30B-A3B fit on


def kinds_for(tag):
    return [k for k in KINDS if tag not in BIG or k in ROOMY]


def gpu_key(name):
    """'NVIDIA A100-SXM4-80GB' -> 'A100'; 'NVIDIA L40S' -> 'L40S'; 'NVIDIA H100 80GB HBM3' -> 'H100'."""
    n = (name or "").upper()
    for k in ("A100", "H100", "B200", "L40S", "A6000", "H200", "L4"):
        if k in n:
            return k
    return n.replace("NVIDIA ", "").strip() or "UNKNOWN"


def median_gen_s(path):
    # The job captures the preflight's whole stdout, so its progress lines come before the JSON block.
    txt = open(path).read()
    i = txt.find("\n{")
    d = json.loads(txt[i + 1:] if i >= 0 else txt)
    rows = [r for r in d.get("rows", []) if r.get("ok")]
    if len(rows) > 2:
        rows = rows[1:]                     # the first call after a fresh server still pays warm-up
    return statistics.median([r["secs"] for r in rows]) if rows else None


def build(root):
    root_abs = root if os.path.isabs(root) else os.path.join(ENG, root)
    per = {}                                 # tag -> gpu -> median seconds
    for f in sorted(glob.glob(os.path.join(root_abs, "calib", "*.json"))):
        base = os.path.basename(f)[:-5]
        if "__" not in base:
            continue
        tag, gpu = base.split("__", 1)
        m = median_gen_s(f)
        if m:
            per.setdefault(tag, {})[gpu_key(gpu)] = round(m, 2)
    out = {"note": "ratio = this card's median generation time / the A100's, same synthetic call "
                   "(8 frames + 2 maps). The rollout charges gen_s / ratio, so every run is charged "
                   "A100-equivalent seconds. 1.0 means charge what was measured.",
           "base": BASE, "seconds": per, "ratio": {}}
    for tag, gs in per.items():
        b = gs.get(BASE)
        out["ratio"][tag] = {g: (1.0 if g == BASE else (round(s / b, 2) if b else 1.0))
                             for g, s in gs.items()}
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    json.dump(out, open(TABLE, "w"), indent=1)
    print(f"wrote {TABLE}")
    print(f"{'model':24} " + "".join(f"{g:>10}" for g in ("A100", "L40S", "H100")) + "    ratios")
    for tag in sorted(per):
        secs = "".join(f"{per[tag].get(g, '—'):>10}" for g in ("A100", "L40S", "H100"))
        rat = " ".join(f"{g}={v}" for g, v in sorted(out["ratio"][tag].items()) if g != BASE)
        print(f"{tag:24} {secs}    {rat}")
    miss = [t for t in per if BASE not in per[t]]
    if miss:
        print(f"\nNO A100 ROW (ratio falls back to 1.0, i.e. charged as measured): {', '.join(miss)}")
    return out


def submit(root, repeat, dry):
    n = 0
    for e in roster_vh.runnable():
        if e["kind"] != "local":
            continue
        for kind in kinds_for(e["tag"]):
            cmd = [sys.executable, "-m", "vhhome_bench.cluster.beaker_submit", "calib", e["tag"], "--root", root,
                   "--cluster", KINDS[kind], "--retries", "2", "--suffix", kind.lower()]
            print("  ", " ".join(cmd[1:]))
            if not dry:
                subprocess.run(cmd, cwd=ENG, check=False,
                               env={**os.environ, "CALIB_REPEAT": str(repeat)})
            n += 1
    print(f"{n} calibration jobs" + (" (dry run)" if dry else " submitted"))


def lookup(tag, gpu):
    try:
        t = json.load(open(TABLE))["ratio"]
    except Exception:
        print(1.0); return
    print(t.get(tag, {}).get(gpu_key(gpu), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tmp/gpu_calib")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lookup", default=None)
    ap.add_argument("--gpu", default="")
    ap.add_argument("--repeat", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.lookup:
        lookup(a.lookup, a.gpu)
    elif a.submit:
        submit(a.root, a.repeat, a.dry_run)
    elif a.build:
        build(a.root)
    else:
        ap.error("one of --submit / --build / --lookup")


if __name__ == "__main__":
    main()
