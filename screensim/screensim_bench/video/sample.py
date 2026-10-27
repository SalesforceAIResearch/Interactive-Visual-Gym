"""Randomly sample N episodes from a round's records, replay-film each, write a manifest.

  python screensim_bench/video/sample.py tmp/pilot/sample_r6.json --round 6 --n 10 --seed 6 \
      --site site_gh --subdir videos/round6

--style hand (default): screensim_bench/video/film.py — a hand performs every gesture, in-between screens
move, and every line is voiced while the picture holds. --style replay: the older still-frame cut.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


from multiprocessing import Pool

_CAP = None
_STYLE = "hand"


def _init(style="hand"):
    global _CAP, _STYLE
    from screensim.render.capture import Capturer
    _CAP = Capturer(scale=1.0).__enter__()
    _STYLE = style


def _one(job):
    n, r, path = job
    if _STYLE == "hand":
        from screensim_bench.video.film import film
    else:
        from screensim_bench.video.replay import film
    fid, cues, secs = film(r, path, _CAP)
    return n, fid, cues, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--round", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--site", default="site_gh")
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--notes", default=None, help="json {record key: note html}")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--style", choices=["hand", "replay"], default="hand")
    a = ap.parse_args()
    runs = [r for r in json.load(open(a.records)) if "error" not in r]
    from screensim.interact.beats_design import beats_design
    for r in runs:
        r["beats_design"] = beats_design(r)          # sweep records do not carry them
    pick = random.Random(a.seed).sample(runs, a.n)
    notes = json.load(open(a.notes)) if a.notes else {}
    revs = sorted({r.get("engine_rev") for r in runs})
    eps = []
    os.makedirs(os.path.join(a.site, a.subdir), exist_ok=True)
    jobs, meta = [], {}
    for n, r in enumerate(pick, 1):
        key = f"{r['contestant']}__{r['episode']}__{r['persona']}"
        vid = os.path.join(a.subdir, f"{n:02d}_{key}".replace(".", "-") + ".mp4")
        jobs.append((n, r, os.path.join(a.site, vid)))
        meta[n] = (key, r, vid)
    got = {}
    with Pool(min(a.workers, len(jobs)), initializer=_init, initargs=(a.style,)) as pool:
        for n, fid, cues, secs in pool.imap_unordered(_one, jobs):
            got[n] = (fid, cues, secs)
            print(f"[{len(got)}/{a.n}] #{n} {meta[n][0]} exact={fid['exact']} {secs:.0f}s", flush=True)
    for n in sorted(got):
        key, r, vid = meta[n]
        fid, cues, secs = got[n]
        eps.append({"key": key, "record": r, "video": vid, "poster": vid[:-4] + ".jpg",
                    "cues": cues, "seconds": secs, "fidelity": fid, "note": notes.get(key)})
    json.dump({"round": a.round, "seed": a.seed, "style": a.style, "engine_rev": ",".join(map(str, revs)),
               "human_model": runs[0].get("human_model"), "n_runs": len(runs), "episodes": eps},
              open(a.manifest, "w"))


if __name__ == "__main__":
    main()
