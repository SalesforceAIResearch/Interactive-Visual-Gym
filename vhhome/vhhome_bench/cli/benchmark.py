"""vhhome-benchmark: one assistant over the benchmark's 75 cells (25 cases x 3 personas), resumable.

Records land under <out>/<tag>/ (or <out>/r<k>/<tag>/ with --repeat k); a cell whose record exists is skipped, so a
stopped run resumes. An API assistant can use several simulator instances at once (--engines 8080,8081,...); a model
you serve locally must use one, because its latency is charged to the game clock and two episodes sharing the
server would each be charged the other's compute. The published protocol is three independent runs:
--repeat 1, 2 and 3 into the same --out, then vhhome-score --run <out>/r1 <out>/r2 <out>/r3.
"""
import argparse, json, os, sys, threading

from vhhome import paths
from vhhome_bench.cli._common import ASSISTANT_HELP, add_assistant_args, assistant_entry, require_keys, use_cases


def main():
    ap = argparse.ArgumentParser(prog="vhhome-benchmark", description=__doc__, epilog=ASSISTANT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_assistant_args(ap)
    ap.add_argument("--out", required=True, help="run root")
    ap.add_argument("--repeat", type=int, default=None, help="independent run number k: records go to <out>/r<k>")
    ap.add_argument("--engines", default="8080", help="comma-separated simulator ports to run on at once")
    ap.add_argument("--cells", default=paths.CELLS, help="the cell list (default: the benchmark's 75 cells)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells (a smoke run)")
    ap.add_argument("--shard", default=None, help="k/n: every n-th cell, for splitting a lane over machines")
    a = ap.parse_args()

    entry = assistant_entry(a.assistant, a.base_url, a.profile, a.image_scale, a.max_tok, a.tag)
    require_keys(entry)
    use_cases(a.cases)
    from vhhome_bench import lane as L
    cells = L.load_cells(a.cells, a.shard, a.limit)
    ports = [int(p) for p in a.engines.split(",")]
    if entry["kind"] == "local" and len(ports) > 1:
        sys.exit("a locally served model runs ONE episode at a time: give a single --engines port")
    root = os.path.abspath(os.path.join(a.out, f"r{a.repeat}") if a.repeat else a.out)
    os.makedirs(root, exist_ok=True)
    print(f"lane {entry['tag']}: {len(cells)} cells, simulators {ports}, root {root}", flush=True)

    log, queue, lock = [], list(cells), threading.Lock()

    def worker(port):
        while True:
            with lock:
                if not queue:
                    return
                cell = queue.pop(0)
            try:
                L.run_cell(entry, cell, port, root, log)
            except Exception as e:                      # one bad cell must not end the lane
                print(f"[lane] {cell} raised {type(e).__name__}: {e}", flush=True)

    ts = [threading.Thread(target=worker, args=(p,), daemon=True) for p in ports]
    [t.start() for t in ts]
    [t.join() for t in ts]
    done = sum(1 for x in log if x["ok"])
    json.dump({"tag": entry["tag"], "cells": len(cells), "ran": len(log), "ok": done, "log": log},
              open(os.path.join(root, f"{entry['tag']}.lane.json"), "w"), indent=1)
    print(f"lane {entry['tag']} finished: {done}/{len(log)} ran ok ({len(cells) - len(log)} already present)")
    return 6 if (log and not done) else 0


if __name__ == "__main__":
    sys.exit(main())
