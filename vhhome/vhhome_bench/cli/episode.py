"""vhhome-episode: run one episode.

A simulated person (a Gemini model with a persona) does one household chore in VirtualHome; the engine forces the
case's mistakes at live moments; the assistant watches the first-person frames and the maps and may speak.
The record (every tick, every line, every verdict) is written as one JSON file in the layout vhhome-score reads:
<out>/<tag>/<tag>__<case>__<persona>.json, with the episode's log beside it.
"""
import argparse, os, sys

from vhhome_bench.cli._common import ASSISTANT_HELP, add_assistant_args, assistant_entry, require_keys, use_cases


def main():
    ap = argparse.ArgumentParser(prog="vhhome-episode", description=__doc__, epilog=ASSISTANT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="a case name (vhhome-case list), or <task>__clean for a mistake-free run")
    ap.add_argument("--persona", default="baseline", help="the person's persona (vhhome-case list personas)")
    add_assistant_args(ap, default="silent")
    ap.add_argument("--port", type=int, default=8080, help="the simulator instance to use (simulator/launch_simulators.sh)")
    ap.add_argument("--out", default="runs/demo", help="run root; the record goes under <out>/<tag>/")
    ap.add_argument("--paradigm", default="combo", choices=["combo", "soft", "hard"],
                    help="how a mistake is forced (combo = the benchmark setting: takeover with an in-character prompt)")
    a = ap.parse_args()

    entry = assistant_entry(a.assistant, a.base_url, a.profile, a.image_scale, a.max_tok, a.tag)
    if a.latency is not None:
        entry["env"]["ASSIST_LATENCY"] = a.latency
    require_keys(entry)
    use_cases(a.cases)
    from vhhome_bench import lane
    for key, value in lane.BASE_ENV.items():
        os.environ.setdefault(key, value)              # explicit caller settings take precedence
    os.environ.update(entry["env"])
    os.environ["VH_RUN_TAG"] = f"{entry['tag']}-{os.getpid()}"
    out = os.path.join(a.out, entry["tag"], f"{entry['tag']}__{a.case}__{a.persona}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    from vhhome_bench import episode as E                # imported after the environment is set
    rep = E.run(a.port, a.case, a.persona, out, a.paradigm)
    print(f"record: {out}")
    return 0 if rep else 1


if __name__ == "__main__":
    sys.exit(main())
