"""cooksim-episode: run ONE benchmark episode -- a case, a cook persona, an assistant -- and write its record.

    cooksim-episode --case b3_easy_nops_medium_map_2 --persona baseline \
        --assistant gemini:gemini-3.8-flash --out runs/demo

The assistant is `<backend>:<model>` (gemini, openai, anthropic, live, realtime, live_native, realtime_native) or a
roster tag from `cooksim-roster` (its exact flags are used). The record lands at
<out>/frames/<persona>/<tag>/<case>.json, the layout every scoring tool reads.
"""
import argparse
import os
import runpy
import sys

from cooksim_bench import roster as R
from cooksim_bench.cli._common import apply_bench_env, require_renderer


def assistant_flags(spec, base_url=None, profile=None):
    """Turn `backend:model` or a roster tag into the rollout's --assist-* flags, plus the record tag."""
    entry = R.by_tag(spec) if any(e["tag"] == spec for e in R.ROSTER) else None
    if entry:
        flags = list(entry["flags"])
        if base_url:
            flags += ["--assist-base-url", base_url]
        if profile:
            flags += ["--assist-profile", profile]
        return flags, spec
    if ":" not in spec:
        sys.exit(f"--assistant must be <backend>:<model> or a roster tag; got {spec!r}  (cooksim-roster lists the tags)")
    backend, model = spec.split(":", 1)
    flags = ["--assist-backend", backend, "--assist-model", model]
    if base_url:
        flags += ["--assist-base-url", base_url]
    if profile:
        flags += ["--assist-profile", profile]
    return flags, model.replace("/", "_")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cooksim-episode", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="case name from the suite, e.g. b3_easy_nops_medium_map_2")
    ap.add_argument("--persona", default="baseline", help="cook persona: baseline | classic_novice | classic_expert | ... (cooksim-roster --personas)")
    ap.add_argument("--assistant", required=True, help="<backend>:<model> or a roster tag")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for backend 'openai' (local vLLM servers)")
    ap.add_argument("--profile", default=None, help="request profile for backend 'openai' (gpt5 | default | ...)")
    ap.add_argument("--out", default="runs/demo", help="run root; the record goes under <out>/frames/<persona>/<tag>/")
    ap.add_argument("--tag", default=None, help="record tag (default: the model name or roster tag)")
    ap.add_argument("--cases", default=None, help="a case file instead of the shipped suite (cooksim-case writes these)")
    ap.add_argument("--latency", default=None, help="2: fixed two-tick delay (main benchmark); measured: inference-time delay. Omitted: COOKSIM_ASSIST_LATENCY or measured.")
    ap.add_argument("--obs", default="frames", choices=["frames", "agent+++"], help="what the assistant sees: rendered frames or a text state map")
    ap.add_argument("--dump-states", action="store_true", help="also write the per-tick kitchen state (exact grading evidence, needed for films)")
    args, extra = ap.parse_known_args(argv)

    apply_bench_env({"COOKSIM_CASES": os.path.abspath(args.cases) if args.cases else None,
                     "COOKSIM_ASSIST_LATENCY": args.latency,
                     "COOKSIM_DUMP_PER_EPISODE": "1" if args.dump_states else None})
    require_renderer()
    flags, tag = assistant_flags(args.assistant, args.base_url, args.profile)
    tag = args.tag or tag
    out = os.path.join(args.out, "frames", args.persona, tag, f"{args.case}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.dump_states:
        os.environ["COOKSIM_DUMP_STATES"] = out[:-5] + ".states.jsonl"
    sys.argv = ["e2e_stepin_rollout", "--case", args.case, "--persona-v2", args.persona, "--paradigm", "hardsoft",
                "--sup-obs", args.obs, "--out", out] + flags + extra
    runpy.run_module("cooksim_bench.e2e_stepin_rollout", run_name="__main__")
    print(f"record: {out}")


if __name__ == "__main__":
    main()
