"""screensim-benchmark: run the benchmark against one or more assistants.

    screensim-benchmark --assistant openai:gpt-5.4-mini --out results/gpt
    screensim-benchmark --assistant openai:my-7b --base-url http://127.0.0.1:8000/v1 --out results/mine

90 cells per seat: 30 designed episodes x 3 personas. Resumable — re-running the same command fills in
only what is missing. `--limit N` runs the first N cells as a setup check before spending the rest.
"""
import argparse
import json

from screensim_bench.cli._common import apply_bench_env, assistant_config, output_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assistant", required=True, help="<backend>:<model> or a roster tag")
    ap.add_argument("--out", required=True, help="results file or directory")
    ap.add_argument("--base-url", default="", help="for an OpenAI-compatible server")
    ap.add_argument("--tasks", default="", help="comma list of task ids; default every benchmark task")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N cells: a setup check")
    ap.add_argument("--no-manual", action="store_true", help="assistant without the User Guide (ablation)")
    ap.add_argument("--human-model", default="gemini-3.7-flash", help="the model that plays the person")
    ap.add_argument("--shard", default="", help="i/n: run slice i of n, for splitting a run across machines")
    ap.add_argument("--dry-run", action="store_true", help="list selected cells without calling models")
    a = ap.parse_args(argv)
    apply_bench_env()
    try:
        tag, config = assistant_config(a.assistant, a.base_url)
    except ValueError as exc:
        ap.error(str(exc))

    argv2 = ["--models", tag, "--out", output_file(a.out),
             "--workers", str(a.workers), "--human-model", a.human_model]
    if a.tasks:
        argv2 += ["--tasks", a.tasks]
    if a.limit:
        argv2 += ["--limit", str(a.limit)]
    if a.no_manual:
        argv2 += ["--no-manual"]
    if a.shard:
        argv2 += ["--shard", a.shard]
    if config:
        argv2 += ["--assistant-config", json.dumps({tag: config})]
    if a.dry_run:
        argv2 += ["--dry-run"]
    from screensim_bench.runs import sweep
    sweep.main(argv2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
