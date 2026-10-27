"""screensim-episode: run ONE episode — a task, a persona, an assistant — and write its record.

    screensim-episode --task hearing_setup --persona classic_novice \
        --assistant openai:gpt-5.4-mini --out runs/demo

The record holds every tap, both sides of the conversation, the forced mistakes and their windows, and
the engine's verdict. It replays exactly, so it can be scored, graded or filmed later with no model
calls.
"""
import argparse
import json

from screensim_bench.cli._common import apply_bench_env, assistant_config, output_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, help="a task id (screensim-case lists them)")
    ap.add_argument("--persona", default="classic_novice")
    ap.add_argument("--assistant", required=True, help="<backend>:<model> or a roster tag")
    ap.add_argument("--out", required=True, help="run directory or summary JSON filename")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--episode", default="", help="a specific designed episode id, e.g. hearing_setup__c2")
    ap.add_argument("--human-model", default="gemini-3.7-flash")
    ap.add_argument("--dry-run", action="store_true", help="list the selected cell without calling models")
    a = ap.parse_args(argv)
    apply_bench_env()
    try:
        tag, config = assistant_config(a.assistant, a.base_url)
    except ValueError as exc:
        ap.error(str(exc))
    args = ["--models", tag, "--tasks", a.task, "--out", output_file(a.out),
            "--workers", "1", "--human-model", a.human_model, "--limit", "1",
            "--persona", a.persona]
    if a.episode:
        args += ["--episode", a.episode]
    if config:
        args += ["--assistant-config", json.dumps({tag: config})]
    if a.dry_run:
        args += ["--dry-run"]
    from screensim_bench.runs import sweep
    sweep.main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
