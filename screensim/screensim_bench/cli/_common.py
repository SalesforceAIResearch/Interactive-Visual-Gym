"""Shared pieces of the ScreenSim command-line tools."""
import os
import re
import sys
from pathlib import Path

from screensim.interact.roster import ROSTER, make_backend          # noqa: F401  (re-exported)

# the setting every published ScreenSim number was produced with
BENCH_ENV = {
    "SCREENSIM_GRADER": "person",        # the simulated person judges whether a warning named the mistake
    "SCREENSIM_HUMAN_MODEL": "gemini-3.7-flash",
}

BENCHMARK_PERSONAS = ("classic_novice", "classic_expert", "baseline")


def apply_bench_env():
    for k, v in BENCH_ENV.items():
        os.environ.setdefault(k, v)


def roster_tags():
    return [e["tag"] for e in ROSTER]


def resolve_assistant(spec):
    """`<backend>:<model>` or a roster tag -> (backend, model, tag)."""
    for e in ROSTER:
        if e["tag"] == spec:
            return e.get("backend"), e.get("model"), spec
    if ":" not in spec:
        sys.exit(f"--assistant must be <backend>:<model> or a roster tag; got {spec!r} "
                 f"(screensim-roster lists them)")
    backend, model = spec.split(":", 1)
    if backend not in ("openai", "gemini", "anthropic", "live", "realtime") or not model.strip():
        raise ValueError("Use openai:<model>, anthropic:<model>, gemini:<model>, live:<model>, realtime:<model>, or a roster tag")
    for e in ROSTER:
        if e["backend"] == backend and e["model"] == model:
            return backend, model, e["tag"]
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("._")
    if not tag:
        raise ValueError("The assistant model must contain a usable name")
    return backend, model, tag


def assistant_config(spec, base_url=""):
    """Resolve explicit provider routing without changing published roster entries."""
    backend, model, tag = resolve_assistant(spec)
    entry = next((dict(e) for e in ROSTER if e["tag"] == tag
                  and e["backend"] == backend and e["model"] == model), None)
    if base_url and backend != "openai":
        raise ValueError("--base-url is supported only for OpenAI-compatible assistants")
    if entry is not None and not base_url:
        return tag, None
    entry = entry or dict(tag=tag, backend=backend, model=model, profile="default",
                          base_url="https://api.openai.com/v1", max_tokens=4000, image_scale=1.0)
    if base_url:
        entry["base_url"] = base_url
    return tag, entry


def output_file(value):
    """Accept an explicit JSON file or a run directory, including existing directories."""
    path = Path(value).expanduser()
    if path.is_dir() or path.suffix.lower() != ".json":
        path = path / "composite_results_v2.json"
    return str(path)


def record_roots(values):
    roots = [Path(value).expanduser().resolve() for value in values]
    if len(set(roots)) != len(roots):
        raise ValueError("Each run directory must be distinct")
    for root in roots:
        if not root.is_dir() or not any(root.glob("*.records.jsonl")):
            raise ValueError(f"No *.records.jsonl files in run directory: {root}")
    return [str(root) for root in roots]
