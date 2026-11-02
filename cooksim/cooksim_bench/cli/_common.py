"""Shared pieces of the CookSim command-line tools: the benchmark environment, the renderer check, key loading."""
import os
import sys
import urllib.request

from cooksim_bench.paths import ROOT, SUITE, GRID

RENDER_URL = "http://127.0.0.1:17773/static/headless_render.html"

# Backward-compatible runner defaults; select --latency 2 for the main benchmark.
BENCH_ENV = {
    "COOKSIM_SCHED": "v5",              # scheduler v5: live-state anchors, soft-hard takeovers
    "COOKSIM_PACE": "tight",            # the fire physics of the formal runs
    "COOKSIM_ASSIST_LATENCY": "measured",   # the assistant's own thinking time is charged to the game clock
    "COOKSIM_GRADER": "cook",           # the simulated cook decides whether a mistake call named the mistake
    "GEMINI_MODEL": "gemini-3.7-flash", # the model behind the simulated cook
}


def apply_bench_env(overrides=None):
    """Set the benchmark environment, keeping anything the caller already exported."""
    os.environ.setdefault("COOKSIM_CASES", SUITE)
    os.environ.setdefault("COOKSIM_GRID", GRID)
    os.environ.setdefault("COOKSIM_PAGE", RENDER_URL)
    for k, v in BENCH_ENV.items():
        os.environ.setdefault(k, v)
    for k, v in (overrides or {}).items():
        if v is not None:
            os.environ[k] = str(v)


def renderer_up(url=None, timeout=3.0):
    url = url or os.environ.get("COOKSIM_PAGE", RENDER_URL)
    try:
        urllib.request.urlopen(url, timeout=timeout).read(64)
        return True
    except Exception:                          # noqa: BLE001
        return False


def require_renderer():
    if not renderer_up():
        page = os.environ.get("COOKSIM_PAGE", RENDER_URL)
        sys.exit(f"the renderer is not answering at {page}\n"
                 f"start it in another terminal:  cooksim-server --port 17773   (or set COOKSIM_PAGE)")


def require_keys(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit("missing API key(s): " + ", ".join(missing) + "  (export them, see docs/GETTING_STARTED.md)")
