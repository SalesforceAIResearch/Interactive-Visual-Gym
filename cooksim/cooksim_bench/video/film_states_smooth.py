#!/usr/bin/env python3
"""Re-film an episode's state dump as SUB-TICK frames, staged for present_video.py.

    . ./gpu_env.sh
    python -m cooksim_bench.video.film_states_smooth --states tmp/run/<ep>.states.jsonl \
        --out tmp/v5/stage/<ep> --sub 20 --width 960 --height 720

Why a second filming pass exists now. The standing pipeline reuses the rollout's own
observation frames as film material — one composite per tick, which is right because it
guarantees the video shows exactly what the supervisor saw. But the observation is
DEFINED as one snapped frame per tick, so reusing it means the cook hops between cells and
the picture is capped at the observation's capture resolution.

So the observation stays untouched and the film is rebuilt from COOKSIM_DUMP_STATES, which
is the authoritative per-tick record the evaluation already re-derives ground truth from.
Same states, same order, same tick indices — only interpolated and larger. Nothing here
feeds the assistant.

present_video.py already understands this: its tickmap carries `subs`, it sets each
frame's duration to TICK_S/subs, and it maps utterance ticks through `ticks`, so captions
and voice stay on the tick they were spoken at whatever sub-frame rate is used.
"""
from __future__ import annotations
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, base64, io, json, os, sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

# FILM ON THE SOFTWARE RASTERISER, ALWAYS.
#
# The GPU path (--use-gl=angle --use-angle=vulkan) LOSES ITS WEBGL CONTEXT after the first
# render of a frame. Console: "WebGL: CONTEXT_LOST_WEBGL: loseContext: context lost", preceded
# by "GPU stall due to ReadPixels". Every side-by-side frame needs TWO renders — top-down, then
# ego — so the second one silently did nothing and the ego half was a byte-identical copy of the
# top-down. Measured mean|left-right|: 2.9 on the GPU path, 86.5 on SwiftShader; scene brightness
# 18.6 vs 47.3. Every video and every supervisor observation frame captured on the GPU path has a
# duplicated FIRST PERSON pane and a near-black scene.
#
# SwiftShader is ~0.86 s/frame against ~0.1 on the GPU, which is the price of a correct picture.
# Override with COOKSIM_FILM_GPU=1 only if the context-loss bug is actually fixed.
if os.environ.get("COOKSIM_FILM_GPU") != "1":
    os.environ["COOKSIM_FORCE_SWIFTSHADER"] = "1"

from PIL import Image
from playwright.sync_api import sync_playwright
from cooksim_bench.backends.gemini_stream_rgb import GPU

TICK_S = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", type=int, default=20)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--page", default=os.environ.get(
        "COOKSIM_PAGE", "http://127.0.0.1:17773/static/headless_render.html"))
    a = ap.parse_args()
    sub = max(1, a.sub)

    states = [json.loads(l) for l in open(a.states) if l.strip()]
    if not states:
        raise SystemExit(f"no states in {a.states}")
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(states)} ticks x {sub} sub-frames -> {a.out}", flush=True)

    n = 0
    with sync_playwright() as p:
        b = p.chromium.launch(args=GPU)
        page = b.new_page(viewport={"width": min(a.width, 1600), "height": min(a.height, 1200)})
        sep = "&" if "?" in a.page else "?"
        page.goto(f"{a.page}{sep}w={a.width}&h={a.height}", wait_until="load", timeout=180000)
        page.wait_for_function("() => window.__renderFilmSbs !== undefined", timeout=90000)
        # Renderer and __envReady are built lazily on the first render call; the photoreal
        # HDRI then arrives asynchronously. Capturing before it lands gives near-black PBR.
        page.evaluate("(s)=>window.__renderFrame(s)", json.dumps(states[0]))
        page.wait_for_function("() => window.__envReady === true && (window.__loadsPending || 0) === 0", timeout=180000)

        dt = TICK_S / sub
        for i, st in enumerate(states):
            js = json.dumps(st)
            for k in range(1, sub + 1):
                url = page.evaluate("([s,al,d])=>window.__renderFilmSbs(s,al,d,0.9)",
                                    [js, k / sub, dt])
                img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                img.save(os.path.join(a.out, f"{n:06d}.jpg"), quality=90)
                n += 1
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(states)} ticks ({n} frames)", flush=True)
        w, h = img.size
        b.close()

    # The dump is written once per tick in rollout order, so frame index == tick index —
    # the same mapping cut_v5_videos.py derives from the harness's frame filenames.
    json.dump({"ticks": list(range(len(states))), "subs": sub, "w": w, "h": h, "ext": "jpg"},
              open(os.path.join(a.out, "tickmap.json"), "w"))
    print(f"wrote {n} frames ({w}x{h}), tickmap subs={sub}", flush=True)


if __name__ == "__main__":
    main()
