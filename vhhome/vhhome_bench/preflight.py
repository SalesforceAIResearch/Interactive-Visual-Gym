#!/usr/bin/env python3
"""Prove an assistant arm answers through THE HARNESS'S OWN path, and warm its server.

    python -m vhhome_bench.preflight qwen3.8-27b [--repeat 3] [--json]

Why not a hand-rolled request: a probe that builds its own call passes while the real path 401s, sends
the frames in the wrong order, or returns reasoning-only content — which is how a mute arm gets past a
green preflight (tools/assist_sweep.py's PREFLIGHT guard exists for exactly that). This imports
iv_protocol under the arm's environment and calls assist_frames, so the arm's profile, image scale, auth,
schema and JSON extraction are the ones the lane will use.

It doubles as the warm-up the Beaker job runs between "server is up" and the first episode: a fresh
server pays one-time costs (schema compiled to a grammar, kernels chosen, weights paged in from the shared filesystem) and
the assistant's latency is charged to the game clock.
"""
from vhhome import keys
import argparse, io, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

from vhhome_bench import roster as roster_vh


def frames(n=8, maps=2):
    """The SAME SHAPE an episode sends: n first-person frames plus the two maps (whole-house overview and
    the room the person is in) = 10 images, at episode size.

    The count matters as much as the bytes. A preflight that sent only the 8 frames passed against servers
    capped at 8 images, and then every real call failed with "At most 8 image(s) may be provided in one
    prompt" — a whole Beaker check round of mute assistants that looked green here (2026-09-20)."""
    from PIL import Image, ImageDraw
    out = []
    for i in range(n):
        im = Image.new("RGB", (1280, 480), (22, 24, 30))
        d = ImageDraw.Draw(im)
        d.rectangle([40 + i * 30, 150, 420 + i * 30, 430], fill=(150, 120, 80))     # "furniture"
        d.rectangle([660, 40, 1240, 440], outline=(57, 217, 138), width=3)          # "map"
        d.text((30, 30), f"tick {i}", fill=(255, 255, 255))
        b = io.BytesIO(); im.save(b, "JPEG", quality=80); out.append(b.getvalue())
    for k in range(maps):                       # the two map images the assistant turn appends
        im = Image.new("RGB", (560, 560), (15, 17, 23))
        d = ImageDraw.Draw(im)
        d.rectangle([40, 40, 520, 520], outline=(57, 217, 138), width=3)
        d.text((60, 60), "MAP" if k else "HOUSE", fill=(57, 217, 138))
        b = io.BytesIO(); im.save(b, "JPEG", quality=80); out.append(b.getvalue())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    entry = roster_vh.by_tag(a.tag)
    miss = roster_vh.missing_keys(entry)
    if miss:
        sys.exit(f"{a.tag}: missing {', '.join(miss)}")
    for k, v in entry.get("env", {}).items():
        os.environ[k] = v
    keys.load_all()
    for var, path in ():
        p = os.path.expanduser(path)
        if not os.environ.get(var) and os.path.exists(p):
            os.environ[var] = open(p).read().strip()

    from vhhome_bench import protocol as P  # imported AFTER the arm's env is in place
    sys_prompt = P.e2e_supervisor_prompt("routine_hard", "baseline")
    user = ("OBSERVATION t=42: the last 8 frames are attached (first-person view beside the top-down map).\n"
            "The person is in the kitchen and has just closed the fridge holding a carton.\n"
            "Reply with the strict JSON described above.")
    imgs = frames()
    print(f"[preflight] {a.tag}: sending {len(imgs)} images per call (8 frames + 2 maps, as an episode does)",
          flush=True)
    rows = []
    for i in range(a.repeat):
        t0 = time.time()
        out = P.assist_frames(P.make_client() if not P.ASSIST_BASE_URL and not P.ASSIST_BACKEND else None,
                              sys_prompt, user, imgs)
        secs = time.time() - t0
        ok = isinstance(out, dict) and bool(out)
        rows.append({"call": i + 1, "ok": ok, "secs": round(secs, 1),
                     "keys": sorted(out)[:6] if ok else [],
                     "said": (out.get("text") or "")[:80] if ok else "",
                     "flag": (out or {}).get("flag")})
        print(f"[preflight] {a.tag} call {i+1}/{a.repeat}: {'ok' if ok else 'EMPTY'} in {secs:.1f}s"
              + (f" keys={sorted(out)}" if ok else ""), flush=True)
    st = dict(P.ASSIST_STATS)
    good = sum(1 for r in rows if r["ok"])
    print(f"[preflight] {a.tag}: {good}/{len(rows)} replies parsed, "
          f"prompt_tokens={st.get('prompt_tokens')}, retries={st.get('retries')}, "
          f"schema_fail={st.get('schema_fail')}", flush=True)
    if a.json:
        print(json.dumps({"tag": a.tag, "rows": rows, "stats": st}, indent=1))
    sys.exit(0 if good else 5)


if __name__ == "__main__":
    main()
