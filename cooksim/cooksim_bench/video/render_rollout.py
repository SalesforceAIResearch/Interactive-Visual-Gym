#!/usr/bin/env python3
"""Renderer for error ROLLOUTS (see exp_error_rollout.py): full-episode video with a ground-truth
overlay (normal -> DEVIATION -> FAILED), a live step/plate readout, an outcome end-card, and a
sidecar manifest carrying the per-command timeline for the website's trajectory panel."""
import io, json, os
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright
from cooksim_bench.backends.gemini_stream_rgb import GPU, PAGE, render_frame, encode
from cooksim.core.items import Plate

TITLE = {"wrong_location": "WRONG LOCATION", "raw_on_plate": "RAW ON PLATE", "wrong_order": "WRONG ORDER",
         "wrong_ingredient": "WRONG INGREDIENT", "omission": "OMISSION", "premature_serve": "PREMATURE SERVE",
         "stall": "STALL"}
OUTCOME_TXT = {"wrong_serve": "✗ SERVE REJECTED — WRONG PLATE", "timeout": "✗ TIMEOUT — TASK NEVER COMPLETED",
               "fire": "✗ KITCHEN FIRE — IRRECOVERABLE FAILURE", "won": "⚠ DELIVERED (variant leak!)"}
def _success_txt(total, base):
    return f"✓ DELIVERED — time {total} vs {base} plan (+{total - base} lost)"


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def plate_str(cook):
    h = cook.holding
    if h is None:
        return "empty-handed"
    if isinstance(h, Plate):
        seq = getattr(h, "content_seq", None) or [(i.name, getattr(i.state, "value", i.state)) for i in h.contents]
        return "plate: [" + ", ".join(f"{n}({s})" for n, s in seq) + "]" if seq else "plate: [empty]"
    return f"holding {getattr(h,'name',h)} ({getattr(getattr(h,'state',None),'value','')})"


def overlay(png, dev, phase_lab, tick, mode, step_txt, plate, outcome=None):
    im = Image.open(io.BytesIO(png)).convert("RGB")
    W, H = im.size
    bar = 58
    canvas = Image.new("RGB", (W, H + bar + 26), (24, 30, 43))
    canvas.paste(im, (0, bar))
    d = ImageDraw.Draw(canvas)
    col = {"normal": (150, 200, 160), "dev": (255, 180, 84), "failed": (255, 106, 90)}[mode]
    mtxt = {"normal": "on-plan", "dev": "⚠ DEVIATING", "failed": "✗ FAILED"}[mode]
    d.text((12, 8), f"ERROR ROLLOUT — {TITLE[dev]} · {phase_lab}", font=_font(18), fill=(244, 238, 231))
    d.text((12, 34), step_txt, font=_font(13), fill=(182, 171, 194))
    d.text((W - 150, 8), mtxt, font=_font(14), fill=col)
    d.text((W - 150, 28), f"time {tick}", font=_font(12), fill=(182, 171, 194))
    d.rectangle([0, bar - 3, W, bar], fill=col)
    d.text((12, H + bar + 5), plate, font=_font(13), fill=(233, 226, 214))
    if outcome:                                              # end card (outcome = display string here)
        ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rectangle([0, bar + H // 2 - 40, W, bar + H // 2 + 40], fill=(20, 16, 24, 215))
        good = outcome.startswith("✓")
        od.text((22, bar + H // 2 - 16), outcome, font=_font(21),
                fill=(95, 207, 148) if good else (255, 106, 90))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")
    return canvas


async def render(args, build):
    from cooksim_bench.exp_error_rollout import baseline_time
    game, plan, v, ro = build(args.cell, args.dev, args.phase)
    baseline = baseline_time(args.cell)
    cook = game.cooks[0]
    frames = []
    dev_frame = fail_frame = None
    async with async_playwright() as p:
        b = await p.chromium.launch(args=GPU)
        page = await b.new_page(viewport={"width": 800, "height": 580})
        await page.goto(PAGE, wait_until="load")
        await page.wait_for_function("window.__ready===true", timeout=90000)
        await page.wait_for_timeout(400)
        alive = True
        while alive:
            png = await render_frame(page, game.render_state())
            mode = "dev" if (ro.dev_tick is not None and ro.tick >= ro.dev_tick) else "normal"
            if mode == "dev" and dev_frame is None:
                dev_frame = len(frames)
            cur = ro.log[-1] if ro.log else None
            done_n = sum(1 for l in ro.log if l["status"] in ("done", "rejected", "illegal-skipped"))
            step_txt = (f"step {min(done_n + 1, len(ro.cmds))}/{len(ro.cmds)}: {cur['cmd']}" if cur
                        else f"step 1/{len(ro.cmds)}")
            frames.append(overlay(png, args.dev, args.phase, ro.tick, mode, step_txt, plate_str(cook)))
            alive = ro.step()
        # outcome end-card (~2s)
        png = await render_frame(page, game.render_state())
        fail_frame = len(frames)
        base = baseline
        good = ro.outcome == "won" and v.get("succeeds")
        card_txt = _success_txt(ro.tick, base) if good else OUTCOME_TXT[ro.outcome]
        end_mode = "dev" if good else "failed"
        for _ in range(args.fps * 2):
            frames.append(overlay(png, args.dev, args.phase, ro.tick, end_mode,
                                  f"outcome: {ro.outcome} · total time {ro.tick} (plan {base})",
                                  plate_str(cook), outcome=card_txt))
        await b.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    encode(frames, args.out, fps=args.fps)
    manifest = dict(
        dev=args.dev, phase=args.phase, cell=args.cell, fps=args.fps,
        n_frames=len(frames), duration=round(len(frames) / args.fps, 2),
        k=v["k"], outcome=ro.outcome, succeeds=bool(v.get("succeeds")),
        total_time=ro.tick, baseline_time=baseline,
        time_overhead=(ro.tick - baseline) if baseline else None,
        dev_time=round(dev_frame / args.fps, 2) if dev_frame is not None else None,
        fail_time=round(fail_frame / args.fps, 2),
        gt_plan=plan,
        commands=[dict(cmd=l["cmd"], kind=l["kind"], status=l["status"],
                       t_issue=round(l["t_issue"] / args.fps, 2),
                       t_done=(round(l["t_done"] / args.fps, 2) if l["t_done"] is not None else None))
                  for l in ro.log],
    )
    json.dump(manifest, open(os.path.splitext(args.out)[0] + ".json", "w"), indent=2)
    print(f"RESULT {args.dev}/{args.phase}: k={v['k']} outcome={ro.outcome} frames={len(frames)} "
          f"dev={manifest['dev_time']}s fail={manifest['fail_time']}s -> {args.out}")
