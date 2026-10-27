"""Film one interaction episode into an mp4: the phone screen on the left, a
live transcript + running state on the right, and a verdict card at the end.

Fed the episode's `_film` frames (png bytes, caption, tick, kind), its timed
transcript, and its grade. Produces a wide (phone + panel) reel — the CookSim
eval-video analogue for the screen domain.

  from screensim_bench.video.episode_film import film_episode
  film_episode(ep, out_path)          # ep carries _film + verdict fields
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

FONT = "/weka/oe-training-default/jasonr/chenhao_prior/home/.local/share/fonts/InterVariable.ttf"
PHONE_W, PHONE_H = 412, 892
PANEL_W = 520
PAD = 28
GAP = 24
W = PHONE_W + GAP * 3 + PANEL_W
H = PHONE_H + PAD * 2
BG = (14, 16, 23)
CARD = (22, 27, 36)
INK = (232, 236, 245)
MUT = (152, 162, 182)
ACC = (77, 163, 255)
GOOD = (78, 201, 138)
WARN = (224, 168, 63)
BAD = (229, 118, 118)


def _font(sz, bold=False):
    try:
        f = ImageFont.truetype(FONT, sz)
        if bold:
            try:
                f.set_variation_by_name("Bold")
            except Exception:
                pass
        return f
    except Exception:
        return ImageFont.load_default()


F_H = _font(24, bold=True)
F_M = _font(19)
F_S = _font(16)
F_T = _font(15)
F_TAG = _font(14, bold=True)


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _round(draw, box, r, fill):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def _verdict_color(ep):
    v = ep.get("detection_verdict")
    if ep["family"] != "error":
        return (GOOD if ep.get("restraint_verdict") != "false_alarm" else WARN)
    if v in ("correct", "early_correct"):
        return GOOD
    return BAD


def _panel(ep, upto_lines, header_note):
    """Render the right panel: header + transcript so far + running verdict."""
    img = Image.new("RGB", (PANEL_W, PHONE_H), CARD)
    d = ImageDraw.Draw(img)
    x = 22
    y = 22
    # header
    d.text((x, y), ep["contestant"], font=F_H, fill=INK)
    y += 32
    fam = ep["family"].upper()
    famc = BAD if fam == "ERROR" else GOOD
    _round(d, (x, y, x + 74, y + 22), 6, famc)
    d.text((x + 10, y + 3), fam, font=F_TAG, fill=(12, 14, 20))
    d.text((x + 86, y + 2), f"{ep['task']}", font=F_S, fill=MUT)
    y += 34
    if ep.get("err") and ep["err"] != "none":
        d.text((x, y), f"slip: {ep['err']}  ·  {header_note}", font=F_S, fill=MUT)
        y += 26
    d.line((x, y, PANEL_W - 22, y), fill=(40, 46, 58), width=1)
    y += 14
    # transcript
    for who, text in upto_lines[-14:]:
        col = ACC if who == "assistant" else INK
        tag = "assistant" if who == "assistant" else "you"
        d.text((x, y), tag.upper(), font=F_TAG, fill=(col[0], col[1], col[2]))
        y += 20
        for ln in _wrap(d, text, F_T, PANEL_W - 44):
            d.text((x + 4, y), ln, font=F_T, fill=(210, 216, 228) if who ==
                   "assistant" else (200, 206, 220))
            y += 20
        y += 6
        if y > PHONE_H - 60:
            break
    return img


def _verdict_card(ep):
    img = Image.new("RGB", (PANEL_W, PHONE_H), CARD)
    d = ImageDraw.Draw(img)
    x, y = 30, 40
    d.text((x, y), "EVALUATION", font=F_TAG, fill=MUT)
    y += 34
    col = _verdict_color(ep)
    if ep["family"] == "error":
        v = ep.get("detection_verdict")
        head = ("Caught & repaired" if ep.get("repaired") else
                "Detected" if v in ("correct", "early_correct") else
                "Missed — error committed" if ep.get("deviation_committed") else
                "Missed")
    else:
        head = ("Stayed quiet (correct)" if ep.get("restraint_verdict")
                != "false_alarm" else "Nagged on a fine change")
    d.text((x, y), head, font=_font(28, bold=True), fill=col)
    y += 52
    rows = [
        ("family", ep["family"]),
        ("detection", str(ep.get("detection_verdict"))),
        ("repaired", str(ep.get("repaired"))),
        ("error committed", str(ep.get("deviation_committed"))),
        ("goal", "met" if ep.get("goal_ok") else "not met"),
        ("nags on this run", str(ep.get("nags", 0))),
        ("coordinate leaks", str(ep.get("leaks", 0))),
        ("interface", ep.get("interface", "sv1")),
    ]
    for k, val in rows:
        d.text((x, y), k, font=F_S, fill=MUT)
        d.text((x + 220, y), val, font=F_M, fill=INK)
        y += 30
    y += 14
    d.line((x, y, PANEL_W - 30, y), fill=(40, 46, 58), width=1)
    y += 18
    d.text((x, y), ep.get("goal_detail", "")[:60], font=F_S, fill=MUT)
    return img


def _compose(phone_png, panel_img, caption):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    phone = Image.open(io.BytesIO(phone_png)).convert("RGB")
    if phone.size != (PHONE_W, PHONE_H):
        phone = phone.resize((PHONE_W, PHONE_H))
    canvas.paste(phone, (GAP, PAD))
    canvas.paste(panel_img, (GAP + PHONE_W + GAP, PAD))
    # caption strip at the very bottom of the phone
    if caption:
        who = caption.split(":", 1)[0]
        col = ACC if who == "assistant" else INK
        cy = H - 44
        _round(d, (GAP, cy - 4, GAP + PHONE_W, cy + 30), 8, (0, 0, 0))
    return canvas


def film_episode(ep, out_path, fps=2, scale=1):
    film = ep.get("_film") or []
    if not film:
        return None
    transcript = ep.get("transcript") or []
    tmp = tempfile.mkdtemp()
    # map each film frame to the transcript lines spoken up to its tick
    said = []
    frames = []
    tl = [e for e in ep.get("timeline", []) if e["kind"] == "say"]
    for i, (png, caption, tick, kind) in enumerate(film):
        upto = [(e["who"], e["text"]) for e in tl if e["t"] <= tick]
        panel = _panel(ep, upto, f"tick {tick}")
        img = _compose(png, panel, caption)
        # hold the "say" frames a beat longer so lines are readable
        reps = 3 if kind == "say" else 2
        for r in range(reps):
            fp = os.path.join(tmp, f"f{len(frames):04d}.png")
            img.save(fp)
            frames.append(fp)
    # verdict card, held
    vc = _verdict_card(ep)
    blank = Image.new("RGB", (PHONE_W, PHONE_H), (10, 12, 18))
    vimg = _compose(_png(blank), vc, "")
    for r in range(6):
        fp = os.path.join(tmp, f"f{len(frames):04d}.png")
        vimg.save(fp)
        frames.append(fp)
    # encode
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(tmp, "f%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_path], check=True)
    for fp in frames:
        os.remove(fp)
    os.rmdir(tmp)
    return out_path


def _png(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()
