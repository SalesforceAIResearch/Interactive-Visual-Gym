"""Presentation film of a recorded ScreenSim episode: a hand performs every gesture, the
screen moves the way a phone does, and every spoken line is voiced while the picture holds.

The engine is deterministic, so the record's timeline replays exactly on a fresh phone (the
same fidelity check as screensim_bench/video/replay.py). What this adds is the motion in between:

  hand     a pointing hand travels to each touch point, presses and lifts. Taps press a
           row, drags pull the content or a slider knob, typing taps the keyboard.
  screen   in-between frames are REAL renders or exact composites of them: a scroll is the
           content moving with the finger, a slider knob follows it, typed text appears
           letter by letter, opening a page slides it in from the right and Back slides it
           out. A switch that flips cross-fades.
  voice    each utterance is spoken (Gemini TTS, the same voices as the CookSim and
           VirtualHome cuts: assistant = Kore, person = Fenrir). While a line plays the
           picture HOLDS, the clock shows HOLD, and a caption band shows who is speaking.

    python screensim_bench/video/film.py tmp/pilot/sample_r8.json --contestant gpt-5.5 \\
        --episode lost_phone_lockdown__c2 --persona classic_novice --out ep.mp4
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave


import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from screensim.core.actions import LONG_PRESS, PRESS_KEY, SWIPE, TAP, TYPE, ground, parse
from screensim.core.engine import Device
from screensim.core.layout import content_bottom, content_top
from screensim.core.model import KEYBOARD_H, SCREEN_H, SCREEN_W, STATUS_H
from screensim.tasks import by_id, resolve_dynamic
import screensim_bench.video.episode_film as fe
import screensim_bench.video.replay as rf

FFMPEG = "ffmpeg"
FPS = 20

# ------------------------------------------------------------------ layout
PAD, GAP = 24, 24
PHONE_X, PHONE_Y = PAD, PAD
PANEL_X = PAD + SCREEN_W + GAP
PANEL_W = rf.PANEL_W
STAGE_H = PAD + SCREEN_H + 14          # the hand may hang below the phone, down to the band
PANEL_H = STAGE_H - PAD
BAND_Y, BAND_H = STAGE_H, 124
W = PANEL_X + PANEL_W + PAD
H = BAND_Y + BAND_H + 12
H += H % 2
W += W % 2
BG = rf.BG

SPEAKER_COLOR = {"assistant": (79, 163, 255), "human": (255, 201, 79)}
SPEAKER_LABEL = {"assistant": "ASSISTANT", "human": "PERSON"}

# ------------------------------------------------------------------ voice (as CookSim / VH)
SR = 24000
TTS_MODEL = "gemini-2.5-flash-preview-tts"
VOICE = {"assistant": "Kore", "human": "Fenrir"}
STYLE = ("Say this like a real person talking about their phone to someone next to them — "
         "casual everyday speech, quick natural pace, relaxed and conversational, "
         "not like reading from a script: ")
CACHE = os.environ.get("SCREENSIM_TTS_CACHE", "tmp/tts_cache_film")
# the TTS quota is per project per minute and shared with the CookSim and VH cuts on this
# box, so clips go through the same lock-file semaphore those tools use
LOCKDIR = os.path.join(tempfile.gettempdir(), "cooksim_tts_slots")
SLOTS = int(os.environ.get("COOKSIM_TTS_SLOTS", "4"))
GAP_S = 0.25                            # a breath after each line


@contextlib.contextmanager
def _tts_slot():
    os.makedirs(LOCKDIR, exist_ok=True)
    while True:
        for i in range(SLOTS):
            f = open(os.path.join(LOCKDIR, f"{i}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(0.7)


def tts_client():
    from google import genai
    return genai.Client(api_key=open(os.path.expanduser("~/.gemini_key")).read().strip())


def clean_speech(text):
    return re.sub(r"\*\*|__|`", "", text or "").strip()


def tts(client, text, who):
    """float32 mono 24 kHz for one line, disk-cached. Raises rather than returning silence:
    a silent 'voiced' line would misrepresent the transcript."""
    voice = VOICE[who]
    key = hashlib.md5((voice + "|screensim-film-v1|" + text).encode()).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    disk = os.path.join(CACHE, key + ".npy")
    if os.path.exists(disk):
        try:
            return np.load(disk)
        except Exception:
            with contextlib.suppress(OSError):
                os.remove(disk)
    from google.genai import types
    last = None
    for k in range(8):
        try:
            with _tts_slot():
                r = client.models.generate_content(
                    model=TTS_MODEL, contents=STYLE + text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))))
            pcm = r.candidates[0].content.parts[0].inline_data.data
            a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(a) < SR // 10 or np.abs(a).max() < 0.01:
                raise RuntimeError(f"degenerate audio ({len(a)} samples)")
            idx = np.where(np.abs(a) > 0.012)[0]
            if len(idx):
                a = a[max(0, idx[0] - SR // 20): idx[-1] + SR // 20]
            tmp = f"{disk}.{os.getpid()}.tmp.npy"
            np.save(tmp, a)
            os.replace(tmp, disk)                     # atomic: concurrent cuts share the cache
            return a
        except Exception as e:
            last = e
            time.sleep((12 + 8 * k) if "RESOURCE_EXHAUSTED" in str(e) else (3 + 3 * k))
    raise RuntimeError(f"TTS failed ({voice}): {last}")


# ------------------------------------------------------------------ the hand
HAND_ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hand.png")
HAND_SCALE = 0.36                       # hand.png is a 2x raster of a 240x380 drawing
HAND_HOT = (158, 32)                    # the fingertip, in hand.png pixels


class Hand:
    def __init__(self):
        base = Image.open(HAND_ASSET).convert("RGBA")
        self.hot = (HAND_HOT[0] * HAND_SCALE, HAND_HOT[1] * HAND_SCALE)
        self.img = {}
        self.shadow = {}
        for press in (0, 1):
            s = HAND_SCALE * (1.0 - 0.05 * press)
            im = base.resize((max(1, int(base.width * s)), max(1, int(base.height * s))), Image.LANCZOS)
            alpha = im.split()[3].point(lambda v: int(v * 0.93))
            im.putalpha(alpha)
            sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
            sh.putalpha(alpha.point(lambda v: int(v * 0.32)))
            self.img[press] = (im, (HAND_HOT[0] * s, HAND_HOT[1] * s))
            self.shadow[press] = sh.filter(ImageFilter.GaussianBlur(6))

    def draw(self, stage, x, y, press=0.0, lift=1.0):
        """Fingertip at phone coords (x, y). press 0..1 squashes the hand; lift 0..1 raises it
        off the glass, which moves its shadow away."""
        p = 1 if press >= 0.5 else 0
        im, hot = self.img[p]
        sx = int(round(PHONE_X + x - hot[0]))
        sy = int(round(PHONE_Y + y - hot[1]))
        dx, dy = int(5 + 12 * lift), int(8 + 16 * lift)
        stage.paste(self.shadow[p], (sx + dx, sy + dy), self.shadow[p])
        stage.paste(im, (sx, sy), im)


def ease(f):
    f = max(0.0, min(1.0, f))
    return f * f * (3 - 2 * f)


# ------------------------------------------------------------------ screen helpers
_MASK = None


def _screen_mask():
    global _MASK
    if _MASK is None:
        m = Image.new("L", (SCREEN_W, SCREEN_H), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, SCREEN_W - 1, SCREEN_H - 1), radius=52, fill=255)
        _MASK = m
    return _MASK


def snap(cap, dev):
    return Image.open(io.BytesIO(cap.snap(dev, fx_override=[]))).convert("RGB")


def slide(pre, post, f, push=True):
    """iOS navigation: the new page slides in over the old one, which drifts a third of the way."""
    f = ease(f)
    out = Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))
    if push:
        back, front, bx, fx_ = pre, post, -int(0.3 * SCREEN_W * f), int(SCREEN_W * (1 - f))
    else:
        back, front, bx, fx_ = post, pre, -int(0.3 * SCREEN_W * (1 - f)), int(SCREEN_W * f)
    out.paste(back, (bx, 0))
    dim = Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))
    out = Image.blend(out, dim, 0.10 * (f if push else 1 - f))
    shadow = Image.new("L", (18, SCREEN_H), 0)
    for i in range(18):
        ImageDraw.Draw(shadow).line((i, 0, i, SCREEN_H), fill=int(70 * (i / 18)))
    out.paste((0, 0, 0), (fx_ - 18, 0), shadow)
    out.paste(front, (fx_, 0))
    out.paste(post.crop((0, 0, SCREEN_W, STATUS_H)), (0, 0))      # the status bar never moves
    final = Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))
    final.paste(out, (0, 0), _screen_mask())
    return final


def scrolled(pre, post, top, bottom, d, f):
    """The content between the bars at an intermediate scroll: exact composite of the two real
    renders, since scrolling only translates the content (row y shows pre[y+o] or post[y+o-d])."""
    a, b = np.asarray(pre), np.asarray(post)
    out = b.copy()
    o = int(round(d * ease(f)))
    ys = np.arange(top, bottom)
    src_pre = ys + o
    use_pre = (src_pre >= top) & (src_pre < bottom)
    src_post = np.clip(ys + o - d, top, bottom - 1)
    out[top:bottom] = np.where(use_pre[:, None, None], a[np.clip(src_pre, 0, SCREEN_H - 1)], b[src_post])
    return Image.fromarray(out)


def _leaves(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _leaves(v, path + (k,))
    else:
        yield path, d


def _set(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def changed_leaves(a, b):
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    return [(p, la.get(p), lb[p]) for p in lb if la.get(p) != lb[p]]


# ------------------------------------------------------------------ gesture geometry
BACK_PT = (40, STATUS_H + 22)


def geometry(dev, text):
    """Where the finger goes for this gesture, from the screen it is performed on."""
    g = parse(text) if isinstance(text, str) else text
    if g is None:
        return {"kind": "none"}
    scr = dev.laid_out_screen()
    if g.verb in (TAP, LONG_PRESS):
        gr = ground(g, scr.widgets)
        if gr.ok and gr.widget is not None and gr.widget.bounds is not None:
            w, b = gr.widget, gr.widget.bounds
            x = (b.x2 - 41) if w.kind == "toggle" else b.cx
            return {"kind": "tap", "pt": (x, b.cy), "bounds": b, "wkind": w.kind}
        if gr.ok and gr.point:
            return {"kind": "tap", "pt": tuple(gr.point)}
        return {"kind": "tap", "pt": (SCREEN_W // 2, 470), "miss": True}
    if g.verb == PRESS_KEY:
        if g.key == "home":
            return {"kind": "home"}
        if g.key == "back":
            return {"kind": "tap", "pt": BACK_PT}
        return {"kind": "tap", "pt": (SCREEN_W // 2, SCREEN_H - 20)}
    if g.verb == SWIPE:
        if g.anchor_text:
            gr = ground(g, scr.widgets)
            if gr.ok and gr.widget is not None and gr.widget.bounds is not None:
                return {"kind": "slider", "bounds": gr.widget.bounds, "dir": g.direction,
                        "label": gr.widget.label}
        return {"kind": "scroll", "dir": g.direction}
    if g.verb == TYPE:
        return {"kind": "type", "text": g.text}
    return {"kind": "none"}


_QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def key_point(ch):
    """Roughly where a character sits on the on-screen keyboard."""
    top = SCREEN_H - KEYBOARD_H
    c = ch.lower()
    for r, row in enumerate(_QWERTY):
        if c in row:
            i = row.index(c)
            return (21 + i * 41 + r * 20, top + 38 + r * 54)
    if c.isdigit():
        n = 9 if c == "0" else int(c) - 1
        return (70 + (n % 3) * 136, top + 38 + (n // 3) * 54) if c != "0" else (206, top + 200)
    return (206, top + 200)                                          # space bar and the rest


# ------------------------------------------------------------------ panel and band
F_BAND_LAB = fe._font(16, True)
F_BAND_TXT = fe._font(22)
F_BAND_META = fe._font(14)
F_CLOCK = fe._font(16, True)

BAND_KIND = {
    "say_assistant": ("ASSISTANT", SPEAKER_COLOR["assistant"]),
    "say_human": ("PERSON", SPEAKER_COLOR["human"]),
    "slip": ("ENGINE · FORCED SLIP", rf.BAD), "takeover": ("ENGINE", rf.BAD),
    "blocked": ("PHONE DID NOT RESPOND", rf.WARN), "released": ("RELEASED", rf.GOOD),
    "call": ("PERSON'S PRIVATE JUDGEMENT", rf.GOOD), "move": ("PERSON", rf.MUT),
    "flag": ("ASSISTANT FLAG", rf.ACC), "unfirable": ("ENGINE", rf.FAINT),
    "tap": ("", rf.MUT), "start": ("", rf.MUT), "end": ("GRADE", rf.INK),
}


def band_image(record, kind, text, tick, hold):
    img = Image.new("RGB", (W, BAND_H), (16, 17, 20))
    d = ImageDraw.Draw(img)
    meta = f"{record['contestant']} · persona {record['persona']} · {record['task']} · engine_rev {record.get('engine_rev', '?')}"
    d.text((22, 10), meta, font=F_BAND_META, fill=(120, 126, 140))
    clock = f"tick {tick:02d}" + ("   || HOLD" if hold else "")
    d.text((W - 22 - d.textlength(clock, font=F_CLOCK), 9), clock, font=F_CLOCK,
           fill=(255, 120, 120) if hold else (150, 154, 165))
    label, col = BAND_KIND.get(kind, ("", rf.MUT))
    if text:
        d.rectangle((0, 0, 7, BAND_H), fill=col)
        if label:
            d.text((22, 36), label, font=F_BAND_LAB, fill=col)
        y = 58 if label else 40
        for ln in fe._wrap(d, text, F_BAND_TXT, W - 60)[:2]:
            d.text((22, y), ln, font=F_BAND_TXT, fill=(238, 240, 245))
            y += 29
    return img


def panel_image(record, log, ev, tick, final=False):
    rf.CAP_H = PANEL_H - rf.PHONE_H - 10        # the log fills exactly the panel's height here
    return rf._panel(record, log, ev, tick, final=final)


# ------------------------------------------------------------------ the film
class Film:
    def __init__(self, record, out_path, cap, client):
        self.r, self.out, self.cap, self.client = record, out_path, cap, client
        self.hand = Hand()
        self.hx, self.hy, self.lift = SCREEN_W * 0.72, SCREEN_H * 0.86, 1.0
        self.n = 0
        self.audio = []                 # (second, samples)
        self.cues = []
        self.log = []
        self.panel = None
        self.band = None
        self.phone = None
        self.tick = 0
        self.poster = None
        tmp_dir = tempfile.mkdtemp(prefix="screensim_film_")
        self.tmp_video = os.path.join(tmp_dir, "video.mp4")
        self.tmp_wav = os.path.join(tmp_dir, "audio.wav")
        self.proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "24", "-pix_fmt", "yuv420p", "-threads", "2", self.tmp_video],
            stdin=subprocess.PIPE)

    # ---- frame writing
    def frame(self, phone=None, hand=None, ripple=None, highlight=None, reps=1):
        phone = phone if phone is not None else self.phone
        canvas = Image.new("RGB", (W, H), BG)
        stage = Image.new("RGB", (PANEL_X - 6, STAGE_H), BG)
        ph = phone
        if highlight is not None:
            ph = phone.copy()
            ImageDraw.Draw(ph, "RGBA").rectangle(
                (highlight.x, highlight.y, highlight.x2, highlight.y2), fill=(0, 0, 0, 22))
        stage.paste(ph, (PHONE_X, PHONE_Y))
        if ripple is not None:
            (rx, ry), rad, col = ripple
            ImageDraw.Draw(stage, "RGBA").ellipse(
                (PHONE_X + rx - rad, PHONE_Y + ry - rad, PHONE_X + rx + rad, PHONE_Y + ry + rad),
                outline=col, width=3, fill=(col[0], col[1], col[2], 40))
        hx, hy, press, lift = hand if hand is not None else (self.hx, self.hy, 0.0, self.lift)
        self.hand.draw(stage, hx, hy, press, lift)
        canvas.paste(stage, (0, 0))
        canvas.paste(self.panel, (PANEL_X, PAD))
        canvas.paste(self.band, (0, BAND_Y))
        data = canvas.tobytes()
        for _ in range(reps):
            self.proc.stdin.write(data)
        self.n += reps
        return canvas

    def seconds(self):
        return round(self.n / FPS, 2)

    # ---- hand motion
    def travel(self, tx, ty, frames=7, phone=None):
        x0, y0, l0 = self.hx, self.hy, self.lift
        for i in range(1, frames + 1):
            f = ease(i / frames)
            self.frame(phone, hand=(x0 + (tx - x0) * f, y0 + (ty - y0) * f, 0.0, l0 + (1.0 - l0) * f))
        self.hx, self.hy, self.lift = tx, ty, 1.0

    def rest(self):
        """Lift off the glass and drift a little down and right, out of the way."""
        tx, ty = min(SCREEN_W - 40, self.hx + 26), min(SCREEN_H - 60, self.hy + 70)
        x0, y0 = self.hx, self.hy
        for i in range(1, 5):
            f = ease(i / 4)
            self.frame(hand=(x0 + (tx - x0) * f, y0 + (ty - y0) * f, 0.0, 0.3 + 0.7 * f))
        self.hx, self.hy, self.lift = tx, ty, 1.0

    # ---- events
    def set_event(self, ev, tick, band_kind=None, band_text=None, hold=False):
        self.tick = tick
        if ev is not None and ev["k"] not in ("end",):
            self.log.append(ev)
        self.panel = panel_image(self.r, self.log, ev, tick)
        self.band = band_image(self.r, band_kind or "", band_text or "", tick, hold)
        if ev is not None:
            tag, _col, body, _cap = rf._label(ev, self.r)
            self.cues.append({"s": self.seconds(), "tick": tick, "tag": tag, "text": body})

    def speak(self, ev, tick, samples):
        who = "assistant" if ev["k"] == "say_assistant" else "human"
        self.set_event(ev, tick, band_kind=ev["k"], band_text=ev["text"], hold=True)
        self.audio.append((self.n / FPS, samples))
        self.frame(reps=int(math.ceil((len(samples) / SR + GAP_S) * FPS)))

    def note(self, ev, tick, kind, text, seconds):
        self.set_event(ev, tick, band_kind=kind, band_text=text)
        self.frame(reps=max(1, int(round(seconds * FPS))))

    # ---- gestures
    def gesture(self, pre_dev, post_dev, geo, ev, slip=False, ok=True):
        pre, post = snap(self.cap, pre_dev), snap(self.cap, post_dev)
        self.phone = pre
        tone = rf.BAD if slip else (rf.ACC if ok else rf.WARN)
        kind = geo.get("kind")
        if kind == "scroll":
            self._scroll(pre_dev, post_dev, pre, post, geo, tone)
        elif kind == "slider":
            self._slider(pre_dev, post_dev, pre, post, geo, tone)
        elif kind == "type":
            self._type(pre_dev, post_dev, pre, post, geo)
        elif kind == "home":
            self._home(pre, post)
        else:
            self._tap(pre_dev, post_dev, pre, post, geo, tone)
        self.phone = post
        if self.poster is None and slip:
            self.poster = self.frame(reps=1)

    def _press(self, x, y, tone, phone, highlight=None, frames=3):
        for i in range(1, frames + 1):
            f = i / frames
            self.frame(phone, hand=(x, y, f, 1.0 - f),
                       ripple=((x, y), 10 + 10 * f, tone + (int(200 * f),)), highlight=highlight)

    def _tap(self, pre_dev, post_dev, pre, post, geo, tone):
        x, y = geo.get("pt", (SCREEN_W // 2, 470))
        self.travel(x, y, phone=pre)
        hl = geo.get("bounds") if geo.get("wkind") in ("row", "account", "button", "listitem") else None
        self._press(x, y, tone, pre, highlight=hl)
        self.frame(pre, hand=(x, y, 1.0, 0.0), ripple=((x, y), 22, tone + (160,)), highlight=hl, reps=2)
        a, b = pre_dev.st, post_dev.st
        pushed = (a.app == b.app and len(b.stack) > len(a.stack))
        popped = (a.app == b.app and len(b.stack) < len(a.stack))
        changed_screen = (a.app, a.cur_screen_key()) != (b.app, b.cur_screen_key())
        steps = 7 if (pushed or popped) else (5 if changed_screen else (3 if pre.tobytes() != post.tobytes() else 1))
        for i in range(1, steps + 1):
            f = i / steps
            if pushed or popped:
                ph = slide(pre, post, f, push=pushed)
            elif steps > 1:
                ph = Image.blend(pre, post, ease(f))
            else:
                ph = post
            self.frame(ph, hand=(x, y, 1.0 - f, f * 0.6),
                       ripple=((x, y), 22 + 8 * f, tone + (int(160 * (1 - f)),)))
        self.phone = post
        self.hx, self.hy, self.lift = x, y, 0.6
        self.rest()

    def _scroll(self, pre_dev, post_dev, pre, post, geo, tone):
        scr = pre_dev.laid_out_screen()
        top = content_top(scr.title is not None)
        bottom = content_bottom(bool(pre_dev.st.keyboard))
        key = pre_dev.st.cur_screen_key()
        d = post_dev.st.scroll.get(key, 0) - pre_dev.st.scroll.get(key, 0)
        same_screen = key == post_dev.st.cur_screen_key()
        up = geo.get("dir") != "down"
        travel = abs(d) if (same_screen and 60 <= abs(d) <= (bottom - top) - 20) else 180
        x = SCREEN_W // 2 + 30
        y0 = (bottom - 60) if up else (top + 60)
        y0 = min(max(y0, top + travel + 20) if up else y0, SCREEN_H - 60)
        y1 = y0 - travel if up else y0 + travel
        self.travel(x, y0, phone=pre)
        self._press(x, y0, tone, pre, frames=2)
        frames = 12
        for i in range(1, frames + 1):
            f = i / frames
            ph = scrolled(pre, post, top, bottom, d, f) if (same_screen and d) else pre
            self.frame(ph, hand=(x, y0 + (y1 - y0) * ease(f), 1.0, 0.0))
        self.phone = post
        self.hx, self.hy, self.lift = x, y1, 0.0
        self.rest()

    def _slider(self, pre_dev, post_dev, pre, post, geo, tone):
        b = geo["bounds"]
        diffs = [(p, v0, v1) for p, v0, v1 in changed_leaves(pre_dev.st.data, post_dev.st.data)
                 if isinstance(v1, (int, float)) and isinstance(v0, (int, float))]
        track0, track1 = b.x + 16, b.x2 - 16
        cy = b.y + b.h * 0.62
        if diffs:
            _p, v0, v1 = diffs[0]
            x0 = track0 + (track1 - track0) * max(0, min(100, v0)) / 100.0
            x1 = track0 + (track1 - track0) * max(0, min(100, v1)) / 100.0
        else:
            x0 = (track0 + track1) / 2
            x1 = x0 + (70 if geo.get("dir") == "right" else -70)
        self.travel(x0, cy, phone=pre)
        self._press(x0, cy, tone, pre, frames=2)
        mids = {}
        if diffs:
            for k in (1, 2, 3):
                dv = copy.deepcopy(post_dev)
                for p, a, c in diffs:
                    _set(dv.st.data, p, type(c)(a + (c - a) * k / 4))
                mids[k] = snap(self.cap, dv)
        frames = 10
        for i in range(1, frames + 1):
            f = i / frames
            ph = post if f >= 1 else (mids.get(min(3, int(f * 4))) or pre if diffs else pre)
            self.frame(ph, hand=(x0 + (x1 - x0) * ease(f), cy, 1.0, 0.0))
        self.phone = post
        self.hx, self.hy, self.lift = x1, cy, 0.0
        self.rest()

    def _type(self, pre_dev, post_dev, pre, post, geo):
        text = geo.get("text") or ""
        grown = [(p, v0 or "", v1) for p, v0, v1 in changed_leaves(pre_dev.st.data, post_dev.st.data)
                 if isinstance(v1, str) and v1.startswith(v0 or "") and len(v1) > len(v0 or "")]
        steps = []
        if grown:
            p, v0, v1 = grown[0]
            new = v1[len(v0):]
            stride = max(1, math.ceil(len(new) / 12))
            for k in range(stride, len(new) + stride, stride):
                dv = copy.deepcopy(post_dev)
                _set(dv.st.data, p, v0 + new[:min(k, len(new))])
                steps.append((new[min(k, len(new)) - 1], snap(self.cap, dv)))
        else:
            steps = [(c, post) for c in text[:12]] or [(" ", post)]
        cur = pre
        for ch, ph in steps:
            kx, ky = key_point(ch)
            x0, y0 = self.hx, self.hy
            for i in range(1, 3):
                f = ease(i / 2)
                self.frame(cur, hand=(x0 + (kx - x0) * f, y0 + (ky - y0) * f, 0.0, 0.4))
            self.frame(cur, hand=(kx, ky, 1.0, 0.0), ripple=((kx, ky), 14, rf.ACC + (170,)))
            cur = ph
            self.frame(cur, hand=(kx, ky, 0.0, 0.3))
            self.hx, self.hy, self.lift = kx, ky, 0.3
        self.phone = post
        self.frame(post, reps=2)
        self.rest()

    def _home(self, pre, post):
        x, y0, y1 = SCREEN_W // 2, SCREEN_H - 14, SCREEN_H - 220
        self.travel(x, y0, phone=pre)
        for i in range(1, 9):
            f = i / 8
            self.frame(Image.blend(pre, post, ease(f)), hand=(x, y0 + (y1 - y0) * ease(f), 1.0, 0.0))
        self.phone = post
        self.hx, self.hy, self.lift = x, y1, 0.0
        self.rest()

    def blocked(self, dev, geo, ev):
        ph = snap(self.cap, dev)
        self.phone = ph
        x, y = geo.get("pt", (SCREEN_W // 2, 470))
        if geo.get("kind") == "slider":
            b = geo["bounds"]
            x, y = (b.x + b.x2) / 2, b.y + b.h * 0.62
        self.travel(x, y, phone=ph)
        self._press(x, y, rf.WARN, ph)
        self.frame(ph, hand=(x, y, 1.0, 0.0), ripple=((x, y), 22, rf.WARN + (170,)), reps=6)
        self.hx, self.hy, self.lift = x, y, 0.0
        self.rest()

    # ---- finish
    def close(self, fidelity):
        self.proc.stdin.close()
        self.proc.wait()
        total = int(math.ceil(self.n / FPS * SR))
        track = np.zeros(total + SR, dtype=np.float32)
        for sec, a in self.audio:
            o = int(sec * SR)
            track[o:o + len(a)] += a[: max(0, len(track) - o)]
        peak = float(np.abs(track).max()) or 1.0
        pcm = (track[:total] / peak * 0.89 * 32767).astype(np.int16)
        with wave.open(self.tmp_wav, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SR)
            f.writeframes(pcm.tobytes())
        os.makedirs(os.path.dirname(os.path.abspath(self.out)) or ".", exist_ok=True)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", self.tmp_video, "-i", self.tmp_wav,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                        self.out], check=True)
        if self.poster is not None:
            self.poster.save(self.out[:-4] + ".jpg", quality=82)
        for p in (self.tmp_video, self.tmp_wav):
            with contextlib.suppress(OSError):
                os.remove(p)
        with contextlib.suppress(OSError):
            os.rmdir(os.path.dirname(self.tmp_video))
        return fidelity, self.cues, self.n / FPS


def film(record, out_path, cap, client=None):
    """Replay the record on a fresh phone and film it with the hand and the voices.
    Returns (fidelity, cues, seconds) like tools.replay_film.film."""
    client = client or tts_client()
    events = rf._events(record)
    for ev in events:
        if ev["k"].startswith("say_"):
            ev["text"] = clean_speech(ev["text"])
    clips = {}
    for ev in events:                                # synthesize first: know every hold up front
        if ev["k"].startswith("say_") and ev["text"]:
            who = "assistant" if ev["k"] == "say_assistant" else "human"
            clips[ev["i"], ev["text"]] = tts(client, ev["text"], who)
    by_idx = {}
    for ev in events:
        by_idx.setdefault(ev["i"], []).append(ev)

    task = by_id(record["task"])
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    fm = Film(record, out_path, cap, client)
    fm.phone = snap(cap, dev)
    fm.set_event(None, 0, band_kind="start", band_text="“" + record.get("instruction", "") + "”")
    fm.cues.append({"s": 0.0, "tick": 0, "tag": "START", "text": "episode starts"})
    fm.frame(reps=int(1.8 * FPS))

    mism = []
    for idx, e in enumerate(record["timeline"]):
        evs = by_idx.get(idx, [])
        if e["kind"] == "gesture":
            pre_dev = copy.deepcopy(dev)
            geo = geometry(pre_dev, resolve_dynamic(pre_dev, e["text"]))
            res = dev.act(resolve_dynamic(dev, e["text"]))
            if bool(res.ok) != bool(e.get("ok")) or dev.st.tick != e["t"]:
                mism.append({"t": e["t"], "text": e["text"], "logged_ok": e.get("ok"),
                             "replay_ok": bool(res.ok), "replay_tick": dev.st.tick})
            ev = evs[0] if evs else None
            if ev is not None and ev["k"] == "slip":
                b = (record.get("beats_design") or [{}] * 9)[ev["beat"]] if ev.get("beat") is not None else {}
                fm.set_event(ev, dev.st.tick, band_kind="slip",
                             band_text=f"{e['text']}   (meant: {b.get('correct', '?')})")
            elif ev is not None:
                fm.set_event(ev, dev.st.tick, band_kind="tap",
                             band_text=e["text"] + ("" if res.ok else "   (no effect)"))
            fm.gesture(pre_dev, dev, geo, ev, slip=bool(e.get("slip")), ok=bool(res.ok))
            continue
        if e["kind"] == "blocked":
            geo = geometry(dev, resolve_dynamic(dev, e["text"]))
            for ev in evs:
                fm.set_event(ev, dev.st.tick, band_kind="blocked",
                             band_text=f"{e['text']} — the slip is still held, so nothing happens")
            dev._tick(1)
            fm.blocked(dev, geo, evs[0] if evs else None)
            continue
        for ev in evs:
            k = ev["k"]
            if k.startswith("say_"):
                a = clips.get((ev["i"], ev["text"]))
                if a is not None:
                    fm.speak(ev, dev.st.tick, a)
                continue
            tag, col, body, _cap = rf._label(ev, record)
            if k == "takeover":
                fm.note(ev, dev.st.tick, "takeover", body, 0.9)
            elif k == "released":
                fm.note(ev, dev.st.tick, "released", body, 0.9)
            elif k == "call":
                fm.note(ev, dev.st.tick, "call", body, 1.0)
            elif k == "move":
                fm.note(ev, dev.st.tick, "move", body, 0.6)
            elif k == "flag":
                fm.note(ev, dev.st.tick, "flag", body, 0.8)
            elif k == "unfirable":
                fm.note(ev, dev.st.tick, "unfirable", body, 0.8)
            else:                                    # plans: the log updates, the film moves on
                fm.set_event(ev, dev.st.tick)
                fm.frame(reps=3)

    goal_ok = bool(task.goal(dev.st)[0])
    fidelity = {"mismatches": mism, "final_tick_logged": record.get("final_tick"),
                "final_tick_replay": dev.st.tick, "goal_logged": record.get("goal_ok"),
                "goal_replay": goal_ok,
                "exact": (not mism and goal_ok == bool(record.get("goal_ok"))
                          and dev.st.tick == record.get("final_tick"))}
    head = (f"goal {'met' if record.get('goal_ok') else 'NOT met'} · success {record.get('success')} · "
            f"{record.get('false_flags', 0)} false alarm(s)")
    fm.panel = panel_image(record, fm.log, None, dev.st.tick, final=True)
    fm.band = band_image(record, "end", head, dev.st.tick, False)
    fm.cues.append({"s": fm.seconds(), "tick": dev.st.tick, "tag": "GRADE", "text": head})
    last = fm.frame(reps=int(3.5 * FPS))
    if fm.poster is None:
        fm.poster = last
    return fm.close(fidelity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--contestant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = next(r for r in json.load(open(a.records)) if r.get("episode") == a.episode
               and r.get("persona") == a.persona and r.get("contestant") == a.contestant)
    from screensim.render.capture import Capturer
    with Capturer(scale=1.0) as cap:
        fid, cues, secs = film(rec, a.out, cap)
    json.dump({"fidelity": fid, "cues": cues, "seconds": secs}, open(a.out[:-4] + ".cues.json", "w"), indent=1)
    print(json.dumps(fid), f"{secs:.0f}s")


if __name__ == "__main__":
    main()
