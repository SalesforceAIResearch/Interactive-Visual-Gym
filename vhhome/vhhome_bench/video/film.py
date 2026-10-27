#!/usr/bin/env python3
"""Presentation cut of a VH interaction rollout: freeze-on-speech + TTS voice + captions.

Port of the CookSim `present_video.py` FREEZE treatment to VirtualHome. At every tick that
carries an utterance the picture HOLDS on that tick's last frame for exactly as long as the
speech clip runs, so each line is heard against a still, and the tick counter visibly stops.
Non-speaking ticks play at native speed (1 tick = 1 s = FPT frames at 5 fps).

Inputs are what iv_recovery_rollout.py already writes: the report JSON (tick-stamped
`conversation`) and its sibling .mp4 (ego | top-down composite).

  python -m vhhome_bench.video.film RUN.json [-o OUT.mp4]

Conventions kept from CookSim so the two benchmarks' cuts look like one family:
  assistant (vlm) = voice Kore,   caption #4FA3FF
  human           = voice Fenrir, caption #FFC94F

TTS quota is per PROJECT per MINUTE and shared with any other process on this box (the cook-sim
agent hits the same project). Clips go through a counting semaphore in lock files and are
disk-cached, so a re-cut costs only the encode.
"""
from vhhome import paths
import argparse, hashlib, json, math, os, re, subprocess, sys, time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 24000
FPS = 5                     # source fps = frames per tick (iv_world.FPT = 5 since engine f75: 1 tick = 1 s of animation)
TTS_MODEL = "gemini-2.5-flash-preview-tts"
VOICE = {"vlm": "Kore", "human": "Fenrir"}
LABEL = {"vlm": "ASSISTANT (VLM)", "human": "HUMAN AGENT"}
COLOR = {"vlm": (79, 163, 255), "human": (255, 201, 79)}
STYLE = ("Say this in a natural, conversational tone, at a brisk everyday pace, "
         "as one person talking to another in their kitchen: ")
CACHE = os.path.join(paths.RUNS, "tts_cache")
LOCKDIR = os.path.join(paths.RUNS, ".tts_slots")
SLOTS = int(os.environ.get("COOKSIM_TTS_SLOTS", 4))
GAP_S = 0.25                # breath between consecutive lines on the same tick
BAND_H = 132                # caption band drawn under the video


# ---------------------------------------------------------------- TTS
def _slot():
    import fcntl, contextlib
    @contextlib.contextmanager
    def _cm():
        os.makedirs(LOCKDIR, exist_ok=True)
        while True:
            for i in range(SLOTS):
                f = open(os.path.join(LOCKDIR, f"{i}.lock"), "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    f.close(); continue
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN); f.close()
                return
            time.sleep(0.7)
    return _cm()


def tts(client, text, speaker):
    """float32 mono 24k for one line, disk-cached. Raises after retries rather than
    returning silence -- a silent 'voiced' line would misrepresent the transcript."""
    key = hashlib.md5((VOICE[speaker] + "|iv1|" + text).encode()).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    disk = os.path.join(CACHE, key + ".npy")
    if os.path.exists(disk):
        return np.load(disk)
    from google.genai import types
    last = None
    for k in range(8):
        try:
            with _slot():
                # spoken form only (the caption keeps the written line): "bedroom #2" makes the voice model
                # return an empty candidate, "bedroom number 2" does not
                spoken = re.sub(r"#\s*(\d+)", r"number \1", text)
                r = client.models.generate_content(
                    model=TTS_MODEL, contents=STYLE + spoken,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE[speaker])))))
            pcm = r.candidates[0].content.parts[0].inline_data.data
            a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(a) < SR // 10 or np.abs(a).max() < 0.01:
                raise RuntimeError(f"degenerate audio ({len(a)} samples)")
            idx = np.where(np.abs(a) > 0.012)[0]
            if len(idx):
                a = a[max(0, idx[0] - SR // 20): idx[-1] + SR // 20]
            np.save(disk, a)
            return a
        except Exception as e:
            last = e
            # a 429 is shared-quota contention: back off much harder than a transient error
            time.sleep((12 + 8 * k) if "RESOURCE_EXHAUSTED" in str(e) else (3 + 3 * k))
    raise RuntimeError(f"TTS failed ({speaker}): {last}")


# ---------------------------------------------------------------- drawing
def _font(sz, mono=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("Mono" if mono else ""),
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _wrap(draw, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines[:3]


def compose(frame, tick, speaker, text, frozen, meta):
    """Video frame + caption band. `frozen` drives the HOLD badge and stops the tick clock."""
    h, w = frame.shape[:2]
    im = Image.new("RGB", (w, h + BAND_H), (16, 17, 20))
    im.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(im)
    f_small, f_lab, f_txt = _font(15), _font(17), _font(21)
    f_mono = _font(16, mono=True)

    d.text((12, h + 10), meta, font=f_small, fill=(150, 154, 165))
    # ASCII only: DejaVu has no U+23F8 and it renders as tofu
    clock = f"t={tick:03d}" + ("  || HOLD" if frozen else "")
    d.text((w - 12 - d.textlength(clock, font=f_mono), h + 10), clock, font=f_mono,
           fill=(255, 120, 120) if frozen else (150, 154, 165))
    if text:
        col = COLOR[speaker]
        d.rectangle([0, h, 6, h + BAND_H], fill=col)
        d.text((16, h + 34), LABEL[speaker], font=f_lab, fill=col)
        y = h + 58
        for ln in _wrap(d, text, f_txt, w - 32):
            d.text((16, y), ln, font=f_txt, fill=(238, 240, 245)); y += 26
    return np.asarray(im)


# ---------------------------------------------------------------- main
def _atempo(a, tempo):
    """Pitch-preserving speed-up of a float32 mono clip via ffmpeg atempo (chained above 2x)."""
    chain = []
    t = tempo
    while t > 2.0:
        chain.append("atempo=2.0"); t /= 2.0
    chain.append(f"atempo={t:.4f}")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                        "-filter:a", ",".join(chain), "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                       input=np.asarray(a, dtype=np.float32).tobytes(), stdout=subprocess.PIPE, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def build(run_json, out_mp4):
    import imageio.v2 as imageio
    rep = json.load(open(run_json))
    src = run_json.replace(".json", ".mp4")
    if not os.path.exists(src):
        sys.exit(f"missing source video {src}")

    # utterances per tick, in order, skipping the empty 'silent decision' entries
    per_tick = {}
    for c in rep.get("conversation", []):
        t, sp, tx = c.get("t"), c.get("speaker"), (c.get("text") or "").strip()
        if t is None or not tx:
            continue
        per_tick.setdefault(int(t), []).append((sp if sp in VOICE else "vlm", tx))

    from vhhome_bench import protocol as P
    client = P.make_client()

    # synthesize first: fail before encoding, and know every hold length up front
    clips = {}
    n_lines = sum(len(v) for v in per_tick.values())
    print(f"[present] {os.path.basename(run_json)}: {n_lines} spoken lines", flush=True)
    tempo = float(os.environ.get("IV_TTS_TEMPO", "1.0") or 1.0)     # talk deck 2026-10-08: faster speech, pitch kept
    for t, uts in sorted(per_tick.items()):
        for i, (sp, tx) in enumerate(uts):
            a = tts(client, tx, sp)
            if abs(tempo - 1.0) > 1e-3:
                a = _atempo(a, tempo)
            clips[(t, i)] = a
    print(f"[present] TTS done ({len(clips)} clips)", flush=True)

    meta = (f"{rep.get('scenario','')} · {rep.get('persona','')} · "
            f"{rep.get('err','')} · {rep.get('episode','')}")
    rd = imageio.get_reader(src)
    tmp = out_mp4 + ".video.mp4"
    audio_parts, n_out = [], 0
    with imageio.get_writer(tmp, fps=FPS, macro_block_size=None,
                            output_params=["-threads", "2"]) as w:
        tick, last = 0, None
        tick_start = {}                     # tick -> second in the finished film (clip cutting)
        for k, fr in enumerate(rd):
            tick = k // FPS
            last = fr
            if k % FPS == 0:
                tick_start[tick] = round(n_out / FPS, 3)
            w.append_data(compose(fr, tick, None, None, False, meta)); n_out += 1
            # end of this tick -> hold for each of its lines
            if (k + 1) % FPS == 0:
                for i, (sp, tx) in enumerate(per_tick.get(tick, [])):
                    a = clips[(t := tick, i)]
                    hold = int(math.ceil((len(a) / SR + GAP_S) * FPS))
                    audio_parts.append((n_out / FPS, a))
                    for _ in range(hold):
                        w.append_data(compose(last, tick, sp, tx, True, meta)); n_out += 1
    rd.close()

    # audio bed
    total = int(math.ceil(n_out / FPS * SR)) + SR
    track = np.zeros(total, dtype=np.float32)
    for off_s, a in audio_parts:
        o = int(off_s * SR)
        track[o:o + len(a)] += a[: max(0, total - o)]
    peak = float(np.abs(track).max()) or 1.0
    track = (track / peak * 0.89 * 32767).astype(np.int16)
    wav = out_mp4 + ".wav"
    import wave
    with wave.open(wav, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(track.tobytes())

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", wav,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-threads", "2", "-c:a", "aac", "-b:a", "128k", "-shortest", out_mp4],
                   check=True)
    os.remove(tmp); os.remove(wav)
    # sidecar: where each tick starts in THIS film, so a clip can be cut by transcript tick
    json.dump({"fps": FPS, "tick_start_s": tick_start}, open(out_mp4.replace(".mp4", ".ticks.json"), "w"))
    # never trust the writer's own "wrote ..." -- probe the finalised file (CookSim lesson)
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", out_mp4], capture_output=True, text=True)
    print(f"[present] wrote {out_mp4}  frames={n_out}  dur={p.stdout.strip()}s", flush=True)
    return out_mp4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    build(a.run_json, a.out or a.run_json.replace(".json", "_present.mp4"))
