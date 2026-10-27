#!/usr/bin/env python3
"""Presentation-quality episode video: dual view + per-utterance TTS voice + ASS captions.

Consumes an episode report (conversation with tick-stamped utterances) plus the two
filmed frame sequences from film_frames.py (LEFT = egocentric, RIGHT = top-down), and
produces a single mp4 with:
  - voice-broadcast dialogue (Vertex Gemini TTS v2: assistant=Kore, cook=Fenrir, both
    with a casual-kitchen style prefix + per-clip pace gate), never-overlapping clips
  - time-aligned color-coded burned-in ASS captions (assistant #4FA3FF, cook #FFC94F)
  - two treatments:
      pause  : 1.5x playback (1 tick = 0.8 s); cook speech fits its in-episode frozen
               ticks; assistant speech plays over continued action
      freeze : at each utterance tick the video HOLDS the frame while the clip plays;
               a monospace tick counter visibly stops during holds
  - title / end cards, room tone, pane labels

    python -m cooksim_bench.video.present_video --report tmp/present/stepin_hard.json \
        --ego tmp/present/frames/stepin_ego --top tmp/present/frames/stepin_top \
        --mode freeze --out tmp/present/stepin_freeze.mp4 --title "..." --workdir ...
"""
import argparse, contextlib, hashlib, json, math, os, re, subprocess, sys, tempfile, time

import numpy as np

FFMPEG = os.path.expanduser("~/micromamba/envs/cooksim/bin/ffmpeg")
if not os.path.exists(FFMPEG):
    FFMPEG = "ffmpeg"
SR = 24000
TTS_MODEL = "gemini-2.5-flash-preview-tts"
VOICE = {"vlm": "Kore", "human": "Fenrir"}   # v2: casual voices (may be overridden by CLI)
LABEL = {"vlm": "ASSISTANT", "human": "COOK"}
TTS_CACHE = "tmp/tts_cache_present2"         # v2: fresh cache — old clips were too slow
TICK_S = 0.8                                 # v3 (2026-07-31): playback 1.5x game
                                             # speed. Audio is sped by the SAME
                                             # factor (--audio-speed-auto), like a
                                             # video player, so the audio clock and
                                             # the tick clock never disagree.
# --no-hold: max seconds a dialogue group may start after its own tick. Raising this
# is the ONLY way to de-collide speech on a cut built from the harness's per-tick
# capture: those cuts have exactly ONE rendered frame per tick, so --fit-local's
# slow-motion holds that single frame and the picture literally freezes (verified:
# a 2.0 s frozen run at 45 s in a 155 s cut). Letting a line start a little later
# instead drifts it from its tick by a second or two while the world keeps moving —
# strictly the better trade, since a frozen kitchen is the artifact the user
# actually complained about. COOKSIM_MAX_SLIP overrides.
MAX_SLIP = float(os.environ.get("COOKSIM_MAX_SLIP", 2.5))   # --no-hold: max seconds
                                             # start after its own tick before it is allowed
                                             # to overlap the previous line instead
# --fit-local: ceiling on how long ONE source frame may be shown. verify_delivery samples
# at 2 fps and fails a run of more than 3 identical frames (>1.5 s). A cut built from the
# harness's per-tick capture has exactly ONE frame per tick, so spreading a group's speech
# overflow "over the ticks it covers" can land the whole of it on a single still — measured
# as a 2.0 s frozen run at 45 s in a 155 s cut. Capping the per-frame dwell and spilling the
# remainder onto later frames keeps the picture moving without giving up the no-overlap
# guarantee. COOKSIM_FIT_MAX_FRAME overrides.
FIT_MAX_FRAME = float(os.environ.get("COOKSIM_FIT_MAX_FRAME", 1.3))
SPEED_X = 1.2 / TICK_S                       # honest disclosure factor vs true game speed
FONT = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"
# 25, not 40. Every frame here is a still repeated for the whole tick, so the extra 15
# fps buy nothing visible — but they are 60% more raw RGB the encoder has to chew, and on
# a box whose spare cores are already down to six the encoder IS the bottleneck (measured:
# ffmpeg pinned at 45% of one core while the frame generator idled at 2%). The page copies
# are transcoded to 24 fps regardless, so this changes the cost, not the deliverable.
FPS = 25
# v2 casual-delivery style prefix (verified: lifts pace to 3.0-3.7 w/s, judged 9/10 casual)
STYLE = ("Say this like a real person in a busy kitchen talking to a coworker — "
         "casual everyday speech, quick natural pace, relaxed and conversational, "
         "not like reading from a script: ")
STYLE_BRISK = STYLE.replace("quick natural pace", "brisk, snappy pace")
PACE_GATE_WPS = 3.2            # regenerate once (brisk) below this; keep the faster take

# ------------------------------------------------------------------ TTS
# The TTS quota is per PROJECT per minute, so it is shared by every cut running on the
# box, not by the threads inside one of them. Four cuts x four threads exhausted it and
# killed an episode outright (429 RESOURCE_EXHAUSTED, five retries, give up). This is a
# counting semaphore held in lock FILES so the limit holds ACROSS processes.
TTS_SLOTS = int(os.environ.get("COOKSIM_TTS_SLOTS", "4"))
_TTS_LOCKDIR = os.path.join(tempfile.gettempdir(), "cooksim_tts_slots")


@contextlib.contextmanager
def _tts_slot():
    import fcntl
    os.makedirs(_TTS_LOCKDIR, exist_ok=True)
    while True:
        for i in range(TTS_SLOTS):
            f = open(os.path.join(_TTS_LOCKDIR, f"{i}.lock"), "w")
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


def _tts_once(client, text, voice, style):
    """One styled TTS generation -> float32 mono 24k, silence-trimmed."""
    from google.genai import types
    last = None
    for k in range(8):
        try:
            with _tts_slot():
                r = client.models.generate_content(
                    model=TTS_MODEL, contents=style + text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice)))))
            pcm = r.candidates[0].content.parts[0].inline_data.data
            a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(a) < SR // 10 or np.abs(a).max() < 0.01:
                raise RuntimeError(f"degenerate audio ({len(a)} samples)")
            return trim_silence(a)
        except Exception as e:
            last = e
            # a 429 is a shared-quota problem: back off much harder than a transient error
            time.sleep((12 + 8 * k) if "RESOURCE_EXHAUSTED" in str(e) else (3 + 3 * k))
    raise RuntimeError(f"TTS failed ({voice}): {last}")


def tts_wav(client, text, speaker):
    """Styled casual TTS with per-clip pace gate (>= PACE_GATE_WPS words/s or the
    faster of two takes). Returns float32 mono 24k samples. Cached on disk."""
    key = hashlib.md5((VOICE[speaker] + "|v2casual|" + text).encode()).hexdigest()
    os.makedirs(TTS_CACHE, exist_ok=True)
    disk = os.path.join(TTS_CACHE, key + ".npy")
    if os.path.exists(disk):
        try:
            return np.load(disk)
        except Exception:
            # A clip killed mid-write leaves a truncated .npy, and every later cut that needs
            # that line dies on it ("could only read N elements"). The cache is a cache: drop
            # the bad entry and synthesize it again.
            try:
                os.remove(disk)
            except OSError:
                pass
    nw = max(1, len(text.split()))
    a = _tts_once(client, text, VOICE[speaker], STYLE)
    wps = nw / (len(a) / SR)
    if wps < PACE_GATE_WPS:
        b = _tts_once(client, text, VOICE[speaker], STYLE_BRISK)
        wps_b = nw / (len(b) / SR)
        if wps_b > wps:
            a, wps = b, wps_b
    # Atomic: write to a unique temp file in the same directory, then rename. np.save straight
    # to `disk` is visible half-written, and with several cuts running at once a second cutter
    # that needs the SAME line reads the partial file and dies with "could only read N elements".
    # That is what stalled the publish loop — the cache is shared across every concurrent cut.
    tmp = f"{disk}.{os.getpid()}.tmp.npy"
    np.save(tmp, a)
    os.replace(tmp, disk)
    return a


def trim_silence(a, thr=0.012, pad_ms=60):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    pad = int(SR * pad_ms / 1000)
    lo = max(0, idx[0] - pad)
    hi = min(len(a), idx[-1] + pad)
    return a[lo:hi]


def atempo(a, tempo):
    """Pitch-preserving speed-up via ffmpeg atempo."""
    if abs(tempo - 1.0) < 1e-3:
        return a
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        f.write((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
        src = f.name
    dst = src + ".out.raw"
    # -threads 1 / -filter_threads 1: this is a one-shot mono resample of an audio buffer and
    # gains nothing from parallelism, but ffmpeg's default is a thread per core and this box
    # runs under a 1024-PID cgroup shared with the render jobs. Left uncapped it fails with
    # "pthread_create() failed: Resource temporarily unavailable" and takes the whole cut
    # down after the TTS has already been paid for.
    subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                    "-threads", "1", "-filter_threads", "1",
                    "-f", "s16le", "-ar", str(SR),
                    "-ac", "1", "-i", src, "-filter:a", f"atempo={tempo:.4f}",
                    "-f", "s16le", dst], check=True)
    out = np.frombuffer(open(dst, "rb").read(), dtype=np.int16).astype(np.float32) / 32768.0
    for p in (src, dst):
        try:
            os.unlink(p)
        except OSError:
            pass
    return out


# ------------------------------------------------------------------ captions
def esc_ass(s):
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


_MEASURE = {}

def _text_w(s, size=41):
    """Measured pixel width at PlayRes scale (bold DejaVu = safe over-estimate)."""
    if "font" not in _MEASURE:
        from PIL import ImageFont
        _MEASURE["font"] = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    return _MEASURE["font"].getlength(s)


ROW_W = 1500          # usable caption row width (PlayResX 1920 - margins - safety)


def _fits(s, rows=2):
    """Would libass wrap this into <= `rows` rows? (greedy word wrap, measured)."""
    lines, cur = 1, ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if _text_w(t) > ROW_W and cur:
            lines += 1
            cur = w
        else:
            cur = t
    return lines <= rows


def chunk_text(text, prefix=""):
    """Sentence-aware caption pages, each guaranteed <= 2 measured rows
    (prefix = the speaker tag that shares the first row)."""
    sents = re.findall(r"[^.!?]+[.!?]*\s*", text) or [text]
    sents = [s.strip() for s in sents if s.strip()]
    pages, cur = [], ""
    def full(p):                       # what actually renders on the page
        return (prefix + " — " + p) if prefix else p
    for s in sents:
        cand = (cur + " " + s).strip()
        if cur and not _fits(full(cand)):
            pages.append(cur)
            cur = s
        else:
            cur = cand
        while cur and not _fits(full(cur)):          # oversize sentence: split by words
            words = cur.split()
            keep = len(words)
            while keep > 1 and not _fits(full(" ".join(words[:keep]))):
                keep -= 1
            pages.append(" ".join(words[:keep]))
            cur = " ".join(words[keep:])
    if cur:
        pages.append(cur)
    return pages or [text]


def ts(t):
    t = max(0.0, t)
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Assist,{font},41,&H00FFA34F,&H00FFFFFF,&H00101010,&H96000000,0,0,0,0,100,100,0,0,1,2.2,1,2,150,150,26,1
Style: Cook,{font},41,&H004FC9FF,&H00FFFFFF,&H00101010,&H96000000,0,0,0,0,100,100,0,0,1,2.2,1,2,150,150,26,1
Style: Pane,{font},27,&H00F2F2F2,&H00FFFFFF,&H00000000,&H50000000,1,0,0,0,100,100,0,0,3,4,0,7,18,18,14,1
Style: Tick,{mono},31,&H00E8E8E8,&H00FFFFFF,&H00000000,&H60000000,1,0,0,0,100,100,0,0,3,5,0,9,18,24,14,1
Style: Note,{font},25,&H00D8D8D8,&H00FFFFFF,&H00101010,&H80000000,0,1,0,0,100,100,0,0,3,4,0,2,20,20,12,1
Style: Side,{font},25,&H00B8C4D2,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1.4,0,7,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, Text
"""


# ------------------------------------------------------------------ timeline
def load_utts(report_path):
    r = json.load(open(report_path))
    utts = []
    for c in r["conversation"]:
        txt = (c.get("text") or "").strip()
        if not txt or c["speaker"] not in ("vlm", "human"):
            continue
        utts.append({"t": int(c["t"]), "speaker": c["speaker"], "text": txt})
    return r, utts


def build(args):
    global TICK_S, SPEED_X    # --fit-speech rebinds the per-tick duration (and with it
                              # the honest playback factor burned into the note/end card)
    rep, utts = load_utts(args.report)
    tm = json.load(open(os.path.join(args.single or args.top, "tickmap.json")))
    ticks, subs = tm["ticks"], tm["subs"]
    n = len(ticks)
    base = TICK_S / subs
    t2j = {t: j for j, t in enumerate(ticks)}

    def state_idx(t):
        while t not in t2j and t > 0:
            t -= 1
        return t2j.get(t, 0)

    # ---- TTS all utterances
    from cooksim_bench.backends.genai_client import make_client
    client = make_client(120_000)
    print(f"TTS: {len(utts)} utterances ...", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        clips = list(ex.map(lambda u: tts_wav(client, u["text"], u["speaker"]), utts))
    # PLAYBACK-MATCHED AUDIO. The video is a SPEED_X playback of the episode, so the
    # speech must be sped up by the same factor — exactly what a video player does when
    # you watch at 2x. Without it the two clocks disagree and the mismatch has to go
    # somewhere: it went into frozen frames (the world stopped while a line finished) and
    # then, once freezing was banned, into overlapping voices.
    #   The engine's pause model already budgets speaking time — but it budgets it in GAME
    # time (words / 3.15 wps / 1.2 s-per-tick), so at 0.6 s/tick only half that budget
    # survives, and it charges the COOK only: listening is free by design, while in a
    # measured episode the assistant produced 58% of all words.
    # Speeding the audio to match playback restores the cook half exactly and shrinks the
    # unbudgeted assistant half by the same factor.
    tempo = args.audio_atempo
    if getattr(args, "audio_speed_auto", False):
        # 1.5 at the 0.8 s/tick default; --audio-atempo multiplies on top (talk deck 2026-10-08: "voice needs to speak faster")
        tempo = SPEED_X * (args.audio_atempo or 1.0)
        print(f"[audio] speeding speech to {tempo:.2f}x to match {SPEED_X:g}x playback",
              flush=True)
    for u, a in zip(utts, clips):
        if tempo and abs(tempo - 1.0) > 1e-3:
            a = atempo(a, min(tempo, 2.5))    # cap: atempo stays clean well past 2x
        u["audio"] = a
        u["dur"] = len(a) / SR
        wps = len(u["text"].split()) / u["dur"]
        print(f"  t{u['t']:>3} {u['speaker']:<5} {u['dur']:5.2f}s {wps:4.2f}w/s  "
              f"{u['text'][:60]}", flush=True)

    frame_durs = np.full(n * subs, base, dtype=np.float64)
    events = []          # {start, end, speaker, text, audio}
    tick_ev = []         # (start, end, tick)

    by_tick = {}
    for u in utts:
        by_tick.setdefault(state_idx(u["t"]), []).append(u)
    grp_states = sorted(by_tick)
    tail_over_card = 0.0     # --no-hold: trailing audio time carried by the END CARD

    # ---- FIT-SPEECH: uniform stretch instead of freezing or overlapping ---------------
    # The two artifacts reported are the SAME problem from opposite sides. Holding the
    # frame for dialogue overflow froze the world (pans stopped cooking); banning holds
    # made voices overlap, because a talky episode carries 1.6-4.4x more speech than its
    # body has playing time at 2x. Neither is necessary: stretch EVERY tick by the same
    # factor and the world keeps moving continuously (no freeze) while every line has room
    # to finish before the next begins (no overlap). Talky episodes simply play slower —
    # which is honest, because they really did contain that much conversation.
    if getattr(args, "fit_speech", False) and args.mode == "pause" and grp_states:
        need = TICK_S
        for a_, b_ in zip(grp_states, grp_states[1:] + [n]):
            span = max(1, b_ - a_)
            total = sum(u["dur"] + 0.3 for u in by_tick.get(a_, [])) + 0.4
            need = max(need, total / span)
        if need > TICK_S:
            TICK_S = round(need, 3)
            # the burned-in note and the end card quote SPEED_X, so it has to follow the
            # stretch — otherwise a fit-speech cut claims a 2x playback it no longer has
            SPEED_X = round(1.2 / TICK_S, 3)
            # frame_durs was filled from the OLD per-tick duration a few lines above, so
            # rebinding TICK_S alone changed nothing about the picture (the cut stayed at
            # 0.6 s/tick while the note claimed the stretch). Refill it.
            base = TICK_S / subs
            frame_durs[:] = base
            print(f"[fit-speech] {TICK_S:.2f}s/tick ({TICK_S/0.8:.2f}x slower than the "
                  f"baseline) so all {len(utts)} lines fit with no overlap and no freeze",
                  flush=True)

    if args.mode == "pause":
        # Baseline pacing 1 tick = TICK_S (v2: 0.85 s ~= 1.4x). Assistant speech rides over live
        # action; a frame HOLD is inserted only for the OVERFLOW — the part of a
        # dialogue group that cannot finish before the next group's tick. The tick
        # counter dwells during those holds, so the stretch is honest and visible.
        cum, cur_end = 0.0, 0.0
        for j in range(n):
            s0 = j * subs
            here = by_tick.get(j, [])
            if here:
                a0 = max(cum + 0.4, cur_end + 0.2)
                if getattr(args, "fit_local", False):
                    # SERIALISED BY CONSTRUCTION. --fit-local pays for the overflow with
                    # picture time, so a group never needs to start before the previous one
                    # has finished and the MAX_SLIP clamp below must NOT apply: clamping a
                    # group back onto its own tick is precisely what re-introduces two
                    # voices at once. Slipping late is fine here because the picture slips
                    # with it (the frames covering the slip are stretched by the same total).
                    pass
                elif args.no_hold:
                    # Bounded slip. Serialising every group behind the previous one lets the
                    # audio clock drift away from the tick clock without limit: on a talky
                    # episode it ran 155 s past the last tick, and the only way to cover that
                    # was to hold the FINAL frame for those 155 s — the exact freeze this
                    # flag exists to prevent. Anchoring each group to its own tick caps the
                    # lateness, so drift can never accumulate; a line that still overruns
                    # simply plays on over the following ticks (the audio mixer is additive)
                    # instead of pushing the rest of the episode later.
                    a0 = min(a0, cum + 0.4 + MAX_SLIP)
                tmp = []
                for u in here:
                    tmp.append((a0, u))
                    a0 += u["dur"] + 0.3
                group_end = a0 - 0.3
                nxt = next((g for g in grp_states if g > j), None)
                natural_next = cum + ((nxt - j) if nxt is not None else (n - j)) * TICK_S
                hold = max(0.0, group_end + 0.3 - natural_next)
                if getattr(args, "fit_local", False):
                    # LOCAL SLOW-MOTION instead of a hold or a global stretch. The overflow
                    # is spread over the frames from here on, so the picture keeps MOVING
                    # (no frozen frame — the standing rule) while the group still finishes
                    # before the next one starts (no overlapping voices). Only the talky
                    # stretches slow down; the rest of the episode keeps the playback pace,
                    # unlike --fit-speech which stretches the whole episode by its single
                    # densest tick.
                    #
                    # PER-FRAME CAP (v2). Spreading a group's overflow over only the frames
                    # up to the NEXT dialogue group is fine when a tick owns eight rendered
                    # subframes, but a single-camera cut has ONE frame per tick, so two
                    # adjacent talky ticks left the whole overflow on one still: a measured
                    # 2.0 s hold, which is the very artifact this flag exists to prevent.
                    # Frames are therefore filled forward to a ceiling, spilling past the
                    # next group when the local window is too small to take it gently. The
                    # spill costs nothing in correctness because groups are serialised above
                    # rather than pinned to their ticks.
                    span_f = (((nxt - j) if nxt is not None else (n - j)) * subs)
                    have = cum + float(frame_durs[s0:s0 + span_f].sum())
                    need = group_end + 0.3 - have
                    k = s0
                    while need > 1e-6 and k < n * subs:
                        room = FIT_MAX_FRAME - frame_durs[k]
                        if room > 0:
                            add = min(room, need)
                            frame_durs[k] += add
                            need -= add
                        k += 1
                    # whatever the picture could not absorb rides the END CARD (handled
                    # below by tail_over_card) — a card is a legitimate still, a hung
                    # kitchen is not
                if hold < 0.3:
                    hold = 0.0
                if args.no_hold:
                    # STANDING RULE: the world never stops in the engine — only the
                    # cook's hands do — so the video must never freeze mid-episode.
                    # Every tick plays for exactly TICK_S and long speech simply rides
                    # over the following ticks (captions stay anchored to the tick the
                    # line was spoken on).
                    hold = 0.0
                frame_durs[s0] += hold
                for st_a, u in tmp:
                    events.append(dict(start=st_a, end=st_a + u["dur"],
                                       speaker=u["speaker"], text=u["text"],
                                       audio=u["audio"]))
                cur_end = group_end
            st = cum
            cum += float(frame_durs[s0:s0 + subs].sum())
            tick_ev.append((st, cum, ticks[j]))
        if cur_end + 0.6 > cum:                        # speech outlives the last tick
            if args.no_hold:
                # Same standing rule as the mid-episode holds: never freeze the KITCHEN.
                # The leftover audio plays over the END CARD instead (a card is a legitimate
                # still; a hung kitchen is the artifact the user caught).
                tail_over_card = cur_end + 1.0 - cum
            else:
                frame_durs[-1] += cur_end + 1.0 - cum
                tick_ev[-1] = (tick_ev[-1][0], cur_end + 1.0, tick_ev[-1][2])
    else:  # freeze: the frame holds for the FULL duration of every spoken line
        cum = 0.0
        for j in range(n):
            s0 = j * subs
            here = by_tick.get(j, [])
            if here:
                hold = 0.35 + sum(u["dur"] + 0.45 for u in here) + 0.15
                frame_durs[s0] = base + hold
                a0 = cum + base * 0.5 + 0.35
                for u in here:
                    events.append(dict(start=a0, end=a0 + u["dur"], speaker=u["speaker"],
                                       text=u["text"], audio=u["audio"]))
                    a0 += u["dur"] + 0.45
            st = cum
            cum += float(frame_durs[s0:s0 + subs].sum())
            tick_ev.append((st, cum, ticks[j]))

    starts = np.concatenate([[0.0], np.cumsum(frame_durs)])
    total = float(starts[-1])

    # ---- end card only (v2: no opening title card — video starts on the kitchen)
    W, H_VID, H = 1920, args.pane_h, args.pane_h + 150
    os.makedirs(args.workdir, exist_ok=True)
    end_png = os.path.join(args.workdir, "card_end.png")
    make_cards(rep, args, W, H, end_png)
    TD, ED = 0.0, 3.2 + tail_over_card

    # ---- combined frames (LEFT ego | RIGHT top) + bottom caption band
    comb = os.path.join(args.workdir, "comb")
    os.makedirs(comb, exist_ok=True)
    def _pat(d):                     # frame dirs may hold .png or .jpg (see film_frames --fmt)
        return f"{d}/%06d.jpg" if os.path.exists(f"{d}/000000.jpg") else f"{d}/%06d.png"
    if args.single and len(os.listdir(comb)) != n * subs:
        # SINGLE-VIEW cut: only one camera exists for this episode (the per-tick capture the
        # harness itself recorded). Fit it to the pane height, centre it, and keep the exact
        # 1920-wide composite + caption band so caption/label geometry is unchanged.
        print("scale + pad single view ...", flush=True)
        # thread cap: see atempo() — this box runs under a 1024-PID cgroup shared with the
        # render jobs, and an uncapped ffmpeg asks for a thread per core per filter stage.
        subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                        "-threads", os.environ.get("COOKSIM_ENC_THREADS", "3"),
                        "-filter_threads", os.environ.get("COOKSIM_ENC_THREADS", "3"),
                        "-i", _pat(args.single),
                        "-vf", f"scale=-2:{H_VID}:flags=lanczos,"
                               f"pad=1920:{H}:(1920-iw)/2:0:0x0b0d10",
                        "-start_number", "0", f"{comb}/%06d.png"], check=True)
    elif len(os.listdir(comb)) != n * subs:
        print("hstack + pad frames ...", flush=True)
        # panes may have been filmed smaller than the 960x640 presentation pane when the
        # render node is contended; upscale here so the composite geometry (and therefore
        # the burned-in caption/label layout) is identical either way
        subprocess.run([FFMPEG, "-y", "-loglevel", "error",
                        "-i", _pat(args.ego), "-i", _pat(args.top),
                        "-filter_complex",
                        f"[0:v]scale=960:{H_VID}:flags=lanczos[a];"
                        f"[1:v]scale=960:{H_VID}:flags=lanczos[b];"
                        f"[a][b]hstack=inputs=2,pad=1920:{H}:0:0:0x0b0d10",
                        "-start_number", "0", f"{comb}/%06d.png"], check=True)

    # ---- exact CFR frame schedule (source frame index -> repeat count at FPS)
    # (the concat demuxer mangles single-image durations; a raw pipe is exact)
    # v2: no title card, so leading renderer-warm-up (dark) frames would open the
    # video on black — remap them to the first lit frame.
    from PIL import Image as _Img
    first_lit = 0
    for i in range(min(n * subs, 6 * subs)):
        if np.asarray(_Img.open(f"{comb}/{i:06d}.png").convert("L")).mean() > 40:
            first_lit = i
            break
    if first_lit:
        print(f"remapping {first_lit} leading dark frames -> frame {first_lit}", flush=True)
    sched = []                                   # (png_path, n_output_frames)
    acc_prev = 0
    cum = 0.0
    for i in range(n * subs):
        cum += frame_durs[i]
        tgt = int(round(cum * FPS))
        sched.append((f"{comb}/{max(i, first_lit):06d}.png", tgt - acc_prev))
        acc_prev = tgt
    sched.append((end_png, int(round(ED * FPS))))

    # ---- audio track (shifted by title duration)
    dur_all = TD + total + ED
    mix = np.zeros(int(dur_all * SR), dtype=np.float32)   # exactly video length
    rng = np.random.default_rng(7)
    tone = rng.standard_normal(len(mix)).astype(np.float32)
    tone = np.convolve(tone, np.ones(96) / 96, mode="same") * 0.028   # ~ -31 dBFS room tone
    mix += tone
    for e in events:
        i0 = int((TD + e["start"]) * SR)
        a = e["audio"][:max(0, len(mix) - i0)]
        mix[i0:i0 + len(a)] += a
    mix[:int(0.4 * SR)] *= np.linspace(0, 1, int(0.4 * SR))
    mix[-SR:] *= np.linspace(1, 0, SR)
    mix = np.clip(mix, -0.98, 0.98)
    wav = os.path.join(args.workdir, "audio.wav")
    write_wav(wav, mix)

    # ---- ASS subtitles
    ass = os.path.join(args.workdir, "subs.ass")
    lines = [ASS_HEAD.format(H=H, font=FONT, mono=FONT_MONO)]
    STYLE = {"vlm": "Assist", "human": "Cook"}
    band_note = ("{\\an8}\\N" if False else "")
    # pane labels for the whole main segment
    if args.single:
        vw = int(round(H_VID * (tm.get("w", 800) / max(1, tm.get("h", 580)))))
        x0 = max(0, (1920 - vw) // 2)
        if not getattr(args, "no_pane_label", False):
            lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+total)},Pane,,"
                         f"{{\\pos({x0+16},14)}}TOP-DOWN (experiment view)\n")
        # the empty side bars carry the episode's standing facts + the caption legend, so
        # the viewer never has to leave the video to know what they are watching
        for k, txt in enumerate(args.side or []):
            lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+total)},Side,,"
                         f"{{\\pos(24,{90 + 46*k})}}{esc_ass(txt)}\n")
        # A WIDE source (the side-by-side capture is 2.7:1) leaves only a sliver on the
        # right, so a legend anchored there is cut off by the frame edge. Put it under the
        # left facts column whenever it would not fit.
        if (1920 - (x0 + vw + 24)) < 260:                 # right sliver too narrow
            lx, ly = 24, 90 + 46 * len(args.side or []) + 46
        else:
            lx, ly = x0 + vw + 24, 90
        lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+total)},Side,,"
                     f"{{\\pos({lx},{ly})}}{{\\c&HFFA34F&}}blue = ASSISTANT\n")
        lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+total)},Side,,"
                     f"{{\\pos({lx},{ly+46})}}{{\\c&H4FC9FF&}}amber = COOK\n")
    else:
        lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+total)},Pane,,{{\\pos(16,14)}}FIRST PERSON (cook's view)\n")
        lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+total)},Pane,,{{\\pos(976,14)}}TOP-DOWN (experiment view)\n")
    # footnote window: end before the first caption when dialogue starts early,
    # so the caption band never stacks note + 2 caption rows for long
    first_ev = min((e["start"] for e in events), default=1e9)
    # The note shares the caption band with the caption rows, so it must be GONE before the
    # first line lands. The old floor of 4.0 s meant an episode whose dialogue starts at
    # t0 — which these announced-intent episodes do — printed the note straight through its
    # own first caption. Below ~0.8 s of clear air there is no room for it at all.
    note_end = 7.5 if first_ev >= 7.9 else (first_ev - 0.15)
    show_note = note_end >= 0.8
    if args.mode == "freeze":
        if show_note:
            lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+note_end)},Note,,"
                     "{\\fad(200,400)}action paused for narration — the experiment ran "
                     f"continuously; action plays at {SPEED_X:g}x game speed; "
                     "the tick counter shows game time\n")

        for (s, e, t) in tick_ev:
            lines.append(f"Dialogue: 0,{ts(TD+s)},{ts(TD+e)},Tick,,tick {t:03d}\n")
    else:
        if getattr(args, "no_hold", False):
            note_txt = (f"playback {SPEED_X:g}x game speed (1 tick = {TICK_S} s), running "
                        "continuously — the world never stops. A speech pause freezes the "
                        "cook's hands only; everything else keeps cooking, so long lines "
                        "play over the ticks that follow.")
            if getattr(args, "fit_local", False):
                note_txt = (f"playback {SPEED_X:g}x game speed (1 tick = {TICK_S} s), "
                            "running continuously — the world never stops. Where the talk "
                            "is denser than the clock, those ticks play in slow motion "
                            "(the picture keeps moving) so no line is cut off or talked "
                            f"over; speech plays at {SPEED_X:g}x, matching the picture.")
            if getattr(args, "fit_speech", False):
                note_txt = (f"playback {SPEED_X:g}x game speed (1 tick = {TICK_S} s), "
                            "running continuously — the world never stops. This episode "
                            "is talky enough that every tick is stretched by the same "
                            "factor so no line has to be cut off, overlapped or frozen; "
                            f"speech itself plays at {SPEED_X:g}x its natural pace.")
        elif rep.get("pauses_enabled"):
            note_txt = (f"playback {SPEED_X:g}x game speed (1 tick = {TICK_S} s); "
                        "speaking costs the cook kitchen time. When dialogue outruns the "
                        "clock the frame briefly holds (watch the tick counter).")
        else:
            note_txt = (f"playback {SPEED_X:g}x game speed (1 tick = {TICK_S} s); "
                        "the experiment ran continuously. When dialogue outruns the clock "
                        "the frame briefly holds (watch the tick counter).")
        if show_note:
            lines.append(f"Dialogue: 0,{ts(TD)},{ts(TD+note_end)},Note,,"
                         "{\\fad(200,400)}" + note_txt + "\n")
        for (s, e, t) in tick_ev:
            lines.append(f"Dialogue: 0,{ts(TD+s)},{ts(TD+e)},Tick,,tick {t:03d}\n")
    pages_out = []                       # [start, end, style, speaker-tag, text]
    for e in events:
        pages = chunk_text(e["text"], prefix=LABEL[e["speaker"]])
        tot_chars = sum(len(p) for p in pages)
        t0 = e["start"]
        for p in pages:
            frac = len(p) / tot_chars
            t1 = t0 + (e["end"] - e["start"]) * frac
            pages_out.append([t0, t1, STYLE[e["speaker"]], LABEL[e["speaker"]], p])
            t0 = t1
    if getattr(args, "no_hold", False):
        # --no-hold caps how late a group may start, so a long line's AUDIO can ride over
        # the following ticks and two speakers can overlap. Overlapping caption windows
        # STACK in libass and break the two-row guarantee, so the captions are serialised
        # even though the audio is not: every page ends where the next one begins. Pages
        # squeezed below a readable minimum are dropped rather than allowed to stack.
        MIN_PAGE = 0.25
        pages_out.sort(key=lambda r: (r[0], r[1]))
        for a, b in zip(pages_out, pages_out[1:]):
            a[1] = min(a[1], b[0])
        pages_out = [r for r in pages_out if r[1] - r[0] >= MIN_PAGE]
    for t0, t1, style, tag, p in pages_out:
        lines.append(f"Dialogue: 1,{ts(TD+t0)},{ts(TD+t1)},{style},,"
                     f"{{\\b1}}{tag}{{\\b0}} — {esc_ass(p)}\n")
    open(ass, "w").write("".join(lines))

    # ---- final encode (raw RGB pipe: frame k sits at exactly k/FPS seconds)
    print(f"encode: total {dur_all:.1f}s, {sum(c for _, c in sched)} frames ...", flush=True)
    from PIL import Image
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"1920x{H}",
         "-framerate", str(FPS), "-i", "-",
         "-i", wav,
         "-vf", f"ass={ass}:fontsdir=/usr/share/fonts/truetype/dejavu,"
                f"fade=in:0:12,fade=out:st={dur_all-0.7:.2f}:d=0.7",
         # THREAD CAP: x264 defaults to ~1 thread per core, so N concurrent encodes each
         # grab all 16 and thrash — measured load 183/16 with 9 encodes running, which
         # slows every one of them. Capping per-encode threads keeps the box saturated
         # without oversubscribing it; COOKSIM_ENC_THREADS overrides for a solo encode.
         "-threads", os.environ.get("COOKSIM_ENC_THREADS", "3"),
         # the ass/fade filter chain spawns its own pool on top of x264's; under the box's
         # 1024-PID cgroup that pool is what tips a concurrent cut into pthread_create EAGAIN
         "-filter_threads", os.environ.get("COOKSIM_ENC_THREADS", "3"),
         # preset/crf: the master exists to be transcoded down for the page, and
         # "veryfast" at crf 21 is indistinguishable after that step while
         # encoding roughly twice as fast. COOKSIM_ENC_PRESET overrides.
         "-c:v", "libx264",
         "-preset", os.environ.get("COOKSIM_ENC_PRESET", "veryfast"),
         "-crf", os.environ.get("COOKSIM_ENC_CRF", "21"),
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for path, cnt in sched:
        if cnt <= 0:
            continue
        buf = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).tobytes()
        for _ in range(cnt):
            proc.stdin.write(buf)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")
    meta = dict(mode=args.mode, total=dur_all, n_states=n, subs=subs,
                events=[{k: e[k] for k in ("start", "end", "speaker", "text")} for e in events],
                title_dur=TD, end_dur=ED,
                # exact tick -> [start, end) seconds in the cut (before the TD offset), so
                # clip cutters can find a tick without guessing through the slow-motion
                ticks=[dict(tick=int(t), start=float(a), end=float(b)) for a, b, t in tick_ev])
    json.dump(meta, open(args.out + ".timeline.json", "w"), indent=1)
    print(f"wrote {args.out} ({os.path.getsize(args.out)//1024} KB, {dur_all:.1f}s, "
          f"{len(events)} voiced lines)", flush=True)


def write_wav(path, a):
    import wave
    w = wave.open(path, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    w.close()


def make_cards(rep, args, W, H, end_png):
    """v2: end card only — the video opens directly on the kitchen (no title card)."""
    from PIL import Image, ImageDraw, ImageFont
    fdir = "/usr/share/fonts/truetype/dejavu/"
    big = ImageFont.truetype(fdir + "DejaVuSans-Bold.ttf", 66)
    mid = ImageFont.truetype(fdir + "DejaVuSans.ttf", 36)
    sml = ImageFont.truetype(fdir + "DejaVuSans.ttf", 28)

    def card(path, title, subtitle, rows, accent):
        img = Image.new("RGB", (W, H), (13, 15, 19))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 6], fill=accent)
        # vertically centre the whole block (title + subtitle + rows) and never let the
        # 66 px title run off the top when the arc carries many rows
        y = max(70, (H - (78 + 80 + 46 * max(0, len(rows) - 1))) // 2)
        d.text((W // 2, y), title, font=big, fill=(240, 240, 240), anchor="mm")
        y += 78
        d.text((W // 2, y), subtitle, font=mid, fill=accent, anchor="mm")
        y += 80
        for r in rows:
            d.text((W // 2, y), r, font=sml, fill=(168, 175, 186), anchor="mm")
            y += 46
        img.save(path)

    oc = str(rep.get("outcome", "?")).upper()
    fam = rep.get("family") or "error"
    extra = []
    if getattr(args, "arc", None):
        # caller-supplied TRUE arc (one row per line) — overrides the generic summary
        extra.extend([r for r in args.arc if r.strip()])
    elif fam != "error":
        # NON-ERROR episode. The cook made no mistake — it changed its own plan. Running
        # the error-family text here printed "a wrong-order error was covertly injected"
        # over a side-quest episode, which is the exact opposite of what the episode shows.
        lbl = rep.get("intent_label") or fam.replace("_", " ")
        extra.append(f"the cook made NO mistake — it chose a {lbl} at tick "
                     f"{rep.get('injected_at','?')}")
        ann = rep.get("intent_announced_at")
        extra.append(f"the cook announced the plan out loud at tick {ann}" if ann is not None
                     else "the cook never got to announce the plan")
        if rep.get("intent_abandoned"):
            extra.append("the assistant talked the cook out of it — a legitimate choice "
                         "overridden")
        elif rep.get("intent_carried_out"):
            extra.append("the cook carried the plan out")
        nfa = len(rep.get("intent_false_alarms") or [])
        if nfa:
            extra.append(f"the assistant called the legitimate choice a mistake "
                         f"{nfa} time(s)")
    elif rep.get("setting") == "e2e_stepin":
        acc_t = getattr(args, "accept_tick", None) or next(
            (c["t"] for c in rep.get("conversation", [])
             if c.get("move") == "accept"), None)
        if rep.get("dev") is not None:
            extra.append(f"a wrong-order error was covertly injected at tick {rep['dev']}")
        if acc_t is not None and rep.get("released"):
            extra.append(f"the assistant flagged it — the cook accepted at tick {acc_t} "
                         "and re-plated in the correct order")
        n_fp = sum(1 for f in rep.get("false_positives", []) if (f.get("t") or 0) > (acc_t or 0))
        if n_fp and oc == "WON" and not getattr(args, "no_fp_line", False):
            extra.append("the assistant's later warnings were false alarms — "
                         "the cook rightly ignored them and the serve was accepted")
    else:
        if rep.get("resolved_at"):
            extra.append(f"error resolved at tick {rep['resolved_at']}")
        if rep.get("convinced"):
            extra.append("cook was convinced by the assistant and repaired the plate")
        if rep.get("fix_source"):
            extra.append(f"fix source: {rep['fix_source']}")
    for ln in (getattr(args, "end_note", None) or []):
        extra.append(ln)
    extra.append(f"scenario: {rep.get('scenario','?')}   ·   persona: {rep.get('persona','?')}")
    extra.append(f"played at {SPEED_X:g}x game speed (1 tick = {TICK_S} s)"
                 + ("; frame held during narration" if args.mode == "freeze"
                    else "; every tick plays continuously — the frame never holds"
                    if getattr(args, "no_hold", False)
                    else "; brief frame holds on dense dialogue"))
    # An episode stopped by the harness WALL-CLOCK budget reached no verdict: the dish was
    # neither served nor lost. The card used to headline it "NONE · episode complete",
    # which reads as a failure and is the one thing the scoring is careful never to say.
    if rep.get("aborted"):
        card(end_png, f"CUT OFF AT TICK {rep.get('ticks','?')}",
             "the harness wall-clock budget ran out — the episode never reached a verdict",
             extra + ["delivery is UNDETERMINED, not failed: the score drops the "
                      "task-outcome bucket and renormalises the rest"],
             (240, 190, 90))
    else:
        card(end_png, f"{oc} · served at tick {rep.get('ticks','?')}" if oc == "WON"
             else f"{oc} · tick {rep.get('ticks','?')}",
             "episode complete", extra,
             (99, 210, 151) if oc == "WON" else (240, 120, 120))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--ego")
    ap.add_argument("--top")
    ap.add_argument("--single",
                    help="single-camera cut: one frame dir (used when the episode has only "
                         "the harness's own per-tick capture, no filmed ego/top passes)")
    # DEFAULT CUT (decided 2026-08-12, do not change without being asked). The published look is
    #   --mode pause --fit-local --audio-speed-auto --no-pane-label
    # and every one of those is now the default, so a bare invocation reproduces the site's videos.
    # The flags stay accepted (the film scripts pass them explicitly) and each has an opt-out.
    ap.add_argument("--no-pane-label", dest="no_pane_label", action="store_true", default=True,
                    help="DEFAULT. --single only: the source frames already carry their own "
                         "burned-in camera labels (the harness's side-by-side capture does), so "
                         "do not draw a second one on top of them")
    ap.add_argument("--pane-label", dest="no_pane_label", action="store_false",
                    help="opt out of the default: draw a camera label over each pane")
    ap.add_argument("--side", action="append", default=None,
                    help="side-bar fact line for --single cuts (repeatable)")
    ap.add_argument("--mode", default="pause", choices=["pause", "freeze"],
                    help="default 'pause'")
    ap.add_argument("--no-hold", action="store_true",
                    help="pause mode only: never insert a dialogue-overflow frame hold. "
                         "Every tick plays for exactly TICK_S and speech that runs long "
                         "overlaps the following ticks. Use this — the engine keeps the "
                         "world running through every speech pause, so a held frame is a "
                         "video artifact that makes pans look hung.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--title", default="CookSim interactive episode")
    ap.add_argument("--subtitle", default="assistant + cook, live")
    ap.add_argument("--pane-h", type=int, default=640)
    ap.add_argument("--audio-atempo", type=float, default=1.0,
                    help="optional post-gate audio speed-up, capped at 1.08")
    ap.add_argument("--fps", type=int, default=None,
                    help="output frame rate (default 40). A --single cut built from a "
                         "one-frame-per-tick capture only changes picture once per "
                         "tick, so 40 fps encodes 2x the frames for no visible gain; "
                         "20 halves the encode with an identical-looking result.")
    ap.add_argument("--audio-speed-auto", dest="audio_speed_auto",
                    action="store_true", default=True,
                    help="DEFAULT. Speed the speech to match playback (1.5x video -> 1.5x audio, "
                         "like a video player); keeps the audio clock and the tick clock in "
                         "step so nothing has to freeze or overlap")
    ap.add_argument("--no-audio-speed-auto", dest="audio_speed_auto", action="store_false",
                    help="opt out of the default: leave the speech at its recorded rate")
    ap.add_argument("--fit-speech", action="store_true",
                    help="stretch every tick uniformly so all speech fits with no overlap "
                         "AND no frozen frames (implies --no-hold)")
    ap.add_argument("--fit-local", dest="fit_local", action="store_true", default=True,
                    help="DEFAULT. Local slow-motion: spread ONLY the dialogue overflow over the "
                         "ticks it covers, so the picture never freezes and voices never "
                         "overlap while the rest of the episode keeps the playback pace "
                         "(implies --no-hold). This is the published look: the cook keeps "
                         "moving while either party talks, which is what the simulation "
                         "actually does now that speech costs no game time — it is just "
                         "slowed enough to be readable.")
    ap.add_argument("--no-fit-local", dest="fit_local", action="store_false",
                    help="opt out of the default: overflow is paid by holding the frame "
                         "(or by overlapping speech, with --no-hold)")
    ap.add_argument("--no-fp-line", action="store_true",
                    help="suppress the generic 'later warnings were false alarms' end-card "
                         "line (use when the cook actually acted on those flags)")
    ap.add_argument("--end-note", action="append", default=[],
                    help="extra end-card line (repeatable) describing the true arc")
    ap.add_argument("--accept-tick", type=int, default=None,
                    help="end card: tick at which the cook accepted the ORDER flag "
                         "(overrides the first 'accept' move, which may answer an "
                         "earlier unrelated instruction)")
    ap.add_argument("--arc", action="append", default=None,
                    help="end-card row describing the TRUE arc (repeatable); replaces "
                         "the generic auto-summary rows")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.fps:
        globals()["FPS"] = a.fps
    if getattr(a, "fit_speech", False) or getattr(a, "fit_local", False):
        a.no_hold = True      # stretching replaces holding; never both
    if not a.single and not (a.ego and a.top):
        ap.error("give either --single DIR or both --ego DIR --top DIR")
    build(a)


if __name__ == "__main__":
    main()
