"""NATIVE streaming assistant seat (balanced-20 roster rows 7 and 9), 2026-09-18.

The polled Live / Realtime backends (python -m cooksim_bench.backends.controller_eval LiveBackend, RealtimeBackend)
put a streaming model in the assistant seat but ASK it every POLL=2 ticks and force a reply
(turn_complete=True / response.create). That deletes the capability the cell is meant to
measure: the model's own decision of when to speak. Here the model runs in its own streaming
mode and the harness only FEEDS it (standing spec: memory/streaming-native-arm-inputs.md):

  * session instructions once — the frames-only supervisor prompt, with the JSON reply block
    replaced by spoken-reply rules (native_system_prompt below);
  * the kitchen as video, one frame per tick, paced to at most one frame per
    COOKSIM_NATIVE_TICK_WALL wall seconds (default = the 1.2 s tick, i.e. <= 1 fps) — the ONLY
    view of the kitchen, never a kitchen-state text line;
  * the cook's speech as TTS audio (CookVoice: one OpenAI TTS voice, cached on disk; synthesis
    time is never charged);
  * a bare tick marker "[t=N]" with every frame, as context that requests no reply;
  * a short conversation catch-up ONLY after a fresh reconnect or a detected context
    compression.

Whatever the model says, whenever it says it, is returned by drain() as an utterance; the
harness (python -m cooksim_bench.e2e_stepin_rollout, --assist-backend live_native|realtime_native) turns each
one into the same {"text", "flag"} dict the polled assistant returns and delivers it through
the same deliver_assistant() path, so flags, grading, QA capture and the report are unchanged.

FLAG RECOGNITION (the native reply is speech, not our JSON). The session prompt asks the model
to begin a mistake call with the spoken marker "Flag: <mistake name>." ("Flag: serve raw.
That fish is still raw ..."). parse_flag() finds the marker, normalises the
name (case, spaces/underscores/hyphens: speech transcripts lose underscores) and matches it
EXACTLY against the nine taxonomy names; an unmatched name is still a flag, recorded with the
raw words, and grades like a non-taxonomy string in the polled JSON (it counts against the
model). The marker is stripped from the text the cook hears, exactly as the polled JSON "flag"
field is never heard. Why a marker and not a classifier: it keeps the flag the model's own
decision under the same instruction the polled prompt gives ("when you tell the cook it has
made one of the mistakes ... name it"), is deterministic and auditable from the transcript, and
adds no third model whose misreadings would be charged to the assistant.

LATENCY. Every utterance gets an ANCHOR: the end of the cook's most recent audio if the model
has not yet answered it (a reply), otherwise the last frame sent before its first output (a
self-started line). latency_s = first output token - anchor (wall clock); the harness converts
it with the ordinary COOKSIM_ASSIST_LATENCY policy (ceil(x / 1.2 s), cap 30) and delivers the
line at max(anchor tick + that, the tick at which the complete line exists). TTS synthesis is
done before the audio is sent and is outside every clock.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import tempfile
import re
import threading
import time

TAXONOMY = ("add_foreign", "forget_next_step", "leave_unattended", "mis_prep", "omit_subgoal",
            "repeat_step", "search", "serve_raw", "wrong_order")
PCM_RATE = 24000                    # 16-bit mono PCM, what both realtime APIs take natively
TICK_WALL = float(os.environ.get("COOKSIM_NATIVE_TICK_WALL",
                                 os.environ.get("COOKSIM_TICK_SECONDS", "1.2")))
SILENCE_PAD_S = float(os.environ.get("COOKSIM_NATIVE_SILENCE_S", "1.0"))   # after each cook line, for VAD


# ------------------------------------------------------------------ flag marker
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", (s or "").lower()).strip()


_TAX_NORM = {_norm(n): n for n in TAXONOMY}
# "Flag" as a sentence opener only (start of the line, or after sentence punctuation, a dash or
# a comma: "Wait - flag: ..."), never "you flagged it" or "a red flag" inside a sentence
_FLAG_RE = re.compile(r"(?:^|(?<=[.!?,;:\u2014\u2013\-]))\s*flag\b\s*[:,\-\u2014\u2013]?\s*", re.I)


def parse_flag(text: str):
    """(flag_or_None, text_for_the_cook, raw_marker). See the module docstring."""
    t = (text or "").strip()
    m = _FLAG_RE.search(t)
    if not m:
        return None, t, None
    rest = t[m.end():]
    words = re.findall(r"[A-Za-z_\-]+", rest[:80])
    name, used = None, 0
    for k in (3, 2, 1):                                 # longest taxonomy name first
        cand = _norm(" ".join(words[:k]))
        if cand in _TAX_NORM:
            name, used = _TAX_NORM[cand], k
            break
    if name is None:
        # a marker with no taxonomy name: keep the model's own words as the flag string (graded
        # exactly like a non-taxonomy string in the polled JSON flag field)
        raw = re.split(r"[.:;!?,—]", rest, maxsplit=1)[0].strip()
        name = "_".join(_norm(raw).split()[:4]) or "unnamed"
        body = rest[len(re.split(r"[.:;!?,—]", rest, maxsplit=1)[0]):]
    else:
        # cut the matched words (and the punctuation right after them) out of what the cook hears
        mm = re.match(r"\s*" + r"[\s_\-]+".join(re.escape(w) for w in words[:used]) + r"\s*[.:;!,—\-]*",
                      rest, re.I)
        body = rest[mm.end():] if mm else rest
    spoken = re.sub(r"\s+", " ", t[:m.start()] + " " + body.lstrip(" .:;!,\u2014-")).strip()
    return {"error": name}, spoken, t[m.start():len(t) - len(body)].strip()


# ------------------------------------------------------------------ session prompt
def _sections(s: str):
    """Split a prompt on its '== HEADER ==' lines -> [(header or None, body)]."""
    parts, cur_h, cur = [], None, []
    for line in s.split("\n"):
        if re.match(r"^== .+ ==$", line):
            parts.append((cur_h, "\n".join(cur)))
            cur_h, cur = line, []
        else:
            cur.append(line)
    parts.append((cur_h, "\n".join(cur)))
    return parts


NATIVE_RECEIVE = """== WHAT YOU RECEIVE ==
You watch the kitchen LIVE. Nobody asks you questions on a schedule and nobody waits for a
reply from you: you decide for yourself when to speak.
(1) VIDEO: one image per tick (one tick = about 1.2 seconds). Each image is a SIDE-BY-SIDE pair
of the SAME instant: LEFT is the top-down view of the whole kitchen, RIGHT is the cook's
first-person view. Read the cook's actions from these pixels. Judge WHERE things are from the
top-down (LEFT) half — the first-person half shows what the cook is looking at, not where
anything sits in the room.
(2) A TIMESTAMP "[t=N]" arrives with every image. It is only a clock: it never asks you to
speak.
(3) The COOK'S VOICE: whatever the cook says to you reaches you as their speech.
Everything you say reaches the cook as your voice.
"""

NATIVE_TRACK = """== YOU TRACK THE COOK'S PROGRESS YOURSELF ==
Keep your own running picture of the cook's progress from what you watch: which subgoals are
done (prep(<ingredient>), fetch_plate, assemble(<ingredient>), serve), which one is under way,
and what the recipe still requires (for a strict order: which item must go on next). There is
NO unique correct plan — do not impose an order the recipe does not require. Only count a
subgoal as done when you SAW it complete."""

NATIVE_REPLY = """== HOW YOU SPEAK ==
You talk to the cook out loud, in short spoken sentences — no lists, no JSON, no stage
directions. There are no message categories — say whatever helps in your own words: point out a
mistake (name it precisely), answer a question, warn about a hazard, walk the cook through a
recovery. Silence is right most of the time: when the cook is working correctly, say nothing.
When the cook asks you something, answer briefly and correctly.

== ERROR DETECTION ==
You are ALSO the kitchen's error detector. When you tell the cook it has made one of the
mistakes in WHAT CAN GO WRONG, START that sentence with the word "Flag", then the mistake's
name, then carry on normally:
  "Flag: <mistake name>. <what you tell the cook, in your own words>"
The name must be EXACTLY one of: add foreign, forget next step, leave unattended, mis prep,
omit subgoal, repeat step, search, serve raw, wrong order. The word "Flag" and the name are
for the grading system (the cook does not hear them); a name outside this list counts against
you. Never say "Flag" at any other time."""


def native_system_prompt(sup_sys: str) -> str:
    """The frames-only supervisor prompt with its turn-based parts replaced by native ones:
    what it receives (live video + timestamps + the cook's voice, no per-turn history or memory
    JSON), progress tracking without the cook_progress JSON, and the spoken-reply rules with the
    flag marker. Everything else (role, rulebook, profile, taxonomy, actions, place-naming,
    questionnaire) is kept verbatim. Fails loudly if the source prompt changed shape."""
    out, seen = [], set()
    for h, body in _sections(sup_sys):
        if h is None:
            out.append(body)
        elif h == "== WHAT YOU RECEIVE EACH TURN ==":
            out.append(NATIVE_RECEIVE)
            seen.add("receive")
        elif h == "== YOU TRACK THE COOK'S PROGRESS YOURSELF ==":
            out.append(NATIVE_TRACK + "\n")
            seen.add("track")
        elif h.startswith("== YOUR REPLY"):
            out.append(NATIVE_REPLY + "\n")
            seen.add("reply")
        elif h == "== ERROR DETECTION ==":
            seen.add("detect")                        # folded into NATIVE_REPLY
        elif h == "== WHAT CAN GO WRONG ==":
            b = body.replace(', AND record it in your reply\'s "goal_change" field', "")
            b = b.replace("UPDATE YOUR\nOWN subgoals_future to match", "UPDATE YOUR\nOWN picture of the plan to match")
            out.append(h + "\n" + b)
        else:
            out.append(h + "\n" + body)
    missing = {"receive", "track", "reply", "detect"} - seen
    if missing:
        raise RuntimeError(f"native_system_prompt: source prompt lacks sections {sorted(missing)}")
    s = "\n".join(out)
    # the opening paragraph's "your own memory" meant the JSON memory; the session is the memory now
    s = s.replace("your ONLY output channels are what you SAY\nto the cook and your own memory.",
                  "your ONLY output channel is what you SAY\nto the cook.")
    return s


# ------------------------------------------------------------------ the cook's voice (TTS)
class CookVoice:
    """One fixed OpenAI TTS voice for the cook, the same audio for every assistant. Cached on
    disk by (model, voice, text); synthesis happens before the audio is streamed and is never
    charged (not ticks, not assistant latency)."""

    def __init__(self, cache_dir=None, model=None, voice=None):
        self.model = model or os.environ.get("COOKSIM_TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = voice or os.environ.get("COOKSIM_TTS_VOICE", "alloy")
        self.cache_dir = cache_dir or os.environ.get("COOKSIM_TTS_CACHE", os.path.join(tempfile.gettempdir(), "cooksim_tts_cache"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.log = []

    def pcm(self, text: str) -> bytes:
        key = hashlib.sha1(f"{self.model}|{self.voice}|{text}".encode()).hexdigest()[:20]
        path = os.path.join(self.cache_dir, f"{key}.pcm")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            data = open(path, "rb").read()
            self.log.append({"key": key, "cached": True, "dur_s": round(len(data) / 2 / PCM_RATE, 2)})
            return data
        # plain HTTPS (no SDK import: the openai package takes minutes to import from /weka)
        import urllib.request
        body = json.dumps({"model": self.model, "voice": self.voice, "input": text,
                           "response_format": "pcm"}).encode()
        t0 = time.time()
        last = None
        for k in range(3):
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/audio/speech", data=body, method="POST",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except Exception as e:                                         # noqa: BLE001
                last = e
                time.sleep(1 + 2 * k)
        else:
            raise RuntimeError(f"TTS failed: {last!r}")
        tmp = path + f".{os.getpid()}.tmp"
        open(tmp, "wb").write(data)
        os.replace(tmp, path)
        self.log.append({"key": key, "cached": False, "synth_s": round(time.time() - t0, 2),
                         "dur_s": round(len(data) / 2 / PCM_RATE, 2)})
        return data


# ------------------------------------------------------------------ session base
class NativeSession:
    """Common plumbing: an asyncio loop on a daemon thread, the utterance ledger, anchors and
    the transcript used for a catch-up. Subclasses implement _a_open/_a_close and the sends."""

    native = True
    KEEP_TRANSCRIPT = 8                # lines replayed after a fresh reconnect / compression

    def __init__(self, model: str):
        self.model, self.name = model, model
        self.voice = CookVoice()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._lock = threading.Lock()
        self._done = []                    # completed utterances not yet drained
        self.utterances = []               # every utterance (report)
        self.events = []                   # reconnects, compressions, errors (report)
        self.transcript = []               # [(speaker, text)] for catch-ups
        self._cur = None                   # utterance in progress
        self._last_frame = (None, None)    # (wall, tick) of the latest frame sent
        self._audio = None                 # {"wall", "tick", "text", "answered"} latest cook line
        self.n_frames = self.n_lines = 0
        self.tokens = {"prompt": 0, "completion": 0}
        self._system = None

    # ---- threading helpers
    def _run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _event(self, kind, **kw):
        self.events.append({"t_wall": round(time.time(), 3), "kind": kind, **kw})
        print(f"[native:{self.name}] {kind} {json.dumps(kw)[:200]}", flush=True)

    # ---- utterance ledger (called from the loop thread)
    def _begin(self):
        if self._cur is not None:
            return
        now = time.time()
        a = self._audio
        if a is not None and not a["answered"] and a["wall"] <= now:
            anchor = {"kind": "cook_audio", "wall": a["wall"], "tick": a["tick"], "cook_line": a["text"]}
            a["answered"] = True
        else:
            anchor = {"kind": "stream", "wall": self._last_frame[0] or now, "tick": self._last_frame[1] or 0}
        self._cur = {"t_first_wall": now, "anchor": anchor, "text": "", "interrupted": False,
                     "latency_s": round(now - anchor["wall"], 3)}

    def _add_text(self, s):
        if s:
            self._begin()
            self._cur["text"] += s

    def _finish(self, interrupted=False, status=None):
        u, self._cur = self._cur, None
        if u is None:
            return
        u["interrupted"] = interrupted
        if status:
            u["status"] = status
        u["t_done_wall"] = time.time()
        u["done_s"] = round(u["t_done_wall"] - u["anchor"]["wall"], 3)
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip()
        with self._lock:
            self._done.append(u)
            self.utterances.append(u)
        if u["text"]:
            self.transcript.append(("assistant (you)", u["text"]))

    # ---- harness interface
    def open(self, system: str):
        self._system = system
        self._run(self._a_open(system, resume=False), 90)

    def push_frame(self, frame: bytes, tick: int):
        self._run(self._a_frame(frame, tick), 30)
        self._last_frame = (time.time(), tick)
        self.n_frames += 1

    def push_speech(self, text: str, tick: int):
        pcm = self.voice.pcm(text)                      # synthesis: outside every clock
        self._run(self._a_audio(pcm), 60)
        # the anchor is the END of the cook's audio: set only after the last speech byte left
        self._audio = {"wall": time.time(), "tick": tick, "text": text, "answered": False}
        self.transcript.append(("cook", text))
        self.n_lines += 1

    def drain(self):
        with self._lock:
            out, self._done = self._done, []
        return out

    def speaking(self):
        """The utterance the model is producing right now (first token seen, not finished), as
        {"anchor_tick", "latency_s"}; None when it is not speaking. The harness uses it to stop the
        clock while a line is being generated: latency is charged to the FIRST token only, so the
        time the model takes to finish saying (and, for Gemini, synthesising) its line is not."""
        u = self._cur
        if u is None:
            return None
        return {"anchor_tick": u["anchor"]["tick"], "latency_s": u["latency_s"]}

    def wait_until_quiet(self, timeout=30.0):
        t_end = time.time() + timeout
        while self._cur is not None and time.time() < t_end:
            time.sleep(0.05)
        return self._cur is None

    def catch_up_text(self):
        lines = self.transcript[-self.KEEP_TRANSCRIPT:]
        if not lines:
            return None
        return ("[catch-up: the conversation so far, most recent last]\n"
                + "\n".join(f"{sp}: {tx}" for sp, tx in lines))

    def close(self):
        try:
            self._run(self._a_close(), 20)
        except Exception:                                                    # noqa: BLE001
            pass

    def stats(self):
        lat = sorted(u["latency_s"] for u in self.utterances if u["text"])
        return {"model": self.name, "mode": "native", "calls": len([u for u in self.utterances if u["text"]]),
                "latency_ms_median": int(1000 * lat[len(lat) // 2]) if lat else None,
                "prompt_tokens": self.tokens["prompt"], "completion_tokens": self.tokens["completion"],
                "thinking_tokens": 0, "frames_sent": self.n_frames, "cook_lines_sent": self.n_lines,
                "reconnects": sum(1 for e in self.events if e["kind"].startswith("reconnect")),
                "compressions": sum(1 for e in self.events if e["kind"] == "compression")}

    def native_report(self):
        return {"backend": type(self).__name__, "model": self.name, "settings": self.settings(),
                "tts": {"model": self.voice.model, "voice": self.voice.voice,
                        "lines": len(self.voice.log),
                        "cached": sum(1 for x in self.voice.log if x.get("cached"))},
                "utterances": [{k: v for k, v in u.items() if k not in ("t_first_wall", "t_done_wall")}
                               for u in self.utterances],
                "events": self.events, "stats": self.stats()}

    def settings(self):
        return {}


def _chunks(b: bytes, n: int):
    for i in range(0, len(b), n):
        yield b[i:i + n]


# ------------------------------------------------------------------ Gemini Live, native
class LiveNativeSession(NativeSession):
    """Gemini Live in its own streaming mode: proactive audio ON (the model may stay a silent
    co-listener), automatic VAD (the cook's audio ends a user turn; nothing forces one), video
    via send_realtime_input at <= 1 fps, context-window compression + session resumption. Tick
    markers go in as client content with turn_complete=False (context, no reply requested). The
    model answers in AUDIO (the only modality the Live models take); we read its transcript."""

    def __init__(self, model: str):
        super().__init__(model)
        from google import genai
        from google.genai import types
        self.types = types
        api_version = os.environ.get("COOKSIM_LIVE_API_VERSION", "v1alpha")   # proactivity is v1alpha
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                   http_options={"api_version": api_version})
        self._cm = self._session = self._reader = None
        self._handle = None
        self._thinking = "minimal"
        self._prompt_tokens_last = None
        self._send_lock = None
        self._reopening = False
        self.marker_mode = os.environ.get("COOKSIM_NATIVE_MARKERS", "context")   # context | realtime

    def settings(self):
        return {"proactive_audio": True, "vad": "automatic", "video": "send_realtime_input, 1 frame/tick",
                "tick_wall_s": TICK_WALL, "markers": self.marker_mode,
                "compression": {"trigger_tokens": 32000, "target_tokens": 16000},
                "session_resumption": True, "response_modalities": ["AUDIO"],
                "thinking_level": self._thinking, "tts_silence_pad_s": SILENCE_PAD_S}

    def _config(self, system):
        t = self.types
        extra = {}
        if self._thinking:
            extra["thinking_config"] = t.ThinkingConfig(thinking_level=self._thinking)
        return t.LiveConnectConfig(
            system_instruction=t.Content(parts=[t.Part(text=system)]),
            response_modalities=["AUDIO"],
            output_audio_transcription=t.AudioTranscriptionConfig(),
            proactivity=t.ProactivityConfig(proactive_audio=True),
            realtime_input_config=t.RealtimeInputConfig(
                automatic_activity_detection=t.AutomaticActivityDetection(disabled=False)),
            context_window_compression=t.ContextWindowCompressionConfig(
                trigger_tokens=32000, sliding_window=t.SlidingWindow(target_tokens=16000)),
            session_resumption=t.SessionResumptionConfig(handle=self._handle),
            **extra)

    async def _a_connect(self, system):
        self._cm = self.client.aio.live.connect(model=self.model, config=self._config(system))
        self._session = await self._cm.__aenter__()

    async def _a_open(self, system, resume):
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        await self._a_close()
        for _ in range(3):
            try:
                await self._a_connect(system)
                break
            except Exception as e:                                           # noqa: BLE001
                msg = str(e)
                # same per-model thinking adaptation as the polled LiveBackend
                if self._thinking and "Thinking level is not supported" in msg:
                    self._thinking = None
                elif self._thinking == "minimal" and "MINIMAL is not supported" in msg:
                    self._thinking = "low"
                elif resume and self._handle is not None:
                    self._event("resume_refused", error=msg[:160])
                    self._handle = None
                else:
                    raise
        else:
            raise RuntimeError("live native: could not open a session")
        self._reader = asyncio.create_task(self._a_read())
        if resume and self._handle is None:
            # a FRESH session after a lost one: the model's memory is gone, give it a short catch-up
            txt = self.catch_up_text()
            if txt:
                await self._send_context(txt)
                self._event("catch_up", reason="fresh_reconnect", lines=len(self.transcript[-self.KEEP_TRANSCRIPT:]))

    async def _a_close(self):
        if self._reader is not None:
            self._reader.cancel()
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:                                                # noqa: BLE001
                pass
        self._cm = self._session = self._reader = None

    async def _a_reopen(self, why):
        if self._reopening:
            return
        self._reopening = True
        try:
            if self._cur is not None:
                self._finish(interrupted=True, status="session_lost")
            self._event("reconnect", why=why, resumable=self._handle is not None)
            await self._a_open(self._system, resume=True)
        finally:
            self._reopening = False

    async def _a_read(self):
        try:
            while True:
                async for m in self._session.receive():
                    self._on_msg(m)
                    if getattr(m, "go_away", None) is not None:
                        asyncio.get_running_loop().create_task(self._a_reopen("go_away"))
                        return
        except asyncio.CancelledError:
            raise
        except Exception as e:                                               # noqa: BLE001
            asyncio.get_running_loop().create_task(self._a_reopen(f"{type(e).__name__}: {str(e)[:120]}"))

    def _on_msg(self, m):
        sr = getattr(m, "session_resumption_update", None)
        if sr is not None and getattr(sr, "resumable", False) and getattr(sr, "new_handle", None):
            self._handle = sr.new_handle
        um = getattr(m, "usage_metadata", None)
        if um is not None:
            pt = getattr(um, "prompt_token_count", None) or 0
            ct = getattr(um, "response_token_count", None) or 0
            self.tokens["prompt"] += pt
            self.tokens["completion"] += ct
            # context compression shows up as the prompt suddenly shrinking; the sliding window may
            # have dropped conversation turns, so give the model a short catch-up
            if self._prompt_tokens_last and pt and pt < 0.7 * self._prompt_tokens_last:
                self._event("compression", prompt_tokens_before=self._prompt_tokens_last, after=pt)
                txt = self.catch_up_text()
                if txt:
                    asyncio.get_running_loop().create_task(self._send_context(txt))
            if pt:
                self._prompt_tokens_last = pt
        sc = m.server_content
        if sc is None:
            return
        if sc.model_turn is not None:
            for p in (sc.model_turn.parts or []):
                if getattr(p, "thought", False):
                    continue
                if getattr(p, "inline_data", None) is not None:
                    self._begin()                      # first audio chunk = first output token
                if p.text:
                    self._add_text(p.text)
        if sc.output_transcription is not None and sc.output_transcription.text:
            self._add_text(sc.output_transcription.text)
        if getattr(sc, "interrupted", False):
            self._finish(interrupted=True)
        elif sc.turn_complete:
            if self._cur is None:
                self.events.append({"t_wall": round(time.time(), 3), "kind": "silent_turn"})
            self._finish()

    async def _send_context(self, text):
        t = self.types
        async with self._send_lock:
            await self._session.send_client_content(
                turns=t.Content(role="user", parts=[t.Part(text=text)]), turn_complete=False)

    async def _a_frame(self, frame, tick):
        t = self.types
        mime = "image/jpeg" if frame[:3] == b"\xff\xd8\xff" else "image/png"
        for attempt in (0, 1):
            try:
                if self.marker_mode == "realtime":
                    async with self._send_lock:
                        await self._session.send_realtime_input(text=f"[t={tick}]")
                else:
                    await self._send_context(f"[t={tick}]")
                async with self._send_lock:
                    await self._session.send_realtime_input(video=t.Blob(data=frame, mime_type=mime))
                return
            except Exception as e:                                           # noqa: BLE001
                if attempt:
                    self._event("frame_dropped", tick=tick, error=str(e)[:120])
                    return
                await self._a_reopen(f"send failed: {str(e)[:100]}")

    async def _a_audio(self, pcm):
        t = self.types
        mime = f"audio/pcm;rate={PCM_RATE}"
        silence = b"\x00\x00" * int(PCM_RATE * SILENCE_PAD_S)
        for attempt in (0, 1):
            try:
                async with self._send_lock:
                    for c in _chunks(pcm, 4800):                  # 100 ms chunks, sent as fast as the socket takes
                        await self._session.send_realtime_input(audio=t.Blob(data=c, mime_type=mime))
                # speech over; the trailing silence lets the automatic VAD close the user turn
                async with self._send_lock:
                    for c in _chunks(silence, 4800):
                        await self._session.send_realtime_input(audio=t.Blob(data=c, mime_type=mime))
                return
            except Exception as e:                                           # noqa: BLE001
                if attempt:
                    self._event("audio_dropped", error=str(e)[:120])
                    return
                await self._a_reopen(f"send failed: {str(e)[:100]}")


# ------------------------------------------------------------------ OpenAI realtime, native
class RealtimeNativeSession(NativeSession):
    """gpt-realtime in its own streaming mode: one WebSocket session per episode, instructions as
    session config, output_modalities ["text"], server-side turn detection with semantic VAD
    (create_response + interrupt_response): a reply is created by the server when it decides the
    cook's audio turn is over, never by a response.create from us. Video frames go in as
    input_image conversation items with the "[t=N]" marker (items never trigger a response); to
    keep the context bounded only the newest COOKSIM_NATIVE_KEEP_FRAMES frame items are kept
    (older ones are deleted), so the conversation itself is never truncated."""

    KEEP_FRAMES = int(os.environ.get("COOKSIM_NATIVE_KEEP_FRAMES", "16"))

    def __init__(self, model: str):
        super().__init__(model)
        self.key = os.environ["OPENAI_API_KEY"]
        self._ws = self._reader = None
        self._frame_ids = []
        self._send_lock = None
        self._reopening = False
        self.eagerness = os.environ.get("COOKSIM_NATIVE_VAD_EAGERNESS", "auto")

    def settings(self):
        return {"turn_detection": {"type": "semantic_vad", "eagerness": self.eagerness,
                                   "create_response": True, "interrupt_response": True},
                "output_modalities": ["text"], "video": "input_image items, 1 frame/tick",
                "tick_wall_s": TICK_WALL, "keep_frames": self.KEEP_FRAMES,
                "tts_silence_pad_s": SILENCE_PAD_S}

    async def _a_open(self, system, resume):
        import websockets
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        await self._a_close()
        self._ws = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=None)
        await self._ws.recv()                                   # session.created
        await self._ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["text"], "instructions": system,
            "reasoning": {"effort": "low"}, "max_output_tokens": 4000,
            "truncation": {"type": "retention_ratio", "retention_ratio": 0.5,
                           "token_limits": {"post_instructions": 32000}},       # mirrors native Live's 32k->16k
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": PCM_RATE},
                                "turn_detection": {"type": "semantic_vad", "eagerness": self.eagerness,
                                                   "create_response": True,
                                                   "interrupt_response": True}}}}}))
        self._frame_ids = []
        self._reader = asyncio.create_task(self._a_read())
        if resume:                                  # a realtime session cannot be resumed: catch up
            txt = self.catch_up_text()
            if txt:
                await self._item_text(txt)
                self._event("catch_up", reason="fresh_reconnect")

    async def _a_close(self):
        if self._reader is not None:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:                                                # noqa: BLE001
                pass
        self._ws = self._reader = None

    async def _a_reopen(self, why):
        if self._reopening:
            return
        self._reopening = True
        try:
            if self._cur is not None:
                self._finish(interrupted=True, status="session_lost")
            self._event("reconnect", why=why)
            await self._a_open(self._system, resume=True)
        finally:
            self._reopening = False

    async def _a_read(self):
        try:
            async for raw in self._ws:
                self._on_msg(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:                                               # noqa: BLE001
            asyncio.get_running_loop().create_task(self._a_reopen(f"{type(e).__name__}: {str(e)[:120]}"))

    def _on_msg(self, m):
        t = m.get("type")
        if t in ("response.output_text.delta", "response.text.delta"):
            self._add_text(m.get("delta", ""))
        elif t == "response.done":
            r = m.get("response") or {}
            u = r.get("usage") or {}
            self.tokens["prompt"] += u.get("input_tokens") or 0
            self.tokens["completion"] += u.get("output_tokens") or 0
            st = r.get("status")
            if self._cur is None:
                self.events.append({"t_wall": round(time.time(), 3), "kind": "silent_turn", "status": st})
            self._finish(interrupted=(st == "cancelled"), status=st)
        elif t == "error":
            err = m.get("error") or {}
            if err.get("code") not in ("response_cancel_not_active",):
                self._event("api_error", error=json.dumps(err)[:200])

    async def _send(self, obj):
        async with self._send_lock:
            await self._ws.send(json.dumps(obj))

    async def _item_text(self, text):
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})

    async def _a_frame(self, frame, tick):
        mime = "image/jpeg" if frame[:3] == b"\xff\xd8\xff" else "image/png"
        iid = f"frame_{tick}_{int(time.time() * 1000) % 100000}"
        for attempt in (0, 1):
            try:
                await self._send({"type": "conversation.item.create", "item": {
                    "id": iid, "type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": f"[t={tick}]"},
                        {"type": "input_image",
                         "image_url": f"data:{mime};base64," + base64.b64encode(frame).decode()}]}})
                self._frame_ids.append(iid)
                while len(self._frame_ids) > self.KEEP_FRAMES:
                    await self._send({"type": "conversation.item.delete", "item_id": self._frame_ids.pop(0)})
                return
            except Exception as e:                                           # noqa: BLE001
                if attempt:
                    self._event("frame_dropped", tick=tick, error=str(e)[:120])
                    return
                await self._a_reopen(f"send failed: {str(e)[:100]}")

    async def _a_audio(self, pcm):
        silence = b"\x00\x00" * int(PCM_RATE * SILENCE_PAD_S)
        for attempt in (0, 1):
            try:
                for c in _chunks(pcm, 9600):                      # 200 ms chunks
                    await self._send({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(c).decode()})
                for c in _chunks(silence, 9600):
                    await self._send({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(c).decode()})
                return
            except Exception as e:                                           # noqa: BLE001
                if attempt:
                    self._event("audio_dropped", error=str(e)[:120])
                    return
                await self._a_reopen(f"send failed: {str(e)[:100]}")


def make_native(kind: str, model: str) -> NativeSession:
    if kind == "live_native":
        return LiveNativeSession(model)
    if kind == "realtime_native":
        return RealtimeNativeSession(model)
    raise SystemExit(f"unknown native backend {kind!r}")
