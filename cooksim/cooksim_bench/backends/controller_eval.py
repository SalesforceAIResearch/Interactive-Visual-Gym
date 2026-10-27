#!/usr/bin/env python3
"""VLM-as-CONTROLLER evaluation across model backends — the "human agent alone" test.

The cook is driven by ONE model, no assistant, no injection: does the model finish the recipe,
how fast, at what latency and cost? Same controller as python -m cooksim_bench.backends.gemini_stream_v3 (decision every
POLL=2 ticks; the model's previous JSON is its only memory; full engine feedback on every
command; strict v2 grounding), with the model behind a pluggable backend:

    --backend gemini  --model gemini-3-flash-preview
    --backend openai  --model <served name> --base-url http://127.0.0.1:18100/v1     (vLLM / SGLang / any OpenAI-compatible local server)

Observation modes (--mode):
    text_state_map   the cook's OWN v2 state map (ids + coordinates, ASCII grid) + full feedback
    agent+++         last 8 rendered top-down frames + position/facing + exact plate order + full feedback

Cells: the benchmark's 10 (5 recipes x {medium, hard} maps) by default.

    source tmp/present/env.sh; . ./gpu_env.sh
    $PY python -m cooksim_bench.backends.controller_eval --backend gemini --model gemini-3-flash-preview --mode text_state_map \
        --out-dir tmp/ctrl_eval/gemini-3-flash-preview
Writes one JSON per (mode, cell) with every turn (action, disposition, feedback, structure,
latency_ms, tokens) plus a summary line. python -m cooksim_bench.scoring.controller_eval_report aggregates.
"""
from __future__ import annotations
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, base64, json, os, re, sys, time
try:
    from asyncio import timeout as timeout_context
except ImportError:  # Python 3.10
    from async_timeout import timeout as timeout_context

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

from cooksim.core.enums import Action                                             # noqa: E402
from cooksim.agents.highlevel import HighLevelController                          # noqa: E402
from cooksim.agents.observe_text import observe_text                              # noqa: E402
from cooksim_bench.backends.gemini_stream_rgb import make_gen_game, GPU, PAGE                            # noqa: E402
from cooksim_bench.agent_context import dir_name, plate_in_order                                # noqa: E402
from cooksim_bench.backends.cmd_feedback import FeedbackTracker                                          # noqa: E402
from cooksim_bench.backends.planact_v3_prompts import build_prompt, build_turn                           # noqa: E402

POLL, WINDOW = 2, 8
CAP = int(os.environ.get("CTRL_CAP", "300"))
NOOP = {"", "continue", "none", "no-op", "noop", "wait", "stay", "null"}
CELLS10 = [f"{n}_nops_{m}_map" for n in ("easy", "medium", "hard", "expert", "master") for m in ("medium", "hard")]

# v2 grounding contract for the TEXT mode: the map prints ids; the parser accepts them.
V2_TEXT_ADDENDUM = """
GROUNDING (v2): the state map prints every station/source/counter with a public ID ("pan 2",
"cutting board 1", "potato source 2") and its (x,y). When a kind has MORE THAN ONE instance you
MUST say which — by ID ("go to pan 2"), by coordinate ("go to the pan at (6,4)"), or by contents/
state when that picks out exactly one ("the pan with the steak", "the empty fryer"). A bare name
for a duplicated kind is REFUSED with the list of ids; re-issue naming one. Never guess."""


# ---------------------------------------------------------------- backends
class Backend:
    name = "?"
    def call(self, system: str, user: str, images: list[bytes] | None) -> tuple[dict, dict]:
        raise NotImplementedError


def _mime(f: bytes) -> str:
    """the frame's real format: the renderer emits JPEG; Anthropic rejects a mislabelled image outright."""
    return "image/jpeg" if f[:3] == b"\xff\xd8\xff" else "image/png"


def _parse_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    # a reply that lost its opening brace ('"cook_progress": {...') would otherwise parse as the inner
    # object and drop text/flag (seen on the realtime seat, which has no JSON mode)
    if t.startswith('"') and '":' in t[:40]:
        t = "{" + t
    i = t.find("{")
    if i < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r'"action"\s*:\s*"([^"]*)"', t)
        return {"action": m.group(1) if m else "", "reason": "(salvaged from malformed JSON)"}


class GeminiBackend(Backend):
    """Gemini, stateless. Per Google's Gemini 3 guide: the system prompt goes in system_instruction, the
    temperature stays at its default (lower values can loop or degrade), and thinking is set by level, not
    budget, on the 3.x models ("low": latency is charged to the game clock). Older models keep a budget."""
    def __init__(self, model: str, thinking_budget=640):
        from google import genai
        from google.genai import types
        self.genai, self.types = genai, types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model, self.name = model, model
        if thinking_budget == 640 and model.startswith("gemini-3"):
            thinking_budget = "low"           # the harness default budget, translated for the 3.x models
        self.thinking = thinking_budget

    def call(self, system, user, images):
        types = self.types
        parts = [user] + [types.Part.from_bytes(data=f, mime_type=_mime(f)) for f in (images or [])]
        ck = dict(system_instruction=system, response_mime_type="application/json", max_output_tokens=6000)
        if self.thinking is not None:
            ck["thinking_config"] = (types.ThinkingConfig(thinking_level=self.thinking) if isinstance(self.thinking, str)
                                     else types.ThinkingConfig(thinking_budget=self.thinking))
        last = None
        for k in range(5):
            t0 = time.monotonic()
            try:
                r = self.client.models.generate_content(model=self.model, contents=parts,
                                                        config=types.GenerateContentConfig(**ck))
                ms = int((time.monotonic() - t0) * 1000)
                um = getattr(r, "usage_metadata", None)
                meta = {"latency_ms": ms,
                        "prompt_tokens": getattr(um, "prompt_token_count", None),
                        "completion_tokens": getattr(um, "candidates_token_count", None),
                        "thinking_tokens": getattr(um, "thoughts_token_count", None),
                        "total_tokens": getattr(um, "total_token_count", None)}
                return _parse_json(r.text or ""), meta
            except Exception as e:
                last = e; msg = str(e)
                if "thinking" in msg.lower() and "thinking_config" in ck:
                    ck.pop("thinking_config"); self.thinking = None; continue    # model has no thinking
                if any(x in msg for x in ("503", "500", "429", "RESOURCE", "UNAVAIL", "INTERNAL", "disconnect", "Connection")):
                    time.sleep(2 + 2 * k); continue
                raise
        raise RuntimeError(f"gemini call failed after retries: {last}")


def _reply_schema():
    """The assistant reply schema (cook_progress / goal_change / text / flag), shared by every schema profile."""
    global _REPLY_SCHEMA
    try:
        return _REPLY_SCHEMA
    except NameError:
        _REPLY_SCHEMA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant_reply_schema.json")))
        return _REPLY_SCHEMA


# Per-model request profiles for OpenAI-compatible local servers. Each model's own docs
# decide the sampling; the harness must not impose Gemini's temperature on a model whose
# guide says "do not run it greedy".
PROFILES = {
    "default": dict(temperature=0.2, extra_body={}, prefill=None),
    # Muse Glimmer 30B (meta prompting guide): T 1.0 / top_p 0.95 / top_k 64; reasoning
    # strength via chat_template_kwargs; NO-THINK by pre-filling the assistant turn and
    # continuing it -- ~125 completion tokens instead of ~450, the main latency lever.
    "museglimmer-nothink": dict(temperature=1.0, top_p=0.95,
        extra_body={"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "low"},
                    "add_generation_prompt": False, "continue_final_message": True},
        prefill={"role": "assistant", "recipient": "user", "content": ""}),
    "museglimmer-low": dict(temperature=1.0, top_p=0.95,
        extra_body={"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "low"}}, prefill=None),
    # MiniCPM-o 4.5 (model card + smoke test): images BEFORE the text part; thinking off via
    # chat_template_kwargs; greedy-ish sampling was 30/30 valid JSON in the smoke test.
    # MiniCPM-o sampling = the CookBook's AWQ-on-vLLM values (audit 2026-09-19; T0.1 let vLLM add generation_config's
    # top_p/top_k silently). 1.81 s median vs 2.10 s, 24/24 both.
    "minicpmo": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 100, "repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}},
                     prefill=None, images_first=True),
    # Local pass@3 eval (2026-09-15). Plain json_object mode left most local replies without the reply schema
    # (setup bench on 24 real round-4 calls: Qwen3.8-27B 8/24, Muse Glimmer 11/24 had cook_progress/text/flag);
    # guided decoding on cooksim_bench/assistant_reply_schema.json -- the exact reply the assistant prompt asks for --
    # fixes the format and changes nothing the model is asked.
    # Qwen3.8-27B non-thinking sampling per its model card: T 0.7 / top_p 0.8 / top_k 20 / presence 1.5.
    "qwen38-schema": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 20, "presence_penalty": 1.5},
                          prefill=None, reply_schema=True),
    "museglimmer-nothink-schema": dict(temperature=1.0, top_p=0.95,
        extra_body={"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "low"},
                    "add_generation_prompt": False, "continue_final_message": True},
        prefill={"role": "assistant", "recipient": "user", "content": ""}, reply_schema=True),
    "minicpmo-schema": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 100, "repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}},
                            prefill=None, images_first=True, reply_schema=True),
    # Molmo2-8B (setup bench 2026-09-15): default sampling + the reply schema; 24/24 valid, median 5.1 s with frames
    # at --assist-image-scale 0.5 (13.7k -> 8.5k prompt tokens). The server's chat template folds the system text
    # into the first user turn (the shipped template rejects a system message).
    # greedy = AllenAI's own vLLM demo (audit 2026-09-19): T0.2 ran to the token cap once in 96 calls, greedy never
    "molmo2-schema": dict(temperature=0.0, extra_body={}, prefill=None, reply_schema=True),
    # GLM-4.6V-Flash: same shape as the other served open-weight models (schema on, no thinking).
    # the model card's sampling (audit 2026-09-19): caught the call-2 wrong_order in 4/4 runs vs 0/3 at T0.2; +0.25 s
    "glm46v-flash-schema": dict(temperature=0.8, top_p=0.6,
                                extra_body={"top_k": 2, "repetition_penalty": 1.1,
                                            "chat_template_kwargs": {"enable_thinking": False}},
                                prefill=None, reply_schema=True),
    # --- setup batch 2026-09-18 (local_models/<name>/SETUP.md has every number) ---------------------------
    # InternVideo3-8B (transformers shim, frames as one clip): the model's own generation_config sampling.
    # 22/24 valid; the 2 failures are runaways that repeat subgoals_completed to the token cap, so run it with
    # --assist-max-tokens 1024 (longest valid reply 668 tokens). T0.2 was worse: 21/24, p90 158 s.
    # + repetition_penalty 1.05 (audit 2026-09-19): the official settings alone still ran away on 6/24 stress calls, 0/24 with it
    "internvideo3-schema": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 20, "repetition_penalty": 1.05},
                                prefill=None, reply_schema=True),
    # Qwen3-Omni-30B-A3B-Instruct (plain vLLM, text only; the talker is skipped at load): model-card sampling.
    # 24/24 valid, median 3.1 s.
    "qwen3-omni-schema": dict(temperature=0.6, top_p=0.95, extra_body={"top_k": 20}, prefill=None, reply_schema=True),
    # Kimi-VL-A3B-Thinking-2506 (needs the ◁think▷ reasoning-parser plugin in serving/): thinking on is the
    # roster entry; the two others are the row-14 ablation (does thinking help when the clock runs?).
    # on: 23/24, median 10.5 s, caught both textbook errors · budget 256: 24/24, 6.2 s · off: 24/24, 3.7 s.
    # frames first (the authors' order) and a 2,048-token thinking budget (audit 2026-09-19): 24/24 vs 20/24, max 30.8 s vs 60.4 s
    "kimi-vl-think-schema": dict(temperature=0.8, extra_body={"thinking_token_budget": 2048}, prefill=None,
                                 reply_schema=True, images_first=True),
    "kimi-vl-think256-schema": dict(temperature=0.8, extra_body={"thinking_token_budget": 256}, prefill=None,
                                    reply_schema=True, images_first=True),
    "kimi-vl-nothink-schema": dict(temperature=0.8,
                                   extra_body={"add_generation_prompt": False, "continue_final_message": True},
                                   prefill={"role": "assistant", "content": "\u25c1think\u25b7\u25c1/think\u25b7"},
                                   reply_schema=True, images_first=True),
    # OpenAI API, GPT-5.x: temperature must stay at its default, the reply budget is
    # max_completion_tokens, and reasoning effort is a request field ("low" for latency).
    # the same, on the Responses API (OpenAI's recommended surface for reasoning models; audit 2026-09-18:
    # gpt-5.5 6.0-6.6 s vs 8.0-10.2 s on chat completions, 25-35% fewer reasoning tokens)
    "gpt5-responses": dict(temperature=None, extra_body={}, prefill=None, responses_api=True,
                           responses_body={"reasoning": {"effort": "low"}, "prompt_cache_key": "cooksim-assistant"}),
    "gpt5": dict(temperature=None, extra_body={"reasoning_effort": "low", "prompt_cache_key": "cooksim-assistant"},
                 prefill=None,
                 max_completion_tokens=True),
    # ---- COOK-SEAT profiles (cook-model ablation 2026-09-22) ------------------------------
    # The cook does NOT emit the assistant reply schema (it emits utterance / move /
    # close_conversation / plan_update / assistant_error_call), so a cook profile must leave
    # reply_schema OFF -- plain json_object mode, which is what the shared Gemini cook has
    # always used. Sampling stays each model's own card values, exactly as in the assistant
    # roster; only the schema constraint and the cache key differ from the assistant twin.
    "gpt5-cook": dict(temperature=None, extra_body={"reasoning_effort": "low", "prompt_cache_key": "cooksim-cook"},
                      prefill=None, max_completion_tokens=True),
    # Qwen3.8-27B non-thinking card sampling (== "qwen38-schema" minus the assistant schema)
    "qwen38-cook": dict(temperature=0.7, top_p=0.8, extra_body={"top_k": 20, "presence_penalty": 1.5},
                        prefill=None),
    # Muse Glimmer 30B card sampling, no-think prefill (== "museglimmer-nothink-schema" minus the schema)
    "museglimmer-cook": dict(temperature=1.0, top_p=0.95,
        extra_body={"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "low"},
                    "add_generation_prompt": False, "continue_final_message": True},
        prefill={"role": "assistant", "recipient": "user", "content": ""}),
}


class RealtimeBackend(Backend):
    """OpenAI Realtime API (GA shape) as a text cook: one WebSocket session per episode,
    output_modalities=["text"], the system prompt as session instructions, each ask as a
    user message + response.create. Mirrors LiveBackend so the two "live" seats compare."""

    ASK_TIMEOUT = 60.0

    def __init__(self, model: str):
        import asyncio, threading
        self.asyncio = asyncio
        self.model, self.name = model, model
        self.key = os.environ["OPENAI_API_KEY"]
        self._ws = None
        self._sys = None
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def _run(self, coro, timeout):
        return self.asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _a_open(self, system):
        import websockets
        await self._a_close()
        self._ws = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            additional_headers={"Authorization": f"Bearer {self.key}"}, max_size=None)
        await self._ws.recv()                                   # session.created
        # every ask carries the full state, so old turns are dead weight: bound the context (median prompt was
        # 47k tokens, max 134k, before), pin the reasoning effort the GPT seats use, cap the reply
        await self._ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["text"], "instructions": system,
            "reasoning": {"effort": "low"}, "max_output_tokens": 4000,
            "truncation": {"type": "retention_ratio", "retention_ratio": 0.5,
                           "token_limits": {"post_instructions": 16000}}}}))

    async def _a_close(self):
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def _a_ask(self, user, images=None):
        t0 = time.time()
        content = []
        for f in (images or []):
            mime = "image/jpeg" if f[:3] == b"\xff\xd8\xff" else "image/png"
            content.append({"type": "input_image",
                            "image_url": f"data:{mime};base64," + base64.b64encode(f).decode()})
        content.append({"type": "input_text", "text": user})
        await self._ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": content}}))
        await self._ws.send(json.dumps({"type": "response.create",
                                        "response": {"output_modalities": ["text"]}}))
        text, usage = "", None
        async with timeout_context(self.ASK_TIMEOUT):
            while True:
                m = json.loads(await self._ws.recv())
                t = m.get("type")
                if t in ("response.output_text.delta", "response.text.delta"):
                    text += m.get("delta", "")
                elif t == "response.done":
                    usage = (m.get("response") or {}).get("usage") or {}
                    break
                elif t == "error":
                    raise RuntimeError(f"realtime error: {json.dumps(m)[:200]}")
        return text, usage, int((time.time() - t0) * 1000)

    def reset_episode(self, system):
        self._sys = system
        self._run(self._a_open(system), 60)

    def call(self, system, user, images):
        for attempt in (0, 1):
            try:
                text, usage, ms = self._run(self._a_ask(user, (images or [])[-2:]), self.ASK_TIMEOUT + 10)
                break
            except Exception:
                if attempt:
                    raise
                self._run(self._a_open(self._sys or system), 60)
        meta = dict(latency_ms=ms, prompt_tokens=(usage or {}).get("input_tokens"),
                    completion_tokens=(usage or {}).get("output_tokens"), thinking_tokens=0)
        return _parse_json(text), meta


class LiveBackend(Backend):
    """gemini Live API (stateful WebSocket session) as a controller backend.

    One session per EPISODE (reset_episode). The model only accepts AUDIO response modality, so it
    SPEAKS its JSON and we read the output_audio_transcription text -- the probe showed this parses
    75/75 on a long run when the ask goes via send_client_content(turn_complete=True). The session
    itself is the memory (context compression 16k->8k keeps it bounded); the turn text still carries
    prev_json exactly like the stateless backends so the information per ask is identical.
    In frame mode only the frames NEW since the last ask are pushed (send_realtime_input(video=...));
    older ones are already in the session. A dropped connection reconnects once (full state is in
    every turn, so the lost session costs plan memory only)."""

    ASK_TIMEOUT = 45.0

    def __init__(self, model: str):
        import asyncio, threading
        from google import genai
        from google.genai import types
        self.asyncio, self.genai, self.types = asyncio, genai, types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model, self.name = model, model
        self._cm = self._session = self._reader = None
        self._sys = None
        self._q = None
        self._thinking = "minimal"      # the lowest level the model accepts; see _a_open
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def _run(self, coro, timeout):
        return self.asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # ---- session lifecycle (all _a* run on the loop thread) ----
    def _config(self, system):
        t = self.types
        extra = {}
        if self._thinking:
            extra["thinking_config"] = t.ThinkingConfig(thinking_level=self._thinking)
        return t.LiveConnectConfig(
            system_instruction=t.Content(parts=[t.Part(text=system)]),
            response_modalities=["AUDIO"],
            output_audio_transcription=t.AudioTranscriptionConfig(),
            context_window_compression=t.ContextWindowCompressionConfig(
                trigger_tokens=16000, sliding_window=t.SlidingWindow(target_tokens=8000)),
            **extra)

    async def _a_open(self, system):
        await self._a_close()
        self._q = self.asyncio.Queue()
        try:
            self._cm = self.client.aio.live.connect(model=self.model, config=self._config(system))
            self._session = await self._cm.__aenter__()
        except Exception as e:                                          # noqa: BLE001
            # The Live models disagree about thinking: gemini-3.8-live refuses any level, the 3.1 Live
            # model takes "minimal", the 3.8 extended-thinking model requires more than "minimal".
            # Adjust once on refusal and keep the setting for the rest of the episode.
            msg = str(e)
            if self._thinking and "Thinking level is not supported" in msg:
                self._thinking = None
            elif self._thinking == "minimal" and "MINIMAL is not supported" in msg:
                self._thinking = "low"
            else:
                raise
            self._cm = self.client.aio.live.connect(model=self.model, config=self._config(system))
            self._session = await self._cm.__aenter__()
        self._reader = self.asyncio.create_task(self._a_read())

    async def _a_read(self):
        try:
            while True:
                async for m in self._session.receive():
                    await self._q.put(m)
        except Exception as e:                            # noqa: BLE001 -- surfaced to ask() as None
            self._err = f"{type(e).__name__}: {str(e)[:200]}"
            await self._q.put(None)

    async def _a_close(self):
        if self._reader is not None:
            self._reader.cancel()
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._cm = self._session = self._reader = None

    async def _a_ask(self, user, new_frames):
        t = self.types
        while not self._q.empty():
            self._q.get_nowait()
        t0 = time.time()
        # The frames go INLINE in the question turn. Sent as realtime video (send_realtime_input) they are
        # not ordered with the turn: the ask was answered before they arrived (0 images on the first ask,
        # the previous ask's frames afterwards; probe 2026-09-18, both Live models).
        parts = [t.Part(inline_data=t.Blob(data=f, mime_type=_mime(f))) for f in (new_frames or [])]
        await self._session.send_client_content(
            turns=t.Content(role="user", parts=parts + [t.Part(text=user)]), turn_complete=True)
        text, usage = "", None
        async with timeout_context(self.ASK_TIMEOUT):
            while True:
                m = await self._q.get()
                if m is None:
                    raise ConnectionError(getattr(self, "_err", "live session dropped"))
                if m.usage_metadata is not None:
                    usage = m.usage_metadata
                sc = m.server_content
                if sc is not None:
                    if sc.model_turn is not None:
                        text += "".join(p.text or "" for p in (sc.model_turn.parts or [])
                                        if not getattr(p, "thought", False))
                    if sc.output_transcription is not None and sc.output_transcription.text:
                        text += sc.output_transcription.text
                    if sc.turn_complete:
                        break
        ms = int((time.time() - t0) * 1000)               # clock stops at turn_complete, not at the drain
        if usage is None or not getattr(usage, "response_token_count", None):
            try:                                          # usage often trails the turn_complete
                async with timeout_context(0.6):
                    while True:
                        m = await self._q.get()
                        if m is not None and m.usage_metadata is not None:
                            usage = m.usage_metadata
                            if getattr(usage, "response_token_count", None):
                                break
            except TimeoutError:
                pass
        return text, usage, ms

    # ---- Backend interface ----
    def reset_episode(self, system):
        self._sys = system
        self._sent = 0                                    # frames already in the session
        self._run(self._a_open(system), 60)

    def call(self, system, user, images):
        new = None
        if images:
            new = images[max(0, len(images) - max(1, POLL)):]   # only frames since the last ask
        for attempt in (0, 1):
            try:
                text, usage, ms = self._run(self._a_ask(user, new), self.ASK_TIMEOUT + 10)
                break
            except Exception:
                if attempt:                                # second failure ends up as a backend error
                    raise
                self._run(self._a_open(self._sys), 60)     # reconnect once, retry the same ask
        meta = dict(latency_ms=ms,
                    prompt_tokens=getattr(usage, "prompt_token_count", None) or 0,
                    completion_tokens=getattr(usage, "response_token_count", None) or 0,
                    thinking_tokens=getattr(usage, "thoughts_token_count", None) or 0)
        return _parse_json(text), meta


class OpenAIBackend(Backend):
    """Any OpenAI-compatible chat-completions server (vLLM, SGLang, a FastAPI wrapper)."""
    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY", json_mode: bool = True,
                 max_tokens: int = 1200, profile: str = "default", image_scale: float = 1.0,
                 image_format: str = "png"):
        import requests
        self.requests = requests
        self.model, self.name = model, model
        self.url = base_url.rstrip("/") + "/chat/completions"
        if "api.openai.com" in base_url and api_key == "EMPTY":
            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        elif "dashscope" in base_url and api_key == "EMPTY":      # Alibaba Model Studio (Qwen API models)
            api_key = os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        self.key, self.json_mode, self.max_tokens = api_key, json_mode, max_tokens
        self.profile = PROFILES[profile]
        self.image_scale, self.image_format = image_scale, image_format

    def _img(self, f: bytes) -> str:
        if self.image_scale != 1.0 or self.image_format != "png":
            from PIL import Image; import io
            im = Image.open(io.BytesIO(f)).convert("RGB")
            if self.image_scale != 1.0:
                im = im.resize((int(im.width * self.image_scale), int(im.height * self.image_scale)))
            buf = io.BytesIO(); im.save(buf, format="JPEG" if self.image_format == "jpeg" else "PNG", quality=85)
            f = buf.getvalue()
        return f"data:{_mime(f)};base64," + base64.b64encode(f).decode()

    def _call_responses(self, system, user, images):
        """OpenAI Responses API (profile responses_api=True): OpenAI's recommended surface for its reasoning
        models. Same system prompt, turn, frames and plain JSON mode as the chat path."""
        content = [{"type": "input_text", "text": user}] + \
                  [{"type": "input_image", "image_url": self._img(f)} for f in (images or [])]
        body = {"model": self.model, "instructions": system,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": self.max_tokens, "store": False,
                "text": {"format": {"type": "json_object"}}}
        body.update(self.profile.get("responses_body") or {})
        url = self.url.replace("/chat/completions", "/responses")
        last = None
        for k in range(4):
            t0 = time.monotonic()
            try:
                r = self.requests.post(url, json=body, timeout=180, headers={"Authorization": f"Bearer {self.key}"})
                r.raise_for_status()
                ms = int((time.monotonic() - t0) * 1000)
                j = r.json()
                text = "".join(c.get("text", "") for o in j.get("output") or [] if o.get("type") == "message"
                               for c in o.get("content") or [] if c.get("type") == "output_text")
                u = j.get("usage") or {}
                meta = {"latency_ms": ms, "prompt_tokens": u.get("input_tokens"),
                        "completion_tokens": u.get("output_tokens"),
                        "thinking_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                        "total_tokens": u.get("total_tokens")}
                return _parse_json(text), meta
            except Exception as e:
                last = e; time.sleep(1 + k)
        raise RuntimeError(f"responses call failed: {last}")

    def call(self, system, user, images):
        if self.profile.get("responses_api"):
            return self._call_responses(system, user, images)
        imgs = [{"type": "image_url", "image_url": {"url": self._img(f)}} for f in (images or [])]
        content = (imgs + [{"type": "text", "text": user}]) if self.profile.get("images_first") \
                  else ([{"type": "text", "text": user}] + imgs)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        if self.profile.get("prefill"):
            msgs.append(dict(self.profile["prefill"]))
        body = {"model": self.model, "messages": msgs}
        if self.profile.get("temperature") is not None:
            body["temperature"] = self.profile["temperature"]
        if self.profile.get("max_completion_tokens"):
            body["max_completion_tokens"] = self.max_tokens
        else:
            body["max_tokens"] = self.max_tokens
        if self.profile.get("top_p") is not None:
            body["top_p"] = self.profile["top_p"]
        if self.profile.get("extra_body"):
            body.update(self.profile["extra_body"])
        if self.profile.get("reply_schema"):
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "assistant_reply", "strict": True, "schema": _reply_schema()}}
        elif self.json_mode:
            body["response_format"] = {"type": "json_object"}
        last = None
        for k in range(4):
            t0 = time.monotonic()
            try:
                r = self.requests.post(self.url, json=body, timeout=180,
                                       headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 400 and self.json_mode and "response_format" in r.text:
                    self.json_mode = False; body.pop("response_format", None); continue
                r.raise_for_status()
                ms = int((time.monotonic() - t0) * 1000)
                j = r.json(); text = j["choices"][0]["message"]["content"]
                u = j.get("usage") or {}
                meta = {"latency_ms": ms, "prompt_tokens": u.get("prompt_tokens"),
                        "completion_tokens": u.get("completion_tokens"), "thinking_tokens": None,
                        "total_tokens": u.get("total_tokens")}
                return _parse_json(text), meta
            except Exception as e:
                last = e; time.sleep(1 + k)
        raise RuntimeError(f"openai-compatible call failed: {last}")


# ---------------------------------------------------------------- episode
def mode_extras(mode, tracker, game):
    out = [tracker.block(level="full")]
    cook = game.cooks[0]
    if mode == "agent+++":
        out.append(f"CURRENT POSITION: the cook is at {tuple(cook.pos)} facing {dir_name(cook.direction)}.")
        out.append(f"CURRENT PLATE (components in the exact order they were added): {plate_in_order(cook.holding)}")
    return "\n\n".join(out)


def run_episode(backend: Backend, mode: str, cell: str, page=None, cap: int = CAP, verbose=True):
    game, recipe = make_gen_game(cell)
    ctrl = HighLevelController(game, 0)
    tracker = FeedbackTracker(k=5)
    sys_prompt = build_prompt(mode) + (V2_TEXT_ADDENDUM if mode == "text_state_map" else "")
    if hasattr(backend, "reset_episode"):                # session backends: one session per episode
        backend.reset_episode(sys_prompt)
    prev_json, fbuf, turns, path = None, [], [], []
    start_wins = game.stats.get("deliveries", 0)
    t = 0; wall0 = time.time(); errors = 0
    for t in range(cap):
        path.append(tuple(game.cooks[0].pos))
        if mode != "text_state_map":
            url = page.evaluate("(s)=>window.__renderFrame(s)", json.dumps(game.render_state()))
            fbuf.append(base64.b64decode(url.split(",", 1)[1]))
        if t % POLL == 0:
            tracker.refresh(ctrl)
            user = build_turn(mode, recipe, t, prev_json, mode_extras(mode, tracker, game))
            images = None
            if mode == "text_state_map":
                user += "\n\n" + observe_text(game, path=path[-WINDOW:], viewpoint="cook")
            else:
                images = fbuf[-WINDOW:]
            try:
                out, meta = backend.call(sys_prompt, user, images)
            except Exception as e:                       # a dead backend ends the episode honestly
                errors += 1
                turns.append(dict(t=t, error=f"{type(e).__name__}: {str(e)[:200]}"))
                if verbose: print(f"[t{t:03}] BACKEND ERROR {e}", flush=True)
                if errors >= 5:
                    break
                continue
            act = str(out.get("action", "") or "").strip()
            disp, fb = "noop", ""
            if act and act.lower() not in NOOP:
                _ex, fb, disp = tracker.submit(act, game, ctrl, strict=True)
            prev_json = json.dumps(out) if out else prev_json
            turns.append(dict(t=t, action=act, disposition=disp, feedback=fb, structure=out, **meta))
            if verbose:
                print(f"[t{t:03}] {meta['latency_ms']:5d}ms {act!r:42} -> {disp} {fb[:60]}", flush=True)
        low = ctrl.step() if ctrl.skill is not None else int(Action.STAY)
        acts = [int(Action.STAY)] * len(game.cooks); acts[0] = low
        game.step(acts)
        if game.stats.get("deliveries", 0) > start_wins or game.stats.get("failed_deliveries", 0) > 0 \
                or game.stats.get("kitchen_fire") or any(getattr(s, "status", None) == "burnt" for s in game.stations.values()):
            break
    won = game.stats.get("deliveries", 0) > start_wins
    outcome = ("won" if won else "wrong_serve" if game.stats.get("failed_deliveries", 0) > 0
               else "fire" if game.stats.get("kitchen_fire")
               else "burned" if any(getattr(s, "status", None) == "burnt" for s in game.stations.values())
               else "timeout")
    lat = sorted(x["latency_ms"] for x in turns if "latency_ms" in x)
    ptok = [x.get("prompt_tokens") or 0 for x in turns if "latency_ms" in x]
    ctok = [x.get("completion_tokens") or 0 for x in turns if "latency_ms" in x]
    ttok = [x.get("thinking_tokens") or 0 for x in turns if "latency_ms" in x]
    return dict(schema="controller-eval-v1", backend=backend.name, mode=mode, cell=cell, won=won, outcome=outcome,
                ticks=t + 1, decisions=len([x for x in turns if "latency_ms" in x]),
                backend_errors=errors, wall_s=round(time.time() - wall0, 1),
                latency_ms={"median": lat[len(lat) // 2] if lat else None,
                            "p90": lat[min(len(lat) - 1, int(len(lat) * 0.9))] if lat else None,
                            "mean": round(sum(lat) / len(lat), 1) if lat else None},
                tokens={"prompt": sum(ptok), "completion": sum(ctok), "thinking": sum(ttok)},
                dispositions={d: sum(1 for x in turns if x.get("disposition") == d)
                              for d in ("executing", "refused", "grounding", "noop")},
                turns=turns)


def make_backend(args) -> Backend:
    if args.backend == "live":
        return LiveBackend(args.model)
    if args.backend == "gemini":
        return GeminiBackend(args.model, thinking_budget=(None if args.no_thinking else args.thinking))
    return OpenAIBackend(args.model, args.base_url, json_mode=not args.no_json_mode, max_tokens=args.max_tokens,
                         profile=args.profile, image_scale=args.image_scale, image_format=args.image_format)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["gemini", "openai", "live"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--mode", choices=["text_state_map", "agent+++"], required=True)
    ap.add_argument("--cells", nargs="*", default=CELLS10)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--thinking", type=int, default=640)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--profile", default="default", choices=sorted(PROFILES))
    ap.add_argument("--image-scale", type=float, default=1.0, help="downscale frames before sending (0.5 = 400x290)")
    ap.add_argument("--image-format", default="png", choices=["png", "jpeg"])
    ap.add_argument("--tag", default=None, help="run tag for the filename (default: model)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    backend = make_backend(args)
    from cooksim_bench.engine_stamp import stamp
    rev = stamp()["rev"]

    def _run(page):
        for cell in args.cells:
            outp = os.path.join(args.out_dir, f"{tag}__{args.mode}__{cell}.json")
            if os.path.exists(outp):
                print(f"skip (done) {outp}"); continue
            print(f"=== {backend.name} / {args.mode} / {cell}", flush=True)
            rec = run_episode(backend, args.mode, cell, page=page, cap=args.cap)
            rec["engine_rev"] = rev
            json.dump(rec, open(outp, "w"), indent=1)
            print(f"--> {rec['outcome']} ticks={rec['ticks']} decisions={rec['decisions']} "
                  f"lat_med={rec['latency_ms']['median']}ms tokens={rec['tokens']}", flush=True)

    if args.mode == "text_state_map":
        _run(None)
    else:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(args=GPU)
            page = b.new_page(viewport={"width": 800, "height": 580})
            page.goto(PAGE, wait_until="load")
            page.wait_for_function("window.__ready===true", timeout=90000)
            page.wait_for_timeout(400)
            _run(page)
            b.close()


if __name__ == "__main__":
    main()
