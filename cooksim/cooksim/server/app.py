"""FastAPI + WebSocket server.

Serves the WebGL front-end and exposes:

* ``GET  /``                  the live game / player UI
* ``GET  /editor``            the visual level editor
* ``GET  /api/layouts``       list of built-in layouts
* ``GET  /api/layout/{name}`` one layout as JSON
* ``GET  /api/recipes``       the recipe book
* ``POST /api/validate``      validate a layout dict
* ``POST /api/generate``      procedurally generate a layout
* ``WS   /ws``                a live, controllable game session

WebSocket protocol — client -> server JSON messages::

    {"type":"reset", "layout":"open_kitchen", "n_players":2, "seed":0}
    {"type":"reset", "layout_data":{...}}            # custom layout
    {"type":"reset", "procedural":{"width":11,...}}  # random layout
    {"type":"action", "player":0, "action":4}        # persistent action
    {"type":"set_speed", "tps":7}                    # 0 pauses
    {"type":"add_bot", "player":1, "kind":"greedy"}
    {"type":"remove_bot", "player":1}
    {"type":"step"}                                  # single manual tick

server -> client::

    {"type":"init", "recipes":[...], "layouts":[...], "state":{...}}
    {"type":"state", "state":{...}}
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.layout import Layout, generate_layout
from ..core.recipes import RecipeBook
from ..layouts import all_layouts, list_layouts, load_layout
from .session import GameSession

WEB_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "web"))

app = FastAPI(title="CookSim", version="0.1.0")
_RECIPES = RecipeBook()


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/editor")
def editor():
    return FileResponse(os.path.join(WEB_DIR, "editor.html"))


@app.get("/api/layouts")
def api_layouts():
    return {"layouts": list_layouts()}


@app.get("/api/layout/{name}")
def api_layout(name: str):
    try:
        return load_layout(name).to_dict()
    except KeyError:
        return JSONResponse({"error": f"unknown layout {name}"}, status_code=404)


@app.get("/api/recipes")
def api_recipes():
    return {"recipes": _RECIPES.to_dict()}


@app.post("/api/validate")
async def api_validate(payload: dict):
    try:
        lay = Layout.from_dict(payload)
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "issues": [f"parse error: {e}"]}
    issues = lay.validate()
    return {"valid": not issues, "issues": issues}


@app.post("/api/generate")
async def api_generate(payload: dict):
    lay = generate_layout(**payload)
    return lay.to_dict()


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    session = GameSession()
    await ws.send_json(
        {
            "type": "init",
            "recipes": _RECIPES.to_dict(),
            "layouts": list_layouts(),
            "state": session.state(),
        }
    )

    stop = asyncio.Event()

    async def game_loop():
        while not stop.is_set():
            if session.tps > 0:
                session.tick()
                try:
                    await ws.send_json({"type": "state", "state": session.state()})
                except Exception:
                    stop.set()
                    return
                await asyncio.sleep(1.0 / session.tps)
            else:
                await asyncio.sleep(0.05)

    loop_task = asyncio.create_task(game_loop())

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "action":
                session.set_action(int(msg["player"]), int(msg["action"]))
            elif mtype == "reset":
                session.reset(
                    layout=msg.get("layout"),
                    layout_data=msg.get("layout_data"),
                    procedural=msg.get("procedural"),
                    n_players=msg.get("n_players"),
                    config=msg.get("config"),
                    seed=msg.get("seed", 0),
                    order_recipes=msg.get("order_recipes"),
                    move_ms=msg.get("move_ms"),
                )
                await ws.send_json({"type": "state", "state": session.state()})
            elif mtype == "set_speed":
                session.tps = float(msg.get("tps", 7))
            elif mtype == "add_bot":
                session.add_bot(int(msg["player"]), msg.get("kind", "greedy"), msg.get("role", "any"))
            elif mtype == "remove_bot":
                session.remove_bot(int(msg["player"]))
            elif mtype == "step":
                session.tick()
                await ws.send_json({"type": "state", "state": session.state()})
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        loop_task.cancel()


# Static assets (js, css). Mounted last so it doesn't shadow API routes.
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# =============================== HUMAN-STUDY ASSISTANT BRIDGE ===============================
# POST /api/assist — stateless supervisor turn for the human study (web/study.html).
# The BROWSER holds the supervisor's recurrent memory (prev_json) and posts it back each
# turn together with the last frames (canvas JPEG dataURLs), the public communication
# history, and anything the human just said (via speech-to-text). The Gemini key stays in
# the server environment; the page never sees it.
#
# SINGLE-SOURCE PROMPT: the system prompt is the EXACT e2e benchmark supervisor prompt
# (cooksim.interact.protocol.e2e_supervisor_prompt — the same function tools/e2e_rollout.py
# uses), built on a real UserState for the study scenario, with exactly TWO whitelisted
# human-study deltas applied below. Everything else (game rules, taxonomy, structured
# cook_progress schema, speaking discipline, flag instructions) is byte-identical.

# ---- HUMAN-STUDY DELTA 1 (whitelisted): a short REAL-PARTICIPANT note appended AFTER the
# benchmark persona block (the persona itself is creative_uninterruptible, benchmark-native) ----
_STUDY_HUMAN_NOTE = (
    "\n(This cook is played by a REAL HUMAN participant: keyboard controls, hears you via "
    "voice, replies by voice or typing — expect human reaction times and occasional idling.)")
# ---- HUMAN-STUDY DELTA 2 (whitelisted): rendered per-tick frames -> browser screenshots ----
_STUDY_OBS_BLOCK = (
    "(1) the last few SCREENSHOTS of the cook's own game view, oldest -> newest — your "
    "ONLY window into what the cook is doing; read their actions from the pixels (where "
    "they walk, what they pick up, what lands on the plate). The cook is the blue chef; "
    "they may hold a white plate (food visibly stacks on it once added); pots/pans/boards "
    "show their contents; done food darkens; a neglected hot station shows SMOKE (a grey "
    "plume — your warning window) then FIRE.")

# ---- HUMAN-STUDY DELTA 3 (whitelisted, added 2026-07-31 after a field failure) ----
# A participant scooped RAW meat straight from the crate onto the plate and the assistant
# answered "You're doing great. Just scoop the cheese onto the plate and serve." It had
# read the plate correctly — it simply never compared each component's STATE against the
# order. Order errors were already caught (the plate sequence is checked against the
# recipe list); ingredient state was not. This delta makes that check explicit. It is a
# study-app addition only: the benchmark prompt stays byte-identical so experiment results
# remain comparable.
_STUDY_STATE_CHECK = """

== CHECK THE PLATE BEFORE YOU APPROVE ANYTHING ==
Every component of the order has a required STATE as well as a position — "meat (cooked)"
means a RAW patty in that slot is WRONG even though the ingredient is right. Raw food can
be scooped straight from a crate onto a plate, so this happens easily and it is invisible
unless you look.
Before you tell the cook that things look fine, or to add the next component, or to serve,
check the plate you can see against the order, component by component:
  - is each ingredient the one the order asks for, in the right position, AND in the
    required state (cooked where the order says cooked, chopped where it says chopped)?
  - a plate carrying a component in the wrong state CANNOT be served: serving it is
    rejected and the episode is lost.
If a component is in the wrong state, SAY SO plainly and say what to do about it, instead
of moving on to the next step. Never answer "looks good", "you're on the right track" or
"go ahead and serve" without having made this check."""

_STUDY_SUP_CACHE = {}


def _study_layout(e):
    """The benchmark episode layout = grid.json layout + the SAME two injections
    exp_interact.build_episode(inject=True) applies: a lettuce source at (0,2) and the
    TRASH at (5,2). Without them the participant cannot dump a plate (no trash tile)."""
    import copy
    lay = copy.deepcopy(e["layout"])
    grid = lay["grid"] if isinstance(lay, dict) else lay
    grid[2][0] = "lettuce_source"      # inject_source(game, "lettuce@0,2")  [x,y] -> grid[y][x]
    grid[2][5] = "trash"               # inject_source(game, "trash@5,2")
    return lay


def _study_grid_entry():
    """The study scenario's grid.json entry (shared by /api/study_scenario and the prompt)."""
    import json as _json
    import re as _re
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    grid_file = os.environ.get("COOKSIM_GRID") or os.path.join(root, "tmp", "grid.json")
    with open(grid_file) as f:
        data = _json.load(f)
    sl = lambda s: _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")  # noqa: E731
    return next(x for x in data["grid"] if sl(x["cell"]) == _STUDY_CELL), grid_file


def _study_supervisor_prompt():
    """e2e benchmark supervisor prompt for the study scenario + the two deltas above."""
    if "prompt" in _STUDY_SUP_CACHE:
        return _STUDY_SUP_CACHE["prompt"]
    from ..core.plan_ir import segment_plan
    from ..core.recipes import Recipe
    from ..interact.protocol import OBS_FRAMES, e2e_supervisor_prompt, profile_block_text
    from ..interact.state import RECOVERY_PERSONAS, UserState
    e, _gf = _study_grid_entry()
    rec = Recipe(id=e["recipe_id"], name=e["recipe_name"],
                 contents=tuple((n, s) for n, s in e["contents"]), base_reward=20,
                 ordered=e["ordered"])          # mirrors tools/exp_interact.recipe_of
    clean = list(segment_plan(rec))
    state = UserState(recipe=rec, clean_ir=clean, err_ir=list(clean), records=[],
                      profile=RECOVERY_PERSONAS["creative_uninterruptible"])
    s = e2e_supervisor_prompt(state, hazard_ticks=(240, 60))
    pb = profile_block_text(state.profile)
    assert pb in s, "profile block not found — protocol.py changed shape"
    s = s.replace(pb, pb + _STUDY_HUMAN_NOTE)   # DELTA 1: append the real-participant note
    assert OBS_FRAMES in s, "frames observation block not found — protocol.py changed shape"
    s = s.replace(OBS_FRAMES, _STUDY_OBS_BLOCK)  # DELTA 2
    s = s + _STUDY_STATE_CHECK                   # DELTA 3: verify component STATE
    _STUDY_SUP_CACHE["prompt"] = s
    return s


def _gemini_text(r):
    """All text parts of the first candidate, concatenated.

    ``r.text`` warns "non-text parts in the response: ['thought_signature'], returning
    concatenated text" and has been observed to TRUNCATE the tail of the reply (the
    recurring 502 'unparseable JSON' with a missing closing brace).  Joining the text
    parts ourselves keeps every byte the model produced.
    """
    try:
        parts = r.candidates[0].content.parts or []
    except Exception:  # noqa: BLE001
        parts = []
    txt = "".join(getattr(p, "text", None) or "" for p in parts)
    if txt:
        return txt
    try:
        return r.text or ""
    except Exception:  # noqa: BLE001
        return ""


def _repair_truncated_json(s):
    """Best-effort repair of a TRUNCATED JSON object.

    Scans the prefix with a tiny string-aware bracket stack, then: closes an
    unterminated string literal, drops a dangling ',' (or completes a dangling ':'
    with null), and appends the missing ']'/'}' closers in reverse order.
    """
    stack = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    out = s
    if in_str:
        out += '"'
    out = out.rstrip()
    if out.endswith(","):
        out = out[:-1].rstrip()
    if out.endswith(":"):
        out += " null"
    out += "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return out


def _parse_reply_json(t):
    """Parse the model reply text into a dict.

    Returns ``(obj, repaired)`` where ``repaired`` is True when the truncation
    repair was needed.  Raises on genuinely unparseable input.
    """
    import json as _json
    i = t.find("{")
    if i < 0:
        return {}, False
    frag = t[i:]
    dec = _json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(frag)
        return (obj if isinstance(obj, dict) else {}), False
    except Exception:  # noqa: BLE001
        obj, _ = dec.raw_decode(_repair_truncated_json(frag))
        return (obj if isinstance(obj, dict) else {}), True


def _vertex_mode() -> bool:
    """Salesforce-internal Vertex credentials (2026-07-26 default paradigm): ADC via
    gcloud + GOOGLE_GENAI_USE_VERTEXAI=true. No API key anywhere. Live voice models are
    NOT available on the internal project - live mode falls back to pro."""
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"


def _paradigm_model(name: str) -> str:
    """Map historical model ids onto whatever the active credential paradigm serves.
    On the Salesforce project (location=global) gemini-3.1-pro-preview works as-is;
    only the flash-tier id differs."""
    if not _vertex_mode():
        return name
    n = (name or "").strip()
    if n == "gemini-flash-latest" or ("flash" in n and "live" not in n and "3-flash" not in n):
        return "gemini-3-flash-preview"
    return n or "gemini-3.1-pro-preview"


def _make_genai_client(_genai, _types, timeout_ms: int):
    if _vertex_mode():
        # global location: 3.x previews 404 in regional endpoints on this project
        return _genai.Client(vertexai=True,
                             project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                                    ""),   # set GOOGLE_CLOUD_PROJECT in the environment,
                             location="global",
                             http_options=_types.HttpOptions(timeout=timeout_ms))
    key, _ksrc, kerr = _resolve_gemini_key()
    if not key:
        raise RuntimeError(kerr or "no Gemini credentials")
    return _genai.Client(api_key=key, http_options=_types.HttpOptions(timeout=timeout_ms))


def _resolve_gemini_key():
    """Resolve the Gemini API key for the study assistant.

    Order: env GEMINI_API_KEY -> env GEMINI_KEY_FILE (path to a file) -> ./gemini_key
    (current working directory) -> ~/.gemini_key.  The first source that yields a
    non-empty value wins; its value is validated (39 ASCII chars starting 'AIzaSy') so a
    homoglyph/copy-paste-corrupted key fails fast with a precise message instead of an
    opaque 502 from Google.  Returns (key, source, None) or (None, None, error_message).
    """
    sources = [("env GEMINI_API_KEY", lambda: os.environ.get("GEMINI_API_KEY"))]
    kf = os.environ.get("GEMINI_KEY_FILE")
    if kf:
        sources.append((f"env GEMINI_KEY_FILE ({kf})",
                        lambda: open(os.path.expanduser(kf)).read()))
    sources.append(("./gemini_key", lambda: open("gemini_key").read()))
    sources.append(("~/.gemini_key", lambda: open(os.path.expanduser("~/.gemini_key")).read()))
    for name, get in sources:
        try:
            raw = get()
        except Exception:
            continue
        if not raw or not raw.strip():
            continue
        key = raw.strip()
        if not (key.isascii() and key.startswith("AIzaSy") and len(key) == 39):
            return None, None, (
                f"key found in {name} but looks malformed (len={len(key)}, "
                f"ascii={key.isascii()}) — check for homoglyph/copy-paste corruption; "
                "expected 39 ASCII chars starting with 'AIzaSy'")
        return key, name, None
    return None, None, ("no Gemini key: set GEMINI_API_KEY, or GEMINI_KEY_FILE=<path>, or put "
                        "a 'gemini_key' file next to where the server was started, or "
                        "~/.gemini_key")


@app.post("/api/assist")
async def assist(payload: dict):
    import base64 as _b64
    import json as _json
    try:
        from google import genai as _genai
        from google.genai import types as _types
    except Exception:
        return JSONResponse({"ok": False, "error": "google-genai not installed"}, status_code=501)
    if not _vertex_mode():
        key, _ksrc, kerr = _resolve_gemini_key()
        if not key:
            return JSONResponse({"ok": False, "error": kerr}, status_code=501)
    try:
        sys_prompt = _study_supervisor_prompt()
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        print("[assist] cannot build the study supervisor prompt:", e, flush=True)
        print("".join(_tb.format_exception(type(e), e, e.__traceback__))[-1500:], flush=True)
        return JSONResponse({"ok": False,
                             "error": f"study prompt unavailable: {type(e).__name__}: {e}"},
                            status_code=500)
    frames = payload.get("frames") or []
    # Ground-truth text state map — sent by the fast/DISPLAY channel only (the flash
    # model misreads the 420px frames). The graded channel never sends this field, so
    # its prompt stays byte-identical to the benchmark condition.
    state_text = payload.get("state_text") or ""
    state_block = ("== GROUND-TRUTH KITCHEN STATE (text, straight from the game engine — "
                   "when the frames and this text disagree, TRUST THIS TEXT) ==\n"
                   + str(state_text)[:4000] + "\n\n") if state_text else ""
    # Per-request BREVITY NUDGE, fast/display channel only (payload flag): appended to
    # the USER content — the shared benchmark SYSTEM prompt is untouched, and the graded
    # channel never sends the flag, so its request stays byte-identical to the benchmark.
    brevity_block = ("\n\n(If your previous message still applies and nothing material "
                     "changed, reply with empty text. Never restate advice you already "
                     "gave in other words.)") if payload.get("brevity_note") else ""
    parts = [sys_prompt,
             "CURRENT ORDER: " + _json.dumps(payload.get("order") or "(unknown)") + "\n\n"
             + state_block +
             "== YOUR PREVIOUS OUTPUT ==\n" + (payload.get("prev_json") or "(first turn — "
             "write the plan you expect the cook to follow)") + "\n\n"
             "== COMMUNICATION HISTORY ==\n" + (payload.get("comm") or "(nothing yet)")
             + ("\n\n== THE COOK JUST SAID ==\n" + payload["said"] if payload.get("said") else "")
             + f"\n\nBelow are the last {len(frames)} frames (oldest first). Output your JSON."
             + brevity_block]
    # EMPTY-PART GUARD: a canvas captured before it is ready yields a degenerate dataURL
    # ("data:," or an empty base64 payload) that decodes to ZERO bytes. Appended as an
    # image Part it made Vertex reject the WHOLE request with
    # 400 INVALID_ARGUMENT "empty inlineData parameter" — the field failure participants
    # saw as "[assist] ERROR ClientError: 400". Frames are validated client-side too; this
    # is the server's own line of defence (an old page in a browser cache still reaches it).
    dropped = 0
    for f in frames:
        try:
            raw = _b64.b64decode(str(f).split(",", 1)[1])
        except Exception:
            raw = b""
        if len(raw) < 256:          # smaller than any real JPEG frame
            dropped += 1
            continue
        parts.append(_types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
    if dropped:
        print(f"[assist] dropped {dropped} empty/degenerate frame(s)", flush=True)
    client = _make_genai_client(_genai, _types, 60_000)
    def _call():
        import re as _re
        # per-request model override (used by tools/assist_latency_bench.py; harmless for
        # the study page, which never sends the field)
        req_model = payload.get("model")
        model = (req_model if isinstance(req_model, str) and req_model.startswith("gemini-")
                 else os.environ.get("ASSIST_MODEL", "gemini-3.1-pro-preview"))
        model = _paradigm_model(model)   # vertex serves the 2.5 family only
        # THINKING BUDGET BY TIER: the fast/display channel must fit its 2s cadence —
        # flash with budget=256 measured p50 2.0-5.3s, with budget=0 p50 1.35s on
        # Vertex. The graded/pro channel keeps 256 (benchmark condition, quality first).
        budget = 0 if "flash" in model.lower() else 256

        # CONFIG LADDER (field bug: participants hit a bare
        # `400 INVALID_ARGUMENT — Request contains an invalid argument` that no amount of
        # frame validation removed). A study machine runs its own SDK version against
        # whichever backend its credentials select, and a config this server considers
        # ordinary — a thinking budget, a JSON response mime type — can simply be
        # unsupported there. Rather than guess which knob the participant's stack rejects,
        # the request DEGRADES: full config, then no thinking budget, then no JSON mime
        # type (the reply is parsed out of plain text, which _parse_reply_json already
        # handles), then a bare request with no frames. Whichever rung works is logged, so
        # the cause is answerable from the server log instead of a stack trace.
        def _cfg(thinking=True, json_mime=True):
            kw = dict(temperature=0.2, max_output_tokens=2000)
            if json_mime:
                kw["response_mime_type"] = "application/json"
            if thinking:
                kw["thinking_config"] = _types.ThinkingConfig(thinking_budget=budget)
            return _types.GenerateContentConfig(**kw)

        ladder = [("full", _cfg(), parts),
                  ("no-thinking-budget", _cfg(thinking=False), parts),
                  ("no-json-mime", _cfg(thinking=False, json_mime=False), parts),
                  ("text-only", _cfg(thinking=False, json_mime=False), parts[:1])]

        def _generate():
            first = None
            for name, c, pl in ladder:
                try:
                    out = client.models.generate_content(model=model, contents=pl, config=c)
                    if name != "full":
                        print(f"[assist] recovered on the '{name}' fallback "
                              f"(the full request was rejected by this backend)", flush=True)
                    return out
                except Exception as ex:  # noqa: BLE001
                    msg = str(ex)
                    if "INVALID_ARGUMENT" not in msg and "400" not in msg:
                        raise            # not a request-shape problem: surface it as-is
                    first = first or ex
                    print(f"[assist] config '{name}' rejected: {msg[:200]}", flush=True)
            raise first

        last_err = None
        for attempt in (1, 2):     # model retry is the LAST resort (truncation is repaired)
            r = _generate()
            t = _re.sub(r"^```(?:json)?|```$", "", _gemini_text(r).strip(), flags=_re.M).strip()
            try:
                obj, repaired = _parse_reply_json(t)
                if repaired:
                    print("[assist] repaired truncated JSON tail", flush=True)
                return obj
            except Exception as pe:
                # keep the raw text in the server log so 'why 502' is answerable
                print(f"[assist] JSON parse failure (attempt {attempt}): {pe}; "
                      f"raw={t[:400]!r}", flush=True)
                last_err = ValueError(f"model returned unparseable JSON: {str(pe)[:120]}")
        raise last_err
    try:
        out = await asyncio.get_event_loop().run_in_executor(None, _call)
    except Exception as e:
        # Log the full story server-side (participant machines only see the short string).
        import traceback as _tb
        _sizes = [len(getattr(p, "inline_data", None).data)
                  for p in parts if getattr(p, "inline_data", None)]
        print(f"[assist] ERROR {type(e).__name__}: {e}", flush=True)
        print(f"[assist]   request shape: model={os.environ.get('ASSIST_MODEL', 'default')} "
              f"vertex={_vertex_mode()} parts={len(parts)} images={len(_sizes)} "
              f"image_bytes={sum(_sizes)} text_chars={len(parts[0]) if parts else 0}",
              flush=True)
        print("".join(_tb.format_exception(type(e), e, e.__traceback__))[-2000:], flush=True)
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"},
                            status_code=502)
    return JSONResponse({"ok": True, "reply": out, "prev_json": _json.dumps(out)})


@app.get("/api/assist_health")
async def assist_health():
    """Tiny 1-token Gemini ping so the study page can show key/quota/network problems
    BEFORE the first real assistant turn (bare 502s were undiagnosable for participants)."""
    try:
        from google import genai as _genai
        from google.genai import types as _types
    except Exception:
        return JSONResponse({"ok": False, "error": "google-genai not installed"}, status_code=501)
    if _vertex_mode():
        ksrc = "vertex:" + os.environ.get("GOOGLE_CLOUD_PROJECT", "?")
    else:
        key, ksrc, kerr = _resolve_gemini_key()
        if not key:
            return JSONResponse({"ok": False, "error": kerr}, status_code=501)
    model = _paradigm_model(os.environ.get("ASSIST_MODEL", "gemini-3.1-pro-preview"))

    def _ping():
        client = _make_genai_client(_genai, _types, 25_000)
        client.models.generate_content(
            model=model, contents=["ping"],
            config=_types.GenerateContentConfig(max_output_tokens=1))

    try:
        await asyncio.get_event_loop().run_in_executor(None, _ping)
    except Exception as e:
        import traceback as _tb
        print(f"[assist_health] ERROR {type(e).__name__}: {e}", flush=True)
        print("".join(_tb.format_exception(type(e), e, e.__traceback__))[-2000:], flush=True)
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"},
                            status_code=502)
    return {"ok": True, "model": model, "key_source": ksrc, "live_available": True}


# ------------------------------- HUMAN-STUDY SCENARIO -------------------------------
# The study must be apple-to-apple with the simulated-human benchmark: the
# `wrong_order__after_outcome` e2e case runs on generator cell `medium_nops_medium_map`
# with ONE cook and ONE ordered Cheeseburger recipe (bun raw -> meat cooked -> cheese raw).
# This endpoint hands the /study page everything it needs to reset the ws session onto the
# EXACT same kitchen: it reads the same grid file and mirrors the same layout/recipe/config
# construction as tools/gemini_stream_rgb.make_gen_game (used by tools/exp_interact.build_episode).
_STUDY_CELL = os.environ.get("STUDY_CELL", "medium_nops_medium_map")


@app.get("/api/study_scenario")
def api_study_scenario():
    try:
        e, grid_file = _study_grid_entry()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"error": f"cannot load study cell {_STUDY_CELL!r}: {exc}"},
            status_code=500)
    return {
        "cell": _STUDY_CELL,
        "layout_data": _study_layout(e),
        "recipe_ids": [e["recipe_id"]],
        "recipe": {"id": e["recipe_id"], "name": e["recipe_name"],
                   "contents": e["contents"], "ordered": e["ordered"]},
        "n_players": 1,
        "seed": 0,
        # exactly make_gen_game's GameConfig: ONE standing order that never expires and no
        # episode timer -> a clean sandbox-style "complete the recipe" task
        "config": {"horizon": 10**9, "max_orders": 1,
                   "order_base_time": 10**9, "order_time_per_difficulty": 0,
                   "pot_burn_time": 240, "pan_burn_time": 240, "oven_burn_time": 240,
                   "fire_time": 60},
    }


@app.get("/study")
def study():
    return FileResponse(os.path.join(WEB_DIR, "study.html"))


# =============================== LIVE-MODE VOICE PROXY ===============================
# WS /ws/live — dual-channel live mode for the human study (Issue G):
#   voice/perception channel: browser mic PCM16@16k + 420px JPEG frames stream UP through
#   this proxy into a Gemini Live session; native audio (PCM24k) + input/output
#   transcriptions stream DOWN. The proxy (not ephemeral tokens) was chosen because it
#   works with the installed google-genai (2.10) unconditionally and the key never leaves
#   the server; the browser only ever sees this same-origin websocket.
#   The GRADED bookkeeping channel stays on POST /api/assist (benchmark-identical prompt).
_LIVE_MODEL = os.environ.get("LIVE_MODEL",
    "gemini-live-2.5-flash" if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    else "models/gemini-3.1-flash-live-preview")


# The LIVE voice channel now runs the SAME supervisor prompt as the graded channel
# (_study_supervisor_prompt -> the benchmark e2e prompt), instead of the four-sentence
# brief it used to get. That brief was why live mode fell apart in the field: with no game
# rules, no plating-order discipline, no error taxonomy and no ground-truth state, it
# guessed plate contents from 420px video and contradicted itself turn to turn ("plate the
# cheese" / "no, place the bun" / "you already cooked the meat"). Only the OUTPUT CONTRACT
# differs — a voice model speaks, it cannot emit the benchmark's JSON — so the shared
# prompt is followed by this adapter, which overrides the reply format and nothing else.
_VOICE_ADAPTER = """

=================== VOICE CHANNEL — HOW THIS TURN DIFFERS ===================
Everything above about the game, the recipe, plating order, hazards, what counts as an
error and when to keep quiet applies UNCHANGED. Only these points differ, and they
override the corresponding instructions above:

1. YOU SPEAK. Do NOT produce JSON, field names, or any structured output — you are talking
   out loud to the cook. Say ONE short, natural sentence at a time, the way a colleague
   would. When you need to report an error, just say plainly what is wrong and what to do.
2. MEMORY. You do not receive a "previous output" block or a written communication
   history: you remember this conversation yourself. Stay consistent with what you have
   already said — if you correct yourself, say so explicitly rather than silently
   contradicting an earlier statement.
3. WHAT YOU SEE. Instead of a fixed set of screenshots you receive a continuous low-res
   video of the cook's screen, PLUS [KITCHEN STATE] messages containing the exact contents
   of the cook's hands, of every plate (bottom to top) and of every station, straight from
   the game engine. That text is GROUND TRUTH and outranks the video, which is easy to
   misread. Before saying what is on the plate or what has been cooked, read the most
   recent [KITCHEN STATE]. Never contradict it.
4. NEXT STEP QUESTIONS. If the cook asks what to do next, answer from the kitchen state
   and the required plating order: name ONE concrete action that is possible RIGHT NOW
   (you cannot plate an ingredient that has not been prepared, and the cook cannot scoop
   while its hands are full).
5. ALWAYS SPEAK ENGLISH, whatever language you think you hear.
6. Time passes while you talk, so keep it short. Silence is fine when nothing is wrong.
"""


def _study_voice_brief():
    """System instruction for the LIVE voice channel: the benchmark supervisor prompt
    (single source, identical to the graded /api/assist channel) + the voice adapter."""
    return _study_supervisor_prompt() + _VOICE_ADAPTER


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    import base64 as _b64
    await ws.accept()
    if not _vertex_mode():
        key, _ksrc, kerr = _resolve_gemini_key()
        if not key:
            await ws.send_json({"type": "error", "error": kerr})
            await ws.close()
            return
    try:
        from google import genai as _genai
        from google.genai import types as _types
    except Exception:
        await ws.send_json({"type": "error", "error": "google-genai not installed"})
        await ws.close()
        return
    try:
        brief = _study_voice_brief()
    except Exception as e:  # noqa: BLE001
        await ws.send_json({"type": "error", "error": f"voice brief unavailable: {e}"})
        await ws.close()
        return
    client = _make_genai_client(_genai, _types, 120_000)
    cfg = _types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription={}, output_audio_transcription={},
        system_instruction=_types.Content(parts=[_types.Part(text=brief)]))
    try:
        async with client.aio.live.connect(model=_LIVE_MODEL, config=cfg) as session:
            await ws.send_json({"type": "ready", "model": _LIVE_MODEL})

            _live_debug = os.environ.get("LIVE_DEBUG") == "1"

            async def pump_up():        # browser -> Live
                while True:
                    msg = await ws.receive_json()
                    t = msg.get("type") if isinstance(msg, dict) else None
                    # DECODE separately from SEND: one malformed client frame must NEVER
                    # kill the whole live session (field bug: a dataURL without a comma
                    # raised IndexError and tore the session down mid-episode). A failed
                    # SEND still propagates — that genuinely means the session is gone.
                    if t == "audio":    # PCM16 @16k, base64
                        try:
                            data = _b64.b64decode(msg["data"])
                        except Exception:  # noqa: BLE001
                            continue
                        await session.send_realtime_input(audio=_types.Blob(
                            mime_type="audio/pcm;rate=16000", data=data))
                    elif t == "frame":  # canvas JPEG dataURL (420px)
                        try:
                            raw = _b64.b64decode(msg["data"].split(",", 1)[1])
                        except Exception:  # noqa: BLE001
                            continue
                        # An EMPTY blob is not merely useless: send_realtime_input raises
                        # on it, and a raised SEND tears the whole live session down
                        # (that is the intended behaviour for real transport failures).
                        # Degenerate frames must therefore never reach the wire.
                        if len(raw) < 256:
                            continue
                        # NOTE: video=, not media= — the live backend rejects the
                        # deprecated realtime_input.media_chunks path (APIError 1007)
                        await session.send_realtime_input(video=_types.Blob(
                            mime_type="image/jpeg", data=raw))
                    elif t == "state":  # ground-truth kitchen state (context, not a turn)
                        txt = str(msg.get("text") or "")[:4000]
                        if txt:
                            # turn_complete=False: this is CONTEXT, not something the cook
                            # asked. It updates what the model knows without provoking a
                            # reply — the live channel used to have no state at all and
                            # guessed plate contents from 420px frames.
                            await session.send_client_content(
                                turns=_types.Content(
                                    role="user",
                                    parts=[_types.Part(
                                        text="[KITCHEN STATE — ground truth from the game "
                                             "engine, more reliable than anything you think "
                                             "you see in the video. Do not reply to this.]\n"
                                             + txt)]),
                                turn_complete=False)
                    elif t == "text":   # typed message
                        txt = str(msg.get("text") or "")
                        if txt:
                            await session.send_client_content(
                                turns=_types.Content(role="user",
                                                     parts=[_types.Part(text=txt)]),
                                turn_complete=True)
                    # unknown types are ignored (forward compatibility)

            async def pump_down():      # Live -> browser
                # NOTE: session.receive() is an async iterator that ENDS at each turn
                # boundary — iterate it in a loop or the proxy goes deaf after the first
                # exchange (the exact "audio stops after one reply" field bug).
                while True:
                    await _pump_one_turn()

            # ECHO second line (server): the proxy knows when it streamed audio DOWN.
            # An input transcription arriving within LIVE_ECHO_TAG_MS of downstream
            # audio is tagged echo_risk so the client can suppress with confidence
            # (client playout lags this timestamp, so the window is generous).
            import time as _time
            _echo_tag_s = float(os.environ.get("LIVE_ECHO_TAG_MS", "7000")) / 1000.0
            _last_audio_down = [0.0]

            def _echo_risk():
                return (_time.monotonic() - _last_audio_down[0]) < _echo_tag_s

            async def _pump_one_turn():
                async for m in session.receive():
                    if m.data:          # native audio, PCM 24k
                        _last_audio_down[0] = _time.monotonic()
                        await ws.send_json({"type": "audio",
                                            "data": _b64.b64encode(m.data).decode()})
                    sc = m.server_content
                    if sc is None:
                        continue
                    if _live_debug:
                        bits = []
                        for f in ("input_transcription", "output_transcription",
                                  "interim_input_transcription"):
                            v = getattr(sc, f, None)
                            if v is not None:
                                bits.append(f"{f}={v.text!r}/fin={v.finished}")
                        for f in ("turn_complete", "interrupted", "generation_complete",
                                  "waiting_for_input"):
                            if getattr(sc, f, None):
                                bits.append(f)
                        if bits:
                            print("[live-debug]", " | ".join(bits), flush=True)
                    it = sc.input_transcription
                    if it is not None and it.text:
                        await ws.send_json({"type": "you", "text": it.text,
                                            "echo_risk": _echo_risk()})
                    # the flash-live-preview model streams the USER's words primarily as
                    # interim_input_transcription (low-latency, cumulative hypothesis) —
                    # forwarding only the final input_transcription lost user speech
                    iit = getattr(sc, "interim_input_transcription", None)
                    if iit is not None and iit.text:
                        await ws.send_json({"type": "you_interim", "text": iit.text,
                                            "echo_risk": _echo_risk()})
                    ot = sc.output_transcription
                    if ot is not None and ot.text:
                        await ws.send_json({"type": "ai", "text": ot.text})
                    if sc.interrupted:
                        await ws.send_json({"type": "interrupted"})
                    if sc.turn_complete:
                        await ws.send_json({"type": "turn_complete"})

            up = asyncio.ensure_future(pump_up())
            down = asyncio.ensure_future(pump_down())
            done, pending = await asyncio.wait([up, down],
                                               return_when=asyncio.FIRST_EXCEPTION)
            for p in pending:
                p.cancel()
            for d in done:
                exc = d.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        print(f"[live] ERROR {type(e).__name__}: {e}", flush=True)
        print("".join(_tb.format_exception(type(e), e, e.__traceback__))[-1500:], flush=True)
        try:
            await ws.send_json({"type": "error",
                                "error": f"{type(e).__name__}: {str(e)[:200]}"})
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# =============================== END-OF-GAME EVALUATION ===============================
# POST /api/evaluate — scores a finished /study episode and returns
#   {"ok": true, "evaluation": {"schema", "quantitative", "judge", "computed_at"}}.
#
# QUANTITATIVE: pure server-side extraction, NO LLM — the SPIRIT of tools/eval_assistant.py
# adapted to the APP episode schema, which differs from rollout reports: conversation
# entries are {who, text, channel, t_ms}, timing lives in freeze_intervals /
# serve_success_at, and NO ground-truth error injection exists — so every
# detection-verdict criterion (detection validity/latency, false alarms, recovery
# verdicts) is reported honestly under `not_applicable` instead of being faked.
#
# QUALITATIVE: ONE Gemini call — the 6-rubric judge from tools/eval_judge.py adapted to
# no-ground-truth mode (grounding graded as internal consistency with observable events).
# payload.judge selects the mode: "real" (default) | "mock" (canned scores, used by the
# QA suite so browser tests never wait 15-45s on a judge call) | "skip". Judge failures
# NEVER fail the endpoint: the response still carries the quantitative block plus
# judge = {"error": ...} so the page can render a "judge unavailable" note.

_STUDY_JUDGE_KEYS = ("grounding", "actionability", "timing_manners",
                     "user_respect", "language_economy", "communication")


def _study_token_jaccard(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _study_median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


# Fixed reference constants (worst, best) — study-v1, the reference-anchored
# normalization style of tools/eval_assistant.py (its REF table) re-anchored to the
# app's wall-clock units. N(x) = clip((x - worst) / (best - worst), 0, 1); 1 is better.
_STUDY_REF = {
    "time_to_success_s": (900.0, 120.0),    # active cooking seconds (freeze excluded)
    "talk_words":        (900.0, 80.0),     # same anchors as eval_assistant REF
    "messages":          (60.0, 5.0),       # displayed assistant messages
    "responsiveness_ms": (20000.0, 2000.0),  # human msg -> next displayed assistant msg
}
_STUDY_NAG_BUDGET_PER_10MIN = 6.0            # nags tolerated per 10 min active (floor 3)
# Documented NEUTRAL weight vector. The participant is a real human, not a benchmark
# persona, so eval_assistant's per-persona weight table does not apply; these weights
# mirror its bucket structure (A task / B efficiency / C2 talk / C1 nag) plus a
# responsiveness bucket, renormalized when a bucket is undefined for the episode.
_STUDY_WEIGHTS = {"task": 0.30, "time": 0.20, "economy": 0.20,
                  "nag": 0.15, "responsiveness": 0.15}


def _study_quant_eval(ep):
    """Quantitative criteria + composite 0-100 score for one /study episode dict."""
    conv = [c for c in (ep.get("conversation") or []) if isinstance(c, dict)]

    def t_of(c):
        t = c.get("t_ms")
        return t if isinstance(t, (int, float)) else None

    human = [c for c in conv if c.get("who") == "human" and (c.get("text") or "").strip()]
    # only messages the participant actually saw/heard count as assistant talk
    shown = [c for c in conv if c.get("who") == "assistant"
             and (c.get("text") or "").strip() and c.get("displayed") is not False]

    success = bool(ep.get("serve_success_at"))
    failed = len(ep.get("serve_rejects") or [])
    frozen_ms = sum(max(0, (iv.get("end_ms") or 0) - (iv.get("start_ms") or 0))
                    for iv in (ep.get("freeze_intervals") or []) if isinstance(iv, dict))
    paused_ms = 0
    for iv in (ep.get("assistant_paused_intervals") or []):
        s, e = iv.get("start_t_ms"), iv.get("end_t_ms")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            paused_ms += max(0, e - s)

    # time to success: ACTIVE cooking time, speak-freeze excluded — the page's own
    # serve_success_at.elapsed_ms carries exactly that number
    tts = None
    if success:
        ss = ep["serve_success_at"]
        tts = ss.get("elapsed_ms") if isinstance(ss.get("elapsed_ms"), (int, float)) else None
        if tts is None:
            cs = (ep.get("cook_start_at") or {}).get("t_ms") or 0
            tts = max(0, (ss.get("t_ms") or 0) - cs - frozen_ms)

    talk_words = sum(len(c["text"].split()) for c in shown)

    # nag: near-duplicate (token-Jaccard >= 0.6) of one of the last 3 displayed messages
    # (same rule + threshold as tools/eval_assistant.py)
    nag = 0
    recent = []
    for c in shown:
        x = c["text"]
        if any(_study_token_jaccard(x, r) >= 0.6 for r in recent):
            nag += 1
        recent = (recent + [x])[-3:]

    # question responsiveness: median gap human message -> NEXT displayed assistant msg
    gaps = []
    for h in human:
        th = t_of(h)
        if th is None:
            continue
        nxt = min((t_of(c) for c in shown if t_of(c) is not None and t_of(c) > th),
                  default=None)
        if nxt is not None:
            gaps.append(nxt - th)
    resp = _study_median(gaps)

    balance = (len(human) / (len(human) + len(shown))) if (human or shown) else None

    # active play time (nag-budget denominator): success elapsed, else wall time from
    # cook_start to the last recorded event minus frozen time
    if tts is not None:
        active_ms = tts
    else:
        cs = (ep.get("cook_start_at") or {}).get("t_ms")
        ts_all = [t for t in (t_of(c) for c in conv) if t is not None]
        last = max(ts_all) if ts_all else None
        active_ms = max(0, last - (cs or 0) - frozen_ms) if last is not None else None

    # safety events: only if the hud samples actually carry fire/burn state — the app's
    # hud sampler records {t_ms, score, served, tick} only, so normally N/A (honest)
    hud = [h for h in (ep.get("hud") or []) if isinstance(h, dict)]
    safety_keys = {"fires", "fire", "burnt", "burned"}
    safety_observable = any(safety_keys & set(h.keys()) for h in hud[:200])

    crit = dict(
        task_success=success, deliveries=1 if success else 0, failed_deliveries=failed,
        time_to_success_ms=tts, active_ms=active_ms, frozen_ms=frozen_ms,
        assistant_paused_ms=paused_ms,
        talk_volume_words=talk_words, assistant_message_count=len(shown),
        nag_count=nag, question_responsiveness_ms=resp,
        human_message_count=len(human),
        interaction_balance=round(balance, 4) if balance is not None else None,
        freeze_count=len(ep.get("freeze_intervals") or []),
        stale_replies=ep.get("stale_replies"),
        flags_raised=len(ep.get("flags") or []),
    )
    if safety_observable:
        crit["safety_events"] = sum(
            1 for h in hud for k in safety_keys if h.get(k))

    def norm(key, x):
        if x is None:
            return None
        worst, best = _STUDY_REF[key]
        return max(0.0, min(1.0, (x - worst) / (best - worst)))

    buckets = {
        "task": 1.0 if success else 0.0,
        # no success = worst efficiency by definition (time-to-success undefined)
        "time": norm("time_to_success_s", tts / 1000.0) if tts is not None else 0.0,
        "economy": ((norm("talk_words", talk_words) or 0.0)
                    + (norm("messages", len(shown)) or 0.0)) / 2.0,
    }
    minutes = (active_ms or 0) / 60000.0
    budget = max(3.0, _STUDY_NAG_BUDGET_PER_10MIN * minutes / 10.0)
    buckets["nag"] = max(0.0, min(1.0, 1.0 - min(nag / budget, 1.0)))
    r_norm = norm("responsiveness_ms", resp) if resp is not None else None
    weights = dict(_STUDY_WEIGHTS)
    if r_norm is None:            # participant never asked anything: renormalize
        del weights["responsiveness"]
        tot = sum(weights.values())
        weights = {k: round(v / tot, 4) for k, v in weights.items()}
    else:
        buckets["responsiveness"] = r_norm
    score = round(100.0 * sum(weights[k] * buckets[k] for k in weights), 1)

    na_msg = ("requires injected-error episodes — the human study has no ground-truth "
              "error injection, so flag verdicts cannot be scored")
    not_applicable = {
        "detection_validity": na_msg,
        "detection_latency": na_msg,
        "false_alarms": na_msg,
        "recovery_and_persuasion_verdicts": na_msg,
    }
    if not safety_observable:
        not_applicable["safety_events"] = (
            "not observable: app hud samples carry only score/served/tick "
            "(no fire/burn state is recorded)")

    return {"schema": "study-quant-v1", "criteria": crit,
            "not_applicable": not_applicable,
            "normalized": {k: round(v, 4) for k, v in buckets.items()},
            "weights": weights, "score": score,
            "normalization": ("fixed-reference study-v1 (reference-anchored like "
                              "tools/eval_assistant.py, neutral non-persona weights)"),
            "reference_constants": dict(_STUDY_REF,
                                        nag_budget_per_10min=_STUDY_NAG_BUDGET_PER_10MIN)}


_STUDY_JUDGE_RUBRICS = """\
1. grounding — are the assistant's kitchen-state claims CONSISTENT with the observable \
events (serve rejections, the final serve outcome, what the cook reports, the summary \
stats)? NO ground-truth log exists: judge internal consistency, never invent hidden truth.
   1: repeated confident claims contradicted by later events.
   4: several claims unverifiable or contradicted; correct ones are generic.
   7: essentially consistent throughout; occasional imprecision.
   10: precise, verifiably consistent, never overclaims.
2. actionability — could a FIRST-TIME human participant execute the advice as spoken, \
immediately?
   1: vague or impossible instructions; the cook must guess.
   4: advice usable only after follow-up questions.
   7: directly executable with minor inference.
   10: exactly the right granularity, executable at once.
3. timing_manners — spoke when speaking had value; silent when silence had value; no \
needless interruptions mid-action without new events.
   1: constant interruptions at the worst moments, or total silence during the crisis.
   4: some useful timing, several needless interruptions or long absences.
   7: mostly speaks at natural breakpoints or genuinely urgent moments.
   10: every utterance lands where it helps; silence otherwise.
4. user_respect — respected the participant's autonomy and way of working (guided \
without taking over, never condescending, adapted to how this person actually played).
   1: micromanages or fights the participant's choices.
   4: generic one-size-fits-all assistance.
   7: visibly adapted granularity and frequency to the participant.
   10: guidance exactly matched to what this participant needed.
5. language_economy — value per word. BREVITY IS A VIRTUE: long, flowery, or repeated \
messages are penalized even when correct; short precise ones score high.
   1: walls of text, heavy repetition, tiny information content.
   4: correct but padded; noticeable repetition.
   7: compact messages; little dead weight.
   10: near-minimal wording for the value delivered.
6. communication — overall persuasion/communication quality: clear claims, evidence \
when challenged, graceful under pushback, coherent across turns.
   1: confusing or contradictory; caves instantly or bulldozes.
   4: repeats assertions louder instead of giving evidence.
   7: clear, mostly convincing, reasonable escalation.
   10: minimal rounds to justified agreement; concedes gracefully when wrong."""


def _study_eval_transcript(ep, max_chars=14000):
    """Timestamped transcript from the APP conversation schema ({who,text,channel,t_ms});
    suppressed (never-displayed) assistant records are excluded — the judge grades what
    the participant actually experienced. Key system events stay in as context."""
    lines = []
    for c in ep.get("conversation") or []:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        t = c.get("t_ms")
        ts = f"{t / 1000:7.1f}s" if isinstance(t, (int, float)) else "      ?s"
        who = c.get("who")
        ch = c.get("channel") or "?"
        if who == "assistant":
            if c.get("displayed") is False:
                continue
            lines.append(f"{ts} ASSISTANT[{ch}]: {text}")
        elif who == "human":
            lines.append(f"{ts} COOK[{ch}]: {text}")
        elif who == "system" and any(k in text for k in (
                "serve rejected", "order served", "preview over", "scenario started")):
            lines.append(f"{ts} (event: {text})")
    return "\n".join(lines)[-max_chars:] or "(empty transcript)"


def _study_judge(ep, quant, key):
    """ONE Gemini call: tools/eval_judge.py's third-party judge adapted to
    no-ground-truth mode. Raises on failure (the endpoint converts to judge.error)."""
    import json as _json
    import re as _re

    from google import genai as _genai
    from google.genai import types as _types
    model = _paradigm_model(os.environ.get("STUDY_JUDGE_MODEL", "gemini-3.1-pro-preview"))
    client = _make_genai_client(_genai, _types, 90_000)
    crit = quant.get("criteria") or {}
    stats = {k: crit.get(k) for k in (
        "task_success", "failed_deliveries", "time_to_success_ms", "talk_volume_words",
        "assistant_message_count", "nag_count", "question_responsiveness_ms",
        "human_message_count", "freeze_count")}
    sys_prompt = (
        "You are a strict third-party judge grading an AI kitchen ASSISTANT (the "
        "'ASSISTANT' speaker) that supervised a REAL HUMAN participant playing a cooking "
        "game. You grade the ASSISTANT only — never the cook. IMPORTANT: unlike the "
        "benchmark setting, NO ground-truth error-injection log exists for this episode; "
        "grade ONLY from the transcript and the summary statistics, and judge grounding "
        "as internal consistency with observable events (serve rejections, the final "
        "outcome, what the cook reports). Grade each rubric 1-10 using the anchored "
        "descriptors; intermediate integers interpolate. REMEMBER: BREVITY IS A VIRTUE — "
        "never reward length, hedging, or repetition. An assistant that says little but "
        "exactly the right thing at the right moment is a 10, not a 5. Each rationale is "
        "ONE sentence.")
    user = (
        f"== SUMMARY STATISTICS (computed from the episode log) ==\n"
        f"{_json.dumps(stats, indent=1)}\n\n"
        f"== TIMESTAMPED TRANSCRIPT (what the participant saw/heard) ==\n"
        f"{_study_eval_transcript(ep)}\n\n"
        f"== RUBRICS (score each 1-10) ==\n{_STUDY_JUDGE_RUBRICS}\n\n"
        'Reply with EXACTLY this JSON:\n'
        '{"scores": {"grounding": <1-10>, "actionability": <1-10>, '
        '"timing_manners": <1-10>, "user_respect": <1-10>, "language_economy": <1-10>, '
        '"communication": <1-10>}, "rationales": {"grounding": "<one sentence>", '
        '"actionability": "<one sentence>", "timing_manners": "<one sentence>", '
        '"user_respect": "<one sentence>", "language_economy": "<one sentence>", '
        '"communication": "<one sentence>"}}')
    cfg = _types.GenerateContentConfig(
        temperature=0.2, response_mime_type="application/json", max_output_tokens=2000,
        thinking_config=_types.ThinkingConfig(thinking_budget=768))
    last = None
    for attempt in (1, 2):
        r = client.models.generate_content(model=model, contents=[sys_prompt, user],
                                           config=cfg)
        t = _re.sub(r"^```(?:json)?|```$", "", _gemini_text(r).strip(), flags=_re.M).strip()
        try:
            obj, repaired = _parse_reply_json(t)
            raw = obj.get("scores") or {}
            scores = {}
            for k in _STUDY_JUDGE_KEYS:
                v = raw.get(k)
                if not isinstance(v, (int, float)):
                    raise ValueError(f"judge missing/non-numeric rubric {k}: {raw}")
                scores[k] = int(max(1, min(10, round(v))))
            rat = obj.get("rationales") or {}
            rationales = {k: str(rat.get(k, ""))[:300] for k in _STUDY_JUDGE_KEYS}
            if repaired:
                print("[evaluate] repaired truncated judge JSON tail", flush=True)
            return {"schema": "study-judge-v1", "model": model,
                    "mode": "no_ground_truth", "scores": scores,
                    "rationales": rationales}
        except Exception as pe:  # noqa: BLE001
            print(f"[evaluate] judge parse failure (attempt {attempt}): {pe}; "
                  f"raw={t[:300]!r}", flush=True)
            last = pe
    raise last


@app.post("/api/evaluate")
async def api_evaluate(payload: dict):
    import datetime as _dt
    ep = payload.get("episode")
    if not isinstance(ep, dict):
        return JSONResponse(
            {"ok": False, "error": "payload must carry `episode` (the /study episode "
             "JSON object)"}, status_code=400)
    try:
        quant = _study_quant_eval(ep)
    except Exception as e:  # noqa: BLE001
        import traceback as _tb
        print(f"[evaluate] quantitative FAILED {type(e).__name__}: {e}", flush=True)
        print("".join(_tb.format_exception(type(e), e, e.__traceback__))[-1500:], flush=True)
        return JSONResponse(
            {"ok": False, "error": f"quantitative eval failed: {type(e).__name__}: "
             f"{str(e)[:200]}"}, status_code=500)
    mode = str(payload.get("judge") or os.environ.get("STUDY_JUDGE_MODE") or "real").lower()
    if mode == "mock":
        judge = {"schema": "study-judge-v1", "model": None, "mode": "mock", "mock": True,
                 "scores": {k: 7 for k in _STUDY_JUDGE_KEYS},
                 "rationales": {k: "mock rationale (judge mocked for tests)"
                                for k in _STUDY_JUDGE_KEYS}}
    elif mode in ("skip", "none", "off"):
        judge = {"error": "judge skipped by request"}
    else:
        key, _ksrc, kerr = _resolve_gemini_key()
        if not key:
            judge = {"error": kerr}
        else:
            try:
                judge = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: _study_judge(ep, quant, key))
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                print(f"[evaluate] judge ERROR {type(e).__name__}: {e}", flush=True)
                print("".join(_tb.format_exception(type(e), e, e.__traceback__))[-1500:],
                      flush=True)
                judge = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    return {"ok": True,
            "evaluation": {"schema": "study-eval-v1", "quantitative": quant,
                           "judge": judge,
                           "computed_at": _dt.datetime.now(_dt.timezone.utc)
                           .isoformat(timespec="seconds")}}


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the CookSim server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
