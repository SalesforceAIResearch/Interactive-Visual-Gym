"""Single place that builds a Gemini client and sends a video for critique.
Reads the API key from ~/.gemini_key (or GEMINI_API_KEY). Never commits it."""
from __future__ import annotations

import json
import os
import re
import time

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _key():
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k.strip()
    for p in ("~/.gemini_key", "~/.config/gemini_key"):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return open(p).read().strip()
    raise RuntimeError("no Gemini key (set GEMINI_API_KEY or ~/.gemini_key)")


def make_client():
    return genai.Client(api_key=_key())


def _strip_json(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.S)
    return m.group(0) if m else t


def critique_video(client, video_path: str, prompt: str, schema_hint: str = "",
                   model: str = DEFAULT_MODEL, max_retries: int = 4) -> dict:
    """Send an mp4 + text prompt, expect a JSON object back."""
    with open(video_path, "rb") as f:
        data = f.read()
    parts = [types.Part.from_bytes(data=data, mime_type="video/mp4"),
             types.Part(text=prompt + ("\n\n" + schema_hint if schema_hint else ""))]
    cfg = types.GenerateContentConfig(
        temperature=0.3, max_output_tokens=4000,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=1024))
    last = None
    for attempt in range(max_retries):
        try:
            r = client.models.generate_content(
                model=model, contents=parts, config=cfg)
            txt = r.text or ""
            return json.loads(_strip_json(txt))
        except Exception as e:
            last = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"critique failed after {max_retries}: {last}")


def critique_frames(client, frame_paths: list, prompt: str, schema_hint: str = "",
                    model: str = DEFAULT_MODEL, max_retries: int = 4) -> dict:
    """Send a sequence of PNG frames + prompt, expect JSON back. Useful when a
    finer read than the encoded video is wanted."""
    parts = []
    for p in frame_paths:
        with open(p, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
    parts.append(types.Part(text=prompt + ("\n\n" + schema_hint if schema_hint else "")))
    cfg = types.GenerateContentConfig(
        temperature=0.3, max_output_tokens=4000,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=1024))
    last = None
    for attempt in range(max_retries):
        try:
            r = client.models.generate_content(model=model, contents=parts, config=cfg)
            return json.loads(_strip_json(r.text or ""))
        except Exception as e:
            last = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"critique failed after {max_retries}: {last}")
