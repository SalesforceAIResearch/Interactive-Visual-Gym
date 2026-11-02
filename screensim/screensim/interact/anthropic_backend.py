"""Claude for ScreenSim, on the official Anthropic SDK — assistant and rubric-judge interfaces.

Ported from cook-bench-engine/tools/anthropic_backend.py (2026-10-12). Credentials: ANTHROPIC_API_KEY
and, for a user-scoped key (``sk-ant-usr-…``), the workspace it bills to in ANTHROPIC_WORKSPACE_ID.
Both fall back to ``~/.anthropic_key`` and ``~/.anthropic_workspace`` so a judge run needs no exports.
"""
from __future__ import annotations

import os
import time


def _from_file(name):
    p = os.path.expanduser(f"~/{name}")
    try:
        return open(p).read().strip() or None
    except OSError:
        return None


class AnthropicBackend:
    def __init__(self, model="claude-haiku-5-5", max_tokens=6000, effort="low"):
        import anthropic
        self.anthropic = anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or _from_file(".anthropic_key")
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID") or _from_file(".anthropic_workspace")
        if not key:
            raise RuntimeError("no ANTHROPIC_API_KEY and no ~/.anthropic_key")
        hdr = {"anthropic-workspace-id": ws} if ws else None
        self.client = anthropic.Anthropic(api_key=key, timeout=180.0, max_retries=2, default_headers=hdr)
        self.model = self.name = model
        self.max_tokens = max(int(max_tokens), 16000)   # the budget covers adaptive thinking as well as the reply
        self.effort = effort

    def call(self, system, user):
        """The judge needs text back; JSON parsing is the caller's job (same as the other judges)."""
        try:
            r = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                # the grading system prompt is identical for every round of an episode: cache it
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
                output_config={"effort": self.effort})
        except self.anthropic.BadRequestError as e:
            raise RuntimeError(f"anthropic call rejected (not retried): {e.message}") from e
        except self.anthropic.APIStatusError as e:
            raise RuntimeError(f"anthropic call failed after SDK retries: HTTP {e.status_code} {e.message}") from e
        except self.anthropic.APIConnectionError as e:
            raise RuntimeError(f"anthropic connection failed after SDK retries: {e}") from e
        if r.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in r.content if b.type == "text")


# Assistant requests use the same transport settings as CookSim.
import base64
import io
import time

from .backends import Backend, _parse_json


class AnthropicAssistantBackend(Backend):
    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 4000, effort: str = "low",
                 image_scale: float = 1.0, image_format: str = "png"):
        import anthropic
        self.anthropic = anthropic
        import os
        # a user-scoped key (sk-ant-usr-...) must name the workspace it bills to: ANTHROPIC_WORKSPACE_ID (2026-10-10)
        hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else None
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or _from_file(".anthropic_key"),
                                         timeout=180.0, max_retries=2, default_headers=hdr)   # 429 / 5xx / network retried by the SDK
        self.model, self.name = model, model
        # the budget covers adaptive thinking as well as the reply; the harness default (4000) is sized
        # for models that do not think, so keep room above it (only generated tokens are billed)
        self.max_tokens = max(int(max_tokens), 16000)
        self.effort = effort
        self.image_scale, self.image_format = image_scale, image_format

    def _img(self, f: bytes) -> dict:
        mime = "image/jpeg" if f[:3] == b"\xff\xd8\xff" else "image/png"   # the renderer emits JPEG
        if self.image_scale != 1.0 or self.image_format != "png":
            from PIL import Image
            im = Image.open(io.BytesIO(f)).convert("RGB")
            if self.image_scale != 1.0:
                im = im.resize((int(im.width * self.image_scale), int(im.height * self.image_scale)))
            buf = io.BytesIO()
            if self.image_format == "jpeg":
                im.save(buf, format="JPEG", quality=85); mime = "image/jpeg"
            else:
                im.save(buf, format="PNG")
            f = buf.getvalue()
        return {"type": "image", "source": {"type": "base64", "media_type": mime,
                                            "data": base64.standard_b64encode(f).decode()}}

    def call(self, system, user, images):
        content = [self._img(f) for f in (images or [])] + [{"type": "text", "text": user}]
        t0 = time.monotonic()
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # the system prompt is identical for every call of an episode: cache it
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_config={"effort": self.effort},
            )
        except self.anthropic.BadRequestError as e:
            raise RuntimeError(f"anthropic call rejected (not retried): {e.message}") from e
        except self.anthropic.APIStatusError as e:
            raise RuntimeError(f"anthropic call failed after SDK retries: HTTP {e.status_code} {e.message}") from e
        except self.anthropic.APIConnectionError as e:
            raise RuntimeError(f"anthropic connection failed after SDK retries: {e}") from e
        ms = int((time.monotonic() - t0) * 1000)
        u = r.usage
        prompt = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
            + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        meta = {"latency_ms": ms, "prompt_tokens": prompt, "completion_tokens": u.output_tokens,
                "thinking_tokens": None, "total_tokens": prompt + (u.output_tokens or 0),
                "stop_reason": r.stop_reason, "request_id": getattr(r, "_request_id", None)}
        if r.stop_reason == "refusal":
            d = getattr(r, "stop_details", None)
            meta["refusal"] = getattr(d, "category", None) if d else None
            return {}, meta
        text = "".join(b.text for b in r.content if b.type == "text")
        return _parse_json(text), meta
