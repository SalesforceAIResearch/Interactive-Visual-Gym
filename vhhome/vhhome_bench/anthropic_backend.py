"""Claude assistant transport; same Messages API settings as CookSim.

The existing VH-Home system prompt defines the reply schema. We parse the model
reply without repair, as for the other API assistants.
"""
import base64
import io
import time

from vhhome_bench.protocol import _extract_json as _parse_json


class AnthropicBackend:
    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 4000, effort: str = "low",
                 image_scale: float = 1.0, image_format: str = "png"):
        import anthropic
        self.anthropic = anthropic
        import os
        # a user-scoped key (sk-ant-usr-...) must name the workspace it bills to: ANTHROPIC_WORKSPACE_ID (2026-10-10)
        hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else None
        self.client = anthropic.Anthropic(timeout=180.0, max_retries=2, default_headers=hdr)   # 429 / 5xx / network retried by the SDK
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
