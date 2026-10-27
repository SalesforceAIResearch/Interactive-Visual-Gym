"""Claude for ScreenSim, on the official Anthropic SDK — currently used only as a rubric judge.

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
