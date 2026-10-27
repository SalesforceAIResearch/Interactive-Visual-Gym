"""Pluggable ASSISTANT backend for the e2e rollout (2026-08-19).

The e2e harness has always called the assistant through one funnel —
recovery_rollout.gemini_frames(client, sys, user, frames) — with a hard-wired Gemini
client. To benchmark OTHER models in the assistant seat, this module wraps any
python -m cooksim_bench.backends.controller_eval Backend (GeminiBackend, OpenAIBackend for local vLLM servers,
LiveBackend for the Gemini Live session API) in an object gemini_frames recognises by
its .sup_call attribute and routes to. The cook's own calls keep the plain Gemini
client, so the two seats are independent.

Spec strings (for --assist-backend/--assist-model on the e2e CLI):
    gemini  + model id                          stateless Gemini API
    openai  + model id (+ base_url, profile)    any /v1/chat/completions server
    live    + model id                          one Live session per episode
    live_native / realtime_native + model id    NATIVE streaming seat (no polling), see
                                                python -m cooksim_bench.backends.native_assistant
"""
import time


class AssistantBackend:
    def __init__(self, backend, retries=3):
        self.b = backend
        self.name = backend.name
        self.retries = retries
        self.meta_log = []            # per-call latency/tokens, archived into the report
        self._session_open = False

    cook_seat = False        # set by probe_rollout: in the COOK seat the "system" text is the
                             # per-turn world context and the "user" text is the static persona
                             # hint — the opposite of the assistant seat

    def sup_call(self, system, user, frames):
        # session backends (Live / Realtime) hold one session per episode and take the
        # system text ONCE, at open. In the cook seat the context changes every turn, so
        # the session is opened on the static persona hint and the context travels in
        # the per-turn message instead (found by the GPT-realtime audit: both session
        # cooks were blind after their first call).
        if hasattr(self.b, "reset_episode"):
            if self.cook_seat:
                system, user = user, system + "\n\n" + user
            if not self._session_open:
                self.b.reset_episode(system)
                self._session_open = True
        last = None
        for k in range(self.retries):
            try:
                out, meta = self.b.call(system, user, list(frames) or None)
                meta["t_wall"] = time.time()
                self.meta_log.append(meta)
                return out if isinstance(out, dict) else {}
            except Exception as e:                               # noqa: BLE001
                last = e
                time.sleep(2 + 2 * k)
        print(f"[assistant_backend] {self.name}: giving up after {self.retries} tries: "
              f"{last!r}", flush=True)
        return {}

    def stats(self):
        lat = sorted(m["latency_ms"] for m in self.meta_log if m.get("latency_ms"))
        return {"model": self.name, "calls": len(self.meta_log),
                "latency_ms_median": lat[len(lat) // 2] if lat else None,
                "prompt_tokens": sum(m.get("prompt_tokens") or 0 for m in self.meta_log),
                "completion_tokens": sum(m.get("completion_tokens") or 0 for m in self.meta_log),
                "thinking_tokens": sum(m.get("thinking_tokens") or 0 for m in self.meta_log)}


def make_assistant(kind, model, base_url="http://127.0.0.1:18100/v1",
                   profile="default", thinking=640, max_tokens=4000,
                   image_scale=1.0, image_format="png"):
    if kind in ("live_native", "realtime_native"):
        # not a wrapped Backend: a streaming session the harness feeds every tick and reads
        # utterances from (python -m cooksim_bench.backends.native_assistant; e2e_stepin_rollout native_step)
        from cooksim_bench.backends.native_assistant import make_native
        return make_native(kind, model)
    from cooksim_bench.backends.controller_eval import GeminiBackend, OpenAIBackend, LiveBackend, RealtimeBackend
    if kind == "gemini":
        b = GeminiBackend(model, thinking_budget=thinking)
    elif kind == "live":
        b = LiveBackend(model)
    elif kind == "realtime":
        b = RealtimeBackend(model)
    elif kind == "anthropic":
        from cooksim_bench.backends.anthropic_backend import AnthropicBackend
        b = AnthropicBackend(model, max_tokens=max_tokens, image_scale=image_scale, image_format=image_format)
    elif kind == "openai":
        b = OpenAIBackend(model, base_url, profile=profile, max_tokens=max_tokens,
                          image_scale=image_scale, image_format=image_format)
    else:
        raise SystemExit(f"unknown assistant backend {kind!r}")
    return AssistantBackend(b)
