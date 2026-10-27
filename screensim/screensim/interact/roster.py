"""The ScreenSim assistant roster: CookSim's balanced-20 (cook-bench-engine tools/roster.py @ 562aeab,
2026-09-19), reached exactly as CookSim reaches it -- same backend, model id, profile, token cap and
serve script -- minus the slots that cannot run here yet.

    python -m screensim.interact.roster          # print the roster with each entry's status

Not in the ScreenSim run (and why):
  2  claude-opus-5              no ANTHROPIC_API_KEY on this machine
  5  qwen3.8-max                no DASHSCOPE_API_KEY
  7  gemini-3.8-live__native    CookSim's native seat streams the cook's SPEECH (TTS) and paces the game at
  9  gpt-realtime-2.1__native   wall-clock speed; ScreenSim is turn-based with a text person. Open ruling
                                in CookSim too (reactive only).
  10 qwen3.5-omni-plus-realtime no key, and no adapter for Model Studio's realtime image protocol

Deviations from CookSim, both deliberate:
  - image_scale is 1.0 for every model. CookSim sends Molmo2 half-size kitchen frames (1042x396) because
    its latency is charged to the game clock and full size was slower; ScreenSim's frames are 412x892
    phone screens whose text is unreadable at half size, and ScreenSim does not charge latency.
  - Proact-VL and StreamingVLM run on ScreenSim copies of CookSim's shims (screensim/serving/shims/), which
    only replace the kitchen wording ("a video of a cooking game", the recipe line) with the phone's.
"""
from __future__ import annotations

import os

HOME = os.path.expanduser("~")
COOKSIM = os.environ.get("COOKSIM_ENGINE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "cooksim"))
SCREENSIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPENAI_URL = "https://api.openai.com/v1"


def _local(slot, tag, group, port, profile, serve, *, max_tokens=4000, shim=False, own_shim=False):
    return dict(slot=slot, tag=tag, group=group, kind="local", backend="openai", model=tag,
                base_url=f"http://127.0.0.1:{port}/v1", profile=profile, max_tokens=max_tokens,
                image_scale=1.0, port=port,
                serve=os.path.join(SCREENSIM if own_shim else COOKSIM, "serving", serve),
                # vLLM batches concurrent requests; a transformers shim runs one request at a time
                workers=1 if shim else 4)


ROSTER = [
    # ---- A · frontier API
    dict(slot="1", tag="gpt-6-astra", group="A", kind="api", backend="openai", model="gpt-6-astra",
         base_url=OPENAI_URL, profile="gpt5", max_tokens=4000, image_scale=1.0, workers=6),
    dict(slot="3", tag="gemini-3.1-pro-preview", group="A", kind="api", backend="gemini",
         model="gemini-3.1-pro-preview", workers=6),
    dict(slot="4", tag="gemini-3.8-flash", group="A", kind="api", backend="gemini", model="gemini-3.8-flash",
         workers=6),
    # ---- B · streaming protocol, polled (one session per episode)
    dict(slot="6", tag="gemini-3.8-live", group="B", kind="api", backend="live", model="gemini-3.8-live",
         workers=6),
    dict(slot="8", tag="gpt-realtime-2.1", group="B", kind="api", backend="realtime", model="gpt-realtime-2.1",
         workers=6),
    # ---- C · open-weight vision-language (local, one GPU each)
    _local("11", "qwen3.8-27b", "C", 18300, "qwen38-schema", "serve_qwen.sh"),
    _local("12a", "molmo2-8b", "C", 18400, "molmo2-schema", "serve_molmo2.sh"),
    _local("12b", "molmo2-o-7b", "C", 18600, "molmo2-schema", "serve_molmo2_o.sh"),
    _local("13", "internvideo3-8b", "C", 18700, "internvideo3-schema", "serve_internvideo3.sh",
           max_tokens=1024, shim=True),
    _local("14a", "glm-4.6v-flash", "C", 18500, "glm46v-flash-schema", "serve_glm46v_flash.sh"),
    _local("14b", "kimi-vl-thinking", "C", 18650, "kimi-vl-think-schema", "serve_kimi_vl.sh", max_tokens=8000),
    _local("15", "museglimmer", "C", 18200, "museglimmer-nothink-schema", "serve_museglimmer.sh"),
    # ---- D · duplex / proactive open (local)
    _local("16", "minicpmo", "D", 18100, "minicpmo-schema", "serve_minicpmo.sh"),
    _local("17", "qwen3-omni-30b-a3b", "D", 18800, "qwen3-omni-schema", "serve_qwen3_omni.sh"),
    _local("18", "mmduet2", "D", 18900, "default", "serve_mmduet2.sh", shim=True),
    _local("19", "proact-vl", "D", 18950, "default", "serve_proact_vl.sh", shim=True, own_shim=True),
    _local("20", "streamingvlm", "D", 19000, "default", "serve_streamingvlm.sh", shim=True, own_shim=True),
]
TAGS = [r["tag"] for r in ROSTER]


def by_tag(tag):
    for r in ROSTER:
        if r["tag"] == tag:
            return r
    raise KeyError(tag)


def make_backend(tag, base_url=None, entry=None):
    """A fresh assistant backend for ONE episode (session backends hold one session per episode)."""
    from . import backends as B
    e = entry if entry is not None else by_tag(tag)
    if e["backend"] == "gemini":
        b = B.GeminiBackend(e["model"])
    elif e["backend"] == "live":
        b = B.LiveBackend(e["model"])
    elif e["backend"] == "realtime":
        b = B.RealtimeBackend(e["model"])
    else:
        url = base_url or os.environ.get("SCREENSIM_ASSIST_BASE_URL") or e["base_url"]
        b = B.OpenAIBackend(e["model"], url, profile=e["profile"], max_tokens=e["max_tokens"],
                            image_scale=e.get("image_scale", 1.0))
    return b


if __name__ == "__main__":
    for r in ROSTER:
        where = f"local :{r['port']} {os.path.basename(r['serve'])}" if r["kind"] == "local" else r["backend"]
        print(f"{r['slot']:>4} {r['group']} {r['tag']:26s} {where}")
