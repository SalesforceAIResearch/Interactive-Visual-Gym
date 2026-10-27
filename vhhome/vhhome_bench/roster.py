"""The assistant roster for the VH benchmark: ONE definition, used by the lane runner, the setup
check and the Beaker submitter.

It is the roster of the published table. Open-weight models are served by the scripts in `serving/`
(one per model, on a fixed port), and each model's request profile lives in protocol._PROFILES.

The difference is the interface: VH's rollout takes the assistant from the ENVIRONMENT
(IV_ASSIST_MODEL / IV_ASSIST_BASE_URL / IV_ASSIST_PROFILE / IV_ASSIST_BACKEND /
IV_ASSIST_IMAGE_SCALE / IV_ASSIST_MAX_TOK), not from argv, so an entry carries `env` rather than
flags.

    python -m vhhome_bench.roster            # print the roster with each entry's status
"""
from vhhome import paths, keys
import os

SERVING_DIR = os.environ.get("VHHOME_SERVING") or os.path.join(paths.ROOT, "serving")   # the serve_*.sh scripts
OPENAI_URL = "https://api.openai.com/v1"
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _local(port, served, profile, *, scale=1.0, max_tok=4000):
    return {"IV_ASSIST_MODEL": served, "IV_ASSIST_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "IV_ASSIST_PROFILE": profile, "IV_ASSIST_IMAGE_SCALE": str(scale),
            "IV_ASSIST_MAX_TOK": str(max_tok)}


# status: ready | needs_key | not_ported (the native seats have no VH backend yet)
ROSTER = [
    # ---- A · frontier API
    dict(slot="1", tag="gpt-6-astra", group="A", kind="api", status="ready", needs=["OPENAI_API_KEY"],
         env={"IV_ASSIST_MODEL": "gpt-6-astra", "IV_ASSIST_BASE_URL": OPENAI_URL,
              "IV_ASSIST_PROFILE": "gpt5"}),
    # no Anthropic request backend in protocol.assist_frames yet; the seat is not in the published table
    dict(slot="2", tag="claude-opus-5", group="A", kind="api", status="not_ported", needs=["ANTHROPIC_API_KEY"],
         env={"IV_ASSIST_BACKEND": "anthropic", "IV_ASSIST_MODEL": "claude-opus-5"}),
    dict(slot="3", tag="gemini-3.1-pro-preview", group="A", kind="api", status="ready", needs=["GEMINI_API_KEY"],
         env={"IV_ASSIST_MODEL": "gemini-3.1-pro-preview"}),
    dict(slot="4", tag="gemini-3.8-flash", group="A", kind="api", status="ready", needs=["GEMINI_API_KEY"],
         env={"IV_ASSIST_MODEL": "gemini-3.8-flash"}),
    dict(slot="5", tag="qwen3.8-max", group="A", kind="api", status="needs_key", needs=["DASHSCOPE_API_KEY"],
         env={"IV_ASSIST_MODEL": "qwen3.8-max", "IV_ASSIST_BASE_URL": DASHSCOPE_URL,
              "IV_ASSIST_PROFILE": "default"}),
    # ---- B · streaming protocol
    dict(slot="6", tag="gemini-3.8-live", group="B", kind="api", status="ready", needs=["GEMINI_API_KEY"],
         env={"IV_ASSIST_BACKEND": "live", "IV_ASSIST_MODEL": "gemini-3.8-live"}),
    # NATIVE seats (ported 2026-10-04, vhhome_bench/native.py): the model in its own streaming mode. The
    # person's speech reaches it through OpenAI TTS, so both keys are needed for the Gemini seat.
    dict(slot="7", tag="gemini-3.8-live__native", group="B", kind="native", status="ready",
         needs=["GEMINI_API_KEY", "OPENAI_API_KEY"],
         env={"IV_ASSIST_BACKEND": "live_native", "IV_ASSIST_MODEL": "gemini-3.8-live"}),
    dict(slot="8", tag="gpt-realtime-2.1", group="B", kind="api", status="ready", needs=["OPENAI_API_KEY"],
         env={"IV_ASSIST_BACKEND": "realtime", "IV_ASSIST_MODEL": "gpt-realtime-2.1"}),
    dict(slot="9", tag="gpt-realtime-2.1__native", group="B", kind="native", status="ready",
         needs=["OPENAI_API_KEY"],
         env={"IV_ASSIST_BACKEND": "realtime_native", "IV_ASSIST_MODEL": "gpt-realtime-2.1"}),
    dict(slot="10", tag="qwen3.5-omni-plus-realtime", group="B", kind="api", status="needs_key",
         needs=["DASHSCOPE_API_KEY"], env={}),
    # ---- C · open-weight vision-language, one GPU and one port each
    dict(slot="11", tag="qwen3.8-27b", group="C", kind="local", status="ready", serve="serve_qwen.sh",
         port=18300, env=_local(18300, "qwen3.8-27b", "qwen38-schema")),
    dict(slot="12a", tag="molmo2-8b", group="C", kind="local", status="ready", serve="serve_molmo2.sh",
         port=18400, env=_local(18400, "molmo2-8b", "molmo2-schema", scale=0.5)),
    # DROPPED from bench_v1 on project decision (2026-09-20): its replies fail the reply schema in ~3 of 4
    # calls in VH while molmo2-8b on the same profile is clean, and the 8B already represents the family.
    dict(slot="12b", tag="molmo2-o-7b", group="C", kind="local", status="dropped", serve="serve_molmo2_o.sh",
         port=18600, env=_local(18600, "molmo2-o-7b", "molmo2-schema", scale=0.5)),
    dict(slot="13", tag="internvideo3-8b", group="C", kind="local", status="ready", serve="serve_internvideo3.sh",
         port=18700, env=_local(18700, "internvideo3-8b", "internvideo3-schema", max_tok=1024)),
    dict(slot="14a", tag="glm-4.6v-flash", group="C", kind="local", status="ready", serve="serve_glm46v_flash.sh",
         port=18500, env=_local(18500, "glm-4.6v-flash", "glm46v-flash-schema")),
    dict(slot="14b", tag="kimi-vl-thinking", group="C", kind="local", status="ready", serve="serve_kimi_vl.sh",
         port=18650, env=_local(18650, "kimi-vl-thinking", "kimi-vl-think-schema", max_tok=8000)),
    dict(slot="15", tag="museglimmer", group="C", kind="local", status="ready", serve="serve_museglimmer.sh",
         port=18200, env=_local(18200, "museglimmer", "museglimmer-nothink-schema")),
    # ---- D · duplex / proactive open. The last three are shim servers: they wrap their own loop's
    # speech into the reply JSON and never set a flag, so read them on what they SAY.
    dict(slot="16", tag="minicpmo", group="D", kind="local", status="ready", serve="serve_minicpmo.sh",
         port=18100, env=_local(18100, "minicpmo", "minicpmo-schema")),
    dict(slot="17", tag="qwen3-omni-30b-a3b", group="D", kind="local", status="ready", serve="serve_qwen3_omni.sh",
         port=18800, env=_local(18800, "qwen3-omni-30b-a3b", "qwen3-omni-schema")),
    dict(slot="18", tag="mmduet2", group="D", kind="local", status="ready", serve="serve_mmduet2.sh",
         port=18900, env=_local(18900, "mmduet2", "shim")),
    dict(slot="19", tag="proact-vl", group="D", kind="local", status="ready", serve="serve_proact_vl.sh",
         port=18950, env=_local(18950, "proact-vl", "shim")),
    dict(slot="20", tag="streamingvlm", group="D", kind="local", status="ready", serve="serve_streamingvlm.sh",
         port=19000, env=_local(19000, "streamingvlm", "shim")),
    # ---- the floor every restraint number is read against
    dict(slot="F", tag="silent", group="control", kind="control", status="ready", needs=[],
         env={"IV_ASSIST_MODEL": "silent"}),
]


def by_tag(tag):
    for r in ROSTER:
        if r["tag"] == tag:
            return r
    raise KeyError(tag)


def missing_keys(entry):
    """Keys the entry needs that neither the environment nor the keys file provides (see vhhome.keys)."""
    return [k for k in entry.get("needs", []) if not keys.get(k, required=False)]


def runnable(tags=None):
    """Entries we can actually run here, in roster order."""
    out = []
    for e in ROSTER:
        if tags and e["tag"] not in tags:
            continue
        if e["status"] in ("needs_key", "not_ported", "dropped") or missing_keys(e):
            continue
        out.append(e)
    return out


def serve_path(entry):
    return os.path.join(SERVING_DIR, entry["serve"])


def main():
    """vhhome-roster: the roster with each entry's status here (keys present, serving script)."""
    print(f"{'slot':5} {'tag':26} {'grp':4} {'kind':7} {'status':11} where")
    for e in ROSTER:
        miss = missing_keys(e)
        st = e["status"] if e["status"] != "ready" else ("MISSING:" + ",".join(miss) if miss else "ready")
        where = (f"port {e['port']} · {e['serve']}" if e["kind"] == "local"
                 else e.get("env", {}).get("IV_ASSIST_MODEL", "—"))
        print(f"{e['slot']:5} {e['tag']:26} {e['group']:4} {e['kind']:7} {st:11} {where}")
    print(f"\nrunnable here: {len(runnable())} of {len(ROSTER)}")


if __name__ == "__main__":
    main()
