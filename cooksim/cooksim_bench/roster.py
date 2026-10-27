"""The balanced-20 assistant roster: ONE definition, used by the run drivers, the roster check and the docs.

Every entry says how the assistant is reached (`--assist-*` flags passed to python -m cooksim_bench.e2e_stepin_rollout),
what it needs (API keys, or a local server: serve script, port, GPU memory), and where its setup notes are.
Slots follow the model-selection page (hellomuffin.github.io/cooksim-results-v6/model_selection.html):
A frontier API · B streaming protocol · C open-weight vision-language · D duplex / proactive open.

    python -m cooksim_bench.roster                  # print the roster with each entry's status
"""
import os

OPENAI_URL = "https://api.openai.com/v1"
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _local(port, served, profile, *, scale=1.0, max_tokens=4000):
    return ["--assist-backend", "openai", "--assist-model", served,
            "--assist-base-url", f"http://127.0.0.1:{port}/v1", "--assist-profile", profile,
            "--assist-max-tokens", str(max_tokens), "--assist-image-scale", str(scale)]


# status: ready | needs_key | blocked | built (built = works, open ruling on what it measures)
#         benched = set up and checked, but kept OUT of the formal run (user's call)
ROSTER = [
    # ---- A · frontier API
    dict(slot="1", tag="gpt-6-astra", group="A", kind="api", status="ready", needs=["OPENAI_API_KEY"],
         flags=["--assist-backend", "openai", "--assist-model", "gpt-6-astra", "--assist-base-url", OPENAI_URL,
                "--assist-profile", "gpt5"]),
    dict(slot="2", tag="claude-opus-5", group="A", kind="api", status="needs_key", needs=["ANTHROPIC_API_KEY"],
         flags=["--assist-backend", "anthropic", "--assist-model", "claude-opus-5"]),
    dict(slot="3", tag="gemini-3.1-pro-preview", group="A", kind="api", status="ready", needs=["GEMINI_API_KEY"],
         flags=["--assist-backend", "gemini", "--assist-model", "gemini-3.1-pro-preview"]),
    dict(slot="4", tag="gemini-3.8-flash", group="A", kind="api", status="ready", needs=["GEMINI_API_KEY"],
         flags=["--assist-backend", "gemini", "--assist-model", "gemini-3.8-flash"]),
    dict(slot="5", tag="qwen3.8-max", group="A", kind="api", status="needs_key", needs=["DASHSCOPE_API_KEY"],
         flags=["--assist-backend", "openai", "--assist-model", "qwen3.8-max", "--assist-base-url", DASHSCOPE_URL,
                "--assist-profile", "default"]),
    # ---- B · streaming protocol (the same weights polled by us, or deciding for themselves)
    dict(slot="6", tag="gemini-3.8-live", group="B", kind="api", status="ready", needs=["GEMINI_API_KEY"],
         flags=["--assist-backend", "live", "--assist-model", "gemini-3.8-live"]),
    dict(slot="7", tag="gemini-3.8-live__native", group="B", kind="native", status="built",
         needs=["GEMINI_API_KEY", "OPENAI_API_KEY"],     # the cook's voice is OpenAI TTS
         flags=["--assist-backend", "live_native", "--assist-model", "gemini-3.8-live"]),
    dict(slot="8", tag="gpt-realtime-2.1", group="B", kind="api", status="ready", needs=["OPENAI_API_KEY"],
         flags=["--assist-backend", "realtime", "--assist-model", "gpt-realtime-2.1"]),
    dict(slot="9", tag="gpt-realtime-2.1__native", group="B", kind="native", status="built", needs=["OPENAI_API_KEY"],
         flags=["--assist-backend", "realtime_native", "--assist-model", "gpt-realtime-2.1"]),
    dict(slot="10", tag="qwen3.5-omni-plus-realtime", group="B", kind="api", status="needs_key",
         needs=["DASHSCOPE_API_KEY"], flags=[]),       # needs an input_image_buffer adapter too
    # ---- C · open-weight vision-language, one GPU and one port each
    dict(slot="11", tag="qwen3.8-27b", group="C", kind="local", status="ready", serve="serve_qwen.sh", port=18300,
         notes="serving/notes/qwen.md", flags=_local(18300, "qwen3.8-27b", "qwen38-schema")),
    dict(slot="12a", tag="molmo2-8b", group="C", kind="local", status="ready", serve="serve_molmo2.sh", port=18400,
         flags=_local(18400, "molmo2-8b", "molmo2-schema", scale=0.5)),
    # benched 2026-09-20: set up and checked, but the formal run uses Molmo2-8B only (user's call)
    dict(slot="12b", tag="molmo2-o-7b", group="C", kind="local", status="benched", serve="serve_molmo2_o.sh", port=18600,
         notes="serving/notes/molmo2-o.md", flags=_local(18600, "molmo2-o-7b", "molmo2-schema", scale=0.5)),
    dict(slot="13", tag="internvideo3-8b", group="C", kind="local", status="ready", serve="serve_internvideo3.sh",
         port=18700, notes="serving/notes/internvideo3.md",
         flags=_local(18700, "internvideo3-8b", "internvideo3-schema", max_tokens=1024)),
    dict(slot="14a", tag="glm-4.6v-flash", group="C", kind="local", status="ready", serve="serve_glm46v_flash.sh",
         port=18500, notes="serving/notes/glm46v-flash.md", flags=_local(18500, "glm-4.6v-flash", "glm46v-flash-schema")),
    dict(slot="14b", tag="kimi-vl-thinking", group="C", kind="local", status="ready", serve="serve_kimi_vl.sh",
         port=18650, notes="serving/notes/kimi-vl.md",
         flags=_local(18650, "kimi-vl-thinking", "kimi-vl-think-schema", max_tokens=8000)),
    dict(slot="15", tag="museglimmer", group="C", kind="local", status="ready", serve="serve_museglimmer.sh",
         port=18200, flags=_local(18200, "museglimmer", "museglimmer-nothink-schema")),
    # ---- D · duplex / proactive open
    dict(slot="16", tag="minicpmo", group="D", kind="local", status="ready", serve="serve_minicpmo.sh", port=18100,
         flags=_local(18100, "minicpmo", "minicpmo-schema")),
    dict(slot="17", tag="qwen3-omni-30b-a3b", group="D", kind="local", status="ready", serve="serve_qwen3_omni.sh",
         port=18800, notes="serving/notes/qwen3-omni.md",
         flags=_local(18800, "qwen3-omni-30b-a3b", "qwen3-omni-schema")),
    dict(slot="18", tag="mmduet2", group="D", kind="local", status="ready", serve="serve_mmduet2.sh", port=18900,
         notes="serving/notes/mmduet2.md (replaces Streamo, whose weights were never released; silent on the rulebook prompt)",
         flags=_local(18900, "mmduet2", "default")),
    dict(slot="19", tag="proact-vl", group="D", kind="local", status="ready", serve="serve_proact_vl.sh", port=18950,
         notes="serving/notes/proact-vl.md", flags=_local(18950, "proact-vl", "default")),
    dict(slot="20", tag="streamingvlm", group="D", kind="local", status="ready", serve="serve_streamingvlm.sh",
         port=19000, notes="serving/notes/streamingvlm.md", flags=_local(19000, "streamingvlm", "default")),
]

# the fixed 3-episode end-to-end check: one per persona, easy / hard / master recipe, medium map
CHECK_EPISODES = [("baseline", "b3_easy_nops_medium_map_2"),
                  ("classic_novice", "b3_hard_nops_medium_map_3"),
                  ("classic_expert", "b3_master_nops_medium_map_3")]


# ---- COOK-MODEL ABLATION roster (2026-09-22) ---------------------------------------------
# Four models in the COOK (simulated-human) seat, one fixed assistant, one fixed 30-episode
# subset. Everything the harness does is identical across the four; only the endpoint behind
# the cook's voice changes. Sampling is each model's own card values (an OpenAI reasoning
# model cannot take temperature at all), and NO cook profile carries the assistant reply
# schema -- the cook emits its own move/plan_update JSON.
COOK_ABLATION_ASSISTANT = ["--assist-backend", "gemini", "--assist-model", "gemini-3.7-flash"]

COOK_ABLATION = [
    dict(tag="gpt-6-astra", kind="api", needs=["OPENAI_API_KEY"],
         flags=["--cook-backend", "openai", "--cook-model", "gpt-6-astra",
                "--cook-base-url", OPENAI_URL, "--cook-profile", "gpt5-cook",
                "--cook-max-tokens", "4000"]),
    # the Realtime seat keeps its NATIVE shape: one WebSocket session per episode, session
    # instructions = the static persona hint, each turn's world context as the user message
    # (assistant_backend.AssistantBackend.sup_call, cook_seat=True)
    dict(tag="gpt-realtime-2.1", kind="api", needs=["OPENAI_API_KEY"],
         flags=["--cook-backend", "realtime", "--cook-model", "gpt-realtime-2.1"]),
    dict(tag="qwen3.8-27b", kind="local", serve="serve_qwen.sh", port=18300,
         flags=["--cook-backend", "openai", "--cook-model", "qwen3.8-27b",
                "--cook-base-url", "http://127.0.0.1:18300/v1",
                "--cook-profile", "qwen38-cook", "--cook-max-tokens", "4000"]),
    dict(tag="museglimmer", kind="local", serve="serve_museglimmer.sh", port=18200,
         flags=["--cook-backend", "openai", "--cook-model", "museglimmer",
                "--cook-base-url", "http://127.0.0.1:18200/v1",
                "--cook-profile", "museglimmer-cook", "--cook-max-tokens", "4000"]),
]

# the fixed cell list every cook runs (python -m cooksim_bench.suite.draw_cook_ablation_cells wrote it)
COOK_ABLATION_CELLS = "benchmark/cook_ablation_cells.json"


def cook_by_tag(tag):
    for r in COOK_ABLATION:
        if r["tag"] == tag:
            return r
    raise KeyError(tag)


def by_tag(tag):
    for r in ROSTER:
        if r["tag"] == tag:
            return r
    raise KeyError(tag)


def missing_keys(entry):
    return [k for k in entry.get("needs", []) if not os.environ.get(k)]


def main(argv=None):
    """cooksim-roster: print the assistant roster (tags accepted by --assistant), or the cook personas."""
    import argparse
    ap = argparse.ArgumentParser(prog="cooksim-roster")
    ap.add_argument("--personas", action="store_true", help="list the cook personas instead")
    a = ap.parse_args(argv)
    if a.personas:
        from cooksim.interact.persona_v2 import ROSTER as PERSONAS
        for p in PERSONAS:
            print(f"{p.name:24s} skill={p.skill:6s} trust={p.trust:6s} questionnaire={p.q1},{p.q2},{p.q3}")
        return
    print(f"{'slot':>4} grp {'tag':30s} {'status':10s} how")
    for r in ROSTER:
        where = f"local :{r['port']} {r['serve']}" if r["kind"] == "local" else r["kind"]
        print(f"{r['slot']:>4} {r['group']} {r['tag']:30s} {r['status']:10s} {where}")


if __name__ == "__main__":
    main()
