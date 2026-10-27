# Assistants

This page is the guide to the *assistant seat* of CookSim: which models we evaluated, how the harness reaches each one, what each needs (an API key or a locally served model), and how to put a model of your own in the seat. For running episodes and scoring them once an assistant is reachable, see [REPRODUCE.md](REPRODUCE.md); for what is measured, see [BENCHMARK.md](BENCHMARK.md).

```bash
cooksim-roster              # every evaluated assistant: tag, group, status, how it is reached
cooksim-roster --personas   # the cook personas
```

The roster is defined once, in `cooksim_bench/roster.py`. A roster *tag* passed to `--assistant` expands to the exact flags that produced the published numbers; `<backend>:<model>` runs anything else.

```bash
cooksim-episode --case b3_easy_nops_medium_map_2 --persona baseline --assistant gpt-6-astra --out runs/demo
cooksim-episode --case b3_easy_nops_medium_map_2 --persona baseline --assistant gemini:gemini-3.8-flash --out runs/demo
cooksim-benchmark --assistant qwen3.8-27b --out runs/qwen --workers 2
```

The renderer must be running in another terminal (`cooksim-server --port 17773`).

## 1. What every assistant receives

In the default **polled** setting the harness calls the assistant once every 2 ticks (a tick is 1.2 s of game time) with:

- a fixed system prompt (the assistant's role, the kitchen rulebook, the recipe, the cook's persona questionnaire, the mistake taxonomy);
- a user turn with the assistant's previous reply, the recent conversation and the engine's feedback;
- the last 8 frames, each a 1042x396 JPEG showing the top-down kitchen beside the cook's first-person view.

It must answer with one JSON object (`cooksim_bench/backends/assistant_reply_schema.json`): `cook_progress` (completed / current / future subgoals), `goal_change`, `text` (what the cook hears; empty = stay silent) and `flag` (the name of the mistake it is calling, if any). The seconds a reply takes are charged to the game clock (`COOKSIM_ASSIST_LATENCY=measured`, the default), so a slow model's warning lands late in the kitchen. `--latency 2` replaces the measured time by a fixed 2 ticks for every reply (the latency ablation).

Every seat is parsed by the same JSON reader in plain JSON mode. No provider-side schema enforcement is used for the API seats, because the streaming seats cannot have it and parity matters more than a few parse failures; locally served models are given the reply schema as guided decoding (section 5), which fixes their output format without changing what they are asked.

A silent baseline needs no model: `COOKSIM_SILENT_ASSISTANT=1` never speaks, and every score can be read against that floor.

## 2. The roster

Twenty slots in four groups. *Status* is the roster's own word: **ready** = runs with the listed key or server; **needs_key** = the backend is built, we did not hold the key; **built** = runs, but see the note on what the native setting measures (section 4); **benched** = set up and checked, kept out of the published run.

### A. Frontier API

| Slot | Tag | Reached by | Needs | Status |
|---|---|---|---|---|
| 1 | `gpt-6-astra` | `--assist-backend openai --assist-model gpt-6-astra --assist-base-url https://api.openai.com/v1 --assist-profile gpt5` | `OPENAI_API_KEY` | ready |
| 2 | `claude-opus-5` | `--assist-backend anthropic --assist-model claude-opus-5` | `ANTHROPIC_API_KEY` (+ `ANTHROPIC_WORKSPACE_ID` if your account needs it) | needs_key |
| 3 | `gemini-3.1-pro-preview` | `--assist-backend gemini --assist-model gemini-3.1-pro-preview` | `GEMINI_API_KEY` | ready |
| 4 | `gemini-3.8-flash` | `--assist-backend gemini --assist-model gemini-3.8-flash` | `GEMINI_API_KEY` | ready |
| 5 | `qwen3.8-max` | `--assist-backend openai --assist-model qwen3.8-max --assist-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 --assist-profile default` | `DASHSCOPE_API_KEY` | needs_key |

### B. Streaming protocol (the same weights, polled by us or deciding for themselves)

| Slot | Tag | Reached by | Needs | Status |
|---|---|---|---|---|
| 6 | `gemini-3.8-live` | `--assist-backend live --assist-model gemini-3.8-live` | `GEMINI_API_KEY` | ready |
| 7 | `gemini-3.8-live__native` | `--assist-backend live_native --assist-model gemini-3.8-live` | `GEMINI_API_KEY`, `OPENAI_API_KEY` (the cook's voice is OpenAI TTS) | built |
| 8 | `gpt-realtime-2.1` | `--assist-backend realtime --assist-model gpt-realtime-2.1` | `OPENAI_API_KEY` | ready |
| 9 | `gpt-realtime-2.1__native` | `--assist-backend realtime_native --assist-model gpt-realtime-2.1` | `OPENAI_API_KEY` | built |
| 10 | `qwen3.5-omni-plus-realtime` | no flags: not runnable in this release (Model Studio's realtime protocol takes images through `input_image_buffer.append`, which the realtime backend does not yet speak) | `DASHSCOPE_API_KEY` | needs_key |

### C. Open-weight vision-language (served locally, one GPU and one port each)

| Slot | Tag | Weights | Serving script | Port | Profile | Weights in GPU memory | Status |
|---|---|---|---|---|---|---|---|
| 11 | `qwen3.8-27b` | `Qwen/Qwen3.8-27B` bf16 | `serving/serve_qwen.sh` | 18300 | `qwen38-schema` | 52 GB (needs an 80 GB card) | ready |
| 12a | `molmo2-8b` | `allenai/Molmo2-8B` bf16 | `serving/serve_molmo2.sh` | 18400 | `molmo2-schema`, `--assist-image-scale 0.5` | 16.1 GiB | ready |
| 12b | `molmo2-o-7b` | `allenai/Molmo2-O-7B` (fp32 on disk, served bf16) | `serving/serve_molmo2_o.sh` | 18600 | `molmo2-schema`, `--assist-image-scale 0.5` | 14.5 GiB | benched |
| 13 | `internvideo3-8b` | `yanziang/InternVideo3-8B-Instruct` bf16 | `serving/serve_internvideo3.sh` (shim) | 18700 | `internvideo3-schema`, `--assist-max-tokens 1024` | about 18 GB | ready |
| 14a | `glm-4.6v-flash` | `zai-org/GLM-4.6V-Flash` bf16 | `serving/serve_glm46v_flash.sh` | 18500 | `glm46v-flash-schema` | 19.2 GiB | ready |
| 14b | `kimi-vl-thinking` | `moonshotai/Kimi-VL-A3B-Thinking-2506` bf16 | `serving/serve_kimi_vl.sh` (reasoning-parser plugin) | 18650 | `kimi-vl-think-schema`, `--assist-max-tokens 8000` | 30.7 GiB | ready |
| 15 | `museglimmer` | `RedHatAI/Muse-Glimmer-30B-INT4` + `meta-models/Muse-Glimmer-30B-assistant` (speculative drafter) | `serving/serve_museglimmer.sh` | 18200 | `museglimmer-nothink-schema` | 32 GiB (fits a 46 GB card) | ready |

### D. Duplex / proactive open models (served locally)

| Slot | Tag | Weights | Serving script | Port | Profile | Weights in GPU memory | Status |
|---|---|---|---|---|---|---|---|
| 16 | `minicpmo` | `openbmb/MiniCPM-o-4_5-awq` | `serving/serve_minicpmo.sh` | 18100 | `minicpmo-schema` | 6.9 GiB | ready |
| 17 | `qwen3-omni-30b-a3b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` bf16 (thinker only) | `serving/serve_qwen3_omni.sh` | 18800 | `qwen3-omni-schema` | 58 GiB (needs an 80 GB card) | ready |
| 18 | `mmduet2` | `wangyueqian/MMDuet2` | `serving/serve_mmduet2.sh` (shim) | 18900 | `default` | about 9 GB | ready |
| 19 | `proact-vl` | `chenjoya/LiveCC-7B-Base` + LoRA `oaaoaa/proact_lora_base_liveccbase_final` | `serving/serve_proact_vl.sh` (shim) | 18950 | `default` | about 17 GB | ready |
| 20 | `streamingvlm` | `mit-han-lab/StreamingVLM` bf16 | `serving/serve_streamingvlm.sh` (shim) | 19000 | `default` | 16.6 GB | ready |

Every local slot is reached the same way, which is what the tag expands to:

```
--assist-backend openai --assist-model <served name> --assist-base-url http://127.0.0.1:<port>/v1
--assist-profile <profile> --assist-max-tokens 4000 --assist-image-scale 1.0
```

with the per-model `--assist-max-tokens` / `--assist-image-scale` overrides shown in the tables. The "weights in GPU memory" column is the loaded size we observed; the vLLM scripts reserve 90% of the card by default (`GPU_UTIL`), so a card needs headroom beyond the weights for the KV cache.

## 3. API assistants

**Gemini** (`--assist-backend gemini`). The system prompt goes in `system_instruction`; temperature stays at the provider default; thinking is set to the lowest level for the 3.x models.

**OpenAI GPT** (`--assist-backend openai --assist-base-url https://api.openai.com/v1 --assist-profile gpt5`). Chat Completions with `reasoning_effort: low`, temperature at its default, `max_completion_tokens` as the reply budget, and a stable `prompt_cache_key` so the fixed system prompt is served from the provider's cache. `OPENAI_API_KEY` is read when the base URL is `api.openai.com`.

**Claude** (`--assist-backend anthropic`). The official SDK; frames labelled with their real format; effort `low`, the counterpart of the GPT seats' reasoning effort; the system prompt cached; no temperature (the model rejects it). There is deliberately no refusal fallback: a benchmark reply must come from the model under test, so a refusal is recorded as silence.

**Alibaba Model Studio** (`--assist-backend openai --assist-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1`). Any OpenAI-compatible chat model on the endpoint; `DASHSCOPE_API_KEY` is read when the base URL contains `dashscope`. Confirm the model id with your key.

## 4. The two streaming settings

The streaming APIs (Gemini Live, GPT realtime) can be run two ways, and the roster carries both because they measure different things.

**Polled** (`--assist-backend live | realtime`, tags without a suffix). One streaming session per episode, but the harness drives it exactly like a chat model: every 2 ticks it sends the last frames with the question turn and forces a reply. The model gets no say in *when* to speak; the setting isolates what the streaming weights see and know from how they manage the floor.

- `live`: audio-native session; the model speaks its JSON reply and the transcript is read. The frames go inline in the question turn (sent as realtime video they would arrive after the question was answered). The Live models differ about thinking levels; the backend adjusts once on refusal.
- `realtime`: one WebSocket session per episode, text output, reasoning effort `low`, replies capped at 4,000 tokens, and a bounded context (`truncation` with retention ratio 0.5 at 16k tokens), because every ask carries the full state.

**Native** (`--assist-backend live_native | realtime_native`, tags with `__native`; `cooksim_bench/backends/native_assistant.py`). The model runs in its own streaming mode and nobody polls it. The session is given the instructions once; then the harness only *feeds* it:

- the kitchen as video, one frame per tick, paced to at most one frame per 1.2 s of wall time (at most 1 fps) and the only view of the kitchen it gets;
- the cook's speech as audio (one fixed OpenAI TTS voice, the same for every assistant, synthesised ahead and outside every clock);
- a bare `[t=N]` marker with each frame, as context that asks for no reply.

Whatever the model says, whenever it says it, is delivered through the same path as a polled reply, so flags, grading and reports are unchanged. Since the native reply is speech rather than JSON, a mistake call is recognised by a spoken marker: the prompt asks the model to begin one with "Flag: <mistake name>." The marker is matched against the taxonomy, stripped from what the cook hears, and an unmatched name still counts as a flag. Latency is charged from the end of the cook's audio (for a reply) or the last frame sent (for a self-started line) to the model's first output token, then converted to ticks by the ordinary policy. Native episodes run at wall-clock pace (at least 1.2 s per tick), so they are slower to produce than polled ones.

What the native setting measures is the model's own turn-taking. In our runs both native seats spoke almost only in reply to the cook and rarely on their own from the video; that is their native behaviour, and it is the reason their detection numbers are far below the polled seats in the results.

## 5. Request profiles

`--profile` (or `--assist-profile`) selects a request profile in `cooksim_bench/backends/controller_eval.py` (`PROFILES`): how the OpenAI-compatible request is shaped for a given model. A profile sets sampling (temperature, top_p, provider-specific `extra_body` such as `top_k`, `repetition_penalty`, `chat_template_kwargs`), whether images go before or after the text, an optional assistant prefill, and whether the reply is constrained to the reply schema by guided decoding (`reply_schema=True`).

- **`default`** — temperature 0.2, nothing else; plain JSON mode. The profile for any OpenAI-compatible endpoint you have no model-specific guidance for, and the one the three shim-wrapped models use (their shim builds the JSON itself).
- **`gpt5`** — for OpenAI's reasoning models: no temperature (they reject one), `reasoning_effort: low`, `prompt_cache_key`, and the reply budget sent as `max_completion_tokens`. `gpt5-responses` is the same request on the Responses API; the roster stays on Chat Completions.
- **`<model>-schema`** — one per locally served model, following that model's own card: its recommended sampling, thinking off where the model has a thinking mode, and the reply schema as guided decoding. For example `qwen38-schema` is the Qwen3.8 non-thinking card sampling (T 0.7, top_p 0.8, top_k 20, presence penalty 1.5); `molmo2-schema` is greedy; `kimi-vl-think-schema` keeps thinking on with a 2,048-token budget and frames first (`kimi-vl-think256-schema` / `kimi-vl-nothink-schema` are the thinking ablation).
- **`*-cook`** profiles are for the cook seat in the cook-model ablation (same sampling, no reply schema).

To use your own OpenAI-compatible server:

```bash
cooksim-episode --case b3_easy_nops_medium_map_2 --persona baseline \
    --assistant openai:my-model --base-url http://127.0.0.1:8000/v1 --profile default --out runs/mine
```

Add a profile to `PROFILES` if the model's card calls for specific sampling or the reply schema.

## 6. Serving the open-weight assistants

### 6.1 Rules shared by every local model

- **One Python environment per model** at `$ROOT/envs/<name>` and one weights cache at `$ROOT/hf_cache/<name>` (`ROOT` defaults to `$HOME`). Never install a model's stack into the `cooksim` environment: the models pin incompatible versions of torch and transformers.
- **One GPU per server, one port per model**, bound to `127.0.0.1`. The scripts read `CUDA_VISIBLE_DEVICES`, `PORT`, `ROOT`, `HF_HOME` and `LOGDIR` from the environment and write a pid file under `$LOGDIR`.
- **One episode at a time per server** (`--workers 1`, or 2-3 at most). The model's latency is charged to the game clock, so two episodes sharing a GPU would each be charged the other's compute.
- **All of them are slower than the 1.2 s tick.** At the median a reply lands 1 tick late (MMDuet2) to several ticks late (Kimi-VL with thinking on), and those ticks are charged.
- The vLLM scripts use vLLM 0.19.1. A `/tokenize` request that contains images corrupts that version's image cache; every later chat request with the same frames fails with HTTP 500 until a restart. Never call it on a server that is running episodes.
- The shims in `serving/shims/` are small OpenAI-compatible servers around the authors' own inference code. Three of them (`mmduet2`, `proact-vl`, `streamingvlm`) import the authors' repository from `serving/shims/<name>/src`, which you clone there; each model's note in `serving/notes/<name>.md` names the repository and the exact package versions. `serving/shims/internvideo3/shim_server.py` is a plain transformers server; `serving/shims/kimi-vl/kimi_vl_reasoning_parser.py` is a vLLM reasoning-parser plugin, not a server.

### 6.2 Starting a model

```bash
export ROOT=$HOME                                  # where envs/<name> and hf_cache/<name> live
CUDA_VISIBLE_DEVICES=0 bash serving/serve_qwen.sh &
curl -s http://127.0.0.1:18300/v1/models           # wait until it answers (minutes on a network disk)
cooksim-episode --case b3_easy_nops_medium_map_2 --persona baseline --assistant qwen3.8-27b --out runs/qwen
```

Each model's environment recipe, weights download and fixes are in `serving/notes/<name>.md`; the item-by-item check of each setting against the model's own guidance is in `serving/notes/<name>.audit.md`.

### 6.3 The models

| Model | Served by | What matters, and why |
|---|---|---|
| **Qwen3.8-27B** | vLLM | `--chat-template-content-format openai`: without it vLLM treats the template as "string" and silently moves all 8 frames in front of the text. Built-in MTP speculative decoding at depth 5. Thinking off server-wide; the card's non-thinking sampling. |
| **Molmo2-8B** | vLLM | Molmo2 has no system role: `serving/molmo2_chat_template_system.jinja` folds the instructions into the first user turn (byte-identical to the original for conversations without a system message). Greedy decoding. Half-size frames on purpose: full size was slower and caught fewer mistakes. |
| **Molmo 2-O 7B** | vLLM | Same template and profile as the 8B. Checked but kept out of the published run. |
| **InternVideo3-8B** | transformers shim | Not a vLLM architecture. The 8 frames go in as one clip (`FRAMES=video`, 1.2 s apart; `FRAMES=images` sends them separately); flash-attention 2; the model's own sampling plus `repetition_penalty 1.05`, without which replies loop to the token cap. Run with `--assist-max-tokens 1024`. |
| **GLM-4.6V-Flash** | vLLM | The card's sampling (T 0.8, top_p 0.6, top_k 2, repetition 1.1); thinking off via `enable_thinking=false`; `--reasoning-parser glm45`. |
| **Kimi-VL-A3B-Thinking-2506** | vLLM | vLLM has no parser for its `◁think▷` markers: `serving/shims/kimi-vl/kimi_vl_reasoning_parser.py` keeps the thinking out of the answer and applies the schema after it. Frames first, a 2,048-token thinking budget, T 0.8; `--assist-max-tokens 8000`. |
| **Muse Glimmer 30B** | vLLM (nightly) | INT4 with the DFlash speculative drafter rather than the official BF16 on purpose: BF16 fits an 80 GB card but was slower with no detection gain. `--structured-outputs-config {"enable_in_reasoning":true}` is required because the no-think profile pre-fills the answer channel. |
| **MiniCPM-o 4.5** | vLLM | The maintainers' own AWQ checkpoint (`VARIANT=bf16` serves the original); one slice per frame (`MAX_SLICE_NUMS=1`); thinking off. |
| **Qwen3-Omni-30B-A3B** | vLLM | Plain vLLM loads the thinker only and skips the audio talker (we read text), so no audio weights are needed. The card's sampling. |
| **MMDuet2** | shim | The authors' demo loop: the frames replayed as 2-frame user turns, greedy, "NO REPLY" → empty text. It cannot write the reply JSON; the shim wraps its speech. With the authors' prompt shape it stays silent on the benchmark rulebook, although it answers short "what is happening" questions. |
| **Proact-VL** | shim | The ungated LoRA of the authors' recipe on LiveCC-7B-Base, FLAG head included. The authors' per-chunk loop with **their own system prompt** (`PROACT_PROMPT=ego4d`; `PROACT_PROMPT=bench` restores the benchmark rulebook, which the model recited back). Reply = what it says on the 2 newest chunks; JSON wrapped by the shim. |
| **StreamingVLM** | shim | The authors' streaming loop (1-s chunks, sink/window KV cache); no system turn, context in the loop's "previous text" slot (`SVLM_PROMPT=native`; `SVLM_PROMPT=bench` restores the benchmark system turn). JSON wrapped by the shim. |

Three consequences to keep in mind when reading the results:

- MMDuet2, Proact-VL and StreamingVLM never set a `flag` (their reply is wrapped speech), so flag-based detection reads zero for them by construction; judge them on what they say.
- The same three run with their authors' prompt shape and therefore **do not see the benchmark rulebook**. That is the faithful way to run them; whether to hold them to the rulebook is a benchmark-design choice, and the environment variables above let you make the other one.
- Where the tag's flags deviate from a model's published guidance (frame order, sampling, a repetition penalty), the deviation was measured and kept only where it was no worse; the per-model `*.audit.md` notes record each comparison.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `the renderer is not answering` | Start `cooksim-server --port 17773` in another terminal, or point `COOKSIM_PAGE` at a running one. |
| `--assistant must be <backend>:<model> or a roster tag` | The tag is not in `cooksim-roster`; use `backend:model` with `--base-url` / `--profile` for anything else. |
| A local model answers without looking at the right frames | Check the rendered prompt order (Qwen3.8's "string" template format, section 6.3). |
| HTTP 500 "Expected a cached item for mm_hash" | Someone called `/tokenize` with images on vLLM 0.19.1; restart the server. |
| Molmo2 HTTP 400 "Conversation roles must alternate" | The server was started without `molmo2_chat_template_system.jinja`. |
| Kimi-VL replies contain `◁think▷` text or run out of tokens | The reasoning-parser plugin is not loaded, or the profile lacks the thinking budget. |
| InternVideo3 calls take about 40 s now and then | Repetition runaways; use `internvideo3-schema` and `--assist-max-tokens 1024`. |
| StreamingVLM import error | transformers must be 4.52.4 (not the 4.51.3 the authors' environment script installs). |
| Claude replies are always empty | Frames must be labelled with their real format (JPEG); the backend sniffs it. |
| `gemini-3.8-live` session fails at setup | It refuses a thinking level; the backend drops it automatically. |
| Native seat says nothing for a whole episode | Expected often (section 4): the model only replies to the cook. Check `OPENAI_API_KEY` is set for the cook's TTS voice. |
