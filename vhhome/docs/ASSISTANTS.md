# Assistants

This page is the guide to the *assistant seat* of VH-Home: which models we evaluated, how the harness reaches each
one, what each needs (an API key or a locally served model), and how to put a model of your own in the seat. For
running episodes and scoring them once an assistant is reachable, see [REPRODUCE.md](REPRODUCE.md) and
[EVALUATE_YOUR_MODEL.md](EVALUATE_YOUR_MODEL.md); for what is measured, see [BENCHMARK.md](BENCHMARK.md).

```bash
vhhome-roster               # every evaluated assistant: slot, tag, group, kind, status, how it is reached
```

The roster is defined once, in `vhhome_bench/roster.py`. A roster *tag* passed to `--assistant` expands to the exact
backend, model, request profile and (for local models) port that produced the published numbers;
`<backend>:<model>` runs anything else.

```bash
vhhome-episode --latency 2 --case cx_tidy_missing_rep --persona baseline --assistant gpt-6-astra --out runs/demo
vhhome-episode --latency 2 --case cx_tidy_missing_rep --persona baseline --assistant gemini:gemini-3.8-flash --out runs/demo
vhhome-benchmark --latency 2 --assistant qwen3.8-27b --out runs/qwen --engines 8080
```

A simulator instance must be running (`bash simulator/launch_simulators.sh 1`) and `GEMINI_API_KEY` must be set
for every assistant, because the simulated person is a Gemini model.

## 1. What every assistant receives

In the default **polled** setting the harness calls the assistant once every 2 ticks (a tick is one second of
animation; a reply's wall time is converted to ticks at 1.2 s per tick) with:

- a fixed system prompt: the assistant's role, the task instruction and its public plan, the house rules, the
  person's questionnaire answers, the six mistake classes with their definitions, the commands the person's body
  accepts, the task's true ordering constraints, the reply format;
- a user turn with the current tick, the assistant's previous JSON reply (its only memory), the conversation so far
  and the person's latest line;
- the last 8 first-person frames (each with the house plan beside it), one per second, followed by a whole-house
  map and a map of the room the person is in: 10 images.

It must answer with one JSON object (`vhhome/data/assistant_reply_schema.json`): `human_progress` (completed /
current / future subgoals), `key_moments` (its own notes, at most 10), `text` (what the person hears; empty = stay
silent) and `flag` (`null` or `{"error": "<mistake name>"}`). The seconds a reply takes are charged to the game
clock (`ceil(x / 1.2)` ticks, at most 15), so a slow model's warning lands late. The protocol is spelled out in
[EVALUATE_YOUR_MODEL.md](EVALUATE_YOUR_MODEL.md).

Every seat is parsed by the same JSON reader (`protocol._extract_json`). The API seats are run in plain JSON mode
without provider-side schema enforcement, because the streaming seats cannot have it and parity matters more than a
few parse failures; locally served models are given the reply schema as guided decoding (section 5), which fixes
their output format without changing what they are asked.

A silent baseline needs no model: `--assistant silent` never speaks, and every score can be read against that floor.

## 2. The roster

Twenty slots in four groups, plus the control. *Status* is the roster's own word: **ready** = runs with the listed
key or server; **needs_key** = the backend is built, we did not hold the key; **not_ported** = no request backend
for it in this harness; **dropped** = set up and checked, kept out of the published run. The "reached by" column
gives the environment the tag expands to (the harness takes the assistant from `IV_ASSIST_*` variables).

### A. Frontier API

| Slot | Tag | Reached by | Needs | Status |
|---|---|---|---|---|
| 1 | `gpt-6-astra` | `IV_ASSIST_MODEL=gpt-6-astra`, `IV_ASSIST_BASE_URL=https://api.openai.com/v1`, `IV_ASSIST_PROFILE=gpt5` | `OPENAI_API_KEY` | ready |
| 2 | `claude-opus-5` | `IV_ASSIST_BACKEND=anthropic` | `ANTHROPIC_API_KEY` | ready |
| 3 | `gemini-3.1-pro-preview` | `IV_ASSIST_MODEL=gemini-3.1-pro-preview` (Gemini `generateContent`) | `GEMINI_API_KEY` | ready |
| 4 | `gemini-3.8-flash` | `IV_ASSIST_MODEL=gemini-3.8-flash` | `GEMINI_API_KEY` | ready |
| 5 | `qwen3.8-max` | `IV_ASSIST_MODEL=qwen3.8-max`, `IV_ASSIST_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, `IV_ASSIST_PROFILE=default` | `DASHSCOPE_API_KEY` | needs_key |

### B. Streaming protocol (the same weights, polled by us or deciding for themselves)

| Slot | Tag | Reached by | Needs | Status |
|---|---|---|---|---|
| 6 | `gemini-3.8-live` | `IV_ASSIST_BACKEND=live`, `IV_ASSIST_MODEL=gemini-3.8-live` | `GEMINI_API_KEY` | ready |
| 7 | `gemini-3.8-live__native` | `IV_ASSIST_BACKEND=live_native`, `IV_ASSIST_MODEL=gemini-3.8-live` | `GEMINI_API_KEY`, `OPENAI_API_KEY` (the person's voice is OpenAI TTS) | ready |
| 8 | `gpt-realtime-2.1` | `IV_ASSIST_BACKEND=realtime`, `IV_ASSIST_MODEL=gpt-realtime-2.1` | `OPENAI_API_KEY` | ready |
| 9 | `gpt-realtime-2.1__native` | `IV_ASSIST_BACKEND=realtime_native`, `IV_ASSIST_MODEL=gpt-realtime-2.1` | `OPENAI_API_KEY` | ready |
| 10 | `qwen3.5-omni-plus-realtime` | no environment: not runnable in this release | `DASHSCOPE_API_KEY` | needs_key |

### C. Open-weight vision-language (served locally, one GPU and one port each)

| Slot | Tag | Weights | Serving script | Port | Profile | Per-seat overrides | Status |
|---|---|---|---|---|---|---|---|
| 11 | `qwen3.8-27b` | `Qwen/Qwen3.8-27B` bf16 | `serving/serve_qwen.sh` | 18300 | `qwen38-schema` | | ready |
| 12a | `molmo2-8b` | `allenai/Molmo2-8B` bf16 | `serving/serve_molmo2.sh` | 18400 | `molmo2-schema` | image scale 0.5 | ready |
| 12b | `molmo2-o-7b` | (script not shipped) | `serving/serve_molmo2_o.sh` | 18600 | `molmo2-schema` | image scale 0.5 | dropped |
| 13 | `internvideo3-8b` | `yanziang/InternVideo3-8B-Instruct` bf16 | `serving/serve_internvideo3.sh` (shim) | 18700 | `internvideo3-schema` | max tokens 1024 | ready |
| 14a | `glm-4.6v-flash` | `zai-org/GLM-4.6V-Flash` bf16 | `serving/serve_glm46v_flash.sh` | 18500 | `glm46v-flash-schema` | | ready |
| 14b | `kimi-vl-thinking` | `moonshotai/Kimi-VL-A3B-Thinking-2506` bf16 | `serving/serve_kimi_vl.sh` (reasoning-parser plugin) | 18650 | `kimi-vl-think-schema` | max tokens 8000 | ready |
| 15 | `museglimmer` | `RedHatAI/Muse-Glimmer-30B-INT4` + `meta-models/Muse-Glimmer-30B-assistant` (speculative drafter) | `serving/serve_museglimmer.sh` | 18200 | `museglimmer-nothink-schema` | | ready |

### D. Duplex / proactive open models (served locally)

| Slot | Tag | Weights | Serving script | Port | Profile | Status |
|---|---|---|---|---|---|---|
| 16 | `minicpmo` | `openbmb/MiniCPM-o-4_5-awq` (`VARIANT=bf16` serves `openbmb/MiniCPM-o-4_5`) | `serving/serve_minicpmo.sh` | 18100 | `minicpmo-schema` | ready |
| 17 | `qwen3-omni-30b-a3b` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` bf16 (thinker only) | `serving/serve_qwen3_omni.sh` | 18800 | `qwen3-omni-schema` | ready |
| 18 | `mmduet2` | `wangyueqian/MMDuet2` | `serving/serve_mmduet2.sh` (shim) | 18900 | `shim` | ready |
| 19 | `proact-vl` | LiveCC-7B-Base + LoRA `oaaoaa/proact_lora_base_liveccbase_final` | `serving/serve_proact_vl.sh` (shim) | 18950 | `shim` | ready |
| 20 | `streamingvlm` | `mit-han-lab/StreamingVLM` bf16 | `serving/serve_streamingvlm.sh` (shim) | 19000 | `shim` | ready |

### The control

| Slot | Tag | Reached by | Needs | Status |
|---|---|---|---|---|
| F | `silent` | `IV_ASSIST_MODEL=silent`: never speaks | nothing | ready |

Every local slot is reached the same way, which is what the tag expands to:

```
IV_ASSIST_MODEL=<served name>  IV_ASSIST_BASE_URL=http://127.0.0.1:<port>/v1
IV_ASSIST_PROFILE=<profile>    IV_ASSIST_IMAGE_SCALE=1.0   IV_ASSIST_MAX_TOK=4000
```

with the per-seat image-scale and max-token overrides shown in the tables. The same settings are what
`--assistant openai:<served name> --base-url http://127.0.0.1:<port>/v1 --profile <profile>` gives you by hand.

## 3. API assistants

**Gemini** (`gemini:<model>`, the default backend). One `generateContent` call per turn: the system prompt and the
user text as the first parts, then the images; JSON response mode, temperature 0.2, a 512-token thinking budget,
4000 output tokens. Transient errors (rate limits, 5xx, timeouts) are retried; a network outage is waited out for up
to about six minutes before the turn is given up.

**OpenAI GPT** (`openai:<model>`, base URL `https://api.openai.com/v1`, profile `gpt5`). Chat Completions with
`reasoning_effort: low`, no temperature (the reasoning models reject one), the reply budget sent as
`max_completion_tokens`, and a stable `prompt_cache_key` so the fixed system prompt is served from the provider's
cache. `OPENAI_API_KEY` (or `IV_ASSIST_API_KEY`) is sent to any non-loopback base URL.

**Alibaba Model Studio** (`openai:<model>` with the DashScope compatible-mode base URL, profile `default`). Any
OpenAI-compatible chat model on that endpoint; set `IV_ASSIST_API_KEY` to your `DASHSCOPE_API_KEY`. We did not hold
the key, so no number of that seat is published. Confirm the model id with your key.

**Claude.** Install `pip install -e ".[anthropic]"` and use `--assistant anthropic:<model-id>`. The Messages API receives the same system prompt, user text, and frames as other API assistants. The existing prompt specifies the JSON reply schema; parsing does not repair model output or substitute another model. Effort is low, matching CookSim, and the token budget includes thinking. `ANTHROPIC_API_KEY` is required; user-scoped credentials may also need `ANTHROPIC_WORKSPACE_ID`.

## 4. The two streaming settings

The streaming APIs (Gemini Live, GPT Realtime) can be run two ways, and the published table carries both because
they measure different things.

**Polled** (`live:<model>`, `realtime:<model>`; tags without a suffix). The harness drives the streaming model
exactly like a chat model: every 2 ticks it opens a session, sends the system prompt, the user text and the images,
and forces a reply; then the session is closed. One session per turn, deliberately: a persistent socket would hand
these seats a conversation memory no other seat has, and any gap that opened up would measure the memory rather than
the model. The handshake latency this costs is charged like any other latency. The model gets no say in *when* to
speak; the setting isolates what the streaming weights see and know from how they manage the floor.

- `live`: the Gemini Live model is audio-out; it speaks its JSON reply and the server's own transcription of what it
  said is parsed. Its JSON compliance is therefore not comparable to a text model's: the taxonomy label has to
  survive being spoken aloud and transcribed back. The Live models differ about thinking levels; the backend adjusts
  once on refusal and keeps the setting for the process.
- `realtime`: one WebSocket session per turn, text output.

**Native** (`live-native:<model>`, `realtime-native:<model>`; tags with `__native`; `vhhome_bench/native.py`). The
model runs in its own streaming mode and nobody polls it. The session is given its instructions once (the same
supervisor prompt, with its turn-based parts replaced by live-stream ones); then the harness only *feeds* it:

- the house as video, one composite image per tick (first-person view beside the house plan, the same picture the
  polled assistant gets eight of), paced to at most one frame per 1.2 s of wall time, and nothing else about the
  house: no state line, no map text;
- the person's speech as audio (one fixed OpenAI TTS voice, the same for every assistant, synthesised ahead of time
  and cached; synthesis is outside every clock);
- a bare `[t=N]` marker with each frame, as context that asks for no reply.

Whatever the model says, whenever it says it, is delivered through the same path as a polled reply, so flags,
grading and records are unchanged. Since the native reply is speech rather than JSON, a mistake call is recognised
by a spoken marker: the prompt asks the model to begin such a sentence with "Flag: <mistake name>." The marker is
matched against the six names, stripped from what the person hears, and a name outside the list still counts as a
flag. Latency is charged from the end of the person's audio (for a reply) or the last frame sent (for a
self-started line) to the model's first output token, then converted to ticks by the ordinary rule. Native
episodes run at wall-clock pace (at least 1.2 s per tick), so they are slower to produce than polled ones. A
session the provider refuses is recorded as a mute episode, never retried.

What the native setting measures is the model's own turn-taking. In the published run the two native seats finish
far fewer chores than their polled counterparts (`won` 0.05 and 0.07 against 0.28 and 0.36) while their
interaction-quality scores are among the highest: when they do speak, they speak well, but they rarely speak from
the video on their own.

## 5. Request profiles

`--profile` selects a request profile in `vhhome_bench/protocol.py` (`_PROFILES`): how the OpenAI-compatible request
is shaped for a given model. A profile sets sampling (temperature, top_p, provider-specific `extra_body` such as
`top_k`, `repetition_penalty`, `chat_template_kwargs`), whether the images go before or after the text, an optional
assistant prefill the server continues, whether every image is letterboxed onto one canvas, and whether the reply is
constrained to the reply schema by guided decoding (`response_format: json_schema`, strict).

| profile | for | request shape |
|---|---|---|
| `default` | any OpenAI-compatible endpoint you have no model-specific guidance for | temperature 0.2, plain JSON |
| `gpt5` | OpenAI reasoning models | no temperature, `reasoning_effort: low`, `prompt_cache_key`, reply budget as `max_completion_tokens` |
| `qwen38-schema` | Qwen3.8-27B | the card's non-thinking sampling (T 0.7, top_p 0.8, top_k 20, presence penalty 1.5); schema |
| `molmo2-schema` | Molmo2-8B | greedy (T 0); schema |
| `glm46v-flash-schema` | GLM-4.6V-Flash | T 0.8, top_p 0.6, top_k 2, repetition penalty 1.1, thinking off; schema |
| `internvideo3-schema` | InternVideo3-8B | T 0.7, top_p 0.8, top_k 20, repetition penalty 1.05; every image letterboxed onto the first frame's canvas (the shim stacks the images into one clip, so the square maps and the wide frames must share a shape); schema |
| `kimi-vl-think-schema` | Kimi-VL-A3B-Thinking | T 0.8, a 2048-token thinking budget, images first; schema |
| `museglimmer-nothink-schema` | Muse Glimmer 30B | T 1.0, top_p 0.95, top_k 64, `reasoning_strength: low`, an empty assistant prefill the server continues; schema |
| `minicpmo-schema` | MiniCPM-o 4.5 | T 0.7, top_p 0.8, top_k 100, repetition penalty 1.05, thinking off, images first; schema |
| `qwen3-omni-schema` | Qwen3-Omni-30B-A3B | T 0.6, top_p 0.95, top_k 20; schema |
| `shim` | MMDuet2, Proact-VL, StreamingVLM | nothing: the shim runs the model's own loop and builds the JSON itself |

`minicpmo` and `museglimmer-nothink` (without `-schema`) are kept as the pre-audit shapes some earlier cells were
run with; they are not the published setting.

To use your own OpenAI-compatible server:

```bash
vhhome-episode --latency 2 --case cx_tidy_missing_rep --persona baseline \
    --assistant openai:my-model --base-url http://127.0.0.1:8000/v1 --profile default --out runs/mine
```

Add a profile to `_PROFILES` if the model's card calls for specific sampling or the reply schema.

## 6. Serving the open-weight assistants

### 6.1 Rules shared by every local model

- **One Python environment per model** at `$VHHOME_MODELS/envs/<name>` and one weights cache at
  `$VHHOME_MODELS/hf_cache/<name>` (`VHHOME_MODELS` defaults to `$HOME`; the older `ROOT` variable is still read).
  Never install a model's stack into the `vhhome` environment: the models pin incompatible versions of torch and
  transformers.
- **One GPU per server, one port per model**, bound to `127.0.0.1`. The scripts read `CUDA_VISIBLE_DEVICES`,
  `PORT`, `VHHOME_MODELS`, `HF_HOME` and `LOGDIR` from the environment and write a pid file under `$LOGDIR`.
- **One episode at a time per server** (`vhhome-benchmark --engines <one port>`; the CLI refuses several ports for a
  local model). The model's latency is charged to the game clock, so two episodes sharing a GPU would each be
  charged the other's compute.
- **All of them are slower than the 1.2 s tick**, and those ticks are charged. The measured per-GPU generation
  times of each model on one synthetic call (8 frames + 2 maps) are in `benchmark/gpu_latency_ratio.json`.
- **12 images per request.** Every vLLM script passes `--limit-mm-per-prompt '{"image": 12}'`: an assistant turn
  carries 8 frames and 2 maps, and a server started with a lower cap answers HTTP 400 on every call.
- The shims in `serving/shims/` are small OpenAI-compatible servers around the authors' own inference code. Three
  of them (`mmduet2`, `proact-vl`, `streamingvlm`) import the authors' repository from `serving/shims/<name>/src`,
  which you clone there (it is not vendored and is git-ignored); `serving/shims/internvideo3/shim_server.py` is a
  plain transformers server; `serving/shims/kimi-vl/kimi_vl_reasoning_parser.py` is a vLLM reasoning-parser plugin,
  not a server.

### 6.2 Starting a model

```bash
export VHHOME_MODELS=$HOME                          # where envs/<name> and hf_cache/<name> live
CUDA_VISIBLE_DEVICES=0 bash serving/serve.sh qwen3.8-27b &
curl -s http://127.0.0.1:18300/v1/models            # wait until it answers (minutes on a network disk)
python -m vhhome_bench.preflight qwen3.8-27b        # one real turn through the harness's own path; warms the server
vhhome-episode --latency 2 --case cx_tidy_missing_rep --persona baseline --assistant qwen3.8-27b --out runs/qwen
```

`serving/serve.sh <tag> [port]` looks the tag up in the roster and runs its `serve_<model>.sh` with that port, logs
under `$LOGDIR` (default `runs/serving/`). Each script's header states what it needs; [serving/README.md](../serving/README.md)
collects them per model.

### 6.3 The models

| Model | Served by | What matters, and why |
|---|---|---|
| **Qwen3.8-27B** | vLLM | `--chat-template-content-format openai`: vLLM otherwise detects the template's content format as "string" and moves every image in front of the text, discarding the order the harness sends. Built-in MTP speculative decoding at depth 5 (`MTP_K`). Thinking off server-wide (`THINK=1` turns it on); the card's non-thinking sampling. The gated-delta-net prefill kernel is pinned to the Triton path (`GDN_BACKEND`) so different GPU generations use the same kernel. |
| **Molmo2-8B** | vLLM | Molmo2's shipped chat template rejects a system message and the benchmark always sends one: `serving/molmo2_chat_template_system.jinja` folds the system text into the first user turn and is byte-identical to the original for conversations without one. Greedy decoding. Half-size images (`IV_ASSIST_IMAGE_SCALE=0.5`). |
| **InternVideo3-8B** | transformers shim | Not a vLLM architecture (its attention was converted to multi-head latent attention, shipped as trust-remote-code only). The images go in as one clip (`FRAMES=video`, 1.2 s apart; `FRAMES=images` sends them separately), which is why the profile letterboxes the maps onto the frame canvas. Flash-attention 2 by default (`ATTN`); the model's own sampling plus `repetition_penalty 1.05`; the reply schema enforced with xgrammar; one request at a time. Run with max tokens 1024. |
| **GLM-4.6V-Flash** | vLLM | Native vLLM support, no remote code. `--reasoning-parser glm45`; with thinking off (the profile) `reasoning_content` stays empty and latency is unchanged, with thinking on the parser keeps the reply schema off the think block. |
| **Kimi-VL-A3B-Thinking-2506** | vLLM | The model thinks between `◁think▷` markers that are several tokens each, which vLLM's built-in parsers cannot match: `serving/shims/kimi-vl/kimi_vl_reasoning_parser.py` splits the thinking into `reasoning_content` and applies the reply schema only after the closing marker. `--reasoning-config` enables the per-request thinking budget the profile sets (2048). Max tokens 8000. |
| **Muse Glimmer 30B** | vLLM | The INT4 (W4A16) checkpoint with the DFlash speculative drafter (`SPEC=1`; `MODEL=meta-models/Muse-Glimmer-30B` serves the official BF16). `--structured-outputs-config {"enable_in_reasoning":true}` is required because the no-think profile pre-fills the answer channel and a schema applied "after reasoning" would never apply. `--reasoning-parser muse_glimmer`, `--generation-config auto`. |
| **MiniCPM-o 4.5** | vLLM | The maintainers' AWQ checkpoint by default (`VARIANT=bf16` serves the original); one slice per image (`MAX_SLICE_NUMS=1`); thinking off. |
| **Qwen3-Omni-30B-A3B** | vLLM | Plain vLLM loads the thinker only and skips the audio talker (we read text), so no audio weights are needed. The Instruct model has no thinking mode. |
| **MMDuet2** | shim | The authors' own demo loop, imported: the frames are replayed as 2-frame user turns, the model answers or says "NO REPLY" after every turn (greedy), and the last turn's output is returned ("NO REPLY" becomes empty text). With `MMDUET2_SYSTEM=official` (the default) the model keeps its own proactive system prompt and the benchmark's system message rides as the question on the first frame turn; `MMDUET2_SYSTEM=bench` puts the benchmark system message in the system turn instead. It cannot write the reply JSON; the shim wraps its text. |
| **Proact-VL** | shim | The released LoRA, FLAG head included, on LiveCC-7B-Base, through the authors' inference code. The frames are replayed as the model's native 1-second chunks; per chunk its own FLAG head decides whether to speak; the reply is what it says on the newest chunks (`PROACT_NEW_CHUNKS=2`). With `PROACT_PROMPT=ego4d` (the default) it keeps the authors' own system prompt, because it recited the benchmark rulebook back; `PROACT_PROMPT=bench` restores the benchmark prompt. JSON wrapped by the shim. |
| **StreamingVLM** | shim | The authors' streaming loop (1-second chunks, attention-sink plus sliding-window KV cache, contiguous-RoPE positions), not vLLM. With `SVLM_PROMPT=native` (the default) the context goes in the loop's "previous text" slot and the rulebook is left out, because that slot is truncated to about a thousand tokens; `SVLM_PROMPT=bench` restores the benchmark system turn. The reply is the commentary of the newest chunks (`SVLM_NEW_CHUNKS=2`). JSON wrapped by the shim. |

Three consequences to keep in mind when reading the results:

- MMDuet2, Proact-VL and StreamingVLM never set a `flag` (their reply is wrapped speech), so flag-based detection
  reads zero for them by construction; judge them on what they say. MMDuet2 said nothing in the published run, so
  it has no interaction-quality score either.
- The same three run with their authors' prompt shape and therefore **do not see the benchmark rulebook** in the
  default setting. That is the faithful way to run them; whether to hold them to the rulebook is a benchmark-design
  choice, and the environment variables above let you make the other one.
- Where a script's flags deviate from a model's published guidance (frame order, sampling, a repetition penalty,
  batched-token settings), the script header records the measurement that justified it.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `the simulator on port N is not answering` or the episode never starts | Start `bash simulator/launch_simulators.sh 1` in another terminal (HTTP 501 on the port means healthy) and pass its port with `--port` / `--engines`. See [SIMULATOR.md](SIMULATOR.md). |
| `unknown assistant '...'` | The tag is not in `vhhome-roster`; use `<backend>:<model>` with `--base-url` / `--profile` for anything else. |
| `<tag>: missing GEMINI_API_KEY` (or another key) | Export it or add a line to `~/.config/vhhome/keys.env`. The person always needs `GEMINI_API_KEY`. |
| `<tag>: roster status 'not_ported'` / `'dropped'` | The seat cannot be run in this release (`molmo2-o-7b`). |
| `a locally served model runs ONE episode at a time` | Give `vhhome-benchmark` a single `--engines` port for a local model. |
| HTTP 400 "At most 8 image(s) may be provided in one prompt" | The server's image cap is below the 10 images a turn carries. Start it from the shipped script (`--limit-mm-per-prompt '{"image": 12}'`) or raise your own server's cap. |
| Every assistant call fails and the cell lands in `<tag>/void/` | Transport failure: the server refused or never answered (retries, no prompt tokens ever returned). The cell is voided rather than recorded as silence, because silence would be data and this is the harness. Check the serve log under `$LOGDIR`, `curl .../v1/models`, then rerun; the voided cell is simply missing and is picked up by the same command. |
| `assistant.stats.schema_fail` is high, the model is read as silent | The model's text yields no JSON object. Use the model's `*-schema` profile (guided decoding) on a vLLM server, or `default` plus a prompt your server honours; a reasoning model must keep its thinking out of `content` (Kimi-VL needs its parser plugin). After three failed attempts a turn is empty, which is silence. |
| The person is frozen for many ticks after each assistant turn | Expected: the reply's latency is charged (`ceil(x / 1.2)` ticks, at most 15). Check `assistant_latency[]` in the record. `ASSIST_LATENCY=0` removes the charge for debugging only. |
| A local model answers without looking at the right frames | Check the rendered prompt order (Qwen3.8's "string" template format, section 6.3). |
| Molmo2 HTTP 400 "Conversation roles must alternate" | The server was started without `molmo2_chat_template_system.jinja`. |
| Kimi-VL replies contain `◁think▷` text or run out of tokens | The reasoning-parser plugin is not loaded, or the profile lacks the thinking budget; use max tokens 8000. |
| InternVideo3 fails with "all input arrays must have the same shape" | The `internvideo3-schema` profile (which letterboxes the maps) is not in use. |
| `gemini-3.8-live` session fails at setup | It refuses a thinking level; the backend drops it automatically and keeps that for the process. |
| Native seat says nothing for a whole episode | Often expected (section 4): the model replies mostly to the person. Check `OPENAI_API_KEY` is set for the person's TTS voice; a refused session is recorded as a mute episode. |
