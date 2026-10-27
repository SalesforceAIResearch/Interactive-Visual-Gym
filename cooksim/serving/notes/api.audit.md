# API assistants vs. their providers' official guidance (2026-09-18)

Scope: request construction and settings of the ten API/streaming assistant seats of the CookSim
benchmark. No engine edits, no episodes. Evidence = the official doc page (URL + section) and, where a
doc left the behaviour open, a single cheap probe call with the real assistant system prompt (12,953
chars, from `tmp/roster_check/.../b3_easy_nops_medium_map_2.json`), a realistic user turn and 8 real
1042x396 JPEG frames (`tmp/stepin_zcfc8a62/00040-00047.jpg`). Probe scripts:
`/tmp/claude-0/.../scratchpad/apiaudit/{gem,oai,rt*,live*}.py`. Latency numbers from 1-3 probe calls
are indicative only (n is tiny); token numbers are exact.

Terms. *Seat* = one way a model is plugged in: `gemini` (stateless GeminiBackend), `openai`
(OpenAIBackend, Chat Completions), `anthropic` (AnthropicBackend), `live` / `realtime` (polled
session: we ask every 2 ticks), `live_native` / `realtime_native` (the model streams and decides when
to speak). *Parity* = every assistant gets the same prompt, frames and output contract (plain JSON, no
schema enforcement), each at its provider's own defaults otherwise.

Code: `cook-bench-engine/cooksim_bench/controller_eval.py` (GeminiBackend L72, PROFILES L124, RealtimeBackend
L184, LiveBackend L266, OpenAIBackend L417), `cooksim_bench/anthropic_backend.py`, `cooksim_bench/assistant_backend.py`,
`cooksim_bench/native_assistant.py`, `cooksim_bench/roster.py` (the flags each roster entry is run with).

## 0. What the frames actually are

`e2e_stepin_rollout.snap()` stores `window.__renderFrameSbs` output = **JPEG** bytes (`ff d8 ff`),
1042x396. Three backends label them `image/png`:

| Backend | Label sent | Provider behaviour | Consequence |
|---|---|---|---|
| GeminiBackend (`Part.from_bytes(..., mime_type="image/png")`) | image/png | accepted (probe) | harmless today |
| OpenAIBackend `_img` (no re-encode at scale 1.0/png) | `data:image/png` | accepted (probe) | harmless today |
| **AnthropicBackend `_img`** (no re-encode at scale 1.0/png) | `media_type: image/png` | **400 "Image does not match the provided media type image/png"** (known API validation; see anthropics/claude-code issues #11936, #12009) | **every Claude call fails -> BadRequest is not retried -> AssistantBackend returns {} -> Claude is silent for the whole episode** |

Per-frame token cost at each provider's default (exact, from usage):

| Model | Setting | Tokens / frame | 8 frames |
|---|---|---|---|
| gemini-3.8-flash / 3.7-flash / 3.1-pro (stateless) | media_resolution unspecified (=high) | 1,080 | 8,640 |
| same | MEDIUM | 532 | 4,256 |
| same | LOW | 270 | 2,160 |
| gpt-6-astra, gpt-5.5 | detail auto (33x13 patches x1.2) | ~515 | ~4,120 |
| same | detail low (512 px cap) | ~135 | ~1,080 |
| gpt-realtime-2.1 | detail auto | 452 | (2 per ask) |
| same | detail low | 65 | |
| claude-opus-5 (computed: ceil(1042/28) x ceil(396/28)) | high-res tier, no resize | 570 | 4,560 |
| gemini-3.8-live (video via send_realtime_input, or inline) | default | 270 | |
| gemini-3.1-flash-live-preview | video via send_realtime_input | 65 | |
| gemini-3.1-flash-live-preview | inline image in client turn, default / MEDIUM / LOW | 1,080 / 540 / 65 | |

Past runs (tmp/sample_r6*, tmp/assist_bench_cookfix_pass3, ~30 episodes each), per call:

| Assistant | median latency | prompt tokens / call (median, max) | completion | thinking |
|---|---|---|---|---|
| gemini-3.8-flash | 3.6 s | 12.3k, 13.0k | 175 | 55 |
| gemini-3.1-pro-preview | 9.6 s | 12.2k, 13.0k | 280 | 300 |
| gpt-6-astra | 6.4 s | 7.8k, 8.3k | 280 | n/r |
| gpt-realtime-2.1 (polled) | 5.0 s | **47.3k, 133.9k** | 331 | n/r |
| gemini-3.1-flash-live-preview (polled) | 1.1 s | 12.6k, 15.2k | 0 (not captured) | 0 |

---

## 1. gemini-3.8-flash / gemini-3.7-flash / gemini-3.1-pro-preview — seat `gemini` (GeminiBackend)

Sources: G3 = https://ai.google.dev/gemini-api/docs/gemini-3 ; TH = https://ai.google.dev/gemini-api/docs/thinking ;
MR = https://ai.google.dev/gemini-api/docs/media-resolution ; IU = https://ai.google.dev/gemini-api/docs/image-understanding ;
SO = https://ai.google.dev/gemini-api/docs/structured-output ; CA = https://ai.google.dev/gemini-api/docs/caching ;
MO = https://ai.google.dev/gemini-api/docs/models

| # | Item | Ours | Official | Source | Verdict |
|---|---|---|---|---|---|
| 1 | API surface | `models.generate_content`, google-genai 1.64 | same | G3 | matches |
| 1b | System prompt | passed as the first **content part** (`contents=[system, user, *images]`) | `config.system_instruction` | G3 / text-generation docs "System instructions" | **should change** (small; also the documented place for a fixed prefix) |
| 2 | Thinking control | `ThinkingConfig(thinking_budget=640)` | Gemini 3: `thinking_level`; "you cannot use both"; budget kept only for backward compatibility; 3.8/3.7-flash levels low/medium/high (default **medium**, no `minimal`); 3.1-pro low/medium/high (default **high**) | G3 "Thinking level"; TH table "Thinking levels by model" | **should change** to `thinking_level="low"` (probe: budget 640 accepted and gives 0-390 thought tokens, level low 144-546; latency indistinguishable at n=3: 3.8-flash budget 1.8/3.6/2.2 s vs low 3.9/3.2/3.0 s; pro 7.5/12.8 vs 8.7/10.6 s). Official knob, same name as the other seats' "low". Note `minimal` is rejected on 3.8-flash (probe: 400 "Thinking level MINIMAL is not supported") |
| 2b | Temperature | 0.2 | "For all Gemini 3 models, we strongly recommend keeping the temperature parameter at its default value of 1.0"; lower values "may lead to unexpected behavior, such as looping or degraded performance" | G3 "Temperature" | **should change** (drop the field). Also the parity-preserving choice: GPT-5.x/6 and Claude Opus 5 reject temperature, so they already run at provider default |
| 3 | Image resolution | not set = 1,080 tok/frame (8.6k of the 12.1k prompt) | Images: `high` (1120) "for most analysis tasks"; video: low/medium (70/frame) | MR "Choosing the right resolution" | matches the image recommendation. Option: MEDIUM (532/frame, prompt 12.1k->7.7k) puts Gemini at the same tokens/frame as GPT (~515) and Claude (570); latency gain not measurable at n=3. Leave unless a re-baseline is planned |
| 3b | MIME label | `image/png` for JPEG bytes | JPEG is a supported type; label should describe the bytes | IU "Supported formats" | should change (harmless today, one-line fix) |
| 3c | Order | text then images | "place the text prompt before the image" (single image with text) | IU "Prompt placement" | matches |
| 4 | JSON output | `response_mime_type="application/json"`, no schema | structured output = mime + `response_json_schema` | SO | deviates on purpose (parity: no API assistant gets schema enforcement; Live/Realtime cannot). Parity-preserving option if ever wanted: schema for all three stateless providers (Gemini `response_json_schema`, OpenAI `json_schema strict`, Claude `output_config.format`) — Live/Realtime would still lack it |
| 5 | Sampling / rejected params | temperature 0.2, max_output_tokens 6000 | temp 1.0; nothing else rejected (probe) | G3 | see 2b |
| 6 | Caching | implicit (automatic) | implicit caching needs >= **4,096** tokens of common prefix on 3.8/3.7 Flash and 3.1 Pro | CA "Implicit caching" | not reachable: the fixed prefix is the ~3.4k-token system prompt, the user turn changes every call and the frame window slides. Nothing to do |
| 6b | Model status | 3.8-flash stable, 3.7-flash stable (previous gen), 3.1-pro-preview preview | — | MO | ok |
| 6c | Vertex path | `genai_client.resolve_model` maps any id containing "flash" (not "3-flash") to `gemini-3-flash-preview` when GOOGLE_GENAI_USE_VERTEXAI=true | — | — | GeminiBackend builds its own API-key client, so the assistant seat is unaffected; the COOK seat (GEMINI_MODEL=gemini-3.7-flash) would be silently swapped on Vertex. Note only |

## 2. gpt-6-astra and gpt-5.5 — seat `openai`, profile `gpt5` (OpenAIBackend, Chat Completions)

Sources: RS = https://developers.openai.com/api/docs/guides/reasoning ; LM = https://developers.openai.com/api/docs/guides/latest-model ;
MA = https://developers.openai.com/api/docs/models/gpt-6-astra ; M55 = https://developers.openai.com/api/docs/models/gpt-5.5 ;
IV = https://developers.openai.com/api/docs/guides/images-vision ; PC = https://developers.openai.com/api/docs/guides/prompt-caching ;
FM = https://developers.openai.com/api/docs/guides/priority-processing (Fast mode) ; LO = https://developers.openai.com/api/docs/guides/latency-optimization

Probe (same request, 8 frames): 

| Model | Variant | latency (s) | prompt | cached | output | reasoning |
|---|---|---|---|---|---|---|
| gpt-6-astra | chat, effort low (current) | 7.4, 8.6, 6.2, 6.0 | 7,446 | 0/7296/7296/0 | 272-300 | 31-56 |
| gpt-6-astra | Responses, effort low | 5.9, 5.5, 5.6 | 7,454 | 0/7451/7451 | 216-260 | 22-39 |
| gpt-6-astra | chat, service_tier priority (served "fast") | 4.5 | | | 201 | 27 |
| gpt-6-astra | Responses, priority | 5.0 | | | 234 | 58 |
| gpt-6-astra | chat, detail low | 5.1 | 4,406 | | 147 | 18 |
| gpt-6-astra | chat, verbosity low | 5.1 | | | 196 | 21 |
| gpt-6-astra | effort minimal / none | **400** "Supported values are: 'low', 'medium', 'high', and 'xhigh'" | | | | |
| gpt-5.5 | chat, effort low (current) | 8.0, 10.2 | 7,446 | 0, 0 | 481-538 | 300-409 |
| gpt-5.5 | Responses, effort low | 5.9, 6.6 | 7,454 | 0, 6400 | 330-419 | 193-295 |
| gpt-5.5 | chat, effort none | 3.0 | | | 179 | 0 |
| gpt-5.5 | chat, priority | 8.4 | | | 554 | 395 |
| gpt-5.5 | chat, none + prompt_cache_key | 2.7, 2.9, 3.1 | 7,446 | 0, 6400, 6400 | | |

| # | Item | Ours | Official | Source | Verdict |
|---|---|---|---|---|---|
| 1 | API surface | `/v1/chat/completions` | "Reasoning models work better with the Responses API. While the Chat Completions API is still supported, you'll get improved model intelligence and performance by using Responses." GPT-6 Astra: "Set model to gpt-6-astra in a Responses API request"; both endpoints listed as supported | RS "Recommended API"; LM; MA "Endpoints"; M55 | **should change**: add a Responses path. Probe: gpt-5.5 used 25-35% fewer reasoning tokens and was 2-4 s faster; astra ~0.5 s faster. Same inputs and the same json_object output -> parity unaffected |
| 2 | Reasoning | `reasoning_effort="low"` both models | astra: low/medium/high/xhigh/max, **no none/minimal**; gpt-5.5: none/low/medium(default)/high/xhigh; "none" for "latency-critical tasks that do not benefit from any reasoning", "low is ideal for ... planning ... multi-step decision making, while optimizing for speed" | RS "Reasoning effort"; MA; M55 | matches (low is astra's floor and the documented latency setting for decision tasks; keeps the "low" row across providers). gpt-5.5 at `none` is 3 s instead of 8-10 s — a legitimate ablation, not a parity default |
| 2b | Reply budget | `max_completion_tokens=4000` (e2e default) | reserve >= 25k for reasoning + output "when you start experimenting"; truncation -> incomplete | RS "Token management" | deviates on purpose, low risk: max observed reasoning+output 950 tokens at low. Optional: 8000 (only generated tokens are billed) |
| 3 | Image detail | not set = `auto` (~515 tok/frame) | default auto in both APIs; `high` standard fidelity; `low` coarse (512 px cap); `original` for dense/coordinate work | IV "The detail parameter", "Patch-based tokenization" | matches. `low` would downscale 1042 -> 512 px and blur the two views — not recommended |
| 3b | MIME label | `data:image/png` for JPEG | — | — | accepted; fix with the shared sniff (one line) |
| 4 | JSON output | `response_format={"type":"json_object"}` | Structured Outputs (`json_schema`, strict) supported on both models | MA "Key features"; M55 | deviates on purpose (parity). In Responses: `text.format={"type":"json_object"}` (verified accepted) |
| 5 | Sampling / rejected | temperature omitted (profile `temperature=None`) | "Remove temperature, top_p, top_logprobs" (astra); not available with reasoning models | LM "Unsupported parameters"; RS | matches |
| 6a | Prompt caching | automatic, no key | caching automatic >= 1,024 tokens; `prompt_cache_key`: "use a stable key for requests that share a reusable prefix to help route them to the same cache" (pre-5.6 models); 5.6+: accounting key | PC "Minimum cacheable length", "Purpose of prompt_cache_key" | **should change**: add a stable `prompt_cache_key`. Probe: without key the identical request hit cache 2/4 (astra) and 0/2 (gpt-5.5 chat); with key 2/2 after the first. Our layout (system first, then the user text, frames last) already follows "put stable instructions first" |
| 6b | Service tier | default | Fast mode (`service_tier: "priority"`/`"fast"`): "up to 2.5x faster", 2x price; "Fast mode for GPT-6 Astra does not include a latency SLA" | FM; LM | deviates on purpose — keep standard unless every provider gets its fast tier (Claude Opus 5 `speed:"fast"`); one astra probe 4.5 s vs ~6 s, gpt-5.5 no gain |
| 6c | Streaming | off | streaming is the top UX latency lever | LO "Make your users wait less" | not applicable: the harness needs the whole JSON before acting; streaming does not shorten time-to-complete |
| 6d | Verbosity | not set | `verbosity` exists (low shortens output) | LM | leave: changes the answer, not the plumbing (probe: 196 vs 272-300 output tokens) |

## 3. gpt-realtime-2.1 — seat `realtime` (RealtimeBackend, polled) and `realtime_native` (RealtimeNativeSession)

Sources: MR21 = https://developers.openai.com/api/docs/models/gpt-realtime-2.1 ; CE = https://developers.openai.com/api/reference/resources/realtime/client-events (session.update) ;
RC = https://developers.openai.com/api/docs/guides/realtime-conversations ; VAD = https://developers.openai.com/api/docs/guides/realtime-vad ;
VC = https://developers.openai.com/api/docs/guides/voice-latency-cost ; ANN = https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/

Probe: `session.created` defaults = reasoning null, max_output_tokens "inf", truncation "auto",
server_vad (threshold 0.5, prefix 300 ms, silence 500 ms). `reasoning:{effort:"low"|"minimal"}`,
`max_output_tokens:1500` and `truncation:{type:"retention_ratio", retention_ratio:0.5,
token_limits:{post_instructions:16000}}` are all accepted by session.update. Default effort, first ask:
242 reasoning tokens / 4.6 s; `minimal`: 56 / 3.1 s. The text channel has **no JSON mode**; 14 probe
replies: 11 fenced ```json (parsed fine by `_parse_json`), 2 with junk before the brace (`>{`, `({`,
parsed fine), **1 without the opening brace** (`"cook_progress": {...` -> `_parse_json` returns the
inner cook_progress object, so text and flag are lost).

| # | Item | Ours | Official | Source | Verdict |
|---|---|---|---|---|---|
| 1 | API surface | WebSocket `/v1/realtime`, GA session shape, `output_modalities:["text"]` | text-only output via output_modalities; WebRTC recommended for voice apps | RC "Text-only sessions", "Latency" | matches (server-to-server; WebRTC is for browsers) |
| 2 | Reasoning | not set (server default) | `reasoning.effort` minimal/low/medium/high/xhigh, "low as the default" | CE; ANN | matches the default; **pin `low` explicitly** (a default can move; same row as the others) |
| 3 | Image input | `input_image`, detail not set = auto (452 tok/frame); polled seat sends the 2 newest frames per ask | `detail` auto/low/high, default auto | CE "Conversation item content" | matches |
| 4 | JSON output | prompt-only JSON, `_parse_json` | no response_format in Realtime (tools are the only structured channel) | CE | deviates by necessity. **Should change (harness-wide, parity-neutral)**: make `_parse_json` recover a reply whose opening brace is missing (see change R4) |
| 5 | Output cap | `max_output_tokens` inf | 1-4096 or inf | CE | should change: 4000, the same cap the stateless seats get |
| 6 | Context (polled) | one session per episode, every ask appended, truncation "auto" | "When the number of tokens in a conversation exceeds the model's input token limit the conversation [will] be truncated"; retention_ratio "allows you to amortize truncations across multiple turns and help improve cached token usage" | VC "Truncation"; CE | **should change**: past runs show 47k median / 134k max prompt tokens per call (the 128k context fills; latency 5.0 s median). Every ask already carries the full state (prev JSON + conversation), so the old turns are redundant. `retention_ratio 0.5` + `post_instructions 16000` mirrors the Live seat's 16k->8k compression (parity between the two polled session seats) |
| 6b | Caching | automatic | automatic in Realtime; keep history static | VC "Prompt caching" | matches (probe: 4.2k-5.6k cached per ask); retention_ratio keeps it working after a truncation |
| 7a | Native: turn detection | `semantic_vad`, eagerness auto, create_response true, interrupt_response true | semantic_vad eagerness low/medium/high/auto (auto = medium); create/interrupt used in speech-to-speech | VAD | matches |
| 7b | Native: audio format | `audio/pcm` 24 kHz | pcm16 24 kHz | CE | matches |
| 7c | Native: context | frame items pruned to KEEP_FRAMES=16, truncation auto, no reasoning pin | as above | CE; VC | should change: pin `reasoning.effort:"low"` and add retention_ratio truncation (e.g. post_instructions 32000, mirroring the native Live 32k->16k) so a long episode never hits the 128k wall mid-sentence |

## 4. gemini-3.8-live and gemini-3.1-flash-live-preview — seat `live` (LiveBackend, polled) and `live_native` (LiveNativeSession)

Sources: LG = https://ai.google.dev/gemini-api/docs/live-guide ; LS = https://ai.google.dev/gemini-api/docs/live-session ;
LV = https://ai.google.dev/gemini-api/docs/live ("Technical specifications") ; MO ; MR

**P0 finding (polled `live` seat): the model answers before it has the frames.** LiveBackend pushes the
new frames with `send_realtime_input(video=...)` and immediately sends the question with
`send_client_content(turn_complete=True)`. The docs warn realtime input is "optimized for
responsiveness at the expense of deterministic ordering" (LG "Send methods"). Probe, same call pattern,
question "what is in the right half of the most recent image, or say NO IMAGE":

| Model | How frames go in | ask 1 | ask 2 |
|---|---|---|---|
| gemini-3.8-live | realtime video, no wait (= LiveBackend) | **"NO IMAGE."**, 0 image tokens | 1 frame (270 tok) of the 4 sent so far |
| gemini-3.8-live | realtime video + 2 s wait | 1 frame | 2 of 4 frames |
| gemini-3.8-live | **inline parts in the client turn** | 2 frames (540 tok), correct | 4 frames, correct |
| gemini-3.1-flash-live-preview | realtime video, no wait (= LiveBackend) | **"NO IMAGE."**, 0 image tokens | 1 frame (65 tok) |
| gemini-3.1-flash-live-preview | inline parts in the client turn | 2 frames (2,160 tok), correct | 4 frames, correct |

So the polled Live seat sees at most ~1 frame per ask, one ask late; its 1.1 s median latency in past
runs is partly the latency of a model that has not looked. (Realtime video is also documented at
<= 1 frame/s, LV; we push 2 frames in one burst.)

| # | Item | Ours | Official | Source | Verdict |
|---|---|---|---|---|---|
| 1 | Model choice | 3.8-live (stable) and 3.1-flash-live-preview | 3.1-flash-live-preview: "Legacy audio-to-audio preview model. We recommend updating to Gemini 3.8 Live" | MO | ok as a roster row; label it legacy |
| 1b | Response modality | AUDIO + output_audio_transcription, JSON read from the transcript | native-audio models support only AUDIO; use output transcription for text | LG "Response modalities" | matches (by necessity). Parity note: latency is measured to turn_complete, i.e. until the whole JSON has been *spoken* |
| 2 | Thinking | polled + native: start `minimal`, auto-drop on refusal | 3.8 Live standard: omit thinkingLevel; 3.8 Live Extended Thinking: low/medium/high | LG "Thinking configuration" | matches (probe: 3.8-live refuses any level -> dropped; 3.1-flash-live accepts minimal) |
| 3 | Frames, polled | `send_realtime_input(video)` then immediate ask | video as images "max 1 frame per second"; `send_client_content` for turn-by-turn and context, deterministic | LV "Technical specifications"; LG "Send methods" | **should change (P0)**: put the new frames as `inline_data` parts in the same `send_client_content` turn |
| 3b | Media resolution | not set | `mediaResolution` in session setup | LG "Media resolution"; MR | with inline frames **set MEDIUM** (probe: inline cost 3.8-live 270/frame at default and MEDIUM; 3.1-flash-live 1,080 default -> 540 MEDIUM -> 65 LOW), keeping tokens/frame near the other seats (~450-570) and the 16k compression window from churning every 3 asks |
| 3c | Frames, native | 1 frame/tick via `send_realtime_input(video)`, paced to >= 1.2 s | <= 1 fps | LV | matches |
| 4 | JSON | prompt-only (no JSON mode in Live) | — | LG | deviates by necessity (as Realtime) |
| 6 | Context | polled: compression trigger 16k -> 8k; native: 32k -> 16k + session resumption + GoAway reconnect | audio+video sessions limited to 2 min without compression; connection ~10 min; resumption handles valid 2 h; GoAway carries timeLeft | LS | matches. With inline MEDIUM frames on 3.1-flash-live a 2-frame ask is ~1.5k tokens; consider 32k -> 16k for the polled seat too (same as native) |
| 7a | Native: proactivity | `ProactivityConfig(proactive_audio=True)` on API version **v1alpha** | docs say "with API version v1beta" | LG "Proactive audio" | deviates on purpose, verified: v1beta rejects the field ("Unknown name \"proactivity\""), v1alpha opens. Keep v1alpha |
| 7b | Native: VAD | automatic, default sensitivities, 1.0 s trailing silence after each TTS line | automatic VAD; silenceDurationMs recommended 500-800 ms | LG "Voice Activity Detection" | matches (the 1.0 s pad exceeds the server's silence window, so the turn always closes) |
| 7c | Native: audio | PCM 24 kHz, `audio/pcm;rate=24000` | input "raw 16-bit PCM audio, 16kHz"; rate declared in the MIME | LV | acceptable (rate declared); 16 kHz would match the documented native rate exactly — optional |
| 7d | Native: media resolution | not set (270 / 65 tok per video frame) | video: low/medium recommended | MR | matches |

## 5. claude-opus-5 — seat `anthropic` (AnthropicBackend), code-only (no key on this machine)

Sources: VI = https://platform.claude.com/docs/en/build-with-claude/vision ; plus the bundled claude-api
skill reference (Thinking & Effort, Prompt Caching, Fast Mode, Common Pitfalls), which mirrors
https://platform.claude.com/docs/en/build-with-claude/{effort,adaptive-thinking,prompt-caching,fast-mode}.

| # | Item | Ours | Official | Source | Verdict |
|---|---|---|---|---|---|
| 1 | API surface | official SDK, `messages.create` | Messages API | skill "Architecture" | matches |
| 2 | Thinking / effort | adaptive by default (thinking param omitted), `output_config.effort="low"` | Opus 5 runs adaptive when `thinking` is omitted; effort low..max, default high; `low` for "latency-sensitive routes"; disabling thinking on Opus 5 has two failure modes — "prefer low/medium effort instead" | skill "Thinking & Effort", "Common Pitfalls" | matches |
| 3 | Image input | base64, **labelled image/png for JPEG bytes** | supported JPEG/PNG/GIF/WebP; the base64 data must match `media_type` (400 otherwise) | VI "Supported formats"; claude-code issues #11936/#12009 | **should change (P0)**: every call would 400 and the seat would be silent |
| 3b | Resolution | 1042x396, no resize -> 570 visual tokens/frame | high-res tier (Opus 5): 2576 px long edge / 4784 tokens; cost = ceil(w/28) x ceil(h/28) | VI "Resolution and token cost" | matches |
| 3c | Order | images first, then the text | "Claude works best when images come before text" | VI tip | matches |
| 4 | JSON | plain JSON parsed by `_parse_json`, no `output_config.format` | structured outputs via `output_config.format`; prefill rejected on Opus 5 | skill "Structured outputs", "Prefill removed" | deviates on purpose (parity) |
| 5 | Sampling | none sent | temperature/top_p/top_k rejected (400) on Opus 5 | skill table | matches |
| 5b | max_tokens | max(4000, 16000) = 16000, non-streaming | ~16000 default for non-streaming | skill "max_tokens defaults" | matches |
| 6a | Caching | `cache_control: ephemeral` on the system block | min cacheable prefix on Opus 5 = 512 tokens; render order tools -> system -> messages | skill "Prompt Caching" | matches (system ~3.4k tokens caches) |
| 6b | Tier | standard | Fast mode (`speed:"fast"`, beta header, 2x price, up to 2.5x output tok/s); Priority Tier NOT supported on Opus 5 | skill "Fast Mode" | deviates on purpose (parity with standard-tier GPT/Gemini) |
| 6c | Refusal fallback | none; refusal = silent reply, recorded | skill default is `fallbacks:"default"` | skill "refusal stop reason" | deviates on purpose (a benchmark reply must come from the model under test) |

## 6. Parity summary after the recommended changes

Every stateless seat: provider-default sampling, its lowest documented "low" reasoning setting
(Gemini thinking_level low, GPT reasoning low, Claude effort low), plain JSON mode without schema,
frames at the provider's default image resolution (~450-1,080 tokens/frame), a stable cached prefix.
Both polled session seats: the 2 new frames each ask, correctly ordered before the question, and a
bounded context (16k -> 8k). Nothing recommended here gives one assistant help another lacks; the
changes that alter model behaviour (Gemini temperature/thinking/system_instruction, OpenAI Responses,
Live inline frames, Realtime truncation) need a fresh baseline for those rows — results from before
and after must not be mixed.
