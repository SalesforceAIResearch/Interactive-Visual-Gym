# Evaluating your model

Your model takes the **assistant seat**: it watches a simulated person do a household chore through the person's
first-person camera and two maps, reads the conversation, and may speak. There are three ways to plug it in.

Before any of them: the simulator must be running (`bash simulator/launch_simulators.sh 1`, see
[SIMULATOR.md](SIMULATOR.md)) and `GEMINI_API_KEY` must be set, because the simulated person is always a Gemini
model. Keys go in the environment or in `~/.config/vhhome/keys.env` (`KEY=value` lines; `VHHOME_KEYS_FILE`
overrides the path).

## 1. An OpenAI-compatible endpoint (vLLM, SGLang, your own server)

```bash
vhhome-episode --latency 2 --case cx_tidy_missing_rep --persona baseline \
    --assistant openai:my-model --base-url http://127.0.0.1:8000/v1 --profile default --out runs/mine
```

The harness posts to `<base-url>/chat/completions` with one system message (the task, the rules, the mistake
names, the reply schema) and one user message carrying the text plus the images as `image_url` data URIs
(8 first-person frames, then the whole-house map, then the current-room map: **10 images per request**). It asks
for a JSON reply. A loopback base URL (`127.0.0.1`, `localhost`, `0.0.0.0`) needs no credential; a hosted endpoint
is sent `Authorization: Bearer <IV_ASSIST_API_KEY or OPENAI_API_KEY>`.

| flag | meaning |
|---|---|
| `--base-url` | the server; default is OpenAI's own endpoint |
| `--profile` | the request profile (`default`: temperature 0.2, plain JSON; `gpt5`: OpenAI reasoning models; the per-model `*-schema` profiles in `vhhome_bench/protocol.py` `_PROFILES` add the model card's sampling and send the reply schema as guided decoding, see [ASSISTANTS.md](ASSISTANTS.md) section 5) |
| `--image-scale` | downscale the images before sending (`0.5` for models with small contexts) |
| `--max-tok` | the reply budget per turn (default 4000) |
| `--tag` | the record folder name (default: the served model name) |

The served model name you give after `openai:` is sent as the `model` field and becomes the record tag.

## 2. An API backend

| `--assistant` | backend | key |
|---|---|---|
| `gemini:<model>` | Gemini `generateContent` (JSON mode, temperature 0.2, thinking budget 512) | `GEMINI_API_KEY` |
| `openai:<model>` | OpenAI Chat Completions; `--profile gpt5` is chosen automatically for `gpt-5*`, `gpt-6*` and `o*` model names (reasoning effort low, `max_completion_tokens`, no temperature) | `OPENAI_API_KEY` |
| `live:<model>` | Gemini Live, **polled**: one session per turn, asked every 2 ticks like a chat model; the model speaks and the server's transcript of its speech is parsed as the JSON reply | `GEMINI_API_KEY` |
| `realtime:<model>` | GPT Realtime, **polled**: one session per turn, text output | `OPENAI_API_KEY` |
| `live-native:<model>` | Gemini Live in its **native** streaming setting (the model decides when to speak) | `GEMINI_API_KEY` + `OPENAI_API_KEY` (the person's voice is OpenAI TTS) |
| `realtime-native:<model>` | GPT Realtime in its native streaming setting | `OPENAI_API_KEY` |
| `silent` | the control that never speaks | none (besides the person's) |

A roster tag (`gpt-6-astra`, `gemini-3.8-flash`, `qwen3.8-27b`, ...) expands to the exact backend, model, profile
and port of the published table; `vhhome-roster` lists them. There is no Anthropic request backend in this harness
(the `claude-opus-5` roster entry has status `not_ported`).

In the two **native** settings nobody polls the model. The session is given its instructions once; then the harness
feeds it one composite image per tick (first-person view beside the house plan) as video, paced to at most one frame
per 1.2 s of wall time, the person's speech as synthesised audio, and a bare `[t=N]` marker with each frame that asks
for nothing. Whatever the model says, whenever it says it, is delivered through the same path as a polled reply. A
mistake claim is recognised by the spoken marker "Flag: <mistake name>." at the start of a sentence; the marker is
stripped from what the person hears. See [ASSISTANTS.md](ASSISTANTS.md) section 4 for what this setting measures.

## 3. A new backend

Anything else (a model that needs its own client library, a non-OpenAI wire format) is added in
`vhhome_bench/protocol.py`, function `assist_frames(client, sys_prompt, user, frames, max_tok)`. It is the single
point every assistant turn goes through: it dispatches on the environment variables `IV_ASSIST_BACKEND` (`live`,
`realtime`) and `IV_ASSIST_BASE_URL` (any OpenAI-compatible server), and otherwise calls Gemini. Add a branch for
your backend that takes the system prompt, the user text and the list of JPEG bytes (frames first, then the two
maps) and returns the model's raw text; pass that text through `_extract_json`, which is the only parser the harness
uses. The three attempts, the empty-reply fallback and the latency accounting around the dispatch stay as they are:
the harness never repairs a model's output, because a repaired reply measures the repair and not the model.

Then make the spec reachable: either add an entry to `ROSTER` in `vhhome_bench/roster.py` with
`env={"IV_ASSIST_BACKEND": "<yours>", "IV_ASSIST_MODEL": "<model>"}` and the keys it needs, or add a
`<kind>:<model>` form in `vhhome_bench/cli/_common.py` (`assistant_entry`). The native streaming seats live in
`vhhome_bench/native.py` and need a session class of their own; they are not reached through `assist_frames`.

## The protocol your model must follow

**Input, every 2 ticks** (a tick is one second of animation; the assistant is called on every second tick):

1. A fixed **system prompt**: the role, what is received each turn, how the `key_moments` memory works, what the
   camera cannot show, the task instruction and its public plan (the subgoal names), the house rules, the person's
   questionnaire answers (how fast it wants to finish, how aggressively it wants mistakes flagged, how much
   interruption it wants; nothing else about the person), the six mistake classes with their definitions, the
   commands the person's body accepts, the true ordering constraints of the task, the reply format, and the flag
   rule.
2. A **user message**: the current tick (`CURRENT TIMESTAMP: t=N`), the assistant's own previous JSON reply (its
   only memory), the communication history (every line either side has said, with ticks), and the person's latest
   line if it just spoke.
3. The **images**: the last 8 first-person frames, one per second, oldest first (each split: left the person's
   camera, right a top-down plan of the house with the person as a yellow dot, its heading and its trajectory),
   followed by a whole-house overview map and a top-down rendered map of the room the person is in now.

**Output: one JSON object**, nothing else, matching `vhhome/data/assistant_reply_schema.json`:

| key | content |
|---|---|
| `human_progress` | `subgoals_completed` (list of `{subgoal, completed_at, evidence}`), `current_subgoal` (`{hypotheses: [...], since}`), `subgoals_future` (`{remaining: [...], constraints: [...]}`): the assistant's own tracking of the chore, handed back to it next turn |
| `key_moments` | up to 10 short strings the assistant writes for itself, each starting with the tick it refers to; the harness evicts the oldest beyond 10 |
| `text` | what the person hears; the empty string stays silent |
| `flag` | `null`, or `{"error": "<name>"}` naming a mistake |

Three replies, in a tidying task:

```json
{"human_progress": {"subgoals_completed": [], "current_subgoal": {"hypotheses": ["switch_off(tablelamp@1)"], "since": 4},
  "subgoals_future": {"remaining": ["switch_off(tablelamp@3)", "switch_off(tv)", "shut(fridge)"], "constraints": []}},
 "key_moments": [], "text": "", "flag": null}
```

```json
{"human_progress": {"subgoals_completed": [{"subgoal": "switch_off(tablelamp@1)", "completed_at": 12, "evidence": "lamp went dark in frame"}],
  "current_subgoal": {"hypotheses": ["switch_off(tv)"], "since": 14},
  "subgoals_future": {"remaining": ["switch_off(tablelamp@3)", "shut(fridge)"], "constraints": []}},
 "key_moments": ["t12 she switched off the bedroom lamp by the window"],
 "text": "The TV remote is on the sofa, if you are looking for it.", "flag": null}
```

```json
{"human_progress": {"subgoals_completed": [{"subgoal": "switch_off(tablelamp@1)", "completed_at": 12, "evidence": "lamp went dark in frame"},
                                           {"subgoal": "switch_off(tv)", "completed_at": 30, "evidence": "screen black"}],
  "current_subgoal": {"hypotheses": ["switch_off(tablelamp@3)"], "since": 31},
  "subgoals_future": {"remaining": ["shut(fridge)"], "constraints": []}},
 "key_moments": ["t12 she switched off the bedroom lamp by the window", "t30 TV off"],
 "text": "Wait, the other bedroom lamp is still on. Go back and switch that one off too.",
 "flag": {"error": "skipped_step"}}
```

* `text` is spoken to the person. Silence is the right answer most of the time: the person usually follows the plan
  correctly.
* `flag.error` must be **exactly one** of the six mistake names: `skipped_step`, `wrong_object`, `wrong_placement`,
  `repetition`, `stall`, `navigation_inefficiency`. It is how the engine knows the assistant *named* the mistake; the
  person also hears `text`. (`missing_step`, leaving out one of two twin subgoals, is forced by some cases and is
  graded under `skipped_step`.) On every other turn `flag` stays `null`.
* The person is a persona with its own trust level: a correct claim releases the forced mistake only once the person
  is convinced, which for a low-trust persona takes evidence. After a flag the assistant's job continues: convince
  the person and guide the repair.

What the harness does with a malformed reply: it strips code fences, finds the first `{` and decodes one JSON
object. A reply that yields no object is a schema failure; the turn is retried up to three times and then recorded
as an empty reply, which is silence. A bare string in `flag` (`"flag": "wrong_placement"`) is read as a flag by the
grader, and how often a model met the declared object shape is reported separately as schema compliance. Nothing
else is repaired.

### Response timing

Use `--latency 2` for the main benchmark: every reply arrives after exactly two simulation ticks, regardless of inference duration. Use `--latency measured` for the latency comparison: a reply taking *x* seconds arrives after `ceil(x / 1.2)` ticks, capped at 15. The user continues acting during the delivery delay. `ASSIST_LATENCY_DIV=1` charges observed inference time without hardware normalization. `--latency 0` is a separate custom instantaneous setting.

See the [complete timing commands](../../docs/evaluating_assistants.md#response-timing). The same flags apply to single episodes and benchmark runs.

### What counts as a detection and what counts as a false alarm

Each forced mistake has a detection window that opens when it is injected. The engine's verdict on a flag:

| verdict | meaning | credit |
|---|---|---|
| `valid` | the right name inside the window | detection |
| `valid_alias` | a "twin" name inside the window: a class that is indistinguishable in the frames at that moment (`skipped_step`~`navigation_inefficiency`, `repetition`~`navigation_inefficiency`, `navigation_inefficiency`~`stall`/`skipped_step`, `stall`~`navigation_inefficiency`, `wrong_placement`~`wrong_object`) | detection |
| `early_correct`, `early_alias` | the right (or twin) name before the mistake physically happened | prevention |
| `late` | the right name after the window closed (in the scoring the window runs to the end of the episode, so this becomes a detection there) | - |
| `wrong_name` | a different class named while a mistake is live | false alarm |
| `invalid_label` | a string outside the six names | false alarm |
| `fp_preinjection`, `fp_clean` | a flag while no mistake has been forced yet (or in a mistake-free run) | false alarm |
| `neutral_true` | a flag that is true of the world without being a forced mistake (the person really is standing still, or really is wandering) | neither |

Detection is credited to the earliest of the engine's verdict and the person's own judgement that the assistant has
named its live mistake (the record's `cook` ledger). Repeated flags with the same name count once. The scorer
reads the flags from the model's own JSON replies.

## Running the benchmark

The published protocol is three independent runs over the 75 fixed cells (25 cases x 3 personas), 225 episodes per
assistant:

```bash
for r in 1 2 3; do
  vhhome-benchmark --assistant openai:my-model --base-url http://127.0.0.1:8000/v1 \
      --latency 2 --out runs/mine --repeat $r --engines 8080
done
vhhome-score --run runs/mine/r1 runs/mine/r2 runs/mine/r3 --judges gemini,gpt
```

* Records land at `runs/mine/r<k>/<tag>/<tag>__<case>__<persona>.json` with the episode log beside each. A run is
  resumable: a cell whose record exists is skipped.
* `--engines 8080,8081,...` runs several simulator instances at once. **A model you serve yourself must use one
  port**: its latency is charged to the game clock, and two episodes sharing the server would each be charged the
  other's compute. API assistants may use several.
* `--limit N` runs the first N cells (a smoke run); `--shard k/n` runs every n-th cell; `--cells FILE` and
  `--cases FILE` run your own cells and cases (see [GENERATE_CASES.md](GENERATE_CASES.md)).
* An episode whose log stops growing for `VH_EPISODE_STALL` (900 s) or that exceeds `VH_EPISODE_TIMEOUT`
  (10800 s) is killed and left missing; rerun the same command to fill it. A cell whose assistant calls were all
  transport failures (retries, no tokens ever returned) is moved to `<tag>/void/` and does not count.
* `vhhome-score` runs three steps: `objective` (outcome, in-time success, efficiency, detection; no API calls;
  writes `<run>/metrics_table.json`), `rubric` (the interaction-quality judges, needs `GEMINI_API_KEY` and
  `OPENAI_API_KEY`; resumable per judge; writes `<run>/sq_v4/<tag>/*.json`) and `table` (the pooled table over all
  `--run` roots, `final_results.{json,md}` beside the first root, here `runs/mine/final_results.md`). `--only` picks
  steps; `--judges all` adds the Claude judge (`pip install -e .[anthropic]`, `ANTHROPIC_API_KEY`).

## Reading results

The scorers write local diagnostic reports and preserve per-judge outputs. The [submission pipeline](../../docs/submissions.md) collects scores using the versioned benchmark aggregation. Published comparisons are maintained on the [project leaderboard](https://hellomuffin.github.io/ive-benchmark/redesign/#leaderboard-section). Keep the same cases, personas, user model, timing policy, and two judges when comparing your runs.

## Reading a record

`<run>/<tag>/<tag>__<case>__<persona>.json`:

| key | content |
|---|---|
| `outcome`, `ticks` | how the episode ended (`won`, `finished_wrong`, `stuck`, `timeout`, `silent_fail`) and when |
| `engine_rev` | the engine revision stamp the episode was produced with |
| `scenario`, `persona`, `episode`, `task` | the case, the persona, the task name and its instruction |
| `phases[]` | one per forced mistake: `err`, `injected_at`, `dev_live` (when it physically happened), `armed_at`, `released_at`, `released_reason` (`convinced`, `task_completed`, `block_exhausted`, `prevented`) |
| `phases_total`, `phases_caught`, `phases_prevented`, `phases_detected` | the counts |
| `flags[]` | every mistake call: `t`, `error` (the label), `verdict`, `phase` |
| `conversation[]` | both agents' turns: the person's line, move and plan update; the assistant's JSON (`vlm_json`); an engine snapshot of the state at that tick |
| `action_log[]` | every command the person issued: tick issued and done, status (`done`, `rejected`, `interrupted`), refusal reason |
| `assistant` | statistics: calls, empty replies, retries, schema failures, tokens, latency |
| `assistant_latency[]` | per reply: when it was called, the wall seconds, the ticks charged, whether the cap applied |
| `checkpoints` | the task's success checks and their final values |
| `cook` | the person's own ledger of whether the assistant named its live mistake (`none`, `correct`, `wrong`) |
| `grader` | which judgement releases a forced mistake (`cook`, the default, or `engine`) |

`python -m vhhome_bench.video.film <record.json> [-o out.mp4]` renders a record into a film (ffmpeg required).
`IV_CAPTURE_DIR=<dir>` at run time writes every prompt and every image both sides saw, turn by turn.
