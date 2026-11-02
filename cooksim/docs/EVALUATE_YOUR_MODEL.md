# Evaluating your model

Your model takes the **assistant seat**: it watches the cook through frames, reads the conversation, and may speak.
There are three ways to plug it in.

## 1. An OpenAI-compatible endpoint (vLLM, SGLang, your own server)

```bash
cooksim-episode --latency 2 --case b3_easy_nops_medium_map_2 --persona baseline \
    --assistant openai:my-model --base-url http://127.0.0.1:8000/v1 --profile default --out runs/mine
```

The harness posts to `<base-url>/chat/completions` with the system prompt, one user message carrying the frames
(as images) and the text, and asks for JSON. `--profile` selects request details per model family
(`default`; `gpt5` for OpenAI reasoning models). The served model name is the record tag; override it with `--tag`.

## 2. An API backend

`--assistant gemini:<model>`, `openai:<model> --base-url https://api.openai.com/v1 --profile gpt5`,
`anthropic:<model>`; the streaming settings `live:<gemini live model>`, `realtime:<gpt realtime model>` (polled) and
`live_native:` / `realtime_native:` (the model drives turn-taking). Keys: `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` (plus `ANTHROPIC_WORKSPACE_ID` for a user-scoped key).

## 3. A new backend class

Anything else, including models that need their own client library: see [ADD_A_BACKEND.md](ADD_A_BACKEND.md).

## The protocol your model must follow

**Input, every 2 ticks:** a system prompt (kitchen rules, the recipe, the cook's questionnaire answers, the flag
vocabulary, the reply schema), then the latest frames of the top-down and first-person views and the conversation
since the last turn, with the cook's lines marked. With `--obs agent+++` the frames are replaced by a text state map.

**Output:** JSON only.

```json
{"text": "The pan on the left is smoking, take the meat out.", "flag": null}
{"text": "Stop, that is the wrong order: the bun goes down first.", "flag": {"error": "wrong_order"}}
{"text": "", "flag": null}
```

* `text` is spoken to the cook; the empty string stays silent. Silence is the right answer most of the time.
* `flag.error` names a forced mistake from the vocabulary (`wrong_order`, `serve_raw`, `add_foreign`,
  `misidentify_assemble`, `omit_subgoal`, `repeat_step`, `forget_next_step`, `leave_unattended`, `mis_prep`). It is how
  the engine knows the assistant *named* the mistake; the cook also hears `text`. A wrong name, or a flag when no
  mistake is live, is a false alarm.
* Response timing is selected with `--latency`: `2` delivers every response after two simulation ticks (main benchmark); `measured` charges inference time at 1.2 seconds per tick, rounded up and capped at 30 ticks. A 10-second response therefore arrives 9 ticks later in measured mode.

Replies that are not valid JSON, or that nest `text`/`flag` inside another object, are tolerated once (the harness
lifts nested keys) but a model that cannot keep the schema will be read as silent.

## Running the benchmark

```bash
for r in 1 2 3; do
  cooksim-benchmark --assistant openai:my-model --base-url http://127.0.0.1:8000/v1 \
      --latency 2 --out runs/mine_fixed2 --workers 1 --repeat "$r"
done
cooksim-score --run runs/mine_fixed2
```

This runs the main benchmark with **fixed two-tick response delay**, ignoring measured inference latency. For the measured-latency comparison, replace `--latency 2` with `--latency measured` and use a separate output root such as `runs/mine_measured`. Keep the cases, personas, model settings, and judges unchanged. See [Response timing](../../docs/evaluating_assistants.md#response-timing) and [Reproduce](REPRODUCE.md).

Benchmark scores are maintained on the [project leaderboard](https://hellomuffin.github.io/ive-benchmark/redesign/#leaderboard-section), not duplicated in this repository.

## Reading a record

`<run>/frames/<persona>/<tag>/<case>.json`:

| key | content |
|---|---|
| `outcome`, `ticks` | how the episode ended and when |
| `conversation` | every line, who said it (`vlm` = assistant, `human` = cook), the tick, the raw model JSON |
| `injections` | each sub-case: when it armed, fired, released, whether the mistake committed, the cook's verdict on the assistant's call |
| `flags`, `false_positives` | the assistant's mistake calls and how they were judged |
| `actions`, `kitchen_feedback` | the cook's commands and the engine's responses, tick by tick |
| `assist_latency` | the charged latency per reply |
| `persona_v2`, `questionnaire_v2` | the persona and what the assistant was told |

`--dump-states` adds `<case>.states.jsonl`, the full kitchen state per tick, which the rubric grader uses as exact
evidence and the film tools replay.
