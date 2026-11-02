# Adding an assistant backend

A backend is a small class that turns the harness's turn into a model call. Everything else, the prompts, the frame
selection, the latency charge, the record, is shared.

## The interface

```python
class MyBackend:
    name = "my-model"                      # the record tag

    def call(self, system: str, user: str, images: list[bytes] | None) -> tuple[dict, dict]:
        """One turn. `images` are JPEG frames (oldest first). Return (reply, meta):
        reply = the parsed JSON the model produced ({"text": ..., "flag": ...}); {} if it produced nothing
        meta  = {"latency_ms": int, "prompt_tokens": int, "completion_tokens": int, ...}  (latency is charged)"""
```

`cooksim_bench/backends/controller_eval.py` has the reference implementations: `OpenAIBackend` (any
chat-completions server; image encoding, JSON mode, the request profiles) and `GeminiBackend`;
`anthropic_backend.py` is the smallest complete example (40 lines of logic). The harness wraps your backend in
`AssistantBackend` (`assistant_backend.py`), which retries a failed call three times and archives `meta` into the
record.

Session backends (a model that keeps state across turns) add `reset_episode(system)`: it is called once per episode
with the system prompt, and `call()` then receives only the new user text and frames.

## Registering it

1. Add the class to `cooksim_bench/backends/` and a branch to `make_assistant()` in `assistant_backend.py`
   (`backend == "mybackend"` → `MyBackend(model, ...)`).
2. Add the choice to `--assist-backend` in `cooksim_bench/e2e_stepin_rollout.py` (search for `choices=["gemini", "openai"`).
3. Add a roster entry to `cooksim_bench/roster.py` so `cooksim-episode --assistant <tag>` and `cooksim-roster` know it:

```python
dict(slot="21", tag="my-model", group="C", kind="api", status="ready", needs=["MY_API_KEY"],
     flags=["--assist-backend", "mybackend", "--assist-model", "my-model"]),
```

For a locally served model, no code is needed: serve it behind an OpenAI-compatible endpoint and use
`--assistant openai:<served-name> --base-url ...`, or add a `serving/serve_<name>.sh` and a roster entry of kind
`local` with its port (see the existing entries).

## Native streaming backends

`cooksim_bench/backends/native_assistant.py` implements the setting in which the model, not the harness, decides when
to speak: a session receives one frame per tick (at most 1 fps on the wall clock) and the cook's lines as audio,
and emits utterances whenever it wants; each utterance is charged from the end of the cook's audio (or the last
frame) to the first output token. The Gemini Live and GPT realtime sessions are the two implementations; a new one
follows the same `NativeSession` shape (`open`, `push_frame`, `push_speech`, `drain`, `speaking`, `close`).

## Checking a backend

```bash
cooksim-episode --latency 2 --case b3_easy_nops_medium_map_2 --persona baseline --assistant <tag> --out runs/check
```

Read the record's `conversation` for the raw replies: the JSON must parse and carry `text` and `flag`; the
`assist_latency` field shows what the clock charged.
