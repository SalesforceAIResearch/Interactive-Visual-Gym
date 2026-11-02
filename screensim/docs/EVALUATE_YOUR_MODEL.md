# Evaluating your model

Your model serves as the visual assistant: it observes phone screenshots and communicates with the simulated user, but does not tap controls.

## 1. Choose an assistant configuration

`screensim-roster` lists the configured assistants. Use a roster tag to preserve its backend and model-specific request settings.

```bash
screensim-benchmark --assistant gemini-3.8-flash \
    --out runs/my-model/r1/composite_results_v2.json
```

For an OpenAI-compatible model server:

```bash
screensim-episode --task hearing_setup --episode hearing_setup__c2 \
    --persona baseline --assistant openai:my-model \
    --base-url http://localhost:8000/v1 --out runs/my-model/smoke
```

The model name and endpoint are forwarded together to the OpenAI backend. `--base-url` also overrides the endpoint of an existing OpenAI roster entry without changing the stored roster. Use `--dry-run` to inspect the selected case–persona cells before starting a run.

For a new backend, inspect [the roster](../screensim/interact/roster.py), its `make_backend` function, and the adapters in [screensim_bench/backends](../screensim_bench/backends/). Test routing explicitly before running the benchmark.

## 2. Observation and response interface

The assistant receives screenshots, conversation context, and its previous structured response. The existing harness controls how these are assembled. The reply includes an utterance and an optional typed error flag:

```json
{"text": "Turn Night Mode back on, then turn off Motion to Wake.",
 "flag": {"error": "wrong_target"}}
```

The full contract is in [assistant_reply_schema.json](../screensim/interact/assistant_reply_schema.json). Preserve the schema and error vocabulary when integrating a model; an empty utterance represents silence.

## 3. Score saved records

```bash
screensim-score --results runs/my-model/r1
screensim-grade --results runs/my-model/r1 --judge gemini,gpt
```

Objective scoring is offline. Interaction-quality grading calls the configured judges for missing judgments. See [Metrics](../../docs/metrics.md).

## 4. Compare consistently

Use the same cases, persona assignments, simulated-user model, manual context, and judge configuration. Keep independent runs in separate directories. The `--no-manual` option changes the assistant's information and is an ablation, not a default-equivalent setup.

See [Reproduce results](REPRODUCE.md), [Benchmark](BENCHMARK.md), and [Getting started](GETTING_STARTED.md).

## Claude

Install `pip install -e ".[all]"`, export `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`,
and use `--assistant anthropic:<model-id>` with the episode or benchmark command.
Claude receives the existing assistant prompt and screenshot stream through the Messages API.
The simulated person remains Gemini 3.7 Flash. See the shared
[Claude setup](../../docs/evaluating_assistants.md#claude-assistants) for credentials and scoring.
