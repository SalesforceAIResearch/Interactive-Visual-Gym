# Assistants

List the configured model backends from the installed release:

```bash
screensim-roster
```

The [roster implementation](../screensim/interact/roster.py) defines the available tags and model-specific settings. The returned list is the source of truth; model availability and serving requirements are not implied by a successful listing.

## Evaluate a configured model

See [Evaluate your model](EVALUATE_YOUR_MODEL.md) for the response contract, custom endpoints, run commands, and scoring.

## Model serving

Local-model launch scripts and adapters are under [serving](../serving/). Use the model's required environment and verify its endpoint before evaluating it. A working server is separate from successful routing through the benchmark wrapper.

## Credentials

Hosted assistants, the simulated user, and judges may use separate clients and keys. See [Getting started](GETTING_STARTED.md) and [Interaction quality](INTERACTION_QUALITY_GRADING.md). Do not place credentials in roster entries, benchmark files, or logs.
