# The engine

A deterministic iPhone. Same seed, same taps, same pixels.

## Shape

```
screensim/
  core/     state, actions, gesture grounding, the step loop, the OS shell, pacing
  apps/     Settings (hand-built pages + ~950 synthesized from the User Guide), QuickPay,
            Messages, Mail, Phone, Photos, Shop; inventory.py walks every reachable page
  render/   scene graph -> HTML phone screen, captured headlessly at 412x892
  interact/ the benchmark layer: personas, prompts, the episode designer, the runner,
            the person's screen readout, verdicts and grading
  tasks.py  the benchmark tasks; tasks_rl.py and task_splits.py the training library
```

## How a tap works

A gesture names what is on the screen — `tap Flash on Silent`, `swipe up`, `press back`,
`type "trailmix88"`. The engine resolves it against the laid-out widgets: if the label is not on
screen, the tap is refused rather than guessed. That is what makes "the assistant told them to tap
something that is not there" a measurable event rather than an opinion.

## Determinism and the stamp

`engine_rev.py` hashes the engine and the harness into every record, so a result can always be traced
to the code that produced it — and a record from an older engine can be spotted before it is
published. Films are replays of saved records: no model calls, no re-runs.

## The phone's pages

Most of Settings is generated from a local mirror of the iPhone User Guide, so the page tree and the
control names match the real device rather than a hand-drawn approximation. `benchmark/auto_tasks.jsonl`
holds the single-control tasks derived from that mirror; `screensim.apps.inventory` walks the whole
tree and is what the in-phone search uses.

## Rendering

The screen is real HTML rendered in headless chromium (`pip install ".[render]"` then
`python -m playwright install chromium`). Everything except rendering and video works without it.
