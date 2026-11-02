# Interaction quality: how an episode is graded

ScreenSim evaluates assistant responses against the recorded phone state and conversation. The grader is
[screensim_bench/scoring/rubric.py](../screensim_bench/scoring/rubric.py); the five-category aggregation is
[screensim_bench/scoring/five_categories.py](../screensim_bench/scoring/five_categories.py).

## The unit: one round

Every assistant line is one **round**: the line, the person's reply and every tap until the next
line, and the exact screen the person was looking at at that tick — every row and its value, read
out of the engine, so no judge has to interpret pixels.

## Two judges on every episode, averaged

```
gemini-3.7-flash   +   gpt-5.4-mini
```

(ruling 2026-10-12). A third judge, `claude-haiku-5-5`, is wired up and can be added with
`--judge all`, but it is **not** part of the reported numbers.

Each judge scores the round independently. Historical diagnostic fields retain the following merged view; these fields are not the reported five-category score:

| quantity | rule |
|---|---|
| round score | the **mean** of the scores the judges returned (a judge that errored contributes nothing) |
| the gate | the round is gated if **any** judge gates it |
| an item (T1, S2, H3 …) | passes only if **every** judge that ruled on it passed it; `null` if none ruled |
| reasons | the judges' reasons joined with ` \|\| ` |

Why two rather than one: a GPT seat graded by a GPT judge cannot rule out self-preference, and that
is the one bias a single judge cannot be argued out of. With a judge from each family on every seat,
the same-family exclusion CookSim used to need disappears, and the mean is not dominated by either.

## Per-judge resume

Grading is resumable **one judge at a time**. A judge that errored, was rate-limited, or was added
later is filled in on the next run; judges that already returned a score for that round are kept and
not re-called. An episode is skipped entirely only when every requested judge has scored every round **and** the
file was written for the same judge set. A file graded with a different set is re-merged from the
verdicts it already holds, with no new API calls, so one table never mixes two judge panels.

```bash
screensim-grade --results runs/my-model/r1 --judge gemini,gpt
# Optional third judge:
screensim-grade --results runs/my-model/r1 --judge all
```

Output: `<root>/<outdir>/<seat>/<episode>__<persona>.json`, where `outdir` is `aq_v4` for the full
sixteen-item rubric and `aq_v4five` under `--subset five`. Each round carries a `judges` block with
every judge's own verdict alongside the merged one, so a disagreement can always be reopened.

## Credentials

| judge | key |
|---|---|
| gemini-3.7-flash | the Gemini key the harness already uses |
| gpt-5.4-mini | `~/.openai_key` |
| claude-haiku-5-5 | `~/.anthropic_key`, plus `~/.anthropic_workspace` — a user-scoped key (`sk-ant-usr-…`) must name the workspace it bills to, sent as the `anthropic-workspace-id` header |

Both Anthropic files are read automatically by `screensim/interact/anthropic_backend.py` when the
matching environment variables are unset.

**Rate limits.** The Anthropic workspace allows far fewer parallel calls than we run graders, and
CookSim grades against the same workspace. The Claude judge therefore caps its own in-flight calls
process-wide (`AQ_CLAUDE_CONCURRENCY`, default 16) and backs off on HTTP 429 — up to six attempts,
doubling from 10 s — instead of recording an error for the round. Seeing 429s in a first smoke test
is normal and is not a credential problem.

## What the five-category view does on top

`screensim_bench/scoring/five_categories.py` reduces each round to the five reported categories (Factual grounding,
Situational relevance, Actionable guidance, User intent uptake, Guidance conciseness) and applies the
**Factual-grounding gate**: a line that fails T1 or T2 scores 0 for the round, whatever else it got
right **for that judge**. We then average the Gemini and GPT scores, pool rounds within each run, and average runs equally. Category scores also average the two judges. The reducer reads the saved per-judge items, not merged flags. `legacy_rounds` retains the old calculation for reproducing historical analyses.

After `screensim-grade`, produce the paper-comparable score with:

```bash
python -m screensim_bench.scoring.aq_contract runs/r1 runs/r2 runs/r3 \
  --model YOUR_ASSISTANT_TAG --json runs/interaction_quality.json
```

The output reports per-run counts, categories, overall quality, and the number of rounds with only one usable judge. Repair missing judgments with `screensim-grade` before comparing to the benchmark. New episode files also include `paper_quality`; their top-level `AQ` remains a historical diagnostic.
