# Interaction quality grading

This is the formal grading of an assistant's *interaction quality* in CookSim: how well each thing the assistant says
holds up against what was actually true in the kitchen at that moment and against what the cook needed. It is the
"interaction quality" column of the benchmark table, next to task success. The number is the mean of independent
LLM judges, read on five categories, with a factual-grounding gate. **The formal setting (ruling 2026-10-12) averages
two judges, Gemini and GPT**; a third judge, Claude, is implemented and can be added per episode with `--judge all`, but
it is not part of the reported two-judge score. Everything below is implemented in
`cooksim_bench/scoring/sq_v4_score.py` (per-episode grading) and `cooksim_bench/scoring/sq_v4_three_runs.py` (seat-level aggregation).

## 1. What one grading unit is

A **round** is one assistant line together with its context:

| the judge sees | where it comes from |
|---|---|
| the assistant line (and the lines of the same round) | the episode record's `conversation` |
| the frame the assistant was looking at when it decided | the episode video at the decision tick (`called_at`), not the arrival tick |
| the exact kitchen state at that tick: stations, items, plate ledger, who holds what | the per-tick state dump, or the recorded evidence when no dump exists |
| the live situation: a fired mistake and whether it has committed, a pending cook question or request, a live announced intent, a standing "stop interrupting" request | the record's injection and conversation logs |
| the cook's reaction until the next round: lines, commands and their results, engine aftermath | the record |
| the assistant's own previous lines (ledger of up to 40) | the record |
| the recipe, the kitchen inventory, the cook's persona answers (skill, interruption preference, trust) | suite + persona |

Every assistant line in every episode is one round. Nothing is sampled.

## 2. The rubric

The judge classifies the line into one or more **utterance types** (mistake claim, instruction, answer, alert, plan
comment, reply to pushback, acknowledgement), lists every factual claim with a verdict (true / false / unverifiable against
the state it was shown), lists every advised action with "can it be started now", and then answers the rubric items.
Sixteen items are asked; the reported score uses the ten that belong to the five categories below.

| category (paper name) | item | question (short form) | weight | applies to |
|---|---|---|---|---|
| **Factual grounding** (`truthful`) | T1 | every factual claim about the current kitchen state or the cook's completed actions is true at this tick | gate | all lines |
| | T2 | the line refers to no station, item, place, rule or recipe requirement that does not exist in this kitchen and order | gate | all lines |
| **Situational relevance** (`sensible`) | S1 | the line is about the current situation: the cook's current action or subgoal, a live or imminent mistake, a pending question or objection | 2 | all lines |
| | S2 | if a cook question, request or objection is pending, the line answers, accepts, rejects with a reason, or defers it for a stated urgent reason | 2 | all lines |
| **Actionable guidance** (`helpful`) | H1 | each advised action can be started now in the current state (or the prerequisite is stated) | fail = category fails | instruction, mistake claim, answer |
| | H2 | each advised action is useful for the current plan, the pending request or the needed repair | 2 | instruction, mistake claim |
| | H3 | a mistake claim comes with what to do about it | 2 | mistake claim |
| **User intent uptake** (`listens`) | L1 | a reply to cook pushback adds something new (evidence, a re-check) rather than repeating | 2 | reply to pushback, mistake claim |
| | L2 | a live valid announced intent, or a request to stop non-urgent interruptions, is respected unless a current safety reason is stated | fail = category fails | all lines |
| **Guidance conciseness** (`economy`) | E1 | the line is concise for its function and for the cook's interruption preference | 1 | all lines |

Engine-side overrides are applied on top of the judge's item verdicts, so the parts that can be checked mechanically are
not left to the model: a claim the judge marked false against the engine state fails T1; the first *real* advised action
(monitoring advice such as "keep watching" is not an action) decides H1; L2 is marked not-applicable when there is no
live intent or stop request. The remaining six items (specificity, timing, honesty, safety, consistency) are recorded
in the output for analysis but are not part of the reported score.

## 3. The judges

| judge key | model | role |
|---|---|---|
| `gemini` | gemini-3.7-flash | independent judge, **in the reported score** |
| `gpt` | gpt-5.4-mini | independent judge, **in the reported score** |
| `claude` | claude-haiku-5-5 | independent judge, optional (`--judge all`), not in the reported score |

The judges are **independent**: each receives the identical prompt and frame, in a separate call, and never sees
another judge's answer. Every episode of every seat is graded by the same judge set (`--judge gemini,gpt`); there is no
same-family exclusion, because the same judges grade every seat and any self-preference is diluted identically
everywhere. The aggregation (`sq_v4_three_runs.py`) reads a FIXED judge set (`SQ_JUDGES`, default `gemini,gpt`), so an
episode that happens to carry a Claude verdict as well is averaged over the same two judges as every other episode.
Why more than one judge: on the 1,229 episodes of the previous run that two judges both graded, the Gemini judge
scored strong API seats 0.07 to 0.17 higher than the GPT judge and weak open-weight seats 0.09 to 0.16 lower, while the
run-to-run spread of a seat was only 0.02. A single judge therefore moves the ranking more than the benchmark's own
noise does; averaging judges of different families cancels the family-specific lean.

Judges are flash-tier on purpose: the full run is about 58,000 rounds per judge for the three benchmark runs alone.
Each call is capped at 420 s and retried three times; a judge that still returns nothing is recorded as an error for that
round and the round is scored from the judges that answered.

## 4. Aggregation

Let a round be judged by the set `J` of judges that answered. For judge `j` and category `c`, with `I_c` the items of
`c` that applied to this line (an item applies when the line's types include one of its types):

```
clean_j(c)  = 1  if no item in I_c failed for judge j, else 0        (absent if I_c is empty)
gate_j      = 0  if judge j failed T1 or T2, else 1
clean(c)    = mean over j in J of clean_j(c)                          judges averaged
judge score = gate_j × mean over categories applicable to judge j of clean_j(c)
round score = mean over j in J of judge score
```

So a line whose facts are wrong scores 0 for every judge that caught it, whatever else it did right; a line whose
facts are right scores the share of its applicable categories that were clean, averaged over judges. A category
with no applicable item (for example User intent uptake on a line with no live intent and no pushback) is simply absent
for that round and does not pull the score either way.

Seat level:

```
AQ(seat, run)      = mean of round score over ALL rounds of that seat in that run (pooled, not per episode)
category(seat,run) = mean of clean(c) over the rounds where c applied
AQ(seat)           = mean over the three independent runs, reported with the SD over runs
```

The episode-mean variant (mean of per-episode means) is kept in the JSON as `AQ_episode_mean` but the table uses the
pooled round mean, so a chatty seat is weighed by every line it spoke.

## 5. Running it

Run from `interactive_visual_gym/cooksim/` after completing [Getting started](GETTING_STARTED.md):

```bash
cooksim-score --run runs/my_run --only rubric --judges gemini,gpt --workers 8
cooksim-score --run runs/my_run --only table
```

The rubric step calls the configured judges. The table step reads cached judgments. To add the optional Claude judge, use `--judges all` and configure its credentials.

Resume rules. An episode's output lives at `<root>/sq_v4/frames/<persona>/<seat>[__r2|__r3]/<episode>.json`. If it
exists, only the judges missing from it (or that errored) are called, per round, and the round is re-merged; a judge
never grades the same episode twice. `--judge all` adds the third judge to already-graded episodes. `--model <seat>`
and `--limit N` restrict a pass. The orchestrator (`cooksim-score`, step `rubric`) runs exactly the
`--judge gemini,gpt` command once a phase is complete and verified.

**Validity of a stored judgement.** A judge entry counts as a judgement only if it carries a `score` and at least one Boolean
item verdict, or if it is a documented not-applicable round (`na_round: true`, written when the judge answers but marks every
item not applicable, e.g. for an empty or malformed assistant line). A `score` key alone does not; the resume logic re-calls
such slots. The rule is `valid_judgement()` in `cooksim_bench/scoring/sq_v4_score.py`.

## 6. Output format

Per episode (`sq_v4/frames/.../<episode>.json`):

```
schema, episode, mode, persona, model, judges: [...], stamp_source (exact per-tick state | recorded evidence)
aggregate: AQ, rounds, rounds_scored, must_fail_share, by_category, by_type        # single-file view, all 16 items
rounds[]: round, t, assistant lines, cook reply, types, score, must_fail, items (merged), judges: {
            gemini: {items, score, must_fail, types, raw}, gpt: {...}, claude: {...} }
```

The per-judge blocks are the primary data; `items` and `score` at round level are the merged view (reasons of the
judges joined with ` || `). The paper number comes from `sq_v4_three_runs.py`, which re-aggregates from the per-judge
item verdicts exactly as in section 4 and writes `<root>/sq_v4/three_runs_five.json`
(`{seat: {runs: {1,2,3: {n, rounds, AQ, AQ_episode_mean, by_category, applicable}}, mean, sd}}`).

## 7. Provenance and history

* `RUBRIC_VERSION = sq-v4-2026-08-27-flash`; rubric items and the engine overrides date from the 2026-08-28 audit.
* 2026-09-22: the five reported categories and their paper names; 2026-09-24: round-level view with the
  Factual-grounding gate; pooled rounds.
* 2026-10-09: independent judges averaged on every episode, per-judge resume, same-family exclusion dropped. Before
  that a single judge graded each seat (Gemini for GPT seats, GPT for the rest, every fifth episode double-graded).
* 2026-10-12: the reported score is the mean of the Gemini and GPT judges (`--judge gemini,gpt`, `SQ_JUDGES`); the
  Claude judge remains available but is not reported.
* Judge claims are read against the frame at the decision tick, not the arrival tick (ruling 2026-09-25).
