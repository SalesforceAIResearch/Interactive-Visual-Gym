# Interaction quality grading (the per-round rubric)

This is the grading of an assistant's *interaction quality* in VH-Home: how well each thing the assistant says
holds up against what was actually true in the house at that moment and against what the person needed. It is the
`AQ` column of the benchmark table, next to task success and detection. The number is the mean of independent
LLM judges on a per-round rubric with a factual-grounding gate. Everything below is implemented in
`vhhome_bench/scoring/rubric.py` (per-episode grading, rubric version `sq-v4-vh-2026-09-22`),
`vhhome_bench/scoring/rubric_table.py` (the per-run arm table) and `vhhome_bench/scoring/final_table.py` (the
pooled table). `vhhome-score` runs them; see [REPRODUCE.md](REPRODUCE.md) section 3.

The rubric is the household port of the one CookSim uses, with the same items reworded for a house instead of a
kitchen, so the two benchmarks' `AQ` columns are built the same way.

## 1. What one grading unit is

A **round** is one conversation round: the assistant's line (or lines) of that round, the person's reply, and what
the person then did until the next round. One judge call per round. Every round of every episode is graded;
nothing is sampled. **Silence is not graded**: a round the assistant stayed out of contributes nothing, and the
obligations it ignored (a missed mistake, a late warning) are engine metrics, not rubric items. The judge grades
only the assistant's line; the person's reaction is evidence about the line, never the grade. A defiant persona
refusing good advice is not the assistant's failure, but an advised command that the engine refused is evidence
that the advice could not be carried out.

## 2. The rubric

Sixteen yes/no items in ten categories. Each item has a weight, or is a **must** item: a must item that fails zeroes
the round. An item applies to a round only when the round's utterance type is one the item lists (an empty list
means all types); the judge first classifies the line into one or more types from `mistake_claim`, `instruction`,
`answer`, `alert`, `plan_comment`, `reply_to_pushback`, `acknowledgement`. The item texts below are the ones the
judge is given, verbatim.

| item | category | weight | applies to | question |
|---|---|---|---|---|
| T1 | truthful | must | all | Every factual claim in the line about the house's observable state or about what the person has already done is true at this tick (whether a flagged mistake is the RIGHT mistake is verdicted by the engine and excluded here; the surrounding state and action facts are not). |
| T2 | truthful | must | all | The line refers to no room, object, appliance, door or task requirement that is absent from this house and this task. |
| S1 | sensible | 2 | all | The line is about the current situation — what the person is doing or about to do, a live or imminent mistake, a pending question or objection, or a plan the person just announced — not a finished step or a distant future step (whether an advised step is the RIGHT one is H2). |
| S2 | sensible | 2 | all | If the person has a pending question, request or objection, the line answers it, accepts it, rejects it with a reason, or defers it for a stated urgent reason. |
| P1 | specific | 2 | mistake_claim, instruction, answer, alert | Every referent with more than one candidate in this house (two mugs, three rooms with a table, either of two doors) is uniquely identified by room, position, contents or state. |
| P2 | specific | 2 | mistake_claim | The mistake is named as the concrete thing that is wrong (which object, which step), not a vague 'something's off'. |
| H1 | helpful | must | instruction, mistake_claim, answer | Each advised action can be started NOW from where the person is and what they hold, or the line includes the prerequisite that makes it startable (put the thing down, go through the door, enter the named room); the place exists and accepts the object in its current state. |
| H2 | helpful | 2 | instruction, mistake_claim | Each advised action is useful for finishing the chore or repairing the mistake — not merely legal while irrelevant, premature or counterproductive. |
| H3 | helpful | 2 | mistake_claim | The mistake claim comes with what to do about it (or an explicit 'stop' when stopping is the fix). |
| W1 | well_timed | 2 | all | This was an appropriate moment to speak: a mistake repair or urgent alert is spoken while it can still change the outcome; non-urgent tips, narration, praise or small talk are not spoken while the person is speaking, carrying out a correct step, or facing a live unaddressed mistake. |
| L1 | listens | 2 | reply_to_pushback, mistake_claim | If this replies to the person pushing back on an earlier assistant claim or instruction, it adds something new — concrete evidence (what it saw, where), a different reason, a correction, or an alternative step — rather than repeating or intensifying. |
| L2 | listens | must | all | If the person has announced a legitimate plan change, or has asked the assistant to stop interrupting, the line respects that unless it states a current task-critical reason. |
| U1 | honest | 2 | mistake_claim, answer, reply_to_pushback | Stated confidence matches what the assistant could actually see through the camera: visible facts stated plainly when true; things out of view framed as checks or questions; a claim the person or the house contradicts is conceded, not defended. |
| SF1 | safe | must | instruction, mistake_claim, answer, alert | The advised action is safe and sane for a home: it does not leave an appliance, tap or hob running unattended when that matters, does not put an object somewhere that ruins it or blocks a doorway, and a genuinely hazardous condition is marked urgent. |
| C1 | consistent | 1 | all | The line is consistent with the assistant's earlier relevant statements unless it marks a correction or the house changed enough that the earlier statement no longer applies. |
| E1 | economy | 1 | all | The line is concise for its function and this person's tolerance for interruption: what is needed to act or understand now, without narration, praise or future steps that are not currently useful. |

### The reported subset

The reported setting (`vhhome-score` passes `--subset` to the grader) asks the judge only the **ten items of the
five reported categories**: truthful (T1, T2), sensible (S1, S2), helpful (H1, H2, H3), listens (L1, L2) and
economy (E1). The other five categories (specific, well_timed, honest, safe, consistent) are not reported: asking
for them costs a third of the prompt and the output for numbers nothing reads, and `specific` and `safe` in
particular were passed by every seat, so they carry no information about any seat. Grading all sixteen is one
flag away (`python -m vhhome_bench.scoring.rubric <run>` without `--subset`).

### The repair table

So that the judge grades H2 and H3 against the same repair the engine scores, not against its own idea of one, the
judge is told what the engine considers the repair for each forced mistake class (`REPAIR` in `rubric.py`):

| forced mistake | the repair |
|---|---|
| `skipped_step` | go back and do the step that was skipped, before carrying on |
| `missing_step` | do the step that was left out before finishing |
| `wrong_object` | put the wrong object back where it was and take the one the task names |
| `misidentify_fetch` | put the wrong object back and fetch the one the task names |
| `wrong_placement` | pick the object up again and put it in the place the task names |
| `repetition` | stop: that step is already done; move on to the next one |
| `stall` | resume the chore with the next step |
| `navigation_inefficiency` | take the direct route to the room the next step is in |

## 3. What the judge is given

Every assistant turn in a VH-Home record carries the engine's own state stamp for that tick, so the situation block
the judge reads is exact rather than reconstructed from frames. For one round the prompt contains:

| the judge sees | where it comes from |
|---|---|
| the task instruction, the task and case names, the person's persona card (ground truth; the assistant was not told it verbatim) | the record and `protocol.PERSONA_CARDS` |
| the situation at the round's tick: what the person is holding, what is open, what is switched on, where the task objects are, the person's own next commands (its plan), its mode | the engine's state stamp in the record's `conversation[]` |
| the person's last commands in the 60 ticks before, each with the engine's result and the refusal reason | `action_log[]` |
| the live forced mistake (what the engine made the person do wrong) and its window | `phases[]` |
| the engine's verdict on any mistake the assistant named in this round | `flags[]` |
| what is pending from the person (a question, a request, an objection) and whether it has asked for less interruption | the person's moves in `conversation[]` |
| the previous 8 rounds, and a ledger of up to 40 of the assistant's earlier lines (for consistency) | the record |
| the round itself: the assistant's line(s), the person's reply with its move | the record |
| what happened until the next round: the person's commands and the engine's results (evidence only) | `action_log[]` |
| the rubric items in force, with category, weight and applicable types | `rubric.py` |
| optionally (`--frames`), the rendered frame at that tick | the episode's film beside the record |

The judge answers with JSON: the round's types; every factual claim with a verdict (true, false, unverifiable
against the situation); every advised action in the simulator's command form with whether it can be started now;
the purpose (urgent or not); and pass / fail / not-applicable for every item with a one-sentence reason quoting the
evidence.

**Engine-side overrides** are applied on top of the judge's verdicts, so the parts that can be checked mechanically
are not left to the model: a claim the judge marked false fails T1; a line that advises a real action is typed
`instruction` whatever the judge called it (monitoring advice such as "keep watching" is not an action); the first
real advised action decides H1 (later steps are sequenced behind it); if the person tried the advised command and
the engine refused it, H1 fails on the refusal, which outranks the judge either way; L2 is not applicable when the
person has not asked for less interruption.

## 4. The judges

| judge key | model | role |
|---|---|---|
| `gemini` | `gemini-3.7-flash`, JSON mode, temperature 0 | independent judge, in the reported score |
| `gpt` | `gpt-5.4-mini`, JSON mode | independent judge, in the reported score |
| `claude` | `claude-haiku-5-5`, effort low | independent judge, optional (`--judges all`; needs `pip install -e .[anthropic]`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_WORKSPACE_ID` for a user-scoped key) |

The judges are **independent**: each receives the identical prompt in a separate call and never sees another
judge's answer. The same judge set grades every seat; there is no same-family exclusion, because any
self-preference is diluted identically everywhere. Judges are flash-tier on purpose: the full protocol is one call
per round per judge over 19 seats, 225 episodes each. Each call is capped at 420 s (`SQ_CALL_TIMEOUT`) and retried
three times; a judge that still returns nothing is recorded as an error for that round, and the round is scored
from the judges that answered. Claude calls are throttled (`CLAUDE_CONCURRENCY`, default 20 in flight) and a rate
limit waits rather than giving the round up.

The reported setting uses Gemini and GPT on every round. Historical files may also
contain Claude verdicts, which are retained for analysis but excluded from the paper score.
The judges receive the transcript and engine-state context, without rendered frames.

## 5. Aggregation

For each judge separately, an applicable category passes if every applicable item in that
category passes. The five categories are factual grounding, situational relevance, actionable
guidance, user intent uptake, and guidance conciseness. Failure of T1 or T2 gates that judge's
round score to zero. Otherwise, the score is the mean of its applicable categories.

```
judge score  = factual-grounding gate × mean of applicable category pass flags
round score  = mean of Gemini and GPT judge scores
AQ(run)      = mean of scored rounds pooled across episodes in that run
AQ(model)    = mean of the three run scores
SD           = population standard deviation of the three run scores
```

The same two judges are used for category scores. `final_table.py` reads their saved items
and recomputes this score; it does not trust historical `aggregate.AQ` fields. The diagnostic
`rubric_table.py` additionally prints the episode-mean variant and its bootstrap interval,
which are not the headline score. Missing judge calls should be retried before comparing
results; the submission grader requires both judges for each round.

## 6. Running it, caching and resume

`vhhome-score --run <r1> [<r2> <r3>] --only rubric --judges gemini,gpt --workers 32` runs, per run root:

```bash
python -m vhhome_bench.scoring.rubric <run> --subset --judge gemini,gpt --workers 32
python -m vhhome_bench.scoring.rubric_table <run>
```

Needs `GEMINI_API_KEY` and `OPENAI_API_KEY` (from the environment or `~/.config/vhhome/keys.env`).

* Output per episode: `<run>/sq_v4/<tag>/<tag>__<case>__<persona>.json`. If it exists, only the judges missing from
  it (or that errored) are called, per round, and the round is re-merged; **a judge never grades a round twice**.
  `--judge all` therefore adds the third judge to already-graded episodes without touching the first two.
* `--only <tag>[,<tag>]` restricts a pass to some seats; `--limit N` to N episodes per seat; `--frames` sends the
  rendered frame at each round (needs the episode's film beside the record).
* One grader per run root at a time: a lock file under `<run>/sq_v4/` makes a second pass wait, so an orchestrated
  scoring step and a hand-run pass cannot pay for the same rounds twice.
* At the end the grader prints each judge's call count and token totals, so what a pass cost is measured rather
  than guessed.

### Output format

Per episode (`<run>/sq_v4/<tag>/<episode>.json`):

```
schema (rubric version, "-sub5" when the subset was graded, "-3j"), items_graded, episode, arm, scenario, persona,
judges: [...], judge, with_frames, engine_rev, outcome
aggregate: AQ, rounds, rounds_scored, must_fail_share, by_category, by_type
rounds[]: round, t, assistant (the lines), person_reply [(move, text)], live_error, stamp_source,
          judges: { gemini: {items, score, must_fail, types, raw}, gpt: {...}, claude: {...} },
          score, must_fail, types, items (the merged view: pass only if no judge failed it; reasons joined with " || ")
```

The per-judge blocks are the primary data. Round-level `score` and `aggregate.AQ` retain historical
diagnostics; `final_table.py` recomputes the paper score from the per-judge items as described in section 5.

## 7. Limitations

* The judges are language models. They are given the engine's exact state, and the mechanically checkable parts
  are overridden from it, but the judgement of relevance, usefulness and economy is theirs; averaging two judges
  of different families is the mitigation, not a cure.
* Silence is not graded, so AQ says how good the assistant's lines were, not whether it spoke when it should have.
  Read it beside `DET` (did it name the mistakes, in time) and `won`: a seat can rank high on detection and low on
  AQ, which is the divergence the rubric exists to expose, and a seat that speaks rarely can carry a high AQ on a
  handful of lines.
* The gate is strict: one false claim zeroes the round for the judge that caught it, whatever else the line did.
  That is intended (a wrong statement about the house is worse than an unhelpful one), but it makes AQ sensitive to
  the judges' reading of the state stamp.
* Seats that never speak (`silent`, and `mmduet2` in the published run) have no AQ.
* The published AQ had a third judge on most rounds and two on the rest; a reproduction with two judges throughout
  is comparable but will not match to the third decimal.
