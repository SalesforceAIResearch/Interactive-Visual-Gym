# The benchmark

CookSim measures whether a visual assistant **helps a person who is doing something**: it must notice mistakes as
they happen, say the right thing at the right moment, answer questions about what it sees, honour standing
requests, and otherwise keep quiet. Everything it says is checked against the simulator's ground truth.

## 1. Anatomy of an episode

| piece | what it is |
|---|---|
| **tick** | the unit of simulated time; one tick is 1.2 s of wall clock when a model's latency is charged |
| **cook** | an LLM (gemini-3.7-flash in the published runs) playing a person with a persona; it plans the dish from the order card and the kitchen rules and acts one command at a time, talking only when it has a reason to |
| **assistant** | the model under test; every 2 ticks it receives the latest frames (top-down and first-person) plus the conversation and may return a line to say, with an optional **flag** naming a mistake |
| **case** | the kitchen, the order, and a schedule of sub-cases: forced mistakes, questions, a standing request, a plan change ([GENERATE_CASES.md](GENERATE_CASES.md)) |
| **takeover** | when a forced mistake fires, the engine executes it through the cook's hands and the cook's voice defends it; the takeover releases when the assistant names the mistake with the right flag (the cook then repairs it) or when the mistake commits in the world (the wrong item is on the plate, the food has smoked) |
| **outcome** | `won`: the correct dish served; `wrong_serve`: a wrong dish served; `burned`: food left on the heat caught fire; `timeout`: the clock ran out |

Heat physics: food that is done stays safe for 40 ticks, then the station **smokes**; 45 ticks later it **burns**
and the episode is over. Scooping the food out is the only way to stop the smoke. Both the cook and the assistant
read these numbers in their kitchen rules.

**Latency.** The assistant's measured thinking time is charged to the game clock: a reply that took 6 s arrives
5 ticks after the frames it saw, and the kitchen kept moving meanwhile. The clock does not wait. `--latency 2`
sets a fixed two-tick delay, the main benchmark setting. Use `--latency measured` for the measured-latency comparison. Zero delay is a separate custom setting.

## 2. The suite

Fifty cases: five recipe tiers × two kitchens × five schedules. Case names read `b3_<tier>_nops_<map>_map_<k>`.

| tier | recipe, plated in this order | forced mistakes per case |
|---|---|---|
| easy | caprese: chopped tomato, cheese | 2 |
| medium | cheeseburger: bun, cooked meat, cheese | 2 |
| hard | fish & chips: cooked fish, fried potato, fried potato | 3 |
| expert | steak frites: cooked steak, fried potato, fried potato | 4 |
| master | burrito: tortilla, rice, steak, chopped tomato, chopped lettuce | 4 |

Each case also carries questions, one standing request and, on most cases, a plan change. At least half the mistakes
of a case are **ruinous** (the dish is wrong if nobody intervenes); the rest cost time. Every mistake was verified to
fire and commit under three cooking orders with the scripted cook before it entered the suite
(`benchmark/cases_composite_v5_ledger.md` records every design decision).

The two kitchens per recipe differ in layout: the `medium` map keeps stations close; the `hard` map spreads them
out, so walking takes longer and a walk-away is harder to recover.

## 3. Personas

The cook's persona has two hidden dials, **skill** and **trust** in the assistant, and three questionnaire answers
the assistant is shown. Skill sets how competent the cook's own plan and hands are; trust sets whether it accepts a
correction without evidence or argues back. The formal runs use `baseline`, `classic_novice` and `classic_expert`
(150 episodes per seat per run); six more personas are registered. The table is in
[GENERATE_CASES.md](GENERATE_CASES.md#1-personas).

## 4. Metrics

Let T be the episode length in ticks, T_ref the reference length of a clean run of the cell, and C = 2·(T_ref +
allowance) the cap.

| metric | definition |
|---|---|
| **in-time success** | 1 if the correct dish was served and T ≤ C |
| **task success** | 1 if the correct dish was served, any time |
| **action efficiency (TT)** | clip((C − T) / (C − T_ref)) on won episodes |
| **conversation efficiency (CC)** | 1 − clip(T_interrupt / T), where T_interrupt is the time the assistant was speaking while the cook was mid-action |
| **prevention timing (DT)** | per fired mistake: 1.0 if the assistant's correct call prevented it, 0.9–0.5 if the call came inside the window before the mistake committed (scaled by latency), 0.3 after it committed, 0 if never named; averaged over the episode |
| **detection precision / recall / F1** | a call counts when it names the fired mistake inside its window or within 30 ticks after; every other flag is a false alarm |
| **QA accuracy** | the cook's questions, graded against the world state at the moment of asking (correct / partially correct / incorrect / evasive) |
| **proactive-alert accuracy** | standing requests: did the assistant alert at the right moment |
| **interaction quality** | every assistant line graded on five categories against the exact kitchen state, mean of independent judges: [INTERACTION_QUALITY_GRADING.md](INTERACTION_QUALITY_GRADING.md) |

`cooksim-score` writes all of them (`eval_v2.json`, `qa_regrade.json`, `proactive_v2.json`, `sq_v4/`).

## 5. Settings

* **polled** (every seat): the harness calls the assistant every 2 ticks with the last frames and the transcript.
* **native** (streaming APIs): nobody polls; frames go into a live session at up to 1 fps, the cook's lines arrive
  as audio, and the model decides itself when to speak. Both settings are reported for the streaming models.
* **observation**: rendered frames (default) or `agent+++`, a text state map, for models without vision.
* **cook-model ablation**: the same cases with a different model behind the cook ([REPRODUCE.md](REPRODUCE.md)).

## 6. What the assistant receives and returns

The system prompt carries the kitchen rules, the recipe, the persona's questionnaire answers and the flag
vocabulary; each turn carries the latest frames and the conversation since the last turn. The reply is JSON:

```json
{"text": "Stop, the cheese goes on after the meat.", "flag": {"error": "wrong_order"}}
```

`text` is what the cook hears (empty to stay silent); `flag.error` names a mistake from the vocabulary and is `null`
on every other turn. Naming the wrong mistake or flagging when nothing is wrong counts against the assistant.
