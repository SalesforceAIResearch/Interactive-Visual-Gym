# Generating new episodes

An episode is **a case** (one order on one kitchen with a schedule of events), played by **a persona** (the simulated
cook), with **an assistant** in the seat. All three are chosen independently, so there are three ways to make new
episodes: pick a different persona for an existing case, change the schedule of a case, or both.

```bash
cooksim-case list                                     # cells, personas, verified mistakes, question kinds
cooksim-case show b3_medium_nops_medium_map_3         # what a case schedules
cooksim-case derive --from b3_medium_nops_medium_map_3 --name burger_q \
    --add-question next_step --out my_cases.json      # a new case
cooksim-case verify my_cases.json                     # does every forced mistake fire, in every cooking order?
cooksim-episode --latency 2 --cases my_cases.json --case burger_q --persona classic_novice \
    --assistant gemini:gemini-3.8-flash --out runs/burger_q
```

## 1. Personas

The persona is an episode-level choice, `--persona`. It sets the cook's skill and trust in the assistant, and three
questionnaire answers that the assistant is shown (the assistant knows the answers, not the hidden skill and trust):

| persona | skill | trust | questionnaire |
|---|---|---|---|
| `baseline` | medium | medium | 2, 2, 2 |
| `classic_novice` | low | high | 1, 3, 1 |
| `classic_expert` | high | low | 2, 1, 3 |
| `defiant_novice` | low | low | 2, 2, 1 |
| `credulous_expert` | high | high | 3, 3, 2 |
| `speed_demon` | medium | low | 3, 2, 3 |
| `safety_stepper` | medium | high | 1, 3, 1 |
| `quiet_perfectionist` | high | medium | 1, 1, 3 |
| `eager_parallel_novice` | low | medium | 3, 1, 2 |

The three questions (`cooksim/interact/persona_v2.py`) ask how much guidance the person wants, how they react to being
corrected, and how they feel about interruptions; each answer is one of three options. The formal runs use the first
three personas. To add a persona, append a `PersonaV2(skill, trust, q1, q2, q3, name=...)` entry to `ROSTER` in that
file; `cooksim-roster --personas` lists what is registered.

## 2. Cases

A case names a **cell** (recipe tier × map, `benchmark/grid.json`) and a list of **sub-cases** that the engine fires in
order. Each sub-case waits for its **anchor**, a condition on the live kitchen (what is prepped, what is plated, what is on
the heat), never on a timer, so the same case plays out at the cook's own pace. The engine arms sub-case k+1 only after
sub-case k has released.

| family | what happens | example from the suite |
|---|---|---|
| `error` | the engine takes over the cook's hands and makes the mistake; the cook's voice defends it until the assistant names it or it commits | `wrong_order(a=bun, b=cheese)`: the cheese goes on the plate before the bun |
| `qa_probe` | the cook asks the assistant a question about the kitchen | "What is my next step?" |
| `qa_proactive` | a standing request the assistant must act on later | "Give me a shout the moment the meat is done" |
| `plan_change` | the cook announces a change of plan and the assistant must adapt | "I'll do the lettuce on the other board" |

### Mistake types

| `err` | the cook ... | ruinous? |
|---|---|---|
| `wrong_order` | plates item b before item a | yes: the dish is served wrong |
| `serve_raw` | plates an item before it is prepared or cooked | yes |
| `add_foreign` / `misidentify_assemble` | scoops an off-recipe ingredient onto the plate | yes |
| `omit_subgoal` (`premature_serve`) | carries the plate to the pass with items missing | yes, and it ends the episode |
| `repeat_step` | plates an item twice | yes |
| `leave_unattended` | walks away from food on the heat until it smokes | yes if it burns |
| `mis_prep` | preps an off-recipe item for a component | no: the cook recovers; time is lost |
| `forget_next_step` | stalls and forgets the next step | no: time is lost |

A forced mistake is only usable where it is **verified**: the suite ships an inventory (`benchmark/error_inventory_v5.json`)
of mistakes that were replayed on each cell with the scripted cook and shown to fire and commit. `cooksim-case list`
prints, per cell, the mistakes you can add; `--add-error <err>[:<component>]` picks one.

### Composing a case

`cooksim-case derive` starts from a suite case and edits its schedule:

| flag | effect |
|---|---|
| `--from <case>` `--name <new>` | the base case and the new name (every sub-case is renamed with it) |
| `--drop <err or name suffix> ...` | remove sub-cases |
| `--add-error <err>[:<component>] ...` | add a verified mistake of the same cell (its anchors come from the inventory) |
| `--add-question <kind> ...` | add a question kind that exists on this cell (`next_step`, `pan_contents`, ...); the verified question of another case of the cell is reused |
| `--add-request <item> ...` | add "tell me when `<item>` is done" |
| `--order <names> ...` | force the fire order |
| `--out <file>` | the case file (several cases can accumulate in one file) |

Without `--order`, sub-cases are ordered by their **moment**: how far the dish is when the mistake fires, read off the
designer's reference run; questions keep their place relative to the mistakes they followed; a `premature_serve`
always comes last because it ends the episode. Two rules the suite designer enforces are worth keeping in mind:
at most one mistake per plate slot, and no two forced mistakes at adjacent moments (a takeover needs room to
release before the next one arms).

### Verifying

```bash
cooksim-case verify my_cases.json [--policies template,plate_early,reverse] [--only burger_q]
```

Replays each case with the **scripted cook**, a rule-following cook with no model behind it, under each legal cooking
order (`template`: the reference order; `plate_early`: takes the plate first; `reverse`: the components in reverse
order), with the assistant silent. The verdict is the suite's own acceptance rule: **every forced mistake fires and
commits under every cooking order**. It needs no API key and no browser, and takes a minute or two per case.

Questions, requests and plan changes are listed as `n/a`: the scripted cook has no voice, so they are exercised only
by the LLM cook; their anchors were verified when the suite was built and are reused unchanged.

When a mistake is reported as `skipped(anchor_never_reached ...)` under some order, the schedule does not fit that
order: typically the anchor asks for food on the heat that this order has not started yet. Move the mistake, pick
another, or restrict the case to the orders where it fires and say so when you report results.

## 3. The case file

```json
{"schema": "cooksim-cases", "cases": [
  {"name": "burger_q", "cell": "medium_nops_medium_map", "family": "composite", "n_errors": 2, "n_ruinous": 2,
   "derived_from": "b3_medium_nops_medium_map_3",
   "subs": [ ...sub-cases in fire order... ]}]}
```

An **error** sub-case:

| field | meaning |
|---|---|
| `name`, `cell`, `family: "error"` | identity |
| `err` | the mistake label the assistant must name (`wrong_order`, ...) |
| `fn`, `kwargs` | the engine function that forces it and its arguments (`component`, `a`/`b`, `after`) |
| `prep_floor`, `plate_floor` | the anchor: at least these items prepped, at least this many items on the plate |
| `dev`, `identifiable`, `window` | reference ticks from the inventory: when the deviation starts, when it becomes identifiable, the window in which naming it counts as in time |
| `outcome`, `max_smoke` | what the mistake does to the dish if nobody intervenes |
| `ruinous`, `robust_ruinous` | whether the served dish is wrong after it commits |
| `lu_late` | (walk-away only) how late after the food is done the walk-away may still start |

A **question** (`qa_probe`) or **request** (`qa_proactive`) sub-case:

| field | meaning |
|---|---|
| `question` | the cook's line |
| `qa_family` | the question kind (Q1–Q10) used in scoring |
| `gt_kind`, `expected_gt` | what a correct answer must say, read from the world when the question fires |
| `prep_floor`, `trigger_after`, `ask_tick_clean` | the anchor and the reference tick |
| `item` | (request) the item to watch |

A **plan change** sub-case carries `intent_label`, the cook's announcement, and the new goal (`goal_extend`).

## 4. Running and scoring your cases

`cooksim-episode --cases my_cases.json --case <name> ...` runs one; `cooksim-benchmark --suite my_cases.json ...` runs a
whole file with N workers and all personas; `cooksim-score --run <run>` grades the records exactly as the suite is graded.
Scores on custom cases are comparable across assistants on the same cases, not with the published leaderboard.
