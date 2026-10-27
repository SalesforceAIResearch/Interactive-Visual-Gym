# The benchmark

VH-Home measures whether a visual assistant **helps a person who is doing a household chore**: it must notice
mistakes as they happen, name them in time, say the right thing at the right moment, and otherwise keep quiet.
Everything it says is checked against the simulator's ground truth. This page specifies the shipped benchmark:
the tasks, the plan grammar, the mistake classes, the cases, the personas, the cells, the episode protocol,
the flag verdicts, the metrics and the run protocol. The engine itself is described in [ENGINE.md](ENGINE.md);
making new cases in [GENERATE_CASES.md](GENERATE_CASES.md).

## 1. Anatomy of an episode

| piece | what it is |
|---|---|
| **tick** | the unit of simulated time: 1 second of animation, 5 rendered frames; 1.2 s of wall clock when a model's latency is charged |
| **person** | a Gemini model playing a persona; it plans the chore for itself from an empty queue and acts one command at a time through an egocentric, visibility-gated action space; it speaks rarely (under ten words, only when it needs something) |
| **assistant** | the model under test; every 2 ticks it receives the last 8 first-person frames plus two maps and the conversation, and may return a spoken line with an optional **flag** naming a mistake |
| **case** | one task plus a schedule of two or three mistakes of different classes, each anchored on a subgoal of the task's plan |
| **takeover** | when the live world reaches a mistake's anchor the engine takes the person's hands, executes the mistake, and shows the person a system task so it talks about what it is doing in character; the person never repairs a mistake on its own |
| **outcome** | `won`, `finished_wrong`, `stuck`, `timeout` or `silent_fail` (section 7) |

## 2. The tasks

Seventeen chores, each in one of VirtualHome's apartments (scenes 1 to 5). A task is a **setup** that leaves
things out of place, a reference **plan** (the step list the engine executes in a clean run), and programmatic
**checkpoints** that define success; the person is given only the instruction. `vhhome-case list tasks` prints
this table.

| task | apartment | steps | instruction |
|---|---|---|---|
| `relocate_hard` | 2 | 5 | Take a book and put it on the coffee table in the living room. |
| `appliance_hard` | 2 | 10 | Take the milk from the fridge, put it in the microwave, and switch the microwave on. |
| `power_hard` | 2 | 6 | Turn on a table lamp, the stove, and the TV. |
| `posture_hard` | 2 | 9 | Sit on a bed, a chair, and the sofa, standing up between each. |
| `transfer_hard` | 2 | 9 | Take the milk from the fridge and put it in the bedroom cabinet. |
| `search_hard` | 2 | 8 | Find each of these somewhere in the house and go up to it: a TV, a table lamp, the fridge, and the computer. |
| `tidy_hard` | 2 | 8 | Tidy up the house — switch off the lights on both table lamps, switch off the TV, and close the fridge. |
| `drink_hard` | 2 | 3 | Get a juice and drink it. |
| `routine_hard` | 2 | 19 | Turn off the bedroom lamp that is on, put the book that is lying on the bed on the living-room coffee table, heat the milk from the fridge in the microwave, and bring one of the plates from the kitchen table over to the coffee table. |
| `putback_hard` | 1 | 12 | In the living room, a toilet paper roll has been left on the desk, and a book and a pillow on the sofa; they belong in other rooms. Put the toilet paper roll on the bathroom counter, the book on the bedroom desk, and the pillow on the bed. |
| `dishes_hard` | 2 | 20 | Dirty dishes have been left around the flat: a plate and a glass on the kitchen table, a plate and a glass on the living-room coffee table, and a plate on the bed in the bedroom. Put all of them in the dishwasher and close it. |
| `groceries_hard` | 2 | 10 | The milk was left on the bedroom nightstand and the juice on the living-room coffee table. Put them both in the fridge, then close it. |
| `shutdown_hard` | 4 | 10 | Everything has been left on. Switch off all the table lamps, the TV and the computer, and close the fridge. |
| `mugs_hard` | 5 | 12 | Three mugs have been left lying around the flat — one in the bathroom, one in the living room and one on the bedroom nightstand. Collect all three and put them on the kitchen table. |
| `guest_hard` | 1 | 16 | A guest is coming and the living room was left as it was. Put the plate on the coffee table and the mug on the desk into the dishwasher, put the pillow from the sofa back on the bed, and switch the TV off. |
| `aftermeal_hard` | 3 | 20 | Clear up after the meal in the living room: the two plates on the coffee table and the mug on the side table go in the dishwasher, the apple on the side table goes back in the fridge, and close both. |
| `tidybedroom_hard` | 1 | 12 | Tidy up the bedroom: put the book that is lying on the bed back on the desk, turn off the lamp that was left on, close the cabinet, and take the mug from the nightstand to the kitchen table. |

The instructions are precise but minimal: an object is qualified ("kitchen table", "bedroom cabinet") only when
the person could confuse it with another instance; the route between rooms is never spelled out. Grading runs on
the objects the engine actually acted on, never on the words typed, and never depends on the rooms traversed.
Short tasks (two or three subgoals, such as `drink_hard`) cannot host two attributable mistakes and so carry no
shipped case; they remain available as clean runs.

## 3. Plans: the subgoal grammar and anchors

A task's step list is segmented, losslessly, into typed **subgoals** (`vhhome.plan`). Mistakes are attached to
subgoal ids, never to step indices or timers.

| subgoal | the steps it owns |
|---|---|
| `fetch(x)` | walk to x, grab x — or walk to the container, open it, grab x, close it |
| `deposit(x)` | walk to the surface, put x on it — or walk to the container, open it, put x in it, close it |
| `switch_on(d)`, `switch_off(d)` | walk to d, switch it |
| `open(c)`, `shut(c)` | walk to c, open or close it |
| `inspect(o)` | walk to o, turn to it (the search task) |
| `posture(s)` | walk to a seat, sit, stand up |
| `drink(x)` | drink x |
| `waypoint(o)` | a walk with no action on o |

A subgoal id names its object; `@N` picks the N-th instance of that class in the room (`switch_off(tablelamp@3)`),
and `#2` distinguishes a repeated subgoal on the same object (`fetch(mug@1)#2`, the second mug trip).
`vhhome-case plan <task>` prints a task's subgoals with the commands each one owns, for example `tidy_hard`:

```
subgoal id              commands
  switch_off(tablelamp@1) walk to the tablelamp; switch off the tablelamp
  switch_off(tablelamp@3) walk to the tablelamp; switch off the tablelamp
  shut(fridge)           walk to the fridge; close the fridge
  switch_off(tv)         walk to the tv; switch off the tv
```

The only orderings a task requires are data dependencies: `fetch(x)` before any later subgoal that uses x,
`deposit(x into a device)` before `switch_on(device)`, a sit before the stand that follows it. Everything else is
legal in any order, and the assistant is told so; doing independent subgoals in an unexpected order is not a
mistake.

## 4. The mistake classes

Six names, the only ones an assistant may put in a flag. Each class is a transform of the plan with a unique
observable signature; its timing and scope are parameters, not separate classes. `vhhome-case list mistakes`
prints the parameters.

| class | what the person does | what the assistant can see | parameters |
|---|---|---|---|
| `skipped_step` | never does a required subgoal and carries straight on with the next one; the omission's consequences follow (a later deposit refused empty-handed, a door left open) | the person starts the next subgoal without ever going to the omitted target; later, the consequence | `sid` the omitted subgoal; `pass_by` (default false) |
| `missing_step` | leaves out one of two twin subgoals (the same action on identical objects: the second lamp, the second glass); graded under the label `skipped_step` | one twin done, the other left as it was, and the person moving on | `sid` |
| `wrong_object` | takes a look-alike at a fetch (the juice for the milk), believes it, and uses it for the rest of the chore | the wrong thing in hand at the grab, then wherever the chore takes it | `sid` the fetch; `wrong` the look-alike; `room` |
| `wrong_placement` | sets the right object down on the wrong surface; the plan ends there, so nothing runs empty afterwards | the object going down somewhere the task did not say, and the person treating the job as finished | `sid` the deposit; `surf` the wrong surface; `room` |
| `repetition` | goes back to a finished switch, door or deposit and does it again; on a toggle the second press inverts the state (the tidied lamp is lit again, the shut fridge open again); on a deposit the object is taken out and put back | a revisit to a finished target and a real state change | `sid` the finished subgoal; `at` the subgoal after which it happens; `gap=delayed` or `immediate` |
| `stall` | stops moving mid-task: `transient` resumes on its own, `terminal` never does | the frames stop changing while a step is unfinished | `after`; `duration=transient` or `terminal`; `frames` (default 40) |
| `navigation_inefficiency` | a pointless detour before a subgoal: `explore` walks to the given stops in other rooms, `loiter` drifts around the current room; the plan then resumes intact | movement to places the chore has no business in, with nothing changing | `before`; `variant=explore` or `loiter`; `stops=obj@room,...`; `idle` frames paused at each stop (default 8; the shipped cases use 0 because a pause reads as a stall) |

Two families decide what the person is told while a mistake is live. **Commission** mistakes
(`skipped_step`, `missing_step`, `wrong_object`, `wrong_placement`) are named in the person's system task
("The mistake in play: you are picking up the juice where the milk belongs. You know it is wrong and you are doing
it anyway — that is the test."). **Attention** mistakes (`repetition`, `stall`, `navigation_inefficiency`) are
not named: the person is told only that it has lost track of something, and to check what the assistant says
against what it can see before judging it. In both families the person must say nothing about the mistake, must
not fix it itself, and must set `assistant_error_call` to `correct` once the assistant has named it (section 7).

Observation-channel **twins** are labels the grader credits for one another because the two are
indistinguishable in the frames at the decisive moment: `skipped_step` and `repetition` each with
`navigation_inefficiency`; `navigation_inefficiency` with `stall` and `skipped_step`; `stall` with
`navigation_inefficiency`; `wrong_placement` with `wrong_object`; `wrong_object` with nothing.

## 5. The cases

Twenty-five cases, each a task with two or three mistakes of different classes. A **story** is the one-sentence
human reading of the run, the seed for the plausibility audit and the caption on the results page.
`vhhome-case list cases` prints them; `vhhome-case show <case>` prints a case's mistakes and the plan it composes
to.

| case | task | mistakes | story |
|---|---|---|---|
| `cx_routine_skip_obj` | routine_hard | skipped_step + wrong_object | She leaves the bedroom lamp burning, does the book correctly, and then opens the fridge and takes the juice instead of the milk. |
| `cx_routine_skip_obj_stall` | routine_hard | skipped_step + wrong_object + stall | She leaves the lamp on, microwaves the juice instead of the milk, and then freezes with the plate in her hands before carrying it through. |
| `cx_routine_nav_place` | routine_hard | navigation_inefficiency + wrong_placement | On her way to the fridge she wanders over to the sofa and through to the bedroom, and at the end she leaves the plate on the kitchen counter instead of the coffee table. |
| `cx_routine_rep_obj` | routine_hard | repetition + wrong_object | Having switched the bedroom lamp off she goes back and switches it on again, and later takes the juice out of the fridge instead of the milk. |
| `cx_tidy_missing_rep` | tidy_hard | missing_step + repetition | She switches off one bedroom lamp but forgets its twin in the other bedroom, and after the television she goes back to the fridge she already shut and opens it again. |
| `cx_tidy_missing_stall` | tidy_hard | missing_step + stall | She forgets the second bedroom lamp, finishes the fridge and the television, and then stands motionless in the living room. |
| `cx_appliance_obj_stall` | appliance_hard | wrong_object + stall | She opens the fridge and takes the juice instead of the milk, loads it into the microwave, and then stands there doing nothing before starting the machine. |
| `cx_appliance_nav_place` | appliance_hard | navigation_inefficiency + wrong_placement | She wanders through the living room and bedroom before going to the fridge, then sets the milk down on the counter instead of putting it in the microwave. |
| `cx_search_skip_stall` | search_hard | skipped_step + stall | She never goes to look at the fridge at all, and after finding the computer she stops and stands still. |
| `cx_putback_skip_place` | putback_hard | skipped_step + wrong_placement | She never touches the toilet paper roll, carries the book to the bedroom and drops it on the bed instead of the desk. |
| `cx_putback_nav_place` | putback_hard | navigation_inefficiency + wrong_placement | After the toilet paper she wanders through the kitchen and the bedroom before coming back for the book, and then leaves the book on the nightstand instead of the desk. |
| `cx_dishes_missing_stall` | dishes_hard | stall + missing_step | She stands still in the living room with the coffee-table plate in her hands, and at the end forgets the glass on the kitchen table. |
| `cx_dishes_nav_place` | dishes_hard | navigation_inefficiency + wrong_placement | After the bedroom plate she wanders past the sofa and back to the bed before collecting the coffee-table dishes, and sets the last glass on the counter instead of in the machine. |
| `cx_groceries_stall_place` | groceries_hard | stall + wrong_placement | She picks up the milk in the bedroom and stands there holding it, and later leaves the juice on the kitchen table instead of putting it in the fridge. |
| `cx_groceries_skip_stall` | groceries_hard | skipped_step + stall | She never goes to the bedroom for the milk, picks up the juice from the coffee table and stands there holding it. |
| `cx_shutdown_missing_rep` | shutdown_hard | repetition + missing_step | After the television she goes back and opens the fridge she had just shut, and later switches off only one of the two table lamps. |
| `cx_shutdown_skip_stall` | shutdown_hard | stall + skipped_step | She shuts the fridge and stands motionless in front of it, and later walks past the computer without switching it off. |
| `cx_mugs_missing_stall` | mugs_hard | missing_step + stall | She brings the first mug back, skips the second entirely, and stands frozen holding the third. |
| `cx_mugs_nav_place` | mugs_hard | navigation_inefficiency + wrong_placement | Between the first and second mug she wanders past the sofa and the bed, and puts the third mug on the counter instead of the table. |
| `cx_guest_skip_stall` | guest_hard | skipped_step + stall | She loads the plate but never fetches the mug from the desk, and later stands frozen with the pillow in her hands. |
| `cx_guest_stall_place` | guest_hard | stall + wrong_placement | She freezes with the plate in her hands, and later sets the mug on the kitchen table instead of in the dishwasher. |
| `cx_aftermeal_missing_obj` | aftermeal_hard | missing_step + wrong_object | She clears only one of the two plates, and instead of the mug from the side table she takes a wine glass from the coffee table to the dishwasher. |
| `cx_aftermeal_stall_place` | aftermeal_hard | stall + wrong_placement | She stands still holding the second plate, and at the end leaves the apple on the kitchen table instead of in the fridge. |
| `cx_tidybedroom_skip_rep` | tidybedroom_hard | skipped_step + repetition | She leaves the cabinet open, switches the lamp off, and when she reaches for the mug she turns back and switches the lamp on again. |
| `cx_tidybedroom_stall_place` | tidybedroom_hard | stall + wrong_placement | She freezes with the book in her hands, and finally puts the mug on the kitchen counter instead of the kitchen table. |

### The composition rules

Mistakes compose in plan order under a priority rule: an earlier mistake owns the plan, and a later one whose
anchor it removed is skipped with a record. A multi-mistake episode is only meaningful if the mistakes do not
collide, so every case must pass these rules (`vhhome_bench.case_checks`; `python tests/verify_cases.py`
re-checks the shipped set and writes `benchmark/composite_ledger.json`, 25 of 25 admitted):

| rule | requirement |
|---|---|
| R1 direct-twin exclusion | no two mistakes that are observation twins of each other (section 4): a correct flag for one would be creditable for the other and the episode could not say which was seen |
| R2 anchor survival | every declared mistake is still in the composed plan; nothing may be anchored after a `wrong_placement` (it truncates the plan) or on a subgoal an omission removes, and an omission needs a surviving successor subgoal to anchor its detection window on |
| R3 window separation | anchors are distinct subgoals separated by at least one surviving subgoal, so detection windows do not overlap and each flag attributes to exactly one mistake |
| R4 independent observability | no two mistakes act on the same object, and an earlier mistake must not make a later one unobservable |
| R5 semantic plausibility | the combination reads as one person having a plausibly bad run; this is a human and VLM audit (`benchmark/composite_vlm_audit.json`), not a mechanical check |
| R6 visible state | a repetition's inverting press must change the picture: lamps, the TV and doors do; a microwave, dishwasher or stove switch renders identically either way and is refused |
| R7 the misplacement must move something | a wrong placement onto the same kind of surface the object started on is a lift-and-replace, and is refused |

The checker also requires that the composed plan executes in the symbolic twin (an injected command the house
would refuse can never manifest), that a case holding an outcome-breaking class actually breaks the task, and
that each mistake adds at least one command the clean plan lacks, so it is distinguishable from ordinary
execution.

## 6. Personas

The person's persona has five dials. The first two are private traits ("nobody asked you this"); the last three
are the person's answers to a pre-task questionnaire, and the assistant is shown those answers and nothing else
about the person. `vhhome-case list personas` prints the table.

| persona | knows this house | trusts the assistant | speed | alerting wanted | interruption wanted |
|---|---|---|---|---|---|
| `credulous_expert` | high | high | parallel | aggressive | medium |
| `classic_expert` | high | low | soso | confident | none |
| `quiet_perfectionist` | high | medium | safe | confident | none |
| `classic_novice` | low | high | safe | aggressive | stepwise |
| `defiant_novice` | low | low | soso | medium | stepwise |
| `eager_parallel_novice` | low | medium | parallel | confident | medium |
| `safety_stepper` | medium | high | safe | aggressive | stepwise |
| `speed_demon` | medium | low | parallel | medium | none |
| `baseline` | medium | medium | soso | medium | medium |

What each dial changes:

* **Knows this house** (familiarity, not housework skill; everyone can do the chore). `low`: the person does not
  know where things are kept in this house and has to look around, and often ask, to find them. `medium`: most
  things straight away, a few take a moment. `high`: goes straight to what it needs.
* **Trusts the assistant**. Decides how fast the person comes around to a correct claim about a live mistake:
  `high` accepts the first correct warning; `low` only once it sees the mistake with its own eyes or in its own
  hands; `medium` treats the assistant as potentially helpful but not automatically right.
* **Speed** (questionnaire: "How much do you care about finishing the chore fast?"): `safe` = safe first, `soso`,
  `parallel` = as soon as possible, with a lot of parallelism. It sets the pace the person describes itself with.
* **Alerting wanted** (questionnaire: "How do you want the assistant to alert potential mistakes?"):
  `confident` = flag only when confident, `medium`, `aggressive` = spot all possible mistakes aggressively.
* **Interruption wanted** (questionnaire: "What interruption style do you want?"): `stepwise` = step by step,
  `medium`, `none` = don't interrupt unless necessary. `none` also makes the person avoid talk and raises the
  weight of conversation efficiency in the EFF metric (section 9).

## 7. The cells

The formal benchmark plays 75 fixed **cells**, each a case and a persona (`benchmark/cells.json`): the 25 cases
with 3 distinct personas each, drawn once at random with seed 20260920 and balanced so every persona appears
in 8 or 9 cells. Every assistant runs the same 75 cells. `benchmark/clean_cells.json` holds one clean cell per
case (the same task, no mistakes, silent assistant); the median clean completion time per case, measured in
engine ticks, is `benchmark/t_ref.json`, the reference time the success cap uses.

## 8. The episode protocol

**Polling.** The assistant is called every `POLL = 2` ticks. Each call carries the system prompt (the task
instruction, the plan's subgoal names and true order constraints, the six mistake definitions, the person's
questionnaire answers, the command vocabulary the person's body accepts, the reply schema), then the last
`WINDOW = 8` first-person frames with a whole-house map and a map of the current room, the tick, the
conversation so far and the assistant's own previous reply (its only memory). The reply is JSON
(`vhhome/data/assistant_reply_schema.json`):

```json
{"human_progress": {"subgoals_completed": [...], "current_subgoal": {...}, "subgoals_future": {...}},
 "key_moments": ["t12 she took the milk out of the fridge"],
 "text": "Stop — that is the juice, not the milk.",
 "flag": {"error": "wrong_object"}}
```

Only `text` reaches the person (empty to stay silent). `flag.error` must be exactly one of the six names and
is how a mistake claim is graded; `flag` is `null` on every other turn. A wrong name, or a flag when nothing is
live, counts against the assistant.

**Latency.** The assistant's reply is not free. A reply that took x wall-seconds lands `ceil(x / 1.2)` ticks after
the tick it was asked on (`ASSIST_TICK_S = 1.2`), capped at `ASSIST_MAX_TICKS = 15`; the person keeps moving
meanwhile, and the next call starts only when the reply has landed. Only the successful attempt's own generation
time is charged (API stalls and retries are not), and for locally served models the charged time can be divided
by a measured hardware ratio (`benchmark/gpu_latency_ratio.json`, `ASSIST_LATENCY_DIV`) so every run is charged
comparable seconds across cards. The published numbers served every open-weight model on the same card type and
applied no factor (`ASSIST_LATENCY_DIV = 1`). `ASSIST_LATENCY=0` makes the
assistant instantaneous, an ablation.

**Takeover and release.** The person works down the clean plan at its own pace. Each mistake waits for its
**live anchor**: for `wrong_object`, `skipped_step`, `missing_step` and `wrong_placement`, one of the person's
next two queued actions aims at the anchored subgoal; for `navigation_inefficiency`, one of the next two aims at
the subgoal it comes before; for `stall`, the action that completes the anchored subgoal has just finished; for
`repetition`, the repeated subgoal was finished at least 30 ticks ago and the next two actions aim at the `at`
subgoal (or the chore is one step from finishing). Nothing fires on a timer; a mistake whose moment never comes
is skipped at its deadline with the reason recorded. When the anchor is reached the engine replaces the person's
queue with the mistake's action sequence and shows the person a `[SYSTEM TASK]` block, so it talks about what
it is doing in character and never fixes the mistake itself. The block is lifted, and the hands come back, when
the person is convinced by a correct assistant claim (`released_reason = convinced`), when the sequence has
been carried out (`task_completed`), or when the queue is exhausted (`block_exhausted`); a correct claim the
person accepts before the mistake has physically happened releases it as `prevented`.

**The person never self-repairs.** A standing rule in every turn tells the person that a mistake is the
assistant's failure to catch, not the person's to hide: it says nothing about the mistake, fixes only what the
assistant has told it is wrong, and otherwise finishes the chore as it is. Its reply carries a private field
`assistant_error_call` (`none` | `correct` | `wrong`): its own judgement of whether the assistant has named the
live mistake. In the benchmark setting (`VH_GRADER=cook`, the default) that judgement, not the engine's flag
verdict, is what arms the release; the engine's verdict is recorded beside it.

**Outcomes.** An episode ends `won` when every checkpoint is true and the person's queue is empty;
`finished_wrong` when the person has finished its order with a mistake nobody talked it out of and a checkpoint
is false; `stuck` after `VH_NO_PROGRESS = 60` ticks with no change of room, hands or world state;
`silent_fail` after ten consecutive polls in which nobody spoke and nothing was queued (or when the assistant
repeats the same line three times while the house stands still); `timeout` at the tick cap (`STEPIN_CAP`, 520 in
the benchmark) or the wall-clock budget (`STEPIN_WALL`, 5400 s).

## 9. Flag verdicts

Every flag is graded against every mistake whose deviation has fired, and credited to the one that grades it
best:

| verdict | meaning |
|---|---|
| `valid` | the right name inside the mistake's window: a detection |
| `valid_alias` | a twin name inside the window: a detection |
| `early_correct`, `early_alias` | the right name (or a twin) before the window opens, even before the mistake physically happened: a prevention if the person accepts it in time |
| `late` | the right name after the window closed |
| `wrong_name` | a name that is neither the live mistake nor one of its twins |
| `invalid_label` | not one of the six names |
| `fp_preinjection`, `fp_clean`, `fp_skipped` | a flag when no mistake has fired, in a clean run, or in a run whose mistake was skipped: a false alarm |
| `neutral_true` | a claim that is true of the world without being a forced mistake (the person really is standing still or wandering): never charged |

Detection is credited to the earliest of the engine's verdict and the person's own `correct` call. Repeated
flags with the same name count once.

## 10. Metrics

Let T be the episode length in ticks, T_ref the measured clean completion time of the case
(`benchmark/t_ref.json`) and C = 2 (T_ref + 20) the cap (`ALPHA = 20`). `vhhome-score` computes everything from
the records (`vhhome_bench.scoring.metrics`) and reports each arm as the mean over the independent runs, with
plus or minus half the range between runs.

| metric | definition |
|---|---|
| **won** | the share of episodes whose outcome is `won` |
| **S**, in-time success | 1[won] · 1[T ≤ C]: finishing is not enough, it must finish inside twice the reference time plus the allowance |
| **TT**, time efficiency | 0 if T > C, else clip((C − T) / (C − T_ref)) to [0, 1]: the share of the budget left |
| **CC**, conversation efficiency | 1 − clip(2 U / T), where U is the number of assistant turns with a non-empty line (each occupies one 2-tick poll) |
| **EFF**, efficiency | (2 TT + c CC) / (2 + c) over episodes with S = 1 only; c = 3 when the persona answered "don't interrupt unless necessary", else 1 |
| **DT**, detection timing | mean over the fired mistakes of an episode: 1 if prevented, 0.5 if correctly flagged after it landed, 0 if missed |
| **P, R, F1** | over flags: a true positive is a flag credited as a detection or prevention; every other flag that is not neutral is a false alarm; P = TP / (TP + FP), R = TP / fired, F1 their harmonic mean |
| **DET**, detection | mean(DT, F1): prevention is worth twice a late catch, and an assistant that flags everything is refused the credit DT alone would give |
| **AQ**, interaction quality | the per-round rubric ([INTERACTION_QUALITY_GRADING.md](INTERACTION_QUALITY_GRADING.md)): one judge call per conversation round, silence not graded, round score = passed weight / applicable weight, a failed "must" item zeroes the round; episode AQ is the mean over its rounds, and the arm's AQ is pooled over episodes with a 2000-draw bootstrap 95% interval |

EFF is conditional on success, so its n is smaller than the arm's episode count; the n is printed for that
reason.

## 11. The run protocol and the published table

The published protocol is three independent runs of the 75 cells per assistant (225 episodes):

```bash
vhhome-benchmark --assistant <spec> --out runs/mine --repeat 1
vhhome-benchmark --assistant <spec> --out runs/mine --repeat 2
vhhome-benchmark --assistant <spec> --out runs/mine --repeat 3
vhhome-score --run runs/mine/r1 runs/mine/r2 runs/mine/r3
```

`vhhome-score` grades the objective metrics, runs the rubric judges (`--judges gemini,gpt` is the reported
setting; `all` adds the Claude judge), and writes the pooled `final_results.{json,md}` beside the first run root.
The person model is part of the measurement: keep `GEMINI_MODEL` fixed when comparing assistants.

The published table (`results/formal_run_2.md`, data in `results/data/formal_run_2.json`) covers 19 seats on
this protocol: three polled API models, two streaming APIs in both their polled and native settings, six
open-weight vision-language models, five duplex or proactive open models, and the silent control that never
speaks. In that table the best AQ is 0.49 (`gpt-6-astra`), won ranges from 0.05 to 0.36 across the speaking
seats, in-time success S is at most 0.11, and DET is at most 0.57 (`gemini-3.1-pro-preview`). Scores on custom
cases are comparable across assistants on the same cases, not with the published table.

Known limits, measured and kept honest: navigation and visibility still end some episodes before the first
mistake fires; the mistake trigger assumes the clean plan's order, so a person working in its own order can
finish without ever reaching the trigger state; VirtualHome has no consequence dynamics (no smoke, fire or
spoilage), so hazard-type mistakes are not in the taxonomy; and assistants cannot localise the person from the
pictures alone, which is why they receive the maps.
