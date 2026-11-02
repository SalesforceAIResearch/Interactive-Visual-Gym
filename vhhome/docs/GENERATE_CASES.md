# Generating new episodes

An episode is **a case** (an existing household task plus a schedule of mistakes), played by **a persona** (the
simulated person), with **an assistant** in the seat. The three are chosen independently, so there are three ways
to make new episodes: a different persona on an existing case, a new schedule of mistakes on an existing task, or
both. A brand-new task is possible too, and is covered last.

```bash
vhhome-case list                                      # tasks, cases, personas, mistake classes
vhhome-case plan tidy_hard                            # the subgoals a mistake can be anchored on
vhhome-case new --name my_tidy_1 --task tidy_hard \
    --mistake "skipped_step sid=switch_off(tablelamp@1)" \
    --mistake "repetition sid=shut(fridge) at=switch_off(tv) gap=delayed" \
    --story "She walks straight past the first bedroom lamp, and after the television she goes back and opens the fridge she had already shut." \
    --out my_cases.json                               # checks the composition rules, writes the case
vhhome-case verify my_cases.json                      # re-check a case file
vhhome-case --cases my_cases.json show my_tidy_1      # the plan the case composes to
vhhome-episode --latency 2 --cases my_cases.json --case my_tidy_1 --persona classic_novice \
    --assistant gemini:gemini-3.8-flash --out runs/mine
```

Every mistake is a transform of the task's plan, anchored on one of its **subgoals**, that the engine forces at
the moment the person reaches that subgoal. A case names no persona; the persona is picked when the episode runs.

## 1. Looking around: `list` and `plan`

`vhhome-case list` prints the tasks with their instruction, the shipped cases with their stories, the personas
with their five dials, and the mistake classes with their parameters. The last block, which is what `new` needs:

```
MISTAKE CLASSES and their parameters (anchors are subgoal ids from `vhhome-case plan <task>`)
  skipped_step             sid pass_by=False
      the person never does that subgoal and carries on with the next one
  missing_step             sid
      one of two twin subgoals is left out (the task names two identical targets)
  wrong_object             sid wrong room
      the person takes the look-alike `wrong` instead and uses it for the rest of the chore
  wrong_placement          sid surf room
      the object is put down on `surf` in `room` instead of where the task says; the plan ends there
  repetition               sid at=None gap='delayed'
      an already finished switch/door/deposit subgoal is done a second time (`at`: after which subgoal)
  stall                    after duration='terminal' frames=40
      the person freezes for `frames` frames after that subgoal (transient) or never resumes (terminal)
  navigation_inefficiency  before variant='explore' stops=None idle=8
      a detour through `stops` (obj@room,...) before that subgoal
```

`vhhome-case plan <task>` prints the task's subgoals, the commands each one owns, where the person starts, and
the setup that left things out of place. The subgoal ids in the first column are the anchors:

```
$ vhhome-case plan tidybedroom_hard
tidybedroom_hard: TASK: Tidy up the bedroom: put the book that is lying on the bed back on the desk, turn off the lamp that was left on, close the cabinet, and take the mug from the nightstand to the kitchen table.

subgoal id              commands
  fetch(book)            walk to the book; grab the book
  deposit(book)          walk to the desk; put the book on the desk
  shut(cabinet)          walk to the cabinet; close the cabinet
  switch_off(tablelamp@1) walk to the tablelamp; switch off the tablelamp
  fetch(mug)             walk to the mug; grab the mug
  deposit(mug)           walk to the kitchentable; put the mug on the kitchentable

the person starts in the kitchen; the task's setup (what is out of place):
   walk to the book
   grab the book
   walk to the bed
   put the book on the bed
   walk to the mug
   grab the mug
   walk to the nightstand
   put the mug on the nightstand
   walk to the tablelamp
   switch on the tablelamp
   walk to the cabinet
   open the cabinet
```

`@N` in an id picks the N-th instance of a class in the room (`switch_off(tablelamp@3)` in `tidy_hard` is the
second bedroom's lamp); `#2` marks a repeated subgoal on the same object (`fetch(mug@1)#2` in `mugs_hard` is the
second mug trip). The grammar itself is in [BENCHMARK.md](BENCHMARK.md#3-plans-the-subgoal-grammar-and-anchors).

## 2. The mistake parameters, with one verified example each

A `--mistake` argument is the class name followed by `key=value` pairs; `stops` takes a comma-separated list of
`object@room`. Mistakes are given in plan order. Every example below was written with `vhhome-case new` and
admitted by the checks; the full commands are in section 3.

| class | grammar | example (task) | what happens in the composed plan |
|---|---|---|---|
| `skipped_step` | `skipped_step sid=<subgoal> [pass_by=true]` | `skipped_step sid=switch_off(tablelamp@1)` (`tidy_hard`) | the subgoal is deleted; the person does the next one instead; `pass_by=true` makes it walk past the omitted target first |
| `missing_step` | `missing_step sid=<one of two twin subgoals>` | `missing_step sid=switch_off(tablelamp@1)` (`shutdown_hard`, whose two lamps are `switch_off(tablelamp@1)` and `switch_off(tablelamp@1)#2`) | the named twin is deleted; graded as `skipped_step` |
| `wrong_object` | `wrong_object sid=<fetch subgoal> wrong=<look-alike class> room=<its room>` | `wrong_object sid=fetch(milk) wrong=juice room=kitchen` (`routine_hard`) | the fetch becomes `wrong_grab(juice)` (keeping the fridge open/close), and every later command on the milk is rewritten to the juice |
| `wrong_placement` | `wrong_placement sid=<deposit subgoal> surf=<surface class> room=<its room>` | `wrong_placement sid=deposit(mug) surf=kitchencounter room=kitchen` (`tidybedroom_hard`) | the deposit becomes `put_wrong(mug)` and the plan is cut off after it |
| `repetition` | `repetition sid=<finished switch/door/deposit> at=<later subgoal> gap=delayed` or `gap=immediate` (no `at`) | `repetition sid=shut(cabinet) at=switch_off(tablelamp@1) gap=delayed` (`tidybedroom_hard`) | a `redo(shut(cabinet))` node is inserted after `at`: walk back, open the cabinet, stand still; a deposit would be taken out and put back |
| `stall` | `stall after=<subgoal> duration=transient\|terminal frames=<n>` | `stall after=fetch(mug@1) duration=transient frames=60` (`mugs_hard`) | a `lost` node (stand still for n frames; 5 frames = 1 tick) is inserted after the subgoal; `terminal` drops the rest of the plan and never resumes |
| `navigation_inefficiency` | `navigation_inefficiency before=<subgoal> variant=explore\|loiter stops=<obj@room,...> idle=<frames>` | `navigation_inefficiency before=fetch(juice) variant=explore stops=sofa@livingroom,bed@bedroom idle=0` (`groceries_hard`) | a `search` node (walk to each stop) is inserted before the subgoal; `loiter` drifts between same-room stops with pauses and glances; use `idle=0` with `explore`, because standing at each stop reads as a stall, which is a twin |

Anchors must come from `vhhome-case plan` for that task. Which look-alikes and surfaces exist in an apartment is
a scene question: the shipped cases use `juice` for `milk` and `wineglass` for `mug` as look-alikes, and
`kitchencounter`, `kitchentable`, `nightstand` and `bed` as wrong surfaces; `vhhome-case show` on a shipped case
of the same task is the quickest guide.

## 3. Writing a case: `new`

`vhhome-case new` checks the schedule against the composition rules and, if admitted, appends the case to the
`--out` file (creating it if needed; a case of the same name is replaced). The two cases shipped as
`examples/my_cases.json`:

```
$ vhhome-case new --name my_tidy_1 --task tidy_hard \
    --mistake "skipped_step sid=switch_off(tablelamp@1)" \
    --mistake "repetition sid=shut(fridge) at=switch_off(tv) gap=delayed" \
    --story "She walks straight past the first bedroom lamp, and after the television she goes back and opens the fridge she had already shut." \
    --out my_cases.json
my_tidy_1: admitted — admitted
wrote my_cases.json  (1 case(s)); run it with: vhhome-episode --cases my_cases.json --case my_tidy_1 ...

$ vhhome-case new --name my_bedroom_1 --task tidybedroom_hard \
    --mistake "repetition sid=shut(cabinet) at=switch_off(tablelamp@1) gap=delayed" \
    --mistake "wrong_placement sid=deposit(mug) surf=kitchencounter room=kitchen" \
    --story "After switching off the lamp she goes back and opens the cabinet she had just closed, and at the end she leaves the mug on the kitchen counter instead of the kitchen table." \
    --out my_cases.json
my_bedroom_1: admitted — admitted
wrote my_cases.json  (2 case(s)); run it with: vhhome-episode --cases my_cases.json --case my_bedroom_1 ...
```

The other examples of section 2, each admitted the same way:

```bash
vhhome-case new --name doc_missing --task shutdown_hard \
    --mistake "missing_step sid=switch_off(tablelamp@1)" \
    --mistake "stall after=switch_off(tv) duration=transient frames=60" \
    --story "She switches off the television and then stands still in front of it, and later switches off only one of the two table lamps." \
    --out my_cases.json
vhhome-case new --name doc_wrongobj --task routine_hard \
    --mistake "wrong_object sid=fetch(milk) wrong=juice room=kitchen" \
    --mistake "navigation_inefficiency before=fetch(plate) variant=explore stops=sofa@livingroom,bed@bedroom idle=0" \
    --story "She takes the juice out of the fridge instead of the milk and microwaves it, then wanders past the sofa and the bed before coming back for the plate." \
    --out my_cases.json
vhhome-case new --name doc_wrongplace --task groceries_hard \
    --mistake "navigation_inefficiency before=fetch(juice) variant=explore stops=sofa@livingroom,bed@bedroom idle=0" \
    --mistake "wrong_placement sid=deposit(juice) surf=kitchentable room=kitchen" \
    --story "After the milk she wanders through the living room and the bedroom before fetching the juice, and then leaves the juice on the kitchen table instead of in the fridge." \
    --out my_cases.json
vhhome-case new --name doc_stall --task mugs_hard \
    --mistake "stall after=fetch(mug@1) duration=transient frames=60" \
    --mistake "wrong_placement sid=deposit(mug@1)#3 surf=kitchencounter room=kitchen" \
    --story "She picks up the bathroom mug and freezes with it in her hands, and at the end puts the third mug on the counter instead of the table." \
    --out my_cases.json
```

The `--story` is one sentence saying how the run reads to a viewer. It is the seed of the plausibility audit
(rule R5, which no replay can check) and the caption of the case on a results page, so write it as a person
having a plausibly bad run, not as two unrelated glitches.

### A rejected case

A mistake on the **last** subgoal of a task cannot be an omission: there is no later subgoal left for the
person to do instead, so nothing anchors the detection window.

```
$ vhhome-case new --name doc_rejected --task tidy_hard \
    --mistake "stall after=shut(fridge) duration=transient frames=60" \
    --mistake "skipped_step sid=switch_off(tv)" \
    --story "She shuts the fridge, stands still, and never switches the television off." \
    --out my_cases.json
doc_rejected: REJECTED — R2 — trailing omission of switch_off(tv): no successor subgoal survives to anchor a detection window on
not written (use --force to write it anyway)
```

Two mistakes that are observation twins are refused too, because a correct flag for one would be creditable
for the other and the episode could not say which the assistant saw:

```
$ vhhome-case new --name doc_twins --task tidy_hard \
    --mistake "stall after=switch_off(tablelamp@1) duration=transient frames=60" \
    --mistake "navigation_inefficiency before=switch_off(tv) variant=explore stops=sofa@livingroom,bed@bedroom idle=0" \
    --story "She freezes after the first lamp and later wanders before the television." \
    --out my_cases.json
doc_twins: REJECTED — R1 direct-twin exclusion — stall and navigation_inefficiency are twins, a correct flag for one is creditable for the other
not written (use --force to write it anyway)
```

`--force` writes a rejected case anyway (it is marked `"admitted": false` in the file); use it only to study a
rejection, never for a case you intend to score.

### What the checks reject, and why

The checker (`vhhome_bench.case_checks`) replays the composed plan through the symbolic twin and applies the
rules of [BENCHMARK.md](BENCHMARK.md#the-composition-rules). In practice a rejection says one of:

| message begins | cause | fix |
|---|---|---|
| `R1 direct-twin exclusion` | two classes the grader credits for one another (`stall` with `navigation_inefficiency`, `skipped_step` or `repetition` with `navigation_inefficiency`, `wrong_placement` with `wrong_object`) | pick a class from the other family |
| `R2 anchor survival` | a mistake's anchor was removed by an earlier one (anything after a `wrong_placement`, anything on an omitted subgoal), or an omission has no surviving successor | reorder, or anchor elsewhere; `wrong_placement` must be last |
| `R3 window separation` | the anchors are adjacent, or identical | leave at least one untouched subgoal between them |
| `R4 independent observability` | two mistakes act on the same object | choose different targets |
| `R6 visible state` | a repetition on a microwave, dishwasher or stove switch, which renders the same either way | repeat a lamp, the TV or a door instead |
| `R7 the misplacement must move something` | the wrong surface is the same kind the object started on | choose a surface of another kind |
| `R8 per-phase distinctiveness` | a mistake adds no command the clean plan lacks | change its parameters so it is visible |
| `not executable` | an injected command the house would refuse (grabbing from a closed container, putting down something not held) | check the plan order with `show` |
| `the composite does not change the outcome` | an outcome-breaking class is present but the task still completes | anchor it on a subgoal the checkpoints depend on |

`vhhome-case verify my_cases.json` re-checks every case in a file (exit status 1 if any is rejected), and
`vhhome-case --cases my_cases.json show <name>` prints the composed plan, marking the nodes a mistake changed:

```
$ vhhome-case --cases my_cases.json show my_bedroom_1
my_bedroom_1  task=tidybedroom_hard
  After switching off the lamp she goes back and opens the cabinet she had just closed, and at the end she leaves the mug on the kitchen counter instead of the kitchen table.
mistakes:
  repetition               {"sid": "shut(cabinet)", "at": "switch_off(tablelamp@1)", "gap": "delayed"}
  wrong_placement          {"sid": "deposit(mug)", "surf": "kitchencounter", "room": "kitchen"}

composed plan (* = changed by a mistake):
    fetch(book)            walk to the book; grab the book
    deposit(book)          walk to the desk; put the book on the desk
    shut(cabinet)          walk to the cabinet; close the cabinet
    switch_off(tablelamp@1) walk to the tablelamp; switch off the tablelamp
  * redo(shut(cabinet))    walk to the cabinet; open the cabinet; stand still
    fetch(mug)             walk to the mug; grab the mug
  * put_wrong(mug)         walk to the kitchencounter; put the mug on the kitchencounter

verdict: admitted — admitted
```

## 4. The case file

`vhhome-case new` writes `{"cases": [...]}`; several cases accumulate in one file. Each entry is the name, the
task, the mistakes as `[class, {parameters}]` pairs in plan order, the story, and the verdict:

```json
{"cases": [
  {"name": "my_tidy_1", "task": "tidy_hard",
   "mistakes": [["skipped_step", {"sid": "switch_off(tablelamp@1)"}],
                ["repetition", {"sid": "shut(fridge)", "at": "switch_off(tv)", "gap": "delayed"}]],
   "story": "She walks straight past the first bedroom lamp, and after the television she goes back and opens the fridge she had already shut.",
   "admitted": true, "reason": "admitted"}]}
```

A case file is made visible to the tools with `--cases FILE` (on `vhhome-case`, `vhhome-episode` and
`vhhome-benchmark`), which sets `VHHOME_CASES` for the process and its episode subprocesses; the file's cases
are appended to the shipped set when the harness loads, so every tool that resolves a case by name sees them.
Several files can be joined in `VHHOME_CASES` with the path separator.

## 5. Running a case

One episode:

```bash
vhhome-episode --latency 2 --cases my_cases.json --case my_tidy_1 --persona classic_novice \
    --assistant gemini:gemini-3.8-flash --port 8080 --out runs/mine
```

The record lands at `runs/mine/gemini-3.8-flash/gemini-3.8-flash__my_tidy_1__classic_novice.json`, with the log
and the film beside it. `--assistant silent` is the control that never speaks. `--paradigm` chooses how the
mistake is forced: `combo` (the benchmark setting: takeover with the in-character system task), `soft` (prompt
only, no takeover; the mistake happens only if the person enacts it) and `hard` (silent takeover, no prompt).

A whole run over your own cells:

```bash
vhhome-benchmark --latency 2 --assistant gemini:gemini-3.8-flash --cases my_cases.json --cells custom_cells.json \
    --out runs/mine --engines 8080,8081 --repeat 1
vhhome-score --run runs/mine/r1 --only objective
```

`--cells` takes a cells file, the same format as `benchmark/cells.json`: a JSON object whose `cells` list holds
one `{"case": ..., "persona": ...}` per episode (other keys are ignored; the shipped file also records its seed
and draw). `examples/custom_cells.json` plays the two example cases with two personas each:

```json
{"cells": [
  {"case": "my_tidy_1", "persona": "classic_novice"},
  {"case": "my_tidy_1", "persona": "classic_expert"},
  {"case": "my_bedroom_1", "persona": "classic_novice"},
  {"case": "my_bedroom_1", "persona": "classic_expert"}]}
```

The run is resumable (a cell whose record exists is skipped), and `vhhome-score` grades it exactly as the
benchmark is graded. Scores on custom cases are comparable across assistants on the same cells, not with the
published table.

## 6. Choosing a persona

`--persona` (default `baseline`) picks one of the nine personas of
[BENCHMARK.md](BENCHMARK.md#6-personas). Two dials matter most when designing an episode. **Knows this house**
decides whether the person finds things itself (`high`) or has to look around and ask (`low`), so a `low`
persona produces more conversation and more refused commands before the first mistake. **Trusts the assistant**
decides how fast a correct claim releases the mistake: `high` comes around at the first correct warning, `low`
only once it sees the mistake itself, so a `low` persona tests whether the assistant can make its case, not just
name the mistake. The three questionnaire dials are shown to the assistant and shape what a well-judged line
looks like. To compare assistants, keep the persona fixed across them; the shipped cells draw three distinct
personas per case for that reason.

## 7. Filming the design

Two reference films prove what a case looks like before any model plays it. Both are executed by the engine
itself, with no person model and no assistant, from the plan: the clean plan, or the case's composed faulty
plan. They need a running simulator and ffmpeg.

```bash
# the clean reference: the task done right
python -m vhhome_bench.video.film_reference 8080 tidy_hard runs/ref/tidy_hard.json

# the designed faulty run of a case (a user case needs VHHOME_CASES so the module can find it)
VHHOME_CASES=my_cases.json python -m vhhome_bench.video.film_reference 8080 tidy_hard runs/ref/my_tidy_1.json --case my_tidy_1
```

Each writes the JSON (outcome, ticks, checkpoints, the action log with the engine's status per command) and an
`.mp4` beside it, opening with a survey of every object the setup displaced. The faulty film is the check that
rule R5 asks for: watch it and ask whether the two mistakes read as one person's bad run and whether each is
visible on camera. The clean film is the companion a reader needs to see what the mistakes took away.

A clean run with the person model in the loop is `--case <task>__clean` (any task), which runs the task with
nothing injected and a silent assistant by default:

```bash
vhhome-episode --latency 2 --case tidybedroom_hard__clean --persona baseline --port 8080 --out runs/ref
```

## 8. Adding a brand-new task

A task is two entries: its steps in `vhhome/tasks.py` and its instruction and checkpoints in `vhhome/goals.py`.

In `vhhome/tasks.py`, add a key to `EPISODES`:

```python
"mytask_hard": dict(family="...", diff="hard", scene=1, init="kitchen",
   # optional: dedupe={"mug": {"room": "bedroom", "keep": 1}} keeps one instance of a class so a bare
   # reference cannot bind to a decoy; targets={"milk": "fridge"} pins the instance the checkpoints mean
   setup=[("Walk","mug","bedroom"),("Grab","mug","bedroom"),("Walk","nightstand","bedroom"),
          ("Put","mug","bedroom","nightstand","bedroom")],          # leaves things out of place
   steps=[("Walk","mug","bedroom"),("Grab","mug","bedroom"),
          ("Walk","kitchentable","kitchen"),("Put","mug","bedroom","kitchentable","kitchen")]),
```

A step is `(Action, class, room)` or, for `Put`/`PutIn`, `(Action, class, room, class2, room2)`; `("StandUp",)`
has no object. Actions are `Walk`, `Grab`, `Open`, `Close`, `Put`, `PutIn`, `SwitchOn`, `SwitchOff`, `Sit`,
`StandUp`, `Drink`, `TurnTo`, `LookAt`. The steps must follow the walk-anchored grammar of section 1, so that
`vhhome-case plan mytask_hard` segments them into the subgoals you intend; `scene` is the apartment (1 to 5) and
`init` the room the person starts in. Keep one natural goal with a perturbed initial state, not a chain of
unrelated subtasks.

In `vhhome/goals.py`, add the same key with the instruction and a checkpoints function over the resolved events,
built from the helpers in that file (`grabbed`, `put_on`, `put_in`, `put_on_n`, `put_in_n`, `opened`, `closed`,
`turned_on`, `turned_off`, `turned_off_n`, `sat_on`, `drank`, `reached`):

```python
"mytask_hard": dict(
   instruction="TASK: Take the mug from the bedroom nightstand to the kitchen table.",
   checkpoints=lambda ev: {"mug on the kitchen table": put_on(ev, "mug", "kitchentable")}),
```

Checkpoints latch once true; success is all of them true. Grading is on the classes the engine actually acted
on, so a `cabinet` is not one of the kitchen cabinets and a `tablelamp` is not a wall lamp; a goal over several
identical objects needs the `_n` helpers (`turned_off_n(ev, "tablelamp", 2)`).

Then verify, in this order: `vhhome-case plan mytask_hard` (the segmentation is what you meant),
`python tests/test_engine.py` (the offline suite still passes), and the clean run in the real apartment, which is
the only proof that the scene contains reachable instances of everything the task names:

```bash
python -m vhhome_bench.video.film_reference 8080 mytask_hard runs/ref/mytask_hard.json   # must end "won"
vhhome-episode --latency 2 --case mytask_hard__clean --persona baseline --port 8080 --out runs/ref       # with the person model
```

Watch the film: an object the person stands over on a low table, a lamp on a wall, or a surface with no free
spot all fail only in the real scene. When the clean run wins and looks right, compose cases on the task with
`vhhome-case new` as above; the composition rules then apply to it like any shipped task. A new task has no
entry in `benchmark/t_ref.json`, so in-time success falls back to the symbolic replay's time until you measure a
clean run and add it.
