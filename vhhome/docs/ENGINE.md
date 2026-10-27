# The engine

VH-Home's engine (`vhhome`) is a tick-stepped household world over the VirtualHome simulator, with a symbolic
twin of the state, an egocentric action space, a plan representation that mistakes are transforms of, and a
harness (`vhhome_bench`) that runs a simulated person and an assistant in it. This page covers the engine on its
own; the benchmark built on it is in [BENCHMARK.md](BENCHMARK.md), the simulator installation in
[SIMULATOR.md](SIMULATOR.md).

## What is simulated

| | |
|---|---|
| Apartments | VirtualHome v2.3.0, scenes 1 to 5: several rooms (kitchen, living room, bedrooms, bathroom), doors and open passages, furniture, appliances, small objects |
| Objects | instance-unique: an apartment holds several plates, books, mugs and lamps, and the engine never collapses them onto one name |
| State | what is held (two hands), what is open, what is switched on, where every task object is (on a surface, inside a container) |
| Actions | walk, grab, open, close, put on, put in, switch on, switch off, sit, stand up, drink, turn to, look at; plus the egocentric movement primitives below |
| Dynamics | none of its own: VirtualHome has no smoke, fire or spoilage, so a mistake leaves a persistent wrong state (a lamp on, a door open, a device running empty) rather than a consequence that unfolds |
| Rendering | the person's first-person camera (a wide field of view with three pitch levels) and two top-down maps (the whole house, the current room), rendered with software OpenGL |

## The world and the tick

`vhhome.world.LiveWorld` owns one Unity session on one port; one episode is one fresh session. It executes one
closed-grammar command at a time, rendering the frames live. Time is discrete: one **tick** is 1 second of
animation, recorded as `FPT = 5` frames. A command spans several ticks (a walk across a room costs many), and
idle ticks freeze the last frame, which is the honest analogue of "the world keeps running" in a world with no
autonomous dynamics. Long actions are rendered once and then streamed tick by tick, so a person who changes its
plan mid-walk is put back at the frame where it changed its mind.

Every executed command produces a log entry: the text, the engine step, the status (`done`, `rejected`, or
`interrupted`), the tick issued and done, the room, and a refusal reason when refused. The whole log is in the
episode record.

## The symbolic twin

Beside the scene graph the world keeps `SymState`, a symbolic state with the same verb semantics as the offline
plan executor: `holding`, `opened`, `on`, and `loc` (where each object is), keyed on instance ids. The twin does
three jobs. It **checks preconditions** before a command reaches Unity and refuses in words: "the fridge is
closed — open it first", "my hands are full", "I'm not holding the milk — my hands are empty", "the tablelamp is
already off", "I'm not sitting". It **grades**: the task's checkpoints run on the twin's resolved events
(verb, class, instance, target), so grading is on what the body actually did, not on what anyone typed. And it
is the **reference** the person's exact skills and the live mistake anchors read from.

The same executor runs offline without a simulator (`vhhome.plan.simulate`): it replays a compiled plan
symbolically, reports the outcome and the checkpoints, and is what the case checker and the engine test suite
use.

## The egocentric action space and its two gates

The person acts in a **mid-level, egocentric, visibility-gated** action space (`vhhome.actions`,
`vhhome.gates`). The body can only act on what is in its first-person view, every object reference resolves to
the nearest visible instance of that class, and rooms are changed by walking through a visible door. The exact
footsteps belong to the engine; where to look and which way to go belong to the agent, which is given the maps
for that purpose. The vocabulary both agents are shown is rendered from the parser's own constants:

```
MOVING / LOOKING (no target needs to be visible)
  "turn left" · "turn right"            — 30 deg each; "turn right 3" = three turns, one look after each
  "go straight"                         — advance about 1 m (at most 12 in one command)
  "look down" · "look up"               — pitch the view between 0, 14 and 28 deg below level
  "go through the visible door"         — step through a doorway or open passage you can see now

ACTING ON SOMETHING (the target must be visible first)
  "turn to <object>"  "go to <object>"  (approach until about 1.5 m away)
  "pick up <object>" · "open <object>" · "close <object>"
  "switch on <object>" · "switch off <object>" · "drink <object>" · "sit on <object>"
  "put <object> on <surface>" · "put <object> in <container>"
  "stand up"
```

Two gates decide whether a command works, and both are pure geometry that executes nothing:

* **Visibility gate**: `go to X` and every manipulation need an instance of X in the first-person view (about
  48 degrees either side of the facing direction; walls and doorways block sight). Refusal: "I cannot see any
  fridge from here — I need to turn to look around, or go through a door toward it (check the map)".
* **Reach gate**: a manipulation needs the target visible and within 1.5 m (`NEAR_M`). Refusal: "the fridge is
  2.4 m away — too far to reach; I need to go to it first". `go to` stops at that distance, close enough to act
  and far enough to keep the target in view.

Which instance is acted on is the one the gate saw, so where the body stands and which way it faces decides
which mug it picks up. A held object is resolved from the hands, not by sight (you can feel what you carry), so
`put` and `drink` gate only the destination. Commands with a removed phrasing ("turn around", "go through the
nearest door") are refused with the allowed form instead of being dropped. The engine's own scripted plan steps,
a person following its routine in its own house, do not pass through these gates; only agent-typed commands do.

## The plan IR and its transforms

A task's step list is segmented losslessly into typed subgoal nodes (`vhhome.plan.segment_steps`): `fetch`,
`deposit`, `switch_on`, `switch_off`, `open`, `shut`, `inspect`, `posture`, `drink`, `waypoint`, each owning the
steps it consists of (the grammar is in [BENCHMARK.md](BENCHMARK.md#3-plans-the-subgoal-grammar-and-anchors)).
Compiling the nodes back yields the original steps byte for byte. Mistakes are typed **transforms** over subgoal
ids, never over raw indices: `skip`, `insert`, `replace`, `repeat`, `swap`, `permute`, `truncate_done`,
`forget_rest`, plus injected node factories (`wrong_grab`, `put_wrong`, `search`, `loiter`, `lost`, `redo`). The
six graded classes in `vhhome.mistakes.BEHAVIORS` are built from them (`err_skipped_step`, `err_missing_step`,
`err_wrong_object`, `err_wrong_placement`, `err_repetition`, `err_stall`, `err_navigation_inefficiency`).
Transforms **compose in order** under a priority rule: an earlier transform owns the plan, and a later one whose
anchor it removed is skipped with a record rather than raised. The real dependency graph of a plan
(`vhhome.plan.deps`) lists the only orderings a task requires, which both agents are told.

## The person's runner

The simulated person (`vhhome_bench.person`, prompts in `vhhome_bench.protocol`) is a language model with a
persona card. It starts with an **empty queue** and plans the chore for itself from the instruction, its own
eyes (its last frames) and the map images; it carries a private order of the subgoals and works straight down
it. It has two free **skills** that cost no game time and answer from the live house: `resolve_subgoal`
(a subgoal id to its exact ordered commands) and `expand_action` (one step to the command block it belongs to).
Its reply JSON carries a **move** (`ask_why`, `question_correctness`, `ask_for_more_guidance`,
`ask_for_less_interruption`, `accept`, `reject`, `answer`), an utterance, a `plan_update` with one of three ops
(`drop`: queue unchanged; `append`: add commands; `replace`: swap the remaining plan), a `close_conversation`
bit, and the private `assistant_error_call` (`none` | `correct` | `wrong`): its own judgement of whether the
assistant has named its live mistake. Only the utterance crosses to the assistant.

Two standing rules shape it. The **silence rule**: real people work in silence, so most turns the utterance is
empty; it speaks only when a command was refused and it is stuck, when its queue is empty and it does not know
the next step, or when asked a question, and then under ten words. The **no-self-repair rule**: a mistake is the
assistant's failure to catch, not the person's to hide; it says nothing about the mistake, fixes only what the
assistant has told it is wrong, and otherwise finishes the chore as it is. Its words and its actions must match:
agreeing out loud with `op: drop` changes nothing in the world, and the engine records every such case.

## The takeover block and its release

When a scheduled mistake fires, the engine replaces the person's queue with the mistake's action sequence and
shows it a `[SYSTEM TASK]` block, re-rendered every turn, in one of two framings chosen by the mistake's family.
For a **commission** mistake (`skipped_step`, `missing_step`, `wrong_object`, `wrong_placement`), in short:

> You are testing whether the AI assistant can spot a mistake and convince you it is real. Your hands are
> already committed: the action sequence below is what you are doing, and you cannot change it here. [...]
> The mistake in play: you are picking up the juice where the milk belongs. You know it is wrong and you are
> doing it anyway — that is the test. Do not say anything to the assistant about this mistake [...] do not fix
> anything yourself. Once it has found it, your trust level decides how fast you come around: HIGH — the first
> correct warning; LOW — only after you see it with your own eyes or in your own hands. When convinced, say so
> in your own words and set "move": "accept" — your hands return to you and this task ends.

For an **attention** mistake (`repetition`, `stall`, `navigation_inefficiency`), which is not named:

> You are testing whether the AI assistant notices that you have LOST TRACK of something and brings you back.
> Your hands are already committed: the action sequence below is the errand you drifted into; you cannot change
> it here, and you have not noticed the problem. [...] Until someone brings it up, carry on and do not fix
> anything yourself. When the assistant does and what you can see confirms it, come back at once whatever your
> persona: say so in your own words and set "move": "accept".

The block dissolves, and the hands come back, for one of these reasons, recorded per mistake as
`released_reason`:

| reason | when |
|---|---|
| `convinced` | the person's `assistant_error_call` was `correct` (or, with `VH_GRADER=engine`, the engine's flag verdict was a detection) and it then moved `accept` |
| `prevented` | the correct claim was accepted before the mistake physically happened; the clean plan is restored |
| `convinced_early_missed` | the correct claim was made before the window opened but the mistake had already happened when the person accepted |
| `task_completed` | the forced sequence has been carried out |
| `block_exhausted` | the queue ran out |

After release the person's own plan stands again; a step it left wrong on purpose is marked as such in its
private order so it does not go back to it unprompted, and a correction the assistant gives later, in any words,
can still be carried out.

## Live injection

A mistake fires on a **live anchor** in the person's own flow, never on a timer. The engine watches the
person's queue against the plan's **progress key** (which subgoals are done, read from the twin) and the
next-two-actions rule: for `wrong_object`, `skipped_step`, `missing_step` and `wrong_placement`, one of the next
two queued actions aims at the anchored subgoal's objects; for `navigation_inefficiency`, one of the next two
aims at the subgoal it comes before; for `stall`, the action that completes the anchored subgoal has just
finished; for `repetition`, the repeated subgoal was finished at least 30 ticks ago (`STEPIN_REP_GAP`) and the
next two actions aim at the `at` subgoal, or the chore is one step from finishing. Because the person works in
its own order, the mistake's block is cut from the plan at that moment and the detection window opens on the
tick of injection (`dev_live`). A mistake whose moment never comes is skipped at its deadline with the reason,
and the episode moves on to the next one.

## Latency charging

The assistant's reply is not free. A reply that took x wall-seconds is delivered `ceil(x / ASSIST_TICK_S)` ticks
after the tick it was asked on (`ASSIST_TICK_S = 1.2`), capped at `ASSIST_MAX_TICKS = 15`; the world advances in
between and the next call starts only when the reply has landed. Only the successful attempt's own generation
time is charged, so API stalls and retries do not freeze the person; the raw wall time is recorded per call. For
a locally served model the charged time is divided by `ASSIST_LATENCY_DIV`, the measured ratio of this
hardware's generation time to a reference GPU's (`benchmark/gpu_latency_ratio.json`), so the delay describes
the model rather than the card it ran on; the published runs served every open-weight model on the same card type
and applied no factor. `ASSIST_LATENCY=0` restores an instantaneous assistant.

## What a record contains

One JSON per episode, written beside its log and film:

| field | content |
|---|---|
| `outcome`, `ticks`, `engine_rev` | the outcome, the episode length, the engine revision stamp |
| `scenario`, `persona`, `episode`, `task` | the case, the persona, the task key, the instruction |
| `phases[]` | one per mistake: `err`, `injected_at`, `dev_live` (when it physically happened), `armed_at`, `released_at`, `released_reason`, the window, and the flags credited to it |
| `phases_total`, `phases_caught`, `phases_prevented`, `phases_detected` | the counts |
| `flags[]` | every assistant flag: `t`, `label`, `verdict`, `phase` |
| `conversation[]` | both agents' turns: the person's utterance, move and plan_update; the assistant's full JSON; an engine snapshot of the state at that tick |
| `action_log[]` | every command: tick issued and done, status `done`, `rejected` or `interrupted`, the refusal reason |
| `assistant` | call statistics: calls, empty replies, retries, schema failures, tokens, latency |
| `assistant_latency[]` | per call: wall time, generation time, ticks charged, whether capped |
| `checkpoints` | the task's checkpoints at the end |
| `cook` | the person's `assistant_error_call` ledger |
| `grader` | which grader armed the releases (`cook` or `engine`) |

## Environment switches

| variable | meaning |
|---|---|
| `VHHOME_ROOT`, `VHHOME_SIMULATOR`, `VHHOME_UNITY_DIR`, `VHHOME_VIRTUALHOME`, `VHHOME_RUNS` | where the checkout, the simulator directory, the Unity build, the VirtualHome client and the run output live (`vhhome/paths.py`) |
| `VHHOME_KEYS_FILE` | the keys file (default `~/.config/vhhome/keys.env`) |
| `VHHOME_MODELS`, `VHHOME_SERVING` | the weights and environments root for served models; the directory of the serving scripts |
| `VHHOME_CASES` | user case files (path-separated) appended to the shipped cases |
| `VH_MIDLEVEL=1` | the egocentric action space; always on in the console scripts |
| `VH_GRADER` | `cook` (default: the person's own call arms the release) or `engine` (the flag verdict does) |
| `GEMINI_MODEL` | the model behind the person |
| `ASSIST_LATENCY=0` | an instantaneous assistant (ablation) |
| `ASSIST_LATENCY_DIV`, `ASSIST_MAX_TICKS` | the hardware ratio and the cap on charged ticks |
| `STEPIN_WALL`, `STEPIN_CAP` | the wall-clock budget in seconds and the tick cap (5400 and 520 in the benchmark) |
| `VH_NO_PROGRESS` | ticks without progress before an episode ends `stuck` (60) |
| `IV_CAPTURE_DIR` | dump every prompt and image both sides see |
| `UNITY_JOB_WORKERS`, `LP_NUM_THREADS` | simulator threads (see [SIMULATOR.md](SIMULATOR.md)) |

## Using the engine without the benchmark

The episode runner is a module with a positional interface, the same one the benchmark lane calls:

```bash
python -m vhhome_bench.episode PORT CASE PERSONA OUT.json [combo|soft|hard]
```

It runs one case with the assistant named by the environment (`IV_ASSIST_MODEL`, `IV_ASSIST_BASE_URL`,
`IV_ASSIST_PROFILE`, `IV_ASSIST_BACKEND`; `IV_ASSIST_MODEL=silent` for no assistant) and writes the record and
its film to `OUT.json` and `OUT.mp4`. `vhhome-episode` is the same call with the assistant resolved from
`--assistant` and the benchmark's episode settings applied.

To drive the world with no person model at all, `film_reference` executes a task's clean plan, or a case's
composed faulty plan, step by step through `LiveWorld.exec_step` and scores the same checkpoints:

```bash
python -m vhhome_bench.video.film_reference PORT TASK OUT.json [--case NAME]
```

`python tests/test_engine.py` runs the offline engine suite (the VirtualHome client must be importable, but no
simulator is needed) and ends with `0 failure(s)`.

## Package map

```
vhhome/
  world.py        LiveWorld (one Unity session, one command per tick, frames), SymState (the symbolic twin)
  actions.py      the mid-level controller: primitives, the natural-language parser, the vocabulary text
  gates.py        the visibility and reach gates (pure geometry, executes nothing)
  tasks.py        the 17 tasks: scene, start room, setup, steps
  goals.py        instructions and checkpoints
  plan.py         the subgoal IR, transforms, the symbolic executor
  mistakes.py     the behaviour library: the graded classes as transforms
  topdown_map.py, render.py   the maps and the first-person camera
  paths.py, keys.py, data/    locations, API keys, the assistant reply schema
vhhome_bench/
  protocol.py     prompts, persona cards, the assistant call, flag verdicts
  person.py       the person's runner (queue, skills, turns)
  episode.py      one rollout: polling, latency, injection, takeover, release, outcome, the record
  lane.py, roster.py, cases_spec.py, cases.py, case_checks.py, native.py, native_backend.py
  scoring/        metrics, metrics_table, results_table, rubric, rubric_table, rubric_batch, final_table
  video/          film (a record into a film), film_reference (the engine executes a plan)
  cli/            vhhome-episode, vhhome-benchmark, vhhome-score, vhhome-case
```
