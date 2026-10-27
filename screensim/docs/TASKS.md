# The RL training task library — and why it is kept apart from the benchmark

This repository now contains **two task sets that must never be confused**:

| Set | Where | Tasks | Designed episodes | × 3 personas | Purpose |
|---|---|---|---|---|---|
| **`BENCH_TEST`** | `screensim.task_splits.BENCH_TEST` | 11 | 30 | 90 | **held out.** The `HARD_TASKS` every formal run and every published number was scored on. Frozen. |
| **`RL_TRAIN`** | `screensim.task_splits.RL_TRAIN` | 67 | 168 | 504 | training. Hand-written, replay-verified, split into four complexity bands. |

```python
from screensim.task_splits import RL_TRAIN, BENCH_TEST, RL_TRAIN_BY_BAND
from screensim.tasks import by_id

for task_id in RL_TRAIN:            # train on these
    task = by_id(task_id)
```

**Do not train on `BENCH_TEST`, and do not report benchmark numbers on `RL_TRAIN`.** The two lists are
disjoint by construction and a test asserts it. Everything else in `screensim/tasks.py` (the older v2
library) belongs to neither set — it is neither frozen nor leakage-checked, so leave it out of both
unless you check it yourself.

## What a training task is

A task is a natural request an owner could make of a phone, worded as an **outcome** rather than as a
list of switches, with a reference plan and a goal state:

> *"I'm abroad for a fortnight and roaming is extortionate. Only the browser and the maps may touch
> mobile data, nothing may quietly refresh itself in the background except my calendar. Mobile data
> itself stays on — I need those two apps working."* — `roaming_full_lockdown`, band 4, 15 gestures,
> 10 state changes

Every task also carries a **trap**: a plausible wrong move that satisfies the surface of the request
and breaks it. For the task above it is the master *Cellular Data* switch, which is exactly what a
careless assistant reaches for and exactly what must stay on.

### The four bands

| Band | Shape | Tasks | Gestures (mean / max) | State changes (mean / max) |
|---|---|---|---|---|
| 1 | one page, a couple of controls | 17 | 5.8 / 9 | 2.0 / 3 |
| 2 | one page plus a leave-alone constraint | 19 | 5.9 / 8 | 2.2 / 3 |
| 3 | several pages and/or a real trap | 21 | 11.5 / 24 | 4.2 / 20 |
| 4 | above benchmark difficulty | 10 | 17.8 / 27 | 8.5 / 20 |
| — | *benchmark, for comparison* | *11* | *11.4 / 18* | *6.2 / 20* |

Band 4 is deliberately harder than the test set on both measures.

## How the split is enforced

`tools/rl_split.py` compares every training task with every benchmark task on its **goal** — the set
of `(state path, value)` pairs the reference solution produces — not on its name or its wording:

* **leak → rejected.** The training goal is a *subset* of a benchmark goal, or a *superset* of one.
  Either way the benchmark task is contained in the training task and can be memorised.
* **overlap → kept, and reported.** Some paths in common, but neither contains the other.

Two tasks were rejected by this rule and are excluded from `RL_TRAIN` (they still exist in the
library, and could re-qualify if deepened so neither containment holds):

* `stolen_device_guard` — its goal sits entirely inside `lost_phone_lockdown`.
* `privacy_deep_clean` — it contains `stop_profiling` whole.

Of the 67 kept, 21 share at least one control with a benchmark task and 46 share none. The sharing is
intentional: the phone has a finite number of switches, and a training set that avoided every control
the test set touches would teach nothing transferable. What must not be shared is the *task*.

## How a task is written, and what "verified" means

The division of labour is the point. **Hand-written** (`tools/rl_task_specs.py`): the situation, the
outcome wording, the trap, and which controls the owner wants. **Derived from the engine**
(`tools/build_rl_tasks.py`): the route to each page including the scrolls, the full gesture plan, and
the goal predicate.

```python
dict(id="volume_safety_for_kid", family="sounds", band=2,
     instruction="My son wears headphones for hours. Cap how loud the music can get and hold the "
                 "ceiling at the quietest setting the phone offers.",
     trap="the cap has its own switch as well as a level; setting one without the other does nothing",
     mirrors="Apple Support: Reduce loud sounds",
     steps=[("settings.sounds", ["tap Volume Limit", "tap Reduce Loud Audio",
                                 "tap Max Decibel Level", "tap 75 Decibels"])]),
```

A step is either `("<settings page key>", [actions])`, where the compiler routes there and scrolls to
each control, or `("raw", [gestures])`, where the gestures run exactly as written — that is how
dialogs, sliders, typing and the non-Settings apps are expressed. `start_app` selects the app the
episode begins in (`"settings"` by default, `"bank"` for the payments flow).

Then the gate: every spec is replayed on a fresh phone and **rejected** unless the plan lands, the
goal holds, and nothing outside the goal moved. Nothing enters the library on trust.

```bash
python tools/build_rl_tasks.py --check    # verify only
python tools/build_rl_tasks.py            # verify, then write screensim/tasks_rl.py
python tools/rl_split.py                  # re-check leakage, write screensim/task_splits.py
python tools/author_task.py card "lock screen"   # find a page, see its controls and state paths
```

`screensim/tasks_rl.py` and `screensim/task_splits.py` are **generated — never edit them by hand.**
Edit the specs and rebuild.

## Errors are generated per task, not written by hand

The existing designer (`screensim/interact/composite.py`) injects mistakes into these tasks with no
changes: 168 certified episodes, two designed mistakes each, drawn from each page's real structure —
sibling switches give `wrong_target` and `extra_write`, radio lists give `wrong_value`, the pages
passed en route give `wrong_page`. A sample:

```
location_audit__c1
   wrong_page     instead of 'tap Location Services'   the person taps 'Tracking'
   wrong_target   instead of 'tap Precise Location'    the person taps 'While Using the App'
mum_new_phone__c2
   wrong_target   instead of 'tap On/Off Labels'       the person taps 'Button Shapes'
   wrong_value    instead of 'tap 5 Minutes'           the person taps '2 Minutes'
```

## Coverage

The training tasks visit **80 screens**; the benchmark visits 32, and **all 32 are covered**. They also
exercise every interaction type the benchmark uses: switches, radio choices, a slider, typing, a
one-time code, and a confirmation dialog.

## Known gaps

* **Error mix is skewed**: `wrong_page` 131 and `wrong_target` 133 against `wrong_value` 19 and
  `extra_write` 19. `extra_write` needs a page where a sibling switch is genuinely tempting, which is a
  hand-design judgement the designer cannot discover on its own.
* **QuickPay's `Request` flow is unused** by any task; only *send* is trained.
* Two screens carry ambiguous labels — the Settings root renders both a search *field* and a *row*
  called "Search", and the Photos page renders two rows called "Cellular Data". Tasks avoid both.
