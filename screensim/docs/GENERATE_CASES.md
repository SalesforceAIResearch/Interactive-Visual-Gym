# Making new episodes

Three independent choices: the **task**, the **mistakes and where they fire**, and the **person**.

## Existing task, different person

No new files:

```bash
screensim-benchmark --assistant openai:gpt-5.4-mini --tasks hearing_setup --out results/persona-study
```

Personas come from `screensim/interact/personas.py`; the benchmark three are `classic_novice`,
`classic_expert`, `baseline`.

## Existing task, your own mistakes

Mistakes are designed onto a task's reference plan. `screensim-case --episodes` shows what the
designer chose for each task; to place your own, give the engine a page, the controls, and where in
the plan it fires:

```python
from screensim.interact.composite import design_composite_episodes
from screensim.tasks import by_id

ir, episodes = design_composite_episodes(by_id("hearing_setup"), per_task=3)
for e in episodes:
    for b in e.beats:
        print(e.id, b.kind, "instead of", b.correct_gesture, "the person does", b.alt_gesture)
```

The designer only proposes mistakes the page can actually host: a `wrong_target` needs a sibling
switch, a `wrong_value` needs a choice list, an `extra_write` needs a control the task never touches.
That is why a mistake is chosen *per page* rather than written by hand.

## A new task

A task is hand-written: an outcome-worded instruction, a trap, and the controls the owner wants. The
route and the goal predicate are derived from the engine, and anything that does not replay is
rejected:

```python
# screensim_bench/suite/task_specs.py
dict(id="camera_framing_guides", family="camera", band=1,
     instruction="My photos come out crooked and my selfies come out flipped the wrong way round. "
                 "Put the lines across the viewfinder that split it into thirds, and make the front "
                 "camera save selfies the way they looked on screen. Leave the rest alone.",
     trap="'Level' is already on and looks like the framing aid",
     steps=[("settings.stub_camera", ["tap Grid", "tap Mirror Front Camera"])]),
```

```bash
python -m screensim_bench.suite.build_tasks --check    # verify every spec replays
python -m screensim_bench.suite.build_tasks            # compile them
python -m screensim_bench.suite.author card "lock screen"   # find a page and its controls
```

`author card` prints a page's route and every control with its state path — write specs against that
rather than guessing labels.

## Keeping training and benchmark apart

If you are generating tasks to train on, keep them out of the test set. `screensim-case --rl` lists
the 67-task training library, and `python -m screensim_bench.suite.split` enforces the rule: a
training task is rejected when its goal contains, or is contained by, a benchmark task's goal. See
[TASKS.md](TASKS.md).
