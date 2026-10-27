# Contributing to ScreenSim

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q            # 206 tests, no API keys needed
```

## The two rules that matter

**Anything that enters the task library must replay.** `python -m screensim_bench.suite.build_tasks`
rejects a task whose plan does not land or whose goal does not hold. Never hand-edit the generated
`screensim/tasks_rl.py` or `screensim/task_splits.py`; edit the specs and rebuild.

**Training and benchmark stay apart.** `python -m screensim_bench.suite.split` fails if a training
task's goal contains, or is contained by, a benchmark task's. A test enforces it.

## Changing the engine

Every record carries an engine stamp (`screensim/engine_rev.py`). If you change engine behaviour, say
so in the pull request: results produced before and after are not comparable, and published episodes
must be re-shot rather than mixed.
