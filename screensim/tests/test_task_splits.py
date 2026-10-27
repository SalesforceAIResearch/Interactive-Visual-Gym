"""The RL training set and the benchmark set must stay apart — see docs/RL_TASKS.md."""
import sys, os

from screensim.task_splits import BENCH_TEST, RL_TRAIN, RL_TRAIN_BY_BAND
from screensim.tasks import HARD_TASKS, by_id
from screensim_bench.suite.split import goal_of


def test_sets_are_disjoint_and_resolve():
    assert set(RL_TRAIN) & set(BENCH_TEST) == set()
    assert sorted(BENCH_TEST) == sorted(HARD_TASKS), "the benchmark set is frozen"
    for tid in RL_TRAIN + BENCH_TEST:
        assert by_id(tid) is not None
    assert sorted(t for ids in RL_TRAIN_BY_BAND.values() for t in ids) == sorted(RL_TRAIN)


def test_no_training_goal_contains_or_is_contained_by_a_benchmark_goal():
    bench = {tid: goal_of(by_id(tid)) for tid in BENCH_TEST}
    bad = []
    for tid in RL_TRAIN:
        g = goal_of(by_id(tid))
        assert g, f"{tid} has an empty goal"
        for bid, bg in bench.items():
            if g <= bg or bg <= g:
                bad.append((tid, bid))
    assert not bad, f"leaked training tasks: {bad}"


def test_every_training_task_replays_to_its_goal():
    for tid in RL_TRAIN:
        t = by_id(tid)
        from screensim.core.engine import Device
        from screensim.tasks import resolve_dynamic
        d = Device(t.seed)
        if t.start_app != "home":
            d.open_app(t.start_app)
        for g in t.plan:
            d.act(resolve_dynamic(d, g))
        ok, why = t.goal(d.st)
        assert ok, f"{tid} did not reach its goal: {why}"
