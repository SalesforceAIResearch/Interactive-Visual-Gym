"""The command line a user meets first (no model calls, no keys)."""
import screensim_bench.cli.case as case_cli
import screensim_bench.cli.roster as roster_cli
from screensim.tasks import HARD_TASKS
from screensim.task_splits import BENCH_TEST, RL_TRAIN


def test_case_listing(capsys):
    case_cli.main([])
    out = capsys.readouterr().out
    for t in HARD_TASKS:
        assert t in out


def test_roster_listing(capsys):
    roster_cli.main([])
    out = capsys.readouterr().out
    assert "gpt-6-astra" in out and "seats" in out


def test_rl_listing_is_separate_from_the_benchmark(capsys):
    case_cli.main(["--rl"])
    out = capsys.readouterr().out
    assert "training tasks" in out
    assert set(RL_TRAIN) & set(BENCH_TEST) == set(), "training and benchmark sets must stay disjoint"


def test_paths_resolve_to_shipped_data():
    import os

    from screensim_bench.paths import AUTO_TASKS, BENCHMARK
    assert os.path.isdir(BENCHMARK)
    assert os.path.exists(AUTO_TASKS), "the derived User Guide tasks ship with the release"
