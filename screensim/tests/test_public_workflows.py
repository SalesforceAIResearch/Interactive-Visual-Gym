"""Public CLI regression tests: real selection/scoring, mocked provider boundaries."""
import json
import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from screensim.interact import agents, backends, roster
from screensim.interact.personas import PERSONAS, bench_personas_for
from screensim.interact.composite import design_composite_episodes
from screensim.tasks import HARD_TASKS, by_id
from screensim_bench.cli import episode, benchmark, score, grade
from screensim_bench.cli._common import assistant_config, output_file
from screensim_bench.runs import sweep

_DESIGNS = {}


@pytest.fixture(autouse=True)
def reuse_immutable_test_designs(monkeypatch):
    def cached(task, per_task=3):
        key = (task.id, per_task)
        if key not in _DESIGNS:
            _DESIGNS[key] = design_composite_episodes(task, per_task=per_task)
        return copy.deepcopy(_DESIGNS[key])
    monkeypatch.setattr(sweep, "design_composite_episodes", cached)


def keys(jobs):
    return [(c, e.id, p.name) for c, t, ir, e, p in jobs]


def test_full_benchmark_selection_is_unchanged():
    expected = []
    for tid in HARD_TASKS:
        _, cases = sweep.design_composite_episodes(by_id(tid), per_task=3)
        expected.extend(("test", case.id, p) for case in cases for p in bench_personas_for(case.id))
    assert keys(sweep.build_jobs(["test"])) == expected
    assert len(expected) == 90


def test_selected_case_and_persona_precede_limit():
    jobs = sweep.build_jobs(["test"], ["hearing_setup"], limit=1,
                            episode="hearing_setup__c2", persona="baseline")
    assert keys(jobs) == [("test", "hearing_setup__c2", "baseline")]


def test_explicit_cells_precede_limit():
    cell = keys(sweep.build_jobs(["test"]))[-1]
    jobs = sweep.build_jobs(["test"], limit=1,
                            cells=[dict(zip(("contestant", "episode", "persona"), cell))])
    assert keys(jobs) == [cell]


@pytest.mark.parametrize("kwargs", [dict(persona="missing"), dict(episode="missing"),
    dict(limit=-1), dict(shard=(0, 3)), dict(shard=(4, 3)), dict(cells=[]),
    dict(cells=[dict(contestant="test", episode="missing", persona="baseline")])])
def test_invalid_selection_rejected_before_execution(kwargs):
    with pytest.raises(ValueError):
        sweep.build_jobs(["test"], **kwargs)


def test_shards_are_disjoint_and_exhaustive():
    full = set(keys(sweep.build_jobs(["test"])))
    parts = [set(keys(sweep.build_jobs(["test"], shard=(i, 3)))) for i in (1, 2, 3)]
    assert set.union(*parts) == full
    assert sum(map(len, parts)) == len(full)


def test_episode_dry_run_calls_no_provider(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "one", Mock(side_effect=AssertionError("provider called")))
    out = tmp_path / "run"
    episode.main(["--task", "hearing_setup", "--episode", "hearing_setup__c2",
                  "--persona", "baseline", "--assistant", "gemini-3.8-flash",
                  "--out", str(out), "--dry-run"])
    selected = json.loads(capsys.readouterr().out)
    assert selected[0]["episode"] == "hearing_setup__c2"
    assert selected[0]["persona"] == "baseline"
    assert not out.exists()


def test_benchmark_forwards_custom_provider_and_endpoint(tmp_path, monkeypatch):
    received = {}
    def run(*args, **kwargs):
        received.update(kwargs)
        received["config"] = dict(sweep.ASSISTANT_CONFIG)
    monkeypatch.setattr(sweep, "run", run)
    benchmark.main(["--assistant", "openai:org/custom-vlm", "--base-url", "http://127.0.0.1:9999/v1",
                    "--out", str(tmp_path), "--limit", "2"])
    assert received["contestants"] == ["org_custom-vlm"]
    entry = received["config"]["org_custom-vlm"]
    assert entry["backend"] == "openai" and entry["model"] == "org/custom-vlm"
    assert entry["base_url"] == "http://127.0.0.1:9999/v1"


def test_roster_defaults_not_mutated_and_override_reaches_backend(monkeypatch):
    before = json.dumps(roster.ROSTER, sort_keys=True)
    tag, entry = assistant_config("openai:gpt-6-astra", "http://localhost:9999/v1")
    backend = Mock()
    monkeypatch.setattr(backends, "OpenAIBackend", backend)
    roster.make_backend(tag, entry=entry)
    assert backend.call_args.args == ("gpt-6-astra", "http://localhost:9999/v1")
    assert backend.call_args.kwargs["profile"] == "gpt5"
    assert json.dumps(roster.ROSTER, sort_keys=True) == before
    assert assistant_config("gemini:gemini-3.8-flash") == ("gemini-3.8-flash", None)


@pytest.mark.parametrize("spec,url", [("bad:model", ""), ("openai:", ""),
    ("gemini:some-model", "http://localhost:9999")])
def test_invalid_backend_arguments(spec, url):
    with pytest.raises(ValueError):
        assistant_config(spec, url)


def test_environment_credentials_work_without_legacy_files(monkeypatch):
    import google.genai
    import openai
    gemini, gpt = Mock(), Mock()
    monkeypatch.setattr(google.genai, "Client", gemini)
    monkeypatch.setattr(openai, "OpenAI", gpt)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setattr(agents.os.path, "isfile", lambda _: False)
    agents.LLMClient("test-model")
    agents.OpenAIClient("test-model")
    gemini.assert_called_once_with(api_key="test-gemini")
    gpt.assert_called_once_with(api_key="test-openai")


def test_missing_credential_is_actionable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(agents.os.path, "isfile", lambda _: False)
    with pytest.raises(ValueError, match="Missing GEMINI_API_KEY"):
        agents._credential("GEMINI_API_KEY", "~/.gemini_key")


def test_output_directory_and_json_file(tmp_path):
    assert output_file(str(tmp_path)) == str(tmp_path / "composite_results_v2.json")
    assert output_file(str(tmp_path / "run.json")) == str(tmp_path / "run.json")


def test_real_scripted_rollout_resume_and_three_run_scoring(tmp_path, monkeypatch):
    # This is the actual engine, event injection, records, and scorers; no API calls.
    monkeypatch.setattr(sweep, "INTERFACE", "v1")
    roots = []
    for i in range(3):
        root = tmp_path / f"r{i+1}"
        output = root / "composite_results_v2.json"
        sweep.run(1, str(output), ["oracle ceiling"], tasks=["hearing_setup"], limit=1)
        records = output.with_suffix(".records.jsonl")
        original = records.read_bytes()
        sweep.run(1, str(output), ["oracle ceiling"], tasks=["hearing_setup"], limit=1)
        assert records.read_bytes() == original
        roots.append(str(root))
    argv = sys.argv
    score.main(["--results", roots[0]])
    assert sys.argv is argv
    objective = json.loads((Path(roots[0]) / "objective_scores.json").read_text())
    assert objective[0]["episodes"] == 1
    score.main(["--results", *roots, "--three-runs"])
    repeat = json.loads((Path(roots[0]) / "three_runs.json").read_text())[0]
    assert repeat["runs"] == 3 and repeat["cases"] == 1
    assert repeat["S_mean"] == objective[0]["S"]
    assert repeat["flips"] == 0


def test_failed_episode_is_saved_and_command_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "one", Mock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="1 episode"):
        sweep.run(1, str(tmp_path / "out.json"), ["test"], tasks=["hearing_setup"], limit=1)
    record = json.loads((tmp_path / "out.records.jsonl").read_text())
    assert record["error"] == "provider unavailable"


@pytest.mark.parametrize("module", [score, grade])
def test_scorers_reject_missing_records(tmp_path, module):
    with pytest.raises(SystemExit) as error:
        module.main(["--results", str(tmp_path)])
    assert error.value.code == 2


def test_three_run_flag_requires_three_runs(tmp_path):
    (tmp_path / "test.records.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit) as error:
        score.main(["--results", str(tmp_path), "--three-runs"])
    assert error.value.code == 2
