from cooksim_bench.cli.episode import assistant_flags
from cooksim_bench import roster
import subprocess
import pytest
import json
from cooksim_bench.scoring import post_episode


def test_roster_default_flags_are_unchanged():
    entry = roster.ROSTER[0]
    flags, tag = assistant_flags(entry["tag"])
    assert flags == entry["flags"] and tag == entry["tag"]
    assert flags is not entry["flags"]


def test_explicit_roster_endpoint_and_profile_are_forwarded():
    entry = next(e for e in roster.ROSTER if "--assist-backend" in e["flags"]
                 and e["flags"][e["flags"].index("--assist-backend")+1] == "openai")
    original = list(entry["flags"])
    flags, _ = assistant_flags(entry["tag"], "http://localhost:9999/v1", "default")
    assert flags[-4:] == ["--assist-base-url", "http://localhost:9999/v1", "--assist-profile", "default"]
    assert entry["flags"] == original


def test_post_episode_uses_installed_modules_and_propagates_failure(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
    monkeypatch.setattr(post_episode.subprocess, "run", run)
    post_episode.score_record("record.json", judge=True)
    assert [c[2] for c in calls] == ["cooksim_bench.scoring.eval_assistant",
        "cooksim_bench.scoring.eval_judge", "cooksim_bench.scoring.eval_assistant"]
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(post_episode.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        post_episode.score_record("record.json")


def test_fresh_complete_record_is_scored_immediately(tmp_path, monkeypatch):
    from cooksim_bench.scoring import eval_v2_sweep as sweep
    case = sweep.all_cases()[0]["name"]
    record = tmp_path / "frames" / "baseline" / "test-model" / f"{case}.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    output = tmp_path / "eval_v2.json"
    monkeypatch.setattr(sweep, "BENCH", str(tmp_path))
    monkeypatch.setattr(sweep, "OUT", str(output))
    seen = []
    def score(data, key, *args, **kwargs):
        seen.append(key)
        return {"assistance_quality": {"SQ": None}}
    monkeypatch.setattr(sweep, "score_episode", score)
    sweep.main()
    assert seen == [f"frames/baseline/test-model/{case}"]
    assert json.loads(output.read_text())["n"] == 1
