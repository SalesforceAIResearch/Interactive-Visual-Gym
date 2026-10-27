"""The command-line tools parse, list the suite, and compose a case (no model, no renderer, no keys)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(*args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


def test_help_of_every_command():
    for mod in ("cooksim_bench.cli.episode", "cooksim_bench.cli.benchmark", "cooksim_bench.cli.score",
                "cooksim_bench.cli.case", "cooksim_bench.roster"):
        r = run(mod, "--help")
        assert r.returncode == 0, (mod, r.stderr[-500:])
        assert "usage:" in r.stdout


def test_roster_and_personas():
    r = run("cooksim_bench.roster")
    assert "gemini-3.8-flash" in r.stdout and "gpt-6-astra" in r.stdout
    r = run("cooksim_bench.roster", "--personas")
    assert "classic_novice" in r.stdout and "classic_expert" in r.stdout


def test_case_list_and_show():
    r = run("cooksim_bench.cli.case", "list")
    assert r.returncode == 0 and "medium_nops_medium_map" in r.stdout and "leave_unattended" in r.stdout
    r = run("cooksim_bench.cli.case", "show", "b3_medium_nops_medium_map_3")
    assert r.returncode == 0 and "wrong_order" in r.stdout


def test_case_derive_writes_a_runnable_case(tmp_path):
    out = tmp_path / "cases.json"
    r = run("cooksim_bench.cli.case", "derive", "--from", "b3_medium_nops_medium_map_3", "--name", "t_burger",
            "--add-question", "next_step", "--out", str(out))
    assert r.returncode == 0, r.stderr[-500:]
    case = json.load(open(out))["cases"][0]
    assert case["name"] == "t_burger" and case["cell"] == "medium_nops_medium_map"
    fams = [s["family"] for s in case["subs"]]
    assert fams[-1] == "error" and case["subs"][-1]["fn"] == "premature_serve"     # a premature serve is always last
    assert "qa_probe" in fams
    assert all(s["name"].startswith("t_burger") for s in case["subs"])
