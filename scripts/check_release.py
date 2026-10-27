#!/usr/bin/env python3
"""Offline release checks; no assistant/judge calls or live simulator rollouts.

Run from any directory using the Python environment containing the selected engines.
Snapshots preserve CLI output and runtime-file hashes, not stochastic LLM outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("cooksim", "vhhome", "screensim")
GUIDES = ("GETTING_STARTED", "EVALUATE_YOUR_MODEL", "BENCHMARK", "REPRODUCE",
          "GENERATE_CASES", "ASSISTANTS", "INTERACTION_QUALITY_GRADING", "ENGINE")
LISTINGS = {
    "cooksim": [("cooksim-roster", []), ("cooksim-case", ["list"])],
    "vhhome": [("vhhome-roster", []), ("vhhome-case", ["list"])],
    "screensim": [("screensim-roster", []), ("screensim-case", ["--episodes"])],
}


def tracked_files():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, capture_output=True, check=True)
    return [Path(p) for p in result.stdout.decode().split("\0") if p]


def runtime_hashes(engines):
    result = {}
    for p in tracked_files():
        if p.parts[0] not in engines or "docs" in p.parts or p.suffix == ".md":
            continue
        if p.name in ("LICENSE", "CITATION.cff"):
            continue
        path = ROOT / p
        if path.is_file():
            result[str(p)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_python_syntax(engines):
    errors = []
    for path in tracked_files():
        if path.parts[0] in engines and path.suffix == ".py":
            try:
                compile((ROOT / path).read_bytes(), str(path), "exec")
            except SyntaxError as exc:
                errors.append(f"{path}:{exc.lineno}: {exc.msg}")
    return errors


def check_docs():
    errors = []
    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    files += list((ROOT / "docs").glob("*.md"))
    for engine in ENGINES:
        files.append(ROOT / engine / "README.md")
        files += list((ROOT / engine / "docs").rglob("*.md"))
        for guide in GUIDES:
            if not (ROOT / engine / "docs" / (guide + ".md")).is_file():
                errors.append(f"{engine}: missing {guide}.md")
    for path in files:
        if not path.exists():
            errors.append(f"Missing document: {path.relative_to(ROOT)}")
            continue
        # Exclude fenced source examples; validate actual Markdown link destinations.
        text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip().split(' "', 1)[0].strip("<>")
            if urlsplit(target).scheme or target.startswith(("#", "//")):
                continue
            dest = unquote(target.split("#", 1)[0])
            if dest and not (path.parent / dest).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {dest}")
    return errors


def entrypoints(engine):
    # This small section consists only of string assignments in all three packages.
    # Parsing it directly keeps this checker usable on Python 3.10 without tomli.
    text = (ROOT / engine / "pyproject.toml").read_text()
    block = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return dict(re.findall(r'^([\w-]+)\s*=\s*"([\w.:]+)"', block, re.M))


def environment(engine, root=ROOT):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / engine)
    env["PYTHONHASHSEED"] = "0"
    # Checks never need provider credentials. Do not put secrets into reports.
    for key in list(env):
        if key.endswith(("API_KEY", "API_TOKEN")):
            env.pop(key)
    return env


def run(command, cwd, env, timeout=180):
    try:
        p = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": f"Timed out after {timeout}s"}


def check_commands(engine):
    checks = {}
    entries = entrypoints(engine)
    for name in entries:
        # Resolve the console script next to this interpreter; do not accidentally
        # validate another checkout's executable from the caller's PATH.
        script = Path(sys.executable).parent / name
        checks[f"{engine}/{name} --help"] = run([str(script), "--help"], ROOT / engine, environment(engine)) if script.exists() else {
            "returncode": 127, "stdout": "", "stderr": f"Install {engine} in this Python environment first"}
    for name, args in LISTINGS[engine]:
        module, function = entries[name].split(":")
        launcher = f"from {module} import {function}; {function}()"
        checks[f"{engine}/{name} {' '.join(args)}".strip()] = run(
            [sys.executable, "-c", launcher, *args], ROOT / engine, environment(engine))
    return checks


def check_tests(engine):
    if engine != "vhhome":
        return {f"{engine}/pytest": run([sys.executable, "-m", "pytest", "-q"],
                ROOT / engine, environment(engine), timeout=900)}
    # These standalone suites rewrite benchmark ledgers. Run them in a temporary
    # copy, never against the user's original cases/results. Unity itself is not run.
    source = ROOT / engine
    with tempfile.TemporaryDirectory(prefix="ivg-vhhome-check-") as temp:
        target = Path(temp) / engine
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            "simulator", "runs", "tmp", "__pycache__", "*.egg-info", ".pytest_cache"))
        env = environment(engine, Path(temp))
        env["VHHOME_ROOT"] = str(target)
        env["VHHOME_VIRTUALHOME"] = os.environ.get("VHHOME_VIRTUALHOME", str(source / "simulator" / "virtualhome"))
        return {f"vhhome/{script}": run([sys.executable, "tests/" + script], target, env, timeout=900)
                for script in ("test_engine.py", "verify_cases.py")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", action="append", choices=ENGINES)
    ap.add_argument("--docs-only", action="store_true")
    ap.add_argument("--tests", action="store_true", help="also run offline engine/scoring suites")
    ap.add_argument("--snapshot", type=Path, help="save runtime hashes and CLI output")
    ap.add_argument("--compare", type=Path, help="compare runtime hashes and CLI output with a snapshot")
    ap.add_argument("--report", type=Path, help="write detailed results as JSON")
    args = ap.parse_args()
    if args.docs_only and (args.tests or args.snapshot or args.compare):
        ap.error("--docs-only cannot be combined with tests or snapshots")
    engines = args.engine or list(ENGINES)
    errors = check_docs()
    report = {"engines": engines, "documentation_errors": list(errors), "commands": {}, "tests": {}}
    if not args.docs_only:
        report["syntax_errors"] = check_python_syntax(engines)
        errors.extend(report["syntax_errors"])
        report["runtime_hashes"] = runtime_hashes(engines)
        for engine in engines:
            print(f"Checking {engine} command interfaces...", flush=True)
            report["commands"].update(check_commands(engine))
            if args.tests:
                print(f"Running {engine} offline tests...", flush=True)
                report["tests"].update(check_tests(engine))
    for kind in ("commands", "tests"):
        for name, result in report[kind].items():
            print(f"{'PASS' if result['returncode'] == 0 else 'FAIL'} {name}", flush=True)
            if result["returncode"]:
                errors.append(f"{name}: exit {result['returncode']}")
                print((result["stdout"] + result["stderr"])[-4000:])
    snapshot = {k: report[k] for k in ("engines", "runtime_hashes", "commands") if k in report}
    if args.compare:
        before = json.loads(args.compare.read_text())
        if before["engines"] != engines:
            errors.append("Snapshot engine selection differs")
        for k in ("runtime_hashes", "commands"):
            old, new = before[k], snapshot[k]
            for name in sorted(old.keys() | new.keys()):
                if old.get(name) != new.get(name):
                    errors.append(f"Changed {k}: {name}")
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, indent=2) + "\n")
    if args.report:
        report["errors"] = list(errors)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for error in errors:
        print("FAIL", error)
    print(f"{len(errors)} failure(s); {len(report['commands'])} CLI checks; {len(report['tests'])} test suites")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
