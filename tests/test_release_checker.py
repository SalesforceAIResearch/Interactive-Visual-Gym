"""Tests for the release validator; standard library only, no provider calls."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "release_checker", Path(__file__).resolve().parents[1] / "scripts" / "check_release.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class ReleaseCheckerTests(unittest.TestCase):
    def test_repository_documentation(self):
        self.assertEqual(checker.check_docs(), [])

    def test_all_engines_have_entrypoints(self):
        for engine in checker.ENGINES:
            entries = checker.entrypoints(engine)
            self.assertIn(engine + "-episode", entries)
            self.assertIn(engine + "-benchmark", entries)

    def test_source_packages_are_not_ignored_as_generated_runs(self):
        files = {str(p) for p in checker.tracked_files()}
        self.assertIn("screensim/screensim_bench/runs/__init__.py", files)
        self.assertIn("screensim/screensim_bench/runs/sweep.py", files)

    def test_shipped_python_sources_compile(self):
        self.assertEqual(checker.check_python_syntax(checker.ENGINES), [])

    def test_missing_document_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(checker, "ROOT", Path(tmp)):
            errors = checker.check_docs()
            self.assertTrue(any("missing GETTING_STARTED.md" in e for e in errors))
            self.assertTrue(any("Missing document: README.md" in e for e in errors))

    def test_environment_removes_api_secrets(self):
        with patch.dict(checker.os.environ, {"SAMPLE_API_KEY": "not-a-real-key"}):
            env = checker.environment("cooksim")
        self.assertNotIn("SAMPLE_API_KEY", env)
        self.assertEqual(env["PYTHONHASHSEED"], "0")

    def test_runtime_snapshot_excludes_documentation(self):
        hashes = checker.runtime_hashes(checker.ENGINES)
        self.assertGreater(len(hashes), 100)
        self.assertTrue(all(not p.endswith((".md", "CITATION.cff")) for p in hashes))
        self.assertTrue(all("/docs/" not in p for p in hashes))

    def test_runtime_snapshot_detects_content_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / "cooksim" / "example.py"
            code.parent.mkdir()
            code.write_text("value = 1\n")
            with patch.object(checker, "ROOT", root), patch.object(
                    checker, "tracked_files", return_value=[Path("cooksim/example.py")]):
                before = checker.runtime_hashes(["cooksim"])
                code.write_text("value = 2\n")
                self.assertNotEqual(before, checker.runtime_hashes(["cooksim"]))
                code.unlink()
                self.assertEqual(checker.runtime_hashes(["cooksim"]), {})

    def test_subprocess_failure_is_preserved(self):
        result = checker.run([checker.sys.executable, "-c", "raise SystemExit(7)"],
                             checker.ROOT, checker.environment("cooksim"))
        self.assertEqual(result["returncode"], 7)

    def test_subprocess_timeout_is_failure(self):
        result = checker.run([checker.sys.executable, "-c", "import time; time.sleep(10)"],
                             checker.ROOT, checker.environment("cooksim"), timeout=0.05)
        self.assertEqual(result["returncode"], 124)


if __name__ == "__main__":
    unittest.main()
