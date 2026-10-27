"""Installer/path/launcher regression checks, without Unity or provider credentials."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paths = load("vh_paths", ROOT / "vhhome/vhhome/paths.py")
launcher = load("vh_launcher", ROOT / "vhhome/simulator/launch_simulators.py")
INSTALL = ROOT / "vhhome/simulator/install_virtualhome.sh"


class SimulatorInstallationTests(unittest.TestCase):
    def test_both_client_layouts(self):
        for relative in ("simulation", "virtualhome/simulation"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                expected = Path(tmp) / relative
                (expected / "unity_simulator").mkdir(parents=True)
                with patch.object(paths, "VIRTUALHOME", tmp):
                    self.assertEqual(paths.virtualhome_client(), str(expected))

    def test_client_install_is_idempotent_for_both_layouts(self):
        for relative in ("simulation", "virtualhome/simulation"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/"virtualhome"/relative/"unity_simulator").mkdir(parents=True)
                env = dict(os.environ, VHHOME_SIMULATOR=tmp)
                for _ in range(2):
                    run = subprocess.run(["bash", str(INSTALL), "--client-only"], env=env,
                                         capture_output=True, text=True, timeout=15)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertIn("client present", run.stdout)

    def test_existing_unknown_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"virtualhome"
            directory.mkdir()
            marker = directory/"keep.txt"
            marker.write_text("user data")
            run = subprocess.run(["bash", str(INSTALL), "--client-only"],
                env=dict(os.environ, VHHOME_SIMULATOR=tmp), capture_output=True, timeout=15)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(marker.read_text(), "user data")

    def test_fresh_client_clone_pins_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"
            source.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.DEVNULL).decode().strip()
            git("init")
            package = source/"virtualhome/simulation/unity_simulator"
            package.mkdir(parents=True)
            (package/"__init__.py").write_text("")
            git("add", ".")
            git("-c", "user.name=Release test", "-c", "user.email=release-test@example.invalid", "commit", "-m", "fixture")
            revision = git("rev-parse", "HEAD")
            target = Path(tmp)/"installed"
            run = subprocess.run(["bash", str(INSTALL), "--client-only"], capture_output=True, text=True,
                env=dict(os.environ, VHHOME_SIMULATOR=str(target), VHHOME_CLIENT_URL=source.as_uri(),
                         VHHOME_CLIENT_REV=revision), timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            installed = subprocess.check_output(["git", "-C", str(target/"virtualhome"), "rev-parse", "HEAD"]).decode().strip()
            self.assertEqual(installed, revision)

    def test_busy_port_is_detected_without_touching_owner(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            self.assertTrue(launcher.port_busy(sock.getsockname()[1]))

    def test_archive_install_and_wrong_version_rejection(self):
        for version in ("2.3.0", "2.0.0"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/"virtualhome/simulation/unity_simulator").mkdir(parents=True)
                archive = root/"build.zip"
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr(f"build/linux_exec.v{version}.x86_64", "test fixture, not Unity")
                    z.writestr(f"build/linux_exec.v{version}_Data/test.txt", "fixture")
                run = subprocess.run(["bash", str(INSTALL), "--unity-archive", str(archive)],
                    env=dict(os.environ, VHHOME_SIMULATOR=tmp), capture_output=True, text=True, timeout=15)
                self.assertEqual(run.returncode == 0, version == "2.3.0", run.stderr)
                if version == "2.3.0":
                    self.assertTrue(os.access(root/"unity_vol/linux_exec.v2.3.0.x86_64", os.X_OK))
                    repeated = subprocess.run(["bash", str(INSTALL)], env=dict(os.environ, VHHOME_SIMULATOR=tmp),
                                              capture_output=True, timeout=15)
                    self.assertEqual(repeated.returncode, 0)

    def test_stale_pid_is_never_signalled(self):
        record = {"processes": [{"pid": os.getpid(), "start_token": "wrong-token"}]}
        with patch.object(launcher.os, "killpg") as kill:
            launcher.stop_owned(record)
            kill.assert_not_called()

    def test_owned_process_is_stopped(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            token = launcher.start_token(process.pid)
            self.assertIsNotNone(token)
            launcher.stop_owned({"processes": [{"pid": process.pid, "start_token": token}]})
            process.wait(timeout=5)
            self.assertNotEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    def test_bad_launcher_arguments_fail(self):
        for args in (["0"], ["1", "--port", "65536"], ["1", "--timeout", "0"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                launcher.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
