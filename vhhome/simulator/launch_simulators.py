"""Launch/stop only owned Unity and Xvfb processes, with explicit readiness checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def start_token(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_owned(record):
    for entry in reversed(record.get("processes", [])):
        pid = entry["pid"]
        if not entry["start_token"] or start_token(pid) != entry["start_token"]:
            continue
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def port_busy(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return None


def launch(args):
    indices = [int(os.environ["ONLY_INDEX"])] if os.environ.get("ONLY_INDEX") else list(range(args.count))
    if any(i < 0 or i >= args.count for i in indices):
        raise ValueError("ONLY_INDEX must lie between 0 and count - 1")
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.stop:
        for i in indices:
            record_path = args.state_dir / f"{args.port+i}.json"
            if record_path.exists():
                stop_owned(json.loads(record_path.read_text()))
                record_path.unlink()
        return
    simulator_dir = Path(os.environ.get("VHHOME_SIMULATOR", str(Path(__file__).parent)))
    unity_dir = Path(os.environ.get("VHHOME_UNITY_DIR", str(simulator_dir / "unity_vol"))).resolve()
    executable = unity_dir / "linux_exec.v2.3.0.x86_64"
    if not os.access(executable, os.X_OK):
        raise ValueError(f"Missing executable: {executable}; install VirtualHome v2.3.0 first")
    xvfb = shutil.which("Xvfb")
    if not xvfb:
        raise ValueError("Xvfb is missing; install system dependencies or run simulator/build_gl_stack.sh")
    for i in indices:
        port, display = args.port+i, args.display+i
        if port_busy(port):
            raise ValueError(f"Port {port} is occupied; choose --port or stop its owner explicitly")
        if Path(f"/tmp/.X{display}-lock").exists() or Path(f"/tmp/.X11-unix/X{display}").exists():
            raise ValueError(f"Display :{display} is occupied; choose --display")
    env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe",
               LP_NUM_THREADS=os.environ.get("LP_NUM_THREADS", "9"))
    runtime = args.state_dir / "runtime"
    runtime.mkdir(exist_ok=True, mode=0o700)
    env["XDG_RUNTIME_DIR"] = str(runtime)
    started = []
    try:
        for i in indices:
            port, display = args.port+i, args.display+i
            record = {"port": port, "display": display, "processes": []}
            record_path = args.state_dir / f"{port}.json"
            started.append((record, record_path))
            commands = [
                ("xvfb", [xvfb, f":{display}", "-screen", "0", "1280x1024x24", "+extension", "GLX"]),
                ("unity", [str(executable), "-batchmode", "-force-glcore", "-job-worker-count",
                           os.environ.get("UNITY_JOB_WORKERS", "16"), f"-http-port={port}"]),
            ]
            processes = []
            for label, command in commands:
                logfile = args.state_dir / f"{label}-{port}.log"
                with logfile.open("ab") as log:
                    process = subprocess.Popen(command, cwd=unity_dir, env=dict(env, DISPLAY=f":{display}"),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                processes.append(process)
                record["processes"].append({"pid": process.pid, "start_token": start_token(process.pid)})
                record_path.write_text(json.dumps(record))
                if label == "xvfb":
                    deadline = time.monotonic() + 15
                    while not Path(f"/tmp/.X11-unix/X{display}").exists():
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError(f"Xvfb failed to start; see {logfile}")
                        time.sleep(0.1)
            deadline = time.monotonic() + args.timeout
            while health(port) != 501:
                if any(p.poll() is not None for p in processes) or time.monotonic() >= deadline:
                    raise RuntimeError(f"Unity on port {port} is not ready; see {args.state_dir}/unity-{port}.log")
                time.sleep(1)
            print(f"READY port={port} display=:{display} logs={args.state_dir}", flush=True)
    except BaseException:
        for record, path in started:
            stop_owned(record)
            path.unlink(missing_ok=True)
        raise
    print("PORTS_READY " + " ".join("501" for _ in indices), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, nargs="?", default=4)
    parser.add_argument("--port", type=int, default=8080, help="first HTTP port")
    parser.add_argument("--display", type=int, default=10, help="first X display")
    parser.add_argument("--timeout", type=float, default=180, help="readiness timeout per instance in seconds")
    parser.add_argument("--state-dir", type=Path, default=Path.home()/".cache"/"vhhome"/"instances")
    parser.add_argument("--stop", action="store_true", help="stop only processes recorded by this launcher")
    args = parser.parse_args(argv)
    if args.count < 1 or args.display < 0 or args.port < 1 or args.port+args.count > 65536 or args.timeout <= 0:
        parser.error("count/timeout must be positive; ports must be in 1..65535 and display nonnegative")
    args.state_dir = args.state_dir.expanduser().resolve()
    try:
        launch(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
