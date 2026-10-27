# The simulator

VH-Home's world is [VirtualHome](https://github.com/xavierpuigf/virtualhome) v2.3.0, a Unity apartment simulator with a Python client. This page covers installing it, launching headless instances, the software OpenGL stack they render with, and what goes wrong.

## 1. What is installed

Two pieces, both put under `simulator/` by `install_virtualhome.sh` and never committed:

| Piece | Where | What it is |
|---|---|---|
| The Python client | `simulator/virtualhome/` | a pinned VirtualHome checkout; flat and nested `simulation/` layouts are detected automatically |
| The Unity build | `simulator/unity_vol/` | the v2.3.0 Linux player: `linux_exec.v2.3.0.x86_64` (the executable), `linux_exec.v2.3.0_Data/` (its data), and `Output/`, where Unity writes the rendered frames |

The engine finds them through `vhhome/paths.py`, and every location can be moved with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `VHHOME_ROOT` | the checkout | the repository root |
| `VHHOME_SIMULATOR` | `<root>/simulator` | the directory holding `virtualhome/` and `unity_vol/` |
| `VHHOME_UNITY_DIR` | `<simulator>/unity_vol` | the Unity build directory itself |
| `VHHOME_VIRTUALHOME` | `<simulator>/virtualhome` | the VirtualHome client checkout |
| `VHHOME_RUNS` | `<root>/runs` | run output: records, films, per-episode frame scratch |

Set `VHHOME_SIMULATOR` (or the two finer ones) to keep the multi-gigabyte build outside the checkout, or to share one build between several checkouts.

## 2. Installing: `install_virtualhome.sh`

```bash
bash simulator/install_virtualhome.sh --unity-archive /path/to/linux_exec.zip
# For offline tests without Unity:
bash simulator/install_virtualhome.sh --client-only
```

The client is pinned to revision `58970fd80951c2eaa1af713e0917d1a105353ad8`. Existing supported client layouts are reused. The Unity archive must contain `linux_exec.v2.3.0.x86_64` and its matching data directory. The installer checks the version, refuses to overwrite a nonempty installation, and supports an optional `VHHOME_UNITY_SHA256` checksum. It requires `git`, `unzip`, and `curl` when downloading; `VHHOME_SIMULATOR` changes the destination.

**Unity asset prerequisite.** The upstream v2.3.0 download is currently unavailable. Supply an existing local archive or set `VHHOME_UNITY_URL` to an accessible copy of that version. The installer does not silently replace it with another version. An already installed executable is reused without downloading.

## 3. Launching: `launch_simulators.sh`

```bash
bash simulator/launch_simulators.sh [N]              # N instances, default 4
bash simulator/launch_simulators.sh 2 --port 9080 --display 30 --state-dir /tmp/vh-demo
bash simulator/launch_simulators.sh 2 --port 9080 --state-dir /tmp/vh-demo --stop
```

For each index `k` in `0..N-1` the script:

1. verifies that port `8080 + k` and display `:10 + k` are unused; unrelated processes are never killed;
2. starts `Xvfb :<display> -screen 0 1280x1024x24 +extension GLX`, logging to `xvfb-<port>.log` in the state directory;
3. starts the Unity player **inside `unity_vol/`** with `-batchmode -force-glcore -job-worker-count $UNITY_JOB_WORKERS -http-port=<port>`, logging to `unity-<port>.log`;
4. polls `http://127.0.0.1:<port>/` for up to three minutes until it answers HTTP **501**.

`501` is the healthy answer: Unity rejects a plain GET. `PORTS_READY 501 501 ...` is printed only after every instance is ready. A startup failure exits nonzero and cleans up processes started by that invocation. Instances otherwise survive the launching shell. The default state directory is `~/.cache/vhhome/instances/`; `--stop` uses its PID/start-time records to stop only owned processes. Use `ONLY_INDEX=k` with `--stop`, then without it, to restart one instance.

Unity writes frames to `Output/` under its own working directory and the engine reads them from `<unity dir>/Output`, which is why the process must start inside `unity_vol/`. If you launch Unity by hand, `cd` there first.

**One episode per instance at a time.** An instance holds one scene. Two episodes on one port reset each other's scene and both records are corrupt. Start as many instances as episodes you want to run at once, and give each episode its own port (`vhhome-episode --port`, `vhhome-benchmark --engines 8080,8081,...`).

## 4. The software-GL stack

Rendering is software OpenGL, Mesa's `llvmpipe` rasterizer, under a virtual X display (Xvfb). No GPU is needed; one instance uses a few CPU cores. The launcher exports `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` so that a GPU, if present, is ignored.

### With root: system packages

The launcher uses the system stack whenever `/usr/bin/Xvfb` and Mesa's software rasterizer (`swrast_dri.so`) are installed. On Ubuntu 22.04 or 24.04:

```bash
apt-get install -y --no-install-recommends xvfb libgl1 libglx-mesa0 libgl1-mesa-dri libglu1-mesa \
  mesa-utils libxext6 libxrender1 libxtst6 libxi6 libxcursor1 libxrandr2 libxinerama1 libxxf86vm1 \
  libxfixes3 libxdamage1 libxcomposite1 libxss1 libasound2t64 libnss3 libgbm1 curl
```

On Ubuntu 20.04 the names are `libgl1-mesa-glx` and `libasound2`.

### Without root: `build_gl_stack.sh`

```bash
bash simulator/build_gl_stack.sh          # once; needs network, about two minutes
```

The script downloads the Ubuntu `.deb` packages the stack needs (`apt-get download`, no root), unpacks them with `dpkg -x` under `~/.vhhome/glstack` (`GLSTACK_PREFIX` overrides), and writes `simulator/gl_stack_env.sh`, which puts the private binaries, libraries, DRI drivers, keyboard data and fonts on the relevant paths. The launcher sources that file automatically **when the system stack is missing**. Two details the script's comments explain: Xvfb shells out to the `xkbcomp` binary to build its keymap, so the `x11-xkb-utils` package must be in the stack or the display aborts; and recent Mesa splits the renderer into `mesa-libgallium` with its own LLVM, so those packages are in the list too.

### The shadowing warning

A private stack must never shadow a working system stack. Its loader paths (`LD_LIBRARY_PATH`, `PATH`) win over `/usr`; on a machine that already has Xvfb and Mesa, sourcing `gl_stack_env.sh` pairs the system Xvfb with the private libraries (or the reverse), the display comes up with no GLX visual, and Unity logs `Unable to find a supported OpenGL core profile`. That message reads like a GPU problem and is not. The launcher guards against it by preferring the system stack whenever it is present; keep your own shell from sourcing `gl_stack_env.sh` on such a machine, and if you see the message, check `echo $LD_LIBRARY_PATH` and `which Xvfb` before anything else.

## 5. Threads: `UNITY_JOB_WORKERS` and `LP_NUM_THREADS`

| Variable | Default | What it sets |
|---|---|---|
| `UNITY_JOB_WORKERS` | 16 | Unity's job-worker thread count per instance (`-job-worker-count`). Without it Unity sizes the pool to the machine, and on a node with hundreds of cores it then never opens its port. |
| `LP_NUM_THREADS` | 9 | `llvmpipe` render threads per instance. |

Together they bound the CPU one instance uses. Lower them to pack more instances onto a machine; raise `LP_NUM_THREADS` on an otherwise idle one.

## 6. The container image: `simulator/Dockerfile`

The Dockerfile builds an Ubuntu 20.04 image with the same software-GL stack: Xvfb, the Mesa packages (the 20.04 Mesa gives `llvmpipe` an OpenGL 4.5 core profile, which is enough for the v2.3.0 player in batch mode), the X11 and audio libraries the player links against, `curl` and certificates. It sets `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe`, `LP_NUM_THREADS=9` and `DISPLAY=:1`, and its working directory is `/unity_vol`. It does **not** contain the Unity build or this repository: mount or copy `unity_vol/` to `/unity_vol` (and point `VHHOME_UNITY_DIR` at it) and the checkout alongside, then run `launch_simulators.sh` inside the container.

## 7. Throughput

A 20-step episode renders in roughly 10 to 30 minutes on one instance; every command the person issues is animated and rendered at 5 frames per tick before the assistant sees it, so the simulator, not the models, sets the floor. Instances are independent, so throughput scales with the number of instances a machine can host under the thread settings above. API assistants may use several instances at once (`--engines`); a model served locally must use exactly one, because its reply latency is charged to the game clock and two episodes sharing the server would each be charged the other's compute.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `missing .../unity_vol/linux_exec.v2.3.0.x86_64 -- run simulator/install_virtualhome.sh first` | The build is not where the launcher looks. Run the installer, or set `VHHOME_UNITY_DIR` to the directory that contains the executable. |
| `Xvfb not found. Install the packages listed at the top of this script, or build a private stack` | No X server on the machine. Install the packages (section 4) or run `build_gl_stack.sh`. |
| Unity readiness timeout | Read `unity-<port>.log` in the state directory; on a many-core host see `UNITY_JOB_WORKERS` (section 5). |
| `Unable to find a supported OpenGL core profile` | No GLX visual on the display: a private GL stack is shadowing the system one, or the Mesa packages are incomplete. Section 4. |
| The Xvfb log mentions `xkbcomp` | Rebuild the private stack with `build_gl_stack.sh`, which includes `x11-xkb-utils`, or install the system package. The launcher does not modify system directories. |
| Frames missing mid-episode (`FileNotFoundError: Action_..._normal.png`) | Two episodes rendered on the same instance at the same time and deleted each other's frames. One episode per port. |
| An instance that stopped answering | Run `ONLY_INDEX=k bash simulator/launch_simulators.sh N --stop`, then repeat without `--stop`. |
