#!/usr/bin/env bash
# Start independent headless Unity instances; never kill unrelated processes.
# bash simulator/launch_simulators.sh 2 [--port 8080 --display 10]
# bash simulator/launch_simulators.sh 2 --stop
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -x /usr/bin/Xvfb ] && [ -e /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so ]; then
  : # Prefer the working system GL stack.
elif [ -f "$HERE/gl_stack_env.sh" ]; then
  . "$HERE/gl_stack_env.sh"
fi
exec "${VHHOME_PYTHON:-python3}" "$HERE/launch_simulators.py" "$@"
