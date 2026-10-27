#!/usr/bin/env bash
# Serve one open-weight assistant of the roster on its port, with its own script:
#
#   bash serving/serve.sh <tag> [port]          e.g.  bash serving/serve.sh qwen3.8-27b
#
# Each model has its own Python environment and weights under $VHHOME_MODELS (default $HOME): see
# serving/README.md for what to install per model. The server is OpenAI-compatible on 127.0.0.1:<port>;
# the roster entry carries the port and request profile, so after this the model is run with
# `vhhome-benchmark --assistant <tag>`.
set -euo pipefail
TAG=${1:?usage: serving/serve.sh <tag> [port]}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENG="$(dirname "$HERE")"
PY=${PY:-python3}
read_entry() { PYTHONPATH="$ENG" "$PY" -c "from vhhome_bench import roster as r; print(r.by_tag('$TAG').get('$1', ''))"; }
SERVE=$(read_entry serve); PORT=${2:-$(read_entry port)}
[ -n "$SERVE" ] || { echo "$TAG is not a locally served model (see vhhome-roster)"; exit 2; }
export LOGDIR=${LOGDIR:-$ENG/runs/serving}; mkdir -p "$LOGDIR"
echo "[serve] $TAG -> 127.0.0.1:$PORT  ($HERE/$SERVE, logs in $LOGDIR)"
cd "$HERE"
exec env PORT="$PORT" bash "$HERE/$SERVE"
