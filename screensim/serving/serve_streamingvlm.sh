#!/usr/bin/env bash
# StreamingVLM (mit-han-lab/StreamingVLM, Qwen2.5-VL-7B fine-tune), bf16, the authors' streaming inference code
# (transformers 4.51.3 + their patched attention / contiguous-RoPE KV cache), behind an OpenAI-compatible shim on
# 127.0.0.1:19000. Not served by vLLM: the streaming KV policy is custom code. shim_server.py replays the <=8 frames
# as native 1-s chunks, lets the model comment per second, returns the commentary of the 2 newest chunks, and WRAPS
# it into the reply schema (the model does not emit the JSON itself).
#   start: flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_streamingvlm.sh' > logs/serve_streamingvlm.log 2>&1 &
#   stop:  kill $(cat logs/serve_streamingvlm.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$HOME}                  # where envs/<name> and hf_cache/<name> live
LMD="$HERE/shims/streamingvlm"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/streamingvlm} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PORT=${PORT:-19000} SERVED_NAME=streamingvlm SVLM_NEW_CHUNKS=${SVLM_NEW_CHUNKS:-2}
# audit 2026-09-19: the authors' loop has no system turn -- context goes in "previous text" (SVLM_PROMPT=native: one-line
# video description + the recipe line; the rulebook is left out). SVLM_PROMPT=bench restores the benchmark system turn.
export SVLM_PROMPT=${SVLM_PROMPT:-native} SVLM_QUERY_AT=${SVLM_QUERY_AT:-first}
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/cooksim_serving}" "$LMD/logs"; echo $$ > "$LOGDIR/serve_streamingvlm.pid"
exec "$ROOT/envs/streamingvlm/bin/python" "$LMD/shim_server.py"
