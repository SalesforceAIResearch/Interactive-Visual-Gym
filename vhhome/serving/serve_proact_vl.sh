#!/usr/bin/env bash
# Proact-VL (LiveCC-7B-Base = Qwen2-VL-7B + released LoRA incl. the <|FLAG|> response head), bf16, transformers
# 4.57.1 via the authors' own inference code, behind an OpenAI-compatible shim on 127.0.0.1:18950.
# Not a vLLM architecture: shim_server.py replays the <=8 frames as the model's native 1-s chunks, reads the FLAG
# head per chunk, and WRAPS the free-text utterance into the reply schema (the model cannot emit the JSON itself).
#   start: flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_proact_vl.sh' > logs/serve_proact_vl.log 2>&1 &
#   stop:  kill $(cat logs/serve_proact_vl.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
LMD="$HERE/shims/proact-vl"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/proact-vl} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PORT=${PORT:-18950} SERVED_NAME=proact-vl
export PROACT_THRESHOLD=${PROACT_THRESHOLD:-0.5} PROACT_NEW_CHUNKS=${PROACT_NEW_CHUNKS:-2} PROACT_MAX_NEW=${PROACT_MAX_NEW:-12} PROACT_QUERY_AT=${PROACT_QUERY_AT:-new}
# audit 2026-09-19: the authors' Ego4D guidance system prompt (PROACT_PROMPT=ego4d) replaces the 13k-char rulebook, which the
# model recited; PROACT_PROMPT=bench restores the benchmark system prompt. See $LMD/AUDIT.md.
export PROACT_PROMPT=${PROACT_PROMPT:-ego4d}
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"; echo $$ > "$LOGDIR/serve_proact_vl.pid"
exec "$ROOT/envs/proact-vl/bin/python" "$LMD/shim_server.py"
