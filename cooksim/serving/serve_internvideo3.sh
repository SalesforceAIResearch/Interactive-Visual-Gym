#!/usr/bin/env bash
# InternVideo3-8B-Instruct (bf16; Qwen3-VL vision tower + M2LA latent-attention LLM, trust_remote_code), served by a
# transformers shim (vLLM 0.19.1 has no `internvideo3` architecture) OpenAI-compatible on 127.0.0.1:18700.
# One request at a time; response_format json_schema/json_object is enforced with xgrammar. See serving/notes/internvideo3.md.
#   start: flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_internvideo3.sh' > logs/serve_internvideo3.log 2>&1 &
#   stop:  kill $(cat logs/serve_internvideo3.pid)
# FRAMES=video (default: the 8 frames as one clip, TICK_SECONDS apart) | FRAMES=images (8 separate images).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$HOME}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/internvideo3-8b} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PORT=${PORT:-18700} SERVED_NAME=internvideo3-8b FRAMES=${FRAMES:-video} TICK_SECONDS=${TICK_SECONDS:-1.2}
export ATTN=${ATTN:-flash_attention_2}   # audit 2026-09-19: the authors' eval scripts use FA2 (card: sdpa); FA2 is ~3% faster, same greedy text 6/8
MODEL=$(ls -d $HF_HOME/hub/models--yanziang--InternVideo3-8B-Instruct/snapshots/*/ | head -1)
export MODEL   # local snapshot of yanziang/InternVideo3-8B-Instruct: a repo id makes transformers 4.57.3 call the Hub (model_info) even offline
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/cooksim_serving}"; echo $$ > "$LOGDIR/serve_internvideo3.pid"
exec "$ROOT/envs/internvideo3-8b/bin/python" "$HERE/shims/internvideo3/shim_server.py"
