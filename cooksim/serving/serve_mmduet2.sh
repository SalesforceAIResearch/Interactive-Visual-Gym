#!/usr/bin/env bash
# MMDuet2 (wangyueqian/MMDuet2 = Qwen2.5-VL-3B + SFT + multi-turn RL for proactive replies), bf16, the authors' own
# demo loop (src/demo/api_server.py ProactiveInferenceAPIClient on src/proactive_eval/qwen2_5_vl; transformers 4.49.0,
# torch 2.4.0, flash-attn 2.6.3, qwen-vl-utils 0.0.8 = demo/requirements.txt) behind an OpenAI-compatible shim on
# 127.0.0.1:18900. shim_server.py replays the <=8 frames as 2-frame user turns (official num_frames_per_turn=2), lets
# the model answer or say "NO REPLY" after every turn (greedy, official), returns the LAST turn's output
# ("NO REPLY" -> text ""), and WRAPS plain text into the reply schema.
#   start: flock -w 10800 /tmp/gpu1.lock bash -c 'CUDA_VISIBLE_DEVICES=1 exec bash serve_mmduet2.sh' > logs/serve_mmduet2.log 2>&1 &
#   stop:  kill $(cat logs/serve_mmduet2.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$HOME}                  # where envs/<name> and hf_cache/<name> live
LMD="$HERE/shims/mmduet2"
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/mmduet2} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PORT=${PORT:-18900} SERVED_NAME=mmduet2
export MMDUET2_SYSTEM=${MMDUET2_SYSTEM:-official} MMDUET2_MAX_PIXELS=${MMDUET2_MAX_PIXELS:-100352} MMDUET2_FRAME_INTERVAL=${MMDUET2_FRAME_INTERVAL:-1.2}
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/cooksim_serving}" "$LMD/logs"; echo $$ > "$LOGDIR/serve_mmduet2.pid"
exec "$ROOT/envs/mmduet2/bin/python" "$LMD/shim_server.py"
