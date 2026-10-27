#!/usr/bin/env bash
# MiniCPM-o 4.5 (AWQ W4A16 default | VARIANT=bf16), vLLM 0.19.1, OpenAI-compatible on 127.0.0.1:18100, GPU 2.
#   start: nohup bash serve_minicpmo.sh > logs/serve_minicpmo.log 2>&1 &     stop: kill $(cat logs/serve_minicpmo.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/minicpmo} HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export VLLM_USE_FLASHINFER_SAMPLER=0
VARIANT=${VARIANT:-awq}
if [ "$VARIANT" = "awq" ]; then R=models--openbmb--MiniCPM-o-4_5-awq; else R=models--openbmb--MiniCPM-o-4_5; fi
MODEL_DIR=$(ls -d $HF_HOME/hub/$R/snapshots/*/ | head -1)
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"; echo $$ > "$LOGDIR/serve_minicpmo.pid"
exec "$ROOT/envs/minicpmo-vllm/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name minicpmo --host 127.0.0.1 --port 18100 \
  --dtype bfloat16 --trust-remote-code \
  --max-model-len ${MAXLEN:-16384} --max-num-seqs ${MAX_SEQS:-2} \
  --gpu-memory-utilization ${GPU_UTIL:-0.85} \
  --limit-mm-per-prompt '{"image": 12, "video": 0, "audio": 0}' \
  --mm-processor-kwargs "{\"max_slice_nums\": ${MAX_SLICE_NUMS:-1}}" \
  --enable-prefix-caching --no-enable-log-requests
