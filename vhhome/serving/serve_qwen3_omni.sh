#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B-Instruct (bf16 MoE, 3B active), vLLM 0.19.1, OpenAI-compatible on 127.0.0.1:18800.
# Text output only: plain vLLM serves the THINKER alone (Qwen3OmniMoeThinkerForConditionalGeneration);
# the talker / code2wav weights are skipped at load (and their two talker-only shards were never
# downloaded), so no audio is generated and no talker memory is spent.
# The Instruct model has no thinking mode, so no chat_template_kwargs / reasoning parser are needed.
#   start: flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_qwen3_omni.sh' > logs/serve_qwen3_omni.log 2>&1 &
#   stop:  kill $(cat logs/serve_qwen3_omni.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/qwen3-omni-30b-a3b} HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0          # no nvcc here: avoid the JIT sampler
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"; echo $$ > "$LOGDIR/serve_qwen3_omni.pid"
exec "$ROOT/envs/qwen3-omni-30b-a3b/bin/vllm" serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --served-model-name qwen3-omni-30b-a3b --host 127.0.0.1 --port 18800 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len ${MAXLEN:-32768} --max-num-seqs ${MAX_SEQS:-2} \
  --gpu-memory-utilization ${GPU_UTIL:-0.92} \
  --limit-mm-per-prompt '{"image": 12, "video": 0, "audio": 0}' \
  --enable-prefix-caching --no-enable-log-requests ${EXTRA:-}
