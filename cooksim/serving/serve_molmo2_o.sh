#!/usr/bin/env bash
# Molmo 2-O 7B (allenai/Molmo2-O-7B: OLMo text backbone + SigLIP2, weights shipped fp32, served bf16),
# vLLM 0.19.1 native Molmo2ForConditionalGeneration, OpenAI-compatible on 127.0.0.1:18600.
# Its shipped chat template is byte-identical to Molmo2-8B's and rejects a system message, which every
# benchmark call sends; molmo2_chat_template_system.jinja (the Molmo2-8B fix) folds the system text into the
# first user turn and renders byte-identically when there is no system message.
#   start: flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_molmo2_o_7b.sh' \
#            > logs/serve_molmo2_o_7b.log 2>&1 &            stop: kill $(cat logs/serve_molmo2_o_7b.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$HOME}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/molmo2-o-7b} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0          # no nvcc on this box: avoid the JIT sampler
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/cooksim_serving}"; echo $$ > "$LOGDIR/serve_molmo2_o_7b.pid"
exec "$ROOT/envs/molmo2-o-7b/bin/vllm" serve allenai/Molmo2-O-7B \
  --served-model-name molmo2-o-7b --host 127.0.0.1 --port 18600 \
  --dtype bfloat16 --trust-remote-code \
  --max-model-len ${MAXLEN:-32768} --max-num-seqs ${MAX_SEQS:-2} \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --chat-template "$HERE/molmo2_chat_template_system.jinja" \
  --enable-prefix-caching --no-enable-log-requests ${EXTRA:-}
