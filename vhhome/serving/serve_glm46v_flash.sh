#!/usr/bin/env bash
# GLM-4.6V-Flash (bf16, Glm4vForConditionalGeneration), vLLM 0.19.1, OpenAI-compatible on 127.0.0.1:18500, GPU from CUDA_VISIBLE_DEVICES (default 1).
#   start: nohup bash serve_glm46v_flash.sh > logs/serve_glm.log 2>&1 &     stop: kill $(cat logs/serve_glm.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/glm} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0          # no nvcc on this box: avoid the JIT sampler
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"; echo $$ > "$LOGDIR/serve_glm.pid"
# Env: $ROOT/envs/glm (dedicated venv: vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0); its vLLM
# registry contains Glm4vForConditionalGeneration — GLM-4.6V-Flash is native vLLM support, no
# remote code. VENV=<other env> overrides.
# Audit 2026-09-18 (serving/notes/glm46v-flash.audit.md): --reasoning-parser glm45 is the vLLM recipe's flag for GLM-4.xV
# (recipes.vllm.ai/zai-org/GLM-4.6V). With thinking off (the benchmark profile) reasoning_content stays empty and
# latency is unchanged (paired A/B); with thinking on it keeps the reply schema off the <think> block.
VENV=${VENV:-$ROOT/envs/glm}
exec "$VENV/bin/vllm" serve zai-org/GLM-4.6V-Flash \
  --served-model-name glm-4.6v-flash \
  --host 127.0.0.1 --port 18500 \
  --tensor-parallel-size 1 --dtype auto \
  --max-model-len ${MAXLEN:-32768} \
  --max-num-seqs ${MAX_SEQS:-2} \
  --limit-mm-per-prompt '{"image": 12, "video": 0}' \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --reasoning-parser glm45 \
  --enable-prefix-caching \
  --no-enable-log-requests ${EXTRA:-}
