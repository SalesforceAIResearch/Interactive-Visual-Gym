#!/usr/bin/env bash
# Qwen3.8-27B (bf16) + built-in MTP speculative decoding, OpenAI-compatible on 127.0.0.1:18300, GPU 0.
# Thinking is OFF server-wide (the harness profile sends no chat_template_kwargs); a request can
# still turn it on with chat_template_kwargs={"enable_thinking": true}.
# Audit 2026-09-18 (local_models/qwen/AUDIT.md): vLLM auto-detects this template's content format as 'string' and
# then moves every image in front of the text (joined by "\n"), discarding the order the harness sends.
# --chat-template-content-format openai renders token-for-token what the official HF processor renders.
# MTP depth 5 (MTP_K; recipe default 3, its rtx_pro_6000/GB10 override 5): 4.73 s median vs 5.6-5.8 s at depth 3,
# 86 vs 68 tok/s decode, mean acceptance length 5.2 of 6, 24/24 schema-valid.
#   start:  nohup bash serve_qwen38_27b.sh > logs/serve_qwen38_27b.log 2>&1 &
#   stop:   kill $(cat logs/serve_qwen38_27b.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/qwen}
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0          # no nvcc here: avoid the JIT sampler
# H100 (2026-09-30): on sm90 vLLM picks the flashinfer gated-delta-net prefill kernel, whose JIT needs nvcc
# ("Could not find nvcc and default cuda_home=/usr/local/cuda doesn't exist"); on the A100 it always used the
# Triton/FLA path. Pin that path everywhere so H100 and A100 runs use the same kernel.
GDN_BACKEND=${GDN_BACKEND:-triton}
SPEC=${SPEC:-1}
EXTRA=${EXTRA:-}
[ "$SPEC" = "1" ] && EXTRA="$EXTRA --speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K:-5}}"
[ "${THINK:-0}" != "1" ] && EXTRA="$EXTRA --default-chat-template-kwargs {\"enable_thinking\":false}"
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"
echo $$ > "$LOGDIR/serve_qwen38_27b.pid"
exec "$ROOT/envs/qwen/bin/vllm" serve Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 18300 \
  --tensor-parallel-size 1 --dtype auto \
  --max-model-len ${MAXLEN:-32768} \
  --max-num-seqs ${MAX_SEQS:-2} \
  --limit-mm-per-prompt '{"image": 12}' \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --chat-template-content-format openai \
  --no-enable-log-requests \
  --additional-config "{\"gdn_prefill_backend\":\"$GDN_BACKEND\"}" \
  $EXTRA
