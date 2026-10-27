#!/usr/bin/env bash
# Molmo2-8B (bf16, Qwen3-8B backbone + SigLIP2), vLLM 0.19.1, OpenAI-compatible on 127.0.0.1:18400, GPU 3.
# The shipped chat template rejects a system message (roles must alternate from user), and the benchmark
# always sends one: molmo2_chat_template_system.jinja prepends the system text to the first user turn and is
# byte-identical to the original for conversations without a system message.
# Audit 2026-09-18 (serving/notes/molmo2.audit.md): AllenAI's vLLM demo uses --max-num-batched-tokens 36864, but on this A100
# it is SLOWER than vLLM's default (2048): uncached prefill 1.72 -> 2.00 s median in two alternating rounds, so the
# default is kept on purpose. Sampling is set in the request profile (greedy, per the demo).
#   start: nohup bash serve_molmo2.sh > logs/serve_molmo2.log 2>&1 &     stop: kill $(cat logs/serve_molmo2.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/molmo2} HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"; echo $$ > "$LOGDIR/serve_molmo2.pid"
exec "$ROOT/envs/molmo2/bin/vllm" serve allenai/Molmo2-8B \
  --served-model-name molmo2-8b --host 127.0.0.1 --port 18400 \
  --dtype auto --trust-remote-code \
  --max-model-len ${MAXLEN:-32768} --max-num-seqs ${MAX_SEQS:-2} \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --limit-mm-per-prompt '{"image": 12, "video": 0}' \
  --chat-template "$HERE/molmo2_chat_template_system.jinja" \
  --enable-prefix-caching --no-enable-log-requests ${EXTRA:-}
