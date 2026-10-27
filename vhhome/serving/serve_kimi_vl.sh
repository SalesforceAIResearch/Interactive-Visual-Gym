#!/usr/bin/env bash
# Kimi-VL-A3B-Thinking-2506 (moonshotai; MoE 16B total / ~3B active + MoonViT, bf16), vLLM 0.19.1 native
# KimiVLForConditionalGeneration, OpenAI-compatible on 127.0.0.1:18650.
# The model thinks in "◁think▷ … ◁/think▷" (multi-token markers; vLLM 0.19.1 has no parser for them), so a
# reasoning-parser plugin ($HERE/shims/kimi-vl/kimi_vl_reasoning_parser.py) splits thinking into reasoning_content and
# applies the reply json_schema only after ◁/think▷. --reasoning-config enables the per-request
# `thinking_token_budget` (short-budget variant). Thinking off = assistant prefill "◁think▷◁/think▷" with
# continue_final_message (see the kimi-vl-nothink profile in serving/notes/kimi-vl.md).
# Audit 2026-09-18 (serving/notes/kimi-vl.audit.md): --max-num-batched-tokens 32768 is the authors' own serve flag
# (github.com/MoonshotAI/Kimi-VL README, "OpenAI-Compatible Server"); vLLM 0.19.1's A100 default is 2048.
# Uncached prefill 857 -> 740 ms median, faster on 22/22 calls.
#   start: flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_kimi_vl_thinking.sh' \
#            > logs/serve_kimi_vl_thinking.log 2>&1 &        stop: kill $(cat logs/serve_kimi_vl_thinking.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${VHHOME_MODELS:-${ROOT:-$HOME}}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/kimi-vl-thinking} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0          # no nvcc on this box: avoid the JIT sampler
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/vhhome_serving}"; echo $$ > "$LOGDIR/serve_kimi_vl_thinking.pid"
exec "$ROOT/envs/kimi-vl-thinking/bin/vllm" serve moonshotai/Kimi-VL-A3B-Thinking-2506 \
  --served-model-name kimi-vl-thinking --host 127.0.0.1 --port 18650 \
  --dtype bfloat16 --trust-remote-code \
  --max-model-len ${MAXLEN:-32768} --max-num-seqs ${MAX_SEQS:-2} --max-num-batched-tokens ${MNBT:-32768} \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --limit-mm-per-prompt '{"image": 12, "video": 0}' \
  --reasoning-parser-plugin "$HERE/shims/kimi-vl/kimi_vl_reasoning_parser.py" --reasoning-parser kimi_vl \
  --reasoning-config '{"reasoning_start_str": "◁think▷", "reasoning_end_str": "◁/think▷"}' \
  --enable-prefix-caching --no-enable-log-requests ${EXTRA:-}
