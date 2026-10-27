#!/usr/bin/env bash
# Muse Glimmer 30B: RedHatAI INT4 (W4A16 GPTQ, Marlin on A100) + DFlash speculative drafter,
# OpenAI-compatible on 127.0.0.1:18200, GPU 1.  A100 has no FP8, so the FP8-block variant is out.
# Audit 2026-09-19 (local_models/museglimmer/AUDIT.md): the official BF16 checkpoint fits one A100-80GB
# (MODEL=meta-models/Muse-Glimmer-30B GPU_UTIL=0.92) but was +0.9 s median with no demonstrable detection gain,
# so INT4 + DFlash stays the default.
#   start:  nohup bash serve_museglimmer.sh > logs/serve_museglimmer.log 2>&1 &
#   stop:   kill $(cat logs/serve_museglimmer.pid)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$HOME}                  # where envs/<name> and hf_cache/<name> live
export HF_HOME=${HF_HOME:-$ROOT/hf_cache/museglimmer}
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=${MODEL:-RedHatAI/Muse-Glimmer-30B-INT4}
SPEC=${SPEC:-1}
EXTRA=${EXTRA:-}
[ "$SPEC" = "1" ] && EXTRA="$EXTRA --speculative-config {\"method\":\"dflash\",\"model\":\"meta-models/Muse-Glimmer-30B-assistant\",\"num_speculative_tokens\":15}"
# Guided JSON must apply from the first token: the nothink profile pre-fills the answer channel, so the
# muse_glimmer reasoning parser never sees reasoning end and a schema applied "after reasoning" is never
# applied at all (bench 2026-09-15: 11/24 valid without this flag, 24/24 with it, median 4.1 s).
EXTRA="$EXTRA --structured-outputs-config {\"enable_in_reasoning\":true}"
mkdir -p "${LOGDIR:=${TMPDIR:-/tmp}/cooksim_serving}"
echo $$ > "$LOGDIR/serve_museglimmer.pid"
exec "$ROOT/envs/museglimmer/bin/vllm" serve "$MODEL" \
  --served-model-name museglimmer \
  --host 127.0.0.1 --port 18200 \
  --tensor-parallel-size 1 --dtype auto \
  --max-model-len ${MAXLEN:-32768} \
  --max-num-seqs ${MAX_SEQS:-2} \
  --limit-mm-per-prompt '{"image": 8}' \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --enable-prefix-caching \
  --reasoning-parser muse_glimmer \
  --generation-config auto \
  --no-enable-log-requests \
  $EXTRA
