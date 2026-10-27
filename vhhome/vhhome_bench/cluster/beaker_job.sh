#!/usr/bin/env bash
# What a VH Beaker batch job runs: one roster assistant, on the job's own GPU, from the engine checkout on
# the shared mount (no git clone — the job sees the same code, envs, weights and outputs as a session).
#
#   bash tools/beaker_job_vh.sh check <tag> <root>            # 3 cells of the benchmark table
#   bash tools/beaker_job_vh.sh lane  <tag> <root> [shard]    # the full 75-cell lane, or shard k/n
#   bash tools/beaker_job_vh.sh calib <tag> <root>            # latency only: no engine, no episodes
#
# SHARD may be "auto/<n>", which a REPLICA job resolves to "$((BEAKER_REPLICA_RANK+1))/<n>". One
# experiment with n replicas is one row in the Beaker UI instead of n rows (project decision 2026-09-22), and
# each replica still runs its own disjoint slice of the 75 cells.
#
# Two servers come up inside the job:
#   * the VirtualHome engine — Unity on a virtual display, software GL (llvmpipe) on the job's CPUs. The
#     GPU is left to the model. tools/launch_native_hostgl.sh caps Unity's job workers, which a 256-CPU
#     node needs (it otherwise hangs before opening its port).
#   * the assistant's own server, from CookSim/serving/<serve script>, on its roster port, with
#     that model's env and weights (both on the shared filesystem). API arms skip this.
# Submitted by vhhome_bench/cluster/beaker_submit.py; everything the job writes lands under <root> on the shared filesystem.
set -uo pipefail
MODE=${1:?mode}; TAG=${2:?tag}; ROOT_OUT=${3:?root}; SHARD=${4:-}; CELLS=${5:-}
[ "$SHARD" = "-" ] && SHARD=""      # "-" is the empty placeholder: an empty WORD shifts the arguments
[ "$CELLS" = "-" ] && CELLS=""
ENG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"      # the checkout (this script is vhhome_bench/cluster/)
cd "$ENG"
export CUDA_VISIBLE_DEVICES=0                       # the job's one GPU is the model's
export LOGDIR="$ENG/$ROOT_OUT/serve_logs"; mkdir -p "$LOGDIR"
LOGSFX="${BEAKER_JOB_ID:-$$}-${BEAKER_REPLICA_RANK:-0}"   # one log per job: siblings must not overwrite
export PYTHONUNBUFFERED=1
PY=${PY:-python3}
PORT_ENGINE=${PORT_ENGINE:-8080}

case "$SHARD" in
  auto/*) SHARD="$(( ${BEAKER_REPLICA_RANK:-0} + 1 ))/${SHARD#auto/}" ;;
esac
echo "[job] $(date -u +%FT%TZ) host=${BEAKER_NODE_HOSTNAME:-?} gpus=${BEAKER_ASSIGNED_GPU_COUNT:-?} mode=$MODE tag=$TAG root=$ROOT_OUT shard=${SHARD:-all} replica=${BEAKER_REPLICA_RANK:-none}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ---- 0. what card did we get, and what does a second on it cost in A100 seconds?
# A lane may land on an A100, an L40S or an H100 depending on which cluster had room. The model's
# generation time is charged to the game clock, so the same arm would be judged differently on different
# hardware. The job exports the measured ratio for ITS card (docs/GPU_LATENCY_RATIO.json, written by
# -m vhhome_bench.cluster.latency_ratio) and the rollout charges gen_s / ratio — A100-equivalent seconds everywhere.
VH_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
export VH_GPU_NAME
# VH_LAT_NORMALISE=0 (project decision 2026-10-05): every served model runs on the same card (the H100 cluster, H100), so its
# generation time is charged as measured — no A100-equivalent factor.
if [ "${VH_LAT_NORMALISE:-1}" = "0" ]; then
    export ASSIST_LATENCY_DIV=1
    echo "[job] gpu='$VH_GPU_NAME' ASSIST_LATENCY_DIV=1 (VH_LAT_NORMALISE=0: gen time charged as measured on this card)"
else
export ASSIST_LATENCY_DIV="$("$PY" -m vhhome_bench.cluster.latency_ratio --lookup "$TAG" --gpu "$VH_GPU_NAME" 2>/dev/null || echo 1)"
echo "[job] gpu='$VH_GPU_NAME' ASSIST_LATENCY_DIV=$ASSIST_LATENCY_DIV (gen time charged as A100 seconds)"
fi

read_entry() { "$PY" -c "from vhhome_bench import roster as r; e=r.by_tag('$TAG'); print(e.get('$1',''))"; }
SERVE=$(read_entry serve); MPORT=$(read_entry port)

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

# ---- 1. the VirtualHome engine (CPU, software GL). A calibration job plays no episodes, so it skips
# the engine entirely and only times the model's replies.
if [ "$MODE" != "calib" ]; then
setsid bash simulator/launch_simulators.sh 1 > "$LOGDIR/engine.$LOGSFX.log" 2>&1 < /dev/null & PIDS+=($!)
for i in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT_ENGINE/)" = "501" ] && break
    sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT_ENGINE/)" = "501" ] || {
    echo "[job] VH engine never came up, see $LOGDIR/engine.$LOGSFX.log"; tail -20 "$LOGDIR/engine.$LOGSFX.log"; exit 3; }
echo "[job] VH engine ready on :$PORT_ENGINE"
fi

# ---- 2. the assistant's server (local models only)
if [ -n "$SERVE" ]; then
    # THROUGH THE VH WRAPPER, never CookSim's script directly: VH sends 8 frames + 2 maps and the
    # cook scripts cap vLLM at 8 images, so a directly-started server answers "At most 8 image(s) may be
    # provided in one prompt" to every call. The harness then retries 3x and records an empty reply, so a
    # whole check round looked like 12 mute models (round 1, 2026-09-20).
    PORT_M=$MPORT setsid bash serving/serve.sh "$TAG" "$MPORT" > "$LOGDIR/serve_$TAG.$LOGSFX.log" 2>&1 < /dev/null & PIDS+=($!)
    for i in $(seq 1 2400); do
        curl -sf "http://127.0.0.1:$MPORT/v1/models" >/dev/null && break
        sleep 1.5
    done
    curl -sf "http://127.0.0.1:$MPORT/v1/models" >/dev/null || {
        echo "[job] model server never came up, see $LOGDIR/serve_$TAG.$LOGSFX.log"; tail -30 "$LOGDIR/serve_$TAG.$LOGSFX.log"; exit 3; }
    echo "[job] $TAG serving on :$MPORT"
    # WARM-UP, for the same reason CookSim warms every driver: a fresh server pays one-time costs (the
    # reply schema compiled to a grammar, kernels chosen, weights paged in from the shared filesystem) and the assistant's
    # latency is charged to the game clock, so the first episode must not carry them.
    "$PY" -m vhhome_bench.preflight "$TAG" --repeat 3 || { echo "[job] preflight failed"; exit 4; }
fi

# ---- 3a. calibration: time the SAME synthetic call on this card and write the row
if [ "$MODE" = "calib" ]; then
    OUT="$ENG/$ROOT_OUT/calib"; mkdir -p "$OUT"
    SAFE_GPU="$(echo "$VH_GPU_NAME" | tr ' /' '__')"
    "$PY" -m vhhome_bench.preflight "$TAG" --repeat "${CALIB_REPEAT:-6}" --json \
        > "$OUT/${TAG}__${SAFE_GPU}.json"
    rc=$?
    echo "[job] calib rc=$rc -> $OUT/${TAG}__${SAFE_GPU}.json"
    exit $rc
fi

# ---- 3. the lane
ARGS=(--root "$ROOT_OUT" --engines "$PORT_ENGINE")
[ -n "$SHARD" ] && ARGS+=(--shard "$SHARD")
[ -n "$CELLS" ] && ARGS+=(--cells "$CELLS")     # a fill job runs only the cells still missing
[ "$MODE" = "check" ] && ARGS+=(--limit 3)
"$PY" -m vhhome_bench.lane "$TAG" "${ARGS[@]}"
rc=$?
echo "[job] $(date -u +%FT%TZ) $MODE finished rc=$rc"
exit $rc
