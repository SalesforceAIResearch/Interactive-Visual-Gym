#!/usr/bin/env bash
# What a Beaker batch job runs: one roster assistant, on the job's own GPU, from the engine checkout on the
# shared weka mount (no git clone -- the job sees the same code, envs, weights and outputs as a session).
#
#   bash cooksim_bench/cluster/beaker_job.sh check <tag> <root>                 # the 3-episode setup check (run_roster_check.py)
#   bash cooksim_bench/cluster/beaker_job.sh lane  <tag> <root> [repeats] [r4|r5]   # the 10-episode lane (run_lane.py)
#   bash cooksim_bench/cluster/beaker_job.sh bench <tag> <root> [repeat] [shard/nshards]
#        the FORMAL run: the whole suite x 3 personas. `repeat` (1,2,3...) writes to <tag>__r<k> so the
#        repeats are independent samples; `shard` splits one seat's queue over several jobs (e.g. 2/3).
#        Every mode skips episodes whose report already exists, so a preempted job resumes where it stopped.
#
# Submitted by -m cooksim_bench.cluster.beaker_submit; everything the job writes lands under <root> on weka.
set -uo pipefail
MODE=$1; TAG=$2; ROOT_OUT=$3; REPEATS=${4:-3}; EPSET=${5:-r4}
# bench mode reads the same two positional slots differently: repeat index and shard spec
REPEAT=${4:-1}; SHARDSPEC=${5:-1/1}; SHARD=${SHARDSPEC%%/*}; NSHARDS=${SHARDSPEC##*/}
# "auto/N": one Beaker experiment with N REPLICAS instead of N separate jobs -- the replica's own rank
# picks its slice, so the whole seat is a single row in the Beaker UI. BEAKER_REPLICA_RANK is 0-based.
if [ "$SHARD" = "auto" ]; then SHARD=$(( ${BEAKER_REPLICA_RANK:-0} + 1 )); fi
ENG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# manifest mode: ONE experiment covers every remaining run. This replica reads its own line of the
# manifest (-m cooksim_bench.cluster.make_replica_manifest) by BEAKER_REPLICA_RANK and becomes that item.
#   bash cooksim_bench/cluster/beaker_job.sh manifest <manifest path> <unused>
if [ "$MODE" = "manifest" ]; then
    MF="$ENG/$TAG"; RANK=${BEAKER_REPLICA_RANK:-0}
    LINE=$(python3 -c "
import json, sys
it = json.load(open('$MF'))['items']
i = it[$RANK % len(it)]
print(i['kind'], i['tag'], i['root'], i['repeat'], i['shard'], i['nshards'],
      ' '.join(f'{k}={v}' for k, v in (i.get('env') or {}).items()))")
    read -r MODE TAG ROOT_OUT REPEAT SHARD NSHARDS EXTRA_ENV <<< "$LINE"    # kind may be bench|cookabl|latcal
    for kv in ${EXTRA_ENV:-}; do export "${kv?}"; done
    echo "[job] manifest rank=$RANK -> mode=$MODE tag=$TAG root=$ROOT_OUT repeat=$REPEAT shard=$SHARD/$NSHARDS env='${EXTRA_ENV:-}'"
fi
cd "$ENG"
echo "[job] $(date -u +%FT%TZ) host=${BEAKER_NODE_HOSTNAME:-?} gpus=${BEAKER_ASSIGNED_GPU_COUNT:-?} mode=$MODE tag=$TAG root=$ROOT_OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source tmp/present/env.sh >/dev/null 2>&1          # $PY, COOKSIM_GRID, the Gemini key, gpu_env.sh
export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
export CUDA_VISIBLE_DEVICES=0                       # the job's one GPU
export LOGDIR="$ENG/$ROOT_OUT/serve_logs"; mkdir -p "$LOGDIR"

if [ "$MODE" = "check" ]; then
    # run_roster_check starts its own renderer and the model server (lock files are private to the container)
    exec "$PY" -m cooksim_bench.cluster.run_roster_check --tags "$TAG" --gpus 0 --root "$ROOT_OUT"
fi

# sqgrade: the rubric grader for ONE seat. Nothing is served and nothing is rendered, so this task asks
# for no GPU at all -- it is API calls against episodes already on weka.
if [ "$MODE" = "sqgrade" ]; then
    export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
    exec env ASSIST_ROOT="$ROOT_OUT" "$PY" -m cooksim_bench.scoring.sq_v4_score --judge gpt-main --double-every 5 \
        --first-run-only --model "$TAG" --workers "${SQ_WORKERS:-12}"
fi
if [ "$MODE" != "lane" ] && [ "$MODE" != "bench" ] && [ "$MODE" != "cookabl" ]; then echo "unknown mode $MODE"; exit 2; fi
# cookabl looks the seat up in the COOK roster (an API cook has no serve script at all)
LOOKUP="roster.by_tag('$TAG')"; [ "$MODE" = "cookabl" ] && LOOKUP="roster.cook_by_tag('$TAG')"
SERVE=$("$PY" -c "from cooksim_bench import roster; e=$LOOKUP; print(e.get('serve',''))")
PORT=$("$PY" -c "from cooksim_bench import roster; e=$LOOKUP; print(e.get('port',''))")
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
setsid "$PY" -m cooksim.server.app --port 17773 > "$LOGDIR/render.log" 2>&1 < /dev/null & PIDS+=($!)
export COOKSIM_PAGE="http://127.0.0.1:17773/static/headless_render.html"
if [ -n "$SERVE" ]; then
    PORT=$PORT setsid bash "serving/$SERVE" > "$LOGDIR/serve_$TAG.log" 2>&1 < /dev/null & PIDS+=($!)
    for i in $(seq 1 2400); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
        kill -0 "${PIDS[-1]}" 2>/dev/null || { echo "[job] server died, see $LOGDIR/serve_$TAG.log"; exit 3; }
        sleep 1.5
    done
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "[job] server never came up"; exit 3; }
    echo "[job] $TAG serving on :$PORT"
    "$PY" -m cooksim_bench.backends.warmup_server "$TAG"        # same warm-up as every interactive path
fi
for i in $(seq 1 120); do curl -sf "$COOKSIM_PAGE" >/dev/null && break; sleep 1; done
if [ "$MODE" = "cookabl" ]; then
    # COOK-MODEL ABLATION (2026-09-22): identical harness for all four cooks -- same suite, same
    # scheduler/pace/latency/grader, same 30 cells, same assistant. Only --cook-* changes.
    export COOKSIM_CASES=${COOKSIM_CASES:-bench/v5/cases_composite_v5.json}
    export COOKSIM_SCHED=${COOKSIM_SCHED:-v5} COOKSIM_PACE=${COOKSIM_PACE:-tight}
    export COOKSIM_ASSIST_LATENCY=${COOKSIM_ASSIST_LATENCY:-measured} COOKSIM_GRADER=${COOKSIM_GRADER:-cook}
    CELLS=$("$PY" -c "from cooksim_bench import roster; print(roster.COOK_ABLATION_CELLS)")
    COOKFLAGS=$("$PY" -c "from cooksim_bench import roster; print(' '.join(roster.cook_by_tag('$TAG')['flags']))")
    ASSISTFLAGS=$("$PY" -c "from cooksim_bench import roster; print(' '.join(roster.COOK_ABLATION_ASSISTANT))")
    echo "[job] cookabl cook=$TAG cells=$CELLS shard=$SHARD/$NSHARDS"
    echo "[job] cook flags:      $COOKFLAGS"
    echo "[job] assistant flags: $ASSISTFLAGS"
    echo "[job] suite=$COOKSIM_CASES sched=$COOKSIM_SCHED pace=$COOKSIM_PACE latency=$COOKSIM_ASSIST_LATENCY grader=$COOKSIM_GRADER"
    "$PY" -m cooksim_bench.run_assist_bench --tag "$TAG" --sup-obs frames --paradigm hardsoft \
        --persona-v2 baseline classic_novice classic_expert --workers 1 \
        --cells-file "$CELLS" \
        --shard "$((SHARD - 1))" --num-shards "$NSHARDS" \
        --out-root "$ROOT_OUT/frames" $ASSISTFLAGS $COOKFLAGS
    echo "[job] $(date -u +%FT%TZ) cookabl finished rc=$?"
elif [ "$MODE" = "bench" ]; then
    # the formal benchmark: every case of $COOKSIM_CASES x the three personas, one episode at a time
    # (a local model owns its GPU, and its reply latency is charged to the game clock)
    # the formal-run harness settings, set here so a job never inherits a stale shell
    export COOKSIM_CASES=${COOKSIM_CASES:-bench/v5/cases_composite_v5.json}
    export COOKSIM_SCHED=${COOKSIM_SCHED:-v5} COOKSIM_PACE=${COOKSIM_PACE:-tight}
    export COOKSIM_ASSIST_LATENCY=${COOKSIM_ASSIST_LATENCY:-measured} COOKSIM_GRADER=${COOKSIM_GRADER:-cook}
    echo "[job] bench suite=$COOKSIM_CASES sched=$COOKSIM_SCHED pace=$COOKSIM_PACE latency=$COOKSIM_ASSIST_LATENCY grader=$COOKSIM_GRADER"
    FLAGS=$("$PY" -c "from cooksim_bench import roster; print(' '.join(roster.by_tag('$TAG')['flags']))")
    OUTTAG="$TAG"; [ "$REPEAT" != "1" ] && OUTTAG="${TAG}__r${REPEAT}"
    echo "[job] bench repeat=$REPEAT shard=$SHARD/$NSHARDS out_tag=$OUTTAG"
    "$PY" -m cooksim_bench.run_assist_bench --tag "$OUTTAG" --sup-obs frames --paradigm hardsoft \
        --persona-v2 baseline classic_novice classic_expert --workers 1 \
        --shard "$((SHARD - 1))" --num-shards "$NSHARDS" \
        --out-root "$ROOT_OUT/frames" $FLAGS
    echo "[job] $(date -u +%FT%TZ) bench finished rc=$?"
else
    "$PY" -m cooksim_bench.cluster.run_lane "$TAG" --root "$ROOT_OUT" --repeats "$REPEATS" --episode-set "$EPSET" --no-warmup
    echo "[job] $(date -u +%FT%TZ) lane finished rc=$?"
fi
