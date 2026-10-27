#!/usr/bin/env bash
# Evaluate a model you serve yourself on the full VH-Home protocol: three independent runs over the 75
# benchmark cells, then the pooled table.
#
# The assistant is any OpenAI-compatible chat endpoint (vLLM, SGLang, a shim around your own loop):
#   --assistant openai:<served name> --base-url http://127.0.0.1:<port>/v1
# The endpoint's key, if it wants one, is IV_ASSIST_API_KEY (OPENAI_API_KEY is used when that is unset).
# GEMINI_API_KEY is needed in every case: the simulated person is a Gemini model.
#
# Run from the repository root. Usage:
#   MODEL=my-vlm BASE_URL=http://127.0.0.1:8000/v1 bash examples/evaluate_served_model.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL to the model name your server answers to}
BASE_URL=${BASE_URL:?set BASE_URL to your server, e.g. http://127.0.0.1:8000/v1}
PROFILE=${PROFILE:-default}            # request profile: default | gpt5 | one of the per-model profiles in protocol._PROFILES
OUT=${OUT:-runs/$MODEL}

# 1. Serve the model. For one of the open-weight models of the published roster the serving scripts do it
#    (each listens on its roster port; `vhhome-roster` prints them):
#       bash serving/serve.sh qwen3.8-27b          # then --assistant qwen3.8-27b, no --base-url needed
#    For your own model, start your server on BASE_URL before running this script.

# 2. One simulator instance. A locally served model runs ONE episode at a time on ONE port: its latency
#    is charged to the game clock, and two episodes sharing the server would each be charged the other's
#    compute. (API assistants may spread over several instances: --engines 8080,8081.)
bash simulator/launch_simulators.sh 1

# 3. Three independent runs of the 75 cells. Each run is resumable: a cell whose record exists is skipped,
#    so a stopped run continues where it was. Records land under runs/<model>/r<k>/<model>/.
#    An episode with no log growth for VH_EPISODE_STALL seconds (900) is killed and left missing.
for k in 1 2 3; do
  vhhome-benchmark --assistant "openai:$MODEL" --base-url "$BASE_URL" --profile "$PROFILE" \
      --engines 8080 --out "$OUT" --repeat "$k"
done

# 4. Score: the objective metrics (won, S, EFF, DET) per run, the interaction-quality rubric with the two
#    reported judges (needs GEMINI_API_KEY and OPENAI_API_KEY; per-judge results are cached, so a rerun
#    resumes), and the pooled table over the three runs, written as final_results.{json,md} beside r1.
vhhome-score --run "$OUT/r1" "$OUT/r2" "$OUT/r3" --judges gemini,gpt

# To charge latency as reference-GPU seconds instead of what this card measured, export
# ASSIST_LATENCY_DIV=<this card's generation time / an A100's> before step 3 (benchmark/gpu_latency_ratio.json
# holds the measured ratios for the roster models).
