#!/usr/bin/env bash
# Run one VH-Home episode twice, once with the silent control and once with a Gemini assistant,
# then score both records objectively (no judge calls).
#
# Run from the repository root after `pip install -e .` and `bash simulator/install_virtualhome.sh`.
# Needs GEMINI_API_KEY (the simulated person is always a Gemini model; the Gemini assistant uses it too),
# either exported or in ~/.config/vhhome/keys.env.
set -euo pipefail

CASE=${CASE:-cx_tidy_missing_rep}      # any case from `vhhome-case list cases`
PERSONA=${PERSONA:-baseline}           # any persona from `vhhome-case list personas`
ASSISTANT=${ASSISTANT:-gemini:gemini-3.8-flash}
OUT=${OUT:-runs/demo}

# 1. One headless simulator instance on port 8080 (software OpenGL under Xvfb; no GPU needed).
#    The launcher waits until the port answers HTTP 501, which is the healthy reply. Leave it running.
bash simulator/launch_simulators.sh 1

# 2. The control: nobody speaks. This is the floor every other number is read against.
#    Record: runs/demo/silent/silent__<case>__<persona>.json, with its .log and .mp4 beside it.
vhhome-episode --case "$CASE" --persona "$PERSONA" --assistant silent --port 8080 --out "$OUT"

# 3. The same cell with an assistant in the seat. Its reply latency is charged to the game clock.
#    Record: runs/demo/gemini-3.8-flash/gemini-3.8-flash__<case>__<persona>.json.
vhhome-episode --case "$CASE" --persona "$PERSONA" --assistant "$ASSISTANT" --port 8080 --out "$OUT"

# 4. Objective scoring of everything under runs/demo: outcome, in-time success S, efficiency EFF,
#    detection DET, one row per assistant. Writes runs/demo/metrics_table.json. The interaction-quality
#    rubric is the `rubric` step (vhhome-score --run runs/demo --only rubric; needs GEMINI_API_KEY and
#    OPENAI_API_KEY for the two judges).
vhhome-score --run "$OUT" --only objective
