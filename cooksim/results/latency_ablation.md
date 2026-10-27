# CookSim · Experiment 4 — the assistant's reply latency: measured vs a fixed 2-tick delay (ablation)

## Measured reply latency (measured arm, run 1)
Each assistant call is timed from the tick it was issued to the tick its words arrive; the game keeps running meanwhile. Charged ticks = ⌈seconds / 1.2⌉ (one tick = 1.2 s, the human app's tick), capped at 30 ticks per reply (API hangs at the HTTP timeout are not charged in full). Local models are charged in A100-equivalent seconds. The last two columns restrict to calls that produced a mistake warning (flag), which are longer than routine replies.

| Seat | calls / episode | median s | p90 s | median ticks | p90 ticks | share capped (30 ticks) | median ticks on a warning line | n warnings |
|---|---|---|---|---|---|---|---|---|
| Muse Glimmer | 116 | 1.5 | 2.1 | 2 | 2 | 0.00 | 2 | 372 |
| Qwen3.8-27B | 81 | 2.1 | 3.9 | 2 | 4 | 0.00 | 3 | 274 |
| Gemini 3.8 Flash | 55 | 3.4 | 6.9 | 3 | 6 | 0.01 | 4 | 469 |
| GPT-6 Astra | 39 | 6.8 | 8.8 | 6 | 8 | 0.00 | 7 | 422 |
| Qwen3-Omni 30B | 97 | 1.4 | 2.1 | 2 | 2 | 0.00 | 2 | 222 |
| Gemini 3.1 Pro | 27 | 9.5 | 15.3 | 8 | 13 | 0.02 | 10 | 695 |
| Gemini 3.8 Live | 97 | 2.2 | 2.9 | 2 | 3 | 0.00 | 3 | 383 |
| GPT-realtime 2.1 | 46 | 4.4 | 7.1 | 4 | 6 | 0.00 | 4 | 502 |
| GPT-realtime 2.1 (native) | 8 | 1.8 | 3.0 | 2 | 3 | 0.00 | 2 | 168 |
| MiniCPM-o | 71 | 1.3 | 1.9 | 2 | 2 | 0.00 | 2 | 456 |
| Molmo2-8B | 66 | 2.5 | 2.9 | 3 | 3 | 0.00 | 3 | 23 |
| GLM-4.6V Flash | 124 | 1.3 | 1.7 | 2 | 2 | 0.00 | 2 | 70 |
| Kimi-VL Thinking | 58 | 2.7 | 5.8 | 3 | 5 | 0.00 | 3 | 1460 |
| InternVideo3-8B | 60 | 3.0 | 3.9 | 3 | 4 | 0.00 | 3 | 233 |
| StreamingVLM | 93 | 1.3 | 1.7 | 2 | 2 | 0.00 | – | 0 |
| Gemini 3.8 Live (native) | 7 | 1.9 | 2.6 | 2 | 3 | 0.00 | 2 | 60 |
| Proact-VL | 89 | 1.0 | 1.1 | 1 | 1 | 0.00 | – | 0 |
| MMDuet2 | 67 | 0.6 | 0.7 | 1 | 1 | 0.00 | – | 0 |

**Setting.** Same episodes, two arms: **measured** (`COOKSIM_ASSIST_LATENCY=measured`, run 1: the kitchen keeps running while the model thinks; a warning lands ⌈latency / 1.2 s⌉ ticks after the decision, capped at 30; the per-seat table above gives the measured delays) vs **fixed 2 ticks** (`COOKSIM_ASSIST_LATENCY=2`: every reply is charged exactly 2 ticks = 2.4 s, whatever the model's real speed, so every seat is compared under one common delay; for the two native streaming arms the fixed delay is a floor — a line that physically arrived later than 2 ticks lands when it arrived, 13–15% of their lines). Only persona × case cells completed by both arms are paired: 2547 episodes. Values are *measured → fixed 2 ticks*.

| Seat | paired n | In-time success | Task success | Recall | Prevention rate | DT | F1 | FA / ep | assistant turns / ep |
|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | 148 | 0.28 → 0.48 | 0.57 → 0.71 | 0.61 → 0.78 | 0.30 → 0.49 | 0.32 → 0.50 | 0.65 → 0.79 | 0.15 → 0.24 | 56 → 119 |
| Gemini 3.1 Pro | 148 | 0.20 → 0.47 | 0.43 → 0.75 | 0.53 → 0.83 | 0.21 → 0.57 | 0.27 → 0.63 | 0.56 → 0.74 | 0.47 → 1.20 | 28 → 127 |
| Muse Glimmer | 149 | 0.44 → 0.44 | 0.58 → 0.56 | 0.63 → 0.61 | 0.45 → 0.44 | 0.47 → 0.46 | 0.65 → 0.63 | 0.41 → 0.32 | 117 → 125 |
| GPT-6 Astra | 150 | 0.28 → 0.42 | 0.49 → 0.82 | 0.63 → 0.81 | 0.39 → 0.50 | 0.38 → 0.48 | 0.68 → 0.83 | 0.13 → 0.13 | 40 → 139 |
| Qwen3.8-27B | 147 | 0.33 → 0.37 | 0.45 → 0.48 | 0.56 → 0.63 | 0.39 → 0.48 | 0.40 → 0.49 | 0.61 → 0.67 | 0.18 → 0.15 | 82 → 109 |
| GPT-realtime 2.1 | 148 | 0.17 → 0.34 | 0.31 → 0.53 | 0.56 → 0.70 | 0.35 → 0.47 | 0.34 → 0.50 | 0.59 → 0.69 | 0.47 → 0.79 | 46 → 116 |
| Qwen3-Omni 30B | 145 | 0.24 → 0.28 | 0.29 → 0.34 | 0.41 → 0.40 | 0.30 → 0.31 | 0.31 → 0.32 | 0.44 → 0.45 | 0.24 → 0.19 | 97 → 110 |
| Gemini 3.8 Live | 146 | 0.19 → 0.23 | 0.51 → 0.50 | 0.52 → 0.56 | 0.21 → 0.24 | 0.26 → 0.29 | 0.58 → 0.60 | 0.28 → 0.40 | 97 → 122 |
| GPT-realtime 2.1 (native) | 144 | 0.09 → 0.14 | 0.17 → 0.24 | 0.13 → 0.17 | 0.02 → 0.02 | 0.02 → 0.03 | 0.15 → 0.21 | 0.10 → 0.06 | 8 → 9 |
| InternVideo3-8B | 141 | 0.04 → 0.10 | 0.08 → 0.17 | 0.15 → 0.23 | 0.05 → 0.11 | 0.08 → 0.15 | 0.18 → 0.26 | 0.27 → 0.62 | 59 → 110 |
| MiniCPM-o | 132 | 0.08 → 0.10 | 0.08 → 0.10 | 0.40 → 0.38 | 0.19 → 0.20 | 0.25 → 0.26 | 0.34 → 0.36 | 0.96 → 0.98 | 72 → 76 |
| Kimi-VL Thinking | 147 | 0.05 → 0.10 | 0.11 → 0.15 | 0.48 → 0.56 | 0.10 → 0.13 | 0.25 → 0.30 | 0.32 → 0.35 | 3.22 → 3.84 | 58 → 103 |
| GLM-4.6V Flash | 130 | 0.07 → 0.09 | 0.09 → 0.13 | 0.13 → 0.15 | 0.09 → 0.12 | 0.10 → 0.13 | 0.16 → 0.18 | 0.07 → 0.07 | 104 → 102 |
| Molmo2-8B | 145 | 0.06 → 0.06 | 0.07 → 0.09 | 0.04 → 0.05 | 0.03 → 0.04 | 0.03 → 0.05 | 0.05 → 0.07 | 0.05 → 0.08 | 63 → 105 |
| Gemini 3.8 Live (native) | 131 | 0.04 → 0.05 | 0.11 → 0.11 | 0.07 → 0.07 | 0.01 → 0.02 | 0.01 → 0.02 | 0.10 → 0.08 | 0.03 → 0.04 | 6 → 5 |
| StreamingVLM | 136 | 0.04 → 0.05 | 0.04 → 0.06 | 0.02 → 0.01 | 0.01 → 0.01 | 0.01 → 0.01 | 0.02 → 0.02 | 0.00 → 0.00 | 91 → 97 |
| Proact-VL | 147 | 0.04 → 0.03 | 0.05 → 0.03 | 0.01 → 0.01 | 0.01 → 0.01 | 0.01 → 0.01 | 0.02 → 0.01 | 0.00 → 0.00 | 91 → 96 |
| MMDuet2 | 113 | 0.03 → 0.02 | 0.03 → 0.02 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 82 → 85 |

Pooled: in-time 0.153 → 0.216; task success 0.256 → 0.334; timed-out share 0.155 → 0.159; prevented/fired 0.185 → 0.245; turns 66.3 → 98.6.


Provenance: the benchmark run, run 1 (measured latency) vs the delay-ablation run (fixed 2-tick delay, `--latency 2`); scorer: `cooksim-score` (objective step).


---
_generated 2026-10-14 15:14 UTC from the run records_
