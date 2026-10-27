# VH-Home formal run 2 — final table (three independent runs, 75 episodes per seat per run)

| seat | AQ [95% CI] | AQ r1 / r2 / r3 | won | S (in-time success) | EFF | DET | DT | F1 |
|---|---|---|---|---|---|---|---|---|
| gpt-6-astra | 0.492 [0.47, 0.51] | 0.483 / 0.482 / 0.511 | 0.21 ± 0.02 | 0.08 ± 0.01 | 0.60 ± 0.01 | 0.50 ± 0.03 | 0.29 ± 0.02 | 0.71 ± 0.03 |
| gpt-realtime-2.1__native | 0.487 [0.45, 0.52] | 0.495 / 0.484 / 0.479 | 0.07 ± 0.03 | 0.01 ± 0.00 | 0.54 ± 0.11 | 0.39 ± 0.01 | 0.12 ± 0.02 | 0.67 ± 0.03 |
| gemini-3.8-flash | 0.483 [0.46, 0.50] | 0.497 / 0.488 / 0.463 | 0.33 ± 0.03 | 0.11 ± 0.03 | 0.55 ± 0.06 | 0.52 ± 0.02 | 0.34 ± 0.01 | 0.71 ± 0.02 |
| gemini-3.8-live__native | 0.447 [0.41, 0.48] | 0.478 / 0.411 / 0.451 | 0.05 ± 0.03 | 0.00 ± 0.00 | — | 0.32 ± 0.08 | 0.03 ± 0.01 | 0.62 ± 0.15 |
| gemini-3.8-live | 0.397 [0.38, 0.41] | 0.400 / 0.395 / 0.397 | 0.28 ± 0.02 | 0.10 ± 0.03 | 0.46 ± 0.05 | 0.38 ± 0.05 | 0.13 ± 0.03 | 0.63 ± 0.07 |
| gpt-realtime-2.1 | 0.373 [0.36, 0.39] | 0.378 / 0.362 / 0.378 | 0.36 ± 0.03 | 0.10 ± 0.03 | 0.47 ± 0.08 | 0.46 ± 0.02 | 0.27 ± 0.01 | 0.66 ± 0.04 |
| qwen3.8-27b | 0.366 [0.35, 0.38] | 0.360 / 0.354 / 0.384 | 0.25 ± 0.01 | 0.09 ± 0.01 | 0.52 ± 0.04 | 0.43 ± 0.03 | 0.21 ± 0.03 | 0.65 ± 0.05 |
| glm-4.6v-flash | 0.344 [0.33, 0.36] | 0.353 / 0.346 / 0.333 | 0.16 ± 0.03 | 0.04 ± 0.03 | 0.44 ± 0.18 | 0.38 ± 0.02 | 0.09 ± 0.01 | 0.66 ± 0.03 |
| museglimmer | 0.337 [0.32, 0.35] | 0.327 / 0.331 / 0.354 | 0.31 ± 0.03 | 0.09 ± 0.04 | 0.53 ± 0.03 | 0.40 ± 0.01 | 0.10 ± 0.01 | 0.69 ± 0.02 |
| gemini-3.1-pro-preview | 0.335 [0.32, 0.35] | 0.338 / 0.337 / 0.328 | 0.22 ± 0.07 | 0.06 ± 0.03 | 0.55 ± 0.06 | 0.57 ± 0.03 | 0.40 ± 0.02 | 0.73 ± 0.04 |
| qwen3-omni-30b-a3b | 0.287 [0.27, 0.30] | 0.281 / 0.271 / 0.310 | 0.19 ± 0.03 | 0.07 ± 0.00 | 0.51 ± 0.14 | 0.29 ± 0.05 | 0.02 ± 0.01 | 0.57 ± 0.09 |
| internvideo3-8b | 0.193 [0.18, 0.21] | 0.191 / 0.186 / 0.202 | 0.13 ± 0.01 | 0.07 ± 0.01 | 0.48 ± 0.05 | 0.00 ± 0.00 | 0.00 ± 0.00 | — |
| minicpmo | 0.174 [0.16, 0.18] | 0.167 / 0.166 / 0.187 | 0.14 ± 0.01 | 0.07 ± 0.01 | 0.45 ± 0.06 | 0.18 ± 0.04 | 0.06 ± 0.02 | 0.30 ± 0.06 |
| molmo2-8b | 0.140 [0.12, 0.16] | 0.140 / 0.123 / 0.158 | 0.11 ± 0.05 | 0.05 ± 0.01 | 0.52 ± 0.09 | 0.00 ± 0.00 | 0.00 ± 0.00 | — |
| proact-vl | 0.130 [0.12, 0.14] | 0.130 / 0.129 / 0.131 | 0.08 ± 0.03 | 0.04 ± 0.03 | 0.42 ± 0.12 | 0.00 ± 0.00 | 0.00 ± 0.00 | — |
| kimi-vl-thinking | 0.124 [0.11, 0.14] | 0.130 / 0.117 / 0.125 | 0.20 ± 0.06 | 0.11 ± 0.05 | 0.52 ± 0.04 | 0.51 ± 0.02 | 0.37 ± 0.01 | 0.65 ± 0.02 |
| streamingvlm | 0.118 [0.11, 0.12] | 0.119 / 0.118 / 0.118 | 0.12 ± 0.01 | 0.03 ± 0.01 | 0.44 ± 0.13 | 0.00 ± 0.00 | 0.00 ± 0.00 | — |
| mmduet2 | — | — / — / — | 0.01 ± 0.00 | 0.00 ± 0.01 | 0.49 ± — | 0.00 ± 0.00 | 0.00 ± 0.00 | — |
| silent | — | — / — / — | 0.01 ± 0.01 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00 ± 0.00 | — |

- S, EFF, DET: mean over the three runs, ± half the range between runs (vhhome_bench/scoring/metrics_table.py).
- AQ: SQ v4 per-episode interaction quality pooled over the three runs (n = 225 per seat), bootstrap 95% interval over episodes; per-run means in order r1 / r2 / r3.
- AQ is the mean over the judges that answered on each round: gemini-3.7-flash and gpt-5.4-mini on every round, claude-haiku-5-5 on 99% / 99% / 44% of the rounds of runs 1 / 2 / 3 (the Anthropic balance ran out on 2026-10-12; project decision: average the two judges where Claude is missing).
- won: share of episodes the person finished the chore correctly, mean ± half-range over runs; S additionally requires finishing inside the time cap C = 2(T_ref + alpha).
- silent and mmduet2 never speak, so they have no AQ; silent is the floor every other number is read against.
