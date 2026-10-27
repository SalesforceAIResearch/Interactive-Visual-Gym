# CookSim · Experiment 1 — task outcome, efficiency, detection

**Detection definitions (definition).** *Recall* = share of fired mistakes whose right name was called inside the live window **or within 30 ticks after the commit** (`COOKSIM_LATE_GRACE`, default 30); a late-but-correct call is a detection. *Precision* = detections / (detections + false alarms). *Prevention rate* (Experiment 4) = share of fired mistakes flagged before the commit **and** abandoned by the cook. *DT* keeps the timing tiers (1.0 prevented, 0.5 correct call before the commit, 0 otherwise). Timing therefore lives in DT and prevention, not in recall.

**Setting.** Suite `benchmark/cases_composite_v5.json` (50 cases × 3 cook personas = 150 episodes per assistant seat), engine `41ada1ea80` (heat 40 ticks done→smoke, 45 smoke→burnt, no smoke hold, explicit non-disclosure prompts), cook = the simulated human agent (self-planning, `COOKSIM_SCHED=v5`, pace `tight`, grader `cook`), assistant reply latency **charged** to the game clock in A100-equivalent seconds. Gates: an error actually fired (G2) and the episode reached a verdict (G3). Run 1 only (repeats 2/3 exist for CIs; not tabulated here).

**Definitions** (metrics.html §3). C = 2·(T_ref + allowance). In-time success S = 1[won]·1[T ≤ C]. Action efficiency TT = clip((C − T)/(C − T_ref)), 0 if T > C. Conversation efficiency CC = 1 − clip(Σ speech-ticks landing mid-action / T). Prevention timing DT = mean over fired errors of {1 prevented, 0.5 correctly flagged in window but committed, 0}. Detection F1 from precision TP/(TP+FP) and recall TP/fired (a flag after the commit is neither TP nor FP). FA = false alarms per episode.

| Seat | n | In-time success | Task success (any time) | TT (action eff.) | CC (conversation eff.) | DT (prevention timing) | F1 | Precision | Recall (pooled) | FA / ep |
|---|---|---|---|---|---|---|---|---|---|---|
| Muse Glimmer | 149 | 0.44 | 0.58 | 0.49 | 0.02 | 0.47 | 0.65 | 0.74 | 0.63 | 0.41 |
| Qwen3.8-27B | 148 | 0.33 | 0.45 | 0.48 | 0.20 | 0.40 | 0.61 | 0.76 | 0.56 | 0.18 |
| Gemini 3.8 Flash | 149 | 0.29 | 0.58 | 0.41 | 0.75 | 0.32 | 0.65 | 0.79 | 0.60 | 0.15 |
| GPT-6 Astra | 150 | 0.28 | 0.49 | 0.41 | 0.83 | 0.38 | 0.68 | 0.82 | 0.63 | 0.13 |
| Qwen3-Omni 30B | 149 | 0.24 | 0.29 | 0.61 | 0.00 | 0.31 | 0.44 | 0.56 | 0.40 | 0.25 |
| Gemini 3.1 Pro | 150 | 0.21 | 0.43 | 0.41 | 0.72 | 0.28 | 0.56 | 0.67 | 0.53 | 0.48 |
| Gemini 3.8 Live | 147 | 0.19 | 0.52 | 0.40 | 0.81 | 0.26 | 0.58 | 0.72 | 0.52 | 0.28 |
| GPT-realtime 2.1 | 149 | 0.17 | 0.32 | 0.49 | 0.03 | 0.34 | 0.59 | 0.69 | 0.56 | 0.47 |
| GPT-realtime 2.1 (native) | 148 | 0.10 | 0.18 | 0.57 | 0.70 | 0.02 | 0.16 | 0.22 | 0.13 | 0.09 |
| MiniCPM-o | 140 | 0.07 | 0.08 | 0.79 | 0.00 | 0.25 | 0.34 | 0.35 | 0.39 | 0.97 |
| Molmo2-8B | 150 | 0.07 | 0.07 | 0.61 | 0.02 | 0.03 | 0.05 | 0.08 | 0.04 | 0.05 |
| GLM-4.6V Flash | 138 | 0.07 | 0.09 | 0.58 | 0.65 | 0.09 | 0.15 | 0.21 | 0.12 | 0.07 |
| Kimi-VL Thinking | 149 | 0.06 | 0.11 | 0.59 | 0.00 | 0.25 | 0.31 | 0.24 | 0.48 | 3.23 |
| InternVideo3-8B | 146 | 0.05 | 0.09 | 0.56 | 0.45 | 0.08 | 0.18 | 0.27 | 0.15 | 0.31 |
| StreamingVLM | 146 | 0.05 | 0.05 | 0.64 | 0.49 | 0.01 | 0.02 | 0.03 | 0.01 | 0.00 |
| Gemini 3.8 Live (native) | 137 | 0.04 | 0.13 | 0.59 | 0.94 | 0.01 | 0.10 | 0.16 | 0.07 | 0.03 |
| Proact-VL | 148 | 0.04 | 0.05 | 0.67 | 0.48 | 0.01 | 0.02 | 0.03 | 0.01 | 0.00 |
| MMDuet2 | 117 | 0.03 | 0.03 | 0.70 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Pooled over 2610 episodes: in-time 0.154, task success 0.256, DT 0.199, F1 0.344.

## Three independent runs
Every seat was run three times end to end (run 1 = `<tag>`, runs 2 and 3 = `<tag>__r2`, `<tag>__r3`; same suite, same engine, fresh sampling), so the spread across runs is the run-to-run noise of a 150-episode seat. Values are per run, then mean ± SD over the runs that exist (n per run shows how complete each repeat is at dump time).

| Seat | run | n | In-time S | Task S | DT | F1 | TT | CC |
|---|---|---|---|---|---|---|---|---|
| Muse Glimmer | r1 | 149 | 0.44 | 0.58 | 0.47 | 0.65 | 0.49 | 0.02 |
|  | r2 | 149 | 0.40 | 0.56 | 0.45 | 0.60 | 0.48 | 0.04 |
|  | r3 | 149 | 0.46 | 0.56 | 0.48 | 0.64 | 0.50 | 0.02 |
|  | **mean ± SD** | 447 | **0.43** ± 0.03 | **0.57** ± 0.01 | **0.47** ± 0.01 | **0.63** ± 0.02 | **0.49** ± 0.01 | **0.03** ± 0.01 |
| Qwen3.8-27B | r1 | 148 | 0.33 | 0.45 | 0.40 | 0.61 | 0.48 | 0.20 |
|  | r2 | 150 | 0.24 | 0.39 | 0.41 | 0.64 | 0.49 | 0.21 |
|  | r3 | 148 | 0.25 | 0.41 | 0.42 | 0.64 | 0.47 | 0.19 |
|  | **mean ± SD** | 446 | **0.27** ± 0.04 | **0.41** ± 0.02 | **0.41** ± 0.01 | **0.63** ± 0.01 | **0.48** ± 0.01 | **0.20** ± 0.01 |
| Gemini 3.8 Flash | r1 | 149 | 0.29 | 0.58 | 0.32 | 0.65 | 0.41 | 0.75 |
|  | r2 | 147 | 0.33 | 0.62 | 0.34 | 0.65 | 0.38 | 0.76 |
|  | r3 | 147 | 0.29 | 0.55 | 0.34 | 0.68 | 0.41 | 0.76 |
|  | **mean ± SD** | 443 | **0.30** ± 0.02 | **0.58** ± 0.03 | **0.33** ± 0.01 | **0.66** ± 0.01 | **0.40** ± 0.02 | **0.76** ± 0.00 |
| GPT-6 Astra | r1 | 150 | 0.28 | 0.49 | 0.38 | 0.68 | 0.41 | 0.83 |
|  | r2 | 149 | 0.26 | 0.49 | 0.33 | 0.66 | 0.39 | 0.83 |
|  | r3 | 149 | 0.32 | 0.56 | 0.39 | 0.71 | 0.37 | 0.84 |
|  | **mean ± SD** | 448 | **0.29** ± 0.03 | **0.51** ± 0.04 | **0.37** ± 0.03 | **0.68** ± 0.02 | **0.39** ± 0.02 | **0.83** ± 0.00 |
| Qwen3-Omni 30B | r1 | 149 | 0.24 | 0.29 | 0.31 | 0.44 | 0.61 | 0.00 |
|  | r2 | 149 | 0.25 | 0.30 | 0.37 | 0.48 | 0.59 | 0.00 |
|  | r3 | 146 | 0.19 | 0.25 | 0.34 | 0.44 | 0.58 | 0.01 |
|  | **mean ± SD** | 444 | **0.23** ± 0.03 | **0.28** ± 0.02 | **0.34** ± 0.03 | **0.45** ± 0.02 | **0.59** ± 0.01 | **0.00** ± 0.00 |
| Gemini 3.1 Pro | r1 | 150 | 0.21 | 0.43 | 0.28 | 0.56 | 0.41 | 0.72 |
|  | r2 | 149 | 0.22 | 0.40 | 0.23 | 0.53 | 0.42 | 0.75 |
|  | r3 | 149 | 0.18 | 0.34 | 0.21 | 0.51 | 0.43 | 0.73 |
|  | **mean ± SD** | 448 | **0.21** ± 0.02 | **0.39** ± 0.04 | **0.24** ± 0.03 | **0.53** ± 0.02 | **0.42** ± 0.01 | **0.73** ± 0.01 |
| Gemini 3.8 Live | r1 | 147 | 0.19 | 0.52 | 0.26 | 0.58 | 0.40 | 0.81 |
|  | r2 | 149 | 0.22 | 0.52 | 0.30 | 0.58 | 0.41 | 0.79 |
|  | r3 | 147 | 0.27 | 0.53 | 0.29 | 0.59 | 0.42 | 0.79 |
|  | **mean ± SD** | 443 | **0.23** ± 0.03 | **0.52** ± 0.01 | **0.28** ± 0.02 | **0.58** ± 0.01 | **0.41** ± 0.01 | **0.80** ± 0.01 |
| GPT-realtime 2.1 | r1 | 149 | 0.17 | 0.32 | 0.34 | 0.59 | 0.49 | 0.03 |
|  | r2 | 148 | 0.23 | 0.38 | 0.38 | 0.64 | 0.46 | 0.01 |
|  | r3 | 150 | 0.26 | 0.40 | 0.39 | 0.64 | 0.49 | 0.02 |
|  | **mean ± SD** | 447 | **0.22** ± 0.04 | **0.36** ± 0.04 | **0.37** ± 0.02 | **0.62** ± 0.02 | **0.48** ± 0.01 | **0.02** ± 0.01 |
| GPT-realtime 2.1 (native) | r1 | 148 | 0.10 | 0.18 | 0.02 | 0.16 | 0.57 | 0.70 |
|  | r2 | 146 | 0.07 | 0.18 | 0.01 | 0.19 | 0.57 | 0.74 |
|  | r3 | 148 | 0.07 | 0.20 | 0.02 | 0.23 | 0.56 | 0.69 |
|  | **mean ± SD** | 442 | **0.08** ± 0.02 | **0.18** ± 0.01 | **0.02** ± 0.01 | **0.19** ± 0.03 | **0.57** ± 0.00 | **0.71** ± 0.02 |
| MiniCPM-o | r1 | 140 | 0.07 | 0.08 | 0.25 | 0.34 | 0.79 | 0.00 |
|  | r2 | 140 | 0.05 | 0.06 | 0.22 | 0.31 | 0.75 | 0.01 |
|  | r3 | 142 | 0.06 | 0.06 | 0.24 | 0.32 | 0.77 | 0.00 |
|  | **mean ± SD** | 422 | **0.06** ± 0.01 | **0.07** ± 0.01 | **0.24** ± 0.01 | **0.33** ± 0.01 | **0.77** ± 0.02 | **0.00** ± 0.00 |
| Molmo2-8B | r1 | 150 | 0.07 | 0.07 | 0.03 | 0.05 | 0.61 | 0.02 |
|  | r2 | 145 | 0.04 | 0.08 | 0.05 | 0.06 | 0.63 | 0.03 |
|  | r3 | 145 | 0.08 | 0.11 | 0.04 | 0.05 | 0.57 | 0.02 |
|  | **mean ± SD** | 440 | **0.06** ± 0.01 | **0.09** ± 0.02 | **0.04** ± 0.01 | **0.05** ± 0.00 | **0.61** ± 0.02 | **0.02** ± 0.00 |
| GLM-4.6V Flash | r1 | 138 | 0.07 | 0.09 | 0.09 | 0.15 | 0.58 | 0.65 |
|  | r2 | 138 | 0.07 | 0.10 | 0.10 | 0.14 | 0.56 | 0.64 |
|  | r3 | 142 | 0.06 | 0.10 | 0.12 | 0.17 | 0.56 | 0.64 |
|  | **mean ± SD** | 418 | **0.06** ± 0.00 | **0.10** ± 0.01 | **0.10** ± 0.01 | **0.15** ± 0.01 | **0.57** ± 0.01 | **0.64** ± 0.01 |
| Kimi-VL Thinking | r1 | 149 | 0.06 | 0.11 | 0.25 | 0.31 | 0.59 | 0.00 |
|  | r2 | 148 | 0.06 | 0.14 | 0.22 | 0.33 | 0.58 | 0.00 |
|  | r3 | 147 | 0.07 | 0.17 | 0.23 | 0.33 | 0.58 | 0.01 |
|  | **mean ± SD** | 444 | **0.06** ± 0.00 | **0.14** ± 0.02 | **0.24** ± 0.01 | **0.32** ± 0.01 | **0.58** ± 0.01 | **0.00** ± 0.00 |
| InternVideo3-8B | r1 | 146 | 0.05 | 0.09 | 0.08 | 0.18 | 0.56 | 0.45 |
|  | r2 | 145 | 0.04 | 0.10 | 0.10 | 0.19 | 0.53 | 0.47 |
|  | r3 | 148 | 0.06 | 0.11 | 0.06 | 0.15 | 0.56 | 0.45 |
|  | **mean ± SD** | 439 | **0.05** ± 0.01 | **0.10** ± 0.01 | **0.08** ± 0.02 | **0.17** ± 0.01 | **0.55** ± 0.02 | **0.45** ± 0.01 |
| StreamingVLM | r1 | 146 | 0.05 | 0.05 | 0.01 | 0.02 | 0.64 | 0.49 |
|  | r2 | 142 | 0.02 | 0.04 | 0.00 | 0.01 | 0.64 | 0.51 |
|  | r3 | 142 | 0.04 | 0.04 | 0.00 | 0.01 | 0.63 | 0.49 |
|  | **mean ± SD** | 430 | **0.03** ± 0.01 | **0.04** ± 0.01 | **0.01** ± 0.00 | **0.02** ± 0.00 | **0.64** ± 0.00 | **0.50** ± 0.01 |
| Gemini 3.8 Live (native) | r1 | 137 | 0.04 | 0.13 | 0.01 | 0.10 | 0.59 | 0.94 |
|  | r2 | 141 | 0.04 | 0.09 | 0.02 | 0.10 | 0.61 | 0.93 |
|  | r3 | 140 | 0.04 | 0.08 | 0.01 | 0.10 | 0.62 | 0.94 |
|  | **mean ± SD** | 418 | **0.04** ± 0.00 | **0.10** ± 0.02 | **0.01** ± 0.00 | **0.10** ± 0.00 | **0.61** ± 0.01 | **0.94** ± 0.00 |
| Proact-VL | r1 | 148 | 0.04 | 0.05 | 0.01 | 0.02 | 0.67 | 0.48 |
|  | r2 | 148 | 0.03 | 0.03 | 0.01 | 0.02 | 0.66 | 0.49 |
|  | r3 | 147 | 0.03 | 0.03 | 0.00 | 0.01 | 0.66 | 0.49 |
|  | **mean ± SD** | 443 | **0.03** ± 0.01 | **0.04** ± 0.01 | **0.01** ± 0.00 | **0.01** ± 0.01 | **0.66** ± 0.01 | **0.49** ± 0.01 |
| MMDuet2 | r1 | 117 | 0.03 | 0.03 | 0.00 | 0.00 | 0.70 | 1.00 |
|  | r2 | 114 | 0.01 | 0.01 | 0.00 | 0.00 | 0.70 | 1.00 |
|  | r3 | 115 | 0.01 | 0.01 | 0.00 | 0.00 | 0.70 | 1.00 |
|  | **mean ± SD** | 346 | **0.01** ± 0.01 | **0.01** ± 0.01 | **0.00** ± 0.00 | **0.00** ± 0.00 | **0.70** ± 0.00 | **1.00** ± 0.00 |

Provenance: `<run>/frames/<persona>/<tag>[__r2|__r3]/<case>.json`; scorer: `cooksim-score` (objective step); page `cooksim-results-v6/formal_results.html` (unpublished draft).


---
_generated 2026-10-14 15:14 UTC from the run records_
