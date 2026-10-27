# CookSim · Experiment 3 — question answering and proactive alerts (run 1)

**What is asked.** The simulated cook asks scripted questions mid-task (families Q1–Q9, the question library) and makes standing requests "tell me the moment the X is done" (Q10). Kept families (user ruling 2026-09-23; Q5 set/count, Q7 referent, Q8 self-recall, Q9 next-step dropped):
- **Q1 order** — "Did I put the rice on the heat before or after I chopped the steak?" → graded on the relation word.
- **Q2 count** — "How many times have I walked over to a cutting board so far?" → exact integer.
- **Q3 timing on heat** — "How long has the steak been cooking?" / "Is anything on the heat already done and just sitting there?" → station must match, ticks within 5.
- **Q4 did-it-happen** — "Have I been over to the trash at any point so far?" → yes/no against world truth; hedges are incorrect.
- **Q10 proactive** — binary: 1 if the assistant asserted readiness inside the window [done, done + smoke_grace = 40 ticks) (window closes when the food starts smoking; `qa_v3_lib.ready_horizon`, `COOKSIM_Q10_HORIZON=burn` restores the old 85-tick horizon); early / late / missed = 0; pre-empted and no-window requests excluded.

Answer score: correct 1, partially correct 0.5, incorrect / evasive / no answer 0; ungradable and asked-too-late excluded. Judge Gemini 3.7 Flash with per-family rubric (`cooksim-score (qa)`, attr-v6, per-item chop guard and per-kind Q5 rubric fixed 2026-09-23). **ALL** = pooled mean over all scored items; **macro** = mean of the five sub-scores.

| Seat | n items | ALL (pooled) | macro | Q1 order | Q2 count | Q3 timing | Q4 did-it | Q10 proactive |
|---|---|---|---|---|---|---|---|---|
| GPT-6 Astra | 188 | 0.78 | 0.71 | 0.67 | 0.23 | 0.91 | 0.88 | 0.87 |
| Gemini 3.8 Flash | 190 | 0.61 | 0.57 | 0.60 | 0.39 | 0.61 | 0.54 | 0.70 |
| Gemini 3.8 Live | 182 | 0.47 | 0.43 | 0.27 | 0.23 | 0.51 | 0.67 | 0.48 |
| Qwen3.8-27B | 185 | 0.42 | 0.42 | 0.36 | 0.23 | 0.51 | 0.71 | 0.30 |
| GPT-realtime 2.1 | 172 | 0.41 | 0.38 | 0.30 | 0.18 | 0.45 | 0.52 | 0.42 |
| Gemini 3.1 Pro | 191 | 0.40 | 0.38 | 0.27 | 0.26 | 0.43 | 0.52 | 0.41 |
| GPT-realtime 2.1 (native) | 169 | 0.40 | 0.42 | 0.42 | 0.30 | 0.54 | 0.65 | 0.21 |
| Gemini 3.8 Live (native) | 133 | 0.28 | 0.32 | 0.28 | 0.12 | 0.40 | 0.69 | 0.10 |
| Muse Glimmer | 189 | 0.28 | 0.22 | 0.07 | 0.05 | 0.33 | 0.30 | 0.34 |
| Qwen3-Omni 30B | 174 | 0.27 | 0.19 | 0.14 | 0.00 | 0.32 | 0.12 | 0.38 |
| Kimi-VL Thinking | 162 | 0.23 | 0.13 | 0.00 | 0.00 | 0.36 | 0.00 | 0.32 |
| Molmo2-8B | 152 | 0.16 | 0.15 | 0.00 | 0.00 | 0.12 | 0.45 | 0.16 |
| GLM-4.6V Flash | 147 | 0.14 | 0.09 | 0.00 | 0.00 | 0.09 | 0.15 | 0.23 |
| InternVideo3-8B | 159 | 0.08 | 0.06 | 0.00 | 0.00 | 0.08 | 0.10 | 0.11 |
| MiniCPM-o | 112 | 0.07 | 0.05 | 0.00 | 0.00 | 0.10 | 0.07 | 0.07 |
| StreamingVLM | 170 | 0.03 | 0.04 | 0.07 | 0.00 | 0.00 | 0.08 | 0.03 |
| Proact-VL | 160 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 |
| MMDuet2 | 110 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Sub-category n: Q1 220, Q2 361, Q3 901, Q4 393, Q10 1070. Pooled Q10 = 0.313.

Provenance: `the benchmark run/qa_regrade.json` (answers), `the benchmark run/proactive_v2.json` (alerts: the offline judge re-grade `cooksim-score (proactive)`, superseding `injections[].proactive`; re-scored under the 25-tick horizon).


---
_generated 2026-10-14 15:14 UTC from the run records_
