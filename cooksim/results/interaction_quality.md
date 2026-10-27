# CookSim · Experiment 2 — interaction quality (SQ v4 rubric, five categories, three independent runs)

**What is graded.** Every assistant line in an episode is one grading *round*: the line itself, the cook's reply until the next line, the frame the assistant saw when it decided, and the exact kitchen state at that tick (rebuilt from the action log). A judge (GPT for Gemini seats, Gemini for GPT seats — no seat is graded by its own family; every fifth API episode by both, averaged) answers the rubric's items for that line against the engine state. Run 1 was judged on the full 16-item rubric and re-aggregated here on the five reported categories; runs 2 and 3 were judged on the five categories directly (`cooksim-score (rubric) --categories five`).

**The five categories and their items** :
- **Factual grounding** (`truthful`) — T1 every factual claim about the current kitchen state or the cook's completed actions is true; T2 the line refers to no station, item, rule or recipe requirement that does not exist.
- **Situational relevance** (`sensible`) — S1 the line is about the current situation (the cook's current action, a live hazard, a pending question); S2 if a cook question, request or objection is pending, the line addresses it.
- **Actionable guidance** (`helpful`; applies to instructions, mistake claims and answers) — H1 the first advised action can be started now; H2 the advised action is useful for the current plan; H3 a mistake claim comes with what to do about it.
- **User intent uptake** (`listens`; applies only to replies to pushback and to rounds with a live announced intent or a stop request — 8–30% of rounds) — L1 a reply to pushback re-checks rather than repeats; L2 a live valid announced intent or a "stop" is honoured.
- **Guidance conciseness** (`economy`) — E1 the line is concise for its function and for the cook's stated interruption preference.

**Scoring (round-level view, factual-grounding gate).** For one round and one category, the line is *clean* (1) if none of that category's applicable items failed, 0 if any failed; a category with no applicable item is absent for that round. The two Factual-grounding items are a gate: a line that fails T1 (a claim about the kitchen state or the cook's past actions is false) or T2 (it refers to a station, item, rule or recipe requirement that does not exist) scores **0** as a round, whatever else it got right — advice built on a false picture of the kitchen is not credited for being relevant, actionable or concise. Otherwise the round's score is the mean of its applicable clean flags (Situational relevance, Actionable guidance, User intent uptake, Guidance conciseness, with Factual grounding = 1 by construction). The overall score **AQ** is the mean of round scores over all the seat's lines (every line weighted equally; the episode-mean variant is within ±0.02 and is kept in the JSON). Category columns are the share of lines clean in that category, independent of the gate. What changed from the earlier table: (i) category values used to be *item* pass rates (a line with one false claim and one true one scored 0.5 on Factual grounding), which overstated every category by 0.05–0.15 and hid that a third to two thirds of the top seats' lines fail at least one check; (ii) the earlier must set also zeroed on H1 (action not startable now) and L2 (announced intent not honoured) — those now cost their own category only. Judge check: on double-graded rounds the two judges agree on the ranking of seats; the run-to-run SD of AQ is ≤ 0.04 for every seat.

Column keys: Fac. ground. = Factual grounding, Sit. relev. = Situational relevance, Act. guid. = Actionable guidance, Intent upt. = User intent uptake (its applicability, share of rounds, in the last column), Concise. = Guidance conciseness. One row per independent run (n = rounds graded); the bold row is the mean ± SD over the three runs. Aggregation: `cooksim-score` (table step) → `<run>/sq_v4/three_runs_five.json`.

| Seat | run | rounds | AQ | Fac. ground. | Sit. relev. | Act. guid. | Intent upt. | Concise. | Intent upt. applies to |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6 Astra | r1 | 1335 | 0.65 | 0.74 | 0.88 | 0.69 | 0.95 | 0.88 | 11% of rounds |
|  | r2 | 1299 | 0.63 | 0.72 | 0.86 | 0.67 | 0.90 | 0.87 | 13% of rounds |
|  | r3 | 1463 | 0.68 | 0.76 | 0.89 | 0.73 | 0.93 | 0.90 | 11% of rounds |
|  | **mean ± SD** | 4097 | **0.66** ± 0.02 | **0.74** ± 0.02 | **0.88** ± 0.01 | **0.70** ± 0.02 | **0.92** ± 0.02 | **0.89** ± 0.01 |  |
| Gemini 3.8 Flash | r1 | 3724 | 0.65 | 0.77 | 0.78 | 0.58 | 0.67 | 0.88 | 14% of rounds |
|  | r2 | 3738 | 0.64 | 0.77 | 0.77 | 0.58 | 0.71 | 0.88 | 15% of rounds |
|  | r3 | 3738 | 0.66 | 0.78 | 0.79 | 0.60 | 0.63 | 0.89 | 14% of rounds |
|  | **mean ± SD** | 11200 | **0.65** ± 0.01 | **0.78** ± 0.01 | **0.78** ± 0.01 | **0.58** ± 0.01 | **0.67** ± 0.04 | **0.88** ± 0.00 |  |
| Gemini 3.8 Live | r1 | 3647 | 0.62 | 0.79 | 0.75 | 0.48 | 0.60 | 0.85 | 32% of rounds |
|  | r2 | 3529 | 0.61 | 0.77 | 0.74 | 0.46 | 0.61 | 0.86 | 26% of rounds |
|  | r3 | 3523 | 0.60 | 0.77 | 0.73 | 0.45 | 0.62 | 0.85 | 26% of rounds |
|  | **mean ± SD** | 10699 | **0.61** ± 0.01 | **0.78** ± 0.01 | **0.74** ± 0.01 | **0.46** ± 0.01 | **0.61** ± 0.01 | **0.85** ± 0.00 |  |
| GLM-4.6V Flash | r1 | 10610 | 0.59 | 0.87 | 0.58 | 0.35 | 0.44 | 0.78 | 17% of rounds |
|  | r2 | 10799 | 0.58 | 0.86 | 0.58 | 0.33 | 0.47 | 0.77 | 16% of rounds |
|  | r3 | 11069 | 0.59 | 0.86 | 0.59 | 0.35 | 0.45 | 0.78 | 16% of rounds |
|  | **mean ± SD** | 32478 | **0.59** ± 0.00 | **0.86** ± 0.00 | **0.58** ± 0.01 | **0.34** ± 0.01 | **0.46** ± 0.01 | **0.77** ± 0.01 |  |
| Gemini 3.8 Live (native) | r1 | 860 | 0.56 | 0.71 | 0.76 | 0.48 | 0.75 | 0.80 | 13% of rounds |
|  | r2 | 788 | 0.54 | 0.68 | 0.75 | 0.46 | 0.74 | 0.80 | 14% of rounds |
|  | r3 | 826 | 0.57 | 0.71 | 0.77 | 0.48 | 0.77 | 0.81 | 16% of rounds |
|  | **mean ± SD** | 2474 | **0.56** ± 0.01 | **0.70** ± 0.01 | **0.76** ± 0.01 | **0.47** ± 0.01 | **0.75** ± 0.01 | **0.81** ± 0.01 |  |
| GPT-realtime 2.1 (native) | r1 | 1152 | 0.49 | 0.66 | 0.81 | 0.56 | 0.78 | 0.46 | 15% of rounds |
|  | r2 | 1154 | 0.47 | 0.64 | 0.80 | 0.56 | 0.79 | 0.47 | 17% of rounds |
|  | r3 | 1312 | 0.48 | 0.65 | 0.81 | 0.56 | 0.84 | 0.47 | 16% of rounds |
|  | **mean ± SD** | 3618 | **0.48** ± 0.01 | **0.65** ± 0.01 | **0.81** ± 0.00 | **0.56** ± 0.00 | **0.80** ± 0.02 | **0.46** ± 0.01 |  |
| GPT-realtime 2.1 | r1 | 6238 | 0.44 | 0.65 | 0.70 | 0.46 | 0.55 | 0.44 | 38% of rounds |
|  | r2 | 6427 | 0.44 | 0.66 | 0.71 | 0.47 | 0.54 | 0.44 | 39% of rounds |
|  | r3 | 6541 | 0.45 | 0.66 | 0.72 | 0.47 | 0.54 | 0.44 | 36% of rounds |
|  | **mean ± SD** | 19206 | **0.44** ± 0.00 | **0.66** ± 0.00 | **0.71** ± 0.01 | **0.47** ± 0.01 | **0.54** ± 0.00 | **0.44** ± 0.00 |  |
| Qwen3.8-27B | r1 | 5954 | 0.40 | 0.51 | 0.81 | 0.51 | 0.72 | 0.66 | 21% of rounds |
|  | r2 | 6033 | 0.39 | 0.49 | 0.81 | 0.50 | 0.70 | 0.66 | 21% of rounds |
|  | r3 | 6568 | 0.40 | 0.51 | 0.82 | 0.51 | 0.69 | 0.66 | 21% of rounds |
|  | **mean ± SD** | 18555 | **0.40** ± 0.01 | **0.50** ± 0.01 | **0.81** ± 0.00 | **0.51** ± 0.00 | **0.70** ± 0.01 | **0.66** ± 0.00 |  |
| Gemini 3.1 Pro | r1 | 2020 | 0.32 | 0.38 | 0.68 | 0.38 | 0.78 | 0.68 | 40% of rounds |
|  | r2 | 1814 | 0.34 | 0.41 | 0.69 | 0.39 | 0.79 | 0.69 | 35% of rounds |
|  | r3 | 1733 | 0.32 | 0.39 | 0.67 | 0.38 | 0.82 | 0.68 | 37% of rounds |
|  | **mean ± SD** | 5567 | **0.32** ± 0.01 | **0.40** ± 0.01 | **0.68** ± 0.01 | **0.38** ± 0.01 | **0.80** ± 0.02 | **0.69** ± 0.00 |  |
| StreamingVLM | r1 | 8884 | 0.27 | 0.74 | 0.07 | 0.18 | 0.28 | 0.15 | 42% of rounds |
|  | r2 | 8377 | 0.28 | 0.74 | 0.07 | 0.16 | 0.30 | 0.16 | 38% of rounds |
|  | r3 | 8613 | 0.27 | 0.72 | 0.07 | 0.16 | 0.29 | 0.16 | 42% of rounds |
|  | **mean ± SD** | 25874 | **0.27** ± 0.00 | **0.73** ± 0.01 | **0.07** ± 0.00 | **0.17** ± 0.01 | **0.29** ± 0.01 | **0.16** ± 0.01 |  |
| InternVideo3-8B | r1 | 5933 | 0.26 | 0.45 | 0.34 | 0.16 | 0.20 | 0.53 | 42% of rounds |
|  | r2 | 5816 | 0.26 | 0.45 | 0.35 | 0.17 | 0.17 | 0.54 | 41% of rounds |
|  | r3 | 5668 | 0.25 | 0.44 | 0.33 | 0.16 | 0.17 | 0.51 | 45% of rounds |
|  | **mean ± SD** | 17417 | **0.26** ± 0.00 | **0.45** ± 0.01 | **0.34** ± 0.01 | **0.16** ± 0.00 | **0.18** ± 0.01 | **0.53** ± 0.01 |  |
| Qwen3-Omni 30B | r1 | 12227 | 0.26 | 0.41 | 0.53 | 0.30 | 0.24 | 0.49 | 49% of rounds |
|  | r2 | 12478 | 0.23 | 0.38 | 0.50 | 0.27 | 0.23 | 0.46 | 55% of rounds |
|  | r3 | 12016 | 0.24 | 0.39 | 0.50 | 0.27 | 0.24 | 0.46 | 54% of rounds |
|  | **mean ± SD** | 36721 | **0.24** ± 0.01 | **0.39** ± 0.01 | **0.51** ± 0.01 | **0.28** ± 0.01 | **0.24** ± 0.01 | **0.47** ± 0.01 |  |
| Muse Glimmer | r1 | 13190 | 0.18 | 0.28 | 0.68 | 0.33 | 0.38 | 0.28 | 42% of rounds |
|  | r2 | 13311 | 0.18 | 0.28 | 0.68 | 0.32 | 0.35 | 0.28 | 38% of rounds |
|  | r3 | 13032 | 0.18 | 0.29 | 0.67 | 0.33 | 0.36 | 0.27 | 42% of rounds |
|  | **mean ± SD** | 39533 | **0.18** ± 0.00 | **0.28** ± 0.00 | **0.67** ± 0.00 | **0.33** ± 0.00 | **0.36** ± 0.01 | **0.28** ± 0.01 |  |
| Proact-VL | r1 | 9990 | 0.16 | 0.46 | 0.10 | 0.05 | 0.10 | 0.23 | 55% of rounds |
|  | r2 | 9878 | 0.16 | 0.44 | 0.10 | 0.06 | 0.10 | 0.23 | 53% of rounds |
|  | r3 | 10013 | 0.17 | 0.46 | 0.10 | 0.06 | 0.13 | 0.25 | 51% of rounds |
|  | **mean ± SD** | 29881 | **0.16** ± 0.00 | **0.45** ± 0.01 | **0.10** ± 0.00 | **0.06** ± 0.00 | **0.11** ± 0.01 | **0.24** ± 0.01 |  |
| Molmo2-8B | r1 | 7882 | 0.16 | 0.42 | 0.19 | 0.11 | 0.17 | 0.15 | 47% of rounds |
|  | r2 | 7207 | 0.14 | 0.37 | 0.18 | 0.09 | 0.14 | 0.13 | 50% of rounds |
|  | r3 | 7472 | 0.15 | 0.41 | 0.19 | 0.10 | 0.16 | 0.14 | 47% of rounds |
|  | **mean ± SD** | 22561 | **0.15** ± 0.01 | **0.40** ± 0.02 | **0.19** ± 0.00 | **0.10** ± 0.01 | **0.16** ± 0.01 | **0.14** ± 0.01 |  |
| MiniCPM-o | r1 | 8944 | 0.13 | 0.28 | 0.30 | 0.10 | 0.21 | 0.38 | 61% of rounds |
|  | r2 | 9007 | 0.13 | 0.28 | 0.28 | 0.09 | 0.21 | 0.36 | 58% of rounds |
|  | r3 | 9262 | 0.12 | 0.27 | 0.30 | 0.09 | 0.21 | 0.36 | 63% of rounds |
|  | **mean ± SD** | 27213 | **0.13** ± 0.00 | **0.28** ± 0.00 | **0.30** ± 0.01 | **0.09** ± 0.00 | **0.21** ± 0.00 | **0.37** ± 0.01 |  |
| Kimi-VL Thinking | r1 | 6817 | 0.04 | 0.08 | 0.27 | 0.05 | 0.14 | 0.12 | 69% of rounds |
|  | r2 | 7129 | 0.04 | 0.08 | 0.27 | 0.06 | 0.15 | 0.12 | 64% of rounds |
|  | r3 | 7053 | 0.04 | 0.08 | 0.26 | 0.05 | 0.14 | 0.12 | 65% of rounds |
|  | **mean ± SD** | 20999 | **0.04** ± 0.00 | **0.08** ± 0.00 | **0.27** ± 0.00 | **0.05** ± 0.00 | **0.14** ± 0.00 | **0.12** ± 0.00 |  |

Reading. Situational relevance and Guidance conciseness are high for most API seats (they speak about the present, briefly); Factual grounding and Actionable guidance are where they separate — one line in four from the best seat (GPT-6 Astra) contains a claim the engine state contradicts, two in three from Gemini 3.1 Pro. The human study (Experiment 6) found the two categories people reward are Factual grounding and Guidance conciseness.

Provenance: `the benchmark run/sq_v4/frames/<persona>/<tag>[__r2|__r3]/b3_*.json` (grader `cooksim-score (rubric) --judge gpt-main`; run 1 `--first-run-only`, runs 2/3 `--repeats-only --categories five`).


---
_generated 2026-10-14 15:14 UTC from the run records_
