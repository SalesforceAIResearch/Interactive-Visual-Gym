# Results

**Where these numbers come from.** The run described in [docs/REPRODUCE.md](../docs/REPRODUCE.md): the 50-case suite,
three cook personas (150 episodes per seat per run), three independent runs per seat, the simulated cook
`gemini-3.7-flash`, measured assistant latency charged to the clock, the final heat physics (food done → safe 40 ticks →
smokes → 45 ticks → burnt), and the interaction-quality rubric as the mean of the Gemini and GPT judges. Every table is
copied from the run's experiment files; the same tables are machine-readable in [`data/`](data/).

## Leaderboard (mean ± SD over three independent runs)

| Seat | In-time success | Task success | Prevention timing (DT) | Detection F1 | Interaction quality |
|---|---|---|---|---|---|
| Muse Glimmer | 0.43 ± 0.03 | 0.57 ± 0.01 | 0.47 ± 0.01 | 0.63 ± 0.02 | 0.18 ± 0.00 |
| Qwen3.8-27B | 0.27 ± 0.04 | 0.41 ± 0.02 | 0.41 ± 0.01 | 0.63 ± 0.01 | 0.40 ± 0.01 |
| Gemini 3.8 Flash | 0.30 ± 0.02 | 0.58 ± 0.03 | 0.33 ± 0.01 | 0.66 ± 0.01 | 0.65 ± 0.01 |
| GPT-6 Astra | 0.29 ± 0.03 | 0.51 ± 0.04 | 0.37 ± 0.03 | 0.68 ± 0.02 | 0.66 ± 0.02 |
| Qwen3-Omni 30B | 0.23 ± 0.03 | 0.28 ± 0.02 | 0.34 ± 0.03 | 0.45 ± 0.02 | 0.24 ± 0.01 |
| Gemini 3.1 Pro | 0.21 ± 0.02 | 0.39 ± 0.04 | 0.24 ± 0.03 | 0.53 ± 0.02 | 0.32 ± 0.01 |
| Gemini 3.8 Live | 0.23 ± 0.03 | 0.52 ± 0.01 | 0.28 ± 0.02 | 0.58 ± 0.01 | 0.61 ± 0.01 |
| GPT-realtime 2.1 | 0.22 ± 0.04 | 0.36 ± 0.04 | 0.37 ± 0.02 | 0.62 ± 0.02 | 0.44 ± 0.00 |
| GPT-realtime 2.1 (native) | 0.08 ± 0.02 | 0.18 ± 0.01 | 0.02 ± 0.01 | 0.19 ± 0.03 | 0.48 ± 0.01 |
| MiniCPM-o | 0.06 ± 0.01 | 0.07 ± 0.01 | 0.24 ± 0.01 | 0.33 ± 0.01 | 0.13 ± 0.00 |
| Molmo2-8B | 0.06 ± 0.01 | 0.09 ± 0.02 | 0.04 ± 0.01 | 0.05 ± 0.00 | 0.15 ± 0.01 |
| GLM-4.6V Flash | 0.06 ± 0.00 | 0.10 ± 0.01 | 0.10 ± 0.01 | 0.15 ± 0.01 | 0.59 ± 0.00 |
| Kimi-VL Thinking | 0.06 ± 0.00 | 0.14 ± 0.02 | 0.24 ± 0.01 | 0.32 ± 0.01 | 0.04 ± 0.00 |
| InternVideo3-8B | 0.05 ± 0.01 | 0.10 ± 0.01 | 0.08 ± 0.02 | 0.17 ± 0.01 | 0.26 ± 0.00 |
| StreamingVLM | 0.03 ± 0.01 | 0.04 ± 0.01 | 0.01 ± 0.00 | 0.02 ± 0.00 | 0.27 ± 0.00 |
| Gemini 3.8 Live (native) | 0.04 ± 0.00 | 0.10 ± 0.02 | 0.01 ± 0.00 | 0.10 ± 0.00 | 0.56 ± 0.01 |
| Proact-VL | 0.03 ± 0.01 | 0.04 ± 0.01 | 0.01 ± 0.00 | 0.01 ± 0.01 | 0.16 ± 0.00 |
| MMDuet2 | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | – |

Columns: in-time success = correct dish served within the time cap; task success = correct dish served at any time;
prevention timing = how early the assistant's correct calls came (1 = the mistake was prevented, 0 = never named);
detection F1 = naming the fired mistakes inside their windows, against false alarms; interaction quality = the rubric
score of every assistant line (five categories, factual-grounding gate). Definitions: [docs/BENCHMARK.md](../docs/BENCHMARK.md#4-metrics)
and [docs/INTERACTION_QUALITY_GRADING.md](../docs/INTERACTION_QUALITY_GRADING.md).

Run 1, pooled over 2610 episodes: in-time 0.154, task success 0.256, DT 0.199, F1 0.344.

## Full tables

| page | contents |
|---|---|
| [task_outcome.md](task_outcome.md) | outcome, in-time success, efficiency, prevention timing, detection, per run and pooled |
| [interaction_quality.md](interaction_quality.md) | the rubric's five categories per seat and run |
| [qa_proactive.md](qa_proactive.md) | question answering (Q1–Q4) and proactive-alert accuracy |
| [latency_ablation.md](latency_ablation.md) | measured latency vs a fixed 2-tick delay |
| [cook_model_ablation.md](cook_model_ablation.md) | swapping the model behind the simulated cook, 90 cells |

The streaming APIs appear twice: *polled* (the harness calls them every 2 ticks) and *native* (a live session decides
itself when to speak). Seats run with the text state map instead of frames are marked in the tables.
