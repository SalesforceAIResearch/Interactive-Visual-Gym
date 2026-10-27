# CookSim · Experiment 5 — swapping the model behind the simulated cook (ablation)

**Setting.** Same assistant (Gemini 3.7 Flash), same 90 episodes, four different models in the **cook** seat with identical prompt and harness. Rerun v6: 360 episodes on disk (4 cooks × 30 cells), scored with the standard scorer; the cook's own speed is not a measured quantity in this ablation.

| Cook model | n | In-time success | Task success | DT | F1 | FA / ep | cook words / ep |
|---|---|---|---|---|---|---|---|
| GPT-6 Astra | 87 | 0.24 | 0.51 | 0.20 | 0.55 | 0.15 | 190 |
| GPT-realtime 2.1 | 75 | 0.08 | 0.40 | 0.45 | 0.73 | 0.09 | 389 |
| Muse Glimmer | 61 | 0.00 | 0.07 | 0.18 | 0.62 | 0.20 | 442 |
| Qwen3.8-27B | 71 | 0.03 | 0.35 | 0.16 | 0.69 | 0.14 | 374 |

Provenance: `<run>/frames/<persona>/<cook>/<case>.json` (cells `benchmark/cook_ablation_cells.json`, 30 cases × 3 personas, four models in the cook seat; see docs/REPRODUCE.md §3).


---
_generated 2026-10-14 15:14 UTC from the run records_
