# Results

> **Status of these numbers.** The table on this page comes from the benchmark run described in
> [docs/REPRODUCE.md](../docs/REPRODUCE.md): the 25 shipped cases, the 75 fixed cells (25 cases x 3 personas),
> three independent runs per seat (225 episodes per seat), the assistant's measured reply latency charged to the
> game clock, the code-default Gemini model as the simulated person, and the two-judge interaction-quality rubric
> with a third judge on most rounds. Every value is copied from `formal_run_2.md`; the same rows, machine-readable,
> with the per-run values behind every mean, are in [`data/formal_run_2.json`](data/formal_run_2.json).

## Leaderboard

Nineteen seats; 75 cells per seat per run; three runs. Seats are listed in AQ order, as in the source table.
Metric definitions follow the table; the full account is in [docs/BENCHMARK.md](../docs/BENCHMARK.md).

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

### The columns

Every column except AQ is the mean over the three runs, ± half the range between the runs (the largest run value
minus the smallest, halved). A dash means the value is not defined for that seat (no successful episode for EFF,
no flag ever raised for F1, no spoken line for AQ).

- **AQ [95% CI]** — interaction quality: every assistant line is graded by LLM judges against the engine's exact
  state at that tick on a per-round rubric with a factual-grounding gate (a line that states something false about
  the house scores 0). AQ is the mean per episode, pooled over the 225 episodes of the seat, with a 2000-draw
  bootstrap 95% interval over episodes. **AQ r1 / r2 / r3** are the per-run means. The rubric and the judges are in
  [docs/INTERACTION_QUALITY_GRADING.md](../docs/INTERACTION_QUALITY_GRADING.md).
- **won** — the share of episodes in which the person finished the chore correctly (every checkpoint of the task
  true at the end), at any time.
- **S (in-time success)** — won *and* finished inside the time cap `C = 2 (T_ref + 20)` ticks, where `T_ref` is the
  measured clean-plan time of the case (`benchmark/t_ref.json`). A won episode that overruns the cap does not count.
- **EFF** — efficiency over the successful episodes only: `(2 TT + c CC) / (2 + c)`, with TT the share of the time
  budget left (`(C - T) / (C - T_ref)`, clipped to [0, 1]), CC the conversation efficiency (1 minus the share of the
  episode's ticks the assistant spent talking, each spoken line counted as 2 ticks) and `c` from the persona's
  interruption preference (3 when the person asked not to be interrupted, 1 otherwise). Because it is conditional on
  success, a seat that rarely succeeds has an EFF computed on a handful of episodes, which is why its spread is wide.
- **DET** — detection, the mean of DT and F1.
- **DT** — detection timing: per forced mistake, 1 if the assistant prevented it (named it before it physically
  happened), 0.5 if it named it correctly after it landed, 0 if it missed it; averaged over the forced mistakes.
  Prevention is worth twice a late catch.
- **F1** — the harmonic mean of precision and recall over the assistant's flags, so that a seat which flags
  everything is not rewarded: recall is the share of fired mistakes it named (the right name or a defensible twin
  name, at any time after the mistake was injected), precision is detections over detections plus false alarms (a
  wrong name, a name outside the six, or a flag before any mistake was forced that is not true of the world).
  Repeated flags with the same name count once.

### The seats

The seats are the 19 of the roster that were run (`vhhome-roster`; [docs/ASSISTANTS.md](../docs/ASSISTANTS.md)):

- **Frontier API, polled**: `gpt-6-astra`, `gemini-3.1-pro-preview`, `gemini-3.8-flash`. One call every 2 ticks.
- **Streaming APIs, polled and native**: `gemini-3.8-live` and `gpt-realtime-2.1` are the streaming models driven
  like chat models (one session per turn, forced reply every 2 ticks); `gemini-3.8-live__native` and
  `gpt-realtime-2.1__native` are the same weights in their own streaming mode, fed video, the person's speech as
  audio and a tick marker, speaking when they choose. The native seats carry the second- and fourth-highest AQ in the
  table while finishing almost no chore (`won` 0.05 and 0.07): what they say is good, but they say little.
- **Open-weight vision-language models, served locally**: `qwen3.8-27b`, `molmo2-8b`, `internvideo3-8b`,
  `glm-4.6v-flash`, `kimi-vl-thinking`, `museglimmer`. One GPU and one port each, measured latency charged.
- **Duplex / proactive open models, served locally**: `minicpmo`, `qwen3-omni-30b-a3b`, `mmduet2`, `proact-vl`,
  `streamingvlm`. The last three run through shims around their authors' own streaming loops with their authors'
  prompt shape; they never set a mistake `flag`, so their DT, DET and F1 read zero by construction, and they should
  be read on AQ and `won`.
- **The floor**: `silent` never speaks. Its `won` (0.01) is what the person achieves on its own against the forced
  mistakes, and every other number is read against it.

Notes on the table:

- AQ was graded by two judges (`gemini-3.7-flash`, `gpt-5.4-mini`) on every round and by a third
  (`claude-haiku-5-5`) on most rounds (99%, 99% and 44% of the rounds of runs 1, 2 and 3); each round's score is the
  mean over the judges that answered it, so the two judges are averaged where the third is missing.
- `silent` and `mmduet2` never spoke, so they have no AQ.
- `S` requires the time cap; `won` does not. The highest `won` is 0.36 (`gpt-realtime-2.1`), the highest `S` 0.11;
  most won episodes overrun twice the clean-plan time.
- The highest detection is `gemini-3.1-pro-preview` (DET 0.57, DT 0.40), which is not the seat with the highest AQ
  or the highest `won`: naming mistakes, finishing chores and speaking well are measured separately on purpose.

## Regenerating the table

`vhhome-score` writes this table (as `final_results.md`, with `final_results.json` beside it) from the three run
roots, without regrading anything that is already graded:

```bash
vhhome-score --run runs/formal/r1 runs/formal/r2 runs/formal/r3 --only table
```

See [docs/REPRODUCE.md](../docs/REPRODUCE.md) for the full procedure (the three runs per seat and the scoring
steps) and [docs/ASSISTANTS.md](../docs/ASSISTANTS.md) for how each seat is reached.
