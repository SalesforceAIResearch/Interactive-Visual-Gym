# Molmo 2-O 7B — official-guidance audit (2026-09-18)

Served name `molmo2-o-7b`, port 18600, env `$ROOT/envs/molmo2-o-7b` (vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0),
weights `allenai/Molmo2-O-7B` @ `7844106`. The weights ship in fp32 and are served in bf16.

Molmo2-O uses the same `Molmo2ForConditionalGeneration` class as Molmo2-8B, with an OLMo-3-7B-Instruct text backbone. It also has a **byte-identical chat template** and a **byte-identical `preprocessor_config.json`** (checked with `diff`). So every Molmo2 source and verdict in `../molmo2/AUDIT.md` applies here, including the system-message analysis. This file lists only what is specific to O-7B and the O-7B measurements.

**Outcome.**
- **Sampling:** greedy decoding (temperature 0), AllenAI's own vLLM setting. This uses the same shared `molmo2-schema` PROFILES change as Molmo2-8B.
- **Scheduler budget:** `--max-num-batched-tokens 36864`, the demo's value, is **rejected**. It is slower on this A100 in every alternating restart: 1.83 s against 2.23 s uncached prefill.

## O-7B-specific sources
- **[card-o]** https://huggingface.co/allenai/Molmo2-O-7B: "Molmo2-O-7B is based on Olmo3-7B-Instruct … uses SigLIP 2". The examples are identical to Molmo2-8B's: single user turn, `dtype="auto"`, no sampling parameters.
- **[repo]** https://github.com/allenai/molmo2 README, "Convert Checkpoint": "N: … 65536 for Molmo2-O-7B" (max model length).
- **[demo]** `scripts/unified_demo.py`: `dtype="bfloat16"` (fp32 only with `--use_float32`), `temperature=0`, `max_num_batched_tokens=36864`.
- `generation_config.json` holds token ids only, with no sampling fields. `tokenizer_config.json`: `bos_token = eos_token = <|endoftext|>`.

## Item table
| # | item | ours (before) | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint / dtype | `allenai/Molmo2-O-7B`, fp32 weights served as bf16 | the demo serves bf16 by default; the card uses `dtype="auto"` (fp32 in HF) | [demo], [card-o] | **matches** (bf16 is the demo's default) |
| 2 | engine / version / flags | vLLM 0.19.1, native class, `--trust-remote-code`, `--max-model-len 32768` | vLLM ≥ 0.15.0; conversion max_model_len 65536 | [repo], vLLM supported-models row lists `allenai/Molmo2-O-7B` | **matches in effect** (longest prompt 13.7k tokens) |
| 3 | chat template / system | the same fold template as 8B | no system role; instructions go in the user turn | as for 8B | **matches.** `render_fold.txt`: identical to the official template given the folded user turn. The server adds BOS `<|endoftext|>` exactly as the processor's `insert_bos` does. |
| 4a | image tiling | `--assist-image-scale 0.5` (2 crops plus global per frame) | up to 8 crops per image | as for 8B | **deviates on purpose.** Official full size costs **+3.5 s** paired (7.9 s against 4.0 s median) and is *less* reliable on O-7B: **22/24** schema-valid in both full-size runs, because calls 3 and 21 loop until the 4,000-token cap ("I'll make sure to scoop raw cheese onto the plate, then head to the serving pass and serve. I'll make sure to scoop…"). |
| 5 | sampling | T 0.2 | greedy (temperature 0) | [demo] | **fixed**, adopted |
| 6 | thinking | none | not a reasoning model | [card-o] | **matches** |
| 7 | scheduler budget | vLLM A100 default of 2048 | 36864 | [demo] | **deviates on purpose.** Prefill median was 1.83 / 1.83 s at 2048 against 2.23 / 2.22 s at 36864, in two alternating rounds (plus the phase-2 probe at 1.89 against 2.38 s). |
| 8 | structured output | `json_schema` | not discussed. The schema is *needed* here: `json_object` alone gave 20/24 in setup. | – | **matches** |

## Bench (24 real calls, `max_tokens 4000`; interleaved A/B, same method as `../molmo2/AUDIT.md`)
| run (tag, `_audit`) | server | frames | median | p90 | paired Δ | compl tok | schema ok | spoke | flags | length-cut |
|---|---|---|---|---|---|---|---|---|---|---|
| **before**: `molmo2o_ab_before` (T 0.2) | current | 0.5 | 3.98 s | 4.98 s | 0 | 200 | 24/24 | 24 | 1 | 0 |
| **after**: greedy `molmo2o_ab_t0` | current | 0.5 | 3.37 s | 5.24 s | **−0.77 s** | 214 | 23/24 | 23 | 2 | 1 |
| greedy, official full tiling `molmo2o_ab_t0_s1.0` | current | 1.0 | 7.88 s | 10.38 s | +3.47 s | 254 | 22/24 | 22 | 1 | 2 |
| before `molmo2o_ab2_before` (T 0.2) | + mnbt 36864 | 0.5 | 4.54 s | 5.33 s | 0 | 225 | 23/24 | 23 | 2 | 1 |
| greedy `molmo2o_ab2_t0` | + mnbt 36864 | 0.5 | 3.96 s | 7.10 s | −0.05 s | 222 | 24/24 | 24 | 2 | 0 |
| prefill only, phase 3, alternating restarts | 2048 vs 36864 | 0.5 | **1.83 / 1.83 vs 2.23 / 2.22 s** | 1.89 / 1.89 vs 2.32 / 2.33 s | – | – | – | – | – | – |
| prefill only (max_tokens 1), calls 2–23 | current → + mnbt 36864 | 0.5 | 1.89 → 2.38 s | 1.99 → 3.67 s | +0.45 s, 0/22 calls faster | – | – | – | – | – |

Sequential phase-1 runs, all at half frames:

| tag (`_audit`) | T | server | median | schema ok |
|---|---|---|---|---|
| `molmo2o_before` | 0.2 | current | 4.52 s | 24/24 |
| `molmo2o_t0` | 0 | current | 4.35 s | 24/24 |
| `molmo2o_t0_mnbt` | 0 | + mnbt 36864 | 5.11 s | 24/24 |
| `molmo2o_before_mnbt` | 0.2 | + mnbt 36864 | 4.78 s | 24/24 |

- **Greedy.** Greedy is the official setting.
  - It is not slower: paired −0.77 s and −0.05 s.
  - Its validity equals T 0.2's: 95/96 over four runs each.
  - The single failure in each case is **call 3**, which loops to the cap under both temperatures ("…the walls are gray. The pass is in the center of the room…"). This is a model weakness, not a sampling one.
  - Greedy is adopted, and the profile entry is shared with 8B.
- **Scheduler budget.** The phase-2 probes disagreed in sign between 8B and O-7B, so both models were re-measured in alternating server restarts (`audit_plan3.sh`: 2048, 36864, 2048, 36864). 36864 was slower every time:
  - O-7B: 1.83 / 1.83 s against 2.23 / 2.22 s;
  - 8B: 1.72 / 1.73 s against 2.01 / 2.00 s.

  The vLLM default is kept.
- **Sanity read** (greedy, half frames; reference = round-4 API reply):
  - [2] easy_medium_1 t32: "I see the cheese on the plate already. That's a mistake… Clear the plate and start over." It catches the error in every half-frame run but labels it `add_foreign` where the reference says `wrong_order`. At full size it misses: "No need to flag anything yet."
  - [5] t94: "You're still standing by the cheese crate with the correct plate, not at the serving pass yet. Go to the serving pass, then serve." The advice is correct but it raises no `forget_next_step` flag.
  - [0] t16: "Good catch! Let's fix this. First, trash the cheese…" It gives the right fix but no flag, followed by an unrequested recipe walk-through.
  - [13] and [18] are misses.
  - It spoke 24/24 against the reference's 15/24: a chatty narrator, as in setup.

## Changes to apply
- **PROFILES:** `"molmo2-schema": dict(temperature=0.0, extra_body={}, prefill=None, reply_schema=True)`. This is the same entry as Molmo2-8B, so apply it once.
- **Serve script:** unchanged (`pass3_setup/serve_molmo2_o_7b.sh`).
- **Benchmark flags:** unchanged: `--assist-profile molmo2-schema --assist-max-tokens 4000 --assist-image-scale 0.5`.
