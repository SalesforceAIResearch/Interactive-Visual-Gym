# Molmo 2-O 7B: setup (2026-09-18, roster row 12)

**Status: ready.** It is served as `molmo2-o-7b` on 127.0.0.1:18600 and was verified on the 24 real calls.

## What it is
- **Weights:** `allenai/Molmo2-O-7B` at commit `784410650d12`. It is public, not gated, and 31.1 GB on disk because the weights ship in **fp32**. It is served in bf16, which takes 14.5 GiB on the GPU.
- **Architecture:** `Molmo2ForConditionalGeneration`. The text model is OLMo-style: 32 layers, OLMo qk-norm, norm-after, and YaRN on 24 of the 32 layers. The vision encoder is a SigLIP2-style ViT. The architecture class is the same as Molmo2-8B's, and vLLM 0.19.1's `molmo2.py` implements the OLMo variants (`qk_norm_type="olmo"`, `norm_after`, `rope_scaling_layers`), so it needs **no shim**.
- **Thinking:** none. It is not a reasoning model.

## Environment
- **venv:** `$ROOT/envs/molmo2-o-7b`, created with `/opt/conda/bin/python3 -m venv` and filled with the venv's own pip. It holds vllm 0.19.1, torch 2.10.0+cu128 and transformers 5.17.0, the same versions as the working Molmo2-8B env.
- **Weights cache:** `HF_HOME=$ROOT/hf_cache/molmo2-o-7b`.
- **Serve script:** `$LM/pass3_setup/serve_molmo2_o_7b.sh`. Its flags are `--dtype bfloat16 --trust-remote-code --max-model-len 32768 --max-num-seqs 2 --gpu-memory-utilization 0.90 --limit-mm-per-prompt {"image":8,"video":0} --enable-prefix-caching --chat-template molmo2_chat_template_system.jinja`. To launch it:
  `flock -w 10800 /tmp/gpu0.lock bash -c 'CUDA_VISIBLE_DEVICES=0 exec bash serve_molmo2_o_7b.sh' > logs/serve_molmo2_o_7b.log 2>&1 &`
- **Cold start:** about 6.5 min. The weight load took 338 s.

## Fix needed: the system message (checked, not assumed)
- The shipped `chat_template.jinja` is **byte-identical** to Molmo2-8B's (checked with `diff`).
- Rendered through the model's own tokenizer, a conversation that starts with a system message raises `TemplateError: Conversation roles must alternate user/assistant/...`.
- The existing `pass3_setup/molmo2_chat_template_system.jinja` works unchanged:
  - it renders `<|im_start|>user\nSYS\n\nhi<|im_end|>\n<|im_start|>assistant\n`;
  - for conversations with no system message (single-turn and multi-turn) it renders byte-identically to the shipped template.
- The server stops cleanly: `finish_reason=stop`, no run-on output.

## Bench: 24 real calls, one request in flight
| variant | median | p90 | prompt tok | completion tok | schema ok | spoke |
|---|---|---|---|---|---|---|
| **half frames (`image-scale 0.5`), schema: chosen** | **4.74 s** | **6.65 s** | 8,438 | 211 | **24/24** | 24 |
| full frames, schema | 7.56 s | 10.31 s | 13,662 | 236 | 23/24 | 23 |
| half frames, no schema (`default`, json_object) | 4.25 s | 6.32 s | 8,438 | 203 | 20/24 | 22 |

The runs are saved as `results/molmo2o_schema_s0.5.json`, `results/molmo2o_schema_s1.0.json` and `results/molmo2o_noschema_s0.5.json`.

- **Full frames fail once.** In the full-frame run the one failure is call 12 (hard_hard_1 t76). The reply degenerated into a loop, repeating "The plate is empty on the counter. You can now pick up a potato…" until it hit 4,000 tokens.
- **The schema is needed.** Unlike Molmo2-8B, the O-7B model drops fields without it: 20/24 valid under json_object mode.
- **It cannot keep the game's pace.** At about 4.7 s median, each reply lands about 4 ticks late at 1.2 s per tick, similar to Molmo2-8B (5.1 s).

## Sanity read (half frames, schema; reference = the round-4 API reply in calls.jsonl)
- **[0] easy_hard_1 t16.** REF: "You just chopped the cheese instead of trashing it!…" O-7B: "Good catch! Let's fix this. Trash the cheese and grab a tomato to chop…" It gives the right fix but does not name the error, sets no flag, and adds a long run of unrequested steps.
- **[2] easy_medium_1 t32.** REF: wrong order, cheese before tomato. O-7B: "I see you put cheese on the plate before the tomato was chopped. That's a mistake. Clear the plate and start over…" It caught the error but flagged it as `add_foreign` rather than `wrong_order`.
- **[5] easy_hard_2 t94.** REF: "You have not moved yet — …go to the serving pass". O-7B: "You're still standing by the cheese crate with the correct plate, not at the serving pass yet. Go to the serving pass, then serve." Correct, but no `forget_next_step` flag.
- **[13] hard_hard_1 t254.** REF: "take it to the serving pass". O-7B: "The fish is done and smoking in the top fryer. Grab the plate and scoop it out…" This is a miss that invents a hazard.
- **[18] master_hard_4 t408.** REF: "Stop! You just picked up an onion…" O-7B: "You've got the steak on the plate. Now, park the plate…" A miss.
- **Overall:** it spoke 24/24, against 15/24 for the reference, and raised 1 flag in 24 calls. At [20] it recited the whole recipe (about 250 words) where the reference stayed silent. The profile to expect is a chatty narrator with little detection, below Molmo2-8B, which raised 3 flags.

## Proposed PROFILES entry and benchmark flags
The existing `molmo2-schema` profile is exactly what was benched, so reuse it. An alias could be added if wanted:
```python
"molmo2-o-schema": dict(temperature=0.2, extra_body={}, prefill=None, reply_schema=True),
```
```
--assist-backend openai --assist-model molmo2-o-7b --assist-base-url http://127.0.0.1:18600/v1 \
--assist-profile molmo2-schema --assist-max-tokens 4000 --assist-image-scale 0.5
```
Molmo2-O publishes no sampling guidance; its generation_config holds only token ids.
