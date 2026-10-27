# Kimi-VL-A3B-Thinking-2506: setup (2026-09-18, roster row 14)

**Status: ready.** It is served as `kimi-vl-thinking` on 127.0.0.1:18650. One server supports three modes, chosen per request: thinking on, thinking with a short token budget, and thinking off.

## Which model
- **Weights:** `moonshotai/Kimi-VL-A3B-Thinking-2506` at commit `aa1730989e75`. It is public, not gated, and 32.8 GB in bf16 (30.7 GiB on the GPU).
- **Architecture:** a MoE with 16B total and about 3B active parameters, plus the MoonViT vision encoder.
- **Newest release:** this is the newest Kimi-VL thinking release. The Hugging Face listing for `moonshotai` shows only `Kimi-VL-A3B-Thinking` (2025-04) and `-2506` (2025-06).
- **Newer open Moonshot vision models exist, but none fits here.** `Kimi-K2.5` and `Kimi-K2.6` are natively multimodal thinking models at 595 GB each. `Kimi-K3` is 1.56 TB. None fits on one or two A100-80GB GPUs, so they are not candidates for this box.

## Environment
- **venv:** `$ROOT/envs/kimi-vl-thinking`, created with `/opt/conda/bin/python3 -m venv` and filled with the venv's own pip. It holds vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0, tiktoken and blobfile.
- **Weights cache:** `HF_HOME=$ROOT/hf_cache/kimi-vl-thinking`. The `figures/*` files must be present too: huggingface_hub 1.x refuses an incomplete snapshot when `HF_HUB_OFFLINE=1`.
- **Serve script:** `$LM/pass3_setup/serve_kimi_vl_thinking.sh`. vLLM supports `KimiVLForConditionalGeneration` natively. The flags are `--dtype bfloat16 --trust-remote-code --max-model-len 32768 --max-num-seqs 2 --gpu-memory-utilization 0.90 --limit-mm-per-prompt {"image":8,"video":0} --enable-prefix-caching`, plus the reasoning plugin described below. Cold start is about 3 min.
- **Chat template:** the shipped template has a `system` role branch, so it needs no fix.

## Fix needed: a reasoning-parser plugin (`$LM/kimi-vl/kimi_vl_reasoning_parser.py`)
Kimi-VL writes `◁think▷ … ◁/think▷` and then its answer. Neither marker is a single token: they are 5 and 6 tiktoken tokens. vLLM 0.19.1 has no parser for them, and its `BaseThinkingReasoningParser` needs single-token markers. Without a parser, two things break:
- the benchmark's `_parse_json` would pick the first `{` out of the thinking text;
- the json_schema grammar would constrain the output from the first token, which silently turns thinking off.

The plugin, registered as `kimi_vl`, matches the markers on decoded tail text. It does three things:
- It moves the thinking into `reasoning_content`, so `content` holds only the JSON.
- It applies the reply schema only after `◁/think▷`.
- It treats a prompt that already ends in `◁think▷◁/think▷` as "reasoning ended". That is how thinking off works: the request prefills an empty think block and continues it, so the grammar applies from the first token.

`--reasoning-config '{"reasoning_start_str":"◁think▷","reasoning_end_str":"◁/think▷"}'` enables vLLM's per-request `thinking_token_budget`. A smoke test confirmed the budget cuts the thinking short and forces the end marker.

All three modes were checked on the server: the content is always clean JSON, and the reasoning is split out.

## Bench: 24 real calls, one request in flight, sampling T=0.8 (model card and Moonshot blog), max_tokens 4000
| variant | median | p90 | prompt tok | completion tok | schema ok | spoke | flags raised |
|---|---|---|---|---|---|---|---|
| **thinking on**, full frames | **10.48 s** | **32.40 s** (max 58.9 s) | 8,752 | 730 | 23/24 | 22 | 7 |
| thinking on, half frames | 10.50 s | 26.07 s | 5,408 | 769 | 23/24 | 23 | – |
| **thinking budget 256 tokens**, full frames | **6.18 s** | **10.44 s** | 8,752 | 457 | **24/24** | 24 | 8 |
| **thinking off** (prefill), full frames | **3.69 s** | **7.49 s** | 8,763 | 275 | **24/24** | 24 | 3 |
| thinking off, half frames | 3.95 s | 7.11 s | 5,419 | 292 | 24/24 | 24 | – |

The runs are saved as `results/kimivl_think_schema_s1.0.json`, `kimivl_think-schema_s0.5.json`, `kimivl_budget256_schema_s1.0.json`, `kimivl_nothink_schema_s1.0.json` and `kimivl_nothink-schema_s0.5.json`.

- **Thinking-on failure.** The one failure is call 18 (master_hard_4 t408): 17,202 characters of thinking hit the 4,000-token cap before any answer, which took 58.9 s. The median thinking length is about 2,450 characters.
- **Frame size.** Halving the frames does not help, because latency here is decode-bound. Full frames are kept.
- **Pace.** No mode keeps the 1.2 s tick. With thinking on, a reply is about 9 ticks late at the median and 27 at p90. The 256-token budget cuts that to about 5 ticks, and thinking off to about 3.

## Sanity read (reference = the round-4 API reply in calls.jsonl)
- **[0] easy_hard_1 t16.** REF: "You just chopped the cheese instead of trashing it!…"
  - Thinking on: "The cheese you're chopping isn't the tomato. Toss it in the trash and grab a tomato…", with flag `mis_prep`. **Correct.**
  - Thinking off: "Looks like you're chopping the cheese instead of the tomato!…" It sees the error but raises no flag and says the tomato must be raw.
  - Budget 256: it says to trash the cheese but raises no flag.
- **[2] easy_medium_1 t32.** REF: wrong order, cheese before tomato.
  - Thinking on: "Warning: The cheese was added to the plate before the tomato was chopped, which violates the required order…", with flag `wrong_order`. **Correct.**
  - Thinking off: "The cheese needs to be cooked first…" This is wrong.
  - Budget 256: "Great! You've cleared the plate…" This is false.
- **[5] easy_hard_2 t94.** REF: the cook has not moved; go to the serving pass.
  - Thinking off: "You're still standing by the cheese crate… Go to the serving pass, then serve." Correct.
  - Thinking on and budget 256: "Proceed to the serving pass and serve." Correct advice, but none of the modes raises a flag.
- **[13] hard_hard_1 t254.** REF: go to the serving pass. All three modes invent a smoking fryer.
- **[18] master_hard_4 t408.** REF: "Stop! You just picked up an onion…"
  - Thinking on: no reply, because it hit the length cap.
  - Budget 256: "Stop! You just scooped cheese onto your plate, but the recipe… doesn't include cheese…", with flag `mis_prep`. It gets the right error class but names the wrong ingredient.
  - Thinking off: it narrates the recipe order.

**The answer to row 14's question:** yes, full thinking catches the two textbook errors that thinking off misses (calls 0 and 2). The price is about 7 s of extra median latency and a 30–60 s tail. The 256-token budget keeps the flag rate but loses those two specific catches.

## Proposed PROFILES entries and benchmark flags
```python
# Kimi-VL-A3B-Thinking-2506: card sampling T=0.8; the server's kimi_vl reasoning parser applies the schema after ◁/think▷.
"kimi-vl-think-schema": dict(temperature=0.8, extra_body={}, prefill=None, reply_schema=True),
"kimi-vl-think256-schema": dict(temperature=0.8, extra_body={"thinking_token_budget": 256}, prefill=None, reply_schema=True),
# thinking OFF: prefill an empty think block and continue the assistant turn.
"kimi-vl-nothink-schema": dict(temperature=0.8,
    extra_body={"add_generation_prompt": False, "continue_final_message": True},
    prefill={"role": "assistant", "content": "◁think▷◁/think▷"}, reply_schema=True),
```
The main row, thinking on:
```
--assist-backend openai --assist-model kimi-vl-thinking --assist-base-url http://127.0.0.1:18650/v1 \
--assist-profile kimi-vl-think-schema --assist-max-tokens 4000 --assist-image-scale 1.0
```
For the ablations, swap in `--assist-profile kimi-vl-think256-schema` or `kimi-vl-nothink-schema`.

`bench_kimi.py` in this folder registers these profiles in-process to reproduce the bench; the engine was not edited.

**For the orchestrator:**
- `OpenAIBackend` does not pass `thinking_tokens` through. Reasoning is returned in `message.reasoning_content`, which the engine ignores. That is harmless, but if thinking cost should be logged, read `reasoning_content`.
- With max_tokens 4000, about 1 in 24 thinking-on calls runs out of tokens with no answer. Consider `--assist-max-tokens 8000`, or accept this as part of the model's behaviour.
