# GLM-4.6V-Flash — official-guidance audit (2026-09-18)

Served name `glm-4.6v-flash`, port 18500, env `$ROOT/envs/glm` (vllm 0.19.1, torch 2.10.0+cu128, transformers 5.17.0),
weights `zai-org/GLM-4.6V-Flash` @ `411bb4d`, bf16.

**Outcome.** Two changes, both taken from the official guidance:
- **Sampling:** switch to the model card's decoding parameters. This needs a PROFILES change.
- **Server:** add `--reasoning-parser glm45`. This is already in `serve_glm46v_flash.sh`.

Everything else either matches the guidance, or deviates on purpose with a measurement behind it (thinking off, text before images).

## Sources
- **[card]** https://huggingface.co/zai-org/GLM-4.6V-Flash (README in the local snapshot):
  - "Environment Installation": `pip install vllm>=0.12.0`, `pip install transformers>=5.0.0rc0`.
  - "Evaluation Settings": "We primarily use vLLM as the backend … we recommend the following decoding parameters: top_p: 0.6, top_k: 2, temperature: 0.8, repetition_penalty: 1.1, max_generate_tokens: 16K".
  - The Quick Start message lists `{"type":"image"}` before `{"type":"text"}`.
- **[gencfg]** `generation_config.json`: `do_sample: true, top_p 0.6, temperature 0.8, top_k 2`. vLLM loads these as server defaults; the serve log says "Default vLLM sampling parameters have been overridden by the model's generation_config.json: {'temperature': 0.8, 'top_k': 2, 'top_p': 0.6}".
- **[glmv]** https://github.com/zai-org/GLM-V README:
  - "The model also introduces a **Thinking Mode** switch … This switch works the same as in the `GLM-4.5` language model."
  - "Serve GLM-V Series Models Locally … vLLM (v0.25.0+) — see recipes".
- **[recipe]** https://recipes.vllm.ai/zai-org/GLM-4.6V:
  - Serve command: `vllm serve zai-org/GLM-4.6V-FP8 --tensor-parallel-size 4 --tool-call-parser glm45 --reasoning-parser glm45 --enable-auto-tool-choice --enable-expert-parallel --allowed-local-media-path / --mm-encoder-tp-mode data --mm-processor-cache-type shm`.
  - "Required vLLM: vLLM 0.12.0+".
  - "max-num-batched-tokens=32768 for prompt-heavy workloads".
- **[vllm-reasoning]** https://docs.vllm.ai/en/latest/features/reasoning_outputs/, supported-models table: "GLM-4.5 series | `glm45` | `json`, `regex` | ✅".
- **[tmpl]** `chat_template.jinja` in the snapshot. With `enable_thinking=false` it appends `/nothink` to the user turn and pre-fills `<think></think>\n` after `<|assistant|>`. It has a `system` branch that renders `<|system|>`.
- **[pp]** `preprocessor_config.json`: `size {shortest_edge: 12544, longest_edge: 9633792}` in pixels, `patch_size 14`, `merge_size 2`.

## Item table
| # | item | ours (before) | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint / dtype | `zai-org/GLM-4.6V-Flash` bf16 (`--dtype auto`) | same repo, `torch_dtype="auto"` (bf16). There is no official quantized Flash build, and FP8 is not usable on A100 anyway. | [card] Quick Start | **matches** |
| 2a | engine / version | vLLM 0.19.1, native `Glm4vForConditionalGeneration`, no remote code | vLLM ≥ 0.12.0, transformers ≥ 5.0.0rc0. The repo-wide README now says "v0.25.0+". | [card], [recipe], [glmv] | **matches** the card and the recipe floor. The v0.25 line is repo-wide and not specific to Flash; not upgraded (pending, optional). |
| 2b | reasoning parser | none | `--reasoning-parser glm45` | [recipe], [vllm-reasoning] | **fixed**, adopted (see bench) |
| 2c | tool parser, `--mm-encoder-tp-mode data`, `--mm-processor-cache-type shm`, expert parallel | not set | set in the recipe | [recipe] | **deviates on purpose.** These are for tool calls and for TP=4 multi-process serving. We use TP=1 with no tools, so none of them applies. |
| 2d | max-model-len / max-num-batched-tokens | 32768 / vLLM default (2048 on A100 in 0.19.1) | 65536 default / 32768 "for prompt-heavy workloads" | [recipe] | **matches in effect.** Tested 32768: uncached prefill (TTFT) was 1.090 s before and 1.102 s after (22 calls each), so no gain and not adopted. The prompt is about 8.3k tokens, so 32k context is plenty. |
| 3 | chat template | shipped template, system message → `<|system|>` | same | [tmpl] | **matches.** `render_diff.py` sent the benchmark's call 0 through the server's `/tokenize` and through the official `apply_chat_template`. With image runs collapsed the two are identical (7,708 tokens, 8 image blocks), in both text-first and images-first order (`render_textfirst.txt`, `render_imagesfirst.txt`). |
| 4a | visual input size | 8 × 1042×396 frames at native size (`--assist-image-scale 1.0`), about 518 visual tokens each | native resolution between 12,544 and 9.6M pixels | [pp] | **matches.** 1042×396 = 0.41 MP is well within range; the processor resizes it only to a multiple of 28. |
| 4b | image position | text first, then 8 images | the card example puts the image first | [card] | **deviates on purpose.** In a paired A/B, images-first was **+0.76 s** slower (median paired difference), gave 24/24 either way, and caught no extra error: it lost the one catch card sampling makes (call 2, wrong_order). |
| 5 | sampling | T 0.2 in the request. top_p 0.6 / top_k 2 came silently from [gencfg]; no repetition penalty. | T 0.8, top_p 0.6, top_k 2, repetition_penalty 1.1 | [card], [gencfg] | **fixed**, adopted (see bench) |
| 6 | thinking | off: `chat_template_kwargs.enable_thinking=false` (→ `/nothink` plus an empty think block) | the official switch is `enable_thinking`, as in GLM-4.5. The model's default is thinking on. | [glmv], [tmpl] | the switch **matches**. Keeping thinking off **deviates on purpose**: with thinking on (glm45 parser, card sampling, 12 calls) the median was 9.99 s and p90 13.5 s, against 3.5 s off. It gave 12/12 valid replies but 0 correct flags. |
| 7 | efficiency features | prefix caching, CUDA graphs (default), `max-num-seqs 2` | no MTP/EAGLE drafter is published for GLM-4.6V-Flash. The recipe's only speed flags are 2c and 2d. | [recipe], [card] | **matches** |
| 8 | structured output | `response_format json_schema` (xgrammar) | the glm45 parser supports `json` structured output | [vllm-reasoning] | **matches.** With glm45 and thinking off, `reasoning_content` was empty on 48/48 calls and the schema was valid 48/48. With thinking on, the schema applied after `</think>` (12/12). |

## Bench (24 real calls, `max_tokens 4000`, full frames)
The shared host drifts by up to ±1.8 s in median between identical sequential runs, because of load from the other agents. So request-level changes were measured with **interleaved A/B** (`$LM/molmo2/ab_bench.py`): each call is sent to every variant in rotating order on the same server, and the "paired Δ" column is the median over calls of (variant − before).

| run (tag, `_audit`) | server | median | p90 | paired Δ | compl tok | schema ok | spoke | flags | correct flag (of 5 in ref) |
|---|---|---|---|---|---|---|---|---|---|
| **before**: `glm_ab_before` (profile `glm46v-flash-schema`) | current | 3.49 s | 6.86 s | 0 | 245 | 24/24 | 23 | 0 | 0 |
| card sampling `glm_ab_card` | current | 3.68 s | 7.01 s | +0.25 s | 258 | 24/24 | 19 | 1 | 1 |
| card sampling + images first `glm_ab_card_imgfirst` | current | 4.36 s | 7.56 s | +0.76 s | 252 | 24/24 | 21 | 0 | 0 |
| before `glm_ab2_before` | + glm45 + mnbt 32768 | 3.50 s | 6.86 s | 0 | 245 | 24/24 | 23 | 0 | 0 |
| **after**: card sampling `glm_ab2_card` | + glm45 + mnbt 32768 | **3.61 s** | **7.23 s** | +0.27 s | 251 | **24/24** | 21 | 1 | 1 |
| sequential: `glm_card_rp`, `glm_card_imgfirst_rp` | + glm45 + mnbt | 4.50 / 4.59 s | 7.83 / 7.85 s | – | 260 / 252 | 24/24 | 21 / 19 | 1 / 0 | 1 / 0 |
| thinking ON, card, glm45 (`glm_think_card_rp`, 12 calls) | + glm45 + mnbt | 9.99 s | 13.53 s | – | 699 | 12/12 | 11 | 2 | 0 |
| prefill only (max_tokens 1), calls 2–23 | current / + mnbt | 1.090 / 1.102 s | 1.205 / 1.291 s | – | – | – | – | – | – |

Reference replies spoke on 15/24 calls. Raw rows are in `pass3_setup/results/<tag>.json`; logs are `pass3_setup/logs/ab_glm_p2s{0,1}.log` and `audit_run*.log`.

**Why card sampling is adopted despite +0.25 s.** The audit rule allows a latency cost when the official setting improves detection on the sanity read and the improvement can be shown:
- At call 2 (easy_medium_1 t32, cheese plated before the tomato; ref flag `wrong_order`), card sampling raised `wrong_order` in **4 of 4 runs**: "You messed up the order. Cheese shouldn't be on the plate yet. Clear the plate and start again." T 0.2 missed it in **3 of 3** runs, saying "Okay, you're heading to the trash. That's the right first step to undo the mistake."
- It spoke less (19–21 against 23), which is closer to the reference's 15.
- The latency cost is about 10 extra tokens, and validity is unchanged at 24/24.

Calls 0, 5, 13 and 18 are missed by every variant.
- Call 5 is a false location claim: "You're at the serving pass…" when the cook is still at the crate.
- These misses are the model's, not the setup's.

## Changes to apply
**Serve script** (already applied): `pass3_setup/serve_glm46v_flash.sh` gains `--reasoning-parser glm45`. Nothing else changed.

**PROFILES** (engine, for the orchestrator):
```python
# GLM-4.6V-Flash: model-card decoding (top_p 0.6, top_k 2, T 0.8, repetition_penalty 1.1), thinking off via the
# official enable_thinking switch; server runs --reasoning-parser glm45. Audit 2026-09-18: 24/24, paired +0.25 s,
# catches the wrong_order at call 2 in 4/4 runs (T 0.2: 0/3).
"glm46v-flash-schema": dict(temperature=0.8, top_p=0.6,
    extra_body={"top_k": 2, "repetition_penalty": 1.1, "chat_template_kwargs": {"enable_thinking": False}},
    prefill=None, reply_schema=True),
```
The benchmark flags are unchanged: `--assist-profile glm46v-flash-schema --assist-max-tokens 4000 --assist-image-scale 1.0`.

## Side finding (applies to every vLLM server here)
In vLLM 0.19.1, a `POST /tokenize` that carries images corrupts the engine's multimodal cache. Every later chat request with the same frames fails with HTTP 500: `AssertionError: Expected a cached item for mm_hash=…` in `multimodal/cache.py`. This cost the first S0 pass of this audit. Never call `/tokenize` with images on a server that is benching or running episodes. The render checks here were moved to run after the benches.

## Pending
- A vLLM ≥ 0.25 env, the GLM-V README's current floor, was not built. The card and the recipe accept 0.19.1, and nothing in the recipe needs a newer version at TP=1.
