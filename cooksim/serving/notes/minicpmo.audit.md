# MiniCPM-o 4.5 (`minicpmo`, port 18100): audit against the official guidance (2026-09-18)

**Verdict.** The setup follows the official guidance on every item except one: sampling.
- **Sampling was a hybrid.** The profile sent T 0.1, and vLLM filled in top_p 0.95 / top_k 20 from `generation_config.json`. Neither is an official MiniCPM-o recipe.
- **Fix adopted.** The official vLLM-with-AWQ recipe: T 0.7, top_p 0.8, top_k 100, repetition_penalty 1.05.
- **Effect.** Schema validity stays 24/24, and latency falls: median 2.10 → 1.81 s, p90 3.61 → 3.17 s.
- **Detection is unchanged.** 0 of the 5 reference flags are caught either before or after the fix.
- **The serve script is unchanged.** The fix is in the request profile only.

Sources (local copies are in `docs_cache/`):
- **[card]** HF model card, `openbmb/MiniCPM-o-4_5` and `openbmb/MiniCPM-o-4_5-awq` (identical text): `hf_cache/minicpmo/.../README.md`.
- **[cb-vllm]** CookBook, `deployment/vllm/minicpm-o4_5_vllm.md` (github.com/OpenSQZ/MiniCPM-V-CookBook).
- **[cb-awq]** CookBook, `quantization/awq/minicpm-o4_5_awq_quantize.md`.
- **[code]** The model's own `modeling_minicpmo.py` and `processing_minicpmo.py`.
- **vLLM recipes.** The vLLM recipes repo (github.com/vllm-project/recipes, `models/openbmb/`) has only MiniCPM-V-4.6 and MiniCPM5. There is **no MiniCPM-o 4.5 recipe**.

## Item table

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint | `openbmb/MiniCPM-o-4_5-awq` (W4A16 AWQ LLM; vision and resampler bf16), rev 02b87bc | the AWQ repo is openbmb's own upload (`base_model_relation: quantized`). [cb-awq] "Method 1 (Use the pre-quantized model with vllm)" serves exactly this repo. | [card] front matter; [cb-awq] §Method 1 | **matches (official build)** |
| 1b | quantization cost | not stated by the authors | the only statement is [card] "Easy Usage": "**Basic usage, recommended for 100% precision:** PyTorch inference with Nvidia GPU". It gives no AWQ number. [cb-awq] and the quantization README rate AWQ "Accuracy: High" with no figures. | [card]; cookbook `quantization/README.md` table | **deviates on purpose.** Our own bf16 run on the same 24 calls (`results/minicpmo_bf16_schema.json`) gives 24/24, the same misses and 2/24 flags, but is 1.5 s slower at the median. The authors' "100% precision" path is transformers, not vLLM, and would be far slower. |
| 1c | dtype | `--dtype bfloat16` | bf16 (`torch_dtype=torch.bfloat16` in the card examples); [cb-vllm] `--dtype auto`, which resolves to bf16 | [card] Quickstart; [cb-vllm] §2.1 | matches |
| 2 | engine | vLLM 0.19.1 (torch 2.10, transformers 4.57.6) | "MiniCPM-o 4.5 support has been merged into the official vLLM … starting from vLLM **v0.16.0**" | [cb-vllm] §1.1 | matches |
| 2b | flags | `--trust-remote-code --max-model-len 16384 --limit-mm-per-prompt {"image":8,"video":0,"audio":0} --max-num-seqs 2 --enable-prefix-caching` | `--trust-remote-code`, `--limit-mm-per-prompt` for multi-image input; `--max-model-len 2048/4096` in the examples, which is too short for our 4.7k-token prompts; `stop_token_ids [151643, 151645]` | [cb-vllm] §2.1, §2.5 | matches. The stop ids are the eos list in `generation_config.json`, which vLLM applies itself. |
| 3 | chat template | the server renders the exact request | the official processor (`model.chat` path: images as `<image>./</image>` joined by `\n`, `tokenizer.apply_chat_template(..., enable_thinking=False)`, `processor(..., max_slice_nums=1)`) | [code] `MiniCPMO.chat` | **matches, byte for byte.** Call 0 decodes to the same text through both paths (4,129 tokens): system turn, `<image_id>i</image_id><image>…</image>` ×8 before the text, then `<|im_start|>assistant\n<think>\n\n</think>`. |
| 4 | visual input | 8 images, `max_slice_nums 1` (64 tokens per frame), image ids on | "Chat with Video": frames with `max_slice_nums=1, use_image_id=False`. The streaming/duplex API also uses `max_slice_nums=1  # Increase for HD mode`. The multi-image example uses the processor default (9 slices). | [card] "Chat with Video", "Streaming Inference"; `preprocessor_config.json` (`max_slice_nums 9`, `use_image_id true`) | **matches the video recipe, with one exception: image ids.** vLLM 0.19.1 cannot turn image ids off for *image* inputs (`MiniCPMVMultiModalProcessor.get_image_prompt_texts` does not forward `use_image_id`). Verified: `--mm-processor-kwargs {"max_slice_nums":1,"use_image_id":false}` leaves the prompt at 4,129 tokens. The official processor without ids gives 4,105 tokens, which differs only in the 8 `<image_id>` tags. Only a `video_url` input would drop them. HD slicing (9) was benched: +0.7 s, same replies (`minicpmo_awq_slice9_schema`). |
| 5 | sampling | T 0.1 plus vLLM-injected top_p 0.95 / top_k 20 (server log: "Default vLLM sampling parameters have been overridden by the model's generation_config.json: {temperature 0.6, top_k 20, top_p 0.95}") | [cb-awq] vLLM on the AWQ model: `temperature=0.7, top_p=0.8, top_k=100, repetition_penalty=1.05`. [code] `chat()` default sampling: T 0.7, top_p 0.8, top_k 100, rep 1.02. `generation_config.json`: T 0.6, top_p 0.95, top_k 20. | [cb-awq] §Method 1 step 2; [code] `prepare_generation_config` | **fixed** (see bench). The AWQ+vLLM recipe is used because it is the one for our exact stack. |
| 6 | thinking | off (`chat_template_kwargs.enable_thinking=false`) | the model supports instruct and thinking modes in one model; "All results are reported in instruct mode"; every example has `enable_thinking=False` | [card] Evaluation note; examples | matches. No reasoning parser is needed: the template emits an empty `<think>\n\n</think>`. |
| 7 | efficiency | awq_marlin kernel, FLASH_ATTN (FA2) for the LLM and ViT, CUDA graphs, prefix caching, chunked prefill (V1 default) | no drafter (MTP/EAGLE) is published for o4.5. [cb-vllm] only adds `--max-num-batched-tokens 2048`, a memory knob. | [cb-vllm] §2.1 | matches |
| 8 | structured output | `response_format json_schema` (xgrammar) | not discussed by the authors; the vLLM default backend works with this model | — | matches. 24/24 valid, 0 length-capped. |

## Bench (24 real calls, one in flight, `--max-tokens 4000 --image-scale 1.0`, GPU 0, AWQ, 1 slice)

| tag | sampling | median | p90 | prompt tok | compl tok | schema ok | spoke | flags (match ref) |
|---|---|---|---|---|---|---|---|---|
| `minicpmo_awq_schema_audit_base` (before) | T 0.1 (+ top_p 0.95 / top_k 20 from gen-config) | 2.10 s | 3.61 s | 4,741 | 276 | 24/24 | 24 | 3 (0) |
| `minicpmo_awq_schema_audit_cardsampling` (**after**) | T 0.7 / top_p 0.8 / top_k 100 / rep 1.05 | **1.81 s** | **3.18 s** | 4,741 | 277 | 24/24 | 24 | 1 (0) |
| `…_base_r2`, `…_cardsampling_r2` (repeats) | same | 2.10 / 1.81 s | 3.60 / 3.17 s | | | 24/24 | 24 | 3 / 1 |

- **Repeat runs.** vLLM's fixed seed makes the repeats reproduce the first runs exactly, so the latency gain is not noise.
- **Why it is faster.** The completion lengths are equal, so decode is not the cause. The likely cause is the sampler path: top_k 100 with an explicit rep-penalty instead of top_k 20 plus the generation-config merge. The gain is consistent across calls.

**Sanity read.** Detection does not change. The reference flags 5 calls, and the model catches none of them before or after the fix.

| call | reference | before | after |
|---|---|---|---|
| easy_hard_1 t16 (mis_prep) | "You just chopped the cheese instead of trashing it!" | "go to the plate stack…" | "pick up the tomato crate… another raw tomato" |
| easy_medium_1 t32 (wrong_order) | "cheese went on before the chopped tomato" | "Now you need raw cheese…" | "Now grab a slice of raw cheese…" |
| easy_hard_2 t94 (forget_next_step) | "you're still at the cheese crate… go to the serving pass" | "Order completed successfully." | "The caprese is being served!" |
| master_hard_4 t408 (mis_prep) | "you just picked up an onion…" | "cook and add the rice" | "move toward getting steaks ready" |

Unflagged-reference calls where the model raised a flag:
- expert_hard_2 t128: before `leave_unattended`, after `wrong_order`. The reference text says "the plate order is now broken", so the after flag is the better match.
- expert_medium_3 t214: before `leave_unattended` (the fryer is smoking; the reference talks about it), after none.
- master_hard_2 t38: before `mis_prep` on a dropped steak, after none.

Net: an equal number of plausible flags either way.

**Adopt** under the audit rule: validity is not lower and latency is not worse; both are better.

## Changes to apply
- **Engine** (`cooksim_bench/controller_eval.py` PROFILES). Replace `minicpmo-schema`:
```python
# MiniCPM-o 4.5 (audit 2026-09-18): the CookBook's vLLM-on-AWQ sampling (quantization/awq/minicpm-o4_5_awq_quantize.md):
# T 0.7 / top_p 0.8 / top_k 100 / repetition_penalty 1.05; thinking off (instruct mode, as the card reports).
# Before: T 0.1 + vLLM-injected top_p 0.95 / top_k 20. 24/24 both; median 2.10 -> 1.81 s, p90 3.61 -> 3.17 s.
"minicpmo-schema": dict(temperature=0.7, top_p=0.8,
                        extra_body={"top_k": 100, "repetition_penalty": 1.05,
                                    "chat_template_kwargs": {"enable_thinking": False}},
                        prefill=None, images_first=True, reply_schema=True),
```
  The plain `minicpmo` profile (no schema) should get the same sampling if it is still used.
- **Serve script** (`pass3_setup/serve_minicpmo.sh`): no change.
  - Optional: add `--generation-config vllm`, so a request that omits a sampling field no longer inherits the Qwen3 generation-config values. The new profile sets every field, so this is only a safeguard.
- **Operational hazard.** `POST /tokenize` with image content on a *serving* vLLM 0.19.1 instance puts the frames into the API-side multimodal cache but not the engine's. The next chat call with the same frames then fails with 500 "Expected a cached item for mm_hash=…", and a restart is needed. Do not call `/tokenize` with images against a server a benchmark is using.

Raw rows: `pass3_setup/results/minicpmo_awq_schema_audit_{base,cardsampling}{,_r2}.json`. Summaries: `pass3_setup/audit_summary.py`.
