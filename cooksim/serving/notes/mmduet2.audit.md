# MMDuet2: official-guidance audit (checklist of AUDIT_2026-09-18.md)

Audited 2026-09-18 while the model was being set up, so the served setup was built to the guidance from the start.

**Sources**
- **[card]** huggingface.co/wangyueqian/MMDuet2, README "Usage".
- **[repo]** github.com/yellow-binary-tree/MMDuet2 @ 7bd91d5 (local copy `src/`). The files cited are:
  - `demo/requirements.txt`
  - `demo/inference.py`, which is identical to `proactive_eval/inference.py`
  - `demo/api_server.py` and `demo/frontend.py`
  - `train/scripts/train.sh`
  - `rl/verl/recipe/proactive/run_scripts/*.sh`
- **[paper]** arXiv 2512.06810 (HTML version), sections §4.1 and §4.3, Fig. 2, §5.1 and §5.3 (Table 7), and
  Appendix A.
- **[data]** huggingface.co/datasets/wangyueqian/MMDuet2-data. The files used are
  `evaluate/ego-frame_input_format.json` (copy in `data/`) and the head of
  `sft/egoexo4d-…-2_sec_per_frame-sft.jsonl`.

The model is not a vLLM or SGLang serving target in any official doc. The card says to use the repo's demo and
evaluation code for proactive use ("we recommend using the demo and evaluation code provided in MMDuet2 Github Repo
to experience its ability of actively generating replies while a video is playing online"). So the "engine" here
is the authors' Python loop, and the shim imports it.

| # | item | ours | official | source | verdict |
|---|---|---|---|---|---|
| 1 | checkpoint & precision | `wangyueqian/MMDuet2` @ 3224235, bf16, no quantization | the one released checkpoint; loaded `torch_dtype=torch.bfloat16`; tensors are stored BF16 (all 825) | [card]; [repo] `demo/inference.py` L48–50 (`from_pretrained(…, torch_dtype=torch.bfloat16, attn_implementation=…)`) | matches |
| 2 | engine & version | the authors' `ProactiveInferenceAPIClient` (`demo/api_server.py`) on their `proactive_eval/qwen2_5_vl` model class; transformers 4.49.0, torch 2.4.0, flash-attn 2.6.3, qwen-vl-utils 0.0.8, Python 3.10 | the same loop and pins; Python 3.10 (`conda create -n mmduet2-infer python=3.10`) | [repo] README "Quick Start", `demo/requirements.txt`, `demo/inference.py` L13 (`from qwen2_5_vl import …`) | matches. One addition: `setuptools`, needed by triton 3.0 inside a uv venv. No change in behaviour. |
| 3 | chat template | the checkpoint's own `chat_template.json` via `processor.apply_chat_template(history, add_generation_prompt=True)`, called by the authors' `add_image` | the same. Format: `system` = proactive prompt; user turn = `<image><image>` + optional text **after** the frames; the assistant's reply or `NO REPLY` after every turn | [paper] Fig. 2; [data] SFT head (`{"role":"user","content":"<image>What checks are done related to the brakes?"}`); [repo] `api_server.py` `add_image` | matches. The rendered prompt for call 0 is in `probe/rendered_call0.txt`; it follows Fig. 2 token for token. **Adaptation (on purpose):** the benchmark's 13k-character system message has no slot in this format, so it becomes the first turn's "question" and the benchmark user text rides on the newest turn. Two alternatives were benched: everything on the last turn (0/24 spoke), and the benchmark prompt in the system slot, which replaces the RL'd system prompt (see the note below). |
| 4 | visual input | 2 frames per user turn; 8 frames → 4 turns; 1 frame per tick (≈1.2 s); frames `smart_resize`d to `max_pixels` 128·28·28, i.e. 504×168 = 108 tokens per frame, as images (not video) | "we use 128 tokens per frame, 2 frames per user turn" (SFT and RL); inference "we use 2 frames per user turn" for EGO/TV/VAD; "recommend … a relatively high-frequency inference frame rate (1 second per frame)"; images, one `{"type":"image"}` per frame | [paper] §4.3 (twice), §5.1, §5.3 / Table 7; [repo] `train/scripts/train.sh` `MAX_PIXELS=$((128*28*28))`; [data] `ego-frame_input_format.json` (2 images per turn, 1 fps frames) | matches the training budget. The inference scripts do not pin a size: the demo frontend resizes to a ≤640 px side and the eval uses pip qwen-vl-utils defaults. Both were benched (≈207 and 518 tokens/frame) and gave the same 0/24 at 1.20 s and 1.77 s, so the training budget (fastest, in-distribution) is kept. |
| 5 | sampling | greedy (`do_sample=False`), `max_new_tokens` 512, the checkpoint's `repetition_penalty` 1.05 from `generation_config.json` | `do_sample: bool = False`, `temperature 1.0`, `top_k 40` (only used if sampling), `max_new_tokens=512` | [repo] `demo/inference.py` `ProactiveTestArguments` L33–35, `generate(…max_new_tokens=512…)` L179 | matches. The code comment says sampling "greatly reduces the probability of the model remaining silent". That path (T 1.0 / top_k 40) was benched: 1/24 spoke, and the one reply was wrong ("Do not serve yet…" on a complete plate). Not adopted, because greedy is the default. |
| 6 | thinking / reasoning | none | none. It is a non-thinking Qwen2.5-VL-3B; the reply is plain text or `NO REPLY` | [paper] §4.1 | matches (n/a) |
| 7 | efficiency | incremental KV reuse inside a request (`past_key_values`, with only the new turn's pixels through the vision tower); flash-attention-2; bf16; token drop off (`drop_method='none'`) | the same. The paper measures wall time on an H100 against MMDuet, and App. A names a cheaper decision format (predict `<vis_start>` vs `<im_end>`) as future work that is **not released** | [repo] `api_server.py` `generate(… past_key_values=…, drop_method='none' …)`; `modeling_qwen2_5_vl_DTD.py` `prepare_inputs_for_generation` L2375–2390; [paper] §5.1 "Inference Speed", App. A | matches. Measured 1.08–1.19 s median per stateless call (4 turns, ~5.1k tokens), within the 2-tick poll. The released code has no other speed feature (no drafter, no prefix cache across requests). A persistent per-episode session would be the real gain; see SETUP.md. |
| 8 | structured output | none; the shim wraps the plain-text reply into the schema (`NO REPLY` → `text ""`, `flag null`) | no structured or JSON output in any official material; the output space is free text or `NO REPLY` | [paper] §4.1; [repo] `evaluate.py` `NO_REPLY_STR = 'NO REPLY'` | deviates on purpose: the model cannot be asked for the schema without replacing its system prompt. Guided JSON would force it to speak in a format it was never trained on, and would erase the `NO REPLY` decision that is the point of the model. |

**Note on the non-official system-prompt variant.**
- `mmduet2_system=bench` puts the benchmark system prompt in the system slot. The eval loader permits that
  ("using system prompt in data instead of default system prompt"), but it drops the RL'd proactive prompt.
- The model then tries to write the JSON. 16 of 24 replies were a reworded silence (`NO ACTION`); 5 were broken
  JSON that repeated to the 512-token cap.
- Median latency was 5.6 s and p90 22.5 s. Only 1 of the 24 was an error-relevant line, and it came too late.
- It is worse on latency, is not the official protocol, and detects nothing, so it was not adopted.

## Before / after
Nothing was changed after the first bench, because the setup was built to the guidance from the start. The
baseline was re-run under the `_audit` suffix alongside every alternative:

| tag | median | p90 | schema ok | spoke |
|---|---|---|---|---|
| `mmduet2_official_t128` (adopted) | 1.19 s | 1.39 s | 24/24 | 0/24 |
| `mmduet2_official_t128_audit` (same, re-run) | 1.08 s | 1.12 s | 24/24 | 0/24 |
| `mmduet2_native_px_audit` | 1.77 s | 1.78 s | 24/24 | 0/24 |
| `mmduet2_demo640_audit` | 1.20 s | 1.24 s | 24/24 | 0/24 |
| `mmduet2_textlast_audit` | 1.09 s | 1.12 s | 24/24 | 0/24 |
| `mmduet2_sample_audit` | 1.10 s | 1.13 s | 24/24 | 1/24 |
| `mmduet2_mustreply_probe` | 1.06 s | 1.09 s | 24/24 | 0/24 |
| `mmduet2_benchsys_audit` (not official) | 5.57 s | 22.5 s | 24/24 | 24/24 (16 = "NO ACTION") |

**Adopted:** official system prompt, rulebook as the first-turn question, user text on the newest turn, 128 tokens
per frame, greedy decoding. This is the serve-script default. No engine change is needed beyond the optional
named profile in SETUP.md.
