# SETUP — Muse Glimmer 30B on the L40S (2026-08-18)

Everything lives under
`$LM/museglimmer/` (this dir),
env under `$ROOT/envs/museglimmer`,
weights under `HF_HOME=$ROOT/hf_cache/museglimmer`.
Host: Python 3.12.12, driver 570.124.06 (CUDA 12.8 driver), no `nvcc`, no `docker`, `uv` at ~/.local/bin/uv.

## 1. Environment (uv venv, own dir)

```bash
cd $ROOT/envs
uv venv --python 3.12 museglimmer                    # 0.2 s
export UV_CACHE_DIR=$ROOT/hf_cache/museglimmer/.uv_cache
```

### Attempt 1 — vLLM nightly, default index, cu128 torch  → FAILED (resolver)
```bash
uv pip install --python ./museglimmer/bin/python --prerelease=allow -U vllm --torch-backend=cu128 \
    --extra-index-url https://wheels.vllm.ai/nightly
# × No solution found when resolving dependencies:
# ╰─▶ Because there is no version of torch==2.13.0 and vllm==0.26.1rc1.dev911+g9842d7014 depends on torch==2.13.0 ...
```
(torch 2.13.0 exists only as cu126 / cu129 / cu130 wheels; there is no cu128 build.)

### Attempt 2 — vLLM nightly, default index, cu129 torch  → installed but does not import
```bash
uv pip install --python ./museglimmer/bin/python --prerelease=allow -U vllm --torch-backend=cu129 \
    --extra-index-url https://wheels.vllm.ai/nightly          # 2m05s, vllm==0.26.1rc1.dev911+g9842d7014
python -c "import vllm.model_executor.models.muse_glimmer"
# ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```
Root cause: the default nightly variant is now built for **CUDA 13.0** (`vllm/envs.py: VLLM_MAIN_CUDA_VERSION="13.0"`;
`ldd vllm/_C_stable_libtorch.so → libcudart.so.13 => not found`) which needs driver ≥ 580. Our driver is 570.

### Attempt 3 — vLLM nightly **cu129 variant**  → OK (this is the working env)
```bash
uv pip install --python ./museglimmer/bin/python --prerelease=allow -U \
    "vllm==0.27.2rc1.dev193+gaa9903490.cu129" --torch-backend=cu129 \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129        # 36 s (torch already cached)
```
Result (`python -c "import importlib.metadata as m; ..."`):
```
vllm 0.27.2rc1.dev193+gaa9903490.cu129   (vLLM main commit aa99034, 2026-08-18; Muse Glimmer support merged 2026-08-14, PR #51655)
torch 2.13.0+cu129     torchvision 0.28.0+cu129     triton 3.7.1
transformers 5.15.0    compressed-tensors 0.17.0    xgrammar 0.2.6rc1    llguidance 1.7.6
flashinfer-python 0.6.16.post3   huggingface-hub 1.28.0   openai 3.2.0   pillow 12.3.0   numpy 2.3.5
```
Verification: `torch.cuda.is_available()==True` on the L40S (8,9); `vllm.model_executor.models.muse_glimmer` imports;
`ReasoningParserManager.get_reasoning_parser('muse_glimmer')` and `ToolParserManager.get_tool_parser('muse_glimmer')`
resolve; `transformers.MuseGlimmerForConditionalGeneration` exists.

Why nightly at all: no released wheel (PyPI latest 0.27.1, 2026-08-11) contains `vllm/model_executor/models/muse_glimmer.py`
(checked raw.githubusercontent for tags v0.27.0/v0.27.1 → 404; main → 200). The recipe's supported path is the docker
image `vllm/vllm-openai:muse-glimmer`, unavailable here (no docker).

## 2. Weights (no HF token required — repos are ungated; nothing was needed from ~/.huggingface/token)

```bash
export HF_HOME=$ROOT/hf_cache/museglimmer
hf download RedHatAI/Muse-Glimmer-30B-FP8-block            # 34.4 GB, ~3 min
#  -> hub/models--RedHatAI--Muse-Glimmer-30B-FP8-block/snapshots/8ed2e29141d4fef439b9a0e15e0a2678bc190a82
hf download RedHatAI/Muse-Glimmer-30B-INT4                 # 22.2 GB (latency comparison)
#  -> hub/models--RedHatAI--Muse-Glimmer-30B-INT4/snapshots/f5b410ce4234fad70eef8be99b4680ee4e30b418
hf download meta-models/Muse-Glimmer-30B-assistant         # 5.11 GB DFlash drafter (speculative decoding)
#  -> hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/e8192f3a8f617f74be2ce220360c89ef4789f39f
```
Not downloaded: the 59.6 GB BF16 base (does not fit 46 GB), NVFP4 builds (Blackwell-only kernels), GGUF (llama.cpp path).

## 3. Serving (see serve.sh)

```bash
cd $LM/museglimmer
nvidia-smi --query-gpu=memory.used --format=csv          # must be ~1 MiB
flock /tmp/gpu0.lock -c "bash serve.sh > logs/serve_fp8.log 2>&1"                                  # FP8-block (default)
flock /tmp/gpu0.lock -c "bash serve.sh RedHatAI/Muse-Glimmer-30B-INT4 > logs/serve_int4.log 2>&1"  # INT4
EXTRA='--speculative-config {"method":"dflash","model":"meta-models/Muse-Glimmer-30B-assistant","num_speculative_tokens":15}' \
  flock /tmp/gpu0.lock -c "bash serve.sh RedHatAI/Muse-Glimmer-30B-INT4 > logs/serve_int4_dflash.log 2>&1"
# ready when: curl -s http://127.0.0.1:18200/v1/models   -> {"data":[{"id":"museglimmer",...}]}
# stop: kill <APIServer pid>   (pid printed in the log as "(APIServer pid=NNNN)"; the EngineCore child exits with it)
```

### Serve attempt 1 (FP8-block) → engine crashed at the warm-up sampling step
```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
  (flashinfer/jit/cpp_ext.py get_cuda_path <- flashinfer.sampling.top_k_top_p_sampling_from_logits <- vllm topk_topp_sampler)
RuntimeError: Engine core initialization failed. See root cause above.
```
Cause: vLLM defaults to the FlashInfer top-k/top-p sampler, whose kernel is JIT-compiled at first use and needs nvcc.
Fix: `export VLLM_USE_FLASHINFER_SAMPLER=0` in serve.sh (torch-native sampler; log confirms
"FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0"). Log kept as
`logs/serve_fp8_attempt1_nvcc_error.log`.

### Serve attempt 2 (FP8-block) → OK
Key log lines: `Selected TritonFp8BlockScaledMMKernel for CompressedTensorsW8A8Fp8` (Ada has no DeepGEMM/cutlass block-FP8
path); `Using FLASH_ATTN attention backend` (FA2); `Model loading took 32.06 GiB memory` (weights load in 9–39 s);
`torch.compile took 85.96 s`; `GPU KV cache size: 318,401 tokens, Maximum concurrency for 32,768 tokens per request: 9.72x`;
`Free memory on device (43.98/44.4 GiB) on startup ... Actual usage is 32.22 GiB for weights`. Cold start to `/v1/models`
≈ 3.5–4 min (compile + CUDA-graph capture); warm restarts reuse the torch.compile cache (~2 min).
Harmless warnings: deep_gemm `_find_cuda_home ... AssertionError` (DeepGEMM unused on sm_89), `cuda.cudart deprecated`.

### Serve attempt 3 (INT4) → OK
`Using MarlinLinearKernel for CompressedTensorsWNA16`; `Model loading took 20.71 GiB`; `torch.compile took 64 s`;
`GPU KV cache size: 899,109 tokens (27.4x at 32k)`; ready ~2.5 min after start.

### Serve attempt 4 (INT4 + DFlash drafter) → OK  (current default of serve.sh)
Flag: `--speculative-config '{"method":"dflash","model":"meta-models/Muse-Glimmer-30B-assistant","num_speculative_tokens":15}'`.
Log: `Overriding draft model max model len from 131072 to 32768`; `max_num_scheduled_tokens is set to 2048 based on the
speculative decoding settings` (warning, harmless for batch-1); `Draft model DFlashQwen3ForCausalLM does not support
external multimodal embeddings. Embeddings from the target model will not be passed to the drafter` (warning — the drafter
still works on image prompts, mean acceptance length 3.1–3.7); `Model loading took 25.6 GiB`; `GPU KV cache size:
449,335 tokens (13.7x at 32k)`; `Capturing dflash CUDA graphs`; ready ~2.5 min after start.

### Chat-template gotcha found while testing
The RedHatAI repos' `chat_template.jinja` (7,167 B, older "Onyx" revision) always appends `Reasoning strength: high.`
(or the `reasoning_strength` kwarg) after the system prompt; the current meta-models template (docs_raw/base_chat_template.jinja,
9,992 B) honours a line already present. Consequence: the FP8 and first INT4 runs that wrote `Reasoning strength: low.`
into the system prompt were actually rendered with a trailing `Reasoning strength: high.`. Fix in smoke.py: pass
`extra_body={"chat_template_kwargs": {"reasoning_strength": "low"}}` (works with either template). serve.sh can also
serve the meta-models template with `BASE_TEMPLATE=1` (untested path). Rendering check:
`AutoProcessor.from_pretrained("RedHatAI/Muse-Glimmer-30B-INT4").tokenizer.apply_chat_template(...)`.

### Things that did not work (verbatim last errors)
* per-request `extra_body={"mm_processor_kwargs": {"max_image_tokens": 256}}` →
  `BadRequestError: Error code: 400 - {'error': {'message': 'Unexpected keyword argument max_image_tokens.', 'type': 'BadRequestError', 'param': None, 'code': 400}}`
* default nightly wheel → `ImportError: libcudart.so.13: cannot open shared object file: No such file or directory` (CUDA-13 build vs driver 570)
* first serve without `VLLM_USE_FLASHINFER_SAMPLER=0` → `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
* uv resolver with `--torch-backend=cu128` → `Because there is no version of torch==2.13.0 and vllm==0.26.1rc1.dev911+g9842d7014 depends on torch==2.13.0 ...`

## 4. Smoke test
```bash
$ROOT/envs/museglimmer/bin/python smoke.py --n 5 --cases img8 --strength low
$ROOT/envs/museglimmer/bin/python smoke.py --n 5 --cases img8_nothink,img8,text2k,img8_json,img8_nothink_half,text2k_nothink --strength low --out logs/smoke_results_int4_dflash.json
# results appended to logs/smoke_results.json; console logs in logs/smoke_*.log ; summary in RESULTS.md
```
