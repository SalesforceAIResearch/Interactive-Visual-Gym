# SETUP — MiniCPM-o 4.5 (`minicpmo`) on this box, reproducible commands

Host facts: NVIDIA L40S 46 GB, driver 570.124.06 (CUDA 12.8; no `nvcc`), system python 3.12.12
(`/opt/conda/bin/python3`; base env has torch 2.9.1+cu128 / vllm 0.15.1 — NOT used, vllm 0.15.1 predates o4.5).
No `~/.huggingface/token` (not needed; the repo is public).

    export HF_HOME=$ROOT/hf_cache/minicpmo
    ROOT=$ROOT
    mkdir -p $ROOT/simulation_home/local_models/minicpmo $HF_HOME $ROOT/envs

## 1. Model download (bf16 original ~19 GB + AWQ int4 ~8 GB)

    /opt/conda/bin/hf download openbmb/MiniCPM-o-4_5 --exclude "assets/audio_cases/*" "*.mp4"
    /opt/conda/bin/hf download openbmb/MiniCPM-o-4_5-awq --exclude "assets/*"      # W4 AWQ, LLM only; vpm/resampler bf16
    # -> $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582  (54 s)
    # assets/token2wav/* is kept (needed by init_tts()/as_duplex()); *.mp4 demo clips are skipped.

## 2. Env A — transformers path (`envs/minicpmo`): model.chat + streaming/duplex APIs

    /opt/conda/bin/python3 -m venv $ROOT/envs/minicpmo
    $ROOT/envs/minicpmo/bin/pip install --upgrade pip
    $ROOT/envs/minicpmo/bin/pip install "transformers==4.51.0" accelerate "torch==2.8.0" "torchaudio==2.8.0" \
        "torchvision==0.23.0" "minicpmo-utils[all]>=1.0.5" pillow numpy fastapi "uvicorn[standard]" openai
    # FIX (hit on first run): librosa 0.9 -> "No module named 'pkg_resources'" -> transformers refuses to load
    # the remote code ("This modeling file requires the following packages that were not found: librosa").
    $ROOT/envs/minicpmo/bin/pip install "setuptools<80"
    $ROOT/envs/minicpmo/bin/pip install pypdf          # only used to read the technical report
    # optional flash-attn (prebuilt wheel matching torch 2.8 / cu12 / cp312 / cxx11abi=TRUE):
    $ROOT/envs/minicpmo/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

Result: torch 2.8.0+cu128, transformers 4.51.0, minicpmo-utils 1.0.6 (with stepaudio2), accelerate 1.14.0,
pillow 10.4.0, numpy 2.5.2, flash_attn 2.8.3 (optional; `--attn flash_attention_2` in the smoke scripts).

## 3. Env B — vLLM path (`envs/minicpmo-vllm`): OpenAI-compatible server

    /opt/conda/bin/python3 -m venv $ROOT/envs/minicpmo-vllm
    $ROOT/envs/minicpmo-vllm/bin/pip install --upgrade pip
    $ROOT/envs/minicpmo-vllm/bin/pip install "vllm==0.19.1" "transformers==4.57.6" openai pillow

Why 0.19.1: MiniCPM-o 4.5 needs vllm >= 0.16.0; vllm >= 0.20 pins torch >= 2.11 whose PyPI wheels are cu130
(driver >= 580 required; this driver is 570). 0.19.1 pins torch 2.10.0 (+cu128 default wheel) — works.
transformers pinned to 4.57.6 (< 5) because the remote-code processor was written for 4.x.
Result: vllm 0.19.1, torch 2.10.0+cu128, transformers 4.57.6, flashinfer-python 0.6.6, xgrammar 0.2.3.

## 4. Smoke tests (all GPU work under the lock; the lock blocks while the other agent's server runs)

    cd $ROOT/simulation_home/local_models/minicpmo
    ENV=$ROOT/envs/minicpmo
    $ENV/bin/python make_frames.py                                            # 8 x 800x580 PNGs -> frames/
    HF_HUB_OFFLINE=1 flock /tmp/gpu0.lock -c "$ENV/bin/python smoke_stateless.py"  > logs_smoke_stateless_sdpa.txt 2>&1
    HF_HUB_OFFLINE=1 flock /tmp/gpu0.lock -c "$ENV/bin/python smoke_streaming.py --duplex" > logs_smoke_streaming.txt 2>&1
    HF_HUB_OFFLINE=1 flock /tmp/gpu0.lock -c "$ENV/bin/python smoke_stateless.py --attn flash_attention_2 --out results_stateless_fa2.json" > logs_smoke_stateless_fa2.txt 2>&1

## 5. Server (vLLM, 127.0.0.1:18100) and client check

    # start (holds the lock while it runs; log in logs_serve_vllm.txt)
    nohup flock /tmp/gpu0.lock -c 'bash serve.sh' > logs_serve_vllm.txt 2>&1 &
    # first attempt failed: "vllm: error: unrecognized arguments: --disable-log-requests" (flag removed in 0.19) -> dropped
    # wait for "Application startup complete", then:
    curl -s http://127.0.0.1:18100/v1/models
    $ROOT/envs/minicpmo-vllm/bin/python smoke_vllm_client.py            # results_vllm.json (identical repeats -> prefix-cache hits)
    $ROOT/envs/minicpmo-vllm/bin/python smoke_vllm_client.py --vary --out results_vllm_vary.json   # per-call varied images
    # AWQ int4 server (recommended):   VARIANT=awq nohup flock /tmp/gpu0.lock -c 'bash serve.sh' > logs_serve_vllm_awq.txt 2>&1 &
    #   -> results_vllm_awq_vary.json ; HD variant: VARIANT=awq MAX_SLICE_NUMS=9 ... -> results_vllm_awq_slice9_vary.json
    # stop: kill the `vllm serve` python pid you started (ps aux | grep "vllm serve"), never pkill by pattern
