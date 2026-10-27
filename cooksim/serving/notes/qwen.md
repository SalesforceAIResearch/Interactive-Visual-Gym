# SETUP — Qwen3.8-27B / Qwen3.5-9B / Qwen3.5-2B (`qwen`), reproducible commands

Host: 4x A100-SXM4-80GB (ours: GPU 3 — `export CUDA_VISIBLE_DEVICES=3` before ANY GPU process),
driver 570.124.06 (CUDA 12.8, no nvcc), system python 3.12.12 (/opt/conda/bin/python3).
No ~/.huggingface/token (not needed; all three repos are public, Apache-2.0).

    ROOT=$ROOT
    export HF_HOME=$ROOT/hf_cache/qwen
    mkdir -p $ROOT/simulation_home/local_models/qwen/logs $HF_HOME $ROOT/envs

## 1. Env (one venv for everything)

    /opt/conda/bin/python3 -m venv $ROOT/envs/qwen
    $ROOT/envs/qwen/bin/pip install "vllm==0.19.1" openai pillow requests
    # -> vllm 0.19.1, torch 2.10.0+cu128, transformers 5.15.0, xgrammar 0.2.3, flashinfer-python 0.6.6

Why vllm 0.19.1 (PyPI release, no nightly): the Qwen3.8-27B vLLM recipe requires vllm >= 0.17.0 /
transformers >= 5.8.0. 0.19.1 is the newest PyPI vLLM whose torch (2.10.0+cu128) runs on driver 570
(>= 0.20 pins torch cu130 -> needs driver >= 580; the nightly cu129 wheels also work on this driver but
are not needed — unlike Muse Glimmer, no Qwen3.x parser/model code is missing from 0.19.1).
Verified before serving: vllm 0.19.1's model registry contains `Qwen3_5ForConditionalGeneration`, and all
three checkpoints — including Qwen3.8-27B — declare exactly that architecture (`model_type: qwen3_5`) in
config.json, i.e. Qwen3.8-27B reuses the 3.5 hybrid-attention architecture and is servable by the release wheel.

## 2. Model downloads (~75 GB total; the FS was under heavy load — downloads are I/O-bound)

    /opt/conda/bin/hf download Qwen/Qwen3.8-27B      # 52 GB bf16
    /opt/conda/bin/hf download Qwen/Qwen3.5-9B       # 19 GB bf16
    /opt/conda/bin/hf download Qwen/Qwen3.5-2B       # 4.3 GB bf16

NOT downloaded: `Qwen/Qwen3.8-27B-FP8` (sm_80 has no FP8 — useless here), GPTQ-Int4 variants (bf16 of every
size fits in 80 GB). Course note: three Qwen3-VL checkpoints (~83 GB) were downloaded first under the initial
"Qwen3-VL is newest" assumption, then deleted after the correction to Qwen3.8 (`rm -rf $HF_HOME/hub/models--Qwen--Qwen3-VL-*`).

## 3. Serving (one size at a time, GPU 3, port 18300)

    cd $ROOT/simulation_home/local_models/qwen
    bash serve.sh 2b|9b|27b          # logs/serve_<size>.txt; stop with: kill $(cat logs/serve.pid)

serve.sh: `vllm serve <repo> --served-model-name qwen-<size> --host 127.0.0.1 --port 18300
--tensor-parallel-size 1 --dtype auto --max-model-len 32768 --max-num-seqs 4
--limit-mm-per-prompt '{"image": 8}' --gpu-memory-utilization 0.90 --reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking": false}'`
(thinking OFF server-wide because cooksim_bench/controller_eval.py's default profile sends no chat_template_kwargs;
`THINK=1 bash serve.sh <size>` re-enables it. `--reasoning-parser qwen3` keeps any `<think>` block out of
`message.content` if thinking is ever on.)

## 4. Smoke test

    $ROOT/envs/qwen/bin/python smoke.py --model qwen-<size> --out results_<size>.json

## 5. Controller sweep (server running; render server for agent+++ mode)

    cd $ROOT/simulation_home/cook-bench-engine
    conda activate cooksim
    $PY cooksim_bench/controller_eval.py --backend openai --base-url http://127.0.0.1:18300/v1 \
        --model qwen-<size> --mode text_state_map --out-dir tmp/ctrl_eval/qwen --tag qwen-<size> --max-tokens 600
    $PY cooksim_bench/controller_eval.py --backend openai --base-url http://127.0.0.1:18300/v1 \
        --model qwen-<size> --mode agent+++ --out-dir tmp/ctrl_eval/qwen --tag qwen-<size> --max-tokens 600

The render server (needed by agent+++) was down at 17773; brought up with the engine's documented command
(`. ./gpu_env.sh && $PY -m cooksim.server.app --port 17773`, CUDA_VISIBLE_DEVICES=3 so our chromium renders
on our own GPU) and killed by pid after the last sweep.
