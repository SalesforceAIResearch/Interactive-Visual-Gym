# Serving the open-weight assistants

The eleven open-weight assistants of the published table are served locally as OpenAI-compatible chat servers on
`127.0.0.1:<port>`, one model per GPU and per port, and reached by the harness exactly like any other
OpenAI-compatible endpoint. This page is the practical setup per model, taken from the headers of the
`serve_*.sh` scripts in this directory. For which seat uses which script, profile and port, see
[docs/ASSISTANTS.md](../docs/ASSISTANTS.md) section 2; for the rules every local model shares, section 6.

```bash
bash serving/serve.sh <tag> [port]            # e.g.  bash serving/serve.sh qwen3.8-27b
```

`serve.sh` looks the tag up in `vhhome_bench/roster.py`, takes the script name and port from the entry, and runs
`serve_<model>.sh` with `PORT` set. `PY` names the Python used to read the roster (default `python3`); `LOGDIR` is
where the pid file goes (default `<checkout>/runs/serving`). After the server answers `GET /v1/models`, the model is
run with `vhhome-benchmark --assistant <tag> --engines <one simulator port>`.

## Layout: `VHHOME_MODELS`

Every script reads the same root:

```
$VHHOME_MODELS/                 default: $HOME   (the older name ROOT is still honoured)
  envs/<name>/                  one Python environment per model (a venv with its own torch, transformers, vLLM)
  hf_cache/<name>/              that model's HF_HOME: the weights, downloaded once
```

The scripts set `HF_HUB_OFFLINE=1` (and, for the shims, `TRANSFORMERS_OFFLINE=1`), so the weights must be in
`hf_cache/<name>` before the first start: download them into that cache with the Hugging Face tooling of your choice
(`HF_HOME=$VHHOME_MODELS/hf_cache/<name>`), with the environment's `HF_HUB_OFFLINE` unset. Two scripts
(`serve_minicpmo.sh`, `serve_internvideo3.sh`) resolve the local snapshot directory under
`hf_cache/<name>/hub/models--<org>--<repo>/snapshots/*/` and fail at start if it is missing.

Never install a model's stack into the `vhhome` environment: the models pin incompatible versions of torch and
transformers, and the benchmark itself needs none of them.

Common knobs read by every script: `CUDA_VISIBLE_DEVICES` (each script has its own default GPU index; set it
explicitly), `PORT` (set by `serve.sh`; the script's own default is the roster port), `HF_HOME`, `LOGDIR`
(default `${TMPDIR:-/tmp}/vhhome_serving` when a script is run directly), and for the vLLM scripts `MAXLEN`
(context length), `MAX_SEQS` (concurrent sequences, default 2), `GPU_UTIL` (fraction of the card reserved; the
weights need headroom for the KV cache) and `EXTRA` (extra vLLM flags). The vLLM scripts export
`VLLM_USE_FLASHINFER_SAMPLER=0` so that no CUDA JIT compiler is needed on the serving host.

## The image cap: 12 per request

An assistant turn carries 10 images (8 first-person frames and 2 maps). Every vLLM script therefore passes
`--limit-mm-per-prompt '{"image": 12, ...}'`. This is the one setting that differs from the sibling kitchen
benchmark, CookSim, whose turns carry 8 images; a server started with the lower cap answers HTTP 400
("At most 8 image(s) may be provided in one prompt") on every call. The shims take the images as the request
carries them.

## Per model

What each script expects: the environment name under `envs/`, the cache under `hf_cache/`, the server, the weights,
the flags that matter, and the port. Versions are those the script headers name; where a header names none, pin the
version your environment was built with and record it.

### `serve_qwen.sh` — Qwen3.8-27B (port 18300, served name `qwen3.8-27b`)

- Environment `envs/qwen`, cache `hf_cache/qwen`, vLLM (`$VHHOME_MODELS/envs/qwen/bin/vllm`; the header does not
  name a vLLM version, pin the one your environment was built with). Weights `Qwen/Qwen3.8-27B`, bf16.
- Flags: `--chat-template-content-format openai` (required: without it vLLM detects the template as "string"
  format and moves every image in front of the text), `--reasoning-parser qwen3`, `--enable-prefix-caching`,
  `--max-model-len 32768`, `--limit-mm-per-prompt '{"image": 12}'`, prefill kernel pinned with
  `--additional-config {"gdn_prefill_backend": "triton"}` (`GDN_BACKEND`).
- Speculative decoding: built-in MTP at depth 5 (`SPEC=1`, `MTP_K=5`); `SPEC=0` turns it off.
- Thinking off server-wide via `--default-chat-template-kwargs {"enable_thinking": false}`; `THINK=1` leaves it
  on (a request can still turn it on with `chat_template_kwargs`).
- Pid file `$LOGDIR/serve_qwen38_27b.pid`.

### `serve_molmo2.sh` — Molmo2-8B (port 18400, served name `molmo2-8b`)

- Environment `envs/molmo2`, cache `hf_cache/molmo2`, vLLM 0.19.1. Weights `allenai/Molmo2-8B`, bf16,
  `--trust-remote-code`.
- Flags: `--chat-template serving/molmo2_chat_template_system.jinja` (required: the shipped template rejects a
  system message, and the benchmark always sends one; the shipped file prepends the system text to the first user
  turn and is byte-identical to the original otherwise), `--max-model-len 32768`, `--enable-prefix-caching`,
  `--limit-mm-per-prompt '{"image": 12, "video": 0}'`. vLLM's default batched-token setting is kept on purpose (the
  demo's larger value measured slower on the serving card). Sampling is greedy, set by the request profile.
- Pid file `$LOGDIR/serve_molmo2.pid`.

### `serve_internvideo3.sh` — InternVideo3-8B-Instruct (port 18700, served name `internvideo3-8b`)

- Environment `envs/internvideo3-8b`, cache `hf_cache/internvideo3-8b`, a **transformers shim**
  (`shims/internvideo3/shim_server.py`; FastAPI), not vLLM: vLLM 0.19.1 has no `internvideo3` architecture. The
  header mentions transformers 4.57.3. Weights `yanziang/InternVideo3-8B-Instruct`, bf16, trust-remote-code; the
  script resolves the local snapshot directory and exports it as `MODEL`.
- Serves `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`; one request at a time; `response_format`
  `json_schema` is enforced with xgrammar, `json_object` accepts any JSON object.
- `FRAMES=video` (default): the images go in as one clip, `TICK_SECONDS=1.2` apart; `FRAMES=images` sends them as
  separate images. `ATTN=flash_attention_2` by default (the authors' evaluation scripts use it; `sdpa` is the card's
  quickstart). `MAX_NEW_CAP` caps generation (4000).
- Pid file `$LOGDIR/serve_internvideo3.pid`.

### `serve_glm46v_flash.sh` — GLM-4.6V-Flash (port 18500, served name `glm-4.6v-flash`)

- Environment `envs/glm` (`VENV` overrides), cache `hf_cache/glm`. The header names the environment's versions:
  vLLM 0.19.1, torch 2.10.0+cu128, transformers 5.17.0. Weights `zai-org/GLM-4.6V-Flash`, bf16; native vLLM
  support, no remote code.
- Flags: `--reasoning-parser glm45`, `--max-model-len 32768`, `--enable-prefix-caching`,
  `--limit-mm-per-prompt '{"image": 12, "video": 0}'`.
- Pid file `$LOGDIR/serve_glm.pid`.

### `serve_kimi_vl.sh` — Kimi-VL-A3B-Thinking-2506 (port 18650, served name `kimi-vl-thinking`)

- Environment `envs/kimi-vl-thinking`, cache `hf_cache/kimi-vl-thinking`, vLLM 0.19.1 (native
  `KimiVLForConditionalGeneration`). Weights `moonshotai/Kimi-VL-A3B-Thinking-2506`, bf16, `--trust-remote-code`.
- Flags: `--reasoning-parser-plugin shims/kimi-vl/kimi_vl_reasoning_parser.py --reasoning-parser kimi_vl` (the
  model's `◁think▷ ... ◁/think▷` markers are several tokens each and vLLM 0.19.1 has no parser for them; the plugin
  moves the thinking into `reasoning_content` and applies the reply schema only after the closing marker),
  `--reasoning-config '{"reasoning_start_str": "◁think▷", "reasoning_end_str": "◁/think▷"}'` (enables the
  per-request `thinking_token_budget` the profile sets), `--max-num-batched-tokens 32768` (`MNBT`; the authors' own
  serve flag), `--max-model-len 32768`, `--limit-mm-per-prompt '{"image": 12, "video": 0}'`.
- Pid file `$LOGDIR/serve_kimi_vl_thinking.pid`.

### `serve_museglimmer.sh` — Muse Glimmer 30B (port 18200, served name `museglimmer`)

- Environment `envs/museglimmer`, cache `hf_cache/museglimmer`, vLLM (the header names no version; it must provide
  the `muse_glimmer` reasoning parser, the `dflash` speculative method and `--structured-outputs-config`).
- Weights `RedHatAI/Muse-Glimmer-30B-INT4` (W4A16 GPTQ) by default (`MODEL`), with the DFlash speculative drafter
  `meta-models/Muse-Glimmer-30B-assistant` (`SPEC=1`, 15 speculative tokens). `MODEL=meta-models/Muse-Glimmer-30B
  GPU_UTIL=0.92` serves the official BF16 checkpoint, which fits an 80 GB card but measured slower with no detection
  gain. The FP8 variant needs a card with FP8 support.
- Flags: `--reasoning-parser muse_glimmer`, `--generation-config auto`,
  `--structured-outputs-config {"enable_in_reasoning": true}` (required: the no-think profile pre-fills the answer
  channel, so a schema applied only after reasoning would never apply), `--max-model-len 32768`,
  `--limit-mm-per-prompt '{"image": 12}'`.
- Pid file `$LOGDIR/serve_museglimmer.pid`.

### `serve_minicpmo.sh` — MiniCPM-o 4.5 (port 18100, served name `minicpmo`)

- Environment `envs/minicpmo-vllm`, cache `hf_cache/minicpmo`, vLLM 0.19.1. Weights `openbmb/MiniCPM-o-4_5-awq`
  (`VARIANT=awq`, the default) or `openbmb/MiniCPM-o-4_5` (`VARIANT=bf16`); the script resolves the local snapshot
  directory. `--trust-remote-code`.
- Flags: `--mm-processor-kwargs {"max_slice_nums": 1}` (`MAX_SLICE_NUMS`; one slice per image),
  `--max-model-len 16384`, `GPU_UTIL` default 0.85, `--limit-mm-per-prompt '{"image": 12, "video": 0, "audio": 0}'`.
- Pid file `$LOGDIR/serve_minicpmo.pid`.

### `serve_qwen3_omni.sh` — Qwen3-Omni-30B-A3B-Instruct (port 18800, served name `qwen3-omni-30b-a3b`)

- Environment `envs/qwen3-omni-30b-a3b`, cache `hf_cache/qwen3-omni-30b-a3b`, vLLM 0.19.1. Weights
  `Qwen/Qwen3-Omni-30B-A3B-Instruct`, bf16.
- Text output only: plain vLLM serves the thinker alone and skips the talker and code2wav weights at load, so the
  talker-only shards need not be downloaded. The Instruct model has no thinking mode, so no parser or template
  arguments are needed.
- Flags: `--max-model-len 32768`, `GPU_UTIL` default 0.92,
  `--limit-mm-per-prompt '{"image": 12, "video": 0, "audio": 0}'`.
- Pid file `$LOGDIR/serve_qwen3_omni.pid`.

### `serve_mmduet2.sh` — MMDuet2 (port 18900, served name `mmduet2`)

- Environment `envs/mmduet2`, cache `hf_cache/mmduet2`, a **shim** (`shims/mmduet2/shim_server.py`) around the
  authors' own demo loop. The header names the authors' environment: transformers 4.49.0, torch 2.4.0,
  flash-attn 2.6.3, qwen-vl-utils 0.0.8 (their `demo/requirements.txt`). Weights `wangyueqian/MMDuet2`
  (Qwen2.5-VL-3B post-trained for proactive replies), bf16.
- Upstream code: the MMDuet2 authors' repository, cloned into `shims/mmduet2/src` (git-ignored, not vendored).
  The shim imports `src/demo/api_server.py` (`ProactiveInferenceAPIClient`) and `src/proactive_eval/qwen2_5_vl`
  from it. The header names no commit: pin the commit your clone was tested with.
- Knobs: `MMDUET2_SYSTEM=official|bench` (the model's own proactive system prompt, or the benchmark's system
  message, in the system turn; default `official`), `MMDUET2_MAX_PIXELS` (per-frame resize before the model, default
  100352, i.e. 128 tokens per frame; 0 = no pre-resize), `MMDUET2_FRAME_INTERVAL` (1.2).
- Output: the last turn's text; "NO REPLY" becomes empty text; plain text is wrapped into the reply object
  (`text` set, `flag` null). The per-turn outputs are returned in an extra `x_mmduet2` field and logged to
  `shims/mmduet2/logs/requests.jsonl`.
- Pid file `$LOGDIR/serve_mmduet2.pid`.

### `serve_proact_vl.sh` — Proact-VL (port 18950, served name `proact-vl`)

- Environment `envs/proact-vl`, cache `hf_cache/proact-vl`, a **shim** (`shims/proact-vl/shim_server.py`) around
  the authors' own inference code; the header names transformers 4.57.1. Weights: LiveCC-7B-Base (Qwen2-VL-7B) with
  the released LoRA including the `<|FLAG|>` response head, `oaaoaa/proact_lora_base_liveccbase_final`
  (`PROACT_CKPT`), bf16.
- Upstream code: the Proact-VL authors' repository, cloned to `shims/proact-vl/src/Proact-VL` (git-ignored). The
  shim imports the `proactvl` package from it (`model.modeling_proact`, `infer.assistant`, `utils.conversations`).
  The header names no commit: pin the commit your clone was tested with.
- Knobs: `PROACT_PROMPT=ego4d|bench` (the authors' own system prompt, or the benchmark's; default `ego4d`),
  `PROACT_THRESHOLD` (FLAG-head threshold, 0.5), `PROACT_NEW_CHUNKS` (the chunks whose utterances are returned, 2),
  `PROACT_MAX_NEW` (12 tokens per utterance), `PROACT_QUERY_AT=new|first` (where the user text rides).
- Output: the utterances on the new chunks, joined; silence on all of them is empty text; wrapped into the reply
  object (`flag` null). Per-chunk FLAG scores are returned in an extra `x_proact` field and logged to
  `shims/proact-vl/logs/requests.jsonl`.
- Pid file `$LOGDIR/serve_proact_vl.pid`.

### `serve_streamingvlm.sh` — StreamingVLM (port 19000, served name `streamingvlm`)

- Environment `envs/streamingvlm`, cache `hf_cache/streamingvlm`, a **shim** (`shims/streamingvlm/shim_server.py`)
  around the authors' streaming inference code (their patched attention and contiguous-RoPE KV cache); the header
  names transformers 4.51.3. Weights `mit-han-lab/StreamingVLM` (a Qwen2.5-VL-7B fine-tune), bf16.
- Upstream code: the StreamingVLM authors' repository (`mit-han-lab/StreamingVLM`), cloned into
  `shims/streamingvlm/src` (git-ignored); the shim imports `streaming_vlm.inference` from it. The header names no
  commit: pin the commit your clone was tested with.
- Knobs: `SVLM_PROMPT=native|bench` (context in the loop's "previous text" slot without the rulebook, or the
  benchmark system turn; default `native`), `SVLM_NEW_CHUNKS` (2), `SVLM_QUERY_AT`.
- Output: the commentary of the new chunks with the silence markers stripped; wrapped into the reply object
  (`flag` null). Per-chunk commentary in an extra `x_streamingvlm` field, logged to
  `shims/streamingvlm/logs/requests.jsonl`.
- Pid file `$LOGDIR/serve_streamingvlm.pid`.

## Logs and pid files

- `serve.sh` sets `LOGDIR` to `<checkout>/runs/serving` and prints where it logs; a script run directly uses
  `${TMPDIR:-/tmp}/vhhome_serving`. Each script writes `$LOGDIR/serve_<name>.pid` (names above); stop a server with
  `kill $(cat $LOGDIR/serve_<name>.pid)`.
- The scripts write their own stdout/stderr to wherever you redirect them (`nohup bash serving/serve.sh <tag>
  > runs/serving/<tag>.log 2>&1 &` is the shape the headers show).
- The three loop shims additionally log every request to `shims/<name>/logs/requests.jsonl`.
- The harness records, per assistant call, the wall seconds and the ticks charged (`assistant_latency[]` in the
  episode record). `python -m vhhome_bench.preflight <tag>` sends one real turn through the harness's own path and
  doubles as the warm-up: a fresh server pays one-time costs (schema compiled to a grammar, kernels chosen, weights
  paged in) that would otherwise be charged to the first episode.
