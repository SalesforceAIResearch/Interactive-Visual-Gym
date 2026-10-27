#!/usr/bin/env python3
"""OpenAI-compatible shim for InternVideo3-8B-Instruct (yanziang/InternVideo3-8B-Instruct).

Why a shim: the model is `InternVideo3ForConditionalGeneration` (Qwen3-VL-8B vision tower + an LLM whose attention
was converted to M2LA, a DeepSeek-style multi-head latent attention with per-layer kv_lora_rank). It ships as
trust_remote_code transformers code only; vLLM 0.19.1 has no `internvideo3` architecture, so it cannot be served there.

What it serves: POST /v1/chat/completions (+ GET /v1/models, GET /health) for the benchmark's request shape --
one system message, one user message with text plus up to 8 `image_url` data URIs, optional `response_format`
(`json_schema` -> grammar-constrained decoding with xgrammar on that schema; `json_object` -> any JSON object).
One request at a time (a lock around generate). Transformers inference, bf16, sdpa attention, as on the model card.

Frames: FRAMES=video (default) passes the images as ONE short clip through the model's video processor, which is
what the model card uses for temporal input: frames are kept in order (no resampling), timestamps are TICK_SECONDS
apart, and the clip placeholder sits where the first image was. FRAMES=images passes each frame as its own image.

Env: MODEL (repo id), PORT (18700), SERVED_NAME (internvideo3-8b), FRAMES, TICK_SECONDS (1.2).
"""
import base64, io, json, os, re, time, uuid

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, LogitsProcessorList

MODEL = os.environ.get("MODEL", "yanziang/InternVideo3-8B-Instruct")
SERVED = os.environ.get("SERVED_NAME", "internvideo3-8b")
FRAMES = os.environ.get("FRAMES", "video")
TICK = float(os.environ.get("TICK_SECONDS", "1.2"))
MAX_NEW_CAP = int(os.environ.get("MAX_NEW_CAP", "4000"))
# ATTN: sdpa is the model card / README quickstart; the authors' own eval scripts (InternVideo3_eval/scripts/eval_*.sh)
# load it with attn_implementation=flash_attention_2 (needs flash-attn in the env).
ATTN = os.environ.get("ATTN", "sdpa")

print(f"[shim] loading {MODEL} (frames={FRAMES}, attn={ATTN})", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN,
                                             device_map="cuda:0", trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
tok = processor.tokenizer
print(f"[shim] loaded in {time.time() - t0:.0f}s", flush=True)

# ---- grammar-constrained JSON (xgrammar), compiled once per distinct schema
import xgrammar as xgr
_vocab = getattr(model.config, "text_config", model.config).vocab_size
_tinfo = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=_vocab)
_compiler = xgr.GrammarCompiler(_tinfo)
_cache = {}


def _grammar(rf):
    if not rf:
        return None
    t = rf.get("type")
    if t == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
        key = json.dumps(schema, sort_keys=True)
        if key not in _cache:
            _cache[key] = _compiler.compile_json_schema(key, strict_mode=True)
        return _cache[key]
    if t == "json_object":
        if "__obj__" not in _cache:
            _cache["__obj__"] = _compiler.compile_json_schema('{"type":"object"}', strict_mode=False)
        return _cache["__obj__"]
    return None


def _image(url):
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    raise ValueError("only data: image URLs are supported")


def _convert(messages):
    """OpenAI messages -> (HF messages with image/video placeholders, list of PIL images)."""
    out, images = [], []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str) or c is None:
            out.append({"role": m["role"], "content": [{"type": "text", "text": c or ""}]})
            continue
        parts, video_placed = [], False
        for p in c:
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "image_url":
                images.append(_image(p["image_url"]["url"]))
                if FRAMES == "images":
                    parts.append({"type": "image"})
                elif not video_placed:                                   # FRAMES=video | video_dup
                    parts.append({"type": "video"}); video_placed = True
        out.append({"role": m["role"], "content": parts})
    return out, images


def _inputs(messages):
    hf, images = _convert(messages)
    text = processor.apply_chat_template(hf, tokenize=False, add_generation_prompt=True)
    if not images:
        return processor(text=text, return_tensors="pt")
    if FRAMES == "images":
        return processor(text=text, images=images, do_resize=True, return_tensors="pt")
    from transformers.video_utils import VideoMetadata
    frames = np.stack([np.asarray(im) for im in images])            # (T, H, W, 3), oldest first
    fps = 1.0 / TICK
    if FRAMES == "video_dup":                                      # each tick frame fills a whole temporal patch
        frames = np.repeat(frames, 2, axis=0); fps = 2.0 / TICK   # (2 identical frames), timestamps stay per tick
    if len(frames) % 2:                                            # temporal patch is 2 frames
        frames = np.concatenate([frames, frames[-1:]])
    meta = VideoMetadata(total_num_frames=len(frames), fps=fps, frames_indices=list(range(len(frames))),
                         width=frames.shape[2], height=frames.shape[1], duration=len(frames) / fps)
    return processor(text=text, videos=[frames], video_metadata=[meta], do_sample_frames=False,
                     return_tensors="pt")


import asyncio
_lock = asyncio.Lock()          # one request in flight
app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED, "object": "model", "owned_by": "local"}]}


def _generate(body):
    inputs = _inputs(body["messages"]).to(model.device)
    n_in = inputs["input_ids"].shape[1]
    temp = float(body.get("temperature", 0.7) or 0.0)
    gk = dict(max_new_tokens=min(int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024),
                                 MAX_NEW_CAP), use_cache=True)
    if temp > 0:
        gk.update(do_sample=True, temperature=temp, top_p=float(body.get("top_p", 1.0)),
                  top_k=int(body.get("top_k", 0) or 0))
    else:
        gk.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    if body.get("repetition_penalty"):
        gk["repetition_penalty"] = float(body["repetition_penalty"])
    g = _grammar(body.get("response_format"))
    if g is not None:
        gk["logits_processor"] = LogitsProcessorList([xgr.contrib.hf.LogitsProcessor(g)])
    with torch.inference_mode():
        out = model.generate(**inputs, **gk)
    gen = out[0, n_in:]
    text = tok.decode(gen, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()
    n_out = int(gen.shape[0])
    finish = "length" if n_out >= gk["max_new_tokens"] else "stop"
    return text, n_in, n_out, finish


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    t0 = time.time()
    try:
        async with _lock:
            text, n_in, n_out, finish = await asyncio.get_running_loop().run_in_executor(None, _generate, body)
    except Exception as e:  # surface as HTTP 400/500 like an OpenAI server
        import traceback; traceback.print_exc()
        return JSONResponse({"error": {"message": repr(e)[:1000]}}, status_code=500)
    print(f"[shim] {time.time() - t0:.2f}s in={n_in} out={n_out} {finish}", flush=True)
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:16], "object": "chat.completion", "created": int(time.time()),
            "model": SERVED,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "18700")), log_level="warning")
