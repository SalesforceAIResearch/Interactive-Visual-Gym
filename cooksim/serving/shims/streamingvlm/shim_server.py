#!/usr/bin/env python3
"""OpenAI-compatible shim for StreamingVLM (mit-han-lab/StreamingVLM, Qwen2.5-VL-7B fine-tune), served for the
CookSim assistant benchmark, using the authors' own streaming inference code (streaming_vlm/inference).

What one POST /v1/chat/completions does (stateless, one request at a time):
  * The up-to-8 frames (one per game tick, oldest first) are replayed as StreamingVLM's native stream: one 1-second
    chunk per frame, 2 frames per chunk (FPS=2 in qwen_vl_utils; the tick frame is duplicated), resized with the
    authors' _spatial_resize_video, "Time=a-b s" prompt per chunk, interleaved user(chunk) / assistant(commentary)
    turns, the patched streaming model (contiguous-RoPE "shrink" positions), process_past_kv every chunk (attention
    sink 512 + text window 512; vision window 16 chunks -- nothing is evicted inside an 8-chunk call).
  * Chunk 0 is exactly the native first turn: [system], "previous text" (empty), user = [Time, video, query]. The
    query is the benchmark's user text verbatim; the benchmark's system prompt goes into the system turn (the native
    demo uses the default Qwen system prompt; the 'previous text' slot would be truncated to sink+window = 1024
    tokens, so the ~3.3k-token rulebook is kept out of it).
  * The model generates its native per-second commentary on EVERY chunk (MAX_TOKEN_PER_DURATION = 20, T 0.9,
    repetition 1.05, the defaults of streaming_vlm/inference/inference.py). A chunk whose commentary is only the
    " ..." continuation marker is the model staying silent for that second.
  * The reply text = the commentary of the chunks the benchmark has not shown before (the last SVLM_NEW_CHUNKS=2,
    since the benchmark polls every 2 ticks), " ..." markers stripped, joined. Silent on all new chunks => "".

StreamingVLM does not emit the benchmark JSON in its streaming mode (it writes <=20-token commentary per second).
The shim WRAPS the text into the reply schema: {"cook_progress": <empty lists, since = current t>, "goal_change":
null, "text": ..., "flag": null}. That is an adapter, not the model's own JSON. The per-chunk commentary is returned
in the top-level "x_streamingvlm" field and logged to logs/requests.jsonl.
"""
import base64, io, json, os, re, sys, threading, time

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
os.chdir(os.path.join(HERE, "logs") if os.path.isdir(os.path.join(HERE, "logs")) else HERE)  # patch_model mkdirs ./output_image

from huggingface_hub import snapshot_download                                      # noqa: E402
from streaming_vlm.inference.inference import (                                     # noqa: E402
    load_model_and_processor, process_past_kv, SYSTEM_PROMPT_OFFSET, TOKEN_IDS, DEFAULT_TEMPERATURE,
    DEFAULT_REPETITION_PENALTY, MAX_TOKEN_PER_DURATION, DEFAULT_TEXT_SINK, DEFAULT_TEXT_SLIDING_WINDOW,
    DEFAULT_WINDOW_SIZE, DEFAULT_TEXT_ROUND)
from streaming_vlm.inference.streaming_args import StreamingArgs                    # noqa: E402
from livecc_utils.video_process_patch import _spatial_resize_video                  # noqa: E402
from qwen_vl_utils.vision_process import FPS                                         # noqa: E402

SERVED = os.environ.get("SERVED_NAME", "streamingvlm")
NEW_CHUNKS = int(os.environ.get("SVLM_NEW_CHUNKS", "2"))
TEMP = float(os.environ.get("SVLM_TEMPERATURE", DEFAULT_TEMPERATURE))
MAX_NEW = int(os.environ.get("SVLM_MAX_NEW", MAX_TOKEN_PER_DURATION))
# Audit options (2026-09-18), also settable per request via extra_body svlm_query_at / svlm_prompt:
#  SVLM_QUERY_AT "first" (default, the native first turn) | "new": the benchmark user text is the extra user text of
#    the first NEW chunk instead (chunk 0 still opens the native way: previous text + Time + video, no query).
#  SVLM_PROMPT "bench" (default: benchmark system prompt in a system turn) | "native": the authors' loop has no system
#    turn -- context goes into "previous text" (inference.py --previous_text, a one-line video description); here the
#    recipe line of the benchmark system prompt. The 13k-char rulebook is left out.
QUERY_AT = os.environ.get("SVLM_QUERY_AT", "first")
PROMPT = os.environ.get("SVLM_PROMPT", "bench")
LOG = os.path.join(HERE, "logs", "requests.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

local = snapshot_download("mit-han-lab/StreamingVLM", local_files_only=True)
model, processor = load_model_and_processor(local, "Qwen2_5")
model.eval()
DEV = model.device
ASSIST_START_BIAS = len(processor(text="<|im_start|>assistant\n")["input_ids"][0])
ASSIST_END_BIAS = len(processor(text=" ...<|im_end|>")["input_ids"][0])
LOCK = threading.Lock()
print(f"[shim] loaded {local}; FPS={FPS}; temp={TEMP}; max_new={MAX_NEW}", flush=True)


def chunk_video(img: Image.Image) -> torch.Tensor:
    """One tick frame -> the (T=FPS, C, H, W) clip _read_video_decord_plus returns for a 1-s chunk, then the
    authors' spatial resize."""
    t = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1)
    clip = torch.stack([t] * int(FPS), 0)
    return _spatial_resize_video(clip)


def parse(body):
    system, text, imgs = "", "", []
    for m in body.get("messages", []):
        c = m.get("content")
        if m.get("role") == "system":
            system += c if isinstance(c, str) else "".join(p.get("text", "") for p in c)
        elif m.get("role") == "user":
            if isinstance(c, str):
                text += c
                continue
            for p in c:
                if p.get("type") == "text":
                    text += p["text"]
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
                    imgs.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    return system, text, imgs


def wrap(text, t_now):
    return {"cook_progress": {"subgoals_completed": [], "current_subgoal": {"hypotheses": [], "since": t_now},
                              "subgoals_future": {"remaining": [], "constraints": []}},
            "goal_change": None, "text": text, "flag": None}


@torch.no_grad()
def run(system, query, imgs, t_now, query_at="first", previous_text=""):
    """streaming_vlm.inference.inference.streaming_inference(), with in-memory frames instead of a video file."""
    streaming_args = StreamingArgs(pos_mode="shrink", all_text=False)
    past_key_values, full_hist, prev_ids = None, [], None
    clips, pix = [], []
    t0s = t_now - len(imgs) + 1
    first_new = max(0, len(imgs) - NEW_CHUNKS)
    trace, said, n_prompt, n_gen = [], [], 0, 0
    for i, img in enumerate(imgs):
        past_key_values, prev_ids, clips, pix = process_past_kv(
            past_key_values, i, text_round=DEFAULT_TEXT_ROUND, visual_round=DEFAULT_WINDOW_SIZE,
            full_conversation_history=full_hist, prev_generated_ids=prev_ids,
            assistant_start_bias=ASSIST_START_BIAS, assistant_end_bias=ASSIST_END_BIAS,
            recent_video_window_clips=clips, recent_pixel_values_videos=pix,
            text_sink=DEFAULT_TEXT_SINK, text_sliding_window=DEFAULT_TEXT_SLIDING_WINDOW)
        clips.append(chunk_video(img))
        st = float(t0s + i)
        prompt = f"Time={st:.1f}-{st + 1:.1f}s"
        q_here = query if i == (0 if query_at == "first" else first_new) else ""
        if i == 0:
            user_content = [{"type": "text", "text": prompt}, {"type": "video", "video": "frames"}] + (
                [{"type": "text", "text": q_here}] if q_here else [])
            full_hist = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "previous text", "content": previous_text}, {"role": "user", "content": user_content}]
            text = processor.apply_chat_template(full_hist, tokenize=False, add_generation_prompt=True)
        else:
            user_content = [{"type": "text", "text": prompt}, {"type": "video", "video": "frames"}] + (
                [{"type": "text", "text": q_here}] if q_here else [])
            full_hist.append({"role": "user", "content": user_content})
            text = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False,
                                                 add_generation_prompt=True)
            text = "\n" + text[SYSTEM_PROMPT_OFFSET:]
        inputs = processor(text=[text], videos=clips[-1], padding=True, return_tensors="pt").to(DEV)
        n_prompt += inputs["input_ids"].shape[1]
        if prev_ids is not None:
            if prev_ids[:, -1].item() != TOKEN_IDS["\n"]:
                inputs["input_ids"] = torch.cat([prev_ids, inputs["input_ids"]], dim=1)
            else:
                inputs["input_ids"] = torch.cat([prev_ids, inputs["input_ids"][:, 1:]], dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        pix.append(inputs["pixel_values_videos"])
        streaming_args.input_ids = inputs["input_ids"]
        if i == 0:
            streaming_args.video_grid_thw = inputs["video_grid_thw"]
            streaming_args.second_per_grid_ts = inputs.get("second_per_grid_ts")
        else:
            streaming_args.video_grid_thw = torch.cat([streaming_args.video_grid_thw, inputs["video_grid_thw"]], 0)
            streaming_args.second_per_grid_ts = (torch.cat([streaming_args.second_per_grid_ts,
                                                            inputs["second_per_grid_ts"]], 0)
                                                 if "second_per_grid_ts" in inputs else None)
        cur_len = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, past_key_values=past_key_values, max_new_tokens=MAX_NEW, use_cache=True,
                             return_dict_in_generate=True, do_sample=True,
                             repetition_penalty=DEFAULT_REPETITION_PENALTY, streaming_args=streaming_args,
                             pad_token_id=151645, temperature=TEMP)
        gen = out.sequences
        if gen[0, -1].item() != 151645:
            gen = torch.cat([gen, torch.tensor([[151645]], device=DEV)], dim=1)
        new_ids = gen[:, cur_len:]
        n_gen += new_ids.shape[1]
        resp = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        past_key_values = out.past_key_values
        prev_ids = gen.clone()
        full_hist.append({"role": "assistant", "content": resp})
        utt = resp.replace("...", " ").strip()
        utt = re.sub(r"\s+", " ", utt)
        trace.append({"chunk": i, "new": i >= first_new, "time": prompt, "raw": resp, "said": utt or None})
        if i >= first_new and utt:
            said.append(utt)
    return " ".join(said).strip(), trace, n_prompt, n_gen


app = FastAPI()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED, "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    system, text, imgs = parse(body)
    m = re.search(r"CURRENT TIMESTAMP:\s*t=(\d+)", text)
    t_now = int(m.group(1)) if m else len(imgs) - 1
    query_at = body.get("svlm_query_at", QUERY_AT); prompt_mode = body.get("svlm_prompt", PROMPT)
    prev = ""
    if prompt_mode == "native":
        rm = re.search(r"The cook was asked to make:[^\n]*", system)
        prev = ("This is a video of a cooking game: top-down view of the kitchen (left) and the cook's first-person "
                "view (right). " + (rm.group(0) if rm else "")).strip()
        system = ""
    t0 = time.time()
    with LOCK:
        said, trace, n_prompt, n_gen = run(system, text, imgs, t_now, query_at, prev)
        torch.cuda.empty_cache()
    reply = wrap(said, t_now)
    dt = time.time() - t0
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": t0, "t": t_now, "cfg": {"query_at": query_at, "prompt": prompt_mode}, "n_frames": len(imgs), "latency_s": round(dt, 3), "text": said,
                            "trace": trace}) + "\n")
    return JSONResponse({
        "id": f"svlm-{int(t0 * 1000)}", "object": "chat.completion", "created": int(t0), "model": SERVED,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(reply)}}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen},
        "x_streamingvlm": {"adapter": "shim-wrapped (model emits free commentary; JSON built by the shim)",
                           "new_chunks": NEW_CHUNKS, "query_at": query_at,
                           "prompt": prompt_mode, "trace": trace},
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "19000")), log_level="warning")
