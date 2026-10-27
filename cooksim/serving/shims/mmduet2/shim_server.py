#!/usr/bin/env python3
"""OpenAI-compatible shim for MMDuet2 (wangyueqian/MMDuet2, Qwen2.5-VL-3B post-trained with multi-turn RL for
proactive replies), served for the CookSim assistant benchmark.

It runs the AUTHORS' OWN demo loop, imported, not copied: src/demo/api_server.py ProactiveInferenceAPIClient
(add_text / add_image / reset) on top of src/proactive_eval/qwen2_5_vl (their Qwen2.5-VL class whose
prepare_inputs_for_generation feeds only the NEW frames' pixels while the KV cache carries the rest).

What one POST /v1/chat/completions does (stateless, one request at a time):
  1. client.reset(): history = [system]. The system turn is the model's official proactive system prompt
     (the one it was SFT'd + RL'd with: "... If the information in this segment of the video cannot answer the
     question, output "NO REPLY".") when MMDUET2_SYSTEM=official (default). MMDUET2_SYSTEM=bench puts the
     benchmark's system message there instead (the eval code's "system prompt in data" path).
  2. The <=8 frames (one per game tick, oldest first) are pushed through client.add_image(); the official
     num_frames_per_turn=2 makes each pair of frames one user turn, and after EVERY turn the model generates
     (greedy, max_new_tokens 512 = the official defaults) either a reply or "NO REPLY". Its earlier replies stay in
     the conversation / KV cache exactly as in a live session.
  3. Text placement follows the paper's chat format (Fig. 2: "<image><image> What are people doing?" — text rides
     in the user turn after that turn's frames) via client.add_text():
       - MMDUET2_SYSTEM=official: the benchmark system message (the task / rulebook = the "question") rides on
         the FIRST frame turn; the benchmark user text (timestamp, previous JSON, conversation) rides on the LAST
         turn, i.e. the moment it is sent.
       - MMDUET2_SYSTEM=bench: only the user text, on the last turn.
  4. Each frame is pre-resized with qwen-vl-utils' smart_resize to MMDUET2_MAX_PIXELS (default 128*28*28 =
     "128 tokens per frame", the paper's SFT/RL setting, train/scripts/train.sh MAX_PIXELS). 0 = no pre-resize
     (the demo server's own default: pip qwen-vl-utils MAX_PIXELS).
  5. Reply decision = the model's own text: exactly "NO REPLY" (NO_REPLY_STR of the RL reward / the eval
     post-processing) -> spoken text "". Only the LAST turn's output (the 2 frames this poll adds) is returned.

JSON: MMDuet2 is trained to write plain-text replies. If its last-turn output happens to parse as a JSON object
with a string "text", that object is used (missing schema keys filled); otherwise the shim WRAPS the plain text
into the reply schema: {"cook_progress": <empty, since = current t>, "goal_change": null, "text": ..., "flag": null}.
The per-turn outputs are returned in "x_mmduet2" and logged to logs/requests.jsonl.
"""
import base64, io, json, os, re, sys, threading, time, types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, os.path.join(SRC, "proactive_eval"))     # their qwen2_5_vl package (DTD model class)
sys.path.insert(0, os.path.join(SRC, "demo"))               # demo/inference.py + demo/api_server.py

import torch                                                   # noqa: E402
from PIL import Image                                          # noqa: E402
from fastapi import FastAPI, Request                           # noqa: E402
from fastapi.responses import JSONResponse                     # noqa: E402
import uvicorn                                                 # noqa: E402
from huggingface_hub import snapshot_download                  # noqa: E402
from qwen_vl_utils.vision_process import smart_resize          # noqa: E402
import api_server as A                                         # noqa: E402  (official demo server module)

SERVED = os.environ.get("SERVED_NAME", "mmduet2")
CKPT = os.environ.get("MMDUET2_CKPT", "wangyueqian/MMDuet2")
SYSMODE = os.environ.get("MMDUET2_SYSTEM", "official")        # official | bench
MAXPIX = int(os.environ.get("MMDUET2_MAX_PIXELS", str(128 * 28 * 28)))
FRAME_INTERVAL = float(os.environ.get("MMDUET2_FRAME_INTERVAL", "1.2"))   # one frame per game tick
NO_REPLY = "NO REPLY"
LOG = os.path.join(HERE, "logs", "requests.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

# Official defaults, read from the authors' own argument dataclass (not retyped).
F = A.ProactiveTestAPIArguments.__dataclass_fields__
args = types.SimpleNamespace(
    llm_pretrained=snapshot_download(CKPT, local_files_only=True),
    attn_implementation=F["attn_implementation"].default,       # flash_attention_2
    system_prompt=F["system_prompt"].default,
    input_assistant_turns=F["input_assistant_turns"].default,
    do_sample=F["do_sample"].default, temperature=F["temperature"].default, top_k=F["top_k"].default,
    num_frames_per_turn=F["num_frames_per_turn"].default,       # 2
)
OFFICIAL_SYSTEM = args.system_prompt
client = A.ProactiveInferenceAPIClient(args)
client.set_fps(frame_interval=FRAME_INTERVAL)
LOCK = threading.Lock()
print(f"[shim] loaded {CKPT} -> {args.llm_pretrained}; system={SYSMODE} max_pixels={MAXPIX} "
      f"frames/turn={args.num_frames_per_turn} do_sample={args.do_sample} attn={args.attn_implementation}", flush=True)


def prep(img, maxpix):
    img = img.convert("RGB")
    if maxpix > 0:
        w, h = img.size
        rh, rw = smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=maxpix)
        if (rw, rh) != (w, h):
            img = img.resize((rw, rh), Image.BICUBIC)
    return img


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


def empty_reply(t_now):
    return {"cook_progress": {"subgoals_completed": [], "current_subgoal": {"hypotheses": [], "since": t_now},
                              "subgoals_future": {"remaining": [], "constraints": []}},
            "goal_change": None, "text": "", "flag": None}


def to_reply(out, t_now):
    """-> (reply dict, mode). mode: silent | model_json | wrapped."""
    s = out.strip()
    if s == NO_REPLY or not s:
        return empty_reply(t_now), "silent"
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict) and isinstance(o.get("text"), str):
                r = empty_reply(t_now)
                r.update({k: v for k, v in o.items() if k in r})
                if not isinstance(r["cook_progress"], dict):
                    r["cook_progress"] = empty_reply(t_now)["cook_progress"]
                for k, v in empty_reply(t_now)["cook_progress"].items():
                    r["cook_progress"].setdefault(k, v)
                return r, "model_json"
        except Exception:
            pass
    r = empty_reply(t_now)
    r["text"] = s
    return r, "wrapped"


@torch.no_grad()
def run(system, text, imgs, sysmode=SYSMODE, maxpix=MAXPIX, must_reply=False):
    client.reset()                                    # history = [official system prompt]
    if sysmode == "bench":
        client.history = [{"role": "system", "content": system}]
    n = len(imgs)
    k = args.num_frames_per_turn
    lead = n % k                                      # odd count: the oldest frame is dropped so turns end on NOW
    imgs = imgs[lead:]
    n_turns = max(1, len(imgs) // k)
    turns = []
    for ti in range(n_turns):
        if sysmode == "official" and ti == 0 and system:
            client.add_text(system)
        if sysmode == "official_last" and ti == n_turns - 1 and system:
            client.add_text(system)
        if ti == n_turns - 1 and text:
            client.add_text(text)
        if must_reply and ti == n_turns - 1:          # probe only: the eval loader's must_reply suffix on the prompt
            _orig = client.processor.apply_chat_template
            client.processor.apply_chat_template = lambda *a, **kw: _orig(*a, **kw) + client.must_reply_prompt
        res = None
        try:
            for img in imgs[ti * k:(ti + 1) * k]:
                res = client.add_image(prep(img, maxpix))
        finally:
            if must_reply and ti == n_turns - 1:
                client.processor.apply_chat_template = _orig
        out = client.history[-1]["content"] if client.history and client.history[-1]["role"] == "assistant" else ""
        turns.append({"turn": ti, "frames": f"{ti * k}-{ti * k + k - 1}", "out": out,
                      "demo_response_flag": bool(res and res.get("response"))})
    n_prompt = client.past_key_values.get_seq_length() if client.past_key_values is not None else 0
    return turns, n_prompt


def ntoks(s):
    return len(client.processor.tokenizer(s)["input_ids"]) if s else 0


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
    t_now = int(m.group(1)) if m else 0
    t0 = time.time()
    # Probe-only overrides (bench.py --extra); defaults = the serve-script settings.
    sysmode = body.get("mmduet2_system", SYSMODE)
    maxpix = int(body.get("mmduet2_max_pixels", MAXPIX))
    must_reply = bool(body.get("mmduet2_must_reply", False))
    sample = body.get("mmduet2_sample")               # [temperature, top_k]: the code's optional sampling path
    with LOCK:
        if sample:
            client.do_sample, client.temperature, client.top_k = True, float(sample[0]), int(sample[1])
        try:
            turns, kv_len = run(system, text, imgs, sysmode, maxpix, must_reply)
        finally:
            client.do_sample, client.temperature, client.top_k = args.do_sample, args.temperature, args.top_k
    last = turns[-1]["out"] if turns else ""
    reply, mode = to_reply(last, t_now)
    n_gen = sum(ntoks(t["out"]) for t in turns)
    dt = time.time() - t0
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": t0, "t": t_now, "n_frames": len(imgs), "latency_s": round(dt, 3), "mode": mode,
                            "system": sysmode, "max_pixels": maxpix, "must_reply": must_reply, "sample": sample,
                            "text": reply["text"], "turns": turns}) + "\n")
    return JSONResponse({
        "id": f"mmduet2-{int(t0 * 1000)}", "object": "chat.completion", "created": int(t0), "model": SERVED,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(reply)}}],
        "usage": {"prompt_tokens": max(0, kv_len - n_gen), "completion_tokens": n_gen, "total_tokens": kv_len},
        "x_mmduet2": {"mode": mode, "system": sysmode, "max_pixels": maxpix, "must_reply": must_reply,
                      "turns": turns},
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "18900")), log_level="warning")
