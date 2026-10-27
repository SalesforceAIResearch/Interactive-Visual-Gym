"""ScreenSim copy of CookSim's shim (cook-bench-engine serving/shims/proact-vl/shim_server.py @ 562aeab). Changes, and only these:
  - the authors' code is imported from CookSim's clone (SHIM_SRC, default ../../../../cook-bench-engine/serving/shims/proact-vl/src);
  - the kitchen wording is replaced by the phone's (the task line is read from "== WHAT THEY'RE TRYING TO DO ==");
  - the wrapped reply carries task_progress (ScreenSim's schema) instead of cook_progress.
Everything else (sampling, chunking, thresholds, prompt mode defaults) is CookSim's audited setup.
"""
#!/usr/bin/env python3
"""OpenAI-compatible shim for Proact-VL (LiveCC-7B-Base + the released LoRA/FLAG head), served for the CookSim
assistant benchmark.

What one POST /v1/chat/completions does (stateless, one request at a time):
  1. The benchmark's system message primes a fresh Proact-VL session (Assistant.prime_system_prompt, the model's
     own code).
  2. The up-to-8 frames (one per game tick, oldest first) are replayed as the model's native stream: one 1-second
     chunk per frame. Proact-VL's Qwen2-VL video path takes 2 frames per 1-s chunk (temporal patch 2), so each tick
     frame is duplicated into its chunk.
  3. Per chunk the model's own <|FLAG|> response head is read (sigmoid > state_threshold => speak). When it fires
     the model generates its utterance for that second (native max_new_tokens 12, T 0.7, top_p 0.9, rep 1.15);
     when it does not, the native silence continuation " ..." is prefilled -- exactly infer_one_chunk().
  4. The benchmark's user text is the "user query" (<|query_start|>...<|query_end|>) of the first NEW chunk --
     the chunks the benchmark has not shown before (it polls every 2 ticks, so the last PROACT_NEW_CHUNKS=2).
     Earlier chunks are context: the model decides on them natively, but what it says there is not returned
     (a previous poll already covered those frames).
  5. The reply text = the model's utterances on the new chunks, joined. FLAG silent on all new chunks => "".

Proact-VL cannot emit the benchmark's JSON (it generates <=12-token commentary per second). The shim therefore
WRAPS the free text into the reply schema: {"task_progress": <empty lists, since = current t>, "goal_change": null,
"text": <utterance or "">, "flag": null}. This is an adapter, not the model's own JSON. The per-chunk FLAG scores and
utterances are returned in the top-level "x_proact" field of the response and logged to logs/requests.jsonl.
"""
import base64, io, json, os, re, sys, threading, time

import torch
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("SHIM_SRC", os.path.join(HERE, "..", "..", "..", "..", "cook-bench-engine", "serving", "shims",
                                               "proact-vl", "src"))
sys.path.insert(0, os.path.join(SRC, "Proact-VL"))
from huggingface_hub import snapshot_download                      # noqa: E402
from qwen_vl_utils.vision_process import smart_resize               # noqa: E402
from proactvl.model.modeling_proact import ProAct_OmniModel, ProActConfig   # noqa: E402
from proactvl.infer.assistant import Assistant, MIN_PIXELS, MAX_PIXELS     # noqa: E402
from proactvl.utils.conversations import EGO4D_SYSTEM_PROMPT               # noqa: E402

CKPT = os.environ.get("PROACT_CKPT", "oaaoaa/proact_lora_base_liveccbase_final")
SERVED = os.environ.get("SERVED_NAME", "proact-vl")
THRESH = float(os.environ.get("PROACT_THRESHOLD", "0.5"))
NEW_CHUNKS = int(os.environ.get("PROACT_NEW_CHUNKS", "2"))
QUERY_AT = os.environ.get("PROACT_QUERY_AT", "new")      # "new": first new chunk (default); "first": chunk 0
ATTN = os.environ.get("PROACT_ATTN", "flash_attention_2")
# PROACT_PROMPT: "bench" (default) = the benchmark system prompt primes the session, the benchmark user text is the query.
# "ego4d" (audit ablation) = the authors' own guidance system prompt for egocentric step-by-step guidance
# (proactvl/utils/conversations.py EGO4D_SYSTEM_PROMPT, used by construct_val_system_prompt for ego4d_goal_step) primes
# the session; the query is the recipe line of the benchmark system prompt + the benchmark user text. The 13k-char
# rulebook is left out (the model was trained on one-paragraph system prompts and recites long ones).
PROMPT = os.environ.get("PROACT_PROMPT", "bench")
GEN = dict(do_sample=True, max_new_tokens=int(os.environ.get("PROACT_MAX_NEW", "12")), temperature=0.7, top_p=0.9,
           repetition_penalty=1.15)
LOG = os.path.join(HERE, "logs", "requests.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

# ------------------------------------------------------------------ model
local = snapshot_download(CKPT, local_files_only=True)
cfg = ProActConfig.from_pretrained(local)
cfg.attn_implementation = ATTN
model = ProAct_OmniModel.from_pretrained(local, config=cfg).to("cuda").eval()
proc = model.processor
tok = proc.tokenizer
GEN["eos_token_id"] = tok.eos_token_id
GEN["pad_token_id"] = tok.pad_token_id
SILENCE = model.silence_eos_token              # ' ...'
asst = Assistant(0, model, max_kv_tokens=16384, use_audio_in_video=False, generate_config=GEN, device="cuda")
asst.set_threshold(THRESH)
LOCK = threading.Lock()
print(f"[shim] loaded {CKPT} (base {cfg.model_name_or_path}), attn={ATTN}, threshold={THRESH}, prompt={PROMPT}",
      flush=True)


def user_prompt(query: str) -> str:
    """MultiAssistantStreamInference._make_user_prompt, verbatim logic (no co-speaker history)."""
    content = [{"type": "video", "video": "<|VIDEO|>"}]
    if query:
        content.append({"type": "text", "text": f"<|query_start|>{query}<|query_end|>"})
    content.append({"type": "text", "text": "<|FLAG|>"})
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    use_default_system_prompt=False, add_generation_tokens=False)
    return text[text.index("<|im_start|>user\n"):]


def chunk_video(img: Image.Image) -> torch.Tensor:
    """One tick frame -> a 2-frame 1-s chunk, resized the way qwen_vl_utils.fetch_video does for this model."""
    w, h = img.size
    rh, rw = smart_resize(h, w, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    img = img.convert("RGB").resize((rw, rh), Image.BICUBIC)
    t = torch.from_numpy(__import__("numpy").asarray(img).copy()).permute(2, 0, 1).float()
    return torch.stack([t, t], 0)


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
    return {"task_progress": {"subgoals_completed": [], "current_subgoal": {"hypotheses": [], "since": t_now},
                              "subgoals_future": {"remaining": [], "constraints": []}},
            "goal_change": None, "text": text, "flag": None}


@torch.no_grad()
def run(system, text, imgs):
    asst.clear_session()
    asst.prime_system_prompt(system)
    n_prompt = asst.system_token_cnt
    n_gen = 0
    first_new = max(0, len(imgs) - NEW_CHUNKS)
    trace, said = [], []
    size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    chunks = imgs if imgs else [None]
    for k, img in enumerate(chunks):
        q = text if k == (0 if QUERY_AT == "first" else first_new) else ""
        if img is None:                                 # no frames: text-only chunk (not expected from the bench)
            inputs = proc(text=user_prompt(q).replace("<|vision_start|><|video_pad|><|vision_end|>", ""),
                          return_tensors="pt", padding=True)
        else:
            inputs = proc(text=user_prompt(q), videos=[chunk_video(img)], images=None, return_tensors="pt",
                          padding=True, size=size)
        inputs = inputs.to(model.device).to(model.llm.dtype)
        n_prompt += inputs["input_ids"].shape[-1]
        flag, score = asst.forward_user(inputs, generate_flag=True)
        if flag:
            resp, cnt = asst.forward_assistant()
            n_gen += cnt
            utt = resp.replace(SILENCE, "").replace("<|im_end|>", "").strip()
        else:
            asst.forward_custom_assistant(SILENCE)
            utt = None
        trace.append({"chunk": k, "new": k >= first_new, "flag_score": round(float(score), 4), "active": bool(flag),
                      "said": utt})
        if k >= first_new and utt:
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
    t_now = int(m.group(1)) if m else 0
    # per-request overrides (audit A/B without a 2.5-min reload): extra_body proact_threshold / proact_prompt
    thr = float(body.get("proact_threshold", THRESH)); prompt_mode = body.get("proact_prompt", PROMPT)
    asst.set_threshold(thr)
    if prompt_mode == "ego4d":
        rm = re.search(r"== WHAT THEY'RE TRYING TO DO ==\\n([^\\n]+)", system)
        text = (("The person's request: " + rm.group(1) + "\n\n") if rm else "") + text
        system = EGO4D_SYSTEM_PROMPT
    t0 = time.time()
    with LOCK:
        said, trace, n_prompt, n_gen = run(system, text, imgs)
    reply = wrap(said, t_now)
    dt = time.time() - t0
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": t0, "t": t_now, "cfg": {"thr": thr, "query_at": QUERY_AT, "prompt": prompt_mode}, "n_frames": len(imgs), "latency_s": round(dt, 3), "text": said,
                            "trace": trace}) + "\n")
    return JSONResponse({
        "id": f"proact-{int(t0 * 1000)}", "object": "chat.completion", "created": int(t0), "model": SERVED,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(reply)}}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen},
        "x_proact": {"adapter": "shim-wrapped (model emits free text; JSON built by the shim)",
                     "threshold": thr, "new_chunks": NEW_CHUNKS, "query_at": QUERY_AT, "prompt": prompt_mode,
                     "trace": trace},
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "18950")), log_level="warning")
