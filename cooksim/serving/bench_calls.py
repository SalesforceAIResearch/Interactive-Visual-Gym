#!/usr/bin/env python3
"""Replay the reconstructed genuine assistant calls (calls.jsonl) against a local OpenAI-compatible server,
ONE request in flight, with exactly the request the benchmark sends: tools/controller_eval.py OpenAIBackend
(profile, response_format json_object, max_tokens, image_scale/format, images order) — imported, not copied,
so the measured path is the benchmark's own code.

    python serving/bench_calls.py --model qwen3.8-27b --base-url http://127.0.0.1:18300/v1 --profile default \
        --max-tokens 4000 --image-scale 1.0 --tag qwen_default [--extra '{"chat_template_kwargs":{...}}'] [--n 24]
Results -> results/<tag>.json ; prints median/p90 latency, tokens, JSON/schema validity.
"""
import argparse, base64, json, os, sys, time, statistics as S

ENG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ENG, "tools"))
HERE = os.path.dirname(os.path.abspath(__file__))
CALLS = os.environ.get("BENCH_CALLS", os.path.join(ENG, "bench", "assistant_calls_24.jsonl.gz"))
OUT = os.environ.get("BENCH_OUT", os.path.join(ENG, "tmp", "serving_bench"))
import controller_eval as CE          # noqa: E402


def schema_ok(o):
    """The assistant reply schema of the benchmark: cook_progress{subgoals_completed, current_subgoal,
    subgoals_future}, goal_change, text (string)."""
    if not isinstance(o, dict) or not o:
        return False, "not an object / unparseable"
    cp = o.get("cook_progress")
    if not isinstance(cp, dict):
        return False, "no cook_progress"
    for k in ("subgoals_completed", "current_subgoal", "subgoals_future"):
        if k not in cp:
            return False, f"cook_progress.{k} missing"
    if "text" not in o or not isinstance(o.get("text"), str):
        return False, "text missing/not str"
    if "goal_change" not in o:
        return False, "goal_change missing"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--profile", default="default")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--image-scale", type=float, default=1.0)
    ap.add_argument("--image-format", default="png")
    ap.add_argument("--extra", default=None, help="JSON merged into the profile's extra_body (e.g. thinking on)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-json-mode", action="store_true",
                    help="do not send response_format json_object (needed when extra carries a schema constraint)")
    a = ap.parse_args()
    if a.extra:
        prof = dict(CE.PROFILES[a.profile]); eb = dict(prof.get("extra_body") or {}); eb.update(json.loads(a.extra))
        prof["extra_body"] = eb
        CE.PROFILES["_bench"] = prof
        a.profile_used = "_bench"
    else:
        a.profile_used = a.profile
    b = CE.OpenAIBackend(a.model, a.base_url, max_tokens=a.max_tokens, profile=a.profile_used,
                         image_scale=a.image_scale, image_format=a.image_format, json_mode=not a.no_json_mode)
    # capture raw text too: wrap requests.post
    raw = {}
    _post = b.requests.post
    def post(*args, **kw):
        r = _post(*args, **kw)
        try:
            j = r.json(); m = j["choices"][0]["message"]
            raw["content"] = m.get("content"); raw["reasoning"] = m.get("reasoning_content") or m.get("reasoning")
            raw["finish"] = j["choices"][0].get("finish_reason")
        except Exception:
            raw["content"] = r.text[:500]
        return r
    b.requests = type("R", (), {"post": staticmethod(post)})
    import gzip
    _op = gzip.open if CALLS.endswith(".gz") else open
    calls = [json.loads(l) for l in _op(CALLS, "rt")][: a.n]
    for c in calls[: a.warmup]:                                  # warm-up (not timed): compile/cudagraph/prefix
        b.call(c["system"], c["user"], [base64.b64decode(f) for f in c["frames_b64"]])
    rows = []
    for i, c in enumerate(calls):
        frames = [base64.b64decode(f) for f in c["frames_b64"]]
        raw.clear()
        try:
            out, meta = b.call(c["system"], c["user"], frames)
            ok, why = schema_ok(out)
        except Exception as e:
            out, meta, ok, why = {}, {"latency_ms": None}, False, f"error {e!r}"[:200]
        rows.append(dict(i=i, src=c["src"], t=c["t_issue"], said=c["said"], ok=ok, why=why,
                         latency_ms=meta.get("latency_ms"), prompt_tokens=meta.get("prompt_tokens"),
                         completion_tokens=meta.get("completion_tokens"), finish=raw.get("finish"),
                         reasoning_chars=len(raw.get("reasoning") or ""),
                         text=(out or {}).get("text"), reply=out, raw_content=(raw.get("content") or "")[:3000],
                         ref_text=(c["ref_reply"] or {}).get("text"), ref_model=c["ref_model"]))
        r = rows[-1]
        print(f"[{i+1:2d}] {r['latency_ms']}ms p={r['prompt_tokens']} c={r['completion_tokens']} ok={ok} {why} | {str(r['text'])[:90]!r}", flush=True)
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    summ = dict(tag=a.tag, model=a.model, profile=a.profile, extra=a.extra, max_tokens=a.max_tokens,
                image_scale=a.image_scale, n=len(rows),
                latency_median_ms=(S.median(lat) if lat else None),
                latency_p90_ms=(lat[min(len(lat) - 1, int(0.9 * len(lat)))] if lat else None),
                prompt_tokens_median=S.median([r["prompt_tokens"] or 0 for r in rows]),
                completion_tokens_median=S.median([r["completion_tokens"] or 0 for r in rows]),
                schema_ok=sum(r["ok"] for r in rows), finish_length=sum(1 for r in rows if r["finish"] == "length"),
                spoke=sum(1 for r in rows if r["text"]), ref_spoke=sum(1 for r in rows if r["ref_text"]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, a.tag + ".json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
