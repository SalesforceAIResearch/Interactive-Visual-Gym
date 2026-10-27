#!/usr/bin/env python3
"""The Claude judge through the Message Batches API (2026-10-12).

The workspace allows about 16 concurrent live requests, which made the third judge the bottleneck of the three-judge
pass (33 rounds a minute, two days for the 101k rounds of the three runs). Batches have no concurrency cap and cost half.

    python -m vhhome_bench.scoring.rubric_batch submit runs/formal/r1      # every round without a Claude verdict -> batches
    python -m vhhome_bench.scoring.rubric_batch status runs/formal/r1
    python -m vhhome_bench.scoring.rubric_batch merge  runs/formal/r1      # ended batches -> the per-episode outputs (under the root lock)

Prompts are built exactly as vhhome_bench/scoring/rubric.py builds them (--subset items, no frames); verdicts go through the same
resolve() and the same per-round merge, so a batch-judged round is indistinguishable from a live-judged one. A round
whose batch result errored is left without a Claude verdict; the closing `sq_v4_vh.py --judge all` sweep grades it live.
"""
from vhhome import keys
import argparse, fcntl, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)

from vhhome_bench.scoring import rubric as G  # noqa: E402

G.ACTIVE = [it for it in G.ITEMS if it[0] in G.SUBSET_ITEMS]          # the --subset prompt, as every pass of bench2
MODEL = "claude-haiku-5-5"
SYSTEM = "You are a strict, neutral grader. Reply with JSON only."
PER_BATCH = 4000


def client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY") or keys.get("ANTHROPIC_API_KEY")
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID") or keys.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(api_key=key, timeout=300.0, max_retries=3, default_headers={"anthropic-workspace-id": ws})


def root_of(a):
    return a if os.path.isabs(a) else os.path.join(ENG, a)


def bdir(root):
    d = os.path.join(root, "sq_v4", "claude_batches"); os.makedirs(d, exist_ok=True); return d


def episodes(root):
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        arm = d.rstrip("/").split("/")[-1]
        if arm in ("serve_logs", "_fill", "sq_v4", "sq_v4_frames") or arm.startswith("_"):
            continue
        for f in sorted(glob.glob(d + "*.json")):
            if not f.endswith(".lane.json"):
                yield arm, f


def has_claude(outp, rnd):
    try:
        j = json.load(open(outp))
    except Exception:
        return False
    for r in j.get("rounds") or []:
        if r.get("round") == rnd:
            return "score" in (r.get("judges") or {}).get("claude", {})
    return False


def submit(root, limit=0):
    from vhhome_bench import protocol as P
    c = client(); d = bdir(root)
    ledger_f = os.path.join(d, "batches.json")
    ledger = json.load(open(ledger_f)) if os.path.exists(ledger_f) else []
    pending = {cid for b in ledger if not b.get("merged") for cid in b["ids"]}     # already in flight
    reqs, manifest, n = [], {}, 0
    for arm, f in episodes(root):
        rec = json.load(open(f)); name = os.path.basename(f)[:-5]
        outp = os.path.join(root, "sq_v4", arm, name + ".json")
        rounds = G.rounds_of(rec)
        card = P.PERSONA_CARDS.get(rec.get("persona"), "")
        for i in range(len(rounds)):
            rnd = rounds[i]["round"]
            key = f"{arm}|{name}|{rnd}"
            if key in pending or (os.path.exists(outp) and has_claude(outp, rnd)):
                continue
            prompt = G.build_prompt(rec, rounds, i, card)[0]
            cid = f"q{n:07d}"; n += 1
            manifest[cid] = [arm, name, rnd]
            reqs.append({"custom_id": cid, "params": {
                "model": MODEL, "max_tokens": 16000,
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "output_config": {"effort": "low"}}})
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    print(f"{root}: {n} round(s) need a Claude verdict", flush=True)
    for k in range(0, len(reqs), PER_BATCH):
        chunk = reqs[k:k + PER_BATCH]
        b = c.messages.batches.create(requests=chunk)
        ids = {r["custom_id"]: manifest[r["custom_id"]] for r in chunk}
        json.dump(ids, open(os.path.join(d, f"{b.id}.manifest.json"), "w"))
        ledger.append({"batch": b.id, "n": len(chunk), "submitted": int(time.time()), "merged": False,
                       "ids": [f"{a}|{e}|{r}" for a, e, r in ids.values()]})
        json.dump(ledger, open(ledger_f, "w"), indent=1)
        print(f"  batch {b.id}: {len(chunk)} requests", flush=True)


def status(root):
    c = client(); d = bdir(root); ledger_f = os.path.join(d, "batches.json")
    if not os.path.exists(ledger_f):
        print("no batches"); return
    for b in json.load(open(ledger_f)):
        try:
            m = c.messages.batches.retrieve(b["batch"]); rc = m.request_counts
            print(f"{b['batch']} n={b['n']} {m.processing_status} ok={rc.succeeded} err={rc.errored} exp={rc.expired} "
                  f"processing={rc.processing} merged={b['merged']}")
        except Exception as e:
            print(b["batch"], "retrieve failed:", type(e).__name__, str(e)[:80])


def merge(root, again=False):
    from vhhome_bench import protocol as P
    c = client(); d = bdir(root); ledger_f = os.path.join(d, "batches.json")
    if not os.path.exists(ledger_f):
        print("no batches"); return
    ledger = json.load(open(ledger_f))
    verdicts, failed = {}, 0
    done_batches = []
    for b in ledger:
        if b["merged"] and not again:
            continue
        m = c.messages.batches.retrieve(b["batch"])
        if m.processing_status != "ended":
            print(f"{b['batch']} still {m.processing_status}"); continue
        man = json.load(open(os.path.join(d, f"{b['batch']}.manifest.json")))
        for r in c.messages.batches.results(b["batch"]):
            arm, name, rnd = man[r.custom_id]
            if r.result.type == "succeeded":
                msg = r.result.message
                txt = "".join(x.text for x in msg.content if x.type == "text")
                try:
                    parsed = G.parse(txt)
                    verdicts.setdefault((arm, name), {})[rnd] = parsed
                except Exception:
                    failed += 1
            else:
                failed += 1
        done_batches.append(b["batch"])
    print(f"{len(verdicts)} episode(s) with results, {failed} failed round(s)", flush=True)
    out_root = os.path.join(root, "sq_v4"); os.makedirs(out_root, exist_ok=True)
    lockf = open(os.path.join(out_root, ".grader.lock"), "w")
    print("waiting for the root lock ...", flush=True); fcntl.flock(lockf, fcntl.LOCK_EX)
    n_ep = 0
    for (arm, name), rv in verdicts.items():
        f = os.path.join(root, arm, name + ".json")
        if not os.path.exists(f):
            continue
        rec = json.load(open(f)); rounds = G.rounds_of(rec)
        card = P.PERSONA_CARDS.get(rec.get("persona"), "")
        outp = os.path.join(out_root, arm, name + ".json")
        existing = None
        if os.path.exists(outp):
            try:
                existing = json.load(open(outp))
            except Exception:
                existing = None
        rows = []
        for i in range(len(rounds)):
            rnd = rounds[i]["round"]
            old = next((x for x in (existing or {}).get("rounds") or [] if x.get("round") == rnd), None)
            per = dict((old or {}).get("judges") or {})
            if not per and old and "score" in old and existing.get("judge"):       # single-judge format
                per = {G._old_judge_key(existing): {k: old[k] for k in ("score", "must_fail", "types", "items", "raw") if k in old}}
            if rnd in rv and "score" not in per.get("claude", {}):
                prompt, t, t_next, err, stop, stamp = G.build_prompt(rec, rounds, i, card)
                per["claude"] = G.resolve(rv[rnd], rec, t, t_next, stop, stamp) | {"raw": rv[rnd]}
            if old:
                row = {k: v for k, v in old.items() if k not in ("score", "must_fail", "types", "items", "error", "raw", "judges")}
            else:
                _p, t, t_next, err, stop, stamp = G.build_prompt(rec, rounds, i, card)
                row = {"round": rnd, "t": t, "assistant": [a.get("text") for a in rounds[i]["assistant"]],
                       "person_reply": [(x.get("move"), x.get("text")) for x in rounds[i]["human"] if (x.get("text") or "").strip()],
                       "live_error": (err or {}).get("class"), "stamp_source": stamp.get("source")}
            row["judges"] = per
            merged = G._merge(per)
            row.update(merged if merged else {"score": None, "items": {}, "types": []})
            rows.append(row)
        judges = sorted({jn for r in rows for jn in (r.get("judges") or {})})
        out = {"schema": G.RUBRIC_VERSION + "-sub5-3j", "items_graded": [i for i, *_ in G.ACTIVE], "episode": name, "arm": arm,
               "scenario": rec.get("scenario"), "persona": rec.get("persona"), "judges": judges, "judge": ",".join(judges),
               "with_frames": False, "engine_rev": rec.get("engine_rev"), "outcome": rec.get("outcome"),
               "aggregate": G.aggregate(rows), "rounds": rows}
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump(out, open(outp + ".tmp", "w"), indent=1, default=str); os.replace(outp + ".tmp", outp)
        n_ep += 1
    for b in ledger:
        if b["batch"] in done_batches:
            b["merged"] = True
    json.dump(ledger, open(ledger_f, "w"), indent=1)
    print(f"merged Claude verdicts into {n_ep} episode output(s); {len(done_batches)} batch(es) marked merged", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "status", "merge"])
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=0, help="submit: at most this many rounds (a test)")
    ap.add_argument("--again", action="store_true", help="merge: re-read batches already merged (for rounds still lacking a verdict)")
    a = ap.parse_args()
    {"submit": lambda: submit(root_of(a.root), a.limit), "status": lambda: status(root_of(a.root)),
     "merge": lambda: merge(root_of(a.root), a.again)}[a.cmd]()
