#!/usr/bin/env python3
"""Offline re-grade of every Q10 standing request, with a judge deciding what a line asserts.

The audit of 2026-08-28 found the keyword grader unusable: it scored the assistant's
ACKNOWLEDGEMENT of the request as a failed alert (205 of 247 "early" verdicts), credited lines
that denied readiness ("the steak is not done yet" scored 1.0), matched an item word against a
claim about a different item, missed alerts phrased without a ready-word, and ended the window
when the cook removed the food, so a fast cook lowered the assistant's score.

This re-grade keeps the engine's window (now the item's physical safe period: done -> burnt)
and asks a judge, per candidate line, what that line asserts about THIS item:

    alert   — asserts the item is ready NOW (or tells the cook to take it off now)
    promise — undertakes to tell the cook later
    denial  — says it is not ready yet
    other   — mentions the item for another reason

The earliest line judged `alert` inside the window is the alert. Results are written to
<ASSIST_ROOT>/proactive_v2.json and consumed by python -m cooksim_bench.scoring.eval_v2_sweep.

    $PY python -m cooksim_bench.scoring.regrade_proactive_v2 [--workers 8] [--limit N] [--model gemini-3.7-flash]
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import sys
import threading

from cooksim_bench.paths import ROOT  # the CookSim release root
BENCH = os.environ.get("ASSIST_ROOT", "tmp/assist_bench_v3")
OUT = os.path.join(ROOT, BENCH, "proactive_v2.json")
CACHE = os.path.join(ROOT, BENCH, "proactive_v2_lines.json")
VERSION = "q10-v3-2026-09-15"          # v3: strict clause-level alert rule (qa_v3_lib.strict_alert)

from cooksim_bench.scoring.qa_v3_lib import _pat, grade_proactive, classify_alert_line, strict_alert          # noqa: E402
import inspect as _inspect                                                      # noqa: E402
from cooksim_bench.scoring import qa_v3_lib as _q  # noqa: E402
# a rule edit must re-grade every request: the stored schema carries a hash of the grading code
VERSION += "-" + hashlib.sha1((_inspect.getsource(_q.strict_alert) + _inspect.getsource(_q.grade_proactive)
                               + _inspect.getsource(_q.bind_request) + _inspect.getsource(_q.ready_horizon)
                               + str(_q._HEDGE.pattern) + str(_q._CLAUSE_SPLIT.pattern)).encode()).hexdigest()[:8]

PROMPT = """A cook asked an AI kitchen assistant: "tell me the moment the {item} is done".
Below is ONE line the assistant later said. Decide what that line asserts ABOUT THE {item} — not about any other food.

LINE: "{text}"

Reply with JSON only: {{"kind": "alert" | "promise" | "denial" | "other"}}
  alert   = an UNCONDITIONAL, present-tense statement that the {item} IS cooked/done RIGHT NOW
            (e.g. "The {item} is done!", "{item}'s ready, take it off now").
  promise = it undertakes to tell the cook LATER, or describes what to do WHEN/IF/AS SOON AS the {item} is done
            (e.g. "wait for the {item} to finish", "the moment its bar is full, scoop it", "if it's done, plate it")
  denial  = it says the {item} is NOT ready yet (still cooking, half done, needs longer)
  other   = anything else: an instruction that does not itself claim the {item} is done now (e.g. "go get a plate"),
            a claim about a DIFFERENT food, or a claim that the {item} is chopped/sliced rather than cooked.
A line is NEVER an alert just because it gives a next step, or because it is conditional on the {item} being done.
When unsure between alert and anything else, do not answer alert."""



def station_type_of(rec):
    """Which heat station each item cooks in for this episode's recipe -> the burn horizon."""
    cell = ((rec.get("composite") or {}).get("cell")) or None
    try:
        os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
        from cooksim_bench.exp_interact import recipe_of
        from cooksim.core.generator import cook_station_for
        rid = recipe_of(cell).id if cell else None
        return lambda item: str(cook_station_for(rid, item)).split(".")[-1].lower()
    except Exception:
        return lambda item: ""


class Judge:
    def __init__(self, model):
        from cooksim_bench.backends.genai_client import make_client
        from google.genai import types
        self.c, self.t, self.model = make_client(120_000), types, model

    def __call__(self, prompt):
        cfg = self.t.GenerateContentConfig(temperature=0.0, max_output_tokens=200,
                                           response_mime_type="application/json")
        r = self.c.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return json.loads((r.text or "{}").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gemini-3.7-flash")
    a = ap.parse_args()

    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    lock = threading.Lock()
    judge = Judge(a.model)

    files = sorted(glob.glob(os.path.join(ROOT, BENCH, "*/*/*/b3_*.json")))
    jobs = []
    for f in files:
        try:
            rec = json.load(open(f))
        except Exception:
            continue
        key0 = "/".join(f.split("/")[-4:])[:-5]
        for e in rec.get("injections") or []:
            if e.get("family") != "qa_proactive" or not e.get("injected"):
                continue
            k = f"{key0}#{e.get('idx')}"
            if k in out and out[k].get("schema") == VERSION:
                continue
            jobs.append((k, f, rec, e))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"[q10-v2] {len(jobs)} requests to re-grade ({len(files)} episodes, cache {len(cache)} lines)", flush=True)

    def classify_with_judge(text, item):
        # the cache holds the judge's RAW label; the strict alert rule is applied afterwards, so a rule change
        # never needs a re-query and never reads a stale post-rule label
        h = hashlib.sha1(f"raw\x00{item}\x00{text}".encode()).hexdigest()[:16]
        with lock:
            kind = cache.get(h)
        if kind is None:
            try:
                kind = (judge(PROMPT.format(item=item, text=text[:1200])) or {}).get("kind")
            except Exception:
                kind = None
            if kind not in ("alert", "promise", "denial", "other"):
                kind = classify_alert_line(text, item)          # honest fallback, never a silent guess
            with lock:
                cache[h] = kind
        # whether a line IS an alert is decided by the checkable clause rule (qa_v3_lib.strict_alert); the judge
        # alone credited conditional lines ("if the pan's progress is full, scoop…"). The judge keeps the
        # promise / denial / other labels, and a judged denial is never promoted.
        if strict_alert(text, item):
            return "denial" if kind == "denial" else "alert"
        return "other" if kind == "alert" else kind

    def one(job):
        k, f, rec, e = job
        item = e.get("item") or ""
        comm = [{"t": c.get("t"), "speaker": "vlm", "text": c.get("text")}
                for c in (rec.get("conversation") or []) if c.get("speaker") == "vlm" and c.get("text")]
        st = station_type_of(rec)(item)
        # A6 (2026-09-19): bind the request to the load cooking at the ask, with each load's own station and lit tick
        # (the episode's world: states dump, else an action-log replay) and the engine's burn time for the run's pace
        try:
            from cooksim_bench.scoring.qa_replay import world_for_record
            world = world_for_record(rec, f)[0]
        except Exception:                                   # noqa: BLE001
            world = None
        g = grade_proactive(e.get("qa") or {}, item, comm, rec.get("heat_events") or [],
                            rec.get("ticks") or 0, station_type=st, classify=classify_with_judge,
                            world=world, pace=rec.get("pace"))
        g["schema"] = VERSION
        g["station"] = st
        g["legacy"] = {kk: (e.get("proactive") or {}).get(kk) for kk in ("verdict", "score", "window", "alert_t")}
        return k, g

    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, g in ex.map(one, jobs):
            out[k] = g
            done += 1
            if done % 50 == 0:
                with lock:
                    json.dump(out, open(OUT + ".tmp", "w"), indent=0); os.replace(OUT + ".tmp", OUT)
                    json.dump(cache, open(CACHE + ".tmp", "w")); os.replace(CACHE + ".tmp", CACHE)
                print(f"  {done}/{len(jobs)}", flush=True)
    json.dump(out, open(OUT + ".tmp", "w"), indent=0); os.replace(OUT + ".tmp", OUT)
    json.dump(cache, open(CACHE + ".tmp", "w")); os.replace(CACHE + ".tmp", CACHE)
    import collections
    c = collections.Counter(v.get("verdict") for v in out.values())
    print(f"[q10-v2] wrote {OUT}: {len(out)} requests, verdicts {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
