#!/usr/bin/env python3
"""Does every sub-case of the live suite actually FIRE with the LLM cook? (2026-09-17)

The scripted verifier (verify_cases_v5.py) follows three fixed cooking orders, and the round-4/5 firing table showed
it blind to sub-cases whose moment never arrives in the order an LLM cook really works (serve_raw(meat) 1/27,
wrong_order(tortilla, lettuce) 0/27 — both passed the scripted verifier). This runs every case with the real cook
(GEMINI_MODEL, default gemini-3.7-flash as in the benchmark lanes) and a scripted assistant that names each mistake
the moment it is injected (COOKSIM_SCRIPTED_ASSISTANT=early): the cook accepts, the episode goes on, and every later
sub-case gets its chance.

Two limits found on the first run (2026-09-17), which shape how misses are judged:
  * the scripted assistant says nothing else, so a classic_novice cook ("what should I start with?") gets no answer
    and times out (39/50) — the default personas are therefore baseline and classic_expert;
  * a one-line call at injection cannot stop a short plating block (commission errors were prevented 0-8%), so the
    wrong plate often commits and the episode ends early.
So a miss is not judged by the raw fire rate. Each unfired sub-case is classified by whether the live kitchen PASSED
its moment (e.g. the item it needed "next" got plated anyway) — a trigger bug — or the episode ended before the cook
reached that stage.

    python3 python -m cooksim_bench.suite.verify_firing_llm [--personas baseline,classic_novice,classic_expert] [--workers 20]
                                       [--out tmp/verify_llm] [--only case1,case2] [--summarize-only]
"""
import argparse
import collections
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
from cooksim_bench.paths import ROOT  # the CookSim release root
CASES = "benchmark/cases_composite_v4.json"


def run_one(case, persona, out_dir, env):
    out = os.path.join(out_dir, persona, f"{case}.json")
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if env.get("COOKSIM_DUMP_STATES"):
        env = dict(env, COOKSIM_DUMP_STATES=out.replace(".json", ".states.jsonl"))
    with open(out.replace(".json", ".log"), "w") as lf:
        subprocess.run([sys.executable, "-m", "cooksim_bench.e2e_stepin_rollout", "--case", case, "--persona-v2", persona,
                        "--paradigm", "hardsoft", "--sup-obs", "text_state_map", "--out", out],
                       cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=3600)


def _comp(v):
    return v[len("assemble("):-1] if isinstance(v, str) and v.startswith("assemble(") else None


def moment_passed(sub, d):
    """Did the live kitchen pass this (unfired) sub-case's moment? True = trigger bug, False = episode ended first."""
    plated = [x["item"] for x in d.get("plate_seq") or []]
    served = d.get("outcome") in ("won", "wrong_serve")
    # G1 (round-5): the REALISED type, not the designed label — a designed add_foreign that runs
    # misidentify_assemble has no `before`, and a designed omit_subgoal that runs premature_serve has no `sid`.
    fam, kw = sub.get("family"), sub.get("kwargs") or {}
    err = sub.get("fn") or sub.get("err")
    if fam != "error":
        return None                                  # questions/intents: judged by their waiting reason instead
    if err == "add_foreign" and kw.get("before") == "serve":
        return served
    if err == "omit_subgoal" and "after" in kw:
        x = _comp(kw["after"])
        return x in plated and len(plated) > plated.index(x) + 1
    if err == "forget_next_step":
        a = kw.get("after", "")
        x = _comp(a) or (a[a.find("(") + 1:a.rfind(")")].split("#")[0] if "(" in a else None)
        return (x in plated) if x else served
    if err == "wrong_order":
        return kw.get("a") in plated
    x = kw.get("component") or _comp(kw.get("sid")) or _comp(kw.get("before"))
    return x in plated


def summarize(out_dir, cases):
    subs = {}
    outcomes = collections.Counter()
    narrowed = []
    for c in cases:
        for i, s in enumerate(c["subs"]):
            subs[(c["name"], i)] = {"case": c["name"], "idx": i, "family": s.get("family"),
                                    "what": (s.get("fn") or s.get("err")) if s.get("family") == "error"
                                    else (s.get("gt_kind") or s.get("family")),
                                    "designed": s.get("err"),
                                    "kwargs": s.get("kwargs"), "runs": 0, "fired": 0, "ended_first": 0, "passed_unfired": 0,
                                    "last_why": collections.Counter(), "sub": s}
    for c in cases:
        for pdir in sorted(os.listdir(out_dir)):
            f = os.path.join(out_dir, pdir, c["name"] + ".json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            outcomes[d.get("outcome")] += 1
            narrowed += [(c["name"], pdir, e.get("err"), d.get("outcome")) for e in d.get("injections") or []
                         if (e.get("report_keys") or {}).get("smoke_narrowed") or e.get("smoke_narrowed")]
            for e in d.get("injections") or []:
                k = (c["name"], e.get("idx"))
                if k not in subs:
                    continue
                r = subs[k]
                r["runs"] += 1
                if e.get("injected"):
                    r["fired"] += 1
                    continue
                why = ((e.get("report_keys") or {}).get("injection_skipped") or {}).get("why") or "?"
                last = why.split(" (waiting: ")[-1].rstrip(")")[:70]
                r["last_why"][last] += 1
                mp = moment_passed(r["sub"], d)
                if mp is None:                       # question / intent: a floor never reached means it ended first
                    mp = (not last.startswith("progress floor")) and d.get("outcome") in ("won", "wrong_serve")
                if mp:
                    r["passed_unfired"] += 1
                else:
                    r["ended_first"] += 1
    return subs, outcomes, narrowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="baseline,classic_expert")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="tmp/verify_llm")
    ap.add_argument("--only", default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--silent", action="store_true",
                    help="a SILENT assistant (COOKSIM_SILENT_ASSISTANT=1) instead of the one that names each mistake: "
                         "nothing is prevented, so every commit is visible; per-tick states are dumped (COOKSIM_DUMP_STATES)")
    a = ap.parse_args()
    cases = json.load(open(os.path.join(ROOT, a.cases)))["cases"]
    if a.only:
        cases = [c for c in cases if c["name"] in set(a.only.split(","))]
    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    if not a.summarize_only:
        env = dict(os.environ)
        env.update(COOKSIM_SCHED="v5", COOKSIM_CASES=a.cases, COOKSIM_PACE="tight", COOKSIM_SCRIPTED_ASSISTANT="early",
                   COOKSIM_ASSIST_LATENCY="off", COOKSIM_NO_VIDEO="1", GEMINI_MODEL=env.get("GEMINI_MODEL", "gemini-3.7-flash"))
        if a.silent:
            env.pop("COOKSIM_SCRIPTED_ASSISTANT", None)
            env.update(COOKSIM_SILENT_ASSISTANT="1", COOKSIM_DUMP_STATES="1")
        if not env.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY missing: the cook is a real LLM here (source tmp/present/env.sh)")
        jobs = [(c["name"], p) for c in cases for p in a.personas.split(",")]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: run_one(j[0], j[1], out_dir, env), jobs))
    subs, outcomes, narrowed = summarize(out_dir, cases)
    json.dump({"outcomes": dict(outcomes), "smoke_narrowed": narrowed,
               "subs": [{**{k: v_ for k, v_ in v.items() if k != "sub"}, "last_why": dict(v["last_why"])} for v in subs.values()]},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    lines = [f"# LLM-cook firing check — episodes by outcome {dict(outcomes)}", "",
             "| case | # | sub-case | fired | MOMENT PASSED UNFIRED | episode ended first | last waiting reason |",
             "|---|---|---|---|---|---|---|"]
    by_fam = collections.defaultdict(lambda: [0, 0, 0, 0])
    for v in sorted(subs.values(), key=lambda v: (v["case"], v["idx"])):
        fam = "question/request" if v["family"] in ("qa_probe", "qa_proactive") else v["family"]
        by_fam[fam][0] += v["fired"]
        by_fam[fam][1] += v["runs"]
        by_fam[fam][2] += v["passed_unfired"]
        by_fam[fam][3] += v["ended_first"]
        if v["passed_unfired"]:
            lines.append(f"| {v['case'][3:]} | {v['idx']} | {v['what']} {json.dumps(v['kwargs']) if v['kwargs'] else ''} | "
                         f"{v['fired']}/{v['runs']} | {v['passed_unfired']} | {v['ended_first']} | "
                         f"{v['last_why'].most_common(1)[0][0] if v['last_why'] else ''} |")
    lines.insert(1, "per family — fired / runs, moment passed unfired, ended first: " + "; ".join(
        f"{k} {f}/{n}, {pu}, {ef}" for k, (f, n, pu, ef) in by_fam.items()))
    lines.insert(2, f"smoke ban narrowed {len(narrowed)} injections; burned episodes {outcomes.get('burned', 0)}")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()
