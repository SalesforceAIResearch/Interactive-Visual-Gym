#!/usr/bin/env python3
"""Route-robustness verifier for the benchmark cases under scheduler v5.

    $PY python -m cooksim_bench.suite.verify_cases_v5 [--cases tmp/interact_eval/cases_composite_v3.json] [--policies template,plate_early,reverse]
                                 [--out tmp/verify_v5] [--workers 24] [--only b3_easy_nops_medium_map_1,...]

For every case and every legal cooking order (the scripted verification cook, no LLM, silent
assistant, no browser) it runs the episode and records, per planned sub-case: fired / skipped
(with the reason) / committed (the plate verdict for errors), plus the outcome. A case passes
when every planned error fires and commits under every policy and the questions/requests are
asked. The summary is written as JSON + a markdown table.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
from cooksim_bench.paths import ROOT  # the CookSim release root
PY = sys.executable


def run_one(case, policy, out_dir, env):
    out = os.path.join(out_dir, f"{case}__{policy}.json")
    if os.path.exists(out):
        return out
    e = dict(env)
    e["COOKSIM_SCRIPTED_COOK"] = policy
    log = out.replace(".json", ".log")
    with open(log, "w") as lf:
        subprocess.run([PY, "-m", "cooksim_bench.e2e_stepin_rollout", "--case", case, "--persona-v2", "baseline",
                        "--paradigm", "hardsoft", "--sup-obs", "text_state_map", "--out", out],
                       cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=e, timeout=1500)
    return out


def summarize(pool, out_dir, policies):
    rows, per_case = [], {}
    for c in pool["cases"]:
        name = c["name"]
        planned = [(i, s.get("family"), s.get("err") or s.get("gt_kind") or s.get("family")) for i, s in enumerate(c["subs"])]
        pc = {"case": name, "cell": c["cell"], "planned": planned, "policies": {}}
        for pol in policies:
            f = os.path.join(out_dir, f"{name}__{pol}.json")
            if not os.path.exists(f):
                pc["policies"][pol] = {"missing": True}
                continue
            try:
                d = json.load(open(f))
            except Exception as ex:
                pc["policies"][pol] = {"error": repr(ex)[:80]}
                continue
            inj = {i.get("idx"): i for i in (d.get("injections") or [])}
            subs = []
            ok = True
            for i, fam, kind in planned:
                e = inj.get(i)
                if e is None:
                    subs.append((i, fam, kind, "MISSING", None)); ok = False; continue
                if fam == "error":
                    pv = e.get("plate_verdict_at_release") or {}
                    committed = pv.get("error_committed")
                    st = "fired" if e.get("injected") else f"skipped:{e.get('skipped')}"
                    if not e.get("injected"):
                        ok = False
                    subs.append((i, fam, kind, st, committed))
                elif fam in ("qa_probe", "qa_proactive"):
                    asked = (e.get("qa") or {}).get("asked_at") is not None
                    st = "asked" if asked else f"skipped:{e.get('skipped')}"
                    if not asked:
                        ok = False
                    subs.append((i, fam, kind, st, None))
                else:
                    st = "fired" if e.get("injected") else f"skipped:{e.get('skipped')}"
                    if not e.get("injected"):
                        ok = False
                    subs.append((i, fam, kind, st, None))
            end_t = d.get("ticks")
            if d.get("outcome") != "won":
                subs2 = []
                for (i, fam, kind, st, committed) in subs:
                    e = inj.get(i) or {}
                    sk = (e.get("report_keys") or {}).get("injection_skipped") or {}
                    if st.startswith("skipped:anchor_never_reached") and sk.get("t") == end_t:
                        st = "ended"                      # the episode was lost before this sub-case's turn
                    subs2.append((i, fam, kind, st, committed))
                subs = subs2
                ok = all(st in ("fired", "asked", "ended") and committed is not False for _i, _f, _k, st, committed in subs)
            # pass criterion (ruling 2026-09-09: a sub-case whose precondition the cook's own route never
            # creates is correctly skipped): no MISSING record, every FIRED error committed (or undecidable)
            ok = all(st != "MISSING" and committed is not False for _i, _f, _k, st, committed in subs)
            # round-3 audit: a takeover must not issue commands this kitchen refuses. Count refusals
            # inside the injected blocks (the cook's own refused commands are its business) — any
            # such command fails the case, which is how the old whole-recipe omission block hid.
            block_bad = 0
            for e in (d.get("injections") or []):
                if e.get("family") != "error" or not e.get("injected"):
                    continue
                rk = e.get("report_keys") or {}
                t0, t1 = rk.get("injected_at") or 0, rk.get("released_at")
                for l in d.get("actions") or []:
                    if str(l.get("kind", "")).startswith(("vlm_fix", "scripted")):
                        continue
                    ti = l.get("t_issue") or 0
                    if ti >= t0 and (t1 is None or ti <= t1) and l.get("status") in (
                            "rejected", "failed", "illegal-skipped", "orphan-skipped"):
                        block_bad += 1
            if block_bad:
                ok = False
            n_err = sum(1 for _i, f, _k, _s, _c in subs if f == "error")
            n_fired = sum(1 for _i, f, _k, st, _c in subs if f == "error" and st == "fired")
            pc["policies"][pol] = {"outcome": d.get("outcome"), "ticks": d.get("ticks"), "aborted": d.get("aborted"),
                                   "subs": subs, "all_fired": ok, "errors_planned": n_err, "errors_fired": n_fired,
                                   "block_bad": block_bad,
                                   "bad_cmds": sum(1 for l in d.get("actions", []) if l.get("status") in ("rejected", "failed", "illegal-skipped", "orphan-skipped"))}
        per_case[name] = pc
    return per_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="tmp/interact_eval/cases_composite_v3.json")
    ap.add_argument("--policies", default="template,plate_early,reverse")
    ap.add_argument("--out", default="tmp/verify_v5")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", default=None)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    pool = json.load(open(os.path.join(ROOT, a.cases)))
    policies = a.policies.split(",")
    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    names = [c["name"] for c in pool["cases"]]
    if a.only:
        names = [n for n in names if n in set(a.only.split(","))]
    if not a.summarize_only:
        env = dict(os.environ)
        env.update(COOKSIM_PACE="tight", COOKSIM_CASES=a.cases, COOKSIM_SILENT_ASSISTANT="1", COOKSIM_NO_VIDEO="1",
                   COOKSIM_ASSIST_LATENCY="off", COOKSIM_SCHED="v5")
        env.setdefault("GEMINI_API_KEY", "x"); env.setdefault("OPENAI_API_KEY", "x")
        jobs = [(n, p) for n in names for p in policies]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(lambda j: run_one(j[0], j[1], out_dir, env), jobs))
    per_case = summarize(pool, out_dir, policies)
    json.dump(per_case, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    # markdown
    lines = ["| case | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    n_pass = 0
    fail_reasons = collections.Counter()
    for name in names:
        pc = per_case[name]
        cells = []
        all_ok = True
        for pol in policies:
            r = pc["policies"].get(pol, {})
            if "subs" not in r:
                cells.append("missing"); all_ok = False; continue
            marks = []
            for i, fam, kind, st, committed in r["subs"]:
                if st == "fired":
                    marks.append(f"{kind}:{'C' if committed else ('?' if committed is None else 'x')}")
                    if committed is False:
                        all_ok = False; fail_reasons[f"{kind} fired but not committed"] += 1
                elif st in ("asked", "ended"):
                    marks.append(f"{kind}:{st}")
                else:
                    marks.append(f"{kind}:{st}")
                    fail_reasons[f"{fam}:{st.split(':')[1][:24] if ':' in st else st}"] += 1
                    if st == "MISSING":
                        all_ok = False
            if r.get("block_bad"):
                marks.append(f"BLOCK-REFUSED x{r['block_bad']}")
                all_ok = False
                fail_reasons["takeover issued commands the kitchen refused"] += 1
            cells.append(f"{r['outcome']} t{r['ticks']} [{r.get('errors_fired')}/{r.get('errors_planned')} errors] " + " ".join(marks))
        n_pass += all_ok
        lines.append(f"| {name}{' ✅' if all_ok else ' ❌'} | " + " | ".join(cells) + " |")
    tot_err = sum(r.get("errors_planned", 0) for pc in per_case.values() for r in pc["policies"].values() if "subs" in r)
    tot_fired = sum(r.get("errors_fired", 0) for pc in per_case.values() for r in pc["policies"].values() if "subs" in r)
    lines.insert(0, f"# verify v5 — {n_pass}/{len(names)} cases: every fired error commits, nothing missing (all policies); "
                    f"errors offered {tot_fired}/{tot_err} = {tot_fired / max(1, tot_err):.0%}\n\nskip / non-commit reasons: {dict(fail_reasons.most_common())}\n")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines[:3]))
    print(f"wrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
