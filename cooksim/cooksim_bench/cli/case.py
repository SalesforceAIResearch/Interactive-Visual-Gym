"""cooksim-case: inspect the benchmark suite and compose NEW cases from its verified parts.

A case is one order on one kitchen (a *cell*: recipe tier x map) plus a schedule of sub-cases: scheduled
mistakes the engine forces on the cook (`error`), questions the cook asks (`qa_probe`), standing requests such as
"tell me when the meat is done" (`qa_proactive`), and plan changes. The engine fires each sub-case on live kitchen
anchors (what is prepped and plated), never on a timer, so a sub-case carries its anchors; this tool reuses sub-cases
that were verified to fire, and `verify` replays a new case with the scripted cook to prove the schedule fires.

    cooksim-case list                                   # cells, personas, verified mistakes per cell, question kinds
    cooksim-case show b3_medium_nops_medium_map_3       # one case's schedule
    cooksim-case derive --from b3_medium_nops_medium_map_3 --name my_burger_1 \\
        --drop wrong_order --add-error leave_unattended:meat --add-question next_step --out my_cases.json
    cooksim-case verify my_cases.json                   # scripted cook, every cooking order: did everything fire?
    cooksim-episode --cases my_cases.json --case my_burger_1 --persona classic_novice --assistant gemini:gemini-3.8-flash

The persona is chosen per episode (`--persona`), not stored in the case: the same case is played by every persona.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

from cooksim_bench.paths import ROOT, SUITE, BENCHMARK

INVENTORY = os.path.join(BENCHMARK, "error_inventory_v5.json")
ERROR_DESC = {
    "mis_prep": "preps the wrong (off-recipe) item for a component",
    "add_foreign": "scoops an off-recipe ingredient onto the plate",
    "misidentify_assemble": "plates a look-alike item instead of the right one",
    "serve_raw": "plates an item before it is prepared or cooked",
    "wrong_order": "plates two items in the wrong order",
    "omit_subgoal": "serves with an item missing",
    "repeat_step": "plates an item twice",
    "forget_next_step": "stalls and forgets the next step",
    "leave_unattended": "walks away from food on the heat until it smokes",
}


def load_suite(path=None):
    return json.load(open(path or SUITE))


def load_inventory():
    return json.load(open(INVENTORY))["cases"]


def sub_label(s):
    fam = s.get("family")
    if fam == "error":
        kw = s.get("kwargs") or {}
        return f"error  {s.get('err'):22s} {s.get('fn')}({', '.join(f'{k}={v}' for k, v in kw.items())})"
    if fam == "qa_probe":
        return f"question               {s.get('qa_family', '')} {s.get('question', '')!r}"
    if fam == "qa_proactive":
        return f"request                {s.get('question', '')!r}"
    if fam == "plan_change":
        return f"plan change            {s.get('intent_label') or s.get('label') or ''}"
    return f"{fam}"


def cmd_list(a):
    suite = load_suite(a.suite)
    cells = collections.Counter(c["cell"] for c in suite["cases"])
    print("Cells (recipe tier x map) and their cases:")
    for cell, n in sorted(cells.items()):
        print(f"  {cell:26s} {n} cases")
    from cooksim.interact.persona_v2 import ROSTER as PERSONAS
    print("\nCook personas (--persona):")
    for p in PERSONAS:
        print(f"  {p.name:24s} skill={p.skill:6s} trust={p.trust:6s}")
    inv = load_inventory()
    print("\nVerified mistakes per cell (--add-error <err>[:<component>]):")
    by = collections.defaultdict(set)
    for r in inv:
        comp = (r.get("kwargs") or {}).get("component") or (r.get("kwargs") or {}).get("a") or ""
        by[r["cell"]].add(f"{r['err']}{':' + comp if comp else ''}")
    for cell in sorted(by):
        print(f"  {cell:26s} " + ", ".join(sorted(by[cell])))
    print("\nMistake types:")
    for k, v in ERROR_DESC.items():
        print(f"  {k:22s} {v}")
    kinds = collections.Counter()
    for c in suite["cases"]:
        for s in c["subs"]:
            if s["family"] == "qa_probe":
                kinds[s["name"].split("_q_", 1)[1].rsplit("_", 1)[0]] += 1
            elif s["family"] == "qa_proactive":
                kinds["ready:" + str(s.get("item"))] += 1
    print("\nQuestion / request kinds in the suite (--add-question <kind>, --add-request <item>):")
    for k, n in sorted(kinds.items()):
        print(f"  {k:28s} used {n}x")


def cmd_show(a):
    suite = load_suite(a.suite)
    case = next((c for c in suite["cases"] if c["name"] == a.case), None)
    if not case:
        sys.exit(f"no case {a.case!r} in {a.suite or SUITE}")
    print(f"{case['name']}  cell={case['cell']}  errors={case.get('n_errors')}  ruinous={case.get('n_ruinous')}")
    if case.get("rationale"):
        print("  " + case["rationale"])
    print("schedule (fires in this order, each on its live anchor):")
    for i, s in enumerate(case["subs"]):
        anchor = []
        if s.get("prep_floor"):
            anchor.append(f"prep>={s['prep_floor']}")
        if s.get("plate_floor") is not None and s.get("family") == "error":
            anchor.append(f"plate>={s['plate_floor']}")
        print(f"  [{i}] {sub_label(s)}   {' '.join(anchor)}")


def find_sub_in_suite(suite, cell, pred):
    for c in suite["cases"]:
        if c["cell"] != cell:
            continue
        for s in c["subs"]:
            if pred(s):
                return json.loads(json.dumps(s))
    return None


def cmd_derive(a):
    from cooksim_bench.suite import design_composite_v5 as D
    suite = load_suite(a.suite)
    base = next((c for c in suite["cases"] if c["name"] == getattr(a, "from")), None)
    if not base:
        sys.exit(f"no case {getattr(a, 'from')!r} in the suite")
    case = json.loads(json.dumps(base))
    case["name"] = a.name
    case.pop("design_r4", None)
    subs = [s for s in case["subs"] if not any(d in (s.get("err"), s.get("name"), s.get("fn")) or s["name"].endswith(d) for d in a.drop)]
    for s in subs:
        if not s["name"].startswith(a.name):
            s["name"] = a.name + s["name"][len(base["name"]):] if s["name"].startswith(base["name"]) else f"{a.name}_{s['name']}"
    inv = load_inventory()
    for spec in a.add_error:
        err, _, comp = spec.partition(":")
        rows = [r for r in inv if r["cell"] == case["cell"] and r["err"] == err
                and (not comp or comp in ((r.get("kwargs") or {}).get("component"), (r.get("kwargs") or {}).get("a")))]
        if not rows:
            sys.exit(f"no verified mistake {spec!r} for cell {case['cell']} (cooksim-case list shows what exists)")
        row = rows[0]
        sub = D.error_sub(case, row, f"added by cooksim-case from inventory row {row['name']}")
        sub.pop("_pos", None)
        if any(s.get("err") == err and (s.get("kwargs") or {}) == (sub.get("kwargs") or {}) for s in subs):
            sys.exit(f"the case already contains {spec}")
        subs.append(sub)
    for kind in a.add_question:
        s = find_sub_in_suite(suite, case["cell"], lambda s: s["family"] == "qa_probe" and f"_q_{kind}_" in s["name"])
        if not s:
            sys.exit(f"no verified question of kind {kind!r} on cell {case['cell']}")
        s["name"] = f"{a.name}_q_{kind}"
        subs.append(s)
    for item in a.add_request:
        s = find_sub_in_suite(suite, case["cell"], lambda s: s["family"] == "qa_proactive" and s.get("item") == item)
        if not s:
            sys.exit(f"no verified 'tell me when {item} is ready' request on cell {case['cell']}")
        s["name"] = f"{a.name}_p_ready_{item}"
        subs.append(s)
    # fire order. The engine arms sub-case k+1 only after k released, so the schedule must follow the recipe's
    # progress: every forced mistake has a *moment* (how far the dish is when it fires, from the designer's
    # reference run); a premature serve ends the episode and is always last (designer rule C2); questions and
    # requests keep their place relative to the mistakes they followed in the base case, added ones go last.
    def pos(s):
        if s["family"] == "error":
            if s.get("fn") == "premature_serve":
                return 10 ** 6
            try:
                return float(D.moment(s, case["cell"])[0])
            except Exception:                                     # noqa: BLE001
                return 10 ** 5
        return None
    base_names = [b["name"] for b in base["subs"]]
    last = -1.0
    for s in subs:
        p = pos(s)
        if p is not None:
            s["_pos"] = p
    for i, s in enumerate(subs):
        if s.get("_pos") is None:
            if s["name"] in base_names or any(s["name"].endswith(bn[len(base["name"]):]) for bn in base_names):
                prev = [t.get("_pos") for t in subs[:i] if t.get("_pos") is not None]
                s["_pos"] = (prev[-1] if prev else 0.0) + 0.01
            else:
                s["_pos"] = 10 ** 5 + 0.5
    if a.order:
        want = a.order
        for s in subs:
            for i, w in enumerate(want):
                if w in (s.get("err"), s.get("name")) or s["name"].endswith(w):
                    s["_pos"] = i
    subs.sort(key=lambda s: s["_pos"])
    for s in subs:
        s.pop("_pos", None)
    case["subs"] = subs
    case["n_errors"] = sum(1 for s in subs if s["family"] == "error")
    case["n_ruinous"] = sum(1 for s in subs if s["family"] == "error" and s.get("ruinous"))
    case["derived_from"] = base["name"]
    out = {"schema": "cooksim-cases", "derived_with": "cooksim-case derive", "cases": []}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
        out["cases"] = [c for c in out["cases"] if c["name"] != a.name]
    out["cases"].append(case)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.name} to {a.out} ({len(subs)} sub-cases). Next: cooksim-case verify {a.out}")
    for i, s in enumerate(subs):
        print(f"  [{i}] {sub_label(s)}")


def cmd_verify(a):
    """Replay every case with the SCRIPTED cook under each legal cooking order; report what fired."""
    pool = json.load(open(a.cases))
    out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.cases)), "verify")
    os.makedirs(out_dir, exist_ok=True)
    # the scripted cook follows a fixed cooking order and the assistant seat stays silent: no model is called,
    # no key is needed, no browser is opened -- only the engine's schedule is exercised
    env = dict(os.environ, COOKSIM_CASES=os.path.abspath(a.cases), COOKSIM_SCHED="v5", COOKSIM_PACE="tight",
               COOKSIM_GRID=os.path.join(BENCHMARK, "grid.json"), COOKSIM_NO_VIDEO="1", COOKSIM_SILENT_ASSISTANT="1",
               COOKSIM_ASSIST_LATENCY="measured", COOKSIM_GRADER="cook")
    policies = a.policies.split(",")
    allok = True
    for c in pool["cases"]:
        if a.only and c["name"] not in a.only:
            continue
        print(f"== {c['name']} ({c['cell']})")
        for pol in policies:
            out = os.path.join(out_dir, f"{c['name']}__{pol}.json")
            if not os.path.exists(out):
                e = dict(env, COOKSIM_SCRIPTED_COOK=pol)
                with open(out.replace(".json", ".log"), "w") as lf:
                    subprocess.run([sys.executable, "-m", "cooksim_bench.e2e_stepin_rollout", "--case", c["name"],
                                    "--persona-v2", "baseline", "--paradigm", "hardsoft", "--sup-obs", "text_state_map",
                                    "--out", out], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=e, timeout=1500)
            try:
                d = json.load(open(out))
            except Exception as ex:                                   # noqa: BLE001
                print(f"  {pol:12s} FAILED to run ({ex!r}); see {out.replace('.json', '.log')}"); allok = False; continue
            inj = {i.get("idx"): i for i in (d.get("injections") or [])}
            parts = []
            for i, s in enumerate(c["subs"]):
                e = inj.get(i) or {}
                if s["family"] == "error":
                    # the verdict is about the forced mistakes: each must fire AND commit in the world under every
                    # cooking order (the suite's own acceptance rule)
                    if e.get("injected"):
                        committed = (e.get("plate_verdict_at_release") or {}).get("error_committed")
                        st = "fired+committed" if committed else "fired (not committed)"
                        allok &= bool(committed)
                    else:
                        st = f"skipped({e.get('skipped', '?')})"; allok = False
                else:
                    # the scripted cook has no voice: questions, requests and plan changes are only exercised by the
                    # LLM cook. They are listed for completeness; their anchors were verified when the suite was built.
                    st = "n/a (scripted cook does not speak)"
                parts.append(f"[{i}] {s.get('err') if s['family']=='error' else s['family']}: {st}")
            print(f"  {pol:12s} outcome={d.get('outcome'):11s} " + "; ".join(parts))
    print("VERIFIED: every forced mistake fired and committed under every cooking order" if allok else "NOT VERIFIED: a forced mistake did not fire or did not commit under some cooking order; move it or pick another (see the skip reasons)")
    return 0 if allok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cooksim-case", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("list", help="cells, personas, verified mistakes, question kinds"); p.add_argument("--suite", default=None); p.set_defaults(fn=cmd_list)
    p = sp.add_parser("show", help="print a case's schedule"); p.add_argument("case"); p.add_argument("--suite", default=None); p.set_defaults(fn=cmd_show)
    p = sp.add_parser("derive", help="compose a new case from a suite case and verified parts")
    p.add_argument("--from", required=True, help="the suite case to start from"); p.add_argument("--name", required=True, help="the new case's name")
    p.add_argument("--suite", default=None)
    p.add_argument("--drop", nargs="*", default=[], help="sub-cases to remove (error type, or sub-case name suffix)")
    p.add_argument("--add-error", nargs="*", default=[], help="verified mistakes to add: <err>[:<component>]")
    p.add_argument("--add-question", nargs="*", default=[], help="question kinds to add (cooksim-case list)")
    p.add_argument("--add-request", nargs="*", default=[], help="'tell me when <item> is ready' requests to add")
    p.add_argument("--order", nargs="*", default=None, help="explicit order of sub-cases (error types / name suffixes)")
    p.add_argument("--out", required=True, help="case file to write or extend (JSON)"); p.set_defaults(fn=cmd_derive)
    p = sp.add_parser("verify", help="replay a case file with the scripted cook and report what fired")
    p.add_argument("cases"); p.add_argument("--only", nargs="*", default=None); p.add_argument("--out", default=None)
    p.add_argument("--policies", default="template,plate_early,reverse", help="cooking orders to replay"); p.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
