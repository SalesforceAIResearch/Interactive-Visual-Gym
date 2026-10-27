"""vhhome-case: inspect the benchmark's tasks, cases and personas, and write new cases.

A case is an existing household task plus a schedule of mistakes: each mistake is a transform of the task's plan,
anchored on one of its subgoals, that the engine forces at the moment the person reaches that subgoal. The persona
is chosen when the episode is run, not in the case.

  vhhome-case list [tasks|cases|personas|mistakes]
  vhhome-case plan <task>                    the task's subgoals, the anchors a mistake can name
  vhhome-case show <case> [--cases FILE]     a case's schedule and the plan it composes to
  vhhome-case new --name NAME --task TASK --mistake "<class> key=value ..." [--mistake ...] --story "..." --out FILE
  vhhome-case verify FILE                    re-check a case file against the composition rules
"""
import argparse, inspect, json, os, sys

MEANING = {
    "skipped_step": "the person never does that subgoal and carries on with the next one",
    "missing_step": "one of two twin subgoals is left out (the task names two identical targets)",
    "wrong_object": "the person takes the look-alike `wrong` instead and uses it for the rest of the chore",
    "wrong_placement": "the object is put down on `surf` in `room` instead of where the task says; the plan ends there",
    "repetition": "an already finished switch/door/deposit subgoal is done a second time (`at`: after which subgoal)",
    "stall": "the person freezes for `frames` frames after that subgoal (transient) or never resumes (terminal)",
    "navigation_inefficiency": "a detour through `stops` (obj@room,...) before that subgoal",
}
GRADED = list(MEANING)


def _val(k, v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if k == "stops":                                       # stops=sofa@livingroom,bed@bedroom
        return [x.split("@", 1) for x in v.split(",")]
    try:
        return int(v)
    except ValueError:
        return v


def parse_mistake(text):
    toks = text.split()
    if not toks or toks[0] not in GRADED:
        sys.exit(f"mistake must start with one of {GRADED}: {text!r}")
    kw = {}
    for t in toks[1:]:
        if "=" not in t:
            sys.exit(f"expected key=value, got {t!r}")
        k, v = t.split("=", 1)
        kw[k] = _val(k, v)
    return [toks[0], kw]


def cmd_list(a):
    from vhhome import tasks as T, goals as G, mistakes as M
    from vhhome_bench import protocol as P, cases_spec as SPEC
    what = a.what or "all"
    if what in ("tasks", "all"):
        print("TASKS (household chores; scene = VirtualHome apartment)")
        for k, e in T.EPISODES.items():
            print(f"  {k:18} scene {e['scene']}  {len(e['steps']):>2} steps  {G.EPISODES[k]['instruction'][:110]}")
    if what in ("cases", "all"):
        print(f"\nCASES ({len(SPEC.COMPOSITE)} shipped; each with two or more mistakes of different classes)")
        for name, epk, calls, story in SPEC.COMPOSITE:
            print(f"  {name:28} {epk:16} {' + '.join(t for t, _ in calls)}")
            print(f"      {story}")
    if what in ("personas", "all"):
        print("\nPERSONAS  (knows this house, trusts the assistant, speed, alerting wanted, interruption wanted)")
        for n, d in P.PERSONA_DIALS.items():
            print(f"  {n:22} {' / '.join(d)}")
    if what in ("mistakes", "all"):
        print("\nMISTAKE CLASSES and their parameters (anchors are subgoal ids from `vhhome-case plan <task>`)")
        for n in GRADED:
            fn = M.BEHAVIORS[n][0]
            ps = [f"{p.name}" + (f"={p.default!r}" if p.default is not p.empty else "")
                  for p in list(inspect.signature(fn).parameters.values())[1:] if p.name not in ("ask", "pause")]
            print(f"  {n:24} {' '.join(ps)}")
            print(f"      {MEANING[n]}")


def _nodes(epk, calls=()):
    from vhhome_bench import cases as CS
    return CS.build_deviated(epk, [(t, dict(k)) for t, k in calls])


def cmd_plan(a):
    from vhhome import tasks as T, goals as G
    from vhhome.world import pretty_cmd
    if a.task not in T.EPISODES:
        sys.exit(f"unknown task {a.task!r}; vhhome-case list tasks")
    print(f"{a.task}: {G.EPISODES[a.task]['instruction']}\n")
    print("subgoal id              commands")
    for n in _nodes(a.task)[0]:
        print(f"  {n.sid:22} {'; '.join(pretty_cmd(tuple(c)) for c in n.commands)}")
    e = T.EPISODES[a.task]
    print(f"\nthe person starts in the {e.get('init')}; the task's setup (what is out of place):")
    for st in e.get("setup") or []:
        print("  ", pretty_cmd(tuple(st)))


def _find(name):
    from vhhome_bench import cases_spec as SPEC
    for n, epk, calls, story in SPEC.COMPOSITE:
        if n == name:
            return epk, calls, story
    sys.exit(f"unknown case {name!r} (pass --cases FILE for your own)")


def _verify_one(name, epk, calls, story):
    from vhhome_bench import case_checks as CC
    CC.ROWS.clear(); CC.FAILS.clear()
    try:
        CC.verify(name, epk, [(t, dict(k)) for t, k in calls], story)
    except Exception as e:
        return {"admitted": False, "reason": f"verifier raised {type(e).__name__}: {e}"}
    return CC.ROWS[-1] if CC.ROWS else {"admitted": False, "reason": "no verdict"}


def cmd_show(a):
    from vhhome.world import pretty_cmd
    epk, calls, story = _find(a.case)
    print(f"{a.case}  task={epk}\n  {story}\nmistakes:")
    for t, kw in calls:
        print(f"  {t:24} {json.dumps(kw)}")
    clean, ir, _c, _r = _nodes(epk, calls)
    print("\ncomposed plan (* = changed by a mistake):")
    cs = {n.sid for n in clean}
    for n in ir:
        print(f"  {'*' if n.sid not in cs else ' '} {n.sid:22} {'; '.join(pretty_cmd(tuple(c)) for c in n.commands)}")
    row = _verify_one(a.case, epk, calls, story)
    print(f"\nverdict: {'admitted' if row.get('admitted') else 'REJECTED'} — {row.get('reason')}")


def cmd_new(a):
    from vhhome import tasks as T
    if a.task not in T.EPISODES:
        sys.exit(f"unknown task {a.task!r}; vhhome-case list tasks")
    calls = [parse_mistake(m) for m in a.mistake]
    row = _verify_one(a.name, a.task, calls, a.story)
    print(f"{a.name}: {'admitted' if row.get('admitted') else 'REJECTED'} — {row.get('reason')}")
    if not row.get("admitted") and not a.force:
        sys.exit("not written (use --force to write it anyway)")
    doc = {"cases": []}
    if os.path.exists(a.out):
        doc = json.load(open(a.out))
    doc["cases"] = [c for c in doc["cases"] if c["name"] != a.name]
    doc["cases"].append({"name": a.name, "task": a.task, "mistakes": calls, "story": a.story,
                         "admitted": bool(row.get("admitted")), "reason": row.get("reason")})
    json.dump(doc, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(doc['cases'])} case(s)); run it with: vhhome-episode --cases {a.out} --case {a.name} ...")


def cmd_verify(a):
    doc = json.load(open(a.file))
    bad = 0
    for c in doc["cases"]:
        row = _verify_one(c["name"], c["task"], c["mistakes"], c.get("story", ""))
        bad += not row.get("admitted")
        print(f"  {'OK  ' if row.get('admitted') else 'REJ '}{c['name']:28} {c['task']:16} {row.get('reason')}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(prog="vhhome-case", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=None, help="a case file of your own to include")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("list"); p.add_argument("what", nargs="?", choices=["tasks", "cases", "personas", "mistakes"]); p.set_defaults(fn=cmd_list)
    p = sp.add_parser("plan"); p.add_argument("task"); p.set_defaults(fn=cmd_plan)
    p = sp.add_parser("show"); p.add_argument("case"); p.set_defaults(fn=cmd_show)
    p = sp.add_parser("new")
    p.add_argument("--name", required=True); p.add_argument("--task", required=True)
    p.add_argument("--mistake", action="append", required=True, help='"<class> key=value ...", repeatable, in plan order')
    p.add_argument("--story", required=True, help="one sentence: how the run reads to a viewer")
    p.add_argument("--out", required=True, help="case file to write or extend (JSON)")
    p.add_argument("--force", action="store_true", help="write even if the checks reject it")
    p.set_defaults(fn=cmd_new)
    p = sp.add_parser("verify"); p.add_argument("file"); p.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    os.environ.setdefault("VH_MIDLEVEL", "1")
    from vhhome_bench.cli._common import use_cases
    use_cases(a.cases)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
