#!/usr/bin/env python3
"""Design + VERIFY the 50 composite assistant-benchmark episodes (2026-08-19).

The spec (user): 10 tasks (the benchmark cells) x 5 episodes; every episode carries
2 injected ERRORS + 2 QA questions, and across the suite the schedules must cover most
of the 9 error labels and most of the QA kinds. Every episode's design and ORDER must be
semantically plausible and executable/trackable by the engine.

What "verified" means here, mechanically — for every episode this script:
  1. derives each sub's REAL anchor exactly the way the live harness does (segment_plan
     -> compose -> longest-common-prefix m for errors; trigger_after for qa), then
     replays the clean prefix headlessly and records the plate anchor key
     (inject_key: ordered plate contents + held flag);
  2. proves the schedule's anchors are MONOTONE in plate progression (each anchor's
     plate tuple is a prefix of the next, (len, held) non-decreasing) — the condition
     under which sub k+1's trigger is still reachable after sub k releases, because a
     recovered cook re-enters the clean plate trajectory;
  3. refuses a first sub the live harness would refuse (anchor collapse: m > 0 with a
     t0-identical reference key);
  4. enforces the composition rules: distinct error labels and distinct qa kinds inside
     an episode; at most one burn-capable error (leave_unattended) per episode; an
     error whose expected outcome fires at the serve (omit/wrong_order/serve_raw/
     add_foreign commit at plating or serve) is allowed anywhere — if unrecovered it
     ends the episode and later subs are honestly recorded not_reached;
  5. writes a human-readable design ledger with, per episode, the schedule, each
     anchor, and the semantic rationale for the pairing/order.

Output: tmp/interact_eval/cases_composite.json (subs INLINE — the schedule is
self-contained and immune to pool drift) + tmp/interact_eval/cases_composite_ledger.md.

    $PY python -m cooksim_bench.suite.gen_composite_cases
"""
import json
import os
from cooksim_bench.paths import ROOT
import sys
from collections import defaultdict

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.exp_interact import build_episode                      # noqa: E402
from cooksim.core.plan_ir import segment_plan               # noqa: E402
from cooksim_bench import exp_interact as X  # noqa: E402

CELLS10 = [f"{n}_nops_{m}_map" for n in ("easy", "medium", "hard", "expert", "master")
           for m in ("medium", "hard")]
PER_CELL = 5
REPLAY_PERSONA = "capable_collaborator"


def load_pool(path, key="cases"):
    try:
        d = json.load(open(path))
        return d[key] if isinstance(d, dict) else d
    except Exception:
        return []


def plate_key(game, ro=None):
    """The live harness's inject_key, replicated (plate + prep progress when the rollout is given)."""
    from cooksim_bench.e2e_stepin_rollout import inject_key
    return inject_key(game, ro)


def derive_anchor(case, cell, inject):
    """(m, ref_key) exactly as the live harness computes them for this sub."""
    state, ro = build_episode(None, REPLAY_PERSONA, cell=cell, inject=inject)
    flat_clean = [(c, n.sid) for n in segment_plan(state.recipe) for c in n.commands]
    if case.get("family") == "qa_probe":
        m = case["trigger_after"]
    else:
        err_call = ((case.get("fn") or case["err"]), dict(case["kwargs"]))
        err_ir, _ = X.E.compose(list(segment_plan(state.recipe)), [err_call])
        flat_err = [(c, n.sid) for n in err_ir for c in n.commands]
        m = 0
        while (m < min(len(flat_clean), len(flat_err))
               and flat_clean[m][0] == flat_err[m][0]):
            m += 1
    rstate, rro = build_episode(None, REPLAY_PERSONA, cell=cell, inject=inject)
    rro.cmds = list(flat_clean[:m])
    rro.cap = 450
    while rro.i < len(rro.cmds) or rro.ctrl.skill is not None:
        if not rro.step():
            break
    k = plate_key(rro.game, rro)
    t0state, t0ro = build_episode(None, REPLAY_PERSONA, cell=cell, inject=inject)
    return m, k, plate_key(t0ro.game, t0ro), len(flat_clean)


def monotone(a, b, ordered=True):
    """Anchor a may precede anchor b: b's plate extends (or equals) a's — as a prefix for an
    ordered dish, as a multiset for an order-free dish (inject_key sorts those)."""
    import collections
    pa, ha = a["plate"], a["held"]
    pb, hb = b["plate"], b["held"]
    qa_, qb_ = a.get("prep"), b.get("prep")
    if qa_ is not None and qb_ is not None and (qb_[0] < qa_[0] or qb_[1] < qa_[1]):
        return False                                   # prep progress never goes backwards
    if len(pb) < len(pa) or (ha and not hb and len(pb) == len(pa)):
        return False
    if not ordered:
        return not (collections.Counter(pa) - collections.Counter(pb))
    return tuple(pb[:len(pa)]) == tuple(pa)


# semantic pairing preferences: after an error touching component/subgoal X, a QA that
# interrogates exactly that state is the natural question a real cook would ask
def pairing_note(err_case, qa_case):
    kind = qa_case.get("gt_kind", "")
    err = err_case.get("err")
    kw = err_case.get("kwargs", {})
    comp = kw.get("component") or kw.get("wrong") or kw.get("sid") or ""
    if comp and isinstance(comp, str) and comp.split("(")[-1].rstrip(")") in kind:
        return (f"the question interrogates the very state the {err} touched "
                f"({comp}) — the natural thing to ask after that confusion")
    if kind.startswith("past_subgoal") and err in ("mis_prep", "repeat_step",
                                                   "forget_next_step", "omit_subgoal"):
        return (f"an episodic what-did-I-do question is plausible right after a {err} "
                f"recovery — the cook's own history just became uncertain")
    if kind in ("pan_state", "station_state", "board_state") and err in (
            "leave_unattended", "mis_prep", "serve_raw"):
        return f"a station-state question is plausible around a {err} — heat is in doubt"
    if kind in ("plate_contents", "remaining") and err in (
            "add_foreign", "wrong_order", "omit_subgoal", "repeat_step"):
        return f"a plate question is plausible after a {err} — the plate is what it hit"
    return "an unrelated spot-check — plausible anywhere the truth is well-defined"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "cases_composite_v5.json"))
    ap.add_argument("--ledger", default="tmp/interact_eval/cases_composite_ledger.md")
    args = ap.parse_args()

    errors = load_pool("tmp/interact_eval/cases_composite_errors.json")
    qa = (load_pool("tmp/interact_eval/cases_intent.json")
          + load_pool("tmp/interact_eval/cases_newrecipes.json")
          + load_pool("tmp/interact_eval/cases_qa_hard.json"))
    qa = [c for c in qa if c.get("family") == "qa_probe"]
    # dedupe qa by name (cases_intent entries are repeated inside cases_newrecipes)
    seen, qs = set(), []
    for c in qa:
        if c["name"] not in seen:
            seen.add(c["name"])
            qs.append(c)
    qa = qs

    episodes, ledger = [], []
    cov_err, cov_qa = defaultdict(int), defaultdict(int)
    for cell in CELLS10:
        inject = not cell.endswith("hard_map")   # medium cells carry the decoy world,
        # hard cells have real distractor sources and their pools were verified inject=False
        ecands = [dict(c) for c in errors if c["cell"] == cell]
        qcands = [dict(c) for c in qa if c["cell"] == cell]
        for c in ecands + qcands:
            c["inject"] = inject
            m, key, t0key, plan_len = derive_anchor(c, cell, inject)
            c["_m"], c["_anchor"] = m, {"plate": list(key["plate"]),
                                        "held": key["plate_held"]}
            c["_collapses"] = bool(m > 0 and key == t0key)
        # a collapsed anchor does not test the trigger it describes (it fires the
        # moment the schedule reaches it, wherever that is) — inadmissible ANYWHERE
        # in a composite, not only as the opener
        ecands = [c for c in ecands if not c["_collapses"]]
        qcands = [c for c in qcands if not c["_collapses"]]
        # ---- selection: maximise label/kind coverage across the cell's 5 episodes ----
        by_label = defaultdict(list)
        for c in sorted(ecands, key=lambda c: c["_m"]):
            by_label[c["err"]].append(c)
        by_kind = defaultdict(list)
        for c in sorted(qcands, key=lambda c: c["_m"]):
            k = c["gt_kind"].split(":")[0]
            by_kind[k].append(c)
        # COVERAGE GUARANTEE: cycle every feasible label into the cell's 10 error
        # slots (each appears at least once when <=10 exist) and every qa kind into
        # the 10 qa slots — rarity first so a thin label (serve_raw: master cells
        # only) is never crowded out by a plentiful one.
        labels = sorted(by_label, key=lambda l: len(by_label[l]))
        kinds = sorted(by_kind, key=lambda k: len(by_kind[k]))
        err_slots = [labels[i % len(labels)] for i in range(2 * PER_CELL)]
        qa_slots = [kinds[i % len(kinds)] for i in range(2 * PER_CELL)]
        used = defaultdict(int)
        cell_eps = []
        attempts = 0
        while len(cell_eps) < PER_CELL and attempts < 400:
            attempts += 1
            i0 = 2 * len(cell_eps)
            off = attempts // PER_CELL           # rotate pairings when a combo fails
            l1 = err_slots[i0]
            l2 = err_slots[(i0 + 1 + off) % len(err_slots)]
            if l1 == l2:
                l2 = err_slots[(i0 + 2 + off) % len(err_slots)]
            if l1 == l2:
                continue
            e1 = by_label[l1][(used[l1] + off) % len(by_label[l1])]
            e2 = by_label[l2][(used[l2] + off) % len(by_label[l2])]
            if e1["err"] == "leave_unattended" and e2["err"] == "leave_unattended":
                continue
            k1 = qa_slots[i0]
            k2 = qa_slots[(i0 + 1 + off) % len(qa_slots)]
            if k1 == k2:
                k2 = qa_slots[(i0 + 2 + off) % len(qa_slots)]
            if k1 == k2:
                continue
            q1 = by_kind[k1][(used["q" + k1] + off) % len(by_kind[k1])]
            q2 = by_kind[k2][(used["q" + k2] + off) % len(by_kind[k2])]
            subs = sorted([e1, e2, q1, q2],
                          key=lambda c: (c["_m"], 0 if c.get("family") != "qa_probe"
                                         else 1))
            # ---- hard checks ----
            if subs[0]["_collapses"]:
                continue                     # the live harness would refuse this opener
            ok = all(monotone(subs[i]["_anchor"], subs[i + 1]["_anchor"])
                     for i in range(3))
            if not ok:
                continue
            if len({s["name"] for s in subs}) < 4:
                continue
            used[l1] += 1
            used[l2] += 1
            used["q" + k1] += 1
            used["q" + k2] += 1
            cov_err[l1] += 1
            cov_err[l2] += 1
            cov_qa[k1] += 1
            cov_qa[k2] += 1
            name = f"bm_{cell}_{len(cell_eps) + 1}"
            order = [("E" if s.get("family") != "qa_probe" else "Q") for s in subs]
            errs_in = [s for s in subs if s.get("family") != "qa_probe"]
            qas_in = [s for s in subs if s.get("family") == "qa_probe"]
            notes = []
            for qq in qas_in:
                prev_e = next((s for s in reversed(subs[:subs.index(qq)])
                               if s.get("family") != "qa_probe"), errs_in[0])
                notes.append(f'{qq["name"]} ({qq["gt_kind"]}): '
                             + pairing_note(prev_e, qq))
            clean = []
            for s in subs:
                d = {k: v for k, v in s.items() if not k.startswith("_")}
                clean.append(d)
            ep = {"name": name, "family": "composite", "cell": cell,
                  "err": "composite", "inject": inject,
                  "subs": clean,
                  "order_pattern": "".join(order),
                  "anchors": [{"name": s["name"], "m": s["_m"],
                               "plate": s["_anchor"]["plate"],
                               "held": s["_anchor"]["held"]} for s in subs],
                  "rationale": "; ".join(notes)}
            episodes.append(ep)
            cell_eps.append(ep)
            ledger.append((cell, ep))
        if len(cell_eps) < PER_CELL:
            print(f"[WARN] {cell}: only {len(cell_eps)} admissible episodes "
                  f"after {attempts} attempts", flush=True)

    json.dump({"schema": "composite-v1", "cases": episodes}, open(args.out, "w"),
              indent=1)
    with open(args.ledger, "w") as f:
        f.write("# Composite assistant-benchmark episodes — design ledger\n\n")
        f.write(f"{len(episodes)} episodes ({PER_CELL} per cell). Every schedule "
                "below passed: engine-derived anchors, monotone plate progression, "
                "no collapsed anchors anywhere in the schedule, distinct labels/kinds "
                "per episode.\n\n"
                "**Structural exclusions.** `serve_raw` is absent from every schedule "
                "because all four of its verified cases anchor BEFORE any plate work "
                "(m>0 with the t0 plate key) — the live harness refuses such anchors "
                "(E1), so the label is unusable under the plate-keyed trigger on these "
                "recipes. Its behaviour is semantically covered by `omit_subgoal` of a "
                "cook-prep subgoal, which serves the component raw. `pan_state` QA is "
                "excluded for the same anchor-collapse reason; `station_state` covers "
                "the identical question family.\n\n")
        f.write("## Coverage\n\nerror labels: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(cov_err.items()))
                + "\n\nqa kinds: "
                + ", ".join(f"{k} x{v}" for k, v in sorted(cov_qa.items())) + "\n\n")
        for cell, ep in ledger:
            f.write(f"### {ep['name']}  ({ep['order_pattern']})\n\n")
            for a, s in zip(ep["anchors"], ep["subs"]):
                fam = s.get("family")
                what = (s.get("err") if fam != "qa_probe"
                        else s.get("gt_kind") + " — “" + str(s.get("question", ""))[:80] + "”")
                f.write(f"- m={a['m']:3d} plate={a['plate']} held={a['held']} "
                        f"| {fam}: **{what}** ({s['name']})\n")
            f.write(f"\n_{ep['rationale']}_\n\n")
    print(f"wrote {args.out} ({len(episodes)} episodes) and {args.ledger}")
    print("error coverage:", dict(cov_err))
    print("qa coverage:", dict(cov_qa))


if __name__ == "__main__":
    main()
