#!/usr/bin/env python3
"""Build + verify the hand-designed composite suite (v2).

Every episode is authored individually in python -m cooksim_bench.suite.composite_v2_spec as
    EPISODES = [ {cell, errors: [(label, kwargs) ...], qa: [(kind, question, anchor_m) ...],
                  story: "why these two errors, this order, these questions"} ...]
This script turns each spec into a runnable composite case:
  * errors are looked up in the replay-verified inventory (tmp/interact_eval/error_inventory.json)
    by (cell, label, kwargs) — an error that is not in the inventory is a design error;
  * questions are checked to be well-defined at their anchor (the ground truth is computed by
    replaying the clean prefix), and stored as qa_probe subs with trigger_after = anchor_m;
  * the schedule is ordered by anchor and verified: monotone plate progression, no collapsed
    anchors, distinct labels and distinct question kinds inside the episode;
  * a ledger is written with the story, and the whole suite's diversity is tabulated
    (how many times each (label, kwargs) instance and each question is reused).

    $PY python -m cooksim_bench.suite.design_composite_v2 [--out tmp/interact_eval/cases_composite.json]
"""
import argparse
import collections
import json
import os
from cooksim_bench.paths import ROOT
import sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.suite.gen_composite_cases import derive_anchor, monotone         # noqa: E402
from cooksim_bench.exp_interact import build_episode                          # noqa: E402
from cooksim.core.plan_ir import segment_plan                   # noqa: E402
from cooksim_bench.suite import intent_lib as L  # noqa: E402


def replay_prefix(cell, inject, m):
    state, ro = build_episode(None, "capable_collaborator", cell=cell, inject=inject)
    flat = [(c, n.sid) for n in segment_plan(state.recipe) for c in n.commands]
    ro.cmds = list(flat[:m])
    ro.cap = 600
    term = ("done", "rejected", "illegal-skipped", "orphan-skipped", "failed")
    while True:
        seen = {l.get("i") for l in ro.log if l.get("status") in term}
        if ro.ctrl.skill is None and all(i in seen for i in range(len(ro.cmds))):
            break
        if not ro.step():
            break
    return state, ro, flat


def well_defined(kind, gt):
    if not isinstance(gt, dict) or gt.get("error"):
        return False
    base = kind.split(":", 1)[0]
    if base == "where":
        return bool(gt.get("found"))
    if base == "past_subgoal":
        return gt.get("finished") is not None
    if base == "visited":
        return gt.get("visited") is not None
    return True                                     # scene kinds are always defined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "cases_composite_v5.json"))
    ap.add_argument("--ledger", default="tmp/interact_eval/cases_composite_ledger.md")
    ap.add_argument("--spec", default="composite_v2_spec")
    args = ap.parse_args()
    spec = __import__(args.spec)
    inv = json.load(open("tmp/interact_eval/error_inventory.json"))["cases"]

    def find_error(cell, label, kwargs):
        for c in inv:
            if c["cell"] == cell and c["err"] == label and all(c["kwargs"].get(k) == v for k, v in kwargs.items()) \
                    and len(c["kwargs"]) == len(kwargs):
                return c
        return None

    episodes, problems = [], []
    use_err, use_q = collections.Counter(), collections.Counter()
    per_cell = collections.Counter()
    for ep in spec.EPISODES:
        cell = ep["cell"]
        inject = not cell.endswith("hard_map")
        per_cell[cell] += 1
        name = f"bm_{cell}_{per_cell[cell]}"
        subs, anchors, bad = [], [], []
        for label, kwargs in ep["errors"]:
            c = find_error(cell, label, kwargs)
            if c is None:
                bad.append(f"error not in inventory: {label} {kwargs}")
                continue
            if c["collapses"]:
                bad.append(f"collapsed anchor: {label} {kwargs}")
                continue
            s = {k: v for k, v in c.items() if k not in ("anchor_m", "anchor_plate", "anchor_held", "collapses", "plan_len", "desc")}
            s["name"] = f"{name}_{label}"
            subs.append(s)
            anchors.append({"m": c["anchor_m"], "plate": c["anchor_plate"], "held": c["anchor_held"]})
            use_err[(cell, label, json.dumps(kwargs, sort_keys=True))] += 1
        for kind, question, m in ep["qa"]:
            state, ro, flat = replay_prefix(cell, inject, m)
            try:
                gt = L.ground_truth(kind, ro.game, ro)
            except Exception as e:                                # noqa: BLE001
                gt = {"error": repr(e)}
            if not well_defined(kind, gt):
                bad.append(f"question not well-defined at m={m}: {kind} -> {gt}")
                continue
            from cooksim_bench.e2e_stepin_rollout import inject_key
            key = inject_key(ro.game, ro)
            state0, ro0, _ = replay_prefix(cell, inject, 0)
            if key == inject_key(ro0.game, ro0):
                bad.append(f"question anchor collapses at m={m}: {kind}")
                continue
            subs.append({"name": f"{name}_q_{kind.split(':')[0]}_{m}", "family": "qa_probe", "cell": cell,
                         "err": "no_error", "trigger_after": m, "intent_seq": [], "resume_from": m,
                         "question": question, "gt_kind": kind, "expected_gt": gt, "inject": inject,
                         "auto": False})
            anchors.append({"m": m, "plate": list(key["plate"]), "held": key["plate_held"]})
            use_q[question] += 1
        if bad:
            problems.append((name, bad))
            continue
        # At a shared anchor the QUESTION goes first: it leaves the plate alone and releases
        # in a few ticks, whereas an error that adds to the plate (or a stall whose release
        # coincides with the next plating step) would make a question queued behind it
        # unreachable — 13 of 214 subs in the v1 sweeps died exactly that way (2026-08-26 audit).
        order = sorted(range(len(subs)), key=lambda i: (anchors[i]["m"], 0 if subs[i].get("family") == "qa_probe" else 1))
        subs = [subs[i] for i in order]
        anchors = [anchors[i] for i in order]
        from cooksim_bench.exp_interact import recipe_of
        ordered = bool(getattr(recipe_of(cell), "ordered", False))
        for i in range(len(anchors) - 1):
            if not monotone(anchors[i], anchors[i + 1], ordered=ordered):
                bad.append(f"anchors not monotone: {anchors[i]} -> {anchors[i+1]}")
        # a premature serve ends the episode if it commits -> it must be the last sub
        for i, s in enumerate(subs):
            if s.get("family") != "qa_probe" and s["err"] == "omit_subgoal" and "after" in (s.get("kwargs") or {}) \
                    and i != len(subs) - 1:
                bad.append(f"premature serve is not the last sub (position {i+1} of {len(subs)})")
        labels = [s["err"] for s in subs if s.get("family") != "qa_probe"]
        kinds = [s["gt_kind"].split(":")[0] for s in subs if s.get("family") == "qa_probe"]
        if len(set(labels)) != len(labels):
            bad.append("duplicate error labels in episode")
        if len(set(kinds)) != len(kinds):
            bad.append("duplicate question kinds in episode")
        if bad:
            problems.append((name, bad))
            continue
        for s in subs:
            s["inject"] = inject
        episodes.append({"name": name, "family": "composite", "cell": cell, "err": "composite", "inject": inject,
                         "subs": subs, "order_pattern": "".join("Q" if s.get("family") == "qa_probe" else "E" for s in subs),
                         "anchors": [dict(a, name=s["name"]) for a, s in zip(anchors, subs)],
                         "rationale": ep["story"]})
    tmp_out = args.out + ".tmp"
    json.dump({"schema": "composite-v2", "cases": episodes}, open(tmp_out, "w"), indent=1, default=str)
    os.replace(tmp_out, args.out)                 # atomic: episodes in flight read this file
    reuse_e = sum(1 for v in use_err.values() if v > 1)
    reuse_q = sum(1 for v in use_q.values() if v > 1)
    with open(args.ledger, "w") as f:
        f.write(f"# Composite suite v2 — hand-designed episodes\n\n{len(episodes)} episodes verified"
                f"{', ' + str(len(problems)) + ' rejected' if problems else ''}. "
                f"Distinct error instances used: {len(use_err)} (reused more than once: {reuse_e}); "
                f"distinct questions: {len(use_q)} (reused: {reuse_q}).\n\n")
        for ep in episodes:
            f.write(f"### {ep['name']}  ({ep['order_pattern']})\n\n")
            for a, s in zip(ep["anchors"], ep["subs"]):
                if s.get("family") == "qa_probe":
                    f.write(f"- m={a['m']:3d} plate={a['plate']} held={a['held']} | question: **{s['gt_kind']}** — “{s['question']}” → {json.dumps(s['expected_gt'], default=str)[:120]}\n")
                else:
                    f.write(f"- m={a['m']:3d} plate={a['plate']} held={a['held']} | error: **{s['err']}** {json.dumps(s['kwargs'])} (window {s['window']}, outcome {s['outcome']})\n")
            f.write(f"\n_{ep['rationale']}_\n\n")
        if problems:
            f.write("## Rejected\n\n" + "".join(f"- {n}: {w}\n" for n, w in problems))
    print(f"{len(episodes)} episodes written; {len(problems)} rejected; distinct errors {len(use_err)} (reused {reuse_e}), distinct questions {len(use_q)} (reused {reuse_q})")
    for n, w in problems:
        print("  REJECT", n, w)


if __name__ == "__main__":
    main()
