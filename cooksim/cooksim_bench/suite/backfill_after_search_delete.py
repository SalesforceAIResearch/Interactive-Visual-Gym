#!/usr/bin/env python3
"""Refill the error slots emptied by deleting `search`, and repair the three error sub-cases that never fired.

User rulings 2026-09-17: delete `search`; backfill its 25 slots with VARIED error types that can really fire at that
point, favouring the under-used types; fix the never-firing sub-cases; verify with the LLM cook.

Candidates come from tmp/interact_eval/error_inventory.json (every legal (cell, error, kwargs), replay-checked). A
candidate is admitted into a case only if it passes the rules learned from the round-4/5 firing table:
  R1 same item + heat: no plating-moment error on an item another error in the episode leaves unattended (and no
     leave_unattended on an item another error plates). medium_hard_2 serve_raw(meat) fired 1/27: its moment always
     fell while the unattended meat smoked, and the smoke ban held it until the moment was gone.
  R2 wrong_order(a, b): b is raw or the item right after a. Lettuce (the burrito's last item) was never ready when the
     tortilla or rice was next: master_medium_3 0/27, master_hard_3 2/27.
  R3 a premature serve (omit_subgoal after X) ends the episode, so it stays the last error: a new error's moment must
     come before it, and a case gets at most one.
  R4 no two plating-moment errors at the same anchor; one error type per episode.
  R5 no plating error whose inventory outcome is `won` (a repeat at the last step never spoils the dish).
  R7 no two errors on the same item at the same moment ("x is next on the plate"): when the assistant catches the
     first, the cook plates x properly and the second moment is gone.
  R5b no repeat_step at the last item (the plate is already complete; medium_hard_1 2/2 missed)
  R9 no forget_next_step after the second-to-last item (collapses into "plate complete"; 4 cases 2/2 missed)
  R11 no omission of the second-to-last item when the last item still needs prep (master_medium_4 2/2 missed)
  R8 mis_prep's wrong item is off-recipe (a recipe component cannot be told apart from ordinary prep).
Every rule targets a sub-case that could NOT fire even with an assistant that catches every mistake. Starvation that
only happens when an assistant MISSES a mistake is not guarded: the user ruled (2026-09-17) that a wrong serve stays
terminal and accepted losing later events when the assistant misses one.
Preference: the least-used type in the suite so far, then the moment farthest from the episode's other errors.

    python3 python -m cooksim_bench.suite.backfill_after_search_delete [--apply]      (dry run prints the choices)
"""
import collections
import json
import os
import sys

from cooksim_bench.paths import ROOT  # the CookSim release root
CASES = os.path.join(ROOT, "bench", "v5", "cases_composite_v4.json")
INV = os.path.join(ROOT, "tmp", "interact_eval", "error_inventory.json")
HEAD_BEFORE_DELETE = "fb6d2c3"               # last commit that still had the search sub-cases
BASE_COMMIT = "896c01f"                      # suite after the search deletion, before any backfill (the designer's input)
RUINOUS = {"mis_prep", "misidentify_assemble", "add_foreign", "serve_raw", "wrong_order", "omit_subgoal",
           "repeat_step", "leave_unattended"}          # python -m cooksim_bench.suite.design_composite_v3's own definition
RECIPE = {"easy": ["tomato", "cheese"], "medium": ["bun", "meat", "cheese"], "hard": ["fish", "potato", "potato"],
          "expert": ["steak", "potato", "potato"], "master": ["tortilla", "rice", "steak", "tomato", "lettuce"]}
RAW = {"cheese", "bun", "tortilla"}
PLATING = {"serve_raw", "add_foreign", "omit_subgoal", "wrong_order", "repeat_step", "misidentify_assemble"}
# the three measured never-firing sub-cases: (case, err, kwargs) to replace, same type preferred when a safe one exists
FIX = [("b3_medium_nops_hard_map_2", "serve_raw", {"component": "meat"}),                   # rounds 4-5: 1/27
       ("b3_master_nops_medium_map_3", "wrong_order", {"a": "tortilla", "b": "lettuce"}),  # rounds 4-5: 0/27
       ("b3_master_nops_hard_map_3", "wrong_order", {"a": "rice", "b": "lettuce"}),        # rounds 4-5: 2/27
       # LLM-cook check 2026-09-17 (python -m cooksim_bench.suite.verify_firing_llm): the live kitchen passed the moment in both valid runs
       ("b3_easy_nops_hard_map_2", "forget_next_step", {"after": "assemble(tomato)"}),     # R9
       ("b3_easy_nops_medium_map_1", "forget_next_step", {"after": "assemble(tomato)"}),   # R9
       ("b3_master_nops_medium_map_3", "forget_next_step", {"after": "assemble(tomato)"}), # R9
       ("b3_medium_nops_hard_map_5", "forget_next_step", {"after": "assemble(meat)"}),     # R9
       ("b3_master_nops_hard_map_4", "wrong_order", {"a": "tortilla", "b": "tomato"}),     # R2
       ("b3_master_nops_medium_map_1", "wrong_order", {"a": "tomato", "b": "lettuce"}),    # R2 (last item)
       ("b3_master_nops_medium_map_4", "wrong_order", {"a": "tortilla", "b": "lettuce"}),  # R2
       ("b3_master_nops_medium_map_4", "omit_subgoal", {"sid": "assemble(tomato)"}),       # R11
       ("b3_master_nops_medium_map_5", "wrong_order", {"a": "rice", "b": "tomato"}),       # R2
       ("b3_medium_nops_hard_map_1", "repeat_step", {"sid": "assemble(bun)", "at": "assemble(meat)"})]  # plated back-to-back


def comp_of(err, kw):
    """The item a plating-moment error acts on."""
    for k in ("component", "a"):
        if kw.get(k):
            return {kw[k]} | ({kw["b"]} if err == "wrong_order" else set())
    for k in ("sid", "before", "after"):
        v = kw.get(k) or ""
        if v.startswith("assemble("):
            return {v[len("assemble("):-1]}
    return set()


def is_terminal(err, kw):
    return err == "omit_subgoal" and "after" in kw


def admissible(c, x, errs, anchors_m):
    tier = c["name"].split("_")[1]
    items = RECIPE[tier]
    err, kw = x["err"], x["kwargs"]
    if x.get("collapses"):
        return "collapses to t0"
    if err in {e["err"] for e in errs}:
        return "type already in the episode"
    unattended = {e["kwargs"].get("component") for e in errs if e["err"] == "leave_unattended"}
    plated = set().union(*[comp_of(e["err"], e["kwargs"]) for e in errs if e["err"] in PLATING]) if errs else set()
    if err in PLATING and comp_of(err, kw) & unattended:
        return "R1 plates an item another error leaves unattended"
    if err == "leave_unattended" and kw.get("component") in plated:
        return "R1 leaves unattended an item another error plates"
    if err == "wrong_order":
        a, b = kw["a"], kw["b"]
        if not (b in RAW or (a in items and b in items and items.index(b) == items.index(a) + 1 and b != items[-1])):
            return "R2 the swapped item is late in the recipe"
    terms = [e for e in errs if is_terminal(e["err"], e["kwargs"])]
    if terms:
        mt = min(anchors_m.get(e["name"], 10 ** 6) for e in terms)
        if is_terminal(err, kw) or x["anchor_m"] >= mt:
            return "R3 after the episode's premature serve"
    elif is_terminal(err, kw) and any(anchors_m.get(e["name"], -1) >= x["anchor_m"] for e in errs):
        return "R3 a premature serve must stay last"
    if err in PLATING and any(e["err"] in PLATING and anchors_m.get(e["name"]) == x["anchor_m"] for e in errs):
        return "R4 same plating moment as another error"
    if err in PLATING and x.get("outcome") == "won":
        return "R5 never spoils the dish"
    if err == "repeat_step" and kw.get("at") == f"assemble({items[-1]})":
        return "R5b a repeat at the last item cannot land (the plate is already complete)"
    if err == "forget_next_step" and kw.get("after") == f"assemble({items[-2]})":
        return "R9 forgetting after the second-to-last item collapses into 'plate complete'"
    if err == "omit_subgoal" and kw.get("sid") == f"assemble({items[-2]})" and items[-1] not in RAW:
        return "R11 the omission plates the last item next, which still needs prep"
    # R7 (narrowed): two errors that act on the same item AT THE SAME MOMENT ("x is next on the plate") cannot both fire
    # even when the assistant catches the first: the cook then plates x properly and the second moment is gone.
    NEXT_X = {"serve_raw", "misidentify_assemble", "omit_subgoal", "add_foreign", "wrong_order"}

    def next_items(e_err, e_kw):
        """The plating moments an error uses. An omission of X also plates the item AFTER X, so it uses that moment
        too (easy_*: omit(tomato) plated the cheese and add_foreign(cheese) never got its turn)."""
        if e_err not in NEXT_X or is_terminal(e_err, e_kw):
            return set()
        if e_err == "wrong_order":
            return {e_kw.get("a")}
        if e_kw.get("component"):
            return {e_kw["component"]}
        v = e_kw.get("sid") or e_kw.get("before") or ""
        x_ = v[len("assemble("):-1] if v.startswith("assemble(") else None
        out = {x_} if x_ else set()
        if e_err == "omit_subgoal" and x_ in items and items.index(x_) + 1 < len(items):
            out.add(items[items.index(x_) + 1])
        return out
    if err == "mis_prep" and kw.get("wrong") in items:
        # R8: a wrong item that is ALSO a recipe component cannot be told apart from ordinary prep (the commit verdict
        # needs an excess or an off-path station). The first backfill picked tomato->cheese, fish->potato,
        # steak->potato and the scripted verifier failed 15 of them "fired but not committed" (45/50 cases).
        return "R8 the wrong item is also a recipe component"
    nx = next_items(err, kw)
    if nx and any(nx & next_items(e["err"], e["kwargs"]) for e in errs):
        return "R7 same item at the same moment as another error"
    return None


def remove_sub(c, name, inv_by_key):
    sub = next(s for s in c["subs"] if s["name"] == name)
    c["subs"] = [s for s in c["subs"] if s["name"] != name]
    if any(a["name"] == name for a in c["anchors"]):
        i = [a["name"] for a in c["anchors"]].index(name)
        c["anchors"].pop(i)
        c["order_pattern"] = c["order_pattern"][:i] + c["order_pattern"][i + 1:]
    x = inv_by_key.get((c["cell"], sub["err"], json.dumps(sub["kwargs"], sort_keys=True)))
    head, sep, tail = c["rationale"].partition(". Questions:")
    if x and x["desc"] in head.split("; "):
        c["rationale"] = "; ".join(p for p in head.split("; ") if p != x["desc"]) + sep + tail
    return sub


def add_sub(c, x, note):
    ref = next((s for s in c["subs"] if s.get("family") == "error"), c["subs"][0])
    name = f"{c['name']}_{x['err']}"
    sub = {k: v for k, v in x.items() if not k.startswith("anchor_") and k not in ("collapses", "plan_len", "desc", "name")}
    sub.update(name=name, cell=c["cell"], inject=ref.get("inject", True), persona=ref.get("persona", "capable_collaborator"),
               family="error", ruinous=x["err"] in RUINOUS, **{"backfill_2026_09_17": note})
    anchor = {"m": x["anchor_m"], "plate": list(x["anchor_plate"]), "held": x["anchor_held"], "prep": list(x["anchor_prep"]),
              "name": name}
    pos = sum(1 for a in c["anchors"] if a["m"] <= x["anchor_m"])
    aligned = [s["name"] for s in c["subs"]] == [a["name"] for a in c["anchors"]]
    c["anchors"].insert(pos, anchor)
    c["order_pattern"] = c["order_pattern"][:pos] + "E" + c["order_pattern"][pos:]
    c["subs"].insert(pos if aligned else len(c["subs"]), sub)
    head, sep, tail = c["rationale"].partition(". Questions:")
    c["rationale"] = "; ".join([p for p in head.split("; ") if p] + [x["desc"]]) + sep + tail
    c["n_errors"] = sum(1 for s in c["subs"] if s.get("family") == "error")
    c["n_ruinous"] = sum(1 for s in c["subs"] if s.get("family") == "error" and s.get("ruinous"))


def main():
    apply = "--apply" in sys.argv
    import subprocess
    before = {c["name"]: c for c in json.loads(subprocess.run(["git", "show", f"{HEAD_BEFORE_DELETE}:bench/v5/cases_composite_v4.json"],
                                                           cwd=ROOT, capture_output=True, text=True, check=True).stdout)["cases"]}
    d = json.loads(subprocess.run(["git", "show", f"{BASE_COMMIT}:bench/v5/cases_composite_v4.json"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout)
    inv = json.load(open(INV))["cases"]
    inv_by_key = {(x["cell"], x["err"], json.dumps(x["kwargs"], sort_keys=True)): x for x in inv}
    use = collections.Counter(s["err"] for c in d["cases"] for s in c["subs"] if s.get("family") == "error")
    slots = [(c["name"], "search", None) for c in d["cases"] if any(s.get("err") == "search" for s in before[c["name"]]["subs"])]
    slots += [(n, e, kw) for n, e, kw in FIX]
    by_name = {c["name"]: c for c in d["cases"]}
    tier_use = collections.Counter()
    plan = []
    for name, old_err, old_kw in slots:
        c = by_name[name]
        if old_kw is not None:
            old = next(s for s in c["subs"] if s.get("err") == old_err and s.get("kwargs") == old_kw)
            remove_sub(c, old["name"], inv_by_key)
            use[old_err] -= 1
        errs = [s for s in c["subs"] if s.get("family") == "error"]
        anchors_m = {a["name"]: a["m"] for a in c["anchors"]}
        ok, rejected = [], collections.Counter()
        measured_bad = {(n_.split("_nops_")[1] if "_nops_" in n_ else n_, e_, json.dumps(k_, sort_keys=True))
                        for n_, e_, k_ in FIX}          # (map, err, kwargs) a repair removed as never firing
        tier_map = name.split("_nops_")[1]
        for x in inv:
            if x["cell"] != c["cell"]:
                continue
            if (tier_map, x["err"], json.dumps(x["kwargs"], sort_keys=True)) in measured_bad or \
               (x["cell"].split("_nops_")[0], x["err"], json.dumps(x["kwargs"], sort_keys=True)) in {
                   (n_[3:].split("_nops_")[0], e_, json.dumps(k_, sort_keys=True)) for n_, e_, k_ in FIX
                   if n_.split("_")[1] == name.split("_")[1] and n_.split("_nops_")[1].rsplit("_", 1)[0] == name.split("_nops_")[1].rsplit("_", 1)[0]}:
                rejected["measured never firing in this recipe and map"] += 1
                continue
            why = admissible(c, x, errs, anchors_m)
            if why:
                rejected[why] += 1
            else:
                ok.append(x)
        if not ok:
            plan.append((name, old_err, old_kw, None, dict(rejected)))
            continue
        other_m = [anchors_m.get(e["name"], 0) for e in errs] or [0]

        def score(x):
            # a repair mildly prefers its old type; variety across the suite and within the tier outweighs it
            keep_type = 6 if (old_kw is not None and x["err"] == old_err) else 0
            return (use[x["err"]] + 8 * tier_use[(tier, x["err"])] - keep_type,
                    -min(abs(x["anchor_m"] - m) for m in other_m), x["name"])
        tier = name.split("_")[1]
        pick = min(ok, key=score)
        use[pick["err"]] += 1
        tier_use[(tier, pick["err"])] += 1
        note = ({"replaces": "search (deleted 2026-09-17)"} if old_kw is None else
                {"replaces": {"err": old_err, "kwargs": old_kw}, "why": "never fired with LLM cooks (rounds 4-5)"})
        add_sub(c, pick, note)
        plan.append((name, old_err, old_kw, pick, None))
    for name, old_err, old_kw, pick, rej in plan:
        was = "search" if old_kw is None else f"{old_err}{old_kw}"
        if pick is None:
            print(f"  {name:30s} {was:40s} -> NO ADMISSIBLE CANDIDATE {rej}")
        else:
            print(f"  {name:30s} {was:40s} -> {pick['err']:16s} {json.dumps(pick['kwargs']):46s} m={pick['anchor_m']}")
    print("\nerror types in the suite after:", dict(sorted(use.items(), key=lambda kv: kv[1])), "total", sum(use.values()))
    if apply:
        empty = [p[0] for p in plan if p[3] is None]
        if empty:
            print("left unfilled (no candidate can fire there without breaking a rule):", empty)
        open(CASES, "w").write(json.dumps(d, indent=1, ensure_ascii=True))
        print("written", CASES)


if __name__ == "__main__":
    main()
