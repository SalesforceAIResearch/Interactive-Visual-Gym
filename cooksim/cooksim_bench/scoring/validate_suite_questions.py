#!/usr/bin/env python3
"""Static validator for the questions and standing requests of a composite suite.

Pre-formal-run audit 2026-09-19 (tmp/suite_audit/FIX_PLAN.md A4): the suite asks "have I been over to the pot?"
in kitchens that have no pot, "did I chop the fish?" in recipes that put the fish straight into the fryer, and
"tell me when the potato is done" in recipes with two potatoes. None of these can be answered well, and several
cannot be graded at all. This validator reads each case's kitchen (built exactly as the live harness builds it:
exp_interact.build_episode with the case's `inject` flag) and its recipe plan (plan IR), and flags every question
or request that is unanswerable, ungradable or ambiguous as worded.

Checks (severity):
  R1 referent  (error)  a station/place named by the kind or the wording does not exist in this kitchen
                         (visit(pot) / "the pot" / assistant_said:fryer with no fryer ...)
  R2 item      (error)  the item is not in this recipe (order/after_ref/ready_in/proactive about an absent item)
  V1 verb      (error)  "chop" about an item the recipe never chops (fish, meat, rice: they go on the heat), or
                         "on the heat"/"cook" about an item the recipe never cooks; chop(X)/heat(X) specs likewise
  P1 process   (error)  a cooking question/request (ready_in, elapsed, proactive, longest_on, past_done, burn_first)
                         in a recipe with nothing on the heat, or about an item that is never cooked
  A1 after_ref (error)  "the thing after X" when X is the recipe's last item (nothing can come after it)
  K1 kind      (error)  gt_kind has no truth function
  T1 trivial   (warn)   order fixed by the recipe itself (prep(X) before assemble(Y) with X plated before Y on an
                         ordered dish): the answer does not depend on what happened
  M1 ambiguous (warn)   a request/question about "the X" in a recipe with several units of X (two potatoes):
                         the wording should name the batch or station (grading binds it at the ask, A6)
  W1 wording   (warn)   the question wording names a different item than its gt_kind

    $PY python -m cooksim_bench.scoring.validate_suite_questions [bench/v5/cases_composite_v4.json] [--json out.json]
Exit status 1 when any error is found.
"""
import argparse
import collections
import json
import os
from cooksim_bench.paths import ROOT
import re
import sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))

from cooksim_bench.scoring.qa_world import visit_kind, terrain_kind, chopped_items          # noqa: E402

STATION_WORDS = ("cutting board", "chopping board", "board", "pan", "skillet", "pot", "fryer", "deep fryer", "oven",
                 "trash", "bin", "sink")
COOK_KINDS = ("ready_in", "ready", "elapsed", "longest_on", "past_done", "burn_first")
_KITCHEN = {}


def kitchen(cell, inject):
    """{'kinds': set of place kinds on the map, 'recipe': Recipe, 'chopped': set, 'cooked': set, 'counts': Counter}"""
    key = (cell, bool(inject))
    if key not in _KITCHEN:
        from cooksim_bench.exp_interact import build_episode
        _state, ro = build_episode(None, "capable_collaborator", cell=cell, inject=inject)
        g = ro.game
        kinds = {terrain_kind(getattr(t, "value", t)) for row in g.layout.grid for t in row}
        rec = _state.recipe
        cooked = {n for n, s in rec.contents if getattr(s, "value", s) == "cooked"}
        _KITCHEN[key] = {"kinds": kinds, "recipe": rec, "chopped": chopped_items(rec), "cooked": cooked,
                         "items": [n for n, _ in rec.contents], "counts": collections.Counter(n for n, _ in rec.contents),
                         "ordered": bool(getattr(rec, "ordered", False))}
    return _KITCHEN[key]


def _spec(spec):
    m = re.match(r"^(\w+)(?:\((.*)\))?$", spec.strip())
    return (m.group(1), (m.group(2) or "").strip()) if m else (spec, "")


def check_sub(sub, case):
    """[(code, severity, message)] for one qa_probe / qa_proactive sub-case."""
    out = []
    K = kitchen(case["cell"], case.get("inject", True))
    fam = sub.get("family")
    kind = sub.get("gt_kind") or ""
    q = str(sub.get("question") or "")
    ql = q.lower()
    base, arg = (kind.split(":", 1) + [""])[:2] if kind else ("", "")

    def err(code, msg):
        out.append((code, "error", msg))

    def warn(code, msg):
        out.append((code, "warn", msg))

    # --- K1: a truth function exists
    if fam == "qa_probe":
        try:
            from cooksim_bench.suite import gen_intent_cases as GI
            if base not in GI.GT_PARAM and kind not in GI.GT:
                err("K1", f"no truth function for gt_kind {kind!r}")
        except Exception:                                  # noqa: BLE001
            pass
    # --- R1: every station the kind or the wording names must exist
    named = []
    for m in re.finditer(r"visit\(([^)]*)\)", kind):
        named.append(("kind", m.group(1)))
    if base == "assistant_said" and arg in ("pan", "pot", "fryer", "oven", "board", "cutting board", "trash", "sink"):
        named.append(("kind", arg))
    for w in STATION_WORDS:
        if re.search(r"\b" + re.escape(w) + r"s?\b", ql):
            named.append(("words", w))
    seen = set()
    for n in named:
        k = visit_kind(n[1])
        if k in seen:
            continue
        seen.add(k)
        if k not in K["kinds"]:
            err("R1", f"'{n[1]}' ({'gt_kind' if n[0] == 'kind' else 'question wording'}) — this kitchen has no {k} "
                      f"(it has: {', '.join(sorted(x for x in K['kinds'] if x in ('pan', 'pot', 'fryer', 'oven', 'cutting board', 'trash', 'sink')))})")
    # --- order / happened sides
    if base in ("order", "happened"):
        sides = [s.strip() for s in arg.split("|")]
        for s in sides:
            v, x = _spec(s)
            if v in ("prep", "chop", "heat", "ignite", "cook", "assemble") and x not in K["items"]:
                err("R2", f"{s}: {x} is not in this recipe ({', '.join(K['items'])})")
                continue
            if v == "chop" and x not in K["chopped"]:
                err("V1", f"{s}: the recipe never chops the {x}")
            if v in ("heat", "ignite", "cook") and x not in K["cooked"]:
                err("V1", f"{s}: the recipe never puts the {x} on the heat")
            if v == "prep":
                says_chop = bool(re.search(r"\bchop\w*\b[^?]{0,20}\b" + re.escape(x) + r"\b", ql)) or \
                    bool(re.search(r"\b" + re.escape(x) + r"\b[^?]{0,12}\bchopped\b", ql))
                says_heat = bool(re.search(r"\b(on the heat|in the (pan|pot|fryer|oven)|cook(ed)?|fr(y|ied)|boil(ed)?)\b", ql))
                if says_chop and x not in K["chopped"]:
                    err("V1", f"\"chop the {x}\": the recipe never chops the {x} (it goes straight on the heat); "
                              f"ask \"put the {x} on the heat\"")
                elif says_heat and not says_chop and x not in K["cooked"]:
                    err("V1", f"prep({x}) worded as cooking, but the recipe never cooks the {x}")
                elif not says_chop and not says_heat:
                    want = "chop" if x in K["chopped"] else "put ... on the heat"
                    warn("W1", f"prep({x}) — the wording does not say how the {x} is processed (expected '{want}')")
        if base == "order" and len(sides) == 2:
            (va, xa), (vb, xb) = _spec(sides[0]), _spec(sides[1])
            if K["ordered"] and va in ("prep", "chop", "heat", "ignite", "cook") and vb == "assemble" and xa in K["items"] and xb in K["items"]:
                if K["items"].index(xa) < K["items"].index(xb):
                    warn("T1", f"{sides[0]} | {sides[1]}: the recipe fixes the answer ({xa} must be processed and plated before the {xb})")
            if xa == xb and {va, vb} <= {"prep", "chop", "heat", "ignite", "cook", "assemble"}:
                warn("T1", f"{sides[0]} | {sides[1]}: the same item's own steps, fixed by the recipe")
    # --- cooking questions and requests
    item = sub.get("item") or (arg if base in ("ready_in", "ready", "elapsed") else None)
    if fam == "qa_proactive" or base in COOK_KINDS:
        if not K["cooked"]:
            err("P1", f"{kind or fam}: nothing in this recipe is cooked")
        elif item and item not in K["cooked"]:
            err("P1", f"{kind or fam}: the {item} is never cooked in this recipe"
                      + ("" if item in K["items"] else f" (not in the recipe at all)"))
        if item and K["counts"].get(item, 0) > 1:
            names_station = re.search(r"\b(first|second|other|batch|fryer|pan|pot|oven|both|just put|in the)\b", ql)
            if not names_station:
                warn("M1", f"\"the {item}\": this recipe cooks {K['counts'][item]} of them and the wording does not say which "
                           f"(grading binds the load cooking at the ask)")
    # --- after_ref
    if base == "after_ref":
        if arg not in K["items"]:
            err("R2", f"after_ref:{arg}: not in this recipe")
        elif K["ordered"] and K["items"][-1] == arg and K["counts"][arg] == 1:
            err("A1", f"after_ref:{arg}: the {arg} is the recipe's last item; nothing is plated after it")
    # --- the wording must be about the item of the kind
    for x in set(re.findall(r"\((\w+)\)", kind)) | ({arg} if base in ("ready_in", "after_ref", "elapsed") else set()):
        if x and x not in ("plan", "heat", "board", "plate") and not re.search(r"\b" + re.escape(x) + r"", ql) \
                and visit_kind(x) not in {visit_kind(w) for w in re.findall(r"[a-z]+(?: [a-z]+)?", ql)}:
            warn("W1", f"the wording never names '{x}' from its gt_kind")
    return out


def validate(path):
    d = json.load(open(path))
    cases = d["cases"] if isinstance(d, dict) else d
    rows = []
    for c in cases:
        for s in c.get("subs") or []:
            if s.get("family") not in ("qa_probe", "qa_proactive"):
                continue
            for code, sev, msg in check_sub(s, c):
                rows.append({"case": c["name"], "sub": s.get("name"), "family": s.get("family"), "gt_kind": s.get("gt_kind"),
                             "item": s.get("item"), "question": s.get("question"), "code": code, "severity": sev, "message": msg})
    return cases, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", nargs="?", default="benchmark/cases_composite_v4.json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cases, rows = validate(a.suite)
    n_q = sum(1 for c in cases for s in c.get("subs") or [] if s.get("family") in ("qa_probe", "qa_proactive"))
    bad = {(r["case"], r["sub"]) for r in rows if r["severity"] == "error"}
    print(f"{a.suite}: {len(cases)} cases, {n_q} questions/requests; {len(bad)} with errors, "
          f"{len({(r['case'], r['sub']) for r in rows if r['severity'] == 'warn'})} with warnings")
    by = collections.Counter((r["code"], r["severity"]) for r in rows)
    for (code, sev), n in sorted(by.items()):
        print(f"  {code} {sev:5} {n}")
    for r in rows:
        print(f"[{r['severity']}] {r['code']} {r['sub']}  ({r['gt_kind'] or r['family']}{' ' + r['item'] if r['item'] else ''})\n"
              f"        \"{r['question']}\"\n        {r['message']}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
