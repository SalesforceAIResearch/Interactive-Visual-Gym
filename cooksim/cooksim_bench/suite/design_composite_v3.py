#!/usr/bin/env python3
"""Build + verify the suite-v3 composite cases (docs/SUITE_V3_DESIGN.md).

Rules implemented here
  errors per episode by tier   easy 2 · medium 2 · hard 3 · expert 4 · master 4
  ruinous share                >= half of the errors in every episode are ruinous types
  instance reuse               every (cell, label, kwargs) inventory instance is used once if the pool allows
  label spread                 no label takes more than 40% of a cell's error slots
  anchors                      plate + prep-progress keys (suite v3); distinct or shared only behind a
                               non-plate error; monotone; a premature serve is always the last sub;
                               a question sharing an anchor with an error is asked first
  time-only errors             never two in a row
  heat overlap                 on hot recipes >= 1 ruinous error is anchored while a station is lit
  questions                    2 per episode from the v3 families (python -m cooksim_bench.scoring.qa_v3_lib), different families,
                               CONTENT-HARD by qa_v3_lib.hardness (memory questions mid-plan after an error,
                               long history, no recipe-inferable answers, counts >= 2, real comparisons),
                               family balance and >= NEG_SHARE negative truths; a reask (Q6) pairs a
                               question with its later re-ask when the truth has changed
  proactive (Q10)              one "tell me when the X is done" request per hot-recipe episode, anchored
                               right after X is put on the heat (prep-phase anchor)

    $PY python -m cooksim_bench.suite.design_composite_v3 [--seed 3] [--out tmp/interact_eval/cases_composite_v3.json]
"""
import argparse
import collections
import json
import math
import os
from cooksim_bench.paths import ROOT
import random
import sys

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim_bench.suite.gen_composite_cases import monotone                          # noqa: E402
from cooksim_bench.suite.design_composite_v2 import replay_prefix                     # noqa: E402
from cooksim_bench.exp_interact import recipe_of                                # noqa: E402
from cooksim.core.plan_ir import segment_plan                     # noqa: E402
from cooksim_bench.suite import gen_intent_cases as GI  # noqa: E402
from cooksim_bench.scoring import qa_v3_lib as Q  # noqa: E402

TIER_N = {"easy": 2, "medium": 2, "hard": 3, "expert": 4, "master": 4}
RUINOUS = {"mis_prep", "misidentify_assemble", "add_foreign", "serve_raw", "wrong_order", "omit_subgoal",
           "repeat_step", "leave_unattended"}
TIME_ONLY = {"forget_next_step"}
NONPLATE = {"forget_next_step", "leave_unattended"}
CELLS = [f"{t}_nops_{m}_map" for t in ("easy", "medium", "hard", "expert", "master") for m in ("medium", "hard")]
PER_CELL = 5
NEG_SHARE = 0.30
NEG_WORDS = ("no", "neither", "nothing", "0", "not on any", "is not cooking", "has not", "only one")
REASK_BASES = ("missing", "past_done", "longest_on", "count", "elapsed", "empty_boards")   # "happened" dropped with Q4 (2026-08-28)
REF = json.load(open(os.path.join(ROOT, "benchmark", "ref_ticks.json")))

_cache = {}


def replay(cell, m):
    inject = not cell.endswith("hard_map")
    key = (cell, m)
    if key not in _cache:
        state, ro, flat = replay_prefix(cell, inject, m)
        ro.cell = cell
        ro.flat_plan = flat
        _cache[key] = (state, ro, flat)
    return _cache[key]


def key_at(cell, m):
    from cooksim_bench.e2e_stepin_rollout import inject_key
    _, ro, _ = replay(cell, m)
    k = inject_key(ro.game, ro)
    return {"m": m, "plate": list(k["plate"]), "held": k["plate_held"], "prep": list(k.get("prep") or ())}


def tick_at(cell, m):
    _, ro, _ = replay(cell, m)
    return ro.tick


def burn_margin(cell, m):
    """Ticks until the earliest lit station at this anchor burns (tight pace); None if nothing is on heat."""
    _, ro, _ = replay(cell, m)
    lit = [Q._burn_at(st) - st.timer for _, st, _ in Q._stations(ro.game) if Q._lit(st) and Q._contents(st)]
    return min(lit) if lit else None


def err_dur(c):
    return max(0, (c.get("end") or 0) - (c.get("dev") or 0))


OVERRUN_SLACK = 60
HEAT_SAFETY = 10        # an error's replay must end at least this many ticks before anything on the heat burns


def lit_now(cell, m):
    _, ro, _ = replay(cell, m)
    return any(Q._lit(st) and Q._contents(st) for _, st, _ in Q._stations(ro.game))


def words(spec, chopped=None, past=False):
    """Plain words for an event spec, with the verb of the item's OWN process (audit A4, 2026-09-19): prep(X) is
    "chop the X" only when the recipe chops X; fish, meat and rice go straight on the heat ("put the X on the heat").
    The suite used to say "chop the fish" for every prep side, so 7 of its 13 order questions asked about a non-event."""
    return Q._spec_words(spec, chopped, past)


def question_templates(cell):
    """(kind, question text) candidates for a cell, from its recipe and map."""
    rec = recipe_of(cell)
    plan = list(segment_plan(rec))
    prepped = list(dict.fromkeys(n.component for n in plan if n.type == "prep"))
    cooked = list(dict.fromkeys(c for c, st in rec.contents if st == "cooked"))
    asm = list(dict.fromkeys(n.component for n in plan if n.type == "assemble"))
    _, ro0, _ = replay(cell, 0)
    rs = ro0.game.render_state()
    kinds_here = {str(d.get("type")) for d in (rs.get("stations") or [])}
    T = []
    from cooksim_bench.scoring.qa_world import chopped_items
    chopped = chopped_items(rec)
    ev = [f"prep({c})" for c in prepped] + [f"ignite({c})" for c in cooked] + [f"assemble({c})" for c in asm]
    for i, a in enumerate(ev):
        for b in ev[i + 1:]:
            T.append((f"order:{a}|{b}", f"Did I {words(a, chopped)} before or after I {words(b, chopped, past=True)}?"))
    # Q4 (happened:* / exists:* — the never-happened traps) retired 2026-08-28: it scored 0.90 across the
    # sweep against 0.49 for the other families, i.e. the question announced its own answer. See
    # qa_v3_lib.RETIRED_FAMILIES.
    here = {Q._visit_kind_of_type(k) for k in kinds_here}
    for w in ("cutting board", "fryer", "pan", "pot", "counter"):
        if w in here or w == "counter":
            T.append((f"count:visit({w})", f"How many times have I walked over to a {w} so far?"))
    for c in cooked:
        T.append((f"elapsed:{c}", f"How long has the {c} been on the heat?"))
    if cooked:
        T.append(("longest_on:heat", "Which of the things on the heat has been on the longest?"))
        T.append(("past_done:heat", "Is anything on the heat already done and just sitting there?"))
    T.append(("missing:plate", "What still has to go on this plate?"))
    T.append(("empty_boards:board", "How many cutting boards are free right now?"))
    for c in asm:
        T.append((f"after_ref:{c}", f"The thing I put on right after the {c} — what is it, and is it done?"))
    for w in (["fryer"] if "fryer" in kinds_here and cooked else []) + (["pan"] if "pan" in kinds_here and cooked else []) + ["plate"]:
        T.append((f"assistant_said:{w}", f"What did you tell me about the {w} earlier?"))
    T.append(("next_step:plan", "What is my next step?"))
    if cooked:
        T.append(("burn_first:heat", "If I keep going like this, which one burns first?"))
        for c in cooked:
            T.append((f"ready_in:{c}", (f"How long until the {c} I just put on the heat is ready?"
                                        if [n for n, _ in rec.contents].count(c) > 1 else f"How long until the {c} is ready?")))
    # nothing unanswerable leaves the designer: the suite validator's errors (absent station, wrong verb, item not in
    # the recipe, nothing after the last item ...) drop the template here
    from cooksim_bench.scoring.validate_suite_questions import check_sub
    case = {"cell": cell, "inject": not cell.endswith("hard_map")}
    return [(k, q) for k, q in T if not any(sev == "error" for _c, sev, _m in check_sub(
        {"family": "qa_probe", "gt_kind": k, "question": q}, case))]


def negative(gt):
    a = str(gt.get("answer", "")).strip().lower() if isinstance(gt, dict) else ""
    return any(a == w or a.startswith(w + " ") or a.startswith(w + ":") or a.startswith(w + ",") or a.startswith(w + " —") for w in NEG_WORDS)


def well_defined(kind, gt):
    if not isinstance(gt, dict) or gt.get("error"):
        return False
    base = kind.split(":", 1)[0]
    if base == "assistant_said":
        return True
    return "answer" in gt


def question_anchors(cell, pool):
    """Every subgoal boundary of the clean plan (prep phase included, thanks to the prep key) plus the
    inventory's error anchors; deduplicated by key."""
    _, _, flat = replay(cell, 0)
    ms = {0}
    last = None
    for i, (c, sid) in enumerate(flat):
        if sid != last:
            ms.add(i); last = sid
    ms |= {c["anchor_m"] for c in pool}
    out, seen = [], set()
    for m in sorted(ms):
        k = key_at(cell, m)
        sig = (tuple(k["plate"]), k["held"], tuple(k["prep"]))
        if sig in seen or m == 0:
            continue
        seen.add(sig); out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tmp/interact_eval/cases_composite_v3.json")
    ap.add_argument("--ledger", default="tmp/interact_eval/cases_composite_v3_ledger.md")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tries", type=int, default=400)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    inv = [c for c in json.load(open("tmp/interact_eval/error_inventory.json"))["cases"] if not c.get("collapses")]
    by_cell = collections.defaultdict(list)
    for c in inv:
        by_cell[c["cell"]].append(c)
    used, reuse = set(), collections.Counter()
    try:
        intents = json.load(open(os.path.join(ROOT, "benchmark", "intents_v3.json")))["cases"]
    except Exception:
        intents = []
    intent_by_cell = collections.defaultdict(list)
    for it in intents:
        intent_by_cell[it["cell"]].append(it)
    intent_used = set()
    # ruling 2026-08-27: a third of the episodes carry an announced intent (a plan change), >= 1 per cell
    INTENT_EPISODES = 17          # a third of the 50 episodes carry an announced intent; >= 1 per cell, <= 2 per cell
    fam_count = collections.Counter()
    q_truths, hard_scores = [], []
    episodes, problems = [], []
    label_total = collections.Counter()

    for cell in CELLS:
        tier = cell.split("_")[0]
        n_err = TIER_N[tier]
        need_r = math.ceil(n_err / 2)
        rec = recipe_of(cell)
        ordered = bool(getattr(rec, "ordered", False))
        cooked = list(dict.fromkeys(c for c, st in rec.contents if st == "cooked"))
        hot = bool(cooked)
        ref = REF.get(cell) or 100
        pool = by_cell[cell]
        cell_labels = collections.Counter()
        slot_cap = max(1, int(0.4 * n_err * PER_CELL + 0.999))
        templates = question_templates(cell)
        q_anchors = question_anchors(cell, pool)
        _, _, flat = replay(cell, 0)
        for k in range(1, PER_CELL + 1):
            name = f"b3_{cell}_{k}"
            best = None
            for _ in range(args.tries):
                cand = [c for c in pool if (cell, c["err"], json.dumps(c["kwargs"], sort_keys=True)) not in used and cell_labels[c["err"]] < slot_cap]
                if _ >= args.tries // 2:
                    cap_use = 2 if _ < 3 * args.tries // 4 else 3
                    cand = [c for c in pool if reuse[(cell, c["err"], json.dumps(c["kwargs"], sort_keys=True))] < cap_use and cell_labels[c["err"]] < slot_cap]
                rng.shuffle(cand)
                pick, labels = [], set()
                target_r = need_r if rng.random() < 0.6 else min(n_err, need_r + 1)
                for c in sorted(cand, key=lambda c: (0 if c["err"] in RUINOUS else 1, rng.random())):
                    if len(pick) == n_err:
                        break
                    if c["err"] in labels:
                        continue
                    n_r = sum(1 for p in pick if p["err"] in RUINOUS)
                    if c["err"] in RUINOUS and n_r >= target_r and any(x["err"] in TIME_ONLY for x in cand if x["err"] not in labels):
                        continue
                    if c["err"] not in RUINOUS and (n_err - len(pick)) <= (need_r - n_r):
                        continue
                    if c["err"] == "omit_subgoal" and "after" in c["kwargs"] and any(p["err"] == "omit_subgoal" for p in pick):
                        continue
                    pick.append(c); labels.add(c["err"])
                if len(pick) < n_err or sum(1 for p in pick if p["err"] in RUINOUS) < need_r:
                    continue
                pick.sort(key=lambda c: (c["anchor_m"], 0 if c["err"] in NONPLATE else (2 if c["err"] == "omit_subgoal" else 1)))
                anc = [{"m": c["anchor_m"], "plate": list(c["anchor_plate"]), "held": c["anchor_held"], "prep": list(c.get("anchor_prep") or ())} for c in pick]
                if len({a["m"] for a in anc}) < len(anc):
                    continue                                   # audit rule 1: errors never share an anchor
                # audit rule 2: an error (other than the walk-away itself) must not outlast the burn window of
                # anything on the heat at its anchor; rule 3: its replay must not run > OVERRUN_SLACK ticks
                # past the next error's anchor
                bad_heat = False
                for c in pick:
                    if c["err"] == "leave_unattended":
                        continue
                    bm = burn_margin(cell, c["anchor_m"])
                    if bm is not None and err_dur(c) + HEAT_SAFETY > bm:
                        bad_heat = True; break
                if bad_heat:
                    continue
                if any(tick_at(cell, pick[i]["anchor_m"]) + err_dur(pick[i]) > tick_at(cell, pick[i + 1]["anchor_m"]) + OVERRUN_SLACK
                       for i in range(len(pick) - 1) if pick[i]["err"] != "leave_unattended"):
                    continue
                if any(not monotone(anc[i], anc[i + 1], ordered=ordered) for i in range(len(anc) - 1)):
                    continue
                prem = [i for i, c in enumerate(pick) if c["err"] == "omit_subgoal" and "after" in c["kwargs"]]
                if prem and prem[0] != len(pick) - 1:
                    continue
                if any(pick[i]["err"] in TIME_ONLY and pick[i + 1]["err"] in TIME_ONLY for i in range(len(pick) - 1)):
                    continue
                heat_ok = (not hot) or any(c["err"] in RUINOUS and (c["err"] == "leave_unattended" or lit_now(cell, c["anchor_m"])) for c in pick)
                score = (1 if heat_ok else 0, -sum(label_total[c["err"]] for c in pick), rng.random())
                if best is None or score > best[0]:
                    best = (score, pick, anc)
                if heat_ok and best[0][0] == 1 and _ > 40:
                    break
            if best is None:
                problems.append((name, ["no admissible error set"]))
                continue
            _, pick, anc = best
            for c in pick:
                kk = (cell, c["err"], json.dumps(c["kwargs"], sort_keys=True))
                used.add(kk); reuse[kk] += 1
                cell_labels[c["err"]] += 1; label_total[c["err"]] += 1
            subs = []
            for c in pick:
                s = {kk: v for kk, v in c.items() if kk not in ("anchor_m", "anchor_plate", "anchor_held", "anchor_prep", "collapses", "plan_len", "desc")}
                s["name"] = f"{name}_{c['err']}"
                s["ruinous"] = c["err"] in RUINOUS
                subs.append(s)
            first_error_m = min(c["anchor_m"] for c in pick)
            prem_m = min((c["anchor_m"] for c in pick if c["err"] == "omit_subgoal" and "after" in c["kwargs"]), default=None)

            def clear_of_overrun(m_):
                t_ = tick_at(cell, m_)
                for c in pick:
                    if c["err"] == "leave_unattended" or c["anchor_m"] >= m_:
                        continue
                    if tick_at(cell, c["anchor_m"]) + err_dur(c) > t_ + OVERRUN_SLACK:
                        return False
                return True
            # ---- questions: score every (kind, anchor) for truth + content hardness
            cands = []
            for kind, text in templates:
                for a in q_anchors:
                    if prem_m is not None and a["m"] > prem_m:
                        continue
                    if not clear_of_overrun(a["m"]):
                        continue
                    _, ro, _ = replay(cell, a["m"])
                    try:
                        gt = GI.ground_truth(kind, ro.game, ro)
                    except Exception as e:                        # noqa: BLE001
                        gt = {"error": repr(e)}
                    if not well_defined(kind, gt):
                        continue
                    ok, sc, why = Q.hardness(kind, gt, ro, ro.tick, ref, ordered=ordered, first_error_m=first_error_m, m=a["m"])
                    if not ok:
                        continue
                    cands.append({"kind": kind, "text": text, "a": a, "gt": gt, "score": sc, "why": why, "fam": Q.family_of(kind), "tick": ro.tick})
            want_neg = (sum(q_truths) / max(1, len(q_truths))) < NEG_SHARE
            qsubs, qanc = [], []

            def take(c):
                qsubs.append({"name": f"{name}_q_{c['kind'].split(':')[0]}_{c['a']['m']}", "family": "qa_probe", "cell": cell, "err": "no_error",
                              "trigger_after": c["a"]["m"], "intent_seq": [], "resume_from": c["a"]["m"], "question": c["text"], "gt_kind": c["kind"],
                              "expected_gt": c["gt"], "qa_family": c["fam"], "hardness": round(c["score"], 2), "hardness_why": c["why"],
                              "ask_tick_clean": c["tick"], "inject": not cell.endswith("hard_map"), "auto": False})
                qanc.append(dict(c["a"]))
                fam_count[c["fam"]] += 1; q_truths.append(1 if negative(c["gt"]) else 0); hard_scores.append(c["score"])

            fams_here = set()
            order_c = sorted(cands, key=lambda c: (fam_count[c["fam"]], -(c["score"] + (0.4 if (want_neg and negative(c["gt"])) else 0)), rng.random()))
            for c in order_c:
                if len(qsubs) == 2:
                    break
                if c["fam"] in fams_here:
                    continue
                take(c); fams_here.add(c["fam"])
                # try to pair a re-ask (Q6): the same question later, truth changed
                if len(qsubs) == 1 and c["kind"].split(":")[0] in REASK_BASES and fam_count["Q6"] < 8:
                    for a2 in q_anchors:
                        if a2["m"] <= c["a"]["m"] or (prem_m is not None and a2["m"] > prem_m) or not clear_of_overrun(a2["m"]):
                            continue
                        _, ro2, _ = replay(cell, a2["m"])
                        if ro2.tick - c["tick"] < 40:
                            continue
                        ro2.qa_history = [{"kind": c["kind"], "t": c["tick"], "gt": c["gt"]}]
                        try:
                            gt2 = GI.ground_truth("reask:" + c["kind"], ro2.game, ro2)
                        finally:
                            ro2.qa_history = []
                        ok, sc, why = Q.hardness("reask:" + c["kind"], gt2, ro2, ro2.tick, ref, ordered=ordered)
                        if ok:
                            take({"kind": "reask:" + c["kind"], "text": c["text"] + " — and what has changed since I asked you that before?", "a": a2,
                                  "gt": gt2, "score": sc + 0.5, "why": why + f" ({ro2.tick - c['tick']} ticks after the first ask)", "fam": "Q6", "tick": ro2.tick})
                            fams_here.add("Q6"); break
            if len(qsubs) < 2:
                problems.append((name, [f"only {len(qsubs)} content-hard questions available"]))
            # ---- Q10: proactive request right after a cooked component goes on the heat
            psubs, panc = [], []
            if hot:
                # the request anchors on the item's ignition: never pair it with an error that removes that event
                blocked = {c["kwargs"].get("component") for c in pick if c["err"] in ("serve_raw", "mis_prep")}
                choices = [c_ for c_ in cooked if c_ not in blocked] or []
                item = rng.choice(choices) if choices else None
                m_on = next((i + 1 for i, (cmd, sid) in enumerate(flat) if cmd == "turn on" and (sid or "").split("#")[0] in (f"prep({item})", f"ignite({item})")), None)
                if m_on is not None and (prem_m is None or m_on <= prem_m) and clear_of_overrun(m_on):
                    a = key_at(cell, m_on)
                    _, ro, _ = replay(cell, m_on)
                    gt = GI.ground_truth(f"ready:{item}", ro.game, ro)
                    if gt.get("ticks_left", 0) >= 5:
                        psubs.append({"name": f"{name}_p_ready_{item}_{m_on}", "family": "qa_proactive", "cell": cell, "err": "no_error",
                                      "trigger_after": m_on, "intent_seq": [], "resume_from": m_on, "item": item,
                                      # a recipe with several units of the item (two potatoes) names the batch just put on (audit A6)
                                      "question": (f"Give me a shout the moment the {item} I just put on the heat is done, would you?"
                                                   if [n for n, _ in rec.contents].count(item) > 1 else
                                                   f"Give me a shout the moment the {item} is done, would you?"), "gt_kind": f"ready:{item}",
                                      "expected_gt": gt, "qa_family": "Q10", "ask_tick_clean": ro.tick, "inject": not cell.endswith("hard_map"), "auto": False})
                        panc.append(a); fam_count["Q10"] += 1
            # ---- intents: an announced plan change (never a mistake)
            isubs, ianc = [], []
            n_int_total = sum(1 for e_ in episodes if any(s_["family"] == "plan_change" for s_ in e_["subs"]))
            n_int_cell = sum(1 for e_ in episodes if e_["cell"] == cell and any(s_["family"] == "plan_change" for s_ in e_["subs"]))
            want_intent = (n_int_cell == 0) or (n_int_total < INTENT_EPISODES and n_int_cell < 2)
            if want_intent and intent_by_cell.get(cell):
                cands_i = []
                for it in intent_by_cell[cell]:
                    if (cell, it["name"]) in intent_used:          # names can repeat across cells: key by cell
                        continue
                    a = dict(it["anchor"]); m_i = a["m"]
                    if prem_m is not None and m_i > prem_m:
                        continue
                    if not clear_of_overrun(m_i):
                        continue
                    if it["family"] == "plan_change" and prem_m is not None:
                        continue                                  # an added topping cannot follow a premature serve
                    # must not share an anchor with a plate-changing error or another intent
                    clash = any(c["anchor_m"] == m_i for c in pick)
                    if clash:
                        continue
                    cands_i.append(it)
                # alternate families across the suite
                cands_i.sort(key=lambda it: rng.random())
                for it in cands_i:
                    a = dict(it["anchor"])
                    trial = anc + qanc + panc + [a]
                    trial_subs = subs + qsubs + psubs + [it]
                    order_t = sorted(range(len(trial)), key=lambda i: (trial[i]["m"], 0 if trial_subs[i].get("family") != "error" else 1))
                    tr = [trial[i] for i in order_t]
                    if all(monotone(tr[i], tr[i + 1], ordered=ordered) for i in range(len(tr) - 1)):
                        sub_i = {"name": f"{name}_i_{it['family']}_{a['m']}", "family": it["family"], "cell": cell, "err": "no_error",
                                 "trigger_after": it["trigger_after"], "resume_from": it["resume_from"], "intent_label": it["intent_label"],
                                 "intent_seq": list(it["intent_seq"]), "duration": it.get("duration"), "clean_tick": it.get("clean_tick"),
                                 "rationale": it.get("rationale"), "inject": not cell.endswith("hard_map"), "auto": False}
                        if it.get("goal_extend"):
                            sub_i["goal_extend"] = it["goal_extend"]
                        isubs.append(sub_i); ianc.append(a); intent_used.add((cell, it["name"])); fam_count[it["family"]] += 1
                        break
            allsubs = subs + qsubs + psubs + isubs
            allanc = anc + qanc + panc + ianc
            order = sorted(range(len(allsubs)), key=lambda i: (allanc[i]["m"], 0 if allsubs[i].get("family") != "error" else 1))
            allsubs = [allsubs[i] for i in order]; allanc = [allanc[i] for i in order]
            bad = [f"anchors not monotone: {allanc[i]} -> {allanc[i+1]}" for i in range(len(allanc) - 1) if not monotone(allanc[i], allanc[i + 1], ordered=ordered)]
            if bad:
                problems.append((name, bad)); continue
            inject = not cell.endswith("hard_map")
            for s in allsubs:
                s["inject"] = inject
            story = "; ".join(c["desc"] for c in pick) + ". Questions: " + " / ".join(f"“{q['question']}” (m={q['trigger_after']}, {q['qa_family']}, hardness {q.get('hardness')})" for q in qsubs) \
                + (f". Proactive: “{psubs[0]['question']}” (m={psubs[0]['trigger_after']})" if psubs else "") \
                + (f". Intent: {isubs[0]['intent_label']} ({isubs[0]['family']}, m={isubs[0]['trigger_after']})" if isubs else "")
            episodes.append({"name": name, "family": "composite", "cell": cell, "err": "composite", "inject": inject,
                             "subs": allsubs, "n_errors": len(subs), "n_ruinous": sum(1 for s in subs if s["ruinous"]),
                             "order_pattern": "".join({"qa_probe": "Q", "qa_proactive": "P", "plan_change": "I"}.get(s.get("family"), "E") for s in allsubs),
                             "anchors": [dict(a, name=s["name"]) for a, s in zip(allanc, allsubs)], "rationale": story})
            print(f"{name}: {[s['err'] for s in subs]} ruinous {sum(1 for s in subs if s['ruinous'])}/{len(subs)}; "
                  f"q={[(q['gt_kind'], q.get('hardness')) for q in qsubs]} p={[p['gt_kind'] for p in psubs]}", flush=True)

    allow = {e["name"]: int(sum(max(0, (s.get("identifiable") or 0) - (s.get("dev") or 0)) for s in e["subs"] if s["family"] == "error")) for e in episodes}
    json.dump(allow, open("benchmark/case_allowance_v3.json", "w"), indent=1)
    tmp_out = args.out + ".tmp"
    json.dump({"schema": "composite-v3", "cases": episodes}, open(tmp_out, "w"), indent=1, default=str)
    os.replace(tmp_out, args.out)
    n_err = sum(e["n_errors"] for e in episodes)
    n_q = sum(1 for e in episodes for s in e["subs"] if s["family"] == "qa_probe")
    n_p = sum(1 for e in episodes for s in e["subs"] if s["family"] == "qa_proactive")
    n_i = sum(1 for e in episodes if any(s["family"] == "plan_change" for s in e["subs"]))
    with open(args.ledger, "w") as f:
        f.write(f"# Composite suite v3 — {len(episodes)} episodes, {n_err} errors ({sum(e['n_ruinous'] for e in episodes)} ruinous), {n_q} questions, {n_p} proactive requests, intents in {n_i} episodes; {len(problems)} problems\n\n")
        f.write("Labels: " + ", ".join(f"{k} {v}" for k, v in sorted(label_total.items())) + f"; instances used twice or more: {sum(1 for v in reuse.values() if v > 1)}\n\n")
        f.write("Question families: " + ", ".join(f"{k} {v}" for k, v in sorted(fam_count.items())) + f"; negative truths {sum(q_truths)}/{len(q_truths)}; mean hardness {sum(hard_scores)/max(1,len(hard_scores)):.2f}\n\n")
        for e in episodes:
            f.write(f"## {e['name']}  ({e['order_pattern']}, {e['n_ruinous']}/{e['n_errors']} ruinous)\n\n{e['rationale']}\n\n")
            for a, s in zip(e["anchors"], e["subs"]):
                if s["family"] == "qa_probe":
                    f.write(f"- m={a['m']:3d} t≈{s.get('ask_tick_clean')} plate={a['plate']} held={a['held']} prep={a.get('prep')} | question [{s['qa_family']}] **{s['gt_kind']}** — “{s['question']}” → {json.dumps(s['expected_gt'].get('answer'), default=str)[:100]} (hardness {s.get('hardness')}: {s.get('hardness_why')})\n")
                elif s["family"] == "plan_change":
                    f.write(f"- m={a['m']:3d} t≈{s.get('clean_tick')} plate={a['plate']} held={a['held']} prep={a.get('prep')} | intent [{s['family']}] **{s['intent_label']}** — {' → '.join(s['intent_seq'])} (≈{s.get('duration')} ticks)\n")
                elif s["family"] == "qa_proactive":
                    f.write(f"- m={a['m']:3d} t≈{s.get('ask_tick_clean')} prep={a.get('prep')} | proactive [Q10] **{s['gt_kind']}** — “{s['question']}” (ready in {s['expected_gt'].get('ticks_left')} ticks at the request)\n")
                else:
                    f.write(f"- m={a['m']:3d} plate={a['plate']} held={a['held']} prep={a.get('prep')} | error **{s['err']}** {json.dumps(s['kwargs'])} ({'ruinous' if s['ruinous'] else 'time only'}; outcome {s['outcome']})\n")
            f.write("\n")
        if problems:
            f.write("## Problems\n\n" + "\n".join(f"- {n}: {b}" for n, b in problems) + "\n")
    print(f"wrote {args.out}: {len(episodes)} episodes, {n_err} errors, {n_q} questions, {n_p} proactive; problems {len(problems)}")
    print("labels:", dict(sorted(label_total.items())))
    print("families:", dict(sorted(fam_count.items())), "negative", sum(q_truths), "/", len(q_truths), "mean hardness %.2f" % (sum(hard_scores) / max(1, len(hard_scores))))


if __name__ == "__main__":
    main()
