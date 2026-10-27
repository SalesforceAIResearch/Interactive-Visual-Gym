#!/usr/bin/env python3
"""Evaluation v2 (docs/PERSONA_EVAL_V2.md §2 — the user's design) over the whole sweep.

Four headline scores per episode, weights a, b, c = the cook's three questionnaire answers
(1 = low, 2 = medium, 3 = high):

    task_success        = engine verdict: 1 if the dish was won; graded partial credit
                          otherwise (correct plate prefix / recipe length for a stacked dish,
                          multiset overlap for an order-free dish); 0 on a hazard; null when
                          the episode was aborted without a verdict (not measured, never 0)
    task_efficiency     = (a·TT + c·CC) / (a + c)         raw form a·TT + c·CC stored too
        TT = clip01(ref_ticks / ticks)                    ref = perfect clean-plan run, per cell
        CC = 1 − clip01(T_interrupt / T_action)           T_interrupt = assistant speech time
                                                          (study-app TTS formula) landing while
                                                          the cook is mid-action; T_action = ticks
    assistance_quality  = (b·(DT + DQ) + 2·SQ) / (2b + 2)  raw form b·(DT+DQ) + 2·SQ stored too
        DT = detection timing tier ladder, averaged over the episode's fired error sub-cases:
             prevented 1.0; valid flag before the outcome sealed 0.9–0.5 by latency within the
             window; valid only after the outcome 0.3; right name but outside the window 0.5;
             never 0.0
        DQ = 1 − (½·miss + ½·fa_norm); miss = share of fired error sub-cases with no valid
             flag; fa_norm = false alarms / max(3, 3·ticks/100)
        SQ = (judge − 1) / 9, one anchored suggestion-quality rubric per episode (tools/
             run_sq_judge.py; mean over the judges available), null until judged
    interaction_quality = the cook's OWN exit answers, satisfaction and reuse, (x − 1) / 6,
                          reported separately (never ranked across personas)

This is python -m cooksim_bench.scoring.eval_v2 generalised to composite episodes (2 errors + 2 questions per
episode, each sub-case archived under `injections`) and to episodes with no recorded speech
pauses (T_interrupt is rebuilt from the transcript + the engine snapshot attached to each
assistant turn). Every component is stored unweighted next to the weights, so any future
questionnaire re-weights without recomputation. Question-answer correctness (the composite
suite's second measurement, from python -m cooksim_bench.scoring.regrade_qa) is stored as evidence beside the four
scores; it is not part of the user's formulas.

Audit fixes (tmp/interact_eval/audits/eval_v2_audit.md, 2026-08-26): task_efficiency is null
unless the dish was won (the hard gate); assistance_quality is null until SQ exists (never a
silently re-weighted number); DT scales latency from the correct flag's own tick inside a
window clipped to the episode end, gives no prevention credit on an undetermined plate
verdict, and uses the 0.3 rung when the error had committed before the flag; `late` flags
are not false alarms; a judge model never scores its own episodes; G2 requires an ERROR
sub-case to have fired; G3 also honours eval.incomplete.

    $PY python -m cooksim_bench.scoring.eval_v2_sweep            -> tmp/assist_bench/eval_v2.json  {key0: {...}}
"""
import collections
import glob
import json
import os
import sys
import time

from cooksim_bench.paths import ROOT  # the CookSim release root
os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from cooksim.interact.checks import CORRECT_VERDICTS, NEUTRAL_VERDICTS   # noqa: E402
from cooksim.interact import windows as windows_lib                      # noqa: E402
from cooksim_bench.scoring.eval_v2 import clip01, wmean                                        # noqa: E402

BENCH = os.environ.get("ASSIST_ROOT", "tmp/assist_bench")   # v3 pilot / sweep: ASSIST_ROOT=tmp/assist_bench_v3
OUT = os.path.join(ROOT, BENCH, "eval_v2.json")
QA_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0, "no_answer": 0.0, "evasive": 0.0}


def speech_ticks(text):
    """study-app parity: TTS ~3.15 words/s and 1 tick ~ 1.2 s (same as the harness)."""
    return max(1, min(20, round(len(str(text).split()) / 3.15 / 1.2)))


def load_side(name):
    p = os.path.join(ROOT, BENCH, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def all_cases():
    """v2.2 suite + v3 suite (names are disjoint: bm_ vs b3_)."""
    out = []
    pools = [os.path.join(ROOT, "benchmark", "cases_composite_v5.json"), "tmp/interact_eval/cases_composite_v3.json"]
    if os.environ.get("COOKSIM_CASES"):                       # v5 curated suite takes precedence (same names)
        pools.insert(0, os.environ["COOKSIM_CASES"])
    seen = set()
    for pf in pools:
        try:
            for c in json.load(open(os.path.join(ROOT, pf)))["cases"]:
                if c["name"] not in seen:
                    seen.add(c["name"]); out.append(c)
        except Exception:
            pass
    return out


def recipe_table():
    from cooksim_bench.exp_interact import recipe_of
    cases = all_cases()
    cells = {c["name"]: c["cell"] for c in cases}
    rec = {}
    for cell in set(cells.values()):
        r = recipe_of(cell)
        rec[cell] = {"contents": [c for c, _ in r.contents], "ordered": bool(getattr(r, "ordered", False))}
    return cells, rec


def task_success(d, recipe):
    """Ruling 2026-08-27 (no partial credit): 1 only when the engine verdict is won; every other verdict is 0
    (a wrong serve whose plate happened to contain every recipe item plus a foreign one used to score 1.0
    through the multiset-overlap path — 48 such episodes in the suite-v3 sweep); None without a verdict."""
    if d.get("outcome") == "won":
        return 1.0
    if d.get("outcome") is None or (d.get("aborted") and d.get("outcome") is None):
        return None
    if d.get("outcome"):
        return 0.0
    plate = list(d.get("plate_first_order") or [])
    goal = list((recipe or {}).get("contents") or [])
    if not goal:
        return None
    if recipe.get("ordered"):
        k = 0
        for want, got in zip(goal, plate):
            if want == got:
                k += 1
            else:
                break
        return k / len(goal)
    want = collections.Counter(goal); got = collections.Counter(plate)
    return sum(min(want[x], got[x]) for x in want) / len(goal)


def cc(d):
    """Conversation efficiency rebuilt from the transcript: assistant lines that land while
    the cook has a running action (engine snapshot on that turn) count as interruptions."""
    ticks = d.get("ticks")
    if not ticks:
        return None
    t_int = 0
    for c in d.get("conversation") or []:
        if c.get("speaker") != "vlm" or not (c.get("text") or "").strip():
            continue
        acts = ((c.get("engine") or {}).get("human") or {}).get("past_actions") or []
        mid_action = any(a.get("status") == "running" for a in acts)
        if mid_action:
            t_int += speech_ticks(c["text"])
    return 1.0 - clip01(t_int / ticks), t_int


def sub_dt(e, ticks=None):
    """Detection timing tier for one fired error sub-case (the v1 §3c ladder on the archived
    fields; audit fixes 2026-08-26: the latency scale uses the FLAG's own tick inside the
    window (clipped to the episode end), an undetermined plate verdict earns no prevention
    credit, and a correct flag that only landed after the error committed is the 0.3 rung)."""
    rk = e.get("report_keys") or {}
    flags = e.get("flags") or []
    pv = e.get("plate_verdict_at_release") or {}
    committed = pv.get("error_committed")
    determinate = committed is not None and not pv.get("error_undetermined")
    correct = e.get("flagged_valid") or any(f.get("verdict") in CORRECT_VERDICTS for f in flags)
    if e.get("prevented"):
        return 1.0
    if determinate and committed is False and correct:
        return 1.0                                   # prevention-equivalent: nothing was ever wrong
    win = rk.get("window")
    tf = [f.get("t") for f in flags if f.get("verdict") in CORRECT_VERDICTS and isinstance(f.get("t"), (int, float))]
    if e.get("flagged_valid") or correct:
        if windows_lib.window_rule(rk) == windows_lib.LIVE:
            # LIVE WINDOWS (2026-09-20): the window CLOSES at the commit, so "the error had
            # already happened" is no longer a property of the episode — it is a property of
            # the CALL. The 0.3 post-hoc rung is charged when the correct call itself landed
            # after the window closed (which is also when verdict_live says `late`); a call
            # inside the window is in time by construction, whatever the cook then did with it.
            if (tf and win and min(tf) > win[1]) or (not tf and any(
                    f.get("verdict") == "late" for f in flags)):
                return 0.3
        elif determinate and committed is True and (rk.get("prevention_missed") is not None or e.get("flagged_valid")):
            # right call, but the error had already happened in the world
            return 0.3
        if tf and win and (win[1] - win[0]) > 0:
            lo, hi = win[0], min(win[1], ticks) if ticks else win[1]
            if hi > lo:
                return 0.9 - 0.4 * clip01((min(tf) - lo) / (hi - lo))
            return 0.7
        return 0.7
    if any(f.get("verdict") in ("early_correct", "early_alias", "alias_in_window") for f in flags):
        return 0.5
    if any(f.get("verdict") == "late" for f in flags):
        return 0.3                                   # right error, named only after it had landed (post-hoc rung)
    return 0.0


REATTR_VERDICTS = ("true_observation", "false_alarm", "fp_preinjection", "wrong_name", "early_alias")


def reattribute_release_flags(inj):
    """Audit fix (self_repair_audit.md rec. 1): a flag raised on the tick an error sub-case is
    released is archived under the NEXT sub-case (often a question) as true_observation /
    false_alarm, although it is the catch that drove the repair. Move such flags back to the
    released error sub when they name its error label or its wrong item. Returns
    {err_idx: [flags moved]} and the set of (next_idx, flag id) to ignore as false alarms."""
    moved, ignore = {}, set()
    for k in range(len(inj) - 1):
        e, nxt = inj[k], inj[k + 1]
        if e.get("family") != "error" or not e.get("injected"):
            continue
        t_rel = e.get("archived_at")
        if t_rel is None:
            continue
        kw = (e.get("kwargs") or {}) if isinstance(e.get("kwargs"), dict) else {}
        names = {str(v).lower() for v in kw.values() if isinstance(v, str)}
        label = e.get("err")
        for j, f in enumerate(nxt.get("flags") or []):
            if f.get("t") is None or not (t_rel <= f["t"] <= t_rel + 2) or f.get("verdict") not in REATTR_VERDICTS:
                continue
            utt = (f.get("utt") or "").lower()
            if f.get("error") == label or any(n and n in utt for n in names) or \
                    (label == "wrong_order" and "order" in utt) or (label == "omit_subgoal" and ("serve" in utt or "missing" in utt or "skip" in utt)):
                moved.setdefault(k, []).append(dict(f, verdict="late", reattributed_from=k + 1))
                ignore.add((k + 1, j))
    return moved, ignore


def _jacc(a, b):
    import re as _re
    A = set(_re.findall(r"[a-z']+", a.lower())); B = set(_re.findall(r"[a-z']+", b.lower()))
    return len(A & B) / max(1, len(A | B))


def friction(d):
    """Objective interaction friction (v2.1): assistant lines that near-repeat one of its last
    three lines (nagging), the cook's explicit 'less interruption' requests, and the cook's
    push-backs (question_correctness / reject). Each is a count per episode."""
    conv = d.get("conversation") or []
    recent, nag = [], 0
    for c in conv:
        if c.get("speaker") == "vlm" and (c.get("text") or "").strip():
            t = c["text"]
            if any(_jacc(t, r) >= 0.6 for r in recent):
                nag += 1
            recent = (recent + [t])[-3:]
    ask_less = sum(1 for c in conv if c.get("speaker") == "human" and c.get("move") == "ask_for_less_interruption")
    pushback = sum(1 for c in conv if c.get("speaker") == "human" and c.get("move") in ("question_correctness", "reject"))
    return {"nag_repeats": nag, "ask_less": ask_less, "pushback": pushback}


FA_VERDICTS = ("false_alarm", "fp_preinjection", "wrong_name", "intent_false_alarm", "invalid_label")
# ruling 2026-09-24: recall/detected credit a correct-but-late call up to this many ticks after the commit
LATE_GRACE = int(os.environ.get("COOKSIM_LATE_GRACE", "30") or 0)


def plate_at(d, t):
    """Plate contents at tick t, rebuilt from the plate history and the cook's trash actions
    (a trash empties the plate; items plated after the last trash before t are on it)."""
    trashes = set()
    for c in d.get("conversation") or []:
        for a in ((c.get("engine") or {}).get("human") or {}).get("past_actions") or []:
            if a.get("cmd") == "trash" and a.get("status") == "done" and a.get("t_done") is not None and a["t_done"] <= t:
                trashes.add(a["t_done"])
    last_trash = max(trashes) if trashes else -1
    return [e["item"] for e in (d.get("plate_seq") or []) if last_trash < (e.get("t") or 0) <= t]


def progress(plate, recipe):
    goal = list((recipe or {}).get("contents") or [])
    if not goal:
        return None
    if recipe.get("ordered"):
        k = 0
        for want, got in zip(goal, plate):
            if want == got:
                k += 1
            else:
                break
        return k / len(goal)
    want = collections.Counter(goal); got = collections.Counter(plate)
    return sum(min(want[x], got[x]) for x in want) / len(goal)



def late_detections(d, fired, grace=None):
    """ruling 2026-09-24: the set of fired errors whose right name was called within `grace` ticks AFTER the commit.
    Late calls are recorded on the episode's false_positives list (verdict `late`), not on the injection — they
    usually land after the injection was archived — so each is attributed to the fired error of the same realized
    type whose window closed most recently before the call."""
    grace = LATE_GRACE if grace is None else grace
    out = set()
    hi_of = []
    for i, e in enumerate(fired):
        hi = ((e.get("report_keys") or {}).get("window") or [None, None])[1]
        if hi is not None:
            hi_of.append((hi, i, e.get("err")))
    for x in (d.get("false_positives") or []):
        if x.get("verdict") != "late" or not isinstance(x.get("t"), (int, float)):
            continue
        cands = [(hi, i) for hi, i, err in hi_of if err == x.get("error") and hi < x["t"] <= hi + grace]
        if cands:
            out.add(max(cands)[1])
    for i, e in enumerate(fired):          # the few late calls that did get attached to the injection
        hi = ((e.get("report_keys") or {}).get("window") or [None, None])[1]
        if hi is not None and any(f.get("verdict") == "late" and isinstance(f.get("t"), (int, float)) and f["t"] <= hi + grace for f in (e.get("flags") or [])):
            out.add(i)
    return out

def v3_block(d, fired, ts, ticks, ref_ticks, allowance, recipe, a, b, c, CC, SQ):
    """The user's redesign (2026-08-27): in-time task success under a 2x(reference + allowance)
    action-time cap, no partial credit (ruling 2026-08-27); TT = 0 on timeout, else linear from
    1 at the reference to 0 at the cap; DT = prevented 1.0 / correct flag before the error
    committed 0.5 / else 0; DQ = F1 of precision and recall over the episode's flags.
    2026-09-20: under live windows the commit boundary is the window's own hi (the measured
    commit tick), not the release tick that used to stand in for it."""
    cap = (2 * (ref_ticks + (allowance or 0))) if ref_ticks else None
    timed_out = bool(cap and ticks and ticks > cap)
    # user ruling 2026-08-27: no partial credit — in-time success is 1 only for a dish won within the cap
    progress_at_cap = progress(plate_at(d, cap), recipe) if timed_out else None
    in_time = 0.0 if timed_out else (1.0 if ts == 1.0 else 0.0)
    if cap and ticks:
        TT3 = 0.0 if timed_out else clip01((cap - ticks) / max(1, cap - ref_ticks))
    else:
        TT3 = None
    # DT3 per error
    dts = []
    tp = 0
    late_ok_idx = late_detections(d, fired)
    for ei, e in enumerate(fired):
        rk = e.get("report_keys") or {}
        t_rel = rk.get("released_at") if rk.get("released_at") is not None else e.get("archived_at")
        live_hi = (rk.get("window") or [None, None])[1] \
            if windows_lib.window_rule(rk) == windows_lib.LIVE else None
        # LIVE WINDOWS (2026-09-20): "the tick the error committed" is now MEASURED, not
        # approximated by the release. The window's own hi IS that tick (and falls back to the
        # release when the mistake never committed), so DT keeps its published definition — a
        # correct flag BEFORE the error committed — and stops depending on how long the
        # takeover happened to hold afterwards. hi is inclusive, exactly as verdict_live reads it.
        def flag_in_time(ft, _hi=live_hi, _rel=t_rel):      # NB: not `in_time` -- that name is the episode's
            # own in-time-success flag (line ~285) and shadowing it here broke every score (2026-09-20)
            return (ft <= _hi) if _hi is not None else (_rel is None or ft < _rel)
        flags = e.get("flags") or []
        # ruling 2026-09-24 (supersedes 2026-08-27): RECALL counts the mistake as detected when the right error is
        # named inside its window OR within COOKSIM_LATE_GRACE ticks (default 30) after it committed — a late
        # call is still a detection; TIMING is the job of DT and the prevention rate, not of recall.
        correct = [f for f in flags if f.get("verdict") in CORRECT_VERDICTS]
        # flagged_valid / prevented are the engine's own record of a correct in-window call (the cook grader
        # does not always leave a verdict entry on the flag list), so they count as detections too
        if correct or e.get("flagged_valid") or e.get("prevented") or ei in late_ok_idx:
            tp += 1
        if e.get("prevented"):
            dts.append(1.0)
        elif any(f.get("verdict") in CORRECT_VERDICTS and isinstance(f.get("t"), (int, float))
                 and flag_in_time(f["t"]) for f in flags):
            dts.append(0.5)
        else:
            dts.append(0.0)
    DT3 = (sum(dts) / len(dts)) if dts else None
    # DQ3 = F1
    fp = sum(1 for e in (d.get("injections") or []) for f in (e.get("flags") or []) if f.get("verdict") in FA_VERDICTS)
    recall = (tp / len(fired)) if fired else None
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else (0.0 if fired else None)
    F1 = (2 * precision * recall / (precision + recall)) if (recall is not None and precision is not None and (precision + recall) > 0) else (0.0 if recall is not None else None)
    te_raw, te_norm = wmean([(a, TT3), (c, CC)]) if (in_time is not None and in_time >= 1.0) else (None, None)
    aq_raw, aq_norm = wmean([(b, DT3), (b, F1), (2, SQ)]) if (SQ is not None and DT3 is not None and F1 is not None) else (None, None)
    return {"cap_ticks": cap, "allowance": allowance, "timed_out": timed_out, "in_time_success": in_time,
            "progress_at_cap": progress_at_cap,
            "TT": TT3, "DT": DT3, "DQ_f1": F1, "precision": precision, "recall": recall, "tp": tp, "fp": fp,
            "task_efficiency": te_norm, "task_efficiency_raw": te_raw,
            "assistance_quality": aq_norm, "assistance_quality_raw": aq_raw}


def score_episode(d, key0, ref_ticks, recipe, sq_table, regrade, n_err_planned=2, allowance=0, pro_v2=None):
    q = {it["key"]: it.get("weight_value") or it.get("choice") for it in ((d.get("questionnaire_v2") or {}).get("items") or [])}
    pv = d.get("persona_v2") or {}
    a, b, c = q.get("q1") or pv.get("q1") or 2, q.get("q2") or pv.get("q2") or 2, q.get("q3") or pv.get("q3") or 2
    ticks = d.get("ticks")
    inj = d.get("injections") or []
    moved, ignore = reattribute_release_flags(inj)
    inj = [dict(e, flags=[f for j, f in enumerate(e.get("flags") or []) if (i, j) not in ignore] + moved.get(i, []))
           for i, e in enumerate(inj)]
    errs = [e for e in inj if e.get("family") == "error"]           # suite v3 adds qa_proactive subs: not errors
    fired = [e for e in errs if e.get("injected")]
    qs = [e for e in inj if e.get("family") == "qa_probe" and e.get("injected")]
    # suite v3: per-question family + judge verdict, and the proactive (Q10) grades, for the results page
    try:
        from cooksim_bench.scoring.qa_v3_lib import family_of as _fam
    except Exception:                                            # pragma: no cover
        _fam = lambda k: None
    qa_detail = [{"idx": e.get("idx"), "gt_kind": e.get("gt_kind"), "family": _fam(e.get("gt_kind") or "") or "v2",
                  "verdict": (regrade.get(f"{key0}#{e['idx']}") or {}).get("verdict") or (e.get("qa_grade") or {}).get("verdict"),
                  "answered": bool((e.get("qa") or {}).get("answer"))} for e in qs]
    # Q10: the offline re-grade (python -m cooksim_bench.scoring.regrade_proactive_v2) supersedes the in-episode
    # keyword grader wherever it exists — audit 2026-08-28.
    proactive = []
    for e in inj:
        if e.get("family") != "qa_proactive" or not e.get("injected"):
            continue
        g = dict(e.get("proactive") or {})
        v2 = (pro_v2 or {}).get(f"{key0}#{e.get('idx')}")
        if v2:
            g = dict(v2)
        proactive.append({"idx": e.get("idx"), "item": e.get("item"), **g})
    ts = task_success(d, recipe)
    # TT / CC — task_success is the hard gate: a dish that was not won has no efficiency
    TT = clip01(ref_ticks / ticks) if (ticks and ref_ticks) else None
    ccv = cc(d)
    CC, t_int = (ccv if ccv else (None, None))
    cc_soft = (1.0 - (t_int / (t_int + ticks))) if (t_int is not None and ticks) else None
    gated = ts == 1.0
    # DT / DQ
    DT = (sum(sub_dt(e, ticks) for e in fired) / len(fired)) if fired else None
    # a real problem FOUND counts for detection quality even when found late (timing is DT's job);
    # the audit showed every late catch led to an assistant-directed repair
    def _late_ok(e):
        hi = ((e.get("report_keys") or {}).get("window") or [None, None])[1]
        return hi is not None and any(f.get("verdict") == "late" and isinstance(f.get("t"), (int, float)) and f["t"] <= hi + LATE_GRACE for f in (e.get("flags") or []))
    # ruling 2026-09-24: a late call within LATE_GRACE ticks of the commit is a detection (timing is DT's job)
    _late_idx = late_detections(d, fired)
    hits = [1 if (e.get("flagged_valid") or e.get("prevented") or any(f.get("verdict") in CORRECT_VERDICTS for f in (e.get("flags") or [])) or ei in _late_idx) else 0 for ei, e in enumerate(fired)]
    late = [1 if (not (e.get("flagged_valid") or e.get("prevented") or any(f.get("verdict") in CORRECT_VERDICTS for f in (e.get("flags") or []))) and any(f.get("verdict") == "late" for f in (e.get("flags") or []))) else 0 for e in fired]
    # false alarms = wrong-problem flags; `late` is a correct name after the window (charged in DT/miss, not here)
    fa = sum(1 for e in inj for f in (e.get("flags") or []) if f.get("verdict") not in NEUTRAL_VERDICTS and f.get("verdict") not in ("fp_skipped", "late"))
    budget = max(3.0, 3.0 * (ticks or 100) / 100.0)
    fa_norm = clip01(fa / budget)
    miss = (1 - sum(hits) / len(hits)) if hits else None
    DQ = (1.0 - (0.5 * miss + 0.5 * fa_norm)) if miss is not None else None
    # SQ — no self-judging: a judge model never scores its own episodes (BENCHMARK_DESIGN D10)
    sj = sq_table.get(key0) or {}
    model_under_test = key0.split("/")[2]
    usable = {j: v for j, v in sj.items() if isinstance(v, dict) and isinstance(v.get("score"), (int, float)) and j != model_under_test}
    sqs = [(v["score"] - 1) / 9 for v in usable.values()]
    SQ = clip01(sum(sqs) / len(sqs)) if sqs else None
    judge_gap = (abs(list(usable.values())[0]["score"] - list(usable.values())[1]["score"]) if len(usable) == 2 else None)
    # IQ
    ex = d.get("exit_survey") or {}
    IQ = {k: (clip01((ex[k] - 1) / 6) if isinstance(ex.get(k), (int, float)) else None) for k in ("satisfaction", "reuse")}
    # evidence
    qa_scores = []
    try:
        from cooksim_bench.scoring.qa_v3_lib import qa_score as _qa_score        # the one published QA definition (every family counts)
    except Exception:                                        # pragma: no cover
        _qa_score = lambda v, fam=None: QA_SCORE.get(v)     # noqa: E731
    for e in qs:
        v = (regrade.get(f"{key0}#{e['idx']}") or {}).get("verdict") or (e.get("qa_grade") or {}).get("verdict")
        sc = _qa_score(v, _fam(e.get("gt_kind") or ""))
        if sc is not None:
            qa_scores.append(sc)
    # ---- v2.1: experience-coupled variants (same weights, stricter components) ----
    fr = friction(d)
    fa21 = fa + fr["nag_repeats"] + fr["ask_less"]              # a repeated warning or an over-alerted cook = a false alarm
    # the friction budget follows what the cook ASKED for (Q3): a step-by-step cook wants to be
    # talked through and tolerates repeated instructions; a don't-interrupt cook tolerates none
    budget21 = budget * {1: 4.0, 2: 2.0, 3: 1.0}.get(int(c) if isinstance(c, (int, float)) else 2, 2.0)
    fa21_norm = clip01(fa21 / budget21)
    hit_share = (sum(hits) / len(hits)) if hits else None
    DQ21 = (hit_share * (1.0 - fa21_norm)) if hit_share is not None else None   # multiplicative: both needed
    sq_min = [(v["score"] - 1) / 9 for v in usable.values()]
    SQ21 = clip01(min(sq_min)) if sq_min else None                              # the stricter judge decides
    aq21_raw, aq21_norm = wmean([(b, DT), (b, DQ21), (2, SQ21)]) if (SQ21 is not None and DT is not None and DQ21 is not None) else (None, None)
    te_raw, te_norm = wmean([(a, TT), (c, CC)]) if gated else (None, None)
    # the design's formula needs all three terms; until the judge has scored the episode the
    # headline is null (never a silently re-weighted number)
    aq_raw, aq_norm = wmean([(b, DT), (b, DQ), (2, SQ)]) if (SQ is not None and DT is not None and DQ is not None) else (None, None)
    return {
        "schema": "eval2-v2-sweep", "weights": {"a": a, "b": b, "c": c}, "qa_detail": qa_detail, "proactive": proactive,
        "pace": d.get("pace", "human"),
        "task_success": ts,
        "task_efficiency": {"raw": te_raw, "normalized": te_norm, "TT": TT, "CC": CC, "CC_soft": cc_soft,
                            "t_interrupt": t_int, "ref_ticks": ref_ticks, "gated_on_success": True},
        "assistance_quality": {"raw": aq_raw, "normalized": aq_norm, "DT": DT, "DQ": DQ, "SQ": SQ, "miss": miss,
                               "false_alarms": fa, "fa_norm": fa_norm, "judges": sorted(usable.keys()),
                               "judge_gap": judge_gap, "all_judges": {j: v.get("score") for j, v in sj.items() if isinstance(v, dict)}},
        "interaction_quality": {**IQ, "elicitation": ex.get("elicitation")},
        "v3": v3_block(d, fired, ts, ticks, ref_ticks, allowance, recipe, a, b, c, CC, SQ),
        "v21": {"friction": fr, "fa21": fa21, "fa21_norm": fa21_norm, "DQ21": DQ21, "SQ21": SQ21,
                "assistance_quality": aq21_norm, "assistance_quality_raw": aq21_raw,
                "iq": ((IQ["satisfaction"] + IQ["reuse"]) / 2) if (IQ["satisfaction"] is not None and IQ["reuse"] is not None) else None},
        "evidence": {"outcome": d.get("outcome"), "ticks": ticks, "errors_fired": len(fired), "errors_planned": n_err_planned,
                     "detected": sum(hits), "detected_late": sum(late), "prevented": sum(1 for e in fired if e.get("prevented")),
                     "reattributed_flags": sum(len(v) for v in moved.values()),
                     "committed_caught": sum(1 for e, h in zip(fired, hits) if h and (e.get("plate_verdict_at_release") or {}).get("error_committed") is True),
                     "committed_uncaught": sum(1 for e, h in zip(fired, hits) if not h and (e.get("plate_verdict_at_release") or {}).get("error_committed") is True),
                     "convinced": sum(1 for e in fired if e.get("convinced")),
                     "questions_asked": len(qs), "qa_scores": qa_scores,
                     "words": sum(len((c.get("text") or "").split()) for c in (d.get("conversation") or []) if c.get("speaker") == "vlm"),
                     "engine_rev": (d.get("engine_rev") or {}).get("rev"),
                     # G2: at least one ERROR sub-case fired (a question-first schedule that died before
                     # its first error measured nothing); G3: ran to a verdict, not aborted mid-way
                     "g2": len(fired) >= 1,
                     "g3": d.get("outcome") is not None and not bool((d.get("eval") or {}).get("incomplete"))},
    }


def main():
    cells, rec = recipe_table()
    n_err = {c["name"]: sum(1 for s_ in c["subs"] if s_.get("family") == "error") for c in all_cases()}
    ref = json.load(open(os.path.join(ROOT, "benchmark/ref_ticks.json")))
    allow = json.load(open(os.path.join(ROOT, "benchmark/case_allowance.json")))
    try:
        allow.update(json.load(open(os.path.join(ROOT, "benchmark/case_allowance_v3.json"))))
    except Exception:
        pass
    sq_table = load_side("sq_judge.json")
    pro_v2 = load_side("proactive_v2.json")
    regrade = load_side("qa_regrade.json")
    out, skipped = {}, []
    files = sorted(glob.glob(os.path.join(ROOT, BENCH, "*/*/*/bm_*.json")) + glob.glob(os.path.join(ROOT, BENCH, "*/*/*/b3_*.json")))
    for f in files:
        if "_stale" in f or "_smoke" in f or "_audit" in f:
            continue
        # Public rollouts publish complete JSON with atomic rename. Parse immediately;
        # an interrupted legacy writer is still rejected by the JSON error below.
        key0 = "/".join(f.split("/")[-4:])[:-5]
        try:
            d = json.load(open(f))
        except Exception as e:                              # noqa: BLE001
            skipped.append((f, repr(e)[:80])); continue
        case = key0.split("/")[-1]
        cell = cells.get(case)
        out[key0] = score_episode(d, key0, ref.get(cell), rec.get(cell), sq_table, regrade,
                                  n_err_planned=n_err.get(case, 2), allowance=allow.get(case, 0),
                                  pro_v2=pro_v2)
        out[key0]["cell"] = cell
    tmp = OUT + ".tmp"
    json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": len(out),
               "skipped": [os.path.basename(f) + ": " + why for f, why in skipped], "episodes": out},
              open(tmp, "w"), indent=0)
    os.replace(tmp, OUT)
    with_sq = sum(1 for v in out.values() if v["assistance_quality"]["SQ"] is not None)
    print(f"scored {len(out)} episodes ({with_sq} with SQ); skipped {len(skipped)}; wrote {OUT}")


if __name__ == "__main__":
    main()
