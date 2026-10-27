#!/usr/bin/env python3
"""Quantitative assistant evaluation (design doc docs/ASSISTANT_EVAL_DESIGN.md §2, §4, §5).

    python -m cooksim_bench.scoring.eval_assistant <report.json> [--write]

Pure extraction — NO LLM calls. Reads one rollout report (settings: recovery /
recovery_pause / e2e / e2e_stepin — all share conversation/moves/ticks/outcome) and
computes the Group A/B/C/E criteria, normalizes them against FIXED v1 reference
constants (documented in REF below; silent/oracle anchor runs are future work, §7.3),
applies the persona weight table (§5) and the safety gate, and prints (or with
--write embeds, after backing the report up to <report>.json.bak) an `eval` block.

Group D (experience) comes from python -m cooksim_bench.scoring.eval_judge's `eval_qualitative` block; if the
report already carries one, its questionnaire+judge means fill the D bucket, otherwise
the score is computed with the remaining weights renormalized (`score_includes_D`
says which happened). Run order: eval_assistant → eval_judge --write → eval_assistant
--write again to fold D in.
"""
from __future__ import annotations
import json, sys, os, shutil, collections

POLL = 2                       # ticks between supervisor observation snapshots

# ---------------------------------------------------------------- normalization v1
# Fixed reference constants (worst, best) per criterion — "reference-anchored" §4, with
# constants standing in for the silent/oracle anchor runs until those are cached.
# N(x) = clip((x - worst) / (best - worst), 0, 1); 1 is always better.
REF = {
    "ticks":            (320.0,  60.0),   # worst: dawdling episode; best: oracle-fast solve
    "latency":          (40.0,    0.0),   # valid-flag tick - dev
    "repair_time":      (150.0,  10.0),   # end tick - correction release/flag tick
    "blocked_ticks":    (40.0,    0.0),   # ticks the human sat blocked (status "waiting")
    "talk_words":       (900.0,  80.0),   # total supervisor words
    "turns":            (20.0,    2.0),   # conversation rounds with any speech
    "rounds_to_accept": (6.0,     1.0),   # persuasion: human moves from valid flag to accept
}
# Count-like costs normalize against an episode-length-scaled budget (§4):
# N = 1 - min(count / budget, 1)
FA_BUDGET_PER_100 = 3.0        # false alarms tolerated per 100 ticks (min 2)
NAG_BUDGET_PER_100 = 6.0       # nag events tolerated per 100 ticks (min 3)

# F4b (2026-08-14 fix batch): SINGLE SOURCE. This tuple, the report finaliser's
# `false_positives` and build_new_task's FP tally are the same set by construction —
# they were three hand-maintained literals, and `alias_in_window` had already been added
# to two of them and not the third (X9). `true_observation` joins them all at once.
from cooksim.interact.checks import NEUTRAL_VERDICTS, CORRECT_VERDICTS
OK_VERDICTS = NEUTRAL_VERDICTS                                    # not false alarms
HAZARD_OUTCOMES = ("fire", "hazard", "burned")
PUSHBACK_MOVES = ("question_correctness", "reject", "ignore")

# §5 persona weight table — buckets: A task outcome, B efficiency, C1 false_alarms+nag,
# C2 talk_volume+turns (bad-timing proxy), P persuasion, D experience, E safety+calibration.
WEIGHTS = {
    "novice_stepwise":          dict(A=.30, B=.10, C1=.08, C2=.05, P=.07, D=.20, E=.20),
    "clever_skeptic":           dict(A=.22, B=.15, C1=.18, C2=.10, P=.10, D=.15, E=.10),
    "clever_uninterruptible":   dict(A=.22, B=.15, C1=.15, C2=.15, P=.10, D=.13, E=.10),
    "creative_skeptic":         dict(A=.18, B=.12, C1=.20, C2=.12, P=.08, D=.17, E=.13),
    "creative_uninterruptible": dict(A=.15, B=.12, C1=.18, C2=.22, P=.08, D=.12, E=.13),
}


def clip01(x):
    return max(0.0, min(1.0, x))


def norm(key, x):
    if x is None:
        return None
    worst, best = REF[key]
    return clip01((x - worst) / (best - worst))


def budget_norm(count, per_100, floor, ticks):
    budget = max(floor, per_100 * ticks / 100.0)
    return clip01(1.0 - min(count / budget, 1.0))


def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def token_jaccard(a, b):
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def detect_setting(d):
    s = d.get("setting")
    if s:
        return s                                   # "e2e" | "e2e_stepin"
    return "recovery_pause" if "pauses" in d else "recovery"


# ---------------------------------------------------------------- extraction
def extract(d):
    setting = detect_setting(d)
    conv = d.get("conversation", [])
    moves = list(d.get("moves", []))
    ticks = d.get("ticks")
    outcome = d.get("outcome")
    persona = d.get("persona", "custom")
    is_recovery = setting.startswith("recovery")

    vlm_texts = [(_i(c.get("t")), (c.get("text") or "").strip())
                 for c in conv if c.get("speaker") == "vlm"]
    sup_said = [(t, x) for t, x in vlm_texts if x]

    # ---- Group A ------------------------------------------------------------
    task_success = 1 if outcome == "won" else 0
    prevented = 1 if d.get("prevented") else 0
    if is_recovery:
        # the flag is SCRIPTED ground truth at flag_tick — detection is not under test
        detection_valid = None
        detected_early_missed = None
        recovered = 1 if task_success else 0
        mistake_missed = 0
    else:
        detection_valid = 1 if d.get("flagged_valid") else 0
        detected_early_missed = 1 if (not detection_valid and any(
            f.get("verdict") in ("early_correct", "early_alias")
            for f in d.get("flags", []))) else 0
        recovered = 1 if ((detection_valid or detected_early_missed)
                          and task_success and not prevented) else 0
        mistake_missed = 1 if d.get("failed_at") == "detection" else 0

    # ---- Group B ------------------------------------------------------------
    latency = d.get("latency")
    if is_recovery:
        end = d.get("resolved_at") or ticks
        repair_time = end - d["flag_tick"] if d.get("flag_tick") is not None else None
    else:
        start = d.get("released_at") or d.get("entered_at") or d.get("armed_at")
        repair_time = (ticks - start) if start is not None else None

    # blocked time: engine snapshots whose past_actions contain a "waiting" entry
    waiting_ticks = set()
    for c in conv:
        eng = c.get("engine") or {}
        for a in ((eng.get("human") or {}).get("past_actions")) or []:
            if a.get("status") == "waiting":
                t = _i(c.get("t"))
                if t is not None:
                    waiting_ticks.add(t)
    blocked_ticks = len(waiting_ticks) * POLL

    # ---- Group C ------------------------------------------------------------
    flags = d.get("flags", [])
    false_alarms = sum(1 for f in flags
                       if f.get("verdict") not in OK_VERDICTS
                       and f.get("verdict") != "fp_skipped")
    # nag: flag_repeats + supervisor texts near-duplicating one of its last 3 texts
    nag_repeat = 0
    recent = []
    for _, x in sup_said:
        if any(token_jaccard(x, r) >= 0.6 for r in recent):
            nag_repeat += 1
        recent = (recent + [x])[-3:]
    nag_count = (d.get("flag_repeats") or 0) + nag_repeat

    talk_volume = sum(len(x.split()) for _, x in sup_said)
    turns = len({_i(c.get("round")) for c in conv if (c.get("text") or "").strip()})
    pushback = collections.Counter(m for m in moves if m in PUSHBACK_MOVES)

    # persuasion: human moves from the first landed-correct flag until the first accept
    rounds_to_accept = None
    if is_recovery:
        if "accept" in moves:
            rounds_to_accept = moves.index("accept") + 1
    else:
        seen_flag = False
        n = 0
        for c in conv:
            if c.get("speaker") == "vlm" and c.get("flag_verdict") in CORRECT_VERDICTS:
                seen_flag = True
            elif seen_flag and c.get("speaker") == "human" and c.get("move"):
                n += 1
                if c["move"] == "accept":
                    rounds_to_accept = n
                    break

    # ---- Group E ------------------------------------------------------------
    # pair each flagged supervisor utterance with the next human move
    pairs = []
    pending = None
    for c in conv:
        if c.get("speaker") == "vlm" and c.get("flag_verdict"):
            pending = c["flag_verdict"]
        elif c.get("speaker") == "human" and c.get("move") and pending is not None:
            pairs.append((pending, c["move"]))
            pending = None
    tp = [(v, m) for v, m in pairs if v in CORRECT_VERDICTS]
    fp = [(v, m) for v, m in pairs if v not in OK_VERDICTS and v != "fp_skipped"]
    if is_recovery:
        # the single scripted flag is correct by construction: compliance = did the human
        # ever accept it (or end up convinced); no FPs exist in this setting
        compliance_correct = 1.0 if ("accept" in moves or d.get("convinced")) else 0.0
        rejection_incorrect = None
    else:
        compliance_correct = (sum(1 for _, m in tp if m == "accept") / len(tp)) if tp else None
        rejection_incorrect = (sum(1 for _, m in fp if m in PUSHBACK_MOVES) / len(fp)) if fp else None
    cal_parts = [x for x in (compliance_correct, rejection_incorrect) if x is not None]
    calibration = sum(cal_parts) / len(cal_parts) if cal_parts else None

    hazard = outcome in HAZARD_OUTCOMES

    return dict(
        setting=setting, persona=persona, scenario=d.get("scenario"),
        outcome=outcome, ticks=ticks,
        # A
        task_success=task_success, prevented=prevented, recovered=recovered,
        mistake_missed=mistake_missed, detection_valid=detection_valid,
        detected_early_missed=detected_early_missed,
        # B
        detection_latency=latency, repair_time=repair_time, blocked_ticks=blocked_ticks,
        # C
        false_alarms=false_alarms, nag_count=nag_count, talk_volume=talk_volume,
        turns=turns, rounds_to_accept=rounds_to_accept,
        pushback={k: pushback.get(k, 0) for k in PUSHBACK_MOVES},
        # E
        hazard=hazard, compliance_correct=compliance_correct,
        rejection_incorrect=rejection_incorrect, calibration=calibration,
        convinced=bool(d.get("convinced")),
    )


# ---------------------------------------------------------------- buckets + score
def bucketize(c):
    """Normalized [0,1] value per weight-table bucket (1 = better)."""
    is_recovery = c["setting"].startswith("recovery")

    if is_recovery:
        A = 0.5 * c["task_success"] + 0.5 * (1.0 if c["convinced"] else 0.0)
    else:
        detect = 1.0 if (c["detection_valid"] or c["prevented"]) else \
                 (0.5 if c["detected_early_missed"] else 0.0)
        resolve = 1.0 if (c["prevented"] or c["recovered"]) else 0.0
        A = (c["task_success"] + detect + resolve) / 3.0

    b_parts = [norm("ticks", c["ticks"]), norm("repair_time", c["repair_time"]),
               norm("blocked_ticks", c["blocked_ticks"])]
    if c["detection_latency"] is not None:
        b_parts.append(norm("latency", c["detection_latency"]))
    b_parts = [x for x in b_parts if x is not None]
    B = sum(b_parts) / len(b_parts) if b_parts else 0.5

    C1 = 0.5 * budget_norm(c["false_alarms"], FA_BUDGET_PER_100, 2.0, c["ticks"]) \
       + 0.5 * budget_norm(c["nag_count"], NAG_BUDGET_PER_100, 3.0, c["ticks"])

    C2 = 0.5 * norm("talk_words", c["talk_volume"]) + 0.5 * norm("turns", c["turns"])

    if c["rounds_to_accept"] is not None:
        P = norm("rounds_to_accept", c["rounds_to_accept"])
    elif is_recovery and c["convinced"]:
        P = 0.7            # convinced without an explicit accept move
    elif is_recovery or c["detection_valid"] or c["detected_early_missed"]:
        P = 0.0            # a correct claim landed but the human was never persuaded
    else:
        P = 0.5            # nothing correct to persuade about: neutral

    safety = 0.0 if c["hazard"] else 1.0
    E = 0.5 * safety + 0.5 * (c["calibration"] if c["calibration"] is not None else 0.5)
    return dict(A=A, B=B, C1=C1, C2=C2, P=P, E=E)


def d_bucket(qual):
    """Group D from an eval_qualitative block: mean questionnaire (1-10) and mean judge
    rubric (1-10), each mapped (x-1)/9, averaged."""
    if not qual:
        return None
    parts = []
    ans = ((qual.get("questionnaire") or {}).get("answers")) or {}
    vals = [v for v in ans.values() if isinstance(v, (int, float))]
    if vals:
        parts.append((sum(vals) / len(vals) - 1) / 9.0)
    js = ((qual.get("judge") or {}).get("scores")) or {}
    vals = [v for v in js.values() if isinstance(v, (int, float))]
    if vals:
        parts.append((sum(vals) / len(vals) - 1) / 9.0)
    return clip01(sum(parts) / len(parts)) if parts else None


def score_report(d):
    c = extract(d)
    buckets = bucketize(c)
    # WALL-CLOCK ABORT != FAILURE. An episode cut off by the harness budget never reached
    # a verdict: the dish was neither served nor lost. Scoring it as a failed task (A=0,
    # and on the e2e side a missed detection) reports a busy machine as a bad assistant,
    # and makes such runs incomparable with completed ones. They are marked INCOMPLETE
    # instead: bucket A drops out of the weighted mean and the report says so.
    incomplete = bool(d.get("aborted")) and c["outcome"] not in ("won", "lost", "fire")
    if incomplete:
        buckets["A"] = None
    D = d_bucket(d.get("eval_qualitative"))
    w = dict(WEIGHTS.get(c["persona"],
                         {k: 1 / 7 for k in ("A", "B", "C1", "C2", "P", "D", "E")}))
    if D is not None:
        buckets["D"] = D
        weights = w
    else:                       # renormalize the remaining weights (D pending)
        rest = {k: v for k, v in w.items() if k != "D"}
        tot = sum(rest.values())
        weights = {k: v / tot for k, v in rest.items()}
    live = {k: v for k, v in buckets.items() if v is not None}
    tot_w = sum(weights[k] for k in weights if k in live) or 1.0
    weights = {k: v / tot_w for k, v in weights.items() if k in live}
    terms = {k: weights[k] * live[k] for k in weights}
    score = sum(terms.values())
    gated = bool(c["hazard"])
    if gated:                   # SafetyGate (§1): a fire caps the episode at 0
        score = 0.0
    return dict(schema="assist-eval-v1", incomplete=incomplete,
                aborted=d.get("aborted"), **{"criteria": c},
                normalized=buckets, weights=weights,
                weighted_terms={k: round(v, 4) for k, v in terms.items()},
                score=round(score, 4), score_includes_D=D is not None,
                safety_gated=gated,
                normalization="fixed-reference-v1",
                reference_constants=dict(REF, fa_budget_per_100=FA_BUDGET_PER_100,
                                         nag_budget_per_100=NAG_BUDGET_PER_100))


def main():
    args = [a for a in sys.argv[1:] if a != "--write"]
    write = "--write" in sys.argv
    if not args:
        sys.exit(__doc__)
    for path in args:
        d = json.load(open(path))
        # W4 (2026-08-12 audit): episodes whose injection never fired measured NOTHING
        # about the case, yet were scored (0.33-0.38) and charged false alarms for an
        # error that never existed. Not measured is not a score.
        # E13 (round-2 review): a v5 composite is measured when any of its error sub-cases fired — read injections[],
        # not the singleton fields (which describe only the last sub-case; older records carry the stale label)
        _v5_fired = (d.get("scheduler") == "v5" and (d.get("composite") or {}).get("was_composite")
                     and any(e.get("family") == "error" and e.get("injected") for e in (d.get("injections") or [])))
        if (d.get("failed_at") == "injection_skipped" or d.get("measured") is False) and not _v5_fired:
            print(f"{os.path.basename(path)}  NOT MEASURED — injection never fired "
                  f"({(d.get('injection_skipped') or {}).get('why', 'unknown')}); no score")
            if write:
                d["eval"] = {"not_measured": "injection never fired",
                             "why": (d.get("injection_skipped") or {}).get("why")}
                json.dump(d, open(path, "w"), indent=1)
            continue
        ev = score_report(d)
        c = ev["criteria"]
        print(f"{os.path.basename(path)}  [{c['setting']}/{c['persona']}]  "
              f"outcome={c['outcome']}  score={ev['score']:.3f}"
              f"{' (D pending)' if not ev['score_includes_D'] else ''}"
              f"{' SAFETY-GATED' if ev['safety_gated'] else ''}")
        if ev.get("incomplete"):
            print(f"    INCOMPLETE — cut off by {ev.get('aborted')}; bucket A is undefined "
                  f"and its weight was redistributed, so this score is NOT comparable "
                  f"with a completed episode's")
        for k in ("A", "B", "C1", "C2", "P", "D", "E"):
            if k not in ev["normalized"]:
                continue
            v = ev["normalized"][k]
            if v is None:                      # bucket dropped (e.g. A on an abort)
                print(f"    {k:>2} = —      (not applicable)")
                continue
            print(f"    {k:>2} = {v:.3f} × w {ev['weights'][k]:.3f}"
                  f" = {ev['weighted_terms'][k]:.4f}")
        if write:
            bak = path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            d["eval"] = ev
            json.dump(d, open(path, "w"), indent=1)
            print(f"    -> eval block written into {path} (backup: {bak})")


if __name__ == "__main__":
    main()
