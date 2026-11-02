#!/usr/bin/env python3
"""Assistant metrics for the household benchmark, following the CookSim v6 metrics contract.

Every rate is reported with its denominator, and a metric the household engine cannot produce is
reported as NOT MEASURED with the reason rather than as a zero — a metric that cannot vary is a
permanent zero that looks like a result.

  python -m vhhome_bench.scoring.metrics SWEEPDIR [--json out.json]

Implemented (formulas verbatim from the contract):
  S   in-time success      1[won] * 1[T <= C],  C = 2*(T_ref + alpha)
  TT  total-time efficiency 0 if T > C else clip_[0,1]((C-T)/(C-T_ref))
  CC  conversation eff.     1 - clip_[0,1](sum_u tau(u) / T)
  E   efficiency            (a*TT + c*CC)/(a+c)   over S=1 runs only
  DT  detection timing      mean over injected errors: 1 prevented, 0.5 correct flag, else 0
  P,R,F1 detection          R over errors that fired; P = TP/(TP+FP)

Not measured here, with reasons:
  QA  answer accuracy   the household episodes ask the assistant no scored questions
  A   alert score       there is no "tell me when it is done" standing request in this suite
  AQ  assistance quality v4  needs the 15-item rubric judge; not run in this sweep
"""
from vhhome import paths
import glob, json, os, statistics, sys

ENG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from vhhome_bench import protocol as P
from vhhome_bench import cases as CS
from vhhome import plan as pir

ALPHA = 20          # allowance in the cap C = 2*(T_ref + alpha)
POLL = 2            # ticks between assistant turns; one utterance occupies about one poll

# Persona weights for E. `c` (interruption cost) comes from the persona's own interaction
# preference — the questionnaire answer the contract weights by. `a` (speed) is not a dial in
# this engine, so it is held neutral and that is stated rather than invented.
C_WEIGHT = {"avoidant": 3, "ok": 1}
A_WEIGHT = 2


def clip(x):
    return max(0.0, min(1.0, x))


_TREF_CLEAN = None


def t_ref_measured(case):
    """Reference time of a CASE: the clean plan executed in the engine (plan seeded, no error, silent assistant), from
    benchmark/t_ref.json. None for a case that was never measured."""
    global _TREF_CLEAN
    if _TREF_CLEAN is None:
        f = paths.T_REF
        _TREF_CLEAN = json.load(open(f)).get("by_case", {}) if os.path.exists(f) else {}
    v = _TREF_CLEAN.get(case)
    return float(v) if v else None


def t_ref(epk):
    """Reference time from the clean plan replayed SYMBOLICALLY — the fallback only. It runs 1.6-2.4x shorter than the
    plan executed in the engine (walks and door passes cost more ticks than the symbolic model charges), so on its own
    it put the cap C = 2(T_ref + 20) below the clean execution time of half the chores and in-time success was ~0 for
    every seat. episode_metrics therefore takes the MEASURED case time first (t_ref_measured) — the definition the
    experiment dumps have used since 528baa4a — and this only when a case has no measurement."""
    try:
        cmp_ = CS.build_deviated(epk, [])[2]
        r = pir.simulate([(s, sid, False) for s, sid, _ in cmp_], epk)
        return float(r.get("time") or 0) or float(len(cmp_) * 3)
    except Exception:
        return 60.0


# WINDOW CLOSE (2026-10-15, the version audit). The window used to run to the end of the episode for every class,
# so flags about the person's own later wandering were credited as catching a mistake whose takeover had ended long
# before (a navigation flag 80 ticks after a skipped step's release counted as its valid_alias; stall flags while the
# person was walking counted for a stall released 40 ticks earlier). Now:
#   * a TRANSIENT mistake (stall, navigation_inefficiency) leaves nothing in the world once its takeover ends, so its
#     window closes RELEASE_GRACE ticks after the release, for the exact name as well as for its twins;
#   * a PERSISTENT mistake (skipped/missing step, wrong object, wrong placement, repetition) stays true in the world, so
#     the exact name is still credited until the end of the episode, but a TWIN name only up to RELEASE_GRACE ticks
#     after the release (a later twin-named flag is about something else).
# VH_WINDOW_CLOSE=0 restores the open windows.
RELEASE_GRACE = int(os.environ.get("VH_RELEASE_GRACE", "30"))
WINDOW_CLOSE = os.environ.get("VH_WINDOW_CLOSE", "1") != "0"
TRANSIENT = {"stall", "navigation_inefficiency"}
PERSON_CREDIT = os.environ.get("VH_PERSON_CREDIT", "1") != "0"


def _phase_verdict(p, nm, t):
    lo = (p.get("window") or [10**9])[0]
    rel = p.get("released_at")
    close = (rel + RELEASE_GRACE) if (WINDOW_CLOSE and rel is not None) else None
    hi = close if (close is not None and p["err"] in TRANSIENT) else 10**9
    hi = max(hi, lo)
    v = P.verdict_of({"err": p["err"], "dev": p["dev_live"], "window": (lo, hi)}, nm, t)
    if v == "valid_alias" and close is not None and t > close:
        v = "wrong_name"
    return v, lo


def episode_metrics(d):
    """Metrics for one rollout report."""
    epk, pers = d.get("episode"), d.get("persona")
    T = float(d.get("ticks") or 0)
    Tref = t_ref_measured(d.get("scenario")) or t_ref(epk)
    C = 2.0 * (Tref + ALPHA)
    won = (d.get("outcome") == "won")
    S = 1.0 if (won and T <= C) else 0.0
    TT = 0.0 if T > C else clip((C - T) / max(C - Tref, 1e-9))
    spoke = sum(1 for c in (d.get("conversation") or [])
                if c.get("speaker") == "vlm" and (c.get("text") or "").strip())
    CC = 1.0 - clip((spoke * POLL) / max(T, 1e-9))
    prof = P.PROFILES.get(pers, {})
    c_w = C_WEIGHT.get(prof.get("interaction_preference", "ok"), 1)   # derived from the interrupt dial
    E = (A_WEIGHT * TT + c_w * CC) / (A_WEIGHT + c_w) if S == 1.0 else None

    # REGRADE from the stored ticks rather than trusting the verdict the rollout wrote.
    # The rollout closed an earlier phase's window at the next error's onset; the metrics
    # contract runs a window to acceptance/rejection or END OF EPISODE, and closing early throws
    # away a correct-but-late detection (an assistant naming the forgotten lamp two ticks after
    # the next error began was scored a miss). Overlap is safe because a flag is credited to the
    # phase that grades it BEST — naming B exactly earns `valid` for B, which outranks the
    # `valid_alias` it would earn against a twin phase A.
    phases = d.get("phases") or []
    fired = [p for p in phases if p.get("dev_live") is not None]
    # Read the flags from the model's OWN replies, not from the rollout's flag list. The rollout
    # accepts a flag only as {"error": "<label>"}; one local model emits the bare label as a
    # string ("flag": "wrong_placement"), so 115 of its 118 genuine detections were discarded and
    # it scored a flat zero. A bare taxonomy label is unambiguous, so it is graded here — and how
    # often a model met the declared shape is reported separately as schema compliance, which is
    # the thing that actually varied.
    flags, seen, shape_ok, shape_tot = [], set(), 0, 0
    for c in (d.get("conversation") or []):
        if c.get("speaker") != "vlm":
            continue
        fl = (c.get("vlm_json") or {}).get("flag")
        if not fl:
            continue
        shape_tot += 1
        if isinstance(fl, dict):
            shape_ok += 1
            nm = str(fl.get("error") or "").strip()
        else:
            nm = str(fl).strip()
        if not nm or nm in seen:
            continue          # repeated flags count once, as the contract requires
        seen.add(nm)
        flags.append({"error": nm, "t": c.get("t")})
    RANK = {"valid": 0, "valid_alias": 1, "early_correct": 2, "early_alias": 3,
            "late": 4, "early": 5, "wrong_name": 6, "invalid_label": 7}
    caught_ph, fp = set(), 0
    for f in flags:
        nm, t = f.get("error"), f.get("t")
        cands = []
        for i, p in enumerate(fired):
            v, lo = _phase_verdict(p, nm, t)
            # tie-break toward the phase whose window actually contains the flag
            cands.append((RANK.get(v, 99), 0 if lo <= t else 1, i, v))
        if not cands:
            continue
        cands.sort()
        _r, _c, i, v = cands[0]
        f["_regraded"], f["_phase"] = v, i
        if v in P.ARMS_DETECTION or v in P.ARMS_PREVENTION:
            caught_ph.add(i)
        elif v not in P.NEUTRAL:
            fp += 1
    # PREVENTION (2026-10-15). VH never records a `prevented` release, so DT could not exceed 0.5 even when the
    # assistant named a mistake before it happened. Now a flagged mistake is PREVENTED (DT 1) if a detection-grade
    # flag, or the convinced release it led to, came at or before the tick the mistake physically happened
    # (dev_live). VH_PERSON_CREDIT=0 restores the old rule.
    early_flag = {}
    for f in flags:
        i = f.get("_phase")
        if i is None or f.get("t") is None:
            continue
        if (f.get("_regraded") in P.ARMS_DETECTION or f.get("_regraded") in P.ARMS_PREVENTION) and f["t"] <= fired[i]["dev_live"]:
            early_flag[i] = True
    dt, tp = [], 0
    for i, p in enumerate(fired):
        # a convinced release counts only when the assistant also FLAGGED this mistake (caught_ph): the person's own
        # call alone is unreliable (the version audit found it wrong in ~40% of episodes, often `correct` with no
        # accusation at all), and crediting it let seats that never flag score DET 0.4-0.55.
        conv = (PERSON_CREDIT and i in caught_ph and p.get("released_reason") == "convinced"
                and p.get("released_at") is not None)
        prevented = (p.get("released_reason") == "prevented"
                     or (conv and p["released_at"] <= p["dev_live"])
                     or (PERSON_CREDIT and early_flag.get(i)))
        if prevented:
            dt.append(1.0); tp += 1
        elif i in caught_ph:
            dt.append(0.5); tp += 1
        else:
            dt.append(0.0)
    DT = statistics.mean(dt) if dt else None
    R = (tp / len(fired)) if fired else None
    Pr = (tp / (tp + fp)) if (tp + fp) else None
    F1 = (2 * Pr * R / (Pr + R)) if (Pr and R) else (0.0 if (Pr is not None and R is not None) else None)
    schema = (shape_ok / shape_tot) if shape_tot else None
    return dict(arm=d.get("_arm"), episode=epk, persona=pers, T=T, T_ref=Tref, C=C,
                schema=schema, flags_raw=shape_tot,
                outcome=d.get("outcome"), S=S, TT=TT, CC=CC, E=E, DT=DT, P=Pr, R=R, F1=F1,
                utterances=spoke, phases=len(phases), fired=len(fired), tp=tp, fp=fp,
                prevented_n=sum(1 for x in dt if x == 1.0),
                caught=d.get("phases_caught"), prevented=d.get("phases_prevented"),
                detected=d.get("phases_detected", d.get("phases_caught")), total=d.get("phases_total"))


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def main():
    d0 = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(d0, "*.json"))):
        if os.path.basename(f).startswith("METRICS"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        d["_arm"] = os.path.basename(f).split("__")[0]
        rows.append(episode_metrics(d))
    if not rows:
        print("no reports yet"); return
    arms = sorted({r["arm"] for r in rows})
    METRICS = [("S", "In-time success"), ("E", "Efficiency"), ("CC", "Conversation efficiency"),
               ("DT", "Detection timing"), ("P", "Precision"), ("R", "Recall"),
               ("F1", "Detection F1"), ("schema", "Flag schema compliance")]
    print(f"\n{'metric':24}" + "".join(f"{a:>18}" for a in arms))
    print("-" * (24 + 18 * len(arms)))
    for k, label in METRICS:
        line = f"{label:24}"
        for a in arms:
            m, n = agg([r for r in rows if r["arm"] == a], k)
            line += f"{('—' if m is None else f'{m:.3f}') + f' (n={n})':>18}"
        print(line)
    print("-" * (24 + 18 * len(arms)))
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        print(f"{a:24} rollouts={len(sub)}  errors_fired={sum(r['fired'] for r in sub)}  "
              f"caught={sum(r['tp'] for r in sub)}  false_alarms={sum(r['fp'] for r in sub)}  "
              f"utterances={sum(r['utterances'] for r in sub)}")
    print("\nNOT MEASURED in this suite: QA (no scored questions are asked), "
          "A (no standing alert request), AQ (15-item rubric judge not run).")
    if "--json" in sys.argv:
        json.dump({"rows": rows}, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
