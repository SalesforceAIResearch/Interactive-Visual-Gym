#!/usr/bin/env python3
"""Judgement for the NON-ERROR intent scenarios (plan_change / qa_probe).

    python -m cooksim_bench.scoring.eval_intent tmp/intent/*.json [--write] [--judge]

python -m cooksim_bench.scoring.eval_assistant cannot score these: every criterion it owns is anchored on an
injected ERROR (detection validity, latency, recovery). Here nothing is wrong, so the
questions are different — and the headline one is the user's: **was the final delivery
judged successfully?**

  A  delivery        — did the correct Cheeseburger still reach the pass? The cook's plan
                       change / side quest / question must not cost it the order. This is
                       the pass-fail gate: everything else is conduct.
  B  restraint       — did the assistant call a legitimate choice a MISTAKE? Every flag
                       raised while the deviation was live is a false alarm by
                       construction (`intent_false_alarm`), because the scenario contains
                       no error to find. 1.0 = never falsely accused the cook.
  C  autonomy        — did the cook keep its own plan? `intent_carried_out` true and
                       `intent_abandoned` absent means the assistant respected a decision
                       that was the cook's to make. (A cook talked out of a correct plan
                       is the failure mode this family exists to expose.)
  D  answer accuracy — qa_probe only: the assistant's first reply after the question is
                       graded against ground truth read straight from the engine, at BOTH
                       the moment the question was asked and the moment it was answered
                       (the kitchen moves; an answer true when given is not wrong).
                       Deterministic checks first; --judge adds a Gemini adjudication for
                       phrasing, and disagreements are reported rather than hidden.
  E  economy         — words spoken and turns taken around the deviation (an assistant
                       that lectures through a legitimate plan change is still costly).

The composite is deliberately NOT a persona-weighted score like assist-eval-v1: these
scenarios have no error-recovery trade-off to weigh. It is the mean of the buckets that
apply to the family, with delivery as a hard gate.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys

import os
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
TALK_REF = (900.0, 80.0)          # same reference anchors as assist-eval-v1
TURN_REF = (20.0, 2.0)


def norm(ref, x):
    worst, best = ref
    if x is None:
        return None
    return max(0.0, min(1.0, (x - worst) / (best - worst)))


# ------------------------------------------------------------------ QA grading
def _nums(t):
    return [int(x) for x in re.findall(r"\b\d+\b", str(t))]


# Yes/no cues for the history probes. Negation is tested FIRST: "no, you haven't done that
# yet" contains no affirmation, but "yes" appears inside plenty of negative sentences.
_NEGATE = re.compile(r"\b(no|nope|not yet|never|haven'?t|hasn'?t|had not|have not|"
                     r"didn'?t|did not|don'?t|doesn'?t|isn'?t|nowhere|none)\b")
# STRONG affirmations assert the thing happened. WEAK ones ("done", "finished") appear
# just as readily inside a denial — "you haven't done that yet" is a NO that contains the
# word "done", and counting it as an affirmation made every clean denial ungradable.
_AFFIRM = re.compile(r"\b(yes|yeah|yep|yup|already|"
                     # "you have NOT" is a denial that literally contains "you have". The
                     # contracted form was already safe (the trailing \b fails inside
                     # "haven't"), so only the spaced form leaked, and it turned every
                     # plainly-worded "No, you have not." into `ungradable`.
                     r"you (?:have|did|went|visited)\b(?!\s+not\b)|"
                     r"you'?ve(?!\s+not\b)|that'?s right|correct)\b")
_AFFIRM_WEAK = re.compile(r"\b(done|finished|completed)\b")

# "right" is the most common false friend in English prose ("right now", "you're right"),
# and a spatial grader that counts those reads a correct answer as naming the right-hand
# side of the kitchen. Strip the idioms before looking for directions.
_RIGHT_IDIOM = re.compile(r"\b(right (?:now|away|there|then|before|after)|all right|"
                          r"alright|that'?s right|you'?re right|right\?)")
_VER = {"top": "top", "upper": "top", "above": "top",
        "bottom": "bottom", "lower": "bottom", "below": "bottom"}
_HOR = {"left": "left", "right": "right"}
_MID = ("middle", "centre", "center", "central")


def _norm_answer(a):
    a = _RIGHT_IDIOM.sub(" ", a)
    return re.sub(r"[\s_\-]+", " ", a)


def _region_words(a):
    """(vertical words, horizontal words, middle?) named anywhere in the answer."""
    t = _norm_answer(a)
    ver = {v for k, v in _VER.items() if re.search(rf"\b{k}\b", t)}
    hor = {v for k, v in _HOR.items() if re.search(rf"\b{k}\b", t)}
    mid = any(re.search(rf"\b{m}\b", t) for m in _MID)
    return ver, hor, mid


def _match_region(label, a):
    """How well the answer names GT region `label`. True / "partial" / False / None."""
    ver, hor, mid = _region_words(a)
    if not (ver or hor or mid):
        return None                        # no spatial claim at all -> nothing to grade
    if label == "center":
        return mid and not (ver or hor)
    parts = label.split("-")
    want_v = parts[0] if parts[0] in ("top", "bottom") else None
    want_h = parts[1] if len(parts) > 1 else None
    got = 0
    need = (1 if want_v else 0) + (1 if want_h else 0)
    if want_v and want_v in ver:
        got += 1
    if want_h and want_h in hor:
        got += 1
    if got == need:
        return True
    return "partial" if got else False


def grade_answer(gt_kind, gt_ask, gt_ans, answer):
    """Deterministic grade of the assistant's answer against engine ground truth.

    Returns (verdict, why). Truth at ask-time OR answer-time both count as correct — the
    kitchen moves, and an answer that was true when it was given is not wrong.

    Every kind in the case pool has a matcher here. There used to be three, behind a dict
    lookup that raised KeyError on anything else — and it was reached only because
    `_kind_of` silently defaulted every probe to `plate_count`, so a correct answer about
    a pan was graded against a plate count. Both halves are fixed; a kind with no matcher
    now returns `ungradable` (which `--judge` can still adjudicate) rather than guessing.
    """
    a = (answer or "").lower()
    if not a.strip():
        return "no_answer", "the assistant never answered"
    notes = []

    def _named(names):
        """Every required name appears in the answer."""
        return all(n in a for n in names)

    def ok_plate_count(gt):
        n = gt.get("count")
        if n is None:
            return None
        said = _nums(a)
        words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        for w, v in words.items():
            if re.search(rf"\b{w}\b", a):
                said.append(v)
        return (n in said) if said else None

    def ok_plate_contents(gt):
        want = gt.get("held_plate_contents_in_order") or []
        if not want:
            return None
        pos = [a.find(w) for w in want]
        if any(p < 0 for p in pos):
            return False
        return pos == sorted(pos)          # every component named, in the order stated

    def _cooking(names, done):
        if not _named(names):
            return False
        if done is True:
            return bool(re.search(r"\b(done|ready|cooked|finished)\b", a))
        if done is False:
            return bool(re.search(r"\b(not|still|isn'?t|no[t]? yet|cooking|almost)\b", a))
        return True

    def ok_pan_state(gt):
        pans = gt.get("pans_with_food") or []
        names = {n for p in pans for n in p["contents"]}
        return _cooking(names, gt.get("done")) if names else None

    def ok_station_state(gt):
        st = gt.get("cooking_stations_with_food") or []
        names = {n for s in st for n in s["contents"]}
        if not names:
            return None
        # loaded-but-unlit is the distinction this kind exists for: every station raw at
        # zero progress and an answer claiming it is cooking is wrong, not merely vague.
        cold = [s for s in st if s.get("status") == "raw" and not (s.get("progress") or 0)]
        if len(cold) == len(st) and re.search(r"\b(cooking|is on|heating)\b", a):
            notes.append("every station is LOADED BUT UNLIT "
                         f"({[s['contents'] for s in cold]}) and the answer says it is cooking")
            return False
        return _cooking(names, gt.get("all_done"))

    def ok_board_state(gt):
        boards = gt.get("boards") or []
        names = {c["name"] for b in boards for c in (b.get("contents") or [])}
        if not names:
            return bool(_NEGATE.search(a)) or None
        return _named(names)

    def ok_holding(gt):
        if "holding" not in gt:
            return None
        h = gt.get("holding")
        if h is None:
            return bool(re.search(r"\b(nothing|empty|empty-handed|not holding|no)\b", a))
        if h == "plate":
            if "plate" not in a:
                return False
            cs = gt.get("plate_contents_in_order") or []
            return _named(cs) if cs else True
        return h in a

    def ok_remaining(gt):
        left = gt.get("still_to_plate_in_order")
        if left is None:
            on = gt.get("on_plate_in_order")
            return _named(on) if on else None
        if not left:
            return bool(re.search(r"\b(nothing|done|complete|all there|everything)\b", a))
        return _named(left)

    def ok_where(gt):
        found = gt.get("found")
        if found is None:
            return None
        if not found:                       # the honest "it is not in this kitchen"
            ver, hor, mid = _region_words(a)
            if _NEGATE.search(a):
                return True
            return False if (ver or hor or mid) else None
        best = None
        order = {True: 3, "partial": 2, False: 1, None: 0}
        for hit in found:                   # any of the real locations will do
            r = _match_region(hit.get("region") or "", a)
            if best is None or order[r] > order[best]:
                best = r
        return best

    def _yes_no(truth):
        if truth is None:
            return None
        # These assistants answer and then coach: "No, you didn't go to the trash, you just
        # went to the plate source. You have the bun on the plate now, so go to the pan."
        # The "you have" belongs to the ADVICE, not to the answer, and scanning the whole
        # reply made a clean, correct denial look self-contradictory (ungradable). Read the
        # response clause first; only fall back to the whole reply if it carries no signal.
        head = re.split(r"(?<=[.!?])\s", a.strip(), maxsplit=1)[0]
        for seg in (head, a):
            neg, aff = _NEGATE.search(seg), _AFFIRM.search(seg)
            if neg and aff:
                continue                     # genuinely mixed here — try the wider text
            if neg:
                return truth is False        # a denial, weak "done"/"finished" and all
            if aff or _AFFIRM_WEAK.search(seg):
                return truth is True
        return None

    def ok_past_subgoal(gt):
        return _yes_no(gt.get("finished"))

    def ok_visited(gt):
        return _yes_no(gt.get("visited"))

    table = {"plate_count": ok_plate_count, "plate_contents": ok_plate_contents,
             "pan_state": ok_pan_state, "station_state": ok_station_state,
             "board_state": ok_board_state, "holding": ok_holding,
             "remaining": ok_remaining, "where": ok_where,
             "past_subgoal": ok_past_subgoal, "visited": ok_visited}
    base = (gt_kind or "").split(":", 1)[0]
    fn = table.get(base)
    if fn is None:
        return "ungradable", f"no deterministic matcher for gt_kind={gt_kind!r}"

    r_ask, r_ans = fn(gt_ask or {}), fn(gt_ans or {})
    if r_ask is True or r_ans is True:
        return "correct", ("matches ground truth at "
                           + ("ask time" if r_ask is True else "answer time"))
    if "partial" in (r_ask, r_ans):
        return "partially_correct", "names part of the ground-truth location"
    if r_ask is False or r_ans is False:
        return "incorrect", ("contradicts the engine state"
                             + (" — " + notes[0] if notes else ""))
    return "ungradable", "no deterministic anchor in the answer"


JUDGE_PROMPT = """You are grading ONE answer an AI kitchen assistant gave a cook.

The assistant only ever sees camera frames of the kitchen — it cannot query the engine.
The cook asked a question about what is visible. Below is the question, the assistant's
answer, and the TRUE state of the kitchen taken from the engine at the moment the
question was asked and at the moment it was answered.

QUESTION (as the cook actually put it): {q}
ASSISTANT'S ANSWER: {a}
GROUND TRUTH WHEN ASKED:    {g1}
GROUND TRUTH WHEN ANSWERED: {g2}

Grade ONLY factual correctness against either ground truth (the kitchen moves; an answer
that was true when given is correct). Ignore politeness, extra advice, and phrasing.
Reply with JSON: {{"verdict": "correct"|"partially_correct"|"incorrect"|"no_answer",
"why": "<one sentence>"}}"""


def judge_answer(rep, client=None):
    qa = rep.get("qa") or {}
    if not qa.get("answer"):
        return None
    from cooksim_bench.backends.genai_client import make_client, default_model
    from google.genai import types
    client = client or make_client()
    r = client.models.generate_content(
        model=default_model(),
        contents=[JUDGE_PROMPT.format(
            q=qa.get("question_as_asked") or rep.get("intent_question"),
            a=qa["answer"], g1=json.dumps(qa.get("gt_at_ask")),
            g2=json.dumps(qa.get("gt_at_answer")))],
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           temperature=0.0))
    try:
        return json.loads(r.text)
    except Exception:  # noqa: BLE001
        return {"verdict": "unparsed", "why": (r.text or "")[:200]}


# ------------------------------------------------------------------ scoring
def _dump_for(path):
    """The state dump belonging to a report, or None.

    `path.replace(".json", ".states.jsonl")` only works when the report sits beside its
    dump. In the v5 pipeline it never does — reports are written to `tmp/v5/cur/` and
    dumps to `tmp/run/` — so this recomputation silently did nothing for every episode of
    every sweep, while the docstring claimed the dump was authoritative. Look beside the
    report first (keeps the old behaviour and any test fixture working), then in tmp/run.
    The FULL dump is preferred over the `.clip` one, which is trimmed to the experiment
    window and need not contain the tick the question was asked at.
    """
    sib = path.replace(".json", ".states.jsonl")
    if os.path.exists(sib):
        return sib
    tag = os.path.basename(path)[:-5] if path.endswith(".json") else os.path.basename(path)
    for cand in (os.path.join(ROOT, "tmp/run", tag + ".states.jsonl"),
                 os.path.join(ROOT, "tmp/run", tag + ".clip.states.jsonl")):
        if os.path.exists(cand):
            return cand
    return None


def regt(rep, path):
    """Recompute qa ground truth from the episode's per-tick state dump.

    The dump is the authoritative record of what the kitchen actually looked like, and it
    is immune to whatever the harness computed at run time — an early ground-truth
    extractor filtered stations on a field they do not carry and reported an empty
    kitchen, which would have graded a correct answer as wrong. Grading now reads the
    dump whenever it exists and falls back to the stored value only if it does not."""
    qa = rep.get("qa") or {}
    dump = _dump_for(path)
    if not (qa and dump):
        return qa
    from cooksim_bench.suite.intent_lib import ground_truth_from_dump, is_history_kind
    kind = _kind_of(rep)
    if kind is None:
        return qa
    # A HISTORY probe ("did I already prep the meat", "have I been to the bin") is answered
    # from the cook's execution log. The dump records the SCENE and carries no log, so it
    # cannot re-derive one — recomputing here would not degrade the ground truth, it would
    # replace a correct one with a fabricated one. Leave the live value alone.
    if is_history_kind(kind):
        qa["gt_from_dump"] = "skipped — history kind, the state dump carries no execution log"
        return qa
    st = {}
    for ln in open(dump):
        try:
            d = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        st.setdefault(d.get("tick"), d)
    for when, key in (("asked_at", "gt_at_ask"), ("answered_at", "gt_at_answer")):
        t = qa.get(when)
        if t is not None and t in st:
            fresh = ground_truth_from_dump(kind, st[t])
            if fresh is not None and fresh != qa.get(key):
                qa.setdefault("gt_recomputed", {})[key] = {"stored": qa.get(key),
                                                           "from_state_dump": fresh}
                qa[key] = fresh
    return qa


def score(rep, judge=False, path=None):
    fam = rep.get("family")
    conv = rep.get("conversation") or []
    vlm = [c for c in conv if c.get("speaker") == "vlm" and (c.get("text") or "").strip()]
    words = sum(len((c.get("text") or "").split()) for c in vlm)
    turns = len(vlm)
    inj, rel = rep.get("injected_at"), rep.get("released_at")
    fa = rep.get("intent_false_alarms") or []
    other_fa = rep.get("other_false_alarms") or []

    buckets, detail = {}, {}
    # An episode that ran out of WALL-CLOCK budget never reached a verdict: the dish was
    # neither served nor lost. Scoring that as a failed delivery (and gating the composite
    # to 0) would report a harness timeout as an assistant failure, so it is marked
    # INCOMPLETE instead and the delivery bucket is left out of the mean.
    aborted = rep.get("aborted")
    if aborted and not rep.get("delivery_ok"):
        buckets["A_delivery"] = None
        detail["A_delivery"] = (f"INCOMPLETE — the episode was cut off ({aborted}) at tick "
                                f"{rep.get('ticks')} before any dish reached the pass; "
                                f"delivery is undetermined, not failed")
    else:
        buckets["A_delivery"] = 1.0 if rep.get("delivery_ok") else 0.0
        detail["A_delivery"] = ("the correct dish still reached the pass"
                                if rep.get("delivery_ok") else
                                f"order NOT delivered (outcome={rep.get('outcome')})")

    # RESTRAINT is graded on what the assistant did AFTER it was TOLD. The cook announces
    # its intention out loud before acting ("heads up, I'm going to use the far pan"), so
    # a flag raised after that moment is not a misread frame — it is the assistant
    # contradicting something it was explicitly told. Flags raised BEFORE the announcement
    # are reported separately and cost half as much: at that point the assistant genuinely
    # had only pixels to go on.
    ann = rep.get("intent_announced_at")
    after = rep.get("false_alarms_after_announcement")
    if after is None:                       # episode predates the announcement design
        after = fa if ann is None else [f for f in fa if f["t"] >= ann]
    before = [f for f in fa if f not in after]
    penalty = 0.5 * len(after) + 0.25 * len(before)
    buckets["B_restraint"] = max(0.0, 1.0 - penalty)
    if not fa:
        detail["B_restraint"] = "never called the cook's legitimate choice a mistake"
    else:
        parts = []
        if after and ann is None:
            # SILENT run: the cook never announced, so no flag can be "after" one. These
            # episodes are still charged the full 0.5 — the deviation was legitimate either
            # way — but the sentence must not claim an announcement that never happened.
            parts.append(f"{len(after)} false accusation(s); the cook never announced its "
                         f"intention (silent deviation — the assistant had only frames)")
        elif after:
            parts.append(f"{len(after)} false accusation(s) AFTER the cook announced what "
                         f"it was doing" + (f" (announced at t{ann})" if ann else ""))
        if before:
            parts.append(f"{len(before)} before the announcement (frames only)")
        detail["B_restraint"] = "; ".join(parts)

    if fam != "qa_probe":
        carried = rep.get("intent_carried_out")
        abandoned = bool(rep.get("intent_abandoned"))
        buckets["C_autonomy"] = 1.0 if (carried and not abandoned) else (
            0.0 if abandoned else 0.5)
        detail["C_autonomy"] = ("the cook carried out its own plan" if
                                (carried and not abandoned) else
                                "the assistant talked the cook out of a legitimate plan"
                                if abandoned else
                                "the cook never finished its own plan (inconclusive)")
    else:
        qa = regt(rep, path) if path else (rep.get("qa") or {})
        rep["qa"] = qa
        if qa.get("asked_at") is None and not (qa.get("answer") or "").strip():
            # The probe never fired, so the assistant was never asked anything. `grade_answer`
            # sees only the empty answer string and calls that `no_answer` -> 0.0, which
            # charges the model a wrong answer for a question that was never put to it: a
            # trigger that never matched, or a case whose checkpoint the live episode never
            # reached, silently becomes a model failure in the aggregate. The page already
            # says a trigger that never matched tests nothing; the eval must agree, so this
            # scores like `ungradable` (None = excluded) rather than like a miss.
            verdict, why = "not_asked", "the probe never fired — nothing was asked"
        else:
            verdict, why = grade_answer(_kind_of(rep), qa.get("gt_at_ask"),
                                        qa.get("gt_at_answer"), qa.get("answer"))
        jd = judge_answer(rep) if judge else None
        v = verdict
        if jd and jd.get("verdict") and verdict == "ungradable":
            v = jd["verdict"]
        buckets["D_answer"] = {"correct": 1.0, "partially_correct": 0.5,
                               "incorrect": 0.0, "no_answer": 0.0,
                               "ungradable": None, "unparsed": None,
                               "not_asked": None}.get(v)
        detail["D_answer"] = f"{v} — {why}" + (f" | judge: {jd.get('why')}" if jd else "")
        rep.setdefault("qa", {})["deterministic_verdict"] = verdict
        if jd:
            rep["qa"]["judge"] = jd

    econ = [x for x in (norm(TALK_REF, words), norm(TURN_REF, turns)) if x is not None]
    buckets["E_economy"] = sum(econ) / len(econ) if econ else None
    detail["E_economy"] = f"{words} words over {turns} assistant turns"

    vals = [v for v in buckets.values() if v is not None]
    composite = round(sum(vals) / len(vals), 4) if vals else None
    if not rep.get("delivery_ok") and not aborted:
        composite = 0.0            # delivery is a hard gate — but only for a real loss
    return {"schema": "intent-eval-v1", "family": fam, "buckets": buckets,
            "explain": detail, "composite": composite,
            "incomplete": bool(aborted and not rep.get("delivery_ok")),
            "aborted": aborted,
            "delivery_gate": bool(rep.get("delivery_ok")),
            "words": words, "turns": turns,
            "intent_false_alarms": len(fa), "other_false_alarms": len(other_fa),
            "injected_at": inj, "released_at": rel,
            "released_reason": rep.get("released_reason")}


_LEGACY_SCENARIO_KIND = {"qa_pan_contents": "pan_state", "qa_plate_count": "plate_count",
                         "qa_plate_contents": "plate_contents"}
# longest first: "plate_contents" must win over "plate_count" on a scenario naming both
_PLAIN_KINDS = ("plate_contents", "station_state", "board_state", "plate_count",
                "pan_state", "remaining", "holding")


def _kind_of(rep):
    """Which ground-truth kind this qa_probe is asking about.

    The report now carries `gt_kind` outright, which is the only reliable source: it is
    the case's own field, including the parametrised forms (`where:cheese`,
    `past_subgoal:prep(meat)`, `visited:trash`).

    This used to be a three-entry lookup on `scenario` that DEFAULTED to `plate_count`,
    and real scenarios are named `qa_probe__cheeseburger__station_state__t4` — so every
    probe outside the three original shapes was silently graded as a plate count. One
    measured consequence: a correct answer about a raw pan scored 0.0 `incorrect` because
    the word "one" in "pick one up" was read as a plate count. The scenario map is kept
    only to grade the archived reports that predate the `gt_kind` field, and an unknown
    scenario now returns None (-> `ungradable`) instead of guessing.
    """
    k = rep.get("gt_kind")
    if k:
        return k
    scen = rep.get("scenario") or ""
    if scen in _LEGACY_SCENARIO_KIND:
        return _LEGACY_SCENARIO_KIND[scen]
    for kind in _PLAIN_KINDS:
        if kind in scen:
            return kind
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    rows = []
    for f in a.reports:
        rep = json.load(open(f))
        if not rep.get("family"):
            print(f"skip {f}: not an intent episode"); continue
        ev = score(rep, judge=a.judge, path=f)
        rows.append((os.path.basename(f), ev))
        print(f"\n== {os.path.basename(f)}  [{ev['family']}]  composite="
              f"{ev['composite']}")
        for k, v in ev["buckets"].items():
            print(f"   {k:<12} {'—' if v is None else f'{v:.2f}'}  {ev['explain'][k]}")
        if a.write:
            if not os.path.exists(f + ".bak"):
                shutil.copy(f, f + ".bak")
            rep["eval_intent"] = ev
            json.dump(rep, open(f, "w"), indent=1, default=str)
    if rows:
        ok = sum(1 for _, e in rows if e["delivery_gate"])
        inc = sum(1 for _, e in rows if e.get("incomplete"))
        cl = sum(1 for _, e in rows if e["intent_false_alarms"] == 0)
        print(f"\n{len(rows)} episodes: {ok} delivered the order"
              + (f", {inc} cut off before a verdict" if inc else "")
              + f", {cl} never falsely accused the cook")
    return 0


if __name__ == "__main__":
    sys.exit(main())
