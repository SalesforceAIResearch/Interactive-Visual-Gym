#!/usr/bin/env python3
"""Offline re-grade of every question answer in the assistant benchmark with an LLM judge.

Why (2026-08-26 sweep audit): the live grader is a deterministic per-kind matcher written
for the earlier probe phrasing. On the sweep it marked plainly right answers wrong —
"yes, the chips are done" against a fryer reading cooked/all_done (it wanted the word
"potato"), "there's no tomato on a board yet" against a board holding cheese (it wanted
the board contents named) — and the `past_subgoal` truth it graded against came from the
plan log, which says "not finished" for any step the cook redid after a recovery. This
judge sees the question as asked, the engine truth at ask time AND at answer time, the
plate history (to settle assemble(X) questions), and a synonym list, and grades the
answer's FACTUAL claims only. Advice attached to the answer is ignored unless it
contradicts the truth.

    $PY python -m cooksim_bench.scoring.regrade_qa [--workers 8] [--limit N] [--model gemini-3.7-flash]
Writes tmp/assist_bench/qa_regrade.json  {"<mode>/<persona>/<model>/<case>#<idx>": {...}}
Resume-safe: existing entries are kept.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import sys
import threading

from cooksim_bench.paths import ROOT  # the CookSim release root
BENCH = os.environ.get("ASSIST_ROOT", "tmp/assist_bench")
OUT = os.path.join(ROOT, BENCH, "qa_regrade.json")
PROMPT_VERSION = "qa-v2-2026-08-26"

SYN = ("Synonyms that count as the same thing: potato = chips = fries; meat = patty = burger "
       "patty; steak = steak; serving = the pass = serving window = serving counter; cutting "
       "board = board = chopping board; ignite/turn on/switch on/start = the station was lit; "
       "fryer/pan/pot are stations; 'done'/'ready'/'cooked'/'finished' = status cooked; 'still "
       "cooking'/'not done'/'needs longer'/'almost' = status cooking; 'not yet'/'no' for a "
       "step = not finished; 'nothing'/'empty'/'clear' = no contents. Regions are approximate: "
       "'top', 'top-left', 'bottom-right', 'center' etc.; 'at the top' matches 'top-left' or "
       "'top-right' loosely (count as correct); a wrong side (left vs right, top vs bottom) is wrong.")

PROMPT = """You are grading one spoken answer from a kitchen assistant against the simulator's ground truth.

QUESTION the cook asked: "{question}"
Question kind: {kind}
Ground truth at the moment of asking (JSON from the engine): {gt_ask}
Ground truth at the moment of answering (JSON): {gt_ans}
{extra}
ASSISTANT'S ANSWER: "{answer}"

{syn}

Grade ONLY the factual claims the answer makes about the thing asked. An answer that is true at ask time OR at answer time is correct (the kitchen moves). Extra advice ("next, grab the cheese") does not count for or against unless it asserts a fact that contradicts the truth. If the answer asserts something the truth contradicts (wrong status, wrong place, wrong yes/no, wrong contents or count), it is incorrect. If it is right on the main point but wrong on a secondary factual detail, it is partially_correct. If the ground truth does not determine the answer (the truth is empty or unknown for what was asked), reply ungradable.

If the answer asserts NOTHING about the thing asked (it dodges, changes the subject, or answers a different question), reply evasive.

Reply with JSON only: {{"verdict": "correct" | "partially_correct" | "incorrect" | "evasive" | "ungradable", "reason": "<one sentence>"}}"""


def plate_fact(r, sid, t):
    """What the plate history proves about a past_subgoal question (audit 2026-08-26: generalised
    beyond assemble(X) — a prepped/cooked X that reached the plate before the ask proves prep/
    ignite/cook happened; an X first plated long after the ask, or never, makes 'not yet done' the
    likely truth for assemble/serve-type steps only, so that is stated as likelihood, not fact)."""
    m = re.match(r"(\w+)\((\w+)\)(?:#\d+)?$", sid or "")
    if not m or t is None:
        return None
    kind, comp = m.group(1), m.group(2)
    seq = [e for e in (r.get("plate_seq") or []) if e.get("item") == comp]
    before = [e for e in seq if (e.get("t") or 0) <= t]
    if kind == "assemble":
        if before:
            return f"Plate history says {comp} WAS put on the plate before the question was asked (t={t}); so assemble({comp}) was done."
        first = min((e.get("t") or 0) for e in seq) if seq else None
        return (f"Plate history says {comp} was NOT on the plate at the question (t={t}); it first reached the plate at t={first}." if first
                else f"Plate history says {comp} never reached the plate; so assemble({comp}) was NOT done at the question.") + " Treat that as the truth for 'finished'."
    if kind in ("prep", "ignite", "cook"):
        cooked = [e for e in before if str(e.get("state", "")) in ("chopped", "cooked", "burnt", "fried")]
        if cooked:
            return f"Plate history says {comp} ({cooked[0].get('state')}) reached the plate at t={cooked[0].get('t')}, before the question (t={t}); so {kind}({comp}) WAS done. Treat that as the truth."
        return None
    return None


def extra_for(r, e):
    kind = e.get("gt_kind") or ""
    qa = e.get("qa") or {}
    gt = qa.get("gt_at_ask") or {}
    lines = []
    if kind.startswith("past_subgoal:"):
        sid = kind.split(":", 1)[1]
        if gt.get("world_evidence") is True:
            lines.append(f"The world state proves the step {sid} WAS done (finished = true).")
        else:
            pf = plate_fact(r, sid, qa.get("asked_at"))
            if pf:
                lines.append(pf)
            elif gt.get("finished") is False:
                lines.append("CAUTION: 'finished: false' here comes from a plan log that misses steps the cook "
                             "redid on its own; it is NOT reliable evidence that the step was not done. If the "
                             "answer says the step was done, reply ungradable; if it says not done, reply ungradable too.")
    if kind == "board_state":
        lines.append("Board positions in the truth are grid coordinates only (no region names): do NOT grade the answer's "
                     "position words (left/right/top) for boards; grade only WHICH items are on boards and how many boards are in use.")
    if kind in ("holding", "plate_contents"):
        lines.append("Naming an item that is NOT on the plate (or a wrong count) is incorrect, not partially_correct — the cook would act on it.")
    try:                                                       # suite v3 families (docs/SUITE_V3_DESIGN.md §3)
        from cooksim_bench.scoring.qa_v3_lib import family_of, RUBRIC
        fam = family_of(kind)
        # a family may hold question kinds with different answer shapes (Q5: a SET for `missing`, an INTEGER for
        # `empty_boards`); the kind's own rubric wins when there is one
        rub = RUBRIC.get(f"{fam}/{kind.split(':', 1)[0]}") or RUBRIC.get(fam)
        if fam and rub:
            lines.append(f"[{fam}] " + rub + " The 'answer' field of the truth is the gold answer in words; 'answer_text' if present is its long form.")
    except Exception:
        pass
    if kind.startswith("visited:"):
        lines.append("'visited' means the cook physically walked to that thing at any point so far, even during "
                     "a detour that was later undone.")
    return "\n".join(lines)


# ANSWER ATTRIBUTION (audit 2026-09-15, local pass@3 eval). Two harness artefacts graded assistants on questions they
# could not have answered:
#  * the live harness files each assistant reply under the OLDEST unanswered question (late_qa FIFO); a question the
#    assistant never answered therefore captured the reply to the NEXT question (Qwen3.8-27B hard_hard_map_4: the
#    t91 question received the t257 reply to the t245 question, and the t245 question graded no_answer);
#  * 46 of 218 questions were asked within 10 ticks of the episode's end, inside the assistant's own charged latency.
# Rule: a captured reply that arrived after a LATER question was asked belongs to that later question; a question
# with no reply of its own that was asked <= LATE_TICKS before the end is `asked_too_late` (unscored, like a
# pre-empted proactive request). Re-attributed entries keep the harness's original answer time as `attr_from`.
ATTR_VERSION = "attr-v6-2026-09-19"
# v6: world-history truths (order / happened / count / next_step) are recomputed for EVERY record from its states dump
# or an action-log replay (fixed_truth); a question whose gold answer changed is re-judged (the note says old -> new).
# v5: every prep(X) question is worded "did I CHOP X ..." (question generator), so its event is the CHOP, not the dish's
# final state: comparing cooked ticks turned "steak chopped t20, potato chopped t64" into "after" (gpt-6-astra
# expert_medium_map_1 r2). An item whose recipe plan never chops it (meat, rice) makes a "chop" question about a
# non-event: it is `ungradable` (unscored), since "meat doesn't get chopped" is a true reply.
# v4: GROUND TRUTH of prep(X) in order/happened questions. qa_v3_lib finds a step's finish only from commands TAGGED with
# that step id; the v5 self-planning cook's commands carry tags like "vlm_fix:literal", so prep(X) NEVER finished and
# 23 of 64 order/happened questions in the local pass@3 root said "prep has not happened" although the processed item
# was already on the plate (molmo2 easy_medium_map_2: tomato chopped t17, plated t39, truth at t135 "not happened").
# The truth is recomputed from the per-tick state dump: prep(X) finishes the first tick a station shows X in the
# recipe's wanted state (chopped on a board, cooked in a heat station). assemble/visit truths were already right.
# v3: a chain -- the harness FIFO gave question A the t48 reply and question B the NEXT reply (t73, "you're all set"),
# although the t48 reply answers B (gemini-3.8-flash easy_medium_map_2 r3). When the judge says an earlier reply answers
# B, it replaces B's own captured reply if that one arrived later.
# v2: timing alone mis-assigned a late but on-topic reply (gemini-3.1-pro easy_medium_map_2: "There is one free cutting
# board right now" answers the t17 board question although it arrived at t50, after the t43 question). A reply that
# arrives after a later question is now attributed by a judge that reads BOTH questions and the reply.
ATTRIBUTE_PROMPT = """Two questions were asked by a cook, in this order, and the assistant then said ONE reply.
QUESTION A (earlier): "{qa}"
QUESTION B (later): "{qb}"
REPLY: "{reply}"
Which question does the reply answer? Reply with JSON only: {{"answers": "A" | "B" | "neither"}}"""
LATE_TICKS = int(os.environ.get("QA_LATE_TICKS", 10))


def chopped_in_plan(r):
    """Items whose recipe plan contains a chop step (from the plan IR, not from what one episode did)."""
    try:
        os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
        from cooksim_bench.exp_interact import recipe_of
        from cooksim.core.plan_ir import segment_plan
        rec = recipe_of((r.get("composite") or {}).get("cell"))
        return {n.component for n in segment_plan(rec) if n.type == "prep" and any(c == "chop" for c in n.commands)}
    except Exception:                                      # noqa: BLE001
        return None


def said_chopped(asked, item):
    """True when the question words THIS item as chopped ("chopped the steak", "chopping steak", "the chopped
    steak"), rather than merely mentioning chopping somewhere in a two-event question."""
    i = re.escape(item)
    return bool(re.search(rf"\bchop\w*\s+(?:the\s+|my\s+|that\s+)?{i}\b", asked, re.I)
                or re.search(rf"\b{i}\b[^.?!]{{0,12}}\bchop\w*", asked, re.I))


WORLD_KINDS = ("order", "happened", "count", "next_step")


def fixed_truth(e, r, record_path, at_tick):
    """The truth of a world-history question (order / happened / count / next_step) recomputed at `at_tick` with the
    live truth functions (qa_v3_lib) on the episode's own world: its states dump when present, else a replay of
    its action log (python -m cooksim_bench.scoring.qa_replay, exact on every verify_llm_formal episode). None when nothing to fix.

    v6 (2026-09-19 audit A1-A3): the old fix covered prep(X) only and needed a states dump; count:visit, the
    visit/ignite/assemble sides and next_step:plan kept the live truth, which matched command words and tags."""
    gk = e.get("gt_kind") or ""
    base = gk.split(":", 1)[0]
    if base not in WORLD_KINDS or at_tick is None:
        return None
    asked = str((e.get("qa") or {}).get("question_as_asked") or "")
    if base in ("order", "happened"):
        preps = re.findall(r"(?:prep|chop)\((\w+)\)", gk)
        if preps and re.search(r"\bchop", asked, re.I):
            choppable = chopped_in_plan(r)
            # v7 (2026-09-23): the test is PER ITEM, and only for the item the question itself calls chopped.
            # An order question names two preps and words each one the way its recipe does ("did I put the RICE
            # ON THE HEAT before or after I CHOPPED THE STEAK"): the old any() flagged rice -- never chopped, but
            # never called chopped either -- and made every Q1 instance in the formal run ungradable (124 answers).
            bad = [x for x in preps if choppable is not None and x not in choppable and said_chopped(asked, x)]
            if bad:
                return {"ungradable": True, "why": f"question says 'chop' but the recipe never chops {bad}",
                        "fixed_from_world": True, "answer": "ungradable"}
    try:
        from cooksim_bench.scoring.qa_replay import offline_truth
        gt = offline_truth(gk, r, record_path, at_tick)
    except Exception as ex:                                 # noqa: BLE001
        return {"error": f"offline truth failed: {type(ex).__name__}: {ex}"} if os.environ.get("QA_TRUTH_STRICT") else None
    if isinstance(gt, dict):
        gt["fixed_from_world"] = True
    return gt


def attributed_answers(r, which=None):
    """{idx: (answer_text, gt_at_answer, note)} for every asked qa_probe in one episode record.
    `which(question_a, question_b, reply) -> "A"|"B"|"neither"` settles a reply that arrived after the next question."""
    qs = sorted([e for e in (r.get("injections") or []) if e.get("family") == "qa_probe" and e.get("injected")
                 and (e.get("qa") or {}).get("asked_at") is not None], key=lambda e: e["qa"]["asked_at"])
    out = {e["idx"]: [((e["qa"].get("answer") or "").strip(), e["qa"].get("gt_at_answer"), None)] for e in qs}
    for i, e in enumerate(qs):
        qa = e["qa"]
        nxt = qs[i + 1] if i + 1 < len(qs) else None
        at = qa.get("answered_at")
        if nxt is not None and at is not None and (qa.get("answer") or "").strip() and at > nxt["qa"]["asked_at"]:
            reply = (qa.get("answer") or "").strip()
            pick = which(qa.get("question_as_asked") or "", nxt["qa"].get("question_as_asked") or "", reply) if which else "B"
            if pick == "A":
                out[e["idx"]][0] = (reply, qa.get("gt_at_answer"),
                                    f"reply at t{at} came after the next question (t{nxt['qa']['asked_at']}); judge: it answers THIS question")
                continue
            out[e["idx"]][0] = ("", None, f"reply at t{at} came after the next question (t{nxt['qa']['asked_at']}); judge: it answers {'the next question' if pick == 'B' else 'neither'}")
            nxt_at = nxt["qa"].get("answered_at")
            if pick == "B" and (not (nxt["qa"].get("answer") or "").strip() or (nxt_at is not None and nxt_at > at)):
                was = f"; replaces its own later capture at t{nxt_at}" if (nxt["qa"].get("answer") or "").strip() else ""
                out[nxt["idx"]][0] = (reply, None, f"reply at t{at} re-attributed from question idx {e['idx']} (judge: it answers this question){was}")
    end = r.get("ticks") or 0
    res = {}
    for e in qs:
        ans, gt_ans, note = out[e["idx"]][0]
        if not ans and end - e["qa"]["asked_at"] <= LATE_TICKS:
            note = (note + "; " if note else "") + f"asked at t{e['qa']['asked_at']}, episode ended t{end}"
            res[e["idx"]] = (None, None, note)
        else:
            res[e["idx"]] = (ans, gt_ans, note)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "gemini-3.7-flash"))
    ap.add_argument("--first-run-only", action="store_true",
                    help="skip the independent repeats (<tag>__r2, <tag>__r3). A run-1 episode carries a state "
                         "dump, so its truths are READ; without one the scan replays the episode, which is the "
                         "difference between minutes and hours over a sweep this size")
    ap.add_argument("--case", action="append", default=[],
                    help="only episodes whose case name contains this substring (repeatable). Used to re-grade "
                         "one question family after a truth/guard fix without rescanning the whole run")
    args = ap.parse_args()
    from cooksim_bench.backends.genai_client import make_client
    from google.genai import types
    client = make_client(120_000)
    def which_question(qa_text, qb_text, reply):
        for k in range(3):
            try:
                resp = client.models.generate_content(model=args.model, contents=ATTRIBUTE_PROMPT.format(qa=qa_text, qb=qb_text, reply=reply[:800]),
                                                      config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", max_output_tokens=200))
                a = json.loads((resp.text or "{}").strip()).get("answers")
                if a in ("A", "B", "neither"):
                    return a
            except Exception:                              # noqa: BLE001
                import time
                time.sleep(2 + 2 * k)
        return "neither"
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    done = {k: v for k, v in done.items() if v.get("verdict") != "judge_error"}   # retry transient failures
    lock = threading.Lock()
    jobs = []
    _files = sorted(glob.glob(os.path.join(ROOT, BENCH, "*/*/*/bm_*.json"))
                    + glob.glob(os.path.join(ROOT, BENCH, "*/*/*/b3_*.json")))
    if args.first_run_only:
        _files = [x for x in _files if "__r" not in x.split("/")[-2]]
    if args.case:
        _files = [x for x in _files if any(c in os.path.basename(x) for c in args.case)]
    print(f"scanning {len(_files)} episodes for questions to re-grade", flush=True)
    for f in _files:
        if "_stale" in f or "_smoke" in f or "_audit" in f:
            continue
        key0 = "/".join(f.split("/")[-4:])[:-5]
        try:
            r = json.load(open(f))
        except Exception:                                   # noqa: BLE001
            continue
        attr = attributed_answers(r, which=which_question)
        for e in r.get("injections") or []:
            if e.get("family") != "qa_probe" or not e.get("injected"):
                continue
            key = f"{key0}#{e['idx']}"
            qa = dict(e.get("qa") or {})
            ans, gt_ans, note = attr.get(e["idx"], ((qa.get("answer") or "").strip(), qa.get("gt_at_answer"), None))
            fx = fixed_truth(e, r, f, qa.get("asked_at"))
            if fx is not None:
                old_ans = (qa.get("gt_at_ask") or {}).get("answer")
                qa["gt_at_ask"] = fx
                fx2 = fixed_truth(e, r, f, qa.get("answered_at")) if gt_ans is not None else None
                if fx2 is not None:
                    gt_ans = fx2
                if fx["answer"] != old_ans:
                    note = (note + "; " if note else "") + f"truth recomputed from the world ({fx.get('world_source', 'n/a')}): {old_ans} -> {fx['answer']}"
            prev = done.get(key)
            if prev is not None and prev.get("attr") == ATTR_VERSION:
                continue
            if prev is not None and note is None and prev.get("verdict") not in ("no_answer",):
                prev["attr"] = ATTR_VERSION                 # attribution unchanged: the existing verdict stands
                continue
            if prev is not None:
                prev_verdict = prev.get("verdict")
            else:
                prev_verdict = None
            qa["answer"] = ans
            qa["gt_at_answer"] = gt_ans
            jobs.append((key, e, qa, extra_for(r, e), note, prev_verdict))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(done)} graded already; {len(jobs)} to grade with {args.model}", flush=True)

    def grade(job):
        key, e, qa, extra, note, prev_verdict = job
        res = _grade(key, e, qa, extra)
        res["attr"] = ATTR_VERSION
        if note:
            res["attr_note"] = note
        if prev_verdict is not None and prev_verdict != res.get("verdict"):
            res["verdict_before_attr"] = prev_verdict
        return key, res

    def _grade(key, e, qa, extra):
        if (qa.get("gt_at_ask") or {}).get("ungradable"):
            return {"verdict": "ungradable", "reason": (qa.get("gt_at_ask") or {}).get("why", "question about a non-event"),
                    "live": (e.get("qa_grade") or {}).get("verdict")}
        ans = qa.get("answer")
        if ans is None:
            return {"verdict": "asked_too_late", "reason": f"asked within {LATE_TICKS} ticks of the episode end with no reply (unscored)",
                    "live": (e.get("qa_grade") or {}).get("verdict")}
        ans = ans.strip()
        if not ans:
            return {"verdict": "no_answer", "reason": "the assistant never answered", "live": (e.get("qa_grade") or {}).get("verdict")}
        prompt = PROMPT.format(question=qa.get("question_as_asked") or "", kind=e.get("gt_kind"),
                               gt_ask=json.dumps(qa.get("gt_at_ask"), default=str)[:1500],
                               gt_ans=json.dumps(qa.get("gt_at_answer"), default=str)[:1500],
                               extra=extra, answer=ans[:1200], syn=SYN)
        err = "judge returned no valid verdict"
        for k in range(4):
            try:
                resp = client.models.generate_content(
                    model=args.model, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                                       max_output_tokens=1500 if k else 600))
                txt = (resp.text or "").strip()
                obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
                v = obj.get("verdict")
                if v in ("correct", "partially_correct", "incorrect", "ungradable", "evasive"):
                    return {"verdict": v, "reason": obj.get("reason", ""), "live": (e.get("qa_grade") or {}).get("verdict"), "prompt": PROMPT_VERSION}
            except Exception as ex:                          # noqa: BLE001
                err = repr(ex)
                import time
                time.sleep(3 + 3 * k)
        return {"verdict": "judge_error", "reason": err[:200], "live": (e.get("qa_grade") or {}).get("verdict")}

    n = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for key, res in ex.map(grade, jobs):
            with lock:
                done[key] = res
                n += 1
                if n % 50 == 0:
                    json.dump(done, open(OUT + ".tmp", "w"), indent=0)
                    os.replace(OUT + ".tmp", OUT)
                    print(f"  {n}/{len(jobs)}", flush=True)
    json.dump(done, open(OUT + ".tmp", "w"), indent=0)
    os.replace(OUT + ".tmp", OUT)
    import collections
    print("verdicts:", dict(collections.Counter(v["verdict"] for v in done.values())))
    agree = collections.Counter((v.get("live"), v["verdict"]) for v in done.values())
    print("live vs judge:", dict(agree.most_common(12)))


if __name__ == "__main__":
    main()
