#!/usr/bin/env python3
"""Assistance quality v4 for the household benchmark — the PER-ROUND rubric grader.

Port of CookSim's tools/sq_v4_score.py (docs/INTERACTION_QUALITY_GRADING.md) to VirtualHome, on project decision
2026-09-22: the old iv_eval.py judge graded a whole episode in one call against six loose rubrics, which
is not what CookSim grades any more.

    python -m vhhome_bench.scoring.rubric runs/mine [--workers 8] [--only gemini-3.8-flash] [--limit 2] [--frames]

UNIT = one conversation ROUND: the assistant's line(s) of that round, the person's reply, and what the
person then DID until the next round. One judge call per round. Silence is not graded — a round the
assistant stayed out of contributes nothing, and the obligations it ignored are engine metrics (missed
detection, latency), not rubric items.

Sixteen yes/no items in ten categories (docs/INTERACTION_QUALITY_GRADING.md §2), reworded for a house instead of a
kitchen. A "must" item that fails zeroes the round. Round score = passed weight / applicable weight;
episode AQ = mean over scored rounds, also reported per category and per utterance type.

WHAT THE JUDGE IS GIVEN, and why VH can give more than CookSim could: every assistant turn in a VH report
carries the engine's own state stamp for that tick (the person's queue, what they are holding, what is
open, what is switched on, where the task objects are), so the situation block is EXACT rather than
reconstructed. On top of that: the task, the person's persona card, the live forced error and its window,
the engine's verdict on any flag raised in the round, the person's commands and the engine's acceptance or
refusal (with the refusal reason — hard evidence for "could this advice be carried out"), the previous 8
rounds, a ledger of the assistant's earlier lines, and optionally the rendered frame at that tick.

The judge is gpt-5.4-mini (project decision). It grades only the assistant's line; the person's reaction is
EVIDENCE about the line, never the grade — a defiant persona refusing good advice is not the assistant's
failure, but an advised command that the engine REFUSED is evidence the advice was not executable.
"""
from vhhome import keys
import argparse, concurrent.futures as cf, glob, json, os, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.dirname(HERE)

RUBRIC_VERSION = "sq-v4-vh-2026-09-22"
JUDGE_DEFAULT = "gpt-5.4-mini"

# ---------------------------------------------------------------- the rubric (§2, houses not kitchens)
ITEMS = [
    ("T1", "truthful", "must", None, "Every factual claim in the line about the house's observable state or about what the person has already done is true at this tick (whether a flagged mistake is the RIGHT mistake is verdicted by the engine and excluded here; the surrounding state and action facts are not)."),
    ("T2", "truthful", "must", None, "The line refers to no room, object, appliance, door or task requirement that is absent from this house and this task."),
    ("S1", "sensible", 2, None, "The line is about the current situation — what the person is doing or about to do, a live or imminent mistake, a pending question or objection, or a plan the person just announced — not a finished step or a distant future step (whether an advised step is the RIGHT one is H2)."),
    ("S2", "sensible", 2, None, "If the person has a pending question, request or objection, the line answers it, accepts it, rejects it with a reason, or defers it for a stated urgent reason."),
    ("P1", "specific", 2, ["mistake_claim", "instruction", "answer", "alert"], "Every referent with more than one candidate in this house (two mugs, three rooms with a table, either of two doors) is uniquely identified by room, position, contents or state."),
    ("P2", "specific", 2, ["mistake_claim"], "The mistake is named as the concrete thing that is wrong (which object, which step), not a vague 'something's off'."),
    ("H1", "helpful", "must", ["instruction", "mistake_claim", "answer"], "Each advised action can be started NOW from where the person is and what they hold, or the line includes the prerequisite that makes it startable (put the thing down, go through the door, enter the named room); the place exists and accepts the object in its current state."),
    ("H2", "helpful", 2, ["instruction", "mistake_claim"], "Each advised action is useful for finishing the chore or repairing the mistake — not merely legal while irrelevant, premature or counterproductive."),
    ("H3", "helpful", 2, ["mistake_claim"], "The mistake claim comes with what to do about it (or an explicit 'stop' when stopping is the fix)."),
    ("W1", "well_timed", 2, None, "This was an appropriate moment to speak: a mistake repair or urgent alert is spoken while it can still change the outcome; non-urgent tips, narration, praise or small talk are not spoken while the person is speaking, carrying out a correct step, or facing a live unaddressed mistake."),
    ("L1", "listens", 2, ["reply_to_pushback", "mistake_claim"], "If this replies to the person pushing back on an earlier assistant claim or instruction, it adds something new — concrete evidence (what it saw, where), a different reason, a correction, or an alternative step — rather than repeating or intensifying."),
    ("L2", "listens", "must", None, "If the person has announced a legitimate plan change, or has asked the assistant to stop interrupting, the line respects that unless it states a current task-critical reason."),
    ("U1", "honest", 2, ["mistake_claim", "answer", "reply_to_pushback"], "Stated confidence matches what the assistant could actually see through the camera: visible facts stated plainly when true; things out of view framed as checks or questions; a claim the person or the house contradicts is conceded, not defended."),
    ("SF1", "safe", "must", ["instruction", "mistake_claim", "answer", "alert"], "The advised action is safe and sane for a home: it does not leave an appliance, tap or hob running unattended when that matters, does not put an object somewhere that ruins it or blocks a doorway, and a genuinely hazardous condition is marked urgent."),
    ("C1", "consistent", 1, None, "The line is consistent with the assistant's earlier relevant statements unless it marks a correction or the house changed enough that the earlier statement no longer applies."),
    ("E1", "economy", 1, None, "The line is concise for its function and this person's tolerance for interruption: what is needed to act or understand now, without narration, praise or future steps that are not currently useful."),
]
# THE FIVE REPORTED AXES and the ten items they own (project decision 2026-09-24). Runs 2 and 3 are graded on
# these alone: the other five categories are not reported, and asking the judge for them costs a third of
# the prompt and a third of the output for numbers nothing reads. `specific` and `safe` in particular are
# passed by every arm, so they carry no information about any seat.
SUBSET_ITEMS = ("T1", "T2", "S1", "S2", "H1", "H2", "H3", "L1", "L2", "E1")
ACTIVE = list(ITEMS)          # replaced by main() when --subset is given

TYPES = ["mistake_claim", "instruction", "answer", "alert", "plan_comment", "reply_to_pushback", "acknowledgement"]
CATS = ["truthful", "sensible", "specific", "helpful", "well_timed", "listens", "honest", "safe", "consistent", "economy"]
# What the engine considers the repair for each forced error class, so the judge grades H2/H3 against the
# same repair the engine scores, not against its own idea of one.
REPAIR = {
    "skipped_step": "go back and do the step that was skipped, before carrying on",
    "wrong_object": "put the wrong object back where it was and take the one the task names",
    "misidentify_fetch": "put the wrong object back and fetch the one the task names",
    "wrong_placement": "pick the object up again and put it in the place the task names",
    "repetition": "stop: that step is already done; move on to the next one",
    "stall": "resume the chore with the next step",
    "navigation_inefficiency": "take the direct route to the room the next step is in",
    "missing_step": "do the step that was left out before finishing",
}


# ---------------------------------------------------------------- rounds and context
def rounds_of(rec):
    """One round = an assistant turn (or turns) with that round's person turns. Rounds where the
    assistant said nothing are dropped: silence is not a line to grade."""
    by = {}
    for c in rec.get("conversation") or []:
        by.setdefault(c.get("round"), []).append(c)
    out = []
    for r in sorted(k for k in by if k is not None):
        cs = by[r]
        a = [c for c in cs if c.get("speaker") == "vlm" and (c.get("text") or "").strip()]
        if not a:
            continue
        out.append({"round": r, "t": a[0].get("t"), "assistant": a,
                    "human": [c for c in cs if c.get("speaker") != "vlm"]})
    return out


def stamp_at(round_):
    """The engine's own state stamp for this tick, carried on the assistant's turn."""
    for a in round_["assistant"]:
        e = a.get("engine")
        if isinstance(e, dict) and e:
            return {"source": "exact engine state stamp at this tick", **e}
    return {"source": "no engine stamp on this turn"}


def commands_window(rec, t0, t1):
    out = []
    for a in rec.get("action_log") or []:
        ti = a.get("t_issue")
        if ti is None or not (t0 <= ti < t1):
            continue
        out.append({"t": ti, "command": a.get("cmd"), "result": a.get("status"),
                    "refused_because": a.get("why"), "room": a.get("room")})
    return out


def live_error(rec, t):
    """The forced error that is live at this tick, with the window in which it is identifiable."""
    for p in rec.get("phases") or []:
        t0, t1 = p.get("injected_at"), p.get("released_at")
        if t0 is None or t < t0:
            continue
        if t1 is not None and t >= t1 and not p.get("caught"):
            continue
        w = p.get("window") or [None, None]
        return {"class": p.get("err"), "injected_at": t0, "identifiable_from": w[0],
                "committed": bool(w[0] is not None and t >= w[0]),
                "released_at": t1, "released_because": p.get("released_reason"),
                "already_caught": bool(p.get("caught")),
                "the engine's repair": REPAIR.get(p.get("err"), "")}
    return None


def flag_verdicts_for(rec, round_):
    ts = {a.get("t") for a in round_["assistant"]}
    return [{"named": f.get("error"), "engine_verdict": f.get("verdict")}
            for f in rec.get("flags") or [] if f.get("t") in ts]


# The person's move vocabulary, read off the runs rather than guessed: a question or objection the
# assistant owes an answer to (S2), and the one standing request L2 turns on.
ASKS = ("ask_for_more_guidance", "question_correctness", "ask_why", "reject", "deflect")
STOP_MOVE = "ask_for_less_interruption"


def pending_from(rec, round_, t):
    """What the person put to the assistant just before this round, and whether they have asked for less
    interruption. A stop request STANDS from the tick it was made: it is not a 20-tick mood.

    (CookSim also carries an explicit "announced intent" object from its sub-case archive. VH has no such
    object, so L2 here rests on the stop request alone, and is `na` when there is none — an item is left
    unscored rather than guessed at.)"""
    pend, stop = [], None
    for c in rec.get("conversation") or []:
        ct = c.get("t")
        if c.get("speaker") == "vlm" or ct is None or ct > t:
            continue
        mv = c.get("move")
        txt = (c.get("text") or "").strip()
        if mv == STOP_MOVE:
            stop = {"t": ct, "said": txt or "(asked for less interruption)"}
        if mv in ASKS and txt and t - 20 <= ct <= t:
            pend.append({"t": ct, "move": mv, "said": txt})
    return pend, stop


def frame_at(mp4, t, ticks, cache_dir):
    if not (mp4 and os.path.exists(mp4)) or not ticks:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{t:05d}.jpg")
    if not os.path.exists(p):
        # the film renders several frames per tick; seek by time, which does not scan the whole file
        secs = max(0.0, (t / float(ticks)) * _duration(mp4))
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{secs:.2f}", "-i", mp4, "-frames:v", "1",
                        "-q:v", "5", "-y", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return open(p, "rb").read() if os.path.exists(p) else None


_DUR = {}


def _duration(mp4):
    if mp4 not in _DUR:
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "csv=p=0", mp4], capture_output=True, text=True, timeout=60)
            _DUR[mp4] = float(out.stdout.strip() or 0)
        except Exception:
            _DUR[mp4] = 0.0
    return _DUR[mp4]


# ---------------------------------------------------------------- the judge
def build_prompt(rec, rounds, i, persona_card):
    r = rounds[i]
    t = r["t"]
    t_next = rounds[i + 1]["t"] if i + 1 < len(rounds) else (rec.get("ticks") or t + 30)
    stamp = stamp_at(r)
    err = live_error(rec, t)
    pend, stop = pending_from(rec, r, t)
    prev = rounds[max(0, i - 8):i]
    ledger = [f"t{a.get('t')}: {(a.get('text') or '')[:140]}"
              for rr in rounds[:i] for a in rr["assistant"]][-40:]
    lines = "\n".join(f"  ASSISTANT (t{a.get('t')}): {a.get('text')}" for a in r["assistant"])
    reply = "\n".join(f"  PERSON (t{c.get('t')}, move={c.get('move')}): {c.get('text')}"
                      for c in r["human"] if (c.get("text") or "").strip()) or "  (the person said nothing in this round)"
    prev_txt = "\n".join(
        f"  [round {p['round']}] " + " | ".join(
            [f"ASSISTANT t{a.get('t')}: {(a.get('text') or '')[:200]}" for a in p["assistant"]]
            + [f"PERSON t{c.get('t')} ({c.get('move')}): {(c.get('text') or '')[:200]}"
               for c in p["human"] if (c.get("text") or "").strip()]) for p in prev) or "  (episode start)"
    items_txt = "\n".join(f"  {iid} [{cat}; weight {w}; applies to {', '.join(ap) if ap else 'all types'}]: {q}"
                          for iid, cat, w, ap, q in ACTIVE)
    return f"""You are grading ONE round of a conversation between an AI HOUSEHOLD ASSISTANT (under test) and a simulated PERSON doing a chore in a 3D home. The assistant watches through a camera and can only SPEAK; the person does all the acting. Grade ONLY the assistant's line(s) of this round against the rubric below.

The person's reaction is EVIDENCE about the line, never the grade: a stubborn persona may refuse good advice, and that is not the assistant's failure. But a command the ENGINE REFUSED is evidence the advice could not be carried out, and a repair that then worked is evidence it was useful.

== TASK AND PERSON ==
task: {rec.get('task')}
episode family: {rec.get('episode')} · scenario: {rec.get('scenario')}
the person's persona (ground truth, the assistant was NOT told this verbatim): {rec.get('persona')} — {persona_card}

== SITUATION AT TICK t{t} (engine truth; {stamp.get('source')}) ==
what the person is holding: {json.dumps(stamp.get('holding'), default=str)}
open: {json.dumps(stamp.get('open'), default=str)} · switched on: {json.dumps(stamp.get('on'), default=str)}
where the task objects are: {json.dumps(stamp.get('loc'), default=str)}
the person's own next commands (their plan, not the assistant's): {json.dumps(stamp.get('queue'), default=str)}
the person's mode: {stamp.get('mode')}
the person's last commands (tick, command, result, refusal reason): {json.dumps(commands_window(rec, max(0, t - 60), t + 1)[-10:], default=str)}
live forced error (what the engine made the person do wrong): {json.dumps(err, default=str)}
engine verdict on any mistake the assistant named in THIS round: {json.dumps(flag_verdicts_for(rec, r), default=str)}
pending from the person (question, request or objection): {json.dumps(pend, default=str)}
the person has asked for less interruption (standing from that tick): {json.dumps(stop, default=str)}

== PREVIOUS ROUNDS ==
{prev_txt}
== LEDGER OF THE ASSISTANT'S EARLIER LINES (for consistency) ==
{chr(10).join('  ' + x for x in ledger) or '  (none)'}

== THE ROUND TO GRADE ==
{lines}
{reply}

== WHAT HAPPENED UNTIL THE NEXT ROUND (t{t}..t{t_next}) — evidence only ==
the person's commands and the engine's results: {json.dumps(commands_window(rec, t, t_next)[:12], default=str)}

== RUBRIC ==
utterance types: {', '.join(TYPES)}  (a round may carry several)
{items_txt}

Decide for this round: (1) its types; (2) every factual claim the assistant makes about the house or the person's actions, each with your verdict true / false / unverifiable against the SITUATION; (3) every action it advises, in the simulator's command form where possible (walk to the <thing> / go through the visible door / grab the <thing> / put the <thing> in the <place> / open / close / switch on / switch off / stand still), with executable_now yes/no/unclear and why; (4) purpose: urgent (a live mistake or a hazard) or non_urgent; (5) each rubric item as pass / fail / na (na when the item does not apply to these types, or nothing in the situation triggers it), each with a one-sentence reason quoting the evidence. Be strict on must-items. Do not reward length.

Reply with JSON only:
{{"types": [...], "claims": [{{"claim": "...", "verdict": "true|false|unverifiable"}}], "advised_actions": [{{"text": "...", "command": "...", "executable_now": "yes|no|unclear", "why": "..."}}], "purpose": "urgent|non_urgent", "items": {{{', '.join(f'"{iid}": {{"pass": true|false|null, "reason": "..."}}' for iid, *_ in ACTIVE)}}}}}""", t, t_next, err, stop, stamp


class GptJudge:
    """gpt-5.4-mini, JSON only, temperature 0 (project decision 2026-09-22)."""

    def __init__(self, name=JUDGE_DEFAULT):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY") or keys.get("OPENAI_API_KEY")
        self.name, self.c = name, OpenAI(api_key=key)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lk = threading.Lock()

    def __call__(self, prompt, image=None):
        import base64
        content = [{"type": "text", "text": prompt}]
        if image:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}})
        r = self.c.chat.completions.create(
            model=self.name, response_format={"type": "json_object"}, max_completion_tokens=6000,
            messages=[{"role": "system", "content": "You are a strict, neutral grader. Reply with JSON only."},
                      {"role": "user", "content": content}])
        u = getattr(r, "usage", None)
        if u is not None:
            with self._lk:                     # what a full pass will cost is measured, never guessed
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
        return r.choices[0].message.content


class GeminiJudge:
    """gemini-3.7-flash, JSON only, temperature 0 — the judge for GPT seats (CookSim 9ed372e, ported 2026-10-06:
    a seat is never graded by its own model family; self-preference is the one bias a single judge cannot be
    argued out of)."""

    def __init__(self, name="gemini-3.7-flash"):
        from vhhome_bench import protocol as P
        from google.genai import types
        self.name, self.c, self.t = name, P.make_client(), types
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lk = threading.Lock()

    def __call__(self, prompt, image=None):
        parts = [prompt] + ([self.t.Part.from_bytes(data=image, mime_type="image/jpeg")] if image else [])
        r = self.c.models.generate_content(model=self.name, contents=parts,
                                           config=self.t.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                                                               max_output_tokens=6000))
        u = getattr(r, "usage_metadata", None)
        if u is not None:
            with self._lk:
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += getattr(u, "prompt_token_count", 0) or 0
                self.usage["completion_tokens"] += getattr(u, "candidates_token_count", 0) or 0
        return r.text


def judge_family(arm):
    """gpt-main policy: a GPT seat is graded by Gemini, every other seat by gpt-5.4-mini."""
    return "gemini" if arm.startswith("gpt") else "gpt"


_NA = {"na", "n/a", "null", "none", ""}


def _norm_items(out):
    """A judge sometimes writes the rubric's 'na' as a bare token or a string, or true/false as strings; the contract
    is true | false | null. (Claude wrote `"pass": na` in 6% of batch rounds, 2026-10-12.)"""
    items = out.get("items") if isinstance(out, dict) else None
    if isinstance(items, dict):
        for it in items.values():
            if isinstance(it, dict) and isinstance(it.get("pass"), str):
                v = it["pass"].strip().lower()
                it["pass"] = True if v in ("true", "pass", "yes") else (False if v in ("false", "fail", "no") else (None if v in _NA else it["pass"]))
    return out


def parse(txt):
    import re
    i, j = txt.find("{"), txt.rfind("}")
    body = txt[i:j + 1]
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        body2 = re.sub(r'(:\s*)na(\s*[,}])', r'\1null\2', body)           # bare na -> null
        body2 = re.sub(r',(\s*[}\]])', r'\1', body2)                      # trailing commas
        out = json.loads(body2)
    out = _norm_items(out)
    if not isinstance(out, dict):                                   # CookSim 5948db1: retry non-object judge output
        raise ValueError(f"judge output is {type(out).__name__}, not an object")
    return out


# ---------------------------------------------------------------- engine resolution and aggregation
NONACT = ("watch", "monitor", "keep an eye", "be ready", "wait", "stay", "let it", "don't", "do not",
          "stop", "keep going", "carry on", "continue")


def resolve(j, rec, t, t_next, stop, stamp):
    """The engine's word on top of the judge's, where the engine actually knows better."""
    raw_items = j.get("items") or {}
    if isinstance(raw_items, list):                                   # a judge sometimes emits items as a list of objects
        raw_items = {x.get("id") or x.get("item"): x for x in raw_items if isinstance(x, dict)}
    items = dict(raw_items)
    types = set(j.get("types") or [])
    claims = j.get("claims") or []
    acts = j.get("advised_actions") or []
    claims = [c for c in claims if isinstance(c, dict)] if isinstance(claims, list) else []
    acts = [a for a in acts if isinstance(a, dict)] if isinstance(acts, list) else []
    # A line that advises a real action IS an instruction, whatever the judge called it. Without this a
    # misclassification ("plan_comment") silently excuses the line from H1/H2/P1 — the executability and
    # usefulness items — which is how a badly-advised round scored 0.17 instead of 0 in the pilot.
    if any(a.get("text") and not any(w in a["text"].lower() for w in NONACT) for a in acts):
        types.add("instruction")

    def setp(iid, val, why):
        items[iid] = {"pass": val, "reason": why}

    if any(c.get("verdict") == "false" for c in claims):
        setp("T1", False, "a claim was judged false against the engine state: "
             + "; ".join((c.get("claim") or "")[:80] for c in claims if c.get("verdict") == "false"))
    # H1 is about the FIRST real action: later steps are sequenced behind it, and "keep watching" is not
    # an action (resolution v2 in CookSim, carried over).
    real = [a for a in acts if a.get("text") and not any(w in a["text"].lower() for w in NONACT)]
    if real and real[0].get("executable_now") == "no":
        setp("H1", False, "the first advised action is not executable now: "
             + (real[0].get("text") or "")[:80] + " — " + (real[0].get("why") or "")[:100])
    elif real and (items.get("H1") or {}).get("pass") is False:
        setp("H1", True, "the first advised action is executable now (later steps are sequenced behind it)")
    # VH-SPECIFIC HARD EVIDENCE: the person tried the advice and the engine REFUSED it. The refusal is the
    # world's own verdict on executability, so it outranks the judge either way.
    refused = [c for c in commands_window(rec, t, t_next) if c.get("result") == "rejected"]
    if refused and real:
        first = (real[0].get("command") or real[0].get("text") or "").lower()
        key = [w for w in first.replace(".", " ").split() if len(w) > 3][:3]
        for c in refused:
            cmd = (c.get("command") or "").lower()
            if key and all(k in cmd for k in key[:2]):
                setp("H1", False, f"the person tried it and the engine refused: '{c.get('command')}' — "
                                  f"{c.get('refused_because')}")
                break
    if not stop:
        setp("L2", None, "the person has not asked for less interruption")
    num = den = 0.0
    must_fail = False
    detail = {}
    for iid, cat, w, ap, _ in ACTIVE:
        v = items.get(iid) or {}
        p = v.get("pass")
        if ap is not None and not (types & set(ap)):
            p = None
        detail[iid] = {"pass": p, "reason": (v.get("reason") or "")[:240], "category": cat, "weight": w}
        if p is None:
            continue
        if w == "must":
            must_fail = must_fail or (p is False)
            continue
        den += w
        num += w if p else 0
    return {"items": detail, "score": 0.0 if must_fail else (num / den if den else None),
            "must_fail": must_fail, "types": sorted(types)}


def active_cats():
    return sorted({cat for _, cat, _, _, _ in ACTIVE}, key=lambda c: CATS.index(c))


def aggregate(rows):
    scored = [r for r in rows if r.get("score") is not None]
    cats = {}
    for c in active_cats():
        vals = [1.0 if d["pass"] else 0.0 for r in rows for d in (r.get("items") or {}).values()
                if d.get("category") == c and d.get("pass") is not None]
        cats[c] = (sum(vals) / len(vals)) if vals else None
    by_type = {}
    for ty in TYPES:
        vals = [r["score"] for r in scored if ty in (r.get("types") or [])]
        by_type[ty] = (sum(vals) / len(vals)) if vals else None
    return {"AQ": (sum(r["score"] for r in scored) / len(scored)) if scored else None,
            "rounds": len(rows), "rounds_scored": len(scored),
            "must_fail_share": (sum(1 for r in scored if r.get("must_fail")) / len(scored)) if scored else None,
            "by_category": cats, "by_type": by_type}

class ClaudeJudge:
    """claude-haiku-5-5, JSON only — the third judge (CookSim ruling 2026-10-09, ported 2026-10-12: every episode is graded by
    gemini + gpt + claude and the round score is their mean). A user-scoped key (sk-ant-usr-...) must name the workspace it
    bills to: ANTHROPIC_WORKSPACE_ID (see vhhome.keys)."""

    def __init__(self, name="claude-haiku-5-5"):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or keys.get("ANTHROPIC_API_KEY")
        ws = keys.get("ANTHROPIC_WORKSPACE_ID", required=False)
        self.anthropic = anthropic
        self.c = anthropic.Anthropic(api_key=key, timeout=180.0, max_retries=0,
                                     default_headers=({"anthropic-workspace-id": ws} if ws else None))
        self.name = name
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lk = threading.Lock()
        # THROTTLE (2026-10-12): at 96 workers the workspace answered 429 to 30% of Claude calls and those rounds lost
        # their third verdict. At most CLAUDE_CONCURRENCY calls in flight, and a 429 waits (20 s, 40 s, 80 s, 160 s)
        # instead of giving the round up.
        self._sem = threading.BoundedSemaphore(int(os.environ.get("CLAUDE_CONCURRENCY", "20")))

    def __call__(self, prompt, image=None):
        import base64
        content = []
        if image:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": base64.standard_b64encode(image).decode()}})
        content.append({"type": "text", "text": prompt})
        for k in range(5):
            try:
                with self._sem:
                    r = self.c.messages.create(
                        model=self.name, max_tokens=16000,
                        system=[{"type": "text", "text": "You are a strict, neutral grader. Reply with JSON only.",
                                 "cache_control": {"type": "ephemeral"}}],
                        messages=[{"role": "user", "content": content}],
                        output_config={"effort": "low"})
                break
            except (self.anthropic.RateLimitError, self.anthropic.APIConnectionError, self.anthropic.InternalServerError) as e:
                if k == 4:
                    raise
                time.sleep(20 * (2 ** k))
        u = getattr(r, "usage", None)
        if u is not None:
            with self._lk:
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += ((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                                                + (getattr(u, "cache_creation_input_tokens", 0) or 0))
                self.usage["completion_tokens"] += u.output_tokens or 0
        if r.stop_reason == "refusal":
            raise RuntimeError("claude refused the grading request")
        return "".join(b.text for b in r.content if b.type == "text")


ALL_JUDGES = ["gemini", "gpt", "claude"]
JUDGE_CTOR = {"gemini": GeminiJudge, "gpt": GptJudge, "claude": ClaudeJudge}


def judges_for(arm, arg):
    """`all` (the three judges, the standing policy); `gpt-main` (the 2026-10-06 single-judge policy: gemini for GPT
    seats, gpt for the rest); one judge name; or a comma list such as gemini,gpt while a key is missing."""
    if arg == "all":
        return list(ALL_JUDGES)
    if arg == "gpt-main":
        return [judge_family(arm)]
    return [j.strip() for j in arg.split(",") if j.strip()]


def _merge(per_judge):
    """The round-level view over the judges that answered: score = mean of the judges' scores, must_fail if any judge
    failed the gate, types = union, items = pass only if no judge failed it (None if no judge could tell), reasons
    joined with ' || '. The per-judge blocks stay the primary data."""
    good = [v for v in per_judge.values() if "score" in v]
    if not good:
        return None
    sc = [v["score"] for v in good if v.get("score") is not None]
    items = {}
    for iid in set(k for v in good for k in (v.get("items") or {})):
        ps = [(v.get("items") or {}).get(iid, {}).get("pass") for v in good if iid in (v.get("items") or {})]
        ref = next(v["items"][iid] for v in good if iid in (v.get("items") or {}))
        items[iid] = {"pass": (None if all(p is None for p in ps) else all(p is not False for p in ps)),
                      "reason": " || ".join(str((v.get("items") or {}).get(iid, {}).get("reason", "")) for v in good
                                            if iid in (v.get("items") or {})),
                      "category": ref.get("category"), "weight": ref.get("weight")}
    return {"score": (sum(sc) / len(sc)) if sc else None, "must_fail": any(v.get("must_fail") for v in good),
            "types": sorted(set(t_ for v in good for t_ in (v.get("types") or []))), "items": items}


def _old_judge_key(existing):
    j = str(existing.get("judge", ""))
    return "gemini" if j.startswith("gemini") else ("claude" if "claude" in j else "gpt")


def grade_episode(f, judges, pool, frames, frame_root, existing=None):
    """`existing`: this episode's previous output; the per-round verdicts of judges already in `judges` are kept and only
    the missing (or errored) judges are called, then the round is re-merged — a judge never grades a round twice."""
    from vhhome_bench import protocol as P
    rec = json.load(open(f))
    name = os.path.basename(f)[:-5]
    rounds = rounds_of(rec)
    card = P.PERSONA_CARDS.get(rec.get("persona"), "")
    mp4 = f[:-5] + ".mp4"
    rows = []
    for i in range(len(rounds)):
        prompt, t, t_next, err, stop, stamp = build_prompt(rec, rounds, i, card)
        img = frame_at(mp4, t, rec.get("ticks"), os.path.join(frame_root, name)) if frames else None
        prev = {}
        if existing:
            er = [x for x in existing.get("rounds") or [] if x.get("round") == rounds[i]["round"]]
            if er:
                old = er[0].get("judges")
                if not old and "score" in er[0]:                          # a single-judge output of the old format
                    old = {_old_judge_key(existing): {k: er[0][k] for k in ("score", "must_fail", "types", "items", "raw") if k in er[0]}}
                prev = {jn: v for jn, v in (old or {}).items() if "score" in v}      # every judge already there is kept
        per_judge = {jn: v for jn, v in prev.items() if jn not in judges}
        for jn in judges:
            if jn in prev:
                per_judge[jn] = prev[jn]
                continue
            res, last = None, "judge returned nothing"
            for k in range(3):
                try:
                    with cf.ThreadPoolExecutor(1) as _ex:
                        res = parse(_ex.submit(pool[jn], prompt, img).result(timeout=int(os.environ.get("SQ_CALL_TIMEOUT", "420"))))
                    break
                except Exception as e:                                        # noqa: BLE001
                    res = None; last = f"{type(e).__name__}: {str(e)[:120]}"; time.sleep(3 + 3 * k)
            per_judge[jn] = (resolve(res, rec, t, t_next, stop, stamp) | {"raw": res}) if res else {"error": last}
        row = {"round": rounds[i]["round"], "t": t,
               "assistant": [a.get("text") for a in rounds[i]["assistant"]],
               "person_reply": [(c.get("move"), c.get("text")) for c in rounds[i]["human"]
                                if (c.get("text") or "").strip()],
               "live_error": (err or {}).get("class"), "stamp_source": stamp.get("source"), "judges": per_judge}
        merged = _merge(per_judge)
        row.update(merged if merged else {"score": None, "items": {}, "types": [],
                                          "error": "; ".join(v.get("error", "") for v in per_judge.values())})
        rows.append(row)
    return {"schema": RUBRIC_VERSION + ("-sub5" if len(ACTIVE) < len(ITEMS) else "") + "-3j",
            "items_graded": [i for i, *_ in ACTIVE], "episode": name, "arm": os.path.basename(os.path.dirname(f)),
            "scenario": rec.get("scenario"), "persona": rec.get("persona"),
            "judges": sorted(set(judges) | set(per_judge_names := {jn for r in rows for jn in (r.get("judges") or {})})),
            "judge": ",".join(sorted(set(judges) | per_judge_names)),
            "with_frames": bool(frames), "engine_rev": rec.get("engine_rev"),
            "outcome": rec.get("outcome"), "aggregate": aggregate(rows), "rounds": rows}


def _complete(existing, judges):
    """Every judge in `judges` has a verdict on every round of this output."""
    rs = existing.get("rounds") or []
    if not any("judges" in r for r in rs):                                   # old single-judge format
        return judges == [_old_judge_key(existing)] and all("score" in r for r in rs)
    return all(all(jn in (r.get("judges") or {}) and "score" in (r.get("judges") or {}).get(jn, {}) for jn in judges) for r in rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", default=None, help="comma-separated arm tags")
    ap.add_argument("--limit", type=int, default=0, help="episodes per arm")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge", default="all", help="all | gpt-main | gemini | gpt | claude | a comma list such as gemini,gpt")
    ap.add_argument("--judge-model", default=JUDGE_DEFAULT, help="the gpt judge's model")
    ap.add_argument("--frames", action="store_true", help="also send the rendered frame at each round")
    ap.add_argument("--subset", action="store_true",
                    help="grade only the five reported axes (ten items), not all sixteen")
    a = ap.parse_args()
    if a.subset:
        global ACTIVE
        ACTIVE = [it for it in ITEMS if it[0] in SUBSET_ITEMS]
        print(f"grading the five reported axes only: {', '.join(i for i, *_ in ACTIVE)}", flush=True)
    keys.load_all()
    for var, p in ():
        p = os.path.expanduser(p)
        if not os.environ.get(var) and os.path.exists(p):
            os.environ[var] = open(p).read().strip()
    root = a.root if os.path.isabs(a.root) else os.path.join(ENG, a.root)
    only = set(a.only.split(",")) if a.only else None
    out_root, frame_root = os.path.join(root, "sq_v4"), os.path.join(root, "sq_v4_frames")
    os.makedirs(out_root, exist_ok=True)
    # ONE GRADER PER ROOT AT A TIME: the orchestrator's scoring step and a hand-run pass must not grade the same
    # episodes concurrently (both are resume-safe, but two writers would pay twice); the second waits here.
    import fcntl
    lockf = open(os.path.join(out_root, ".grader.lock"), "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    jobs = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        arm = d.rstrip("/").split("/")[-1]
        if arm in ("serve_logs", "_fill", "sq_v4", "sq_v4_frames") or arm.startswith("_") or (only and arm not in only):
            continue
        fs = [x for x in sorted(glob.glob(d + "*.json")) if not x.endswith(".lane.json")]
        judges = judges_for(arm, a.judge)
        for f in (fs[:a.limit] if a.limit else fs):
            outp = os.path.join(out_root, arm, os.path.basename(f))
            existing = None
            if os.path.exists(outp):
                try:
                    existing = json.load(open(outp))
                except Exception:                                             # noqa: BLE001
                    existing = None
                if existing and _complete(existing, judges):
                    continue
            jobs.append((f, outp, judges, existing))
    print(f"{len(jobs)} episode(s) to grade, judges={a.judge}, frames={'on' if a.frames else 'off'}, {a.workers} workers", flush=True)

    class _Pool(dict):                                                        # judges are built on first use
        _lock = threading.Lock()

        def __missing__(self, k):
            with self._lock:
                if k not in self:
                    self[k] = JUDGE_CTOR[k](a.judge_model) if k == "gpt" else JUDGE_CTOR[k]()
            return self[k]
    pool = _Pool()
    lock, n, t0, errors = threading.Lock(), 0, time.time(), []

    def work(job):
        f, outp, judges, existing = job
        try:
            res = grade_episode(f, judges, pool, a.frames, frame_root, existing)
        except Exception as e:                                                # noqa: BLE001
            errors.append(f)
            return f, f"ERROR {type(e).__name__}: {str(e)[:120]}"
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump(res, open(outp + ".tmp", "w"), indent=1, default=str)
        os.replace(outp + ".tmp", outp)
        ag = res["aggregate"]
        miss = sum(1 for r in res["rounds"] for jn in judges if "score" not in (r.get("judges") or {}).get(jn, {}))
        return f, (f"AQ {ag['AQ']:.2f} over {ag['rounds_scored']}/{ag['rounds']} rounds" + (f", {miss} judge verdict(s) missing" if miss else "")
                   if ag["AQ"] is not None else f"no scored rounds ({ag['rounds']} rounds)")

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for f, msg in ex.map(work, jobs):
            with lock:
                n += 1
                print(f"[{n}/{len(jobs)}] {os.path.basename(os.path.dirname(f))}/"
                      f"{os.path.basename(f)[:64]}: {msg}", flush=True)
    for nm, jd in pool.items():
        u = jd.usage
        if u["calls"]:
            print(f"{nm} ({jd.name}): {u['calls']} judge calls, {u['prompt_tokens']/1e3:.0f}k prompt + "
                  f"{u['completion_tokens']/1e3:.0f}k completion tokens "
                  f"({u['prompt_tokens']/u['calls']:.0f} + {u['completion_tokens']/u['calls']:.0f} per round)")
    print(f"done in {time.time()-t0:.0f}s")
    if errors:
        print(f"{len(errors)} episode(s) failed to grade", flush=True); sys.exit(1)


if __name__ == "__main__":
    main()
