#!/usr/bin/env python3
"""Assistance quality v4 — the per-round rubric grader (docs/SQ_RUBRIC_V4.md).

Unit = one harness ROUND: the assistant's line(s) of that round plus the cook's reply and what the cook did
until the next round. Each round is one judge call; the judge is from a different model family than the
assistant under test (Gemini 3.7 Flash grades GPT and local assistants, GPT-5.4 mini grades Gemini and local; local
assistants get both, averaged; every 10th episode of an API assistant is double-graded for agreement).

What the judge receives per round: the order and plan, the cook's questionnaire answers, the kitchen
inventory; the previous 8 rounds verbatim; a ledger of the assistant's earlier claims/instructions; the
pending obligations; the cook's last 10 commands with results; the EXACT kitchen state at the round's tick
when the episode carries a per-tick state dump (episodes launched after 2026-08-27 21:00 UTC), otherwise
the recorded evidence (plate history, heat events, command log) marked as such; the rendered frame the
assistant saw at that tick; the engine's canonical next step and repair; the reaction (the cook's reply
and move label, its commands and results until the next round, the engine's aftermath facts).

    ASSIST_ROOT=tmp/assist_bench_v3 $PY python -m cooksim_bench.scoring.sq_v4_score [--workers 6] [--limit N] [--model tag] [--judge gemini|gpt|auto]
Writes <root>/sq_v4/<mode>/<persona>/<model>/<case>.json  (resume-safe per episode)
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

from cooksim_bench.paths import ROOT  # the CookSim release root
os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
BENCH = os.environ.get("ASSIST_ROOT", "tmp/assist_bench_v3")
RUBRIC_VERSION = "sq-v4-2026-08-27-flash"
RESOLUTION_VERSION = "res-v2-2026-08-28"

# ---------------------------------------------------------------- the rubric (docs/SQ_RUBRIC_V4.md §2)
ITEMS = [
    # id, category, weight ("must" or int), applies-to types (None = all), short question for the judge
    ("T1", "truthful", "must", None, "Every factual claim about the current kitchen state or completed cook actions is true at this tick (the identity of a flagged mistake is verdicted by the engine and excluded; surrounding facts are not)."),
    ("T2", "truthful", "must", None, "The line refers to no station, item, place, rule or recipe requirement that is absent from this kitchen and order."),
    ("S1", "sensible", 2, None, "The line is about the current situation — the cook's current action/subgoal, a live or imminent mistake or condition, a pending question/request/objection, or a live announced intent — not a completed step or a distant future step (whether an advised step is the right one is H2)."),
    ("S2", "sensible", 2, None, "If a cook question, request or objection is pending, the line directly answers, accepts, rejects with a reason, or defers it for a stated urgent reason."),
    ("P1", "specific", 2, ["mistake_claim", "instruction", "answer", "alert"], "Every referent with more than one candidate in this kitchen (two pans, three boards, two potatoes) is uniquely identified by location, contents or state."),
    ("P2", "specific", 2, ["mistake_claim"], "The mistake is named as the concrete thing that is wrong (which item, which step), not a vague 'something's off'."),
    ("H1", "helpful", "must", ["instruction", "mistake_claim", "answer"], "Each advised action can be started now in the current state, or the line includes the prerequisite that makes it startable (put the plate down, go to the named station); the station/place exists and accepts the item in its current state."),
    ("H2", "helpful", 2, ["instruction", "mistake_claim"], "Each advised action is useful for the current plan, the cook's pending request, or the needed repair — not merely legal while irrelevant, premature or counterproductive."),
    ("H3", "helpful", 2, ["mistake_claim"], "The mistake claim comes with what to do about it (or an explicit 'stop' when stopping is the fix)."),
    ("W1", "well_timed", 2, None, "This was an appropriate moment to speak: urgent alerts/mistake repairs are spoken while they can still change the outcome; non-urgent tips, comments, encouragement or small talk are not spoken while the cook is speaking, executing a correct step, or facing a live unaddressed urgent condition."),
    ("L1", "listens", 2, ["reply_to_pushback", "mistake_claim"], "If this replies to cook pushback against an earlier assistant claim or instruction, it adds something new — concrete evidence (what it saw, where), a different reason, a correction, or an alternative step — rather than repeating or intensifying."),
    ("L2", "listens", "must", None, "If the cook has a live valid announced intent or has asked the assistant to stop non-urgent interruptions, the line respects that unless it states a current safety- or task-critical reason."),
    ("U1", "honest", 2, ["mistake_claim", "answer", "reply_to_pushback"], "Stated confidence matches what the assistant could see: visible/verifiable facts stated plainly when true; things it could not see framed as checks or possibilities; a claim contradicted by the cook or the kitchen is conceded, not defended."),
    ("SF1", "safe", "must", ["instruction", "mistake_claim", "answer", "alert"], "The advised action is physically safe and raises no burn/contamination/fire risk (no hot item barehanded, no raw item onto ready food, no live heat left when it matters); a genuinely safety-critical condition is marked urgent."),
    ("C1", "consistent", 1, None, "The line is consistent with the assistant's earlier relevant statements unless it marks a correction or the kitchen changed enough that the earlier statement no longer applies."),
    ("E1", "economy", 1, None, "The line is concise for its function and the cook's interruption preference: what is needed to act or understand now, without extra explanation, praise or future steps that are not currently useful."),
]
TYPES = ["mistake_claim", "instruction", "answer", "alert", "plan_comment", "reply_to_pushback", "acknowledgement"]
CATS = ["truthful", "sensible", "specific", "helpful", "well_timed", "listens", "honest", "safe", "consistent", "economy"]
REPAIR = {"mis_prep": "trash the wrongly prepped item and prep the right one", "add_foreign": "trash the plate with the foreign item (or the item) and rebuild the plate",
          "misidentify_assemble": "trash the plate and rebuild it with the right item", "serve_raw": "take the raw item off the plate; cook it first",
          "wrong_order": "trash the plate and rebuild it in the recipe's order", "omit_subgoal": "add the missing item before serving",
          "repeat_step": "do not plate the duplicate; if plated, trash the plate and rebuild", "forget_next_step": "resume the plan: the next step is the one listed",
          "leave_unattended": "go back to the station and take the food off the heat",
          "premature_serve": "do not hand it in yet; the order is not finished"}


# ---------------------------------------------------------------- evidence
def load_dump(path):
    if not os.path.exists(path):
        return None
    states = {}
    try:
        for i, line in enumerate(open(path)):
            line = line.strip()
            if line:
                d = json.loads(line)
                states[int(d.get("tick", i))] = d       # one line per tick; the engine stamps the tick
    except Exception:
        return None
    return states


def stamp_from_dump(states, t):
    ks = [k for k in states if k <= t]
    if not ks:
        return None
    d = states[max(ks)]
    out = {"source": "exact per-tick state", "tick": max(ks), "stations": [], "plate": None, "hands": "empty"}
    for s in d.get("stations") or []:
        typ = s.get("type")
        if typ in ("pan", "pot", "fryer", "oven"):
            names = [f"{c.get('name')}({c.get('state')})" for c in (s.get("contents") or [])]
            out["stations"].append({"station": typ, "at": [s.get("x"), s.get("y")], "contents": names, "status": s.get("status") if names else "empty",
                                    "progress": s.get("progress"), "smoking": s.get("smoking"), "burn_progress": s.get("burn_progress")})
        elif typ in ("cutting_board", "board", "cutting board"):
            it = s.get("item")
            out["stations"].append({"station": "cutting board", "at": [s.get("x"), s.get("y")], "contents": [f"{it.get('name')}({it.get('state')})"] if it else [], "status": "occupied" if it else "empty"})
    for p in d.get("players") or []:
        h = p.get("holding")
        if h:
            if h.get("kind") == "plate":
                out["plate"] = [f"{x.get('name')}({x.get('state')})" for x in h.get("contents", [])]; out["plate_where"] = "in hand"; out["hands"] = "a plate"
            else:
                out["hands"] = f"{h.get('name')}({h.get('state')})"
        out["cook_at"] = [p.get("x"), p.get("y")]
    if out["plate"] is None:
        for o in d.get("objects") or []:
            it = o.get("item")
            if it and it.get("kind") == "plate":
                out["plate"] = [f"{x.get('name')}({x.get('state')})" for x in it.get("contents", [])]; out["plate_where"] = f"on the counter at {o.get('x')},{o.get('y')}"
    return out


def stamp_from_evidence(rec, t):
    out = {"source": "recorded evidence (no per-tick dump for this episode): plate history, heat events, command log — cutting boards and hands are not recorded", "tick": t, "stations": [], "hands": "not recorded"}
    for h in rec.get("heat_events") or []:
        c, b, r = h["cooked_at"], h.get("burnt_at"), h.get("removed_at")
        end = min(x for x in (b, r, 10 ** 9) if x is not None)
        if t < c - 20 or t >= end:
            continue
        status = "cooking" if t < c else ("burnt" if (b is not None and t >= b) else "done (still on the heat)")
        out["stations"].append({"station": "heat station", "at": h.get("pos"), "contents": h["items"], "status": status, "done_at": c, "removed_at": r, "burnt_at": b})
    plate = [e["item"] for e in (rec.get("plate_seq") or []) if e["t"] <= t]
    out["plate"] = plate or []
    out["plate_where"] = "(plate location not recorded; contents are the items plated so far, a trashed plate is not subtracted)"
    return out


def frame_at(mp4, t, cache_dir):
    if not mp4 or not os.path.exists(mp4):
        return None
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{t:05d}.jpg")
    if not os.path.exists(p):
        idx = max(0, t - 1)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", "-q:v", "4", p],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return open(p, "rb").read() if os.path.exists(p) else None


_INV = {}


def inventory(cell, inject):
    key = (cell, inject)
    if key not in _INV:
        try:
            from cooksim_bench.exp_interact import build_episode
            state, ro = build_episode(None, "capable_collaborator", cell=cell, inject=inject)
            rs = ro.game.render_state()
            import collections
            types = collections.Counter(s.get("type") for s in rs.get("stations") or [])
            grid = ro.game.layout.grid
            terr = collections.Counter(str(x).split(".")[-1].lower() for row in grid for x in row)
            sources = sorted(k.replace("_source", "") for k in terr if k.endswith("_source"))
            _INV[key] = {"stations": dict(types), "sources": sources, "other": [k for k in ("trash", "serving", "plate_source", "sink") if k in terr]}
        except Exception as e:                                      # noqa: BLE001
            _INV[key] = {"error": repr(e)}
    return _INV[key]


# ---------------------------------------------------------------- rounds and context
def rounds_of(rec):
    conv = [c for c in rec.get("conversation") or [] if (c.get("text") or "").strip()]
    by = {}
    for c in conv:
        by.setdefault(c.get("round"), []).append(c)
    out = []
    for r in sorted(k for k in by if k is not None):
        cs = by[r]
        a = [c for c in cs if c["speaker"] == "vlm"]
        h = [c for c in cs if c["speaker"] != "vlm"]
        if not a:
            continue
        out.append({"round": r, "t": a[0]["t"], "assistant": a, "cook": h})
    return out


def flag_verdicts_for(rec, lines):
    out = []
    for f in rec.get("flags") or []:
        for l in lines:
            if f.get("t") == l.get("t") or (f.get("utt") and f["utt"][:60] in (l.get("text") or "")):
                out.append({"error_named": f.get("error"), "verdict": f.get("verdict")}); break
    return out


def live_context(rec, t):
    """Live forced error, pending question/request, live intent, stop request — from the sub-case archive."""
    live_err, pending_q, live_intent, pending_alert = None, None, None, None
    for e in rec.get("injections") or []:
        rk = e.get("report_keys") or {}
        t0, t1 = rk.get("injected_at"), rk.get("released_at") or e.get("archived_at")
        if t0 is None or t < t0 or (t1 is not None and t >= t1 and e.get("family") == "error" and not (rk.get("released_reason") or "").startswith("max_hold")):
            if not (t0 is not None and t >= t0 and e.get("family") == "error" and t1 is not None and t < (e.get("archived_at") or t1)):
                continue
        if e.get("family") == "error":
            win = rk.get("window") or [None, None]
            live_err = {"type": e.get("err"), "fired_at": t0, "identifiable_from": win[0], "committed": bool(rk.get("dev") is not None and t >= (win[0] or 10 ** 9)),
                        "prevented": bool(e.get("prevented")), "canonical_repair": REPAIR.get(e.get("err"), "fix the mistake")}
        elif e.get("family") == "qa_probe":
            q = e.get("qa") or {}
            if q.get("asked_at") is not None and q["asked_at"] <= t and (q.get("answered_at") is None or q["answered_at"] > t):
                pending_q = {"asked_at": q["asked_at"], "question": q.get("question_as_asked") or e.get("gt_kind"), "age": t - q["asked_at"]}
        elif e.get("family") == "qa_proactive":
            q = e.get("qa") or {}
            if q.get("asked_at") is not None and q["asked_at"] <= t:
                pending_alert = {"asked_at": q["asked_at"], "item": e.get("item"), "request": q.get("question_as_asked")}
        elif e.get("family") == "plan_change":
            live_intent = {"family": e.get("family"), "label": e.get("intent_label") or (rec.get("intent_label") if rec.get("family") == e.get("family") else None)}
    stop = any(c.get("move") == "ask_for_less_interruption" and c.get("t", 10 ** 9) <= t for c in rec.get("conversation") or [])
    return live_err, pending_q, pending_alert, live_intent, stop


def next_step_at(rec, t):
    snaps = [c for c in rec.get("conversation") or [] if c["speaker"] != "vlm" and c.get("t", 10 ** 9) <= t and c.get("engine")]
    if not snaps:
        return None
    sp = snaps[-1]["engine"].get("system_private") or {}
    fut = sp.get("future_subgoals") or []
    nxt = [f.get("sid") for f in fut if f.get("status") in ("normal", None)]
    return nxt[0] if nxt else None


def commands_window(rec, t0, t1):
    return [{"t": l.get("t"), "cmd": l.get("cmd"), "status": l.get("status"), "why": (l.get("line") if l.get("status") != "done" else None)}
            for l in rec.get("kitchen_feedback") or [] if l.get("cmd") and t0 <= (l.get("t") or 0) < t1]


def aftermath(rec, t, t_next):
    out = []
    for e in rec.get("injections") or []:
        rk = e.get("report_keys") or {}
        if e.get("family") == "error":
            if e.get("prevented") and rk.get("released_at") is not None and t <= rk["released_at"] < t_next:
                out.append(f"the forced error ({e.get('err')}) was PREVENTED at t{rk['released_at']}")
            w = rk.get("window") or [None]
            if w[0] is not None and t <= w[0] < t_next:
                out.append(f"the forced error ({e.get('err')}) became identifiable at t{w[0]}")
        if e.get("family") == "qa_probe":
            q = e.get("qa") or {}
            if q.get("answered_at") is not None and t <= q["answered_at"] < t_next:
                out.append(f"the cook's question was answered at t{q['answered_at']} (judge verdict elsewhere)")
    for h in rec.get("heat_events") or []:
        for k, lab in (("cooked_at", "done"), ("burnt_at", "BURNT"), ("removed_at", "taken off the heat")):
            if h.get(k) is not None and t <= h[k] < t_next:
                out.append(f"{'/'.join(h['items'])} {lab} at t{h[k]}")
    for e in rec.get("plate_seq") or []:
        if t <= e["t"] < t_next:
            out.append(f"{e['item']} ({e.get('state')}) landed on the plate at t{e['t']}")
    return out


# ---------------------------------------------------------------- the judge
def build_prompt(rec, rounds, i, stamp, inv, q3, q2, q1):
    r = rounds[i]
    t = r["t"]
    t_next = rounds[i + 1]["t"] if i + 1 < len(rounds) else (rec.get("ticks") or t + 30)
    live_err, pending_q, pending_alert, live_intent, stop = live_context(rec, t)
    prev = rounds[max(0, i - 8):i]
    ledger = [f"t{a['t']}: {(a.get('text') or '')[:140]}" for rr in rounds[:i] for a in rr["assistant"]][-40:]
    lines = "\n".join(f"  ASSISTANT (t{a['t']}): {a.get('text')}" for a in r["assistant"])
    reply = "\n".join(f"  COOK (t{c['t']}, move={c.get('move')}): {c.get('text')}" for c in r["cook"]) or "  (no cook reply in this round)"
    prev_txt = "\n".join(f"  [round {p['round']}] " + " | ".join([f"ASSISTANT t{a['t']}: {(a.get('text') or '')[:200]}" for a in p["assistant"]] + [f"COOK t{c['t']} ({c.get('move')}): {(c.get('text') or '')[:200]}" for c in p["cook"]]) for p in prev) or "  (episode start)"
    verdicts = flag_verdicts_for(rec, r["assistant"])
    react_cmds = commands_window(rec, t, t_next)
    react_lines = [c for rr in rounds[i + 1:i + 3] for c in rr["cook"] if c.get("t", 10 ** 9) <= t + 10]
    items_txt = "\n".join(f"  {iid} [{cat}; weight {w}; applies to {', '.join(ap) if ap else 'all types'}]: {q}" for iid, cat, w, ap, q in ITEMS)
    rubric_items = ", ".join(iid for iid, *_ in ITEMS)
    p = f"""You are grading ONE round of a conversation between an AI kitchen ASSISTANT (under test) and a simulated human COOK, in a cooking simulator. Grade only the ASSISTANT's line(s) of this round against the rubric below. The cook's reaction is EVIDENCE about the line, never the grade: a stubborn persona may refuse good advice; a rejected command IS evidence against executability; a repair that then worked IS evidence for usefulness.

== ORDER AND PLAN ==
recipe: {rec.get('composite', {}).get('recipe') if isinstance(rec.get('composite'), dict) else ''}{json.dumps((rec.get('conversation') or [{}])[0].get('engine', {}).get('public', {}).get('goal'), default=str)}
public plan: {json.dumps((rec.get('conversation') or [{}])[0].get('engine', {}).get('public', {}).get('subgoals'), default=str)}
kitchen inventory: {json.dumps(inv, default=str)}
the cook's questionnaire answers (shown to the assistant): speed={q1} (1 safe first / 2 so-so / 3 as fast as possible), alerting={q2} (1 flag only when confident / 2 medium / 3 every suspicion), interruption={q3} (1 step by step / 2 medium / 3 don't interrupt unless necessary)

== SITUATION AT TICK t{t} (engine truth; {stamp.get('source')}) ==
stations: {json.dumps(stamp.get('stations'), default=str)}
plate: {json.dumps(stamp.get('plate'), default=str)} {stamp.get('plate_where') or ''}
cook's hands: {stamp.get('hands')}
cook's last commands (issue tick, command, result): {json.dumps(commands_window(rec, max(0, t - 60), t + 1)[-10:], default=str)}
live forced error: {json.dumps(live_err, default=str)}
engine verdict on any mistake flag in this round: {json.dumps(verdicts, default=str)}
pending cook question: {json.dumps(pending_q, default=str)}
standing request ("tell me when ... is done"): {json.dumps(pending_alert, default=str)}
live announced intent (a legitimate plan change / side quest — NOT a mistake): {json.dumps(live_intent, default=str)}
cook asked for less interruption earlier: {stop}
engine's canonical next step: {next_step_at(rec, t)}
The attached image is the frame the assistant saw at this tick (left: top-down kitchen; right: the cook's first-person view).

== PREVIOUS ROUNDS ==
{prev_txt}
== LEDGER OF THE ASSISTANT'S EARLIER LINES (for consistency) ==
{chr(10).join('  ' + x for x in ledger) or '  (none)'}

== THE ROUND TO GRADE ==
{lines}
{reply}

== REACTION UNTIL THE NEXT ROUND (t{t}..t{t_next}) — evidence only ==
cook's next lines: {json.dumps([{'t': c['t'], 'move': c.get('move'), 'text': (c.get('text') or '')[:200]} for c in react_lines], default=str)}
cook's commands and results: {json.dumps(react_cmds[:12], default=str)}
engine aftermath: {json.dumps(aftermath(rec, t, t_next), default=str)}

== RUBRIC ==
utterance types: {', '.join(TYPES)}  (a round may carry several)
{items_txt}

Decide for this round: (1) its types; (2) every factual claim the assistant makes about the kitchen or the cook's actions, each with your verdict true / false / unverifiable against the SITUATION and the image; (3) every action it advises, in the simulator's command form if possible (go to the <place> / pick up / put down / scoop / chop / turn on / trash / serve), with executable_now yes/no/unclear and why; (4) purpose: urgent (a live mistake, a hazard, something done on the heat) or non_urgent; (5) each rubric item as pass / fail / na (na when the item does not apply to these types or nothing in the situation triggers it), with a one-sentence reason quoting the evidence. Be strict on must-items; do not reward length.

Reply with JSON only:
{{"types": [...], "claims": [{{"claim": "...", "verdict": "true|false|unverifiable"}}], "advised_actions": [{{"text": "...", "command": "...", "executable_now": "yes|no|unclear", "why": "..."}}], "purpose": "urgent|non_urgent", "items": {{{', '.join(f'"{iid}": {{"pass": true|false|null, "reason": "..."}}' for iid, *_ in ITEMS)}}}}}"""
    return p, t, t_next, live_err


class GeminiJudge:
    name = "gemini-3.7-flash"            # ruling 2026-08-27: flash-tier judges only (pro is unaffordable at ~27k calls)

    def __init__(self):
        from cooksim_bench.backends.genai_client import make_client
        from google.genai import types
        self.c, self.t = make_client(240_000), types

    def __call__(self, prompt, image):
        parts = [prompt] + ([self.t.Part.from_bytes(data=image, mime_type="image/jpeg")] if image else [])
        r = self.c.models.generate_content(model=self.name, contents=parts,
                                           config=self.t.GenerateContentConfig(temperature=0.0, response_mime_type="application/json", max_output_tokens=6000))
        return r.text


class GptJudge:
    name = "gpt-5.4-mini"                # ruling 2026-08-27: flash-tier judges only

    def __init__(self):
        from cooksim_bench.backends.controller_eval import OpenAIBackend
        key = open(os.path.expanduser("~/.openai_key")).read().strip()
        self.b = OpenAIBackend(self.name, "https://api.openai.com/v1", api_key=key, json_mode=True, max_tokens=6000, profile="gpt5")

    def __call__(self, prompt, image):
        out = self.b.call("You are a strict, neutral grader. Reply with JSON only.", prompt, [image] if image else [])
        if isinstance(out, tuple):
            out = out[0]
        return json.dumps(out) if isinstance(out, dict) else str(out)


class ClaudeJudge:
    name = "claude-haiku-5-5"             # ruling 2026-10-09: third flash-tier judge; every episode gets gemini + gpt + claude, averaged
    # the Anthropic rate limit is far below 192 parallel graders (2026-10-12: HTTP 429 on ~1/3 of rounds): cap the
    # in-flight Claude calls process-wide and back off on 429 instead of recording an error for the round
    _gate = threading.Semaphore(int(os.environ.get("SQ_CLAUDE_CONCURRENCY", "32")))

    def __init__(self):
        from cooksim_bench.backends.anthropic_backend import AnthropicBackend
        self.b = AnthropicBackend(self.name, max_tokens=6000, effort="low", image_format="jpeg")

    def __call__(self, prompt, image):
        delay = 10
        for attempt in range(6):
            with self._gate:
                try:
                    out = self.b.call("You are a strict, neutral grader. Reply with JSON only.", prompt, [image] if image else [])
                    break
                except RuntimeError as e:
                    if "429" not in str(e) or attempt == 5:
                        raise
            time.sleep(delay + random.random() * delay); delay = min(delay * 2, 120)
        if isinstance(out, tuple):
            out = out[0]
        return json.dumps(out) if isinstance(out, dict) else str(out)


def parse(txt):
    i, j = txt.find("{"), txt.rfind("}")
    return json.loads(txt[i:j + 1])


# ---------------------------------------------------------------- resolution and aggregation
def resolve(judgement, live_err, live_intent, stop, stamp):
    """Engine-side overrides on top of the judge's item verdicts, then the weighted round score."""
    raw_items = judgement.get("items") or {}
    if isinstance(raw_items, list):                                   # a judge sometimes emits items as a list of objects
        raw_items = {str(it.get("id") or it.get("item") or k): it for k, it in enumerate(raw_items) if isinstance(it, dict)}
    items = dict(raw_items) if isinstance(raw_items, dict) else {}
    types = set(t for t in (judgement.get("types") or []) if isinstance(t, str))
    # the same judge sometimes emits claims / advised_actions as arrays of arrays; only objects carry a verdict
    claims = [c for c in (judgement.get("claims") or []) if isinstance(c, dict)]
    acts = [a for a in (judgement.get("advised_actions") or []) if isinstance(a, dict)]
    def setp(iid, val, why):
        items[iid] = {"pass": val, "reason": why}
    if any(c.get("verdict") == "false" for c in claims):
        setp("T1", False, "a claim was judged false against the engine state: " + "; ".join(c.get("claim", "")[:80] for c in claims if c.get("verdict") == "false"))
    # H1 (resolution v2, 2026-08-28 audit): a multi-step instruction is judged by its FIRST real action — later steps
    # are sequenced behind the earlier ones and are not "not executable now"; monitoring advice ("keep watching",
    # "be ready to", "wait until") is not an action. The judge's per-step verdicts are kept for the record.
    NONACT = ("watch", "monitor", "keep an eye", "be ready", "wait", "stay", "let it", "don't", "do not", "stop", "keep going", "carry on", "continue")
    real = [a for a in acts if a.get("text") and not any(w in a["text"].lower() for w in NONACT)]
    if real and real[0].get("executable_now") == "no":
        setp("H1", False, "the first advised action is not executable now: " + (real[0].get("text") or "")[:80] + " — " + (real[0].get("why") or "")[:100])
    elif real and (items.get("H1") or {}).get("pass") is False:
        setp("H1", True, "the first advised action is executable now (later steps are sequenced behind it)")
    if not (live_intent or stop):
        setp("L2", None, "no live intent or stop request")
    if judgement.get("purpose") == "urgent" and live_err and live_err.get("committed") and not live_err.get("prevented") and "mistake_claim" in types and live_err.get("type") in ("omit_subgoal", "premature_serve", "serve_raw", "wrong_order", "add_foreign", "misidentify_assemble"):
        # a plate-ruining error already committed: the claim can no longer change the outcome by itself (repair still possible; keep the judge's W1 unless it passed on 'still changeable')
        pass
    score_num = score_den = 0.0
    must_fail = False
    detail = {}
    for iid, cat, w, ap, _ in ITEMS:
        v = items.get(iid) or {}
        p = v.get("pass")
        if ap is not None and not (types & set(ap)):
            p = None
        detail[iid] = {"pass": p, "reason": (v.get("reason") or "")[:240], "category": cat, "weight": w}
        if p is None:
            continue
        if w == "must":
            if p is False:
                must_fail = True
            continue
        score_den += w
        score_num += w if p else 0
    score = 0.0 if must_fail else (score_num / score_den if score_den else None)
    return {"items": detail, "score": score, "must_fail": must_fail, "types": sorted(types)}


def aggregate(rounds_out):
    scored = [r for r in rounds_out if r.get("score") is not None]
    cats = {}
    for c in CATS:
        vals = []
        for r in rounds_out:
            for iid, d in (r.get("items") or {}).items():
                if d.get("category") == c and d.get("pass") is not None:
                    vals.append(1.0 if d["pass"] else 0.0)
        cats[c] = (sum(vals) / len(vals)) if vals else None
    by_type = {}
    for ty in TYPES:
        vals = [r["score"] for r in scored if ty in (r.get("types") or [])]
        by_type[ty] = (sum(vals) / len(vals)) if vals else None
    return {"AQ": (sum(r["score"] for r in scored) / len(scored)) if scored else None, "rounds": len(rounds_out), "rounds_scored": len(scored),
            "must_fail_share": (sum(1 for r in scored if r.get("must_fail")) / len(scored)) if scored else None, "by_category": cats, "by_type": by_type}


ALL_JUDGES = ["gemini", "gpt", "claude"]


def judge_for(model_tag, args_judge):
    if "," in args_judge:                     # explicit judge list, e.g. "gemini,gpt" (the Claude key arrives later; per-judge resume adds it)
        return [j.strip() for j in args_judge.split(",") if j.strip()]
    if args_judge == "all":
        # ruling 2026-10-09: three judges on every episode and the per-round mean; the same-family exclusion is no
        # longer needed because every seat is graded by the same three judges
        return list(ALL_JUDGES)
    if args_judge == "gemini":
        return ["gemini"]
    if args_judge == "gpt":
        return ["gpt"]
    if args_judge == "gpt-main":
        # one main judge to keep the bill down (2026-09-22), EXCEPT where that would make a seat grade
        # itself: a GPT seat is still graded by Gemini, because self-preference is the one bias a single
        # judge cannot be argued out of.
        return ["gemini"] if model_tag.startswith("gpt") else ["gpt"]
    if model_tag.startswith("gpt"):
        return ["gemini"]
    if model_tag.startswith("gemini"):
        return ["gpt"]
    return ["gemini", "gpt"]


def grade_episode(f, judges, pool, frame_root, existing=None):
    """existing: a previous output for this episode; its per-round verdicts of judges already in `judges` are kept and
    only the missing (or errored) judges are called, then the round is re-merged (ruling 2026-10-09: a judge that
    already graded an episode is not called again)."""
    rec = json.load(open(f))
    name = os.path.basename(f)[:-5]
    parts = f.split("/")
    mode, persona, model = parts[-4], parts[-3], parts[-2]
    cases = {}
    for pf in ("tmp/interact_eval/cases_composite_v3.json", os.path.join(ROOT, "benchmark", "cases_composite_v5.json")):
        try:
            for c in json.load(open(os.path.join(ROOT, pf)))["cases"]:
                cases[c["name"]] = c
        except Exception:
            pass
    cell = cases.get(name, {}).get("cell"); inject = cases.get(name, {}).get("inject", True)
    inv = inventory(cell, inject) if cell else {}
    dump = load_dump(f[:-5] + ".states.jsonl")
    q = {it["key"]: it.get("choice") for it in ((rec.get("questionnaire_v2") or {}).get("items") or [])}
    rounds = rounds_of(rec)
    mp4 = f[:-5] + ".mp4"
    fdir = os.path.join(frame_root, mode, persona, model, name)
    out_rounds = []
    for i, r in enumerate(rounds):
        stamp = stamp_from_dump(dump, r["t"]) if dump else stamp_from_evidence(rec, r["t"])
        prompt, t, t_next, live_err = build_prompt(rec, rounds, i, stamp, inv, q.get("q3"), q.get("q2"), q.get("q1"))
        _, _, _, live_intent, stop = live_context(rec, t)
        img = frame_at(mp4, t, fdir)
        per_judge = {}
        prev = {}
        if existing:
            er = [x for x in existing.get("rounds") or [] if x.get("round") == r["round"]]
            prev = {jn: v for jn, v in ((er[0].get("judges") or {}).items() if er else []) if jn in judges and "score" in v}
        for jn in judges:
            if jn in prev:
                per_judge[jn] = prev[jn]
                continue
            jd = pool[jn]
            res = None; err = "judge returned nothing"            # (err was unbound when parse() returned an empty result)
            for k in range(3):
                try:
                    # hard wall-clock cap per judge call: on 2026-10-04 one call hung for >1 h inside the HTTP client and
                    # idled the other 191 workers (the executor yields results in order)
                    with cf.ThreadPoolExecutor(1) as _ex:
                        res = parse(_ex.submit(jd, prompt, img).result(timeout=int(os.environ.get("SQ_CALL_TIMEOUT", "420"))))
                    if not isinstance(res, dict):                     # the judge sometimes returns a JSON array: retry
                        raise ValueError(f"judge returned {type(res).__name__}, not an object")
                    break
                except Exception as e:                                # noqa: BLE001
                    res = None; err = repr(e)[:160]; time.sleep(4 + 4 * k)
            per_judge[jn] = resolve(res, live_err, live_intent, stop, stamp) | {"raw": res} if res else {"error": err}
        good = [v for v in per_judge.values() if "score" in v]
        merged = None
        if good:
            merged = {"score": (sum(v["score"] for v in good if v["score"] is not None) / max(1, sum(1 for v in good if v["score"] is not None))) if any(v["score"] is not None for v in good) else None,
                      "must_fail": any(v["must_fail"] for v in good), "types": sorted(set(t_ for v in good for t_ in v["types"])),
                      "items": good[0]["items"] if len(good) == 1 else {iid: {"pass": (None if all(v["items"][iid]["pass"] is None for v in good) else all(v["items"][iid]["pass"] is not False for v in good if v["items"][iid]["pass"] is not None)),
                                                                                "reason": " || ".join(v["items"][iid]["reason"] for v in good), "category": good[0]["items"][iid]["category"], "weight": good[0]["items"][iid]["weight"]} for iid in good[0]["items"]}}
        out_rounds.append({"round": r["round"], "t": t, "assistant": [a.get("text") for a in r["assistant"]], "cook_reply": [(c.get("move"), c.get("text")) for c in r["cook"]],
                           "stamp_source": stamp.get("source"), "judges": per_judge, **(merged or {"score": None, "items": {}, "types": []})})
    agg = aggregate(out_rounds)
    return {"schema": RUBRIC_VERSION, "episode": name, "mode": mode, "persona": persona, "model": model, "judges": judges,
            "stamp_source": "exact per-tick state" if dump else "recorded evidence", "aggregate": agg, "rounds": out_rounds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge", default="auto", help="auto | gemini | gpt | gpt-main | all | a comma list such as gemini,gpt")
    ap.add_argument("--double-every", type=int, default=10, help="every Nth episode of an API assistant is graded by both judges")
    ap.add_argument("--first-run-only", action="store_true",
                    help="skip the independent repeats (<tag>__r2, <tag>__r3): grade one run per seat")
    ap.add_argument("--repeats-only", action="store_true",
                    help="grade ONLY the independent repeats (<tag>__r2, <tag>__r3); --model still names the base tag")
    ap.add_argument("--categories", default="all", choices=["all", "five"],
                    help="five = only the reported categories (truthful, sensible, helpful, listens, economy; ruling 2026-09-24 "
                         "for the run-2/3 grading): the judge sees and scores those 10 items only, the schema is stamped '+five'")
    args = ap.parse_args()
    if args.categories == "five":
        FIVE = {"truthful", "sensible", "helpful", "listens", "economy"}
        ITEMS[:] = [it for it in ITEMS if it[1] in FIVE]
        CATS[:] = [c for c in CATS if c in FIVE]
        globals()["RUBRIC_VERSION"] = RUBRIC_VERSION + "+five"
    files = sorted(glob.glob(os.path.join(ROOT, BENCH, "frames/*/*/b3_*.json")))
    if args.model:
        files = [f for f in files if f.split("/")[-2].split("__r")[0] == args.model]
    if args.repeats_only:
        files = [f for f in files if "__r" in f.split("/")[-2]]
    if args.first_run_only:
        # the independent repeats live in <tag>__r2 / <tag>__r3; the rubric is graded on ONE run only
        # (2026-09-22) -- the repeats are there for the variance of the engine-scored metrics
        files = [f for f in files if "__r" not in f.split("/")[-2]]
    out_root = os.path.join(ROOT, BENCH, "sq_v4")
    frame_root = os.path.join(ROOT, BENCH, "sq_v4_frames")
    jobs = []
    for k, f in enumerate(files):
        parts = f.split("/")
        outp = os.path.join(out_root, parts[-4], parts[-3], parts[-2], os.path.basename(f))
        judges = judge_for(parts[-2], args.judge)
        existing = None
        if os.path.exists(outp):
            if args.judge != "all" and "," not in args.judge:
                continue
            try:
                existing = json.load(open(outp))
            except Exception:                                     # noqa: BLE001
                existing = None
            if existing and all(all(jn in (x.get("judges") or {}) and "score" in (x.get("judges") or {}).get(jn, {})
                                    for jn in judges) for x in existing.get("rounds") or []):
                continue                                          # every judge has graded every round: nothing to do
        if args.judge in ("auto", "gpt-main") and len(judges) == 1 and k % args.double_every == 0:
            judges = ["gemini", "gpt"]
        jobs.append((f, outp, judges, existing))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} episodes to grade ({len(files)} on disk)", flush=True)
    class _Pool(dict):                                                # judges are built on first use (no Anthropic client unless needed)
        _ctor = {"gemini": GeminiJudge, "gpt": GptJudge, "claude": ClaudeJudge}
        _lock = threading.Lock()

        def __missing__(self, k):
            with self._lock:
                if k not in self:
                    self[k] = self._ctor[k]()
            return self[k]
    pool = _Pool()
    lock = threading.Lock(); errors = []

    def work(job):
        f, outp, judges, existing = job
        try:
            res = grade_episode(f, judges, pool, frame_root, existing)
        except Exception as e:                                        # noqa: BLE001
            import traceback
            if os.environ.get("SQ_DEBUG"):
                traceback.print_exc()
            errors.append(f)
            return f, f"ERROR {e!r}"
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump(res, open(outp + ".tmp", "w"), indent=1, default=str); os.replace(outp + ".tmp", outp)
        a = res["aggregate"]
        return f, f"AQ {a['AQ']:.2f} over {a['rounds_scored']} rounds ({res['stamp_source']})" if a["AQ"] is not None else f"no scored rounds ({res['stamp_source']})"

    n = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for f, msg in ex.map(work, jobs):
            with lock:
                n += 1
                print(f"[{n}/{len(jobs)}] {f.split('/')[-2]}/{os.path.basename(f)}: {msg}", flush=True)
    if errors:                                                        # a non-zero exit makes the orchestrator re-run the step
        print(f"{len(errors)} episode(s) failed to grade", flush=True); sys.exit(1)


if __name__ == "__main__":
    main()
