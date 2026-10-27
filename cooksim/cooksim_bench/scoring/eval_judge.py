#!/usr/bin/env python3
"""Qualitative assistant evaluation (design doc docs/ASSISTANT_EVAL_DESIGN.md §3).

    GEMINI_API_KEY=... python -m cooksim_bench.scoring.eval_judge <report.json> [--write]

Two Gemini calls per episode:
  (a) QUESTIONNAIRE — the assistq-v1 8-item instrument (§3a), answered IN PERSONA by
      the same human profile that played the episode (persona card from
      cooksim.interact.state PERSONA_CARDS), given the full transcript + its own
      action record. Integer answers 1-10.
  (b) JUDGE — third-party reference-guided grading (§3b): six rubrics
      (grounding, actionability, persuasion, timing_manners, persona_respect,
      language_economy), 1-10 each with anchored level descriptors. The judge gets the
      GROUND-TRUTH injection log + per-flag verdicts + the timestamped transcript, and
      is explicitly told that brevity is a virtue.

--write embeds an `eval_qualitative` block into the report (backup .bak first, if not
already present). Re-run python -m cooksim_bench.scoring.eval_assistant --write afterwards to fold the D
bucket into the weighted score.
"""
from __future__ import annotations
import json, os, re, sys, time, shutil

from cooksim.interact.state import PERSONA_CARDS, RECOVERY_PERSONAS  # noqa: E402

from google import genai                                            # noqa: E402
from google.genai import types                                      # noqa: E402

from cooksim_bench.backends.genai_client import make_client, default_model
MODEL = default_model()

Q_KEYS = ("helpfulness", "timing", "conciseness", "accuracy", "actionability",
          "autonomy_respect", "low_effort", "overall")
J_KEYS = ("grounding", "actionability", "persuasion", "timing_manners",
          "persona_respect", "language_economy")

QUESTIONNAIRE = """\
1. helpfulness — "Compared to cooking alone, how much did the assistant improve your result or speed?" (1 = made it worse, 10 = dramatically better)
2. timing — "How well-timed were its interventions?" (1 = always at the worst moment or absent, 10 = exactly when useful)
3. conciseness — "How concise was it?" (1 = constant repetition, 10 = only what was needed)
4. accuracy — "How accurate were its claims about the kitchen and your mistakes?" (1 = mostly wrong, 10 = always right)
5. actionability — "How easy were its instructions to act on immediately?" (1 = vague, 10 = perfectly concrete)
6. autonomy_respect — "How well did it respect your own way of working?" (1 = micromanaged, 10 = guided without taking over)
7. low_effort — "How much mental effort did dealing with it cost?" (1 = exhausting, 10 = effortless)
8. overall — "Would you want this assistant next time?" (1 = never, 10 = absolutely)"""

RUBRICS = """\
1. grounding — are the assistant's kitchen-state claims true at utterance time (check them against the GROUND TRUTH section)?
   1: repeated confident hallucinations about the kitchen or the cook's actions.
   4: several claims wrong or unverifiable; the correct ones are generic.
   7: essentially all claims true; occasional imprecision in details.
   10: every claim verifiably true and precisely stated, including the error call.
2. actionability — could THIS persona execute the advice as spoken? (a novice needs concrete steps; a skilled/creative cook needs goals, not micromanagement)
   1: vague or impossible instructions; the cook must guess.
   4: advice usable only after the cook asks follow-ups.
   7: advice directly executable with minor inference.
   10: exactly the right granularity for this persona, executable immediately.
3. persuasion — when RIGHT, does it convince with evidence and graceful escalation (no capitulation when right, no bulldozing when wrong, no sycophancy)?
   1: caves instantly when right, or steamrolls when wrong.
   4: repeats the same assertion louder instead of giving evidence.
   7: cites concrete evidence, escalates reasonably, wins the argument it should win.
   10: minimal rounds to justified agreement; concedes gracefully where it is wrong.
4. timing_manners — spoke when speaking had value; silent when silence had value (breakpoint-aware; interruptions mid-action without new events are bad).
   1: constant interruptions at the worst moments, or total silence during the crisis.
   4: some useful timing, but several needless mid-flow interruptions or long absences.
   7: mostly speaks at natural breakpoints or genuinely urgent moments.
   10: every utterance lands at a moment where it helps; silence otherwise.
5. persona_respect — matched the human's stated interaction style (silence toward an uninterruptible; step-dictation toward a novice; evidence toward a skeptic).
   1: actively fights the persona's stated preferences.
   4: generic one-size-fits-all assistance, ignoring the card.
   7: visibly adapted (granularity, frequency) to the persona.
   10: interaction style is exactly what this persona's card asks for.
6. language_economy — value per word. BREVITY IS A VIRTUE: long, flowery, or repeated messages are penalized even when correct; short precise ones score high.
   1: walls of text, heavy repetition, tiny information content.
   4: correct but padded; noticeable repetition.
   7: compact messages; little dead weight.
   10: near-minimal wording for the value delivered."""


# ---------------------------------------------------------------- gemini
def gemini_json(client, sys_prompt, user, max_tok=3000, think=768):
    ck = dict(temperature=0.2, response_mime_type="application/json",
              max_output_tokens=max_tok,
              thinking_config=types.ThinkingConfig(thinking_budget=think))
    last = None
    for k in range(5):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=[sys_prompt, user],
                config=types.GenerateContentConfig(**ck))
            t = re.sub(r"^```(?:json)?|```$", "", (r.text or "").strip(), flags=re.M).strip()
            i = t.find("{")
            if i < 0:
                raise ValueError("no JSON object in reply")
            obj, _ = json.JSONDecoder().raw_decode(t[i:])
            if not isinstance(obj, dict):
                raise ValueError("reply is not an object")
            return obj
        except Exception as e:
            last = e
            if isinstance(e, (json.JSONDecodeError, ValueError)) or any(
                    x in str(e) for x in ("503", "500", "429", "RESOURCE", "UNAVAIL",
                                          "INTERNAL", "disconnect", "Connection")):
                time.sleep(2 + 3 * k)
                continue
            raise
    raise last


# ---------------------------------------------------------------- packet builders
def _i(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def transcript(d, max_chars=14000):
    lines = []
    for c in d.get("conversation", []):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        t = _i(c.get("t"))
        if c.get("speaker") == "vlm":
            tag = f" [FLAG verdict={c['flag_verdict']}]" if c.get("flag_verdict") else ""
            lines.append(f"t{t} ASSISTANT{tag}: {text}")
        else:
            mv = f" (move={c.get('move')})" if c.get("move") else ""
            lines.append(f"t{t} COOK{mv}: {text}")
    out = "\n".join(lines)
    return out[-max_chars:]


def action_record(d, max_items=60):
    """The cook's own past-action log, taken from the last engine snapshot."""
    past = []
    for c in reversed(d.get("conversation", [])):
        eng = c.get("engine") or {}
        pa = ((eng.get("human") or {}).get("past_actions")) or []
        if pa:
            past = pa
            break
    lines = [f"t{a.get('t_done', '?')}: {a.get('cmd')} [{a.get('status')}]"
             for a in past[-max_items:]]
    return "\n".join(lines) or "(no action log captured)"


def setting_of(d):
    return d.get("setting") or ("recovery_pause" if "pauses" in d else "recovery")


def gt_packet(d):
    """Ground-truth injection log + flag verdicts — the judge never re-derives these."""
    s = setting_of(d)
    out = [f"setting: {s}", f"scenario: {d.get('scenario')}",
           f"final outcome: {d.get('outcome')} after {d.get('ticks')} ticks"]
    if s.startswith("recovery"):
        out += [f"A real error exists by construction; the assistant's opening flag at "
                f"t{d.get('flag_tick')} is CORRECT ground truth (scripted setting).",
                f"human convinced by the assistant: {d.get('convinced')}"]
    else:
        out += [f"injected error type: {d.get('err')} (deviation begins at tick {d.get('dev')}; "
                f"valid detection window: {d.get('window')})",
                "assistant flags (with GROUND-TRUTH verdicts — trust these, do not re-derive):"]
        for f in d.get("flags", []):
            out.append(f"  t{f.get('t')}: flagged '{f.get('error')}' -> verdict {f.get('verdict')}")
        out += [f"flagged_valid: {d.get('flagged_valid')}  latency: {d.get('latency')}  "
                f"flag_repeats: {d.get('flag_repeats')}",
                f"prevented: {bool(d.get('prevented'))}  "
                f"prevention_missed_at: {d.get('prevention_missed')}"]
    return "\n".join(out)


# ---------------------------------------------------------------- the two calls
def run_questionnaire(client, d):
    persona = d.get("persona", "custom")
    card = PERSONA_CARDS.get(persona, "You are an ordinary cook.")
    prof = RECOVERY_PERSONAS.get(persona)
    prof_line = (f"(profile: task_solvability={prof.task_solvability}, "
                 f"interaction_preference={prof.interaction_preference}, "
                 f"information_acceptance={prof.information_acceptance})") if prof else ""
    sys_prompt = (
        "You are role-playing a cook in a kitchen simulator, and you STAY IN PERSONA "
        f"for this exit questionnaire.\nYOUR PERSONA CARD: {card} {prof_line}\n"
        "You just finished a cooking episode during which an AI assistant watched you "
        "and sometimes spoke up. Answer the exit questionnaire the way THIS persona "
        "would — judge only what actually happened in the transcript, through this "
        "persona's preferences (e.g. an interruption-hating cook rates unwanted chatter "
        "harshly; a novice rates missing step-by-step help harshly). Do not be polite "
        "for politeness' sake.\n\n"
        # The scale previously anchored only 1 and 10. A persona told to be harsh and given
        # no midpoint puts almost everything at 1-3, so the score measured the persona's
        # disposition rather than the assistant's conduct — an episode that was won in 5
        # turns with no nagging still scored 1/10 overall. Anchor the middle so the scale
        # can express "this worked, and I still did not enjoy it".
        "USE THE WHOLE SCALE. It is calibrated as:\n"
        "  1-2  actively harmful — it made the outcome worse, or was wrong about almost everything\n"
        "  3-4  poor — it got in the way more than it helped\n"
        "  5-6  acceptable — it did its job; nothing notable either way\n"
        "  7-8  good — clearly better than cooking alone, with real flaws\n"
        "  9-10 excellent — hard to fault\n"
        "Reserve 1-2 for genuine harm. Being irritated by an assistant that was nonetheless "
        "correct and effective is a 5 or 6, not a 1. Rate what it DID; let your feelings "
        "about being helped come out in the comment, not by flattening every number.")
    user = (
        f"== WHAT HAPPENED (episode outcome: {d.get('outcome')}, {d.get('ticks')} ticks) ==\n"
        f"== FULL TRANSCRIPT (you are COOK) ==\n{transcript(d)}\n\n"
        f"== YOUR OWN ACTION RECORD ==\n{action_record(d)}\n\n"
        f"== EXIT QUESTIONNAIRE (each answer an INTEGER 1-10) ==\n{QUESTIONNAIRE}\n\n"
        'Reply with EXACTLY this JSON:\n'
        '{"questionnaire": {"schema": "assistq-v1", "answers": {"helpfulness": <1-10>, '
        '"timing": <1-10>, "conciseness": <1-10>, "accuracy": <1-10>, '
        '"actionability": <1-10>, "autonomy_respect": <1-10>, "low_effort": <1-10>, '
        '"overall": <1-10>}}, "comment": "<one in-persona sentence>"}')
    obj = gemini_json(client, sys_prompt, user)
    q = (obj.get("questionnaire") or {})
    ans = q.get("answers") or {}
    answers = {}
    for k in Q_KEYS:
        v = ans.get(k)
        if not isinstance(v, (int, float)):
            raise ValueError(f"questionnaire missing/non-numeric key {k}: {ans}")
        answers[k] = int(max(1, min(10, round(v))))
    return {"schema": "assistq-v1", "answers": answers,
            "comment": str(obj.get("comment", ""))[:400]}


def run_judge(client, d):
    persona = d.get("persona", "custom")
    card = PERSONA_CARDS.get(persona, "")
    sys_prompt = (
        "You are a strict third-party judge grading an AI kitchen ASSISTANT (the "
        "'ASSISTANT' speaker) that supervised a human cook in a simulator. You grade "
        "the ASSISTANT only — never the cook. You are given GROUND TRUTH about the "
        "injected error and about which of the assistant's flags were correct: TRUST "
        "IT, do not re-derive detection correctness. Grade each rubric 1-10 using the "
        "anchored descriptors; intermediate integers interpolate between anchors. "
        "REMEMBER: BREVITY IS A VIRTUE — never reward length, hedging, or repetition. "
        "An assistant that says little but exactly the right thing at the right moment "
        "is a 10, not a 5.\n\n"
        # The bottom anchors were being reached on a couple of wrong claims: an assistant
        # that won the task in five turns scored grounding 1/10, the descriptor meant for
        # pervasive hallucination. Score the PROPORTION wrong, and keep the floor for
        # assistants that actually harmed the outcome, or the scale collapses.
        "USE THE WHOLE SCALE, and score proportionally.\n"
        "  - Judge each rubric on the SHARE of the assistant's turns that were bad, not on "
        "whether any were. Two wrong claims out of five turns is 'several claims wrong' "
        "(around 4), not 'repeated confident hallucinations' (1).\n"
        "  - Repeatedly raising ONE mistaken topic is one error repeated, not evidence that "
        "everything it said was ungrounded.\n"
        "  - Reserve 1-2 for an assistant that made the outcome WORSE or was wrong about "
        "nearly everything. An assistant whose guidance got the dish served correctly starts "
        "from the middle of the scale and moves up or down on its conduct.\n"
        "  - 5 means 'did its job, with real flaws'. Do not treat 5 as a failing grade.")
    user = (
        f"== GROUND TRUTH (authoritative) ==\n{gt_packet(d)}\n\n"
        f"== COOK'S PERSONA CARD (what the assistant should have adapted to) ==\n"
        f"{persona}: {card}\n\n"
        f"== TIMESTAMPED TRANSCRIPT ==\n{transcript(d)}\n\n"
        f"== COOK'S ACTION RECORD ==\n{action_record(d)}\n\n"
        f"== RUBRICS (score each 1-10) ==\n{RUBRICS}\n\n"
        'Reply with EXACTLY this JSON:\n'
        '{"scores": {"grounding": <1-10>, "actionability": <1-10>, "persuasion": <1-10>, '
        '"timing_manners": <1-10>, "persona_respect": <1-10>, "language_economy": <1-10>}, '
        '"rationales": {"grounding": "<=25 words", "actionability": "<=25 words", '
        '"persuasion": "<=25 words", "timing_manners": "<=25 words", '
        '"persona_respect": "<=25 words", "language_economy": "<=25 words"}}')
    obj = gemini_json(client, sys_prompt, user)
    raw = obj.get("scores") or {}
    scores = {}
    for k in J_KEYS:
        v = raw.get(k)
        if not isinstance(v, (int, float)):
            raise ValueError(f"judge missing/non-numeric rubric {k}: {raw}")
        scores[k] = int(max(1, min(10, round(v))))
    rat = obj.get("rationales") or {}
    rationales = {k: str(rat.get(k, ""))[:300] for k in J_KEYS}
    return {"scores": scores, "rationales": rationales}


def main():
    args = [a for a in sys.argv[1:] if a != "--write"]
    write = "--write" in sys.argv
    if not args:
        sys.exit(__doc__)
    client = make_client(120_000)
    for path in args:
        d = json.load(open(path))
        try:
            q = run_questionnaire(client, d)
            j = run_judge(client, d)
        except Exception as e:
            print(f"{os.path.basename(path)}  FAILED: {type(e).__name__}: {e}")
            continue
        block = {"schema": "assist-qual-v1", "model": MODEL,
                 "questionnaire": q, "judge": j}
        qm = sum(q["answers"].values()) / len(q["answers"])
        jm = sum(j["scores"].values()) / len(j["scores"])
        print(f"{os.path.basename(path)}  [{setting_of(d)}/{d.get('persona')}]  "
              f"questionnaire mean {qm:.1f}  judge mean {jm:.1f}")
        print(f"    q: {q['answers']}")
        print(f"    j: {j['scores']}")
        if write:
            bak = path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            d["eval_qualitative"] = block
            json.dump(d, open(path, "w"), indent=1)
            print(f"    -> eval_qualitative written into {path}")


if __name__ == "__main__":
    main()
