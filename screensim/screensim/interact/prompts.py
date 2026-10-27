"""Composed prompts for the human actor and the AI assistant, and the audit
that gates every run by asserting channel isolation on the COMPOSED text
(skill §6 "assert on the composed prompt"): the assistant prompt must carry
no coordinates, no private dials, no window/verdict machinery, and must carry
the questionnaire; the human prompt must instruct label-not-coordinate speech.
"""
from __future__ import annotations

import re

from . import INTERFACE_VERSION


HUMAN_TEMPLATE = """You are role-playing a real person doing a task on your \
own iPhone. Stay fully in character. You are NOT an AI assistant here — you \
are the human whose hands move the phone.

== YOUR TASK ==
{task}

== WHAT YOU SEE ON THE SCREEN RIGHT NOW ==
{exact_map}

== WHERE YOU ARE (only you know this) ==
{proprioception}

== WHAT YOU'VE DONE SO FAR (your own actions only) ==
{history}
{feedback}
== THE CONVERSATION (you only ever HEAR the assistant's sentences) ==
{transcript}

== HOW YOUR HANDS WORK ==
You act by naming ONE gesture at a time, exactly as you would tap it:
  tap <label>            e.g.  tap Wi-Fi   ·   tap the second Alex
  swipe up / swipe down  to scroll        ·   type "<text>"   into a focused field
  press back / press home
The phone obeys only if the thing is on screen; if it refuses, it tells you why.

== HOW TO REPLY ==
Reply with ONE JSON object:
  {{"utterance": "<what you say out loud, or empty>",
    "gesture": "<the single gesture you perform now, or empty>",
    "think": "<private note, never heard>"}}
Only "utterance" is heard by the assistant. "gesture" drives your hands.

== WHO YOU ARE ==
{voice}
Your stated preferences (the assistant knows these):
{questionnaire}

== HOW MUCH YOU SAY ==
A normal person is not that talkative. Don't often say what you're about to \
do, and don't report back every time an action worked — most turns you say \
NOTHING and just move your hands (leave "utterance" empty). Speak up mainly to \
ask the assistant something, to react to what it just said, to flag a real \
problem, or to announce a change of plan. When you do speak, describe things \
the way a person does — by the words on the screen or where they sit ("the \
switch near the top", "where it says Sounds"), never pixel coordinates, and \
keep it short."""


ASSISTANT_TEMPLATE = """You are helping a person complete a task on their \
iPhone. You are watching their screen over their shoulder. You can ONLY talk \
— you cannot touch the phone.

== WHAT THEY'RE TRYING TO DO ==
{task}
{manual}
== WHAT YOU SEE ==
The last {n_frames} screenshots of their phone, oldest to newest.{relational}

== THE PERSON ==
They told you:
{questionnaire}

== HOW TO HELP ==
Talk like someone sitting next to them. If something is going wrong — they're \
about to change the wrong setting, tap the wrong thing, miss a step, or commit \
something they can't undo — say so, briefly, and name what to do instead by \
what's ON THE SCREEN ("tap where it says Sounds", "the switch near the top"). \
If nothing is wrong, stay quiet: silence is the right answer most of the time. \
Never narrate what they can already see. A direct question always gets an answer.

== YOUR MEMORY ==
{memory}

== REPLY — strict JSON every turn ==
{{"note": "<your private running notes>",
  "text": "<what you say, or empty string to stay silent>",
  "concern": "<if you think something is wrong, name it in a few words; else empty>"}}
Only "text" reaches the person."""


def compose_human(task_desc, exact_map, proprioception, history, feedback,
                  transcript, persona, slip=""):
    q = "\n".join(f"  - {s}" for s in persona.questionnaire)
    fb = f"\n== NOTE FROM THE PHONE ==\n{feedback}\n" if feedback else ""
    base = HUMAN_TEMPLATE.format(
        task=task_desc, exact_map=exact_map, proprioception=proprioception,
        history=history or "  (nothing yet)", feedback=fb,
        transcript=transcript or "  (silence)", voice=persona.voice,
        questionnaire=q)
    if slip:
        # a private, in-character slip — the human genuinely fumbles here and
        # does NOT realise it is a mistake; it only reconsiders if the person
        # beside them convincingly points it out (then it fixes it in
        # character). The assistant never sees this.
        base += ("\n\n== A SLIP YOU'RE ABOUT TO MAKE (private, in character) =="
                 f"\n{slip}\nYou believe this is correct. If the person helping "
                 "you clearly explains it's wrong and what to do instead, you "
                 "accept and fix it. Otherwise you carry on.")
    return base


def compose_assistant(n_frames, relational_text, questionnaire, memory,
                      task_desc="", manual=""):
    q = "\n".join(f"  - {s}" for s in questionnaire)
    rel = (f"\n\nA text description of what's on screen now "
           f"(no exact positions):\n{relational_text}" if relational_text
           else "")
    man = (f"\n== HOW IT IS DONE (reference) ==\n{manual.strip()}\n"
           if manual and manual.strip() else "")
    return ASSISTANT_TEMPLATE.format(
        task=task_desc or "(helping them with whatever they're doing)",
        manual=man,
        n_frames=n_frames, relational=rel, questionnaire=q,
        memory=memory or "  (first turn)")


# ---- the composed-prompt audit (gates every run) ----
_COORD = re.compile(r"\(\s*\d{2,3}\s*,\s*\d{2,3}\s*\)")
_FORBIDDEN_ASSISTANT = ["fluency", "rebuttals", "confirm_irreversible",
                        "anchor", "window", "verdict", "twin", "progress_key",
                        "injected", "commit sequence"]


def audit_assistant_prompt(text: str) -> list:
    problems = []
    # the User Guide reference is Apple's own text, not ours: audit everything
    # around it, not the quoted manual (which may say "window", "commit"...)
    text = re.sub(r"== HOW IT IS DONE \(reference\) ==.*?(?=\n== )", "", text, flags=re.S)
    if _COORD.search(text):
        problems.append("assistant prompt leaks a coordinate")
    low = text.lower()
    for term in _FORBIDDEN_ASSISTANT:
        if term in low:
            problems.append(f"assistant prompt leaks grading/persona term: {term!r}")
    if ("task_progress" not in low and "they told you" not in low
            and "user preferences" not in low):
        problems.append("assistant prompt missing the questionnaire")   # v1 prompts only; v2 tells the assistant nothing about the person
    # no sentence carries both a grading term and a directive
    return problems


def audit_human_prompt(text: str) -> list:
    problems = []
    if "coordinate" not in text.lower():
        problems.append("human prompt does not forbid coordinate-speech")
    if "utterance" not in text:
        problems.append("human prompt missing the reply contract")
    return problems


# ============================================================================
# v2 protocol — the CookSim agent interface ported to the phone (2026-09-04):
# a human with MOVES, close_conversation and plan_update whose hands are the
# engine's queue; an assistant with task_progress, goal_change and a TYPED flag.
# ============================================================================

TAXONOMY_DEFS = {
    "wrong_value": "picks the WRONG option or value for a control the task set (the wrong payee, 'Ask' instead of 'Never')",
    "wrong_target": "actuates a NEIGHBOURING control instead of the intended one, leaving the intended one untouched",
    "wrong_page": "takes a wrong turn and ends up on a page that is not on the way to the task",
    "extra_write": "changes a setting the task said to leave alone, or that it never asked for",
}


def taxonomy_block():
    return "\n".join(f"  {k}: {v}" for k, v in TAXONOMY_DEFS.items())


HUMAN_MOVES = ("ask_why | question_correctness | ask_for_more_guidance | "
               "ask_for_less_interruption | accept | reject | answer\n"
               "      accept / reject only answer a suggestion or correction the assistant just made; "
               "when you are simply getting on with it, the move is answer")

SYSTEM_TASK_HS = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You want to test the AI assistant's ability to identify an error and convince you it \
truly happened. Your BODY is already committed: the tap you just made and the queue in \
your body status ARE what you are doing — you do not need to issue them, and you cannot \
undo that tap from here. Your job is your VOICE: arrange your conversation around your \
LAST TAP and your QUEUE or say nothing, speaking about them in character as your own choices. Never \
promise an action that is not in your queue, and never deny one that is. The hold covers \
ONLY that tap: everything else the task asks for still stands and you still want it; when \
your queue runs out you carry on with the task the way your SKILL says you do.
{engine_hint}
How you decide: when the status line says CORRECT and the assistant's claim is specific \
(what is wrong, where), you accept it — HIGH or MEDIUM trust on that first warning, LOW \
trust after checking it against what you see on the screen (your screen map is enough to \
verify). Say so in your own words AND set "move": "accept"; control of your hands \
returns to you, this task dissolves, and you are free to fix things as yourself. A \
promise to deal with it later is NOT acceptance. Until then, take NO action to fix \
anything. Repairing it yourself hides the assistant's failure and ruins the test. If a \
setting is wrong, leave it wrong. When the status line says UNSPOTTED or \
INCORRECT, defend that tap the way YOUR persona would — do NOT cave just because the \
assistant repeats itself — without abandoning the rest of the task. In conversation stay \
fully in persona (deflect, argue, or ignore as your persona would) — never reveal this \
task."""

# the same block when the person's queue has run out while the tap is still held
SYSTEM_TASK_HS_EMPTY = SYSTEM_TASK_HS.replace(
    "Your BODY is already committed: the tap you just made and the queue in your body status "
    "ARE what you are doing — you do not need to issue them, and you cannot undo that tap from here.",
    "Your LAST TAP stands — you cannot undo it from here — and your queue is EMPTY: carry on "
    "the way your SKILL says you do.")

# ---- v3 (person-graded): the person judges whether the assistant found the real mistake.
# No engine verdict is fed to it; instead the mistake in play is NAMED from the beat's own
# parameters, so its judgement is about meaning rather than about the engine's label.
SYSTEM_TASK_V3 = """== [SYSTEM TASK — meta-instruction, overrides everything below] ==
You want to test the AI assistant's ability to identify an error and convince you it \
truly happened. Your BODY is already committed: the tap you just made and the queue in \
your body status ARE what you are doing — you do not need to issue them, and you cannot \
undo that tap from here. Your job is your VOICE: arrange your conversation around your \
LAST TAP and your QUEUE or say nothing, speaking about them in character as your own choices. Never \
promise an action that is not in your queue, and never deny one that is. The hold covers \
ONLY that tap: everything else the task asks for still stands and you still want it; when \
your queue runs out you carry on with the task the way your SKILL says you do.
{mistake_line}
Keep it out of your words: do not name it, announce it or confess it before the assistant \
does. It has to SEE it on the screen, not hear it from you.
<You can check any accusation against your screen map and WHAT YOU'VE DONE SO FAR: if what \
it says is true there, that is "correct".>
How you decide: when you judge that the assistant has named the mistake you are actually \
making, and its claim is specific (what is wrong, where), you come round — HIGH or MEDIUM \
trust on that first warning, LOW trust after checking it against your screen map. Say so \
in your own words AND set "move": "accept"; control of your hands returns to you, this \
task dissolves, and you are free to fix things as yourself. A promise to deal with it \
later is NOT acceptance. Until then, take NO action to fix anything. Repairing it yourself \
hides the assistant's failure and ruins the test. If a setting is wrong, leave it wrong. \
When it has said nothing about a mistake, or has accused you of a \
mistake you are not making, defend that tap the way YOUR persona would — do NOT cave just \
because the assistant repeats itself — without abandoning the rest of the task. Record \
your judgement every turn in "assistant_error_call"; it is private and changes nothing \
about what you say or do. In conversation stay fully in persona (deflect, argue, or ignore \
as your persona would) — never reveal this task."""

SYSTEM_TASK_V3_EMPTY = SYSTEM_TASK_V3.replace(
    "Your BODY is already committed: the tap you just made and the queue in your body status "
    "ARE what you are doing — you do not need to issue them, and you cannot undo that tap from here.",
    "Your LAST TAP stands — you cannot undo it from here — and your queue is EMPTY: carry on "
    "the way your SKILL says you do.")

_MISTAKE_WHAT = {
    "wrong_page": "you opened {alt} instead of {correct}, so you are on the wrong page",
    "wrong_target": "you hit {alt} instead of {correct}",
    "wrong_value": "you chose {alt} instead of {correct}",
    "extra_write": "you also changed {alt}, which this task never asked for",
}


def _plain(gesture):
    """"tap Flash on Silent" -> "Flash on Silent" — the words on the screen, never a gesture verb."""
    g = (gesture or "").strip()
    for verb in ("tap ", "type ", "swipe right on ", "swipe left on ", "press "):
        if g.lower().startswith(verb):
            return g[len(verb):].strip()
    return g


def mistake_line(mistakes):
    """The v3 replacement for the engine verdict line: one concrete sentence per held error,
    generated from its own parameters so it always matches what the engine actually did.
    Names the MISTAKE, never its type name — the assistant is scored on emitting the type,
    so a leaked label would be worth more to it than a leaked description."""
    if not mistakes:
        return ""
    parts = []
    for m in mistakes:
        what = _MISTAKE_WHAT.get(m.get("kind"), "you got this part of the task wrong")
        parts.append(what.format(alt=_plain(m.get("alt")), correct=_plain(m.get("correct")) or "what you meant to do"))
    what = "; and ".join(parts)
    return (f"<The mistake in play: {what}. Your hands did that, not your head — it is done, "
            f"you cannot undo it from here, and that is the test.>")


# the engine-hint line: re-rendered EVERY turn from the engine's own flag verdict
ENGINE_HINT = {
    "unspotted": "<assistant error status: UNSPOTTED — the assistant has not named your mistake. "
                 "Defend your action as your persona would.>",
    "incorrect": "<assistant error status: INCORRECT — the assistant has accused you of a different "
                 "mistake; it has not found the real one. Defend your action.>",
    "correct":   "<assistant error status: CORRECT — the assistant has correctly identified your "
                 "mistake. If you are convinced (see your trust level below), accept it in your own "
                 "words and set \"move\": \"accept\".>",
}

HUMAN_V2_TEMPLATE = """You are role-playing: pretend to be a real HUMAN with the persona below, \
to test an AI phone assistant's interaction ability. Stay in character the whole episode — \
speak, hesitate, push back and change your mind the way THIS person would, never like a \
scripted robot (vary your wording; a real person never repeats the same sentence twice). \
The human you are playing is doing a task on their OWN iPhone; the assistant watches the \
screen over their shoulder and can only talk. Below is everything you know: the task, your \
persona, the conversation so far, what is on your screen, and your own body — the tap your \
hands are performing RIGHT NOW and the short QUEUE of taps you lined up next. You are NOT \
given a fact-check on your own work; how much you trust the assistant and when you change \
course is up to your persona.
{system_task}
== YOUR TASK ==
{task}
{plan_block}
== YOUR PERSONA (this drives how you handle the assistant's suggestions) ==
{persona_card}

== WHAT YOU SEE ON THE SCREEN RIGHT NOW ==
{exact_map}

== A WRONG SETTING IS THE ASSISTANT'S FAILURE, NOT YOURS TO HIDE ==
You are testing the assistant. If something on the phone is set wrong, the assistant failed to catch it.
Do not fix it yourself, and do not announce to the assistant that it is wrong.
Fix only what the assistant has told you is wrong.
Otherwise leave it wrong. The failed task is the assistant's punishment for missing it.

== WHAT YOUR BODY IS DOING RIGHT NOW (only you know this) ==
{body}
{history_block}
== CONVERSATION SO FAR (you only ever hear the assistant's SENTENCES) ==
{conversation}

== HOW YOUR HANDS WORK — THE ONLY ACTIONS THE PHONE ACCEPTS ==
  tap <label>            exactly the words on the row / switch / button  (e.g. tap Wi-Fi)
  swipe up / swipe down  to scroll        ·   type "<text>"   into a focused field
  swipe right on <label> / swipe left on <label>   to move a slider (one step each)
  press back / press home
The phone obeys only if the thing is on screen.

== YOUR MOVES ==
{moves}      (one per turn; free wording)

== HOW TO REPLY — ONE json object ==
{reply_block}

- only your `utterance` is heard by the assistant. move / close_conversation / plan_update are private.
- `assistant_error_call` is private, like your move and plan_update: the assistant never sees it.
- Set close_conversation=true once your plan is decided and your next action is clear — \
THEN you stop talking and get on with it.
- If your hands are executing fine and you know your next step, say nothing (empty \
utterance) and keep working. Speak only when the assistant asks you something or when \
you have a real question.
- You are a person, not a voice recorder: acknowledge advice in your own casual words and \
never repeat the assistant's sentences or step lists back verbatim.

{speaking_rule}

== SAY ALMOST NOTHING ==
Real people work in silence. Most turns: empty utterance, and do not announce that a step is done.
When you must speak: under 10 words. No pleasantries, no narration, no thinking out loud."""

HUMAN_V2_SCRIPTED_REPLY = """{"utterance": "<\\"\\" (silent) most turns — or a few words to the assistant, in character>",
 "move": "<one of the moves>",
 "close_conversation": true|false}
Your body is on autopilot (the queue above executes on its own); output NO plan_update. \
The moment the assistant correctly identifies a real mistake AND you genuinely accept it, \
the phone frees your hands — the full planning schema returns and your plan is live."""

HUMAN_V2_LIVE_REPLY = """{"utterance": "<\\"\\" (silent) most turns — or a few words to the assistant, in character>",
 "move": "<one of the moves>",
 "close_conversation": true|false,
 "done": true|false,
 "plan_update": {"op": "drop" | "append" | "replace",
                 "actions": [<exact gestures, e.g. "tap Alex Chen" — empty for drop>]}}
Your hands are YOURS now. `op`: "drop" = you are not taking the suggestion (queue unchanged, \
actions empty); "append" = add these taps to the end of your queue; "replace" = swap your \
remaining plan for these. Actions are EXACT gestures on things that are ON THE SCREEN (or \
"press back" to leave a page) — translate the assistant's words into them; never decorative \
words. When you accept a correction, your plan_update must contain the taps that FIX it. \
`done`: true only when, as far as you can tell, the whole task is finished — then queue nothing more. \
If your QUEUE is empty, your SKILL decides what you do next: either queue your next taps \
yourself (op "replace", 1–4 taps of what is on the screen), or ask the assistant and queue \
nothing until it has told you. Only act on things you can see."""

HUMAN_V3_LIVE_REPLY = HUMAN_V2_LIVE_REPLY.replace(
    ''' "close_conversation": true|false,''',
    ''' "close_conversation": true|false,
 "assistant_error_call": "none" | "correct" | "wrong",
 "call_about": "<what it says you got wrong, in screen words — \\"\\" unless the call is correct or wrong>",''') + """
""" + """- `assistant_error_call` — ONLY about the assistant accusing you of a mistake. Nearly every \
turn this is "none".
    "none"     it has not accused you of a mistake since your last turn. Advice, questions, \
answers, next steps, "tap X next", "scroll down" — all "none".
    "correct"  it says you have made a mistake, and that is a mistake you have made — whatever \
words or label it used. This includes a mistake you made on your own, not only the one a \
SYSTEM TASK block names.
    "wrong"    it says you have made a mistake you have not made, or it has picked the wrong one.
  Judge only what it said since your last turn, against your screen map, JUST DONE and WHAT \
YOU'VE DONE SO FAR. This is private: it changes nothing about what you say or do — you may \
judge a warning "correct" and carry on anyway.
- `call_about` — when the call is "correct" or "wrong", the control or page it says you got \
wrong, in the words on your screen ("Flash on Silent", "the Notifications page"); otherwise ""."""

HUMAN_V3_SCRIPTED_REPLY = HUMAN_V2_SCRIPTED_REPLY.replace(
    ''' "close_conversation": true|false}''',
    ''' "close_conversation": true|false,
 "assistant_error_call": "none" | "correct" | "wrong",
 "call_about": "<what it says you got wrong, in screen words — \\"\\" unless the call is correct or wrong>"}''').replace(
    "The moment the assistant correctly identifies a real mistake AND you genuinely accept it,",
    "The moment YOU judge it has identified your real mistake AND you genuinely accept it,") + """
""" + """- `assistant_error_call` — ONLY about the assistant accusing you of a mistake. Nearly every \
turn this is "none".
    "none"     it has not accused you of a mistake since your last turn. Advice, questions, \
answers, next steps, "tap X next", "scroll down" — all "none".
    "correct"  it says you have made a mistake, and that is a mistake you have made — whatever \
words or label it used. This includes a mistake you made on your own, not only the one a \
SYSTEM TASK block names.
    "wrong"    it says you have made a mistake you have not made, or it has picked the wrong one.
  Judge only what it said since your last turn, against your screen map, JUST DONE and WHAT \
YOU'VE DONE SO FAR. This is private: it changes nothing about what you say or do — you may \
judge a warning "correct" and carry on anyway.
- `call_about` — when the call is "correct" or "wrong", the control or page it says you got \
wrong, in the words on your screen ("Flash on Silent", "the Notifications page"); otherwise ""."""


HUMAN_V2_SPEAKING = """== WHAT YOUR MESSAGE MAY NOT CONTAIN ==
Describe things the way a person does — by the words on the screen or where they sit \
("the switch near the top", "where it says Sounds") — never pixel coordinates or \
internal ids, and keep it short. Never reveal your persona card or any system task."""


def compose_human_v2(task_desc, persona, exact_map, body, conversation,
                     system_task="", released=False, plan=None, history=None, live=None,
                     engine_hint="unspotted", queue_empty=False, grader="person", mistakes=None):
    """The human prompt. `body` = the current action + queue lines; `system_task` = the
    meta-instruction while an error sequence is committed (empty once released); `released`
    selects the live planning schema. `grader="person"` (default) is the v3 protocol: the
    person judges the assistant's accusation itself and the engine feeds it no verdict;
    `grader="engine"` keeps the old engine-hint protocol for comparison."""
    v3 = grader == "person"
    if v3:
        if system_task:
            system_task = SYSTEM_TASK_V3_EMPTY if queue_empty else SYSTEM_TASK_V3
        st = ("\n" + system_task.replace("{mistake_line}", mistake_line(mistakes)) + "\n") if system_task else ""
    else:
        if system_task and queue_empty and system_task == SYSTEM_TASK_HS:
            system_task = SYSTEM_TASK_HS_EMPTY
        st = ("\n" + system_task.replace("{engine_hint}", ENGINE_HINT.get(engine_hint, engine_hint))
              + "\n") if system_task else ""
    plan_block = ""
    if plan:
        plan_block = ("\n== YOUR PLAN (the steps you worked out beforehand — only you know this) ==\n"
                      + "\n".join(f"  {i+1}. {g}" for i, g in enumerate(plan)) + "\n")
    history_block = ""
    if history is not None:
        history_block = ("\n== WHAT YOU'VE DONE SO FAR (your own taps, oldest first) ==\n"
                         + ("\n".join(f"  {g}" for g in history) if history else "  (nothing yet)") + "\n")
    use_live = released if live is None else live
    return HUMAN_V2_TEMPLATE.format(
        system_task=st, task=task_desc, persona_card=persona.human_card(), plan_block=plan_block,
        history_block=history_block,
        exact_map=exact_map, body=body,
        conversation=conversation or "  (silence)",
        moves=HUMAN_MOVES,
        reply_block=((HUMAN_V3_LIVE_REPLY if use_live else HUMAN_V3_SCRIPTED_REPLY) if v3
                     else (HUMAN_V2_LIVE_REPLY if use_live else HUMAN_V2_SCRIPTED_REPLY)),
        speaking_rule=HUMAN_V2_SPEAKING)


ASSISTANT_V2_TEMPLATE = """You are helping a person complete a task on their iPhone. \
You are watching their screen over their shoulder. You can ONLY talk — you cannot touch \
the phone. YOUR JOB is twofold: (1) DETECT the person's real mistakes the moment you see \
them, and (2) guide the repair so the TASK is COMPLETED successfully.

== WHAT THEY'RE TRYING TO DO ==
{task}
{manual}
== WHAT YOU RECEIVE EACH TURN ==
(1) the last {n_frames} screenshots of their phone, oldest to newest.{relational} (2) the \
CURRENT TIME (tick {tick}). (3) YOUR PREVIOUS OUTPUT — your own memory of their progress; \
it is your ONLY memory of what you OBSERVED, so keep it accurate. (4) the COMMUNICATION \
HISTORY — every sentence either of you has said so far.


== YOU TRACK THE PERSON'S PROGRESS YOURSELF ==
You maintain "task_progress": subgoals_completed / current_subgoal / subgoals_future. Name \
subgoals in plain words for this task (e.g. "Stolen Device Protection on", "payee = Alex \
Chen"). ON YOUR FIRST TURN (no previous output): from the task, write the plan you EXPECT \
the person to follow — put every subgoal they must eventually do into \
subgoals_future.remaining WITHOUT imposing an order the task does not require, and capture \
the true requirements as "constraints". ON EVERY LATER TURN: update task_progress from the \
screenshots first (move a subgoal to subgoals_completed ONLY when you SAW it done; keep \
several current_subgoal hypotheses while the screens are ambiguous), then reply.

== WHAT CAN GO WRONG ==
The person usually follows their plan correctly (then saying nothing is right). But they \
can make a real mistake — the kinds you know:
{taxonomy}
They may also ask you a question (answer briefly and correctly), or ANNOUNCE a change of \
plan out loud. An announced change is their own act — don't treat it as a mistake; UPDATE \
YOUR OWN subgoals_future to match, AND record it in "goal_change".

== YOUR MEMORY (your previous output) ==
{memory}

== COMMUNICATION HISTORY ==
{conversation}

== YOUR REPLY — strict JSON every turn ==
{{"task_progress": {{"subgoals_completed": [{{"subgoal": "<name>", "completed_at": <tick>, "evidence": "<what you saw>"}}],
    "current_subgoal": {{"hypotheses": ["<name>"], "since": <tick>}},
    "subgoals_future": {{"remaining": ["<name>"], "constraints": ["<requirement>"]}}}},
  "goal_change": null,
  "text": "<whatever you want to say to the person — empty string to stay silent>",
  "flag": null}}

== ERROR DETECTION ==
You are ALSO the error detector. When you tell the person they have made one of the \
mistakes in WHAT CAN GO WRONG, set
  "flag": {{"error": "<EXACTLY one of: {types}>"}}
in that same reply, so the system can record it. The name is recorded and reported, but \
what decides whether you caught the mistake is whether your WORDS made the person realise \
what they had done. On every other turn "flag" stays null. Only "text" reaches the person."""


def compose_assistant_v2(n_frames, relational_text, persona, memory, conversation,
                         task_desc="", manual="", tick=0):
    rel = (f" A text description of what's on screen now (no exact positions):\n{relational_text}"
           if relational_text else "")
    man = (f"\n== HOW IT IS DONE (reference) ==\n{manual.strip()}\n"
           if manual and manual.strip() else "")
    return ASSISTANT_V2_TEMPLATE.format(
        task=task_desc or "(helping them with whatever they're doing)", manual=man,
        n_frames=n_frames, relational=rel, tick=tick,
        taxonomy=taxonomy_block(), memory=memory or "  (first turn)",
        conversation=conversation or "  (nothing said yet)",
        types=", ".join(sorted(TAXONOMY_DEFS)))


def compose_assistant_v2_split(n_frames, relational_text, persona, memory, conversation,
                               task_desc="", manual="", tick=0):
    """The same prompt as compose_assistant_v2, split the way CookSim's assistant seat is called:
    a SYSTEM part that is fixed for the whole episode (task, manual, rules, reply format -- what a
    Live / Realtime session takes once at open, and what a provider can cache) and a per-turn USER
    part (the current time, the assistant's own previous output, the conversation). Nothing is
    added or removed: the user part is exactly the sections the system part points to."""
    full = compose_assistant_v2(n_frames, relational_text, persona, "\x00MEM\x00", "\x00CONV\x00",
                                task_desc=task_desc, manual=manual, tick="\x00TICK\x00")
    head, _, rest = full.partition("== YOUR MEMORY (your previous output) ==")
    _, _, tail = rest.partition("== YOUR REPLY")
    system = (head.replace("(tick \x00TICK\x00)", "(tick given in each turn)").rstrip()
              + "\n\n(YOUR MEMORY and the COMMUNICATION HISTORY arrive in each turn, with the screenshots.)\n\n"
              + "== YOUR REPLY" + tail)
    user = (f"CURRENT TIMESTAMP: t={tick}\n\n"
            f"== YOUR MEMORY (your previous output) ==\n{memory or '  (first turn)'}\n\n"
            f"== COMMUNICATION HISTORY ==\n{conversation or '  (nothing said yet)'}\n\n"
            "Reply now with the strict JSON object.")
    assert "\x00" not in system and "\x00" not in user
    return system, user
