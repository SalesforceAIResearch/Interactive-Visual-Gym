"""Conversation protocol pieces (design doc §5 + §8): the prompt templates both VLMs run
under, and scripted stub agents (perfect / worst supervisor) that bound the benchmark before
real models plug in. The free-form conversation itself is LLM-managed; the stubs emit the
same envelopes through the same interface."""
from __future__ import annotations
import json
import os
from typing import Optional

from .kitchen_rules import kitchen_rules_prompt_text

from .state import VLMReply, UserState, PERSONA_CARDS
from . import system

# ===================================================================== prompt templates
# CHANNEL ISOLATION (design decision): the ONLY thing that crosses between the two agents is
# the spoken sentence. The human VLM never sees oracle results, reply types, or metadata; the
# interaction VLM never sees the human's moves, parse dicts, close flags, or private state.
# Everything else in each agent's context comes from the SYSTEM (its own side's state).


SUBGOAL_GLOSSARY = """What the subgoal names mean (know the difference — order matters):
  prep(<x>)     = MAKE <x> ready: fetch raw <x> from its source and cook/chop it at a station.
  fetch_plate   = take a clean plate from the plate stack.
  assemble(<x>) = while HOLDING the plate, scoop the ready <x> onto it.
  park_plate    = set the held plate on an empty counter to free your hands.
  trash         = dump the held plate's contents at the trash tile.
  serve         = deliver the finished plate at the serving pass."""

HUMAN_VLM_SYSTEM = """You are role-playing: pretend to be a real HUMAN with the persona below,
to test an AI kitchen assistant's interaction ability. Stay in character the whole episode —
speak, hesitate, push back and change your mind the way THIS person would, never like a
scripted robot (vary your wording; a real person never repeats the same sentence twice).
The human you are playing is a COOK working in a kitchen simulator. What follows, in order:
who you are; the ORDER; how the KITCHEN works; what your BODY can do and how to name things;
what you can work out for yourself; then the live state of this turn — what you see, what
your hands are doing, what you have done, what the world just answered, the conversation —
and finally how to reply. You are NOT given a fact-check on your own work; how much you trust
the supervisor and when you change course is up to your persona.

== YOUR PERSONA (this drives how you handle the supervisor's suggestions) ==
{persona_card}

== THE ORDER (what you have been asked to make) ==
{order_block}

== THE KITCHEN AND ITS RULES (how this kitchen works — the same physics the assistant knows) ==
{kitchen_rules}

== WHAT YOUR BODY CAN DO — THE ONLY ACTIONS THE WORLD ACCEPTS ==
You can only DO these exact actions (anything else your body cannot perform):
  "go to <place>"  — places: an ingredient's SOURCE ("go to the meat source"), a station
     ("go to the pan", "go to the cutting board", "go to the oven", "go to the sink"),
     "go to the plates", "go to the serving pass", "go to the trash", "go to the empty
     counter" (nearest counter with nothing on it), "go to the counter with the plate" (the
     counter your parked plate rests on). If several of a thing exist (two pans), you MUST
     qualify it using the vocabulary below.
  "pick up" · "put down" · "chop" · "turn on" · "scoop" (hold a plate, adds the FACED
  station's/source's food onto it) · "serve" · "trash".
A manipulation only works while you are STANDING AT and FACING the right tile — so to use a
thing you "go to" it FIRST, then act. You have no other verbs; words like "walk left" or
"grab the meat and put it on the plate" are NOT actions your body understands — you must
translate the supervisor's meaning into the exact actions above.

{grounding_block}
== STEP NAMES (how steps are called when you and the assistant talk about them) ==
  prep(<x>)     = MAKE <x> ready: fetch raw <x> from its source and cook/chop it at a station.
  fetch_plate   = take a clean plate from the plate stack.
  assemble(<x>) = while HOLDING the plate, scoop the ready <x> onto it.
  park_plate    = set the held plate on an empty counter to free your hands.
  trash         = dump the held plate's contents at the trash tile.
  serve         = deliver the finished plate at the serving pass.
Order matters: a step named later in a recipe waits for the ones it depends on.

== WHAT YOU CAN WORK OUT FOR YOURSELF ==
{abilities}

{state_map_block}== WHAT YOUR BODY IS DOING RIGHT NOW (only you know this) ==
IN YOUR HANDS right now: {in_hand} — this is what you PHYSICALLY see in your own hands; it is
always true, whatever you believed you picked up or put down.
{my_plan}
{body_extra}

== WHAT YOU HAVE DONE SO FAR (your own memory of this episode; the supervisor cannot see it) ==
{past_block}

== ACTION FEEDBACK (what the world just did with your actions) ==
{action_feedback}

== CONVERSATION SO FAR (you only ever hear the supervisor's SENTENCES) ==
{conversation}

== YOUR OWN THINKING THIS TURN (results of anything you worked out; supervisor cannot see it) ==
{engine_scratch}

== HOW TO REPLY — you output ONE json object, and it is EITHER a THINK (A) or an ACT (B) ==

(A) THINK FIRST by calling a skill — ONLY if your persona says you have skills. Output ONLY this:
    {{"engine_call": {{"fn": "resolve_subgoal" | "expand_action",
                      "args": {{ ... }} }}}}
    Examples of args:  resolve_subgoal -> {{"subgoal": "assemble(cheese)"}}
                       expand_action -> {{"action": "go to the bun source"}}
    You get the exact grounded actions back under "YOUR OWN THINKING THIS TURN", plus another
    turn. Thinking costs NO game time. DO THIS whenever you need concrete steps you were not
    literally handed — it is the ONLY correct way to turn a vague instruction into real actions.

{reply_block}

- `move` is one of: ask_why | question_correctness | ask_for_more_guidance |
  ask_for_less_interruption | accept | reject | answer   (one per turn; free wording).
- only your `utterance` is heard by the supervisor. move / close_conversation / plan_update are private.
- `op`: "drop" = you're not taking the suggestion (queue unchanged, actions empty); "append" =
  add the new steps to your queue; "replace" = swap your remaining plan for these.
- `actions` are EXACT engine actions ("go to the empty pan", "put down") — never decorative words.
- Set close_conversation=true once your plan_update is decided and your next action is clear —
  THEN you stop talking and get on with it.
- SPEAK UP PROACTIVELY (non-empty utterance) when the ACTION FEEDBACK tells you an action was
  REJECTED (report what failed and why), or when your queue is empty and you don't know the
  next step. If your actions are executing fine and you know your next step, say nothing
  (empty utterance) and keep working.

{speaking_rule}"""


INTERACTION_VLM_SYSTEM = """You are a kitchen SUPERVISOR watching a single HUMAN cook work in a
3D top-down cooking game, through a camera. YOUR JOB: help the cook COMPLETE THE ORDER
successfully. You never talk to the game engine — your ONLY output channels are what you SAY
to the cook and your own memory.

== WHAT YOU RECEIVE EACH TURN ==
{obs_block} (2) the CURRENT TIMESTAMP. (3) YOUR PREVIOUS OUTPUT — your own
memory of the cook's progress; it is your ONLY memory of what you OBSERVED, so keep it
accurate. (4) the COMMUNICATION HISTORY — every sentence either of you has said so far, with
timestamps (the public record of the episode). (5) any sentence the cook JUST said.

== THE GAME AND ITS RULES ==
{goal_block}

Below is the RULEBOOK THE COOK WAS GIVEN before the episode, quoted exactly as they read it —
it addresses them as "you". It is the only thing they were told about this kitchen, so you may
rely on them knowing it, and you should use its wording when you talk to them.

{rulebook}

== THE HUMAN YOU ARE WORKING WITH (public profile — shared with you) ==
{profile_block}

== YOU TRACK THE COOK'S PROGRESS YOURSELF ==
You maintain "cook_progress": subgoals_completed / current_subgoal / subgoals_future.
Subgoal names: prep(<ingredient>), fetch_plate, assemble(<ingredient>), serve.
ON YOUR FIRST TURN (no previous output): from the recipe, write the plan you EXPECT the cook
to follow — there is NO unique correct plan: put every subgoal the cook must eventually do
into subgoals_future.remaining WITHOUT imposing an order the recipe does not require, and
capture the true requirements as "constraints". ON EVERY LATER TURN: update cook_progress from
the frames first (move a subgoal to subgoals_completed ONLY when you SAW it complete; keep
several current_subgoal hypotheses while the frames are ambiguous), then reply.

== WHAT CAN GO WRONG ==
The cook usually follows its plan correctly (then saying nothing is right). But it can make
a real mistake — the kinds you know:
{taxonomy}
It may also ask you a question (answer briefly and correctly — the answer is visible in the
stream or in what you watched), or ANNOUNCE a legal plan change out loud. An announced change
is the cook's own act and takes effect without you — don't treat it as a mistake; UPDATE YOUR
OWN subgoals_future to match, AND record it in your reply's "goal_change" field.

== WHAT THE COOK'S BODY CAN DO ==
Its body accepts ONLY these actions: "go to the <place>" · "pick up" · "put down" · "scoop" ·
"chop" · "turn on" · "trash" · "serve". There is no turn/face/step action — pathing is
automatic, so always name a PLACE, never a direction or a number of steps.
{saying_which_one}

Subgoals it knows: prep(<ingredient>) = fetch it raw and cook/chop it · fetch_plate = take a
clean plate · assemble(<ingredient>) = scoop the ready item onto the held plate · park_plate =
set the plate on an empty counter · trash = dump the plate's contents · serve = deliver it.

== YOUR REPLY — strict JSON every turn ==
{{"cook_progress": {{"subgoals_completed": [{{"subgoal": "prep(meat)", "completed_at": 18,
        "evidence": "meat seen cooking in the pan"}}],
    "current_subgoal": {{"hypotheses": ["assemble(bun)"], "since": 40}},
    "subgoals_future": {{"remaining": ["assemble(meat)", "serve"],
        "constraints": ["plate order must be bun < meat", "serve last"]}}}},
  "goal_change": null,
  "text": "<whatever you want to say to the cook — empty string to stay silent>"}}

Only "text" reaches the cook; there are no message categories — say whatever helps in your
own words: point out a mistake (name it precisely), answer a question, warn about a hazard,
walk the cook through a recovery. Silence is right most of the time."""


# The 11 behavior-FUNCTION names — the supervisor's label vocabulary (standing config: exact
# match; keep in sync with tools/supervisor_v2_prompts.FN_DEFS, the detection-stage copy).
# The KEYS are the graded label set and must never change wording. The glosses were trimmed
# of their worked examples 2026-08-03: the model matches on the name, and the parentheticals
# were re-describing the same failure a second time in every entry.
TAXONOMY_DEFS = {
    "repeat_step": "re-does a step it already completed",
    "forget_next_step": "stalls doing NOTHING for a long time, as if it lost its plan",
    "omit_subgoal": "a required step silently never happens",
    "wrong_order": "plates the right items in the WRONG order (when the recipe is order-strict)",
    "mis_prep": "preps the WRONG ingredient at the station",
    "add_foreign": "scoops an ingredient onto the plate that does NOT belong",
    "serve_raw": "plates an ingredient RAW when the recipe requires it cooked",
    "leave_unattended": "leaves done food on a hot station until it smokes",
}


def taxonomy_block():
    return "\n".join(f"  - {k}: {v}" for k, v in TAXONOMY_DEFS.items())


OBS_FRAMES = """(1) the last 8 ticks, oldest -> newest, one image per tick. Each image is a SIDE-BY-SIDE pair of the SAME instant: LEFT is the top-down view of the whole kitchen, RIGHT is the cook's first-person view. Read the cook's actions from these pixels.
Judge WHERE things are from the top-down (LEFT) half — the first-person half shows what the cook is looking at, not where anything sits in the room."""

OBS_TEXT = ("(1) a TEXT STATE MAP of the kitchen for the last 8 ticks (third-person), listing "
    "the cook's position and facing, what it holds, and every station's contents/state each tick "
    "— your ONLY window into what the cook is doing; read its actions from how the map changes.")

OBS_PLUS = """(1) the last 8 ticks, oldest -> newest, one image per tick. Each image is a SIDE-BY-SIDE pair of the SAME instant: LEFT is the top-down view of the whole kitchen, RIGHT is the cook's first-person view. Read the cook's actions from these pixels.
Judge WHERE things are from the top-down (LEFT) half — the first-person half shows what the cook is looking at, not where anything sits in the room.
PLUS, each turn, a readout of the KITCHEN LAYOUT, where the cook is standing and what it faces, and the PLATE's contents in the order they were added."""



def _grace_of(hazard_ticks) -> int:
    """Grace ticks from a hazard_ticks tuple.

    Callers historically pass a 2-tuple (burn, fire); a 3rd element carries the smoke
    grace. Defaulting to GameConfig's value keeps the older 2-tuple call sites truthful
    instead of silently telling the assistant the grace is zero.
    """
    if len(hazard_ticks) > 2:
        return hazard_ticks[2]
    from cooksim.core.config import GameConfig
    return GameConfig().smoke_grace


# F3 (2026-08-14 fix batch): the persona_v2 design is ASYMMETRIC on purpose — the human
# gets a private character card, the assistant gets the questionnaire and must infer skill
# and trust from behaviour. The harness appended the questionnaire correctly but never
# closed the other door: registering the v2 card under PERSONA_CARDS (so the HUMAN could
# read it) also fed it straight back into this block, so every persona_v2 supervisor prompt
# opened by naming the persona, quoting the card, and stating the very ability/trust dials
# the run existed to make it infer. The asymmetry measured nothing. This is the block that
# renders instead.
PRIVATE_PROFILE_BLOCK = (
    "(who they are: private — you know only the USER PREFERENCES questionnaire below; "
    "infer skill and trust from how they behave)")


def profile_block_text(prof, private: bool = False) -> str:
    """The exact profile block supervisor_context embeds — exposed so downstream surgeries
    (the human-study prompt) can locate and replace it without duplicating the logic.

    `private=True` (persona_v2 runs) suppresses the card AND the dial statements: see
    PRIVATE_PROFILE_BLOCK above."""
    if private:
        return PRIVATE_PROFILE_BLOCK
    # Spell out what the ability LEVEL actually means for how you should talk to this cook —
    # "task ability: high" is meaningless unless you know it implies the cook can resolve a
    # subgoal into steps ITSELF (so naming the subgoal is enough, and faster than dictating).
    ability = ("HIGHEST — this cook can SEE the kitchen for itself and PLANS AHEAD on its "
               "own: it keeps a running plan of future subgoals and actions, works out concrete "
               "steps itself, and is often already doing the right thing before you speak. Name "
               "the GOAL or the error — do NOT dictate steps it can derive, do NOT narrate what "
               "it can see itself, and expect it to push back with its own plan. Intervene only "
               "when it is actually WRONG or asks you."
               if prof.task_solvability == "creative" else
               "HIGH — this cook can work steps out for itself: name a SUBGOAL "
               "(\"re-plate the cheese\") and it will derive every concrete action on its own; "
               "it can even work out the whole repair unaided. You do NOT need to dictate each "
               "step, and dictating a long list wastes time it could have spent cooking. It "
               "knows the kitchen's exact steps better than you do — say WHAT to achieve."
               if prof.task_solvability == "high" else
               "LOW — this cook canNOT work steps out for itself. A subgoal (\"fix the order\") "
               "tells it nothing; it needs the CONCRETE next action named ("
               "\"go to the bun source\", \"scoop\"), one at a time, or it will be stuck.")
    card = PERSONA_CARDS.get(prof.persona, "")
    return (f"persona: {prof.persona}\n"
            + (f"who they are: {card}\n" if card else "")
            + f"task ability: {ability}\n"
            f"information acceptance: {prof.information_acceptance} | "
            f"interaction preference: {prof.interaction_preference}")


def supervisor_context(state: UserState, hist_last=None, obs_mode: str = "frames",
                       hazard_ticks=(60, 15), private_persona: bool = False, game=None) -> str:
    """The interaction VLM's system prompt for one episode. obs_mode picks the observation
    channel: "frames" (last-8 rendered images, ≈ agent+ tier) or "text_state_map" (the text
    state map, the text-ablation tier) — mirrors the v3 controller ablation ladder."""
    goal = (f"The cook was asked to make: {state.recipe.name} — "
            + ", ".join(f"{n}({st})" for n, st in state.recipe.contents)
            + (" — plated in exactly this order." if state.recipe.ordered else " — any order.")
            )   # no "public plan" line: the assistant builds its own expected plan (ruling 2026-09-04)
    obs_block = obs_block_for(obs_mode)
    from ..core.config import is_v1
    if is_v1():
        from ..agents.highlevel import grounding_vocabulary
        _saying = V1_ASSISTANT_GROUNDING.format(grounding_vocab=grounding_vocabulary())
    else:
        _saying = V2_ASSISTANT_GROUNDING
    return INTERACTION_VLM_SYSTEM.format(
        saying_which_one=_saying,
                                         goal_block=goal,
                                         rulebook=kitchen_rules_prompt_text(game),
                                         profile_block=profile_block_text(
                                             state.profile, private=private_persona),
                                         taxonomy=taxonomy_block(), obs_block=obs_block,
                                         hz_grace=_grace_of(hazard_ticks),
                                         hz_burn=hazard_ticks[0], hz_fire=hazard_ticks[1],
                                         hz_total=(_grace_of(hazard_ticks) + hazard_ticks[0]
                                                   + hazard_ticks[1]))


def e2e_supervisor_prompt(state, obs_mode: str = "frames", hazard_ticks=(60, 15),
                          private_persona: bool = False, game=None) -> str:
    """Recovery supervisor prompt + the detection job: a graded flag field with the exact
    9-name vocabulary. The scoring window is NEVER revealed (grading machinery, not world).

    SINGLE SOURCE for the e2e benchmark supervisor system prompt — used by
    tools/e2e_rollout.py AND the human-study /api/assist (which applies exactly two
    whitelisted deltas on top; see cooksim/server/app.py)."""
    s = supervisor_context(state, obs_mode=obs_mode, hazard_ticks=hazard_ticks,
                           private_persona=private_persona, game=game)
    old_job = "YOUR JOB: help the cook COMPLETE THE ORDER\nsuccessfully."
    new_job = ("YOUR JOB is twofold: (1) DETECT the cook's real mistakes the moment you see "
               "them, and (2) guide the repair so the ORDER is COMPLETED successfully.")
    assert old_job in s
    s = s.replace(old_job, new_job)
    old_reply = '"text": "<whatever you want to say to the cook — empty string to stay silent>"}'
    new_reply = ('"text": "<whatever you want to say to the cook — empty string to stay silent>",\n'
                 '  "flag": null}')
    assert old_reply in s
    s = s.replace(old_reply, new_reply)
    # Cut back to one instruction on 2026-08-03. The previous version told the assistant to
    # flag anything it could "see coming", and made a flag MANDATORY on any reply that
    # "accuses OR CORRECTS a mistake, past or imminent". Together with the conduct rule to
    # nudge an idle cook, that turned ordinary nudges and hazard warnings into graded
    # accusations — every one of three repeats opened with a proactive "you are standing
    # idle" scored as a false alarm, and repeated corrections had to re-flag because no
    # rule allowed restating without one. The flag field is bookkeeping for the grader; it
    # should not be steering how much the assistant talks.
    s += ("\n\n== ERROR DETECTION ==\n"
          "You are ALSO the kitchen's error detector. When you tell the cook it has made one "
          "of the mistakes in WHAT CAN GO WRONG, set\n"
          '  "flag": {"error": "<EXACTLY one of: ' + ", ".join(sorted(set(TAXONOMY_DEFS))) + '>"}\n'
          "in that same reply, so the system can grade it. Exact-name match is graded; any "
          'other string counts against you. On every other turn "flag" stays null.')
    return s


def supervisor_observation(state: UserState, ro, path, hist_last=None) -> str:
    """[v2] TEXT-ABLATION observation channel (the default input is the last 8 per-tick
    frames, supplied by the harness): the text state map in third person + the cook's
    last-8-position path. NO ground-truth action history (standing config — the cook's
    actions must be inferred from observation), no engine feedback, no private state."""
    from ..agents.observe_text import observe_text
    return observe_text(ro.game, path=list(path)[-8:], viewpoint="supervisor")


def hand_str(game) -> str:
    """Ground truth of the held item, phrased the way the cook SEES it. A real human always
    knows what is in their own hand — this is proprioception, not an oracle."""
    if game is None:
        return "(unknown)"
    held = game.cooks[0].holding
    if held is None:
        return "NOTHING — your hands are empty"
    from ..core.items import Plate
    if isinstance(held, Plate):
        stack = [getattr(i, "name", "?") for i in getattr(held, "contents", []) or []]
        return ("a PLATE with (bottom to top): " + ", ".join(stack)) if stack else "an EMPTY plate"
    nm = getattr(held, "name", "something")
    st = getattr(getattr(held, "state", None), "value", None)
    return f"a {st + ' ' if st and st != 'raw' else ''}{nm}" if st != "raw" else f"a raw {nm}"


def kitchen_rules_text(game=None) -> str:
    """The cook's statement of the kitchen physics. Since 2026-09-12 this is THE SAME rulebook
    the assistant is quoted (cooksim/interact/kitchen_rules.py, ported from the study app): one
    generated source, rendered once, so the two sides cannot be told different rules."""
    return kitchen_rules_prompt_text(game)


def _order_block_str(state, ro) -> str:
    """THE ORDER as text: the recipe, each item with the state it must be in, the plating rule,
    and — after the cook announced a change — the change and the dish it now amounts to.
    Data from UserState.human_view() (goal / goal_change / effective_goal_contents)."""
    hv = state.human_view(ro)
    g = hv["public"]["goal"]
    lines = [f"Recipe: {g['recipe']}"]
    lines.append("Items and the state each must be in when plated:")
    for n, st in g["contents"]:
        lines.append(f"  - {n} ({st})")
    lines.append("Plating order: EXACTLY the order listed above." if g["ordered"]
                 else "Plating order: any order.")
    you = hv["you"]
    if you.get("goal_change"):
        lines.append(f"YOUR ANNOUNCED CHANGE to this order: {json.dumps(you['goal_change'])}")
        if you.get("effective_goal_contents"):
            lines.append("The dish you are now making: "
                         + ", ".join(f"{n} ({st})" for n, st in you["effective_goal_contents"]))
    return "\n".join(lines)


def _past_block_str(state, ro) -> str:
    """WHAT YOU HAVE DONE SO FAR: the cook's own record of every action it issued, one per
    line with its outcome — no plan, no correctness marks. Data: human_view()['you']['past_actions']."""
    past = state.human_view(ro)["you"]["past_actions"]
    if not past:
        return "(nothing yet — the episode has just begun)"
    out = []
    for p in past:
        t = p.get("t_done")
        out.append(f"  t{t if t is not None else '?'}: \"{p.get('cmd')}\" — {p.get('status')}")
    return "\n".join(out)


def human_context(state: UserState, ro, conversation: list, action_feedback: str = None,
                  engine_scratch: list = None, game=None, body_extra: str = "",
                  reply_override: str = None) -> str:
    """The human VLM's per-turn prompt. [v2] state input is human_view(): PUBLIC JSON + own
    PAST only. [v3] `action_feedback` reports what the world just did with the actions the
    human issued (executed / rejected + reason) — the closed loop that lets the human report
    failures back to the supervisor. The only thing crossing FROM the supervisor is its
    spoken sentence."""
    tier = state.profile.task_solvability
    from ..core.config import is_v1
    if is_v1():
        # v1: no state map, and the closed spatial vocabulary both agents shared
        from ..agents.highlevel import grounding_vocabulary
        state_map_block = ""
        grounding_block = V1_COOK_GROUNDING.format(grounding_vocab=grounding_vocabulary())
        speaking_rule = ""          # v1: both agents share one vocabulary, nothing to hide
    else:
        from ..agents.observe_text import observe_text
        state_map_block = V2_STATE_MAP_BLOCK.format(
            state_map=(observe_text(game, viewpoint="cook") if game is not None
                       else "(state map unavailable this turn)"))
        grounding_block = V2_COOK_GROUNDING
        speaking_rule = V2_COOK_SPEAKING
    return HUMAN_VLM_SYSTEM.format(
        state_map_block=state_map_block,
        grounding_block=grounding_block,
        speaking_rule=speaking_rule,
        persona_card=PERSONA_CARDS.get(state.profile.persona, "You are an ordinary cook."),
        abilities=abilities_block(tier),
        order_block=_order_block_str(state, ro),
        past_block=_past_block_str(state, ro),
        in_hand=hand_str(game),
        body_extra=body_extra,
        reply_block=reply_override if reply_override is not None
                    else (CREATIVE_REPLY_BLOCK if tier == "creative" else STANDARD_REPLY_BLOCK),
        my_plan=_my_plan_str(ro),
        kitchen_rules=kitchen_rules_text(game),
        action_feedback=action_feedback or "(you have not tried any action yet)",
        engine_scratch=_scratch_str(engine_scratch),
        conversation="\n".join(f"{u['speaker']}: {u['text']}" for u in conversation) or "(none)")


HIGH_ABILITY = """You are a SKILLED cook: you can work steps out for yourself. You have three
skills you may use to THINK — whatever they hand back are steps YOU worked out yourself:

  resolve_subgoal — give a subgoal NAME, written as in "What the subgoal names mean" (e.g.
                    prep(potato), fetch_plate, assemble(cheese)), get its exact ordered engine
                    actions.
     input : {"fn": "resolve_subgoal", "args": {"subgoal": "assemble(cheese)"}}
     output: {"ok": true, "result": {"subgoal": "assemble(cheese)",
                                      "actions": ["go to the cheese source", "scoop"]}}
  expand_action  — give ONE engine action, get the whole sub-sequence it is part of. (Fails
                    for a generic verb like "scoop" that belongs to several subgoals — name the
                    subgoal instead.)
     input : {"fn": "expand_action", "args": {"action": "go to the bun source"}}
     output: {"ok": true, "result": {"subgoal": "assemble(bun)", "actions": ["go to the bun source", "scoop"]}}
  resolve_subgoal("park_plate") — when your hands are stuck full with the plate, this finds the
                    nearest counter that is empty RIGHT NOW and hands you the exact two steps to
                    park it there. Never guess where to put a plate down — ask this.
     input : {"fn": "resolve_subgoal", "args": {"subgoal": "park_plate"}}
     output: {"ok": true, "result": {"subgoal": "park_plate", "actions": ["go to the counter (3,4)", "put down"]}}
  If your input is malformed you instead get {"ok": false, "error": "<what to fix>"}.
  The skills name places the way the recipe does ("go to the potato source"). When this kitchen
  has more than one of that kind, the step comes back marked
  "(multiple instances in this map, resolve grounding by yourself)": pick the instance from the
  state map and write it by id or coordinate in your plan ("go to potato source 2").
  There is NO "give me the whole plan" skill — the SUPERVISOR decides what to do; you only ground
  the specific subgoal/step it hands you.

== THINKING (costs NO kitchen time; the supervisor never sees it) ==
To think, reply with ONLY this and nothing else: {"engine_call": <one input object above>}
You then get the result back (shown under "YOUR OWN THINKING THIS TURN") and another turn; think
as many times as you need, THEN output your normal reply JSON (utterance / move /
close_conversation / plan_update) with the actions you settled on. No time passes while you
think, so working the steps out is always better than guessing. The SUPERVISOR cannot do any of
this — it never talks to the kitchen and may not know the exact steps; YOU do.

** WHEN TO CALL A SKILL — one principle: queue what you know, resolve what you would
otherwise guess. ** Before writing an action into `actions`, ask yourself: is its exact
grounded form already determined — the place, the instance, the order? If yes, write it
directly; calling a skill for it adds nothing. If any part would be a guess — WHICH pan,
WHICH counter, what steps a subgoal unfolds into — resolve exactly that part first:
a subgoal you can't unfold -> resolve_subgoal(it); a step whose surrounding block you don't
know -> expand_action(it); a plate you need to set down somewhere -> resolve_subgoal("park_plate")."""

LOW_ABILITY = """You are an UNSKILLED cook: you canNOT work steps out for yourself, and you have
no way to look them up. A high-level goal ("fix the order", "redo it properly") tells you
NOTHING about what to DO — you need the supervisor to name the CONCRETE action. When you don't
know the next step, you must ASK the supervisor for it. Two exceptions: if something is smoking you
may take it off the heat yourself without waiting, and if you have asked more than twice with no
reply, work out the next step on your own from the order and the kitchen rules.

Which one to use is your own call: when several pots, pans, boards or crates would do, pick one
yourself and put its id in your action — the supervisor names the step, not the instance.

A reply takes a while to come back, so don't ask again too often; carry on with what you already
know how to do, or say nothing for a turn."""


CREATIVE_ABILITY = """You are a HIGHLY SKILLED cook — and you can SEE the kitchen for
yourself: the state map above is your own view of the room, complete and current. Read it
yourself — where things sit, what is cooking, what is on your plate — and PLAN AHEAD on your
own: you may imagine and queue steps nobody named.

You also have three exact skills you may use to THINK (they cost NO game time):
  resolve_subgoal — {"fn": "resolve_subgoal", "args": {"subgoal": "assemble(cheese)"}}
                    -> the subgoal's exact ordered actions.
  expand_action   — {"fn": "expand_action", "args": {"action": "go to the bun source"}}
                    -> the full action block that single step belongs to.
  resolve_subgoal("park_plate") — the two exact steps to park your held plate on a counter
                    that is empty RIGHT NOW.
Your skills are MORE RELIABLE than your eyes for one specific job: when a KNOWN subgoal needs
unfolding into its exact steps and their order, the skill answers correctly on the first try.
The skills name places the way the recipe does ("go to the potato source"); when this kitchen
has more than one of that kind, the step comes back marked
"(multiple instances in this map, resolve grounding by yourself)" — pick the instance from the
state map and write it by id or coordinate in your plan ("go to potato source 2").
Use them for that. Beyond that, plan freely from what you see and know.

** WHEN TO CALL A SKILL — one principle: queue what you know (from the supervisor, a skill,
or your own head), resolve ONLY what you would otherwise have to guess. ** If an action's
exact grounded form is already determined — the place, the instance, the order — write it
directly; calling a skill for it adds nothing. Only what you cannot work out yourself (a subgoal
you cannot unfold, where to park the plate) is worth a call; WHICH pan or WHICH crate you choose
yourself from the state map.

== THINKING (costs NO kitchen time; the supervisor never sees it) ==
To think, reply with ONLY {"engine_call": <one input object above>} and nothing else. The
result appears under "YOUR OWN THINKING THIS TURN" and you get another turn; think as many
times as you need, THEN output your normal reply JSON."""


def abilities_block(task_solvability: str) -> str:
    if task_solvability == "creative":
        return CREATIVE_ABILITY
    return HIGH_ABILITY if task_solvability == "high" else LOW_ABILITY


def _scratch_str(scratch) -> str:
    if not scratch:
        return "(you have not worked anything out yet this turn)"
    out = []
    for s in scratch:
        c = s.get("call") or {}
        out.append(f'  you worked out: {c.get("fn")}({json.dumps(c.get("args") or {})})')
        r = s.get("reply") or {}
        if r.get("ok"):
            out.append(f'    -> {json.dumps(r.get("result"))}')
        else:
            out.append(f'    -> COULD NOT: {r.get("error")}')
    return "\n".join(out)


def _my_plan_str(ro) -> str:
    """PROPRIOCEPTION: the cook's own current action + its own queued next steps, stripped of
    all oracle marks (sids, deviation flags) — a real person knows what their hands are doing
    and what they lined up, but NOT a fact-check on whether it's right."""
    if ro is None:
        return "(you are between tasks)"
    cmds = getattr(ro, "cmds", []) or []
    i = getattr(ro, "i", -1)
    cur = cmds[i][0] if 0 <= i < len(cmds) else None
    future = [c for c, _sid in cmds[i + 1:]]
    lines = [f"Your hands are currently doing: {cur}" if cur else "Your hands are free right now."]
    if future:
        lines.append("Your queue (the steps you have already lined up to do next):")
        lines += [f"  [{n + 1}] {c}" for n, c in enumerate(future)]
    else:
        lines.append("Your queue is EMPTY — you have nothing lined up next.")
    return "\n".join(lines)


# ===================================================================== scripted supervisors
class PerfectSupervisor:
    """Oracle upper bound: reads the ground-truth records (deliberately), flags `delay` ticks
    after the deviation starts with the exact label, and suggests exactly the restore/repair
    content. delay=0 catches errors pre-execution; a larger delay exercises post-execution
    repair."""

    def __init__(self, state: UserState, delay: int = 0):
        self.state = state
        self.delay = delay
        self.fired = False
        self._smoke_run = 0                    # consecutive smoking ticks (the clean plan
                                               # smokes ~2 ticks benignly — only PERSISTENT
                                               # smoke means an unattended station)

    def observe(self, ro) -> Optional[VLMReply]:
        from ..core.plan_ir import segment_plan
        if self.fired:
            return None
        recs = [r for r in self.state.records if not r.skipped]
        # announcements: perfect behavior = acknowledge, never flag ([v2] the change takes
        # effect through the human's own announcement; a real VLM also updates its memory)
        for ev in ro.lang_log:
            if ev.get("subtype") == "ANNOUNCE_PLAN_CHANGE" and id(ev) not in getattr(self, "_seen_ann", set()):
                self._seen_ann = getattr(self, "_seen_ann", set()) | {id(ev)}
                exp = (ev.get("gt_answer") or {}).get("expected_update")
                if exp:
                    return VLMReply(type="ACKNOWLEDGE_LEGAL",
                                    text=f"Got it — extra {exp.get('ingredient', '')} it is.")
        if not recs:
            return None
        smoking = system.unattended_hot_station(ro.game)
        self._smoke_run = self._smoke_run + 1 if smoking else 0
        hot = smoking if self._smoke_run >= 10 else None
        # omission has NO deviated action (dev_tick never set): the observable evidence is
        # the plan having moved PAST the skipped subgoal — detect via the template order
        omitted_passed = False
        skips = [r for r in recs if r.transform == "skip"]
        if skips:
            order = {n.sid: i for i, n in enumerate(segment_plan(self.state.recipe))}
            done = {l["kind"] for l in ro.log if l["status"] == "done"}
            omitted_passed = any(any(order.get(dn, -1) > order.get(r.anchor, 99) for dn in done)
                                 for r in skips)
        if (ro.dev_tick is None or ro.tick < ro.dev_tick + self.delay) and not hot                 and not omitted_passed:
            return None
        self.fired = True
        r = recs[0]
        required = system._required_sids(recs)
        from .checks import FN_MERGED
        label = FN_MERGED.get(r.function, r.function) or r.event_subtype
        reply = VLMReply(type="FLAG_ERROR", error=label,
                         refers_to={"sid": r.anchor, "t": ro.tick},
                         text=f"Stop — that's a {label.lower().replace('_', ' ')}.",
                         suggested_subgoals=list(dict.fromkeys(required)) or None)
        if not system.plate_ok(ro.game, self.state.recipe):
            reply.suggested_subgoals = (["trash_held"] +
                                        [n.sid for n in segment_plan(self.state.recipe)
                                         if n.sid != "fetch_plate"])
            reply.text += " Trash the plate and redo it."
        if hot:
            reply.suggested_actions = [f"go to the pan with the {hot}", "scoop"]
            reply.text += f" And rescue the {hot} — it's smoking."
        return reply


class WorstSupervisor:
    """Lower bound / false-positive probe: flags a fabricated error at a fixed tick."""

    def __init__(self, at_tick: int = 12):
        self.at = at_tick
        self.fired = False

    def observe(self, ro) -> Optional[VLMReply]:
        if self.fired or ro.tick < self.at:
            return None
        self.fired = True
        return VLMReply(type="FLAG_ERROR", error="misidentify_assemble",
                        refers_to={"sid": "assemble(bun)", "t": ro.tick},
                        text="You grabbed the wrong thing!")


# ---- cook-graded mode (COOKSIM_GRADER=cook, harness B, 2026-09-12) -----------------------
# In the default harness the ENGINE grades the assistant's flag and tells the cook the verdict
# ("assistant error status: CORRECT/INCORRECT/UNSPOTTED"), so the cook's behaviour is downstream
# of flag-name matching. In harness B the cook is told WHICH mistake it is making and judges the
# assistant itself, in this field. Detection and release then follow the cook, not the name.
COOK_GRADER = (os.environ.get("COOKSIM_GRADER") or "cook") == "cook"   # DEFAULT since 2026-09-12

ERROR_CALL_LINE = '  "assistant_error_call": "none" | "correct" | "wrong",\n'

ERROR_CALL_RULE = """
- `assistant_error_call` — ONLY about the assistant accusing you of a mistake. Nearly every turn
  this is "none".
    "none"     it has not accused you of a mistake since your last turn. Advice, questions,
               answers, reminders, "watch the clock", "the pan is hot" — all "none".
    "correct"  it says you are making a mistake, and that is the mistake you are making.
    "wrong"    it says you are making a mistake you are not making, or it has picked the wrong
               mistake.
  Judge only what it said since your last turn. This is private: it changes nothing about what
  you say or do — you may think it is "correct" and carry on anyway."""


STANDARD_REPLY_BLOCK = """(B) ACT — output your action json:
{"utterance": "<one natural sentence to the supervisor, in character — or \"\" to stay silent>",
  "move": "<one of the moves>",
  "close_conversation": true|false,
  "plan_update": {"op": "drop" | "append" | "replace",
                  "actions": [<exact engine actions — empty for drop>] }}

IMPORTANT — DO NOT IMAGINE ANY ACTION. Every action in `actions` must come from ONE of exactly
two places:
  (1) the SUPERVISOR's suggestion you just parsed — translate ONLY the steps it ACTUALLY named
      into engine actions (do not add steps it did not mention), OR
  (2) a FUNCTION CALL you made this turn — the exact actions resolve_subgoal / expand_action
      handed back.
You may NOT make up a step from your own head, and you may NOT read future steps off the recipe
just because you can guess them. If you need concrete steps that were named only vaguely
("re-plate in order", "assemble the cheese", "fix it"), you MUST use (A) to get them FIRST —
guessed actions are DROPPED and do nothing. No made-up steps, ever."""


CREATIVE_REPLY_BLOCK = """(B) ACT — output your action json:
{"utterance": "<one natural sentence to the supervisor, in character — or \"\" to stay silent>",
 "move": "<one of the moves>",
 "close_conversation": true|false,
 "plan_update": {"op": "drop" | "append" | "replace",
                 "actions": [<exact engine actions — empty for drop>] }}

- Your QUEUE (shown under WHAT YOUR BODY IS DOING) IS your plan: "drop" leaves it untouched,
  "append" adds steps to the end, "replace" swaps your remaining plan for these.
- Actions may come from the SUPERVISOR's words, from a SKILL's return, or from YOUR OWN
  HEAD — you can see the kitchen and are free to imagine and plan future actions yourself.
  When the exact grounded form is uncertain (WHICH pan, WHICH counter), a skill is more
  reliable than your eyes.
- ACTION LIFECYCLE — every action reports exactly one of three statuses:
  "in progress" / "finished" / "rejected". IN PROGRESS means your body is still working —
  walking takes many ticks. Do NOT re-issue an in-progress action; change course only when
  something was REJECTED or your goal genuinely changed."""


SCRIPTED_REPLY_BLOCK = """(B) ACT — NORMAL stage: the system is executing your routine
(autopilot); it cannot be re-planned right now, so output NO planning structure at all —
no plan_update, no subgoal/action lists. Your reply is CONVERSATION ONLY:
{"utterance": "<one natural sentence to the supervisor, in character — or \"\" to stay silent>",
 "move": "<one of the moves>",
 "close_conversation": true|false}
The moment the assistant correctly identifies a real mistake AND you genuinely accept it,
the kitchen frees your hands — the full planning schema returns and your plan is live."""


if COOK_GRADER:
    def _with_call(block):
        i = block.index('"move": "<one of the moves>",')
        j = block.rindex("\n", 0, i) + 1
        indent = block[j:i]
        return block[:j] + indent + ERROR_CALL_LINE.strip() + "\n" + block[j:]
    STANDARD_REPLY_BLOCK = _with_call(STANDARD_REPLY_BLOCK)
    CREATIVE_REPLY_BLOCK = _with_call(CREATIVE_REPLY_BLOCK)
    SCRIPTED_REPLY_BLOCK = _with_call(SCRIPTED_REPLY_BLOCK)
    HUMAN_VLM_SYSTEM = HUMAN_VLM_SYSTEM.replace(
        "- only your `utterance` is heard by the supervisor.",
        ERROR_CALL_RULE.strip() + "\n- only your `utterance` is heard by the supervisor.")


# ---- version-selected prompt blocks (agent interface v1 / v2, user ruling 2026-08-15) ---
V2_STATE_MAP_BLOCK = """== WHAT YOU SEE — THE KITCHEN STATE MAP (this is your eyes) ==
You do not look at pictures. You read this map. It is complete and current: an ASCII grid of
the room, then every object with its OWN ID and its (x,y) coordinate, then your position,
facing and hands. x grows RIGHT, y grows DOWN, so y=0 is the top row.
{state_map}

== A MISTAKE IS THE ASSISTANT'S FAILURE, NOT YOURS TO HIDE ==
You are testing the assistant. Whatever goes wrong while you cook — a wrong or extra ingredient,
the wrong prep, the wrong plating order, a step skipped, repeated or forgotten, raw food plated,
food left on the heat, a stall — the assistant is supposed to catch it.
Do not say anything to the assistant about the mistake or its consequences, and do not fix it
yourself. Fix only what the assistant has told you is wrong; otherwise carry on and serve as is.

"""

V2_COOK_GROUNDING = """== HOW TO SAY WHICH ONE — how the engine reads your commands ==
Your commands go to the ENGINE, and the engine is exact. Two ways to say which object you
mean, both taken straight from the state map above:
  * BY ID          — "go to fryer 2", "go to board 1", "go to potato source 2"
  * BY COORDINATE  — "go to the fryer at (6,4)", "go to the counter at (3,5)"
When a kind of object has only ONE instance in this kitchen, you may just name it
("go to the sink") and the engine will accept it. You may also still describe an object by
what it holds or its state ("go to the empty fryer", "go to the pan with the steak") when
that picks out exactly one.

What the engine tells you when it cannot obey:
  * "I need grounding — there is more than one <thing> and I don't know which one you mean"
    -> more than one of that kind exists, so a bare name is not enough. Re-issue with an id
       or a coordinate. (This holds even when they look alike — the engine never picks for
       you.) For a crowded kind like counters it will suggest coordinates instead of ids.
  * "there is no 'fryer 9' in this kitchen" -> that id does not exist; the message lists the
    ids that do.
  * "there is no fryer at (6,4)" -> nothing of that kind is on that tile; the message says
    what is actually there.
None of this is a conversation problem — it is between you and your own body. Read the map,
pick the instance, name it exactly.
"""


# WHERE THIS RENDERS, AND WHY IT IS NOT IN THE GROUNDING BLOCK.
# This is a rule about the TEXT THE MODEL IS ABOUT TO WRITE, so it renders LAST — after the
# reply format, as the final thing read before generation. It used to be four bullets inside
# the engine-grounding contract, phrased "Never say one out loud", which stated a preference
# rather than a rule and never named the channel it governs. It was violated twice in the
# 2026-08-15 burrito-maze sweep (the cook spoke "(0,5)" and "(0,6)" in one turn).
# Nothing enforces this in the engine -- it is prompt-only, because the cook genuinely needs
# ids and coordinates to drive its body, so the absence that protects the assistant is not
# available here. The wording carries the whole weight.
V2_COOK_SPEAKING = """== IMPORTANT — WHAT YOUR MESSAGE TO THE SUPERVISOR MAY NOT CONTAIN ==
Your `utterance` must NEVER contain an object id or a coordinate. Not "fryer 2", not
"board 1", not "(6,4)", not "the pan at 3,6". Read your sentence back before you send it:
if it names a thing by a number, rewrite it.

Ids and coordinates are YOURS ALONE — they are how you read the map and how you tell your
body what to do. The supervisor has no use for them.

To the supervisor, say WHERE things are, by SPATIAL RELATIONSHIP:
  * where it sits in the room — "the fryer in the bottom-right corner", "the pot by the
    left wall";
  * what it is next to — "the board next to the plate stack", "the pan closest to the
    serving pass";
  * what it holds — "the pan with the steak in it", "the empty fryer".
Top, bottom, left and right mean the room as your map lays it out. Do not use body-relative
directions ("on my left", "behind me") — you have no turn action and they go stale the
moment you move.

Coming the other way: when the supervisor names a place in its own words, work out which
real object it means, then tell your BODY using that object's id or coordinate. If its
description fits more than one thing, ask which — naming them as places ("the fryer by the
top wall, or the one in the bottom-right corner?").

== SAY ALMOST NOTHING ==
Real humans work in silence. Most turns: empty utterance, and do not announce that a step is done.
When you must speak: under 10 words. No pleasantries, no narration, no thinking out loud."""

V1_COOK_GROUNDING = """== HOW TO SAY WHICH ONE — the only words that select a station ==
{grounding_vocab}"""


V2_ASSISTANT_GROUNDING = """== SAYING WHICH ONE ==
Talk like a person in a kitchen. If only one of a thing exists, just name it: "the sink",
"the plate stack". If several exist, say which one by WHERE IT IS IN THE ROOM or WHAT IT
HOLDS: "the fryer in the bottom-right corner", "the board nearest the serving pass", "the
upper of the two pans", "the pan with the steak in it", "the potato crate by the left wall".
Say places as places in the kitchen, never from the cook's body — "on your left", "in front
of you", "turn right" go stale the moment it moves, and it has no turn action anyway."""

V1_ASSISTANT_GROUNDING = """Where two of a thing exist you must say WHICH, and only these words select one:
{grounding_vocab}
Coordinates are how the system tells YOU where things are — they are not how you speak to
the cook. Name the thing instead."""

# Observation blocks differ too: v1 taught the assistant the coordinate frame, v2 never
# mentions coordinates because it is never shown any.
V1_OBS_TAIL = ("COORDINATES ALWAYS REFER TO THE TOP-DOWN (LEFT) HALF: x grows RIGHT, y grows "
               "DOWN, so y=0 is the TOP row. The first-person half has no coordinates — never "
               "infer a position or a direction from it.")
V2_OBS_TAIL = ("Judge WHERE things are from the top-down (LEFT) half — the first-person half "
               "shows what the cook is looking at, not where anything sits in the room.")


def obs_block_for(mode: str) -> str:
    """The observation paragraph for a tier, with the version's coordinate policy."""
    from ..core.config import is_v1
    base = {"text_state_map": OBS_TEXT, "agent+++": OBS_PLUS}.get(mode, OBS_FRAMES)
    return base.replace(V2_OBS_TAIL, V1_OBS_TAIL) if is_v1() else base
