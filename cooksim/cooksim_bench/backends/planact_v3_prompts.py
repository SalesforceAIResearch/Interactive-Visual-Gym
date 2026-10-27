#!/usr/bin/env python3
"""VLM-as-controller prompt suite v3 — the SUBGOAL-STRUCTURED streaming controller.

Standing observation config: decision every 2 ticks; last 8 execution frames (or the text
state map with the last-8-tick path); current timestamp provided every turn.

Every mode now uses the same STRUCTURED OUTPUT: the model maintains its own plan as
completed/current/future SUBGOALS, and completed/current/future ACTIONS inside the current
subgoal, with per-turn self-evaluation of its last action (finished / rejected / ongoing)
and timestamps on everything completed. Modes differ ONLY in the input/feedback channel.

    from cooksim_bench.backends.planact_v3_prompts import build_prompt, MODES
    sys_prompt = build_prompt("agent++")
"""

MODES = ["basic", "agent", "agent+", "agent++", "agent+++", "text_state_map"]

# ---------------------------------------------------------------- per-turn message
# THE STRUCTURE IS THE RECURRENT STATE: every call is stateless; the model's previous output
# JSON is fed back verbatim as input at t+1 and is its ONLY memory of the episode. We do NOT
# replay conversation history or past observations — memory is exactly what the model wrote,
# so its bookkeeping errors persist, compound, and are measurable against engine ground truth.
PER_TURN_TEMPLATE = """{recipe}

CURRENT TIMESTAMP: t={t}

== YOUR PREVIOUS OUTPUT (your plan state — judge its current_action, then update it) ==
{prev_json}

{mode_extras}{observation_note}
Output your updated JSON for NOW."""

FIRST_TURN_TEMPLATE = """{recipe}

CURRENT TIMESTAMP: t={t}

This is your FIRST turn: no previous output exists. Reason out the complete subgoal plan for
the recipe, then output the full JSON with subgoals_future populated, the first subgoal
promoted to current_subgoal, and its first action issued in "action".

{mode_extras}{observation_note}
Output your JSON for NOW."""

OUTPUT_SCHEMA = '''{
  "subgoals_completed": [
      {"subgoal": "prep(meat)", "completed_at": 18}
  ],
  "current_subgoal": {
      "subgoal": "assemble(bun)",
      "actions_completed": [
          {"action": "go to the bun source", "status": "finished", "at": 44},
          {"action": "scoop",                "status": "rejected", "at": 45}
      ],
      "current_action": "scoop",
      "actions_future": []
  },
  "subgoals_future": ["assemble(meat)", "assemble(cheese)", "serve"],
  "action": "<the command to issue NOW, or \\"\\" to keep executing the current one>",
  "reason": "<one short sentence>"
}'''

IN_CONTEXT_EXAMPLE = '''EXAMPLE (mid-episode; you are told CURRENT TIMESTAMP: t=46. Your PREVIOUS
output had current_action = "scoop" under assemble(bun); the new observation shows the bun now
sits on your plate — so that scoop is judged finished and recorded, the subgoal is done, and
the next subgoal is promoted):
{
  "subgoals_completed": [
      {"subgoal": "prep(meat)", "completed_at": 18},
      {"subgoal": "fetch_plate", "completed_at": 33},
      {"subgoal": "assemble(bun)", "completed_at": 46}
  ],
  "current_subgoal": {
      "subgoal": "assemble(meat)",
      "actions_completed": [],
      "current_action": "go to the pan with the meat",
      "actions_future": ["scoop"]
  },
  "subgoals_future": ["assemble(cheese)", "serve"],
  "action": "go to the pan with the meat",
  "reason": "bun landed on the plate at t=46; next component is the cooked meat in the far pan"
}
(Notes: the finished scoop was recorded inside assemble(bun) before the subgoal moved to
subgoals_completed. Had the scoop been REFUSED instead, it would be appended to
actions_completed as {"action": "scoop", "status": "rejected", "at": 46}, stay as
current_action, and "action" would re-issue it after fixing the cause. While still WALKING to
the pan next turn, current_action stays "go to the pan with the meat" and "action" is "".)'''

CORE = '''You are a STREAMING controller for one cook in a top-down cooking game. A decision is
requested every 2 ticks; you are told the CURRENT TIMESTAMP (t) every turn.

{input_block}

You maintain your own PLAN as a SUBGOAL STRUCTURE and act through it.
A SUBGOAL is a recipe-level block, named: prep(<ingredient>) (fetch it, put it in its station,
start it), fetch_plate, assemble(<ingredient>) (go to where it is, scoop it onto your plate),
serve. Each subgoal contains low-level ACTIONS from the command list below.

ON YOUR FIRST TURN: reason out the COMPLETE plan for the recipe first — write every subgoal in
order into "subgoals_future", promote the first to "current_subgoal" with its action list, and
issue its first action.

ON EVERY LATER TURN, first JUDGE the "current_action" from your PREVIOUS output against the
new observation{fb_eval}, then rewrite the whole structure to match, then act:
- current_action FINISHED (you can see its effect: you arrived; the item is in hand; food is on
  the plate) -> append it to "actions_completed" as {{"action": ..., "status": "finished",
  "at": <current timestamp>}} and promote the next future action to current_action. When it was
  the LAST action of the subgoal, the whole subgoal moves to "subgoals_completed" with
  "completed_at" = the current timestamp, and the next future SUBGOAL is promoted to current.
- current_action REJECTED (nothing happened — refused, or impossible where you stood) -> append
  it as {{"action": ..., "status": "rejected", "at": <current timestamp>}}; keep it (or a fixed
  replacement) as current_action and address the cause before re-issuing.
- current_action ONGOING (still walking there / food still cooking) -> leave the structure
  unchanged and set "action" to "".
If the world surprised you (wrong item on the plate, something burning), REPLAN: rewrite the
current and future subgoals to recover — never leave the structure contradicting what you see.
Finally set "action": the command to issue NOW, or "" (empty) to keep executing.

OUTPUT strict JSON, exactly this schema:
{schema}

{example}

COMMANDS: "go to <object>" (an ingredient source, a pot/pan/oven/cutting board/sink, "the
plates", "the serving pass", "the trash"; when several instances exist QUALIFY: "the pan with
the meat", "the empty pan", or coordinates "(x,y)") | "pick up" | "put down" | "chop" |
"turn on" | "scoop" (hold a plate; adds the faced station's/source's food onto it) | "serve".

RULES — CRITICAL:
- ADD IN ORDER: an ORDERED recipe is accepted ONLY if components are plated in the exact listed
  order; right ingredients in the wrong order are REJECTED.
- ILLEGAL ACTIONS DO NOTHING: a manipulation works only while standing at and FACING the right
  tile; "pick up" with full hands, "scoop" without a clean plate, "turn on" an empty station all
  waste the turn. Verify preconditions before acting.
{fb_block}'''

# ------------------------------------------------------------------ per-mode channels
_FRAMES = ("INPUT: the last 8 rendered frames of the kitchen, one per tick, oldest -> newest — "
           "read everything from the pixels: where you are, which way you face, what you hold "
           "(a white plate; food visibly stacks on it), what is inside each pot/pan/board, "
           "whether food is cooking or done (it darkens), smoke, fire. The frame sequence shows "
           "your MOTION — use it to judge whether you have arrived.")

_TEXTMAP = ("INPUT: the CURRENT KITCHEN STATE AS TEXT — an ASCII map (@ is you), tables of every "
            "station/source with contents and cooking progress, what you are holding, and YOUR "
            "RECENT PATH (your last 8 positions, one per tick — use it to judge whether you are "
            "still moving and in which direction). Trust the tables over your memory: they are "
            "the ground truth of the current tick.")

MODE_NOTES = {
 "basic": dict(
   input_block=_FRAMES,
   fb_eval="",
   fb_block=("- NO FEEDBACK CHANNEL: the engine does NOT tell you whether a command was accepted "
             "— a bad command is SILENTLY refused and simply does nothing. Your step-1 "
             "self-evaluation from the frames is your ONLY way to notice a rejected action: if "
             "the frames show no effect where you expected one, treat it as rejected.")),
 "agent": dict(
   input_block=_FRAMES,
   fb_eval=" and the engine verdicts",
   fb_block=("- FEEDBACK (minimal): each of your recent commands is echoed back with ONLY "
             "\"accepted\" or \"rejected\" — no reason is given. Use it to set your step-1 "
             "verdict; work out the cause of a rejection yourself from the frames.")),
 "agent+": dict(
   input_block=_FRAMES,
   fb_eval=" and the engine feedback",
   fb_block=("- FEEDBACK (full): each of your recent commands is echoed back with the engine's "
             "verdict — \"command received and executing\" / \"continue executing <cmd>\" (still "
             "busy -> output \"\" and wait) / \"command refused because <reason>\" (fix the stated "
             "cause, then re-issue) / \"grounding needed: which <instance>?\" (re-issue NAMING the "
             "instance) / \"no command to execute\" (idle -> issue the next action). Act on it; "
             "never blindly resend a command that is still executing.")),
 "agent++": dict(
   input_block=_FRAMES + "\nYou are ALSO given your CURRENT POSITION and facing as text each turn.",
   fb_eval=" and the engine feedback",
   fb_block=None),   # same as agent+
 "agent+++": dict(
   input_block=_FRAMES + ("\nYou are ALSO given, as text each turn: your CURRENT POSITION and "
                          "facing, and your CURRENT PLATE contents in the exact order they were "
                          "added — trust this line over your memory of what you plated."),
   fb_eval=" and the engine feedback",
   fb_block=None),   # same as agent+
 "text_state_map": dict(
   input_block=_TEXTMAP,
   fb_eval=" and the engine feedback",
   fb_block=None),   # same as agent+
}


def build_prompt(mode: str) -> str:
    n = MODE_NOTES[mode]
    fb = n["fb_block"] if n["fb_block"] is not None else MODE_NOTES["agent+"]["fb_block"]
    return CORE.format(input_block=n["input_block"], fb_eval=n["fb_eval"],
                       schema=OUTPUT_SCHEMA, example=IN_CONTEXT_EXAMPLE, fb_block=fb)


def build_turn(mode: str, recipe: str, t: int, prev_json: str, mode_extras: str = "") -> str:
    """The user message for one decision. mode_extras = the feedback block text (agent/agent+),
    position line (agent++), plate line (agent+++) — already formatted by the harness."""
    obs = ("Below is the CURRENT KITCHEN STATE (text)." if mode == "text_state_map"
           else "Below are your last 8 frames (one per tick, oldest first).")
    tpl = FIRST_TURN_TEMPLATE if prev_json is None else PER_TURN_TEMPLATE
    return tpl.format(recipe=recipe, t=t, prev_json=prev_json or "",
                      mode_extras=(mode_extras + "\n\n") if mode_extras else "",
                      observation_note=obs)


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "basic"
    print(build_prompt(m))
