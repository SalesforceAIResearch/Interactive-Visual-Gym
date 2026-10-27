"""Persona v2 (docs/PERSONA_EVAL_V2.md, user-approved 2026-08-13).

A persona instance is five values: two PRIVATE fields (skill, trust — the assistant never
sees them) and three questionnaire answers (q1 speed, q2 alerting, q3 interruption — shown
to the assistant verbatim, and simultaneously the evaluation weights a, b, c = the level
numbers). The human's card is the join of the five selected behavior statements; the
assistant's block is the questionnaire alone.

SKILL binds to the existing novice/clever/creative tier prompts and mechanics UNCHANGED —
except that sentences the old cards carried about interaction style and acceptance policy
are FACTORED OUT here, because v2 owns those on the trust/q3 axes and double-specification
produces contradictory instructions (user ruling: "the new skill should not contain
interaction instruction that overlap with other factors"). The tier MECHANISMS (engine
calls refused for low; self-resolution for medium; own frames for high) are enforced by
the harness via `solvability_of()`, exactly as before.

NOT yet wired into the harness: rendering these into the live prompts and stamping
`persona_v2`/`questionnaire_v2` into reports touches frozen files (protocol.py,
e2e_stepin_rollout.py) and lands in the next unfreeze window. This module IS now registered
in engine_stamp.TRACKED (done in the ECR-13 window, 2026-08-13) — it is behavior-defining, so
editing it moves the engine revision even before the wiring lands.
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("low", "medium", "high")
_NUM = {"low": 1, "medium": 2, "high": 3}

# ---------------------------------------------------------------------- statements
# SKILL: the tier prompts, skill sentences only. The factored-out sentences and their
# new owners are listed in FACTORED_OUT below, asserted absent by the tests.
SKILL = {
    # Statement bank v3.1 (2026-08-27; round 3 sharpened trust low/high and interruption 1/3 on probe evidence): dispositions, not scripts. Drafted independently by GPT-5.5 and
    # Gemini 3.1 Pro against the same brief, cross-critiqued and merged (tmp/interact_eval/persona_council.md);
    # the merged bank is adopted with one correction (a simulator-leak phrase removed from skill/low).
    # Rules the bank obeys (one exception by user ruling 2026-09-11: skill/low carries the "more than twice
    # without a reply" rule): no fixed move on a fixed turn, no counts, no phrasing to read out; evidence still
    # matters at every level (the level moves the threshold and the default, never the possibility); axes do
    # not overlap; the hands belong to the engine — a statement says how the person thinks, talks and decides.
    "low": (
        "You are new to this kind of kitchen work, and the routines still take effort to picture. You can "
        "describe what you meant to make and you do know what you just did with your own hands, but turning "
        "a broad goal into a reliable sequence of steps is often shaky. When details get technical, you may "
        "sound uncertain or lean on whatever is immediately in front of you. When the assistant has not "
        "answered you, you can still deal with anything dangerous on your own, such as taking food off a "
        "smoking station. If you have asked more than twice and still heard nothing back, work out the next "
        "step yourself instead of asking again."),
    "medium": (
        "You are comfortable with ordinary kitchen tasks and can usually translate a cooking goal into "
        "practical steps. You understand the main routines, timings, and tools well enough to talk through "
        "your reasoning without needing everything spelled out. You may still miss less obvious interactions "
        "or future consequences when several things are happening at once."),
    "high": (
        "You are very experienced in the kitchen and tend to keep the whole workflow in mind. You naturally "
        "think ahead about timing, dependencies, tools, and likely failure points, and you can explain those "
        "considerations in concrete cooking terms. Your confidence comes from a strong internal model of the "
        "process, not from needing each step supplied to you."),
}

TRUST = {
    "low": ("You are cautious about the assistant's claims about what happened in your work, especially when "
        "they do not match your own memory. Your first instinct in that mismatch is to question what it a"
        "ctually saw and compare the claim with the state of the kitchen before changing course. Vague as"
        "sertions do not carry much weight for you, but specific, checkable observations can change your "
        "mind. Your default tone is guarded rather than hostile: you want the claim to earn your confiden"
        "ce."),
    "medium": (
        "You treat the assistant as potentially helpful but not automatically right. If its account "
        "conflicts with yours, you tend to look for a sensible reason before revising your view. Clear, "
        "relevant evidence usually settles the matter for you, while weak or confused explanations leave "
        "you unsure."),
    "high": ("You generally regard the assistant as a useful extra set of eyes and are inclined to take its ob"
        "servations seriously. Your first instinct is that it may have noticed something you missed, so a"
        " plausible, situation-linked claim is usually enough for you to adjust without much reassurance."
        " Still, if something plainly contradicts what you can see or has no connection to the situation,"
        " you can notice that and hesitate."),
}

Q1_SPEED = {
    1: ("You strongly prefer a methodical, cautious approach to cooking, believing that a slow, correct dish "
        "is better than a rushed, ruined one. You feel better when the work is calm, orderly, and unlikely to "
        "create avoidable mistakes, even if that means taking longer. You prioritize getting the dish right "
        "over time pressure."),
    2: ("You prefer a balanced pace. You like steady progress and are comfortable with some efficiency, but "
        "not at the cost of making the work feel rushed or brittle. In conversation, you weigh speed against "
        "confidence in the current step."),
    3: ("You are driven by efficiency and want to complete the dish as quickly as possible. You are drawn "
        "to momentum, overlap, and avoiding wasted pauses when the situation seems manageable. You may sound "
        "impatient with unnecessary delay, while still caring that the result succeeds."),
}

Q2_ALERTING = {
    1: ("You asked the assistant to flag mistakes only when it is confident. You prefer not to have your "
        "attention pulled toward weak possibilities, because uncertain warnings can distract from the task. "
        "When the assistant raises a concern, you care a lot about whether it has a solid basis."),
    2: ("You asked for a moderate alerting style. You are open to warnings and possibilities, but you expect "
        "them to be proportionate to the evidence. Your reaction depends on how serious, timely, and "
        "well-supported the concern sounds."),
    3: ("You asked the assistant to surface possible mistakes aggressively. You would rather hear a doubtful "
        "concern than miss a developing problem, so speculative alerts do not automatically bother you. At the "
        "same time, you treat them as possibilities to sort through, not as guaranteed facts."),
}

Q3_INTERRUPT = {
    1: ("You prefer close guidance as the task unfolds. You are comfortable having the assistant break th"
        "ings down and keep you oriented, because frequent direction reduces uncertainty. When you are be"
        "tween steps or have just finished a piece of work, you would rather ask what comes next than gue"
        "ss from memory. You tend to welcome detailed sequencing and timely prompts."),
    2: ("You prefer a middle-ground interruption style. You can absorb comments while working, but you also "
        "appreciate some respect for the rhythm of the task. Your engagement varies with timing, urgency, and "
        "how relevant the message is to what is happening."),
    3: ("You prefer not to be interrupted unless there is a real need. Extra commentary during active wor"
        "k can feel intrusive, especially if it is not urgent or directly useful. While you are busy, you"
        "r habit is to keep replies brief and get back to the task rather than open a side discussion. Yo"
        "u are more receptive when the interruption clearly prevents a problem or helps with something ti"
        "me-sensitive."),
}

# Axis names as the human sees them (2026-08-27 ruling: the card names each axis, shows the questionnaire
# item and the answer the person gave when there is one, and then the statement as the interpretation).
AXIS_TITLE = {"skill": "YOUR SKILL IN THE KITCHEN", "trust": "HOW FAR YOU TRUST THE AI ASSISTANT",
              "q1": "SPEED", "q2": "ALERTING", "q3": "INTERRUPTION"}

# Anti-parroting house style, persona-neutral (previously buried inside the novice card):
COMMON = ("You are a person, not a voice recorder: acknowledge advice in your own casual "
          "words and never repeat the supervisor's sentences or step lists back verbatim.")

# Sentences REMOVED from the old tier cards and the axis that owns them now. The tests
# assert none of these phrases appear in any SKILL statement.
FACTORED_OUT = {
    "you usually follow what the supervisor suggests": "trust",
    "ask for the next step": "q3",
    "don't like being interrupted": "q3",
    "SILENCE IS YOUR DEFAULT": "q3",
    "WAVE THE SUGGESTION OFF": "trust",
}

QUESTIONS = [
    ("q1", "How much do you care about solving the task fast?",
     {1: "safe first", 2: "soso", 3: "as soon as possible, with a lot of parallelism"}),
    ("q2", "How do you want the assistant to alert potential mistakes?",
     {1: "flag only when confident", 2: "medium",
      3: "spot all possible mistakes aggressively"}),
    ("q3", "What interruption style do you want?",
     {1: "step by step", 2: "medium", 3: "don't interrupt unless necessary"}),
]


@dataclass(frozen=True)
class PersonaV2:
    skill: str          # low | medium | high
    trust: str          # low | medium | high
    q1: int             # 1..3
    q2: int             # 1..3
    q3: int             # 1..3
    name: str = "custom"

    def __post_init__(self):
        if self.skill not in LEVELS or self.trust not in LEVELS:
            raise ValueError(f"skill/trust must be one of {LEVELS}")
        for q in (self.q1, self.q2, self.q3):
            if q not in (1, 2, 3):
                raise ValueError("questionnaire answers must be 1, 2 or 3")

    # -- weights: the level IS the weight (docs §1) --------------------------------
    def weights(self):
        return {"a": self.q1, "b": self.q2, "c": self.q3}

    # -- ability tier: unchanged mechanism binding (docs §4) -----------------------
    def solvability(self) -> str:
        """The existing harness gate values: low tier -> engine calls refused;
        high -> must self-resolve; creative -> own frames + engine calls."""
        return {"low": "low", "medium": "high", "high": "creative"}[self.skill]

    # -- the two renderings --------------------------------------------------------
    def human_card(self) -> str:
        """Five axes, each as: the axis name; the questionnaire item and the answer this person gave (the
        three public axes) or a note that it is simply who they are (the two private axes); then the
        statement — the interpretation of that level as a lived disposition, never a rule."""
        qmap = {k: (q, o) for k, q, o in QUESTIONS}
        blocks = []
        for key, level, bank in (("skill", self.skill, SKILL), ("trust", self.trust, TRUST),
                                 ("q1", self.q1, Q1_SPEED), ("q2", self.q2, Q2_ALERTING), ("q3", self.q3, Q3_INTERRUPT)):
            lvl_num = level if isinstance(level, int) else _NUM[level]
            head = f"AXIS: {AXIS_TITLE[key]} — level {lvl_num} of 3"
            if key in qmap:
                q, opts = qmap[key]
                head += (f" ({level if isinstance(level, str) else opts[level]}).\nQuestionnaire item you answered before the task: “{q}”\n"
                         f"  options: {' / '.join(f'{k} = {opts[k]}' for k in (1, 2, 3))}\n  your answer: {level} = “{opts[level]}” (the assistant is shown this answer)")
            else:
                head += f" ({level}).\nNot in the questionnaire: nobody asked you this — it is simply who you are, and the assistant does not know it."
            blocks.append(head + "\nInterpretation of this level (how a person at this level tends to think, talk and decide — a disposition, not a rule): " + bank[level])
        blocks.append(COMMON)
        return "\n\n".join(blocks)

    def assistant_block(self) -> str:
        lines = ["== USER PREFERENCES (this user filled a pre-task questionnaire; "
                 "these are their answers) =="]
        for i, (key, question, options) in enumerate(QUESTIONS, 1):
            chosen = options[getattr(self, key)]
            lines.append(f'{i}. "{question}"')
            lines.append("     " + " / ".join(options[k] for k in (1, 2, 3)))
            lines.append(f"   -> they chose: {chosen}")
        lines.append("You know nothing else about this user — their skill, and how far "
                     "they will trust you, must be inferred from how they behave.")
        return "\n".join(lines)

    def to_json(self):
        return {"schema": "persona-v2", "name": self.name, "skill": self.skill,
                "trust": self.trust, "q1": self.q1, "q2": self.q2, "q3": self.q3,
                "weights": self.weights()}


# Legacy anchors, expressible exactly as vectors (docs §4). Display continuity only.
LEGACY = {
    "novice_stepwise": PersonaV2("low", "high", 1, 3, 1, name="novice_stepwise_v2"),
    "creative_uninterruptible": PersonaV2("high", "low", 2, 1, 3,
                                          name="creative_uninterruptible_v2"),
}

# The approved 9-persona balanced roster for the eval_ranking sweep (every axis carries
# each level exactly three times).
ROSTER = [
    PersonaV2("medium", "medium", 2, 2, 2, name="baseline"),
    PersonaV2("low", "high", 1, 3, 1, name="classic_novice"),
    PersonaV2("high", "low", 2, 1, 3, name="classic_expert"),
    PersonaV2("low", "low", 2, 2, 1, name="defiant_novice"),
    PersonaV2("high", "high", 3, 3, 2, name="credulous_expert"),
    PersonaV2("medium", "low", 3, 2, 3, name="speed_demon"),
    PersonaV2("medium", "high", 1, 3, 1, name="safety_stepper"),
    PersonaV2("high", "medium", 1, 1, 3, name="quiet_perfectionist"),
    PersonaV2("low", "medium", 3, 1, 2, name="eager_parallel_novice"),
]


def _questionnaire_json(self):
    """The confirmed report contract (2026-08-13): the page audit reads key/choice/
    choice_text and CHECKS weight/weight_value as the writer's independent statement of
    the same fact — a drift here fails the page build loudly."""
    items = []
    for (key, question, options), w in zip(QUESTIONS, ("a", "b", "c")):
        ch = getattr(self, key)
        items.append({"key": key, "question": question,
                      "options": [options[k] for k in (1, 2, 3)],
                      "choice": ch, "choice_text": options[ch],
                      "weight": w, "weight_value": ch})
    return {"schema": "questionnaire-v2", "items": items}


def _profile_dials(self):
    """v2 -> the v1 MECHANICAL dials the harness enforces (ability tier from skill;
    acceptance cap from trust; interaction style from trust/q3). The prompt text comes
    from human_card(); these only set the mechanisms."""
    interaction = ("skeptical" if self.trust == "low"
                   else "avoidant" if self.q3 == 3 else "ok")
    acceptance = "low" if self.trust == "low" else "high"
    return {"task_solvability": self.solvability(),
            "interaction_preference": interaction,
            "information_acceptance": acceptance}


PersonaV2.questionnaire_json = _questionnaire_json
PersonaV2.profile_dials = _profile_dials
BY_NAME = {p.name: p for p in ROSTER}
