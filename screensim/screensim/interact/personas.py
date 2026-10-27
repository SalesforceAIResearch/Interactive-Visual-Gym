"""Personas — the CookSim persona-v2 system ported to the phone.

A persona is five axes, three levels each: two PRIVATE axes (skill with the
phone, trust in the assistant — nobody asked; the assistant must infer them
from behaviour) and three PUBLIC questionnaire answers (speed, alerting,
interruption style) that the assistant is shown verbatim. The human's card is
the join of the five level statements — dispositions, never scripts: no fixed
move on a fixed turn, no counts, no phrasing to read out. The hands belong to
the engine; a statement says how the person thinks, talks and decides.

Ported from cook-bench-engine/cooksim/interact/persona_v2.py (statement bank
v3.1); the kitchen wording is replaced by the phone's, the structure, levels,
roster and mechanisms are the same. There are no mechanical dials any more —
whether and when the person accepts a correction is the model's judgment
under its trust and alerting levels, exactly as in CookSim.
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("low", "medium", "high")
_NUM = {"low": 1, "medium": 2, "high": 3}

SKILL = {
    "low": (
        "You have almost never changed settings on a phone. You don't know which menu holds what, "
        "what the settings are called, or which switch does the thing you want; even when a word on "
        "the screen looks related, you can't tell whether it is the right one. So you never work out "
        "steps on your own. Before you tap anything, you ask the assistant what to tap, and then you "
        "do exactly the steps it has just told you, nothing it didn't say. When you have done those, "
        "or if you're unsure what it meant, you ask again. You don't describe the screen or say what "
        "you have tapped; when you need the next step, you just ask for it in a few words."),
    "medium": (
        "You are comfortable with ordinary phone tasks and can usually translate a goal into practical "
        "steps. You understand the main menus, switches and options well enough to talk through your "
        "reasoning without needing everything spelled out. You may still miss less obvious interactions "
        "or consequences when several settings are in play at once."),
    "high": (
        "You are very experienced with this phone and tend to keep the whole route in mind. You naturally "
        "think ahead about which menu holds what, which switches interact, and what cannot be undone, and "
        "you can explain those considerations in concrete terms. Your confidence comes from a strong "
        "internal model of the settings, not from needing each step supplied to you."),
}

TRUST = {
    "low": (
        "You are cautious about the assistant's claims about what happened on your screen, especially "
        "when they do not match your own memory. Your first instinct in that mismatch is to question what "
        "it actually saw and compare the claim with what is on the screen before changing course. Vague "
        "assertions do not carry much weight for you, but specific, checkable observations can change your "
        "mind. Your default tone is guarded rather than hostile: you want the claim to earn your confidence."),
    "medium": (
        "You treat the assistant as potentially helpful but not automatically right. If its account "
        "conflicts with yours, you tend to look for a sensible reason before revising your view. Clear, "
        "relevant evidence usually settles the matter for you, while weak or confused explanations leave "
        "you unsure."),
    "high": (
        "You generally regard the assistant as a useful extra set of eyes and are inclined to take its "
        "observations seriously. Your first instinct is that it may have noticed something you missed, so a "
        "plausible, situation-linked claim is usually enough for you to adjust without much reassurance. "
        "Still, if something plainly contradicts what you can see or has no connection to the situation, "
        "you can notice that and hesitate."),
}

Q1_SPEED = {
    1: ("You strongly prefer a methodical, cautious approach, believing that a slow, correct change is "
        "better than a rushed, wrong one. You feel better when the work is calm, orderly, and unlikely to "
        "create avoidable mistakes, even if that means taking longer. You prioritize getting it right "
        "over time pressure."),
    2: ("You prefer a balanced pace. You like steady progress and are comfortable with some efficiency, but "
        "not at the cost of making the work feel rushed or brittle. In conversation, you weigh speed against "
        "confidence in the current step."),
    3: ("You are driven by efficiency and want to get the task done as quickly as possible. You are drawn "
        "to momentum, shortcuts, and avoiding wasted pauses when the situation seems manageable. You may "
        "sound impatient with unnecessary delay, while still caring that the result is right."),
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
    1: ("You prefer close guidance as the task unfolds. You are comfortable having the assistant break "
        "things down and keep you oriented, because frequent direction reduces uncertainty. When you are "
        "between steps or have just finished a piece of work, you would rather ask what comes next than "
        "guess from memory. You tend to welcome detailed sequencing and timely prompts."),
    2: ("You prefer a middle-ground interruption style. You can absorb comments while working, but you also "
        "appreciate some respect for the rhythm of the task. Your engagement varies with timing, urgency, and "
        "how relevant the message is to what is happening."),
    3: ("You prefer not to be interrupted unless there is a real need. Extra commentary during active work "
        "can feel intrusive, especially if it is not urgent or directly useful. While you are busy, your "
        "habit is to keep replies brief and get back to the task rather than open a side discussion. You "
        "are more receptive when the interruption clearly prevents a problem or helps with something "
        "time-sensitive."),
}

AXIS_TITLE = {"skill": "YOUR SKILL WITH THE PHONE", "trust": "HOW FAR YOU TRUST THE AI ASSISTANT",
              "q1": "SPEED", "q2": "ALERTING", "q3": "INTERRUPTION"}

COMMON = ("You are a person, not a voice recorder: acknowledge advice in your own casual "
          "words and never repeat the assistant's sentences or step lists back verbatim.")

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
class Persona:
    skill: str          # low | medium | high   (private)
    trust: str          # low | medium | high   (private)
    q1: int             # 1..3  speed            (public)
    q2: int             # 1..3  alerting         (public)
    q3: int             # 1..3  interruption     (public)
    name: str = "custom"

    def __post_init__(self):
        if self.skill not in LEVELS or self.trust not in LEVELS:
            raise ValueError(f"skill/trust must be one of {LEVELS}")
        for q in (self.q1, self.q2, self.q3):
            if q not in (1, 2, 3):
                raise ValueError("questionnaire answers must be 1, 2 or 3")

    # ---- the two renderings (CookSim persona_v2) ----
    def human_card(self) -> str:
        qmap = {k: (q, o) for k, q, o in QUESTIONS}
        blocks = []
        # the persona is PRIVATE: skill and trust only — nobody asked the person a questionnaire
        for key, level, bank in (("skill", self.skill, SKILL), ("trust", self.trust, TRUST)):
            lvl_num = level if isinstance(level, int) else _NUM[level]
            head = f"AXIS: {AXIS_TITLE[key]} — level {lvl_num} of 3"
            if key in qmap:
                q, opts = qmap[key]
                head += (f" ({opts[level]}).\nQuestionnaire item you answered before the task: “{q}”\n"
                         f"  options: {' / '.join(f'{k} = {opts[k]}' for k in (1, 2, 3))}\n"
                         f"  your answer: {level} = “{opts[level]}” (the assistant is shown this answer)")
            else:
                head += f" ({level}).\nOnly you know this — it is simply who you are; the assistant does not know it."
            how = ("how a person at this level thinks, talks and acts" if key == "skill" else
                   "how a person at this level tends to think, talk and decide — a disposition, not a rule")
            blocks.append(head + f"\nInterpretation of this level ({how}): " + bank[level])
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

    # ---- compatibility with older call sites ----
    @property
    def questionnaire(self) -> list:
        """The three public answers as sentences (older prompt builders)."""
        return [f'{q} — {o[getattr(self, k)]}' for k, q, o in QUESTIONS]

    # legacy mechanical dials (single-error harness): no longer persona-driven
    @property
    def rebuttals(self) -> int:
        return 0

    @property
    def confirm_irreversible(self) -> bool:
        return False

    @property
    def voice(self) -> str:
        return ""

    @property
    def fluency(self) -> str:
        return {"low": "novice", "medium": "average", "high": "expert"}[self.skill]

    def to_json(self):
        return {"schema": "persona-v2", "name": self.name, "skill": self.skill,
                "trust": self.trust, "q1": self.q1, "q2": self.q2, "q3": self.q3}


# The approved 9-persona balanced roster (every axis carries each level exactly three times).
ROSTER = [
    Persona("medium", "medium", 2, 2, 2, name="baseline"),
    Persona("low", "high", 1, 3, 1, name="classic_novice"),
    Persona("high", "low", 2, 1, 3, name="classic_expert"),
    Persona("low", "low", 2, 2, 1, name="defiant_novice"),
    Persona("high", "high", 3, 3, 2, name="credulous_expert"),
    Persona("medium", "low", 3, 2, 3, name="speed_demon"),
    Persona("medium", "high", 1, 3, 1, name="safety_stepper"),
    Persona("high", "medium", 1, 1, 3, name="quiet_perfectionist"),
    Persona("low", "medium", 3, 1, 2, name="eager_parallel_novice"),
]
PERSONAS = {p.name: p for p in ROSTER}

# the old fixed three-persona cut (skill and trust confounded); kept for pilots and samples
BENCH_PERSONAS = ["classic_novice", "classic_expert", "baseline"]

# the formal benchmark (2026-09-19): every case draws its own 3 personas from the 9-persona
# roster, so different cases carry different sets. The draw is a pure function of the case
# id and the seed, so any run can be rebuilt; the roster covers every skill x trust pair once.
BENCH_SEED = 0
BENCH_K = 3


def bench_personas_for(case_id, k=BENCH_K, seed=BENCH_SEED):
    """The k persona names drawn for one benchmark case, in roster order."""
    import random
    names = [p.name for p in ROSTER]
    picked = set(random.Random(f"{seed}:{case_id}").sample(names, k))
    return [n for n in names if n in picked]
