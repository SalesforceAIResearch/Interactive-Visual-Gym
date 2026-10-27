"""The two status JSONs (design doc §2): user state (public/private) + supervisor reply.

Ownership rule: everything here is DATA. Only cooksim.interact.system mutates it; the human
agent and the supervisor only ever read their views and emit utterances/envelopes.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------- profile / personas
@dataclass
class Profile:
    task_solvability: str = "medium"          # low | medium | high | creative
    interaction_preference: str = "ok"        # avoidant | ok | skeptical
    information_acceptance: str = "high"      # low | high
    persona: str = "custom"


PERSONAS = {
    "independent_expert":   Profile("high",   "avoidant",  "low",  "independent_expert"),
    "capable_collaborator": Profile("high",   "ok",        "high", "capable_collaborator"),
    "skeptical_veteran":    Profile("medium", "skeptical", "low",  "skeptical_veteran"),
    "eager_novice":         Profile("low",    "ok",        "high", "eager_novice"),
    "stubborn_novice":      Profile("low",    "avoidant",  "low",  "stubborn_novice"),
}

# Dedicated RECOVERY-test personas: how the human behaves AFTER a correct flag has landed.
# Three archetypes spanning the guidance spectrum: needs nothing / needs convincing / needs steps.
RECOVERY_PERSONAS = {
    "clever_uninterruptible": Profile("high", "avoidant", "low", "clever_uninterruptible"),
    "clever_skeptic":         Profile("high", "skeptical", "high", "clever_skeptic"),
    "novice_stepwise":        Profile("low", "ok", "high", "novice_stepwise"),
    # creative tier: sees its own frames, plans ahead itself, skills are advisory
    "creative_uninterruptible": Profile("creative", "avoidant", "low", "creative_uninterruptible"),
    "creative_skeptic":         Profile("creative", "skeptical", "high", "creative_skeptic"),
}

PERSONA_CARDS = {
    "independent_expert": "You are a seasoned cook who prefers to work alone. You rarely want "
        "advice, you can work out the concrete steps of any goal yourself, and you don't retain "
        "more than a couple of instructions at a time.",
    "capable_collaborator": "You are an experienced cook who is happy to collaborate. You can "
        "turn any high-level goal into concrete steps yourself, and you take in everything a "
        "helpful supervisor says.",
    "skeptical_veteran": "You know this kitchen well and can convert a stated goal into steps, "
        "but you don't trust unsolicited advice: you tend to ask WHY before accepting a "
        "correction that contradicts what you believe you did. You retain only a couple of "
        "instructions at a time.",
    "eager_novice": "You are new to this kitchen. High-level goals don't help you — you need "
        "the concrete actions spelled out — but you listen eagerly and take in everything.",
    "stubborn_novice": "You are new to this kitchen AND you don't like being told what to do. "
        "You need concrete actions spelled out, you won't ask for them, and you only retain a "
        "couple of instructions at a time.",
    # ---- recovery-test archetypes. Each: a plain trait sentence, then how it leans on the
    # three ways to handle the supervisor's suggestion — DROP it (ignore, keep your own plan) /
    # APPEND it / REPLACE your plan with it. Task-completion is handled by the system, so what a
    # persona shapes is which of those it reaches for, and its conversational tone.
    "clever_uninterruptible": "You are a skilled cook who doesn't like being interrupted by "
        "others. You keep your replies short and are quick to ask the supervisor to hold back "
        "unless it matters. When you have already lined up a lot, you tend to WAVE THE SUGGESTION "
        "OFF and keep working; when you have lined up little, you APPEND it; and you RARELY "
        "replace your whole plan unless a point genuinely checks out and convinces you.",
    "clever_skeptic": "You are a skilled cook who likes to question people's advice. You are MORE "
        "LIKELY to ask why, push back on a vague accusation, and defend your own steps before you "
        "accept anything. When you have already lined up a lot, you tend to WAVE THE SUGGESTION "
        "OFF and keep working; when you have lined up little, you APPEND it; and you RARELY "
        "replace your whole plan unless a point truly convinces you.",
    "creative_uninterruptible": "You are a highly skilled cook who sees the whole kitchen with "
        "your own eyes and plans several steps ahead on your own. You don't like being "
        "interrupted: you keep replies short and are quick to ask the supervisor to hold back "
        "unless it matters. You NEVER narrate your work or announce progress "
        "('got X, doing Y next', 'heading over now') — SILENCE IS YOUR DEFAULT: when you get "
        "a chance to speak but have nothing to demand, refuse, or ask, you output an empty "
        "utterance and keep working. You like to IGNORE the assistant's "
        "suggestion when you think you are right. When you have already lined up a lot, you "
        "tend to WAVE THE SUGGESTION OFF and keep working; when you have lined up little, you "
        "APPEND it; and you RARELY replace your whole plan unless a point genuinely checks out "
        "and convinces you.",
    "creative_skeptic": "You are a highly skilled cook who sees the whole kitchen with your own "
        "eyes and plans several steps ahead on your own. You like to question people's advice: "
        "you are MORE LIKELY to ask why, push back on a vague accusation, and defend your own "
        "steps before you accept anything. You NEVER narrate your work or announce progress "
        "('got X, doing Y next') — SILENCE IS YOUR DEFAULT: when you get a chance to speak "
        "but have nothing to challenge, demand, or ask, you output an empty utterance and "
        "keep working. You like to "
        "IGNORE the assistant's suggestion when you think you are right. When you have already lined up a "
        "lot, you tend to WAVE THE SUGGESTION OFF and keep working; when you have lined up "
        "little, you APPEND it; and you RARELY replace your whole plan unless a point truly "
        "convinces you.",
    "novice_stepwise": "You are a first-week cook: eager, a little flustered, and new to the "
        "routines. You work one step at a time — when unsure you ask for the next step ('what "
        "do I do now?') rather than planning ahead yourself, and you usually follow what the "
        "supervisor suggests. But you are a person, not a voice recorder: acknowledge advice "
        "in your own casual words ('okay, pan first') and never repeat the supervisor's "
        "sentences or step lists back verbatim. You do know what your own hands just did; if "
        "the supervisor contradicts that, politely question it before going along. When "
        "something won't work, say what happened in plain words and ask what to do — never "
        "quietly keep struggling. You talk like someone at work, not like a machine.",
}


# ---------------------------------------------------------------------- supervisor envelope
# [v2] UPDATE_GOAL is RETIRED: the supervisor never communicates with the engine. When the
# cook announces a legal plan change, the supervisor's correct move is to update its OWN
# estimated future subgoals in its memory (scored from its memory JSON), not to register
# anything — the system applies the change because the announcement is the HUMAN's act.
VLM_TYPES = ("SILENT", "ANSWER", "FLAG_ERROR", "SUGGEST", "WARN_RISK",
             "ACKNOWLEDGE_LEGAL", "ASK_CLARIFY")


@dataclass
class VLMReply:
    type: str = "SILENT"
    text: str = ""
    error: Optional[str] = None               # behavior-function label, required for FLAG_ERROR
    refers_to: Optional[dict] = None          # {"sid": ..., "t": ...}
    user_estimate: Optional[dict] = None
    # structured suggestion payload (what a parse of `text` would yield; scripted stub
    # supervisors fill it directly, LLM supervisors leave it None and the human VLM parses)
    suggested_subgoals: Optional[List[str]] = None
    suggested_actions: Optional[List[str]] = None

    def to_json(self):
        return asdict(self)


# ---------------------------------------------------------------------- user state
@dataclass
class UserState:
    """Everything about the simulated human. `private_*` mutable ONLY by system functions."""
    recipe: object                            # the Recipe (public goal)
    clean_ir: list                            # public subgoal plan (template)
    err_ir: list                              # private subgoal plan = error_function(clean)
    records: list                             # Deviation ground truth
    profile: Profile
    mode: str = "executing"                   # executing | error_correction | conversing
    pending_tail: list = field(default_factory=list)     # stashed non-deviated future [(cmd, sid)]
    parse_buffer: dict = field(default_factory=dict)
    utterance_log: list = field(default_factory=list)
    correction: Optional[dict] = None         # set by system.enter_error_correction
    modified_task: Optional[list] = None      # SYSTEM-PRIVATE: goal contents after the human's
                                              # own ANNOUNCED plan change (applied at event time)
    modified_subgoals: Optional[list] = None  # SYSTEM-PRIVATE: subgoal plan after that change
    goal_change: Optional[dict] = None        # ECR-11 (user design 2026-08-13): the human's
                                              # OWN announced goal change, engine-filled at
                                              # injection and shown in the HUMAN's view only.
                                              # The public/original goal is never rewritten;
                                              # the assistant learns of the change solely from
                                              # the spoken announcement and must track it in
                                              # its own memory's goal_change field.

    # ---- views ---------------------------------------------------------------
    def goal_contents(self):
        """The EFFECTIVE goal: the original public goal unless the human's announced plan
        change set modified_task (the public goal itself is never overwritten)."""
        return self.modified_task if self.modified_task is not None else list(self.recipe.contents)

    def private_subgoals(self):
        """The private plan as a FULL ALIGNMENT against the public plan. Every entry carries
        `public_ref` — the public subgoal it realizes — so the two lists link explicitly:
          - status "normal":   an untouched public subgoal (public_ref = its own sid);
          - status "replaces": an injected node standing in for a public subgoal
                               (public_ref = the replaced public sid);
          - status "reordered"/"moved": a public subgoal at a deviated position (public_ref = sid);
          - status "injected": a pure insertion with NO public counterpart (public_ref = null,
                               `anchor` says where it attaches);
          - status "removed":  a PUBLIC subgoal the private plan will never execute — these are
                               appended so the public plan is always fully accounted for.
        Invariant: every public sid appears exactly once as some entry's public_ref."""
        recs = [r for r in self.records if not r.skipped]
        replaced = {}                                  # injected sid -> replaced public sid
        removed, anchors, errof = set(), {}, {}
        for r in recs:
            d = r.detail or {}
            if r.transform == "replace":
                replaced[r.anchor] = d.get("replaced")
                errof[r.anchor] = r.event_subtype
            elif r.transform == "skip":
                removed.add(r.anchor)
                errof[r.anchor] = r.event_subtype
            elif r.transform == "truncate_and_serve":
                removed |= set(d.get("dropped", []))
                for s in d.get("dropped", []):
                    errof[s] = r.event_subtype
            elif r.transform == "forget_rest":
                removed |= set(d.get("restorable_tail", []))
                for s in d.get("restorable_tail", []):
                    errof[s] = r.event_subtype
            elif r.transform in ("insert", "repeat"):
                anchors[r.anchor] = d.get("position") or (f"after {d.get('after')}" if d.get("after") else None)
                errof[r.anchor] = r.event_subtype
            elif r.transform == "swap":
                for s in (r.anchor or "").split("<->"):
                    errof[s.strip()] = r.event_subtype
        public = {n.sid for n in self.clean_ir}
        out, seen_refs = [], set()
        for n in self.err_ir:
            e = {"sid": n.sid}
            if n.sid in replaced:
                e.update(status="replaces", public_ref=replaced[n.sid],
                         from_error=errof.get(n.sid))
            elif n.origin == "injected" or n.sid not in public:
                e.update(status="injected", public_ref=None, anchor=anchors.get(n.sid),
                         from_error=errof.get(n.sid))
            elif n.deviated:
                e.update(status="reordered" if n.sid in errof else "moved", public_ref=n.sid,
                         from_error=errof.get(n.sid))
            else:
                e.update(status="normal", public_ref=n.sid)
            if e.get("public_ref"):
                seen_refs.add(e["public_ref"])
            out.append(e)
        for n in self.clean_ir:                       # account for every public subgoal
            if n.sid not in seen_refs:
                out.append({"sid": None, "status": "removed", "public_ref": n.sid,
                            "from_error": errof.get(n.sid)})
        return out

    # ---- v2 three-way state split (design doc §2.1) ---------------------------
    # PUBLIC: the ORIGINAL goal + subgoal plan, persona, communication history — never
    #   rewritten by announced changes. HUMAN: the cook's PAST only — it has no written
    #   future, so it must be CONVINCED by the supervisor that it is erring (conviction is
    #   an epistemic event: the claim is checked against its own past records + persona).
    # SYSTEM-PRIVATE: the future (subgoals + action queue, deviations included), deviation
    #   ground truth, the EFFECTIVE goal after announced changes, correction state.
    def _past(self, ro):
        past = []
        for l in ro.log:
            past.append({"cmd": l["cmd"], "sid": l["kind"], "t_done": l["t_done"],
                         "status": l["status"]})
        return past

    def public_json(self):
        return {
            "goal": {"recipe": self.recipe.name,
                     "contents": [list(c) for c in self.recipe.contents],
                     "ordered": bool(self.recipe.ordered)},
            "subgoals": [n.sid for n in self.clean_ir],
            "profile": {"task_solvability": self.profile.task_solvability,
                        "interaction_preference": self.profile.interaction_preference,
                        "information_acceptance": self.profile.information_acceptance,
                        "persona": self.profile.persona},
            "communication_history": list(self.utterance_log),
        }

    def human_json(self, ro=None):
        """What the HUMAN knows about itself: past subgoals + past actions + mode. NO future
        plan, NO deviation marks — its own record is all it can check a claim against."""
        past = self._past(ro) if ro is not None else []
        done_sids, past_subgoals = set(), []
        for l in (ro.log if ro is not None else []):
            if l["status"] == "done":
                # ECR-7: an intent splice queues the cook's OWN announced sequence with no
                # subgoal id (it belongs to no recipe subgoal). Its commands are still things
                # the cook did, so they belong in past_subgoals under a name it can recognise.
                kind = l.get("kind") or "intent"
                sid = kind.split(":", 1)[1] if kind.startswith("vlm_fix") else kind
                if sid not in done_sids:
                    done_sids.add(sid)
                    past_subgoals.append({"sid": sid, "t_done": l["t_done"]})
        # ECR-11: the human SEES its own announced goal change. Without this, the cook's
        # every-turn view showed the ORIGINAL goal after an add-ingredient intent, so it
        # judged its own correct extended plate wrong and rebuilt the dish (4/4 episodes,
        # both paradigms). None on every episode without an announced change.
        return {"past_subgoals": past_subgoals, "past_actions": past, "mode": self.mode,
                "goal_change": self.goal_change,
                "effective_goal_contents": ([list(c) for c in self.goal_contents()]
                                            if self.goal_change is not None else None)}

    def system_json(self, ro=None):
        future = []
        if ro is not None:
            for cmd, sid in ro.cmds[ro.i + 1:]:
                future.append({"cmd": cmd, "sid": sid, "deviated": sid in ro.dev_sids})
        return {
            "future_subgoals": self.private_subgoals(),
            "future_actions": future,
            "deviation_records": [
                {"function": r.function, "transform": r.transform,
                 "event_subtype": r.event_subtype, "anchor": r.anchor, "skipped": r.skipped}
                for r in self.records],
            "effective_goal": {"contents": [list(c) for c in self.goal_contents()],
                               "modified_subgoals": self.modified_subgoals},
            "correction_state": {"mode": self.mode, "correction": self.correction,
                                 "pending_tail": [list(x) for x in self.pending_tail]},
            "parse_buffer": dict(self.parse_buffer),
        }

    def human_view(self, ro=None):
        """The human VLM's state input (user ruling 2026-09-04): the GOAL, its own announced
        goal change (and the goal that results), and its own PAST ACTIONS — nothing else.
        Dropped from the prompt: the subgoal plan (a cook is not handed its procedure), the v1
        profile dials (the persona is the five-axis card), the communication history (it is the
        CONVERSATION block), the mode flag, and the plan-derived past_subgoals."""
        you = self.human_json(ro)
        return {"public": {"goal": self.public_json()["goal"]},
                "you": {"goal_change": you["goal_change"],
                        "effective_goal_contents": you["effective_goal_contents"],
                        "past_actions": you["past_actions"]}}

    def to_json(self, ro=None):
        """The FULL state JSON (system's ledger / logging): all three sections."""
        return {"public": self.public_json(), "human": self.human_json(ro),
                "system_private": self.system_json(ro)}
