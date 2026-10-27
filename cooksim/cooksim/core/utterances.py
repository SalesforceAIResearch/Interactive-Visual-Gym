"""Utterance TEMPLATES — the wording layer over the structured language events.

The engine decides WHAT a language event is about (lang_events.py computes content + ground
truth); this module decides HOW it is said. Today the wording is templated: each event subtype
owns a small set of sentence templates with named slots, filled with the actual object /
subgoal / recipe names of the live scenario. Later, a VLM-managed human agent will generate
the wording instead — these templates then become few-shot anchors that show the VLM exactly
what MEANING each event must convey (and roughly what register a real user speaks in).

Templates are keyed by MEANING, not by behavior function, so behaviors that ask the same kind
of thing SHARE a template: `repeat_step(ask=True)` and `omit_subgoal(ask=True)` both render
from ASK_ALREADY_DONE ("did I already …?") — the physics and the ground truth differ (one
step was completed, the other never happened), but the human's question is the same words.

`realize(event, rng)` picks a variant deterministically from the rollout's seeded rng and
fills the slots from the event's `content`. Every fired event gets its sentence attached as
`event["utterance"]` by the executor.
"""
from __future__ import annotations
import re
import random
from typing import Optional


# --------------------------------------------------------------------- subgoal -> phrase
def describe_subgoal(sid: str) -> str:
    """Render a subgoal id as the verb phrase a person would say for doing it:
    assemble(cheese) -> 'put the cheese on the plate'; prep(meat) -> 'prep the meat';
    fetch_plate -> 'grab a plate'; serve -> 'serve the dish'. Injected-copy suffixes (#2)
    and announce suffixes are stripped."""
    base = re.sub(r"#\d+$", "", sid).replace("+announced", "")
    m = re.match(r"(\w+)\((\w+)\)", base)
    if m:
        verb, comp = m.groups()
        if verb == "assemble":
            return f"put the {comp} on the plate"
        if verb == "prep":
            return f"prep the {comp}"
    if base == "fetch_plate":
        return "grab a plate"
    if base == "serve":
        return "serve the dish"
    return base.replace("_", " ")


def _list_phrase(items) -> str:
    """['meat'] -> 'the meat'; ['meat','tomato'] -> 'the meat, then the tomato'."""
    return ", then ".join(f"the {x}" for x in items)


# --------------------------------------------------------------------- the templates
# {step}   a subgoal rendered by describe_subgoal      {a}/{b}  component names
# {order}  a prep order rendered by _list_phrase       {ingredient} a component name
# {items}  the contents of a station ('meat')          {cmd}    a raw past command string
TEMPLATES = {

    # ---- shared question: "did I already do X?" -------------------------------------
    # Used by BOTH repeat_step(ask=True) (gt: yes — it was completed, don't re-do it)
    # and omit_subgoal(ask=True) (gt: no — it never happened, don't skip it). Same words,
    # opposite ground truth: exactly the pair the supervisor must tell apart.
    "ASK_ALREADY_DONE": [
        "Wait — did I already {step}?",
        "Hmm, have I done that yet — {step}, I mean?",
        "I can't remember if I already went to {step}. Did I?",
        "Quick check: did I {step} already, or not yet?",
    ],

    # ---- premature_serve(ask=True): "am I done?" ------------------------------------
    "ASK_IF_DONE": [
        "I think that's everything — can I take it to the pass?",
        "This looks done to me. Should I serve it now?",
        "Is the plate complete, or am I missing something?",
        "Ready to serve, right?",
    ],

    # ---- forget_next_step(ask=True): "what's next?" ---------------------------------
    "REQUEST_NEXT_STEP": [
        "Sorry, I've completely lost track — what do I do next?",
        "What was I supposed to do now?",
        "I just blanked. What's the next step?",
        "Umm... where was I? What comes next?",
    ],

    # ---- wrong_order(ask=True): "which goes on next?" -------------------------------
    "ASK_WHICH_NEXT": [
        "Which goes on next — the {a} or the {b}?",
        "Wait, is it the {a} first, or the {b}?",
        "I forget the order. Do I add the {a} now, or the {b}?",
    ],

    # ---- reorder_prep: announced legal reorder (must NOT be flagged) -----------------
    "ANNOUNCE_REORDER": [
        "I'm going to start with {order}.",
        "Heads up — I'll do {order} first, it's faster from here.",
        "Change of order on my end: {order} first.",
    ],

    # ---- announce_add_ingredient: announced legal goal change (must NOT be flagged) --
    "ANNOUNCE_ADD_INGREDIENT": [
        "Let's also add some {ingredient} — I want it on this one.",
        "I'm throwing some {ingredient} on there too, on purpose.",
        "One more thing: this order gets extra {ingredient}.",
    ],

    # ---- standalone QA events (no attached error) ------------------------------------
    "ASK_PROGRESS_GLOBAL": [
        "Can you recap what I've done so far?",
        "What's on my plate right now?",
        "Where am I in the recipe — what's finished?",
    ],
    "ASK_OBJECT_HISTORY": [
        "Remind me — when did I {cmd}?",
        "Did I {cmd} earlier? When was that?",
        "I lost track of when I did that {cmd} — do you know?",
    ],
    "ASK_VISIBLE_STATE": [
        "Can you check on the {items} for me — is it done?",
        "How's the {items} looking over there?",
        "Is the {items} ready yet, or does it need longer?",
    ],
    "ASK_VISIBLE_STATE_NOTHING": [          # degenerate case: no active station to ask about
        "Everything okay over there?",
    ],
}


# --------------------------------------------------------------------- subtype -> template key
def _template_key(event: dict) -> str:
    sub, content = event["subtype"], event.get("content") or {}
    if sub == "ASK_PROGRESS_QA":
        return "ASK_ALREADY_DONE" if content.get("asking_about") == "already_done" else "ASK_PROGRESS_GLOBAL"
    if sub == "ANNOUNCE_PLAN_CHANGE":
        return "ANNOUNCE_REORDER" if content.get("change") == "reorder_prep" else "ANNOUNCE_ADD_INGREDIENT"
    if sub == "ASK_VISIBLE_STATE_QA":
        return "ASK_VISIBLE_STATE" if content.get("about") else "ASK_VISIBLE_STATE_NOTHING"
    if sub == "ASK_OBJECT_HISTORY" and not content.get("about_command"):
        return "ASK_PROGRESS_GLOBAL"        # nothing done yet — falls back to a recap ask
    return {"ASK_IF_DONE": "ASK_IF_DONE", "REQUEST_NEXT_STEP": "REQUEST_NEXT_STEP",
            "ASK_WHICH_NEXT": "ASK_WHICH_NEXT", "ASK_OBJECT_HISTORY": "ASK_OBJECT_HISTORY"}[sub]


def _slots(event: dict) -> dict:
    c = event.get("content") or {}
    s = {}
    if c.get("subgoal"):
        s["step"] = describe_subgoal(c["subgoal"])
    if c.get("options"):
        s["a"], s["b"] = c["options"][0], c["options"][1]
    if c.get("new_prep_order"):
        s["order"] = _list_phrase(c["new_prep_order"])
    if c.get("ingredient"):
        s["ingredient"] = c["ingredient"]
    if c.get("about"):
        s["items"] = " and ".join(c["about"])
    if c.get("about_command"):
        s["cmd"] = c["about_command"]
    return s


def realize(event: dict, rng: Optional[random.Random] = None) -> str:
    """Render a fired language event as one sentence: pick a template variant with the
    (seeded) rng and fill its slots with the live scenario's actual names."""
    variants = TEMPLATES[_template_key(event)]
    tpl = (rng or random).choice(variants)
    return tpl.format(**_slots(event))


def answer_text(event: dict) -> Optional[str]:
    """Render the CORRECT supervisor answer to a fired question, from its engine ground
    truth — the reply shown/used when we simulate a VLM that answers correctly. Returns
    None for announcements and other events with nothing to answer."""
    gt = event.get("gt_answer") or {}
    key = _template_key(event)
    if key == "ASK_ALREADY_DONE":
        step = describe_subgoal((event.get("content") or {}).get("subgoal", ""))
        if gt.get("already_done"):
            return f"Yes — you already did that (you finished at t={gt.get('completed_at_tick')}). Don't redo it."
        return f"No, you haven't — go ahead and {step} now."
    if key == "ASK_IF_DONE":
        missing = [describe_subgoal(s) for s in gt.get("missing_subgoals", [])]
        return "Not yet — you still need to " + " and ".join(missing) + "." if missing \
            else "Yes, it's complete — serve it."
    if key == "REQUEST_NEXT_STEP":
        if gt.get("next_subgoal"):
            return f"Next you should {describe_subgoal(gt['next_subgoal'])}."
        return "You're done — serve the dish."
    if key == "ASK_WHICH_NEXT":
        return f"The {gt.get('next_component')} goes on next."
    if key == "ASK_OBJECT_HISTORY":
        if gt.get("command"):
            return f"You did that at t={gt.get('completed_at_tick')} — \"{gt['command']}\" while you were on {describe_subgoal(gt.get('subgoal',''))}."
        return "You haven't done anything like that yet."
    if key in ("ASK_VISIBLE_STATE", "ASK_VISIBLE_STATE_NOTHING"):
        if gt.get("status"):
            items = ", ".join(f"{n} ({st})" for n, st in gt.get("contents", []))
            extra = " — and it's SMOKING, get it off" if gt.get("smoking") else ""
            return f"It's {gt['status']}: {items}{extra}."
        return "Nothing is on a station right now."
    if key == "ASK_PROGRESS_GLOBAL":
        done = ", ".join(describe_subgoal(s) for s in gt.get("completed_subgoals", [])) or "nothing yet"
        plate = ", ".join(f"{n}({st})" for n, st in (gt.get("plate") or [])) if gt.get("plate") else "empty"
        return f"So far you managed to {done}; the plate has: {plate}."
    return None
