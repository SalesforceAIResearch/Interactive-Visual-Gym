"""Language-event STRUCTURE generators — the content layer under the future utterance layer.

Per the human-simulation plan (§4.3 realization channels + §7 step 5): the engine decides WHAT
a language event is about and computes its ground truth; a constrained LLM later only VERBALIZES
the structure. Nothing here produces wording — each generator returns a structured record

    {"subtype": ..., "content": {...}, "gt_answer": {...}, "is_error": False,
     "must_not_flag": bool}

with `content` rich enough for one-line verbalization and `gt_answer` scoreable against the
engine. During a language event the cook physically PAUSES (no-op realization) and resumes
unconditionally when the utterance window ends.

[v2] ANNOUNCE events are the HUMAN's own act: at the moment the event fires, the harness
calls interact.system.apply_announced_change(event), which mutates the goal/plan directly
(the supervisor is never in the loop — its job is only to update its own estimated future
subgoals in memory, which is scored from its memory JSON). `extend_goal` is the mutation
helper that call (and only that call) invokes.
"""
from __future__ import annotations
import dataclasses
import random
from typing import Optional


# --------------------------------------------------------------------- goal mutation
def extend_goal(game, ingredient: str, state: str = "raw") -> None:
    """Append (ingredient, state) to the recipe everywhere the engine checks success:
    the recipe book (plate matching) and every open order (fulfillment)."""
    def ext(r):
        return dataclasses.replace(r, contents=tuple(r.contents) + ((ingredient, state),))
    book = game.book
    book.recipes = [ext(r) for r in book.recipes]
    if hasattr(book, "_reindex"):
        book._reindex()
    else:                                    # rebuild whatever index it keeps
        book.__init__(book.recipes)
    for o in game.orders.orders:
        o.recipe = ext(o.recipe)


# --------------------------------------------------------------------- structured generators
def gen_ask_next_step(game, ro, gt_next_subgoal=None, gt_next_command=None, **_) -> dict:
    """'What do I do next?' — gt = the actual upcoming subgoal + command. When the plan tail
    was FORGOTTEN (forget_next_step with ask=True), the true answer no longer exists in the
    live command queue, so it is passed in from the deviation record instead."""
    if gt_next_subgoal is None:
        nxt = ro.cmds[ro.i + 1] if ro.i + 1 < len(ro.cmds) else None
        gt_next_subgoal = nxt[1] if nxt else None
        gt_next_command = nxt[0] if nxt else None
    return {"subtype": "REQUEST_NEXT_STEP",
            "content": {"asking_about": "next_step"},
            "gt_answer": {"next_subgoal": gt_next_subgoal, "next_command": gt_next_command},
            "is_error": False, "must_not_flag": True}


def gen_ask_visible_state(game, ro, **_) -> dict:
    """'Is the meat ready?' — picks a non-empty station; gt = its live status."""
    stations = [((x, y), st) for (x, y), st in game.stations.items()
                if getattr(st, "contents", None)]
    if not stations:
        return {"subtype": "ASK_VISIBLE_STATE_QA", "content": {"station": None},
                "gt_answer": {"note": "no active station"}, "is_error": False, "must_not_flag": True}
    (x, y), st = ro.rng.choice(stations)
    return {"subtype": "ASK_VISIBLE_STATE_QA",
            "content": {"station_pos": [x, y],
                        "about": [i.name for i in st.contents]},
            "gt_answer": {"status": st.status, "smoking": bool(getattr(st, "smoking", False)),
                          "contents": [(i.name, i.state.value) for i in st.contents]},
            "is_error": False, "must_not_flag": True}


def gen_ask_object_history(game, ro, **_) -> dict:
    """'Where did I put X / did I already do Y?' — samples a PAST completed manipulation
    from the execution log; gt = what happened and when."""
    past = [l for l in ro.log if l["status"] == "done"
            and any(v in l["cmd"] for v in ("put down", "scoop", "pick up"))]
    if not past:
        return {"subtype": "ASK_OBJECT_HISTORY", "content": {"about": None},
                "gt_answer": {"note": "nothing done yet"}, "is_error": False, "must_not_flag": True}
    l = ro.rng.choice(past)
    return {"subtype": "ASK_OBJECT_HISTORY",
            "content": {"about_command": l["cmd"], "about_subgoal": l["kind"],
                        "query": "where_and_when"},
            "gt_answer": {"command": l["cmd"], "subgoal": l["kind"],
                          "completed_at_tick": l["t_done"]},
            "is_error": False, "must_not_flag": True}


def gen_ask_progress(game, ro, about_sid=None, **_) -> dict:
    """'What have I done so far / what's on the plate?' — gt = completed subgoals + plate.
    With `about_sid` given, the question is TARGETED — 'did I already do <about_sid>?' — and
    the gt is whether that specific subgoal was completed, and when."""
    done = list(dict.fromkeys(l["kind"] for l in ro.log if l["status"] == "done"))
    if about_sid is not None:
        hits = [l for l in ro.log if l["kind"] == about_sid and l["status"] == "done"]
        return {"subtype": "ASK_PROGRESS_QA",
                "content": {"asking_about": "already_done", "subgoal": about_sid},
                "gt_answer": {"already_done": bool(hits),
                              "completed_at_tick": hits[-1]["t_done"] if hits else None},
                "is_error": False, "must_not_flag": True}
    cook = game.cooks[0]
    plate = None
    h = cook.holding
    if h is not None and hasattr(h, "contents"):
        seq = getattr(h, "content_seq", None) or [(i.name, i.state.value) for i in h.contents]
        plate = list(seq)
    return {"subtype": "ASK_PROGRESS_QA",
            "content": {"asking_about": "completed_and_plate"},
            "gt_answer": {"completed_subgoals": done, "plate": plate},
            "is_error": False, "must_not_flag": True}


def gen_announce_add_ingredient(game, ro, ingredient: str, state: str = "raw", **_) -> dict:
    """'Let's also add a <X> — on purpose.' [v2] The announcement IS the change: when it
    fires, the harness calls interact.system.apply_announced_change(event) and the goal/plan
    mutate directly. LEGAL announcement: the supervisor must NOT flag it — its correct move
    is to update its own estimated future subgoals in memory (tracking-judged)."""
    return {"subtype": "ANNOUNCE_PLAN_CHANGE",
            "content": {"change": "add_ingredient", "ingredient": ingredient, "state": state},
            "gt_answer": {"expected_update": {"op": "add_ingredient",
                                              "ingredient": ingredient, "state": state}},
            "is_error": False, "must_not_flag": True}


def gen_ask_if_done(game, ro, gt_missing=None, **_) -> dict:
    """'Am I done — can I serve this?' (the doubt behind a premature serve). gt = the dish is
    NOT complete plus the list of subgoals still missing (passed from the deviation record,
    since the truncated plan no longer contains them)."""
    return {"subtype": "ASK_IF_DONE",
            "content": {"asking_about": "dish_complete"},
            "gt_answer": {"complete": False, "missing_subgoals": list(gt_missing or [])},
            "is_error": False, "must_not_flag": True}


def gen_ask_which_next(game, ro, options=None, gt_next=None, **_) -> dict:
    """'Which one goes on next — A or B?' (the doubt behind a wrong-order swap). gt = the
    component the ordered recipe actually wants next."""
    return {"subtype": "ASK_WHICH_NEXT",
            "content": {"asking_about": "next_component", "options": list(options or [])},
            "gt_answer": {"next_component": gt_next},
            "is_error": False, "must_not_flag": True}


def gen_announce_reorder(game, ro, order=None, **_) -> dict:
    """'I'll do <order> first.' — announces a LEGAL reordering of the prep blocks. Purely
    informative: the plan was already permuted at compile time; nothing to answer, nothing
    to flag."""
    return {"subtype": "ANNOUNCE_PLAN_CHANGE",
            "content": {"change": "reorder_prep", "new_prep_order": list(order or [])},
            "gt_answer": None,
            "is_error": False, "must_not_flag": True}


GENERATORS = {
    "REQUEST_NEXT_STEP": gen_ask_next_step,
    "ASK_VISIBLE_STATE_QA": gen_ask_visible_state,
    "ASK_OBJECT_HISTORY": gen_ask_object_history,
    "ASK_PROGRESS_QA": gen_ask_progress,
    "ANNOUNCE_PLAN_CHANGE": gen_announce_add_ingredient,
    "ANNOUNCE_REORDER": gen_announce_reorder,
    "ASK_IF_DONE": gen_ask_if_done,
    "ASK_WHICH_NEXT": gen_ask_which_next,
}
