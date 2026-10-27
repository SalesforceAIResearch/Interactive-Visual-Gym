"""VirtualHome behavior library — the behavior-event taxonomy realized as named functions over the plan IR.

Mirrors cooksim/core/error_lib.py after its catalog surgery (hesitate deleted; wander subsumed by search;
recovery deferred; announced plan change mutates the GOAL so the changed plan counts as success), adapted
to VirtualHome objects and physics. Each function edits the SUBGOAL trajectory via plan_ir_vh transforms
and returns (new IR, [Deviation records]). Functions compose through compose() IN ORDER with the priority
rule: an earlier error owns the plan; a later error whose anchor it removed is skipped-with-record.

CHANNELS (every behavior belongs to exactly one):
  action_only        pure physics; nothing to say; a supervisor catches it only by WATCHING.
  language_optional  the error has a moment of doubt at its trigger boundary; takes ask: bool.
                       ask=True  -> the human PAUSES and asks (structured lang_event with engine ground
                                    truth). A correct, timely VLM answer AVERTS the error (record's
                                    `avertable` says which edit to undo); silence lets it happen.
                       ask=False -> silent; only proactive intervention could stop it.
  language_only      pure utterance; the physical plan is untouched or changed LEGALLY (the announcement
                     makes it legitimate). must_not_flag — flagging these is a false positive.

VH-specific physics notes (vs CookSim): there is no burn clock — safety errors leave PERSISTENT hazard
states (stove ON, fridge door OPEN, microwave running EMPTY) that simulate() reports as `hazards`;
MISIDENTIFY must substitute the look-alike into every LATER command that referenced the original object
(the human believes juice IS the milk, so they microwave the juice); refused engine actions (grab from a
closed fridge) waste the attempt and cascade exactly as the symbolic executor's precondition checks say.
"""
from __future__ import annotations
import copy
from typing import List, Tuple

from vhhome.plan import (Subgoal, Deviation, segment_plan, compile_ir, flat_steps, simulate,   # noqa: F401
                        skip, insert, swap, repeat, replace, truncate_done, permute, forget_rest,
                        wrong_grab_node, wrong_search_node, search_looks_node, stall_node,
                        device_on_node, put_wrong_node, sit_away_node, _idx)

IR = List[Subgoal]
Result = Tuple[IR, List[Deviation]]
PAUSE = 12                    # frames the human stands still while an utterance rides on the boundary


def _chan(recs, channel, asked=None, avertable=None):
    for r in recs:
        r.detail["channel"] = channel
        if channel == "language_optional":
            r.detail["asked"] = bool(asked)
            r.detail["avertable"] = avertable
    return recs


def lang_events_from(records) -> list:
    return [r.detail["lang_event"] for r in records
            if not r.skipped and isinstance(r.detail, dict) and "lang_event" in r.detail]


# =====================================================================================
# LANGUAGE-OPTIONAL — a correct answer averts the error
# =====================================================================================
def err_repeat_step(ir: IR, sid: str, at: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """REPEAT_COMPLETED_STEP — after finishing `at`, the human has lost track of having already done
    the strictly-earlier subgoal `sid` and goes back to re-do it (re-opens the fridge for milk already
    in hand, re-loads a loaded microwave...). Engine preconditions make the repeat partially refuse
    (grab fails: hands full / object moved) — realistic flailing, time lost, plan still completes.
    ask=True: at the trigger the human asks "did I already do <sid>?" (gt: yes, done); a correct answer
    CANCELS the duplicate — only the pause is lost."""
    ir, r = repeat(ir, sid, at)
    avert = {"undo": "cancel_insert", "target": r.anchor}
    r.event_subtype = "REPEAT_DUE_TO_MEMORY_ERROR" if ask else "REPEAT_COMPLETED_STEP"
    if ask:
        r.language_hook = "ASK_PROGRESS_QA"
        r.detail["lang_event"] = {"after": at, "subtype": "ASK_PROGRESS_QA", "pause": pause,
                                  "params": {"about_sid": sid}}
    return ir, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_premature_done(ir: IR, after: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """SKIP_PREREQUISITE (premature completion) — believes the task is finished after `after`: drops
    the remaining subgoals and walks off to the sofa. VH analogue of premature_serve; outcome is a
    SILENT FAIL (checkpoints unmet, human relaxing).
    ask=True: before leaving, asks "am I done?" (gt: NOT done + the missing subgoals); a correct answer
    RESTORES the dropped tail."""
    ir, r = truncate_done(ir, after=after)
    avert = {"undo": "restore_tail", "dropped": list(r.detail.get("dropped", []))}
    if ask:
        r.language_hook = "ASK_IF_DONE"
        r.detail["lang_event"] = {"after": after, "subtype": "ASK_IF_DONE", "pause": pause,
                                  "params": {"gt_missing": list(r.detail.get("dropped", []))}}
    return ir, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_forget_next_step(ir: IR, after: str, ask: bool = False, pause: int = PAUSE,
                         stall_frames: int = 60) -> Result:
    """FORGET_NEXT_STEP — loses the plan after `after`. Silent: stands there LOST (stall node) and the
    episode never finishes. The record keeps the restorable_tail + the specific forgotten next step —
    the hook a future VLM NEXT_STEP_HINT re-appends.
    ask=True: the human asks "what was I doing next?" — gt carries the forgotten step (the runtime can't
    know it: the tail is gone from the live plan, so the deviation record supplies it)."""
    ir, r = forget_rest(ir, after=after, stall_frames=stall_frames)
    avert = {"undo": "restore_tail", "dropped": list(r.detail.get("restorable_tail", []))}
    if ask:
        r.language_hook = "REQUEST_NEXT_STEP"
        r.detail["lang_event"] = {"after": after, "subtype": "REQUEST_NEXT_STEP", "pause": pause,
                                  "params": {"gt_next": r.detail.get("next_step")}}
    return ir, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_omit_subgoal(ir: IR, sid: str, ask: bool = False, pause: int = PAUSE,
                     pass_by: bool = False) -> Result:
    """OMISSION (merged wrong_progress_belief) — believes subgoal `sid` is already done and silently
    skips it. Downstream preconditions cascade realistically (skip fetch(milk) -> the deposit's PutIn
    is refused empty-handed -> silent fail; skip shut(fridge) -> door stays open = hazard).
    pass_by=True (visual-legibility, from the Gemini audit): an omission's evidence is an ABSENCE, so
    the human WALKS RIGHT PAST the omitted target without acting — the skip becomes filmable.
    ask=True: at the boundary before `sid` would have run, asks "did I already do <sid>?" (gt: NO);
    a correct answer REINSERTS the subgoal."""
    i0 = _idx(ir, sid)
    prev = ir[i0 - 1].sid if i0 > 0 else "__start__"
    dropped = copy.deepcopy(ir[i0])
    room = next((c[2] for c in dropped.commands if len(c) > 2), None)
    tgt = (dropped.target or dropped.component)
    ir2, r = skip(ir, sid)
    # @-indexed targets are fine here: _pick resolves 'tablelamp@3' like any token, and the
    # walk-past is exactly what makes a twin-instance omission (missing_step) FILMABLE — the
    # old '@ not in tgt' guard silently produced a pass-by-less deletion with nothing for
    # dev_live to anchor on, so the detection window could never open.
    if pass_by and tgt and room:
        pb = Subgoal(f"pass_by({tgt})", "waypoint", tgt,
                     [("Walk", tgt, room), ("Idle", 10)], origin="injected", deviated=True)
        pb.sid = pb.sid if all(n.sid != pb.sid for n in ir2) else pb.sid + "#2"
        ir2 = ir2[:i0] + [pb] + ir2[i0:]              # spliced where the skipped node would have run
        r.detail["pass_by"] = pb.sid
    avert = {"undo": "reinsert", "node_sid": sid, "after": prev}
    if ask:
        r.language_hook = "ASK_PROGRESS_QA"
        r.detail["lang_event"] = {"after": prev, "subtype": "ASK_PROGRESS_QA", "pause": pause,
                                  "params": {"about_sid": sid, "gt_done": False}}
    r.detail["dropped_node_commands"] = dropped.commands
    return ir2, _chan([r], "language_optional", asked=ask, avertable=avert)


def err_wrong_order(ir: IR, a: str, b: str, ask: bool = False, pause: int = PAUSE) -> Result:
    """WRONG_ACTION (ordering) — swaps two subgoals. On dependency-coupled pairs the engine cascades
    (deposit before its fetch -> refused empty-handed -> the object never moves); on independent pairs
    the swap is harmless (that's what dev_reorder is for — announced and legal).
    ask=True: asks "should I do <b> before <a>?" (gt: the correct order)."""
    ir2, r = swap(ir, a, b)
    if ask:
        r.language_hook = "ASK_WHICH_NEXT"
        r.detail["lang_event"] = {"after": None, "subtype": "ASK_WHICH_NEXT", "pause": pause,
                                  "params": {"options": [a, b], "gt_next": a}}
    return ir2, _chan([r], "language_optional", asked=ask, avertable={"undo": "swap_back", "pair": [a, b]})


# =====================================================================================
# ACTION-ONLY — pure physics; watch or miss it
# =====================================================================================
def err_misidentify_fetch(ir: IR, sid: str, wrong: str, room: str) -> Result:
    """MISIDENTIFY_OBJECT — at fetch subgoal `sid`, grabs the look-alike `wrong` instead, and BELIEVES
    it: every later command that referenced the original object is rewritten to the wrong one (the
    human microwaves the juice they think is milk). The original object never moves — silent fail."""
    node0 = ir[_idx(ir, sid)]
    orig = node0.component
    # keep the container window if the original fetch had one (see wrong_grab_node)
    opens = next((c for c in node0.commands if c[0] == "Open"), None)
    ir2, r = replace(ir, sid, wrong_grab_node(wrong, room,
                                              container=(opens[1] if opens else None),
                                              croom=(opens[2] if opens and len(opens) > 2 else None)))
    out = []
    for n in ir2:                                   # substitute the belief downstream
        n = copy.deepcopy(n)
        if n.origin == "plan" and any(len(c) > 1 and c[1] == orig for c in n.commands):
            n.commands = [tuple([c[0], wrong if (len(c) > 1 and c[1] == orig) else c[1]] + list(c[2:]))
                          if len(c) > 1 else c for c in n.commands]
            n.component = wrong if n.component == orig else n.component
            n.deviated = True
        out.append(n)
    r.event_subtype = "MISIDENTIFY_OBJECT"
    r.detail.update(orig=orig, wrong=wrong, substituted_downstream=True)
    return out, _chan([r], "action_only")


def err_wrong_container(ir: IR, before: str, container: str, room: str) -> Result:
    """CONFUSE_LOCATION_OR_DIRECTION — searches the wrong container/room first (opens it, finds
    nothing, closes it), then the plan resumes. Succeeds, +time."""
    ir2, r = insert(ir, wrong_search_node(container, room), before=before)
    r.event_subtype = "CONFUSE_LOCATION_OR_DIRECTION"
    return ir2, _chan([r], "action_only")


def err_search(ir: IR, before: str, stops, idle: int = 6) -> Result:
    """FAIL_TO_FIND_VISIBLE_OBJECT — hunts N wrong spots (possibly other rooms) while the target is
    in plain sight, then finds it. stops = [(obj, room), ...]. Succeeds, +time."""
    ir2, r = insert(ir, search_looks_node(stops, idle), before=before)
    r.event_subtype = "FAIL_TO_FIND_VISIBLE_OBJECT"
    r.detail["stops"] = list(stops)
    return ir2, _chan([r], "action_only")


def err_wrong_surface(ir: IR, sid: str, surf: str, room: str) -> Result:
    """WRONG_ACTION — replaces deposit subgoal `sid` with parking the object on the wrong surface
    (milk ON the kitchen table instead of IN the microwave). Later operate steps run on the empty
    device — silent fail + running_empty hazard."""
    node = ir[_idx(ir, sid)]
    ir2, r = replace(ir, sid, put_wrong_node(node.component, "kitchen", surf, room))
    r.event_subtype = "WRONG_ACTION"
    r.detail.update(should_have_been=node.target, put_on=surf)
    return ir2, _chan([r], "action_only")


# =====================================================================================
# SAFETY — persistent hazard states (VH's counterpart of smoke/fire)
# =====================================================================================
def err_leave_open(ir: IR, sid: str) -> Result:
    """ATTEMPT_UNSAFE_SHORTCUT — drops the trailing Close from fetch/deposit subgoal `sid`: walks off
    with the door WIDE OPEN and never returns. Task still completes; hazard `left_open` persists."""
    i = _idx(ir, sid)
    node = copy.deepcopy(ir[i])
    closes = [c for c in node.commands if c[0] == "Close"]
    if not closes:
        raise KeyError(f"err_leave_open: '{sid}' has no Close step to drop")
    node.commands = [c for c in node.commands if c[0] != "Close"]
    node.deviated = True
    # visual-legibility (Gemini audit): dwell a moment as she walks off, the door standing open in frame
    linger = Subgoal(f"door_left_open({closes[0][1]})", "stall", closes[0][1],
                     [("Idle", 12)], origin="injected", deviated=True)
    out = ir[:i] + [node, linger] + ir[i + 1:]
    r = Deviation("surgery", "ATTEMPT_UNSAFE_SHORTCUT", sid,
                  {"dropped_close": closes[0][1], "hazard": "left_open"})
    return out, _chan([r], "action_only")


def err_light_and_leave(ir: IR, after: str, dev: str = "stove", room: str = "kitchen") -> Result:
    """FAIL_TO_NOTICE_DANGER — lights a device (default: the STOVE, briefly considering another way to
    warm things), then continues the plan and never turns it off. Hazard `stove_on` persists to the end."""
    ir2, r = insert(ir, device_on_node(dev, room), after=after)
    r.event_subtype = "FAIL_TO_NOTICE_DANGER"
    r.detail["hazard"] = f"{dev}_on"
    return ir2, _chan([r], "action_only")


def err_leave_running(ir: IR, after: str, seat: str = "sofa", room: str = "livingroom") -> Result:
    """LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED — right after starting a device (`after` = the switch_on
    subgoal), walks out and sits down while it runs."""
    ir2, r = insert(ir, sit_away_node(seat, room), after=after)
    ir2[_idx(ir2, after)].deviated = True     # the risky act = STARTING it and leaving: window covers both
    r.event_subtype = "LEAVE_TIME_CRITICAL_OBJECT_UNATTENDED"
    return ir2, _chan([r], "action_only")


# =====================================================================================
# LANGUAGE-ONLY — legal; must NOT be flagged
# =====================================================================================
def dev_reorder(ir: IR, order, pause: int = PAUSE) -> Result:
    """REORDER_SUBGOALS (announced, legal) — permutes independent task blocks and SAYS so.
    The must-not-flag trap: an assistant tracking literal plan order raises a false alarm here."""
    ir2, r = permute(ir, order)
    r.language_hook = "ANNOUNCE_REORDER"
    r.detail["lang_event"] = {"after": None, "subtype": "ANNOUNCE_REORDER", "pause": pause,
                              "params": {"order": list(order)}, "must_not_flag": True}
    return ir2, _chan([r], "language_only")


def dev_announce_change(ir: IR, drop_sids, add_nodes, new_checkpoints, why: str,
                        pause: int = PAUSE) -> Result:
    """ANNOUNCE_PLAN_CHANGE (goal-mutating) — the human announces a change of goal, drops/adds subgoals
    accordingly, and the SUCCESS CRITERION CHANGES WITH IT (pass the returned `new_checkpoints` to
    simulate(..., checkpoints=...)): the announced altered outcome counts as success.
    e.g. appliance_hard: "actually I'll just drink it cold" -> drop deposit+switch_on, add drink(milk)."""
    recs = []
    out = ir
    for s in drop_sids:
        out, r0 = skip(out, s); recs.append(r0)
    anchor = out[-1].sid
    for n in add_nodes:
        out, r1 = insert(out, n, after=anchor); anchor = r1.anchor; recs.append(r1)
    for r0 in recs:                       # the announcement makes these edits LEGAL — never flaggable
        r0.detail["channel"] = "language_only"
        r0.event_subtype = "CHANGE_PLAN"
    rec = Deviation("announce_change", "CHANGE_PLAN", "+".join(drop_sids) or anchor,
                    {"why": why, "new_goal": True,
                     "lang_event": {"after": None, "subtype": "ANNOUNCE_PLAN_CHANGE", "pause": pause,
                                    "params": {"why": why}, "must_not_flag": True}},
                    language_hook="ANNOUNCE_PLAN_CHANGE")
    return out, _chan([rec], "language_only") + recs


# =====================================================================================
BEHAVIORS = {
  # name                      fn                     channel             family
  "repeat_step":              (err_repeat_step,      "language_optional", "C"),
  "premature_done":           (err_premature_done,   "language_optional", "A"),
  "forget_next_step":         (err_forget_next_step, "language_optional", "C"),
  "omit_subgoal":             (err_omit_subgoal,     "language_optional", "C"),
  "wrong_order":              (err_wrong_order,      "language_optional", "A"),
  "misidentify_fetch":        (err_misidentify_fetch, "action_only",      "B"),
  "wrong_container":          (err_wrong_container,  "action_only",       "B"),
  "search":                   (err_search,           "action_only",       "B"),
  "wrong_surface":            (err_wrong_surface,    "action_only",       "A"),
  "leave_open":               (err_leave_open,       "action_only",       "F"),
  "light_and_leave":          (err_light_and_leave,  "action_only",       "F"),
  "leave_running":            (err_leave_running,    "action_only",       "F"),
  "reorder":                  (dev_reorder,          "language_only",     "D"),
  "announce_change":          (dev_announce_change,  "language_only",     "D"),
}


def compose(ir: IR, calls) -> Result:
    """Apply behavior calls IN ORDER (priority rule). calls = [(name, kwargs), ...]. A call whose
    anchor an earlier behavior removed is skipped-with-record, never raised."""
    records = []
    for name, kw in calls:
        fn = BEHAVIORS[name][0]
        try:
            ir, recs = fn(ir, **kw)
        except KeyError as e:
            records.append(Deviation(name, "(not generated)", str(kw), {"reason": str(e)}, skipped=True))
            continue
        records.append(recs[0]) if len(recs) == 1 else records.extend(recs)
    return ir, records


# =====================================================================================
# CATALOG v2 — re-categorized so every class has a UNIQUE OBSERVABLE SIGNATURE.
# Timing/scope are PARAMETERS (variant=, gap=, duration=), not separate classes.
# Discarded as indistinct (per review): FAIL_TO_FIND_VISIBLE_OBJECT, CONFUSE_LOCATION_OR_DIRECTION,
# wrong_container, SKIP_PREREQUISITE-as-a-class (its only visible form IS a skipped step's
# consequences), wrong_order (sequencing itself is invisible; what a watcher sees is the skipped
# prerequisite). REPEAT_x2 merged into repetition(gap=...). HESITATE + FORGET merged into
# stall(duration=...). ABANDON merged into premature_termination. Safety trio merged into
# hazard_left_behind(variant=...).
# =====================================================================================
def loiter_node(stops, idle=10):
    """In-room 'hanging around': drifts between same-room spots with pauses and glances."""
    cmds = []
    for j, (obj, room) in enumerate(stops):
        cmds += [("Walk", obj, room), ("Idle", idle)]
        if j == 0:
            cmds += [("TurnLeft",), ("Idle", 4), ("TurnRight",), ("Idle", 4)]
    return Subgoal("loiter", "waypoint", None, cmds, origin="injected", deviated=True)


def err_wrong_object(ir: IR, sid: str, wrong: str, room: str) -> Result:
    """WRONG_OBJECT — signature: the OBJECT in hand/use is a look-alike; plan structure stays correct.
    The whole fetch subgoal (incl. any container open/close) is replaced as ONE unit — she goes
    straight to the wrong object — and the belief substitutes downstream."""
    ir2, recs = err_misidentify_fetch(ir, sid, wrong, room)
    recs[0].event_subtype = "WRONG_OBJECT"
    return ir2, recs


def err_wrong_placement(ir: IR, sid: str, surf: str, room: str) -> Result:
    """WRONG_PLACEMENT — signature: the right object ends up in the WRONG PLACE, and the error
    COMMITS RIGHT THERE (user decision 2026-08-26). The plan is TRUNCATED after the wrong
    placement: any later device op that depended on the deposit (switch on the microwave) is
    dropped, so she does NOT walk on and run the device empty. The takeover therefore releases
    at the placement itself, and recovery starts from "the milk is on the table"."""
    # Position of the deposit BEFORE the replacement: `replace` swaps a node in place, so this is
    # exactly where THIS transform's put_wrong node ends up. Scanning for "the first injected
    # node" instead was wrong under COMPOSITION — with an earlier error's node (a stall, a
    # detour) already in the plan, the scan found that one and truncated there, silently deleting
    # the rest of the plan including this transform's own placement (found by the composite
    # verifier, 2026-08-28).
    i0 = _idx(ir, sid)
    ir2, recs = err_wrong_surface(ir, sid, surf, room)
    recs[0].event_subtype = "WRONG_PLACEMENT"
    dropped = [n.sid for n in ir2[i0 + 1:]]
    ir2 = ir2[:i0 + 1]
    recs[0].detail["truncated_after"] = dropped
    return ir2, recs


def err_navigation_inefficiency(ir: IR, before: str, variant: str = "explore",
                                stops=None, idle: int = 8) -> Result:
    """NAVIGATION_INEFFICIENCY — signature: extra MOVEMENT with zero state change, then the plan
    resumes intact. variant='explore': multi-room detour through the given stops.
    variant='loiter': random hanging INSIDE the current room (same-room stops + pauses + glances)."""
    node = (search_looks_node(stops, idle) if variant == "explore" else loiter_node(stops, idle))
    ir2, r = insert(ir, node, before=before)
    r.event_subtype = "NAVIGATION_INEFFICIENCY"
    r.detail.update(variant=variant, stops=list(stops or []))
    return ir2, _chan([r], "action_only")


def err_skipped_step(ir: IR, sid: str, ask: bool = False, pass_by: bool = False) -> Result:
    """SKIPPED_STEP — signature: a required step is OMITTED (she walks right past its target), and the
    rest of the plan runs in order with the omission's visible consequences (device runs empty /
    door stays open / refused follow-ups)."""
    ir2, recs = err_omit_subgoal(ir, sid, ask=ask, pass_by=pass_by)
    recs[0].event_subtype = "SKIPPED_STEP"
    return ir2, recs


def err_missing_step(ir: IR, sid: str, ask: bool = False) -> Result:
    """MISSING_STEP — signature: a multi-instance goal left HALF done. The plan holds the same
    action for several identical objects (switch_off(tablelamp@1) and @3); one twin's subgoal is
    omitted ENTIRELY — she never goes to it (she forgot) — and everything else runs in order, so
    the forgotten twin's state stays wrong (one lamp lit) while its sibling got done. Mechanically the
    omission machinery, but a DISTINCT label: the assistant must notice a half-done pair, which
    is an instance-grounding judgement, not a plan-structure one."""
    ir2, recs = err_omit_subgoal(ir, sid, ask=ask, pass_by=False)
    recs[0].event_subtype = "MISSING_STEP"
    return ir2, recs


def err_repetition(ir: IR, sid: str, at: str = None, gap: str = "delayed",
                   ask: bool = False) -> Result:
    """REPETITION — signature: an already-completed subgoal is executed a SECOND time.
    gap='immediate': re-does it right after finishing it (double-check slip).
    gap='delayed': returns to it after intervening subgoals (memory lapse; ask variant available).
    REALIZATION (v3, per user review — the second execution must have a REAL physical effect;
    an engine-refused duplicate is invisible): toggle subgoals (switch_on/off, shut/open) are
    re-done as a SECOND PRESS that INVERTS the state (outcome-bearing — the tidied lamp is back
    on); deposit subgoals are re-done as UNLOAD+RELOAD (visible double state change, benign).
    Other types fall back to the legacy engine-refused duplicate."""
    from vhhome.plan import repeat_of, redo_toggle_node, reload_node, _uniq
    i = _idx(ir, sid)
    node = ir[i]
    if node.type in ("switch_on", "switch_off", "shut", "open"):
        dup, mode = redo_toggle_node(node, immediate=(gap == "immediate")), "toggle_redo"
    elif node.type == "deposit":
        dup, mode = reload_node(node), "reload"
    else:
        dup, mode = repeat_of(node), "refused_duplicate"
    dup.sid = _uniq(dup.sid, {n.sid for n in ir})
    j = i if gap == "immediate" else _idx(ir, at)
    out = ir[:j + 1] + [dup] + ir[j + 1:]
    r = Deviation("repeat", "REPETITION", dup.sid,
                  {"of": sid, "gap": gap, "mode": mode})
    avert = {"undo": "cancel_insert", "target": dup.sid}
    if ask:
        r.language_hook = "ASK_PROGRESS_QA"
        r.detail["lang_event"] = {"after": (at if gap != "immediate" else sid),
                                  "subtype": "ASK_PROGRESS_QA", "pause": PAUSE,
                                  "params": {"about_sid": sid}}
        return out, _chan([r], "language_optional", asked=True, avertable=avert)
    return out, _chan([r], "action_only")


def err_stall(ir: IR, after: str, duration: str = "terminal", frames: int = 40,
              ask: bool = False) -> Result:
    """STALL — signature: STOPS MOVING mid-plan (stationary; contrast with navigation_inefficiency
    which keeps moving). duration='transient': freezes, then resumes on her own.
    duration='terminal': never resumes (the forget case; ask variant carries the restorable tail)."""
    if duration == "transient":
        ir2, r = insert(ir, stall_node(frames), after=after)
        r.event_subtype = "STALL"
        r.detail["duration"] = "transient"
        return ir2, _chan([r], "action_only")
    ir2, recs = err_forget_next_step(ir, after, ask=ask, stall_frames=max(frames, 60))
    recs[0].event_subtype = "STALL"
    recs[0].detail["duration"] = "terminal"
    return ir2, recs


def err_premature_termination(ir: IR, after: str, ask: bool = False) -> Result:
    """PREMATURE_TERMINATION — signature: actively LEAVES the task and settles elsewhere (sofa) with
    the goal unmet (contrast with stall: departing vs freezing). Covers the old premature_done AND
    abandon."""
    ir2, recs = err_premature_done(ir, after, ask=ask)
    recs[0].event_subtype = "PREMATURE_TERMINATION"
    return ir2, recs


def err_hazard_left_behind(ir: IR, variant: str = "door_open", sid: str = None, after: str = None,
                           dev: str = "stove", room: str = "kitchen",
                           seat: str = "sofa", seat_room: str = "livingroom") -> Result:
    """HAZARD_LEFT_BEHIND — signature: a persistent hazardous/wasteful WORLD STATE stays behind her
    while she carries on. variant='door_open' (skips the close, door stands open),
    'device_on' (lights the stove and never returns), 'unattended_running' (starts the device and
    leaves the room while it runs)."""
    if variant == "door_open":
        ir2, recs = err_leave_open(ir, sid)
    elif variant == "device_on":
        ir2, recs = err_light_and_leave(ir, after, dev=dev, room=room)
    else:
        ir2, recs = err_leave_running(ir, after, seat=seat, room=seat_room)
    recs[0].event_subtype = "HAZARD_LEFT_BEHIND"
    recs[0].detail["variant"] = variant
    return ir2, recs


def dev_legal_reorder(ir: IR, order, pause: int = PAUSE) -> Result:
    ir2, recs = dev_reorder(ir, order, pause=pause)
    recs[0].event_subtype = "LEGAL_REORDER"
    return ir2, recs


BEHAVIORS_V2 = {
  # class                      fn                          channel              signature (unique observable)
  "wrong_object":            (err_wrong_object,            "action_only",
                              "the object in hand/use is a look-alike; plan structure correct"),
  "wrong_placement":         (err_wrong_placement,         "action_only",
                              "right object, wrong destination; later device ops run empty"),
  "navigation_inefficiency": (err_navigation_inefficiency, "action_only",
                              "extra movement, zero state change, plan resumes (explore | loiter)"),
  "skipped_step":            (err_skipped_step,            "language_optional",
                              "walks past a required step; visible downstream consequences"),
  "missing_step":            (err_missing_step,            "language_optional",
                              "one twin of a multi-instance goal forgotten; its state stays wrong"),
  "repetition":              (err_repetition,              "language_optional",
                              "a completed subgoal executed twice (immediate | delayed)"),
  "stall":                   (err_stall,                   "language_optional",
                              "stops moving mid-plan (transient resumes | terminal never does)"),
  "legal_reorder":           (dev_legal_reorder,           "language_only",
                              "announced dependency-legal order change — must NOT be flagged"),
  "goal_change":             (dev_announce_change,         "language_only",
                              "announced goal change — success criterion moves with it"),
}
for _n, (_f, _c, _s) in BEHAVIORS_V2.items():
    if _n not in BEHAVIORS:
        BEHAVIORS[_n] = (_f, _c, "v2")
