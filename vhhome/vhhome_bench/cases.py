"""Case definitions for the VH RECOVERY and E2E interaction experiments. Covers skipped_step,
wrong_object, and (added 2026-08-03) repetition + wrong_placement, the latter two requested as
a hard/easy pair: iv_rep_toggle_tidy mirrors detection case det3_rep_toggle_2, and
iv_wrongplace_appliance is a custom milk-to-microwave misplacement.
Port of CookSim gen_recovery_cases / cases_v2
anchoring, with LIVE window re-anchoring (walk durations vary between Unity runs, so dev/lo
are recorded at runtime like the step-in tools' dynamic windows — handover §5.7.5).

Each case carries: the deviated-plan compose calls, the sid anchors that define
  dev_sid  — the first deviated node (dev tick = its first command starting)
  id_step  — (sid, verb) whose COMPLETION opens the scoring window (identifiability)
  flag(before|after) — the sid whose completion is the scripted recovery flag point
plus flag_text, the repair-required predicate, and the per-episode final-state win test.
"""
import sys

from vhhome import plan as pir
from vhhome_bench import protocol as P
from vhhome import mistakes as lib

FIRST_REPLY = {   # scripted persona first replies — verbatim from the handover §4.1
    "novice_stepwise": "Oh no. Okay, okay — what should I do next? Please tell me step by step.",
    "creative_uninterruptible": "Fine, I heard you — no lecture. Just tell me what I should do next.",
}


def build_deviated(episode, calls, post=None):
    """Clean IR (waypoints stripped) + deviated IR + compiled flat steps + records.

    `post` is an optional case-local IR->IR fixup applied AFTER compose. It exists so a single
    case can repair an artifact of a shared transform without editing the transform itself
    (which would silently change every det3_* case built from it).
    """
    base = pir.segment_plan(episode)
    base = [n for n in base if not (n.origin == "plan" and n.type == "waypoint")]
    ir, records = lib.compose([n for n in base], [(n, dict(k)) for n, k in calls])
    if post:
        ir = post(ir)
    compiled = pir.compile_ir(ir)
    return base, ir, compiled, records


def approach_before(sid_target, obj, room, sid=None):
    """Insert an injected waypoint `Walk obj` immediately before sid_target.

    Needed because err_wrong_surface REPLACES the whole deposit node, and in appliance_hard that
    node owned the `Walk microwave`. Without this the agent would fire `SwitchOn microwave` while
    still standing at the kitchentable -- a zero-frame action (ENGINE_NOTES A8) that would abort
    the render. It is also what makes the error legible: she walks to the microwave EMPTY-HANDED
    and runs it with nothing inside.
    """
    def _post(ir):
        out = []
        for n in ir:
            if n.sid == sid_target:
                out.append(pir.Subgoal(sid=sid or f"approach({obj})", type="waypoint",
                                       component=obj, commands=[("Walk", obj, room)],
                                       target=obj, origin="injected"))
            out.append(n)
        return out
    return _post


# final-state win tests (game-rule content; success once the HOUSE is in the goal state)
def _target_in(sym, cls, container):
    """Is the TASK'S OWN instance of `cls` inside `container` now? Class-level fallback when the
    episode pins no target."""
    tid = (getattr(sym, "targets", None) or {}).get(cls)
    if tid is None:
        return sym.loc_cls.get(cls) == container
    return any(_cls(k) == cls and _iid(k) == tid and _cls(v) == container
               for k, v in sym.loc.items())            # instance-level: intentional


def final_ok(episode, sym):
    # Instance-graded episodes must be instance-graded HERE too. final_ok stayed class-level
    # after the checkpoints migrated, so a run that microwaved the DECOY milk reported
    # final_ok=True while its checkpoint "put that milk in microwave" said False — the win was
    # false and the audit caught the incoherence.
    if episode == "appliance_hard":
        return _target_in(sym, "milk", "microwave") and "microwave" in sym.on_cls
    if episode == "tidy_hard":
        # A true finish: NO tablelamp lit (both nightstand lamps off), fridge shut, tv off.
        return ("fridge" not in sym.opened_cls and "tablelamp" not in sym.on_cls
                and "tv" not in sym.on_cls)
    if episode == "settable_hard":
        return sym.loc_cls.get("book") == "coffeetable" and sym.loc_cls.get("plate") == "coffeetable"
    if episode == "transfer_hard":
        return _target_in(sym, "milk", "cabinet")
    cps = sym.checkpoints()
    return all(cps.values())


def node_effect_done(n, sym):
    """Is this CLEAN-plan node's effect already in the world? (pending_tail computation)"""
    if n.type == "waypoint":
        return True
    # compare BASE names: sym's class projections hold 'tablelamp', never 'tablelamp@1'. Comparing the raw
    # token made every numbered switch_off count as done from t0 and every numbered fetch never done (f113:
    # a repetition skipped its wait because the lamp steps looked finished; the recovery tail dropped them)
    import types as _t
    n = _t.SimpleNamespace(type=n.type, commands=n.commands,
                           component=(str(n.component).split("@")[0] if n.component else n.component),
                           target=(str(n.target).split("@")[0] if n.target else n.target))
    if n.type == "fetch":
        tgt = _deposit_target(n.component)
        return n.component in sym.holding_cls or (tgt and sym.loc_cls.get(n.component) == tgt)
    if n.type == "deposit":
        st = next((s for s in n.commands if s[0] in ("Put", "PutIn")), None)
        return bool(st) and sym.loc_cls.get(str(st[1]).split("@")[0]) == str(st[3]).split("@")[0]
    if n.type == "open":
        return n.target in sym.opened_cls
    if n.type == "shut":
        return n.target not in sym.opened_cls
    if n.type == "switch_on":
        return n.target in sym.on_cls
    if n.type == "switch_off":
        return n.target not in sym.on_cls
    if n.type == "drink":
        return any(e[0] == "drink_nearest" and e[1] == n.component for e in sym.events)
    if n.type in ("check", "inspect", "posture"):
        v = {"check": "go_to_nearest", "inspect": "look_at_nearest", "posture": "sit_nearest"}[n.type]
        return any(e[0] == v for e in sym.events)
    return False


_DEP_TGT = {}
def _deposit_target(obj):
    return _DEP_TGT.get(obj)


def pending_tail(clean_nodes, sym, exclude_sids=()):
    """Clean-plan commands still owed, template order, minus already-satisfied blocks and the
    repair anchor (the repair itself covers it). = CookSim pending_tail semantics."""
    global _DEP_TGT
    _DEP_TGT = {}
    for n in clean_nodes:
        if n.type == "deposit":
            st = next((s for s in n.commands if s[0] in ("Put", "PutIn")), None)
            if st:
                _DEP_TGT[str(st[1]).split("@")[0]] = str(st[3]).split("@")[0]
    out = []
    for n in clean_nodes:
        if n.sid in exclude_sids or node_effect_done(n, sym):
            continue
        out.extend([(tuple(s), n.sid) for s in n.commands])
    return out


# ---------------------------------------------------------------- the case table
def all_cases():
    C = []

    def case(name, episode, err, calls, dev_sid, id_step, flag_before_sid, flag_after_sid,
             flag_text, required, repair_note, post=None, reorder=None):
        err = P.LABEL_OF.get(err, err)          # graded label; the transform name stays in calls
        C.append(dict(name=name, episode=episode, err=err, calls=calls, dev_sid=dev_sid,
                      id_step=id_step, flag_before_sid=flag_before_sid,
                      flag_after_sid=flag_after_sid, flag_text=flag_text,
                      required=required, repair_note=repair_note, post=post,
                      reorder=reorder))

    # ---- RESTRAINT TRACK (skill contract §5.1/§7): the non-error family. err=None means no
    # injection ever fires; the assistant is scored on staying quiet (false alarms charged
    # against world truth) and on answering the exit probe. `reorder` is a LEGAL permutation
    # of the clean plan's nodes — a legitimate change of intent the assistant must not
    # "correct" (it holds the TRUE ORDER CONSTRAINTS saying any order is fine).
    case("iv_clean_appliance", "appliance_hard", None, [], dev_sid=None, id_step=None,
         flag_before_sid=None, flag_after_sid=None, flag_text=None,
         required=None, repair_note=None)
    case("iv_clean_tidy", "tidy_hard", None, [], dev_sid=None, id_step=None,
         flag_before_sid=None, flag_after_sid=None, flag_text=None,
         required=None, repair_note=None)
    case("iv_intent_reorder_tidy", "tidy_hard", None, [], dev_sid=None, id_step=None,
         flag_before_sid=None, flag_after_sid=None, flag_text=None,
         required=None, repair_note=None, reorder="reverse")
    case("iv_intent_rotate_tidy", "tidy_hard", None, [], dev_sid=None, id_step=None,
         flag_before_sid=None, flag_after_sid=None, flag_text=None,
         required=None, repair_note=None, reorder="rotate")

    # WRONG_OBJECT: she grabs the juice believing it is the milk; everything downstream uses it
    case("iv_wrongobj_appliance", "appliance_hard", "wrong_object",
         [["wrong_object", {"sid": "fetch(milk)", "wrong": "juice", "room": "kitchen"}]],
         dev_sid="wrong_grab(juice)", id_step=("wrong_grab(juice)", "Grab"),
         flag_before_sid="wrong_grab(juice)", flag_after_sid="switch_on(microwave)",
         flag_text="Wait — that's the JUICE you just grabbed, not the milk.",
         required={"grabbed": "milk", "not_in": ["juice", "microwave"]},
         repair_note="get the real milk (it is in the fridge); if the juice already went into "
                     "the microwave, take it out first")
    # SKIPPED_STEP: a required subgoal silently never happens (she walks right past its target)
    case("iv_skip_appliance", "appliance_hard", "skipped_step",
         [["skipped_step", {"sid": "fetch(milk)"}]],
         dev_sid=None, id_step=None,          # resolved to the successor subgoal below
         flag_before_sid=None, flag_after_sid="switch_on(microwave)",
         flag_text="Hold on — you never took the milk out of the fridge, but you're running "
                   "the microwave.",
         required={"grabbed": "milk", "not_in": None},
         repair_note="the milk is still in the fridge; the microwave must end up running WITH "
                     "the milk inside")
    case("iv_skip_tidy", "tidy_hard", "skipped_step",
         [["skipped_step", {"sid": "switch_off(tablelamp@1)"}]],
         dev_sid=None, id_step=None,          # resolved to the successor subgoal below
         flag_before_sid=None, flag_after_sid=None,
         flag_text="Hold on — you never switched off the first lamp; it's still lit.",
         required={"off": "tablelamp"},
         repair_note="the first lamp is still on; go back and switch it off")
    # REPETITION: a subgoal she already completed is redone, and the redo INVERTS it. This is the
    # recovery-experiment twin of det3_rep_toggle_2 (the hard detection case): after closing the
    # fridge she walks back to the bedroom and presses the lamp again, so the lamp she correctly
    # turned off ends up back ON. Outcome-bearing by construction -- see ARCHITECTURE section 3.
    # MISSING_STEP: she switches off the first nightstand lamp, walks right past its twin
    # without touching it, and carries on with the fridge and the tv — the second lamp stays
    # visibly lit. The assistant must notice a HALF-DONE pair of identical objects.
    case("iv_missing_lamp", "tidy_hard", "missing_step",
         [["missing_step", {"sid": "switch_off(tablelamp@3)"}]],
         dev_sid=None, id_step=None,          # resolved from the composed IR below
         flag_before_sid=None, flag_after_sid=None,
         flag_text="Hold on — one of the lights is still on: you switched off one lamp but the "
                   "other is still lit.",
         required={"off": "tablelamp"},
         repair_note="one lamp was never switched off; send her back to it")

    case("iv_rep_toggle_tidy", "tidy_hard", "repetition",
         [["repetition", {"sid": "switch_off(tablelamp@1)", "at": "switch_off(tablelamp@3)",
                          "gap": "delayed"}]],
         dev_sid="redo(switch_off(tablelamp@1))",
         id_step=("redo(switch_off(tablelamp@1))", "SwitchOn"),
         flag_before_sid="redo(switch_off(tablelamp@1))", flag_after_sid=None,
         flag_text="Hold on — you just switched the first lamp back ON. You already turned "
                   "that off earlier.",
         required={"off": "tablelamp"},
         repair_note="the first lamp is ON again; it must end up OFF")
    # WRONG_PLACEMENT: instead of depositing the milk in the microwave she sets it down on the
    # kitchentable — and the error COMMITS RIGHT THERE (user decision 2026-08-26). The transform
    # truncates the plan after the wrong placement, so she does NOT walk on to the microwave and
    # run it empty; the takeover releases at the placement and recovery starts from "the milk
    # is on the table". (Previously an approach_before() hook added the empty-microwave step.)
    case("iv_wrongplace_appliance", "appliance_hard", "wrong_placement",
         [["wrong_placement", {"sid": "deposit(milk)", "surf": "kitchentable",
                               "room": "kitchen"}]],
         dev_sid="put_wrong(milk)", id_step=("put_wrong(milk)", "Put"),
         flag_before_sid="put_wrong(milk)", flag_after_sid=None,
         flag_text="Hold on — you just set the milk down on the kitchen table. That's meant to "
                   "go inside the microwave.",
         required={"in": ["milk", "microwave"]},
         repair_note="the milk is sitting on the kitchentable; it must end up INSIDE the "
                     "microwave and the microwave must be running with it in there")
    # NAVIGATION_INEFFICIENCY: after taking the milk she wanders the house before going to
    # the microwave. Same episode as the wrong_placement case so the two are comparable.
    #
    # CAVEAT, by construction: this class has NO STATE TO REPAIR -- the plan resumes intact and
    # the goal is still reached, so `final_ok` is True whether or not anyone intervenes
    # (behaviour_lib marks it expect="success"). Recovery "success" is therefore degenerate
    # here; what is actually measurable is DETECTION and how much wandering got cut short.
    # `required` is the episode goal only, so required_met never manufactures a false failure.
    case("iv_nav_appliance", "appliance_hard", "navigation_inefficiency",
         [["navigation_inefficiency", {"before": "deposit(milk)", "variant": "explore",
                                       "stops": [["sofa", "livingroom"], ["bed", "bedroom"]]}]],
         dev_sid=None, id_step=None,          # filled in below from the composed IR
         flag_before_sid=None, flag_after_sid="switch_on(microwave)",
         flag_text="Hold on — you've got the milk, but you're wandering off. The microwave is "
                   "in the kitchen.",
         required={"in": ["milk", "microwave"]},
         repair_note="nothing is broken yet — she is just walking the wrong way with the milk; "
                     "get her back to the microwave")
    # STALL: she freezes mid-plan holding the milk, then resumes on her own. This is the sixth
    # error class and the only one that had no case, so "run every error type" could not be done.
    # `transient` (not `terminal`) keeps the plan intact afterwards, which matches the taxonomy:
    # stall is an error of WASTE, like navigation_inefficiency -- the outcome still lands.
    case("iv_stall_appliance", "appliance_hard", "stall",
         [["stall", {"after": "fetch(milk)", "duration": "transient", "frames": 60}]],
         dev_sid=None, id_step=None,          # filled in below from the composed IR
         flag_before_sid=None, flag_after_sid="switch_on(microwave)",
         flag_text="You've stopped moving — you've been standing still with the milk in your "
                   "hands for a while now.",
         required={"in": ["milk", "microwave"]},
         repair_note="nothing is broken — she has simply stopped; get her moving to the "
                     "microwave again")

    # OMISSION cases (missing_step/skipped_step): pass_by is off (user review 2026-08-25), so the
    # forgotten subgoal is dropped ENTIRELY — there is no injected pass-by node to anchor on. The
    # observable deviation is that she does the SUCCESSOR subgoal instead (walks to the fridge
    # rather than the second lamp); anchor dev/window on that successor's first command.
    for c in C:
        if c["err"] in ("missing_step", "skipped_step") and c["dev_sid"] is None:
            skipped_sid = c["calls"][0][1]["sid"]
            _b, _ir, _cmp, _r = build_deviated(c["episode"], c["calls"])
            clean_sids = [n_.sid for n_ in _b]
            succ = None
            if skipped_sid in clean_sids:
                i_ = clean_sids.index(skipped_sid)
                succ = _b[i_ + 1] if i_ + 1 < len(_b) else None
            if succ is not None:
                c["dev_sid"] = succ.sid
                c["flag_before_sid"] = succ.sid
                c["id_step"] = (succ.sid, succ.commands[0][0] if succ.commands else "Walk")
    for c in C:
        if c["name"] == "iv_stall_appliance" and c["dev_sid"] is None:
            _b, _ir, _cmp, _r = build_deviated(c["episode"], c["calls"])
            st = next((n for n in _ir if n.origin == "injected"), None)
            if st is not None:
                c["dev_sid"] = st.sid
                c["flag_before_sid"] = st.sid
                c["id_step"] = (st.sid, st.commands[0][0] if st.commands else "Idle")
    for c in C:
        if c["name"] == "iv_nav_appliance" and c["dev_sid"] is None:
            _b, _ir, _cmp, _r = build_deviated(c["episode"], c["calls"])
            det = next((n for n in _ir if n.origin == "injected"), None)
            if det is not None:
                c["dev_sid"] = det.sid
                c["flag_before_sid"] = det.sid
                c["id_step"] = (det.sid, det.commands[0][0] if det.commands else "Walk")
    return C


def _iid(tok):
    t = str(tok or "")
    return int(t.split("#", 1)[1]) if "#" in t else None


def _cls(tok):
    return str(tok or "").split("#")[0].split("@")[0]


def required_met(req, sym):
    """Has the repair actually landed?

    Instance-aware wherever the episode pinned a target. Left class-level this disagreed with
    the grading: fetching the DECOY milk satisfies "grabbed: milk" by name, so recovery would
    call itself resolved while final_ok stayed False and the run reported a repair that never
    happened.
    """
    # A composite case declares no single repair predicate: with several errors there is no one
    # thing a repair has to achieve, and the episode's own final-state test is the authority.
    # Returning False here means "not certified by a predicate" rather than crashing.
    if not req:
        return False
    tg = getattr(sym, "targets", None) or {}
    if req.get("grabbed"):
        o = req["grabbed"]
        tid = tg.get(o)
        if tid is not None:
            got = (any(_cls(k) == o and _iid(k) == tid
                       for k in sym.holding)            # instance-level: intentional
                   or any(e[0] == "pick_up_nearest" and e[1] == o and e[2] == tid
                          for e in sym.events))
        else:
            got = o in sym.holding_cls or any(e[0] == "pick_up_nearest" and e[1] == o
                                              for e in sym.events)
        if not got:
            return False
    ni = req.get("not_in")
    if ni and sym.loc_cls.get(ni[0]) == ni[1]:
        return False
    if req.get("closed") and req["closed"] in sym.opened_cls:
        return False
    if req.get("off") and req["off"] in sym.on_cls:          # repetition repair: toggle back off
        return False
    inn = req.get("in")                                   # wrong_placement repair: obj re-homed
    if inn:
        tid = tg.get(inn[0])
        if tid is not None:
            ok = any(_cls(k) == inn[0] and _iid(k) == tid and _cls(v) == inn[1]
                     for k, v in sym.loc.items())       # instance-level: intentional
        else:
            ok = sym.loc_cls.get(inn[0]) == inn[1]
        if not ok:
            return False
    return True

def composite_cases():
    """Cases from the COMPOSITE spec — two or more errors in one episode.

    Kept out of all_cases() so the single-error suite and its ledger stay exactly as they are.
    Each phase gets its own anchors, derived from the composed plan rather than authored:
      dev_sid  — for an omission, the SUCCESSOR subgoal she does instead (the omitted node is
                 gone, so there is nothing else to anchor on); otherwise the node the transform
                 created.
      id_step  — (that sid, the verb whose completion makes the error identifiable).
    """
    from vhhome_bench import cases_spec as SPEC
    OMIT = ("skipped_step", "missing_step")
    out = []
    for name, epk, calls, story in SPEC.COMPOSITE:
        clean = [n.sid for n in build_deviated(epk, [])[0]]
        full = build_deviated(epk, [(t, dict(k)) for t, k in calls])[1]
        full_sids = [n.sid for n in full]
        dev_sids, id_steps, prev = [], [], clean
        for i, (t, kw) in enumerate(calls):
            step_ir = build_deviated(epk, [(t2, dict(k2)) for t2, k2 in calls[:i + 1]])[1]
            step_sids = [n.sid for n in step_ir]
            added = [x for x in step_sids if x not in prev]
            prev = step_sids
            anchor = next((kw[k] for k in ("sid", "before", "after", "at") if k in kw), None)
            if t in OMIT:
                succ = next((x for x in clean[clean.index(anchor) + 1:] if x in full_sids), None)
                sid = succ
            else:
                sid = next((x for x in added if x in full_sids), None)
            node = next((n for n in full if n.sid == sid), None)
            verb = next((c[0] for c in (node.commands if node else []) if c[0] != "Walk"),
                        (node.commands[0][0] if node and node.commands else "Walk"))
            dev_sids.append(sid)
            id_steps.append((sid, verb))
        out.append(dict(name=name, episode=epk, err=P.LABEL_OF.get(calls[0][0], calls[0][0]),
                        errs=[P.LABEL_OF.get(t, t) for t, _ in calls],
                        calls=[[t, dict(k)] for t, k in calls],
                        dev_sid=dev_sids[0], dev_sids=dev_sids,
                        id_step=id_steps[0], id_steps=id_steps,
                        flag_before_sid=None, flag_after_sid=None, flag_text=None,
                        required=None, repair_note=None, post=None, reorder=None,
                        story=story))
    return out

def composite_clean_cases():
    """One CLEAN reference run per composite episode: the same task, nothing injected.

    This is the "what it looks like when it goes right" film. The suite deliberately does not
    film clean episodes during a sweep -- a no-error run is uninformative to score and the person
    works in silence -- but a reviewer reading an episode's design needs the correct completion
    beside the faulty one to see what the injected errors actually took away.

    err=None puts the rollout on its restraint path: the clean plan, no injection, no takeover.
    """
    from vhhome_bench import cases_spec as SPEC
    return [dict(name=f"{name}__clean", episode=epk, err=None, errs=[], calls=[],
                 dev_sid=None, dev_sids=[], id_step=None, id_steps=[],
                 flag_before_sid=None, flag_after_sid=None, flag_text=None,
                 required=None, repair_note=None, post=None, reorder=None, story=story)
            for name, epk, _calls, story in SPEC.COMPOSITE]


def clean_case_for(epk):
    """A zero-error reference case for ANY episode key.

    Used to prove a newly written task actually runs in its apartment: the symbolic twin only
    shows the plan is well formed, it cannot tell you the scene really contains a reachable
    toothbrush. Filming the clean run is what does.
    """
    return dict(name=f"{epk}__clean", episode=epk, err=None, errs=[], calls=[],
                dev_sid=None, dev_sids=[], id_step=None, id_steps=[],
                flag_before_sid=None, flag_after_sid=None, flag_text=None,
                required=None, repair_note=None, post=None, reorder=None,
                story=f"clean reference run of {epk}")
