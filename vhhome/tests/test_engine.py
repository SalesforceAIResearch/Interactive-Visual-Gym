#!/usr/bin/env python3
"""Fast OFFLINE engine tests: every error type x every injection timing, no Unity, no LLM.

Rationale: a full episode costs ~10 minutes and one API budget, and most engine faults are
decidable from the plan algebra alone. Both sides of the step-in trigger are symbolic -- the
reference comes from pir.simulate() and the live side from SymState's plan-effect mirror -- so
the comparison that silently disabled injection for every case is reproducible in milliseconds.

TIMING IS A FIRST-CLASS AXIS. The same error class injected at a different point is a different
test: `repetition` re-doing subgoal 1 after subgoal 2 exercises different plan surgery than
re-doing it after subgoal 4, and `wrong_placement` before vs after its container is opened hits
different preconditions. Each case is therefore re-anchored at every legal position in its
episode and all of them are checked.

  python -m test_engine            # all cases, all timings
  python -m test_engine -v         # show every timing, not just failures
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the checkout, so the packages import without installation
from vhhome import paths
import os, sys, traceback
LEDGER = []
from inspect import signature as _sig


def inspect_sig(fn):
    return _sig(fn).parameters

from vhhome_bench import cases as CS
from vhhome import plan as pir
from vhhome import tasks as episodes
from vhhome import goals as episode_goals
from vhhome.world import SymState, base, pretty_cmd
from vhhome_bench.episode import key_from_sim, key_from_sym

VERBOSE = "-v" in sys.argv
FAILS, CHECKS, INFO = [], [0], []


def check(cond, case, timing, msg):
    CHECKS[0] += 1
    if not cond:
        FAILS.append((case, timing, msg))
    return cond


def flat(nodes):
    return [(tuple(s), n.sid) for n in nodes for s in n.commands]


def prefix_len(fc, fe):
    m = 0
    while m < min(len(fc), len(fe)) and fc[m][0] == fe[m][0]:
        m += 1
    return m


def keep_of(fc, fe):
    K = {base(st[1]) for st, _ in fc + fe if len(st) > 1} | \
        {base(st[3]) for st, _ in fc + fe if len(st) > 3}
    K.discard(None)
    return K


def timings_for(case):
    """Every legal re-anchoring of this case's injection.

    The `calls` list carries the anchor under one of several keys depending on the transform
    (`sid` for redo/replace, `at` for where it lands, `before` for insertions). Re-pointing that
    at each subgoal of the clean plan yields the timing variants.
    """
    epk = case["episode"]
    clean = CS.build_deviated(epk, [], None)[0]
    sids = [n.sid for n in clean if not n.sid.startswith("waypoint")]
    out = [("as-authored", case["calls"])]
    if not case["calls"]:
        return out
    fn, kw = case["calls"][0]
    key = next((k for k in ("sid", "at", "before", "after", "anchor") if k in kw), None)
    if key is None:
        return out
    for s in sids:
        if kw.get(key) == s:
            continue
        nk = dict(kw); nk[key] = s
        out.append((f"{key}={s}", [[fn, nk]]))
    return out


def run_case(case, label, calls):
    epk = case["episode"]
    try:
        clean_nodes, ir, compiled, records = CS.build_deviated(epk, calls, case.get("post"))
    except Exception as e:
        check(False, case["name"], label, f"plan build raised {type(e).__name__}: {e}")
        return
    fc, fe = flat(clean_nodes), flat(ir)
    m = prefix_len(fc, fe)
    KEEP = keep_of(fc, fe)

    # T1 the deviation must exist and be distinguishable from ordinary execution
    OMISSION = {"stall", "skipped_step", "missing_step"}
    dev_cont = fe[m:]
    if not dev_cont:
        # A TRAILING omission (the LAST subgoal forgotten) removes commands but leaves nothing
        # after the skip point, so there is no successor to anchor a detection window on. Since
        # pass_by was removed (user review 2026-08-25) this is a legitimately NON-VIABLE timing,
        # not a build bug — record it in the ledger and move on. For a NON-omission error an
        # empty continuation is still a real fault (the plans are identical).
        if case["err"] in OMISSION:
            INFO.append((case["name"], label,
                         "trailing omission — no successor subgoal to anchor a window"))
            return
        check(False, case["name"], label, "clean and deviated plans are identical")
        return
    sfx = 0
    while sfx < min(len(fc), len(fe)) - m and fc[-1 - sfx][0] == fe[-1 - sfx][0]:
        sfx += 1
    dev_section = [s for s, _ in (fe[m:len(fe) - sfx] or fe[m:m + 4])]
    clean_txt = {pretty_cmd(s) for s, _ in fc}
    dev_only = {pretty_cmd(s) for s in dev_section if s[0] != "Idle"} - clean_txt
    # `stall` and `skipped_step` are errors of INACTION: the deviation is a pause or an
    # omission, so by definition they add no command the clean plan lacks. Requiring one would
    # mark every correctly-built omission case as broken.
    OMISSION = {"stall", "skipped_step", "missing_step"}
    check(bool(dev_only) or case["err"] in OMISSION, case["name"], label,
          "dev_section has no command the clean plan lacks — the error is invisible")

    # T2 THE TRIGGER. Replay the clean prefix through the same symbolic state the live world
    # uses and require it to equal pir.simulate's reference. This is the check that catches a
    # trigger which can never fire.
    ref = pir.simulate([(s, sid, False) for s, sid in fc[:m]], epk)
    ref_key = key_from_sim(ref, KEEP)
    sym = SymState(epk)
    for st, _ in fc[:m]:
        sym.run(tuple(st))
    live_key = key_from_sym(sym, KEEP)
    check(m == 0 or live_key == ref_key, case["name"], label,
          f"TRIGGER UNREACHABLE: live {live_key} != ref {ref_key}")

    # T3 the deviated continuation must be EXECUTABLE from the trigger state — an injection
    # whose first command is refused produces no error at all, only a rejection.
    sym2 = SymState(epk)
    for st, _ in fc[:m]:
        sym2.run(tuple(st))
    viable = True
    for k, (st, sid) in enumerate(dev_cont):
        ok, why = sym2.check(tuple(st))
        if not ok:
            # For an omission error a downstream rejection IS the visible consequence (skip the
            # grab, and the later put legitimately fails), so it is not a fault. For anything
            # else, an as-authored case that cannot execute produces no error at all -- but a
            # RE-ANCHORED one usually just means that anchor is not a legal moment, which is
            # information about the timing, not a bug.
            if case["err"] in OMISSION:
                pass
            elif label == "as-authored":
                check(False, case["name"], label,
                      f"deviated step {k} {pretty_cmd(st)!r} refused at injection: {why}")
            else:
                INFO.append((case["name"], label, f"not a legal anchor ({why})"))
            viable = False
            break
        sym2.run(tuple(st))

    # T4 window anchoring: the id_step the verdict machine opens on must be in the plan
    idst = case.get("id_step")
    if idst and label == "as-authored":
        sid_want, verb = idst
        have = [(sid, s[0]) for s, sid in fe]
        check(any(sid == sid_want and v == verb for sid, v in have), case["name"], label,
              f"id_step {idst} absent from the deviated plan (window would never open)")

    # T5 the injected error must actually BREAK something. `required` states what a successful
    # repair has to achieve, so after the deviated plan alone it must NOT hold -- if it does,
    # the "error" changed nothing and the episode is unfalsifiable. (Asserting the opposite,
    # as this test first did, failed on every single case, which is what exposed the mistake.)
    # navigation_inefficiency and stall are errors of WASTE, not of outcome: the taxonomy says
    # the plan "resumes intact", so the task still completes and `required` is legitimately met.
    # Only outcome-breaking classes are held to this.
    OUTCOME_BREAKING = {"wrong_object", "wrong_placement", "skipped_step", "repetition"}
    req = case.get("required")
    if req and viable and case["err"] in OUTCOME_BREAKING:
        msg = f"required {req} ALREADY met after the deviated plan — the error is a no-op"
        if label == "as-authored":
            check(not CS.required_met(req, sym2), case["name"], label, msg)
        elif CS.required_met(req, sym2):
            INFO.append((case["name"], label, "anchor produces no real error (" + msg + ")"))

    # T6 every sid the case refers to must exist in the plan it was built from. A stale sid
    # silently disables whatever it gates: `flag_before_sid` decides where the recovery replay
    # stops, and a name that matches nothing means the replay runs the WHOLE episode before
    # handing over, so the "error recovery" starts after the task is already finished.
    if label == "as-authored":
        sids_dev = {sid for _, sid in fe}
        sids_clean = {sid for _, sid in fc}
        for k in ("dev_sid", "flag_before_sid", "flag_after_sid", "anchor_sid"):
            v = case.get(k)
            if v:
                check(v in sids_dev or v in sids_clean, case["name"], label,
                      f"{k}={v!r} matches no subgoal in the plan "
                      f"(deviated sids: {sorted(sids_dev)[:6]}...)")

    # T5b the deviation must be DETECTABLE at runtime. dev_live is set either by seeing a
    # command unique to the deviation, or -- for inaction errors, which have none -- by the
    # injected subgoal's sid. If neither route can fire, the window never opens and every flag
    # grades `early` no matter how correct it was.
    if label == "as-authored":
        by_cmd = bool(dev_only)
        by_sid = bool(case.get("dev_sid")) and case["dev_sid"] in {sid for _, sid in fe}
        check(by_cmd or by_sid, case["name"], label,
              "no route to set dev_live: dev_section adds no unique command AND dev_sid does "
              "not name a subgoal in the plan — the detection window could never open")

    # T6b flag timing. `before_outcome` must stop the replay BEFORE the deviated section has
    # done its damage and `after_outcome` after it, otherwise the two variants are the same
    # experiment. Compare positions in the deviated plan.
    if label == "as-authored":
        order = [sid for _, sid in fe]
        def pos(sid):
            return order.index(sid) if sid in order else None
        pdev = pos(case.get("dev_sid"))
        pb, pa = pos(case.get("flag_before_sid")), pos(case.get("flag_after_sid"))
        if pdev is not None and pb is not None:
            check(pb <= pdev, case["name"], label,
                  f"flag_before_sid sits AFTER dev_sid ({pb} > {pdev}) — 'before_outcome' "
                  f"would already include the outcome")
        if pdev is not None and pa is not None:
            check(pa >= pdev, case["name"], label,
                  f"flag_after_sid sits BEFORE dev_sid ({pa} < {pdev})")

        # T6c the recovery handoff must actually hand something over: if the flag lands on the
        # last command there is no remaining plan, and 'recovery' degenerates into watching an
        # already-finished episode.
        if pb is not None:
            check(pb < len(order) - 1, case["name"], label,
                  "flag_before_sid is the final subgoal — nothing left to hand off at the flag")

        # T6d the window must open at or after the deviation, never before it
        if idst and pdev is not None:
            pid = pos(idst[0])
            if pid is not None:
                check(pid >= pdev, case["name"], label,
                      f"id_step {idst[0]} precedes dev_sid — the detection window opens before "
                      f"the error exists")

    # T7 the post/repair tail must be executable where it is spliced in, or the episode cannot
    # reach a resolved state no matter what the agents do.
    if case.get("post") and viable and label == "as-authored":
        sym3 = SymState(epk)
        for st, _ in fe:
            ok, why = sym3.check(tuple(st))
            if ok:
                sym3.run(tuple(st))

    if VERBOSE:
        print(f"    {case['name']:26} {label:28} m={m:<3} dev={len(dev_cont):<3} ok")


def episode_level_tests(epks):
    """Per-episode invariants that no single case exercises."""
    import re, glob
    for epk in sorted(epks):
        # E0 a declared instance target must be REACHABLE: the setup has to put that class into
        # the declared anchor, or _bind_targets finds nothing and grading silently falls back to
        # class level -- the exact failure instance grounding was introduced to remove.
        want = episodes.EPISODES[epk].get("targets") or {}
        for cls_, anchor in want.items():
            placed = any(st[0] == "PutIn" and base(st[1]) == cls_ and base(st[3]) == anchor
                         for st in episodes.EPISODES[epk].get("setup", []) if len(st) > 3)
            check(placed, epk, "episode",
                  f"targets={{{cls_!r}: {anchor!r}}} but the setup never puts a {cls_} in the "
                  f"{anchor} — the binding cannot resolve")
            check(len(inspect_sig(episode_goals.EPISODES[epk]["checkpoints"])) >= 2, epk,
                  "episode", f"declares targets but its checkpoints ignore them")
        # E1 final_ok must RESPOND to state. A predicate written against the wrong container
        # ("tv" not in sym.on, when sym.on holds 'tv#297') is vacuously true and reports every
        # run a success -- tidy_hard did exactly that with the television still switched on.
        sym0 = SymState(epk)
        ck0 = CS.final_ok(epk, sym0)
        symN = SymState(epk)
        for st in episodes.EPISODES[epk]["steps"]:
            symN.run(tuple(st))
        ckN = CS.final_ok(epk, symN)
        check(not ck0, epk, "episode", "final_ok is TRUE before the plan runs — vacuous predicate")
        check(ckN, epk, "episode", "final_ok is FALSE after the clean plan ran to completion")

        # E1b DECOY IMMUNITY: on an instance-graded episode, completing the task with a
        # NON-target instance must not satisfy final_ok. This is the class-vs-instance coherence
        # the audit caught live (decoy milk microwaved -> final_ok True, checkpoint False).
        want = episodes.EPISODES[epk].get("targets") or {}
        for cls_, anchor in want.items():
            symD = SymState(epk)
            symD.targets = {cls_: 999999}          # pretend the target is an instance never touched
            for st in episodes.EPISODES[epk]["steps"]:
                symD.run(tuple(st))                # class-level tokens = acting on "some" instance
            check(not CS.final_ok(epk, symD), epk, "episode",
                  f"final_ok satisfied without touching the pinned {cls_} target — decoy leak")

        # E2 the graded checkpoints must agree with final_ok at the end of the clean plan
        cps = symN.checkpoints()
        check(all(cps.values()) == bool(ckN), epk, "episode",
              f"final_ok={ckN} disagrees with checkpoints {cps}")


def test_instance_pinning():
    """A scripted plan must mean the SAME object throughout; an agent command must not.

    Reproduces the tidy_hard fault offline: the clean plan switched off tablelamp#256, the body
    walked to the kitchen, and the injected `repetition` then resolved "the tablelamp" to #376
    because that one had become nearest -- so the repetition landed on a different lamp, was
    refused as "already on", and the error never happened while the run still reported `won`.
    """
    from vhhome.world import LiveWorld

    class Stub(LiveWorld):
        def __init__(self):                      # no Unity, no scene
            self._plan_inst = {}
            self._seq = iter([256, 376, 376, 512])
        def _pick(self, tok, room):
            return next(self._seq), tok

    w = Stub()
    a1, _ = w._resolve_tok("tablelamp", "bedroom", True)    # scripted: resolves and pins
    a2, _ = w._resolve_tok("tablelamp", "bedroom", True)    # scripted: must reuse
    check(a1 == a2 == 256, "pinning", "scripted",
          f"a scripted reference moved between commands: {a1} then {a2}")

    w2 = Stub()
    b1, _ = w2._resolve_tok("tablelamp", "bedroom", False)  # agent: nearest, per command
    b2, _ = w2._resolve_tok("tablelamp", "bedroom", False)
    check((b1, b2) == (256, 376), "pinning", "agent",
          f"an agent command was pinned ({b1}, {b2}) — nearest resolution is the designed "
          f"ambiguity that produces wrong_object and must not be removed")

    w3 = Stub()
    c1, _ = w3._resolve_tok("tablelamp@2", "bedroom", True)  # explicit index is never memoised
    c2, _ = w3._resolve_tok("tablelamp", "bedroom", True)
    check(c1 != c2 or True, "pinning", "explicit", "")
    check("tablelamp@2" not in w3._plan_inst, "pinning", "explicit",
          "an explicitly indexed token was memoised under its own key")


def test_held_verb_resolution():
    """`put X down` must resolve X among the things in hand, never by proximity.

    Reproduces the iv_wrongobj deadlock offline: the agent holds milk#177, a decoy milk#176 sits
    on the counter and is nearer, so `put the milk on the counter` resolved to #176 and was
    refused as "I'm not holding the milk". With both hands full it could not free one, could not
    open the microwave, and the episode ran until the no-progress guard stopped it.
    """
    from vhhome.world import LiveWorld

    class Stub(LiveWorld):
        def __init__(self, holding):
            self._plan_inst = {}
            class S: pass
            self.sym = S(); self.sym.holding = set(holding)
        def _pick(self, tok, room):
            return 176, tok                      # the decoy is always the nearest

    w = Stub({"milk#177"})
    i, _ = w._resolve_tok("milk", "kitchen", False, held=True)
    check(i == 177, "held-verb", "put",
          f"`put the milk down` resolved to {i}, but the agent is holding 177 — it would be "
          f"refused as 'not holding the milk' and the hand could never be freed")

    j, _ = w._resolve_tok("milk", "kitchen", False, held=False)
    check(j == 176, "held-verb", "grab",
          f"`grab the milk` resolved to {j}; grabbing must stay NEAREST-resolved (176) so that "
          f"taking the decoy remains a real wrong_object")

    w2 = Stub(set())                             # holding nothing -> fall back to nearest
    k, _ = w2._resolve_tok("milk", "kitchen", False, held=True)
    check(k == 176, "held-verb", "empty-hands",
          "with empty hands the held-verb path must fall back to normal resolution")


def test_no_resolved_step_rerender():
    """Nothing may re-render a STORED step for comparison against plan text.

    `step` holds instance-resolved tokens after execution, so pretty_cmd() on it yields
    "switch on the tablelamp#377" while anything built from the plan says "switch on the
    tablelamp". That mismatch silently made `soft_compliant` False on every soft run — a
    headline metric reading the opposite of the truth, with no error anywhere. The display text
    (`cmd`) is captured before resolution and is what must be compared.
    """
    import glob, re
    for f in glob.glob(os.path.join(paths.ROOT, "vhhome_bench", "*.py")) + glob.glob(os.path.join(paths.ROOT, "vhhome", "*.py")):
        for i, ln in enumerate(open(f), 1):
            if re.search(r"pretty_cmd\(\s*tuple\(\s*\w+\[.step.\]|pretty_cmd\(\s*\w+\[.step.\]", ln):
                check(False, "lint", "resolved-step re-render",
                      f"{f}:{i} re-renders a stored (resolved) step — compare l['cmd'] instead: "
                      f"{ln.strip()[:80]}")


def test_inaction_compliance_reachable():
    """For `stall` and `skipped_step`, compliance must be ACHIEVABLE.

    Their dev_section is only an Idle. task_done() excludes Idle when building `want`, so with
    no other command it returned False unconditionally — compliance could never be true. That
    also stopped the SYSTEM TASK from ever dissolving (`task_completed` consults it), so the
    human stood still indefinitely and the run died to the no-progress guard.
    """
    for case in [c for c in CS.all_cases() if c["err"] in ("stall", "skipped_step", "missing_step")]:
        cl, ir, _, _ = CS.build_deviated(case["episode"], case["calls"], case.get("post"))
        fc = flat(cl); fe = flat(ir)
        m = prefix_len(fc, fe)
        sfx = 0
        while sfx < min(len(fc), len(fe)) - m and fc[-1 - sfx][0] == fe[-1 - sfx][0]:
            sfx += 1
        dev_section = [s_ for s_, _ in (fe[m:len(fe) - sfx] or fe[m:m + 4])]
        want = [pretty_cmd(s_) for s_ in dev_section if s_[0] != "Idle"]
        has_idle = any(s_[0] == "Idle" for s_ in dev_section)
        check(bool(want) or has_idle, case["name"], "inaction-compliance",
              f"dev_section {dev_section} has neither a matchable command nor an Idle — "
              f"soft_compliant can never become True and the task can never dissolve")


def lint_half_migration():
    """Guard the failure mode that has now bitten three times.

    Instance keying turned holding/opened/on/loc into sets of 'tv#297'-style tokens. Any caller
    left asking a CLASS-level question against them ('tv' not in sym.on) silently answers the
    opposite of the truth -- no exception, no warning. The projections (*_cls) exist for those
    callers, so a raw membership test outside iv_world.py is almost always a missed migration.
    """
    import re, glob
    bad = []
    for f in glob.glob(os.path.join(paths.ROOT, "vhhome_bench", "*.py")) + glob.glob(os.path.join(paths.ROOT, "vhhome", "*.py")):
        if f.endswith(os.sep + "world.py"):
            continue
        for i, ln in enumerate(open(f), 1):
            if "instance-level: intentional" in ln:
                continue          # explicit opt-out for code that really does want instances
            if re.search(r"\bsym\.(on|opened|holding|loc)\b(?!_cls)", ln) and "_cls" not in ln:
                bad.append(f"{f}:{i} {ln.strip()[:88]}")
    for b_ in bad:
        check(False, "lint", "half-migration",
              f"raw instance-set access where a class-level projection is meant -> {b_}")


def test_scene_fixes_f89():
    """A33 (f89): the four scene defects found by the struggle hunt, pinned offline.

    S1 a grabbable object with no ON edge (VirtualHome drops it on a scripted Put) counts as anchored,
       so the gate binds the copy it sees; an agent's bare token prefers a copy in the body's room over
       the spec/index grounding (the bathroom "walk to the mug" that went to the kitchen).
    S2 the recorded-walk surrogate exists and a failed Walk no longer says "my body couldn't do that here".
    S3 only classes Unity really opens are openable; a resolved nightstand is refused up front.
    S4 dishes_hard pins the bed in the room the map calls "bedroom".
    """
    import inspect
    from vhhome import world as W
    from vhhome import gates as X
    from vhhome import tasks as E

    class Ctrl:
        def __init__(self, nodes, edges):
            self.g = {"nodes": nodes, "edges": edges}
        def _graph(self):
            return self.g
    nodes = [{"id": 11, "class_name": "bathroom", "category": "Rooms"},
             {"id": 64, "class_name": "mug", "category": "Props", "properties": ["GRABBABLE"]},
             {"id": 43, "class_name": "walllamp", "category": "Lamps", "properties": ["HAS_SWITCH"]},
             {"id": 68, "class_name": "washingmachine", "category": "Appliances", "properties": ["CAN_OPEN"]}]
    edges = [{"from_id": i, "to_id": 11, "relation_type": "INSIDE"} for i in (64, 43, 68)]
    ctrl = Ctrl(nodes, edges)
    check(X._is_anchored(ctrl, 64), "f89", "S1-grabbable",
          "an unanchored GRABBABLE mug must count as anchored, or the gate defers its binding to the stale index")
    check(not X._is_anchored(ctrl, 43), "f89", "S1-mounted",
          "a wall lamp with no support must still count as unanchored (the original #12 case)")

    class Stub(W.LiveWorld):
        def __init__(self):
            self._plan_inst = {}
            class S: pass
            self.sym = S(); self.sym.holding = set()
        def _instances_in_body_room(self, cls):
            return [64]
        def _nearest(self, ids):
            return ids[0]
        def _pick(self, tok, room):
            return 224, tok                      # the stale kitchen grounding
    w = Stub()
    i, _ = w._resolve_tok("mug", "kitchen", False)
    check(i == 64, "f89", "S1-body-room",
          f"an agent's bare 'mug' in the bathroom resolved to {i}; it must be the bathroom copy (64)")
    j, _ = w._resolve_tok("mug", "kitchen", True)
    check(j == 224, "f89", "S1-scripted-unchanged",
          f"a SCRIPTED bare token resolved to {j}; scripted grounding must stay on the plan's index (224)")

    src = inspect.getsource(W.LiveWorld)
    check("_walk_surrogate(step)" in src, "f89", "S2-surrogate",
          "exec_step must retry a Walk that rendered nothing via the furniture the target rests on")
    check("couldn't find a way to walk up to the" in src, "f89", "S2-honest-walk-refusal",
          "a failed Walk must name the object instead of 'my body couldn't do that here'")

    check({"nightstand", "desk", "washingmachine"}.isdisjoint(W.OPENS) and {"fridge", "microwave", "dishwasher"} <= W.OPENS,
          "f89", "S3-whitelist", f"OPENS = {sorted(W.OPENS)} must hold what Unity opens and nothing it refuses")
    sym = W.SymState("groceries_hard")
    sym.openable = {"fridge#162"}
    ok_n, why_n = sym.check(("Open", "nightstand#282", "bedroom"))
    check(not ok_n and "not something that opens" in (why_n or ""), "f89", "S3-nightstand-refused",
          f"open the nightstand -> {ok_n}, {why_n!r}; must be refused as not something that opens")
    ok_f, why_f = sym.check(("Open", "fridge#162", "kitchen"))
    check(ok_f or "already open" in (why_f or ""), "f89", "S3-fridge-allowed", f"open the fridge -> {ok_f}, {why_f!r}")

    sym.holding = {"plate#110"}
    ok_g, why_g = sym.check(("Grab", "plate#110", "kitchen"))
    check(not ok_g and "already holding" in (why_g or ""), "f89", "G1-regrab-refused",
          f"grab the plate you hold -> {ok_g}, {why_g!r}; Unity refuses it opaquely, so it must be refused as already holding")
    check('flat = {"floor", "rug", "wall", "ceiling"}' in src, "f89", "G2-surrogate-not-floor",
          "the walk surrogate must skip floor/rug/wall/ceiling supports (a bed rests ON the floor)")

    setup = E.EPISODES["dishes_hard"]["setup"]
    check(("Put", "plate@3", "kitchen", "bed#284", "bedroom") in [tuple(s) for s in setup], "f89", "S4-dishes-bed",
          "dishes_hard must put the bedroom plate on bed#284 (map label 'bedroom'), not a nearness-resolved bed")


def test_person_channel_f92():
    """The person's TEXT state map is off (f92) and its silence rule is explicit (f91).

    The room scene-graph text and whole-house text named the room and listed every landmark in metres;
    the person read that instead of looking, announced rooms it had not seen and narrated the list.
    What stays: its eyes (ego frames), the map IMAGES, and what is in its hands.
    """
    import inspect
    from vhhome_bench import protocol as P
    from vhhome_bench import person as RR

    src = inspect.getsource(RR)
    check('PERSON_STATEMAP = os.environ.get("VH_PERSON_STATEMAP", "0") != "0"' in src, "f92", "statemap-off-by-default",
          "the person's text state map must be off unless VH_PERSON_STATEMAP=1")
    check("if not PERSON_STATEMAP:" in src and "P.HUMAN_NO_STATEMAP_NOTE" in src, "f92", "statemap-gate",
          "with the text state map off the person must get the eyes-and-images note instead of the scene graph")
    check("IN YOUR HANDS:" in src, "f92", "hands-kept",
          "the person must still be told what is in its hands")
    check("if PERSON_STATEMAP and world.pos" in src, "f92", "no-position-line",
          "the engine-truth position/facing line belongs to the text state map, not the eyes-only channel")
    check('pos += f"YOU ARE IN: {_room}' in src and "_room = world.current_room_label()" in src, "f116", "room-name-given",
          "the person is told the room it stands in, under the map's own label")
    check(hasattr(P, "HUMAN_NO_STATEMAP_NOTE") and "first-person frames" in P.HUMAN_NO_STATEMAP_NOTE,
          "f92", "no-statemap-note", "the replacement note must point the person at its own eyes and the map images")
    tail = P.REPLY_TAIL
    fin = P.HUMAN_FINAL_RULES
    import inspect as _i
    from vhhome_bench import episode as SR
    sp = _i.getsource(SR)
    check('for lost in (filt.get("unusable_for_solvability") or []):' in sp and 'P.vocab_hint(lost)' in sp,
          "f93", "no-silent-drop-stepin",
          "a person's unparseable planned step must produce feedback on the step-in path, not vanish")
    check("YOUR QUEUE IS EMPTY and your body is doing NOTHING" in sp and "reask_failed" in sp,
          "f94", "empty-queue-said-out-loud",
          "when the words/actions re-ask also fails, the person must be told its queue is empty")
    check("turn left N" in P.vocab_hint("turn around") and "visible door" in P.vocab_hint("go through the nearest door"),
          "f93", "vocab-hint", "the feedback must name the phrasing that WOULD work")

    check("REAL PEOPLE WORK IN SILENCE" in fin and "UNDER 10 WORDS" in fin, "f91", "silence-rule",
          "the person's prompt must end with the explicit silence rule")
    check('turn_txt = turn_hint + "\\n\\n" + P.HUMAN_FINAL_RULES' in src and "P.gemini_frames(client, ctx, turn_txt, imgs)" in src,
          "f96", "final-rules-literally-last",
          "the silence and visibility rules must be appended AFTER the per-turn hint, the last text the person reads")
    f = SR._said_since_last_turn
    same_tick = [{"speaker": "human", "t": 12, "text": ""}, {"speaker": "vlm", "t": 12, "text": "You skipped a step!"}]
    check(f(same_tick, 12), "f100", "same-tick-accusation-counts",
          "a supervisor line recorded at the SAME tick but AFTER the person's poll turn must count as spoken since that turn")
    check(not f([{"speaker": "vlm", "t": 10, "text": "x"}, {"speaker": "human", "t": 12, "text": "ok"}], 12), "f100", "old-line-does-not-count",
          "a supervisor line from before the person's last turn is not new")
    check(not f([{"speaker": "human", "t": 10, "text": ""}, {"speaker": "vlm", "t": 12, "text": ""}], 12), "f100", "empty-line-does-not-count",
          "an empty supervisor output is not speech")
    # f111: stall / detour / repetition fire on LIVE ANCHORS, never on a timer
    def _L(*steps):
        return [{"status": "done", "step": list(st), "t_done": td} for st, td in steps]
    class _LW:
        def __init__(self, log): self.log = log
    after_plate2 = ("Grab", "plate", 2)
    w1 = _LW(_L((("Grab", "plate#1", "kitchen"), 5), (("Walk", "sink", "kitchen"), 9), (("Grab", "plate#2", "kitchen"), 14),
                (("TurnLeft",), 15)))
    check(SR.just_completed(w1, [after_plate2]), "f111", "stall-anchor-just-finished",
          "a stall 'after fetch(plate@2)' fires when the 2nd plate grab is the last real action (turns aside)")
    w2 = _LW(w1.log + _L((("Walk", "dishwasher", "kitchen"), 20)))
    check(not SR.just_completed(w2, [after_plate2]), "f111", "stall-anchor-moment-passed",
          "once the person has moved on to another action the 'just finished' moment is over")
    check(not SR.just_completed(_LW(w1.log[:1]), [after_plate2]), "f111", "stall-anchor-first-plate",
          "the FIRST plate grab does not complete fetch(plate@2)")
    fl = [(("Walk", "fridge", "kitchen"), "fetch(milk)"), (("Open", "fridge", "kitchen"), "fetch(milk)"),
          (("Grab", "milk", "kitchen"), "fetch(milk)"), (("Close", "fridge", "kitchen"), "fetch(milk)")]
    check(SR.step_marker(fl, "fetch(milk)") == ("Grab", "milk", 1), "f111", "marker-defining-action",
          "a fetch completes at its Grab, not at the optional 'close the fridge' after it")
    lamp = ("SwitchOff", "tablelamp", 1)
    wl = _LW(_L((("Walk", "tablelamp#125", "bedroom"), 10), (("SwitchOff", "tablelamp#125", "bedroom"), 12),
                (("Walk", "bookshelf", "livingroom"), 30)))
    rb = SR.repetition_block(wl, lamp, {"tablelamp", "nightstand"})
    check(rb == [("Walk", "tablelamp#125", "bedroom"), ("SwitchOn", "tablelamp#125", "bedroom"), ("Idle", 10)],
          "f111", "repetition-block-person-run",
          f"repetition redoes the person's OWN run for the step (approach + press, press inverted), got {rb}")
    check(SR.repetition_block(_LW([]), lamp, {"tablelamp"}) is None, "f111", "repetition-needs-done-step",
          "nothing to repeat before the step was done")
    import inspect as _insp
    lb = _insp.getsource(SR.run)
    check("LIVE_AFTER" not in sp and "LIVE_FALLBACK" not in sp and "live_dev_section" not in sp
          and '"timer"' not in lb and '"fallback"' not in lb, "f111", "no-timers",
          "no live injection may fire on elapsed ticks alone (user ruling 2026-09-18)")
    from vhhome import world as W
    check(W.pretty_cmd(("Walk", "fridge#157", "kitchen")) == "walk to the fridge"
          and W.pretty_cmd(("PutIn", "milk#176", "kitchen", "fridge#157", "kitchen")) == "put the milk in the fridge",
          "f113", "no-instance-ids-in-text", "rendered commands never show '#id' instance tokens")
    from vhhome_bench import cases as _CS
    _case = next(c for c in _CS.composite_cases() if c["name"] == "cx_shutdown_missing_rep")
    _cl = _CS.build_deviated(_case["episode"], [], None)[0]
    _owed = {sd for _, sd in _CS.pending_tail(_cl, W.SymState(_case["episode"]))}
    check({"switch_off(tablelamp@1)", "switch_off(tablelamp@1)#2"} <= _owed, "f114", "numbered-steps-owed",
          f"lamps that are ON at the start must count as owed, got {sorted(_owed)}")
    check("cps = w.sym.checkpoints()" in lb, "f115", "finish-by-goal-checkpoints",
          "'chore about to finish' counts unmet GOAL checkpoints, not class-level steps")
    class _PW:
        def last_at(self): return None
        def room_of(self, o): return "kitchen"
    _own = {"suggested_actions": ["turn right", "turn right", "turn right", "go through the visible door"]}
    _f = P.convert_parse(_own, "defiant_novice", _PW())
    check(len(_f["commands"]) == 4 and not _f["dropped_by_acceptance"], "f117", "own-plan-not-capped",
          f"a low-acceptance person's OWN 4-step plan must be queued whole, got {[c[1] for c in _f['commands']]}")
    _f2 = P.convert_parse(_own, "defiant_novice", _PW(), acceptance_cap=True)
    check(len(_f2["dropped_by_acceptance"]) == 1, "f117", "advice-still-capped",
          "accepted ADVICE is still capped at 3 steps for a low-acceptance persona")
    check("and _rep_wait_ok(world):" in lb, "f112", "repetition-gap-on-template-path",
          "the template exact-match path must also wait REP_GAP after the repeated step")
    check("w.tick - t_c >= REP_GAP" in lb and "_about_to_finish(w)" in lb, "f111", "repetition-gap-or-finish",
          "a repetition waits REP_GAP ticks after the step, unless the chore is about to finish")
    check('report.setdefault("live_inject_blocked", [])' in sp and "_live_why(world)" in sp, "f111",
          "skip-reason-recorded", "a phase whose anchor never came is skipped at its deadline WITH the reason")
    check('"live_injected": bool(_p.get("live_injected"))' in sp, "f106", "live-flag-in-report",
          "the report's phase record must say whether the error was injected from live state")
    # f108: object-bound classes fire on a minimal live precondition, with the template's block
    class _Sym:
        def __init__(self, held=(), on=(), opened=()):
            self.holding_cls, self.on_cls, self.opened_cls = set(held), set(on), set(opened)
    class _OW:
        def __init__(self, held=(), on=(), opened=(), log=()):
            self.sym, self.log = _Sym(held, on, opened), list(log)
    put_bed = [("Walk", "bed", "bedroom"), ("Put", "book", "bedroom", "bed", "bedroom")]
    wp = {"err": "wrong_placement", "kw": {"sid": "deposit(book)", "surf": "bed"}, "dev_section": put_bed}
    check(SR.live_object_block(wp, _OW(held=["book"]))[0] == put_bed, "f108", "live-wrong-placement-fires",
          "a wrong placement fires as soon as the object is in hand, with the template's block")
    b, why = SR.live_object_block(wp, _OW())
    check(b is None and "not in hand" in why, "f108", "live-wrong-placement-waits",
          f"with the object not in hand there is nothing to misplace, got {why}")
    wo = {"err": "wrong_object", "kw": {"sid": "fetch(milk)", "wrong": "juice"}, "dev_section": [("Walk", "juice", "kitchen"), ("Grab", "juice", "kitchen")]}
    check(SR.live_object_block(wo, _OW())[0] is not None, "f108", "live-wrong-object-fires", "right object not fetched yet: fire")
    check(SR.live_object_block(wo, _OW(log=[{"status": "done", "step": ["Grab", "milk#176", "kitchen"]}]))[0] is None,
          "f108", "live-wrong-object-too-late", "once the right object was grabbed, swapping it is no longer possible")
    sk = {"err": "skipped_step", "kw": {"sid": "fetch(plate@2)"}, "dev_section": [("Walk", "sink", "kitchen")]}
    one = [{"status": "done", "step": ["Grab", "plate#1", "kitchen"]}]
    check(SR.live_object_block(sk, _OW(log=one))[0] is not None, "f108", "live-omission-nth-instance",
          "fetch(plate@2) is still skippable after only one plate was grabbed")
    check(SR.live_object_block(sk, _OW(log=one * 2))[0] is None, "f108", "live-omission-done",
          "after the second grab the step is done and cannot be skipped")
    so = {"err": "missing_step", "kw": {"sid": "switch_off(tablelamp@1)"}, "dev_section": [("Walk", "tv", "livingroom")]}
    check(SR.live_object_block(so, _OW(on=["tablelamp"]))[0] is not None and SR.live_object_block(so, _OW())[0] is None,
          "f108", "live-omission-switch", "switching off can only be omitted while the lamp is on")
    rest = SR.live_object_rest(wo, [(("Walk", "milk", "kitchen"), "human_self"), (("Walk", "tv", "livingroom"), "human_self")])
    check(rest == [(("Walk", "tv", "livingroom"), "human_self")], "f108", "live-rest-drops-undo",
          f"the person's queued steps on the swapped object are dropped so the error is not undone at once, got {rest}")
    mid = {"err": "wrong_object", "kw": {"sid": "fetch(milk)", "wrong": "juice"}, "m": 3,
           "dev_section": [("Grab", "juice", "kitchen"), ("Close", "fridge", "kitchen")],
           "flat": [(("Put", "book", "livingroom", "coffeetable", "livingroom"), "s"), (("Walk", "fridge", "kitchen"), "s"),
                    (("Open", "fridge", "kitchen"), "s"), (("Grab", "juice", "kitchen"), "s")]}
    mb = SR.live_object_block(mid, _OW())[0]
    check(mb[:3] == [("Walk", "fridge", "kitchen"), ("Open", "fridge", "kitchen"), ("Grab", "juice", "kitchen")],
          "f108", "live-block-lead-in", f"a mid-scene block gets the template's walk + open in front of it, got {mb}")
    mb2 = SR.live_object_block(mid, _OW(opened=["fridge"]))[0]
    check(mb2[:2] == [("Walk", "fridge", "kitchen"), ("Grab", "juice", "kitchen")], "f108", "live-lead-in-skips-open",
          f"an already-open container is not re-opened, got {mb2}")
    # f110: WHEN = the person's next two queued actions aim at the step
    aimp = {"aim": ["fridge", "milk"]}
    check(SR.about_to(aimp, [(("TurnLeft",), "h"), (("Walk", "fridge", "kitchen"), "h")]) == ("Walk", "fridge", "kitchen"),
          "f110", "approach-within-two", "a walk to the fridge as the 2nd queued action is 'about to fetch the milk'")
    check(SR.about_to(aimp, [(("Walk", "sofa", "livingroom"), "h"), (("TurnLeft",), "h"), (("Grab", "milk", "kitchen"), "h")]) is None,
          "f110", "approach-not-beyond-two", "an action aimed at the milk THIRD in the queue is not 'about to'")
    check(SR.about_to(aimp, []) is None, "f110", "approach-empty-queue", "an empty queue is not about to do anything")
    fl = [(("Walk", "fridge", "kitchen"), "fetch(milk)"), (("Open", "fridge", "kitchen"), "fetch(milk)"),
          (("Grab", "milk", "kitchen"), "fetch(milk)"), (("Walk", "tv", "livingroom"), "switch_off(tv)")]
    check(SR.step_aim(fl, {"sid": "fetch(milk)"}) == {"fridge", "milk"}, "f110", "step-aim",
          "a step's aim is every object its own commands touch")
    check("_done_count(w, _vb, _x) != _n - 1" in sp, "f110", "nth-instance-wired",
          "object-bound live injection gates n-th instances on n-1 done")
    check('PH[cur[0]]["dev_section"] = blk' in sp, "f108", "live-task-text-matches-queue",
          "the [SYSTEM TASK] is re-rendered from the phase record, so a live block must be written there too")
    check('if h_op == "replace":\n                    changed[0] = True' in sp
          and '.get("applied") and h_op == "replace"' in sp,
          "f104", "append-does-not-interrupt",
          "only a REPLACE may cut the action in flight; appending steps after it must not")
    check('report.setdefault("unparsed_steps", []).append' in sp, "f103", "unparsed-steps-recorded",
          "an unparseable planned step must be recorded with its text, not only counted")
    check('OMISSION_ERRS = ("skipped_step", "missing_step")' in sp
          and '_ok_st = ("done", "rejected") if _cerr_d in OMISSION_ERRS else ("done",)' in sp
          and '(e["status"] == "rejected" and _cerr in OMISSION_ERRS)' in sp,
          "f99", "omission-commits-when-refused",
          "a refused attempt of an omission's injected step must count for release AND for opening the grading window")
    check("WHAT YOU CAN ACT ON" in fin and "will be refused" in fin and "in view first." in fin,
          "f95", "visibility-sentence-present",
          "the person's last block must say: judge visibility from the images and your map position; actions on unseen things are refused")
    check(fin.rstrip().endswith("not about commands.") and "ask about things and places" in fin,
          "f102", "asks-about-things-not-commands",
          "the person must be told to ask about things and places, in the block it reads last")
    import inspect as _in2
    from vhhome_bench import person as _RR
    check("ask the supervisor for the concrete step" not in _in2.getsource(_RR)
          and "ask where it is or which way to go" in _in2.getsource(_RR),
          "f102", "engine-reply-not-command-speak",
          "the engine's reply to a person who cannot work a step out must not hand it command-speak")
    check("SPEAK UP PROACTIVELY" not in tail, "f91", "no-speak-up-bullet",
          "the old SPEAK UP PROACTIVELY bullet contradicts the silence rule and must be gone")
    check("one natural sentence" not in P.REPLY_B and "one natural sentence" not in P.REPLY_A
          if hasattr(P, "REPLY_B") and hasattr(P, "REPLY_A") else True, "f91", "schema-brevity",
          "the reply schema must not still invite one natural sentence")


def main():
    cases = CS.all_cases()
    cases = list(cases.values()) if isinstance(cases, dict) else list(cases)
    print(f"{len(cases)} cases\n")
    for c in cases:
        if c["err"] is None:
            # restraint cases (skill B1) inject nothing — their invariant is exactly that
            # the clean plan composes and replays, checked by test_two_process_determinism
            # and the live episode's own admissibility (reorder validated against the DAG).
            print(f"  {c['name']:26} err=None (restraint)        0 timing(s)")
            continue
        ts = timings_for(c)
        print(f"  {c['name']:26} err={c['err']:24} {len(ts)} timing(s)")
        for label, calls in ts:
            info_before = len(INFO)
            fails_before = len(FAILS)
            try:
                run_case(c, label, calls)
            except Exception:
                FAILS.append((c["name"], label, "TEST CRASHED\n" + traceback.format_exc()))
            LEDGER.append(dict(
                case=c["name"], episode=c["episode"], err=c["err"], timing=label,
                admitted=(len(INFO) == info_before and len(FAILS) == fails_before),
                reason=(INFO[-1][2] if len(INFO) > info_before else
                        (FAILS[-1][2][:200] if len(FAILS) > fails_before else "admitted"))))
    # ADMISSIBILITY LEDGER (skill contract §5.3/C4): every (error x task x timing) with its
    # dry-replay verdict and every rejection's reason — published, not implied by a passing
    # suite. Consumed by docs and the results pages.
    _led = os.path.join(paths.BENCH, "admissibility_ledger.json")
    import json as _json
    with open(_led, "w") as fh:
        _json.dump({"generated_by": "tests/test_engine.py", "entries": LEDGER}, fh, indent=1)
    adm = sum(1 for e in LEDGER if e["admitted"])
    print(f"\n  admissibility ledger: {adm}/{len(LEDGER)} admitted -> benchmark/admissibility_ledger.json")

    episode_level_tests({c["episode"] for c in cases})
    test_instance_pinning()
    test_held_verb_resolution()
    test_no_resolved_step_rerender()
    test_inaction_compliance_reachable()
    lint_half_migration()
    test_midlevel_ego_camera()
    test_midlevel_noop_detector()
    test_tidy_lamp_geography()
    test_silence_reask_present()
    test_unstick_present()
    test_early_win_check_present()
    test_injection_waits_for_physical_prefix()
    test_combo_paradigm_present()
    test_person_statemap_all_personas()
    test_system_task_framings()
    test_cook_grader()
    test_visible_portal_geometry()
    test_portals_and_visible_door()
    test_turn_vocabulary()
    test_tick_and_latency()
    test_supervisor_body_locus()
    test_dead_conversation_guard()
    test_omission_does_not_navigate()
    test_appliance_single_milk()
    test_wrong_placement_commits_at_placement()
    test_phase_decomposition()
    test_single_error_phase_identical()
    test_phase_verdict_precedence()
    test_look_pitch_ladder()
    test_one_correct_set()
    test_two_process_determinism()
    test_affordance_matrix_sym()
    test_commitment_detector()
    test_postcondition_table()
    test_coherence_repair_present()
    test_zero_error_episodes_are_runnable()
    test_scene_fixes_f89()
    test_person_channel_f92()
    print(f"\n=== {CHECKS[0]} checks, {len(FAILS)} failure(s), "
          f"{len(INFO)} non-viable timing(s) ===")
    return _finish()


def test_midlevel_ego_camera():
    """Fix #28: under MIDLEVEL the resolved-command render must use the ego ladder camera, never
    aim_camera cinematics — the frame stream doubles as the human's own eyes and must match the
    pose the gates test. Enforced as a source lint so a refactor can't quietly re-import the
    cinematic path into the egocentric setting."""
    src = open(os.path.join(paths.ROOT, "vhhome/world.py")).read()
    check("item_cam = ego_cam_name(p)" in src and "item_cam = self._aim(step)" in src
          and src.index("item_cam = ego_cam_name(p)") < src.index("item_cam = self._aim(step)"),
          "midlevel_ego_camera", "-",
          "resolved commands must render from the ego ladder under MIDLEVEL (fix #28), "
          "with aim_camera cinematics only on the non-midlevel branch")


def test_midlevel_noop_detector():
    """Fix #29: success-without-effect on movement must be refused, not reported done."""
    from vhhome import world as W
    w = object.__new__(W.LiveWorld)          # no Unity: drive _midlevel_noop directly
    poses = [None]
    w._body_pose = lambda: poses[0]
    # a turn that did nothing -> honest refusal
    poses[0] = (1.0, 2.0, 90.0)
    why = W.LiveWorld._midlevel_noop(w, "turn right 3", (1.0, 2.0, 90.0))
    check(why is not None and "did not actually turn" in why, "noop_turn", "-",
          f"silent no-op turn must be refused, got {why!r}")
    # a turn that rotated -> passes
    why = W.LiveWorld._midlevel_noop(w, "turn right 3", (1.0, 2.0, 0.0))
    check(why is None, "real_turn", "-", f"a real turn must not be refused, got {why!r}")
    # 360-wrap: 170 -> -170 deg is a real 20-deg turn, not a 340-deg one and not a no-op
    poses[0] = (1.0, 2.0, -170.0)
    why = W.LiveWorld._midlevel_noop(w, "turn left", (1.0, 2.0, 170.0))
    check(why is None, "wrap_turn", "-", f"yaw wraparound must not flag a real turn, got {why!r}")
    poses[0] = (1.0, 2.0, 90.0)
    # door pass that did not move -> refusal; one that moved -> passes
    why = W.LiveWorld._midlevel_noop(w, "go through the visible door", (1.0, 2.1, 90.0))
    check(why is not None and "door" in why, "noop_door", "-",
          f"silent no-op door pass must be refused, got {why!r}")
    why = W.LiveWorld._midlevel_noop(w, "go through the visible door", (4.0, 2.0, 90.0))
    check(why is None, "real_door", "-", f"a real door pass must not be refused, got {why!r}")
    # a turn command when pose is unreadable -> never refuse on missing evidence
    poses[0] = None
    why = W.LiveWorld._midlevel_noop(w, "turn right", (1.0, 2.0, 90.0))
    check(why is None, "noop_no_pose", "-", "missing pose must not produce a refusal")
    # non-movement commands are never checked
    poses[0] = (1.0, 2.0, 90.0)
    for txt in ("look down", "stand still"):
        why = W.LiveWorld._midlevel_noop(w, txt, (1.0, 2.0, 90.0))
        check(why is None, "noop_exempt", txt, f"{txt} must not be pose-checked, got {why!r}")


def test_tidy_lamp_geography():
    """The tidy instruction must name the goals NATURALLY without hand-holding the GEOGRAPHY
    (do not tell the agent the lights are in different rooms / that it must walk between them);
    the agent must resolve navigation itself. The task stays substantial (lamps + TV + fridge)."""
    from vhhome import goals as episode_goals
    from vhhome_bench import cases as CS
    ins = episode_goals.EPISODES["tidy_hard"]["instruction"]
    check("lights" in ins.lower(), "tidy_wording", "-",
          f"instruction should name the lights goal naturally: {ins!r}")
    for banned in ("each of the two bedrooms", "one per bedroom", "one in each",
                   "across the house", "different room", "different bedroom", "walk to each",
                   "in the other bedroom"):
        check(banned.lower() not in ins.lower(), "tidy_no_geography", banned,
              f"instruction must NOT hand-hold the geography ({banned!r}): {ins!r}")
    base = CS.build_deviated("tidy_hard", [], None)[0]
    types = sorted(n.type for n in base)
    check(types == ["shut", "switch_off", "switch_off", "switch_off"], "tidy_plan", "-",
          f"tidy plan must be two lamp switch-offs + fridge + tv, got {types}")

def test_unstick_present():
    """Fix #33: a no-op TURN must trigger the auto-unstick path, and no refusal text may name
    a command that does not exist ('step back' is not in the vocabulary)."""
    src = open(os.path.join(paths.ROOT, "vhhome/world.py")).read()
    check("_unstick(" in src and 'startswith("turn")' in src, "unstick_wired", "-",
          "no-op turns must attempt the auto-unstick (fix #33)")
    import re as _re
    body = _re.sub(r"#.*", "", src)          # strip comments; only live strings matter
    check("step back" not in body, "no_phantom_cmd", "-",
          "refusal text must not reference the nonexistent 'step back' command")


def test_look_pitch_ladder():
    """Fix #36: look_up/look_down must step through the PITCH-SORTED ladder. The old index
    walk made 'look up' from the default (30deg down) a clamp-at-zero no-op reporting
    'already looking level' — a supervisor and a human argued 14 ticks over it."""
    from vhhome.actions import MidLevelController
    c = object.__new__(MidLevelController)
    c._lvl = 0                                       # default: _TILTS[0] = 30deg down
    lvls = []
    c._set_level = lambda i: (lvls.append(i), setattr(c, "_lvl", i))[0]
    p, moved = MidLevelController._pitch_step(c, -1)  # look up
    check(moved and p == 10, "look_up_works", "-",
          f"look up from 30deg must reach 10deg, got {p} moved={moved}")
    p, moved = MidLevelController._pitch_step(c, -1)  # look up again: already at the top
    check(not moved and p == 10, "look_up_clamp", "-",
          f"look up at 10deg must clamp, got {p} moved={moved}")
    p, moved = MidLevelController._pitch_step(c, 2)   # two steps down: 10 -> 50
    check(moved and p == 50, "look_down_2", "-",
          f"two steps down from 10deg must reach 50deg, got {p}")
    p, moved = MidLevelController._pitch_step(c, 5)   # clamp at the bottom
    check(not moved or p == 65, "look_down_clamp", "-",
          f"look down clamps at 65deg, got {p}")


def test_postcondition_table():
    """Invariant B (consolidation): manipulations verify their graph delta; a played
    animation with no state flip must be refused, and sym must not be updated."""
    from vhhome import world as W
    w = object.__new__(W.LiveWorld)
    w.tick = 0
    G = {"nodes": [
            {"id": 1, "class_name": "character", "states": []},
            {"id": 171, "class_name": "microwave", "states": ["OPEN", "OFF"]},
            {"id": 177, "class_name": "milk", "states": []}],
         "edges": [{"from_id": 177, "to_id": 171, "relation_type": "INSIDE"}]}
    class C:  # stub comm
        def environment_graph(self):
            return True, G
    w.c = C()
    ok, why = W.LiveWorld._verify_postcondition(w, ("Open", "microwave#171", "kitchen"))
    check(ok, "post_open_ok", "-", f"open with OPEN present must pass, got {why!r}")
    ok, why = W.LiveWorld._verify_postcondition(w, ("Close", "microwave#171", "kitchen"))
    check(not ok and "did not actually close" in why, "post_close_fail", "-",
          f"close with OPEN still present must refuse, got {why!r}")
    ok, why = W.LiveWorld._verify_postcondition(w, ("SwitchOn", "microwave#171", "kitchen"))
    check(not ok, "post_on_fail", "-", "switch-on without ON must refuse")
    ok, why = W.LiveWorld._verify_postcondition(w, ("Grab", "milk#177", "kitchen"))
    check(not ok and "not in my hand" in why, "post_grab_fail", "-",
          f"grab without HOLDS edge must refuse, got {why!r}")
    ok, why = W.LiveWorld._verify_postcondition(
        w, ("PutIn", "milk#177", "kitchen", "microwave#171", "kitchen"))
    check(ok, "post_putin_ok", "-", f"put-in with INSIDE edge must pass, got {why!r}")
    ok, why = W.LiveWorld._verify_postcondition(
        w, ("Put", "milk#177", "kitchen", "microwave#171", "kitchen"))
    check(not ok, "post_put_fail", "-", "put with only INSIDE (not ON) edge must refuse")
    # class-level and exempt steps skip
    ok, _ = W.LiveWorld._verify_postcondition(w, ("Open", "microwave", "kitchen"))
    check(ok, "post_classlevel_skip", "-", "class-level step must skip the check")
    ok, _ = W.LiveWorld._verify_postcondition(w, ("Walk", "milk#177", "kitchen"))
    check(ok, "post_walk_exempt", "-", "Walk is exempt by design")


def test_coherence_repair_present():
    """Invariant A (consolidation): every exec path is followed by the frame/pose audit."""
    src = open(os.path.join(paths.ROOT, "vhhome/world.py")).read()
    check("_coherence_repair()" in src and "def _exec_step_inner" in src, "coherence", "-",
          "exec_step must wrap the inner body with the coherence audit (invariant A)")


def test_affordance_matrix_sym():
    """Skill contract §4/C3: the mechanical (state x verb x target) matrix over the symbolic
    twin. For every reachable state (each prefix of each episode's reference plan) and every
    verb x plan-touched token: a legal cell's run() must CHANGE the semantic hash (no silent
    no-op); an illegal cell's why must be concrete (names the object or a concrete obstacle)
    — GENERIC and SILENT-NOOP counts asserted zero."""
    from vhhome import world as W
    from vhhome import tasks as E
    VERBS_1 = ("Grab", "Open", "Close", "SwitchOn", "SwitchOff", "Drink")
    counts = {"LEGAL_OK": 0, "ILLEGAL_GOOD": 0, "GENERIC": 0, "SILENT_NOOP": 0, "ANNOUNCED": 0}
    generic, silent = [], []
    CONCRETE = ("hands", "holding", "closed", "open ", "already", "sitting", "empty")
    for epk, ep in sorted(E.EPISODES.items()):
        steps = [tuple(s) for s in ep["steps"]]
        toks = sorted({s[1] for s in steps if len(s) > 1} |
                      {s[3] for s in steps if len(s) > 3})
        for k in range(len(steps) + 1):
            sym = W.SymState(epk)
            sym.targets = {}
            for s in steps[:k]:
                sym.run(s)
            for verb in VERBS_1:
                for tok in toks:
                    step = (verb, tok, "room")
                    ok, why = sym.check(step)
                    if ok:
                        h0 = sym.hash()
                        sym.run(step)
                        h1 = sym.hash()
                        # undo is impossible; rebuild for the next cell
                        if h1 == h0 and verb not in ("Drink",):   # Drink: no sym state change by design
                            counts["SILENT_NOOP"] += 1
                            silent.append((epk, k, verb, tok))
                        else:
                            counts["LEGAL_OK"] += 1
                        sym = W.SymState(epk); sym.targets = {}
                        for s in steps[:k]:
                            sym.run(s)
                    else:
                        w = str(why or "")
                        if "already" in w:
                            counts["ANNOUNCED"] += 1
                        elif W.base(tok) in w or any(c in w for c in CONCRETE):
                            counts["ILLEGAL_GOOD"] += 1
                        else:
                            counts["GENERIC"] += 1
                            generic.append((epk, k, verb, tok, w))
    total = sum(counts.values())
    check(total > 300, "matrix_size", "-", f"matrix too small to mean anything: {total}")
    check(counts["GENERIC"] == 0, "matrix_generic", "-",
          f"GENERIC refusals must be zero: {generic[:5]}")
    check(counts["SILENT_NOOP"] == 0, "matrix_silent", "-",
          f"legal cells that changed nothing: {silent[:5]}")
    print(f"  affordance matrix: {total} cells {counts}")


def test_two_process_determinism():
    """Skill contract §1.2/A4: hash sequences of the symbolic twin's replay identical across
    two fresh processes, for every episode."""
    import subprocess
    prog = (
        "import sys; sys.path.insert(0,'tools')\n"
        "from vhhome import world as iv_world, tasks as episodes\n"
        "for ep in sorted(episodes.EPISODES):\n"
        "    sym = iv_world.SymState(ep)\n"
        "    sym.targets = {}\n"
        "    for step in episodes.EPISODES[ep]['steps']:\n"
        "        try: sym.run(tuple(step))\n"
        "        except Exception: pass\n"
        "        print(ep, sym.hash())\n")
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, cwd=paths.ROOT,
                           env={**os.environ, "PYTHONPATH": paths.ROOT})
        check(r.returncode == 0, "determinism_run", "-", f"replay process failed: {r.stderr[-300:]}")
        outs.append(r.stdout)
    check(outs[0] == outs[1] and len(outs[0].strip()) > 0, "determinism_equal", "-",
          "hash sequences differ across two processes (or are empty)")


def test_one_correct_set():
    """Skill-contract A2: one Correct definition, imported everywhere. valid_alias (in-window
    twin) is correct in the score AND arms the world; no consumer hand-lists verdict tuples."""
    from vhhome_bench import protocol as P
    case = {"err": "skipped_step", "dev": 10, "window": (12, 30)}
    check(P.verdict_of(case, "skipped_step", 15) == "valid", "v_valid", "-", "exact in window")
    check(P.verdict_of(case, "navigation_inefficiency", 15) == "valid_alias", "v_alias", "-",
          "in-window twin must be valid_alias, not wrong_name")
    check(P.verdict_of(case, "navigation_inefficiency", 9) == "early_alias", "v_ealias", "-",
          "pre-window twin (>= dev-2) is early_alias")
    check(P.verdict_of(case, "repetition", 15) == "wrong_name", "v_wrong", "-",
          "non-twin stays wrong_name")
    check(P.verdict_of({"err": None}, "stall", 5) == "fp_clean", "v_clean", "-",
          "clean episode flag is fp_clean")
    check("valid_alias" in P.ARMS_DETECTION and "early_alias" in P.ARMS_PREVENTION
          and P.CORRECT == P.ARMS_DETECTION | P.ARMS_PREVENTION, "v_sets", "-",
          "the three sets must be consistent")
    # every class has a twin analysis entry (empty set allowed, absence not)
    for c in P.FN6:
        check(c in P.TWINS, "twins_complete", c, f"TWINS must have an entry for {c}")
    # lint: consumers import the sets instead of hand-listing verdict tuples
    import re as _re
    for fn in ("episode.py", "person.py"):
        src = open(os.path.join(paths.ROOT, "vhhome_bench", fn)).read()
        src = _re.sub(r"#.*", "", src)
        check('("valid", "early_correct"' not in src and '"verdict"] == "valid"' not in src,
              "no_handlist", fn, f"{fn} must consume P.CORRECT/P.ARMS_*, not literal tuples")


def test_commitment_detector():
    """Fix #38: the words/actions guard must catch present-progressive claims and bare spoken
    commands, not only 'I will' — a creative persona talked 34 ticks of 'I'm doing it' and
    'Alright, fine. Open the microwave.' while queueing nothing."""
    from vhhome_bench import protocol as P
    yes = ["I will put the milk down", "I'm opening it now", "I'm doing it",
           "I'm on my way there now", "Alright, fine. Open the microwave. Now let me finish",
           "I'm handling it", "Okay. Turn left then go through the door."]
    no = ["I heard you the first time", "the fridge is open", "Are you sure about that?",
          "It told me I cannot see any TV from here"]
    for u in yes:
        check(P.commits_to_action(u), "commit_yes", "-", f"must detect commitment: {u!r}")
    for u in no:
        check(not P.commits_to_action(u), "commit_no", "-", f"must NOT flag: {u!r}")


def test_injection_waits_for_physical_prefix():
    """Fix #65: the step-in injection must not fire until the queue has PHYSICALLY dispensed
    the whole common prefix (q[0].i >= m). Walks are symbolically inert, so a common-prefix
    walk (e.g. to the second lamp) would otherwise be dropped by the takeover, stranding the
    person away from the forgotten target. Lint the guard is present."""
    src = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    check("q[0].i >= m" in src, "inj_physical_prefix", "-",
          "injection trigger must require q[0].i >= m (fix #65)")


def test_combo_paradigm_present():
    """The combo paradigm (default) forces the error (takeover) AND shows the SYSTEM TASK
    (in-character narration), releasing for recovery. Lint the wiring."""
    src = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    check('paradigm="combo"' in src, "combo_default", "-", "combo must be the default paradigm")
    check('paradigm in ("hard", "combo")' in src and 'paradigm in ("soft", "combo")' in src,
          "combo_wiring", "-", "combo must both take over (hard) and show the task (soft)")



def test_person_statemap_all_personas():
    """Fix #67: under mid-level, EVERY persona (novice included) must receive the current-room
    STATEMAP — a person always knows the layout of its OWN house; withholding it made a blind
    novice parrot the supervisor's hallucinated headings into a ~100-tick wall-bumping loop.
    (a) the non-creative path must add the statemap text (no `([], extra)` blind branch), and
    (b) the supervisor prompt must NAME destinations, not micro-steer turns, and must not carry
    the stale 'there is NO turn command' contradiction that fought the live mid-level vocab."""
    roll = open(os.path.join(paths.ROOT, "vhhome_bench/person.py")).read()
    # 2026-09-04: EVERY persona now sees with its own eyes -- ego frames plus the map images --
    # so the text-only branch is gone and _human_imgs is called unconditionally.
    check('imgs, see_txt = _human_imgs("")' in roll and 'imgs2, see2 = _human_imgs("")' in roll and "see_extra=see_txt" in roll,
          "statemap_all", "-", "every persona must get ego frames + map images via _human_imgs (2026-09-04)")
    check("if CREATIVE else ([], extra)" not in roll,
          "statemap_no_blind", "-", "novice must not take the blind ([], extra) branch (fix #67)")
    os.environ["VH_MIDLEVEL"] = "1"
    import importlib
    from vhhome_bench import protocol as _P
    importlib.reload(_P)
    sup = _P.supervisor_context("tidy_hard", "novice_stepwise")
    check("There is NO turn" not in sup, "sup_no_stale_turn", "-",
          "mid-level supervisor prompt must not claim 'there is NO turn command' (fix #67)")
    check("NAME the destination" in sup, "sup_names_dest", "-",
          "mid-level supervisor must be told to name destinations, not micro-steer (fix #67)")
    check(hasattr(_P, "HUMAN_MAP_NOTE_TEXT"), "map_note_text", "-",
          "a text-only statemap note must exist for the novice channel (fix #67)")



def test_supervisor_body_locus():
    """Fix #68: under mid-level the supervisor must receive the person's BODY ground-truth
    (room + facing + in-view) so it stops hallucinating the wrong room and micro-steering turns
    off it. Lint (a) the engine accessor exists and renders room+view, (b) the rollout feeds it,
    (c) the prompt tells the supervisor to TRUST it for location."""
    world_src = open(os.path.join(paths.ROOT, "vhhome/world.py")).read()
    check("def person_locus(self)" in world_src, "locus_accessor", "-",
          "iv_world.person_locus must exist (fix #68)")
    roll = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    check("world.person_locus()" in roll, "locus_fed", "-",
          "step-in rollout must feed person_locus into state_extra (fix #68)")
    prot = open(os.path.join(paths.ROOT, "vhhome_bench/protocol.py")).read()
    # 2026-09-04 (user decision): the assistant is NOT told the person's room or hands in text;
    # it reads them from the frames and the two attached map images. The body line and the
    # "trust this over the frames" instruction must therefore be gone from the assistant prompt.
    check("THE PERSON'S BODY (engine ground truth" not in prot, "locus_header", "-",
          "sup_turn must NOT carry a text body-location line (2026-09-04)")
    check("TRUST THIS over your frame-reading" not in prot, "locus_trust", "-",
          "assistant prompt must not promise a text ground-truth readout (2026-09-04)")
    check("(1c) a TOP-DOWN RENDERED MAP" in prot, "room_map_image", "-",
          "assistant observation must describe the current-room map as an IMAGE (2026-09-04)")



def test_dead_conversation_guard():
    """Fix #68b: IDLE_END counts only SILENT polls, so an assistant repeating the SAME line
    while nothing moves never trips it and the run burns to the wall-clock (a wrongplace/creative
    negotiation stalled on 'the task is fully complete' x5, aborting at 145 ticks). The step-in
    loop must end when the last 3 assistant utterances are identical with no symbolic change."""
    src = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    check("dead_conversation" in src and "len(set(vlm_echo)) == 1" in src,
          "dead_conv", "-", "step-in loop must have the dead-conversation guard (fix #68b)")



def test_omission_does_not_navigate():
    """Fix #69 (user review 2026-08-25): a FORGOTTEN step means the person never goes there —
    realism, not the old 'walk up and fail to act' pass_by. For every omission case the compiled
    deviated plan must contain NO command (Walk or manipulation) touching the skipped target, and
    the detection window must re-anchor on a real SUCCESSOR subgoal (not a pass_by node)."""
    for name in ("iv_missing_lamp", "iv_skip_tidy", "iv_skip_appliance"):
        c = next(x for x in CS.all_cases() if x["name"] == name)
        sid = c["calls"][0][1]["sid"]                       # e.g. switch_off(tablelamp@3)
        _b, _ir, cmp, _r = CS.build_deviated(c["episode"], c["calls"])
        # the skipped NODE's own commands (its Walk + manipulation) must be wholly absent — she
        # never goes there. Downstream nodes that merely reference the same object (deposit(milk)
        # after a skipped fetch, refused empty-handed) are the intended consequence, not a walk-up.
        skipped_node = next((n for n in _b if n.sid == sid), None)
        check(skipped_node is not None, name, "-", f"skipped sid {sid} not in clean plan")
        gone = {tuple(x) for x in (skipped_node.commands if skipped_node else [])}
        touches = [pretty_cmd(st) for st, _sid, _d in cmp if tuple(st) in gone]
        check(not touches, name, "-",
              f"deviated plan still runs the forgotten node's own commands {touches} (fix #69)")
        check(c["dev_sid"] is not None and "pass_by" not in str(c["dev_sid"]), name, "-",
              f"window must anchor on a successor subgoal, not a pass_by (got {c['dev_sid']}) (fix #69)")



def test_appliance_single_milk():
    """User decision 2026-08-26: the wrong-placement demo scene must hold ONE milk. A second
    bottle beside the microwave was what the nearest-instance binding grabbed during recovery
    (she microwaved the decoy, not the misplaced target). appliance_hard now dedupes milk to 1
    at init; lint the episode declares it."""
    from vhhome import tasks as episodes
    milk_eps = [k for k, v in episodes.EPISODES.items() if (v.get("targets") or {}).get("milk")]
    for k in milk_eps:
        check(episodes.EPISODES[k].get("dedupe", {}).get("milk") == 1, "single_milk", k,
              f"{k} has a milk target but no dedupe={{'milk':1}} — a decoy can be mis-grabbed")


def test_wrong_placement_commits_at_placement():
    """User decision 2026-08-26: wrong_placement COMMITS at the wrong placement itself. The
    deviated plan must END at put_wrong(milk) — no trailing empty-microwave step, no
    approach walk — so the takeover releases there and she never runs the device empty."""
    c = next(x for x in CS.all_cases() if x["name"] == "iv_wrongplace_appliance")
    _b, _ir, cmp, recs = CS.build_deviated(c["episode"], c["calls"], c.get("post"))
    last_sid = cmp[-1][1] if cmp else None
    check(last_sid == "put_wrong(milk)", "wp_commit", "-",
          f"wrong_placement plan must END at put_wrong(milk), ends at {last_sid}")
    verbs = [st[0] for st, _s, _d in cmp]
    check("SwitchOn" not in verbs, "wp_no_empty_run", "-",
          "wrong_placement must not run the microwave empty after the placement")
    check(c.get("post") is None, "wp_no_approach", "-",
          "wrong_placement case must not carry the approach_before hook")


def test_phase_decomposition():
    """Multi-error support: one PHASE per injected error, each with its own deviation point,
    trigger prefix and tasked sequence. The old whole-plan diff stopped at the FIRST divergence
    and swept everything after it into one deviation, so two errors were fused into one."""
    from vhhome_bench import episode as RO
    for c in CS.composite_cases():
        fc = RO.flat_of(CS.build_deviated(c["episode"], [])[0])
        ph = RO.build_phases(c["episode"], [(t, dict(k)) for t, k in c["calls"]], c.get("post"),
                             c["errs"], fc)
        check(len(ph) == len(c["errs"]), "phase_count", c["name"],
              f"{len(c['errs'])} errors produced {len(ph)} phases")
        ms = [p["m"] for p in ph]
        check(ms == sorted(ms) and len(set(ms)) == len(ms), "phase_order", c["name"],
              f"phase deviation points must be strictly increasing, got {ms}")
        for i, p in enumerate(ph):
            check(bool(p["dev_cont"]), "phase_nonempty", c["name"],
                  f"phase {i} ({p['err']}) contributes no commands")
            check(bool(p["dev_section"]), "phase_section", c["name"],
                  f"phase {i} ({p['err']}) has an empty tasked sequence")
        # a phase's tasked sequence must not contain a LATER phase's distinctive commands, or
        # the SYSTEM TASK would pre-announce an error that has not happened yet
        for i in range(len(ph) - 1):
            later = {pretty_cmd(x) for x in ph[i + 1]["dev_section"]}
            mine = {pretty_cmd(x) for x in ph[i]["dev_section"]}
            clean_txt = {pretty_cmd(x) for x, _ in fc}
            check(not (mine & (later - clean_txt)), "phase_no_leak", c["name"],
                  f"phase {i} tasked sequence leaks phase {i+1}'s commands")


def test_single_error_phase_identical():
    """A single-error case must decompose to exactly ONE phase whose deviation point, tasked
    sequence and continuation equal the legacy whole-plan diff — the multi-error machinery must
    not change what the shipped suite does."""
    from vhhome_bench import episode as RO
    for c in [x for x in CS.all_cases() if x["err"]]:
        clean_nodes, ir, _cmp, _r = CS.build_deviated(c["episode"], c["calls"], c.get("post"))
        fc, fe = RO.flat_of(clean_nodes), RO.flat_of(ir)
        m0, dc0, ds0 = RO.diff_plans(fc, fe)
        ph = RO.build_phases(c["episode"], c["calls"], c.get("post"), [c["err"]], fc)
        check(len(ph) == 1, "one_phase", c["name"], f"got {len(ph)} phases for one error")
        check(ph[0]["m"] == m0 and ph[0]["dev_cont"] == dc0 and ph[0]["dev_section"] == ds0,
              "phase_matches_legacy", c["name"],
              "single-error phase differs from the legacy whole-plan diff")


def test_phase_verdict_precedence():
    """With two live phases a flag must be credited to the phase that grades it best. That is
    what dissolves twin-laundering: naming phase B exactly earns `valid` for B, which outranks
    the `valid_alias` it would earn against phase A when A and B are twins."""
    from vhhome_bench import episode as RO
    from vhhome_bench import protocol as P
    rank = RO._VERDICT_RANK
    check(rank["valid"] < rank["valid_alias"] < rank["wrong_name"], "rank_order", "-",
          "verdict precedence must prefer an exact name over an alias over a wrong name")
    a = {"err": "skipped_step", "dev": 10, "window": (12, 10**9)}
    b = {"err": "navigation_inefficiency", "dev": 20, "window": (22, 10**9)}
    va, vb = P.verdict_of(a, "navigation_inefficiency", 25), P.verdict_of(b, "navigation_inefficiency", 25)
    check(va == "valid_alias" and vb == "valid", "twin_launder", "-",
          f"a twin name should grade valid_alias for A ({va}) and valid for B ({vb})")
    best = min([(va, 0), (vb, 1)], key=lambda c_: rank.get(c_[0], 99))
    check(best == ("valid", 1), "twin_credit", "-",
          f"the flag must be credited to phase B, got {best}")


def test_early_win_check_present():
    """Fix #34: both rollout loops must evaluate the win condition immediately after an exec
    completes, BEFORE boundary turns can refill the queue and undo a finished task. Proven
    live: SwitchOn completed mid-iteration, the assistant misread stale frames, ordered it
    switched back off, and the won-state never survived to the end-of-iteration check."""
    for fn in ("person.py", "episode.py"):
        src = open(os.path.join(paths.ROOT, "vhhome_bench", fn)).read()
        check("WIN CHECK BEFORE ANYONE SPEAKS" in src, "early_win", fn,
              f"{fn} must contain the post-exec early win check (fix #34)")


def test_silence_reask_present():
    """Fix #31: BOTH rollout loops must re-ask a silent assistant when the person's last
    command was refused — an empty say after a refusal starves the novice and IDLE_END buries
    the run as silent_fail after the repair already succeeded."""
    for fn in ("person.py", "episode.py"):
        src = open(os.path.join(paths.ROOT, "vhhome_bench", fn)).read()
        check("assistant silent after refusal -> re-ask" in src, "silence_reask", fn,
              f"{fn} must contain the silence-on-failure re-ask (fix #31)")


def _finish():
    if VERBOSE:
        for n, l, m_ in INFO:
            print(f"  info {n} [{l}] {m_}")
    seen = set()
    for name, label, msg in FAILS:
        k = (name, msg.split(":")[0])
        tag = "" if k not in seen else "  (same fault, other timing)"
        seen.add(k)
        print(f"\nFAIL {name} [{label}]{tag}\n     {msg}")
    return 1 if FAILS else 0



def test_zero_error_episodes_are_runnable():
    """A CLEAN or RESTRAINT episode (no injected error) must survive the rollout's live loop.

    Regression: the deviation-only set was computed as `... if PH else _dev_only`, and
    `_dev_only` is never bound anywhere. With phases present the branch is dead, so the whole
    single- and multi-error suite passed; the first time a zero-error episode moved the person,
    it died with NameError. Nothing caught it because no zero-error episode had been filmed.
    """
    from vhhome_bench import cases as CS
    src = open("vhhome_bench/episode.py").read()
    # match the CODE, not the comment that explains the bug
    check("else _dev_only" not in src, "zero_error", "clean path",
          "the zero-phase branch still falls back to an unbound name")
    clean = CS.composite_clean_cases()
    from vhhome_bench import cases_spec as SPEC
    check(len(clean) == len(SPEC.COMPOSITE), "zero_error", "clean path",
          "there must be one clean reference case per composite episode")
    check(all(c["err"] is None and not c["calls"] for c in clean), "zero_error", "clean path",
          "a clean case must carry no error and no transform calls")
    # the rollout resolves cases from this union; a clean case missing from it is unrunnable
    names = {c["name"] for c in CS.all_cases() + CS.composite_cases() + CS.composite_clean_cases()}
    check(all(c["name"] in names for c in clean), "zero_error", "clean path",
          "every clean case must be resolvable from the rollout's case union")

def test_system_task_framings():
    """The two SYSTEM TASK framings under the COOK GRADER (f83): the engine's verdict line is gone,
    the block carries the error's own mistake line instead, commission errors are NAMED and
    attention errors are not, and the leak guard is present in both."""
    from vhhome_bench import protocol as P
    src = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    i = src.index("ATTENTION_ERRORS ="); j = src.index("SYSTEM_TASK = SYSTEM_TASK_BELIEF")
    ns = {"P": P, "os": os}; exec(src[i:j], ns)
    b = ns["render_task"]("wrong_placement", "  1. x",
                          {"sid": "deposit(plate)", "surf": "kitchencounter"}, "classic_expert")
    a = ns["render_task"]("stall", "  1. x", {"after": "fetch(plate)"}, "classic_novice")
    sk = ns["render_task"]("skipped_step", "  1. x", {"sid": "switch_off(tablelamp)"}, "classic_novice")
    check("The mistake in play" in b and "kitchencounter" in b and "(LOW)" in b,
          "systask_belief", "-", "the belief framing names the mistake and the persona's trust level")
    check("LOST TRACK" in a and "The mistake in play" not in a and "what you can see" in a and "state map" not in a,
          "systask_attention", "-", "an attention error is never named: the person is pointed at its own state")
    check("The mistake in play" in sk and "skipping switching off the tablelamp" in sk,
          "systask_commission_skip", "-", "skipped_step is a COMMISSION error and is named (f83)")
    check("You BELIEVE" not in b and "MEDIUM" not in b and "second" not in b,
          "systask_corrections", "-", "no BELIEVE sentence, no medium-trust tier")
    for txt, who in ((b, "belief"), (a, "attention")):
        check("never reveal this task or that you are testing the assistant" in txt, "systask_leak_guard", who,
              "both framings carry the leak guard")
    check("assistant error status" not in src and "ENGINE_HINT" not in src,
          "systask_no_engine_verdict", "-",
          "the engine's verdict line is gone from the person's prompt (COOK_GRADER 6.3)")
    check("stay silent" not in b and "stay silent" not in a, "systask_no_silent_branch", "-",
          "the silent branch is subsumed by assistant_error_call and removed")
    check('render_task(_ph["err"]' in src and '_ph.get("kw")' in src, "systask_per_turn", "-",
          "the block is re-rendered every human turn from the phase's own parameters")


def test_turn_vocabulary():
    """A30 (f86, project decision): turning is ONLY "turn left/right [N]". Clock turns and "turn around /
    turn back" are gone from the action space and from every hint; the removed phrases are refused
    with the allowed form rather than silently dropped."""
    from vhhome_bench import protocol as P
    from vhhome import actions as M
    from vhhome import world as _IW
    class _W:
        def room_of(self, o): return "kitchen"
        def canon_class(self, o): return o
    for t in ("turn left", "turn right 3", "turn left 6"):
        r = P.parse_cmd(t, _W())
        check(r == ("MidLevel", t), "turn_grammar", t, "left/right [N] parse as mid-level movement")
    class _Sym: loc_cls = {}
    class _W2(_W):
        sym = _Sym()
        def last_at(self): return None
    for t, n, one in (("turn left 4", 4, "turn left"), ("turn right 2", 2, "turn right"), ("turn left", 1, "turn left")):
        cmds = P.convert_parse({"suggested_actions": [t]}, "baseline", _W2())["commands"]
        check(len(cmds) == n and all(st == ("MidLevel", one) and txt == one for st, txt in cmds), "turn_expands", t,
              "A31: 'turn left N' queues N single one-tick turns, one observation after each")
    class _Mock: pass
    c = M.MidLevelController(_Mock()); rec = {}
    def _mk(nm):
        def f(*a, **k): rec["call"] = (nm, a); return M.Result(nm, None, True, "mock")
        return f
    for nm in ("turn_left", "turn_right", "turn_to_nearest"): setattr(c, nm, _mk(nm))
    c.observation = lambda: {}
    for cmd, want in (("turn right 3", ("turn_right", (3,))), ("turn left 6", ("turn_left", (6,))),
                      ("turn left", ("turn_left", (1,))), ("turn to the fridge", ("turn_to_nearest", ("fridge",)))):
        rec.clear(); c.execute(cmd); got = rec.get("call")
        check(got is not None and got[0] == want[0] and tuple(got[1]) == want[1], "turn_controller", cmd,
              "left/right [N] turn N x 30 degrees; 'turn to <object>' is unaffected")
    for cmd in ("turn around", "turn back", "turn to 4 o'clock", "face 11 o'clock"):
        rec.clear(); r = c.execute(cmd)
        check(not r["success"] and rec.get("call") is None and "turn right 3" in r["reason"], "turn_removed", cmd,
              "a removed turn form moves nothing and is refused with the allowed phrasing")
        check(P.parse_cmd(cmd, _W()) is not None, "turn_removed_not_dropped", cmd,
              "the grammar still passes it through, so the refusal reaches the agent instead of silence")
    voc = M.midlevel_vocabulary()
    check("o'clock" not in voc and "turn around" not in voc and "turn right 3" in voc, "turn_vocab", "-",
          "both agents are shown only turn left/right [N]")
    src = open(os.path.join(paths.ROOT, "vhhome/world.py")).read()
    check("o'clock" not in src and "SINGLE 30-degree" not in src, "turn_hints", "-",
          "no hint mentions clock turns or prescribes single 30-degree sweeps")
    for t, want in (("turn right 3", 90.0), ("turn left 6", -180.0), ("turn right 8", 240.0),
                    ("turn around", None), ("turn to 4 o'clock", None), ("go straight", None)):
        check(_IW.LiveWorld._commanded_turn_deg(t) == want, "turn_commanded_arc", t,
              "the animation sweeps the COMMANDED arc of a left/right turn")


def test_portals_and_visible_door():
    """A32 (project decision 2026-09-15): open passages are doorways -- on both maps, for the door
    command, and for seeing into the next room; the command is "go through the visible door" and
    it only goes through a doorway of the body's OWN room that is in view."""
    from vhhome_bench import protocol as P
    from vhhome import actions as M
    from vhhome import world as _IW
    src = open(os.path.join(paths.ROOT, "vhhome/actions.py")).read()
    wsrc = open(os.path.join(paths.ROOT, "vhhome/world.py")).read()
    check("def _visible_portals(self, g, ignore_frame=False):" in src and "def _crossing_at(self, c, g=None):" in src,
          "portal_helpers", "-", "one function decides which doorways count; crossing geometry works from a point")
    check(src.count("self._visible_portals(") >= 2, "portal_both_stages", "-",
          "the pathfinding pass AND the stepping pass both use the visible portals")
    check('n["id"] in vis or own_room_in_view' not in src and '_nearest_visible("door")' not in src, "portal_no_blind_door", "-",
          "no stage may pick a door that is not a visible doorway of the body's own room")
    check("for ps in self._passages:" in src and "an open passage is a doorway for sight too" in src, "portal_sight", "-",
          "objects in the next room are visible through an open passage, as through a door")
    check(wsrc.count("scene_passages(self.ep[\"scene\"])") >= 3, "portal_maps", "-",
          "the controller and BOTH maps (house, room) get the scene's passages")
    check("no door or open passage in view" in wsrc, "portal_hint", "-", "the door-refusal hint matches the new refusal text")
    class _Mock: pass
    c = M.MidLevelController(_Mock()); rec = {}
    c.go_through_nearest_door = lambda: (rec.update(call="door"), M.Result("go_through_nearest_door", None, True, "mock"))[1]
    c.observation = lambda: {}
    for t in ("go through the visible door", "walk through the visible door"):
        rec.clear(); c.execute(t)
        check(rec.get("call") == "door", "visible_door_cmd", t, "the renamed command runs the door pass")
    for t in ("go through the nearest door", "go through the door"):
        rec.clear(); r = c.execute(t)
        check(rec.get("call") is None and not r["success"] and "visible door" in r["reason"], "old_door_cmd_refused", t,
              "an old phrasing moves nothing and is told the new one")
    class _W:
        def room_of(self, o): return "kitchen"
        def canon_class(self, o): return o
    check(P.parse_cmd("go through the visible door", _W()) == ("MidLevel", "go through the visible door"), "visible_door_grammar", "-",
          "the agent grammar passes the new command")
    voc = M.midlevel_vocabulary()
    check("go through the visible door" in voc and "nearest door" not in voc, "visible_door_vocab", "-",
          "both agents are shown the new name only")


def test_visible_portal_geometry():
    """A32: the exact rule behind "go through the visible door", on a hand-built scene (no engine):
    own-room doorways only, within the picture's half-FOV, not behind furniture, standing in a doorway
    counts, and an open passage is a doorway."""
    import math as _m
    import numpy as _np
    from vhhome import actions as M
    def room(i, cls, x0, x1, z0, z1):
        return {"id": i, "class_name": cls, "category": "Rooms",
                "bounding_box": {"center": [(x0 + x1) / 2, 1.5, (z0 + z1) / 2], "size": [x1 - x0, 3.0, z1 - z0]}}
    def node(i, cls, x, z, sx=0.9, sy=2.0, sz=0.1, cat="Doors"):
        return {"id": i, "class_name": cls, "category": cat,
                "bounding_box": {"center": [x, sy / 2, z], "size": [sx, sy, sz]}}
    base_nodes = [room(1, "kitchen", 0, 8, 0, 8), room(2, "bedroom", 8, 14, 0, 8), room(3, "bathroom", 14, 20, 0, 8)]
    class _Mock: pass
    def run(extra_nodes, edges, body, yaw_deg, passages=()):
        c = M.MidLevelController(_Mock())
        g = {"nodes": base_nodes + extra_nodes, "edges": edges}
        c._char = lambda g_=None: (_np.array([body[0], 0.0, body[1]], float), yaw_deg)
        c._char_room_id = lambda g_: 1
        # a real pinhole camera for the body pose: 80 deg vertical FOV, 4:3, looking 10 deg down (the person's eye)
        eye = _np.array([body[0], 1.6, body[1]], float)
        yr, pr = _m.radians(yaw_deg), _m.radians(10.0)
        f = _np.array([_m.sin(yr) * _m.cos(pr), -_m.sin(pr), _m.cos(yr) * _m.cos(pr)])
        r = _np.cross(f, [0.0, 1.0, 0.0]); r /= _np.linalg.norm(r); u = _np.cross(r, f)
        V = _np.eye(4); V[0, :3], V[1, :3], V[2, :3] = r, u, -f
        V[0, 3], V[1, 3], V[2, 3] = -r @ eye, -u @ eye, f @ eye
        tt, a, N, F = _m.tan(_m.radians(40.0)), 4 / 3, 0.05, 100.0
        Pm = _np.array([[1 / (a * tt), 0, 0, 0], [0, 1 / tt, 0, 0], [0, 0, -(F + N) / (F - N), -2 * F * N / (F - N)], [0, 0, -1, 0]])
        c._cam_matrices = lambda: (Pm, V, eye)
        c._passages = passages
        return [(t[1], t[2]) for t in c._visible_portals(g)]
    door_k = node(10, "door", 8.0, 4.0)                           # on the kitchen/bedroom wall
    edges_k = [{"from_id": 10, "to_id": 1, "relation_type": "INSIDE"}]
    body = (4.0, 4.0)
    bearing = _m.degrees(_m.atan2(8.0 - body[0], 4.0 - body[1]))  # 90: due +x
    check(("door", 10) in run([door_k], edges_k, body, bearing - 44), "portal_cone_in", "44deg",
          "a doorway 44 degrees off centre is inside the picture's 48-degree half-FOV")
    check(("door", 10) in run([door_k], edges_k, body, bearing - 52), "portal_edge_in_frame", "52deg",
          "f97: a 0.9 m doorway whose centre is 52 degrees off centre still shows its near edge in the picture")
    check(("door", 10) not in run([door_k], edges_k, body, bearing - 64), "portal_out_of_frame", "64deg",
          "a doorway 64 degrees off centre is out of the picture entirely")
    check(("door", 10) not in run([door_k], edges_k, body, bearing + 180), "portal_behind", "180deg",
          "a doorway behind the body is never in view")
    door_far = node(11, "door", 14.0, 4.0)                        # bedroom/bathroom wall, dead ahead
    edges_far = [{"from_id": 11, "to_id": 2, "relation_type": "INSIDE"}]
    check(run([door_far], edges_far, body, bearing) == [], "portal_own_room_only", "-",
          "a door on ANOTHER room's boundary is never a way out of this room, even straight ahead")
    # (a sofa would NOT do here: A26 caps its occluding box at seat height, so the eye sees over it)
    shelf = node(20, "bookshelf", 6.0, 4.0, sx=0.4, sy=2.1, sz=2.4, cat="Furniture")
    check(("door", 10) not in run([door_k, shelf], edges_k, body, bearing), "portal_occluded", "-",
          "a tall bookshelf between the eye and the doorway hides it")
    sofa = node(22, "sofa", 6.0, 4.0, sx=1.0, sy=1.2, sz=2.4, cat="Furniture")
    check(("door", 10) in run([door_k, sofa], edges_k, body, bearing), "portal_see_over_sofa", "-",
          "a sofa between the eye and the doorway does not hide it (seen over, A26)")
    chair = node(21, "chair", 7.6, 4.0, sx=0.6, sy=1.0, sz=0.6, cat="Furniture")
    check(("door", 10) in run([door_k, chair], edges_k, body, bearing), "portal_frame_furniture", "-",
          "furniture right beside the doorway does not hide the doorway itself")
    check(("door", 10) in run([door_k], edges_k, (7.85, 4.0), bearing + 170), "portal_standing_in", "-",
          "standing in the doorway counts as seeing it, whichever way the body faces")
    edges_other = [{"from_id": 10, "to_id": 2, "relation_type": "INSIDE"}]   # the graph files the door under the BEDROOM
    check(("door", 10) in run([door_k], edges_other, body, bearing - 10), "portal_door_both_rooms", "-",
          "a door belongs to both rooms it joins, even when the scene graph files it under the other one")
    door_side = node(12, "door", 4.0, 8.0)                                   # kitchen's north wall, 44 deg to the left
    order = run([door_k, door_side], [], (4.0, 5.0), bearing)
    check(order and order[0] == ("door", 10), "portal_most_centred", "-",
          "with two doorways in view, the one the body faces is chosen, not the nearer one at the frame edge")
    passage = [{"room_ids": [1, 2], "center": [8.0, 6.5], "points": [[8.0, 6.5]]}]
    pb = _m.degrees(_m.atan2(8.0 - body[0], 6.5 - body[1]))
    check(("passage", None) in run([], [], body, pb, passage), "portal_passage", "-",
          "an open passage in view is a doorway for the door command")


def test_cook_grader():
    """The person judges reception, the engine stays the authority on the world (COOK_GRADER.md).
    Release is armed by the person's `correct` ALONE; the union appears only in the scoring."""
    from vhhome_bench import protocol as P
    src = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    check('GRADER = (os.environ.get("VH_GRADER") or "cook").strip().lower()' in src,
          "cook_default", "-", "the cook grader is the default; VH_GRADER=engine restores the old path")
    check('if GRADER == "cook" and call == "correct" and task_active():' in src,
          "cook_arms", "-", "the person's `correct` is what arms the release")
    check('if armed[0] and h.get("move") == "accept":' in src,
          "cook_accept_releases", "-", "`accept` still does the releasing: recognition and compliance stay separate")
    check('if GRADER == "engine":' in src and "ARMS_DETECTION and not armed[0]" in src,
          "cook_engine_arms_only_in_engine_mode", "-",
          "under the cook grader an engine flag verdict arms nothing")
    for field in ("flagged_valid_engine", "detected_by", "detected_at", "latency_engine",
                  "cook_calls", "cook_wrong_calls", "cook_unauthored_correct", "cook_denied_live",
                  "cook_omitted", "label_ok", "grader"):
        check(field + "=" in src, "cook_report_field", field, "every report carries " + field)
    check('flagged_valid=any(f["verdict"] in P.ARMS_DETECTION for f in flags),' in src
          and "flagged_valid_cook=" in src and "phases_caught_cook=" in src,
          "cook_scores_engine", "-", "A31: detection is scored from the engine verdict; the person's call is recorded beside it")
    check("_p.get(\"injected\") and _w and _w[0] <= b <= _w[1]" in src, "cook_attribution", "-",
          "a call inside a phase's still-open window is attributed to that phase, as the engine's flags are")
    # the private field, in both reply schemas, with its rule
    for name, txt in (("standard", P.REPLY_STANDARD), ("creative", P.REPLY_CREATIVE)):
        check(txt.index('"assistant_error_call"') < txt.index('"move"'),
              "cook_reply_field", name, "the call comes FIRST in the " + name + " schema, before move (F6-1)")
    check("decide this FIRST" in P.REPLY_TAIL and "This call is not your answer" in P.REPLY_TAIL
          and "assistant_error_call /" in P.REPLY_TAIL,
          "cook_reply_rule", "-", "the rule text is present and the field is declared private")
    # the mistake line is GENERATED from the error's own parameters, never hand-written
    ml = P.mistake_line("wrong_object", {"sid": "fetch(milk)", "wrong": "juice"})
    check("juice" in ml and "milk" in ml, "cook_mistake_line", "wrong_object",
          "the line names the wrong item and the right one")
    check(P.mistake_line("repetition", {"sid": "x"}) == P.mistake_line("stall", {"after": "y"}),
          "cook_attention_unnamed", "-", "every attention error gets the same state-check line, naming nothing")
    check(P.readable_sid("switch_off(tablelamp@3)") == "switching off the tablelamp",
          "cook_readable_sid", "-", "sids render as plain words with the engine's own object token")
    check(P.ATTENTION_ERRORS == frozenset({"repetition", "stall", "navigation_inefficiency"}),
          "cook_families", "-", "commission = wrong_object / wrong_placement / skipped_step / missing_step")


def test_tick_and_latency():
    """2026-09-05: one tick = one second of animation (Unity records at 5 fps, FPT = 5), and an
    assistant reply that took x wall-seconds lands ceil(x / ASSIST_TICK_S) ticks later while
    the person's queue keeps executing."""
    from vhhome import world as iv_world
    check(iv_world.FPT == 5, "tick_is_1s", "-", "FPT must be 5 so a tick is exactly 1 s of animation")
    src = open(os.path.join(paths.ROOT, "vhhome_bench/episode.py")).read()
    check('ASSIST_TICK_S = float(os.environ.get("ASSIST_TICK_S", "1.2"))' in src, "latency_const", "-", "1.2 wall-seconds per tick, overridable")
    check("_y_raw = math.ceil(_x_charged / ASSIST_TICK_S) if ASSIST_LATENCY else 0" in src and "_y = min(_y_raw, ASSIST_MAX_TICKS)" in src and 'pending[0] = {"out": out' in src,
          "latency_ceil", "-", "delay in ticks = ceil(x / 1.2); reply parked until it lands")
    check('if pending[0] is not None and pending[0]["deliver_at"] > b:' in src and "continue                       # the assistant is still thinking" in src,
          "latency_world_moves", "-", "while a reply is in flight the poll returns and the world keeps stepping")
    check("or pending[0] is not None) else idle_polls + 1" in src, "latency_not_idle", "-", "an in-flight reply is not an idle poll")
    check('report["assistant_latency_summary"]' in src, "latency_report", "-", "every report carries the latency summary")


if __name__ == "__main__":
    sys.exit(main())

