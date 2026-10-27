"""Composite (multi-error) episode design and certification.

A composite episode carries TWO OR MORE injected errors, at distinct subgoal
anchors, of different types, verified never to conflict. This is the ScreenSim
analogue of a multi-fault benchmark episode.

Design contract (skill §5.1, §5.3):
  * each error is a transform on the plan editing its own dependency closure;
  * two errors must be NON-CONFLICTING — distinct anchors, disjoint committed
    paths, and in the combined trajectory BOTH deviations survive (neither
    masks the other) and the plan still executes with the goal unmet;
  * no two errors are twins (indistinguishable) — different kinds / controls;
  * every composite is certified by dry-replay before it is admitted.

Beat modes
  replace  : an in-place wrong choice (wrong_value / wrong_target) substitutes
             the node's commit gesture — the human commits the wrong thing.
  add      : an ADDITIVE foreign write — the human does the right thing AND an
             extra wrong actuation on the same screen that flips a goal-relevant
             path (e.g. also tapping "Visa" so the payment source is wrong).
  nav      : a navigation error (wrong_page). In the combined dry-replay it is
             injected then inline-recovered (models the assistant steering the
             human back) so the trajectory can reach the later beats; live, the
             takeover holds/releases it like any other error.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..tasks import Task, resolve_dynamic
from ..core.engine import Device
from .plan_ir import PlanIR, compile_plan, deep_diff
from .errors import (generate_cases, siblings_of, dry_replay, _commit_offsets)


@dataclass
class Beat:
    kind: str                 # wrong_value | wrong_target | wrong_page | extra_write
    mode: str                 # replace | add | nav
    node_idx: int
    alt_gesture: str
    correct_gesture: str
    repair: list              # gestures that undo just this beat
    paths: list               # committed paths this beat perturbs (for conflict test)
    label: str                # human-readable "what goes wrong"
    slip: str = ""            # private in-character slip text for the human
    prox: int = 99            # nav beats: row distance between the wrong turn and the right one
    pre: bool = False         # add beats: the extra tap happens BEFORE the node's own gesture


@dataclass
class Composite:
    id: str
    task: str
    beats: list               # ordered by anchor
    certified: dict = field(default_factory=dict)
    ok: bool = False
    reason: str = ""


# ---------------------------------------------------------------- foreign writes
def _foreign_writes(task, ir, node, pre=False):
    """ADDITIVE-error candidates at a node: foreign value-controls on the node's
    screen whose actuation flips a PLAN-TOUCHED path away from its reference
    value (so the goal actually breaks), without removing anything the plan does."""
    _, tgts = siblings_of(task, ir, node)
    touched = set(ir.touched_paths)
    ref = {p: v for p, v in ir.commits}
    # an ADDITIVE error must be a control the reference plan never actuates — a
    # genuinely EXTRA change. A control the plan already sets (a task target) is
    # not "extra"; tapping it again merely undoes a required setting, which is a
    # regression (a wrong_value), not an additive foreign write.
    plan_gestures = {g.lower().strip() for g in ir.flat()}
    # a path is ADDITIVE-eligible only if the plan carries it from initialization
    # and NEVER deliberately sets it at its own step (so flipping it is a genuine
    # EXTRA choice, not a twin of a dedicated control). Concretely: the only
    # commit node that touches it is the FIRST commit (the init) — e.g. the
    # payment source, initialized when the payee is picked but never chosen.
    # ignore the final irreversible commit — it re-writes every field as part of
    # finalizing (e.g. Verify & send), which is not a deliberate CHOICE of the
    # field and would mask an init-carried default.
    commit_nodes = [n for n in ir.nodes if n.kind == "commit" and not n.irreversible]
    first_commit = commit_nodes[0].idx if commit_nodes else -1
    def _init_only(path):
        owners = {n.idx for n in commit_nodes if path in {p for p, _ in n.commit}}
        return owners and owners <= {first_commit}
    out = []
    tail = [node.gesture] if (pre and node.gesture) else []
    for n in ir.nodes:
        if n.idx > node.idx:
            tail += n.gestures
    if pre:
        # the screen the node's gesture is about to act on: its visible controls
        probe = _state_after_node(task, ir, node.idx, pre=True)
        tgts = [f"tap {w.label}" for w in probe.laid_out_screen().widgets
                if w.visible and w.interactive() and w.label
                and w.kind in ("toggle", "radio", "row")]
    for g in tgts:
        if g.lower().strip() in plan_gestures:
            continue                       # a task target, not an extra change
        dev = _state_after_node(task, ir, node.idx, pre=pre)
        before_scr = dev.st.cur_screen_key()
        before = copy.deepcopy(dev.st.data)
        r = dev.act(resolve_dynamic(dev, g))
        if not r.ok or dev.st.cur_screen_key() != before_scr:
            continue                       # refused, or it navigated (not additive)
        changed = deep_diff(before, copy.deepcopy(dev.st.data))
        relevant = [p for p, v in changed
                    if p in touched and ref.get(p) != v and _init_only(p)]
        if not relevant and changed:
            # the authored "keep X alone" traps: an extra change the plan never
            # touches but the GOAL checks — finish the plan and see if it fails
            for gg in tail:
                dev.act(resolve_dynamic(dev, gg))
            ok, _ = task.goal(dev.st)
            if not ok:
                # still never a twin of a dedicated target: a path the plan sets
                # at its own step is a regression of that step, not an extra
                relevant = [p for p, _ in changed
                            if not p.split(".")[-1].startswith("_")
                            and (p not in touched or _init_only(p))]
        if relevant:
            out.append((g, relevant))
    return out


def _state_after_node(task, ir, node_idx, pre=False):
    """The device after node `node_idx` (or, with pre=True, after only its
    approach — the screen the node's own gesture is about to act on)."""
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    for n in ir.nodes:
        if n.idx == node_idx and pre:
            for gg in n.approach:
                dev.act(resolve_dynamic(dev, gg))
            return dev
        for gg in n.gestures:
            dev.act(resolve_dynamic(dev, gg))
        if n.idx == node_idx:
            return dev
    return dev


def _restore_gesture(task, ir, node, path, alt_gesture, pre=False):
    """The gesture that returns `path` to its reference value (the additive
    beat's repair) — the correct value-control on the node's screen.

    Probed from the WRONG state: first actuate the additive error, then find
    the visible control whose tap brings `path` back to the reference value.
    (Probing from the correct state finds nothing — re-selecting the already
    selected row changes no path — which left additive beats with no repair,
    so an accepted detection 'repaired' nothing and the error still committed.)"""
    # the reference is the value the CORRECT trajectory holds right after this
    # node (not the last value in ir.commits — a final commit may reset the form)
    ref = _get(_state_after_node(task, ir, node.idx, pre=pre).st.data, path)
    wrong = _state_after_node(task, ir, node.idx, pre=pre)
    if not wrong.act(resolve_dynamic(wrong, alt_gesture)).ok:
        return []
    if _get(wrong.st.data, path) == ref:
        return []                          # the 'error' changed nothing here
    for w in wrong.laid_out_screen().widgets:
        if not (w.visible and w.interactive() and w.label):
            continue
        probe = _state_after_node(task, ir, node.idx, pre=pre)
        probe.act(resolve_dynamic(probe, alt_gesture))
        scr = probe.st.cur_screen_key()
        r = probe.act(f"tap {w.label}")
        if not r.ok or probe.st.cur_screen_key() != scr:
            continue                       # refused, or it navigated away
        if _get(probe.st.data, path) == ref:
            return [f"tap {w.label}"]
    return []


def _deviated_paths(task, ir, node, alt_gesture):
    """The plan-touched paths whose VALUE actually differs between the correct
    commit and the wrong one — the beat's true perturbation (a payee tap that
    also initialises `source` does not 'perturb' source: both payees set it the
    same)."""
    touched = list(ir.touched_paths)
    def state(alt):
        dev = Device(task.seed)
        if task.start_app != "home":
            dev.open_app(task.start_app)
        for n in ir.nodes:
            if n.idx == node.idx:
                for g in n.approach:
                    dev.act(resolve_dynamic(dev, g))
                dev.act(resolve_dynamic(dev, alt))
                return copy.deepcopy(dev.st.data)
            for g in n.gestures:
                dev.act(resolve_dynamic(dev, g))
        return copy.deepcopy(dev.st.data)
    right, wrong = state(node.gesture), state(alt_gesture)
    return [p for p in touched if _get(right, p) != _get(wrong, p)]


def _navigates(task, ir, node, gesture):
    """Does actuating `gesture` at this node leave the node's screen?"""
    dev = _state_after_node(task, ir, node.idx) if False else None
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    for n in ir.nodes:
        if n.idx == node.idx:
            for g in n.approach:
                dev.act(resolve_dynamic(dev, g))
            here = dev.st.cur_screen_key()
            dev.act(resolve_dynamic(dev, gesture))
            return dev.st.cur_screen_key() != here
        for g in n.gestures:
            dev.act(resolve_dynamic(dev, g))
    return False


# ---------------------------------------------------------------- beats from cases
def _beats_from_cases(task, ir, cases):
    """Turn certified single-error cases into replace/nav beats, plus synthesize
    additive foreign-write beats where the task needs more independent errors."""
    beats = []
    for c in cases:
        if c.family != "error":
            continue
        node = ir.nodes[c.node_idx]
        if c.err in ("wrong_value", "wrong_target"):
            # the wrong choice must NOT be a control the plan actuates elsewhere:
            # re-tapping a target toggle just cancels a prior correct step (the
            # error self-destructs and shows nothing). A wrong toggle that turns
            # ON a NON-target control is the visible, non-cancelling error.
            plan_g = {g.lower().strip() for g in ir.flat()}
            if c.alt_gesture.lower().strip() in plan_g:
                continue
            # keep only substitutions that actually DEVIATE a committed path;
            # a tap that merely navigates elsewhere and commits nothing (an
            # app-row that opens another app) perturbs no path and is dropped —
            # a payee row that navigates AND sets the payee is kept.
            dev_paths = _deviated_paths(task, ir, node, c.alt_gesture)
            if not dev_paths:
                continue
            # PLAUSIBLE only: a wrong choice must be a near-miss — a row within
            # a few of the right one, or one sharing a word with it (the two
            # Alexes). A stranger three rows down is not a human slip.
            dist = _row_distance(task, ir, node, c.alt_gesture)
            if not (dist <= (1 if c.err == "wrong_value" else 3)
                    or _shares_word(c.alt_gesture, node.gesture or "")):
                continue
            beats.append(Beat(
                kind=c.err, mode="replace", node_idx=c.node_idx,
                alt_gesture=c.alt_gesture, correct_gesture=node.gesture or "",
                repair=c.repair, paths=dev_paths,
                label=_label(c, node)))
        elif c.err == "wrong_page":
            if _lands_on_plan_screen(task, ir, node, c.alt_gesture):
                continue                   # a page the plan visits anyway — not lost
            beats.append(Beat(
                kind="wrong_page", mode="nav", node_idx=c.node_idx,
                alt_gesture=c.alt_gesture, correct_gesture=node.gesture or "",
                repair=c.repair or ["press back", node.gesture or ""],
                paths=[], label=_label(c, node),
                prox=_row_distance(task, ir, node, c.alt_gesture)))
    # additive foreign writes: a distinct error type. After a commit node's
    # gesture, or — for a node whose approach changes the screen (dismisses a
    # keyboard, scrolls) — just BEFORE its gesture, where different controls
    # are exposed (the payment speed rows under the keyboard).
    for n in ir.nodes:
        if not n.gesture:
            continue
        probes = []
        if n.kind == "commit":
            probes.append(False)
        if n.approach:
            probes.append(True)
        for pre in probes:
            for g, paths in _foreign_writes(task, ir, n, pre=pre)[:2]:
                repair = _restore_gesture(task, ir, n, paths[0], g, pre=pre)
                if not repair:
                    continue               # an additive error with no undo is unusable
                leaf = paths[0].split(".")[-1]
                fixed = paths[0] in set(ir.touched_paths)
                beats.append(Beat(
                    kind="extra_write", mode="add", node_idx=n.idx,
                    alt_gesture=g, correct_gesture=n.gesture,
                    repair=repair, paths=paths, pre=pre,
                    label=(f"also {g.replace('tap ','taps ')} — changes the {leaf}, "
                           f"which the task fixed" if fixed else
                           f"also {g.replace('tap ','taps ')} — an extra change the "
                           f"task said to leave alone ({leaf})")))
    return beats


_WORD_STOP = {"tap", "the", "a", "an", "and", "of", "to", "on", "off", "in", "for"}


def _shares_word(a, b):
    import re as _re
    wa = {w for w in _re.findall(r"[a-z0-9]+", a.lower()) if w not in _WORD_STOP and len(w) > 2}
    wb = {w for w in _re.findall(r"[a-z0-9]+", b.lower()) if w not in _WORD_STOP and len(w) > 2}
    return bool(wa & wb)


def _plan_screens(task, ir):
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    seen = {dev.st.cur_screen_key()}
    for g in ir.flat():
        dev.act(resolve_dynamic(dev, g))
        seen.add(dev.st.cur_screen_key())
    return seen


def _lands_on_plan_screen(task, ir, node, alt_gesture):
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    for n in ir.nodes:
        if n.idx == node.idx:
            for g in n.approach:
                dev.act(resolve_dynamic(dev, g))
            break
        for g in n.gestures:
            dev.act(resolve_dynamic(dev, g))
    dev.act(resolve_dynamic(dev, alt_gesture))
    return dev.st.cur_screen_key() in _plan_screens(task, ir)


def _row_distance(task, ir, node, alt_gesture):
    """How many rows apart the wrong turn and the right row are on the screen
    where the turn is taken (1 = neighbours). A near-miss is the realistic
    wrong turn; the account banner at the top of Settings is not."""
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    for n in ir.nodes:
        if n.idx == node.idx:
            for g in n.approach:
                dev.act(resolve_dynamic(dev, g))
            break
        for g in n.gestures:
            dev.act(resolve_dynamic(dev, g))
    labels = [w.label for w in dev.laid_out_screen().widgets
              if w.label and w.interactive()]
    def _find(g):
        t = (g or "").lower().replace("tap ", "").strip()
        for i, l in enumerate(labels):
            if l.lower() == t:
                return i
        return None
    a, b = _find(alt_gesture), _find(node.gesture)
    return abs(a - b) if a is not None and b is not None else 99


def _label(case, node):
    right = (node.gesture or "").replace("tap ", "")
    wrong = case.alt_gesture.replace("tap ", "")
    if case.err == "wrong_page":
        return f"takes the wrong turn to reach '{right}' — ends up lost"
    if case.err == "wrong_value":
        return f"picks '{wrong}' instead of '{right}'"
    if case.err == "wrong_target":
        return f"taps '{wrong}' by mistake — so '{right}' is left unset"
    return f"{case.err} at {right}"


# ---------------------------------------------------------------- combined plan
def _combined_plan(ir, beats):
    """Build one flat gesture plan applying every beat. Substitutions replace
    the node's commit; additive beats append after it; nav beats inject the
    wrong turn then inline-recover so the trajectory reaches later beats."""
    by_node = {b.node_idx: b for b in beats}
    out = []
    for n in ir.nodes:
        b = by_node.get(n.idx)
        if b is None:
            out += n.gestures
            continue
        if b.mode == "replace":
            out += n.approach + [b.alt_gesture]
        elif b.mode == "add":
            if b.pre:
                out += n.approach + [b.alt_gesture, n.gesture]
            else:
                out += n.gestures + [b.alt_gesture]
        elif b.mode == "nav":
            # wrong turn, then step back and take the right one (inline recovery)
            out += n.approach + [b.alt_gesture, "press back", n.gesture]
    return out


# ---------------------------------------------------------------- certification
def certify_composite(task, ir, beats, ref_final):
    """Admit a composite iff distinct anchors, disjoint perturbed paths, and the
    combined trajectory executes with BOTH deviations surviving and goal unmet."""
    idxs = [b.node_idx for b in beats]
    if len(set(idxs)) != len(idxs):
        return False, "anchors not distinct", {}
    alts = [b.alt_gesture.lower().strip() for b in beats]
    if len(set(alts)) != len(alts):
        return False, "same control twice (self-cancelling)", {}
    # disjoint perturbed paths (nav beats perturb none)
    seen = set()
    for b in beats:
        ps = set(b.paths)
        if ps & seen:
            return False, f"paths conflict at {ps & seen}", {}
        seen |= ps
    plan = _combined_plan(ir, beats)
    rec = dry_replay(task, plan)
    if rec["refusals"]:
        return False, f"combined plan refused: {rec['refusals'][0]}", {}
    if rec["goal_ok"]:
        return False, "goal still met (an error masked another / cancelled)", {}
    # NON-MASKING: every COMMITTING beat must break the goal on its own — so no
    # error is silently absorbed by another, and each is a genuine fault. (A nav
    # beat is recovered inline by design and graded on detection, not final
    # state, so it is exempt here; its single-error validity was already
    # certified upstream.)
    committing = [b for b in beats if b.mode != "nav"]
    for b in committing:
        p1 = _combined_plan(ir, [b])
        r1 = dry_replay(task, p1)
        if r1["refusals"]:
            return False, f"{b.kind}@{b.node_idx} refused alone", {}
        if r1["goal_ok"]:
            return False, f"{b.kind}@{b.node_idx} does not break the goal alone", {}
        # RECOVERABLE: the beat's repair, spliced right after its slip (what an
        # accepted detection does at run time), must execute and meet the goal.
        # An error with no working undo cannot be 'prevented' by any assistant.
        if not b.repair or b.alt_gesture not in p1:
            return False, f"{b.kind}@{b.node_idx} has no repair", {}
        k = p1.index(b.alt_gesture)
        r2 = dry_replay(task, p1[:k + 1] + list(b.repair) + p1[k + 1:])
        if r2["refusals"] or not r2["goal_ok"]:
            return False, f"{b.kind}@{b.node_idx} repair does not recover the goal", {}
    return True, "", {"plan_len": len(plan),
                      "outcome": [b.kind for b in beats],
                      "committing": len(committing)}


def _get(data, path):
    cur = data
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


# ---------------------------------------------------------------- design
def design_composite_episodes(task, per_task=3):
    """`per_task` certified composite episodes of DISTINCT shapes, each >=2
    errors at distinct anchors and of >=2 different kinds:

      shape A — navigation + downstream slip : get lost on the way, then commit
                a wrong value/target once back on track (an escalating episode).
      shape B — two committed slips          : two wrong value/target choices at
                different subgoals, no navigation error (a 'both toggles wrong').
      shape C — a slip + an unwanted extra   : a wrong choice plus an ADDITIVE
                foreign write the task never asked for.

    Controls are diversified across episodes; every candidate is dry-replay
    certified (executable, goal unmet, both deviations survive, disjoint paths).
    """
    ir = compile_plan(task)
    # a wider per-kind pool so non-target foreign controls are available after
    # the plan-target alts are filtered out (§_beats_from_cases)
    _, cases, _ = generate_cases(task, personas=("distracted_average",),
                                 max_per_kind=8)
    ref_final = dry_replay(task, ir.flat())["dev"].st.data
    beats = _beats_from_cases(task, ir, cases)

    # wrong turns: the NEAREST plausible one first (a neighbouring row), then by position
    nav = sorted([b for b in beats if b.kind == "wrong_page"], key=lambda b: (b.prox, b.node_idx))
    inplace = sorted([b for b in beats if b.kind in ("wrong_value", "wrong_target")],
                     key=lambda b: b.node_idx)
    extra = sorted([b for b in beats if b.kind == "extra_write"], key=lambda b: b.node_idx)

    episodes = []
    used_nodes_sets = set()

    def try_add(chosen, tag):
        if len(episodes) >= per_task or chosen is None or any(b is None for b in chosen):
            return
        nodes = frozenset(b.node_idx for b in chosen)
        if nodes in used_nodes_sets:
            return
        if len({b.kind for b in chosen}) < 2:
            return
        ok, reason, cert = certify_composite(task, ir,
                                             sorted(chosen, key=lambda b: b.node_idx),
                                             ref_final)
        if ok:
            used_nodes_sets.add(nodes)
            episodes.append(Composite(
                id=f"{task.id}__c{len(episodes)+1}", task=task.id,
                beats=sorted(chosen, key=lambda b: b.node_idx),
                certified={**cert, "shape": tag}, ok=True))

    def first_at_other_node(pool, avoid):
        for b in pool:
            if b.node_idx not in avoid:
                return b
        return None

    # shape A — navigation + a downstream committed slip (or additive)
    if nav:
        downstream = [b for b in (inplace + extra) if b.node_idx != nav[0].node_idx]
        downstream.sort(key=lambda b: b.node_idx)
        for d in downstream:                  # first committed slip after the nav
            try_add([nav[0], d], "A · nav→slip")
            if episodes:
                break

    # shape B — two committed slips at different nodes
    for i in range(len(inplace)):
        b2 = first_at_other_node(inplace[i + 1:], {inplace[i].node_idx})
        if b2:
            try_add([inplace[i], b2], "B · two slips")
            if len([e for e in episodes if e.certified.get("shape", "").startswith("B")]):
                break
    # fall back for B: a slip + additive if fewer than two in-place nodes
    if not any(e.certified.get("shape", "").startswith("B") for e in episodes) and inplace and extra:
        e2 = first_at_other_node(extra, {inplace[0].node_idx})
        if e2:                                # every extra write sits on the slip's own node
            try_add([inplace[0], e2], "B · slip+extra")

    # shape C — a committed slip + an unwanted extra write (distinct nodes)
    if inplace and extra:
        for ip in inplace:
            ex = first_at_other_node(extra, {ip.node_idx})
            if ex:
                try_add([ip, ex], "C · slip+extra")
                if any(e.certified.get("shape", "").startswith("C") for e in episodes):
                    break
    # fall back for C: nav + additive
    if not any(e.certified.get("shape", "").startswith("C") for e in episodes) and nav and extra:
        ex = first_at_other_node(extra, {nav[0].node_idx})
        try_add([nav[0], ex], "C · nav+extra")

    # top up to per_task: when a task has little committed-error variety (e.g.
    # a single slippable control), pair the SAME slip with different navigation
    # failures — distinct episodes that get lost at different points.
    # Diversity: rank candidate pairs so that controls not yet used by an
    # admitted episode come first, then near-miss wrong turns, then position;
    # a pair may reuse a node set as long as its two controls are a new pair.
    pool = nav + inplace + extra
    from itertools import combinations
    def _used():
        # settings already broken by an admitted episode (nav beats: their step)
        return {(tuple(sorted(b.paths)) if b.mode != "nav" else ("nav", b.node_idx))
                for e in episodes for b in e.beats}
    cands = [list(c) for c in combinations(pool, 2)
             if c[0].node_idx != c[1].node_idx and len({c[0].kind, c[1].kind}) >= 2]
    def _sig(chosen):
        # what makes an episode distinct: the SETTINGS its committing mistakes
        # break (not which decoy row was tapped) and the STEP of its wrong turn
        # (which neighbouring row it hits does not count)
        return frozenset((b.kind, tuple(sorted(b.paths))) if b.mode != "nav" else ("nav", b.node_idx)
                         for b in chosen)
    while len(episodes) < per_task and cands:
        used = _used()
        seen_sigs = {_sig(e.beats) for e in episodes}
        cands = [c for c in cands if _sig(c) not in seen_sigs]
        if not cands:
            break
        used_alts = {b.alt_gesture for e in episodes for b in e.beats}
        cands.sort(key=lambda c: (sum(1 for b in c if (tuple(sorted(b.paths)) if b.mode != "nav" else ("nav", b.node_idx)) in used),
                                  sum(1 for b in c if b.alt_gesture in used_alts),
                                  sum(b.prox for b in c if b.mode == "nav"),
                                  c[0].node_idx + c[1].node_idx))
        c = cands.pop(0)
        before = len(episodes)
        nodes = frozenset(b.node_idx for b in c)
        used_nodes_sets.discard(nodes)         # a new control pair may share a node set
        try_add(c, "· variant")
        if len(episodes) == before:
            continue

    # last resort (a task with NO navigation errors and one slippable control,
    # e.g. a payment flow): pair a slip with a SECOND additive control on the
    # same screen (the speed row where the source row was used), or a different
    # wrong choice with the first one. Node sets may repeat here; the pair of
    # actuated controls must be new.
    if len(episodes) < per_task:
        seen_alts = {frozenset(b.alt_gesture for b in e.beats) for e in episodes}
        extra2 = []
        for n in ir.nodes:
            if n.kind != "commit" or not n.gesture:
                continue
            for g, paths in _foreign_writes(task, ir, n)[1:3]:
                rep = _restore_gesture(task, ir, n, paths[0], g)
                if rep:
                    extra2.append(Beat(
                        kind="extra_write", mode="add", node_idx=n.idx,
                        alt_gesture=g, correct_gesture=n.gesture, repair=rep,
                        paths=paths, pre=False,
                        label=f"also {g.replace('tap ','taps ')} — an extra change "
                              f"the task did not ask for ({paths[0].split('.')[-1]})"))
        for ip in inplace:
            for ex in extra2 + extra:
                if len(episodes) >= per_task:
                    break
                if ex.node_idx == ip.node_idx:
                    continue
                alts = frozenset([ip.alt_gesture, ex.alt_gesture])
                if alts in seen_alts:
                    continue
                ok, reason, cert = certify_composite(
                    task, ir, sorted([ip, ex], key=lambda b: b.node_idx), ref_final)
                if ok:
                    seen_alts.add(alts)
                    episodes.append(Composite(
                        id=f"{task.id}__c{len(episodes)+1}", task=task.id,
                        beats=sorted([ip, ex], key=lambda b: b.node_idx),
                        certified={**cert, "shape": "C · slip+extra₂"}, ok=True))

    return ir, episodes


def combined_stream(ir, beats):
    """The combined plan as annotated steps for the multi-error runner. Each
    step is (gesture, role, beat) where role ∈ {plain, approach, slip, HOLD}.
    A HOLD is emitted before an irreversible commit; a `slip` step carries its
    Beat so the loop can track per-error injection, detection and repair."""
    by_node = {b.node_idx: b for b in beats}
    steps = []
    for n in ir.nodes:
        b = by_node.get(n.idx)
        if n.kind == "commit" and n.irreversible:
            steps.append({"gesture": None, "role": "HOLD", "beat": None, "node": n.idx})
        if b is None:
            for g in n.gestures:
                steps.append({"gesture": g, "role": "plain", "beat": None, "node": n.idx})
        elif b.mode == "replace":
            for g in n.approach:
                steps.append({"gesture": g, "role": "approach", "beat": None, "node": n.idx})
            steps.append({"gesture": b.alt_gesture, "role": "slip", "beat": b, "node": n.idx})
        elif b.mode == "add":
            if b.pre:
                for g in n.approach:
                    steps.append({"gesture": g, "role": "approach", "beat": None, "node": n.idx})
                steps.append({"gesture": b.alt_gesture, "role": "slip", "beat": b, "node": n.idx})
                steps.append({"gesture": n.gesture, "role": "plain", "beat": None, "node": n.idx})
            else:
                for g in n.gestures:
                    steps.append({"gesture": g, "role": "plain", "beat": None, "node": n.idx})
                steps.append({"gesture": b.alt_gesture, "role": "slip", "beat": b, "node": n.idx})
        elif b.mode == "nav":
            for g in n.approach:
                steps.append({"gesture": g, "role": "approach", "beat": None, "node": n.idx})
            steps.append({"gesture": b.alt_gesture, "role": "slip", "beat": b, "node": n.idx})
            # the wrong turn lands the human elsewhere; the correct nav is the
            # per-beat repair (spliced on an accepted correct detection)
    return steps


def all_composites(per_task=3):
    from ..tasks import by_id
    from ..tasks import HARD_TASKS as HARD
    out = {}
    for tid in HARD:
        ir, eps = design_composite_episodes(by_id(tid), per_task=per_task)
        out[tid] = eps
    return out
