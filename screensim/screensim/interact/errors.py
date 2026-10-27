"""Error transforms on the plan IR, and dry-replay certification of cases.

The taxonomy (transforms on subgoal nodes, approaches always kept so the
navigation closure survives — the dry replay then validates every remaining
gesture still grounds):

  wrong_value   the commit gesture picks a SIBLING OPTION of the same radio
                group ("tap Never" for "tap While Using the App")
  wrong_target  the commit gesture lands on a DIFFERENT CONTROL on the same
                screen ("tap Allow Notifications" for "tap Sounds")
  skip_step     the commit gesture is deleted; the human walks the screen and
                leaves without acting
  early_commit  a contiguous range (>=2) of commit gestures before an
                irreversible confirm is deleted; the human confirms with the
                defaults still in place

Plus the non-error families: `intent` (same edit, but the human ANNOUNCES the
change and the transformed value becomes the goal — the assistant must stay
quiet) and `clean` (no transform).

A case is admitted only if the dry replay under CONVERSATION PACE certifies:
anchor fires (no collapse to t0), every gesture grounds, the goal predicate
fails (error) or the new goal holds (intent), and a repair is validated where
the world allows one. Rejections go to the ledger with reasons (skill §5.3).
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

from ..core import pacing
from ..core.engine import Device
from ..tasks import Task, resolve_dynamic
from .plan_ir import PlanIR, Node, compile_plan, deep_diff

ERROR_KINDS = ("wrong_value", "wrong_target", "skip_step", "early_commit",
               "wrong_page")


@dataclass
class Case:
    name: str
    family: str                 # error | intent | clean
    err: str                    # one of ERROR_KINDS, or "none"
    task: str
    node_idx: int               # -1 for clean
    alt_gesture: str            # replacement commit gesture ("" for skip)
    anchor_k: int               # commits before the edited node
    anchor: list                # the commit prefix itself
    plan: list                  # transformed flat gesture list
    repair: list                # validated repair gestures ([] = prevention-only)
    announce: str = ""          # intent family: what the human says at the switch
    deviated_label: str = ""    # the correct control the error touched/omitted
    certified: dict = field(default_factory=dict)
    persona: str = ""
    interface: str = "sv1"


def _commit_offsets(ir: PlanIR):
    """anchor_k per node: number of commits strictly before node idx."""
    out, k = [], 0
    for n in ir.nodes:
        out.append(k)
        k += len(n.commit)
    return out


def _flat_with_edit(ir: PlanIR, node_idx, replacement):
    """Flatten the plan with node_idx's commit gesture replaced (or dropped
    when replacement is None). Approaches always survive."""
    out = []
    for n in ir.nodes:
        if n.idx == node_idx and n.kind == "commit":
            out += n.approach
            if replacement:
                out.append(replacement)
        else:
            out += n.gestures
    return out


def _flat_drop_range(ir: PlanIR, lo, hi):
    out = []
    for n in ir.nodes:
        if lo <= n.idx <= hi and n.kind == "commit":
            out += n.approach
        else:
            out += n.gestures
    return out


def siblings_of(task: Task, ir: PlanIR, node: Node):
    """Alternative commit gestures on the node's screen, split by kind:

      wrong_value  — controls that change EXACTLY the node's own commit path
                     (a radio's siblings in the same group; a toggle's flip)
      wrong_target — controls that change a DIFFERENT path (a foreign control)

    Membership is decided by dry-tapping each candidate from the pre-node
    state and comparing the path it changes to the node's committed path — so
    a note field's node never treats the source radios as its 'values'.
    """
    node_paths = {p for p, _ in node.commit}

    def _state_at_node():
        dev = Device(task.seed)
        if task.start_app != "home":
            dev.open_app(task.start_app)
        for n in ir.nodes:
            if n.idx == node.idx:
                for raw in n.approach:
                    dev.act(resolve_dynamic(dev, raw))
                return dev
            for raw in n.gestures:
                dev.act(resolve_dynamic(dev, raw))
        return dev

    base = _state_at_node()
    scr = base.laid_out_screen()
    target = (node.gesture or "").replace("tap ", "").strip().lower()
    # radios/toggles are same-screen value choices; a "row" that both navigates
    # AND writes committed state (a payee pick — "the wrong Alex") is a genuine
    # forceable wrong choice too. The dry-tap below keeps only candidates that
    # actually change committed state, so pure-navigation rows are dropped.
    cands = [w for w in scr.widgets
             if w.visible and w.interactive() and w.label
             and w.label.strip().lower() != target
             and w.kind in ("radio", "toggle", "row")]
    vals, tgts = [], []
    for w in cands:
        probe = _state_at_node()
        before = copy.deepcopy(probe.st.data)
        r = probe.act(f"tap {w.label}")
        if not r.ok:
            continue
        changed = {p for p, _ in deep_diff(before, copy.deepcopy(probe.st.data))}
        if changed and changed <= node_paths:
            vals.append(f"tap {w.label}")       # same control/group
        elif changed:
            tgts.append(f"tap {w.label}")       # foreign control
    return vals, tgts


def nav_siblings(task: Task, ir: PlanIR, node: Node):
    """For a REACH subgoal (a required navigation), the other nav rows on the
    same source screen that lead SOMEWHERE ELSE — the plausible wrong turns.
    Returns [(gesture, screen_it_leads_to)]."""
    def _at_from():
        dev = Device(task.seed)
        if task.start_app != "home":
            dev.open_app(task.start_app)
        for n in ir.nodes:
            if n.idx == node.idx:
                for g in n.approach:
                    dev.act(resolve_dynamic(dev, g))
                return dev
            for g in n.gestures:
                dev.act(resolve_dynamic(dev, g))
        return dev

    base = _at_from()
    scr = base.laid_out_screen()
    correct = (node.gesture or "").replace("tap ", "").strip().lower()
    out = []
    for w in scr.widgets:
        if not (w.visible and w.interactive() and w.label and w.on_tap):
            continue
        if w.on_tap[0] != "nav":
            continue
        if w.label.strip().lower() == correct:
            continue
        probe = _at_from()
        before = probe.st.cur_screen_key()
        r = probe.act(f"tap {w.label}")
        dest = probe.st.cur_screen_key()
        if r.ok and dest != before and dest != node.screen:
            out.append((f"tap {w.label}", dest))
    return out


def dry_replay(task: Task, plan, goal=None):
    """Headless replay under conversation pace. Returns a record of per-tick
    commits, refusals, and the goal verdict."""
    pacing.apply_conversation_pace()
    try:
        dev = Device(task.seed)
        if task.start_app != "home":
            dev.open_app(task.start_app)
        prev = copy.deepcopy(dev.st.data)
        commits, refusals, gesture_ticks = [], [], []
        for i, raw in enumerate(plan):
            r = dev.act(resolve_dynamic(dev, raw))
            gesture_ticks.append(dev.st.tick)
            cur = copy.deepcopy(dev.st.data)
            for c in deep_diff(prev, cur):
                commits.append((dev.st.tick, c))
            prev = cur
            if not r.ok:
                refusals.append((i, raw, r.reason))
        ok, detail = (goal or task.goal)(dev.st)
        return {"dev": dev, "commits": commits, "refusals": refusals,
                "gesture_ticks": gesture_ticks, "goal_ok": ok,
                "goal_detail": detail, "end_tick": dev.st.tick}
    finally:
        pacing.reset()


def certify(task: Task, ir: PlanIR, case: Case, ledger: list,
            ref_final=None) -> bool:
    """Dry-replay the transformed plan; fill case.certified or write the
    rejection to the ledger. Returns admitted?"""

    def reject(reason):
        ledger.append({"name": case.name, "task": task.id, "err": case.err,
                       "node_idx": case.node_idx, "rejected": reason})
        return False

    # ---- wrong_page: the human navigates to the WRONG screen and is stuck ----
    if case.err == "wrong_page":
        rec = dry_replay(task, case.plan)          # prefix + the wrong nav
        if rec["refusals"]:
            return reject(f"wrong_nav_refused: {rec['refusals'][0]}")
        if rec["goal_ok"]:
            return reject("goal_met_on_wrong_page")   # not actually lost
        # repair: go back and take the correct turn, then finish
        node = ir.nodes[case.node_idx]
        prefix = []
        for n in ir.nodes:
            if n.idx < case.node_idx:
                prefix += n.gestures
            elif n.idx == case.node_idx:
                prefix += n.approach + [case.alt_gesture]
                break
        tail = [g for n in ir.nodes if n.idx > case.node_idx for g in n.gestures]
        repair = ["press back", node.gesture]
        rec2 = dry_replay(task, prefix + repair + tail)
        if not (rec2["goal_ok"] and not rec2["refusals"]):
            return reject("no_validated_back_repair")
        case.repair = repair
        gt = rec["gesture_ticks"]
        idc = gt[min(len(case.plan) - 1, len(gt) - 1)] if gt else 0
        case.deviated_label = _pretty_screen(node.screen)
        case.certified = {
            "fires": True, "anchor_collapses_to_t0": case.anchor_k == 0,
            "identifiable_at": idc, "window": [idc, rec["end_tick"]],
            "outcome": "wrong_page", "outcome_tick": rec["end_tick"],
            "irreversible": False, "goal_detail": rec["goal_detail"],
            "unique_correct_answer": True, "ref_final_diff": [],
            "wrong_dest": rec["dev"].st.cur_screen_key(), "pace": "conversation",
        }
        return True

    if case.anchor_k == 0 and case.family == "error":
        # anchor collapse: fires at t0 and amputates nothing — the deviation
        # IS the first committed choice. Allowed when this node is the FIRST
        # commit (a leading reach/navigation prefix is fine — e.g. the payee
        # pick after 'Send money'); refuse only when a prior COMMIT exists.
        prior_commit = any(n.kind == "commit" and n.idx < case.node_idx
                           for n in ir.nodes)
        if prior_commit:
            return reject("anchor_collapses_to_t0")

    rec = dry_replay(task, case.plan)
    if rec["refusals"]:
        return reject(f"gesture_refused: {rec['refusals'][0]}")
    if case.family == "error" and rec["goal_ok"]:
        return reject("goal_still_met_after_transform")

    # excess deviation vs the reference's final state — what the error
    # actually changed beyond the plan's intent
    excess = []
    if ref_final is not None:
        excess = deep_diff(rec["dev"].st.data, ref_final)

    if case.family == "intent":
        # an announced change is coherent only if the transformed value is
        # the SOLE deviation — i.e. every differing path belongs to the
        # edited node's own commit set
        node_paths = {p for p, _ in ir.nodes[case.node_idx].commit}
        if not excess:
            return reject("intent_changes_nothing")
        if not {p for p, _ in excess} <= node_paths:
            return reject("intent_not_sole_deviation")

    # where does the deviation become observable?
    edited_flat = case.plan
    ref_flat = ir.flat()
    div = next((i for i, (a, b) in enumerate(zip(edited_flat, ref_flat))
                if a != b), min(len(edited_flat), len(ref_flat)))
    if div >= len(rec["gesture_ticks"]):
        div = len(rec["gesture_ticks"]) - 1
    identifiable_at = rec["gesture_ticks"][div]

    irrev = any(p.startswith(("bank.history", "bank.balance", "shop.order",
                              "mail.sent"))
                for _, (p, _) in rec["commits"])
    commit_tick = rec["end_tick"]
    outcome = "benign" if case.family != "error" else (
        "wrong_commit" if irrev else "wrong_final_state")

    # validate a repair for reversible errors: cheapest candidate that makes
    # the goal true when appended after the flawed plan. Repairability is a
    # property of the DEVIATED NODE (a wrong source before an irreversible
    # confirm is still fixable), not of whether the plan ends irreversibly.
    node_irrev = (case.node_idx >= 0 and ir.nodes[case.node_idx].irreversible)
    repair = []
    if case.family == "error" and not node_irrev and case.node_idx >= 0:
        node = ir.nodes[case.node_idx]
        # replay the fix WHERE IT HAPPENS: everything up through the flawed
        # node, then the repair, then the reference tail (so a wrong source
        # is re-selected before the confirm, not after the money is gone).
        prefix = []
        for n in ir.nodes:
            if n.idx < case.node_idx:
                prefix += n.gestures
            elif n.idx == case.node_idx:
                prefix += n.approach
                if case.alt_gesture:
                    prefix.append(case.alt_gesture)
                break
        tail = []
        for n in ir.nodes:
            if n.idx > case.node_idx:
                tail += n.gestures
        cands = []
        if node.gesture:                       # re-do the node correctly
            cands.append([node.gesture])
            # a wrong choice made via a NAVIGATING row (e.g. the wrong payee)
            # can't be re-selected in place — step back to the picker first
            cands.append(["press back", node.gesture])
            # a wrong TARGET that switched ON a foreign toggle must switch it
            # back OFF (re-tap it) before doing the correct control
            if case.alt_gesture and case.alt_gesture != node.gesture:
                cands.append([case.alt_gesture, node.gesture])
            # a wrong TARGET that moved a RADIO (re-tapping it restores nothing)
            # must re-select the reference row, then do the correct control
            for p, ref_v in excess:
                if isinstance(ref_v, str) and 0 < len(ref_v) <= 40:
                    cands.append([f"tap {ref_v}", node.gesture])
        for p, ref_v in excess:                # restore an excess string value
            if isinstance(ref_v, str) and 0 < len(ref_v) <= 40 \
                    and p not in {q for q, _ in node.commit}:
                cands.append([f"tap {ref_v}"])
        cands.append(list(node.gestures))
        # a repair must RESTORE THE WORLD, not merely satisfy the goal: the
        # cheapest candidate whose final state equals the reference plan's on
        # every path. (Goal-only validation accepted "just do the correct tap"
        # for a wrong target, leaving the foreign toggle the slip switched on —
        # a residual change the task never asked for.)
        ref_data = dry_replay(task, ir.flat())["dev"].st.data
        for cand in cands:
            rec2 = dry_replay(task, prefix + cand + tail)
            if rec2["goal_ok"] and not rec2["refusals"] \
                    and rec2["dev"].st.data == ref_data:
                repair = cand
                break
        if not repair:
            return reject("no_validated_repair")
    case.repair = repair

    # the correct control the error concerns, for the grader's term set
    node = ir.nodes[case.node_idx] if case.node_idx >= 0 else None
    if node and node.gesture:
        case.deviated_label = node.gesture.replace("tap ", "").strip()

    case.certified = {
        "fires": True, "anchor_collapses_to_t0": case.anchor_k == 0,
        "identifiable_at": identifiable_at,
        "window": [identifiable_at, commit_tick],
        "outcome": outcome, "outcome_tick": commit_tick,
        "irreversible": irrev, "goal_detail": rec["goal_detail"],
        "unique_correct_answer": True,
        "ref_final_diff": [list(c) for c in excess],
        "pace": "conversation",
    }
    return True


def generate_cases(task: Task, personas=("plain",), max_per_kind=2):
    """All admissible cases for one task, plus the ledger of rejections."""
    ir = compile_plan(task)
    offsets = _commit_offsets(ir)
    cases, ledger = [], []
    commit_nodes = [n for n in ir.nodes if n.kind == "commit"]
    ref_final = dry_replay(task, ir.flat())["dev"].st.data

    def add(case):
        if certify(task, ir, case, ledger, ref_final=ref_final):
            cases.append(case)

    for n in commit_nodes:
        k = offsets[n.idx]
        anchor = [list(c) for c in ir.commits[:k]]
        vals, tgts = ([], []) if not n.gesture else siblings_of(task, ir, n)
        for alt in vals[:max_per_kind]:
            add(Case(name=f"{task.id}__wv_n{n.idx}_{_slug(alt)}",
                     family="error", err="wrong_value", task=task.id,
                     node_idx=n.idx, alt_gesture=alt, anchor_k=k,
                     anchor=anchor,
                     plan=_flat_with_edit(ir, n.idx, alt), repair=[]))
        for alt in tgts[:max_per_kind]:
            add(Case(name=f"{task.id}__wt_n{n.idx}_{_slug(alt)}",
                     family="error", err="wrong_target", task=task.id,
                     node_idx=n.idx, alt_gesture=alt, anchor_k=k,
                     anchor=anchor,
                     plan=_flat_with_edit(ir, n.idx, alt), repair=[]))
        # skip_step removed from the benchmark (2026-08-26): an OMISSION cannot
        # be forced under soft-hard — the engine can only suppress the toggle
        # once the human is already on the anchor screen, so a free-form human
        # that quits early produces an un-scoreable void, not the designed
        # mistake. Only forceable errors remain (wrong_value/target/page).
        # intent: the same wrong_value edit, announced and adopted as the goal
        for alt in vals[:1]:
            c = Case(name=f"{task.id}__intent_n{n.idx}_{_slug(alt)}",
                     family="intent", err="none", task=task.id,
                     node_idx=n.idx, alt_gesture=alt, anchor_k=k,
                     anchor=anchor,
                     plan=_flat_with_edit(ir, n.idx, alt), repair=[],
                     announce=f"Actually, change of plan — I'll go with "
                              f"{alt.replace('tap ', '')} instead.")
            add(c)

    # wrong_page: at a required navigation, tap a plausible sibling row and
    # land on the WRONG screen (the human is lost). The most realistic errors
    # in a long-horizon task are getting lost, not just fat-fingering a switch.
    reach_nodes = [n for n in ir.nodes if n.kind == "reach"]
    for n in reach_nodes:
        k = offsets[n.idx]
        anchor = [list(c) for c in ir.commits[:k]]
        sibs = nav_siblings(task, ir, n)
        for wrong, dest in sibs[:max_per_kind]:
            add(Case(name=f"{task.id}__wp_n{n.idx}_{_slug(wrong)}",
                     family="error", err="wrong_page", task=task.id,
                     node_idx=n.idx, alt_gesture=wrong, anchor_k=k,
                     anchor=anchor, plan=_flat_wrong_page(ir, n.idx, wrong),
                     repair=[]))

    # early_commit: delete >=2 commit gestures before the irreversible confirm
    irrs = [n.idx for n in commit_nodes if n.irreversible]
    if irrs:
        j = irrs[0]
        prior = [n.idx for n in commit_nodes if n.idx < j]
        if len(prior) >= 3:
            lo, hi = prior[-2], prior[-1]
            k = offsets[lo]
            add(Case(name=f"{task.id}__early_n{lo}_{hi}",
                     family="error", err="early_commit", task=task.id,
                     node_idx=lo, alt_gesture="", anchor_k=k,
                     anchor=[list(c) for c in ir.commits[:k]],
                     plan=_flat_drop_range(ir, lo, hi), repair=[]))

    # clean control
    cases.append(Case(name=f"{task.id}__clean", family="clean", err="none",
                      task=task.id, node_idx=-1, alt_gesture="",
                      anchor_k=0, anchor=[], plan=ir.flat(), repair=[],
                      certified={"fires": False, "outcome": "benign",
                                 "pace": "conversation"}))
    return ir, cases, ledger


def _flat_wrong_page(ir: PlanIR, node_idx, wrong_nav):
    """Prefix up to the reach node's approach, then the WRONG nav — and stop.
    The human has navigated to the wrong screen and is stuck there."""
    out = []
    for n in ir.nodes:
        if n.idx < node_idx:
            out += n.gestures
        elif n.idx == node_idx:
            out += n.approach + [wrong_nav]
            break
    return out


def _pretty_screen(key):
    k = key.split(".", 1)[-1] if "." in key else key
    return k.replace("stub_", "").replace("_", " ").strip()


def _slug(s):
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:28]


def slip_text(case, ir):
    """A private, in-character description of the slip an LLM human makes on
    an error case — phrased as the human's own (mistaken) intention, naming
    on-screen controls only (never the reference plan)."""
    if case.family != "error" or case.node_idx < 0:
        return ""
    right = (ir.nodes[case.node_idx].gesture or "").replace("tap ", "").strip()
    wrong = case.alt_gesture.replace("tap ", "").strip()
    if case.err in ("wrong_value", "wrong_target") and wrong:
        return (f"When you reach the screen with “{right}”, you're going to "
                f"tap “{wrong}” instead — you're convinced that's the right "
                f"choice.")
    if case.err == "wrong_page" and wrong:
        return (f"When you'd normally tap “{right}” to get to the next screen, "
                f"you're going to tap “{wrong}” instead — you're sure that's "
                f"the way to the setting you need, and you'll end up on the "
                f"wrong screen looking for it.")
    if case.err == "skip_step" and right:
        return (f"You're going to move on WITHOUT doing “{right}” — it slips "
                f"your mind and you think you're already done with this screen.")
    if case.err == "early_commit":
        return ("You're going to jump ahead and confirm before finishing the "
                "earlier steps — you think they're already set.")
    return ""
