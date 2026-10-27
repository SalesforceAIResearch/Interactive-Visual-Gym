#!/usr/bin/env python3
"""Offline verifier for the COMPOSITE (multi-error) episode spec.

Dry-replays every composite episode through the symbolic twin and checks the five conflict rules
declared in vhhome_bench/cases_spec.py. No Unity, no model, runs in seconds. Writes
benchmark/composite_ledger.json — one row per episode with a verdict and, on rejection, the reason.

  python -m verify_composite_vh [-v]
"""
import json, os, sys, traceback

from vhhome_bench import cases as CS
from vhhome import plan as pir
from vhhome_bench import protocol as P
from vhhome_bench import cases_spec as SPEC
from vhhome import tasks as episodes
from vhhome.world import SymState, base, pretty_cmd
SymState.LENIENT_NEAREST = True     # offline: nearest tokens name some instance in the right state
from vhhome_bench.episode import flat_of

VERBOSE = "-v" in sys.argv
ROWS, FAILS = [], []

# the class each transform is graded as (transform name == graded label for the seven classes)
GRADED = {"wrong_object", "wrong_placement", "navigation_inefficiency",
          "skipped_step", "missing_step", "repetition", "stall"}
# which transforms are structurally destructive, and of what
TRUNCATING = {"wrong_placement"}          # deletes everything after its own node
OMITTING = {"skipped_step", "missing_step"}   # deletes its own anchor node
INSERTING = {"navigation_inefficiency", "stall", "repetition"}   # adds a node, consumes none


def anchor_of(transform, kw):
    """The subgoal id this transform is anchored on, whichever key it uses."""
    for k in ("sid", "before", "after", "at", "anchor"):
        if k in kw:
            return kw[k]
    return None


# stall and navigation_inefficiency are errors of TIME and ROUTE: their `after`/`before` names
# the neighbour they sit beside, not an object they act on, so they cannot collide with another
# error over an object. Only manipulating transforms have a target.
OBJECT_ACTING = {"wrong_object", "wrong_placement", "skipped_step", "missing_step", "repetition"}


def target_of(transform, kw):
    """The object the error acts on, or None for a time/route error."""
    if transform not in OBJECT_ACTING:
        return None
    sid = anchor_of(transform, kw) or ""
    inner = sid[sid.index("(") + 1:sid.rindex(")")] if "(" in sid and ")" in sid else sid
    return base(inner) if inner else None


def check(cond, name, msg):
    if not cond:
        FAILS.append((name, msg))
    return cond


def verify(name, epk, calls, story):
    row = {"name": name, "episode": epk, "errors": [c[0] for c in calls],
           "anchors": [anchor_of(*c) for c in calls], "story": story}
    # ---------- build ----------
    try:
        base_nodes, ir, compiled, records = CS.build_deviated(epk, [(n, dict(k)) for n, k in calls])
    except Exception as e:
        row.update(admitted=False, reason=f"build raised {type(e).__name__}: {e}")
        ROWS.append(row); check(False, name, row["reason"]); return

    clean_sids = [n.sid for n in base_nodes]
    dev_sids = [n.sid for n in ir]

    # ---------- R2a every declared error actually fired (nothing skipped-with-record) ----------
    skipped = [r for r in records if getattr(r, "skipped", False)]
    if skipped:
        why = "; ".join(f"{r.transform}@{r.anchor}: {r.detail.get('reason', 'anchor removed')}"
                        for r in skipped)
        row.update(admitted=False, reason=f"R2 anchor survival — a transform was skipped: {why}")
        ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- R1 direct-twin exclusion ----------
    labels = [c[0] for c in calls if c[0] in GRADED]
    check(len(labels) == len(calls), name, f"{name}: a transform is not a graded class")
    bad = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]
           if b in P.TWINS.get(a, ()) or a in P.TWINS.get(b, ())]
    if bad:
        row.update(admitted=False,
                   reason=f"R1 direct-twin exclusion — {bad[0][0]} and {bad[0][1]} are twins, "
                          "a correct flag for one is creditable for the other")
        ROWS.append(row); check(False, name, row["reason"]); return
    # shared distant aliases are allowed but recorded, because attribution then relies on R3
    shared = sorted({x for i, a in enumerate(labels) for b in labels[i + 1:]
                     for x in (set(P.TWINS.get(a, ())) & set(P.TWINS.get(b, ())))})
    row["shared_aliases"] = shared

    # ---------- R4 independent observability: distinct target objects ----------
    tgts = [target_of(*c) for c in calls]
    dup = [t for t in set(tgts) if t and tgts.count(t) > 1]
    if dup:
        row.update(admitted=False,
                   reason=f"R4 independent observability — two errors act on the same object "
                          f"({dup[0]}); one can mask the other")
        ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- R2b/R3 anchors present in the composed plan, distinct and separated ----------
    # Position of each error in the composed plan, found by INCREMENTAL DIFF: build with the
    # first i calls and with the first i+1, and the sids that appear are exactly what transform i
    # contributed. This is exact where guessing from a record's fields is not (a replacing
    # transform names the node it REPLACED, an inserting one the node it ADDED), and it doubles as
    # the anchor-survival test — a contribution missing from the FINAL plan was deleted by a
    # later transform.
    idx, contributed = {}, {}
    prev_sids = clean_sids
    for i, (tname, kw) in enumerate(calls):
        step_ir = CS.build_deviated(epk, [(n, dict(k)) for n, k in calls[:i + 1]])[1]
        step_sids = [n.sid for n in step_ir]
        added = [x for x in step_sids if x not in prev_sids]
        removed = [x for x in prev_sids if x not in step_sids]
        contributed[tname] = {"added": added, "removed": removed}
        prev_sids = step_sids
        if tname in OMITTING:
            a = anchor_of(tname, kw)
            if a not in clean_sids:
                row.update(admitted=False, reason=f"R2 — omission anchor {a} not in the clean plan")
                ROWS.append(row); check(False, name, row["reason"]); return
            # the anchor node is deliberately gone; the OBSERVABLE deviation is that she does the
            # SUCCESSOR subgoal instead, so that is where the window anchors
            succ = next((s_ for s_ in clean_sids[clean_sids.index(a) + 1:] if s_ in dev_sids), None)
            if succ is None:
                row.update(admitted=False,
                           reason=f"R2 — trailing omission of {a}: no successor subgoal survives "
                                  "to anchor a detection window on")
                ROWS.append(row); check(False, name, row["reason"]); return
            idx[tname] = dev_sids.index(succ)
        else:
            alive = [x for x in added if x in dev_sids]
            if not alive:
                gone = added[0] if added else anchor_of(tname, kw)
                row.update(admitted=False,
                           reason=f"R2 anchor survival — the node {tname} contributed ({gone}) is "
                                  "not in the final plan: a later transform deleted it")
                ROWS.append(row); check(False, name, row["reason"]); return
            idx[tname] = min(dev_sids.index(x) for x in alive)
    row["contributed"] = contributed
    order = sorted(idx.items(), key=lambda kv: kv[1])
    row["plan_order"] = [k for k, _ in order]
    gaps = [order[i + 1][1] - order[i][1] for i in range(len(order) - 1)]
    row["anchor_gaps"] = gaps
    # R3 as WRITTEN requires "at least one surviving subgoal" between the errors, i.e. a gap of
    # 2 or more. Testing `g < 1` only rejected two errors at the SAME position and admitted every
    # adjacent pair, which is what R3 exists to forbid — the windows still touch (independent
    # audit, 2026-08-28).
    if any(g < 2 for g in gaps):
        row.update(admitted=False,
                   reason=f"R3 window separation — the errors land in adjacent plan positions "
                          f"(gaps {gaps}); with no surviving subgoal between them their detection "
                          "windows touch and a flag cannot be attributed")
        ROWS.append(row); check(False, name, row["reason"]); return
    # ...and the DECLARED anchors must differ too. Comparing only the contributed node positions
    # admitted an episode that anchored both errors on the very same subgoal.
    decl = [anchor_of(*c) for c in calls]
    if len(set(decl)) != len(decl):
        row.update(admitted=False,
                   reason=f"R3 window separation — two errors declare the same anchor {decl}; "
                          "they describe one moment, not two")
        ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- R2c a truncating transform must be last in plan order ----------
    trunc = [t for t, _ in calls if t in TRUNCATING]
    if trunc and row["plan_order"][-1] not in trunc:
        row.update(admitted=False,
                   reason=f"R2 — {trunc[0]} truncates the plan but is not the last error in plan "
                          f"order (last is {row['plan_order'][-1]}), so it deletes that anchor")
        ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- R8 PER-PHASE DISTINCTIVENESS ----------
    # Checking the whole composed plan against clean is not enough: one error can supply the
    # distinctive commands while ANOTHER adds nothing the clean plan lacks, and that second error
    # is then indistinguishable from ordinary execution. A deposit repetition that puts the
    # object back on the SAME surface is the case that bit us — it reads as her adjusting the
    # plate, and the live rollout refused to start the episode at all.
    INACTION = {"stall", "skipped_step", "missing_step"}
    _clean_txt = {pretty_cmd(x) for x, _ in flat_of(base_nodes)}
    _prev = flat_of(base_nodes)
    for i, (tname, kw) in enumerate(calls):
        _cur = flat_of(CS.build_deviated(epk, [(t2, dict(k2)) for t2, k2 in calls[:i + 1]])[1])
        m_ = 0
        while m_ < min(len(_prev), len(_cur)) and _prev[m_][0] == _cur[m_][0]:
            m_ += 1
        sfx_ = 0
        while sfx_ < min(len(_prev), len(_cur)) - m_ and _prev[-1 - sfx_][0] == _cur[-1 - sfx_][0]:
            sfx_ += 1
        sec = [x for x, _ in (_cur[m_:len(_cur) - sfx_] or _cur[m_:m_ + 4])]
        only = {pretty_cmd(x) for x in sec if x[0] != "Idle"} - _clean_txt
        _prev = _cur
        if not only and tname not in INACTION:
            row.update(admitted=False,
                       reason=f"R8 per-phase distinctiveness — phase {i} ({tname}) adds no command "
                              "the clean plan lacks, so it is indistinguishable from ordinary "
                              "execution")
            ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- executability: the composed plan runs in the twin ----------
    sym = SymState(epk)
    refused = []
    for st, sid, dev in compiled:
        ok, why = sym.check(tuple(st))
        if not ok:
            refused.append((pretty_cmd(st), sid, why))
            continue          # a refusal can be the modelled consequence; record and go on
        sym.run(tuple(st))
    row["refusals"] = [{"cmd": c, "sid": s, "why": w} for c, s, w in refused]
    # a refusal on a command belonging to a DEVIATED node means the error itself cannot execute
    dev_nodes = {n.sid for n in ir if n.deviated or n.origin != "plan"}
    fatal = [r for r in refused if r[1] in dev_nodes]
    if fatal:
        row.update(admitted=False,
                   reason=f"not executable — the injected command {fatal[0][0]!r} is refused "
                          f"({fatal[0][2]}), so that error can never manifest")
        ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- the composite must actually break the task ----------
    clean_sym = SymState(epk)
    for st, sid, dev in CS.build_deviated(epk, [])[2]:
        ok, _ = clean_sym.check(tuple(st))
        if ok:
            clean_sym.run(tuple(st))
    broke = CS.final_ok(epk, clean_sym) and not CS.final_ok(epk, sym)
    row["clean_final_ok"] = CS.final_ok(epk, clean_sym)
    row["dev_final_ok"] = CS.final_ok(epk, sym)
    # waste-only classes legitimately still complete; only require breakage when an
    # outcome-breaking class is present
    OUTCOME_BREAKING = {"wrong_object", "wrong_placement", "skipped_step", "repetition",
                        "missing_step"}
    if set(labels) & OUTCOME_BREAKING and not broke:
        row.update(admitted=False,
                   reason="the composite does not change the outcome — an outcome-breaking class "
                          "is present but the task still completes, so the episode is unfalsifiable")
        ROWS.append(row); check(False, name, row["reason"]); return

    # ---------- recorded caveats (do not reject, but must never be silent) ----------
    caveats = []
    # C-a  an omission whose target has DEPENDENTS makes those dependents refuse. In a
    #      single-error case that cascade IS the designed evidence; in a composite it adds
    #      refusals a watcher may read as further errors, so it is recorded.
    deps = pir.deps(base_nodes)
    for tname, kw in calls:
        if tname in OMITTING:
            a = anchor_of(tname, kw)
            dependents = [b for x, b in deps if x == a]
            if dependents:
                caveats.append(f"{tname}@{a} has dependents {dependents} that will refuse; the "
                               "cascade may read as an extra unlabelled error")
    # C-b  wrong_object replaces the WHOLE fetch node. If the original fetch opened a container
    #      and the look-alike does not live in it, the container interaction disappears and a
    #      watcher may read the missing open/close as a skipped step (VLM audit, 2026-08-28).
    for tname, kw in calls:
        if tname == "wrong_object":
            was = next((n for n in base_nodes if n.sid == kw.get("sid")), None)
            now = next((n for n in ir if n.sid == f"wrong_grab({kw.get('wrong')})"), None)
            had = bool(was and any(c[0] == "Open" for c in was.commands))
            keeps = bool(now and any(c[0] == "Open" for c in now.commands))
            if had and not keeps:
                caveats.append(f"wrong_object@{kw.get('sid')} replaced a CONTAINER fetch and the "
                               f"open/close disappeared, which a watcher may read as a skipped "
                               f"step on top of the wrong object")
    # R6  VISIBLE STATE. A repetition is only scorable if its inverting second press CHANGES THE
    #     PICTURE. Engine note A7: a microwave/dishwasher/stove switch "renders identically to not
    #     doing it" — the state flips with no visible consequence and the door is shut, so an
    #     assistant has nothing to see. Lamps and the TV change the room's light; container doors
    #     move geometry. (independent audit, 2026-08-28)
    INVISIBLE_TOGGLE = {"microwave", "dishwasher", "stove"}
    for tname, kw in calls:
        if tname == "repetition":
            tgt = target_of(tname, kw)
            node = next((n for n in base_nodes if n.sid == kw.get("sid")), None)
            is_switch = bool(node and any(c[0] in ("SwitchOn", "SwitchOff") for c in node.commands))
            if is_switch and tgt in INVISIBLE_TOGGLE:
                row.update(admitted=False,
                           reason=f"R6 visible state — a repetition on the {tgt} flips a state that "
                                  "renders identically (engine note A7): the error would have no "
                                  "evidence on camera")
                ROWS.append(row); check(False, name, row["reason"]); return

    # R7  A WRONG PLACEMENT MUST MOVE SOMETHING. Parking the object on the same KIND of surface it
    #     was taken from is a lift-and-replace with no state change — which reads as repetition or
    #     a pointless detour, not as a misplacement. (independent audit, 2026-08-28)
    for tname, kw in calls:
        if tname == "wrong_placement":
            obj = target_of(tname, kw)
            origin = None
            for st, _sid, _d in CS.build_deviated(epk, [])[2]:
                if st[0] in ("Put", "PutIn") and len(st) > 3 and base(st[1]) == obj:
                    continue
            # where the object STARTS: the surface named by the episode's own setup, else the
            # first surface it is taken from
            start = (episodes.EPISODES.get(epk, {}) or {}).get("setup") or []
            for st in start:
                if st[0] in ("Put", "PutIn") and len(st) > 3 and base(st[1]) == obj:
                    origin = base(st[3])
            if origin and base(kw.get("surf", "")) == origin:
                row.update(admitted=False,
                           reason=f"R7 the misplacement must move something — {obj} is put on a "
                                  f"{kw.get('surf')}, the same kind of surface it started on")
                ROWS.append(row); check(False, name, row["reason"]); return

    row["caveats"] = caveats
    row.update(admitted=True, reason="admitted" + (f" (with {len(caveats)} caveat(s))" if caveats else ""))
    ROWS.append(row)


def main():
    print(f"\nCOMPOSITE SPEC — {len(SPEC.COMPOSITE)} multi-error episodes\n")
    for name, epk, calls, story in SPEC.COMPOSITE:
        try:
            verify(name, epk, calls, story)
        except Exception:
            FAILS.append((name, "VERIFIER CRASHED\n" + traceback.format_exc()))
            ROWS.append({"name": name, "episode": epk, "admitted": False,
                         "reason": "verifier crashed"})
    for r in ROWS:
        mark = "OK  " if r.get("admitted") else "REJ "
        print(f"  {mark}{r['name']:28} {r['episode']:15} {'+'.join(r.get('errors', []))}")
        if not r.get("admitted"):
            print(f"        -> {r['reason']}")
        elif VERBOSE:
            print(f"        order={r.get('plan_order')} gaps={r.get('anchor_gaps')} "
                  f"shared_aliases={r.get('shared_aliases')} refusals={len(r.get('refusals', []))}")
    ok = sum(1 for r in ROWS if r.get("admitted"))
    out = "benchmark/composite_ledger.json"
    json.dump({"generated_by": "tests/verify_cases.py",
               "rules": ["R1 direct-twin exclusion", "R2 anchor survival",
                         "R3 window separation", "R4 independent observability",
                         "R5 semantic plausibility (audited separately)"],
               "entries": ROWS}, open(out, "w"), indent=1)
    print(f"\n=== {ok}/{len(ROWS)} admitted -> {out} ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
