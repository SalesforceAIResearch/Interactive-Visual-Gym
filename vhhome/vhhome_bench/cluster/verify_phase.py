#!/usr/bin/env python3
"""Verifier: are a phase's results all present, and do they look right?

    python -m vhhome_bench.cluster.verify_phase runs/formal/r1 [--stamp 2026-10-05.f124] [--n 75]

Per seat. FAIL blocks the phase from being marked done; WARN is shown and does not block.
  present   FAIL  every one of the n cells has a parseable record; nothing sits in the seat's void/ folder
  stamp     FAIL  every record carries an accepted engine stamp (f124, f125 or f126 — the f126 re-run of the frozen-takeover cells; no f124 record ended by the old
                  dead-conversation guard)
  outcome   FAIL  every record has an outcome
  card      FAIL  open-weight seats only: the record's GPU is an H100 and its latency factor is 1.0
  speech    WARN  the assistant spoke in >= 50% of episodes (silent control and mmduet2 exempt); an API seat whose
            every call failed in transport FAILS
  events    WARN  >= 60% of the designed mistakes fired
  outcomes  WARN  timeouts <= 30%; the won rate is not 0 or 1 (silent control exempt)
  floor     WARN  silent control only: with an outcome-breaking mistake fired and nobody speaking, <= 10% of episodes
            may still be won (the no-self-repair property this engine exists for)
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

from vhhome_bench.cluster import plan as P  # noqa: E402

BREAK = {"wrong_object", "wrong_placement", "skipped_step", "missing_step"}


def records(root, tag, cells):
    out = []
    for c in cells:
        f = os.path.join(P.ENG, root, tag, f"{tag}__{c['case']}__{c['persona']}.json")
        out.append((c, f if (os.path.exists(f) and os.path.getsize(f) > 0) else None))
    return out


def check_seat(root, tag, cells, stamp, local_model):
    recs = records(root, tag, cells)
    have = [f for _, f in recs if f]
    r = dict(tag=tag, n=len(have), expected=len(cells), problems=[], warnings=[])
    if len(have) < len(cells):
        r["problems"].append(f"present: {len(have)}/{len(cells)} cells")
    void = glob.glob(os.path.join(P.ENG, root, tag, "void", "*.json"))
    if void:
        r["problems"].append(f"present: {len(void)} voided record(s) (every call a transport failure)")
    bad = stamp_bad = no_out = card_bad = spoke = planned = fired = dead = old_guard = 0
    outc, floor_n, floor_won, ticks, lines_p = {}, 0, 0, 0, 0
    for f in have:
        try:
            d = json.load(open(f))
        except Exception:
            bad += 1
            continue
        stamps = stamp if isinstance(stamp, (list, tuple)) else ([stamp] if stamp else [])
        if stamps and d.get("engine_rev") not in stamps:
            stamp_bad += 1
        # an f124 record that the OLD dead-conversation guard ended must have been re-run on f125
        old_guard += d.get("ended") == "dead_conversation" and d.get("engine_rev") == "2026-10-05.f124"
        o = d.get("outcome")
        no_out += o is None
        outc[o] = outc.get(o, 0) + 1
        ph = d.get("phases") or []
        planned += len(ph)
        fl = [p for p in ph if p.get("dev_live") is not None]
        fired += len(fl)
        conv = d.get("conversation") or []
        spoke += any(c.get("speaker") == "vlm" and (c.get("text") or "").strip() for c in conv)
        lines_p += sum(1 for c in conv if c.get("speaker") != "vlm" and (c.get("text") or "").strip())
        ticks += d.get("ticks") or 0
        s = d.get("assistant_latency_summary") or {}
        if local_model and not ("H100" in str(s.get("gpu") or d.get("gpu") or "") and float(s.get("lat_div", d.get("lat_div", 1)) or 1) == 1.0):
            card_bad += 1
        st = ((d.get("assistant") or {}).get("stats") or {})
        dead += bool(st.get("calls") and st.get("retries") and not st.get("prompt_tokens") and tag not in P.NEVER_SPEAKS and not local_model)
        if tag == "silent" and any(p["err"] in BREAK for p in fl) and not any(p.get("caught") or p.get("prevented") for p in fl):
            floor_n += 1
            floor_won += o == "won"
    n = max(len(have) - bad, 1)
    r.update(outcomes=outc, planned=planned, fired=fired, spoke=spoke, won=outc.get("won", 0),
             person_lines_per_100=round(100 * lines_p / max(ticks, 1), 1), mean_ticks=round(ticks / n))
    if bad:
        r["problems"].append(f"present: {bad} unreadable record(s)")
    if stamp_bad:
        r["problems"].append(f"stamp: {stamp_bad} record(s) not on {stamp}")
    if old_guard:
        r["problems"].append(f"stamp: {old_guard} f124 record(s) ended by the old dead-conversation guard, not yet re-run")
    if no_out:
        r["problems"].append(f"outcome: {no_out} record(s) without an outcome")
    if card_bad:
        r["problems"].append(f"card: {card_bad} record(s) not on an H100 with latency factor 1.0")
    if dead:
        r["problems"].append(f"speech: {dead} episode(s) whose every assistant call failed in transport")
    if len(have) >= 20:
        if tag not in P.NEVER_SPEAKS and spoke / n < 0.5:
            r["warnings"].append(f"speech: the assistant spoke in {spoke}/{n} episodes (< 50%)")
        if planned and fired / planned < 0.6:
            r["warnings"].append(f"events: {fired}/{planned} designed mistakes fired (< 60%)")
        if outc.get("timeout", 0) / n > 0.3:
            r["warnings"].append(f"outcomes: {outc.get('timeout', 0)}/{n} timeouts (> 30%)")
        if tag != "silent" and outc.get("won", 0) in (0, n):
            r["warnings"].append(f"outcomes: won {outc.get('won', 0)}/{n} is degenerate")
    if tag == "silent" and floor_n:
        r["floor"] = [floor_won, floor_n]
        if floor_won / floor_n > 0.10:
            r["warnings"].append(f"floor: {floor_won}/{floor_n} silent episodes with an outcome-breaking mistake were still won (> 10%)")
    r["status"] = "fail" if r["problems"] else ("warn" if r["warnings"] else "pass")
    return r


def verify(root, tags_api, tags_gpu, stamp, cells=None):
    cells = cells or json.load(open(os.path.join(P.ENG, P.CELLS)))["cells"]
    seats = [check_seat(root, t, cells, stamp, False) for t in tags_api] + \
            [check_seat(root, t, cells, stamp, True) for t in tags_gpu]
    st = "fail" if any(s["status"] == "fail" for s in seats) else ("warn" if any(s["status"] == "warn" for s in seats) else "pass")
    return dict(root=root, status=st, seats=seats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stamp", default=None)
    a = ap.parse_args()
    stamp = a.stamp
    sf = os.path.join(P.STATE_DIR, "state.json")
    if not stamp and os.path.exists(sf):
        stamp = json.load(open(sf)).get("stamp")
    v = verify(a.root, P.API_SEATS, P.gpu_seats(), stamp)
    print(f"{a.root}: {v['status'].upper()}  (stamp {stamp})")
    for s in v["seats"]:
        print(f"  {s['tag']:24} {s['n']:3}/{s['expected']} {s['status']:5} won {s['won']:3} {s['outcomes']} fired {s['fired']}/{s['planned']} "
              f"spoke {s['spoke']}" + ("  | " + "; ".join(s["problems"] + s["warnings"]) if s["problems"] or s["warnings"] else ""))
