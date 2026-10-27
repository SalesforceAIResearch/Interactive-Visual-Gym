#!/usr/bin/env python3
"""Film a task's REFERENCE completion: the engine executes the clean plan itself.

The rollout's clean run is driven by the LLM person, who navigates by her own eyes: a cross-room
walk to a small object she cannot see is refused ("no visible juice — turn to scan") and she
spends the episode scanning. That is her behaviour, not the task's. A reference film wants the
plan executed exactly, so this drives LiveWorld.exec_step straight from the episode's steps and
scores the same checkpoints the rollout would.

  python -m vhhome_bench.video.film_reference PORT EPISODE OUT.json      (writes OUT.json and OUT.mp4)
"""
import json, os, sys, time
os.environ.setdefault("VH_MIDLEVEL", "1")

from vhhome import world as W
from vhhome import tasks as E
from vhhome import goals as G

def main():
    port, epk, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    # --case NAME: film a COMPOSITE case's deviated plan instead of the clean plan (D11 2026-09-09):
    # the same engine-executed replay, so a designed error is proven to manifest on camera.
    case_name = None
    if "--case" in sys.argv:
        case_name = sys.argv[sys.argv.index("--case") + 1]
        from vhhome_bench import cases as CS
        from vhhome_bench import cases_spec as SPEC
        from vhhome_bench.episode import flat_of
        _epk, _calls, _story = next((e, c, st) for n, e, c, st in SPEC.COMPOSITE if n == case_name)
        assert _epk == epk, f"{case_name} belongs to {_epk}, not {epk}"
        plan_steps = [st for st, _sid in flat_of(CS.build_deviated(epk, [(t, dict(k)) for t, k in _calls])[1])]
        print(f"[ref] case {case_name}: {len(plan_steps)} deviated commands — {_story}", flush=True)
    else:
        plan_steps = None
    world = W.LiveWorld(port, epk, f"R{port}{(case_name or epk)[:5]}")
    t0 = time.time()
    # The tag names this world's render folders under unity_vol/Output. Two films of the SAME
    # task running at once used to share it and ingested each other's frames (a bedroom-desk
    # frame inside a kitchen walk, 2026-09-09) -- the port makes it unique per engine.
    log = []
    # establishing shot: everything the setup displaced, seen where it now lies, before any step
    displaced = []
    for st in (E.EPISODES[epk].get("setup") or []):
        if st[0] in ("Put", "PutIn") and len(st) > 2:
            try:
                oid = world._pick(st[1], st[2])[0]
                if oid is not None and oid not in displaced:
                    displaced.append(oid)
            except Exception:
                pass
    if displaced:
        world.survey(displaced)
        print(f"[ref] survey of {len(displaced)} displaced object(s)", flush=True)
    for st in (plan_steps if plan_steps is not None else E.EPISODES[epk]["steps"]):
        txt = W.pretty_cmd(st)
        try:
            r = world.exec_step(tuple(st), "plan", txt)
            status = (r or {}).get("status") if isinstance(r, dict) else str(r)[:40]
            why = (r or {}).get("why") if isinstance(r, dict) else None
            if why: status = f"{status} — {str(why)[:90]}"
        except Exception as e:
            status = f"EXC {type(e).__name__}: {str(e)[:80]}"
        log.append({"cmd": txt, "status": status, "t": world.tick})
        print(f"[ref] t{world.tick:3} {txt:44} {status}", flush=True)
    ev = list(getattr(world.sym, "events", []))
    cp_fn = G.EPISODES[epk]["checkpoints"]
    try:
        cps = cp_fn(ev)
    except TypeError:
        cps = cp_fn(ev, getattr(world, "targets", {}) or {})
    cps = {k: bool(v) for k, v in cps.items()}
    won = all(cps.values())
    world.write_video(out[:-5] + ".mp4")
    json.dump({"episode": epk, "scene": E.EPISODES[epk]["scene"], "mode": "reference_scripted",
               "outcome": "won" if won else "silent_fail", "ticks": world.tick,
               "checkpoints": cps, "action_log": log, "wall_s": round(time.time() - t0, 1),
               "engine_rev": getattr(W, "ENGINE_REV", "?")}, open(out, "w"), indent=1)
    print(f"[ref] {epk}: {'won' if won else 'FAIL'} {sum(cps.values())}/{len(cps)} in {world.tick} ticks", flush=True)

if __name__ == "__main__":
    main()
