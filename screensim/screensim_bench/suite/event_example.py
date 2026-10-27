#!/usr/bin/env python3
"""Assemble an `experiment/08_event_examples/` directory for one ScreenSim episode.

Mirrors the VirtualHome example in that folder: a curated record slice, a plain-text timeline, one
rendered frame per tick, a contact sheet and the film. Everything is regenerated from the saved
record by replaying it on a fresh phone — the engine is deterministic, so the frames are the screens
the person actually saw.

    python screensim_bench/suite/event_example.py tmp/refix/bench_v1/gpt-realtime-2.1.records.jsonl \
        --episode siri_lock_screen_privacy__c2 --persona classic_novice \
        --contestant gpt-realtime-2.1 --run 1 --out <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys


from PIL import Image

from screensim.interact.personas import PERSONAS
from screensim.tasks import by_id


def load(path, episode, persona, contestant):
    for line in open(path):
        r = json.loads(line)
        if (r.get("episode") == episode and r.get("persona") == persona
                and r.get("contestant") == contestant and "error" not in r):
            return r
    raise SystemExit("no such episode in " + path)


def timeline_lines(r):
    """One line per event, in the record's order, with the engine's own words."""
    out = []
    for e in r["timeline"]:
        t = f"t{e['t']:>3}"
        k = e["kind"]
        if k == "gesture":
            tail = ""
            if e.get("slip"):
                tail = "                <- forced mistake"
            if e.get("ok") is False:
                tail = " -> refused by the phone"
            out.append(f"{t} [person taps] {e.get('text')}{tail}")
        elif k == "takeover":
            out.append(f"{t} [engine] forces {e.get('text')}")
        elif k == "released":
            out.append(f"{t} [engine] releases the forced mistake: {e.get('text')}")
        elif k == "blocked":
            out.append(f"{t} [engine] blocks: {e.get('text')}")
        elif k == "say":
            who = "ASSISTANT" if e.get("who") == "assistant" else "PERSON"
            flag = f"   [flag {e.get('flag')}]" if e.get("flag") else ""
            out.append(f"{t} {who}: {e.get('text')}{flag}")
        elif k == "move":
            if e.get("text") != "answer":
                out.append(f"{t} [person move] {e.get('text')}")
        elif k == "call":
            out.append(f"{t} [person's private judgement] {e.get('text')} about "
                       f"{e.get('about')} (beat {e.get('beat')})")
    return out


def curated(r, source, run):
    """The record slice the example ships: everything needed to re-read the episode, nothing else."""
    p = PERSONAS[r["persona"]]
    t = by_id(r["task"])
    return {
        "source_record": source, "line_in_source": r.get("_line"), "engine_rev": r.get("engine_rev"),
        "benchmark_run": run, "assistant": r["contestant"], "persona": r["persona"],
        "persona_profile": {"skill": p.skill, "trust": p.trust, "q1": p.q1, "q2": p.q2, "q3": p.q3},
        "human_model": r.get("human_model"), "task": r["task"], "episode": r["episode"],
        "instruction": r.get("instruction"),
        "outcome": {"goal_met": bool(r.get("goal_ok")), "strict_success": bool(r.get("success")),
                    "extra_changes": r.get("extra_changes") or [], "ticks": r.get("final_tick")},
        "designed_errors": r.get("n_designed"), "errors_fired": r.get("n_fired"),
        "beats": r["beats"],
        "detection": {"recall": r.get("recall"), "precision": r.get("precision"), "f1": r.get("f1"),
                      "dt": r.get("dt"), "prevented": r.get("prevented"),
                      "false_flags": r.get("false_flags"), "typed_right": r.get("typed_right")},
        "efficiency": {"cc": r.get("cc"), "polls": r.get("n_polls"), "talk": r.get("n_talk")},
        "person_private_calls": r.get("person_calls"), "person_repairs": r.get("human_fixes"),
        "denials": r.get("denials"), "ambiguous_calls": r.get("ambiguous_calls"),
        "call_health": r.get("call_health"), "conversation": r.get("transcript"),
        "timeline": r["timeline"], "moves": r.get("moves"), "progress_log": r.get("progress_log"),
        "goal_predicate": getattr(t.goal, "__name__", None),
    }


def frames(r, outdir):
    """One PNG per tick, rendered by replaying the record's gestures on a fresh phone."""
    from screensim.render.capture import Capturer
    import screensim_bench.scoring.rubric as A
    os.makedirs(outdir, exist_ok=True)
    rep = A.Replay(r)
    shots = {}
    with Capturer(scale=1.0) as cap:
        for i in range(len(r["timeline"]) + 1):
            rep.advance_to(i)
            shots[rep.dev.st.tick] = cap.snap(rep.dev)   # the tick's last render is what it ends on
    made = []
    for tick, png in sorted(shots.items()):
        path = os.path.join(outdir, f"t{tick:03d}.jpg")
        Image.open(io.BytesIO(png)).convert("RGB").save(path, quality=88)
        made.append(path)
    return made


def contact_sheet(paths, out, cols=6, width=320):
    ims = [Image.open(p) for p in paths]
    w, h = width, int(width * ims[0].height / ims[0].width)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (12, 14, 18))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=88)
    return sheet.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--contestant", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = load(a.records, a.episode, a.persona, a.contestant)
    lo, hi = 0, r["final_tick"]
    os.makedirs(a.out, exist_ok=True)
    tag = f"t{lo}-{hi}"

    head = [f"# {a.contestant} · {a.episode} · {a.persona} · benchmark run {a.run}",
            f"# window t{lo}-{hi}; engine {r.get('engine_rev')}; "
            f"goal met: {bool(r.get('goal_ok'))}, nothing else changed: {not (r.get('extra_changes') or [])}; "
            f"{r.get('n_fired')}/{r.get('n_designed')} designed mistakes fired"]
    open(os.path.join(a.out, f"timeline_{tag}.txt"), "w").write("\n".join(head + timeline_lines(r)) + "\n")
    json.dump(curated(r, a.records, a.run), open(os.path.join(a.out, f"record_{tag}.json"), "w"), indent=1)
    made = frames(r, os.path.join(a.out, "frames"))
    size = contact_sheet(made, os.path.join(a.out, "contact_sheet.jpg"))
    print(f"{a.out}: {len(made)} frames, contact sheet {size}, timeline {len(timeline_lines(r))} lines")
    print("film with: python screensim_bench/video/film.py <json-array> --episode ... --out "
          + os.path.join(a.out, f"{tag}.mp4"))


if __name__ == "__main__":
    main()
