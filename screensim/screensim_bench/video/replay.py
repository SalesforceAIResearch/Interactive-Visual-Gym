"""Film an ALREADY-RUN free-mode episode exactly as it happened, from its record.

The engine is deterministic and a v2 record's timeline logs, in order, every gesture the
engine executed (the person's taps and the engine-forced slips), every blocked tap (which
costs one tick), every spoken line, every move and plan, and every takeover/release.
Replaying that log on a fresh phone reproduces every screen, so the film is the audited
episode itself, with no new model calls. `replay()` checks each gesture's outcome and the
final goal against the record, so a film is only published when the replay is exact.

Frame: phone (left) · per-error status + the running event log (right) · a caption bar
under the phone naming the event on screen. Also returns a cue list (video second → event)
so a web page can sync a clickable event log to the video.

  python screensim_bench/video/replay.py tmp/pilot/sample_r6.json --contestant gpt-5.5 \
      --episode lock_screen_lockdown__c2 --persona baseline --out ep.mp4
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


from PIL import Image, ImageDraw

from screensim.core.engine import Device
from screensim.tasks import by_id, resolve_dynamic
import screensim_bench.video.episode_film as fe

FPS = 4
PHONE_W, PHONE_H = 412, 892
PANEL_W = 660
PAD, GAP, CAP_H = 24, 24, 58
W = PAD + PHONE_W + GAP + PANEL_W + PAD
H = PAD + PHONE_H + 10 + CAP_H + PAD
BG, CARD, CARD2 = (13, 16, 23), (22, 27, 36), (27, 34, 46)
INK, MUT, FAINT, LINE = (232, 236, 245), (152, 162, 182), (107, 118, 136), (37, 44, 56)
ACC, GOOD, WARN, BAD = (77, 163, 255), (78, 201, 138), (224, 168, 63), (229, 118, 118)
KIND = {"wrong_page": "wrong screen", "wrong_value": "wrong value",
        "wrong_target": "wrong control", "extra_write": "unwanted extra change"}
F_H, F_B, F_S, F_T, F_TAG = fe._font(22, True), fe._font(17, True), fe._font(15), fe._font(15), fe._font(12, True)
F_CAP = fe._font(19, True)


# --------------------------------------------------------------------------- replay
def _events(record):
    """Timeline → displayable events, in order. Each: kind, who, text, tick, [beat]."""
    out, last_plan = [], None
    for idx, e in enumerate(record["timeline"]):
        k, t = e["kind"], e["t"]
        if k == "gesture":
            if e.get("slip"):
                out.append({"k": "slip", "t": t, "text": e["text"], "beat": e.get("beat"), "i": idx, "ok": e.get("ok")})
            else:
                out.append({"k": "tap", "t": t, "text": e["text"], "i": idx, "ok": e.get("ok")})
        elif k == "blocked":
            out.append({"k": "blocked", "t": t, "text": e["text"], "beat": e.get("beat"), "i": idx})
        elif k == "say":
            txt = re.sub(r"\*\*|__|`", "", e["text"]).strip()
            out.append({"k": "say_" + e["who"], "t": t, "text": txt, "i": idx,
                        "flag": e.get("flag"), "credited": e.get("credited")})
        elif k == "flag":
            out.append({"k": "flag", "t": t, "text": "", "i": idx,
                        "flag": e.get("flag"), "credited": e.get("credited")})
        elif k == "takeover":
            out.append({"k": "takeover", "t": t, "text": e["text"], "beat": e.get("beat"), "i": idx})
        elif k == "released":
            out.append({"k": "released", "t": t, "text": e["text"], "beat": e.get("beat"), "i": idx})
        elif k == "call":
            out.append({"k": "call", "t": t, "text": e.get("text", ""), "about": e.get("about", ""),
                        "beat": e.get("beat"), "i": idx})
        elif k == "move":
            mv = e.get("text") or ""
            pu = e.get("plan_update") or {}
            acts = tuple(pu.get("actions") or [])
            if mv and mv != "answer":
                out.append({"k": "move", "t": t, "text": mv, "i": idx})
            if pu.get("op") == "replace" and acts and acts != last_plan:
                out.append({"k": "plan", "t": t, "text": " → ".join(acts), "i": idx})
                last_plan = acts
            elif pu.get("op") == "drop":
                last_plan = None
        elif k == "unfirable":
            out.append({"k": "unfirable", "t": t, "text": e.get("text", ""), "beat": e.get("beat"), "i": idx})
    return out


def replay(record, cap=None):
    """Rebuild the episode on a fresh device. Returns (shots, fidelity): shots is
    [(png|None, event|None, tick)] — one shot per displayable event plus start and end."""
    task = by_id(record["task"])
    dev = Device(task.seed)
    if task.start_app != "home":
        dev.open_app(task.start_app)
    evs = _events(record)
    by_idx = {}
    for ev in evs:
        by_idx.setdefault(ev["i"], []).append(ev)
    shots, mism = [], []

    def shot(ev):
        shots.append((cap.snap(dev) if cap is not None else None, ev, dev.st.tick))

    shot(None)
    for idx, e in enumerate(record["timeline"]):
        if e["kind"] == "gesture":
            r = dev.act(resolve_dynamic(dev, e["text"]))
            if bool(r.ok) != bool(e.get("ok")) or dev.st.tick != e["t"]:
                mism.append({"t": e["t"], "text": e["text"], "logged_ok": e.get("ok"),
                             "replay_ok": bool(r.ok), "replay_tick": dev.st.tick})
        elif e["kind"] == "blocked":
            dev._tick(1)
        for ev in by_idx.get(idx, []):
            shot(ev)
    shot({"k": "end", "t": dev.st.tick, "text": "episode ends", "i": len(record["timeline"])})
    goal_ok = bool(task.goal(dev.st)[0])
    fidelity = {"mismatches": mism, "final_tick_logged": record.get("final_tick"),
                "final_tick_replay": dev.st.tick, "goal_logged": record.get("goal_ok"),
                "goal_replay": goal_ok,
                "exact": (not mism and goal_ok == bool(record.get("goal_ok"))
                          and dev.st.tick == record.get("final_tick"))}
    return shots, fidelity


# --------------------------------------------------------------------------- drawing
def _wrap(d, text, font, w):
    return fe._wrap(d, text, font, w)


def _clip(t, n=9):
    w = t.split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def _label(ev, record):
    """(tag, tag colour, body text, caption) for one event."""
    k, t = ev["k"], ev["text"]
    bd = record.get("beats_design") or []
    if k == "tap":
        return "TAP", MUT, t + ("" if ev.get("ok") else "  (no effect)"), f"person taps: {t}"
    if k == "slip":
        b = bd[ev["beat"]] if ev.get("beat") is not None and ev["beat"] < len(bd) else {}
        return ("SLIP", BAD, f"{t}   (meant: {b.get('correct', '?')})",
                f"engine forces the mistake: {t}")
    if k == "takeover":
        b = bd[ev["beat"]] if ev.get("beat") is not None and ev["beat"] < len(bd) else {}
        return ("ENGINE", BAD, f"error {ev['beat'] + 1} fires: {KIND.get(b.get('kind'), b.get('kind'))}",
                f"error {ev['beat'] + 1} fires")
    if k == "blocked":
        return ("BLOCKED", WARN, f"{t}   (phone did not respond; error still held)",
                f"blocked while the error is held: {t}")
    if k == "released":
        return ("RELEASED", GOOD, f"error {ev['beat'] + 1} released: hold lifted, undoing it is up to the person",
                "person accepted the warning, hold released")
    if k in ("say_assistant", "flag") and (ev.get("flag") or ev.get("credited") is not None):
        fl = KIND.get(ev.get("flag"), ev.get("flag")) if ev.get("flag") else "untyped warning"
        verdict = (f"credited to error {ev['credited'] + 1}" if ev.get("credited") is not None
                   else "false alarm")
        body = f"[flags {fl}: {verdict}] " + t
        return ("FLAG", ACC if ev.get("credited") is not None else WARN, body.strip(),
                f"assistant flags {fl}: {verdict}")
    if k == "say_assistant":
        return "ASSISTANT", ACC, t, "assistant: " + _clip(t)
    if k == "say_human":
        return "PERSON", INK, t, "person: " + _clip(t)
    if k == "call":
        verdict = {"correct": "that IS the mistake I am making", "wrong": "that is NOT what I did"}.get(t, t)
        about = f" — {ev['about']}" if ev.get("about") else ""
        return ("JUDGES", GOOD if t == "correct" else WARN, verdict + about,
                f"person judges the warning: {t}")
    if k == "move":
        nice = {"accept": "accepts the suggestion", "reject": "rejects the suggestion",
                "ask_for_more_guidance": "asks for more guidance",
                "question_correctness": "questions the assistant",
                "ask_for_less_interruption": "asks for fewer interruptions"}.get(t, t)
        col = GOOD if t == "accept" else WARN if t in ("reject", "question_correctness") else MUT
        return "MOVE", col, nice, f"person {nice}"
    if k == "plan":
        return "PLAN", FAINT, t, "person plans next taps"
    if k == "unfirable":
        return "ENGINE", FAINT, f"error {ev['beat'] + 1} never fired (its step was not reached)", "error could not fire"
    if k == "end":
        return "END", MUT, "episode ends", "episode ends"
    return k.upper(), MUT, t, t


def _beat_states(record, upto_i, tick):
    """Per designed error, the state as of timeline index upto_i."""
    bd = record.get("beats_design") or []
    beats = record.get("beats") or []
    tl = record["timeline"][:upto_i + 1]
    states = []
    for bi, b in enumerate(bd):
        fired = any(e["kind"] == "takeover" and e.get("beat") == bi for e in tl)
        rel = any(e["kind"] == "released" and e.get("beat") == bi for e in tl)
        unf = any(e["kind"] == "unfirable" and e.get("beat") == bi for e in tl)
        g = beats[bi] if bi < len(beats) else {}
        det = g.get("detected_tick")
        logged = [e for e in record["timeline"] if "credited" in e]
        if logged:
            flagged = fired and any(e.get("credited") == bi for e in tl)
        else:
            flagged = fired and det is not None and (tick > det or (tick == det and rel))
        if unf:
            s = ("never fired", FAINT)
        elif not fired:
            s = ("not yet", FAINT)
        elif rel:
            s = ("released", GOOD)
        elif flagged:
            s = ("flagged, held", ACC)
        else:
            s = ("held, unflagged", BAD)
        states.append((bi, b, s))
    return states


def _panel(record, log, ev, tick, final=False):
    img = Image.new("RGB", (PANEL_W, PHONE_H + 10 + CAP_H), CARD)
    d = ImageDraw.Draw(img)
    x, y, w = 22, 18, PANEL_W - 44
    d.text((x, y), f"assistant: {record['contestant']}", font=F_H, fill=INK)
    y += 32
    d.text((x, y), f"person: {record.get('human_model', '?')} · persona {record['persona']} · {record['task']}",
           font=F_S, fill=MUT)
    y += 24
    for ln in _wrap(d, "“" + record.get("instruction", "") + "”", F_T, w)[:4]:
        d.text((x, y), ln, font=F_T, fill=FAINT)
        y += 19
    y += 8
    d.line((x, y, PANEL_W - 22, y), fill=LINE, width=1)
    y += 12
    upto = ev["i"] if ev else -1
    for bi, b, (s, sc) in _beat_states(record, upto, tick):
        if final:
            g = (record.get("beats") or [{}] * 9)[bi] if bi < len(record.get("beats") or []) else {}
            if g.get("skipped") or s == "never fired":
                s, sc = "never fired", FAINT
            elif g.get("committed"):
                s, sc = "committed (not fixed)", BAD
            elif g.get("prevented"):
                s, sc = "fixed", GOOD
        d.text((x, y), f"error {bi + 1} · {KIND.get(b['kind'], b['kind'])}", font=F_B, fill=INK)
        tw = d.textlength(s, font=F_TAG) + 18
        fe._round(d, (PANEL_W - 22 - tw, y, PANEL_W - 22, y + 21), 6, CARD2)
        d.text((PANEL_W - 22 - tw + 9, y + 4), s, font=F_TAG, fill=sc)
        y += 23
        d.text((x + 2, y), f"slip: {b['alt']}   instead of: {b['correct']}", font=F_T, fill=MUT)
        y += 24
    d.text((x, y), f"tick {tick}", font=F_TAG, fill=FAINT)
    y += 20
    d.line((x, y, PANEL_W - 22, y), fill=LINE, width=1)
    y += 10
    # running log, newest at the bottom; fit as many recent events as the space allows
    rows = []
    for e in log:
        tag, col, body, _ = _label(e, record)
        font = F_T
        lines = _wrap(d, body, font, w - 96)
        rows.append((tag, col, lines, e is ev))
    bottom = PHONE_H + 10 + CAP_H - 14
    need, keep = 0, []
    for r in reversed(rows):
        hgt = 19 * len(r[2]) + 7
        if y + need + hgt > bottom:
            break
        need += hgt
        keep.append(r)
    for tag, col, lines, cur in reversed(keep):
        if cur:
            fe._round(d, (x - 8, y - 3, PANEL_W - 14, y + 19 * len(lines) + 1), 6, CARD2)
        d.text((x, y + 2), tag, font=F_TAG, fill=col)
        for ln in lines:
            d.text((x + 96, y), ln, font=F_T,
                   fill=INK if tag in ("PERSON", "ASSISTANT", "FLAG") else (col if tag in ("SLIP", "BLOCKED") else MUT))
            y += 19
        y += 7
    return img


def _frame(png, panel, caption, capcol):
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    if png is not None:
        ph = Image.open(io.BytesIO(png)).convert("RGB")
        if ph.size != (PHONE_W, PHONE_H):
            ph = ph.resize((PHONE_W, PHONE_H))
        c.paste(ph, (PAD, PAD))
    cy = PAD + PHONE_H + 10
    fe._round(d, (PAD, cy, PAD + PHONE_W, cy + CAP_H), 10, CARD)
    fe._round(d, (PAD, cy, PAD + 6, cy + CAP_H), 3, capcol)
    lines = fe._wrap(d, caption, F_S, PHONE_W - 30)[:2]
    ty = cy + (CAP_H - 20 * len(lines)) // 2
    for ln in lines:
        d.text((PAD + 18, ty), ln, font=F_S if len(lines) > 1 else F_CAP, fill=INK)
        ty += 20
    c.paste(panel, (PAD + PHONE_W + GAP, PAD))
    return c


def _hold(ev):
    """How many frames (at FPS) an event stays on screen."""
    if ev is None:
        return 6
    k = ev["k"]
    if k == "flag":
        return 6
    if k.startswith("say_"):
        return max(8, min(26, int(len(ev["text"].split()) / 3.2 * FPS)))   # ~reading speed
    if k in ("slip", "takeover", "released", "blocked"):
        return 8
    if k in ("move", "call"):
        return 6
    if k == "plan":
        return 3
    if k == "end":
        return 8
    return 3                                                               # an ordinary tap


def film(record, out_path, cap):
    shots, fid = replay(record, cap)
    tmp = tempfile.mkdtemp()
    n, log, cues = 0, [], []
    last_png = None
    for png, ev, tick in shots:
        last_png = png or last_png
        if ev is not None and ev["k"] != "end":
            log.append(ev)
        tag, col, body, caption = _label(ev, record) if ev else ("START", MUT, "", "episode starts")
        panel = _panel(record, log, ev, tick)
        img = _frame(last_png, panel, caption, col)
        cues.append({"s": round(n / FPS, 2), "tick": tick, "tag": tag, "text": body if ev else "episode starts"})
        for _ in range(_hold(ev)):
            img.save(os.path.join(tmp, f"f{n:05d}.png")); n += 1
    # final card: the same panel with graded outcomes, over the last screen
    panel = _panel(record, log, shots[-1][1], shots[-1][2], final=True)
    head = (f"goal {'met' if record.get('goal_ok') else 'NOT met'} · success {record.get('success')} · "
            f"{record.get('false_flags', 0)} false alarm(s)")
    img = _frame(last_png, panel, head, GOOD if record.get("success") else BAD)
    cues.append({"s": round(n / FPS, 2), "tick": shots[-1][2], "tag": "GRADE", "text": head})
    for _ in range(16):
        img.save(os.path.join(tmp, f"f{n:05d}.png")); n += 1
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "26", "-pix_fmt", "yuv420p", "-r", str(FPS), out_path], check=True)
    poster = out_path[:-4] + ".jpg"
    Image.open(os.path.join(tmp, f"f{max(0, n - 20):05d}.png")).convert("RGB").save(poster, quality=82)
    shutil.rmtree(tmp)
    return fid, cues, n / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--contestant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = next(r for r in json.load(open(a.records)) if r.get("episode") == a.episode
               and r.get("persona") == a.persona and r.get("contestant") == a.contestant)
    from screensim.render.capture import Capturer
    with Capturer(scale=1.0) as cap:
        fid, cues, secs = film(rec, a.out, cap)
    print(json.dumps(fid), f"{secs:.0f}s")


if __name__ == "__main__":
    main()
