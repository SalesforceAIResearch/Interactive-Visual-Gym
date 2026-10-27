#!/usr/bin/env python3
"""VLM-as-controller v3: the subgoal-structured plan-act controller (standing config).

Decision every 2 ticks; observation = last 8 per-tick frames (or the text state map with the
last-8 path); the model's previous output JSON is fed back as its ONLY memory. Modes differ
only in the extras channel (see planact_v3_prompts). Every turn's full structure is logged so
bookkeeping accuracy can be scored against engine ground truth later.

    GEMINI_API_KEY=... COOKSIM_PAGE=... python -m cooksim_bench.backends.gemini_stream_v3 \
        --mode agent++ --gen medium_nops_medium_map --out out.json
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, base64, json, os, re, sys, time

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright

from cooksim.core.enums import Action
from cooksim.agents.highlevel import HighLevelController
from cooksim.agents.observe_text import observe_text
from cooksim_bench.backends.gemini_stream_rgb import make_gen_game, GPU, PAGE
from cooksim_bench.agent_context import dir_name, plate_in_order
from cooksim_bench.backends.cmd_feedback import FeedbackTracker
from cooksim_bench.backends.planact_v3_prompts import build_prompt, build_turn, MODES

POLL, WINDOW, CAP = 2, 8, int(os.environ.get("V3_CAP", "300"))
MODEL = "gemini-3.1-pro-preview"
NOOP = {"", "continue", "none", "no-op", "noop", "wait", "stay", "null"}


def gemini_json(client, sys_prompt, user, images=None):
    parts = [sys_prompt, user] + [types.Part.from_bytes(data=f, mime_type="image/png")
                                  for f in (images or [])]
    ck = dict(temperature=0.2, response_mime_type="application/json", max_output_tokens=6000,
              thinking_config=types.ThinkingConfig(thinking_budget=640))
    last = None
    for k in range(5):
        try:
            r = client.models.generate_content(model=MODEL, contents=parts,
                                               config=types.GenerateContentConfig(**ck))
            t = re.sub(r"^```(?:json)?|```$", "", (r.text or "").strip(), flags=re.M).strip()
            i = t.find("{")
            if i < 0:
                return {}
            obj, _ = json.JSONDecoder().raw_decode(t[i:])
            return obj if isinstance(obj, dict) else {}
        except Exception as e:
            last = e
            if isinstance(e, json.JSONDecodeError) or any(x in str(e) for x in
                    ("503", "500", "429", "RESOURCE", "UNAVAIL", "INTERNAL", "disconnect",
                     "Connection", "Expecting")):
                time.sleep(2 + 2 * k)
                continue
            raise
    # salvage: JSON never parsed after retries — recover at least the action field so the
    # episode continues (the structure resets from the last good prev_json next turn)
    try:
        m = re.search(r'"action"\s*:\s*"([^"]*)"', t)
        return {"action": m.group(1) if m else "", "reason": "(salvaged from malformed JSON)"}
    except Exception:
        return {"action": "", "reason": "(unparseable)"}


def mode_extras(mode, tracker, game):
    out = []
    if mode == "agent":
        out.append(tracker.block(level="minimal"))
    elif mode in ("agent+", "agent++", "agent+++", "text_state_map"):
        out.append(tracker.block(level="full"))
    cook = game.cooks[0]
    if mode in ("agent++", "agent+++"):
        out.append(f"CURRENT POSITION: the cook is at {tuple(cook.pos)} facing {dir_name(cook.direction)}.")
    if mode == "agent+++":
        out.append(f"CURRENT PLATE (components in the exact order they were added): {plate_in_order(cook.holding)}")
    return "\n\n".join(out)


def run_episode(mode, cell, client, page=None):
    game, recipe = make_gen_game(cell)
    ctrl = HighLevelController(game, 0)
    tracker = FeedbackTracker(k=5)
    sys_prompt = build_prompt(mode)
    prev_json, fbuf, turns, path = None, [], [], []
    start_wins = game.stats.get("deliveries", 0)
    t = 0
    for t in range(CAP):
        path.append(tuple(game.cooks[0].pos))
        if mode != "text_state_map":
            url = page.evaluate("(s)=>window.__renderFrame(s)", json.dumps(game.render_state()))
            fbuf.append(base64.b64decode(url.split(",", 1)[1]))
        if t % POLL == 0:
            tracker.refresh(ctrl)
            extras = mode_extras(mode, tracker, game)
            user = build_turn(mode, recipe, t, prev_json, extras)
            if mode == "text_state_map":
                user += "\n\n" + observe_text(game, path=path[-WINDOW:])
                images = None
            else:
                images = fbuf[-WINDOW:]
            out = gemini_json(client, sys_prompt, user, images)
            act = str(out.get("action", "") or "").strip()
            disp, fb = "noop", ""
            if act and act.lower() not in NOOP:
                _ex, fb, disp = tracker.submit(act, game, ctrl, strict=(mode != "basic"))
            prev_json = json.dumps(out) if out else prev_json
            turns.append(dict(t=t, action=act, disposition=disp, feedback=fb, structure=out))
            print(f"[t{t:03}] {time.strftime('%H:%M:%S')} {act!r:40} -> {disp} {fb[:50]}", flush=True)
        low = ctrl.step() if ctrl.skill is not None else int(Action.STAY)
        acts = [int(Action.STAY)] * len(game.cooks)
        acts[0] = low
        game.step(acts)
        if game.stats.get("deliveries", 0) > start_wins:
            break
    won = game.stats.get("deliveries", 0) > start_wins
    return dict(mode=mode, cell=cell, won=won, ticks=t + 1, decisions=len(turns),
                fails=game.stats.get("failed_deliveries", 0), turns=turns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if args.mode == "text_state_map":
        rec = run_episode(args.mode, args.gen, client)
    else:
        with sync_playwright() as p:
            b = p.chromium.launch(args=GPU)
            page = b.new_page(viewport={"width": 800, "height": 580})
            page.goto(PAGE, wait_until="load")
            page.wait_for_function("window.__ready===true", timeout=90000)
            page.wait_for_timeout(400)
            rec = run_episode(args.mode, args.gen, client, page)
            b.close()
    json.dump(rec, open(args.out, "w"), indent=1)
    print(f"{args.mode}/{args.gen}: {'WON' if rec['won'] else 'fail'} ticks={rec['ticks']} decisions={rec['decisions']}")
