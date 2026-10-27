#!/usr/bin/env python3
"""Roll out a RECOVERY test case with real VLMs on both sides.

Setup (recovery_cases.json): the episode is replayed to the flag tick; the supervisor
(standard free-form interaction prompt, frames observation) receives the AUGMENTED memory —
an accurate cook_progress whose previous "text" is the flag (right error, no suggestions) —
plus the scripted persona first reply from the human. From there everything is live:
supervisor talks free-form; the human VLM (persona-carded, past-only view) replies with
moves/parse; CONVICTION (accept move or applicable parse) enters error correction and
applies the parsed guidance (or self-repair for high-solvability personas); the supervisor
keeps streaming (poll 2, 8 frames, recurrent memory, full public communication history)
until the episode ends.

    GEMINI_API_KEY=... COOKSIM_PAGE=... python -m cooksim_bench.recovery_rollout \
        --scenario add_foreign__before_outcome --persona novice_stepwise --out f.json
"""
from cooksim_bench.paths import ROOT  # the CookSim release root
import argparse, base64, json, os, re, sys, time

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, "benchmark", "grid.json"))
# The frames this harness snaps ARE the supervisor's observation, and the GPU path
# duplicates the ego pane and darkens the scene (see film_states_smooth.py, the 2026-08-09
# header). Must be set before gemini_stream_rgb resolves GPU at import time.
# Override with COOKSIM_LIVE_GPU=1 only once the WebGL context loss is actually fixed.
if os.environ.get("COOKSIM_LIVE_GPU") != "1":
    os.environ["COOKSIM_FORCE_SWIFTSHADER"] = "1"
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright

from cooksim.interact import system
from cooksim.interact.state import RECOVERY_PERSONAS, PERSONA_CARDS
from cooksim.interact.protocol import supervisor_context, human_context
from cooksim_bench.exp_interact import build_episode
from cooksim_bench.scoring.interact_eval import gemini as _gemini_text


def gemini_text(client, sys_prompt, user, max_tok=1400):
    """Human-VLM call: bigger budget than the eval default (long parse dicts were hitting
    deterministic truncation), and a persistent parse failure degrades to silence instead
    of killing the episode."""
    if hasattr(client, "sup_call"):          # pluggable cook seat (assistant_backend wrapper)
        return client.sup_call(sys_prompt, user, []) or {}
    try:
        return _gemini_text(client, sys_prompt, user, max_tok=max_tok)
    except Exception:
        return {}
from cooksim_bench.backends.gemini_stream_rgb import GPU, PAGE

from cooksim_bench.backends.genai_client import make_client, default_model
MODEL = default_model()
POLL, WINDOW, MAX_ROUNDS = 2, 8, 4
MAX_ENGINE_CALLS = 5         # human "thinking" calls per turn (zero game time; supervisor: none)
# When set, a HIGH-ability cook's NAVIGATION must come from an engine_call return (correct
# grounding), not from memory — this forces the clever cook to actually USE its skills.
FORCE_EC = os.environ.get("COOKSIM_FORCE_ENGINE_CALL") == "1"
_BARE = {"pick up", "put down", "scoop", "turn on", "chop", "trash", "serve", "wash",
         "serve it", "put it down"}   # act on the FACED tile — no grounding needed


def gemini_frames(client, sys_prompt, user, frames):
    # PLUGGABLE ASSISTANT (2026-08-19): when the caller passes an assistant_backend
    # wrapper instead of a Gemini client, route to it — every assistant call in both
    # harnesses funnels through here, so this one branch swaps the assistant seat.
    if hasattr(client, "sup_call"):
        return client.sup_call(sys_prompt, user, frames)
    parts = [sys_prompt, user] + [types.Part.from_bytes(data=f, mime_type="image/png")
                                  for f in frames]
    ck = dict(temperature=0.2, response_mime_type="application/json", max_output_tokens=4000,
              thinking_config=types.ThinkingConfig(thinking_budget=512))
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
            # httpx timeouts (ReadTimeout / ConnectTimeout) can carry an EMPTY message, so a
            # str(e)-only match misses them; test the class name alongside the message.
            _sig = f"{type(e).__name__}: {e}"
            if isinstance(e, json.JSONDecodeError) or type(e).__name__ in ("ServerError", "APIError") \
                    or any(x in _sig for x in
                    ("503", "500", "502", "504", "429", "RESOURCE", "UNAVAIL", "INTERNAL",
                     "timed out", "Timeout", "timeout", "DeadlineExceeded",
                     # 499 CANCELLED is raised as ClientError, so the type-name check above
                     # misses it — and it is transient (the server dropped the call), unlike
                     # the other 4xx. Match the code, not the class: retrying a real 400/403
                     # forever would be worse than failing fast.
                     "499", "CANCELLED", "408",
                     "disconnect", "Connection", "Expecting", "overloaded")):
                time.sleep(2 + 2 * k)
                continue
            raise
    return {}


def sup_turn(client, sup_sys, prev_json, comm, said, frames, now, text_obs=None, state_extra=None):
    hist = "\n".join(f'  t{c["t"]} {c["speaker"]}: "{c["text"]}"' for c in comm)[-2400:]
    user = (f"CURRENT TIMESTAMP: t={now}\n\n"
            "== YOUR PREVIOUS OUTPUT (your memory — update cook_progress, then reply) ==\n"
            f"{prev_json}\n\n"
            "== COMMUNICATION HISTORY (public — everything said so far) ==\n"
            + (hist or "  (nothing yet)")
            + (f'\n\n== THE COOK JUST SAID ==\n  "{said}"' if said else ""))
    if text_obs is not None:                            # text-state-map observation tier
        return gemini_frames(client, sup_sys,
                             user + "\n\n== KITCHEN STATE (last 8 ticks) ==\n" + text_obs
                                  + "\n\nOutput your JSON for NOW.", [])
    extra = ("\n\n== EXACT STATE THIS TICK ==\n" + state_extra) if state_extra else ""
    return gemini_frames(client, sup_sys,
                         user + extra + f"\n\nBelow are your last {len(frames)} frames (one per "
                                "tick, oldest first). Output your JSON for NOW.", frames)


# Hard per-episode wall-clock cap (seconds), env-overridable like the step-in harness.
# 2700 truncated 3 of 4 presentation episodes at ~200 ticks on a contended box (the
# machine sustains ~49 game ticks/min TOTAL across all concurrent rollouts), and a
# truncated episode forfeits the delivery verdict — the headline metric — while also
# making a frames-vs-agent+++ comparison meaningless, since the two tiers get cut at
# different points.
WALL_BUDGET = int(os.environ.get("COOKSIM_WALL_BUDGET", 7200))   # was 2700; concurrent
                             # runs (API-latency dilation lowers ticks/sec) aren't cut short


def run(scenario, persona, client, page, video_path=None, obs_mode="frames", pauses=False):
    t0 = time.time()
    vdir = None
    vcount = [0]
    if video_path:
        import tempfile
        vdir = tempfile.mkdtemp(prefix="recv_", dir="tmp")
    cases = {c["name"]: c for c in json.load(open("tmp/interact_eval/recovery_cases.json"))}
    case = cases[scenario]
    prof = RECOVERY_PERSONAS[persona]
    CREATIVE = prof.task_solvability == "creative"      # sighted, imaginative tier
    # the LAST text the model reads each turn is not an instruction but WHO IT IS — the
    # persona card restated verbatim (turn hints kept overriding the persona; this doesn't).
    PHINT = (PERSONA_CARDS.get(prof.persona, "You are an ordinary cook.")
             + "\n\nStay in character. Output your reply JSON now.")
    state, ro = build_episode(((case.get("fn") or case["err"]), dict(case["kwargs"])),
                              "capable_collaborator", cell=case["cell"], inject=case["inject"])
    state.profile = prof
    ro.cap = 500
    pauses_log = []

    def speech_ticks(text_):
        # calibrated to the study app's actual TTS: SpeechSynthesis u.rate=1.05 on the
        # Chrome default en-US voice ~ 3.15 words/s; 1 tick ~ 1.2 s (move-throttle parity)
        return max(1, min(20, round(len(str(text_).split()) / 3.15 / 1.2)))

    if pauses:
        # study-app parity: 1 benchmark tick ~ one tile-move ~ the app's 1.2s move
        # throttle, and speech plays at ~2.8 words/s - so a sentence of W words costs
        # round(W/2.8/1.2) ticks of standing still (capped; conversation is not free).
        import random as _rnd
        ro.hesitate_p = 0.2
        ro._hes_rng = _rnd.Random(17)
        ro.hesitations = []
        # full study-app parity: conversation eats real kitchen time, so adopt the
        # study's slow-burn config too - otherwise every long exchange ends in a fire
        # the app participant would never see.
        for _k in ("pot_burn_time", "pan_burn_time", "oven_burn_time"):
            setattr(ro.game.config, _k, 480)
        ro.game.config.fire_time = 120
        # stations SNAPSHOT config at construction (game.py) - patch live instances too
        for _st in ro.game.stations.values():
            if hasattr(_st, "burn_time"):
                _st.burn_time = 480
            if hasattr(_st, "fire_time"):
                _st.fire_time = 120
    fbuf, comm, convo, report_moves, path = [], [], [], [], []
    ep_alive = [True]
    sourced = set()        # commands a skill (engine_call) handed back — the only grounded nav a
                           # clever cook may use when FORCE_EC is on

    def enforce_sourced(pd):
        """PROVENANCE (FORCE_EC): a HIGH-ability cook may ONLY use actions a skill RETURNED
        (engine_call) — it may not imagine a plan from memory. EVERY action is dropped unless it
        is in the sourced set, except the two universal verbs the supervisor names directly
        (trash / serve). An imagined plan therefore applies NOTHING, so the cook must actually
        call resolve_subgoal to ground it. Grounding is NOT re-added downstream (see the
        no_prefix flag), so guessed 'go to X' cannot sneak back."""
        if not (FORCE_EC and prof.task_solvability == "high"):
            return pd, None
        from cooksim.interact.checks import is_grounded
        kept, dropped = [], []
        for a in (pd.get("suggested_actions") or []):
            al = str(a).strip().lower()
            if a in sourced or al in _BARE or not al.startswith("go to"):
                kept.append(a); continue
            # a "go to X" the cook wrote itself: grounding only matters to DISAMBIGUATE. If it
            # resolves to a UNIQUE tile (the trash, a source, the serving pass, a qualified pan),
            # there is no wrong-instance risk -> allow. Only an AMBIGUOUS / ungroundable guess
            # ("go to the pan" when two exist) must come from a skill.
            g = is_grounded(a, ro.game, state.recipe)
            if g.get("unique"):
                kept.append(a)
            else:
                dropped.append(a)
        note = None
        if dropped:
            note = ("NOTHING you just listed was applied — you WROTE those steps from memory, but a "
                    "skilled cook WORKS THEM OUT: call resolve_subgoal('<the subgoal the supervisor named>') "
                    "first, then put ONLY the actions it returns into "
                    "`actions`. Dropped: " + ", ".join(map(str, dropped[:5])) + ".")
        return {"suggested_subgoals": pd.get("suggested_subgoals") or [], "suggested_actions": kept}, note

    def sup_obs_text():
        from cooksim.agents.observe_text import observe_text
        return observe_text(ro.game, path=path[-WINDOW:], viewpoint="supervisor")

    def sup_state_extra():
        """agent+++ readout — SINGLE SOURCE in python -m cooksim_bench.agent_context (see the bug it
        fixes there). Never inline a private copy of this."""
        from cooksim_bench.agent_context import supervisor_state_extra
        return supervisor_state_extra(ro.game)
        page.wait_for_function("() => window.__renderFrameSbs !== undefined", timeout=90000)
        # PHOTOREAL LIGHTING IS ASYNC. render3d.js loads env.hdr through RGBELoader and
        # only then applies the image-based lighting; a frame captured before that is lit
        # by the dim fallback and REAL-mode surfaces come out near-black. One render is
        # needed first because the renderer (and the flag) are constructed lazily.
        try:
            page.evaluate("(s)=>window.__renderFrame(s)",
                          json.dumps(ro.game.render_state()))
            page.wait_for_function("() => window.__envReady === true", timeout=120000)
        except Exception as _e:
            print(f"[warn] photoreal env wait failed: {_e}", flush=True)
    _dump = os.environ.get("COOKSIM_DUMP_STATES")
    _fdir = os.environ.get("COOKSIM_FRAME_DIR")
    if _fdir:
        os.makedirs(_fdir, exist_ok=True)
    # CONVERSATION-FROZEN FRAMES (fix): while a speech pause drains, the cook's hands are
    # stopped BY THE CONVERSATION — every tick renders the same motionless cook. Appending
    # those frames flooded the supervisor's 8-frame evidence window with duplicates: on the
    # worst turn of the pause episode 8/8 frames were frozen and identical, so the model
    # could not see that the cook had been walking to the counter, concluded it was being
    # ignored ("You are still empty-handed!"), and spoke again — which bought another
    # pause. A self-feeding stall: 40% of that episode's ticks were conversation-frozen.
    # The study app already excluded frozen frames from its ring; the harness now matches,
    # with one refinement: exactly ONE frozen frame is kept, refreshed in place, so the
    # window is 7 frames of REAL action history + the current situation — hazards that
    # develop during a long pause (a pan starting to smoke) stay visible.
    _pause_slot = [False]

    def snap():
        path.append(tuple(ro.game.cooks[0].pos))
        _st = json.dumps(ro.game.render_state())
        if _dump:                          # env-gated per-tick state dump (JSONL)
            with open(_dump, "a") as _f:
                _f.write(_st + "\n")
        # SIDE-BY-SIDE OBSERVATION (2026-07-31): one image per tick carrying BOTH
        # cameras — top-down LEFT (the coordinate frame), first-person RIGHT — so the
        # supervisor sees a synchronised pair by construction. JPEG: 36 KB vs 196 KB
        # for the old PNG. The same frames are written to disk, so the presentation
        # video needs NO separate filming pass and still gets both views.
        url = page.evaluate("(s)=>window.__renderFrameSbs(s)", _st)
        png = base64.b64decode(url.split(",", 1)[1])
        _frozen = bool(pauses) and getattr(ro, "_pause", 0) > 0
        if _frozen and _pause_slot[0] and fbuf:
            fbuf[-1] = png                 # refresh the single "now" frame in place
        else:
            fbuf.append(png)
        _pause_slot[0] = _frozen
        del fbuf[:-12]                     # supervisor needs only the last 8 frames
        if vdir is not None:
            open(f"{vdir}/{vcount[0]:05d}.jpg", "wb").write(png)
            vcount[0] += 1
        if _fdir:                          # film-ready capture, one composite per tick
            open(f"{_fdir}/{len(path)-1:05d}.jpg", "wb").write(png)

    def world_tick(n, pause=False):
        for _ in range(n):
            if pause:                       # cook is ENGAGED IN CONVERSATION: hands stop,
                ro._pause = 1               # stations/smoke keep running (real time cost)
            snap()
            if not ro.step():
                ep_alive[0] = False
                return

    # ---- fast-forward to the flag tick
    while ro.tick < case["flag_tick"]:
        snap()
        if not ro.step():
            break
    ft = ro.tick
    sup_sys = supervisor_context(state, obs_mode=obs_mode, game=getattr(ro, 'game', None),
                                 hazard_ticks=(ro.game.config.pan_burn_time,
                                               ro.game.config.fire_time,
                                               ro.game.config.smoke_grace))
    prev_json = case["supervisor_prev_json"]         # the augmented memory + flag text
    flag = case["flag_text"]
    first_reply = case["personas"][persona]["human_first_reply"]
    comm.append({"t": ft, "speaker": "vlm (you)", "text": flag})
    comm.append({"t": ft, "speaker": "cook", "text": first_reply})
    convo = [{"speaker": "vlm", "text": flag}, {"speaker": "human", "text": first_reply}]
    report = dict(scenario=scenario, persona=persona, flag_tick=ft, convinced=False,
                  entered=False, fix_source=None, obs_mode=obs_mode,
                  conversation=[{"speaker": "vlm", "t": ft, "text": flag, "round": 1},
                                {"speaker": "human", "t": ft, "text": first_reply, "round": 1}])
    RND = [1]

    def engine_snap():
        try:
            return state.to_json(ro)
        except Exception:
            return None
    state.utterance_log.append({"t": ft, "speaker": "vlm", "text": flag})
    state.utterance_log.append({"t": ft, "speaker": "human", "text": first_reply})

    # ---- live conversation from the flag: supervisor first (it must answer the reply).
    # The cook's hands PAUSE while it is engaged in conversation (world keeps running);
    # parses ACCUMULATE across the whole conversation and are applied ONCE at the end,
    # in dependency (template) order — no half-instruction executes early.
    acc = {"suggested_subgoals": [], "suggested_actions": []}

    def absorb(pd):
        for k in acc:
            for v in (pd or {}).get(k) or []:
                if v not in acc[k]:
                    acc[k].append(v)

    def _safe_why(cmd):
        try:
            return _why(cmd)
        except NameError:
            return "precondition not met"

    def body_status():
        """Creative-tier standing status: position/facing + the three-state action lifecycle
        (in progress / finished / rejected) so the cook never re-issues a step its body is
        still walking."""
        from cooksim_bench.agent_context import dir_name
        c = ro.game.cooks[0]
        term = {"done": "FINISHED", "rejected": "REJECTED", "illegal-skipped": "REJECTED"}
        lines = [f"YOUR POSITION: {tuple(c.pos)}, facing {dir_name(c.direction)}."]
        cur = next((l for l in reversed(ro.log)
                    if l["status"] in ("running", "waiting")), None)
        if cur is not None:
            age = ro.tick - (cur.get("t_issue") or ro.tick)
            blocked = f' — blocked: {_safe_why(cur["cmd"])}' if cur["status"] == "waiting" else ""
            lines.append(f'CURRENT ACTION: "{cur["cmd"]}" — IN PROGRESS ({age} ticks{blocked})')
        else:
            lines.append("CURRENT ACTION: (none — your body is idle)")
        recent = [l for l in ro.log if l["status"] in term][-3:]
        if recent:
            lines.append("RECENT RESULTS: " + " ; ".join(
                f'"{l["cmd"]}" {term[l["status"]]} at t{l.get("t_done") or l.get("t_issue")}'
                + (f' ({_safe_why(l["cmd"])})' if term[l["status"]] == "REJECTED" else "")
                for l in recent))
        left = [c for c, _k in ro.cmds[ro.i + 1:]]
        lines.append("QUEUED AFTER: " + (" / ".join(left) if left else "(nothing)"))
        return "\n".join(lines)

    def human_llm(turn_hint, feedback=None):
        """One human turn, with a THINKING stage in front: a high-ability cook may call engine
        functions to work the steps out ITSELF (resolve_subgoal / expand_action /
        full_repair_plan). Thinking costs NO game time — no world_tick happens in this loop —
        and the supervisor never sees it (it has no engine channel at all). Returns
        (reply_json, scratch)."""
        scratch, h = [], {}

        def _harvest():
            # remember every command a skill handed back this episode -> the "sourced" set
            for e in scratch:
                rep = e.get("reply") or {}
                if rep.get("ok"):
                    for a in (rep.get("result") or {}).get("actions", []) or []:
                        sourced.add(a)

        for _ in range(MAX_ENGINE_CALLS):
            hint = turn_hint
            # STRONG NUDGE: on a skilled cook's FIRST reply of a turn, if it needs concrete steps
            # it wasn't literally handed, force it to THINK (engine_call) rather than guess.
            _q_left = len(ro.cmds) - ro.i - 1
            if FORCE_EC and prof.task_solvability == "high" and not scratch and _q_left <= 1:
                hint = ('You are a SKILLED cook. Queue what you know; resolve what you would '
                        'otherwise guess. If the exact grounded form of your next actions is '
                        'already determined, queue them directly. If any part would be a guess '
                        '(which pan? which counter? what steps?), reply RIGHT NOW with a single '
                        'engine_call — e.g. {"engine_call": {"fn": "resolve_subgoal", "args": '
                        '{"subgoal": "<what the supervisor asked>"}}} — guessed steps are '
                        'dropped. ' + turn_hint)
            ctx = human_context(state, ro, convo, feedback, engine_scratch=scratch,
                                game=ro.game, body_extra=(body_status() if CREATIVE else ""))
            h = (gemini_frames(client, ctx, hint, fbuf[-8:]) if CREATIVE
                 else gemini_text(client, ctx, hint))
            print(f"[t{ro.tick}] human keys={list(h.keys())} ec={bool(h.get('engine_call'))} "
                  f"solv={prof.task_solvability}", flush=True)
            call = h.get("engine_call")
            if not (isinstance(call, dict) and call.get("fn")):
                _harvest()
                return h, scratch
            if prof.task_solvability not in ("high", "creative"):
                # ability is enforced by the mechanism, not just the prompt: an unskilled cook
                # cannot work steps out — it has to ask the supervisor.
                scratch.append({"call": call, "reply": {"ok": False, "error":
                    "you can't work that out for yourself — ask the supervisor for the concrete step"}})
                continue
            rep = system.engine_call(call.get("fn"), call.get("args") or {}, state, ro.game)
            print(f"[t{ro.tick}] ENGINE_CALL {call.get('fn')}({call.get('args')}) -> "
                  f"ok={rep.get('ok')} n={len((rep.get('result') or {}).get('actions', []))}", flush=True)
            scratch.append({"call": call, "reply": rep})
        ctx2 = human_context(state, ro, convo, feedback, engine_scratch=scratch,
                             game=ro.game, body_extra=(body_status() if CREATIVE else ""))
        hint2 = turn_hint + (" You have thought enough — output your REPLY json now, "
                             "with no engine_call.")
        h = gemini_frames(client, ctx2, hint2, fbuf[-8:]) if CREATIVE else gemini_text(client, ctx2, hint2)
        _harvest()
        return h, scratch

    def plan_op_actions(h):
        """New human schema: plan_update {op: drop|append|replace, actions: [...]}. Returns
        (op, parse_dict) where parse_dict is the shape convert_parse consumes and the page
        builder renders. Tolerates the legacy parse_dict shape.

        CREATIVE tier [v5]: there is NO plan_update channel — the plan IS the queue. The
        submission is derived from the structure itself: [current_action] + actions_future.
        Identical to the remaining queue -> drop (no-op); head equal to the running command
        -> the head is not re-issued (no more go-to/go-to duplicates)."""
        pu = h.get("plan_update")
        if isinstance(pu, dict):
            op = str(pu.get("op") or "").strip().lower()
            acts = [a for a in (pu.get("actions") or []) if a]
            if op not in ("drop", "append", "replace"):
                op = "append" if acts else "drop"
            if op == "drop":
                acts = []
            return op, {"suggested_subgoals": [], "suggested_actions": acts}
        pd = h.get("parse_dict") or {}
        items = (pd.get("suggested_actions") or []) + (pd.get("suggested_subgoals") or [])
        return ("append" if items else "drop"), pd

    said, closed = first_reply, False
    for rnd in range(MAX_ROUNDS):
        if time.time() - t0 > WALL_BUDGET:
            report["aborted"] = "wall_clock"
            break
        # pause mode: the supervisor cannot hear an utterance before the human has
        # finished SAYING it - the initial rounds used a fixed 2-tick beat, which read
        # as an instant reply to a long sentence (t -> t+2)
        if pauses and said:
            n_sp = speech_ticks(said)
            pauses_log.append({"t": ro.tick, "who": "human",
                               "words": len(str(said).split()), "ticks": n_sp})
            world_tick(n_sp, pause=True)
        world_tick(2, pause=True)
        if not ep_alive[0]:
            break
        out = sup_turn(client, sup_sys, prev_json, comm, said, fbuf[-WINDOW:], ro.tick,
                       text_obs=(sup_obs_text() if obs_mode=='text_state_map' else None),
                       state_extra=(sup_state_extra() if obs_mode=='agent+++' else None))
        prev_json = json.dumps(out) if out else prev_json
        text = (out.get("text") or "").strip()
        report["conversation"].append({"speaker": "vlm", "t": ro.tick, "text": text,
                                       "round": RND[0], "vlm_json": out, "engine": engine_snap()})
        if text:
            comm.append({"t": ro.tick, "speaker": "vlm (you)", "text": text})
            convo.append({"speaker": "vlm", "text": text})
        world_tick(2, pause=True)
        if not ep_alive[0] or closed:
            break
        h, think = human_llm(PHINT)
        u = h.get("utterance", "")
        report_moves.append(h.get("move"))
        h_op, h_pd = plan_op_actions(h)
        report["conversation"].append({"speaker": "human", "t": ro.tick, "text": u,
                                       "move": h.get("move"), "parse_dict": h_pd, "op": h_op,
                                       "close": h.get("close_conversation"), "engine_calls": think,
                                       "round": RND[0], "human_json": h, "engine": engine_snap()})
        if h.get("close_conversation"):
            RND[0] += 1
        comm.append({"t": ro.tick, "speaker": "cook", "text": u})
        convo.append({"speaker": "human", "text": u})
        said = u
        absorb(h_pd)
        if h.get("close_conversation"):
            closed = True
            said = None
        # conviction check each round: accept move or usable ACCUMULATED content ends it
        filt = system.convert_parse(acc, prof, state.recipe, ro.game, auto_resolve=False)
        if ("accept" in report_moves or filt["anything_applicable"]) and closed:
            break
        if "accept" in report_moves and rnd >= 1:
            break

    # ---- conviction gate + application. `op` is the human's explicit choice (drop/append/
    # replace) from its plan_update; the engine honors it instead of blind full-replace.
    def apply_guidance(filt_now, tag, op="replace"):
        if op == "drop":
            return                                     # human ignored the advice; queue untouched
        if state.mode != "error_correction":
            system.enter_error_correction(ro, state)
            report["entered"] = True
        if filt_now["anything_applicable"]:
            system.apply_fix(ro, state, filt_now["commands"], op=op)
            report["fix_source"] = report.get("fix_source") or tag
        # NO engine self-repair fallback any more: a capable cook that wants the full plan calls
        # full_repair_plan() itself in its thinking stage. The queue only ever holds steps the
        # human actually decided on — engine-injected steps it never authored drove the thrash.
        # The queue/mode change caused by THIS parse belongs to the human turn that produced
        # it — refresh that bubble's engine snapshot so the "queued: ..." effect shows on it,
        # not on the next turn (the snapshot was taken before this apply ran).
        for entry in reversed(report["conversation"]):
            if entry["speaker"] == "human":
                entry["engine"] = engine_snap()
                break

    acc_e, _ec_note0 = enforce_sourced(acc)
    filt = system.convert_parse(acc_e, prof, state.recipe, ro.game, auto_resolve=False,
                                no_prefix=(FORCE_EC and prof.task_solvability=='high'))
    convinced = ("accept" in report_moves) or filt["anything_applicable"]
    report.update(convinced=convinced, moves=report_moves,
                  filter={k: filt[k] for k in ("kept", "dropped_by_acceptance",
                                               "unusable_for_solvability", "anything_applicable")})
    if convinced and ep_alive[0]:
        # flag-time bootstrap: entering correction clears the deviated future, so the accumulated
        # repair is laid in as a replace (the cook commits to the corrected plan at the pause).
        apply_guidance(filt, "conversation", op="replace")

    # ---- STREAMING loop [v3.1, design 2026-07-15]: the supervisor is a NORMAL streaming
    # watcher (polled every POLL ticks, standard prompt, speaks or stays silent on its own).
    # The SYSTEM enforces one invariant so the supervisor never faces a frozen, silent cook:
    #   between two supervisor observations the cook has EITHER moved (a valid action changed
    #   the frame) OR the human has said something. When the cook is stuck-and-silent, the
    #   system makes the human report BEFORE the supervisor's next look.
    from cooksim.interact import checks as _checks
    REASON = {"hands_full": "my hands are full", "no_plate": "I'm not holding a plate",
              "hands_empty": "my hands are empty", "station_empty": "there is nothing at the station",
              "no_finished_plate": "the plate isn't complete enough to serve",
              "no_such_place": "I can't find that place", "unparseable": "that isn't an action I can do",
              "not_cooked_yet": "the food isn't cooked yet"}
    seen_term, fb_buf = set(), []

    def _why(cmd):
        """Best-effort human-readable reason a command failed / is stuck. Manipulations get the
        engine's own precondition message (accurate about the FACED tile — e.g. a plate can't
        rest on a cutting board); other commands fall back to the is_doable blocker map."""
        from cooksim.agents.highlevel import parse as _hl_parse, _validate, MANIPULATIONS
        it = _hl_parse(cmd)
        if it and it.verb in MANIPULATIONS:
            try:
                ok, reason = _validate(it.verb, ro.game, ro.game.cooks[0])
                if not ok and reason:
                    return reason
            except Exception:
                pass
        try:
            b = _checks.is_doable(cmd, ro.game).get("blocker")
        except Exception:
            return "it wouldn't work"
        if b == "no_such_place":
            tgt = str(cmd).strip()
            if tgt.lower().startswith("go to "):
                tgt = tgt[6:]
            return f"I can't find {tgt}"      # e.g. "I can't find the pan with the meat"
        if b and b not in REASON and " " in str(b):
            return str(b)                     # a blocker that is already a sentence IS the reason
        return REASON.get(b, "it wouldn't work")

    def scan_feedback():
        for idx, l in enumerate(ro.log):
            if not str(l.get("kind", "")).startswith(system.FIX):
                continue
            # "waiting" = a manipulation stuck in RETRY because its precondition isn't met (e.g.
            # a put-down whose faced tile can't take the plate). It never used to be surfaced,
            # so the cook sat stuck holding the plate forever with the human none the wiser —
            # now the human hears it and can escalate ("there's nowhere valid to set it down").
            if l["status"] not in ("done", "rejected", "illegal-skipped", "waiting",
                                    "orphan-skipped"):
                continue
            key = (idx, l["status"])
            if key in seen_term:
                continue
            seen_term.add(key)
            cmd = l["cmd"]
            if l["status"] == "done":
                # ECR-13 §D1: a command that legally changed NOTHING (turn on at a
                # station that was already lit) must not read as "you lit it".
                fb_buf.append(f'you DID: "{cmd}"'
                              + (f' — {l["note"]}' if l.get("note") else ""))
            elif l["status"] == "orphan-skipped":
                fb_buf.append(f'SKIPPED "{cmd}" - the go-to before it never arrived, so it was '
                              "not executed (re-plan from where you actually are)")
            elif l["status"] == "waiting":
                fb_buf.append(f'you are STUCK trying "{cmd}" - {_why(cmd)}')
            else:
                fb_buf.append(f'you TRIED "{cmd}" but it was REJECTED - {_why(cmd)}')

    def feedback_text():
        return "\n".join(f"  - {x}" for x in fb_buf) if fb_buf else "(nothing new since you last acted)"

    def snap_state():
        c = ro.game.cooks[0]
        ndone = sum(1 for l in ro.log if l["status"] == "done")
        return (tuple(c.pos), type(c.holding).__name__ if c.holding is not None else None, ndone, ro.i)

    def add_speech_pause(who, text_):
        """SPEAKING costs time: the cook stands still for the duration of its OWN
        utterance (app parity: speak-freeze). LISTENING is free - real humans keep
        working while the assistant talks, and the study app plays TTS over live play.
        (First pause design froze the cook for both directions; a chatty supervisor
        then froze the novice for 73% of the episode and the pan burned - not a
        bookkeeping bug, a realism bug.) No log entry; timestamps simply advance;
        ro.cap grows so pauses never eat the tick budget."""
        if not pauses or not text_ or who != "human":
            return
        words = len(str(text_).split())
        ticks = speech_ticks(text_)
        ro._pause = getattr(ro, "_pause", 0) + ticks
        ro.cap += ticks
        pauses_log.append({"t": ro.tick, "who": who, "words": words, "ticks": ticks})

    def human_turn(idle=False, event=False):
        """Give the human a turn. Three triggers, three hints: the supervisor spoke (reply or
        stay silent in character); an action was rejected/stuck (EVENT — handle it; creative
        cooks may fix it silently); truly idle (report/ask, or self-plan if creative)."""
        scan_feedback()
        turn_hint = PHINT
        h, think = human_llm(turn_hint, feedback_text())
        u = (h.get("utterance") or "").strip()
        h_op, h_pd = plan_op_actions(h)
        rec = {"speaker": "human", "t": ro.tick, "text": u,
               "move": h.get("move"), "parse_dict": h_pd, "op": h_op,
               "close": h.get("close_conversation"), "round": RND[0],
               "engine_calls": think,
               "human_json": h, "engine": engine_snap()}
        if CREATIVE:
            # provenance is LOGGED here, not enforced: skill / parsed / imagined / bare per action
            last_sup = next((m["text"].lower() for m in reversed(convo)
                             if m.get("speaker") == "vlm" and m.get("text")), "")
            _stop = {"go", "to", "the", "a", "an", "up", "down", "on", "with", "of"}
            tags = []
            for a in (h_pd.get("suggested_actions") or []):
                al = str(a).strip().lower()
                if a in sourced:
                    tags.append("skill")
                elif al in _BARE:
                    tags.append("bare")
                else:
                    words = [w for w in re.findall(r"[a-z]+", al) if w not in _stop]
                    tags.append("parsed" if words and all(w in last_sup for w in words)
                                else "imagined")
            rec["action_sources"] = tags
        report["conversation"].append(rec)
        rej_seen[0] = sum(1 for l in ro.log
                          if l["status"] in ("rejected", "illegal-skipped", "waiting"))
        if u:
            comm.append({"t": ro.tick, "speaker": "cook", "text": u})
            convo.append({"speaker": "human", "text": u})
            add_speech_pause("human", u)
        if h.get("close_conversation"):
            RND[0] += 1                     # streaming closes start new rounds too (was never bumped here)
        if h_op != "drop":                  # "drop" = ignore the suggestion; queue left untouched
            h_pd_e, ec_note = enforce_sourced(h_pd)
            if ec_note:
                fb_buf.append(ec_note)
            f = system.convert_parse(h_pd_e, prof, state.recipe, ro.game, auto_resolve=False,
                                     no_prefix=(FORCE_EC and prof.task_solvability=='high'))
            if f["anything_applicable"]:
                apply_guidance(f, "report", op=h_op)
        fb_buf.clear()
        return u or None

    prev_json = [prev_json]
    prev_obs = snap_state()
    said = None
    rej_seen = [0]                     # terminal-failure count last seen by the human (creative)
    while ep_alive[0]:
        if time.time() - t0 > WALL_BUDGET:
            report["aborted"] = "wall_clock"
            break
        snap()
        # While a speech pause is draining, the DIALOGUE clock stops too: the pause IS
        # the time this exchange occupied — polling the supervisor mid-pause let it pile
        # new speech onto an already-frozen cook (pause debt grew faster than it drained).
        if ro.tick % POLL == 0 and (not pauses or getattr(ro, "_pause", 0) == 0):
            unresolved = not (state.mode == "error_correction" and state.correction
                              and state.correction.get("fix_applied") and system.resolved(ro, state))
            # GENUINELY idle = queue empty, no active skill, NOT waiting on a cooking timer,
            # not paused. (Walking keeps ctrl.skill busy; waiting for food to cook keeps
            # ro._wait>0 and the pan visibly changes — neither counts as "nothing to do".)
            idle = (ro.i + 1 >= len(ro.cmds) and ro.ctrl.skill is None
                    and getattr(ro, "_wait", 0) == 0 and getattr(ro, "_pause", 0) == 0
                    and getattr(ro, "_wander", 0) == 0)
            # SYSTEM GUARANTEE: only when the cook has truly nothing to do and hasn't spoken,
            # make the human report first — so the supervisor never faces a frozen, silent cook.
            # (A walking or food-waiting cook produces a changing frame; no report forced.)
            if said is None and unresolved and idle:
                said = human_turn(idle=True)      # poll only; a proper idle human reports itself
            # creative tier is EVENT-DRIVEN on failures too: a fresh rejected/stuck result gives
            # the sighted cook a turn immediately, so "rejected" feedback is acted on, not stale.
            if CREATIVE and said is None:
                nrej = sum(1 for l in ro.log
                           if l["status"] in ("rejected", "illegal-skipped", "waiting"))
                if nrej > rej_seen[0]:
                    said = human_turn(event=True)
            out = sup_turn(client, sup_sys, prev_json[0], comm, said, fbuf[-WINDOW:], ro.tick,
                           text_obs=(sup_obs_text() if obs_mode=='text_state_map' else None),
                       state_extra=(sup_state_extra() if obs_mode=='agent+++' else None))
            prev_json[0] = json.dumps(out) if out else prev_json[0]
            said = None
            text = (out.get("text") or "").strip()
            report["conversation"].append({"speaker": "vlm", "t": ro.tick, "text": text,
                                           "round": RND[0], "vlm_json": out, "engine": engine_snap()})
            prev_obs = snap_state()
            if text:
                comm.append({"t": ro.tick, "speaker": "vlm (you)", "text": text})
                convo.append({"speaker": "vlm", "text": text})
                add_speech_pause("vlm", text)   # the human stops to LISTEN
                said = human_turn()          # human reacts: parse guidance + optional reply
        if state.mode == "error_correction" and state.correction and \
                state.correction.get("fix_applied") and system.resolved(ro, state):
            system.exit_correction(ro, state)
            report["resolved_at"] = ro.tick
        if not ro.step():
            break
    prev_json = prev_json[0]
    report.update(outcome=ro.outcome, ticks=ro.tick)
    if pauses:
        report.update(pauses_enabled=True, pauses=pauses_log,
                      hesitations=getattr(ro, "hesitations", []))
    if vdir is not None:
        # The EPISODE DATA is what matters — a video-encoding failure (missing ffmpeg on a
        # login node, bad codec) must never destroy a completed rollout. Best-effort only.
        import shutil, subprocess as sp
        try:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                if pauses:
                    # REAL-TIME video: 1 tick = 1.2 s (matches the study app's pacing and
                    # the speech-tick formula), with a red SPEAKING banner burned in over
                    # every speech-pause window so the frozen cook is self-explanatory.
                    fr = f"{1/1.2:.4f}"
                    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
                    if pauses_log:
                        en = "+".join(f"between(t\\,{q['t']*1.2:.1f}\\,{(q['t']+q['ticks'])*1.2:.1f})"
                                      for q in pauses_log)
                        vf += f",drawbox=x=0:y=0:w=iw:h=ih:color=red@0.5:t=10:enable='{en}'"
                        _font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                        if os.path.exists(_font):
                            vf += (f",drawtext=fontfile={_font}:text='SPEAKING':"
                                   f"x=(w-text_w)/2:y=24:fontsize=34:fontcolor=white:box=1:"
                                   f"boxcolor=red@0.7:boxborderw=10:enable='{en}'")
                    sp.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", fr,
                            "-i", f"{vdir}/%05d.png", "-movflags", "+faststart",
                            "-vf", vf, "-r", "10", "-pix_fmt", "yuv420p",
                            "-c:v", "libx264", "-crf", "26", video_path], check=True)
                else:
                    sp.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "6",
                            "-i", f"{vdir}/%05d.png", "-movflags", "+faststart",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
                            "-c:v", "libx264", "-crf", "26", video_path], check=True)
            else:
                report["video_error"] = "ffmpeg not found — frames kept at " + vdir
                return report                      # keep the frames so the video can be made later
        except Exception as e:
            report["video_error"] = f"{type(e).__name__}: {e}"
            return report                          # keep frames; data survives
        shutil.rmtree(vdir, ignore_errors=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--persona", required=True, choices=list(RECOVERY_PERSONAS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sup-obs", default="frames", choices=["frames", "text_state_map", "agent+++"],
                    help="supervisor observation channel: rendered frames (agent+ tier) or the "
                         "text state map (text-ablation tier)")
    ap.add_argument("--pauses", action="store_true",
                    help="human-realism pauses: speech/listening cost ticks; 20%% per-step "
                         "hesitation while walking")
    args = ap.parse_args()
    client = make_client(120_000)   # vertex (Salesforce ADC) or legacy API key
    with sync_playwright() as p:
        b = p.chromium.launch(args=GPU)
        page = b.new_page(viewport={"width": 800, "height": 580})
        page.goto(PAGE, wait_until="load", timeout=180000)   # photoreal init is heavy:
        # the HDRI + PBR material library take far longer than the classic style,
        # and a dozen concurrent starts blew the 30s default
        page.wait_for_function("window.__ready===true", timeout=90000)
        page.wait_for_timeout(400)
        rec = run(args.scenario, args.persona, client, page,
                  video_path=args.out.replace(".json", ".mp4"), obs_mode=args.sup_obs,
                  pauses=args.pauses)
        b.close()
    json.dump(rec, open(args.out, "w"), indent=1, default=str)
    # auto-eval: quantitative scoring embeds into the report (no LLM); set EVAL_JUDGE=1
    # to also run the in-persona questionnaire + judge (2 Gemini calls)
    from cooksim_bench.scoring.post_episode import score_record
    score_record(args.out, judge=os.environ.get("EVAL_JUDGE") == "1")

    print(f"{args.scenario} x {args.persona}: outcome={rec['outcome']} convinced={rec['convinced']} "
          f"entered={rec['entered']} fix={rec['fix_source']} ticks={rec['ticks']}")
