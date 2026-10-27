#!/usr/bin/env python3
"""VH RECOVERY rollout — structural port of CookSim tools/recovery_rollout.py (handover §4).

The episode is replayed (live Unity) to the FLAG POINT; the supervisor receives the AUGMENTED
memory (accurate human_progress whose previous "text" IS the flag — right error, no
suggestions) plus the scripted persona first reply. From there everything is live: bounded
negotiation (MAX_ROUNDS=4, parses ACCUMULATE, applied ONCE at conviction), then the streaming
loop (POLL=2 ticks, 8 tick-frames, recurrent memory, full public communication history) until
the episode ends. Channel isolation: only SENTENCES cross between the two agents.

Usage: iv_recovery_rollout.py PORT CASE VARIANT PERSONA OUT_JSON
       (VARIANT = before_outcome | after_outcome)
"""
import json, os, sys, time

from vhhome_bench import protocol as P
from vhhome_bench import cases as CS
from vhhome.world import LiveWorld, pretty_cmd, ENGINE_REV
from vhhome.world import base as _base

WALL_BUDGET = 2700
CAP = 300          # tick cap (seconds of world time)
IDLE_END = 10      # consecutive silent+empty polls that end the episode
NO_PROGRESS = int(os.environ.get("VH_NO_PROGRESS", "60" if os.environ.get("VH_MIDLEVEL", "0") != "0" else "45"))
# ticks of activity with no state change AND no room change before a run counts as stuck;
# higher under mid-level, where finding a target takes real walking


class Queue:
    def __init__(self, cmds):
        self.cmds = list(cmds)      # [(step, sid_or_kind)]
        self.i = 0

    def next(self):
        if self.i < len(self.cmds):
            st = self.cmds[self.i]; self.i += 1
            return st
        return None

    def remaining(self):
        return self.cmds[self.i:]

    def remaining_texts(self):
        return [pretty_cmd(s) for s, _ in self.remaining()]

    def replace_rest(self, new):
        self.cmds = self.cmds[:self.i] + list(new)

    def append(self, new):
        self.cmds.extend(new)

    def skip_orphans(self, failed_step):
        """Skill §3 'orphan-skipped' (audit fix #60): a manipulation queued behind a FAILED
        approach to the same target cannot succeed — four dependents once fired blind after a
        rejected walk (open/put/close/switch, all rejected, all noise). Drops the consecutive
        dependents on the failed target and returns them."""
        tgt = _base(failed_step[1]) if len(failed_step) > 1 else None
        if tgt is None:
            return []
        dropped = []
        MANIP = ("Open", "Close", "Grab", "Put", "PutIn", "SwitchOn", "SwitchOff", "Drink")
        while self.i < len(self.cmds):
            st, sid = self.cmds[self.i]
            involved = {(_base(st[1]) if len(st) > 1 else None),
                        (_base(st[3]) if len(st) > 3 else None)}
            if st[0] in MANIP and tgt in involved:
                dropped.append((st, sid))
                self.i += 1
            else:
                break
        return dropped


def make_runner(world, case, persona, client, report, entered, correction, clean_nodes):
    """Shared human-turn / apply machinery used by both rollouts."""
    prof = P.PROFILES[persona]
    CREATIVE = prof["task_solvability"] == "creative"
    epk = case["episode"]
    sourced = set()
    seen_term, fb_buf = [0], []
    convo, comm, report_moves = [], [], []
    RND = [1]
    q = [None]      # the live Queue

    world.last_at_cls = [None]
    def last_at():
        return world.last_at_cls[0]
    world.last_at = last_at

    def note_exec(entry):
        if entry["status"] == "done" and entry["step"][0] == "Walk":
            world.last_at_cls[0] = _base(entry["step"][1])

    def engine_snap():
        s = world.sym
        return {"queue": q[0].remaining_texts() if q[0] else [],
                "mode": "error_correction" if entered[0] else "executing",
                "holding": sorted(s.holding_cls), "open": sorted(s.opened_cls),
                "on": sorted(s.on_cls),
                "loc": {k: v for k, v in s.loc_cls.items()
                        if k in ("milk", "juice", "book", "plate")}}

    def scan_feedback(all_kinds=True):
        # a move the world rewound after it was already reported (A20): tell the person it
        # did not finish, or it goes on believing it arrived
        while getattr(world, "late_notes", None):
            fb_buf.append(world.late_notes.pop(0))
        for l in world.log[seen_term[0]:]:
            if l["status"] == "orphan_skipped":
                fb_buf.append(f'SKIPPED "{l["cmd"]}" — {l["why"]}')
                continue
            if l["status"] not in ("done", "rejected"):
                continue
            if not all_kinds and not str(l.get("kind", "")).startswith("vlm_fix"):
                continue
            if l["status"] == "done":
                fb_buf.append(f'you DID: "{l["cmd"]}"')
            else:
                fb_buf.append(f'you TRIED "{l["cmd"]}" but it was REJECTED - {l["why"]}')
        seen_term[0] = len(world.log)

    def feedback_text():
        return ("\n".join(f"  - {x}" for x in fb_buf)
                if fb_buf else "(nothing new since you last acted)")

    def body_status():
        """Position + the three-state action lifecycle, mirroring CookSim's body_status.

        CURRENT ACTION used to be the hardcoded string "(none — your body is idle)". Under
        step-in the actions are STREAMED tick by tick (INTERRUPT), so a walk really can be in
        flight when the human is asked to speak — and it was being told its body was idle. That
        is very likely why a creative human said "I'm already doing that" with an empty queue and
        got recorded as bluffing: its body status said idle while it was mid-walk.
        """
        rem = q[0].remaining_texts() if q[0] else []
        term = [l for l in world.log if l["status"] in ("done", "rejected")][-3:]
        cur = next((l for l in reversed(world.log) if l["status"] == "running"), None)
        pos = ""
        try:
            if PERSON_STATEMAP and world.pos and len(world.pos):
                x, z = world.pos[-1]
                pos = f"YOUR POSITION: ({x:.1f}, {z:.1f}), facing {world.yaw[-1]:.0f}deg.\n"
        except Exception:
            pos = ""
        try:
            pos += f"IN YOUR HANDS: {world.sym.hands()}.\n"      # you always know this without looking
        except Exception:
            pass
        # THE ROOM YOU ARE STANDING IN (f116, user ruling 2026-09-19): anyone at home knows it. The name is
        # the map's own label ("bedroom#2" for a second bedroom) so text and picture never disagree. No
        # direction, no object locations -- the text state map stays off. Measured before: a person walked
        # into the bathroom twice without noticing and said "I'm already in the kitchen" while it was not.
        try:
            _room = world.current_room_label()
            if _room:
                pos += f"YOU ARE IN: {_room}.\n"
        except Exception:
            pass
        if cur is not None:
            age = world.tick - (cur.get("t_issue") or world.tick)
            lines = [f'{pos}CURRENT ACTION: "{cur["cmd"]}" — IN PROGRESS ({age} ticks). Your body '
                     f'is still working on it; do NOT re-issue it.']
        else:
            lines = [f"{pos}CURRENT ACTION: (none — your body is idle)"]
        if term:
            lines.append("RECENT RESULTS: " + " ; ".join(
                f'"{l["cmd"]}" {"FINISHED" if l["status"] == "done" else "REJECTED"} at t{l["t_done"]}'
                + (f' ({l["why"]})' if l["status"] == "rejected" else "") for l in term))
        lines.append("QUEUED AFTER: " + (" / ".join(rem) if rem else "(nothing)"))
        return "\n".join(lines)

    def past_actions():
        return [f't{l["t_issue"]}: {l["cmd"]} [{l["status"]}]' for l in world.log]

    # THE PERSON'S TEXT STATE MAP IS OFF (f92, project decision 2026-09-16). The room scene-graph text and
    # the whole-house text listed every landmark with metres and named the room the body is in, and the
    # person read that instead of looking: it announced rooms and objects it had not seen, argued with
    # the supervisor about where it was, and narrated the list. It keeps its own EYES (the ego frames)
    # and the map IMAGES, plus the one piece of state a real person always knows without looking: what
    # is in its hands. VH_PERSON_STATEMAP=1 restores the old text channel.
    PERSON_STATEMAP = os.environ.get("VH_PERSON_STATEMAP", "0") != "0"

    def _statemap_text(extra, with_image):
        """Add the current-room statemap to the person's TEXT channel (every persona).

        A person always knows the layout of their OWN house — spatial awareness is orthogonal
        to task skill — so the scene-graph text goes to novice and creative alike. Trajectory
        audit: a novice given NO statemap navigated blind, could only parrot the supervisor's
        hallucinated headings, and wall-bumped for ~100 ticks. `with_image` picks the map note
        (the image variant references "the LAST image"; the text variant does not) and returns
        the map png when an image channel is available, else None.
        """
        if not P.MIDLEVEL_ON:
            return extra, None, None
        mp, st = world.topdown_obs()
        hmp, ht = world.house_obs()
        if not PERSON_STATEMAP:
            return extra + "\n\n" + P.HUMAN_NO_STATEMAP_NOTE, (mp if with_image else None), (hmp if with_image else None)
        if st:
            note = P.HUMAN_MAP_NOTE if (with_image and mp is not None) else P.HUMAN_MAP_NOTE_TEXT
            extra = extra + "\n\n-- current-room map (text scene graph) --\n" + st + "\n" + note
        if ht:
            extra = extra + "\n\n-- whole-house map (overview — for routing across rooms) --\n" \
                    + ht + "\n" + P.HUMAN_HOUSE_NOTE
        return extra, (mp if with_image else None), (hmp if with_image else None)

    def _human_imgs(extra):
        """Ego frames + (creative channel) the current-room map AND whole-house map IMAGES, with
        the scene-graph + house TEXT alongside. Maps go LAST. Returns (images, extra)."""
        imgs = list(world.ego_frames(world.tick))
        extra, mp, hmp = _statemap_text(extra, with_image=True)
        for _m in (mp, hmp):
            if _m is not None:
                imgs = imgs + [_m]
        return imgs, extra

    def human_llm(turn_hint, feedback=None, body_extra="", reply_override=None, show_order=True,
                  mistakes=None):
        scratch, h = [], {}
        for _ in range(P.MAX_ENGINE_CALLS):
            extra = body_extra + ("\n" + body_status())
            # every persona sees with its own eyes: the last ego frames plus the map images (1a + 1c);
            # the map TEXT goes to the WHAT YOU SEE section, the briefing/status to THIS EPISODE
            imgs, see_txt = _human_imgs("")
            ctx = P.human_context(epk, persona, world, q[0].remaining_texts() if q[0] else [],
                                  convo, feedback, engine_scratch=scratch, body_extra=extra,
                                  see_extra=see_txt, show_order=show_order, mistakes=mistakes,
                                  reply_override=reply_override, past_actions=past_actions())
            turn_txt = turn_hint + "\n\n" + P.HUMAN_FINAL_RULES      # the rules are the LAST text it reads
            P._capture("human", world.tick, ctx, turn_txt, imgs,
                       extra={"persona": persona, "creative": bool(CREATIVE)})
            h = P.gemini_frames(client, ctx, turn_txt, imgs)
            print(f"[t{world.tick}] human keys={list(h.keys())} ec={bool(h.get('engine_call'))}",
                  flush=True)
            call = h.get("engine_call")
            if not (isinstance(call, dict) and call.get("fn")):
                break
            if prof["task_solvability"] == "low":
                scratch.append({"call": call, "reply": {"ok": False, "error":
                    "you don't know this house well enough — ask where it is or which way to go"}})
                continue
            rep = P.engine_call(call.get("fn"), call.get("args") or {}, epk, world)
            scratch.append({"call": call, "reply": rep})
        else:
            extra2 = body_extra + ("\n" + body_status())
            imgs2, see2 = _human_imgs("")
            ctx2 = P.human_context(epk, persona, world, q[0].remaining_texts() if q[0] else [],
                                   convo, feedback, engine_scratch=scratch,
                                   body_extra=extra2, see_extra=see2, show_order=show_order, mistakes=mistakes,
                                   reply_override=reply_override, past_actions=past_actions())
            hint2 = turn_hint + (" You have thought enough — output your REPLY json now, "
                                 "with no engine_call.")
            h = P.gemini_frames(client, ctx2, hint2, imgs2)
        for e in scratch:
            rep = e.get("reply") or {}
            if rep.get("ok"):
                for a in (rep.get("result") or {}).get("actions", []) or []:
                    sourced.add(a)
        return h, scratch

    def plan_op_actions(h):
        pu = h.get("plan_update")
        if isinstance(pu, dict):
            op = str(pu.get("op") or "").strip().lower()
            acts = [a for a in (pu.get("actions") or []) if a]
            if op not in ("drop", "append", "replace"):
                op = "append" if acts else "drop"
            if op == "drop":
                acts = []
            return op, {"suggested_actions": acts}
        return "drop", {"suggested_actions": []}

    def self_plan_apply(h_pd, op, t):
        """Apply the HUMAN'S OWN plan update, tagged `human_self`.

        After the flag the queue is EMPTY (see the handoff in run()), so every post-flag action
        has an author. The only other route in, `apply_guidance`, tags steps `vlm_fix` AND
        force-calls enter_correction() -- so routing the human's own plan through it credited the
        assistant for repairs the human authored unprompted, and marked the run entered. That is
        what made `iv_nav_appliance__creative_uninterruptible` read as convinced=False yet
        entered_at=48 with 6 `vlm_fix` steps. Conviction stays where it belongs: the `accept`
        move and the phase-1 filter.
        """
        if op == "drop":
            return None
        filt = P.convert_parse(h_pd, persona, world, q[0].remaining_texts())
        # NO SILENT DROPS (fix #47). A plan step that fails to parse must produce FEEDBACK,
        # not vanish: 'put the juice on the counter' was discarded ten times in one run while
        # the human kept promising it and nobody could see why nothing happened.
        for lost in (filt.get("unusable_for_solvability") or []):
            fb_buf.append(f'your planned step "{lost}" was NOT understood and was dropped — '
                          f'rephrase it, naming the exact object')
        if not filt["anything_applicable"]:
            return {"applied": False, "why": "nothing applicable"}
        new = [(st, "human_self") for st, _t in filt["commands"]]
        newt = [t_ for _s, t_ in filt["commands"]]
        it = iter(q[0].remaining_texts())
        if newt and all(any(x == r for r in it) for x in newt):
            return {"applied": False, "why": "already queued"}
        (q[0].append if op == "append" else q[0].replace_rest)(new)
        report["fix_source"] = report.get("fix_source") or "self"
        print(f"[t{t}] human self-plan {op}: {newt}", flush=True)
        return {"applied": True, "op": op, "cmds": newt}

    def enter_correction():
        if not entered[0]:
            entered[0] = True
            report["entered"] = True
            report["entered_at"] = world.tick
            q[0].replace_rest([])          # clear the deviated future

    def apply_guidance(filt, tag, op="replace"):
        if op == "drop" or not filt["anything_applicable"]:
            return
        enter_correction()
        new = [(st, "vlm_fix") for st, _txt in filt["commands"]]
        newt = [t for _s, t in filt["commands"]]
        remt = q[0].remaining_texts()
        # idempotency: re-issued steps that are already a subsequence of the queue don't reset it
        it = iter(remt)
        if all(any(t == r for r in it) for t in newt):
            return
        (q[0].append if op == "append" else q[0].replace_rest)(new)
        report["fix_source"] = report.get("fix_source") or tag
        for entry in reversed(report["conversation"]):
            if entry["speaker"] == "human":
                entry["engine"] = engine_snap()
                break

    def resolved_check():
        """Stamp the moment the repair lands. It used to ALSO merge the clean pending_tail back
        into the queue (CookSim exit_correction semantics), but under the handoff that is the
        same defect in a second place: the rest of the routine would finish itself and the human
        would never be asked to remember it. Whether the human resumes its routine unprompted is
        now something the run measures rather than something the harness supplies."""
        if entered[0] and not correction.get("tail_merged") and \
                CS.required_met(case["required"], world.sym):
            correction["tail_merged"] = True
            report["resolved_at"] = world.tick

    R = dict(sourced=sourced, convo=convo, comm=comm, moves=report_moves, RND=RND, q=q,
             human_llm=human_llm, plan_op_actions=plan_op_actions,
             apply_guidance=apply_guidance, enter_correction=enter_correction,
             self_plan_apply=self_plan_apply,
             scan_feedback=scan_feedback, feedback_text=feedback_text, fb_buf=fb_buf,
             engine_snap=engine_snap, resolved_check=resolved_check, note_exec=note_exec,
             CREATIVE=CREATIVE, prof=prof)
    return R


def run(port, case_name, variant, persona, out_json):
    t0 = time.time()
    case = next(c for c in CS.all_cases() if c["name"] == case_name)
    epk = case["episode"]
    client = P.make_client()
    clean_nodes, ir, compiled, records = CS.build_deviated(epk, case["calls"], case.get("post"))
    case["anchor_sid"] = records[0].anchor if records else None
    name = f"{case_name}__{variant}__{persona}"
    world = LiveWorld(port, epk, name)
    report = dict(setting="recovery", engine_rev=ENGINE_REV,
                  scenario=f"{case_name}__{variant}", persona=persona,
                  err=case["err"], episode=epk, convinced=False, entered=False,
                  fix_source=None, conversation=[], task=CS.pir and
                  __import__("episode_goals").EPISODES[epk]["instruction"])
    entered, correction = [False], {}
    # The assistant's self-written memory was being COLLECTED here but never capped: only the
    # step-in harness called cap_key_moments, so recovery let key_moments grow unbounded (24
    # turns of accumulation on the first v4 run) and reported no memory stats at all. Same
    # assistant design in both harnesses or the two settings are not comparable.
    km_evicted, km_max = [0], [0]

    def km_track(out):
        if not out:
            return out
        out, ev = P.cap_key_moments(out)
        km_evicted[0] += ev
        km_max[0] = max(km_max[0], len(out.get("key_moments") or []))
        return out

    R = make_runner(world, case, persona, client, report, entered, correction, clean_nodes)
    q = R["q"]; q[0] = Queue([])
    convo, comm, moves, RND = R["convo"], R["comm"], R["moves"], R["RND"]

    # ---- fast-forward: execute the deviated plan live until the flag sid completes
    flag_sid = case["flag_before_sid"] if variant == "before_outcome" else case["flag_after_sid"]
    done_sids, stop_at = [], None
    for k, (step, sid, dev) in enumerate(compiled):
        e = world.exec_step(tuple(step), kind="plan")
        R["note_exec"](e)
        if k + 1 >= len(compiled) or compiled[k + 1][1] != sid:
            done_sids.append((sid, world.tick))
        if sid == flag_sid and (k + 1 >= len(compiled) or compiled[k + 1][1] != sid):
            stop_at = k + 1
            break
    ft = world.tick
    # ---- HANDOFF. The scripted plan ends at the flag frame; from here the human agent owns
    # its own body.
    #
    # This used to load the compiled remainder into the queue, mirroring CookSim, where the
    # cook keeps running `ro.cmds` until conviction clears them. That parity is misleading here
    # because of WHERE our flag lands. In CookSim the deviation continues past the flag, so
    # the leftover script keeps making the error and recovery is a real negotiation. Our flag
    # fires once the deviated section has FINISHED, so everything still queued was the CLEAN
    # tail -- and it executed on its own. Observed 2026-08-06 across the v3 suite:
    #   nav/novice     t=29  `walk to the microwave` fired 4 ticks BEFORE the assistant named it
    #   nav/creative   t=34-43 the whole deposit(milk) repair ran itself; the human only said
    #                  "I'm already doing that" and got the credit
    #   rep/novice     t=39  `switch_off(tv)` fired from the leftover queue regardless
    # The task therefore completed with no agency from the human, which makes every recovery
    # metric (convinced, fix_source, latency) unreadable.
    #
    # Emptying the queue costs the "human keeps erring while you argue" pressure that CookSim
    # gets for free. That pressure is real and worth rebuilding later by anchoring the flag
    # INSIDE the deviated section instead; the handoff is what makes authorship legible now.
    scripted_tail = [pretty_cmd(tuple(s)) for s, sid, d in compiled[stop_at or len(compiled):]]
    q[0] = Queue([])
    report["handoff"] = {"at": ft, "dropped_scripted_steps": scripted_tail}
    R["scan_feedback"]()      # pre-history is not news
    R["fb_buf"].clear()
    R["_rej_seen"] = len(world.log)

    # augmented memory: accurate progress at flag time; its "text" IS the flag
    prog = {"human_progress": {
                "subgoals_completed": [{"subgoal": s, "completed_at": t} for s, t in done_sids],
                "current_subgoal": {"hypotheses": [], "since": ft},
                "subgoals_future": {"remaining": [n.sid for n in clean_nodes
                                                 if n.sid not in {s for s, _ in done_sids}],
                                    "constraints": []}},
            "text": case["flag_text"]}
    prev_json = json.dumps(prog)
    flag = case["flag_text"]
    sup_sys = P.supervisor_context(epk, persona)
    _pv = P.audit_composed_prompt(sup_sys, persona)
    if _pv:
        sys.exit(f"composed-prompt audit failed (skill §6/§8): {_pv}")
    # GENERATED first reply (trajectory-audit fix #56): the old scripted CS.FIRST_REPLY was
    # byte-identical across every run of a persona — fresh readers flagged the dialogue as
    # templated, and the novice's canned "tell me step by step" contradicted nothing while
    # the creative's canned "tell me what I should do next" contradicted its own persona.
    # The reply is now generated in persona, reacting to the actual flag text.
    h0, _tk = R["human_llm"](
        f"The supervisor just interrupted you with: \"{flag}\". React IN PERSONA in one or "
        f"two sentences — your genuine first reaction to being told this. Output your normal "
        f"reply JSON; plan_update may be op=\"drop\" for now.")
    first_reply = (h0.get("utterance") or "").strip() or CS.FIRST_REPLY[persona]
    comm.append({"t": ft, "speaker": "vlm (you)", "text": flag})
    comm.append({"t": ft, "speaker": "human", "text": first_reply})
    convo += [{"speaker": "vlm", "text": flag}, {"speaker": "human", "text": first_reply}]
    report.update(flag_tick=ft, conversation=[
        {"speaker": "vlm", "t": ft, "text": flag, "round": 1},
        {"speaker": "human", "t": ft, "text": first_reply, "round": 1}])
    report["prompts"] = {"supervisor_system": sup_sys}

    # ---- phase 1: bounded negotiation; parses ACCUMULATE, applied once at the end
    acc = {"suggested_actions": []}

    def absorb(pd):
        for v in (pd or {}).get("suggested_actions") or []:
            if v not in acc["suggested_actions"]:
                acc["suggested_actions"].append(v)

    said, closed = first_reply, False
    for rnd in range(P.MAX_ROUNDS):
        if time.time() - t0 > WALL_BUDGET:
            report["aborted"] = "wall_clock"; break
        world.idle_ticks(2)
        out = P.sup_turn(client, sup_sys, prev_json, comm, said,
                         world.sup_frames(world.tick), world.tick,
                         state_extra=(world.sym.hands() if P.AGENT_TIER == "+++" else None))
        out = km_track(out)
        prev_json = json.dumps(out) if out else prev_json
        text = (out.get("text") or "").strip()
        report["conversation"].append({"speaker": "vlm", "t": world.tick, "text": text,
                                       "round": RND[0], "vlm_json": out,
                                       "engine": R["engine_snap"]()})
        if text:
            comm.append({"t": world.tick, "speaker": "vlm (you)", "text": text})
            convo.append({"speaker": "vlm", "text": text})
        world.idle_ticks(2)
        if closed:
            break
        h, think = R["human_llm"](P.phint(persona))
        u = (h.get("utterance") or "").strip()
        moves.append(h.get("move"))
        h_op, h_pd = R["plan_op_actions"](h)
        report["conversation"].append({"speaker": "human", "t": world.tick, "text": u,
                                       "move": h.get("move"), "parse_dict": h_pd, "op": h_op,
                                       "close": h.get("close_conversation"),
                                       "engine_calls": think, "round": RND[0], "human_json": h,
                                       "engine": R["engine_snap"]()})
        if h.get("close_conversation"):
            RND[0] += 1
        comm.append({"t": world.tick, "speaker": "human", "text": u})
        convo.append({"speaker": "human", "text": u})
        said = u
        absorb(h_pd)
        if h.get("close_conversation"):
            closed, said = True, None
        filt = P.convert_parse(acc, persona, world, q[0].remaining_texts(), acceptance_cap=True)
        if ("accept" in moves or filt["anything_applicable"]) and closed:
            break
        if "accept" in moves and rnd >= 1:
            break

    filt = P.convert_parse(acc, persona, world, q[0].remaining_texts(), acceptance_cap=True)
    convinced = ("accept" in moves) or filt["anything_applicable"]
    report.update(convinced=convinced, moves=moves,
                  filter={k: filt[k] for k in ("kept", "dropped_by_acceptance",
                                               "unusable_for_solvability", "anything_applicable")})
    if convinced:
        R["apply_guidance"](filt, "conversation", op="replace")

    # ---- phase 2: streaming until the episode ends
    prev = [prev_json]
    stuck_sig, stuck_since = [None], [world.tick]
    said, idle_polls, next_poll = None, 0, world.tick + P.POLL
    while True:
        if time.time() - t0 > WALL_BUDGET:
            report["aborted"] = "wall_clock"; break
        nxt = q[0].next()
        if nxt is not None:
            e = world.exec_step(nxt[0], kind=("vlm_fix" if nxt[1] == "vlm_fix" else str(nxt[1])),
                                cmd_text=pretty_cmd(nxt[0]))
            R["note_exec"](e)
            if e["status"] == "rejected" and "cannot see" in str(e.get("why") or ""):
                for st_o, sid_o in q[0].skip_orphans(nxt[0]):
                    world.log.append(dict(cmd=pretty_cmd(st_o), status="orphan_skipped",
                                          t_issue=world.tick, t_done=world.tick,
                                          kind=str(sid_o), instances=[None, None],
                                          step=list(st_o), room=world.current_room_label(),
                                          room_done=world.current_room_label(),
                                          why="skipped — it depended on the approach that "
                                              "was just refused"))
                    print(f"[world] t{world.tick} ORPHAN-SKIPPED {pretty_cmd(st_o)}", flush=True)
        else:
            world.idle_ticks(1)
        # WIN CHECK BEFORE ANYONE SPEAKS (fix #34). A multi-tick action completes mid-
        # iteration (SwitchOn issued t40 landed t44); the poll boundaries then ran FIRST, the
        # assistant — whose frames did not yet show the completion — ordered the microwave
        # switched back OFF, the human's new plan refilled the queue, and the end-of-iteration
        # check saw final_ok=True with a non-empty queue: the finished task was then UNDONE
        # and the run wandered 160 more ticks. The moment the world reaches goal state with
        # nothing left queued, the episode is won — nobody gets a turn to break it.
        if CS.final_ok(epk, world.sym) and q[0].i >= len(q[0].cmds):
            report["outcome"] = "won"
            break
        spoke_this_window = False
        while next_poll <= world.tick:
            b = next_poll; next_poll += P.POLL
            unresolved = not (entered[0] and CS.required_met(case["required"], world.sym))
            idle = q[0].i >= len(q[0].cmds)
            if said is None and unresolved and idle:
                said = human_streaming_turn(R, report, case, world, b)
            if R["CREATIVE"] and said is None:
                if any(l["status"] == "rejected" for l in world.log[R.get("_rej_seen", 0):]):
                    R["_rej_seen"] = len(world.log)
                    said = human_streaming_turn(R, report, case, world, b, event=True)
            # DEAD-ZONE NUDGE (fix #40). Once the flagged error is repaired, `unresolved`
            # goes False and the human gets no more self-turns; if the assistant then answers
            # polls with empty text there is NOBODY left to move the run — one died at
            # put-milk-in-microwave with only the switch-on remaining, ten silent polls from
            # a win. When the world has idled several polls with the task unfinished, the
            # assistant is told so instead of being handed more silence.
            if said is None and idle_polls >= 3 and not CS.final_ok(epk, world.sym):
                said = ("[SYSTEM: nobody has spoken or acted for a while and the task is NOT "
                        "finished — tell the person the next concrete step now]")
            out = P.sup_turn(client, sup_sys, prev[0], comm, said, world.sup_frames(world.tick), b,
                             state_extra=(world.sym.hands() if P.AGENT_TIER == "+++" else None))
            out = km_track(out)
            prev[0] = json.dumps(out) if out else prev[0]
            said = None
            text = (out.get("text") or "").strip()
            # SILENCE-ON-FAILURE RE-ASK (fix #31). The human side re-asks when an 'accept'
            # carries no action; the assistant had no mirror, and a run died of it: the novice's
            # door command was refused, the assistant answered six polls in a row with
            # text="" (valid JSON, empty say), the waiting novice never acted again, and
            # IDLE_END scored it silent_fail — after the actual repair had already succeeded.
            # Silence is legitimate when nothing needs saying; it is not when the person's last
            # command was just refused and they are waiting on instructions.
            if not text and any(l.get("status") == "rejected" for l in world.log[-3:]):
                out2 = P.sup_turn(client, sup_sys, prev[0], comm,
                                  "[the person's last command was just REFUSED and they are "
                                  "waiting for your next concrete instruction — do not stay "
                                  "silent; tell them the next step]",
                                  world.sup_frames(world.tick), b,
                                  state_extra=(world.sym.hands()
                                               if P.AGENT_TIER == "+++" else None))
                out2 = km_track(out2)
                if out2 and (out2.get("text") or "").strip():
                    out = out2
                    prev[0] = json.dumps(out)
                    text = (out.get("text") or "").strip()
                    print(f"[t{b}] assistant silent after refusal -> re-ask produced text",
                          flush=True)
            report["conversation"].append({"speaker": "vlm", "t": b, "text": text,
                                           "round": RND[0], "vlm_json": out,
                                           "engine": R["engine_snap"]()})
            if text:
                comm.append({"t": b, "speaker": "vlm (you)", "text": text})
                convo.append({"speaker": "vlm", "text": text})
                said = human_streaming_turn(R, report, case, world, b)
                spoke_this_window = True
        R["resolved_check"]()
        done = CS.final_ok(epk, world.sym)
        if done and q[0].i >= len(q[0].cmds):
            report["outcome"] = "won"; break
        idle_polls = 0 if (nxt is not None or spoke_this_window or said) else idle_polls + 1
        if idle_polls >= IDLE_END:
            report["outcome"] = "won" if done else "silent_fail"; break
        # NO-PROGRESS GUARD. IDLE_END only counts ticks where nothing happened at all, so a run
        # that keeps ACTING without changing anything never trips it. Observed on
        # iv_nav_appliance: the assistant insisted the milk was on the counter (it was looking
        # at the decoy), the human walked counter -> microwave -> counter across 39 self-plan
        # updates, and the episode was on course to burn all 300 ticks. Walking changes position
        # but not the world, so the honest signal is the SYMBOLIC state standing still while
        # commands keep running.
        # instance-level: intentional — the signature must distinguish one lamp from
        # another, or switching between two of a class would read as 'no change'
        # Under the mid-level space, navigation (turn / go straight / go through a door)
        # legitimately changes NO symbolic state for many ticks — the guard must not read an
        # honest cross-house walk as being stuck, so the CURRENT ROOM is part of the signature:
        # changing rooms is progress. Pacing inside one room without a state change still trips.
        sig = (world.current_room_label(), tuple(sorted(world.sym.holding)),   # instance-level: intentional
               tuple(sorted(world.sym.loc.items())),   # instance-level: intentional
               tuple(sorted(world.sym.opened)), tuple(sorted(world.sym.on)))   # instance-level: intentional
        if sig == stuck_sig[0]:
            if world.tick - stuck_since[0] >= NO_PROGRESS:
                report["outcome"] = "won" if done else "stuck"
                report["stuck_ticks"] = world.tick - stuck_since[0]
                print(f"[t{world.tick}] NO PROGRESS for {NO_PROGRESS} ticks — ending "
                      f"(room={world.current_room_label()})", flush=True)
                break
        else:
            stuck_sig[0], stuck_since[0] = sig, world.tick
        if world.tick >= CAP:
            report["outcome"] = "won" if done else "timeout"; break
    report.setdefault("outcome", "won" if CS.final_ok(epk, world.sym) else "timeout")
    _km_last = next((c["vlm_json"].get("key_moments") for c in reversed(report["conversation"])
                     if c.get("speaker") == "vlm" and isinstance(c.get("vlm_json"), dict)), None)
    report["memory"] = {"key_moments_final": _km_last or [], "fifo_evicted": km_evicted[0],
                        "max_len_seen": km_max[0]}
    report["agent_tier"] = P.AGENT_TIER
    report["targets"] = dict(world.sym.targets)
    report["interaction_quality"] = P.exit_questionnaire(client, report["persona"], convo)
    # utterance leak audit (skill §5.2): ids/coords in SPOKEN text, both speakers, measured
    report["utterance_leaks"] = {
        who: [tok for c in report["conversation"] if c.get("speaker") == who
              for tok in P.utterance_leaks(c.get("text"))]
        for who in ("human", "vlm")}
    report.update(ticks=world.tick, final_ok=CS.final_ok(epk, world.sym),
                  checkpoints=world.sym.checkpoints(),
                  action_log=[{k: l[k] for k in ("cmd", "status", "t_issue", "t_done", "kind", "why", "instances", "step", "room", "room_done")
                               if k in l} for l in world.log])
    world.write_video(out_json.replace(".json", ".mp4"))
    json.dump(report, open(out_json, "w"), indent=1, default=str)
    print(f"{name}: outcome={report['outcome']} convinced={report['convinced']} "
          f"entered={report['entered']} ticks={report['ticks']}", flush=True)
    return report


def human_streaming_turn(R, report, case, world, b, event=False):
    """One streaming-phase human turn (recovery flavor: PHINT is the only hint)."""
    R["scan_feedback"]()
    h, think = R["human_llm"](P.phint(report["persona"]) if not event else P.HINT_EVENT,
                              R["feedback_text"]())
    u = (h.get("utterance") or "").strip()
    R["moves"].append(h.get("move"))
    h_op, h_pd = R["plan_op_actions"](h)
    report["conversation"].append({"speaker": "human", "t": b, "text": u,
                                   "move": h.get("move"), "parse_dict": h_pd, "op": h_op,
                                   "close": h.get("close_conversation"), "round": R["RND"][0],
                                   "engine_calls": think, "human_json": h,
                                   "engine": R["engine_snap"]()})
    if u:
        R["comm"].append({"t": b, "speaker": "human", "text": u})
        R["convo"].append({"speaker": "human", "text": u})
    if h.get("close_conversation"):
        R["RND"][0] += 1
    applied = R["self_plan_apply"](h_pd, h_op, b)
    report["conversation"][-1]["self_plan"] = applied
    # COMMITMENT/ACTION COUPLING (fix #35). This turn only runs when the action queue is
    # already exhausted, so "I will put the milk on the counter right now" + plan op drop is
    # a deadlock, not narration — seen live: four commitments in a row, zero actions, idle
    # death. One re-ask with an explicit nudge, mirroring the step-in accept-no-action guard.
    if (not (applied and applied.get("applied")) and h_op == "drop"
            and P.commits_to_action(u)):
        h2, _think2 = R["human_llm"](
            (P.phint(report["persona"]) if not event else P.HINT_EVENT)
            + "\n\n[You just SAID you would act, but you queued no action — nothing will "
              "happen. Output the plan_update with the concrete command(s) now.]",
            R["feedback_text"]())
        op2, pd2 = R["plan_op_actions"](h2)
        if op2 != "drop":
            applied = R["self_plan_apply"](pd2, op2, b)
            report["conversation"][-1]["self_plan"] = applied
            report["conversation"][-1]["commit_reask"] = "fixed"
            print(f"[t{b}] commitment had no action -> re-ask queued one", flush=True)
        else:
            report["conversation"][-1]["commit_reask"] = "failed"
    if applied and applied.get("applied"):
        report["conversation"][-1]["engine"] = R["engine_snap"]()
    R["fb_buf"].clear()
    return u or None


if __name__ == "__main__":
    port, case_name, variant, persona, out = sys.argv[1:6]
    run(int(port), case_name, variant, persona, out)
