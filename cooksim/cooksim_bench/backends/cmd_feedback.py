#!/usr/bin/env python3
"""Per-command ENGINE FEEDBACK for the streaming controller.

Instead of dumping the whole command history, we keep only the last K commands, each
annotated with the engine's verdict so the VLM knows what actually happened:

  1. "command received and executed"                    -> accepted; just completed ok
  2. "continue executing last command: <cmd>"           -> the last command is still in progress
  3. "command refused because: <reason>"                 -> could not run; engine gives the reason
  4. "grounding needed: which <instance>? regenerate ..."-> several instances exist; name one
  5. "no command to execute ..."                         -> idle; last command already completed

The tracker is engine-driven and has NO dependence on the VLM/browser, so it is fully
unit-testable (see cooksim_bench/test_feedback.py).
"""
import re
from cooksim.agents.highlevel import parse as hl_parse, _validate

# verbs that act on the FACED tile (names as produced by highlevel.parse)
MANIP = {"pickup", "putdown", "chop", "start", "scoop", "serve", "wash", "trash"}


def _instance_word(err: str) -> str:
    m = re.search(r"(pan|pot|oven|board)s?\b", err or "")
    return m.group(1) if m else "station"


class FeedbackTracker:
    """Tracks the VLM's recent commands + the engine's per-command feedback."""

    def __init__(self, k: int = 5):
        self.k = k
        self.log = []          # each: {cmd, fb, status}
        self.active = None     # index of the command whose skill is currently running

    # ---- update the in-flight / last command's feedback from controller state ----
    def refresh(self, ctrl) -> None:
        a = self.active
        if a is not None and a < len(self.log):          # a command is/was in flight
            if ctrl.skill is not None:                   # 2) still running
                self.log[a]["fb"] = f'continue executing last command: "{self.log[a]["cmd"]}" (in progress)'
                self.log[a]["status"] = "running"
            elif ctrl.status == "done":                  # 1) just completed ok
                self.log[a]["fb"] = "command received and executed"
                self.log[a]["status"] = "executed"; self.active = None
            elif ctrl.status == "failed":                # 3) failed mid-execution
                self.log[a]["fb"] = f"command refused because: {ctrl.error}"
                self.log[a]["status"] = "failed"; self.active = None
            return                                       # show 1/2/3 THIS turn; type-5 only once idle
        # idle (no command running) and the last one already completed -> 5) nothing to do
        if self.log and self.log[-1]["status"] == "executed":
            self.log[-1]["fb"] = (f'no command to execute — last command "{self.log[-1]["cmd"]}" already '
                                  f'completed; the cook is idle and ready for your next command')
            self.log[-1]["status"] = "idle"

    # ---- evaluate + (maybe) issue a NEW command; return (executed, feedback, status) ----
    def submit(self, cmd, game, ctrl, strict=True):
        # strict=False (the 'basic' condition, which gets no feedback to reground with) ->
        # ambiguous commands fall back to nearest instead of being rejected.
        it = hl_parse(cmd)
        verb = it.verb if it else None
        # 1) manipulation legality up front -> precise refusal reason
        if verb in MANIP:
            ok, why = _validate(verb, game, game.cooks[0])
            if not ok:
                return self._add(cmd, f"command refused because: {why}", "refused")
        # 2) try to issue (goto grounds here; ambiguity is rejected only under strict)
        accepted = ctrl.issue(cmd, strict=strict)
        if accepted:
            self.active = len(self.log)
            return self._add(cmd, "command received and executing", "executing")
        # 3) rejected -> grounding vs other refusal
        err = ctrl.error or "could not run this command"
        if "AMBIGUOUS" in err:
            inst = _instance_word(err)
            fb = (f"grounding needed: which instance of {inst}? regenerate command naming the specific one "
                  f"(e.g. \"the left {inst}\", \"the right {inst}\", \"the {inst} 2\", or \"the {inst} with the <ingredient>\")")
            return self._add(cmd, fb, "grounding")
        return self._add(cmd, f"command refused because: {err}", "refused")

    def _add(self, cmd, fb, status):
        self.log.append({"cmd": cmd, "fb": fb, "status": status})
        return (status == "executing"), fb, status

    # ---- the block shown to the VLM ----
    # level: "none" (basic: just commands) | "minimal" (agent: accepted/rejected, no reason)
    #        | "full" (agent+/++/+++: the detailed engine feedback).  with_feedback kept for back-compat.
    def block(self, level: str = "full", with_feedback=None) -> str:
        if with_feedback is not None:
            level = "full" if with_feedback else "none"
        recent = self.log[-self.k:]
        if not recent:
            return "YOUR RECENT COMMANDS: (none yet — this is your first command)"
        if level == "none":
            lines = "\n".join(f'  {i}. "{r["cmd"]}"' for i, r in enumerate(recent, 1))
            return f"YOUR LAST {len(recent)} COMMANDS (oldest first):\n{lines}"
        if level == "minimal":
            acc = lambda s: "accepted" if s in ("executing", "executed", "idle", "running") else "rejected"
            lines = "\n".join(f'  {i}. "{r["cmd"]}"  ->  command {acc(r["status"])}' for i, r in enumerate(recent, 1))
            return (f"YOUR LAST {len(recent)} COMMANDS AND WHETHER THE ENGINE ACCEPTED THEM. The engine tells you ONLY "
                    f"\"accepted\" or \"rejected\" — NO reason is given, and a command you let continue (reply \"\") gets "
                    f"NO report. \"accepted\" = it ran; \"rejected\" = it could not run, so re-issue it differently "
                    f"(you must work out how yourself):\n{lines}")
        lines = "\n".join(f'  {i}. "{r["cmd"]}"  ->  {r["fb"]}' for i, r in enumerate(recent, 1))
        return (f"YOUR LAST {len(recent)} COMMANDS (oldest first) AND THE ENGINE'S FEEDBACK ON EACH — ACT ON IT:\n"
                f"  · \"grounding needed\" -> re-issue that command naming the instance\n"
                f"  · \"continue executing\" -> the cook is still busy; reply \"\" and wait\n"
                f"  · \"command refused\" -> fix the stated cause, then re-issue\n"
                f"  · \"no command to execute\" / \"executed\" -> issue the NEXT step\n{lines}")
