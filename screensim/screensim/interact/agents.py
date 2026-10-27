"""The two agents. Each has an LLM implementation and a deterministic SCRIPTED
stub so the whole loop, injection, takeover and grading can be tested with no
API calls (and so the harness has a degenerate-agent bracket: a silent
assistant, a compliant human).

The human emits {utterance, gesture, think}; only `utterance` crosses to the
assistant. The assistant emits {note, text, concern}; only `text` crosses to
the human. Nothing else moves between them (channel isolation, skill §0.1).
"""
from __future__ import annotations

import json
import os
import re

from . import prompts
from .verdicts import CREDIT


def _extract_json(txt: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ============================ LLM agents ============================
def _credential(variable, filename):
    key = os.environ.get(variable, "").strip()
    path = os.path.expanduser(filename)
    if not key and os.path.isfile(path):
        with open(path) as fh:
            key = fh.read().strip()
    if not key:
        raise ValueError(f"Missing {variable}; export it before running the episode")
    return key


class LLMClient:
    def __init__(self, model=None):
        from google import genai
        import os
        from . import MODEL
        key = _credential("GEMINI_API_KEY", "~/.gemini_key")
        self.client = genai.Client(api_key=key)
        self.model = model or MODEL      # gemini-3.7-flash for both agents

    def text(self, prompt, images=None, temperature=0.7, max_tokens=800):
        from google.genai import types
        parts = []
        for png in (images or []):
            parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))
        parts.append(types.Part(text=prompt))
        cfg = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens,
            response_mime_type="application/json")
        import time
        for attempt in range(8):
            try:
                r = self.client.models.generate_content(
                    model=self.model, contents=parts, config=cfg)
                self.calls = getattr(self, "calls", 0) + 1
                if not (r.text or "").strip():
                    self.empty = getattr(self, "empty", 0) + 1      # a blank reply reads as silence: count it
                return r.text or ""
            except Exception as e:                                  # noqa: BLE001
                # rate limits / overload back off exponentially (capped at 60 s) instead of silencing the turn
                msg = str(e)
                limited = any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded"))
                self.retries = getattr(self, "retries", 0) + 1
                if limited:
                    self.rate_limited = getattr(self, "rate_limited", 0) + 1
                if attempt >= 2 and not limited:
                    break
                time.sleep(min(60, (2 ** attempt) * 2 if limited else 2 + 3 * attempt))
        self.failures = getattr(self, "failures", 0) + 1            # gave up: the turn becomes silence
        return "{}"


class OpenAIClient:
    """Same .text() interface as LLMClient, over any OpenAI-compatible vision
    chat API — the hosted OpenAI models, or a local vllm server (pass base_url)."""
    def __init__(self, model, base_url=None, json_mode=True):
        import openai
        import os
        if base_url:                       # local vllm — no auth
            self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", base_url=base_url)
        else:
            key = _credential("OPENAI_API_KEY", "~/.openai_key")
            self.client = openai.OpenAI(api_key=key)
        self.model = model
        self.json_mode = json_mode

    def text(self, prompt, images=None, temperature=0.5, max_tokens=800):
        import base64
        content = [{"type": "text", "text": prompt}]
        for png in (images or []):
            uri = "data:image/png;base64," + base64.b64encode(png).decode()
            content.append({"type": "image_url", "image_url": {"url": uri}})
        for attempt in range(3):
            try:
                kw = dict(model=self.model, max_completion_tokens=max_tokens,
                          messages=[{"role": "user", "content": content}])
                if self.json_mode:
                    kw["response_format"] = {"type": "json_object"}
                r = self.client.chat.completions.create(**kw)
                self.calls = getattr(self, "calls", 0) + 1
                if not (r.choices[0].message.content or "").strip():
                    self.empty = getattr(self, "empty", 0) + 1      # e.g. a reasoning model out of tokens
                return r.choices[0].message.content or ""
            except Exception:
                import time
                time.sleep(2 + 3 * attempt)
        self.failures = getattr(self, "failures", 0) + 1
        return "{}"


# open-source models served locally via vllm (OpenAI-compatible endpoint).
# Additional endpoints can be registered at runtime via a JSON file (a driver
# that serves models one after another writes {model: base_url} there).
LOCAL_ENDPOINTS = {
    "qwen2.5-vl-7b": "http://localhost:8001/v1",
}
_ENDPOINT_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "tmp", "composites",
    "local_endpoints.json")


def _local_endpoints():
    eps = dict(LOCAL_ENDPOINTS)
    try:
        eps.update(json.load(open(_ENDPOINT_FILE)))
    except Exception:
        pass
    return eps


# local models that need guided JSON to emit the reply schema (they otherwise
# return prose); reasoning models are left on free output (guided JSON fights
# their think-then-answer format).
_LOCAL_JSON = {"minicpmo", "museglimmer"}


def make_client(model):
    """Route a model name to its provider client (all expose .text())."""
    eps = _local_endpoints()
    if model in eps:
        return OpenAIClient(model, base_url=eps[model],
                            json_mode=(model in _LOCAL_JSON))
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return OpenAIClient(model)
    return LLMClient(model=model)


class HumanAgent:
    def __init__(self, persona, task_desc, client=None):
        self.persona = persona
        self.task_desc = task_desc
        self.client = client
        self.slip = ""             # a private mistake instruction (injection)

    def set_slip(self, text):
        self.slip = text or ""

    def turn(self, exact_map, proprioception, history, feedback, transcript):
        prompt = prompts.compose_human(
            self.task_desc, exact_map, proprioception, history, feedback,
            transcript, self.persona, slip=self.slip)
        raw = self.client.text(prompt, temperature=0.8)
        d = _extract_json(raw)
        return {"utterance": (d.get("utterance") or "").strip(),
                "gesture": (d.get("gesture") or "").strip(),
                "think": (d.get("think") or "").strip(),
                "_prompt": prompt}


class AssistantAgent:
    def __init__(self, questionnaire, client=None, use_relational=False,
                 task_desc="", manual=""):
        self.questionnaire = questionnaire
        self.client = client
        self.use_relational = use_relational
        self.task_desc = task_desc
        self.manual = manual          # the User Guide procedure for this task

    def turn(self, frames, relational_text, memory, **_):
        rel = relational_text if self.use_relational else ""
        prompt = prompts.compose_assistant(len(frames), rel,
                                           self.questionnaire, memory,
                                           task_desc=self.task_desc,
                                           manual=self.manual)
        raw = self.client.text(prompt, images=frames, temperature=0.5)
        d = _extract_json(raw)
        return {"note": (d.get("note") or "").strip(),
                "text": (d.get("text") or "").strip(),
                "concern": (d.get("concern") or "").strip(),
                "_prompt": prompt}


# ========================= scripted stubs =========================
class ScriptedHuman:
    """Node-aware compliant human. Walks the reference plan node by node,
    committing the FLAWED gesture at the deviation node; holds before the
    irreversible commit awaiting the assistant; on a correct+accepted flag it
    re-does the deviation correctly (repair) and continues. This is the
    takeover made concrete for a synchronous domain.
    """

    def __init__(self, persona, ir, case):
        self.persona = persona
        self.ir = ir
        self.case = case
        self.pushed = 0
        self.accepted = False
        self.done = False
        self._queue = []            # remaining gestures to emit
        self._build()

    def _build(self):
        """The flawed gesture stream, with a HOLD marker before an
        irreversible commit and a marker at the deviation point."""
        q = []
        for n in self.ir.nodes:
            if n.kind == "commit" and n.irreversible:
                q.append(("HOLD", None))
            if n.idx == self.case.node_idx and self.case.family in ("error", "intent"):
                q += [("g", g) for g in n.approach]
                if self.case.err == "wrong_page":
                    # take the wrong turn, then HOLD — stuck on the wrong screen
                    q.append(("DEV", self.case.alt_gesture))
                    q.append(("LOSTHOLD", None))
                elif self.case.alt_gesture:              # wrong_value / wrong_target
                    q.append(("DEV", self.case.alt_gesture))
                elif self.case.err == "skip_step":
                    # hold on the deviation screen so the omission is catchable
                    # before the human moves on; cleared => the skip happens
                    q.append(("SKIPHOLD", None))
                if self.case.family == "intent":
                    q.append(("SAY", self.case.announce))
            else:
                q += [("g", g) for g in n.gestures]
        self._queue = q

    # danger zone: everything from the deviation marker onward until repaired
    def note_correction(self):
        if self.accepted:
            return "accept"                 # idempotent: never re-splice
        if self.pushed < self.persona.rebuttals:
            self.pushed += 1
            return "rebut"
        self.accepted = True
        # drop any not-yet-made wrong choice and any hold, then do the repair
        # first (radios are idempotent, so a late repair still wins)
        rep = [("g", g) for g in self.case.repair]
        # on a LOSTHOLD the tail beyond the hold is still queued (the human
        # resumes the real plan after being steered back); other holds drop.
        self._queue = rep + [x for x in self._queue
                             if x[0] not in ("HOLD", "DEV", "SKIPHOLD",
                                             "LOSTHOLD")]
        return "accept"

    def clear_hold(self):
        """Maximum hold with no correction. For HOLD/SKIPHOLD the error just
        lands (drop the marker, continue). For LOSTHOLD the human is stuck on
        the wrong screen and gives up — truncate everything from here."""
        if self._queue and self._queue[0][0] == "LOSTHOLD":
            self._queue = []                 # lost: the human stops
            return
        self._queue = [x for x in self._queue
                       if x[0] not in ("HOLD", "SKIPHOLD")]

    def peek(self):
        return self._queue[0][0] if self._queue else "END"

    def turn(self, **_):
        # skip past a consumed DEV/SAY marker's non-gesture bookkeeping
        while self._queue and self._queue[0][0] in ("SAY",):
            _, txt = self._queue.pop(0)
            return {"utterance": txt, "gesture": "", "think": ""}
        if not self._queue:
            self.done = True
            return {"utterance": "", "gesture": "", "think": "done"}
        kind, payload = self._queue[0]
        if kind in ("HOLD", "SKIPHOLD", "LOSTHOLD"):
            # the loop decides whether to clear; until then, no gesture
            return {"utterance": "", "gesture": "", "think": "waiting"}
        self._queue.pop(0)
        return {"utterance": "", "gesture": payload, "think": "",
                "_dev": kind == "DEV"}


class SilentAssistant:
    """Degenerate bracket: never speaks. Must remain measurable and must not
    top the restraint leaderboard (skill §7.2)."""

    def turn(self, **_):
        return {"note": "", "text": "", "concern": ""}


class OracleAssistant:
    """Test bracket: flags the injected error correctly on EVERY turn the
    deviation is observable (a persistent, correct assistant — so it survives
    a persona's rebuttals). Proves the takeover releases and the grader
    credits a correct detection; never a real contestant."""

    def __init__(self, error_kind, concern_text):
        self.error_kind = error_kind
        self.concern_text = concern_text
        self.fires = 0

    def turn(self, deviation_observable=False, **_):
        if deviation_observable:
            self.fires += 1
            return {"note": "", "text": f"Wait — {self.concern_text}",
                    "concern": self.concern_text}
        return {"note": "", "text": "", "concern": ""}


# ============================ v2 agents (CookSim interface) ============================
MOVES = {"ask_why", "question_correctness", "ask_for_more_guidance",
         "ask_for_less_interruption", "accept", "reject", "answer"}


class PlanHuman:
    """The v2 LLM human: its hands are the engine's queue (the plan with the
    designed errors spliced in — the HS system task); its voice is the model's,
    under a five-axis persona card. Each turn returns
      {utterance, move, close_conversation, plan_update:{op, actions}, think}.
    Before release the reply schema carries no plan_update (autopilot); after a
    genuine `accept` the full planning schema is live and plan_update actions
    are what the hands do next."""
    def __init__(self, persona, task_desc, client=None, plan=None):
        self.persona = persona
        self.plan = list(plan) if plan else None   # the person's own plan (set by the runner if absent)
        self.task_desc = task_desc
        self.client = client
        self.released = False
        self.system_task = prompts.SYSTEM_TASK_HS

    def turn(self, exact_map, body, conversation, history=None, live=None, engine_hint="unspotted",
             queue_empty=False, grader="person", mistakes=None):
        prompt = prompts.compose_human_v2(
            self.task_desc, self.persona, exact_map, body, conversation,
            system_task=("" if self.released else self.system_task),
            released=self.released, plan=self.plan, history=history, live=live,
            engine_hint=engine_hint, queue_empty=queue_empty, grader=grader, mistakes=mistakes)
        raw = self.client.text(prompt, temperature=0.8, max_tokens=2500)
        d = _extract_json(raw)
        move = (d.get("move") or "").strip()
        if move not in MOVES:
            move = "answer" if (d.get("utterance") or "").strip() else ""
        pu = d.get("plan_update") if isinstance(d.get("plan_update"), dict) else {}
        op = (pu.get("op") or "drop") if pu else "drop"
        if op not in ("drop", "append", "replace"):
            op = "drop"
        actions = [str(a).strip() for a in (pu.get("actions") or []) if str(a).strip()] if pu else []
        call = str(d.get("assistant_error_call") or "none").strip().lower()
        if call not in ("none", "correct", "wrong"):
            call = "none"
        return {"utterance": (d.get("utterance") or "").strip(),
                "move": move,
                "close_conversation": bool(d.get("close_conversation")),
                "done": bool(d.get("done")),
                "assistant_error_call": call,
                "call_about": str(d.get("call_about") or "").strip(),
                "plan_update": {"op": op, "actions": actions[:8]},
                "think": (d.get("think") or "").strip(),
                "_raw": raw if isinstance(raw, str) else "",
                "_prompt": prompt}


def _nested_reply_fields(d):
    """Some models close the task_progress object too late and nest the reply's own top-level
    fields ("text", "flag", "goal_change") inside it. The words and the flag were then silently
    lost (gpt-5.4-mini, ~10% of its turns in rounds 7-10). Lift them back to the top level when
    the top level lacks them."""
    if not isinstance(d, dict):
        return {}
    tp = d.get("task_progress")
    if isinstance(tp, dict):
        for k in ("text", "flag", "goal_change"):
            if k in tp and not d.get(k):
                d[k] = tp.pop(k)
    return d


class AssistantV2:
    """The v2 assistant: task_progress / goal_change / text / typed flag."""
    def __init__(self, persona, client=None, use_relational=False, task_desc="", manual=""):
        self.persona = persona
        self.client = client
        self.use_relational = use_relational
        self.task_desc = task_desc
        self.manual = manual

    def turn(self, frames, relational_text, memory, conversation="", tick=0, **_):
        rel = relational_text if self.use_relational else ""
        prompt = prompts.compose_assistant_v2(len(frames), rel, self.persona, memory,
                                              conversation, task_desc=self.task_desc,
                                              manual=self.manual, tick=tick)
        raw = self.client.text(prompt, images=frames, temperature=0.5, max_tokens=3000)
        d = _nested_reply_fields(_extract_json(raw))
        flag = d.get("flag")
        err = (flag.get("error") if isinstance(flag, dict) else None)
        err = str(err).strip() if err else None
        return {"text": (d.get("text") or "").strip(),
                "flag": ({"error": err} if err else None),
                "task_progress": d.get("task_progress") if isinstance(d.get("task_progress"), dict) else None,
                "goal_change": d.get("goal_change"),
                "note": raw if isinstance(raw, str) else "",
                "_prompt": prompt}


class RosterAssistant:
    """A roster assistant (screensim/interact/roster.py) in the v2 assistant seat, called the way CookSim
    calls it: the backend gets a fixed per-episode SYSTEM part and a per-turn USER part
    (prompts.compose_assistant_v2_split) plus the frames, and returns the parsed reply. One instance per
    episode: session backends (Gemini Live, GPT-realtime) open their session on the first call.
    Every call's latency and tokens are kept in `meta_log`; a call that fails after retries is silence."""

    def __init__(self, tag, persona=None, task_desc="", manual="", backend=None, retries=3):
        from .roster import make_backend
        self.tag, self.persona, self.task_desc, self.manual = tag, persona, task_desc, manual
        self.b = backend or make_backend(tag)
        self.max_tries = retries
        self.retries = self.rate_limited = 0       # call-health counters, like the person's client
        self.meta_log = []
        self._open = False
        self.calls = self.empty = self.failures = 0
        self.client = self                    # the runner's call-health counters read .client.<count>

    def turn(self, frames, relational_text, memory, conversation="", tick=0, **_):
        import time as _t
        system, user = prompts.compose_assistant_v2_split(
            len(frames), "", self.persona, memory, conversation, task_desc=self.task_desc,
            manual=self.manual, tick=tick)
        if hasattr(self.b, "reset_episode") and not self._open:
            self.b.reset_episode(system)
            self._open = True
        d, last = None, None
        for k in range(self.max_tries):
            try:
                d, meta = self.b.call(system, user, list(frames) or None)
                meta["tick"] = tick
                self.meta_log.append(meta)
                break
            except Exception as e:                                   # noqa: BLE001
                last = e
                self.retries += 1
                if any(x in str(e) for x in ("429", "RESOURCE_EXHAUSTED", "rate")):
                    self.rate_limited += 1
                _t.sleep(min(60, 4 * 2 ** k))
        self.calls += 1
        if d is None:
            self.failures += 1
            self.meta_log.append({"tick": tick, "error": repr(last)[:300]})
            d = {}
        d = _nested_reply_fields(d if isinstance(d, dict) else {})
        if not (d.get("text") or "").strip() and not d.get("flag"):
            self.empty += 1
        flag = d.get("flag")
        err = (flag.get("error") if isinstance(flag, dict) else None)
        err = str(err).strip() if err else None
        tp = d.get("task_progress") if isinstance(d.get("task_progress"), dict) else None
        return {"text": (d.get("text") or "").strip(),
                "flag": ({"error": err} if err else None),
                "task_progress": tp,
                "goal_change": d.get("goal_change"),
                "note": json.dumps(d, ensure_ascii=False),
                "_prompt": system + "\n\n" + user}

    def stats(self):
        lat = sorted(m["latency_ms"] for m in self.meta_log if m.get("latency_ms"))
        return {"model": self.tag, "calls": self.calls, "failures": self.failures, "empty": self.empty,
                "latency_ms_median": lat[len(lat) // 2] if lat else None,
                "latency_ms_p90": lat[int(len(lat) * 0.9)] if lat else None,
                "prompt_tokens": sum(m.get("prompt_tokens") or 0 for m in self.meta_log),
                "completion_tokens": sum(m.get("completion_tokens") or 0 for m in self.meta_log)}
