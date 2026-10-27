"""NATIVE streaming assistant seats for the household benchmark (gemini-3.8-live__native,
gpt-realtime-2.1__native) — ported from CookSim/vhhome_bench/native_backend.py (project decision 2026-10-04:
the two streaming APIs are evaluated in their native setting, never the polled one).

The polled Live / Realtime backends ASK the model every POLL ticks and force a JSON reply. That removes the
one thing a streaming model adds: its own decision of when to speak. In the native setting the harness only
FEEDS the session (standing input spec, memory/streaming-native-arm-inputs.md):
  * session instructions once — the supervisor prompt with its turn-based parts (what you receive each turn,
    the key_moments memory, the human_progress JSON, the reply JSON) replaced by live-stream ones;
  * the house as video, ONE frame per tick (first-person | top-down plan, the same composite the polled
    assistant gets eight of), paced to at most one frame per tick of wall time — and nothing else about the
    house: no state line, no map text, no body-state line;
  * the person's speech as TTS audio (one fixed voice, cached; synthesis is outside every clock);
  * a bare "[t=N]" marker with each frame, which asks for nothing.
Whatever the model says, whenever it says it, comes back from drain() as an utterance. A mistake call is
recognised by the spoken marker "Flag: <mistake name>." and graded exactly like the polled JSON flag.

The session classes, the TTS voice and the marker parser are CookSim's own (imported, not copied), so the two
benchmarks' native seats are the same code; only the prompt and the mistake names are the household's.
"""
from vhhome import paths
import os, re, sys


os.environ.setdefault("COOKSIM_TTS_CACHE", os.path.join(paths.RUNS, "tts_cache_native"))
from vhhome_bench import native_backend as NA  # noqa: E402

TAXONOMY = ("navigation_inefficiency", "repetition", "skipped_step", "stall", "wrong_object", "wrong_placement")
NA.TAXONOMY = TAXONOMY
NA._TAX_NORM = {NA._norm(n): n for n in TAXONOMY}               # parse_flag matches against the household names
parse_flag = NA.parse_flag
TICK_WALL = NA.TICK_WALL
SPOKEN = ", ".join(n.replace("_", " ") for n in TAXONOMY)

NATIVE_RECEIVE = """== WHAT YOU RECEIVE ==
You watch the house LIVE. Nobody asks you questions on a schedule and nobody waits for a reply from you: you
decide for yourself when to speak.
(1) VIDEO: one image per tick (one tick = about 1.2 seconds). Each image is split: LEFT = the human's
first-person camera (you see what they see: their hands, what they grab, doors they open), RIGHT = a top-down
floorplan of the house where the YELLOW DOT is the human, the arrow their heading, and the BLUE LINE their
trajectory so far. Read their actions, their room and the state of things from the pictures.
(2) A TIMESTAMP "[t=N]" arrives with every image. It is only a clock: it never asks you to speak.
(3) The HUMAN'S VOICE: whatever the human says to you reaches you as their speech.
Everything you say reaches the human as your voice.
"""

NATIVE_MEMORY = """== WHAT YOU REMEMBER ==
You remember what you have watched in this session; nothing is handed back to you. Keep track of the moments
that matter later — an object changing hands or place, a device switched, a claim the human made about its
own body. One thing worth remembering above all: the human can FEEL what is in its own hands and you cannot.
If it tells you what it is holding and your reading of the picture disagrees, treat your own reading as the
doubtful one.
"""

NATIVE_TRACK = """== YOU TRACK THE HUMAN'S PROGRESS YOURSELF ==
Keep your own running picture of the human's progress from what you watch: which subgoals are done, which one
is under way, and what the task still requires.
{names}
From the task, expect every subgoal the human must eventually do, WITHOUT imposing an order the task does not
require. Count a subgoal as done ONLY when you SAW it complete; keep several hypotheses about the current one
while the frames are ambiguous.
"""

NATIVE_REPLY = """== HOW YOU SPEAK ==
You talk to the human out loud, in short spoken sentences — no lists, no JSON, no stage directions. There are
no message categories — say whatever helps in your own words: point out a mistake (name it precisely), answer
a question, walk the human through a recovery. Silence is right most of the time: when the human is working
correctly, say nothing. When the human asks you something, answer briefly and correctly. Time passes while
you talk.

== ERROR DETECTION ==
When you tell the human they have made one of the mistakes in WHAT CAN GO WRONG, START that sentence with the
word "Flag", then the mistake's name, then carry on normally:
  "Flag: <mistake name>. <what you tell the human, in your own words>"
The name must be EXACTLY one of: """ + SPOKEN + """. The word "Flag" and the name are for the grading system
(the human does not hear them); a name outside this list counts against you. Never say "Flag" at any other
time. The human is confident and WILL push back, so raise a mistake only when you have actually seen it
happen. After you raise one, your job continues: convince the human and guide the repair through conversation
until the task is done."""


def native_system_prompt(sup_sys):
    """The supervisor prompt with every turn-based part swapped for its live-stream counterpart. Fails loudly
    if the source prompt changed shape, so a silent half-conversion cannot reach a run."""
    out, seen = [], set()
    for h, body in NA._sections(sup_sys):
        if h is None:
            out.append(body)
        elif h == "== WHAT YOU RECEIVE EACH TURN ==":
            out.append(NATIVE_RECEIVE); seen.add("receive")
        elif h == "== YOUR MEMORY: key_moments ==":
            out.append(NATIVE_MEMORY); seen.add("memory")
        elif h == "== YOU TRACK THE HUMAN'S PROGRESS YOURSELF ==":
            m = re.search(r"^Subgoal names:.*$", body, re.M)
            out.append(NATIVE_TRACK.format(names=m.group(0) if m else "")); seen.add("track")
        elif h.startswith("== YOUR REPLY"):
            out.append(NATIVE_REPLY + "\n"); seen.add("reply")
        elif h == "== ERROR DETECTION ==":
            seen.add("detect")                              # folded into NATIVE_REPLY
        else:
            b = body.replace(" (your key_moments)", "").replace("reconcile against your key_moments of what you actually watched",
                                                               "reconcile against what you actually watched")
            b = re.sub(r"\s*Log the concession in\s+key_moments so you do not relitigate it\.", " Do not relitigate it.", b)
            b = b.replace("UPDATE YOUR\nOWN subgoals_future to match", "UPDATE YOUR\nOWN picture of the plan to match")
            b = b.replace("UPDATE YOUR OWN subgoals_future to match", "UPDATE YOUR OWN picture of the plan to match")
            out.append(h + "\n" + b)
    missing = {"receive", "memory", "track", "reply", "detect"} - seen
    if missing:
        raise RuntimeError(f"native_system_prompt: the supervisor prompt lacks sections {sorted(missing)}")
    s = "\n".join(out)
    s = s.replace("your ONLY output channels are what you SAY\nto the human and your own memory.",
                  "your ONLY output channel is what you SAY\nto the human.")
    left = re.findall(r"key_moments|human_progress|subgoals_\w+|strict JSON|PREVIOUS OUTPUT|COMMUNICATION HISTORY", s)
    if left:
        raise RuntimeError(f"native_system_prompt: turn-based terms survived the conversion: {sorted(set(left))}")
    return s


def make_native(kind, model):
    return NA.make_native(kind, model)
