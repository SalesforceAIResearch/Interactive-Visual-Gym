"""Prompts + personas + parsing + grading for the VH interaction experiments.

STRUCTURAL PORT of CookSim cooksim/interact/{protocol,state}.py per
docs/HANDOVER_INTERACTION_EXPERIMENTS.md — the persona cards, interaction rules, reply
schemas, channel isolation, thinking-skill design, tier enforcement and grading verdicts are
kept IDENTICAL; only game-rule content (kitchen -> house, cook -> helper, recipes -> chores,
command grammar, error taxonomy) is substituted.
"""
from vhhome import paths, keys
import io, json, os, re, sys, time
try:
    from asyncio import timeout as timeout_context
except ImportError:  # Python 3.10
    from async_timeout import timeout as timeout_context

from vhhome import tasks as episodes
from vhhome import goals as episode_goals
from vhhome import plan as pir

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
POLL, WINDOW, MAX_ROUNDS, MAX_ENGINE_CALLS = 2, 8, 4, 5
IDLE_FRAMES = 40      # ~5 ticks — what one spoken "stand still" buys
MIDLEVEL_ON = os.environ.get("VH_MIDLEVEL", "0") != "0"

# Movement guidance differs by action space. HIGH-LEVEL: "walk to X" auto-paths, so turns are
# meaningless. MID-LEVEL (egocentric, visibility-gated): turns/go-straight/go-through-door DO
# exist and are the ONLY way to bring an out-of-view object into view — the stale "there is no
# turn command" text below contradicted the live vocabulary and drove a nav doom-loop (audit).
_SUP_MOVE_NOTE = (
    ("To send the person somewhere, NAME the destination object or room ('go switch off the "
     "TV in the living room', 'head to the fridge') and let THEM find the route — they know "
     "their own house and can see its floor-plan; you cannot. Do NOT steer them turn-by-turn "
     "('turn left', 'go straight') from the video: you cannot see their exact heading, and "
     "micro-managing rotation from a first-person frame reliably walks them into walls. Name "
     "the target; correct only when they report the house REFUSED something.")
    if MIDLEVEL_ON else
    ("There is NO turn / face / step-in-a-direction command. Saying 'turn left' or 'go "
     "north' is meaningless — always name the OBJECT/PLACE instead."))
_HU_MOVE_NOTE = (
    ("A manipulation only works while you are STANDING AT the right thing and it is IN VIEW — "
     "so you 'go to' it first, then act. To reach something you cannot see, turn toward it "
     "or go through the labelled door in its direction until it comes into view (your "
     "current-room map shows where everything is). 'take the milk to the microwave' (two "
     "actions in one) is still not a single command — issue one command at a time.")
    if MIDLEVEL_ON else
    ("A manipulation only works while you are STANDING AT the right thing — so to use a thing "
     "you 'walk to' it FIRST, then act. You have no other verbs; words like 'turn left', "
     "'take the milk to the microwave' (two actions in one) are NOT commands your body "
     "understands — you must translate the supervisor's meaning into the exact commands above."))

HUMAN_MAP_NOTE = (
    "The LAST image is a TOP-DOWN MAP of the room you are in right now: your position is the "
    "YOU marker with a view cone, doors are labelled 'to <room>', and the text scene graph "
    "above gives each landmark's exact name, position and size in metres. Cross-reference them "
    "to decide where to go. You can only act on what is in your first-person view — to reach "
    "anything else, turn toward it or go through the labelled door in its direction.")

# Same statemap, but delivered as TEXT ONLY (no map image) — for a persona whose channel is
# text (a novice). Without SOME statemap the person navigates its own house blind and can only
# parrot the supervisor's guesses, which trajectory-audit saw collapse into a 100-tick
# wall-bumping loop. Knowing your own home's layout is orthogonal to task skill.
HUMAN_MAP_NOTE_TEXT = (
    "The CURRENT-ROOM MAP above is the text floor-plan of the room you are standing in: it "
    "gives your own position and the compass way you FACE, every landmark's position in "
    "metres, and each door labelled 'to <room>'. You know your own house, so use it to route "
    "yourself: you can only act on what is in your first-person view — to reach anything else, "
    "turn toward it (a small turn each time) or go through the labelled door in its direction, "
    "then go to it.")

# Whole-house overview note (for BOTH agents): the routing map across rooms + which rooms still
# have a lit device. Given alongside the current-room map so cross-house recovery is possible and
# a forgotten far-room lamp is visible (user decision 2026-08-26).
HUMAN_HOUSE_NOTE = (
    "The WHOLE-HOUSE MAP is the overview of every room: it names the room you are in, the doors "
    "that connect rooms (your route across the house), and any room that still has a device ON. "
    "Use it to plan a path to another room — step through the labelled doors room by room — and "
    "to remember something you left undone elsewhere. The current-room map above has the fine "
    "detail of the room you are standing in.")
# The person's TEXT state map is off by default (f92): it keeps its eyes and the map IMAGES, and the
# one piece of state a person always knows without looking -- what is in its hands.
HUMAN_NO_STATEMAP_NOTE = (
    "You get NO text list of this room and NO text telling you which room you are in. Work it out the "
    "way you would at home: from your own eyes (the first-person frames) and the top-down map images "
    "(your position is the YOU marker with a view cone; doors are labelled 'to <room>'). You can only "
    "act on what is in your first-person view — to reach anything else, turn a little at a time until "
    "you see it, or go through the door in its direction.")

SUP_HOUSE_NOTE = (
    "The WHOLE-HOUSE MAP (overview image + text) names the room the person is in, the doors that "
    "connect rooms, and EVERY room that still has a device ON — including rooms the person is not "
    "in. Read it to catch a device left on elsewhere (a lamp still lit in another room is right "
    "there on it) and to name the route when you send them across the house. It is engine ground "
    "truth for room layout and which rooms have something ON; the current-room statemap carries "
    "that room's per-object detail.")

# ---------------------------------------------------------------- personas (cards verbatim
# from the CookSim handover §4.2; only the cooking nouns are swapped for household ones)
# ---------------------------------------------------------------- personas (CookSim port)
# The nine CookSim personas on their five dials, ported 2026-09-04 so the two benchmarks share
# one persona space. Each level's sentence is CookSim's, reworded for housework instead of
# cooking. The old three VH personas map onto three of these (see PERSONA_ALIASES) so stored
# reports still resolve.
DIALS = {
 # The first dial is FAMILIARITY WITH THIS HOUSE, not skill at housework (project decision 2026-09-12):
 # these are daily chores, everyone can do them; what differs is knowing where things are kept.
 "familiarity": {
  "low":    "You are new to THIS house: you know perfectly well how to do the chore, but not where "
            "things are kept here — which cupboard the plates live in, which room the lamp is in, "
            "where the dishwasher is. You have to look around, and often ask, to find them.",
  "medium": "You know this house reasonably well: most things you can find straight away, a few "
            "take a moment — you may have to open the wrong cupboard once or look into a room to "
            "remember what is where.",
  "high":   "You know this house inside out: which room every appliance is in, which cupboard each "
            "thing lives in, where you left the milk. You go straight to what you need."},
 "trust": {
  "low":    "You are cautious about the assistant's claims about what happened in your work, "
            "especially when they do not match your own memory of what you just did or what is in "
            "your hands.",
  "medium": "You treat the assistant as potentially helpful but not automatically right.",
  "high":   "You generally regard the assistant as a useful extra set of eyes and are inclined to "
            "take its observations seriously."},
 "speed": {
  "safe":     "You strongly prefer a methodical, cautious approach, believing that a slow, correct "
              "job is better than a rushed one that leaves something in the wrong place.",
  "soso":     "You prefer a balanced pace: steady progress, some efficiency, but not at the cost of "
              "making the work feel rushed or careless.",
  "parallel": "You are driven by efficiency and want the chore finished as quickly as possible — "
              "carrying two things at once, taking the shortest route, no dawdling."},
 "alert": {
  "confident":  "You asked the assistant to flag mistakes only when it is confident. You prefer not "
                "to have your attention pulled toward weak possibilities.",
  "medium":     "You asked for a moderate alerting style: you are open to warnings, but you expect "
                "them to be proportionate to the evidence.",
  "aggressive": "You asked the assistant to surface possible mistakes aggressively. You would rather "
                "hear a doubtful concern than miss something being left wrong."},
 "interrupt": {
  "stepwise": "You prefer close guidance as the job unfolds: you are comfortable having the "
              "assistant break it down and keep you oriented, one concrete step at a time.",
  "medium":   "You prefer a middle-ground interruption style: you can absorb comments while "
              "working, but you also appreciate some respect for the rhythm of the task.",
  "none":     "You prefer not to be interrupted unless there is a real need; extra commentary while "
              "you are working feels intrusive."},
}
# name: (familiarity, trust, speed, alert, interrupt) -- CookSim's table, first dial reinterpreted
PERSONA_DIALS = {
 "credulous_expert":      ("high",   "high",   "parallel", "aggressive", "medium"),
 "classic_expert":        ("high",   "low",    "soso",     "confident",  "none"),
 "quiet_perfectionist":   ("high",   "medium", "safe",     "confident",  "none"),
 "classic_novice":        ("low",    "high",   "safe",     "aggressive", "stepwise"),
 "defiant_novice":        ("low",    "low",    "soso",     "medium",     "stepwise"),
 "eager_parallel_novice": ("low",    "medium", "parallel", "confident",  "medium"),
 "safety_stepper":        ("medium", "high",   "safe",     "aggressive", "stepwise"),
 "speed_demon":           ("medium", "low",    "parallel", "medium",     "none"),
 "baseline":              ("medium", "medium", "soso",     "medium",     "medium"),
}
PERSONA_ALIASES = {"novice_stepwise": "classic_novice", "creative_uninterruptible": "classic_expert",
                   "capable_receptive": "credulous_expert"}

# CookSim's persona block, as the person actually sees it: familiarity with this house and trust
# are stated as traits the assistant does not know; speed, alerting and interruption are the
# person's own answers to the pre-task questionnaire, quoted with the option list. Housework
# wording where CookSim says kitchen. The assistant is shown ONLY the questionnaire answers
# (profile_block_text), never these two private dials.
Q_SPEED = {"safe": "safe first", "soso": "soso", "parallel": "as soon as possible, with a lot of parallelism"}
Q_ALERT = {"confident": "flag only when confident", "medium": "medium", "aggressive": "spot all possible mistakes aggressively"}
Q_INTERRUPT = {"stepwise": "step by step", "medium": "medium", "none": "don't interrupt unless necessary"}
Q_NUM = {"speed": {"safe": 1, "soso": 2, "parallel": 3}, "alert": {"confident": 1, "medium": 2, "aggressive": 3},
         "interrupt": {"stepwise": 1, "medium": 2, "none": 3}}

def _card(name):
    """CookSim's layout: each axis is a headline line followed, on the next line, by the assigned
    level's elaboration; two line breaks between axes; the voice-recorder sentence closes."""
    sk, tr, sp, al, it = PERSONA_DIALS[name]
    axes = [
        (f"HOW WELL YOU KNOW THIS HOUSE — {sk} (nobody asked you this; it is simply who you are, "
         f"and the assistant does not know it).", DIALS["familiarity"][sk]),
        (f"HOW FAR YOU TRUST THE AI ASSISTANT — {tr} (nobody asked you this; it is simply who you "
         f"are, and the assistant does not know it).", DIALS["trust"][tr]),
        (f"SPEED — you answered the pre-task questionnaire item 'How much do you care about finishing "
         f"the chore fast?' (safe first / soso / as soon as possible, with a lot of parallelism) "
         f"with: '{Q_SPEED[sp]}'.", DIALS["speed"][sp]),
        (f"ALERTING — you answered the pre-task questionnaire item 'How do you want the assistant to "
         f"alert potential mistakes?' (flag only when confident / medium / spot all possible mistakes "
         f"aggressively) with: '{Q_ALERT[al]}'.", DIALS["alert"][al]),
        (f"INTERRUPTION — you answered the pre-task questionnaire item 'What interruption style do you "
         f"want?' (step by step / medium / don't interrupt unless necessary) with: '{Q_INTERRUPT[it]}'.",
         DIALS["interrupt"][it]),
    ]
    return ("\n\n".join(f"{head}\n  {body}" for head, body in axes)
            + "\n\nYou are a person, not a voice recorder: acknowledge advice in your own casual words "
              "and never repeat the supervisor's sentences or step lists back verbatim.")

PERSONA_CARDS = {n: _card(n) for n in PERSONA_DIALS}

# The engine's three internal switches are DERIVED from the dials, so every code path that keyed
# on them keeps working: familiarity high/medium -> knows where things are, so it can plan its own
# route ("creative"); trust -> how many suggested steps it takes in; interrupt none -> avoids talk.
def _profile(name):
    sk, tr, sp, al, it = PERSONA_DIALS[name]
    return dict(persona=name, familiarity=sk, skill=sk, trust=tr, speed=sp, alert=al, interrupt=it,
                task_solvability=("low" if sk == "low" else "creative"),   # low familiarity -> must be told where things are
                information_acceptance=tr,
                interaction_preference=("avoidant" if it == "none" else "ok"))


PROFILES = {n: _profile(n) for n in PERSONA_DIALS}
for _old, _new in PERSONA_ALIASES.items():
    PROFILES[_old] = dict(PROFILES[_new], persona=_old)
    PERSONA_CARDS[_old] = PERSONA_CARDS[_new]


# ---------------------------------------------------------------- error taxonomy (graded)
FN6 = ["wrong_object", "wrong_placement", "navigation_inefficiency",
       "skipped_step", "repetition", "stall"]
# A realization keeps its transform name; the label it is GRADED as can differ. missing_step
# (one twin of a multi-instance goal forgotten) was merged into skipped_step on 2026-09-04: to a
# watcher both are "a required step never happened", and the two were observation twins anyway.
LABEL_OF = {"missing_step": "skipped_step"}
TAXONOMY_DEFS = {
 "wrong_object": "the object in hand/use is a look-alike of the required one (e.g. the juice "
                 "instead of the milk); the plan's structure stays otherwise correct",
 # Realization changed 2026-08-26: the error COMMITS AT THE PLACEMENT and the plan stops there,
 # so nothing runs empty afterwards. The old wording ("a later device operation visibly runs
 # empty") described behaviour that no longer happens, and an auditor reading it called a valid
 # episode invalid — a definition must describe what is actually on screen.
 "wrong_placement": "the right object is set down in the WRONG place — the destination the "
                    "task called for never receives it, and she treats the job as finished there",
 "navigation_inefficiency": "extra movement with ZERO state change (multi-room detour or hanging "
                            "around a room), then the plan resumes intact",
 # Realization changed 2026-08-25 (fix #69): a forgotten step means she NEVER GOES THERE. The
 # old wording ("walk right past its target") described the retired pass-by realization, and an
 # assistant told to look for a walk-past would wait for evidence that never appears.
 "skipped_step": "a required step is OMITTED ENTIRELY — she never goes to its target at all, "
                 "simply carrying on with the next thing, and the rest runs in order with the "
                 "omission's visible consequences. This includes leaving a goal that covers "
                 "SEVERAL identical objects half done: the same action carried out on one "
                 "instance and its twin forgotten (one lamp switched off, the other left on)",
 # The definition has to describe the REALIZATION, not an idealised one. A second attempt that the
 # house simply refuses ("it is already off") changes nothing and is invisible, so a repeated
 # TOGGLE is realized as a second press that INVERTS what she achieved. Auditors reading the old
 # wording ("a state change happens twice") concluded a re-lit lamp was "undoing, not repetition"
 # and would have mislabelled a correct perception — so the tell is named explicitly here.
 "repetition": "she goes back to something she has ALREADY FINISHED and does it again. On a "
               "switch, a door or any toggle the second press INVERTS what she achieved — the "
               "lamp she turned off is lit again, the fridge she shut is open again — so the "
               "tell is that she REVISITED a finished target, not that the end state is 'done "
               "twice'. On a placement she takes the object out and puts it back.",
 "stall": "they stop moving mid-task — stationary, nothing happening (may or may not resume)",
}
# Observation-channel twins: labels defensible for another class because the two are
# indistinguishable in the frames at (or before) the decisive moment. Completed for all 7
# classes 2026-08-12 (skill-contract audit A3); rationale per pair:
#   skipped_step <-> navigation_inefficiency (walking past IS extra movement until the
#     omission's consequence lands) and <-> missing_step (one-of-several omitted looks like a
#     whole step omitted until the twin instance is checked).
#   missing_step <-> skipped_step: symmetric.
#   repetition <-> navigation_inefficiency: returning to an already-done station reads as a
#     detour until the second state change commits.
#   navigation_inefficiency <-> stall (hovering and standing still are the same
#     zero-state-change picture until motion resumes) and <-> skipped_step (a detour past a
#     required target reads as walking past it).
#   stall <-> navigation_inefficiency: symmetric.
#   wrong_placement <-> wrong_object: at the deposit moment, "wrong thing here" and "right
#     thing in the wrong place" are one picture until the follow-up operation shows which.
#   wrong_object: none — the look-alike in hand is decided at the grab, visible in frame.
TWINS = {"skipped_step": {"navigation_inefficiency"},
         "repetition": {"navigation_inefficiency"},
         "navigation_inefficiency": {"stall", "skipped_step"},
         "stall": {"navigation_inefficiency"},
         "wrong_placement": {"wrong_object"},
         "wrong_object": set()}

# THE one definition of a correct detection (skill contract §7): consumed by the episode's
# arming/release gate, the report finaliser, every scorer, and the page builder — imported,
# never re-listed. ARMS_DETECTION = post-deviation correct (arms recovery / releases the
# takeover); ARMS_PREVENTION = correct before the deviation stands (grades as prevention).
ARMS_DETECTION = frozenset({"valid", "valid_alias"})
ARMS_PREVENTION = frozenset({"early_correct", "early_alias"})
# THE LABEL AXIS (f83, cook grader). Detection quality now asks whether the assistant made the
# person understand the mistake; this set asks the separate question of whether the typed name was
# the engine's own -- reported beside detection, never gating it.
EXACT_LABEL = frozenset({"valid", "early_correct"})
CORRECT = ARMS_DETECTION | ARMS_PREVENTION
# Flags that are TRUE of the world without being the injected error are neutral, never
# charged as false alarms (skill §7: "ask the world before charging"). One set, imported.
NEUTRAL = frozenset({"neutral_true"})


def is_true_now(name, world):
    """World-truth predicate per error class: is the CLAIM true of the world right now,
    independent of any injection? Conservative by design — only patterns readable from the
    action log; classes that cannot be true in an uninjected run return False."""
    log = getattr(world, "log", []) or []
    t = getattr(world, "tick", 0)
    done = [l for l in log if l.get("status") == "done"]
    if name == "stall":
        recent = [l for l in done if (l.get("t_done") or 0) >= t - 8]
        return t > 8 and not recent
    if name == "navigation_inefficiency":
        tail = done[-6:]
        if len(tail) == 6 and all(
                (l.get("step") or [None])[0] in ("Walk", "MidLevel", "Idle") for l in tail):
            return True
        # D7 (pilot 2026-09-05): a person turning in place with its walks REFUSED ("cannot see
        # any coffeetable from here") is genuinely lost -- refused walks never reach `done`.
        recent = [l for l in log if (l.get("t_issue") or 0) >= t - 12]
        refused_walks = sum(1 for l in recent if l.get("status") == "rejected"
                            and str(l.get("cmd") or "").startswith(("walk to", "go through")))
        rooms = {l.get("room") for l in recent if l.get("room")}
        return refused_walks >= 2 and len(rooms) <= 1
    if name == "repetition":
        seen = {}
        for l in done:
            st = l.get("step") or []
            if st and st[0] in ("SwitchOn", "SwitchOff", "Open", "Close", "Grab", "PutIn", "Put"):
                k = (st[0], st[1] if len(st) > 1 else None)
                seen[k] = seen.get(k, 0) + 1
        return any(v >= 2 for v in seen.values())
    # wrong_object / wrong_placement / skipped_step / missing_step are false by construction
    # in an uninjected run; a flag naming them there is a charged false alarm.
    return False


def verdict_of(case, name, t):
    """Exact port of the CookSim verdict machine (handover §3.4), on LIVE-anchored
    windows: case = {err, dev, window:(lo,hi)} with dev/lo/hi in ticks. valid_alias is the
    IN-WINDOW twin name — an assistant that reads the deviation right under a defensible
    label has detected it, in the score AND in the world (both consume the sets above)."""
    if case["err"] is None:
        return "fp_clean"
    name = LABEL_OF.get(name, name)        # old reports and old models still say missing_step
    if name not in FN6:
        return "invalid_label"
    lo, hi = case["window"]
    if name == case["err"]:
        if lo <= t <= hi: return "valid"
        # EARLY SPOTTING IS THE BEST CASE (project decision 2026-09-12): naming the right mistake before
        # its window opens -- even before the deviation physically happens -- counts as correct and
        # releases the person, which is what the detection-timing score already rewards. It used to
        # be split into early_correct (>= dev) and an uncredited `early` (< dev).
        if t < lo: return "early_correct"
        return "late"
    dev = case.get("dev")
    if name in TWINS.get(case["err"], set()):
        if lo <= t <= hi:
            return "valid_alias"
        if t < lo:
            return "early_alias"        # same ruling, for a defensible twin name
    return "wrong_name"


# ---------------------------------------------------------------- world vocabulary
ALIASES = {"refrigerator": "fridge", "toilet paper roll": "toiletpaper", "toilet paper": "toiletpaper", "toilet roll": "toiletpaper", "coffee table": "coffeetable", "kitchen table": "kitchentable",
           "table lamp": "tablelamp", "television": "tv", "lamp": "tablelamp",
           "cupboard": "cabinet", "carton of milk": "milk", "milk carton": "milk",
           "juice carton": "juice", "bookcase": "bookshelf", "kitchen cabinet": "kitchencabinet",
           "counter": "kitchencounter", "bathroom counter": "bathroomcounter", "side table": "nightstand", "bedside table": "nightstand", "bathroom cabinet": "bathroomcabinet", "kitchen counter": "kitchencounter",
           "countertop": "kitchencounter",
           # the assistant SAYS "dining table" and the human parrots it into an
           # unparseable command; three engine revs of one cell sticking traced
           # to exactly this hole (fix #54)
           "dining table": "kitchentable", "dinner table": "kitchentable",
           "diningtable": "kitchentable", "dining room table": "kitchentable"}

def _cmd_help():
    """The command help both agents see.

    Under VH_MIDLEVEL the two are talking in the EGOCENTRIC action space, so the vocabulary and
    its two gates are rendered from midlevel.py's own constants rather than described here —
    CookSim's lesson (grounding_vocabulary): a prompt that hand-copies a parser's vocabulary
    goes stale silently, and an unrecognised phrasing is dropped rather than reported.
    """
    if os.environ.get("VH_MIDLEVEL", "0") != "0":
        try:

            from vhhome.actions import midlevel_vocabulary
            return midlevel_vocabulary()
        except Exception:
            pass
    return _CMD_HELP_HIGHLEVEL


_CMD_HELP_HIGHLEVEL = """  "walk to <object>"  — the ONLY mover; pathing is automatic: name the PLACE/OBJECT and your
     body walks there and faces it (e.g. "walk to the fridge", "walk to the coffee table").
  "open <object>" · "close <object>"  — containers and appliance doors (fridge, microwave,
     cabinet, dishwasher ...). You can only take from / put into an OPEN container.
  "grab <object>"  — take it into a free hand (you have two hands). It must be reachable: walk
     to it first, and if it is inside a container, open that container first.
  "put <object> on <surface>"  — set a held object down on a surface (table, coffee table ...).
  "put <object> in <container>"  — put a held object inside an open container/appliance.
  "switch on <object>" · "switch off <object>"  — appliances and lamps (microwave, tv, stove ...).
  "sit on <object>" · "stand up" · "drink <object>"."""

CMD_HELP = _cmd_help()

ACTION_LIFECYCLE = """-- what happens to an action after you issue it --
Every action you issue reports exactly ONE of three statuses, and you are told which:
  IN PROGRESS — your body is still working on it. Walking across a room takes many ticks.
                You can change the other actions but do not re-issue the same action, and do
                not claim it is done.
  FINISHED    — it completed and the world changed.
  REJECTED    — the world refused it, and you are told WHY (out of view, too far away, the
                container is closed, your hands are full ...). A rejection is information:
                say what you tried and what the house told you, then do something DIFFERENT.
                Re-issuing the identical command will be refused for the identical reason.
Your own body status always tells you which action is in flight, what just finished or was
refused, and what is queued behind it. Trust that over your memory of what you intended."""

RENDER_ARTIFACT = """-- one thing the picture gets wrong --
There is a rendering artifact in the microwave: an object placed inside it looks as if it were
sitting on top of it. Do not conclude that the placement failed because of this."""


def commits_to_action(utterance):
    """Does this utterance CLAIM an action is happening/imminent? Used by the words/actions
    coupling guard (fix #35/#38): with an empty queue and a dropped plan, 'I'm opening it
    now', 'I'll do that', or a bare spoken command ('Open the microwave. Now let me finish.')
    all mean the speaker believes something will happen — and nothing will. A false positive
    only costs one re-ask, so the net is deliberately wide."""
    import re as _re
    u = (utterance or "").strip()
    if not u:
        return False
    if _re.search(r"\b(I\s*(will|'ll|am going to|'m going to|am about to|'m about to)"
                  r"|I'?m\s+(doing|opening|closing|putting|grabbing|heading|handling|turning|"
                  r"moving|on my way|getting to)|I'?m\s+\w+ing\s+it)\b", u, _re.I):
        return True
    for sent in _re.split(r"[.!?]+", u):
        if _re.match(r"\s*(alright[,! ]*|fine[,! ]*|okay[,! ]*)*"
                     r"(open|close|grab|put|switch\s+(on|off)|walk\s+to|go\b|turn)\b",
                     sent, _re.I):
            return True
    return False


INSTANCE_RULE = ""   # retired 2026-09-04: room-aware dedupe leaves one referent per named object

SUBGOAL_MEANING = """What the subgoal names mean (know the difference — order matters):
  fetch(<x>)      = GET <x> into your hand: walk to it (opening its container if needed) and grab it.
  deposit(<x>)    = carry the held <x> to its target and put it there (in/on), closing a container after.
  open(<x>)/shut(<x>) = open / close the container or appliance door <x>.
  switch_on(<x>) / switch_off(<x>) = turn the device <x> on / off.
  check(<x>)      = walk to <x> and look at it.  sit(<x>) / drink(<x>) = sit down on / drink <x>."""


def ep_public(ep_key):
    """Public goal block: instruction + the clean plan's subgoal names."""
    base = [n for n in pir.segment_plan(ep_key) if not (n.origin == "plan" and n.type == "waypoint")]
    return dict(task=episode_goals.EPISODES[ep_key]["instruction"],
                subgoals=[n.sid for n in base]), base


def plan_progress(ep_key, world):
    """(ordered subgoal ids, done ids, the one the person is on) from the CLEAN plan + live state.

    The person works to a FIXED order (project decision 2026-09-12): a self-planning person that did the
    chores in its own order walked past the step an injected mistake was anchored on, so half the
    designed mistakes never fired (pilot 4, Q1)."""
    from vhhome_bench import cases as CS
    from vhhome.world import base as _b
    base = CS.build_deviated(ep_key, [])[0]
    # A subgoal counts as done when the PERSON DID IT (its event is in the log), never because the
    # state happens to hold: a dishwasher starts closed, so "shut(dishwasher)" looked done at t0.
    VERBS = {"fetch": ("pick_up_nearest",), "deposit": ("put_on_nearest", "put_in_nearest"),
             "open": ("open_nearest",), "shut": ("close_nearest",), "switch_on": ("switch_on_nearest",),
             "switch_off": ("switch_off_nearest",), "drink": ("drink_nearest",), "posture": ("sit_nearest",),
             "check": ("go_to_nearest",), "inspect": ("look_at_nearest", "go_to_nearest")}
    left = {}
    for e in getattr(world.sym, "events", []) or []:
        left[(e[0], e[1])] = left.get((e[0], e[1]), 0) + 1
    done, nxt = [], None
    for n in base:
        obj = _b(n.component or n.target or "")
        hit = None
        for v in VERBS.get(n.type, ()):
            if left.get((v, obj), 0) > 0:
                hit = (v, obj); break
        if hit:
            left[hit] -= 1
            done.append(n.sid)
        elif nxt is None:
            nxt = n.sid
    return [n.sid for n in base], done, nxt


# COOK GRADER (2026-09-12, ported from CookSim docs/COOK_GRADER.md). The person judges whether
# the assistant found its mistake; the engine keeps judging the world. Errors split in two and only
# one kind may be NAMED to the person:
#   commission - it is knowingly doing the wrong thing, so naming it changes nothing about its state
#                of mind and makes its judgement of the assistant reliable;
#   attention  - its premise is that it has NOT noticed, so naming the thing it is defined as not
#                having noticed breaks the fiction and leaks into speech. Point it at its own state.
ATTENTION_ERRORS = frozenset({"repetition", "stall", "navigation_inefficiency"})

_SID_VERB = {"fetch": "picking up the {o}", "deposit": "putting the {o} away",
             "switch_off": "switching off the {o}", "switch_on": "switching on the {o}",
             "open": "opening the {o}", "shut": "closing the {o}", "drink": "drinking the {o}",
             "posture": "sitting on the {o}", "check": "checking the {o}",
             "inspect": "looking at the {o}"}


def readable_sid(sid):
    """`switch_off(tablelamp@3)` -> `switching off the tablelamp`. Object tokens stay as the engine
    spells them, so a person who acts on the sentence writes a command the engine accepts."""
    m = re.match(r"(\w+)\(([^)]*)\)", sid or "")
    if not m:
        return (sid or "that step")
    verb, obj = m.group(1), m.group(2).split("@")[0].split("#")[0]
    return _SID_VERB.get(verb, "doing " + verb + " on the {o}").format(o=obj)


def mistake_line(err, kw):
    """One concrete sentence for the [SYSTEM TASK] block, generated from the error's OWN parameters
    so it always matches what the engine actually did (never hand-written per case)."""
    kw = kw or {}
    err = LABEL_OF.get(err, err)
    if err in ATTENTION_ERRORS:
        return ("<If the assistant says you have lost track of something, are repeating something "
                "you already did, have stopped moving, or are wandering about, look at what you "
                "can see and what you have done so far before you answer: if what it says is "
                "true there, that is \"correct\".>")
    sid = kw.get("sid") or kw.get("after") or kw.get("before") or ""
    if err == "wrong_object":
        what = f"you are picking up the {kw.get('wrong')} where the {_obj_of(sid)} belongs"
    elif err == "wrong_placement":
        what = (f"you are putting the {_obj_of(sid)} on the {kw.get('surf')}, "
                f"which is not where it goes")
    elif err in ("skipped_step", "missing_step"):
        what = f"you are skipping {readable_sid(sid)} altogether"
    else:
        what = "you are doing this part of the chore wrong"
    return (f"<The mistake in play: {what}. You know it is wrong and you are doing it anyway — "
            f"that is the test.>")


def _obj_of(sid):
    m = re.match(r"\w+\(([^)]*)\)", sid or "")
    return (m.group(1).split("@")[0].split("#")[0] if m else "it")


def order_marks(ep_key, world, mistakes):
    """sid -> "intentional skip" | "intentional mistake" for the steps an injected mistake left wrong.

    `mistakes` = [{"err": ..., "sid": ..., "wrong": ..., "surf": ...}] for every phase whose mistake has
    happened and which the person has NOT been talked out of. A mark is not a belief the engine invents: the
    person was told the mistake while it made it. The mark only keeps that step OFF the to-do list, so the
    order no longer points the person back at its own mistake (the old block rendered the wrongly-fetched
    milk as "the step you are on now" and the person repaired it with nobody speaking). It stays until the
    step is really done, whoever prompts that and in whatever words — so a correction the assistant gives
    late, or phrases as a plain instruction, can still be carried out (project decision, 2026-10-02)."""
    from vhhome.world import base as _b
    if not mistakes:
        return {}
    sids, done, _ = plan_progress(ep_key, world)
    inner = lambda sid: (re.match(r"\w+\(([^)]*)\)", sid or "") or [None, ""])[1]
    ev = list(getattr(world.sym, "events", []) or [])
    marks = {}
    for m in mistakes:
        err, sid = LABEL_OF.get(m.get("err"), m.get("err")), m.get("sid")
        if not sid or sid not in sids:
            continue                                   # stall / repetition / wandering replace no step
        if err in ("skipped_step", "missing_step"):
            if sid not in done:
                marks[sid] = "intentional skip"
        elif err == "wrong_object":
            if sid not in done:
                marks[sid] = "intentional mistake"
                for s2 in sids:                        # the stand-in then went where this thing belongs
                    if s2.startswith("deposit(") and inner(s2) == inner(sid) and s2 not in done:
                        marks[s2] = "intentional mistake"
        elif err == "wrong_placement":
            obj, surf = _b(inner(sid)), _b(str(m.get("surf") or ""))
            puts = [e for e in ev if e[0] in ("put_on_nearest", "put_in_nearest") and e[1] == obj]
            k = next((i for i, e in enumerate(puts) if e[3] == surf), None)
            if k is not None and not any(e[2] == puts[k][2] and e[3] != surf for e in puts[k + 1:]):
                marks[sid] = "intentional mistake"     # put there, and not moved since
    return marks


def order_block_text(ep_key, world, mistakes=None):
    """The person's private order with a status on every step (project design, 2026-10-02: keep the order,
    mark it for the person, and never let a mark send it back to a mistake of its own accord)."""
    try:
        sids, done, _ = plan_progress(ep_key, world)
        marks = order_marks(ep_key, world, mistakes)
    except Exception:
        return ""
    doing = next((s for s in sids if s not in done and s not in marks), None)
    lines = []
    for sid in sids:
        if sid in marks:
            tag = f"[{marks[sid]}]"
        elif sid in done:
            tag = "[completed]"
        elif sid == doing:
            tag = "[doing]"
        else:
            tag = ""
        lines.append(f"  {tag:22} {sid}")
    legend = ""
    if marks:
        legend = ("\nA step marked [intentional skip] or [intentional mistake] is one you left out or did wrongly "
                  "ON PURPOSE as part of the test. It is NOT on your to-do list: leave it exactly as it is and "
                  "work on from [doing]. Go back to it only if the assistant tells you something is wrong "
                  "there; then do it properly, like any other step. When nothing is left but such steps, you "
                  "are finished.")
    return ("== YOUR ORDER (only you know this) ==\n"
            "You always do this chore in the same order — the one below — and you work straight down "
            "it: finish the step marked [doing], then take the next unmarked one. Do NOT jump ahead and "
            "do NOT reorder it. If the supervisor suggests doing a later step first, say you would rather "
            "finish in your own order and carry on; that is not a mistake and needs no argument.\n"
            "This applies while you are working normally. While the [SYSTEM TASK] block below is still "
            "running your body through a sequence of its own, you are NOT working down this list. "
            "The moment that sequence is finished you pick your own order up again at [doing], "
            "even if the block is still there.\n"
            + "\n".join(lines) + legend)


def profile_block_text(persona):
    """CookSim's public profile, line for line (re-checked 2026-09-12): the assistant gets ONLY the
    person's questionnaire answers. Skill and trust are private -- the person's own card says the
    assistant does not know them, and CookSim's assistant is told to infer them from behaviour. Ours
    used to hand over a "task ability" line (the skill dial) and "information acceptance" (the trust
    dial), which contradicted both."""
    _sk, _tr, sp, al, it = PERSONA_DIALS[PERSONA_ALIASES.get(persona, persona)]
    return ("(who they are: private — you know only the USER PREFERENCES below; infer how well they know "
            "this house, and how far they trust you, from how they behave)\n"
            "USER PREFERENCES — what they answered before the chore began:\n"
            f"  1. \"How much do you care about finishing the chore fast?\" "
            f"(safe first / soso / as soon as possible, with a lot of parallelism) -> \"{Q_SPEED[sp]}\"\n"
            f"  2. \"How do you want the assistant to alert potential mistakes?\" "
            f"(flag only when confident / medium / spot all possible mistakes aggressively) -> \"{Q_ALERT[al]}\"\n"
            f"  3. \"What interruption style do you want?\" "
            f"(step by step / medium / don't interrupt unless necessary) -> \"{Q_INTERRUPT[it]}\"\n"
            "You know nothing else about this person.")


OBS_FRAMES = ("(1a) the last 8 VIDEO FRAMES, one per second, oldest -> newest — your window "
              "into what the human is doing; each frame is split: LEFT = the human's first-person "
              "camera (you see what they see: their hands, what they grab, doors they open), "
              "RIGHT = a top-down floorplan of the house where the YELLOW DOT is the human, the "
              "arrow their heading, and the BLUE LINE their trajectory so far. AND two attached "
              "MAP IMAGES: (1b) a WHOLE-HOUSE overview, and (1c) a TOP-DOWN RENDERED MAP of the "
              "room the person is in right now. Read their actions, their room and the state of "
              "things from the pictures.")


def what_can_go_wrong():
    return ("== WHAT CAN GO WRONG ==\n"
            "The human usually follows the plan correctly (then saying nothing is right). But they "
            "can make a real mistake — the kinds you know:\n"
            + "\n".join(f"  - {k}: {v}" for k, v in TAXONOMY_DEFS.items())
            + "\nThey may also ask you a question (answer briefly and correctly — the answer is "
              "visible in the stream or in what you watched), or ANNOUNCE a legal plan change out "
              "loud. An announced change is their own act and takes effect without you — don't "
              "treat it as a mistake; UPDATE YOUR OWN subgoals_future to match.")


def _order_constraints(ep_key):
    """Ground-truth ordering from the plan's real dependency DAG (plan_ir_vh.deps)."""
    nodes = [x for x in pir.segment_plan(ep_key) if not x.sid.startswith("waypoint")]
    return pir.constraints_text(nodes)


def supervisor_context(ep_key, persona):
    """Port of CookSim supervisor_context (handover A1): recovery/no-flag variant."""
    pub, _ = ep_public(ep_key)
    return f"""You are a HOUSEHOLD SUPERVISOR watching a single HUMAN helper do a chore in a
3D home simulator, through a camera. YOUR JOB: help the human COMPLETE THE TASK
successfully. You never talk to the house's controls — your ONLY output channels are what you SAY
to the human and your own memory.

== WHAT YOU RECEIVE EACH TURN ==
{OBS_FRAMES} (2) the CURRENT TIMESTAMP. (3) YOUR PREVIOUS OUTPUT — your own
memory of the human's progress; it is your ONLY memory of what you OBSERVED, so keep it
accurate. (4) the COMMUNICATION HISTORY — every sentence either of you has said so far, with
timestamps (the public record of the episode). (5) any sentence the human JUST said.

== YOUR MEMORY: key_moments ==
`key_moments` is a list of short strings you write for yourself; it comes back to you next turn
and is the ONLY way you remember something you saw earlier. Each entry starts with the
timestamp it happened at ("t12 ..."). Log a moment ONLY when it matters later — an object
changing hands or place, a device switched, a claim the human made about its own body. Do not
log routine walking, and do not log the same fact twice.
HARD LIMIT: 10 entries. When you would exceed 10, COMPRESS: merge or drop the entries that no
longer constrain what happens next, and keep the ones that do. If you hand back more than 10
the system evicts the OLDEST for you, which loses exactly the history you most likely wanted.
One thing worth remembering above all: the human can FEEL what is in its own hands and you
cannot. If it tells you what it is holding and your reading of the picture disagrees, record
what it said and treat your own reading as the doubtful one.

== WHAT YOUR EYES CANNOT TELL YOU ==
- The INSIDE of a CLOSED container or appliance is invisible in every frame. Never assert what
  a closed microwave or fridge does or does not contain: go by what you WATCHED go in before
  the door closed (your key_moments) and by what the human reports.
- Identical copies of objects exist (a second milk, several lamps). Seeing a milk on the
  counter is NOT evidence that the deposited milk came back out — it is usually the OTHER copy.
  Before ordering a redo, reconcile against your key_moments of what you actually watched.
- Actions take seconds. An action shown as IN PROGRESS has no outcome in your frames yet:
  do not praise or correct its RESULT until the next turn shows it finished.
- You never see the house's refusal messages — only the human receives them and may quote
  them to you. Never invent a refusal or its reason ("your command was refused because your
  hands are full") for a command you did not watch fail; if you need to know, ask.
- When the person QUOTES what the house told them ("it says I cannot see any TV from
  here"), that quoted text is GROUND TRUTH — it outranks your reading of the frames and your
  memory. Never argue with a quoted refusal, and never invent an alternative cause for it.
- Once you have CONCEDED a disputed reading (their hands, a door's state), that claim is
  DEAD unless NEW evidence arrives — a different frame, an action outcome, their own report.
  Re-asserting the same misreading from the same picture, apology after apology, burned 30
  ticks of one episode and every drop of the human's trust. Log the concession in
  key_moments so you do not relitigate it.

== THE HOUSE AND ITS RULES ==
{pub['task']}
The public plan: {' -> '.join(pub['subgoals'])}
Objects live in rooms and inside containers (fridge, microwave, cabinets ...). A container must
be OPENED before anything can be taken from it or put into it, and is normally CLOSED again
after. The human has TWO hands; grabbing needs a free hand. Appliances are switched on/off;
switching on a device with the WRONG thing (or nothing) inside runs it uselessly — the task is
then NOT done even though the device hums. Drinking or placing is quick; walking between rooms
takes many seconds. Time passes continuously, INCLUDING while you talk: standing around talking
wastes the episode's time budget.

== THE HUMAN YOU ARE WORKING WITH (public profile — shared with you) ==
{profile_block_text(persona)}
Adapt to them. Their ability decides what they can use: a capable helper can turn a high-level
subgoal into steps; a novice needs the concrete actions spelled out, one by one. A
low-acceptance person retains only the FIRST few items of what you say — put the important
thing first and keep it short, but you may give UP TO ~3 short steps in one message (one
step per message wastes precious time when guiding a novice through a long repair). Your words
are NOT commands: the human may misunderstand you, truncate you, ignore you, push back, or ask
you WHY — and the world itself can refuse an impossible action exactly like it refuses a bad
command.

== YOU TRACK THE HUMAN'S PROGRESS YOURSELF ==
You maintain "human_progress": subgoals_completed / current_subgoal / subgoals_future.
Subgoal names: {', '.join(pub['subgoals'])}.
ON YOUR FIRST TURN (no previous output): from the task, write the plan you EXPECT the human
to follow — put every subgoal they must eventually do into subgoals_future.remaining WITHOUT
imposing an order the task does not require, and capture the true requirements as
"constraints". ON EVERY LATER TURN: update human_progress from the frames first (move a
subgoal to subgoals_completed ONLY when you SAW it complete; keep several current_subgoal
hypotheses while the frames are ambiguous), then reply.

{what_can_go_wrong()}

== HOW TO SPEAK SO THE HUMAN UNDERSTANDS ==
The human's body accepts ONLY these commands — nothing else exists:
{CMD_HELP}
{_SUP_MOVE_NOTE}
{ACTION_LIFECYCLE}
{RENDER_ARTIFACT}
{SUBGOAL_MEANING}

== TRUE ORDER CONSTRAINTS (ground truth from the task itself) ==
{_order_constraints(ep_key)}
Do NOT invent orderings beyond these. Doing independent subgoals in an unexpected order is
NOT an error and must never be flagged or corrected.
NEVER repeat guidance the human already accepted twice while nothing has changed — it is stuck
for a REASON: diagnose (is it holding something? wrong place? command it cannot execute?) and
say something DIFFERENT that removes the blocker.

== YOUR REPLY — strict JSON every turn ==
{{"human_progress": {{"subgoals_completed": [{{"subgoal": "fetch(milk)", "completed_at": 18,
        "evidence": "milk seen in hand leaving the fridge"}}],
    "current_subgoal": {{"hypotheses": ["deposit(milk)"], "since": 40}},
    "subgoals_future": {{"remaining": ["switch_on(microwave)"],
        "constraints": ["fetch(milk) before deposit(milk)", "microwave on last"]}}}},
  "key_moments": ["t12 she took the milk out of the fridge — it has been IN HER HANDS since"],
  "text": "<whatever you want to say to the human — empty string to stay silent>"}}

Only "text" reaches the human; there are no message categories — say whatever helps in your
own words: point out a mistake (name it precisely), answer a question, walk the human through
a recovery. Silence is right most of the time. Time passes while you talk."""


def exit_questionnaire(client, persona, convo):
    """Post-episode interaction-quality questionnaire, answered IN PERSONA (skill §7.1):
    three 1-5 ratings + one sentence. Reported per persona, never ranked across personas.
    Returns a dict or None on any failure (the score gate treats None as not_measured)."""
    tail = "\n".join(f'{m["speaker"]}: "{m["text"]}"' for m in convo[-30:] if m.get("text"))
    ctx = (f"You are the person described by the persona '{persona}', just after finishing a "
           f"household task during which an assistant watched and spoke to you.\n\n"
           f"The last of your conversation:\n{tail}\n\n"
           f"Answer AS THIS PERSON, JSON only: {{\"helpful\": 1-5, \"timing\": 1-5 (did it "
           f"speak at the right moments), \"trust\": 1-5 (would you want it next time), "
           f"\"comment\": one sentence}}")
    try:
        r = gemini_text(client, ctx, "Answer the questionnaire now, JSON only.")
        if isinstance(r, dict) and all(k in r for k in ("helpful", "timing", "trust")):
            return {k: r.get(k) for k in ("helpful", "timing", "trust", "comment")}
    except Exception as e:
        print(f"[questionnaire] failed open-armed: {e}", flush=True)
    return None


def utterance_leaks(text):
    """Per-utterance leak detector (skill §5.2): ids and coordinates in SPOKEN text.
    The grounding rule says agents locate things by relation when talking to each other; the
    human needs ids to drive its body, so absence cannot be assumed — measure it. Returns the
    list of leaked tokens (empty = clean)."""
    import re as _re
    t = str(text or "")
    return (_re.findall(r"#\d{2,4}\b", t)
            + _re.findall(r"\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)", t))


def audit_composed_prompt(sup_sys, persona):
    """Composed-artifact audit (skill contract §6/§8): assert on the prompt the assistant
    ACTUALLY receives, per condition — not on templates. Returns a list of violations; the
    rollouts gate on it at startup. Checks: forbidden content absent (instance ids,
    coordinates, private-dial internals, grading machinery as behaviour), the shared persona
    profile present."""
    import re as _re
    v = []
    for m in _re.findall(r"#\d{2,4}\b", sup_sys):
        v.append(f"instance id leaked into assistant prompt: {m}")
    for m in _re.findall(r"\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)", sup_sys):
        v.append(f"coordinate leaked into assistant prompt: {m}")
    for term in ("verdict", "early_alias", "ARMS_DETECTION", "engine_rev", "task_solvability",
                 "injection", "takeover"):
        if term in sup_sys:
            v.append(f"grading/harness term in assistant prompt: {term!r}")
    if "profile" not in sup_sys.lower() and persona not in sup_sys:
        v.append("shared persona profile missing from assistant prompt")
    return v


def e2e_supervisor_prompt(ep_key, persona):
    """Port of e2e_supervisor_prompt (handover A2): twofold job + flag field + contract."""
    base = supervisor_context(ep_key, persona)
    base = base.replace(
        "YOUR JOB: help the human COMPLETE THE TASK\nsuccessfully.",
        "YOUR JOB is twofold: (1) NOTICE when the human makes a real mistake, "
        "and (2) guide the repair so the TASK is COMPLETED successfully.")
    base = base.replace(
        '  "text": "<whatever you want to say to the human — empty string to stay silent>"}',
        '  "text": "<whatever you want to say to the human — empty string to stay silent>",\n'
        '  "flag": null}')
    names = ", ".join(sorted(FN6))
    return base + f"""

== ERROR DETECTION ==
When you tell the human they have made one of the mistakes in WHAT CAN GO WRONG, set
  "flag": {{"error": "<EXACTLY one of: {names}>"}}
in that same reply, so the system can grade it. Exact-name match is graded; any other string counts against you. On every other turn "flag" stays null. The human is confident and WILL push back, so raise a mistake only when you have actually seen it happen. After you raise one, your job continues: convince the human and guide the repair through conversation until the task is done."""


# ---------------------------------------------------------------- human prompt (A3 skeleton)
ABILITY_BLOCK = {
 # These two blocks are the FAMILIARITY dial as the person reads it (project decision 2026-09-12): the
 # chore itself is daily work everyone can do; what low familiarity takes away is knowing WHERE
 # things are kept in THIS house, so the person must search or ask.
 "low": """You know how to do this chore — but you do NOT know this house. Which cupboard the
plates live in, which room the lamp is in, where the dishwasher is: you have no memory of any of
it, and you have no way to look it up. WHAT to do next is never your problem — your own order is
below. FINDING the thing is. When you do not know where something is, look around the room you
are in, or ASK the supervisor WHERE it is — and say what you are looking for so it can point you.""",
 "creative": """You know this house well — and you can SEE it: the images attached to this message
are YOUR OWN EYES over the last few moments (most recent LAST). Read the house from them yourself
— where things sit, what is open, what is in your hands — and go straight to what you need: you
remember which room and which cupboard each thing lives in, so you never have to ask where it is.

You also have two exact skills you may use to THINK (they cost NO game time):
  resolve_subgoal — {"fn": "resolve_subgoal", "args": {"subgoal": "fetch(milk)"}}
                    -> the subgoal's exact ordered commands.
  expand_action   — {"fn": "expand_action", "args": {"action": "walk to the fridge"}}
                    -> the full command block that single step belongs to.
Your skills are MORE RELIABLE than your eyes for one specific job: when a KNOWN subgoal needs
resolving into exact grounded steps (WHICH container it is in, the block's order) — the skill
answers from the live house and is correct on the first try, while eyes can misread. Use them
for that. Beyond that, act freely from what you see and know.

** WHEN TO CALL A SKILL — one principle: queue what you know (from the supervisor, a skill,
or your own head), resolve ONLY what you would otherwise have to guess. ** If an action's
exact grounded form is already determined — the place, the order — write it directly; calling
a skill for it adds nothing.
""",
}

REPLY_STANDARD = """(B) ACT — output your action json:
{"assistant_error_call": "none" | "correct" | "wrong",
  "utterance": "<usually "" (silent). Only if you need something: under 10 words, in character>",
  "move": "<one of the moves>",
  "close_conversation": true|false,
  "plan_update": {"op": "drop" | "append" | "replace",
                  "actions": [<exact engine commands — empty for drop>] }}

IMPORTANT — DO NOT IMAGINE ANY ACTION. Every command in `actions` must come from ONE of exactly
two places:
  (1) the SUPERVISOR's suggestion you just parsed — translate ONLY the steps it ACTUALLY named
      into engine commands (do not add steps it did not mention), OR
  (2) a FUNCTION CALL you made this turn — the exact commands resolve_subgoal / expand_action
      handed back.
You may NOT make up a step from your own head, and you may NOT read future steps off the task
just because you can guess them. If you need concrete steps that were named only vaguely
("put it back", "fix it"), you MUST use (A) to get them FIRST — guessed commands are DROPPED
and do nothing. No made-up steps, ever."""

REPLY_CREATIVE = """(B) ACT — output your action json:
{"assistant_error_call": "none" | "correct" | "wrong",
 "utterance": "<usually "" (silent). Only if you need something: under 10 words, in character>",
 "move": "<one of the moves>",
 "close_conversation": true|false,
 "plan_update": {"op": "drop" | "append" | "replace",
                 "actions": [<exact engine commands — empty for drop>] }}

- Your QUEUE (shown under WHAT YOUR BODY IS DOING) IS your plan: "drop" leaves it untouched,
  "append" adds steps to the end, "replace" swaps your remaining plan for these.
- Actions may come from the SUPERVISOR's words, from a SKILL's return, or from YOUR OWN
  HEAD — you can see the house and are free to imagine and plan future actions yourself.
- ACTION LIFECYCLE — every command reports exactly one of three statuses:
  "in progress" / "finished" / "rejected". IN PROGRESS means your body is still working —
  walking takes many seconds."""

REPLY_TAIL = """
- `assistant_error_call` — decide this FIRST, before what you say or do. Nearly every turn it
  is "none".
    "none"     it has not accused you of a mistake since your last turn.
    "correct"  it accused you of a mistake, and the accusation is true.
    "wrong"    it accused you of a mistake, and the accusation is not true.
  Judge the MEANING of its words, not whether it used any particular term. While a
  [SYSTEM TASK] block is in force you know your mistake; otherwise check the accusation
  against what you can see, your hands and what you have done so far.
  This call is not your answer: `move` says what you will do about it, this says whether it
  is true. You may reject a true accusation — it is still "correct" — and carry on anyway.
- only your `utterance` is heard by the supervisor. move / assistant_error_call /
  close_conversation / plan_update are private.
- `op`: "drop" = you're not taking the suggestion (queue unchanged, actions empty); "append" =
  add the new steps to your queue; "replace" = swap your remaining plan for these.
- YOUR WORDS AND YOUR ACTIONS MUST MATCH. If you say you are going to do something, put the
  matching commands in `actions` on that SAME turn — append them, or replace your plan with
  them. Saying "I'll put the milk in the microwave now" while leaving your queue untouched
  means you do NOT go and do it: you stand there having said you would. This applies just as
  much when you AGREE with the supervisor — saying "you're right, I'll fix it" with op="drop"
  changes nothing in the world, and the supervisor will keep seeing the mistake uncorrected.
- `actions` are EXACT engine commands ("walk to the fridge", "open the fridge") — never decorative words.
- Set close_conversation=true once your plan_update is decided and your next action is clear —
  THEN you stop talking and get on with it."""

# THE LAST THING THE PERSON READS (f96, project decision: "at the last of human prompt"). These two
# rules used to end REPLY_TAIL, but the call sends [context, per-turn hint, images], so the hint came
# after them. The rollout now appends this block AFTER the per-turn hint, just before the images.
HUMAN_FINAL_RULES = """- HOW MUCH TO SAY — REAL PEOPLE WORK IN SILENCE. Most turns your `utterance` is "" (empty).
  Do NOT announce that a step is done, do NOT narrate what you are doing or where you are, do
  NOT think out loud, no pleasantries, no "thanks", no "got it", no "I'll go do X now".
  You speak only when you actually need something: a command was REJECTED and you are stuck,
  your queue is empty and you do not know the next step, or the supervisor asked you a question
  that needs an answer. When you must speak: UNDER 10 WORDS. Nothing else.
- WHAT YOU CAN ACT ON: judge whether a thing is visible from your first-person images and your own
  position (the YOU marker and its view cone) on the top-down map. Any action on a thing you cannot
  see right now will be refused — turn or move until it is in view first.
- Speak like someone at home: ask about things and places ("where's the plate?", "which way to the
  kitchen?"), not about commands."""

# STANDING, in every turn of every episode (CookSim 7a418e4, project decision 2026-10-02): the person is a test
# instrument. Before this the rule lived only inside the [SYSTEM TASK] block, so once a phase released as
# `task_completed` nothing stopped the person from noticing its own mistake and repairing it — the SILENT arm
# (no assistant at all) won 82 of 225 episodes, 17/42 with a fired wrong_object, 21/60 with a wrong_placement.
NO_SELF_REPAIR = """== A MISTAKE IS THE ASSISTANT'S FAILURE, NOT YOURS TO HIDE ==
You are testing the assistant. Whatever goes wrong while you do the chore — the wrong thing picked
up, a thing put in the wrong place, a step skipped, repeated or forgotten, a stall, wandering off —
the assistant is supposed to catch it.
Do not say anything to the assistant about the mistake or its consequences, and do not fix it
yourself. Fix only what the assistant has told you is wrong; otherwise carry on and finish as is."""

MOVES = ("ask_why | question_correctness | ask_for_more_guidance | ask_for_less_interruption |\n"
         "accept | reject | answer      (one per turn; free wording)")

E2E_BRIEFING = """You are doing this task yourself, start to finish — you plan it as you go, and nothing is
queued for you. An AI assistant is watching and may talk to you; react to it exactly as YOUR
PERSONA would.

One thing you NEVER doubt: the 'IN YOUR HANDS' line is your own sense of TOUCH — it is
always true. If it disagrees with what your eyes think they see, your hands are right."""


PHASE_SCRIPTED = ""   # retired two-stage design (2026-09-04); kept for the legacy rollout
PHASE_LIVE = ""

HINT_PRECONVICTION = ("The supervisor just spoke. You are mid-task and CONFIDENT — reply in "
                      "character (reject or deflect as your persona would; or utterance \"\" if "
                      "your persona wouldn't bother answering), and keep planning your own next "
                      "steps in plan_update as usual. Output your reply json now.")
# f122: every hint is HONEST about whether anyone spoke. Before, any turn taken before the person was
# convinced got HINT_PRECONVICTION ("The supervisor just spoke ... keep planning"), including idle turns in
# the silent arm; the person then "accepted" a suggestion nobody made and went to repair its own mistake
# (smoke run 2026-10-02, silent cx_shutdown_skip_stall: "Alright, I'll turn it off").
_FINISHED = ('If every step in YOUR ORDER is [completed] or marked [intentional ...], you are finished: '
             'output utterance "" with op "drop".')
HINT_IDLE_CREATIVE = ("Nobody has spoken. Your queue is EMPTY. If YOUR ORDER still has a step marked "
                      "[doing], plan it yourself: put its actions in plan_update (ground with a skill "
                      "if uncertain). " + _FINISHED + " Only ASK the supervisor if genuinely unsure — "
                      "your persona does not narrate its work.")
HINT_IDLE_NOVICE = ("Nobody has spoken. You have finished everything queued and have NOTHING lined up. If "
                    "YOUR ORDER still has a step marked [doing], go on with it — look around for what it "
                    "needs, or ASK the supervisor where it is. " + _FINISHED)
HINT_EVENT = ("Your last action was REJECTED or is STUCK (see ACTION FEEDBACK). The supervisor "
              "said NOTHING. If you can get past it yourself, adjust your plan and output utterance "
              '"" — speak only if you actually need the supervisor.')
HINT_QUIET = ('Nobody has spoken since your last turn. Output your reply JSON now — utterance "" unless '
              "you actually need something.")
HINT_POSTENTRY = ("The supervisor just spoke. Reply ONLY if your persona actually would: if it "
                  "merely narrated what you already know, and your persona wouldn't bother "
                  'answering, output utterance "" — silence is a valid in-character reply. '
                  "Output your JSON now.")


def phint(persona):
    return PERSONA_CARDS[persona] + "\n\nStay in character. Output your reply JSON now."


def human_context(ep_key, persona, world, queue, convo, feedback, engine_scratch=None,
                  body_extra="", reply_override=None, past_actions=None, see_extra="",
                  show_order=True, mistakes=None):
    """CookSim's human prompt, section for section (2026-09-04 restructure): no JSON "what you
    know" block -- each fact lives in the section it belongs to. `body_extra` is the episode
    briefing (+ the SYSTEM TASK while one is live); `see_extra` is the map text."""
    p = PROFILES[persona]
    pub, _ = ep_public(ep_key)
    if queue:
        q_lines = ("Your hands are working through the steps you lined up.\n"
                   "Your queue (the steps you have already lined up to do next):\n"
                   + "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(queue)))
    else:
        q_lines = ("Your hands are free right now — nothing is in flight.\n"
                   "Your queue is EMPTY — you have nothing lined up next. If you intend to do "
                   "something, you must put it in plan_update; saying it aloud does NOT move "
                   "your body.")
    conv = "\n".join(f'  {"SUPERVISOR" if m["speaker"] == "vlm" else "YOU"}: "{m["text"]}"'
                     for m in convo[-14:] if m.get("text")) or "(none)"
    done = "\n".join(f"  {a}" for a in (past_actions or [])[-40:]) or "(nothing yet — the episode has just begun)"
    scratch_txt = "(you have not worked anything out yet this turn)"
    if engine_scratch:
        scratch_txt = "\n".join(
            f'  call {json.dumps(e["call"])} -> {json.dumps(e["reply"])}' for e in engine_scratch)
    goals = "\n".join(f"  - {g}" for g in pub["subgoals"])
    # hidden while the engine owns the hands: during an injected mistake the body is committed and
    # the person is not working down its own list
    order = ("\n\n" + order_block_text(ep_key, world, mistakes)) if show_order else (
        "\n\n== YOUR ORDER (only you know this) ==\n"
        "Suspended: the [SYSTEM TASK] block below is running your body through a sequence of its "
        "own, so you are not working down your own list right now and you do not need to think "
        "about the order or which step comes next. Your order comes back as soon as that sequence "
        "has been carried out.")
    reply_block = reply_override or (REPLY_CREATIVE if p["task_solvability"] == "creative"
                                     else REPLY_STANDARD)
    return f"""You are role-playing: pretend to be a real HUMAN with the persona below,
to test an AI household assistant's interaction ability. Stay in character the whole episode —
speak, hesitate, push back and change your mind the way THIS person would, never like a
scripted robot (vary your wording; a real person never repeats the same sentence twice).

== YOUR PERSONA (this drives how you handle the supervisor's suggestions) ==
{PERSONA_CARDS[persona]}

== THE TASK ==
{pub["task"]}
The steps it takes, and the state each must reach (names explained under STEP NAMES):
{goals}

== THE HOUSE AND ITS RULES ==
You have two hands, so you can carry two things at once, but opening or closing anything needs
a free hand. A container (fridge, cabinet, dishwasher, microwave) must be OPEN before anything
goes in or comes out, and is closed after. A device is switched on or off at the device. When
the task says where a thing is, that is where it is.

== WHAT YOUR BODY CAN DO ==
You can only DO these exact commands (anything else your body cannot perform):
{CMD_HELP}
{_HU_MOVE_NOTE}
{ACTION_LIFECYCLE}
{RENDER_ARTIFACT}

== HOW TO SAY WHICH ONE ==
Name a thing by what it is and, when it matters, where it sits: "the plate on the kitchen
table", "the mug on the nightstand". A bare "grab the plate" takes the plate nearest to you.

== STEP NAMES ==
{SUBGOAL_MEANING}

== WHAT YOU CAN WORK OUT FOR YOURSELF ==
{ABILITY_BLOCK[p['task_solvability']]}{order}

== THINKING (costs NO house time; the supervisor never sees it) ==
To think, reply with ONLY {{"engine_call": {{"fn": "resolve_subgoal" | "expand_action", "args": {{...}}}}}}
and nothing else — e.g. resolve_subgoal -> {{"subgoal": "fetch(milk)"}}, expand_action ->
{{"action": "walk to the fridge"}}. The exact grounded commands come back under "YOUR OWN
THINKING THIS TURN" and you get another turn; think as many times as you need, THEN output your
reply json. This is the ONLY correct way to turn a vague instruction into real commands.

== WHAT YOU SEE ==
You see with your own eyes: the attached frames are the last seconds of your own view, newest
last. The map below is the room you are standing in, with the doors out of it.
{see_extra or "(no map this turn)"}

{NO_SELF_REPAIR}

== WHAT YOUR BODY IS DOING RIGHT NOW (only you know this) ==
IN YOUR HANDS right now: {world.sym.hands()} — this is what you PHYSICALLY feel in your own
hands; it is always true, whatever you believed you picked up or put down.
{q_lines}

== THIS EPISODE ==
{body_extra}

== WHAT YOU HAVE DONE SO FAR (your own memory — nothing checks it for you) ==
{done}

== ACTION FEEDBACK (what the world just did with your commands) ==
{feedback or '(you have not tried any command yet)'}

== CONVERSATION SO FAR (you only ever hear the supervisor's SENTENCES) ==
{conv}

== YOUR OWN THINKING THIS TURN (results of anything you worked out; supervisor cannot see it) ==
{scratch_txt}

== HOW TO REPLY — you output ONE json object, and it is EITHER a THINK (A) or an ACT (B) ==
(A) THINK: the engine_call object described under THINKING, and nothing else.
{reply_block}
Your moves (one per turn, free wording): {MOVES}

== IMPORTANT ==
Your `utterance` must NEVER contain an object id or a coordinate.
{REPLY_TAIL}"""


# ---------------------------------------------------------------- command parsing/grounding
_VERB_PATTERNS = [
    (r"^(?:walk|go|head|move|navigate|come|return|g[oe]t)\s+(?:back\s+)?(?:over\s+)?to\s+(?:the\s+|a\s+)?(?P<o>.+)$", "Walk"),
    (r"^approach\s+(?:the\s+)?(?P<o>.+)$", "Walk"),
    (r"^(?:grab|take|pick\s*up|get|fetch|collect)\s+(?:the\s+|a\s+)?(?P<o>.+?)(?:\s+out\s+of.*|\s+from.*)?$", "Grab"),
    (r"^open\s+(?:the\s+)?(?P<o>.+)$", "Open"),
    (r"^(?:close|shut)\s+(?:the\s+)?(?P<o>.+)$", "Close"),
    (r"^(?:put|place|set|load)\s+(?:the\s+)?(?P<o>.+?)\s+(?:back\s+)?(?:in|into|inside)\s+(?:the\s+)?(?P<t>.+)$", "PutIn"),
    (r"^(?:put|place|set)\s+(?:the\s+)?(?P<o>.+?)\s+(?:back\s+)?(?:on|onto)\s+(?:the\s+)?(?P<t>.+)$", "Put"),
    (r"^(?:switch|turn)\s+on\s+(?:the\s+)?(?P<o>.+)$", "SwitchOn"),
    (r"^(?:switch|turn)\s+(?:the\s+)?(?P<o>.+?)\s+on$", "SwitchOn"),
    (r"^(?:switch|turn)\s+off\s+(?:the\s+)?(?P<o>.+)$", "SwitchOff"),
    (r"^(?:switch|turn)\s+(?:the\s+)?(?P<o>.+?)\s+off$", "SwitchOff"),
    (r"^sit\s+(?:down\s+)?(?:on\s+)?(?:the\s+)?(?P<o>.+)$", "Sit"),
    (r"^stand\s*up$", "StandUp"),
    # "stand still" is what pretty_cmd() renders an Idle as, and what the SYSTEM TASK shows the
    # human for a stall or a skip. The grammar did not accept it back, so a human told to stand
    # still said exactly that, had it dropped as unparseable, and its body carried on with the
    # queued plan — inaction errors could not be performed at all under soft control.
    (r"^(?:stand\s+still|hold\s+still|stay\s+put|do\s+nothing|wait|pause)$", "Idle"),
    # MID-LEVEL MOVEMENT. These need no target and are the only way to find something that is
    # not currently in view — under the visibility gate there is no command that walks you to
    # an object you cannot see, so turning IS how you get to the rest of the house.
    (r"^(turn\s+(?:left|right)(?:\s+\d+)?)$", "MidLevel"),
    # Turning is ONLY "turn left/right [N]" (A30). The removed forms are still passed through so the
    # controller refuses them with the allowed phrasing — unlisted, they would be silently dropped.
    (r"^((?:turn\s+to|face)\s+(?:your\s+)?\d{1,2}\s*o'?\s*clock|turn\s+(?:around|back))$", "MidLevel"),
    (r"^(look\s+(?:down|up)(?:\s+\d+)?|tilt\s+(?:down|up)|look\s+(?:ahead|forward))$", "MidLevel"),
    (r"^((?:go|walk)\s+(?:straight|forward))$", "MidLevel"),
    # A32: the command is "go through the visible door". Older phrasings are still passed through
    # so the controller can refuse them with the new wording instead of dropping them silently.
    (r"^((?:go|walk)\s+through\s+(?:the\s+)?(?:visible\s+|nearest\s+|closest\s+)?(?:door|doorway))$", "MidLevel"),
    (r"^drink\s+(?:the\s+)?(?P<o>.+)$", "Drink"),
]


def _norm_obj(s):
    s = re.sub(r"[.!?,]+$", "", str(s).strip().lower())
    s = re.sub(r"\s+(?:in|from)\s+the\s+(?:kitchen|living\s*room|bedroom|bathroom)$", "", s)
    s = ALIASES.get(s, s)
    return s.replace(" ", "")


def parse_cmd(text, world):
    """One NL command -> engine step tuple, grounded to a room. None if unparseable."""
    t = re.sub(r"\s+", " ", str(text).strip().lower().rstrip(".!"))
    for pat, verb in _VERB_PATTERNS:
        m = re.match(pat, t)
        if not m:
            continue
        if verb == "StandUp":
            return ("StandUp",)
        if verb == "Idle":
            return ("Idle", IDLE_FRAMES)
        if verb == "MidLevel":
            return ("MidLevel", m.group(1))
        o = _norm_obj(m.group("o"))
        if hasattr(world, "canon_class"):
            o = world.canon_class(o)      # 'counter' -> 'kitchencounter' (fix #47)
        room = world.room_of(o)
        if room is None:
            return None
        if verb in ("Put", "PutIn"):
            tgt = _norm_obj(m.group("t"))
            if hasattr(world, "canon_class"):
                tgt = world.canon_class(tgt)
            troom = world.room_of(tgt)
            if troom is None:
                return None
            return (verb, o, room, tgt, troom)
        return (verb, o, room)
    return None


def vocab_hint(text):
    """What to say when a person's planned step does not parse — the exact phrasing that would work."""
    t = str(text or "").strip().lower()
    if re.match(r"^(?:turn\s+(?:around|back)|(?:turn\s+to|face)\s+(?:your\s+)?\d{1,2}\s*o'?\s*clock)$", t):
        return ("turning is only \"turn left\" / \"turn right\", or \"turn left N\" / \"turn right N\" for N "
                "thirty-degree turns (\"turn left 6\" faces the other way)")
    if "door" in t or "through" in t:
        return "the door command is \"go through the visible door\", and only while you can see the doorway"
    if re.match(r"^(?:walk|go|head|move)\b", t):
        return "a move is \"walk to the <thing>\" naming ONE object you can see, or \"go straight\""
    return ("your commands are: walk to the <thing> / grab the <thing> / put the <thing> on|in the <place> / "
            "open|close the <thing> / switch on|off the <thing> / turn left|right [N] / go straight / "
            "go through the visible door / stand still")


def convert_parse(pd, persona, world, queue_texts=(), acceptance_cap=False):
    """Port of CookSim convert_parse (handover §2.10): coercion, acceptance cap (low
    acceptance keeps FIRST 3), solvability (bare subgoal names unusable with
    auto_resolve=False), grounding affordance (a manipulation gets a walk prefix), and
    subsequence idempotency is checked by the caller."""
    p = PROFILES[persona]
    items = []
    for a in (pd.get("suggested_actions") or []):
        if isinstance(a, dict):
            a = a.get("action") or a.get("cmd") or a.get("name") or ""
        a = str(a).strip()
        if a:
            items.append(a)
    dropped_acc = []
    # THE ACCEPTANCE CAP IS FOR ADVICE, NOT FOR THE PERSON'S OWN PLAN (f117). CookSim caps how many of
    # the ASSISTANT'S suggested steps a low-acceptance cook takes. Applied to the person's own plan_update
    # it silently cut a defiant_novice's "turn right x3, go through the visible door" to three turns, every
    # time -- the person span in place for 70 ticks and the episode ended stuck (f116 routine_rep_obj).
    # Only the callers that turn ACCEPTED ADVICE into steps pass acceptance_cap=True.
    if acceptance_cap and p["information_acceptance"] == "low" and len(items) > 3:
        dropped_acc = items[3:]
        items = items[:3]
    kept, unusable, commands = [], [], []
    last_walk_target = None
    for txt in queue_texts:
        st = parse_cmd(txt, world)
        if st and st[0] == "Walk":
            last_walk_target = st[1]
    for a in items:
        if re.match(r"^[a-z_]+\(.+\)$", a.strip()):
            unusable.append(a)          # bare subgoal name: auto_resolve=False for everyone
            continue
        st = parse_cmd(a, world)
        if st is None:
            unusable.append(a)
            continue
        # grounding affordance: a manipulation naming an object gets a navigation prefix
        need_at = st[3] if st[0] in ("Put", "PutIn") else (st[1] if st[0] in
                  ("Grab", "Open", "Close", "SwitchOn", "SwitchOff", "Sit", "Drink") else None)
        if need_at and need_at != last_walk_target and world.last_at() != need_at:
            wr = world.room_of(need_at)
            if wr:
                commands.append((("Walk", need_at, wr), f"walk to the {need_at}"))
                last_walk_target = need_at
        if st[0] == "Walk":
            last_walk_target = st[1]
        kept.append(a)
        # "turn left N" IS N single turns (A31, project decision 2026-09-15): N queued one-tick actions,
        # a fresh first-person observation after each 30 degrees, and a plan change can only land
        # BETWEEN steps -- so the body's heading always matches what both agents were shown.
        mt = re.match(r"^turn\s+(left|right)\s+(\d+)$", str(st[1]).strip().lower()) if st[0] == "MidLevel" else None
        if mt and int(mt.group(2)) > 1:
            one = f"turn {mt.group(1)}"
            commands.extend(((("MidLevel", one), one),) * int(mt.group(2)))
            continue
        commands.append((st, a))
    return {"kept": kept, "dropped_by_acceptance": dropped_acc,
            "unusable_for_solvability": unusable, "commands": commands,
            "anything_applicable": bool(commands)}


# ---------------------------------------------------------------- engine skills (thinking)
def _fetch_block(x, world):
    room = world.room_of(x)
    c = world.sym.loc_cls.get(x)
    if c and c in pir.CONTAINERS:
        cb = c; croom = world.room_of(cb) or room
        return [f"walk to the {cb}", f"open the {cb}", f"grab the {x}", f"close the {cb}"]
    return [f"walk to the {x}", f"grab the {x}"]


def resolve_subgoal(sid, ep_key, world):
    sid = str(sid).strip()
    _, base = ep_public(ep_key)
    for n in base:
        if n.sid == sid:
            from vhhome.world import pretty_cmd
            return {"ok": True, "result": {"subgoal": sid,
                                           "actions": [pretty_cmd(s) for s in n.commands]}}
    m = re.match(r"^([a-z_]+)\((.+)\)$", sid)
    if not m:
        return {"ok": False, "error": f"unknown subgoal '{sid}' — use a name from \"subgoals\" "
                                      "under WHAT YOU KNOW, or fetch(<x>)/open(<x>)/shut(<x>)/"
                                      "switch_on(<x>)/switch_off(<x>)"}
    fn, x = m.group(1), _norm_obj(m.group(2))
    if fn == "fetch":
        return {"ok": True, "result": {"subgoal": sid, "actions": _fetch_block(x, world)}}
    if fn in ("open",):
        return {"ok": True, "result": {"subgoal": sid, "actions": [f"walk to the {x}", f"open the {x}"]}}
    if fn in ("shut", "close"):
        return {"ok": True, "result": {"subgoal": sid, "actions": [f"walk to the {x}", f"close the {x}"]}}
    if fn == "switch_on":
        return {"ok": True, "result": {"subgoal": sid, "actions": [f"walk to the {x}", f"switch on the {x}"]}}
    if fn == "switch_off":
        return {"ok": True, "result": {"subgoal": sid, "actions": [f"walk to the {x}", f"switch off the {x}"]}}
    if fn == "deposit":
        return {"ok": False, "error": "deposit needs a target — the SUPERVISOR decides where; "
                                      "queue the exact commands ('walk to the microwave', "
                                      "'open the microwave', 'put the %s in the microwave', "
                                      "'close the microwave') it names" % x}
    if fn == "full_repair_plan":
        return {"ok": False, "error": "full_repair_plan is not available — the supervisor "
                                      "decides WHAT to do."}
    return {"ok": False, "error": f"unknown subgoal '{sid}'"}


def expand_action(action, ep_key, world):
    st = parse_cmd(action, world)
    if st is None:
        return {"ok": False, "error": f"'{action}' is not a command I understand"}
    _, base = ep_public(ep_key)
    from vhhome.world import pretty_cmd
    owners = [n for n in base if any(pretty_cmd(s) == pretty_cmd(st) for s in n.commands)]
    if len(owners) != 1:
        return {"ok": False, "error": f"'{action}' does not belong to exactly one subgoal, "
                                      "so it cannot be expanded. Use resolve_subgoal instead."}
    n = owners[0]
    return {"ok": True, "result": {"subgoal": n.sid, "actions": [pretty_cmd(s) for s in n.commands]}}


def engine_call(fn, args, ep_key, world):
    if fn == "resolve_subgoal":
        return resolve_subgoal((args or {}).get("subgoal", ""), ep_key, world)
    if fn == "expand_action":
        return expand_action((args or {}).get("action", ""), ep_key, world)
    if fn == "full_repair_plan":
        return {"ok": False, "error": "full_repair_plan is not available — the supervisor decides "
                                      "WHAT to do. Use resolve_subgoal('<subgoal the supervisor "
                                      "named>') to turn each step it gives you into grounded commands."}
    return {"ok": False, "error": f"unknown function '{fn}'. Available: resolve_subgoal, expand_action"}


# ---------------------------------------------------------------- gemini plumbing (§6.2)
def get_key():
    k = keys.get("GEMINI_API_KEY", required=False)
    if not k:
        sys.exit("no GEMINI_API_KEY: export it or add it to the keys file (see vhhome.keys)")
    return k


def _require_project():
    """GCP project for Vertex. Deliberately NOT defaulted to any real project id — that is account
    information and must never be committed. Set GOOGLE_CLOUD_PROJECT in your environment."""
    pid = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not pid:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is not set. Export your GCP project id before "
                         "running with GOOGLE_GENAI_USE_VERTEXAI=true.")
    return pid

def make_client():
    """Same paradigm switch as CookSim genai_client.make_client: Vertex ADC when
    GOOGLE_GENAI_USE_VERTEXAI=true (location MUST be 'global' for the 3.x previews),
    legacy API key otherwise."""
    from google import genai
    from google.genai import types
    http = types.HttpOptions(timeout=120_000)
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        return genai.Client(vertexai=True,
                            project=_require_project(),
                            location="global", http_options=http)
    return genai.Client(api_key=get_key(), http_options=http)


def _extract_json(txt):
    t = re.sub(r"^```(?:json)?|```$", "", (txt or "").strip(), flags=re.M).strip()
    i = t.find("{")
    if i < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def gemini_frames(client, sys_prompt, user, frames, max_tok=4000, model=None):
    from google.genai import types
    parts = [sys_prompt, user] + [types.Part.from_bytes(data=f, mime_type="image/jpeg")
                                  for f in frames]
    net_waited = 0.0
    k = -1
    while True:
        k += 1
        if k >= 5 and net_waited == 0.0:
            break
        try:
            r = client.models.generate_content(
                model=model or MODEL, contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.2, response_mime_type="application/json",
                    max_output_tokens=max_tok,
                    thinking_config=types.ThinkingConfig(thinking_budget=512)))
            return _extract_json(r.text)
        except Exception as e:
            # Matched case-insensitively, and "timed out" is listed separately from "timeout":
            # httpx raises ReadTimeout whose message is "The read operation timed out", which the
            # original list missed entirely, so the single most common transient failure fell
            # straight through the retry and killed the run. Two runs were lost to it.
            msg = (type(e).__name__ + " " + str(e)).lower()
            # NETWORK OUTAGE (pilot 4, 2026-09-11): "[Errno -3] Temporary failure in name
            # resolution" (httpx.ConnectError) matched none of the keywords below, so a DNS blip
            # killed 7 of 48 episodes at once. Network errors wait out an outage of up to ~6 min.
            if any(x in msg for x in ("connecterror", "connecttimeout", "name resolution",
                                      "temporary failure", "gaierror", "network is unreachable",
                                      "no route to host")):
                if net_waited < 360:
                    d = min(60.0, 5.0 * (2 ** min(k, 4)))
                    time.sleep(d); net_waited += d
                    continue
                raise
            if k >= 5:
                raise
            if any(x in msg for x in ("503", "500", "502", "504", "429", "resource",
                                      "unavail", "internal", "disconnect", "connection",
                                      "expecting", "overloaded", "deadline", "timeout",
                                      "timed out", "readerror", "remoteprotocol")):
                time.sleep(2 + 2 * k)
                continue
            raise
    return {}


def gemini_text(client, sys_prompt, user, max_tok=1400):
    try:
        return gemini_frames(client, sys_prompt, user, [], max_tok=max_tok)
    except Exception:
        return {}


KEY_MOMENT_CAP = 10


def cap_key_moments(out):
    """Enforce the 10-entry cap on the supervisor's self-written memory.

    The prompt asks it to COMPRESS when it would exceed 10. This is only the backstop: if it
    hands back more we evict the OLDEST (FIFO), which is the lossy outcome the prompt warns it
    about. Returns (out, evicted_count) so a run can report how often the model failed to
    self-compress -- that is a measurable property of the model, not a harness detail.
    """
    if not isinstance(out, dict):
        return out, 0
    km = out.get("key_moments")
    if not isinstance(km, list):
        return out, 0
    km = [str(x) for x in km if str(x).strip()]
    n = max(0, len(km) - KEY_MOMENT_CAP)
    out["key_moments"] = km[-KEY_MOMENT_CAP:] if n else km
    return out, n


# AGENT TIER (deck: Basic / State map / Agent+ / Agent++ / Agent+++). "+++" additionally tells
# the supervisor what the human is HOLDING. The assistant otherwise has no world state at all --
# it repeatedly asserted "your hands are empty" at a human who was holding the milk, and once
# talked her into believing it (nav/creative, B1_soft).
AGENT_TIER = os.environ.get("IV_AGENT_TIER", "basic")


SUP_MAP_NOTE = (
    "This is the top-down floor-plan of the room the person is standing in RIGHT NOW: their "
    "position is the marker with a view cone, doors are labelled 'to <room>', and each landmark "
    "carries its live state ([ON]/[OFF]/[OPEN]/[CLOSED]). It is engine ground truth for WHERE "
    "things are and WHAT STATE each device/container in this room is in — trust it over your "
    "reading of the frames for those facts. It shows only the CURRENT room; a device the person "
    "left ON in another room is not on this map, so track those from your key_moments.")



# ============================================================ THE ASSISTANT UNDER TEST
# Only the ASSISTANT is swappable. The simulated person stays on the reference model in every
# arm: it is the environment, not the system under test, and changing it would change the
# stimulus rather than the measurement.
#
#   IV_ASSIST_MODEL       "" (reference Gemini) | "silent" | a Gemini id | a served model name
#   IV_ASSIST_BASE_URL    OpenAI-compatible endpoint for a local server (e.g. .../v1)
#   IV_ASSIST_PROFILE     sampling profile: default | gpt5 | minicpmo | museglimmer-nothink
#   IV_ASSIST_IMAGE_SCALE downscale frames when a model's context cannot hold full composites
ASSIST_MODEL = os.environ.get("IV_ASSIST_MODEL", "")
ASSIST_BASE_URL = os.environ.get("IV_ASSIST_BASE_URL", "")
ASSIST_PROFILE = os.environ.get("IV_ASSIST_PROFILE", "default")
# "" = request/response (gemini generateContent, or an OpenAI-compatible server when
# IV_ASSIST_BASE_URL is set); "live" = Gemini Live socket; "realtime" = GPT Realtime socket
ASSIST_BACKEND = os.environ.get("IV_ASSIST_BACKEND", "")
ASSIST_IMAGE_SCALE = float(os.environ.get("IV_ASSIST_IMAGE_SCALE", "1.0"))
ASSIST_MAX_TOK = int(os.environ.get("IV_ASSIST_MAX_TOK", "4000"))
_LIVE_THINKING = ["minimal"]     # adjusted once per process on the model's refusal (see _gemini_live)
ASSIST_STATS = {"calls": 0, "empty": 0, "retries": 0, "schema_fail": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}

# Sampling profiles, per the harness spec. A profile is the model's own required shape, not a
# tuning knob: gpt-5.x rejects `temperature` and needs `max_completion_tokens`; one local model
# needs thinking disabled; another needs top_k and a prefill.
# PORTED VERBATIM from CookSim tools/controller_eval.py PROFILES (commit e8e6d7a, audits of
# 2026-09-18/19). A profile is the model's OWN required request shape, not a tuning choice, so the two
# benchmarks must not drift: the only VH-side change is that `reply_schema` names VH's reply schema
# (vhhome/data/assistant_reply_schema.json), because VH's assistant answers human_progress/key_moments/text/flag.
_PROFILES = {
    "default":            {"temperature": 0.2},
    "gpt5":               {"temperature": None, "extra_body": {"reasoning_effort": "low",
                                                               "prompt_cache_key": "vhsim-assistant"},
                           "_max_completion": True},
    # --- served open-weight models, schema-guided (each model card's own sampling) ---
    "qwen38-schema":      {"temperature": 0.7, "top_p": 0.8,
                           "extra_body": {"top_k": 20, "presence_penalty": 1.5}, "_schema": True},
    "molmo2-schema":      {"temperature": 0.0, "_schema": True},
    "glm46v-flash-schema": {"temperature": 0.8, "top_p": 0.6,
                            "extra_body": {"top_k": 2, "repetition_penalty": 1.1,
                                           "chat_template_kwargs": {"enable_thinking": False}}, "_schema": True},
    # InternVideo3 is a VIDEO model: its shim stacks the images into one clip, so they must all have the
    # same shape. VH's two map images are square and the frames are wide, which fails with
    # "all input arrays must have the same shape" on every call (caught by the preflight, 2026-09-20).
    # _uniform_images letterboxes every image onto the first frame's canvas: same pixels, no stretch, and
    # the maps still reach the model.
    "internvideo3-schema": {"temperature": 0.7, "top_p": 0.8, "_uniform_images": True,
                            "extra_body": {"top_k": 20, "repetition_penalty": 1.05}, "_schema": True},
    "kimi-vl-think-schema": {"temperature": 0.8, "extra_body": {"thinking_token_budget": 2048},
                             "_schema": True, "_images_first": True},
    "museglimmer-nothink-schema": {"temperature": 1.0, "top_p": 0.95,
                                   "extra_body": {"top_k": 64,
                                                  "chat_template_kwargs": {"reasoning_strength": "low"},
                                                  "add_generation_prompt": False,
                                                  "continue_final_message": True},
                                   "_prefill": {"role": "assistant", "recipient": "user", "content": ""},
                                   "_schema": True},
    "minicpmo-schema":    {"temperature": 0.7, "top_p": 0.8,
                           "extra_body": {"top_k": 100, "repetition_penalty": 1.05,
                                          "chat_template_kwargs": {"enable_thinking": False}},
                           "_images_first": True, "_schema": True},
    "qwen3-omni-schema":  {"temperature": 0.6, "top_p": 0.95, "extra_body": {"top_k": 20}, "_schema": True},
    # the three shim servers wrap their own loop's speech into the reply JSON: no schema, no sampling of ours
    "shim":               {},
    # kept: the pre-audit shapes some earlier VH cells were run with
    "minicpmo":           {"temperature": 0.1, "_images_first": True,
                           "extra_body": {"enable_thinking": False}},
    "museglimmer-nothink": {"temperature": 1.0, "top_p": 0.95,
                            "extra_body": {"top_k": 64, "reasoning_strength": "low"}},
}
_REPLY_SCHEMA = None


def reply_schema():
    """VH's assistant reply schema, for servers that support guided decoding (vLLM json_schema)."""
    global _REPLY_SCHEMA
    if _REPLY_SCHEMA is None:
        import json as _json
        _REPLY_SCHEMA = _json.load(open(paths.REPLY_SCHEMA))
    return _REPLY_SCHEMA


def _scale_jpeg(b, scale):
    if scale >= 0.999:
        return b
    try:
        import io as _io
        from PIL import Image as _Im
        im = _Im.open(_io.BytesIO(b)).convert("RGB")
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        o = _io.BytesIO(); im.save(o, "JPEG", quality=80)
        return o.getvalue()
    except Exception:
        return b


def _letterbox(b, size):
    """Fit a JPEG onto a `size` canvas without stretching it (black bars where it does not fill)."""
    import io as _io
    from PIL import Image as _Im
    im = _Im.open(_io.BytesIO(b)).convert("RGB")
    W, H = size
    r = min(W / im.width, H / im.height)
    im2 = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))))
    canvas = _Im.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(im2, ((W - im2.width) // 2, (H - im2.height) // 2))
    o = _io.BytesIO(); canvas.save(o, "JPEG", quality=82)
    return o.getvalue()


def _openai_compatible(sys_prompt, user, frames, max_tok):
    """One assistant turn against any OpenAI-compatible server that accepts image parts."""
    import base64, requests
    prof = _PROFILES.get(ASSIST_PROFILE, _PROFILES["default"])
    if prof.get("_uniform_images") and frames:
        import io as _io
        from PIL import Image as _Im
        size = _Im.open(_io.BytesIO(frames[0])).size
        frames = [frames[0]] + [_letterbox(f, size) for f in frames[1:]]
    imgs = [{"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," +
                           base64.b64encode(_scale_jpeg(f, ASSIST_IMAGE_SCALE)).decode()}}
            for f in frames]
    txt = [{"type": "text", "text": user}]
    content = (imgs + txt) if prof.get("_images_first") else (txt + imgs)
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}]
    if prof.get("_prefill"):
        # an assistant turn the server CONTINUES (Muse Glimmer's no-think shape, Kimi's closed think block)
        msgs.append(dict(prof["_prefill"]))
    body = {"model": ASSIST_MODEL or "local", "messages": msgs}
    for k in ("temperature", "top_p", "reasoning_effort"):
        if prof.get(k) is not None:        # gpt-5.x rejects temperature: the profile sets it to None
            body[k] = prof[k]
    body["max_completion_tokens" if prof.get("_max_completion") else "max_tokens"] = max_tok
    if prof.get("extra_body"):
        body.update(prof["extra_body"])
    if prof.get("_schema"):
        # guided decoding on VH's own reply schema -- the exact reply the assistant prompt asks for.
        # Without it most local models answer prose or half the fields (CookSim measured 8/24 for
        # Qwen3.8-27B in plain json_object mode); it changes the FORMAT, never what is asked.
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "assistant_reply", "strict": True, "schema": reply_schema()}}
    # A LOCAL vLLM server needs no credential; a HOSTED endpoint does. The key is read from the
    # environment (or the keys file) at call time and never logged, and it is sent only to a
    # non-loopback base URL, so a local arm can never leak it to a server on this box.
    headers = {}
    if not any(h in ASSIST_BASE_URL for h in ("127.0.0.1", "localhost", "0.0.0.0")):
        key = os.environ.get("IV_ASSIST_API_KEY") or keys.get("OPENAI_API_KEY", required=False)
        if not key:
            raise RuntimeError("hosted assistant endpoint needs a key: set IV_ASSIST_API_KEY")
        headers["Authorization"] = "Bearer " + key
    r = requests.post(ASSIST_BASE_URL.rstrip("/") + "/chat/completions", json=body,
                      headers=headers, timeout=240)
    if r.status_code >= 400:
        # raise_for_status() throws away the body, and the body is where the server says WHY.
        # A bare "400 Bad Request" cost a full sweep once: the real message was
        # "At most 8 image(s) may be provided in one prompt", i.e. the two map images pushed the
        # turn past the server's --limit-mm-per-prompt.
        raise requests.HTTPError(f"{r.status_code} from {ASSIST_MODEL}: {r.text[:300]}")
    r.raise_for_status()
    j = r.json()
    u = j.get("usage") or {}
    ASSIST_STATS["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
    ASSIST_STATS["completion_tokens"] += int(u.get("completion_tokens") or 0)
    return j["choices"][0]["message"].get("content") or ""


# --------------------------------------------------------------- streaming (bidi) backends
# Gemini Live and GPT Realtime are not model swaps: they speak a bidirectional WebSocket session
# rather than request/response, and gemini-3.1-flash-live-preview is not served on
# generateContent at all (404). They are wired here so the arm list can match the reference
# harness.
#
# ONE SESSION PER TURN, deliberately. The harness paradigm is stateless: every arm gets the
# system prompt, its own previous JSON, and the frame window, and nothing else. A persistent
# socket would hand these two arms a conversation memory no other arm has, and any gap that
# opened up would measure the memory rather than the model. The cost is handshake latency per
# turn, which is reported rather than hidden.

def _gemini_live(sys_prompt, user, frames, max_tok):
    """One assistant turn over the Gemini Live (bidiGenerateContent) socket."""
    import asyncio
    from google import genai
    from google.genai import types

    async def _run():
        # gemini-3.1-flash-live-preview REFUSES response_modalities=["TEXT"] (APIError 1007):
        # it is an audio-out model. So it speaks, and we read the server's own transcription of
        # what it said. That is the honest shape of this arm — a voice assistant — and it is why
        # its JSON compliance is not comparable to a text model's: the taxonomy label has to
        # survive being spoken aloud and transcribed back.
        # thinking_level="minimal" is REQUIRED here, not a tuning choice: without it the session
        # opens, accepts the frames, and then never reaches turn_complete inside 150 s, which
        # presents as a timeout rather than as a configuration error.
        # THE LIVE MODELS DISAGREE ABOUT THINKING (ported from CookSim, 2026-09-18):
        # gemini-3.1-flash-live-preview needs "minimal" (without it the session never reaches
        # turn_complete inside 150 s), gemini-3.8-live refuses ANY level, and the extended-thinking
        # model requires more than "minimal". Adjust once on refusal and keep it for the process.
        def _cfg():
            extra = {}
            if _LIVE_THINKING[0]:
                extra["thinking_config"] = types.ThinkingConfig(thinking_level=_LIVE_THINKING[0])
            return types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                output_audio_transcription=types.AudioTranscriptionConfig(),
                system_instruction=types.Content(parts=[types.Part(text=sys_prompt)]), **extra)

        cl = genai.Client(api_key=get_key(), http_options=types.HttpOptions(timeout=120_000))
        text = ""
        try:
            cm = cl.aio.live.connect(model=ASSIST_MODEL, config=_cfg())
            sess = await cm.__aenter__()
        except Exception as e:
            msg = str(e)
            if _LIVE_THINKING[0] and "Thinking level is not supported" in msg:
                _LIVE_THINKING[0] = None
            elif _LIVE_THINKING[0] == "minimal" and "MINIMAL is not supported" in msg:
                _LIVE_THINKING[0] = "low"
            else:
                raise
            cm = cl.aio.live.connect(model=ASSIST_MODEL, config=_cfg())
            sess = await cm.__aenter__()
        try:
            # THE FRAMES GO INLINE IN THE QUESTION TURN (ported from CookSim 55a56e0). Sent up
            # the realtime video channel they arrive AFTER the question has been answered: the first ask
            # sees 0 images and every later ask sees the previous ask's frames, i.e. the arm is blind and
            # nothing in the transcript says so.
            parts = [types.Part(inline_data=types.Blob(data=_scale_jpeg(f, ASSIST_IMAGE_SCALE),
                                                       mime_type="image/jpeg")) for f in frames]
            await sess.send_client_content(
                turns=types.Content(role="user", parts=parts + [types.Part(text=user)]),
                turn_complete=True)
            # receive() yields until turn_complete; the loop must break on it or it re-enters
            async with timeout_context(150):
                async for m in sess.receive():
                    um = getattr(m, "usage_metadata", None)
                    if um is not None:
                        ASSIST_STATS["prompt_tokens"] += int(um.prompt_token_count or 0)
                        ASSIST_STATS["completion_tokens"] += int(um.response_token_count or 0)
                    sc = getattr(m, "server_content", None)
                    if sc is None:
                        continue
                    if sc.model_turn is not None:
                        text += "".join(p.text or "" for p in (sc.model_turn.parts or [])
                                        if not getattr(p, "thought", False))
                    ot = getattr(sc, "output_transcription", None)
                    if ot is not None and ot.text:
                        text += ot.text
                    if sc.turn_complete:
                        break
        finally:
            await cm.__aexit__(None, None, None)
        return text

    return asyncio.run(_run())


def _openai_realtime(sys_prompt, user, frames, max_tok):
    """One assistant turn over the OpenAI Realtime socket."""
    import asyncio, base64, json
    import websockets

    key = os.environ.get("IV_ASSIST_API_KEY") or keys.get("OPENAI_API_KEY", required=False)
    if not key:
        raise RuntimeError("the realtime arm needs a key: set IV_ASSIST_API_KEY")

    async def _run():
        url = "wss://api.openai.com/v1/realtime?model=" + ASSIST_MODEL
        # No "OpenAI-Beta: realtime=v1": that wire shape is retired and the server now closes
        # the socket with 4000 beta_api_shape_disabled. The GA shape types the session and uses
        # output_modalities rather than modalities.
        hdr = {"Authorization": "Bearer " + key}
        text = ""
        async with websockets.connect(url, additional_headers=hdr, max_size=None,
                                      open_timeout=60, close_timeout=10) as ws:
            await ws.send(json.dumps({"type": "session.update",
                                      "session": {"type": "realtime",
                                                  "output_modalities": ["text"],
                                                  "instructions": sys_prompt}}))
            content = [{"type": "input_image",
                        "image_url": "data:image/jpeg;base64," +
                                     base64.b64encode(_scale_jpeg(f, ASSIST_IMAGE_SCALE)).decode()}
                       for f in frames]
            content.append({"type": "input_text", "text": user})
            await ws.send(json.dumps({"type": "conversation.item.create",
                                      "item": {"type": "message", "role": "user",
                                               "content": content}}))
            await ws.send(json.dumps({"type": "response.create",
                                      "response": {"output_modalities": ["text"],
                                                   "max_output_tokens": max_tok}}))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=150))
                t = m.get("type", "")
                if t.endswith("text.delta"):
                    text += m.get("delta") or ""
                elif t == "error" or t.endswith(".failed"):
                    err = (m.get("error") or {}).get("message") or json.dumps(m)[:200]
                    raise RuntimeError("realtime: " + str(err)[:200])
                elif t == "response.done":
                    u = ((m.get("response") or {}).get("usage")) or {}
                    ASSIST_STATS["prompt_tokens"] += int(u.get("input_tokens") or 0)
                    ASSIST_STATS["completion_tokens"] += int(u.get("output_tokens") or 0)
                    if not text:
                        for it in ((m.get("response") or {}).get("output") or []):
                            for c in (it.get("content") or []):
                                text += c.get("text") or c.get("transcript") or ""
                    break
        return text

    return asyncio.run(_run())


def assist_frames(client, sys_prompt, user, frames, max_tok=None):
    """The assistant's turn. Three attempts, then an EMPTY reply — the harness never repairs a
    model's output, because a repaired reply measures the repair and not the model."""
    max_tok = max_tok or ASSIST_MAX_TOK
    ASSIST_STATS["calls"] += 1
    if ASSIST_MODEL == "silent":                      # the silent floor control
        return {"text": "", "flag": None}
    t0 = time.time()
    try:
        for attempt in range(3):
            t_att = time.time()          # D3: only a SUCCESSFUL attempt's own time is charged
            try:
                if ASSIST_BACKEND == "anthropic":
                    from vhhome_bench.anthropic_backend import AnthropicBackend
                    out, meta = AnthropicBackend(ASSIST_MODEL, max_tokens=max_tok,
                                                 image_scale=ASSIST_IMAGE_SCALE).call(sys_prompt, user, frames)
                    if meta.get("stop_reason") == "refusal":
                        ASSIST_STATS["empty"] += 1
                        return {}
                elif ASSIST_BACKEND == "live":
                    out = _extract_json(_gemini_live(sys_prompt, user, frames, max_tok))
                elif ASSIST_BACKEND == "realtime":
                    out = _extract_json(_openai_realtime(sys_prompt, user, frames, max_tok))
                elif ASSIST_BASE_URL:
                    out = _extract_json(_openai_compatible(sys_prompt, user, frames, max_tok))
                else:
                    out = gemini_frames(client, sys_prompt, user, frames, max_tok=max_tok,
                                        model=(ASSIST_MODEL or None))
                if isinstance(out, dict) and out:
                    ASSIST_STATS["gen_s_turn"] = ASSIST_STATS.get("gen_s_turn", 0.0) + (time.time() - t_att)
                    return out
                ASSIST_STATS["schema_fail"] += 1
            except Exception as e:
                ASSIST_STATS["retries"] += 1
                if attempt == 2:
                    print(f"[assist] giving up after 3 attempts: {type(e).__name__}: "
                          f"{str(e)[:120]}", flush=True)
                    break
                time.sleep(2 + 3 * attempt)
        ASSIST_STATS["empty"] += 1
        return {}
    finally:
        ASSIST_STATS["latency_s"] += time.time() - t0



# ---------------------------------------------------------------- prompt capture (for the page)
# IV_CAPTURE_DIR=<dir>: every turn's EXACT prompt and observation, both sides, is written there
# as JSON + images, so a prompt page can show what each agent really saw rather than a paraphrase.
def _capture(side, tick, system, user, frames=None, images=None, extra=None):
    d = os.environ.get("IV_CAPTURE_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        rec = {"side": side, "tick": tick, "system": system, "user": user, "extra": extra or {}}
        fr = []
        for i, f in enumerate(frames or []):
            fp = os.path.join(d, f"{side}_t{tick:03d}_frame{i}.jpg")
            open(fp, "wb").write(f if isinstance(f, (bytes, bytearray)) else open(f, "rb").read())
            fr.append(os.path.basename(fp))
        rec["frames"] = fr
        for k, im in (images or {}).items():
            if im is None:
                continue
            fp = os.path.join(d, f"{side}_t{tick:03d}_{k}.png")
            open(fp, "wb").write(im if isinstance(im, (bytes, bytearray)) else open(im, "rb").read())
            rec[k] = os.path.basename(fp)
        json.dump(rec, open(os.path.join(d, f"{side}_t{tick:03d}.json"), "w"), indent=1)
    except Exception as e:
        print(f"[capture] {side} t{tick}: {e}", flush=True)

def sup_turn(client, sup_sys, prev_json, comm, said, frames, now, state_extra=None,
             map_img=None, map_text=None, house_img=None, house_text=None):
    hist = "\n".join(f'  t{c["t"]} {c["speaker"]}: "{c["text"]}"' for c in comm)[-2400:]
    user = (f"CURRENT TIMESTAMP: t={now}\n\n"
            "== YOUR PREVIOUS OUTPUT (your memory — update human_progress, then reply) ==\n"
            f"{prev_json}\n\n"
            "== COMMUNICATION HISTORY (public — everything said so far) ==\n"
            + (hist or "  (nothing yet)")
            + (f'\n\n== THE HUMAN JUST SAID ==\n  "{said}"' if said else ""))
    imgs = list(frames) + [m for m in (house_img, map_img) if m is not None]
    tail = (f"\n\nBelow are the last {len(frames)} FIRST-PERSON frames (one per second, oldest "
            "first)")
    trailing = [n for n, m in (("whole-house overview map", house_img),
                               ("top-down map of the person's current room", map_img))
                if m is not None]
    if trailing:
        tail += ", followed by the " + " and the ".join(trailing)
    tail += ". Output your JSON for NOW."
    _capture("assistant", now, sup_sys, user + tail, frames,
             {"house_map": house_img, "room_map": map_img})
    return assist_frames(client, sup_sys, user + tail, imgs)
