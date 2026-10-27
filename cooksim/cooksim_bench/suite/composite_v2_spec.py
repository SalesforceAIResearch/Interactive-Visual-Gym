"""Composite suite v2.2 — every episode designed by hand (rebalanced + audited).

RETIRED, KEPT AS HISTORY: the `search` error type was deleted everywhere on 2026-09-17 (user ruling), and
19 episodes below use it, so this spec can no longer be regenerated as-is — design_composite_v2.py would
fail on the missing error function. It is left untouched on purpose: it is the exact design behind the
v2 results pages. The live suite is bench/v5/cases_composite_v4.json.

Format (consumed by python -m cooksim_bench.suite.design_composite_v2):
    EPISODES = [ {"cell": ..., "errors": [(label, kwargs), (label, kwargs)],
                  "qa": [(gt_kind, question, anchor_m), (gt_kind, question, anchor_m)],
                  "story": "why these two errors, this order, these questions"} ... ]

Design rules
  * two errors per episode with different labels; across the suite every (cell, label,
    kwargs) instance is used exactly once;
  * label budget per cell so that no label dominates (v2.0 had 20 omissions and 19 hunts out
    of 100; now no label exceeds 18), subject to what each cell physically allows (the cold
    caprese cells only admit mis_prep, premature serve, forget and search; add_foreign needs
    an off-recipe source on the map; wrong_order and repeat need an ordered recipe;
    leave_unattended needs a component on heat);
  * the two errors sit at different moments of the plan and never at the same anchor;
  * a premature serve (omit 'after' X) is always the LAST sub of its episode: it is a
    commit-or-catch event and, if it commits, the episode ends — nothing may depend on it;
  * an omission of a MIDDLE component is never used on an order-free dish (caprese, fish and
    chips, steak and chips): there the skip is indistinguishable from a premature serve until
    one tick before the serve, and the forced skip re-orders the plate so nothing scheduled
    after it can fire (2026-08-26 audit);
  * two questions per episode of different kinds, anchored where the answer is discriminating;
    at a shared anchor the designer schedules the question BEFORE the error; the live harness
    recomputes the truth at ask time from the live kitchen and the cook's own log, so a
    question asked after a recovery grades against what really happened (some stories rely on
    that on purpose); answers are deliberately mixed (loaded and empty boards, plates with and
    without food, steps done and not yet done, places visited and never visited);
  * questions are in the cook's words.

Anchor m = index into the flat clean command list (plate-keyed in the live harness):
  easy   m=0 prep | 7 plate in hand | 9 tomato on | 11 tomato+cheese on
  medium m=0 prep | 7 plate | 9 bun | 11 bun+meat | 13 bun+meat+cheese
  hard   m=0 prep | 26 plate | 28 fish | 30 fish+2 potato
  expert m=0 prep | 30 plate | 32 steak | 34 steak+2 potato
  master m=0 prep | 28 plate | 30 tortilla | 32 +rice | 34 +steak | 36 +tomato | 38 +lettuce
"""

E = []


def ep(cell, errors, qa, story):
    E.append({"cell": cell, "errors": errors, "qa": qa, "story": story})


# =============================================================================================
# EASY / medium map — caprese (cold, order-free): prep(tomato) → plate → tomato → cheese → serve
# Budget: mis_prep 1, premature serve 1, forget 3, search 5 (the 5-tick empty-plate serve retired).
# =============================================================================================
C = "easy_nops_medium_map"
ep(C, [("mis_prep", {"component": "tomato", "wrong": "cheese"}),
       ("search", {"before": "serve", "looks": 2})],
   [("board_state", "Is there still anything sitting on a cutting board, and which one?", 7),
    ("remaining", "Tell me exactly what still has to go on this plate.", 9)],
   "Opens with the only prep-time mistake this dish allows (cheese goes to the board instead of "
   "the tomato); after the recovery the chopped tomato is still on the far board, which is what "
   "the first question probes (a loaded board, not an empty one). At the end, plate complete, "
   "the cook wanders two wrong counters looking for the pass.")

ep(C, [("forget_next_step", {"after": "fetch_plate"}),
       ("search", {"before": "serve", "looks": 3})],
   [("station_state", "Is anything cooking anywhere right now, or is this dish all cold?", 9),
    ("where:serving", "Which way is the serving window from here?", 11)],
   "The cook picks up the plate and goes idle, is talked back into motion, and then, once the "
   "plate is complete, wanders three wrong counters looking for the pass. The station question "
   "sits between the two errors and is a trap for an assistant that assumes every recipe has "
   "heat; the last question is asked just before the hunt (question first at a shared anchor) "
   "and is the one the lost cook should actually ask. (The premature empty-plate serve that "
   "used to open this episode gave the assistant five ticks on this map — 3 catches in 59 "
   "runs — and was retired as unmeasurable.)")

ep(C, [("search", {"before": "assemble(tomato)", "looks": 3}),
       ("forget_next_step", {"after": "assemble(tomato)"})],
   [("visited:tomato", "Have I already been over to the tomato?", 7),
    ("plate_contents", "Read me back what is on the plate right now, in order.", 11)],
   "Three wrong counters looking for the chopped tomato with the plate in hand, then, tomato "
   "on, a stall with the cheese still to go. The visited question is asked before the hunt and "
   "its honest answer is YES — the cook chopped the tomato at the very start — so an assistant "
   "that only reads the current scene gets it wrong.")

ep(C, [("search", {"before": "assemble(tomato)", "looks": 2}),
       ("omit_subgoal", {"after": "assemble(tomato)"})],
   [("remaining", "What is still left to put on this plate?", 9),
    ("past_subgoal:prep(tomato)", "Did I actually chop the tomato, or did I skip that?", 9)],
   "The cook cannot find the chopped tomato after picking up the plate (two wrong counters), "
   "and then, having found it, walks to the pass with one item on and one missing. Both "
   "questions are asked just before that premature run — a premature serve is always the LAST "
   "sub of its episode, because if it commits the episode is over — and the second is a "
   "memory question about a step done at the very start.")

ep(C, [("search", {"before": "assemble(cheese)", "looks": 3}),
       ("forget_next_step", {"after": "assemble(cheese)"})],
   [("holding", "What am I holding right now and is there anything on it?", 7),
    ("where:trash", "Where is the trash can, in case I need it?", 9)],
   "Wander-then-freeze: with the tomato on, the cook hunts for the cheese in three wrong "
   "places, and then, plate complete, simply stalls — nothing wrong is done, so the assistant "
   "must notice an absence of action at the very end. The holding question is asked the "
   "moment the plate is picked up (a plate, nothing on it); the trash question is a map "
   "question no recipe step ever requires, asked just before the hunt.")

# =============================================================================================
# EASY / hard map — same dish, harder kitchen (cheese far bottom-left, pass top-right).
# Budget: mis_prep 1, premature serve 2 (always the last sub), forget 3, search 4.
# =============================================================================================
C = "easy_nops_hard_map"
ep(C, [("mis_prep", {"component": "tomato", "wrong": "cheese"}),
       ("omit_subgoal", {"after": "fetch_plate"})],
   [("board_state", "Which board has the tomato on it — is it still there?", 7),
    ("where:cheese", "Where is the cheese kept in this kitchen?", 7)],
   "Prep-time slip (cheese to the board instead of the tomato) on a layout where the wrong "
   "ingredient is a long walk away; then, plate in hand, the cook heads straight for the pass "
   "with nothing on it. Both questions are asked at the plate pickup, before that run: the "
   "loaded board and the far-away cheese (bottom-left). The premature serve is the last sub — "
   "if it commits, nothing else is lost.")

ep(C, [("search", {"before": "assemble(cheese)", "looks": 3}),
       ("forget_next_step", {"after": "assemble(cheese)"})],
   [("visited:cheese", "Have I already visited the cheese?", 9),
    ("plate_contents", "What exactly is on the plate now, bottom to top?", 11)],
   "With the tomato on, the cook hunts three wrong counters for the cheese, then stalls with "
   "the completed plate. The visited question is asked just before the hunt (no); the "
   "read-back just before the stall.")

ep(C, [("search", {"before": "assemble(tomato)", "looks": 3}),
       ("omit_subgoal", {"after": "assemble(tomato)"})],
   [("station_state", "Is there anything on the stove I should be watching?", 7),
    ("remaining", "What is still missing from this plate?", 9)],
   "Three-counter hunt for the tomato with the empty plate, then an early run to the pass with "
   "one item on. The stove question is a no-heat trap asked at the plate pickup; the remaining "
   "question is asked with one item on, just before the premature run (last sub).")

ep(C, [("forget_next_step", {"after": "fetch_plate"}),
       ("search", {"before": "assemble(cheese)", "looks": 2})],
   [("holding", "What have I got in my hands?", 7),
    ("where:serving", "Which corner is the serving window in?", 11)],
   "Idle with an empty plate, then lost on the way to the cheese. Both questions are the ones a "
   "disoriented cook asks: what am I holding, and where do I take it.")

ep(C, [("forget_next_step", {"after": "assemble(tomato)"}),
       ("search", {"before": "serve", "looks": 2})],
   [("past_subgoal:assemble(cheese)", "Did the cheese go on already?", 9),
    ("where:trash", "Where is the trash in this kitchen?", 11)],
   "Freezes halfway through plating, then, plate complete, hunts two wrong counters for the "
   "pass. The cheese question is asked with the tomato on, just before the freeze — a memory "
   "question whose answer is NOT YET; the trash question is a map question no recipe step "
   "requires, asked just before the hunt.")

# =============================================================================================
# MEDIUM / medium map — burger (ordered: bun, meat, cheese; lettuce on the map is off-recipe)
# Budget: mis_prep 2, wrong_order 2, add_foreign 2, repeat 1, leave 1, forget 1, omit 1.
# =============================================================================================
C = "medium_nops_medium_map"
ep(C, [("mis_prep", {"component": "meat", "wrong": "lettuce"}),
       ("leave_unattended", {"component": "meat", "loops": 2})],
   [("station_state", "Is the patty done, or is it still cooking?", 7),
    ("remaining", "What still goes on this burger, in order?", 11)],
   "Two errors on the same component at two different stages: the cook first brings lettuce to "
   "the board instead of the meat, then, with the patty on the pan and the bun plated, walks "
   "off for two loops of busywork while it sits on the heat. The station question is asked "
   "while the pan reads COOKING (not done) — the assistant must say so, not guess.")

ep(C, [("wrong_order", {"a": "bun", "b": "meat"}),
       ("add_foreign", {"wrong": "lettuce", "before": "serve"})],
   [("where:trash", "Where's the trash, in case I have to bin something?", 9),
    ("plate_contents", "Read the plate back to me before I serve it.", 13)],
   "Patty heads for the empty plate before the bun — a stacking-order error that looks like "
   "progress — and at the very end lettuce (which this burger does not take) is grabbed on the "
   "way to the pass. The trash question is natural right after a plate had to be discarded; "
   "the read-back is asked BEFORE the lettuce detour (question first at the shared anchor), so "
   "an assistant that answers it well has the plate fresh in mind when the lettuce appears.")

ep(C, [("omit_subgoal", {"sid": "assemble(bun)"}),
       ("forget_next_step", {"after": "assemble(meat)"})],
   [("station_state", "Is the meat still on the pan?", 9),
    ("past_subgoal:assemble(bun)", "Did the bun go on already?", 11)],
   "The bun is skipped entirely, so the plate would start with meat; after the meat the cook "
   "stops dead. Both questions are about the bun/meat confusion: the pan question is asked "
   "with the bun on (the patty is on the pan unless the omission's recovery replaced it — the "
   "truth is read live); the bun question is asked just before the stall.")

ep(C, [("mis_prep", {"component": "meat", "wrong": "cheese"}),
       ("repeat_step", {"sid": "assemble(bun)", "at": "assemble(cheese)"})],
   [("where:bun", "Where do I get the bun from?", 7),
    ("remaining", "Anything else to add before I serve this?", 13)],
   "Cheese is confused with meat at the board at the start; at the end, with the burger "
   "complete, the cook reaches for a second bun. The where question precedes the bun trip; the "
   "last question is asked just before the second-bun detour and its correct answer is "
   "'nothing — it is complete'.")

ep(C, [("wrong_order", {"a": "bun", "b": "cheese"}),
       ("add_foreign", {"component": "meat", "wrong": "lettuce"})],
   [("holding", "What is on the plate I'm holding?", 7),
    ("visited:cheese", "Did I go to the cheese yet?", 11)],
   "Cheese first onto the empty plate, then lettuce plated in place of the patty (a "
   "substitution, not an addition). The visited question is asked with bun+meat on: in the "
   "clean plan the cheese has not been touched yet, but this cook went for the cheese too "
   "early — the truth is recomputed live from the cook's own log, so the honest answer is "
   "'yes, you went there, but nothing from it is on the plate'.")

# =============================================================================================
# MEDIUM / hard map — burger, harder layout (bun bottom-right, cheese top-right, pan centre);
# no trash tile: a spoiled plate is set down and a fresh one fetched. No off-recipe source.
# Budget: mis_prep 1, wrong_order 3, leave 2, repeat 1, search 1, forget 1, omit 1.
# =============================================================================================
C = "medium_nops_hard_map"
ep(C, [("mis_prep", {"component": "meat", "wrong": "cheese"}),
       ("wrong_order", {"a": "meat", "b": "cheese"})],
   [("station_state", "Is the patty ready to come off?", 7),
    ("plate_contents", "What's on the plate so far, in order?", 11)],
   "Cheese is confused with meat twice: first at the board (cheese prepped instead of meat), "
   "then at the plate (cheese laid before the patty). The pan is already COOKED at the plate "
   "anchor on this map, so the answer to the first question is 'yes, take it'.")

ep(C, [("leave_unattended", {"component": "meat", "loops": 1}),
       ("omit_subgoal", {"after": "assemble(meat)"})],
   [("where:serving", "Where's the serving counter in this kitchen?", 7),
    ("remaining", "Cheese still to go, then done?", 11)],
   "A single busywork loop away from the patty on the pan — the shortest leave_unattended — "
   "followed by walking to the pass with bun+meat and no cheese. The serving question is asked "
   "early on a map where the pass is on the far left; the leading question just before the "
   "premature run has a true presupposition.")

ep(C, [("wrong_order", {"a": "bun", "b": "cheese"}),
       ("leave_unattended", {"component": "meat", "loops": 3})],
   [("holding", "What am I carrying right now?", 11),
    ("past_subgoal:prep(meat)", "Did I ever actually cook the meat?", 13)],
   "Cheese heads for the empty plate before the bun; then, bun on, the cook drifts for three "
   "loops while the cooked patty sits on the heat. The holding question comes with bun and "
   "meat on the plate; the memory question at the end asks about the prep step from the very "
   "beginning (truth: yes).")

ep(C, [("repeat_step", {"sid": "assemble(bun)", "at": "assemble(meat)"}),
       ("search", {"before": "serve", "looks": 3})],
   [("where:bun", "Which way to the bun?", 7),
    ("station_state", "Did I leave anything on the pan?", 9)],
   "A second bun right after the meat, then a three-counter hunt for the pass with the finished "
   "burger. Both questions come BEFORE the errors: the bun's location (bottom-right) and the "
   "pan read-out with the cooked patty still on it.")

ep(C, [("wrong_order", {"a": "bun", "b": "meat"}),
       ("forget_next_step", {"after": "assemble(cheese)"})],
   [("remaining", "What's still to go after the bun?", 9),
    ("visited:cheese", "Have I been to the cheese already?", 11)],
   "The patty heads for the bare plate before the bun; later a stall with the finished plate. "
   "The visited-cheese question at bun+meat is 'not yet' — the earlier detour went to the "
   "pan, not the cheese.")

# =============================================================================================
# HARD / medium map — fish & chips (order-free): prep fish, 2x potato, ignite both, plate,
# fish, potato, serve. Budget: leave 4 (fish 2|3, potato 2|3), premature serve 1 (last sub),
# search 4, forget 1. (No skip of the fish: on an order-free dish it is a premature serve in disguise.)
# =============================================================================================
C = "hard_nops_medium_map"
ep(C, [("leave_unattended", {"component": "fish", "loops": 2}),
       ("omit_subgoal", {"after": "assemble(fish)"})],
   [("station_state", "What's in the fryers right now and is any of it done?", 26),
    ("remaining", "How many more things go on this plate, and what are they?", 28)],
   "Two fryers at once: the cook walks away from the fish while it fries, then heads for the "
   "pass with fish alone. The station question (two fryers, fish nearly done, chips further "
   "behind) is the hardest state read-out in this cell; the remaining question, asked just "
   "before the premature run, has a count in its answer (two).")

ep(C, [("search", {"before": "assemble(fish)", "looks": 3}),
       ("leave_unattended", {"component": "potato", "loops": 3})],
   [("past_subgoal:prep(potato)#2", "Did I cut the second potato, or just one?", 26),
    ("where:trash", "Where is the trash?", 30)],
   "Lost looking for the fish fryer with an empty plate, then, with fish on, wandering three "
   "loops while the chips fry. The memory question is about the duplicated prep step.")

ep(C, [("search", {"before": "assemble(fish)", "looks": 2}),
       ("forget_next_step", {"after": "assemble(potato)"})],
   [("visited:lettuce", "Have I gone over to the lettuce at any point?", 26),
    ("plate_contents", "What's on the plate, in order?", 30)],
   "A shorter hunt for the fish fryer with the plate in hand, then a freeze with the full "
   "plate. The visited question is about a distractor ingredient never used (no); the "
   "read-back is asked just before the freeze.")

ep(C, [("leave_unattended", {"component": "potato", "loops": 2}),
       ("search", {"before": "serve", "looks": 3})],
   [("station_state", "Are the chips done yet?", 28),
    ("where:serving", "Which way is the pass from the fryers?", 30)],
   "With one item plated, two loops away from the frying chips; then, plate complete, a "
   "three-counter hunt for the pass. The chips question is asked at exactly the moment the "
   "potato fryer is the only station with food, just before the cook wanders; the pass on this "
   "map is at the top, far from the fryers, and is asked about just before the hunt.")

ep(C, [("leave_unattended", {"component": "fish", "loops": 3}),
       ("search", {"before": "serve", "looks": 2})],
   [("past_subgoal:ignite(potato)", "Is the chip fryer actually switched on already?", 26),
    ("holding", "What am I holding and what's on it?", 28)],
   "Three loops away from the frying fish with an empty plate in hand, then a hunt for the "
   "serving counter with the finished plate. The ignite question is a memory check about the "
   "second fryer while the first is the thing at risk; the holding question comes with the "
   "fish on the plate.")

# =============================================================================================
# HARD / hard map — fish & chips with off-recipe lemon, peas, bun, meat, noodles, shrimp
# Budget: mis_prep 2, add_foreign 4, leave 2, search 1, forget 1.
# =============================================================================================
C = "hard_nops_hard_map"
ep(C, [("mis_prep", {"component": "fish", "wrong": "shrimp"}),
       ("add_foreign", {"wrong": "lemon", "before": "serve"})],
   [("station_state", "Which fryer has what in it, and what's done?", 26),
    ("plate_contents", "Read the plate back before I go to the pass.", 30)],
   "Shrimp goes into the fryer instead of fish — a look-alike substitution at prep time — and "
   "a lemon is added at the very end. Both are 'plausible garnish' mistakes the assistant must "
   "reject on recipe grounds; the read-back is asked before the lemon detour.")

ep(C, [("add_foreign", {"component": "fish", "wrong": "meat"}),
       ("leave_unattended", {"component": "potato", "loops": 3})],
   [("where:meat", "Where is the meat kept — is that the fish?", 26),
    ("remaining", "What's still to go on?", 28)],
   "A burger patty is plated in place of the fish (a cheap recovery: raw meat to the trash, "
   "the fish still in its fryer — which is smoking by then, a second cue); then the cook "
   "abandons the frying chips for three loops. The where question is a leading question asked "
   "just before the detour: the meat is top-left and is not the fish.")

ep(C, [("mis_prep", {"component": "fish", "wrong": "potato"}),
       ("add_foreign", {"wrong": "peas", "before": "assemble(potato)"})],
   [("visited:pan", "Have I used the pan at all so far?", 26),
    ("past_subgoal:prep(potato)#2", "Both potatoes are cut, right — not just one?", 30)],
   "A third potato is cut where the fish should be, and later peas go on before the chips. "
   "The pan question probes memory of a station this dish never uses (everything is fried: "
   "no); the last question is about the duplicated prep step.")

ep(C, [("search", {"before": "assemble(fish)", "looks": 2}),
       ("add_foreign", {"wrong": "noodles", "before": "assemble(potato)"})],
   [("station_state", "Is the fish still in the fryer, and the chips?", 26),
    ("past_subgoal:ignite(fish)", "Did I actually turn the fryer on under the fish?", 30)],
   "A hunt for the fish fryer with the empty plate, then noodles go on before the chips — a "
   "plate that ends up looking full but wrong. The station question is asked before the hunt "
   "(both fryers loaded); the memory question at the end is answered yes.")

ep(C, [("leave_unattended", {"component": "fish", "loops": 3}),
       ("forget_next_step", {"after": "assemble(potato)"})],
   [("holding", "What have I got in hand?", 26),
    ("where:serving", "Where do I serve this?", 30)],
   "Walks away from the fish for three loops, then freezes with the finished plate. The pass on "
   "this map is middle-left, opposite the fryers; the question is asked just before the freeze.")

# =============================================================================================
# EXPERT / medium map — steak & chips (order-free)
# Budget: leave 4 (steak 2|3, potato 2|3), search 3, premature serve 1, mis_prep 1, forget 1.
# =============================================================================================
C = "expert_nops_medium_map"
ep(C, [("mis_prep", {"component": "steak", "wrong": "potato"}),
       ("leave_unattended", {"component": "potato", "loops": 2})],
   [("station_state", "Is the steak done or does it need longer?", 30),
    ("remaining", "What's still to go on the plate?", 32)],
   "A third potato is cut where the steak should be, and later the chips are left in the "
   "fryer for two loops. The steak question is asked when the pan reads COOKED (take it off); "
   "the remaining question is asked just before the chips are abandoned.")

ep(C, [("leave_unattended", {"component": "steak", "loops": 3}),
       ("omit_subgoal", {"after": "assemble(steak)"})],
   [("past_subgoal:ignite(potato)", "Did I switch the fryer on for the chips?", 30),
    ("plate_contents", "What's on the plate right now?", 32)],
   "Abandons the steak on the pan for three loops, then heads for the pass with one item on. "
   "The memory question is about the chip fryer, asked while the steak is the thing at risk; "
   "the read-back is asked just before the premature run (last sub).")

ep(C, [("search", {"before": "assemble(steak)", "looks": 2}),
       ("forget_next_step", {"after": "assemble(potato)"})],
   [("where:pan", "Where's the pan?", 30),
    ("station_state", "Is anything still cooking now, or is it all off?", 34)],
   "Cannot find the pan with the plate in hand (this map has three pans; the steak is on the "
   "centre one), then freezes with the finished plate. The last question is the end-of-service "
   "check (every station empty), asked just before the freeze.")

ep(C, [("search", {"before": "assemble(steak)", "looks": 3}),
       ("leave_unattended", {"component": "potato", "loops": 3})],
   [("holding", "What's in my hands right now?", 32),
    ("visited:lettuce", "Did I ever go over to the lettuce?", 34)],
   "A longer pan hunt with the empty plate, then, with the steak on, three loops away from "
   "the chips. The holding question comes with the steak on the plate, just before the cook "
   "wanders off; the lettuce is a distractor never used (no).")

ep(C, [("leave_unattended", {"component": "steak", "loops": 2}),
       ("search", {"before": "serve", "looks": 3})],
   [("past_subgoal:prep(potato)#2", "Did I cut both potatoes or only one?", 30),
    ("where:serving", "Which way to the serving counter?", 34)],
   "Two loops away from the steak on the pan with an empty plate, then a hunt for the pass "
   "with the finished plate — right after asking where it is.")

# =============================================================================================
# EXPERT / hard map — steak & chips with butter, pepper, egg, meat, mushroom, onion around
# Budget: mis_prep 3, add_foreign 4, leave 3.
# =============================================================================================
C = "expert_nops_hard_map"
ep(C, [("mis_prep", {"component": "steak", "wrong": "mushroom"}),
       ("add_foreign", {"wrong": "butter", "before": "serve"})],
   [("station_state", "Is the steak done, or still cooking? And the chips?", 30),
    ("plate_contents", "Read me the plate before I serve.", 34)],
   "Mushroom goes to the board instead of the steak; a knob of butter goes on at the end. Both "
   "are the kind of thing a real cook might defend ('it's a steak, butter belongs') — the "
   "assistant has to hold the recipe line. The read-back is asked before the butter detour.")

ep(C, [("leave_unattended", {"component": "potato", "loops": 2}),
       ("add_foreign", {"wrong": "pepper", "before": "serve"})],
   [("where:meat", "Is the meat over there the steak?", 30),
    ("remaining", "What else goes on?", 32)],
   "The chips are abandoned for two loops with the steak plated, and pepper is added on top at "
   "the end. The where question is a leading confusion asked at the plate pickup — the meat "
   "source (top) is not the steak, which is on the pan.")

ep(C, [("mis_prep", {"component": "steak", "wrong": "onion"}),
       ("add_foreign", {"wrong": "egg", "before": "assemble(potato)"})],
   [("past_subgoal:prep(potato)#2", "Did I cut both potatoes?", 30),
    ("visited:oven", "Have I been near the oven?", 34)],
   "Onion is prepped in place of the steak, and later a fried egg is slipped in before the "
   "chips. The oven is a distractor station on this map that the plan never uses (visited=no).")

ep(C, [("mis_prep", {"component": "steak", "wrong": "potato"}),
       ("leave_unattended", {"component": "steak", "loops": 2})],
   [("station_state", "Is the steak done, and how far along are the chips?", 30),
    ("where:serving", "Where's the pass on this map?", 34)],
   "A third potato is cut where the steak should be; much later, with the plate in hand, the "
   "cook walks away from the steak on the pan for two loops. The station question is asked "
   "just before that: pan COOKED, fryer still COOKING — two different answers.")

ep(C, [("leave_unattended", {"component": "steak", "loops": 3}),
       ("add_foreign", {"component": "potato", "wrong": "onion"})],
   [("holding", "What's in my hands?", 32),
    ("past_subgoal:assemble(potato)", "Have the chips gone on yet?", 32)],
   "Three loops away from the steak with the empty plate, then onion plated in place of the "
   "chips. Both questions are asked with the steak on the plate, just before the onion "
   "detour: the holding answer is a loaded plate and the chips question's answer is NOT YET.")

# =============================================================================================
# MASTER / medium map — burrito (ordered): tortilla, rice, steak, tomato, lettuce; no trash
# Budget: wrong_order 3, mis_prep 2, repeat 2, leave 2, forget 1.
# =============================================================================================
C = "master_nops_medium_map"
ep(C, [("mis_prep", {"component": "rice", "wrong": "tomato"}),
       ("wrong_order", {"a": "steak", "b": "tomato"})],
   [("station_state", "Is the rice done, and what about the steak?", 28),
    ("remaining", "Tell me the rest of the stack, in order.", 32)],
   "Tomato is brought to the board where rice should be washed; later the tomato is reached "
   "for before the steak. At the plate anchor both the pot and the pan are still COOKING — "
   "the assistant must say neither is ready; the remaining question (three items, ordered) "
   "is asked just before the order slip.")

ep(C, [("leave_unattended", {"component": "steak", "loops": 3}),
       ("repeat_step", {"sid": "assemble(tortilla)", "at": "assemble(tomato)"})],
   [("board_state", "Which boards still have chopped things on them, and what?", 30),
    ("past_subgoal:prep(lettuce)", "Did I already shred the lettuce?", 36)],
   "The cooked steak is left on the pan for three loops; then, near the end, a second tortilla "
   "is reached for right after the tomato. The board question is asked while two boards still "
   "hold tomato and lettuce — the assistant must list both — and the lettuce question is a "
   "memory test on the last prep step, asked just before the second-tortilla slip.")

ep(C, [("wrong_order", {"a": "tortilla", "b": "rice"}),
       ("repeat_step", {"sid": "assemble(tortilla)", "at": "assemble(steak)"})],
   [("where:lettuce", "Where's the lettuce in this kitchen?", 28),
    ("plate_contents", "What's on the tortilla so far?", 34)],
   "Rice toward the bare plate before the tortilla, then a second tortilla reached for right "
   "after the steak. The plate read-back (three deep: tortilla, rice, steak) is asked just "
   "before that.")

ep(C, [("wrong_order", {"a": "rice", "b": "lettuce"}),
       ("forget_next_step", {"after": "assemble(tomato)"})],
   [("station_state", "Is the steak still on the pan?", 32),
    ("past_subgoal:ignite(rice)", "Did I actually turn the pot on under the rice?", 34)],
   "Lettuce reached for right after the tortilla (three layers too early), then a freeze one "
   "step from the end. The pan question is asked with the cooked steak still on it; the pot "
   "question is a memory check on a step done long before.")

ep(C, [("mis_prep", {"component": "rice", "wrong": "lettuce"}),
       ("leave_unattended", {"component": "rice", "loops": 2})],
   [("holding", "What am I holding right now?", 28),
    ("remaining", "After the steak goes on, what remains?", 34)],
   "Lettuce is brought to the board where the rice should be, and later the rice is left "
   "boiling for two loops with the tortilla plated. The remaining question has a two-item "
   "ordered answer.")

# =============================================================================================
# MASTER / hard map — burrito with avocado, corn, bun, cheese, dough, meat, onion, shrimp around
# (the rice pot is already past its grace and smoking at the plate anchor on this map)
# Budget: add_foreign 3, mis_prep 2, wrong_order 2, repeat 1, leave 1, forget 1.
# =============================================================================================
C = "master_nops_hard_map"
ep(C, [("mis_prep", {"component": "rice", "wrong": "onion"}),
       ("add_foreign", {"wrong": "cheese", "before": "assemble(lettuce)"})],
   [("station_state", "Is the rice ready and is the steak still on?", 28),
    ("plate_contents", "Read me the stack so far.", 36)],
   "Onion is prepped in place of the rice; cheese is reached for before the lettuce. Both are "
   "burrito ingredients in other kitchens — the assistant has to reject on this recipe, not on "
   "taste. The four-deep read-back is asked just before the cheese detour.")

ep(C, [("leave_unattended", {"component": "rice", "loops": 1}),
       ("add_foreign", {"component": "steak", "wrong": "meat"})],
   [("board_state", "What's on the three boards right now?", 28),
    ("remaining", "After the rice, what's the order?", 32)],
   "The rice — already smoking on this map — is left for one more loop with the tortilla "
   "plated, then the burger patty is reached for in place of the steak. The board question at "
   "the plate anchor has two loaded boards (tomato, lettuce) and one empty; the remaining "
   "question (three items) is asked just before the patty detour.")

ep(C, [("wrong_order", {"a": "tortilla", "b": "steak"}),
       ("add_foreign", {"wrong": "avocado", "before": "assemble(tomato)"})],
   [("where:tomato", "Where's the tomato I chopped?", 30),
    ("past_subgoal:assemble(lettuce)", "Is the lettuce on yet?", 34)],
   "Steak toward the bare plate before the tortilla, then avocado — the most plausible burrito "
   "addition — before the tomato. The where question is answered by the loaded board (center); "
   "the lettuce question, asked just before the avocado detour, is answered NOT YET.")

ep(C, [("mis_prep", {"component": "rice", "wrong": "cheese"}),
       ("repeat_step", {"sid": "assemble(tortilla)", "at": "assemble(rice)"})],
   [("station_state", "Is the steak done yet?", 32),
    ("visited:pot", "Have I been back to the pot at all?", 36)],
   "Cheese is prepped in place of the rice at the start; later a second tortilla is reached "
   "for right after the rice. The steak question is asked just before that slip (pan COOKED); "
   "the pot question is a memory question whose answer is YES — the cook has been there for "
   "the rice twice.")

ep(C, [("wrong_order", {"a": "rice", "b": "tomato"}),
       ("forget_next_step", {"after": "assemble(lettuce)"})],
   [("holding", "What's in my hand?", 32),
    ("remaining", "Only lettuce left?", 36)],
   "Tomato reached for right after the tortilla, skipping two layers; at the very end the "
   "complete burrito just sits. The holding question comes with tortilla and rice on the "
   "plate; the last question is leading and true: yes, only lettuce.")

EPISODES = E
