"""COMPOSITE (multi-error) benchmark spec for the VirtualHome streaming benchmark.

Every episode here carries TWO OR MORE injected errors of DIFFERENT classes. That is the whole
point of the composite suite: the single-error cases measure "can the assistant catch one thing";
these measure whether it can hold a running model of the task while more than one thing goes
wrong — notice the second error after committing to the first, avoid attributing both to one
cause, and keep the repair of one from masking the other.

Multi-error episodes are only meaningful if the errors do not collide. Five rules, each of which
the verifier checks mechanically (tools/tmp/verify_composite_vh.py):

  R1 DIRECT-TWIN EXCLUSION.  No error may be an observation-channel TWIN of another in the same
     episode (twins are the labels the grader already credits for one another). If e2 is a twin
     of e1, a correct flag naming e2 is creditable for BOTH and the episode cannot say which the
     assistant saw. Sharing a more distant alias is allowed, because R3 separates the windows in
     time and a flag is attributed to the window it lands in.
  R2 ANCHOR SURVIVAL.  Composition applies transforms in order under the priority rule (an
     earlier error owns the plan; a later error whose anchor it removed is skipped-with-record).
     Every declared error must therefore still be present in the composed plan. This is what
     forbids anchoring anything after `wrong_placement`, which TRUNCATES the plan at the
     placement, and anything on a subgoal an omission removes.
  R3 WINDOW SEPARATION.  The errors' anchors must be distinct subgoals separated by at least one
     surviving subgoal, so their detection windows do not overlap and each flag attributes to
     exactly one error.
  R4 INDEPENDENT OBSERVABILITY.  No two errors may act on the same target object, and an earlier
     error must not make a later one unobservable (e.g. skipping the fetch makes a later wrong
     placement of that object impossible — the deposit simply refuses empty-handed).
  R6 VISIBLE STATE.  A repetition's inverting press must CHANGE THE PICTURE. A microwave,
     dishwasher or stove switch renders identically to not pressing it (engine note A7), so a
     repetition there has no evidence on camera; lamps, the TV and container doors do change it.
  R7 THE MISPLACEMENT MUST MOVE SOMETHING.  A wrong placement onto the same kind of surface the
     object started on is a lift-and-replace with no state change, which reads as a pointless
     detour rather than a misplacement.
  R5 SEMANTIC PLAUSIBILITY.  The combination must read as one person having a plausibly bad run,
     not as two unrelated glitches. This is the one rule a replay cannot check; it is audited per
     episode by a VLM and by an independent reader (see docs/COMPOSITE_BENCHMARK.md).

Each entry: (name, episode, [(transform, kwargs), ...], story). `story` is the one-sentence human
reading of the run, used as the seed for the R5 audit and as the caption on the results page.
"""

# ---------------------------------------------------------------- the composite episode set
# Ordering note: transforms are listed in the order compose() should apply them, which is plan
# order. A truncating transform (wrong_placement) is always last.

COMPOSITE = [

 # ---- ROUTINE (8 subgoals, 3 rooms): the long-horizon task, and the only one with room for
 # three well-separated errors.
 ("cx_routine_skip_obj", "routine_hard", [
     ("skipped_step", {"sid": "switch_off(tablelamp)"}),
     ("wrong_object", {"sid": "fetch(milk)", "wrong": "juice", "room": "kitchen"}),
  ], "She leaves the bedroom lamp burning, does the book correctly, and then opens the fridge and "
     "takes the juice instead of the milk."),

 # The stall is anchored on FETCH(plate), not on switch_on(microwave): standing still while the
 # microwave runs is what a person does, and an assistant is told to leave a still person alone.
 # Freezing with the plate in her hands has no innocent reading.
 ("cx_routine_skip_obj_stall", "routine_hard", [
     ("skipped_step", {"sid": "switch_off(tablelamp)"}),
     ("wrong_object", {"sid": "fetch(milk)", "wrong": "juice", "room": "kitchen"}),
     ("stall",        {"after": "fetch(plate)", "duration": "transient", "frames": 60}),
  ], "She leaves the lamp on, microwaves the juice instead of the milk, and then freezes with the "
     "plate in her hands before carrying it through."),

 ("cx_routine_nav_place", "routine_hard", [
     ("navigation_inefficiency", {"before": "fetch(milk)", "variant": "explore", "idle": 0,
                                  "stops": [["sofa", "livingroom"], ["bed", "bedroom"]]}),
     ("wrong_placement",         {"sid": "deposit(plate)", "surf": "kitchencounter",
                                  "room": "kitchen"}),
  ], "On her way to the fridge she wanders over to the sofa and through to the bedroom, and at "
     "the end she leaves the plate on the kitchen counter instead of the coffee table."),

 ("cx_routine_rep_obj", "routine_hard", [
     ("repetition",   {"sid": "switch_off(tablelamp)", "at": "fetch(book)", "gap": "delayed"}),
     ("wrong_object", {"sid": "fetch(milk)", "wrong": "juice", "room": "kitchen"}),
  ], "Having switched the bedroom lamp off she goes back and switches it on again, and later "
     "takes the juice out of the fridge instead of the milk."),

 # ---- TIDY (4 independent state goals; the two identical lamps are the scene's only twin goal,
 # so this is the only home of missing_step).
 ("cx_tidy_missing_rep", "tidy_hard", [
     ("missing_step", {"sid": "switch_off(tablelamp@3)"}),
     ("repetition",   {"sid": "shut(fridge)", "at": "switch_off(tv)", "gap": "delayed"}),
  ], "She switches off one bedroom lamp but forgets its twin in the other bedroom, and after the "
     "television she goes back to the fridge she already shut and opens it again."),

 ("cx_tidy_missing_stall", "tidy_hard", [
     ("missing_step", {"sid": "switch_off(tablelamp@3)"}),
     ("stall",        {"after": "switch_off(tv)", "duration": "transient", "frames": 60}),
  ], "She forgets the second bedroom lamp, finishes the fridge and the television, and then "
     "stands motionless in the living room."),

 # ---- SET-TABLE (two INDEPENDENT fetch/deposit chains): the cleanest place for two errors that
 # cannot mask each other, because they live on different objects.


 ("cx_appliance_obj_stall", "appliance_hard", [
     ("wrong_object", {"sid": "fetch(milk)", "wrong": "juice", "room": "kitchen"}),
     ("stall",        {"after": "deposit(milk)", "duration": "transient", "frames": 60}),
  ], "She opens the fridge and takes the juice instead of the milk, loads it into the microwave, "
     "and then stands there doing nothing before starting the machine."),

 ("cx_appliance_nav_place", "appliance_hard", [
     ("navigation_inefficiency", {"before": "fetch(milk)", "variant": "explore", "idle": 0,
                                  "stops": [["sofa", "livingroom"], ["bed", "bedroom"]]}),
     ("wrong_placement",         {"sid": "deposit(milk)", "surf": "kitchencounter",
                                  "room": "kitchen"}),
  ], "She wanders through the living room and bedroom before going to the fridge, then sets the "
     "milk down on the counter instead of putting it in the microwave."),

 # ---- SEARCH (four independent inspections): a task with no dependencies at all, the hardest
 # setting for an omission to be caught in.
 ("cx_search_skip_stall", "search_hard", [
     ("skipped_step", {"sid": "inspect(fridge)"}),
     ("stall",        {"after": "inspect(computer)", "duration": "transient", "frames": 60}),
  ], "She never goes to look at the fridge at all, and after finding the computer she stops and "
     "stands still."),

 # ---- v4 TASKS (2026-09-09): two composite episodes per task, designed under R1-R7.
 # Ruinous errors (skip / missing / wrong object / wrong placement) in every episode; the benign
 # class (stall / nav / repetition) is the second error where the task has a visible switch or a
 # held object to freeze with.

 # PUT-BACK (apt 1): toilet paper desk->bathroom counter, book sofa->bedroom desk, pillow sofa->bed
 ("cx_putback_skip_place", "putback_hard", [
     ("skipped_step",    {"sid": "fetch(toiletpaper)"}),
     ("wrong_placement", {"sid": "deposit(book)", "surf": "bed", "room": "bedroom"}),
  ], "She never touches the toilet paper roll, carries the book to the bedroom and drops it on "
     "the bed instead of the desk."),
 ("cx_putback_nav_place", "putback_hard", [
     ("navigation_inefficiency", {"before": "fetch(book)", "variant": "explore", "idle": 0,
                                  "stops": [["kitchentable", "kitchen"], ["bed", "bedroom"]]}),
     ("wrong_placement",         {"sid": "deposit(book)", "surf": "nightstand", "room": "bedroom"}),
  ], "After the toilet paper she wanders through the kitchen and the bedroom before coming back "
     "for the book, and then leaves the book on the nightstand instead of the desk."),

 # DISHES (apt 2): three plates and two glasses from three rooms into the dishwasher
 ("cx_dishes_missing_stall", "dishes_hard", [
     ("stall",        {"after": "fetch(plate@2)", "duration": "transient", "frames": 60}),
     ("missing_step", {"sid": "fetch(waterglass@1)"}),
  ], "She stands still in the living room with the coffee-table plate in her hands, and at the "
     "end forgets the glass on the kitchen table."),
 ("cx_dishes_nav_place", "dishes_hard", [
     ("navigation_inefficiency", {"before": "fetch(plate@2)", "variant": "explore", "idle": 0,
                                  "stops": [["sofa", "livingroom"], ["bed", "bedroom"]]}),
     ("wrong_placement",         {"sid": "deposit(waterglass@1)", "surf": "kitchencounter",
                                  "room": "kitchen"}),
  ], "After the bedroom plate she wanders past the sofa and back to the bed before collecting "
     "the coffee-table dishes, and sets the last glass on the counter instead of in the machine."),

 # GROCERIES (apt 2): milk from the nightstand and juice from the coffee table into the fridge
 ("cx_groceries_stall_place", "groceries_hard", [
     ("stall",           {"after": "fetch(milk)", "duration": "transient", "frames": 60}),
     ("wrong_placement", {"sid": "deposit(juice)", "surf": "kitchentable", "room": "kitchen"}),
  ], "She picks up the milk in the bedroom and stands there holding it, and later leaves the "
     "juice on the kitchen table instead of putting it in the fridge."),
 ("cx_groceries_skip_stall", "groceries_hard", [
     ("skipped_step", {"sid": "fetch(milk)"}),
     ("stall",        {"after": "fetch(juice)", "duration": "transient", "frames": 60}),
  ], "She never goes to the bedroom for the milk, picks up the juice from the coffee table and "
     "stands there holding it."),

 # SHUTDOWN (apt 4): fridge shut, TV and computer off, both table lamps off
 ("cx_shutdown_missing_rep", "shutdown_hard", [
     ("repetition",   {"sid": "shut(fridge)", "at": "switch_off(tv)", "gap": "delayed"}),
     ("missing_step", {"sid": "switch_off(tablelamp@1)"}),
  ], "After the television she goes back and opens the fridge she had just shut, and later "
     "switches off only one of the two table lamps."),
 ("cx_shutdown_skip_stall", "shutdown_hard", [
     ("stall",        {"after": "shut(fridge)", "duration": "transient", "frames": 60}),
     ("skipped_step", {"sid": "switch_off(computer)"}),
  ], "She shuts the fridge and stands motionless in front of it, and later walks past the "
     "computer without switching it off."),

 # MUGS (apt 5): three mugs, one per trip, back to the kitchen table
 ("cx_mugs_missing_stall", "mugs_hard", [
     ("missing_step", {"sid": "fetch(mug@1)#2"}),
     ("stall",        {"after": "fetch(mug@1)#3", "duration": "transient", "frames": 60}),
  ], "She brings the first mug back, skips the second entirely, and stands frozen holding "
     "the third."),
 ("cx_mugs_nav_place", "mugs_hard", [
     ("navigation_inefficiency", {"before": "fetch(mug@1)#2", "variant": "explore", "idle": 0,
                                  "stops": [["sofa", "livingroom"], ["bed", "bedroom"]]}),
     ("wrong_placement",         {"sid": "deposit(mug@1)#3", "surf": "kitchencounter",
                                  "room": "kitchen"}),
  ], "Between the first and second mug she wanders past the sofa and the bed, and puts the "
     "third mug on the counter instead of the table."),

 # GUEST (apt 1): plate and mug into the dishwasher, pillow back on the bed, TV off
 ("cx_guest_skip_stall", "guest_hard", [
     ("skipped_step", {"sid": "fetch(mug)"}),
     ("stall",        {"after": "fetch(pillow)", "duration": "transient", "frames": 60}),
  ], "She loads the plate but never fetches the mug from the desk, and later stands frozen "
     "with the pillow in her hands."),
 ("cx_guest_stall_place", "guest_hard", [
     ("stall",           {"after": "fetch(plate)", "duration": "transient", "frames": 60}),
     ("wrong_placement", {"sid": "deposit(mug)", "surf": "kitchentable", "room": "kitchen"}),
  ], "She freezes with the plate in her hands, and later sets the mug on the kitchen table "
     "instead of in the dishwasher."),

 # AFTER-MEAL (apt 3): two plates and the mug into the dishwasher, the apple back in the fridge
 ("cx_aftermeal_missing_obj", "aftermeal_hard", [
     ("missing_step", {"sid": "fetch(plate@1)"}),
     ("wrong_object", {"sid": "fetch(mug)", "wrong": "wineglass", "room": "livingroom"}),
  ], "She clears only one of the two plates, and instead of the mug from the side table she "
     "takes a wine glass from the coffee table to the dishwasher."),
 ("cx_aftermeal_stall_place", "aftermeal_hard", [
     ("stall",           {"after": "fetch(plate@2)", "duration": "transient", "frames": 60}),
     ("wrong_placement", {"sid": "deposit(apple)", "surf": "kitchentable", "room": "kitchen"}),
  ], "She stands still holding the second plate, and at the end leaves the apple on the kitchen "
     "table instead of in the fridge."),

 # TIDY-BEDROOM (apt 1): book bed->desk, cabinet shut, lamp off, mug nightstand->kitchen table
 ("cx_tidybedroom_skip_rep", "tidybedroom_hard", [
     ("skipped_step", {"sid": "shut(cabinet)"}),
     ("repetition",   {"sid": "switch_off(tablelamp@1)", "at": "fetch(mug)", "gap": "delayed"}),
  ], "She leaves the cabinet open, switches the lamp off, and when she reaches for the mug she "
     "turns back and switches the lamp on again."),
 ("cx_tidybedroom_stall_place", "tidybedroom_hard", [
     ("stall",           {"after": "fetch(book)", "duration": "transient", "frames": 60}),
     ("wrong_placement", {"sid": "deposit(mug)", "surf": "kitchencounter", "room": "kitchen"}),
  ], "She freezes with the book in her hands, and finally puts the mug on the kitchen counter "
     "instead of the kitchen table."),
]


# ---------------------------------------------------------------- restraint arm (no injection)
# Not part of the "at least two errors" requirement, but a composite benchmark without a
# restraint denominator scores silence at the top. One clean and one legal-reorder episode per
# task family that hosts a composite episode.
RESTRAINT = [
 ("cx_clean_appliance",  "appliance_hard", None),
 ("cx_clean_settable",   "settable_hard",  None),
 ("cx_clean_tidy",       "tidy_hard",      None),
 ("cx_clean_routine",    "routine_hard",   None),
 ("cx_reorder_tidy",     "tidy_hard",      "reverse"),
 ("cx_reorder_settable", "settable_hard",  "reverse"),
 ("cx_rotate_routine",   "routine_hard",   "rotate"),
 ("cx_rotate_power",     "power_hard",     "rotate"),
]


# ---------------------------------------------------------------- user cases
# A case file written by `vhhome-case new` (any JSON {"cases": [{"name", "task", "mistakes": [[class, {...}], ...],
# "story"}]}) named in VHHOME_CASES (os.pathsep-separated) is appended to the shipped set when this module loads, so
# every tool that resolves a case by name sees it. A case without mistakes is run as "<task>__clean".
import json as _json, os as _os
for _p in [p for p in _os.environ.get("VHHOME_CASES", "").split(_os.pathsep) if p]:
    for _c in _json.load(open(_p)).get("cases", []):
        if _c.get("mistakes") and not any(n == _c["name"] for n, *_ in COMPOSITE):
            COMPOSITE.append((_c["name"], _c["task"], [(t, dict(k)) for t, k in _c["mistakes"]], _c.get("story", "")))
