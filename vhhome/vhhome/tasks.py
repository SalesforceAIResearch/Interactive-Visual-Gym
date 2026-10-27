# 9 Unity-renderable families x {easy, hard}. step = (Action, class, room[, class2, room2]) | (Action,)
# hard episodes are designed to traverse >=3 rooms (objects span bedroom/kitchen/livingroom).
SCENE=2
EPISODES = {
 # 1. RETRIEVE & RELOCATE
 # retrieve a plate from one kitchen table and relocate it to a DIFFERENT table (kitchentable@2)
 "relocate_hard": dict(family="Retrieve & relocate", diff="hard", scene=2, init="bedroom", steps=[
   ("Walk","book","bedroom"),("Grab","book","bedroom"),("Walk","kitchentable","kitchen"),("Walk","coffeetable","livingroom"),("Put","book","bedroom","coffeetable","livingroom")]),
 # 2. APPLIANCE OPERATION
 "appliance_hard": dict(family="Appliance operation", diff="hard", scene=2, init="bedroom",
   targets={"milk": "fridge"},
   # ONE milk in the scene (user decision 2026-08-26): the decoy bottle is DELETED at init (see
   # `dedupe`), not relocated — relocating it into a tracked container broke the symbolic twin.
   # With a single milk the nearest-instance binding can't grab the wrong bottle during recovery
   # (the decoy beside the microwave was what "grab the milk" resolved to). The wrong_object cases
   # are UNAFFECTED — they grab JUICE, a separate object, not a second milk.
   dedupe={"milk": 1},
   setup=[("Walk","milk","kitchen"),("Grab","milk","kitchen"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","milk","kitchen","fridge","kitchen"),("Close","fridge","kitchen"),("Walk","juice","kitchen"),("Grab","juice","kitchen"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","juice","kitchen","fridge","kitchen"),("Close","fridge","kitchen"),("Walk","computer","bedroom")],
   steps=[
   ("Walk","coffeetable","livingroom"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("Grab","milk","kitchen"),("Close","fridge","kitchen"),("Walk","microwave","kitchen"),("Open","microwave","kitchen"),("PutIn","milk","kitchen","microwave","kitchen"),("Close","microwave","kitchen"),("SwitchOn","microwave","kitchen")]),
 # 3. POWER / STANDBY TOGGLE
 "power_hard": dict(family="Power toggle", diff="hard", scene=2, init="bedroom",
   setup=[("SwitchOff","tablelamp@1","bedroom"),("SwitchOff","tablelamp@2","bedroom"),("SwitchOff","tablelamp@3","bedroom")],
   steps=[
   ("Walk","tablelamp","bedroom"),("SwitchOn","tablelamp","bedroom"),("Walk","stove","kitchen"),("SwitchOn","stove","kitchen"),("Walk","tv","livingroom"),("SwitchOn","tv","livingroom")]),
 # 4. POSTURE & ACTIVITY
 "posture_hard": dict(family="Posture & activity", diff="hard", scene=2, init="bedroom", steps=[
   ("Walk","bed","bedroom"),("Sit","bed","bedroom"),("StandUp",),("Walk","chair","kitchen"),("Sit","chair","kitchen"),("StandUp",),("Walk","sofa","livingroom"),("Sit","sofa","livingroom"),("StandUp",)]),
 # 5. ASSEMBLE / SET-TABLE
 # retrieve plate + fork from one table and set them on a DIFFERENT table (kitchentable@2)
  # 6. CONTAINER -> CONTAINER TRANSFER
 # genuine container->container: a setup pass first stows the (counter) milk INTO the fridge,
 # so the recorded task really takes milk out of the fridge and moves it to the microwave.
 "transfer_hard": dict(family="Container→container", diff="hard", scene=2, init="kitchen",
   targets={"milk": "fridge"}, dedupe={"milk": 1},   # single milk (user 2026-08-26): no decoy to mis-grab
   setup=[("Walk","milk","kitchen"),("Grab","milk","kitchen"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","milk","kitchen","fridge","kitchen"),("Close","fridge","kitchen"),("Walk","juice","kitchen"),("Grab","juice","kitchen"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","juice","kitchen","fridge","kitchen"),("Close","fridge","kitchen")],
   steps=[
   ("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("Grab","milk","kitchen"),("Close","fridge","kitchen"),("Walk","coffeetable","livingroom"),("Walk","cabinet","bedroom"),("Open","cabinet","bedroom"),("PutIn","milk","kitchen","cabinet","bedroom"),("Close","cabinet","bedroom")]),
 # 7. SEARCH & INSPECT — agent STARTS FAR from the targets and must traverse the house to find each.
 # easy: TV is in the living room; start in the bedroom (farthest). hard: 4 targets scattered across
 # living/bedroom/kitchen; start in the kitchen and go find them all.
 "search_hard": dict(family="Search & inspect", diff="hard", scene=2, init="kitchen", steps=[
   ("Walk","fridge","kitchen"),("TurnTo","fridge","kitchen"),
   ("Walk","tv","livingroom"),("TurnTo","tv","livingroom"),
   ("Walk","tablelamp","bedroom"),("TurnTo","tablelamp","bedroom"),
   ("Walk","computer","bedroom"),("TurnTo","computer","bedroom")]),
 # 8. STATE-TIDY
 # scene starts with fridge + dishwasher + microwave already OPEN (off-camera setup); stove &
 # cabinets are left closed, so the agent must CHECK state rather than blindly toggle. Task = close
 # the open ones (fridge, dishwasher, microwave).
 "tidy_hard": dict(family="State-tidy", diff="hard", scene=2, init="bedroom",
   # LIGHTS-ONLY tidy (2026-08-24, user redesign): the two operable nightstand lamps are the
   # only lit lights (@1=#256 left bedroom on its nightstand, @3=#377 right bedroom on its
   # nightstand; @2=#376 is a dead wall fixture the engine cannot walk to, kept dark). The
   # task is the natural "turn off all the lights"; forgetting the second lamp is missing_step.
   # One reachable lamp per bedroom is a HARD scene limit: lamps are not graspable (cannot be
   # relocated into one room) and the second bedroom-B lamp + both lightswitches are
   # engine-unreachable (switch-off silently no-ops) -- verified 2026-08-24.
   setup=[("SwitchOff","tablelamp@1","bedroom"),("SwitchOff","tablelamp@2","bedroom"),("SwitchOff","tablelamp@3","bedroom"),
          ("SwitchOn","tablelamp@1","bedroom"),("SwitchOn","tablelamp@3","bedroom"),
          ("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("Walk","tv","livingroom"),("SwitchOn","tv","livingroom")],
   steps=[
   ("Walk","tablelamp@1","bedroom"),("SwitchOff","tablelamp@1","bedroom"),
   ("Walk","tablelamp@3","bedroom"),("SwitchOff","tablelamp@3","bedroom"),
   ("Walk","fridge","kitchen"),("Close","fridge","kitchen"),("Walk","tv","livingroom"),("SwitchOff","tv","livingroom")]),
 # 9. DRINK
 # setup stows the (counter) milk INTO the fridge first, so "get milk from the fridge" is real
 "drink_hard": dict(family="Drink", diff="hard", scene=2, init="bedroom", steps=[
   ("Walk","juice","kitchen"),("Grab","juice","kitchen"),("Drink","juice","kitchen")]),
 # 10. ROUTINE (v3 detection scenario) — long multi-goal chore crossing 3 rooms: lamp off,
 # book -> coffee table, milk from fridge -> microwave -> on, plate -> coffee table. Setup stows
 # the milk, lights the bedroom lamp, and returns the avatar to the kitchen so every plan walk
 # is a real traversal.
 "routine_hard": dict(family="Routine", diff="hard", scene=2, init="kitchen",
   targets={"milk": "fridge"}, dedupe={"milk": 1, "book": {"room": "bedroom", "keep": 1}},   # single milk (user 2026-08-26): no decoy to mis-grab; one bedroom book, laid on the bed so the grab is visible (pickup audit 2026-09-04)
   # Darken ALL THREE default-lit lamps before lighting the one the task is about. Without this,
   # scene 2's other two lamps burn for the whole episode, the whole-house map advertises them
   # every tick, and an assistant can produce a correct-looking "a lamp is still on" flag without
   # ever observing the injected omission (independent audit, 2026-08-28).
   setup=[("SwitchOff","tablelamp@1","bedroom"),("SwitchOff","tablelamp@2","bedroom"),("SwitchOff","tablelamp@3","bedroom"),
          ("Walk","milk","kitchen"),("Grab","milk","kitchen"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","milk","kitchen","fridge","kitchen"),("Close","fridge","kitchen"),("Walk","juice","kitchen"),("Grab","juice","kitchen"),("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","juice","kitchen","fridge","kitchen"),("Close","fridge","kitchen"),("Walk","tablelamp","bedroom"),("SwitchOn","tablelamp","bedroom"),("Walk","book","bedroom"),("Grab","book","bedroom"),("Walk","bed","bedroom"),("Put","book","bedroom","bed","bedroom"),("Walk","kitchentable","kitchen")],
   steps=[
   ("Walk","tablelamp","bedroom"),("SwitchOff","tablelamp","bedroom"),
   ("Walk","book","bedroom"),("Grab","book","bedroom"),
   ("Walk","coffeetable","livingroom"),("Put","book","bedroom","coffeetable","livingroom"),
   ("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("Grab","milk","kitchen"),("Close","fridge","kitchen"),
   ("Walk","microwave","kitchen"),("Open","microwave","kitchen"),("PutIn","milk","kitchen","microwave","kitchen"),("Close","microwave","kitchen"),("SwitchOn","microwave","kitchen"),
   ("Walk","plate","kitchen"),("Grab","plate","kitchen"),
   ("Walk","coffeetable","livingroom"),("Put","plate","kitchen","coffeetable","livingroom")]),



 # ---- v3 (2026-09-04): complexity comes from the HOUSE'S STATE, not from chained subtasks.
 # Each task is one natural goal; the setup leaves the flat wrong (things in other rooms, all of
 # a kind scattered, devices on, doors open) and the plan is the natural way to put it right.

 "putback_hard": dict(family="Put things back", diff="hard", scene=1, init="livingroom",   # apartment 1: ONE bedroom, so "the bedroom desk" is unambiguous
   # CLASS A (many -> many). The pile sits on a close area -- desk + sofa -- and the person starts
   # in that room, so every item is on camera before the first step. Small things never start on the
   # LOW coffee table: the person stands over it and the 40-degree pitch cap hides them (pickup audit 2026-09-04).
   dedupe={"toiletpaper": {"room": "bathroom", "keep": 1}, "book": {"room": "bedroom", "keep": 1},
           "pillow": {"room": "bedroom", "keep": 1}},
   setup=[("Walk","toiletpaper","bathroom"),("Grab","toiletpaper","bathroom"),("Walk","desk","livingroom"),("Put","toiletpaper","bathroom","desk","livingroom"),
          ("Walk","book","bedroom"),("Grab","book","bedroom"),("Walk","sofa","livingroom"),("Put","book","bedroom","sofa","livingroom"),
          ("Walk","pillow","bedroom"),("Grab","pillow","bedroom"),("Walk","sofa","livingroom"),("Put","pillow","bedroom","sofa","livingroom")],
   steps=[
   ("Walk","toiletpaper","bathroom"),("Grab","toiletpaper","bathroom"),
   ("Walk","bathroomcounter","bathroom"),("Put","toiletpaper","bathroom","bathroomcounter","bathroom"),   # the cabinet has no free spot for a roll (engine reject 2026-09-04)
   ("Walk","book","bedroom"),("Grab","book","bedroom"),
   ("Walk","desk","bedroom"),("Put","book","bedroom","desk","bedroom"),
   ("Walk","pillow","bedroom"),("Grab","pillow","bedroom"),
   ("Walk","bed","bedroom"),("Put","pillow","bedroom","bed","bedroom")]),
 "dishes_hard": dict(family="Dishes", diff="hard", scene=2, init="bathroom",
   # CLASS B (everywhere -> one). Each dish starts somewhere visible AND plausible for a dish:
   # the kitchen table, the living-room coffee table, the bedroom desk. The instruction says
   # which room each is in.
   dedupe={"plate": {"room": "kitchen", "keep": 3}, "waterglass": {"room": "kitchen", "keep": 2}},
   setup=[("Walk","plate@2","kitchen"),("Grab","plate@2","kitchen"),("Walk","coffeetable","livingroom"),("Put","plate@2","kitchen","coffeetable","livingroom"),
          ("Walk","waterglass@2","kitchen"),("Grab","waterglass@2","kitchen"),("Walk","coffeetable","livingroom"),("Put","waterglass@2","kitchen","coffeetable","livingroom"),
          ("Walk","plate@3","kitchen"),("Grab","plate@3","kitchen"),("Walk","bed#284","bedroom"),("Put","plate@3","kitchen","bed#284","bedroom")],
   # "the bed in the bedroom" (A33, f89): apartment 2 has TWO bedrooms, and a bare "bed" resolved by
   # nearness to bed#394 in the one the map labels "bedroom#2" -- people searched the room labelled
   # "bedroom", which had no plate (42 refusals over 9 episodes). Pin the bed in "bedroom" (#241).
   steps=[
   ("Walk","plate@3","kitchen"),("Grab","plate@3","kitchen"),
   ("Walk","dishwasher","kitchen"),("Open","dishwasher","kitchen"),("PutIn","plate@3","kitchen","dishwasher","kitchen"),
   ("Walk","plate@2","kitchen"),("Grab","plate@2","kitchen"),("Walk","waterglass@2","kitchen"),("Grab","waterglass@2","kitchen"),
   ("Walk","dishwasher","kitchen"),("PutIn","plate@2","kitchen","dishwasher","kitchen"),("PutIn","waterglass@2","kitchen","dishwasher","kitchen"),
   ("Walk","plate@1","kitchen"),("Grab","plate@1","kitchen"),("Walk","waterglass@1","kitchen"),("Grab","waterglass@1","kitchen"),
   ("Walk","dishwasher","kitchen"),("PutIn","plate@1","kitchen","dishwasher","kitchen"),("PutIn","waterglass@1","kitchen","dishwasher","kitchen"),
   ("Close","dishwasher","kitchen")]),
 "groceries_hard": dict(family="Groceries", diff="hard", scene=2, init="bathroom",
   # scene 2: the juice left on scene 6's living-room table could not be picked up again there.
   dedupe={"milk": {"room": "kitchen", "keep": 1}, "juice": {"room": "kitchen", "keep": 1}},
   setup=[("Walk","milk","kitchen"),("Grab","milk","kitchen"),("Walk","nightstand","bedroom"),("Put","milk","kitchen","nightstand","bedroom"),
          ("Walk","juice","kitchen"),("Grab","juice","kitchen"),("Walk","coffeetable","livingroom"),("Put","juice","kitchen","coffeetable","livingroom")],
   steps=[
   ("Walk","milk","kitchen"),("Grab","milk","kitchen"),
   ("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","milk","kitchen","fridge","kitchen"),
   ("Walk","juice","kitchen"),("Grab","juice","kitchen"),
   ("Walk","fridge","kitchen"),("PutIn","juice","kitchen","fridge","kitchen"),
   ("Close","fridge","kitchen")]),
 "shutdown_hard": dict(family="Shut the house down", diff="hard", scene=4, init="livingroom",   # not the bathroom: its door is never visible to the egocentric gate (pilot 3, A22)
   setup=[("Walk","tablelamp@1","livingroom"),("SwitchOn","tablelamp@1","livingroom"),
          ("Walk","tablelamp@1","bedroom"),("SwitchOn","tablelamp@1","bedroom"),
          ("Walk","tv","livingroom"),("SwitchOn","tv","livingroom"),
          ("Walk","computer","livingroom"),("SwitchOn","computer","livingroom"),
          ("Walk","fridge","kitchen"),("Open","fridge","kitchen")],
   steps=[
   ("Walk","fridge","kitchen"),("Close","fridge","kitchen"),
   ("Walk","tv","livingroom"),("SwitchOff","tv","livingroom"),
   ("Walk","computer","livingroom"),("SwitchOff","computer","livingroom"),
   ("Walk","tablelamp@1","livingroom"),("SwitchOff","tablelamp@1","livingroom"),
   ("Walk","tablelamp@1","bedroom"),("SwitchOff","tablelamp@1","bedroom")]),
  "mugs_hard": dict(family="Mugs", diff="hard", scene=5, init="kitchen",
   # One mug per trip: carrying two, the two Puts resolve among the held mugs by class, and on
   # camera a blue mug goes down as a red one. Tokens carry @1 (a bare class token resolves to
   # the nearest instance -- after a pickup, the one in her hand).
   setup=[("Walk","mug@1","kitchen"),("Grab","mug@1","kitchen"),("Walk","nightstand","bedroom"),("Put","mug@1","kitchen","nightstand","bedroom")],
   steps=[
   ("Walk","mug@1","bathroom"),("Grab","mug@1","bathroom"),("Walk","kitchentable","kitchen"),("Put","mug@1","bathroom","kitchentable","kitchen"),
   ("Walk","mug@1","livingroom"),("Grab","mug@1","livingroom"),("Walk","kitchentable","kitchen"),("Put","mug@1","livingroom","kitchentable","kitchen"),
   ("Walk","mug@1","kitchen"),("Grab","mug@1","kitchen"),("Walk","kitchentable","kitchen"),("Put","mug@1","kitchen","kitchentable","kitchen")]),
 "guest_hard": dict(family="Before a guest", diff="hard", scene=1, init="bedroom",
   dedupe={"plate": {"room": "kitchen", "keep": 1}, "mug": {"room": "bedroom", "keep": 1}, "pillow": {"room": "bedroom", "keep": 1}},
   setup=[("Walk","tv","livingroom"),("SwitchOn","tv","livingroom"),
          ("Walk","plate","kitchen"),("Grab","plate","kitchen"),("Walk","coffeetable","livingroom"),("Put","plate","kitchen","coffeetable","livingroom"),
          ("Walk","mug","bedroom"),("Grab","mug","bedroom"),("Walk","desk","livingroom"),("Put","mug","bedroom","desk","livingroom"),
          ("Walk","pillow","bedroom"),("Grab","pillow","bedroom"),("Walk","sofa","livingroom"),("Put","pillow","bedroom","sofa","livingroom")],
   steps=[
   ("Walk","plate","kitchen"),("Grab","plate","kitchen"),
   ("Walk","dishwasher","kitchen"),("Open","dishwasher","kitchen"),("PutIn","plate","kitchen","dishwasher","kitchen"),
   ("Walk","mug","bedroom"),("Grab","mug","bedroom"),
   ("Walk","dishwasher","kitchen"),("PutIn","mug","bedroom","dishwasher","kitchen"),("Close","dishwasher","kitchen"),
   ("Walk","pillow","bedroom"),("Grab","pillow","bedroom"),("Walk","bed","bedroom"),("Put","pillow","bedroom","bed","bedroom"),
   ("Walk","tv","livingroom"),("SwitchOff","tv","livingroom")]),
 "aftermeal_hard": dict(family="After a meal", diff="hard", scene=3, init="bedroom",
   dedupe={"plate": {"room": "kitchen", "keep": 2}, "mug": {"room": "kitchen", "keep": 1}, "apple": {"room": "kitchen", "keep": 1}},
   setup=[("Walk","plate@1","kitchen"),("Grab","plate@1","kitchen"),("Walk","coffeetable","livingroom"),("Put","plate@1","kitchen","coffeetable","livingroom"),
          ("Walk","plate@2","kitchen"),("Grab","plate@2","kitchen"),("Walk","coffeetable","livingroom"),("Put","plate@2","kitchen","coffeetable","livingroom"),
          ("Walk","mug","kitchen"),("Grab","mug","kitchen"),("Walk","nightstand","livingroom"),("Put","mug","kitchen","nightstand","livingroom"),   # on the desk the mug hid behind the monitor (pickup audit 2026-09-04)
          ("Walk","apple","kitchen"),("Grab","apple","kitchen"),("Walk","nightstand","livingroom"),("Put","apple","kitchen","nightstand","livingroom")],   # mug + apple on the waist-high side table: on the desk the first free spot is under the monitor (pickup audit 2026-09-04)
   steps=[
   ("Walk","plate@1","kitchen"),("Grab","plate@1","kitchen"),
   ("Walk","dishwasher","kitchen"),("Open","dishwasher","kitchen"),("PutIn","plate@1","kitchen","dishwasher","kitchen"),
   ("Walk","plate@2","kitchen"),("Grab","plate@2","kitchen"),
   ("Walk","dishwasher","kitchen"),("PutIn","plate@2","kitchen","dishwasher","kitchen"),
   ("Walk","mug","kitchen"),("Grab","mug","kitchen"),("Walk","apple","kitchen"),("Grab","apple","kitchen"),
   ("Walk","dishwasher","kitchen"),("PutIn","mug","kitchen","dishwasher","kitchen"),("Close","dishwasher","kitchen"),
   ("Walk","fridge","kitchen"),("Open","fridge","kitchen"),("PutIn","apple","kitchen","fridge","kitchen"),("Close","fridge","kitchen")]),

 "tidybedroom_hard": dict(family="Tidy the bedroom", diff="hard", scene=1, init="kitchen",
   # CLASS C, in a flat with ONE bedroom (apartment 2 has two, and the instruction's "the
   # bedroom" was wrong). A pillow put on a desk stays in the hand, so the pillow swap is gone.
   dedupe={"book": {"room": "bedroom", "keep": 1}, "mug": {"room": "bedroom", "keep": 1}},
   setup=[("Walk","book","bedroom"),("Grab","book","bedroom"),("Walk","bed","bedroom"),("Put","book","bedroom","bed","bedroom"),
          ("Walk","mug","bedroom"),("Grab","mug","bedroom"),("Walk","nightstand","bedroom"),("Put","mug","bedroom","nightstand","bedroom"),
          ("Walk","tablelamp@1","bedroom"),("SwitchOn","tablelamp@1","bedroom"),
          ("Walk","cabinet","bedroom"),("Open","cabinet","bedroom")],
   steps=[
   ("Walk","book","bedroom"),("Grab","book","bedroom"),("Walk","desk","bedroom"),("Put","book","bedroom","desk","bedroom"),
   ("Walk","cabinet","bedroom"),("Close","cabinet","bedroom"),
   ("Walk","tablelamp@1","bedroom"),("SwitchOff","tablelamp@1","bedroom"),
   ("Walk","mug","bedroom"),("Grab","mug","bedroom"),
   ("Walk","kitchentable","kitchen"),("Put","mug","bedroom","kitchentable","kitchen")]),

}
