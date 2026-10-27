"""Per-episode NL instruction + programmatic CHECKPOINTS for the 18 VirtualHome benchmark episodes.

Instructions are PRECISE but MINIMAL: an object is qualified (e.g. "kitchen table", "bedroom cabinet") only
when its class has instances the agent could confuse; otherwise it is named bare. The route between rooms is
NOT spelled out — the agent must find and reach things itself. Grading NEVER depends on rooms traversed.

Grading runs on RESOLVED events: each successful action is (verb, cls, id, cls2, id2) where cls/cls2 are the
ACTUAL object classes the gated controller acted on (not the word the VLM typed). So `cabinet` != the 8
`kitchencabinet`s, `tablelamp` != wall/ceiling lamps, `tv` != `tvstand`, `kitchentable` != desk/coffeetable.
Checkpoints latch once true; success = ALL latched.
"""
import inspect

def evaluate(gdef, events, targets=None):
    """Grade an episode, passing the target binding only to episodes that take one.

    Checkpoint lambdas are migrating from `lambda ev:` to `lambda ev, tgt:` one episode at a
    time, so every caller has to tolerate both. Dispatch on the SIGNATURE rather than catching
    TypeError, which would silently swallow a genuine TypeError raised inside the lambda and
    report the episode as ungraded.
    """
    fn = gdef["checkpoints"]
    n = len(inspect.signature(fn).parameters)
    return fn(events, targets or {}) if n >= 2 else fn(events)


def did(ev, verb, cls=None, cls2=None, iid=None):
    """`iid` pins the INSTANCE, not just the class.

    Without it "put a plate in the dishwasher" and "put THE plate in the dishwasher" score
    identically, which is wrong wherever the scene holds more than one of a thing -- scene 2 has
    9 plates, 7 books, 3 tablelamps and 2 milks. Episodes that name a specific object bind it in
    `targets` and pass the id through here; the rest stay class-level.
    """
    return any(e[0] == verb and (cls is None or e[1] == cls) and (cls2 is None or e[3] == cls2)
               and (iid is None or e[2] == iid) for e in ev)

def grabbed_i(ev, o, tgt):   return did(ev, "pick_up_nearest", o, iid=tgt.get(o))
def put_in_i(ev, o, t, tgt): return did(ev, "put_in_nearest", o, t, iid=tgt.get(o))
def put_on_i(ev, o, t, tgt): return did(ev, "put_on_nearest", o, t, iid=tgt.get(o))
def drank_i(ev, o, tgt):     return did(ev, "drink_nearest", o, iid=tgt.get(o))

def _last_toggle_off(ev, o):
    """Instances of `o` whose MOST RECENT toggle was a switch-off.

    A latched "saw a switch_off event" is wrong for a toggle goal: a repetition that re-presses
    the switch leaves the lamp lit while the event is still in the log, so the checkpoint read
    "all lights off" over a world with a lamp burning — the event view and the state view
    disagreed (independent audit, 2026-08-28). Reading the LAST toggle per instance makes them
    agree without giving up event-based grading.
    """
    last = {}
    for e in ev:
        if e[1] == o and e[0] in ("switch_off_nearest", "switch_on_nearest"):
            last[e[2]] = e[0]
    return {i for i, v in last.items() if v == "switch_off_nearest"}


def turned_off_n(ev, o, n):
    """n DISTINCT instances of `o` LEFT switched off — a multi-instance goal ("both bedroom
    lamps") is done only when every twin got its own event and is still off at the end."""
    return len(_last_toggle_off(ev, o)) >= n

def _n_distinct(ev, verb, o, t=None):
    """How many DISTINCT instances of `o` received `verb` (onto/into `t` if given)."""
    return len({e[2] for e in ev if e[0] == verb and e[1] == o and (t is None or e[3] == t)})

def put_in_n(ev, o, t, n):  return _n_distinct(ev, "put_in_nearest", o, t) >= n
def put_on_n(ev, o, t, n):  return _n_distinct(ev, "put_on_nearest", o, t) >= n
def closed_n(ev, o, n):     return _n_distinct(ev, "close_nearest", o) >= n
def grabbed(ev, o):    return did(ev, "pick_up_nearest", o)
def put_on(ev, o, t):  return did(ev, "put_on_nearest", o, t)
def put_in(ev, o, t):  return did(ev, "put_in_nearest", o, t)
def opened(ev, o):     return did(ev, "open_nearest", o)
def closed(ev, o):     return did(ev, "close_nearest", o)
def turned_on(ev, o):  return did(ev, "switch_on_nearest", o)
def turned_off(ev, o): return bool(_last_toggle_off(ev, o))   # last toggle wins
def sat_on(ev, o):     return did(ev, "sit_nearest", o)
def stood(ev):         return did(ev, "stand_up")
def drank(ev, o):      return did(ev, "drink_nearest", o)
def read_it(ev, o):   return did(ev, "read_nearest", o)
def watched(ev, o):   return did(ev, "watch_nearest", o)
def reached(ev, o):    return did(ev, "go_to_nearest", o) or did(ev, "go_through_nearest_door", o)

EPISODES = {
 "relocate_hard": dict(
   instruction="TASK: Take a book and put it on the coffee table in the living room.",
   checkpoints=lambda ev: {"grabbed book": grabbed(ev, "book"),
                           "put book on coffee table": put_on(ev, "book", "coffeetable")}),
 "appliance_hard": dict(
   instruction="TASK: Take the milk from the fridge, put it in the microwave, and switch the microwave on.",
   checkpoints=lambda ev, tgt: {"got the fridge milk": grabbed_i(ev, "milk", tgt),
                                "put that milk in microwave": put_in_i(ev, "milk", "microwave", tgt),
                                "switched microwave on": turned_on(ev, "microwave")}),
 "power_hard": dict(
   instruction="TASK: Turn on a table lamp, the stove, and the TV.",
   checkpoints=lambda ev: {"table lamp on": turned_on(ev, "tablelamp"),
                           "stove on": turned_on(ev, "stove"),
                           "TV on": turned_on(ev, "tv")}),
 "posture_hard": dict(
   instruction="TASK: Sit on a bed, a chair, and the sofa, standing up between each.",
   checkpoints=lambda ev: {"sat on bed": sat_on(ev, "bed"), "sat on chair": sat_on(ev, "chair"),
                           "sat on sofa": sat_on(ev, "sofa")}),
 "routine_hard": dict(
   instruction="TASK: Turn off the bedroom lamp that is on, put the book that is lying on the bed on the living-room coffee table, heat the milk from the fridge in the microwave, and bring one of the plates from the kitchen table over to the coffee table.",
   checkpoints=lambda ev, tgt: {"lamp off": turned_off(ev, "tablelamp"),
                                "book on coffee table": put_on(ev, "book", "coffeetable"),
                                "the fridge milk in microwave": put_in_i(ev, "milk", "microwave", tgt),
                                "microwave on": turned_on(ev, "microwave"),
                                "plate on coffee table": put_on(ev, "plate", "coffeetable")}),
  "transfer_hard": dict(
   instruction="TASK: Take the milk from the fridge and put it in the bedroom cabinet.",
   checkpoints=lambda ev, tgt: {"got the fridge milk": grabbed_i(ev, "milk", tgt),
                                "put that milk in bedroom cabinet": put_in_i(ev, "milk", "cabinet", tgt)}),
 "search_hard": dict(
   instruction="TASK: Find each of these somewhere in the house and go up to it: a TV, a table lamp, the fridge, and the computer.",
   checkpoints=lambda ev: {"reached TV": reached(ev, "tv"), "reached table lamp": reached(ev, "tablelamp"),
                           "reached fridge": reached(ev, "fridge"), "reached computer": reached(ev, "computer")}),
 "tidy_hard": dict(
   # Names the three devices instead of saying "every device that is on". Scene 2 starts with
   # THREE table lamps lit by default and the episode only ever addresses one, so "every device"
   # was ungradeable: the state-based final_ok demanded all three off while the event-based
   # checkpoints latched on the one, and the two disagreed on every run.
   instruction="TASK: Tidy up the house \u2014 switch off the lights on both table lamps, switch off the TV, and close the fridge.",
   checkpoints=lambda ev: {"all lights off": turned_off_n(ev, "tablelamp", 2),
                           "TV off": turned_off(ev, "tv"),
                           "fridge closed": closed(ev, "fridge")}),
 "drink_hard": dict(
   instruction="TASK: Get a juice and drink it.",
   checkpoints=lambda ev: {"got juice": grabbed(ev, "juice"), "drank it": drank(ev, "juice")}),

 # ---- new families (2026-09-03): bathroom, dishwasher, coffee, laundry, office ----
 "putback_hard": dict(
   instruction="TASK: In the living room, a toilet paper roll has been left on the desk, and a book and a pillow on the sofa; they belong in other rooms. Put the toilet paper roll on the bathroom counter, the book on the bedroom desk, and the pillow on the bed.",
   checkpoints=lambda ev: {"toilet paper back on the bathroom counter": put_on(ev, "toiletpaper", "bathroomcounter"), "book back on the bedroom desk": put_on(ev, "book", "desk"), "pillow back on the bed": put_on(ev, "pillow", "bed")}),
 "dishes_hard": dict(
   instruction="TASK: Dirty dishes have been left around the flat: a plate and a glass on the kitchen table, a plate and a glass on the living-room coffee table, and a plate on the bed in the bedroom. Put all of them in the dishwasher and close it.",
   checkpoints=lambda ev: {"all three plates in the dishwasher": put_in_n(ev, "plate", "dishwasher", 3), "both glasses in the dishwasher": put_in_n(ev, "waterglass", "dishwasher", 2), "dishwasher closed": closed(ev, "dishwasher")}),
 "groceries_hard": dict(
   instruction="TASK: The milk was left on the bedroom nightstand and the juice on the living-room coffee table. Put them both in the fridge, then close it.",
   checkpoints=lambda ev: {"milk in the fridge": put_in(ev, "milk", "fridge"), "juice in the fridge": put_in(ev, "juice", "fridge"), "fridge closed": closed(ev, "fridge")}),
 "shutdown_hard": dict(
   instruction="TASK: Everything has been left on. Switch off all the table lamps, the TV and the computer, and close the fridge.",
   checkpoints=lambda ev: {"both lamps off": turned_off_n(ev, "tablelamp", 2), "TV off": turned_off(ev, "tv"), "computer off": turned_off(ev, "computer"), "fridge closed": closed(ev, "fridge")}),
  "mugs_hard": dict(
   instruction="TASK: Three mugs have been left lying around the flat — one in the bathroom, one in the living room and one on the bedroom nightstand. Collect all three and put them on the kitchen table.",
   checkpoints=lambda ev: {"all three mugs on the kitchen table": put_on_n(ev, "mug", "kitchentable", 3)}),
 "guest_hard": dict(
   instruction="TASK: A guest is coming and the living room was left as it was. Put the plate on the coffee table and the mug on the desk into the dishwasher, put the pillow from the sofa back on the bed, and switch the TV off.",
   checkpoints=lambda ev: {"plate in the dishwasher": put_in(ev, "plate", "dishwasher"), "mug in the dishwasher": put_in(ev, "mug", "dishwasher"), "pillow back on the bed": put_on(ev, "pillow", "bed"), "TV off": turned_off(ev, "tv")}),
 "aftermeal_hard": dict(
   instruction="TASK: Clear up after the meal in the living room: the two plates on the coffee table and the mug on the side table go in the dishwasher, the apple on the side table goes back in the fridge, and close both.",
   checkpoints=lambda ev: {"both plates in the dishwasher": put_in_n(ev, "plate", "dishwasher", 2), "mug in the dishwasher": put_in(ev, "mug", "dishwasher"), "apple in the fridge": put_in(ev, "apple", "fridge"), "dishwasher closed": closed(ev, "dishwasher"), "fridge closed": closed(ev, "fridge")}),

 "tidybedroom_hard": dict(
   instruction="TASK: Tidy up the bedroom: put the book that is lying on the bed back on the desk, turn off the lamp that was left on, close the cabinet, and take the mug from the nightstand to the kitchen table.",
   checkpoints=lambda ev: {"book back on the desk": put_on(ev, "book", "desk"), "cabinet closed": closed(ev, "cabinet"), "lamp off": turned_off(ev, "tablelamp"), "mug on the kitchen table": put_on(ev, "mug", "kitchentable")}),
}
