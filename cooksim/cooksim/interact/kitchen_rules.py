"""THE RULES OF THIS KITCHEN — one rulebook, generated from the engine, given to BOTH sides.

Ported verbatim from the study app (hellomuffin/cooksim-study-app @ 0facdb3, 2026-09-09) on the
user's instruction of 2026-09-12: the benchmark cook and the benchmark assistant must read exactly
the wording a study participant reads. Before this there were TWO hand-written copies — the
assistant's `== THE GAME AND ITS RULES ==` paragraph and the cook's `kitchen_rules_text()` — sharing
only `station_rules_text()`; a hand-written copy of the rules drifts from the code (ECR-13: the
duplicate rule set in `checks.is_doable` disagreed with the engine oracle in 1087 of 2162 audited
cells), and both copies left implicit the facts that decide whether advice is even executable:
a plate needs EMPTY hands, cooking is load-then-turn-on, food leaves a heat station only onto a
held plate.

`kitchen_rules_struct(game)` is the rules as DATA, read out of GameConfig / STATION_ACCEPTS /
CHOPPABLE and the tiles this map actually has. `kitchen_rules_prompt_text(game)` renders it in the
cook's second person; the assistant is handed the same text, quoted, so it knows the exact wording
the cook was given and can use those words back.
"""

def kitchen_rules_struct(game=None) -> dict:
    """THE RULES OF THIS KITCHEN, as data, generated from the engine so a rule can never
    contradict the code (station table, timings and capacities all come from GameConfig /
    STATION_ACCEPTS / CHOPPABLE).

    Structure (2026-09-09 rewrite, user ruling): the old version interleaved "what state must
    an item be in" with "how do I work each tool", so a participant had to reassemble the
    procedure themselves. It is now ordered the way a person actually needs it —
        1 GOAL          what you are making and what counts as done
        2 OUTCOME       exactly what ends the episode, win or lose
        3 BODY          how you move and act, and the one-thing-in-your-hands rule
        4 STATIONS      every tile you can face, and what Space does at it in each hand-state
        5 ITEMS         the three states, how to reach each, what each station accepts
        6 PLATE         building and carrying the dish
        7 HEAT          the cooked -> smoking -> burnt timeline
    and it states the facts the previous version left implicit: a plate can only be taken with
    EMPTY hands; a held ingredient can be parked on a counter or binned, not only a plate;
    cooking is TWO steps (load it, then turn it on) unless the station fills up and starts
    itself; food can only leave a heat station onto a held plate, never into your hands.
    """
    from ..core.enums import STATION_ACCEPTS, CHOPPABLE, Terrain
    if game is None:                       # no episode in hand (prompt showcase, audits):
        from ..core.config import GameConfig    # default kitchen, every station shown
        cfg = GameConfig()
        present = {t.value for t in Terrain}
    else:
        cfg = game.config
        present = {t.value for row in game.layout.grid for t in row}
    here = lambda name: name in present
    burn_of = {"pot": cfg.pot_burn_time, "pan": cfg.pan_burn_time,
               "fryer": cfg.fryer_burn_time, "oven": cfg.oven_burn_time}
    cook_of = {"pot": cfg.pot_cook_time, "pan": cfg.pan_cook_time,
               "fryer": cfg.fryer_cook_time, "oven": cfg.oven_cook_time}
    cap_of = {"pot": cfg.pot_capacity, "pan": cfg.pan_capacity,
              "fryer": cfg.fryer_capacity, "oven": cfg.oven_capacity}
    heat = []
    for terrain in (Terrain.POT, Terrain.PAN, Terrain.FRYER, Terrain.OVEN):
        keys = sorted(STATION_ACCEPTS.get(terrain, ()))
        if not keys:
            continue
        n = terrain.value
        heat.append({"name": n, "in_this_kitchen": here(n),
                     "takes": [f"{st} {ing}" if st != "raw" else ing for ing, st in keys],
                     "cook_ticks": cook_of[n], "burn_ticks": burn_of[n], "capacity": cap_of[n]})
    burns = sorted({h["burn_ticks"] for h in heat})

    # a crate tile is <ingredient>_source: this kitchen has one if any such tile exists
    crates = sorted({t.replace("_source", "") for t in present
                     if t.endswith("_source") and t != "plate_source"})

    # every tile you can stand in front of, and what SPACE does there in each hand-state
    def row(name, present_key, what, empty, ingredient, plate):
        ok = bool(crates) if present_key == "__crates__" else here(present_key)
        return {"name": name, "in_this_kitchen": ok, "what": what,
                "empty": empty, "ingredient": ingredient, "plate": plate}

    heat_names = " / ".join(h["name"] for h in heat if h["in_this_kitchen"]) or "pan / pot / fryer / oven"
    stations = [
        row("ingredient crate", "__crates__", "An endless supply of ONE raw ingredient. Its label says which.",
            "take one raw item into your hands",
            "nothing (your hands are already full)",
            "scoop one raw item straight onto the plate"),
        row("plate stack", "plate_source", "Where clean plates come from.",
            "take a clean plate",
            "nothing — you must have EMPTY hands to take a plate, so park or bin what you hold first",
            "nothing (you are already holding a plate)"),
        row("cutting board", "cutting_board", "Chops ONE raw item at a time. Holds one item only.",
            f"CHOP the item lying on it — press Space about {cfg.chop_time} times, and watch the progress bar; "
            f"once it is chopped, the same press picks it up instead",
            "put the raw item down on the board, if the board is empty and the item can be chopped",
            "scoop the finished chopped item onto the plate"),
        row("heat station (" + heat_names + ")", "pan",
            "Cooks what it accepts. Shows a PROGRESS BAR while cooking.",
            "TURN IT ON, if it is loaded and not yet running",
            "put the item in, if this station accepts it in that state; the station starts ITSELF the moment it is full",
            "scoop the finished food out onto the plate — this is the ONLY way food leaves a heat station; "
            "you can never take it back into your hands"),
        row("counter", "counter", "Free surface. Holds one loose item.",
            "pick up whatever is on it",
            "put the item down, if the counter is empty; if a PLATE is sitting there, the item goes onto that plate instead",
            "put the plate down, if the counter is empty; if an INGREDIENT is sitting there, it goes onto your plate instead"),
        row("serving pass", "serving", "Where a finished dish is handed over.",
            "nothing", "nothing (food is served on a plate, never by hand)",
            "SERVE the plate — this ends the episode, win or lose"),
        row("trash", "trash", "Bin. Use it to undo a mistake.",
            "nothing",
            "throw the held item away",
            "dump everything off the plate; you keep the (now empty) plate"),
    ]

    return {
        "goal": ("Cook the ORDER and hand it over at the serving pass. Every item on the list must be on ONE "
                 "plate, in the state the list names, with nothing extra and nothing missing."),
        "goal_points": [
            "The pass accepts an EXACT match only: the right items, each in the right state, no extras.",
            "If the order is marked strict, the items must have been added to the plate in exactly the listed order.",
            "Burnt food never counts, whatever else is on the plate.",
        ],
        "outcome": {
            "win": "You serve a plate that matches the order. That is the only way to win.",
            "lose": [
                "You serve a plate that is wrong, incomplete or empty. The pass rejects it and the episode ends.",
                "Any food burns. The episode ends the moment a station catches fire.",
            ],
            "note": "There is no time limit and no other way for the episode to end. Standing still is safe; "
                    "the only clock that matters is food already on the heat.",
        },
        "body": [
            "An ARROW KEY first TURNS you to face that direction. Pressing it again WALKS you one tile that way.",
            "SPACE acts on the single tile you are FACING. What it does depends on that tile and on what is in your hands.",
            "Your hands hold ONE thing at a time: one ingredient, or one plate. Several actions need your hands EMPTY.",
            "To empty your hands: put the item down on an empty counter, or bin it at the trash.",
        ],
        "stations": stations,
        "crates": crates,
        "states": [
            {"state": "raw", "title": "RAW",
             "how": "Straight out of the crate. Nothing to do to it.",
             "get": "Face the crate with empty hands and press Space to take one, or press Space holding the plate to scoop one straight on."},
            {"state": "chopped", "title": "CHOPPED",
             "how": "Cut up on a cutting board. Some recipes want an item chopped; some heat stations only accept it chopped.",
             "get": f"Carry the raw item to a board, put it down, then press Space about {cfg.chop_time} times to chop it. "
                    f"Take it back with empty hands, or scoop it onto your held plate."},
            {"state": "cooked", "title": "COOKED",
             "how": "Cooked through at a heat station. Watch the progress bar fill.",
             "get": "TWO steps: put the item into a station that accepts it, then turn that station on with EMPTY hands. "
                    "(A station that becomes FULL starts itself.) When the bar is full, scoop it out onto your held plate."},
        ],
        "chop_ok": sorted(CHOPPABLE),
        "chop_note": "The board refuses anything else — a patty is already shaped, a bun and a tortilla are "
                     "used whole, rice and noodles go straight into the pot.",
        "heat": heat,
        "heat_note": "Each station accepts only what is listed, in exactly that state, and refuses everything else.",
        "plate": [
            "Take a clean plate from the plate stack — your hands must be EMPTY.",
            "Add an item by facing its source, board or finished station while HOLDING the plate, and pressing Space.",
            "A strict order must be built in the listed order: what you scoop first sits at the bottom.",
            "You cannot take an item back off a plate. To undo, bin the plate's contents at the trash and rebuild it.",
            f"A plate holds up to {cfg.plate_capacity} items.",
            "To free your hands, put the plate down on an EMPTY counter. It cannot rest on a board, a station or a crate.",
        ],
        "timeline": {"grace": cfg.smoke_grace, "burn": burns,
                     "burn_by_station": {h["name"]: h["burn_ticks"] for h in heat},
                     "cook_by_station": {h["name"]: h["cook_ticks"] for h in heat}},
        "timeline_note": ("Food is only in danger once it has finished cooking. After it is done it sits SAFELY for "
                          f"{cfg.smoke_grace} ticks; then the station SMOKES as a warning; if you still leave it, "
                          "it BURNS and the episode is over. Scooping the food out is the only thing that stops "
                          "smoke — there is no way to turn a station off."),
        "talk": "Time keeps running while you talk to the assistant.",
    }


def kitchen_rules_prompt_text(game=None, interface: str = "commands") -> str:
    """The COOK'S OWN RULEBOOK as plain text, rendered from kitchen_rules_struct() — the same
    single source the participant's page renders (2026-09-09 ruling: the assistant must be told
    exactly what the person was told). Kept in the cook's second person on purpose: the block is
    quoted to the assistant as "the rulebook they were shown", so the assistant knows the exact
    wording the human read and can use the same terms when talking to them."""
    r = kitchen_rules_struct(game)
    L = []
    L.append("1. WHAT THE COOK IS TRYING TO DO")
    L.append("   " + r["goal"])
    L += ["   - " + x for x in r["goal_points"]]
    o = r["outcome"]
    L.append("")
    L.append("2. HOW THE EPISODE ENDS")
    L.append("   WIN:  " + o["win"])
    for x in o["lose"]:
        L.append("   LOSE: " + x)
    L.append("   " + o["note"])
    L.append("")
    L.append("3. THE COOK'S BODY: MOVING AND ACTING")
    L += ["   - " + x for x in r["body"]]
    L.append("")
    L.append("4. THE STATIONS, AND WHAT PRESSING SPACE DOES AT EACH")
    for st in r["stations"]:
        tail = "" if st["in_this_kitchen"] else "   [NOT IN THIS KITCHEN]"
        L.append(f"   {st['name'].upper()}{tail} — {st['what']}")
        L.append(f"       with empty hands   : {st['empty']}")
        L.append(f"       holding an item    : {st['ingredient']}")
        L.append(f"       holding a plate    : {st['plate']}")
    if r.get("crates"):
        L.append("   Crates in this kitchen: " + ", ".join(r["crates"]))
    L.append("")
    L.append("5. ITEMS AND THE THREE STATES THEY CAN BE IN")
    for p_ in r["states"]:
        L.append(f"   {p_['title']} — {p_['how']}")
        L.append(f"       how to get it: {p_['get']}")
    L.append("   Only these can be chopped: " + ", ".join(r["chop_ok"]) + ". " + r["chop_note"])
    L.append("   What each heat station accepts (it refuses everything else):")
    for h in r["heat"]:
        tail = "" if h["in_this_kitchen"] else "  [NOT IN THIS KITCHEN]"
        L.append(f"       {h['name']}: {', '.join(h['takes'])}  (holds {h['capacity']}, "
                 f"cooks in {h['cook_ticks']} ticks){tail}")
    L.append("")
    L.append("6. BUILDING THE PLATE")
    L += ["   - " + x for x in r["plate"]]
    L.append("")
    L.append("7. HEAT, SMOKE AND BURNING")
    tl = r["timeline"]
    burns = " or ".join(str(b) for b in tl["burn"])
    L.append(f"   food DONE -> safe for {tl['grace']} ticks -> station SMOKES -> {burns} more ticks "
             f"-> BURNT, and the episode is over")
    L.append("   per station: " + ", ".join(f"{k} {v}" for k, v in tl["burn_by_station"].items()))
    L.append("   " + r["timeline_note"])
    L.append("   " + r["talk"])
    text = "\n".join(L)
    if game is not None:
        chop = game.config.chop_time
    else:
        from ..core.config import GameConfig
        chop = GameConfig().chop_time
    return text if interface == "keys" else as_commands(text, chop)

# ---------------------------------------------------------------- interface phrasing
# The study app's rulebook addresses a PERSON AT A KEYBOARD ("an ARROW KEY", "press Space").
# Our cook and our assistant speak the engine's command vocabulary and have no keyboard, so the
# text would tell them to do something that does not exist. Every sentence is kept verbatim
# except the interface phrases below, which are swapped one-for-one; nothing else differs from
# hellomuffin/cooksim-study-app @ 0facdb3. Render with interface="keys" to get the study app's
# text byte-for-byte.
COMMAND_PHRASING = [
    ("An ARROW KEY first TURNS you to face that direction. Pressing it again WALKS you one tile that way.",
     "\"go to the <place>\" walks you to a station and leaves you standing at it, facing it."),
    ("SPACE acts on the single tile you are FACING. What it does depends on that tile and on what is in your hands.",
     "Every other action acts on the station you are AT. What each one does depends on that station and on what is in your hands."),
    ("4. THE STATIONS, AND WHAT PRESSING SPACE DOES AT EACH",
     "4. THE STATIONS, AND WHAT YOUR ACTIONS DO AT EACH"),
    ("Face the crate with empty hands and press Space to take one, or press Space holding the plate "
     "to scoop one straight on.",
     "\"pick up\" at the crate with empty hands takes one, or \"scoop\" while holding the plate puts "
     "one straight on."),
    ("put the raw item down on the board", "\"put down\" the raw item on the board"),
    ("then press Space about {chop} times to chop it", "then \"chop\" about {chop} times"),
    ("press Space about {chop} times, and watch the progress bar", "\"chop\" about {chop} times, and watch the progress bar"),
    ("once it is chopped, the same press picks it up instead", "once it is chopped, \"pick up\" takes it instead"),
    ("while HOLDING the plate, and pressing Space", "while HOLDING the plate, and \"scoop\""),
    ("TURN IT ON, if it is loaded", "\"turn on\", if it is loaded"),
    ("take one raw item into your hands", "\"pick up\" takes one raw item into your hands"),
    ("take a clean plate", "\"pick up\" takes a clean plate"),
    ("scoop one raw item straight onto the plate", "\"scoop\" puts one raw item straight onto the plate"),
    ("pick up whatever is on it", "\"pick up\" takes whatever is on it"),
    ("throw the held item away", "\"trash\" throws the held item away"),
    ("SERVE the plate", "\"serve\" hands the plate over"),
]


def as_commands(text: str, chop_time: int) -> str:
    """Swap the keyboard phrasing for the engine's command vocabulary (see COMMAND_PHRASING)."""
    for a, b in COMMAND_PHRASING:
        a = a.replace("{chop}", str(chop_time))
        b = b.replace("{chop}", str(chop_time))
        text = text.replace(a, b)
    return text
