"""Tunable simulation parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GameConfig:
    # Capacities ----------------------------------------------------------
    plate_capacity: int = 5  # fits the 5-layer Loaded Burrito; matching is exact-multiset, so extra room is safe
    pot_capacity: int = 3
    pan_capacity: int = 2
    oven_capacity: int = 3
    fryer_capacity: int = 3      # a basket takes a whole chip batch without auto-starting
    # Timings (in ticks) --------------------------------------------------
    pot_cook_time: int = 20
    pot_burn_time: int = 45      # ruling 2026-10-09: smoke -> burnt in 45 ticks (30 from 09-30 left the forced walk-away unrescuable; 60 before)
    pan_cook_time: int = 15
    pan_burn_time: int = 45
    oven_cook_time: int = 28
    oven_burn_time: int = 45
    fryer_cook_time: int = 14    # deep fryer: fastest heat in the kitchen...
    fryer_burn_time: int = 45    # ruling 2026-09-11 (D-10): same smoke window as every other station
    fire_time: int = 15          # fire duration after the smoke window before the kitchen FAILS
    # Ticks a finished dish sits on the heat before ANY smoke appears. Long enough to walk
    # over and collect it, so smoke always means "left too long" and never "normal timing".
    # F4c (2026-08-14 fix batch): 10 -> 25. At LIVE pace the reference (CLEAN) plan smoked
    # on its own — the walk to the plates and back is slower once a conversation runs
    # alongside it — so "smoke" stopped meaning "something went wrong" and a spotless
    # episode looked hazardous. 25 clears the clean plan while leaving the hazard windows
    # intact: the leave_unattended deviation's 4x idle-8 busywork (32 ticks) still smokes.
    # Ruling 2026-09-30: 25 -> 40 (with burn 60 -> 30, so the dish is lost 70 ticks after it is done instead of
    # 85); the longer clean window replaces the deleted smoke hold — a pending plating mistake now has 40 ticks of
    # honest physics to fire in, and the harness never freezes food again.
    smoke_grace: int = 40
    chop_time: int = 4
    wash_time: int = 5
    # Orders --------------------------------------------------------------
    orders_enabled: bool = True
    max_orders: int = 4
    order_base_time: int = 110
    order_time_per_difficulty: int = 55
    expire_penalty: float = -6.0
    # Plates / serving ----------------------------------------------------
    return_dirty_plates: bool = True
    invalid_serve_penalty: float = -3.0
    trash_penalty: float = 0.0
    # Episode -------------------------------------------------------------
    horizon: int = 800
    # Reward shaping ------------------------------------------------------
    use_shaped_rewards: bool = True
    shaping_factor: float = 1.0
    shaped_rewards: Dict[str, float] = field(
        default_factory=lambda: {
            "place_in_pot": 0.3,
            "start_cooking": 0.2,
            "useful_chop": 0.3,
            "soup_pickup": 1.0,
            "plate_pickup": 0.1,
        }
    )

    def shaped(self, event: str) -> float:
        if not self.use_shaped_rewards:
            return 0.0
        return self.shaped_rewards.get(event, 0.0) * self.shaping_factor


# ---------------------------------------------------------------------------------------
# HUMAN-PACE PHYSICS (E8, hard-map audit 2026-08-15)
#
# Live episodes argue mid-cook, so the default burn timers torch the kitchen during any
# long conversation; the harness has always relaxed them. The relaxation used to be an
# inline patch inside tools/e2e_stepin_rollout.py, which meant the CASE GENERATORS
# validated every case sheet under DEFAULT physics (burn 45-60) while the episodes that
# were scored against those sheets ran under the relaxed physics (burn 480). A sheet
# could therefore certify an outcome the live run could never reach — e.g. every
# leave_unattended sheet declares `burned`, which at 480 needs ~600 ticks.
#
# One definition, applied by both. The inline patch also skipped `fryer_burn_time`, so
# anything reading the CONFIG for a fryer saw 45 while the fryer INSTANCES ran at 480.
HUMAN_PACE_BURN = 480
HUMAN_PACE_FIRE = 120


# TIGHT PACE (user ruling 2026-08-27: "make fire time much tighter, now it is impossible to get
# things on fire"). Burn windows go back to the default physics — a cooked item smokes after
# smoke_grace (40 since 2026-09-30; 25 before) and ignites burn_time later (45 for every station since 2026-10-09, 30 from 09-30;
# 60 before, the fryer 45 before 2026-09-11), i.e. the dish is lost 70 ticks after it is done. Calibration from the 2026-08 sweep: the flawless clean plan
# never leaves a cooked item on heat longer than 37 ticks (master / hard map), a PREVENTED
# leave_unattended keeps the cook away a median 14 ticks (3% exceed 70), an UNCAUGHT one a median
# 133 ticks (95% exceed 70). Selected with COOKSIM_PACE=tight; the default stays "human" (480/120).
TIGHT_PACE_BURN = None          # None = keep each station's default burn_time (45 everywhere)
TIGHT_PACE_FIRE = 30            # ticks of open fire before the kitchen is lost (was 15 by default)
PACE_PRESETS = {"human": (HUMAN_PACE_BURN, HUMAN_PACE_FIRE), "tight": (TIGHT_PACE_BURN, TIGHT_PACE_FIRE)}


def pace_preset(name=None):
    """(burn, fire) for COOKSIM_PACE (default human). burn None keeps per-station defaults."""
    import os
    name = name or os.environ.get("COOKSIM_PACE", "human")
    if name not in PACE_PRESETS:
        raise ValueError(f"COOKSIM_PACE must be one of {sorted(PACE_PRESETS)}, got {name!r}")
    return name, PACE_PRESETS[name]


def apply_pace(game, name=None):
    """Apply the named preset (env COOKSIM_PACE by default) and return its name."""
    name, (burn, fire) = pace_preset(name)
    if burn is None:
        game.config.fire_time = fire
        for _st in game.stations.values():
            if hasattr(_st, "fire_time"):
                _st.fire_time = fire
    else:
        apply_human_pace(game, burn, fire)
    game.stats["pace"] = name
    return name


def apply_human_pace(game, burn: int = HUMAN_PACE_BURN, fire: int = HUMAN_PACE_FIRE):
    """Relax burn/fire timings for a human-paced episode, on the config AND on the live
    station instances (stations snapshot config at construction, so both are required).
    Returns the game for chaining."""
    for _k in ("pot_burn_time", "pan_burn_time", "oven_burn_time", "fryer_burn_time"):
        setattr(game.config, _k, burn)
    game.config.fire_time = fire
    for _st in game.stations.values():
        if hasattr(_st, "burn_time"):
            _st.burn_time = burn
        if hasattr(_st, "fire_time"):
            _st.fire_time = fire
    return game


# ---------------------------------------------------------------------------------------
# AGENT INTERFACE VERSION (user ruling 2026-08-15)
#
# Two agent interfaces live in this codebase and both must stay runnable, so that every
# experiment already published can be reproduced and so that v1 and v2 can be compared
# head to head in one sweep.
#
#   v1  the interface every published page was run on. The cook sees rendered FRAMES on
#       the creative tier and a JSON state elsewhere; both agents share one CLOSED spatial
#       vocabulary (left/right/nearest/...); an unqualified reference to a duplicated
#       station silently resolves to the nearest one; the assistant's readout is written
#       in (x,y) coordinates.
#
#   v2  the 2026-08-15 redesign. The cook is a TEXT agent reading a state map that prints
#       every object's id and coordinate, and it grounds its own body exactly (id or
#       coordinate); more than one instance of a kind ALWAYS requires grounding; the
#       assistant is never shown ids or coordinates at all and describes places in words.
#
# Default v2. Set COOKSIM_AGENT_VERSION=v1 to run the old interface. The value is stamped
# into every episode report, so a report can never be ambiguous about which one produced it.
AGENT_VERSION_DEFAULT = "v2"


def agent_version() -> str:
    """"v1" or "v2" — read at call time so a sweep can switch per episode."""
    import os as _os
    v = (_os.environ.get("COOKSIM_AGENT_VERSION") or AGENT_VERSION_DEFAULT).strip().lower()
    return "v1" if v == "v1" else "v2"


def is_v1() -> bool:
    return agent_version() == "v1"


# ---------------------------------------------------------------------------------------
# FORGET_NEXT_STEP FREEZE (user ruling 2026-09-20)
#
# How many ticks the cook stands lost when `forget_next_step` fires. It used to be written
# out twice as a bare 60 — `plan_ir.forget_rest(stall_ticks=60)` for the scheduled block and
# a literal `"idle 60"` in the live harness's `live_dev_block` — and the measured
# deviation-to-commit window that came out of it (docs/FIRING_AND_GRADING.md §1.7) was 62.0
# ticks (n 30, min 61, max 75): by far the longest catchable window in the suite, 1.7x the
# next longest (wrong_order, 37.0).
#
# The freeze is not only long, it DISPLACES: nothing else may take the cook's body while it
# runs, so every later takeover in the episode slides 60+ ticks to the right. That is what
# starved `leave_unattended`, whose moment is pinned to its own food (it may fire only from
# LU_NEAR ticks before that food is done to LU_LATE ticks after) — the displaced next
# takeover landed clean past the window.
#
# 35 = COOKSIM_ASSIST_LATENCY_MAX (30, the most a single assistant reply may be charged,
# D-1 ruling 2026-09-11) + FORGET_REACT_TICKS (5, the cook's walk back to the food). Even a
# reply that took the maximum chargeable time to arrive still lands inside the freeze, and on
# the measured reply chain (216 API-assistant episodes, 10,098 replies: poll <= 1 tick,
# latency p50 4 / p75 6 / p90 8, correct flag -> first corrective command p50 3 / p75 8) a
# p90 assistant has 35 - 17 = 18 ticks to spare. The window it produces is ~37 ticks, which
# puts forget level with wrong_order instead of half again longer than anything else.
FORGET_FREEZE_TICKS = 35


def forget_freeze_ticks() -> int:
    """Ticks the forget_next_step stall lasts. COOKSIM_FORGET_FREEZE overrides."""
    import os as _os
    v = _os.environ.get("COOKSIM_FORGET_FREEZE")
    return max(1, int(v)) if v else FORGET_FREEZE_TICKS
