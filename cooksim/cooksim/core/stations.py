"""Stateful processing stations: pots, pans, cutting boards and sinks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .enums import PrepState
from .items import Ingredient, Plate


@dataclass
class CookStation:
    """A pot or pan. Holds ingredients and transforms them over time.

    Status lifecycle::

        empty -> filling (status="raw") -> cooking -> cooked -> burnt

    The station is recipe-agnostic: it simply transitions every ingredient
    from RAW/CHOPPED to COOKED, then to BURNT if left on the heat too long.
    """

    capacity: int = 3
    cook_time: int = 20
    burn_time: int = 30          # SMOKE window: ticks after the grace (smoke shown) before fire
    fire_time: int = 10          # ticks of FIRE after the smoke window, then the kitchen fails
    # GRACE: ticks after "cooked" during which the dish sits safely and shows NO smoke.
    # Added 2026-08-03: the dish smoked the tick after it finished, so a cook who walked to
    # the plates and back — the ordinary route on every map — returned to a smoking pan on a
    # run where nothing went wrong. Smoke then read as a mistake when it was just normal
    # timing, and it disqualified whole cells from carrying an error case at all.
    # Declared AFTER burn_time/fire_time on purpose: game.py builds every station with
    # POSITIONAL args (capacity, cook_time, burn_time), so inserting a field earlier would
    # silently bind burn_time to this one.
    smoke_grace: int = 10
    contents: List[Ingredient] = field(default_factory=list)
    cooking: bool = False
    timer: int = 0
    status: str = "empty"        # empty|raw|cooking|cooked|burnt
    auto_start: bool = True      # begin cooking automatically when full

    # -- queries -----------------------------------------------------------
    @property
    def is_full(self) -> bool:
        return len(self.contents) >= self.capacity

    def can_add(self) -> bool:
        return self.status in ("empty", "raw") and not self.is_full

    # -- mutations ---------------------------------------------------------
    def add(self, ing: Ingredient) -> bool:
        if not self.can_add():
            return False
        self.contents.append(ing)
        self.status = "raw"
        if self.auto_start and self.is_full:
            self.start()
        return True

    def start(self) -> bool:
        if self.status == "raw" and self.contents:
            self.cooking = True
            self.status = "cooking"
            self.timer = 0
            return True
        return False

    def take_into_plate(self, plate: Plate, plate_capacity: int = 3) -> bool:
        """Ladle cooked contents onto a clean plate (merging with whatever is
        already plated, e.g. rice + soup), provided there is room."""
        if self.status != "cooked" or plate.dirty:
            return False
        if len(plate.contents) + len(self.contents) > plate_capacity:
            return False
        plate.contents.extend(c.copy() for c in self.contents)
        self.reset()
        return True

    def clear(self) -> Optional[str]:
        """Empty a burnt station (contents are discarded). Returns reason."""
        if self.status == "burnt":
            self.reset()
            return "burnt_cleared"
        return None

    def reset(self) -> None:
        self.contents = []
        self.cooking = False
        self.timer = 0
        self.status = "empty"

    def tick(self) -> None:
        if not self.cooking:
            return
        self.timer += 1
        if self.timer == self.cook_time:
            for c in self.contents:
                c.state = PrepState.COOKED
            self.status = "cooked"
        elif self.timer >= self._fire_at and self.status != "burnt":
            for c in self.contents:
                c.state = PrepState.BURNT
            self.status = "burnt"           # FIRE shown; timer keeps running toward failure

    @property
    def _smoke_at(self) -> int:
        """First tick that shows smoke: cooked, then the grace period."""
        return self.cook_time + self.smoke_grace

    @property
    def _fire_at(self) -> int:
        return self.cook_time + self.smoke_grace + self.burn_time

    @property
    def smoking(self) -> bool:
        """Warning phase: cooked food left on the heat past the grace period."""
        return self.status == "cooked" and self.cooking and self.timer > self._smoke_at

    @property
    def fire_elapsed(self) -> int:
        return max(0, self.timer - self._fire_at)

    def render_state(self) -> dict:
        progress = 0.0
        if self.status == "cooking":
            progress = min(1.0, self.timer / max(1, self.cook_time))
        elif self.status in ("cooked", "burnt"):
            progress = 1.0
        # smoke-warning progress (0..1) once cooked, then fire progress (0..1)
        # 0 through the grace period, then climbs across the smoke window — so the bar only
        # starts moving once the dish is genuinely being left too long.
        burn_progress = 0.0
        if self.status == "cooked":
            burn_progress = min(1.0, max(0.0, self.timer - self._smoke_at) / max(1, self.burn_time))
        fire_progress = 0.0
        if self.status == "burnt":
            fire_progress = min(1.0, self.fire_elapsed / max(1, self.fire_time))
        return {
            "contents": [c.to_dict() for c in self.contents],
            "status": self.status,
            "progress": progress,
            "burn_progress": burn_progress,
            "smoking": self.smoking,
            "fire_progress": fire_progress,
            "capacity": self.capacity,
        }


@dataclass
class CuttingBoard:
    """Chops a single raw ingredient. Progress advances per interaction."""

    chop_time: int = 4
    item: Optional[Ingredient] = None
    progress: int = 0

    def place(self, ing: Ingredient) -> bool:
        if self.item is not None or ing.state != PrepState.RAW:
            return False
        self.item = ing
        self.progress = 0
        return True

    def chop(self) -> bool:
        """One unit of chopping work. Returns True if it did something."""
        if self.item is None or self.item.state != PrepState.RAW:
            return False
        self.progress += 1
        if self.progress >= self.chop_time:
            self.item.state = PrepState.CHOPPED
        return True

    def take(self) -> Optional[Ingredient]:
        item, self.item, self.progress = self.item, None, 0
        return item

    def render_state(self) -> dict:
        return {
            "item": self.item.to_dict() if self.item else None,
            "progress": min(1.0, self.progress / max(1, self.chop_time)),
            "done": self.item is not None and self.item.state == PrepState.CHOPPED,
        }


@dataclass
class Sink:
    """Holds dirty plates and washes them clean over repeated interactions."""

    wash_time: int = 5
    dirty: int = 0
    clean_ready: int = 0
    progress: int = 0

    def deposit_dirty(self) -> bool:
        self.dirty += 1
        return True

    def wash(self) -> bool:
        if self.dirty <= 0:
            return False
        self.progress += 1
        if self.progress >= self.wash_time:
            self.progress = 0
            self.dirty -= 1
            self.clean_ready += 1
        return True

    def take_clean(self) -> bool:
        if self.clean_ready > 0:
            self.clean_ready -= 1
            return True
        return False

    def render_state(self) -> dict:
        return {
            "dirty": self.dirty,
            "clean_ready": self.clean_ready,
            "progress": min(1.0, self.progress / max(1, self.wash_time)),
        }
