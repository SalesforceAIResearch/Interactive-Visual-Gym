"""App interface + registry.

An App owns a slice of `state.data[app_id]`, provides:
  - initial_data(rng) -> dict            seeded starting state
  - view(state) -> Screen                current screen for its keys
  - handle(state, event, gesture) -> None|str   mutate state on an event;
        may return a spoken engine note (rare)

Events are tuples emitted by widgets (their on_tap/on_swipe/on_long) or the
shell. The engine routes an event to the app that owns its screen prefix.
"""
from __future__ import annotations

from typing import Optional

from .model import Screen
from .state import DeviceState

_REGISTRY: dict = {}


class App:
    id: str = ""
    name: str = ""
    icon: str = ""          # emoji or glyph key for the home grid
    color: str = "#5b6b8c"
    root_key: str = ""      # screen key opened when launched

    def initial_data(self, rng) -> dict:
        return {}

    def view(self, state: DeviceState) -> Screen:
        raise NotImplementedError

    def handle(self, state: DeviceState, event: tuple, gesture) -> Optional[str]:
        return None


def register(app: App):
    _REGISTRY[app.id] = app
    return app


def get(app_id: str) -> App:
    return _REGISTRY[app_id]


def get_opt(app_id: str):
    """Like get(), but returns None for the shell/home pseudo-app instead of
    raising — used where a timer or event may fire outside any app."""
    return _REGISTRY.get(app_id)


def all_apps() -> list:
    return list(_REGISTRY.values())


def owner_of(screen_key: str) -> Optional[App]:
    prefix = screen_key.split(".")[0]
    return _REGISTRY.get(prefix)
