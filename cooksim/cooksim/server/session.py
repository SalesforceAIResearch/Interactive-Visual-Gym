"""A single live game session driven by the WebSocket server."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..agents import GreedyChef, RandomAgent
from ..core.config import GameConfig
from ..core.enums import MOVE_ACTIONS, Action
from ..core.game import KitchenGame
from ..core.layout import Layout, generate_layout
from ..layouts import load_layout


class GameSession:
    def __init__(self):
        self.game: Optional[KitchenGame] = None
        self.pending: Dict[int, int] = {}          # player -> action (persistent)
        self.bots: Dict[int, object] = {}          # player -> agent callable
        self.tps: float = 7.0
        self.layout_name: str = "cramped_room"
        # HUMAN-STUDY move throttle (0 = off, the default for every non-study flow):
        # wall-clock cooldown between GRID STEPS for human seats. The engine's
        # orient-first rule means a move action toward a NEW facing only TURNS the cook
        # that tick — we let those through untouched (turning stays instant), and only
        # suppress the tick that would actually WALK while the cooldown is running.
        # Interact/Space is never touched.
        self.move_ms: int = 0
        self._last_walk: Dict[int, float] = {}     # player -> wall-clock ms of last step
        self.reset(layout="cramped_room")

    def reset(
        self,
        layout=None,
        layout_data: Optional[dict] = None,
        procedural: Optional[dict] = None,
        n_players: Optional[int] = None,
        config: Optional[dict] = None,
        seed: Optional[int] = 0,
        keep_bots: bool = True,
        order_recipes: Optional[list] = None,
        move_ms: Optional[int] = None,
    ):
        if move_ms is not None:
            self.move_ms = max(0, min(3000, int(move_ms)))
        self._last_walk = {}
        if layout_data is not None:
            lay = Layout.from_dict(layout_data)
            self.layout_name = lay.name
        elif procedural is not None:
            lay = generate_layout(seed=seed, **procedural)
            self.layout_name = lay.name
        else:
            name = layout or self.layout_name
            lay = load_layout(name)
            self.layout_name = name

        cfg = GameConfig(**config) if config else GameConfig()
        self.game = KitchenGame(lay, config=cfg, n_players=n_players, seed=seed,
                                order_recipe_ids=order_recipes)
        self.pending = {i: int(Action.STAY) for i in range(self.game.n_players)}
        if not keep_bots:
            self.bots = {}
        # drop bots that reference players who no longer exist
        self.bots = {p: b for p, b in self.bots.items() if p < self.game.n_players}

    # -- control ----------------------------------------------------------
    def set_action(self, player: int, action: int):
        if 0 <= player < self.game.n_players:
            self.pending[player] = int(action)

    def add_bot(self, player: int, kind: str = "greedy", role: str = "any"):
        if 0 <= player < self.game.n_players:
            self.bots[player] = (
                GreedyChef(seed=player + 1, role=role) if kind == "greedy"
                else RandomAgent(player)
            )

    def remove_bot(self, player: int):
        self.bots.pop(player, None)

    def _throttle_move(self, player: int, action: int) -> int:
        """Apply the human move cooldown: pure re-orients pass instantly; a tick that
        would WALK is replaced by STAY until move_ms has elapsed since the last step."""
        if self.move_ms <= 0:
            return action
        act = Action(action)
        if act not in MOVE_ACTIONS:
            return action                      # interact / stay: never throttled
        cook = self.game.cooks[player]
        if cook.direction != MOVE_ACTIONS[act]:
            return action                      # this tick only turns — instant
        now = time.time() * 1000.0
        if now - self._last_walk.get(player, 0.0) < self.move_ms:
            return int(Action.STAY)            # cooldown running: suppress the step
        self._last_walk[player] = now
        return action

    def tick(self):
        if self.game is None:
            return
        actions: List[int] = []
        for i in range(self.game.n_players):
            if i in self.bots:
                actions.append(int(self.bots[i](self.game, i)))
            else:
                actions.append(self._throttle_move(i, self.pending.get(i, int(Action.STAY))))
        self.game.step(actions)

    def state(self) -> dict:
        st = self.game.render_state()
        st["layout_name"] = self.layout_name
        st["bots"] = sorted(self.bots.keys())
        st["tps"] = self.tps
        st["n_players"] = self.game.n_players
        return st
