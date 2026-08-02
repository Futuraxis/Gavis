"""Game session management for the Stochastic Gomoku play app.

Each session owns a ``GameEngine`` (v5.0 rules) and an ``MCTS`` solver.
Unlike Moon Chess, every placement is followed by a chance node (50%
vanish), so both the human and the AI move resolve the chance step and
report whether the stone vanished.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import json

from layer2_engine.core.engine import GameEngine
from layer3_solvers.base import SolverConfig
from layer3_solvers.mcts import MCTS

PlayerColor = Literal['p_black', 'p_white']
Difficulty = Literal['easy', 'normal', 'hard']

# MCTS budgets per difficulty tier.  Measured on a 9×9 board:
# ~1.6 ms per iteration → ~0.5 s / ~2.4 s / ~6.5 s per move.
DIFFICULTY_BUDGETS: dict[Difficulty, int] = {
    'easy': 300,
    'normal': 1500,
    'hard': 4000,
}

RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / 'rules' / 'stochastic_gomoku.json'


class PlayError(Exception):
    """Bad play request (unknown game, illegal move, game over, ...)."""


def _resolve_chance(engine: GameEngine, state: dict) -> tuple[dict, Optional[int]]:
    """Resolve all pending chance nodes; returns (state, vanished_cell)."""
    vanished: Optional[int] = None
    while engine.get_node_type(state) == 'chance':
        outcome, state = engine.sample_chance(state)
        if outcome.key == 'vanish':
            vanished = _find_vanished(engine, state)
    return state, vanished


def _find_vanished(engine: GameEngine, state: dict) -> Optional[int]:
    """Locate the cell that vanished (lastActor's most recent piece gone)."""
    board = state['_arrays']['board']
    last_actor = state['env'].get('lastActor')
    last_cell = state['env'].get('lastPlacedCell')
    if last_actor is None or last_cell is None:
        return None
    try:
        _, r, c = str(last_cell).split('_')
        idx = int(r) * 9 + int(c)
    except ValueError:
        return None
    return idx if 0 <= idx < len(board) and board[idx] is None else None


@dataclass
class GameSession:
    """One human-vs-AI Stochastic Gomoku game."""

    game_id: str
    player_color: PlayerColor
    difficulty: Difficulty
    engine: GameEngine
    solver: MCTS
    state: dict = field(init=False)
    last_ai_move: Optional[int] = None
    last_vanish: Optional[int] = None  # cell that vanished on the latest move

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    @property
    def ai_color(self) -> PlayerColor:
        return 'p_white' if self.player_color == 'p_black' else 'p_black'

    @property
    def board(self) -> list:
        return self.state['_arrays']['board']

    @property
    def winner(self) -> Optional[str]:
        return self.state['env'].get('winner')

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def current_player(self) -> Optional[str]:
        return self.engine.get_current_player(self.state)

    # ── Play actions ────────────────────────────────────────────────

    def ai_move(self) -> Optional[int]:
        """AI places a stone and resolves the vanish chance."""
        if self.over or self.current_player != self.ai_color:
            return None
        action = self.solver.select_action(self.state)
        if action is None:
            return None
        self.state = self.engine.apply_action(self.state, action)
        self.last_ai_move = self._cell_index(action)
        self.state, self.last_vanish = _resolve_chance(self.engine, self.state)
        return self.last_ai_move

    def human_move(self, cell_index: int) -> None:
        """Apply the human's move (incl. vanish resolution)."""
        if self.over:
            raise PlayError(f'game {self.game_id} is over')
        if self.current_player != self.player_color:
            raise PlayError('not your turn')
        action = self._find_action(cell_index)
        if action is None:
            raise PlayError(f'cell {cell_index} is not a legal move')
        self.state = self.engine.apply_action(self.state, action)
        self.state, self.last_vanish = _resolve_chance(self.engine, self.state)

    def _find_action(self, cell_index: int) -> Optional[object]:
        for action in self.engine.get_legal_actions(self.state):
            if self._cell_index(action) == cell_index:
                return action
        return None

    @staticmethod
    def _cell_index(action: object) -> int:
        cell = action.params.get('cell', {})
        return int(cell.get('_index', -1))

    # ── Serialization ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            'game_id': self.game_id,
            'player_color': self.player_color,
            'difficulty': self.difficulty,
            'board': self.board,
            'turn': self.current_player,
            'winner': self.winner,
            'over': self.over,
            'last_ai_move': self.last_ai_move,
            'last_vanish': self.last_vanish,
            'round': self.state['env'].get('round', 0),
        }


class PlayManager:
    """Registry of active game sessions (in-memory, single-process)."""

    def __init__(self, seed: int = 42) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._seed = seed

    def start(self, player_color: PlayerColor | str, difficulty: Difficulty) -> GameSession:
        if player_color == 'random':
            player_color = ('p_black' if uuid.uuid4().int % 2 == 0 else 'p_white')
        if player_color not in ('p_black', 'p_white'):
            raise PlayError(f'unknown player color: {player_color}')
        if difficulty not in DIFFICULTY_BUDGETS:
            raise PlayError(f'unknown difficulty: {difficulty}')

        with open(RULES_PATH, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        game_id = uuid.uuid4().hex[:8]
        engine = GameEngine(rules, seed=self._seed)
        solver = MCTS(engine, SolverConfig(seed=self._seed))
        solver.budget = DIFFICULTY_BUDGETS[difficulty]

        session = GameSession(
            game_id=game_id,
            player_color=player_color,  # type: ignore[arg-type]
            difficulty=difficulty,      # type: ignore[arg-type]
            engine=engine,
            solver=solver,
        )
        if session.player_color == 'p_white':
            session.ai_move()
        self._sessions[game_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f'unknown game: {game_id}')
        return session

    def remove(self, game_id: str) -> None:
        self._sessions.pop(game_id, None)
