"""Game session management for the Moon Chess play app.

Holds a ``game_id → GameSession`` registry; each session owns a bare
``GameEngine`` (v5.2 — built from ``rules/moon_chess.json``, no per-game
adapter) and a solver handle.  The solver is never instantiated here —
Layer 4 receives a ``SolverProvider`` (assembled by the app layer,
``train-cli/games.py``) and asks it for a handle, so this module
contains no Layer 3 reference.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal

from layer2_engine.core.engine import GameEngine

from ...solver_provider import SolverHandle, SolverProvider
from ..engine_helpers import engine_from_rules

PlayerColor = Literal["p_black", "p_white"]
Difficulty = Literal["easy", "normal", "hard"]

# MCTS budgets per difficulty tier.  Measured on a 3×3 board with the
# line-scan heuristic rollout: ~0.6 ms per iteration → ~0.15 s / ~0.5 s /
# ~1.3 s per move.  Even 'easy' blocks immediate threats correctly.
DIFFICULTY_BUDGETS: dict[Difficulty, int] = {
    "easy": 200,
    "normal": 800,
    "hard": 2000,
}


class PlayError(Exception):
    """Bad play request (unknown game, illegal move, game over, ...)."""


@dataclass
class GameSession:
    """One human-vs-AI Moon Chess game.

    ``turn`` tracks whose move it is in the HUMAN's perspective; the AI
    responds on the other player's turn.  ``last_ai_move`` is the cell
    index of the most recent AI move (None before the AI has moved).
    """

    game_id: str
    player_color: PlayerColor
    difficulty: Difficulty
    engine: GameEngine
    solver: SolverHandle
    state: dict = field(init=False)
    last_ai_move: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    @property
    def ai_color(self) -> PlayerColor:
        return "p_white" if self.player_color == "p_black" else "p_black"

    @property
    def board(self) -> list:
        """9-element board array (None / 'p_black' / 'p_white')."""
        return self.state["_arrays"]["board"]

    @property
    def winner(self) -> str | None:
        return self.state["env"].get("winner")

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def current_player(self) -> str | None:
        return self.engine.get_current_player(self.state)

    # ── Play actions ────────────────────────────────────────────────

    def ai_move(self) -> int | None:
        """Let the AI choose and apply a move; returns the cell index."""
        if self.over or self.current_player != self.ai_color:
            return None
        action = self.solver.select_action(self.state)
        if action is None:
            return None
        self.state = self.engine.apply_action(self.state, action)
        self.last_ai_move = self._cell_index(action)
        return self.last_ai_move

    def human_move(self, cell_index: int) -> None:
        """Apply the human's move; raises PlayError on illegal moves."""
        if self.over:
            raise PlayError(f"game {self.game_id} is over")
        if self.current_player != self.player_color:
            raise PlayError("not your turn")
        action = self._find_action(cell_index)
        if action is None:
            raise PlayError(f"cell {cell_index} is not a legal move")
        self.state = self.engine.apply_action(self.state, action)

    def _find_action(self, cell_index: int):
        for action in self.engine.get_legal_actions(self.state):
            if self._cell_index(action) == cell_index:
                return action
        return None

    @staticmethod
    def _cell_index(action: object) -> int:
        cell = action.params.get("cell", {})
        cell_id = cell.get("id", "") if isinstance(cell, dict) else str(cell)
        _, r, c = cell_id.split("_")
        return int(r) * 3 + int(c)

    # ── Serialization ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Public view of the session for the API.

        ``round_age`` maps cell index → piece age (1 = oldest … 3 =
        newest, FIFO eviction order); cells without a piece are absent.
        """
        age_map: dict[str, int] = {}
        for entry in self.state["_arrays"].get("pieceOrder", []):
            cid = entry.get("cell_id", "")
            if cid:
                _, r, c = cid.split("_")
                age_map[int(r) * 3 + int(c)] = len(age_map) + 1
        return {
            "game_id": self.game_id,
            "player_color": self.player_color,
            "difficulty": self.difficulty,
            "board": self.board,
            "round_age": age_map,
            "turn": self.current_player,
            "winner": self.winner,
            "over": self.over,
            "last_ai_move": self.last_ai_move,
            "round": self.state["env"].get("round", 0),
        }


class PlayManager:
    """Registry of active game sessions (in-memory, single-process).

    Thread-safe: registry mutations are guarded by one lock; each
    session additionally owns a per-session lock so concurrent ``move``
    calls cannot interleave on the same game.  Finished sessions are
    reclaimed by the caller via :meth:`remove` (or FIFO eviction).
    """

    def __init__(self, provider: SolverProvider, seed: int = 42, max_sessions: int = 128) -> None:
        self._provider = provider
        self._sessions: dict[str, GameSession] = {}
        self._seed = seed
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def start(self, player_color: PlayerColor | str, difficulty: Difficulty) -> GameSession:
        """Create a new session; the AI opens first when the human is white."""
        if player_color == "random":
            player_color = "p_black" if uuid.uuid4().int % 2 == 0 else "p_white"
        if player_color not in ("p_black", "p_white"):
            raise PlayError(f"unknown player color: {player_color}")
        if difficulty not in DIFFICULTY_BUDGETS:
            raise PlayError(f"unknown difficulty: {difficulty}")

        game_id = uuid.uuid4().hex[:8]
        engine = engine_from_rules("moon_chess", self._seed)
        solver = self._provider.create_solver("moon_chess", "mcts", engine, self._seed, DIFFICULTY_BUDGETS[difficulty])

        session = GameSession(
            game_id=game_id,
            player_color=player_color,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            engine=engine,
            solver=solver,
        )
        if session.player_color == "p_white":
            session.ai_move()
        self._register(session)
        return session

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f"unknown game: {game_id}")
        return session

    def remove(self, game_id: str) -> None:
        with self._lock:
            self._sessions.pop(game_id, None)

    # ── Internals ──────────────────────────────────────────────────

    def _register(self, session: GameSession) -> None:
        """Register under the lock, evicting the oldest unfinished session (FIFO)."""
        with self._lock:
            while len(self._sessions) >= self._max_sessions:
                _, oldest = next(iter(self._sessions.items()))
                self._sessions.pop(oldest.game_id, None)
            self._sessions[session.game_id] = session
