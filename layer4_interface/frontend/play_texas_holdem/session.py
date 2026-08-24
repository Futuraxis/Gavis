"""Game session management for the Texas Hold'em play app.

Each session owns a bare ``GameEngine`` (v5.2 — built from
``rules/texas_holdem.json``, no per-game adapter) and a solver handle
running opponent-model search over sampled worlds (the engine's generic
``eval_expr``-driven hybrid PIMC — imperfect-information play without
leaking the opponent's hole cards into the search).  Chance nodes
(hole/community dealing, showdown) are resolved automatically; the human
and the AI only ever see player nodes.

Seat constants and hand/payoff lookups come from the rules JSON via the
frontend ``engine_helpers`` module (``TEXAS_SEATS``, ``texas_hand_name``)
and the generic engine protocol (``get_utility``) — no per-game adapter
exists.  Solvers are injected through a ``SolverProvider``, so Layer 4
holds no ``layer3_solvers`` import.
"""

from __future__ import annotations

import random
import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal

from layer2_engine.core.engine import GameEngine

from ...solver_provider import SolverHandle, SolverProvider
from ..engine_helpers import TEXAS_SEATS, engine_from_rules, resolve_all_chance, texas_hand_name

Seat = Literal["p_sb", "p_bb"]
Difficulty = Literal["easy", "normal", "hard"]

# Hybrid opponent-model search budgets per difficulty tier.  Measured on
# the poker engine: ~1 ms per world simulation → ~0.15 s / ~0.5 s / ~1.5 s.
DIFFICULTY_BUDGETS: dict[Difficulty, int] = {
    "easy": 150,
    "normal": 500,
    "hard": 1200,
}


class PlayError(Exception):
    """Bad play request (unknown game, illegal move, game over, ...)."""


@dataclass
class GameSession:
    """One human-vs-AI Texas Hold'em hand."""

    game_id: str
    player_pid: Seat
    difficulty: Difficulty
    engine: GameEngine
    solver: SolverHandle
    state: dict = field(init=False)
    last_ai_action: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    @property
    def ai_pid(self) -> Seat:
        return TEXAS_SEATS[1] if self.player_pid == TEXAS_SEATS[0] else TEXAS_SEATS[0]

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

    def _env(self, pid: str, field: str) -> object:
        return self.state["env"].get(f"{pid[2:]}_{field}")

    def human_move(self, choice: str, amount: int | None = None) -> str:
        """Apply the human's action, then let the AI reply."""
        if self.over:
            raise PlayError("本局已结束")
        if self.current_player != self.player_pid:
            raise PlayError("还没轮到你")
        action = self._find_action(choice, amount)
        if action is None:
            raise PlayError(f"非法动作: {choice} {amount}")
        self.state = self.engine.apply_action(self.state, action)
        self.state = resolve_all_chance(self.engine, self.state)
        self._ai_turn()
        return action.canonical_key

    def _ai_turn(self) -> None:
        """Let the AI act while it's its turn (including after start)."""
        while not self.over and self.current_player == self.ai_pid:
            action = self.solver.select_action(self.state)
            if action is None:  # search found nothing — random fallback
                actions = self.engine.get_legal_actions(self.state)
                action = random.choice(actions) if actions else None
            if action is None:
                break
            self.last_ai_action = action.canonical_key
            self.state = self.engine.apply_action(self.state, action)
            self.state = resolve_all_chance(self.engine, self.state)

    def _find_action(self, choice: str, amount: int | None) -> object | None:
        for action in self.engine.get_legal_actions(self.state):
            if action.params.get("choice") != choice:
                continue
            if amount is None or action.params.get("amount") == amount:
                return action
        return None

    # ── Serialization ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        env = self.state["env"]
        arrs = self.state["_arrays"]
        over = self.over
        revealed = over and env.get("last_action") == "showdown"
        legal = []
        if not over and self.current_player == self.player_pid:
            for action in self.engine.get_legal_actions(self.state):
                legal.append({"choice": action.params["choice"], "amount": action.params["amount"]})
        raise_amts = sorted({a["amount"] for a in legal if a["choice"] == "raise"})

        def _cards(pid: str) -> list:
            return list(arrs.get(f"{pid[2:]}_hole", []))

        def _hand_name(pid: str) -> str | None:
            cards = [*_cards(pid), *arrs.get("community", [])]
            return texas_hand_name(self.engine, cards) if over else None

        return {
            "game_id": self.game_id,
            "player_pid": self.player_pid,
            "ai_pid": self.ai_pid,
            "difficulty": self.difficulty,
            "over": over,
            "winner": self.winner,
            "turn": self.current_player,
            "phase": env.get("phase"),
            "street": env.get("street", 0),
            "street_name": ("翻前", "翻牌", "转牌", "河牌")[env.get("street", 0)],
            "pot": int(env.get("sb_committed", 0) + env.get("bb_committed", 0)),
            "community": list(arrs.get("community", [])),
            "my_hole": _cards(self.player_pid),
            "ai_hole": _cards(self.ai_pid) if revealed else [],
            "revealed": revealed,
            "my_stack": int(env.get(f"{self.player_pid[2:]}_stack", 0)),
            "ai_stack": int(env.get(f"{self.ai_pid[2:]}_stack", 0)),
            "my_committed": int(env.get(f"{self.player_pid[2:]}_committed", 0)),
            "ai_committed": int(env.get(f"{self.ai_pid[2:]}_committed", 0)),
            "my_folded": bool(env.get(f"{self.player_pid[2:]}_folded")),
            "ai_folded": bool(env.get(f"{self.ai_pid[2:]}_folded")),
            "last_actor": env.get("last_actor"),
            "last_action": env.get("last_action"),
            "last_ai_action": self.last_ai_action,
            "call_to": int(env.get("last_call_to", 0)),
            "my_hand_name": _hand_name(self.player_pid),
            "ai_hand_name": _hand_name(self.ai_pid),
            "payoff": self.engine.get_utility(self.state, self.player_pid) if over else None,
            "legal": legal,
            "raise_amounts": raise_amts,
        }


class PlayManager:
    """Registry of active game sessions (in-memory, single-process).

    Thread-safe: registry mutations are guarded by one lock; each
    session owns a per-session lock so concurrent ``move`` calls cannot
    interleave on the same game.  Finished sessions are reclaimed via
    :meth:`remove` (or FIFO eviction).
    """

    def __init__(self, provider: SolverProvider, seed: int = 42, max_sessions: int = 128) -> None:
        self._provider = provider
        self._sessions: dict[str, GameSession] = {}
        self._seed = seed
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def start(self, player_color: str, difficulty: str) -> GameSession:
        if player_color == "random":
            player_color = TEXAS_SEATS[0] if uuid.uuid4().int % 2 == 0 else TEXAS_SEATS[1]
        if player_color not in TEXAS_SEATS:
            raise PlayError(f"unknown seat: {player_color}")
        if difficulty not in DIFFICULTY_BUDGETS:
            raise PlayError(f"unknown difficulty: {difficulty}")

        game_id = uuid.uuid4().hex[:8]
        engine = engine_from_rules("texas_holdem", self._seed)
        solver = self._provider.create_solver(
            "texas_holdem",
            "hybrid",
            engine,
            self._seed,
            DIFFICULTY_BUDGETS[difficulty],
        )

        session = GameSession(
            game_id=game_id,
            player_pid=player_color,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            engine=engine,
            solver=solver,
        )
        # Blinds/deals are chance nodes — resolve, then let the AI open
        # if the human sits in the big blind (SB acts first preflop).
        session.state = resolve_all_chance(engine, session.state)
        session._ai_turn()
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
