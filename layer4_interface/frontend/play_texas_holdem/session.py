"""Game session management for the Texas Hold'em play app.

Each session owns a ``TexasHoldemAdapter`` (v5.0 rules) and a
``HybridSolver`` running opponent-model search over sampled worlds
(``sample_hidden`` + uniform opponent model — imperfect-information
play without leaking the opponent's hole cards into the search).
Chance nodes (hole/community dealing, showdown) are resolved
automatically; the human and the AI only ever see player nodes.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from layer2_engine.core.poker_utils import PLAYER_BB, PLAYER_SB, poker_hand_name, poker_payoff

from layer2_engine.games.texas_holdem import TexasHoldemAdapter
from layer3_solvers import HybridConfig, HybridSolver

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
    engine: TexasHoldemAdapter
    solver: HybridSolver
    state: dict = field(init=False)
    last_ai_action: Optional[str] = None

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    @property
    def ai_pid(self) -> Seat:
        return PLAYER_BB if self.player_pid == PLAYER_SB else PLAYER_SB

    @property
    def winner(self) -> Optional[str]:
        return self.state["env"].get("winner")

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def current_player(self) -> Optional[str]:
        return self.engine.get_current_player(self.state)

    # ── Play actions ────────────────────────────────────────────────

    def _env(self, pid: str, field: str):
        return self.state["env"].get(f"{pid[2:]}_{field}")

    def human_move(self, choice: str, amount: Optional[int] = None) -> str:
        """Apply the human's action, then let the AI reply."""
        if self.over:
            raise PlayError("本局已结束")
        if self.current_player != self.player_pid:
            raise PlayError("还没轮到你")
        action = self._find_action(choice, amount)
        if action is None:
            raise PlayError(f"非法动作: {choice} {amount}")
        self.state = self.engine.apply_action(self.state, action)
        self.state = self.engine.resolve_chance(self.state)
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
            self.state = self.engine.resolve_chance(self.state)

    def _find_action(self, choice: str, amount: Optional[int]) -> Optional[object]:
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

        def _hand_name(pid: str) -> Optional[str]:
            cards = [*_cards(pid), *arrs.get("community", [])]
            return poker_hand_name(cards) if over else None

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
            "payoff": poker_payoff(self.state, self.player_pid) if over else None,
            "legal": legal,
            "raise_amounts": raise_amts,
        }


class PlayManager:
    """Registry of active game sessions (in-memory, single-process)."""

    def __init__(self, seed: int = 42) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._seed = seed

    def start(self, player_color: str, difficulty: str) -> GameSession:
        if player_color == "random":
            player_color = PLAYER_SB if uuid.uuid4().int % 2 == 0 else PLAYER_BB
        if player_color not in (PLAYER_SB, PLAYER_BB):
            raise PlayError(f"unknown seat: {player_color}")
        if difficulty not in DIFFICULTY_BUDGETS:
            raise PlayError(f"unknown difficulty: {difficulty}")

        game_id = uuid.uuid4().hex[:8]
        engine = TexasHoldemAdapter(seed=self._seed)
        solver = HybridSolver(
            engine,
            HybridConfig(
                seed=self._seed,
                mode="search",
                imperfect_information=True,
                mcts_budget=DIFFICULTY_BUDGETS[difficulty],
                opponent_model="uniform",
            ),
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
        session.state = engine.resolve_chance(session.state)
        session._ai_turn()
        self._sessions[game_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f"unknown game: {game_id}")
        return session

    def remove(self, game_id: str) -> None:
        self._sessions.pop(game_id, None)
