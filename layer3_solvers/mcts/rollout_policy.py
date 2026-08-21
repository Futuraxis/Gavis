"""Rollout policies for MCTS.

The MCTS core is generic over ``SolverAdapter``; game-specific rollout
smarts live here as opt-in policies instead of being hardcoded in the
search (M-04).  ``BoardHeuristicPolicy`` implements the win/block/threat
heuristics for square-board line-connect games.  Every method returns
``None`` / ``0.0`` for states that do not carry a square ``_arrays.board``,
so non-board games fall back to uniform-random rollouts without any
board-specific code ever running.
"""

from __future__ import annotations

import random
import warnings
from typing import Optional

from layer2_engine.interfaces.solver_adapter import ActionInstance, State


def root_player(state: State, adapter, default_player: Optional[str] = None) -> Optional[str]:
    """Rollout-start player id，解析失败时可用 default_player 兜底。"""
    env = state.get("env", {}) if isinstance(state, dict) else {}
    turn = env.get("turn")
    if isinstance(turn, dict):
        turn = turn.get("currentPlayerId", turn)
    if adapter is not None and adapter.get_node_type(state) == "chance":
        turn = env.get("lastActor", turn)
        if turn is None:
            warnings.warn("root_player: chance 节点缺少 env.lastActor，视角可能不正确")
    if turn is None and adapter is not None:
        turn = adapter.get_current_player(state)
    if turn is None:
        turn = default_player
    return turn


class BoardHeuristicPolicy:
    """Win/block/threat heuristics for square-board line-connect games.

    - ``choose``: play an immediate win, else block the opponent's
      immediate win with probability 0.7 (residual randomness keeps the
      search exploratory), else ``None`` (caller falls back to random).
    - ``leaf_value``: signed threat gap (clamped to ±0.5) for
      depth-limited non-terminal rollouts — terminal utilities (±1) keep
      dominance, while the gap captures defensive value that pure-random
      playouts systematically miss.

    All lookups use ``state['_arrays']``/``_constants`` — the policy is
    the only MCTS component allowed to know about board internals.

    注意：当前按格子编号匹配动作，适用于“一格一动作”的游戏；
    同格多动作的游戏需要再细化匹配。
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        block_prob: float = 0.7,
        max_actions: int = 16,
    ) -> None:
        self._rng = rng if rng is not None else random.Random(0)
        self._block_prob = block_prob
        self._max_actions = max_actions

    def choose(self, state: State, actions: list) -> Optional[ActionInstance]:
        """Heuristic move for square-board line-connect games, else None.

        Skipped for large action sets (the scan cost is proportional to
        the action count anyway).
        """
        if len(actions) > self._max_actions:
            return None

        board = state.get("_arrays", {}).get("board", [])
        bs = int(len(board) ** 0.5) if board else 0
        if bs == 0 or bs * bs != len(board):
            return None
        raw_len = state.get("_constants", {}).get("win_length")
        if not raw_len:
            warnings.warn("rollout_policy: 规则缺少 win_length，跳过棋感")
            return None
        win_len = int(raw_len)
        turn = state.get("env", {}).get("turn")
        if not turn:
            return None
        opponent = self._opponent_of(state, turn)
        if opponent is None:
            return None

        my_win = self._scan_line_cells(board, bs, win_len, turn)
        if my_win:
            wins = [a for a in actions if self._action_index(a) in my_win]
            if wins:
                return self._rng.choice(wins)

        blocks = self._scan_line_cells(board, bs, win_len, opponent)
        if blocks and self._rng.random() < self._block_prob:
            block_actions = [a for a in actions if self._action_index(a) in blocks]
            if block_actions:
                return self._rng.choice(block_actions)

        return None

    def leaf_value(self, sim_state: State, player: str) -> float:
        """Threat gap in [-0.5, 0.5] from ``player``'s view; 0.0 off-board.

        Skipped on large boards, where the full line scan is expensive
        and the signal is weak anyway.
        """
        board = sim_state.get("_arrays", {}).get("board", [])
        if not board or len(board) > 25:
            return 0.0
        bs = int(len(board) ** 0.5)
        if bs * bs != len(board):
            return 0.0
        raw_len = sim_state.get("_constants", {}).get("win_length")
        if not raw_len:
            return 0.0
        win_len = int(raw_len)
        players = sim_state.get("_players", [])
        ids = [p["id"] for p in players] if players and isinstance(players[0], dict) else players
        opponent = next((x for x in ids if x != player), None)
        if opponent is None:
            return 0.0

        mine = opp = 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for r in range(bs):
                for c in range(bs):
                    line = [(r + k * dr, c + k * dc) for k in range(win_len)]
                    if any(not (0 <= rr < bs and 0 <= cc < bs) for rr, cc in line):
                        continue
                    vals = [board[rr * bs + cc] for rr, cc in line]
                    if vals.count(None) != 1:
                        continue
                    if vals.count(player) == win_len - 1:
                        mine += 1
                    elif vals.count(opponent) == win_len - 1:
                        opp += 1
        gap = (mine - opp) / max(1, mine + opp) if mine + opp else 0.0
        return max(-1.0, min(1.0, gap)) * 0.5

    @staticmethod
    def _scan_line_cells(board: list, bs: int, win_len: int, player: str) -> set[int]:
        """Cells that complete an ``win_len - 1`` run for ``player``.

        Scans every straight line of length ``win_len`` (4 directions)
        and returns the empty cell of each line that is one piece short
        of a win — the immediate-winning (or threat-blocking) moves.
        """
        cells: set[int] = set()
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for r in range(bs):
                for c in range(bs):
                    line = [(r + k * dr, c + k * dc) for k in range(win_len)]
                    if any(not (0 <= rr < bs and 0 <= cc < bs) for rr, cc in line):
                        continue
                    vals = [board[rr * bs + cc] for rr, cc in line]
                    if vals.count(player) != win_len - 1 or vals.count(None) != 1:
                        continue
                    er, ec = line[vals.index(None)]
                    cells.add(er * bs + ec)
        return cells

    @staticmethod
    def _action_index(action: ActionInstance) -> int:
        """Board index of an action's primary cell parameter.

        Uses the entity's ``_index`` field (produced by view
        materialization) — no cell-id format assumptions.
        """
        cell = action.params.get("cell", {})
        if isinstance(cell, dict) and "_index" in cell:
            return int(cell["_index"])
        return -1

    @staticmethod
    def _opponent_of(state: dict, player: str) -> Optional[str]:
        players = state.get("_players", [])
        ids = [p["id"] for p in players] if players and isinstance(players[0], dict) else players
        for pid in ids:
            if pid != player:
                return pid
        return None
