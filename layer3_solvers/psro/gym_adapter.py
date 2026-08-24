"""GymAdapter — wraps a GameEngine into a Gym-style interface.

PSRO's core algorithms (tabular Q, gamescape) expect a Gym-like
environment with ``reset()`` / ``step(action)`` / ``available_actions()``.
This engine bridges the two worlds.

The wrapper is hard-wired to 3×3 moon-chess-like games (Discrete(9) action
space, ``cell_r_c`` canonical keys, base-3 board encoding); constructing it
for another game shape raises ``ValueError`` instead of silently producing
meaningless results (审查 P2-22).
"""

from __future__ import annotations

import logging

import numpy as np
from gymnasium import spaces

from layer2_engine.core.state_graph import ActionInstance, State
from layer2_engine.core.engine import GameEngine

logger = logging.getLogger(__name__)

_BOARD_SIZE = 3


class GymAdapter:
    """Wrap a ``GameEngine`` into a Gym-style environment.

    Parameters
    ----------
    engine : GameEngine
    state_dim : int
        Size of the encoded observation space (default 19683 for 3×3).
    seed : int, optional
        Seed for chance sampling (审查 P2-23); None keeps the old global
        ``np.random`` behavior.
    """

    def __init__(self, engine: GameEngine, state_dim: int = 19683, seed: int | None = None):
        self._adapter = engine
        self._state: State | None = None
        self._rng = np.random.RandomState(seed)
        raw_players = getattr(engine, "rules", {}).get("players", [])

        if len(raw_players) >= 2:
            first_player = raw_players[0]
            second_player = raw_players[1]

            self._p1 = str(first_player.get("id", "p_black")) if isinstance(first_player, dict) else str(first_player)
            self._p2 = (
                str(second_player.get("id", "p_white")) if isinstance(second_player, dict) else str(second_player)
            )
        else:
            self._p1 = "p_black"
            self._p2 = "p_white"

        self._turn = 0
        self._warned_invalid_action = False

        # 审查 P2-22: 只支持 3×3 月亮棋形状 — 校验 board 尺寸与动作模板
        # 可解析性，失败即抛错（此前非 3×3 游戏静默降级成无意义结果）。
        self._validate_board(engine)

        self.observation_space = spaces.Discrete(state_dim)
        self.action_space = spaces.Discrete(9)
        self.n_actions = 9

    def _validate_board(self, engine: GameEngine) -> None:
        """Reject games whose board is not 3×3 (9 cells).

        The encoding (base-3 over 9 cells) and the int↔action mapping
        (``cell_r_c`` → ``r*3+c``) are hard-wired to that shape; a
        different board silently produced garbage policies before.
        Gated on the rules declaring a ``board`` array so test fakes
        without a groundState still construct.
        """
        rules = getattr(engine, "rules", {})
        ground = rules.get("groundState", {})
        board_def = ground.get("board")
        if not board_def:
            return
        length = board_def.get("length")
        if isinstance(length, dict):
            length = length.get("expr")  # e.g. board_size * board_size
        if isinstance(length, str) and "board_size" in length:
            size = rules.get("constants", {}).get("board_size")
            if size is not None:
                length = int(size) * int(size)
        if length is not None and int(length) != 9:
            raise ValueError(f"GymAdapter only supports 3×3 moon-chess-shaped games, got board length {length}")

    # ── Gym interface ─────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        self._state = self._adapter.create_initial_state()
        self._turn = 0
        return self._encode_state(self._state), {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Step the environment.

        Returns: (obs, reward, terminated, truncated, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._turn += 1

        # Convert int action to ActionInstance
        legal = self._adapter.get_legal_actions(self._state)
        action_instance = self._int_to_action(action, legal)

        if action_instance is None:
            # Fallback: take first legal action（非法/未映射动作 → 显式告警，
            # 不再静默替换；正常路径下 mask 已保证动作合法）
            if not self._warned_invalid_action:
                self._warned_invalid_action = True
                logger.warning("GymAdapter.step: unmapped action %s; falling back to legal[0]", action)
            action_instance = legal[0] if legal else None
            if action_instance is None:
                return self._encode_state(self._state), -1.0, True, False, {}

        # Apply action
        self._state = self._adapter.apply_action(self._state, action_instance)

        # Handle chance nodes (seeded local rng, 审查 P2-23)
        while self._adapter.get_node_type(self._state) == "chance":
            outcomes = self._adapter.get_chance_outcomes(self._state)
            if not outcomes:
                break
            # Uniform sampling (assumes outcomes have probability field)
            r = self._rng.random()
            cumsum = 0.0
            chosen = outcomes[-1]
            for o in outcomes:
                cumsum += o.probability
                if r < cumsum:
                    chosen = o
                    break
            self._state = self._adapter.apply_chance(self._state, chosen)

        done = self._adapter.is_terminal(self._state)
        # Meta-game payoffs are always measured from the row player's perspective.
        reward = self._adapter.get_utility(self._state, self._p1)

        return self._encode_state(self._state), reward, done, False, {}

    def available_actions(self, state: State | None = None) -> np.ndarray:
        """Return boolean mask of legal actions.

        Parameters
        ----------
        state : State, optional
            Compute the mask from this state instead of the internal
            ``_state`` (审查 P1-2: ``select_action`` must mask the state
            it was given, not whatever the gym last stepped on).
        """
        target = state if state is not None else self._state
        if target is None:
            return np.ones(9, dtype=bool)
        legal = self._adapter.get_legal_actions(target)
        mask = np.zeros(9, dtype=bool)
        for a in legal:
            idx = self._action_to_int(a)
            if idx is not None:
                mask[idx] = True
        return mask

    def clone(self) -> "GymAdapter":
        """A copy with its own ``_state`` for parallel match-up evaluation.

        Clones share the wrapped engine (engine) — its methods only
        mutate the passed-in state, so concurrent episodes over separate
        states are safe (audit 3.6: PSRO 评估并行化).
        """
        return GymAdapter(self._adapter, state_dim=int(self.observation_space.n), seed=None)

    def get_current_player(self) -> str | None:
        """Delegated current-player query (who acts at the current state).

        Used by ``estimate_reward`` to route turns instead of assuming
        strict alternation (C-10: chance nodes and multi-action turns
        break the ``steps % 2`` assumption).
        """
        if self._state is None:
            return self._p1
        return self._adapter.get_current_player(self._state)

    @property
    def players(self) -> tuple[str, str]:
        """The two player ids in seat order."""
        return self._p1, self._p2

    # ── Internal ──────────────────────────────────────────────────

    def _encode_state(self, state: State) -> int:
        """Encode board as 3-base integer (matching original PSRO encoding).

        Note: the encoding captures only the board occupancy — pieceOrder
        (FIFO age) and the round counter are dropped.  Acceptable under
        the shared-policy symmetry of PSRO's pool, but states that differ
        only in age/round collapse to the same code.
        """
        board = state.get("_board")
        if board is None:
            board = state.get("_arrays", {}).get("board", [])
        code = 0
        for i, val in enumerate(board):
            if val == "p_black":
                digit = 1
            elif val == "p_white":
                digit = 2
            else:
                digit = 0
            code += digit * (3**i)
        return code

    def _int_to_action(self, action_idx: int, legal: list) -> ActionInstance | None:
        """Map 0-8 int to the corresponding ActionInstance."""
        for a in legal:
            cell = a.params.get("cell", {})
            cell_id = cell.get("id", "") if isinstance(cell, dict) else str(cell)
            try:
                _, r, c = cell_id.split("_")
                idx = int(r) * 3 + int(c)
                if idx == action_idx:
                    return a
            except (ValueError, IndexError):
                pass
        return None

    @staticmethod
    def _action_to_int(action) -> int | None:
        """Convert ActionInstance back to 0-8 int."""
        cell = action.params.get("cell", {})
        cell_id = cell.get("id", "") if isinstance(cell, dict) else str(cell)
        try:
            _, r, c = cell_id.split("_")
            return int(r) * 3 + int(c)
        except (ValueError, IndexError):
            return None
