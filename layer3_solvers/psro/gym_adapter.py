"""GymAdapter — wraps a SolverAdapter into a Gym-style interface.

PSRO's core algorithms (tabular Q, gamescape) expect a Gym-like
environment with ``reset()`` / ``step(action)`` / ``available_actions()``.
This adapter bridges the two worlds.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from layer2_engine.interfaces.solver_adapter import SolverAdapter, State


class GymAdapter:
    """Wrap a ``SolverAdapter`` into a Gym-style environment.

    Parameters
    ----------
    adapter : SolverAdapter
    state_dim : int
        Size of the encoded observation space (default 19683 for 3×3).
    """

    def __init__(self, adapter: SolverAdapter, state_dim: int = 19683):
        self._adapter = adapter
        self._state: State | None = None
        raw_players = getattr(adapter, "rules", {}).get("players", [])

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

        self.observation_space = spaces.Discrete(state_dim)
        self.action_space = spaces.Discrete(9)
        self.n_actions = 9

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
            # Fallback: take first legal action
            action_instance = legal[0] if legal else None
            if action_instance is None:
                return self._encode_state(self._state), -1.0, True, False, {}

        # Apply action
        self._state = self._adapter.apply_action(self._state, action_instance)

        # Handle chance nodes
        while self._adapter.get_node_type(self._state) == "chance":
            outcomes = self._adapter.get_chance_outcomes(self._state)
            if not outcomes:
                break
            # Uniform sampling (assumes outcomes have probability field)
            r = np.random.random()
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

    def available_actions(self) -> np.ndarray:
        """Return boolean mask of legal actions."""
        if self._state is None:
            return np.ones(9, dtype=bool)
        legal = self._adapter.get_legal_actions(self._state)
        mask = np.zeros(9, dtype=bool)
        for a in legal:
            idx = self._action_to_int(a)
            if idx is not None:
                mask[idx] = True
        return mask

    def clone(self) -> "GymAdapter":
        """A copy with its own ``_state`` for parallel match-up evaluation.

        Clones share the wrapped adapter (engine) — its methods only
        mutate the passed-in state, so concurrent episodes over separate
        states are safe (audit 3.6: PSRO 评估并行化).
        """
        return GymAdapter(self._adapter, state_dim=int(self.observation_space.n))

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
        """Encode board as 3-base integer (matching original PSRO encoding)."""
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

    def _int_to_action(self, action_idx: int, legal: list) -> any:
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
