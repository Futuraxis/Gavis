"""Integration tests — combine multiple layers.

These tests verify that the layers work together correctly when
composed, without creating circular dependencies.
"""

from __future__ import annotations

import pytest

from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer2_engine.games.texas_holdem.texas_env_adapter import TexasHoldemAdapter
from layer3_solvers import CFR, CFRConfig, MCTS, MCTSConfig
from layer4_interface.vision_bridge import observation_to_state
from layer4_interface.binding import Observation, MockBinding


class TestBindingEngineSolverIntegration:
    """Full pipeline: Binding → Engine → Solver, no circular deps."""

    @pytest.fixture
    def adapter(self) -> MoonChessAdapter:
        return MoonChessAdapter(seed=42)

    def test_binding_to_state(self, adapter: MoonChessAdapter):
        """Layer 4 → Layer 2: MockBinding → vision_bridge → Engine state."""
        binding = MockBinding()
        obs = binding.parse_image("")
        state = observation_to_state(obs, adapter)
        board = state["_arrays"]["board"]
        black_count = sum(1 for c in board if c == "p_black")
        white_count = sum(1 for c in board if c == "p_white")
        assert black_count == 1
        assert white_count == 1

    def test_state_to_solver(self, adapter: MoonChessAdapter):
        """Layer 2 → Layer 3: Engine state → Solver decision."""
        solver = MCTS(adapter, MCTSConfig(seed=42, budget=200))
        state = adapter.create_initial_state()
        action = solver.select_action(state)
        assert action is not None
        legal_keys = {a.canonical_key for a in adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys

    def test_full_pipeline(self, adapter: MoonChessAdapter):
        """Full pipeline: Binding → Engine → Solver → action."""
        solver = MCTS(adapter, MCTSConfig(seed=42, budget=200))
        binding = MockBinding()

        obs = binding.parse_image("")
        state = observation_to_state(obs, adapter)
        action = solver.select_action(state)

        assert action is not None
        legal_keys = {a.canonical_key for a in adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys


class TestTexasEngineSolverIntegration:
    """Layer 2 (poker engine) + Layer 3 (MCTS / CFR), no circular deps."""

    @pytest.fixture
    def adapter(self) -> TexasHoldemAdapter:
        return TexasHoldemAdapter(seed=42)

    def _resolve(self, adapter: TexasHoldemAdapter, state: dict) -> dict:
        while adapter.get_node_type(state) == 'chance':
            _, state = adapter.sample_chance(state)
        return state

    def test_mcts_plays_full_hand(self, adapter: TexasHoldemAdapter):
        """MCTS drives a full hand to a terminal, zero-sum state."""
        solver = MCTS(adapter, MCTSConfig(seed=42, budget=300))
        state = self._resolve(adapter, adapter.create_initial_state())
        guard = 0
        while not adapter.is_terminal(state) and guard < 60:
            if adapter.get_node_type(state) == 'player':
                action = solver.select_action(state)
                assert action is not None
                state = adapter.apply_action(state, action)
            state = adapter.resolve_chance(state)
            guard += 1
        assert adapter.is_terminal(state)
        u_sb = adapter.get_utility(state, 'p_sb')
        u_bb = adapter.get_utility(state, 'p_bb')
        assert u_sb + u_bb == 0.0

    def test_cfr_info_sets_on_poker(self, adapter: TexasHoldemAdapter):
        """CFR's info-set machinery works with imperfect information."""
        solver = CFR(adapter, CFRConfig(seed=42, iterations=10, depth_limit=10))
        state = self._resolve(adapter, adapter.create_initial_state())
        strategy = solver.solve(state, verbose=False)
        assert strategy
        assert sum(strategy.values()) > 0
        assert len(solver.info_sets) > 0
