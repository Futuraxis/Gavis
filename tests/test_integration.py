"""Integration tests — combine multiple layers.

These tests verify that the layers work together correctly when
composed, without creating circular dependencies.
"""

from __future__ import annotations

import pytest

from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer3_solvers import MCTS, MCTSConfig
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
        # MockBinding returns a board with one X and one O
        black_count = sum(1 for c in state["_board"] if c == "p_black")
        white_count = sum(1 for c in state["_board"] if c == "p_white")
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
        # The action should be legal on the interpreted board
        legal_keys = {a.canonical_key for a in adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys
