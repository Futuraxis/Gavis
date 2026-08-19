"""Tests for VisionBridge — Layer 4 → Layer 2 translation only."""

from __future__ import annotations

import pytest

from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer4_interface.binding import Observation
from layer4_interface.vision_bridge import observation_to_state


@pytest.fixture
def adapter() -> MoonChessAdapter:
    return MoonChessAdapter(seed=42)


class TestObservationToState:
    def test_empty_board(self, adapter: MoonChessAdapter):
        obs = Observation(
            boardObservation=[[None] * 3 for _ in range(3)],
            confidence=[[0.0] * 3 for _ in range(3)],
        )
        state = observation_to_state(obs, adapter)
        board = state["_arrays"]["board"]
        assert len(board) == 9
        assert all(c is None for c in board)

    def test_with_pieces(self, adapter: MoonChessAdapter):
        obs = Observation(
            boardObservation=[["X", None, None], [None, "O", None], [None, None, None]],
            confidence=[[0.9, 0.0, 0.0], [0.0, 0.85, 0.0], [0.0, 0.0, 0.0]],
        )
        state = observation_to_state(obs, adapter)
        board = state["_arrays"]["board"]
        assert board[0] == "p_black"
        assert board[4] == "p_white"

    def test_various_symbols(self, adapter: MoonChessAdapter):
        """Various unicode symbols for X and O should all map correctly."""
        from itertools import product

        symbols_x = ["X", "x", "●"]
        symbols_o = ["O", "o", "○"]
        for sx, so in product(symbols_x, symbols_o):
            obs = Observation(
                boardObservation=[[sx, None], [None, so]],
                confidence=[[0.9, 0.0], [0.0, 0.85]],
            )
            state = observation_to_state(obs, adapter)
            assert state["_arrays"]["board"][0] == "p_black", f"{sx} → p_black failed"
            assert state["_arrays"]["board"][3] == "p_white", f"{so} → p_white failed"

    def test_load_state_preserves_env(self, adapter: MoonChessAdapter):
        obs = Observation(
            boardObservation=[[None] * 3 for _ in range(3)],
            confidence=[[0.0] * 3 for _ in range(3)],
        )
        state = observation_to_state(obs, adapter)
        assert state["env"]["phase"] == "playing"
        assert adapter.get_current_player(state) == "p_black"

    def test_legal_actions_after_load(self, adapter: MoonChessAdapter):
        obs = Observation(
            boardObservation=[["X", None, None], [None, None, None], [None, None, None]],
            confidence=[[0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        state = observation_to_state(obs, adapter)
        actions = adapter.get_legal_actions(state)
        assert len(actions) == 8  # one cell occupied
        for a in actions:
            cell = a.params.get("cell", {})
            cid = cell.get("id", "") if isinstance(cell, dict) else ""
            assert cid != "cell_0_0", f"Occupied cell should not be legal: {a.canonical_key}"
