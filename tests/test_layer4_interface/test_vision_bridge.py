"""Tests for VisionBridge — Layer 4 → Layer 2 translation only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer4_interface.binding import Observation
from layer4_interface.vision_bridge import observation_to_state

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


@pytest.fixture
def engine() -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=42)


class TestObservationToState:
    def test_empty_board(self, engine: GameEngine):
        obs = Observation(
            boardObservation=[[None] * 3 for _ in range(3)],
            confidence=[[0.0] * 3 for _ in range(3)],
        )
        state = observation_to_state(obs, engine)
        board = state["_arrays"]["board"]
        assert len(board) == 9
        assert all(c is None for c in board)

    def test_with_pieces(self, engine: GameEngine):
        obs = Observation(
            boardObservation=[["X", None, None], [None, "O", None], [None, None, None]],
            confidence=[[0.9, 0.0, 0.0], [0.0, 0.85, 0.0], [0.0, 0.0, 0.0]],
        )
        state = observation_to_state(obs, engine)
        board = state["_arrays"]["board"]
        assert board[0] == "p_black"
        assert board[4] == "p_white"

    def test_various_symbols(self, engine: GameEngine):
        """Various unicode symbols for X and O should all map correctly."""
        from itertools import product

        symbols_x = ["X", "x", "●"]
        symbols_o = ["O", "o", "○"]
        for sx, so in product(symbols_x, symbols_o):
            obs = Observation(
                boardObservation=[[sx, None], [None, so]],
                confidence=[[0.9, 0.0], [0.0, 0.85]],
            )
            state = observation_to_state(obs, engine)
            assert state["_arrays"]["board"][0] == "p_black", f"{sx} → p_black failed"
            assert state["_arrays"]["board"][3] == "p_white", f"{so} → p_white failed"

    def test_load_state_preserves_env(self, engine: GameEngine):
        obs = Observation(
            boardObservation=[[None] * 3 for _ in range(3)],
            confidence=[[0.0] * 3 for _ in range(3)],
        )
        state = observation_to_state(obs, engine)
        assert state["env"]["phase"] == "playing"
        assert engine.get_current_player(state) == "p_black"

    def test_legal_actions_after_load(self, engine: GameEngine):
        obs = Observation(
            boardObservation=[["X", None, None], [None, None, None], [None, None, None]],
            confidence=[[0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        state = observation_to_state(obs, engine)
        actions = engine.get_legal_actions(state)
        assert len(actions) == 8  # one cell occupied
        for a in actions:
            cell = a.params.get("cell", {})
            cid = cell.get("id", "") if isinstance(cell, dict) else ""
            assert cid != "cell_0_0", f"Occupied cell should not be legal: {a.canonical_key}"
