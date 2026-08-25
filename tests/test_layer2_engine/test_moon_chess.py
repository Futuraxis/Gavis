"""Tests for the bare Moon Chess engine (Layer 2, v5.2 — no per-game adapter)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


@pytest.fixture
def engine() -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=42)


class TestMoonChessBasics:
    def test_create_initial_state(self, engine: GameEngine):
        state = engine.create_initial_state()
        board = state["_arrays"]["board"]
        assert len(board) == 9  # 3×3
        assert all(c is None for c in board)
        # pieceOrder is a mutable array in ground state
        assert state["_arrays"]["pieceOrder"] == []

    def test_get_node_type(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert engine.get_node_type(state) == "player"

    def test_get_current_player(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert engine.get_current_player(state) == "p_black"

    def test_get_legal_actions(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        assert len(actions) == 9  # all 9 cells empty

    def test_get_chance_outcomes_empty(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert engine.get_chance_outcomes(state) == []

    def test_is_terminal_initial(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert not engine.is_terminal(state)


class TestMoonChessGameplay:
    def test_place_piece(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)

        # Place black at (0,0)
        new_state = engine.apply_action(state, actions[0])
        assert new_state["_arrays"]["board"][0] == "p_black"
        assert len(new_state["_arrays"]["pieceOrder"]) == 1
        assert new_state["_arrays"]["pieceOrder"][0]["cell_id"] == "cell_0_0"

    def test_switch_turn(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        state = engine.apply_action(state, actions[0])
        assert engine.get_current_player(state) == "p_white"

    def test_fifo_eviction(self, engine: GameEngine):
        """After placing 4+ pieces for one player, the oldest should disappear."""
        state = engine.create_initial_state()

        black_count = 0
        max_black = 0
        while black_count < 4 and not engine.is_terminal(state):
            actions = engine.get_legal_actions(state)
            if not actions:
                break
            state = engine.apply_action(state, actions[0])
            if engine.get_current_player(state) == "p_black" or engine.is_terminal(state):
                black_count += 1
            board = state["_arrays"]["board"]
            black_pieces = [i for i, c in enumerate(board) if c == "p_black"]
            max_black = max(max_black, len(black_pieces))

        board = state["_arrays"]["board"]
        black_pieces = [i for i, c in enumerate(board) if c == "p_black"]
        assert len(black_pieces) <= 3
        assert max_black >= 3

    def test_fifo_eviction_clears_correct_cell(self, engine: GameEngine):
        """The evicted piece's own cell must be cleared (row-major index)."""
        state = engine.create_initial_state()
        # Black: 1, 3, 5 → 4th piece at 7 evicts black's oldest (cell_0_1, idx 1).
        placements = [
            ("p_black", "cell_0_1"),
            ("p_white", "cell_0_0"),
            ("p_black", "cell_1_0"),
            ("p_white", "cell_0_2"),
            ("p_black", "cell_1_2"),
            ("p_white", "cell_1_1"),
            ("p_black", "cell_2_1"),
        ]
        for player, cell_id in placements:
            actions = engine.get_legal_actions(state)
            action = next(a for a in actions if a.params.get("cell", {}).get("id", "") == cell_id)
            state = engine.apply_action(state, action)

        board = state["_arrays"]["board"]
        black_po = [e["cell_id"] for e in state["_arrays"]["pieceOrder"] if e["player_id"] == "p_black"]
        assert board[1] is None, "evicted cell_0_1 (idx 1) must be cleared"
        assert board[3] == "p_black", "cell_1_0 (idx 3) must not be collateral"
        assert board[7] == "p_black", "new piece at cell_2_1 (idx 7)"
        assert black_po == ["cell_1_0", "cell_1_2", "cell_2_1"]

    def test_three_in_row_win(self, engine: GameEngine):
        """Black places in top row (0,0), (0,1), (0,2) → should win."""
        state = engine.create_initial_state()

        placements = [
            ("p_black", "cell_0_0"),
            ("p_white", "cell_1_0"),
            ("p_black", "cell_0_1"),
            ("p_white", "cell_1_1"),
            ("p_black", "cell_0_2"),
        ]

        for player, cell_id in placements:
            actions = engine.get_legal_actions(state)
            action = None
            for a in actions:
                cell = a.params.get("cell", {})
                cid = cell.get("id", "") if isinstance(cell, dict) else ""
                if cid == cell_id:
                    action = a
                    break
            assert action is not None, f"Action for {cell_id} not found"

            if player == "p_black":
                assert engine.get_current_player(state) == "p_black"
            state = engine.apply_action(state, action)

        assert engine.is_terminal(state)
        assert state["env"]["winner"] == "p_black"

    def test_action_mask_semantics(self, engine: GameEngine):
        """RL mask semantics live in the obs views (v5.2) — legal cells = 1.

        The adapter's ``get_action_mask`` no longer exists; the derived
        ``cell`` view + legal-actions enumeration is the generic path.
        """
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        assert len(actions) == 9
        obs = engine.get_observation(state, "p_black")
        assert len(obs["cell"]) == 9  # 3×3 cell view (public info)

        state = engine.apply_action(state, actions[0])
        obs2 = engine.get_observation(state, "p_white")
        assert len(obs2["cell"]) == 9

    def test_feature_vector_semantics(self, engine: GameEngine):
        """38-dim feature layout is produced by L3 solvers from obs views.

        The adapter's ``get_feature_vector`` no longer exists; PPO/MARL
        consume the ``cell`` view (covered in test_layer3_solvers).
        """
        state = engine.create_initial_state()
        cells = engine.get_observation(state, "p_black")["cell"]
        assert len(cells) == 9
        assert all("x" in c and "y" in c for c in cells)


class TestGameEngineProtocol:
    """The bare GameEngine is the solver contract (no SolverAdapter class)."""

    def test_is_game_engine(self, engine: GameEngine):
        assert isinstance(engine, GameEngine)

    def test_all_protocol_methods_exist(self, engine: GameEngine):
        methods = [
            "create_initial_state",
            "get_node_type",
            "get_current_player",
            "get_legal_actions",
            "apply_action",
            "get_chance_outcomes",
            "apply_chance",
            "is_terminal",
            "get_utility",
            "get_observation",
            "get_info_set_key",
            "load_state",
            "project_observation",
        ]
        for m in methods:
            assert hasattr(engine, m), f"Missing method: {m}"

    def test_load_state_from_observation(self, engine: GameEngine):
        """Simulate what VisionBridge does (v5.0 format)."""
        board_obs = [
            [None, "X", None],
            [None, None, None],
            ["O", None, None],
        ]
        _board = []
        for row in board_obs:
            for cell in row:
                if cell is None:
                    _board.append(None)
                elif cell == "X":
                    _board.append("p_black")
                else:
                    _board.append("p_white")

        state = engine.load_state(
            {
                "_arrays": {"board": _board},
                "env": {
                    "phase": "playing",
                    "turn": "p_black",
                    "winner": None,
                },
            }
        )
        assert state["_arrays"]["board"][1] == "p_black"
        assert state["_arrays"]["board"][6] == "p_white"
        assert engine.get_current_player(state) == "p_black"
