"""Tests for MoonChessAdapter (Layer 2, v5.0)."""

from __future__ import annotations

import pytest
import numpy as np

from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer2_engine.interfaces.solver_adapter import SolverAdapter, ActionInstance


@pytest.fixture
def adapter() -> MoonChessAdapter:
    return MoonChessAdapter(seed=42)


class TestMoonChessBasics:
    def test_create_initial_state(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        board = state["_arrays"]["board"]
        assert len(board) == 9  # 3×3
        assert all(c is None for c in board)
        # pieceOrder is a mutable array in ground state
        assert state["_arrays"]["pieceOrder"] == []

    def test_get_node_type(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        assert adapter.get_node_type(state) == "player"

    def test_get_current_player(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        assert adapter.get_current_player(state) == "p_black"

    def test_get_legal_actions(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        actions = adapter.get_legal_actions(state)
        assert len(actions) == 9  # all 9 cells empty

    def test_get_chance_outcomes_empty(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        assert adapter.get_chance_outcomes(state) == []

    def test_is_terminal_initial(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        assert not adapter.is_terminal(state)


class TestMoonChessGameplay:
    def test_place_piece(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        actions = adapter.get_legal_actions(state)

        # Place black at (0,0)
        new_state = adapter.apply_action(state, actions[0])
        assert new_state["_arrays"]["board"][0] == "p_black"
        assert len(new_state["_arrays"]["pieceOrder"]) == 1
        assert new_state["_arrays"]["pieceOrder"][0]["cell_id"] == "cell_0_0"

    def test_switch_turn(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        actions = adapter.get_legal_actions(state)
        state = adapter.apply_action(state, actions[0])
        assert adapter.get_current_player(state) == "p_white"

    def test_fifo_eviction(self, adapter: MoonChessAdapter):
        """After placing 4+ pieces for one player, the oldest should disappear."""
        state = adapter.create_initial_state()

        black_count = 0
        max_black = 0
        while black_count < 4 and not adapter.is_terminal(state):
            actions = adapter.get_legal_actions(state)
            if not actions:
                break
            state = adapter.apply_action(state, actions[0])
            if adapter.get_current_player(state) == "p_black" or adapter.is_terminal(state):
                black_count += 1
            board = state["_arrays"]["board"]
            black_pieces = [i for i, c in enumerate(board) if c == "p_black"]
            max_black = max(max_black, len(black_pieces))

        board = state["_arrays"]["board"]
        black_pieces = [i for i, c in enumerate(board) if c == "p_black"]
        assert len(black_pieces) <= 3
        assert max_black >= 3

    def test_fifo_eviction_clears_correct_cell(self, adapter: MoonChessAdapter):
        """The evicted piece's own cell must be cleared (row-major index)."""
        state = adapter.create_initial_state()
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
            actions = adapter.get_legal_actions(state)
            action = next(
                a for a in actions
                if a.params.get("cell", {}).get("id", "") == cell_id
            )
            state = adapter.apply_action(state, action)

        board = state["_arrays"]["board"]
        black_po = [
            e["cell_id"] for e in state["_arrays"]["pieceOrder"]
            if e["player_id"] == "p_black"
        ]
        assert board[1] is None, "evicted cell_0_1 (idx 1) must be cleared"
        assert board[3] == "p_black", "cell_1_0 (idx 3) must not be collateral"
        assert board[7] == "p_black", "new piece at cell_2_1 (idx 7)"
        assert black_po == ["cell_1_0", "cell_1_2", "cell_2_1"]

    def test_three_in_row_win(self, adapter: MoonChessAdapter):
        """Black places in top row (0,0), (0,1), (0,2) → should win."""
        state = adapter.create_initial_state()

        placements = [
            ("p_black", "cell_0_0"),
            ("p_white", "cell_1_0"),
            ("p_black", "cell_0_1"),
            ("p_white", "cell_1_1"),
            ("p_black", "cell_0_2"),
        ]

        for player, cell_id in placements:
            actions = adapter.get_legal_actions(state)
            action = None
            for a in actions:
                cell = a.params.get("cell", {})
                cid = cell.get("id", "") if isinstance(cell, dict) else ""
                if cid == cell_id:
                    action = a
                    break
            assert action is not None, f"Action for {cell_id} not found"

            if player == "p_black":
                assert adapter.get_current_player(state) == "p_black"
            state = adapter.apply_action(state, action)

        assert adapter.is_terminal(state)
        assert state["env"]["winner"] == "p_black"

    def test_action_mask(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        mask = adapter.get_action_mask(state)
        assert mask.shape == (9,)
        assert mask.sum() == 9  # all legal

        actions = adapter.get_legal_actions(state)
        state = adapter.apply_action(state, actions[0])
        state = adapter.apply_action(state, adapter.get_legal_actions(state)[0])

        if not adapter.is_terminal(state):
            mask2 = adapter.get_action_mask(state)
            assert mask2.sum() <= 8  # at least one cell occupied

    def test_feature_vector(self, adapter: MoonChessAdapter):
        state = adapter.create_initial_state()
        vec = adapter.get_feature_vector(state, "p_black")
        assert vec.shape == (38,)
        assert vec.dtype == np.float32
        assert 0.0 <= vec.min() <= 1.0
        assert 0.0 <= vec.max() <= 3.0


class TestMoonChessAdapterProtocol:
    """Verify MoonChessAdapter satisfies SolverAdapter Protocol."""

    def test_is_solver_adapter(self, adapter: MoonChessAdapter):
        assert isinstance(adapter, SolverAdapter)

    def test_all_protocol_methods_exist(self, adapter: MoonChessAdapter):
        methods = [
            "create_initial_state", "get_node_type", "get_current_player",
            "get_legal_actions", "apply_action", "get_chance_outcomes",
            "apply_chance", "is_terminal", "get_utility",
            "get_observation", "get_info_set_key", "load_state",
            "project_observation",
        ]
        for m in methods:
            assert hasattr(adapter, m), f"Missing method: {m}"

    def test_load_state_from_observation(self, adapter: MoonChessAdapter):
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

        state = adapter.load_state({
            "_arrays": {"board": _board},
            "env": {
                "phase": "playing",
                "turn": "p_black",
                "winner": None,
            },
        })
        assert state["_arrays"]["board"][1] == "p_black"
        assert state["_arrays"]["board"][6] == "p_white"
        assert adapter.get_current_player(state) == "p_black"
