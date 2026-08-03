"""Tests for Encoding layer (Layer 4)."""

from __future__ import annotations

import numpy as np
import pytest

from layer4_interface.encoding.game_state_adapter import GameStateAdapter
from layer4_interface.encoding.moon_state_encoder import (
    MoonStateEncoder,
    action_index_to_cell_id,
    cell_id_to_action_index,
)


class TestGameStateAdapter:
    def test_get_board(self):
        adapter = GameStateAdapter()
        state = {"board": [["X", None], [None, "O"]]}
        assert adapter.get_board(state) == [["X", None], [None, "O"]]

    def test_get_current_player(self):
        adapter = GameStateAdapter()
        assert adapter.get_current_player({"currentPlayerId": "player_x"}) == "player_x"

    def test_is_terminal_running(self):
        adapter = GameStateAdapter()
        assert not adapter.is_terminal({"status": "running"})

    def test_is_terminal_finished(self):
        adapter = GameStateAdapter()
        assert adapter.is_terminal({"status": "finished"})


class TestMoonStateEncoder:
    @pytest.fixture
    def encoder(self) -> MoonStateEncoder:
        return MoonStateEncoder()

    def test_feature_dim(self, encoder: MoonStateEncoder):
        assert encoder.FEATURE_DIM == 38

    def test_encode_empty_board(self, encoder: MoonStateEncoder):
        state = {
            "board": [[None, None, None], [None, None, None], [None, None, None]],
            "currentPlayerId": "player_x",
            "pieceOrder": {"player_x": [], "player_o": []},
            "stepCount": 0,
            "status": "running",
            "legalActions": ["cell_0_0", "cell_0_1"],
            "playerSymbols": {"X": "player_x", "O": "player_o"},
        }
        vec = encoder.encode(state, "player_x")
        assert vec.shape == (38,)
        assert vec.dtype == np.float32

    def test_encode_with_pieces(self, encoder: MoonStateEncoder):
        state = {
            "board": [["X", None, None], [None, "O", None], [None, None, None]],
            "currentPlayerId": "player_x",
            "pieceOrder": {
                "player_x": [{"cellId": "cell_0_0", "placedSeq": 1}],
                "player_o": [{"cellId": "cell_1_1", "placedSeq": 2}],
            },
            "stepCount": 2,
            "status": "running",
            "legalActions": ["cell_0_1"],
            "playerSymbols": {"X": "player_x", "O": "player_o"},
        }
        vec = encoder.encode(state, "player_x")
        assert vec.shape == (38,)
        # cell (0,0) should be self (second one-hot active)
        assert vec[1] == 1.0  # self indicator for cell_0_0
        # cell (1,1) should be opponent
        assert vec[3 * 4 + 2] == 1.0  # opponent indicator for cell_1_1 (index 4)

    def test_get_action_mask(self, encoder: MoonStateEncoder):
        state = {
            "board": [[None, None, None], [None, None, None], [None, None, None]],
            "currentPlayerId": "player_x",
            "pieceOrder": {"player_x": [], "player_o": []},
            "stepCount": 0,
            "status": "running",
            "legalActions": ["cell_0_0", "cell_1_1", "cell_2_2"],
            "playerSymbols": {"X": "player_x", "O": "player_o"},
        }
        mask = encoder.get_action_mask(state)
        assert mask.shape == (9,)
        assert mask[0] == 1.0  # cell_0_0
        assert mask[4] == 1.0  # cell_1_1
        assert mask[8] == 1.0  # cell_2_2
        assert mask[1] == 0.0  # cell_0_1 not legal

    def test_action_index_conversion(self):
        assert action_index_to_cell_id(0) == "cell_0_0"
        assert action_index_to_cell_id(4) == "cell_1_1"
        assert action_index_to_cell_id(8) == "cell_2_2"
        assert cell_id_to_action_index("cell_0_0") == 0
        assert cell_id_to_action_index("cell_2_2") == 8

        with pytest.raises(ValueError):
            action_index_to_cell_id(9)

    def test_age_map(self, encoder: MoonStateEncoder):
        """Verify age encoding: newest piece has age 1, oldest has highest age."""
        state = {
            "board": [["X", None, None], [None, "O", "X"], [None, None, None]],
            "currentPlayerId": "player_o",
            "pieceOrder": {
                "player_x": [
                    {"cellId": "cell_0_0", "placedSeq": 1},
                    {"cellId": "cell_1_2", "placedSeq": 3},
                ],
                "player_o": [
                    {"cellId": "cell_1_1", "placedSeq": 2},
                ],
            },
            "stepCount": 3,
            "status": "running",
            "legalActions": ["cell_0_1"],
            "playerSymbols": {"X": "player_x", "O": "player_o"},
        }
        vec = encoder.encode(state, "player_x")
        # Age map: cell_0_0 (seq=1, oldest X) → age=1, cell_1_1 (seq=2, O) → age=1
        #          cell_1_2 (seq=3, newest X) → age=2
        # Index: cell_0_0=27+0=27, cell_1_1=27+4=31, cell_1_2=27+5=32
        assert vec[27] == 1.0  # cell_0_0 age 1 (oldest)
        assert vec[31] == 1.0  # cell_1_1 age 1
        assert vec[32] == 2.0  # cell_1_2 age 2 (newest)
