"""Moon chess state encoder — 38-dimensional feature vector."""

from __future__ import annotations

import numpy as np

from .game_state_adapter import GameStateAdapter


class MoonStateEncoder:
    """Encodes a Moon Chess state into a fixed-length feature vector.

    Feature layout (FEATURE_DIM = 38):
        0-26:    9 cells × 3 one-hot (empty / self / opponent)
        27-35:   9 cells × age encoding (1=latest … 3=oldest)
        36:      whose turn (1 = perspective player)
        37:      normalized step count
    """

    BOARD_SIZE = 3
    ACTION_DIM = 9
    FEATURE_DIM = 38

    def __init__(self, adapter: GameStateAdapter | None = None, max_step_count: int = 32) -> None:
        self.adapter = adapter or GameStateAdapter()
        self.max_step_count = max_step_count

    def encode(self, state: dict, perspective_player_id: str) -> np.ndarray:
        board = self.adapter.get_board(state)
        piece_order = self.adapter.get_piece_order(state)
        current_player_id = self.adapter.get_current_player(state)
        features: list[float] = []

        # 0-26: cell occupancy one-hot
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                cell = self._get_cell(board, r, c)
                if cell is None:
                    features.extend([1.0, 0.0, 0.0])
                elif self._owner_from_symbol(state, cell) == perspective_player_id:
                    features.extend([0.0, 1.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 1.0])

        # 27-35: age encoding
        age_map = self._build_age_map(piece_order)
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                cell_id = f"cell_{r}_{c}"
                features.append(float(age_map.get(cell_id, 0)))

        # 36: turn indicator
        features.append(1.0 if current_player_id == perspective_player_id else 0.0)

        # 37: normalized step count
        features.append(min(1.0, self.adapter.get_step_count(state) / float(self.max_step_count)))

        return np.asarray(features, dtype=np.float32)

    def get_action_mask(self, state: dict) -> np.ndarray:
        legal_actions = set(self.adapter.get_legal_actions(state))
        mask = np.zeros(self.ACTION_DIM, dtype=np.float32)
        if self.adapter.is_terminal(state):
            return mask
        for index in range(self.ACTION_DIM):
            cell_id = action_index_to_cell_id(index)
            if cell_id in legal_actions:
                mask[index] = 1.0
        return mask

    def _build_age_map(self, piece_order: dict[str, list[dict]]) -> dict[str, int]:
        age_map: dict[str, int] = {}
        for entries in piece_order.values():
            sorted_entries = sorted(entries, key=lambda item: item["placedSeq"])
            for age, entry in enumerate(sorted_entries, start=1):
                age_map[entry["cellId"]] = age
        return age_map

    @staticmethod
    def _owner_from_symbol(state: dict, symbol: str) -> str:
        player_symbols = state.get("playerSymbols")
        if player_symbols:
            return player_symbols.get(symbol, "player_x" if symbol == "X" else "player_o")
        return "player_x" if symbol == "X" else "player_o"

    @staticmethod
    def _get_cell(board, r, c):
        if isinstance(board, list):
            if r < len(board) and c < len(board[r]):
                return board[r][c]
        return None


def action_index_to_cell_id(action_index: int) -> str:
    if action_index < 0 or action_index >= 9:
        raise ValueError(f"Action index must be 0-8, got {action_index}.")
    row, col = divmod(action_index, 3)
    return f"cell_{row}_{col}"


def cell_id_to_action_index(cell_id: str) -> int:
    _, row, col = cell_id.split("_")
    return int(row) * 3 + int(col)
