"""月亮棋状态编码器。"""

from __future__ import annotations

import numpy as np

from .game_state_adapter import GameStateAdapter


class MoonStateEncoder:
    """把月亮棋状态编码成固定长度向量。"""

    BOARD_SIZE = 3
    ACTION_DIM = 9
    FEATURE_DIM = 38  # 27(占用 one-hot) + 9(年龄编码) + 1(当前行动方) + 1(归一化步数)

    def __init__(self, adapter: GameStateAdapter | None = None, max_step_count: int = 32) -> None:
        self.adapter = adapter or GameStateAdapter()
        self.max_step_count = max_step_count

    def encode(self, state: dict, perspective_player_id: str) -> np.ndarray:
        board = self.adapter.get_board(state)
        piece_order = self.adapter.get_piece_order(state)
        current_player_id = self.adapter.get_current_player(state)
        features: list[float] = []

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                cell = board[row][col]
                if cell is None:
                    features.extend([1.0, 0.0, 0.0])
                else:
                    owner_id = self._piece_owner_from_symbol(state, cell)
                    if owner_id == perspective_player_id:
                        features.extend([0.0, 1.0, 0.0])
                    else:
                        features.extend([0.0, 0.0, 1.0])

        age_map = self._build_age_map(piece_order)
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                cell_id = f"cell_{row}_{col}"
                features.append(float(age_map.get(cell_id, 0)))

        features.append(1.0 if current_player_id == perspective_player_id else 0.0)
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
    def _piece_owner_from_symbol(state: dict, symbol: str) -> str:
        player_symbols = state.get("playerSymbols")
        if player_symbols:
            return player_symbols.get(symbol, "player_x" if symbol == "X" else "player_o")
        return "player_x" if symbol == "X" else "player_o"


def action_index_to_cell_id(action_index: int) -> str:
    if action_index < 0 or action_index >= 9:
        raise ValueError(f"动作索引必须位于 0 到 8，收到 {action_index}。")
    row, col = divmod(action_index, 3)
    return f"cell_{row}_{col}"


def cell_id_to_action_index(cell_id: str) -> int:
    _, row, col = cell_id.split("_")
    row_index = int(row)
    col_index = int(col)
    if row_index not in range(3) or col_index not in range(3):
        raise ValueError(f"非法 cellId: {cell_id}")
    return row_index * 3 + col_index
