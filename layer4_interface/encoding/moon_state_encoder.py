"""Moon chess state encoder — 38-dimensional feature vector."""

from __future__ import annotations

import numpy as np

from .game_state_adapter import GameStateAdapter


class MoonStateEncoder:
    """Encodes a Moon Chess state into a fixed-length feature vector.

    Feature layout (FEATURE_DIM = 38):
        0-26:    9 cells × 3 one-hot (empty / self / opponent)
        27-35:   9 cells × stack age (1=oldest … 3=latest, 0=empty —
                 FIFO-eviction order, matching the adapter's docstring)
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
        legal_actions = self.adapter.get_legal_actions(state)
        mask = np.zeros(self.ACTION_DIM, dtype=np.float32)
        if self.adapter.is_terminal(state):
            return mask
        # Collect legal CELL IDS only — never hash the action objects
        # themselves (C-02: ActionInstance is an unhashable dataclass, so
        # ``set(actions)`` raised TypeError; comparing cell ids against
        # the action objects also silently produced an all-zero mask).
        legal_cells: set[str] = set()
        for a in legal_actions:
            if isinstance(a, str):
                legal_cells.add(a)
                continue
            cell = a.params.get("cell", {}) if isinstance(getattr(a, "params", None), dict) else None
            if isinstance(cell, dict) and cell.get("id"):
                legal_cells.add(str(cell["id"]))
        for index in range(self.ACTION_DIM):
            cell_id = action_index_to_cell_id(index)
            if cell_id in legal_cells:
                mask[index] = 1.0
        return mask

    def _build_age_map(self, piece_order) -> dict[str, int]:
        """Per-cell piece age (1 = the player's first/oldest placement … N = latest).

        与 L2 ``MoonChessAdapter.get_feature_vector`` 的年龄语义一致（审查
        Minor 1 / M5）：年龄 = 该玩家自己的第 k 次落子（1-based，list 顺序
        即落子顺序），按 cell 记录、后写覆盖——一个 cell 反复落子后其年龄
        为该玩家最近一次落子的序号。``piece_order`` 来自
        ``GameStateAdapter.get_piece_order``：v5.0 下是 ``_arrays.pieceOrder``
        的扁平 record 列表（``{"cell_id", "player_id"}``，旧实现对其调用
        ``.values()`` 会 AttributeError），v4.1 下是按玩家分的 dict
        （``{"player_x": ["cell_0_0", ...]}`` 或 ``{"player_x":
        [{"cellId": ..., "placedSeq": ...}]}``）。
        """
        per_player: dict[str, list] = {}
        if isinstance(piece_order, dict):
            for pid, cells in piece_order.items():
                for cell in cells:
                    if isinstance(cell, str):
                        per_player.setdefault(pid, []).append(cell)
                    else:
                        cell_id = cell.get("cell_id") or cell.get("cellId")
                        if cell_id:
                            per_player.setdefault(pid, []).append(cell_id)
        else:
            for entry in piece_order or []:
                pid = entry.get("player_id", "")
                cell_id = entry.get("cell_id") or entry.get("cellId")
                if pid and cell_id:
                    per_player.setdefault(pid, []).append(cell_id)
        age_map: dict[str, int] = {}
        for cells in per_player.values():
            for age, cid in enumerate(cells, start=1):
                if cid:
                    age_map[cid] = age
        return age_map

    @staticmethod
    def _owner_from_symbol(state: dict, symbol: str) -> str:
        # v5.0 引擎的 _arrays.board 直接存 player id（p_black/p_white，实测
        # 审查 M-5）；v4.1 存符号（X/O）并配 playerSymbols 映射。
        if symbol in ("p_black", "p_white", "player_x", "player_o"):
            return symbol
        player_symbols = state.get("playerSymbols")
        if player_symbols:
            return player_symbols.get(symbol, symbol)
        return "player_x" if symbol == "X" else ("player_o" if symbol == "O" else symbol)

    @staticmethod
    def _get_cell(board, r, c):
        if isinstance(board, list):
            if board and isinstance(board[0], list):
                # v4.1 二维布局
                if r < len(board) and c < len(board[r]):
                    return board[r][c]
            else:
                # v5.0 扁平数组（_arrays.board，row-major 9 元素）
                idx = r * 3 + c
                if 0 <= idx < len(board):
                    return board[idx]
        return None


def action_index_to_cell_id(action_index: int) -> str:
    if action_index < 0 or action_index >= 9:
        raise ValueError(f"Action index must be 0-8, got {action_index}.")
    row, col = divmod(action_index, 3)
    return f"cell_{row}_{col}"


def cell_id_to_action_index(cell_id: str) -> int:
    _, row, col = cell_id.split("_")
    return int(row) * 3 + int(col)
