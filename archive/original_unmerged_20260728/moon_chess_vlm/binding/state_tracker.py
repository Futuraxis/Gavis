"""StateTracker 负责比对观测差异。"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import AmbiguousObservationError, MissingHistoryError
from .schemas import Observation


@dataclass(slots=True)
class StateChange:
    added_cells: list[str]
    removed_cells: list[str]
    inferred_actor_id: str | None
    confidence: float
    ambiguous: bool
    history_incomplete: bool = False


class StateTracker:
    """根据上一帧状态和当前观测推断变化。"""

    def __init__(self, confidence_threshold: float = 0.8) -> None:
        self.confidence_threshold = confidence_threshold

    def infer_state_change(self, previous_state: dict | None, observation: Observation) -> StateChange:
        if previous_state is None:
            raise MissingHistoryError("缺少上一帧完整状态，无法恢复棋子顺序。")

        previous_board = previous_state["board"]
        added_cells: list[str] = []
        removed_cells: list[str] = []
        confidences: list[float] = []

        for row_idx in range(3):
            for col_idx in range(3):
                old_cell = previous_board[row_idx][col_idx]
                new_cell = observation.boardObservation[row_idx][col_idx]
                score = observation.confidence[row_idx][col_idx]
                if score < self.confidence_threshold:
                    continue
                if old_cell == new_cell:
                    continue
                cell_id = f"cell_{row_idx}_{col_idx}"
                confidences.append(score)
                if old_cell is None and new_cell in ("X", "O"):
                    added_cells.append(cell_id)
                    continue
                if old_cell in ("X", "O") and new_cell is None:
                    removed_cells.append(cell_id)
                    continue
                raise AmbiguousObservationError("检测到棋子类型直接变化，无法仅靠观测解释。")

        if not added_cells and not removed_cells:
            return StateChange([], [], None, 1.0, False)

        if len(added_cells) > 1 or len(removed_cells) > 1:
            raise AmbiguousObservationError("单帧包含多个新增或多个消失，无法可靠推断。")

        if added_cells and removed_cells:
            added_cell = added_cells[0]
            removed_cell = removed_cells[0]
            added_piece = self._get_cell_value(observation.boardObservation, added_cell)
            removed_piece = self._get_cell_value(previous_board, removed_cell)
            if added_piece != removed_piece:
                raise AmbiguousObservationError("新增与消失棋子不同色，不符合月亮棋自动移除规则。")

        inferred_actor_id = self._infer_actor_id(previous_state, observation, added_cells)
        avg_confidence = sum(confidences) / len(confidences)
        return StateChange(added_cells, removed_cells, inferred_actor_id, avg_confidence, False)

    def _infer_actor_id(
        self,
        previous_state: dict,
        observation: Observation,
        added_cells: list[str],
    ) -> str | None:
        if not added_cells:
            return None
        piece = self._get_cell_value(observation.boardObservation, added_cells[0])
        mapping = {
            "X": previous_state.get("playerSymbols", {}).get("X", "player_x"),
            "O": previous_state.get("playerSymbols", {}).get("O", "player_o"),
        }
        return mapping.get(piece)

    @staticmethod
    def _get_cell_value(board: list[list[str | None]], cell_id: str) -> str | None:
        _, row, col = cell_id.split("_")
        return board[int(row)][int(col)]
