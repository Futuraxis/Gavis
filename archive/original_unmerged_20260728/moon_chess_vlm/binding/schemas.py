"""Binding Layer 核心数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CellValue = Literal["X", "O"] | None
BoardMatrix = list[list[CellValue]]
ConfidenceMatrix = list[list[float]]


def _validate_square_matrix(
    value: list[list[object]],
    *,
    expected_size: int,
    field_name: str,
) -> list[list[object]]:
    if len(value) != expected_size:
        raise ValueError(f"{field_name} 必须是 {expected_size}x{expected_size} 矩阵。")
    for row in value:
        if len(row) != expected_size:
            raise ValueError(f"{field_name} 必须是 {expected_size}x{expected_size} 矩阵。")
    return value


class Observation(BaseModel):
    """Binding 输出的统一观测结构。"""

    gameId: str
    source: str
    frameSeq: int = Field(ge=0)
    boardObservation: BoardMatrix
    confidence: ConfidenceMatrix
    observedAt: int = Field(ge=0)

    @field_validator("boardObservation")
    @classmethod
    def validate_board(cls, value: BoardMatrix) -> BoardMatrix:
        _validate_square_matrix(value, expected_size=3, field_name="boardObservation")
        for row in value:
            for cell in row:
                if cell not in ("X", "O", None):
                    raise ValueError("boardObservation 中每个格子只能是 'X'、'O' 或 None。")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: ConfidenceMatrix) -> ConfidenceMatrix:
        _validate_square_matrix(value, expected_size=3, field_name="confidence")
        for row in value:
            for score in row:
                if not 0.0 <= score <= 1.0:
                    raise ValueError("confidence 中每个分数必须位于 0 到 1 之间。")
        return value
