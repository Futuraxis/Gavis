"""用于本地联调的 MockBinding。"""

from __future__ import annotations

import time
from typing import Iterable

from pydantic import ValidationError

from .base_binding import BaseBinding
from .exceptions import (
    InvalidBoardError,
    InvalidConfidenceError,
    InvalidFrameSequenceError,
)
from .schemas import CellValue, Observation


class MockBinding(BaseBinding):
    """把 3x3 数组直接包装成 Observation。"""

    def __init__(
        self,
        game_id: str = "moon_demo_001",
        source_name: str = "mock_input",
        default_confidence: float = 1.0,
    ) -> None:
        self.game_id = game_id
        self.source_name = source_name
        self.default_confidence = default_confidence
        self._last_frame_seq = -1

    def parse(
        self,
        source: Iterable[Iterable[CellValue]],
        *,
        confidence: list[list[float]] | None = None,
        frame_seq: int | None = None,
        observed_at: int | None = None,
    ) -> Observation:
        board = self._normalize_board(source)
        confidence_matrix = confidence or [
            [self.default_confidence for _ in range(3)] for _ in range(3)
        ]
        self._validate_frame_seq(frame_seq)

        try:
            observation = Observation(
                gameId=self.game_id,
                source=self.source_name,
                frameSeq=frame_seq if frame_seq is not None else self._last_frame_seq + 1,
                boardObservation=board,
                confidence=confidence_matrix,
                observedAt=observed_at if observed_at is not None else int(time.time() * 1000),
            )
        except ValidationError as exc:
            if "confidence" in str(exc):
                raise InvalidConfidenceError(str(exc)) from exc
            raise InvalidBoardError(str(exc)) from exc

        self._last_frame_seq = observation.frameSeq
        return observation

    def _normalize_board(self, source: Iterable[Iterable[CellValue]]) -> list[list[CellValue]]:
        board = [list(row) for row in source]
        if len(board) != 3 or any(len(row) != 3 for row in board):
            raise InvalidBoardError("MockBinding 只接受 3x3 棋盘输入。")
        for row in board:
            for cell in row:
                if cell not in ("X", "O", None):
                    raise InvalidBoardError("棋盘格子只能是 'X'、'O' 或 None。")
        return board

    def _validate_frame_seq(self, frame_seq: int | None) -> None:
        if frame_seq is None:
            return
        if frame_seq <= self._last_frame_seq:
            raise InvalidFrameSequenceError(
                f"frameSeq 必须严格递增，上一帧为 {self._last_frame_seq}，当前收到 {frame_seq}。"
            )
