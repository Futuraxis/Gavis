"""DomBinding — accepts browser DOM observations.

The browser owns DOM access.  This binding only validates and wraps the
already-structured Observation payload so downstream Layer 4 interfaces
continue to receive the canonical ``Observation`` model.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .exceptions import InvalidBoardError
from .schemas import Observation


class DomBinding:
    """Converts a browser-generated DOM payload into ``Observation``."""

    def __init__(self, game_id: str = "moon_demo_001") -> None:
        self.game_id = game_id
        self._last_frame_seq = -1

    def parse(self, source: str) -> Observation:
        """Parse a JSON string DOM payload into ``Observation``."""
        try:
            payload = json.loads(source)
        except json.JSONDecodeError as exc:
            raise InvalidBoardError(f"DOM payload is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidBoardError("DOM payload must be a JSON object.")
        return self.parse_payload(payload)

    def parse_image(self, image_path: str, **kwargs: Any) -> Observation:
        """Reject image parsing because DOM observations do not use screenshots."""
        raise InvalidBoardError("DomBinding does not parse image files.")

    def parse_bytes(self, data: bytes, mime_type: str, **kwargs: Any) -> Observation:
        """Parse UTF-8 JSON bytes from a browser DOM observation payload."""
        if mime_type not in ("application/json", "text/json", "text/plain"):
            raise InvalidBoardError(f"DomBinding expects JSON bytes, got {mime_type}.")
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidBoardError(f"DOM payload bytes are not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidBoardError("DOM payload must be a JSON object.")
        return self.parse_payload(payload)

    def parse_payload(self, payload: dict[str, Any]) -> Observation:
        """Validate and wrap a browser-generated DOM observation payload."""
        board = payload.get("boardObservation")
        if not self._is_board_observation(board):
            raise InvalidBoardError("boardObservation must be a 3x3 grid containing X, O, or null.")

        confidence = payload.get("confidence")
        if confidence is None:
            confidence = [[1.0] * 3 for _ in range(3)]
        if not self._is_confidence_grid(confidence):
            raise InvalidBoardError("confidence must be a 3x3 numeric grid.")

        frame_seq = int(payload.get("frameSeq", self._last_frame_seq + 1))
        if frame_seq <= self._last_frame_seq:
            raise InvalidBoardError(f"frameSeq must increase, got {frame_seq} after {self._last_frame_seq}.")
        self._last_frame_seq = frame_seq

        return Observation(
            gameId=str(payload.get("gameId", self.game_id)),
            source="dom",
            frameSeq=frame_seq,
            boardObservation=board,
            confidence=confidence,
            observedAt=int(payload.get("observedAt", int(time.time() * 1000))),
        )

    @staticmethod
    def _is_board_observation(value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 3:
            return False
        for row in value:
            if not isinstance(row, list) or len(row) != 3:
                return False
            if any(cell not in ("X", "O", None) for cell in row):
                return False
        return True

    @staticmethod
    def _is_confidence_grid(value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 3:
            return False
        for row in value:
            if not isinstance(row, list) or len(row) != 3:
                return False
            if any(not isinstance(cell, int | float) for cell in row):
                return False
        return True
