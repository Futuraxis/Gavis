"""Mock Binding — returns pre-configured observations for testing."""

from __future__ import annotations

from .base_binding import BaseBinding
from .schemas import Observation


class MockBinding:
    """Returns a pre-configured ``Observation``.  Useful for testing."""

    def __init__(self, observation: Observation | None = None) -> None:
        self._observation = observation or Observation(
            gameId="moon_demo_001",
            source="mock",
            frameSeq=0,
            boardObservation=[
                [None, None, None],
                [None, "X", None],
                [None, None, "O"],
            ],
            confidence=[[0.0, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.92]],
            observedAt=0,
        )
        self._frame_seq = 0

    def parse(self, source: str) -> Observation:
        return self._observation

    def parse_image(self, image_path: str, **kwargs) -> Observation:
        self._frame_seq += 1
        # Return a copy so mutating the caller's copy doesn't affect internal state
        obs = self._observation
        return Observation(
            gameId=obs.gameId,
            source=obs.source,
            frameSeq=self._frame_seq,
            boardObservation=[row[:] for row in obs.boardObservation],
            confidence=[row[:] for row in obs.confidence],
            observedAt=obs.observedAt,
        )

    def parse_bytes(self, data: bytes, mime_type: str, **kwargs) -> Observation:
        return self.parse_image("")
