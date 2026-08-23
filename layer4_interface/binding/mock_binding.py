"""Mock Binding — returns pre-configured observations for testing."""

from __future__ import annotations

import threading

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
        # 帧序号自增是读-改-写，并发下需加锁（审计 3.6）。
        self._seq_lock = threading.Lock()

    def parse(self, source: str) -> Observation:
        # 防御拷贝：与 parse_image 一致，调用方变异结果不污染内部状态（审查 P2）
        obs = self._observation
        return Observation(
            gameId=obs.gameId,
            source=obs.source,
            frameSeq=obs.frameSeq,
            boardObservation=[row[:] for row in obs.boardObservation],
            confidence=[row[:] for row in obs.confidence],
            observedAt=obs.observedAt,
        )

    def parse_image(self, image_path: str, **kwargs) -> Observation:
        with self._seq_lock:
            self._frame_seq += 1
            seq = self._frame_seq
        # Return a copy so mutating the caller's copy doesn't affect internal state
        obs = self._observation
        return Observation(
            gameId=obs.gameId,
            source=obs.source,
            frameSeq=seq,
            boardObservation=[row[:] for row in obs.boardObservation],
            confidence=[row[:] for row in obs.confidence],
            observedAt=obs.observedAt,
        )

    def parse_bytes(self, data: bytes, mime_type: str, **kwargs) -> Observation:
        return self.parse_image("")
