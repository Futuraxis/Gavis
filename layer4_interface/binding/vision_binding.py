"""VisionLLMBinding — uses a Large Vision Model to parse board images.

Unlike ``ImageBinding`` which requires pre-cropped board cells, this
binding sends the entire screenshot to a VLM and lets the model
identify the board and its pieces in one step.
"""

from __future__ import annotations

import time
from typing import Protocol

from .exceptions import VisionModelResponseError
from .schemas import Observation


class VisionModelClient(Protocol):
    """Interface for a vision model API client."""

    def infer_observation(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> dict:
        """Send image to the vision model.

        Must return a dict with:
            - ``boardObservation``: list[list[str | None]]
            - ``confidence``: list[list[float]]
        """
        ...


PROMPT_TEMPLATE = """Please identify the 3×3 game board in this image.

For each cell, determine if it contains:
- "X" (player X's piece)
- "O" (player O's piece)
- None (empty cell)

Return the board as a JSON-like 3×3 grid. Also return a confidence
score (0.0 to 1.0) for each cell.

Only focus on the central 3×3 board area. Ignore any UI elements
outside the board."""


class VisionLLMBinding:
    """Parses a full-page screenshot using a Vision Language Model."""

    def __init__(
        self,
        client: VisionModelClient,
        game_id: str = "moon_demo_001",
        source_name: str = "vision_llm",
        prompt: str = PROMPT_TEMPLATE,
    ) -> None:
        self._client = client
        self.game_id = game_id
        self.source_name = source_name
        self.prompt = prompt
        self._last_frame_seq = -1

    def parse(self, source: str) -> Observation:
        with open(source, 'rb') as f:
            data = f.read()
        return self.parse_bytes(data, "image/png")

    def parse_image(self, image_path: str, **kwargs) -> Observation:
        return self.parse(image_path)

    def parse_bytes(
        self,
        data: bytes,
        mime_type: str,
        *,
        frame_seq: int | None = None,
        observed_at: int | None = None,
    ) -> Observation:
        try:
            response = self._client.infer_observation(
                image_bytes=data,
                mime_type=mime_type,
                prompt=self.prompt,
            )
        except Exception as e:
            raise VisionModelResponseError(f"Vision model call failed: {e}") from e

        board = response.get("boardObservation")
        confidence = response.get("confidence")

        if not board or not confidence:
            raise VisionModelResponseError(
                "Vision model response missing boardObservation or confidence."
            )

        if frame_seq is None:
            self._last_frame_seq += 1
            frame_seq = self._last_frame_seq

        return Observation(
            gameId=self.game_id,
            source=self.source_name,
            frameSeq=frame_seq,
            boardObservation=board,
            confidence=confidence,
            observedAt=observed_at or int(time.time() * 1000),
        )
