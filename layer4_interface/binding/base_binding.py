"""Base binding interface."""

from __future__ import annotations

from typing import Protocol

from .schemas import Observation


class BaseBinding(Protocol):
    """Interface for all binding implementations (image, VLM, mock)."""

    def parse(self, source: str) -> Observation:
        """Parse a source (path, URL, etc.) into an Observation."""
        ...

    def parse_image(self, image_path: str, **kwargs) -> Observation:
        """Parse an image file into an Observation."""
        ...

    def parse_bytes(self, data: bytes, mime_type: str, **kwargs) -> Observation:
        """Parse raw image bytes into an Observation."""
        ...
