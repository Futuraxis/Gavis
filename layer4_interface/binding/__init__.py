"""Binding Layer — converts external visual input to GameState.

Two parallel pipelines:
- ``ImageBinding``: OpenCV-based template matching (fast, no GPU)
- ``VisionLLMBinding``: Large Vision Model API (accurate, needs external API)
"""

from __future__ import annotations

from .base_binding import BaseBinding
from .dom_binding import DomBinding
from .exceptions import (
    AmbiguousObservationError,
    BindingError,
    ImageLoadError,
    InvalidActionMaskError,
    InvalidBoardError,
    InvalidConfidenceError,
    InvalidFrameSequenceError,
    MissingHistoryError,
    VisionModelResponseError,
)
from .image_binding import CellClassifier, ImageBinding, TemplateMatchingClassifier
from .mock_binding import MockBinding
from .qwen_vision import QwenVisionClient
from .schemas import Observation
from .state_tracker import StateChange, StateTracker
from .vision_binding import VisionLLMBinding, VisionModelClient

__all__ = [
    "AmbiguousObservationError",
    "BaseBinding",
    "BindingError",
    "CellClassifier",
    "DomBinding",
    "ImageBinding",
    "ImageLoadError",
    "InvalidActionMaskError",
    "InvalidBoardError",
    "InvalidConfidenceError",
    "InvalidFrameSequenceError",
    "MissingHistoryError",
    "MockBinding",
    "Observation",
    "QwenVisionClient",
    "StateChange",
    "StateTracker",
    "TemplateMatchingClassifier",
    "VisionLLMBinding",
    "VisionModelClient",
    "VisionModelResponseError",
]
