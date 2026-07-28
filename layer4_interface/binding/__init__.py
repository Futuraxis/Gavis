"""Binding Layer — converts external visual input to GameState.

Two parallel pipelines:
- ``ImageBinding``: OpenCV-based template matching (fast, no GPU)
- ``VisionLLMBinding``: Large Vision Model API (accurate, needs external API)
"""

from .base_binding import BaseBinding
from .exceptions import (
    AmbiguousObservationError,
    BindingError,
    ImageLoadError,
    InvalidBoardError,
    InvalidActionMaskError,
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
