"""Layer 4: Interface — VLM binding, state encoding, online learning.

Connects the real world (screenshots, live streams, game apps) to the
Engine (Layer 2) and Solvers (Layer 3).
"""

from .binding import (
    BaseBinding,
    ImageBinding,
    VisionLLMBinding,
    MockBinding,
    Observation,
    StateTracker,
)
from .vision_bridge import observation_to_state

__all__ = [
    "BaseBinding",
    "ImageBinding",
    "VisionLLMBinding",
    "MockBinding",
    "Observation",
    "StateTracker",
    "observation_to_state",
]
