"""Online learning — collects real-game feedback and feeds it back to Solvers.

Layer 4 package: capture (``recorder``), persistence (``store``), signal
shaping (``signals`` / ``feedback_collector``), published model registry
(``models``) and the apply pipeline (``manager``).  No Layer-3 import —
solvers are reached only through the ``SolverProvider`` protocol and the
``IncrementalLearner`` protocol implemented in Layer 3 and assembled in
the app layer.
"""

from __future__ import annotations

from .feedback_collector import OnlineLearner, OnlineLearningSignal
from .manager import ApplyResult, LearningManager
from .models import OnlineModelStore, PublishedModel
from .recorder import LearningHooks, RecordingHandle, TrajectoryRecorder, jsonable
from .signals import outcome_for, signal_from_match
from .store import LearningStore, LearningStoreError

__all__ = [
    "OnlineLearningSignal",
    "OnlineLearner",
    "LearningStore",
    "LearningStoreError",
    "TrajectoryRecorder",
    "RecordingHandle",
    "LearningHooks",
    "OnlineModelStore",
    "PublishedModel",
    "ApplyResult",
    "LearningManager",
    "jsonable",
    "signal_from_match",
    "outcome_for",
]
