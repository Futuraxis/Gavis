"""Online learning — collects real-game feedback and feeds it back to Solvers.

This module defines the interfaces for the online self-learning loop.
Full implementation is future work.
"""

from __future__ import annotations

from .feedback_collector import OnlineLearner, OnlineLearningSignal

__all__ = ["OnlineLearningSignal", "OnlineLearner"]
