"""Online learning — collects real-game feedback and feeds it back to Solvers.

This module defines the interfaces for the online self-learning loop.
Full implementation is future work.
"""
from .feedback_collector import OnlineLearningSignal, OnlineLearner

__all__ = ["OnlineLearningSignal", "OnlineLearner"]
