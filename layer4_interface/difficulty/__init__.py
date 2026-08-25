"""Adaptive difficulty — win-rate driven search-budget control (Layer 4)."""

from __future__ import annotations

from .adaptive import PACING, AdaptiveController, PacingSpec, pacing_scale, pacing_seconds

__all__ = [
    "AdaptiveController",
    "PACING",
    "PacingSpec",
    "pacing_scale",
    "pacing_seconds",
]
