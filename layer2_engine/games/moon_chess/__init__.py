"""Moon Chess — 3×3 board, 3 pieces per player, FIFO eviction, three-in-a-row wins."""

from __future__ import annotations

from .moon_env_adapter import MoonChessAdapter

__all__ = ["MoonChessAdapter"]
