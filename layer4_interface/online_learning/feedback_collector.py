"""Online learning feedback collector.

Collects real-game experiences and feeds them back to the Solver
for continuous improvement.  This module defines the data structures;
the actual training loop is future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OnlineLearningSignal:
    """A single game's experience collected from real human play.

    This is the input to the online self-learning loop.
    """
    game_id: str
    solver_name: str
    controlled_player: str
    state_sequence: list[dict] = field(default_factory=list)
    actions_taken: list[dict] = field(default_factory=list)
    solver_suggestions: list[dict | None] = field(default_factory=list)
    final_outcome: float = 0.0          # +1 (win), 0 (draw), -1 (loss)
    user_rating: Optional[int] = None   # optional user satisfaction (1-5)
    metadata: dict[str, Any] = field(default_factory=dict)


class OnlineLearner:
    """Collects and stores online learning signals.

    In the full implementation, accumulated signals are periodically
    fed back into the Solver (PPO replay buffer, CFR extra iterations,
    PSRO new policy evaluation).
    """

    def __init__(self, buffer_size: int = 10000):
        self._buffer: list[OnlineLearningSignal] = []
        self._buffer_size = buffer_size

    def collect(self, signal: OnlineLearningSignal) -> None:
        """Record one game's experience."""
        self._buffer.append(signal)
        if len(self._buffer) > self._buffer_size:
            self._buffer.pop(0)

    def flush(self, solver) -> int:
        """Feed accumulated signals into a solver for online learning.

        Returns the number of signals processed.
        """
        count = len(self._buffer)
        # Future: implement actual Solver online update here
        self._buffer.clear()
        return count

    @property
    def size(self) -> int:
        return len(self._buffer)
