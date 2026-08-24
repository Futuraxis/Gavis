"""Online learning feedback collector.

Collects real-game experiences and feeds them back to the Solver for
continuous improvement.  This module declares the stable data structures
(``OnlineLearningSignal``) and the in-memory collector; the actual
training pipeline (table building, gate evaluation, publish) lives in
``manager.py`` and runs inside ``layer4_interface/online_learning/``.

Layering: this package never imports ``layer3_solvers``.  Downstream
consumers (an empirical opponent table, a PPO update, ...) are declared
via protocols and assembled in the app layer (``train-cli/games.py``),
mirroring the ``SolverProvider`` dependency-inversion pattern.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


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
    final_outcome: float = 0.0  # +1 (win), 0 (draw), -1 (loss)
    user_rating: int | None = None  # optional user satisfaction (1-5)
    metadata: dict[str, Any] = field(default_factory=dict)


class OnlineLearner:
    """Collects and stores online learning signals.

    ``collect`` keeps the historical in-memory semantics (bounded ring
    buffer, thread-safe).  ``collect_match`` converts one recorded store
    match into a signal and queues it — the same conversion the learning
    pipeline applies on every finished match.
    """

    def __init__(self, buffer_size: int = 10000) -> None:
        self._buffer: list[OnlineLearningSignal] = []
        self._buffer_size = buffer_size
        # 共享 buffer 的读写并发保护（审计 3.6）。
        self._lock = threading.Lock()

    def collect(self, signal: OnlineLearningSignal) -> None:
        """Record one game's experience."""
        with self._lock:
            self._buffer.append(signal)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

    def collect_match(self, game_id: str, solver_name: str, match: dict) -> None:
        """Convert one store match block into a signal and queue it."""
        from .signals import signal_from_match  # local import: avoid package cycle

        self.collect(signal_from_match(game_id, solver_name, match))

    def signals(self) -> list[OnlineLearningSignal]:
        """Snapshot of the buffered signals (newest last)."""
        with self._lock:
            return list(self._buffer)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
