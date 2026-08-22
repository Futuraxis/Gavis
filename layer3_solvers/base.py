"""SolverBase — abstract base for all Layer 3 solvers.

Every solver (MCTS, CFR, PPO, PSRO) implements this interface so that
demos, benchmarks, and the auto-selector can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from layer2_engine.interfaces.solver_adapter import (
    ActionInstance,
    SolverAdapter,
    State,
)


@dataclass
class SolverConfig:
    """Generic solver configuration.

    Individual solvers may extend this with their own parameters.
    """

    seed: Optional[int] = None
    device: str = "cpu"  # "cpu" | "cuda" — used by neural solvers
    verbose: bool = False


@dataclass
class SolverMetrics:
    """Training/benchmark metrics returned by ``train()``."""

    episodes: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    # Solver-specific extras (counts like info_sets/pool_size/steps are ints).
    extra: dict[str, float | int] = field(default_factory=dict)


class SolverBase(ABC):
    """Every solver in Layer 3 implements this interface.

    Usage::

        solver = MCTS(engine, SolverConfig(seed=42))
        action = solver.select_action(state)
        metrics = solver.train(episodes=100)
        solver.save("model.pt")
    """

    def __init__(self, adapter: SolverAdapter, config: SolverConfig):
        self.adapter = adapter
        self.config = config

    # ── Required ──────────────────────────────────────────────────

    @abstractmethod
    def select_action(self, state: State) -> Optional[ActionInstance]:
        """Return the best action for ``state``, or None if no legal moves."""
        ...

    @abstractmethod
    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        """Run training for ``episodes`` self-play or simulated episodes.

        Returns training metrics (win rate, average return, etc.).
        """
        ...

    # ── Optional (save/load) ──────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the solver's learned parameters to ``path``."""
        pass  # default: no-op

    def load(self, path: str) -> None:
        """Load learned parameters from ``path``."""
        pass  # default: no-op

    # ── Name ───────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable solver name (defaults to class name)."""
        return type(self).__name__
