"""Layer 2: Env/Engine — declarative game engine.

Loads ``rules.json`` and provides a full game runtime that all
Layer 3 solvers consume via the ``SolverAdapter`` Protocol.
"""

from __future__ import annotations

from .core.engine import GameEngine
from .interfaces.solver_adapter import (
    ActionInstance,
    ChanceOutcome,
    NodeType,
    SolverAdapter,
    State,
)

__all__ = [
    "SolverAdapter",
    "State",
    "NodeType",
    "ActionInstance",
    "ChanceOutcome",
    "GameEngine",
]
