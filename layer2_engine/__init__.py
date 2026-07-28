"""Layer 2: Env/Engine — declarative game engine.

Loads ``rules.json`` and provides a full game runtime that all
Layer 3 solvers consume via the ``SolverAdapter`` Protocol.
"""

from .interfaces.solver_adapter import (
    SolverAdapter,
    State,
    NodeType,
    ActionInstance,
    ChanceOutcome,
)
from .core.engine import GameEngine

__all__ = [
    "SolverAdapter",
    "State",
    "NodeType",
    "ActionInstance",
    "ChanceOutcome",
    "GameEngine",
]
