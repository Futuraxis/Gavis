"""Layer 2: Engine — declarative game engine (v5.2, adapter-free).

Loads ``rules.json`` and provides the full game runtime that all
Layer 3 solvers consume through the generic ``GameEngine`` protocol.
No per-game adapter classes exist below the rules JSON / the frontend.
"""

from __future__ import annotations

from .core.engine import GameEngine
from .core.state_graph import ActionInstance, ChanceOutcome, NodeType, State

__all__ = [
    "State",
    "NodeType",
    "ActionInstance",
    "ChanceOutcome",
    "GameEngine",
]
