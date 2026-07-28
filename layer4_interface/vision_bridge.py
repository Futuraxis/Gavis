"""VisionBridge — converts visual Observation to Engine State.

Layer 4 (Interface) → Layer 2 (Engine) translation only.
Solver integration is handled at the application/demo layer.
"""

from __future__ import annotations

from layer2_engine.interfaces.solver_adapter import SolverAdapter, State
from .binding.schemas import Observation


def observation_to_state(
    observation: Observation,
    engine: SolverAdapter,
) -> State:
    """Convert a visual ``Observation`` to an Engine-compatible ``State``.

    This is the core translation function of Layer 4 → Layer 2.
    """
    board_grid = observation.boardObservation
    bs = len(board_grid)

    _board: list[str | None] = []
    for row in board_grid:
        for cell in row:
            if cell is None:
                _board.append(None)
            elif cell in ("X", "●", "x"):
                _board.append("p_black")
            elif cell in ("O", "○", "o"):
                _board.append("p_white")
            else:
                _board.append(None)

    state = {
        'board_size': bs,
        '_board': _board,
        'env': {
            'phase': 'playing',
            'turn': {'currentPlayerId': 'p_black', 'round': 0},
            'winner': None,
            'lastPlacedCell': None,
            'lastActor': None,
            'lastAction': None,
            'stepCount': 0,
            'pieceOrder': {'p_black': [], 'p_white': []},
        },
    }

    return engine.load_state(state)
