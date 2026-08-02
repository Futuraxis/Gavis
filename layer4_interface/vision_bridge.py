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
    Builds a ground state in v5.0 format and passes it to ``engine.load_state()``.
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

    state: State = {
        '_arrays': {
            'board': _board,
        },
        'env': {
            'turn': 'p_black',
            'round': 0,
            'phase': 'playing',
            'winner': None,
            'lastPlacedCell': None,
            'lastActor': None,
        },
        '_players': [{'id': 'p_black'}, {'id': 'p_white'}],
        '_constants': {'board_size': bs},
        '_schema': {},
        '_pending_events': [],
        '_pending_effects': [],
    }

    return engine.load_state(state)
