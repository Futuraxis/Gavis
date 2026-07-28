"""Minimal state representation for the v4.1 game engine.

The game state is a plain dict for fast cloning.  Board cells exist
both as nodes (addressable via ``state['nodes']['cell_x_y']``) and as
a flat ``_board`` array for fast adjacency checks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionInstance:
    """A concrete action generated from an ActionTemplate at runtime."""
    template_id: str
    type: str
    actor_id: str
    params: dict          # param_name → full node/value (not just id)
    canonical_key: str


@dataclass
class ChanceOutcome:
    """A single outcome of a chance node."""
    key: str
    probability: float
    effect_ref: str
    canonical_key: str


# ── State factory & clone ─────────────────────────────────────────

def create_gomoku_state(board_size: int = 9) -> dict:
    """Build the initial game state for a board game.

    The state stores board cells as ``nodes.cell_x_y`` entries *and*
    a flat ``_board`` list for O(1) lookup during win-checking.
    """
    nodes = {}
    for y in range(board_size):
        for x in range(board_size):
            idx = y * board_size + x
            nodes[f'cell_{x}_{y}'] = {
                'id': f'cell_{x}_{y}',
                'type': 'board_cell',
                'props': {'x': x, 'y': y, 'occupant': None, 'idx': idx},
            }

    return {
        'board_size': board_size,
        'nodes': nodes,
        '_board': [None] * (board_size * board_size),
        'env': {
            'phase': 'playing',
            'turn': {
                'currentPlayerId': 'p_black',
                'round': 0,
            },
            'lastPlacedCell': None,
            'lastActor': None,
            'lastAction': None,
            'winner': None,
        },
        '_pending_events': [],
        '_pending_effects': [],
    }


def clone_state(state: dict) -> dict:
    """Fast shallow copy of game state for MCTS simulation.

    ``_board`` is the source of truth — ``nodes`` dict is cleared and
    rebuilt lazily only when needed for display.  Extra env keys
    (like ``pieceOrder``) are forwarded automatically.
    """
    base_env = {
        'phase': state['env']['phase'],
        'turn': dict(state['env']['turn']),
        'lastPlacedCell': state['env']['lastPlacedCell'],
        'lastActor': state['env']['lastActor'],
        'lastAction': state['env']['lastAction'],
        'winner': state['env']['winner'],
    }
    # Forward any extra env keys (e.g. pieceOrder, stepCount, gameId)
    for k, v in state['env'].items():
        if k not in base_env:
            base_env[k] = v
    result = {
        'board_size': state['board_size'],
        'nodes': {},
        '_board': list(state['_board']),
        'env': base_env,
    }
    # Preserve win_length if set
    if '_win_length' in state:
        result['_win_length'] = state['_win_length']
    return result


# ── Cell helpers ──────────────────────────────────────────────────

def cell_index(x: int, y: int, board_size: int) -> int:
    """Convert (x, y) to flat array index."""
    return y * board_size + x


def cell_xy(idx: int, board_size: int) -> tuple[int, int]:
    """Convert flat array index to (x, y)."""
    return idx % board_size, idx // board_size


# ── Win check ─────────────────────────────────────────────────────

def check_five_in_row(state: dict, cell_id: str) -> bool:
    """Check whether placing a stone at ``cell_id`` completes five in a row.

    This function is registered in the Engine's expression evaluator
    and called from rules.json effects.
    """
    # Parse cell_id like "cell_3_5"
    parts = cell_id.split('_')
    if len(parts) < 3:
        return False
    x, y = int(parts[-2]), int(parts[-1])
    bs = state['board_size']
    board = state['_board']
    idx = y * bs + x
    player = board[idx]
    if player is None:
        return False

    win_length = state.get('_win_length', 5)
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dx, dy in directions:
        count = 1
        # positive direction
        px, py = x + dx, y + dy
        while 0 <= px < bs and 0 <= py < bs and board[py * bs + px] == player:
            count += 1
            px += dx
            py += dy
        # negative direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < bs and 0 <= ny < bs and board[ny * bs + nx] == player:
            count += 1
            nx -= dx
            ny -= dy
        if count >= win_length:
            return True
    return False
