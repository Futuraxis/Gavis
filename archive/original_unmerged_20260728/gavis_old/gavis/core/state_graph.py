"""Minimal state representation for the v4.1 game engine.

The game state is a plain dict for fast cloning.  Board cells exist
both as nodes (addressable via `state['nodes']['cell_x_y']`) and as
a flat `_board` array for fast adjacency checks.
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


# ------------------------------------------------------------------
# State factory & clone
# ------------------------------------------------------------------

def create_gomoku_state(board_size: int = 9) -> dict:
    """Build the initial game state.

    The state stores board cells as `nodes.cell_x_y` entries *and*
    a flat `_board` list for O(1) lookup during win-checking.
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

    `_board` is the source of truth — `nodes` dict is cleared and
    rebuilt lazily only when needed for display.
    """
    return {
        'board_size': state['board_size'],
        'nodes': {},  # cleared — rebuilt lazily by engine when needed for display
        '_board': state['_board'].copy(),
        'env': deepcopy(state['env']),
        '_pending_events': [],
        '_pending_effects': [],
    }


def _clone_node(node: dict) -> dict:
    return {
        'id': node['id'],
        'type': node['type'],
        'props': dict(node['props']),
    }


# ------------------------------------------------------------------
# Board helpers
# ------------------------------------------------------------------

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def cell_index(x: int, y: int, board_size: int) -> int:
    return y * board_size + x


def cell_xy(index: int, board_size: int) -> tuple[int, int]:
    return index % board_size, index // board_size


def check_five_in_row(state: dict, last_cell_id: str) -> bool:
    """Check if the last-placed stone completed five-in-a-row."""
    bs = state['board_size']
    board = state['_board']

    # last_cell_id is like "cell_3_5"
    # Robust parsing: extract the last two underscore-separated segments
    parts = last_cell_id.split('_')
    try:
        x, y = int(parts[-2]), int(parts[-1])
    except (IndexError, ValueError):
        return False

    idx = cell_index(x, y, bs)
    color = board[idx]
    if color is None:
        return False

    for dx, dy in DIRECTIONS:
        count = 1
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < bs and 0 <= ny < bs and board[cell_index(nx, ny, bs)] == color:
                count += 1
            else:
                break
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < bs and 0 <= ny < bs and board[cell_index(nx, ny, bs)] == color:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False
