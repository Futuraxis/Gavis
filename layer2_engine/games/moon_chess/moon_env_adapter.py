"""Moon Chess — thin GameEngine adapter for Moon Chess.

The FIFO eviction logic now lives entirely in ``rules.json`` via
``listAppend`` and ``trimQueue`` operations.

This adapter only adds:
  1. ``pieceOrder`` initialization in ``create_initial_state()``
  2. RL-friendly methods (``get_feature_vector``, ``get_action_mask``)
  3. ``load_state()`` for importing from VisionBridge
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

from ...core.engine import GameEngine
from ...core.state_graph import clone_state, create_gomoku_state
from ...interfaces.solver_adapter import State


class MoonChessAdapter(GameEngine):
    """GameEngine subclass adding RL-friendly methods for Moon Chess."""

    BOARD_SIZE = 3
    MAX_PIECES = 3
    ACTION_DIM = 9
    FEATURE_DIM = 38

    def __init__(self, seed: Optional[int] = None):
        rules_path = Path(__file__).resolve().parent.parent.parent.parent / 'rules' / 'moon_chess.json'
        import json
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        super().__init__(rules, seed=seed)

    def create_initial_state(self) -> dict:
        state = create_gomoku_state(self.BOARD_SIZE)
        state['env']['pieceOrder'] = {'p_black': [], 'p_white': []}
        state['env']['stepCount'] = 0
        state['env']['gameId'] = 'moon_chess'
        state['_win_length'] = 3
        return state

    def get_observation(self, state: dict, player_id: str = 'p_black') -> dict:
        board_copy = [
            [state['_board'][r * self.BOARD_SIZE + c] for c in range(self.BOARD_SIZE)]
            for r in range(self.BOARD_SIZE)
        ]
        return {
            'board': board_copy,
            'board_size': self.BOARD_SIZE,
            'pieceOrder': deepcopy(state['env'].get('pieceOrder', {})),
            'current_player': state['env']['turn']['currentPlayerId'],
            'stepCount': state['env'].get('stepCount', 0),
            'phase': state['env']['phase'],
        }

    def load_state(self, state: dict) -> dict:
        bs = state.get('board_size', self.BOARD_SIZE)
        result = create_gomoku_state(bs)
        if 'env' in state:
            result['env'].update(state['env'])
        result['env'].setdefault('pieceOrder', {'p_black': [], 'p_white': []})
        result['env'].setdefault('stepCount', 0)
        result['env'].setdefault('gameId', 'moon_chess')
        if '_board' in state:
            result['_board'] = list(state['_board'])
            for idx, occ in enumerate(result['_board']):
                x, y = idx % bs, idx // bs
                result['nodes'][f'cell_{x}_{y}'] = {
                    'id': f'cell_{x}_{y}', 'type': 'board_cell',
                    'props': {'x': x, 'y': y, 'occupant': occ, 'idx': idx},
                }
        result['_win_length'] = 3
        return result

    def get_feature_vector(self, state: dict, perspective_player_id: str) -> np.ndarray:
        obs = self.get_observation(state, perspective_player_id)
        board = obs['board']
        po = obs['pieceOrder']
        cp = obs['current_player']
        sc = obs['stepCount']

        feats: list[float] = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                cell = board[r][c]
                if cell is None:
                    feats.extend([1.0, 0.0, 0.0])
                elif cell == perspective_player_id:
                    feats.extend([0.0, 1.0, 0.0])
                else:
                    feats.extend([0.0, 0.0, 1.0])

        age_map: dict[str, int] = {}
        for entries in po.values():
            for age, entry in enumerate(entries, 1):
                cid = entry if isinstance(entry, str) else entry.get('cellId', '')
                if cid:
                    age_map[cid] = age
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                feats.append(float(age_map.get(f"cell_{c}_{r}", 0)))

        feats.append(1.0 if cp == perspective_player_id else 0.0)
        feats.append(min(1.0, sc / 32.0))
        return np.asarray(feats, dtype=np.float32)

    def get_action_mask(self, state: dict) -> np.ndarray:
        mask = np.zeros(self.ACTION_DIM, dtype=np.float32)
        if self.is_terminal(state):
            return mask
        for a in self.get_legal_actions(state):
            cell = a.params.get('cell', {})
            cid = cell.get('id', '') if isinstance(cell, dict) else str(cell)
            try:
                _, r, c = cid.split('_')
                idx = int(r) * 3 + int(c)
                if 0 <= idx < self.ACTION_DIM:
                    mask[idx] = 1.0
            except (ValueError, IndexError):
                pass
        return mask

    def apply_action(self, state: dict, action) -> dict:
        new_state = super().apply_action(state, action)
        new_state['env']['stepCount'] = state['env'].get('stepCount', 0) + 1
        return new_state
