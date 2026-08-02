"""Moon Chess — thin GameEngine adapter for Moon Chess (v5.0).

The game logic lives entirely in ``rules/moon_chess.json`` via effectors.
This adapter only adds:
  1. RL-friendly methods (``get_feature_vector``, ``get_action_mask``)
  2. ``load_state()`` for importing from VisionBridge
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ...core.engine import GameEngine
from ...core.state_graph import DerivedViewEngine


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

    def get_observation(self, state: dict, player_id: str = 'p_black') -> dict:
        """Return an observation dict suitable for RL/encoding.

        Instead of raw ground state, return a structured observation with
        board as 2D list, piece order, and metadata.
        """
        board = state['_arrays'].get('board', [None] * 9)
        board_2d = [
            [board[r * self.BOARD_SIZE + c] for c in range(self.BOARD_SIZE)]
            for r in range(self.BOARD_SIZE)
        ]
        env = state.get('env', {})

        # Build piece order as dict keyed by player for RL
        raw_po = state['_arrays'].get('pieceOrder', [])
        piece_order: dict[str, list] = {'p_black': [], 'p_white': []}
        for entry in raw_po:
            pid = entry.get('player_id', '')
            if pid in piece_order:
                piece_order[pid].append(entry.get('cell_id', ''))

        return {
            'board': board_2d,
            'board_size': self.BOARD_SIZE,
            'pieceOrder': piece_order,
            'current_player': env.get('turn', 'p_black'),
            'stepCount': env.get('round', 0),
            'phase': env.get('phase', 'playing'),
        }

    def load_state(self, state: dict) -> dict:
        """Import an externally constructed state (e.g. from VisionBridge).

        Accepts both v4.1-style flat dict and v5.0 ground state.
        """
        result = self.create_initial_state()

        # v5.0 format: _arrays.board
        ext_arrays = state.get('_arrays', {})
        ext_board = ext_arrays.get('board')
        if ext_board is not None:
            result['_arrays']['board'] = list(ext_board)
        else:
            # v4.1 fallback: _board or board field
            board = state.get('_board') or state.get('board')
            if board is not None:
                result['_arrays']['board'] = list(board)

        # Merge env
        ext_env = state.get('env', {})
        if ext_env:
            result['env'].update(ext_env)
            if 'turn' in ext_env:
                result['env']['turn'] = ext_env.get('turn', 'p_black')

        # Piece order (migrate from v4.1 dict format if needed)
        ext_po = ext_arrays.get('pieceOrder')
        if ext_po is not None:
            result['_arrays']['pieceOrder'] = list(ext_po)
        else:
            po = state.get('pieceOrder')
            if isinstance(po, dict):
                # Convert v4.1 format: {'p_black': ['cell_0_0', ...]} → [{'cell_id': ..., 'player_id': ...}]
                flat = []
                for pid, cells in po.items():
                    for cell_id in cells:
                        flat.append({'cell_id': cell_id, 'player_id': pid})
                result['_arrays']['pieceOrder'] = flat

        return result

    def get_feature_vector(self, state: dict, perspective_player_id: str) -> np.ndarray:
        """Build 38-dim feature vector for PPO.

        Feature layout:
          0-26:    9 cells × 3 one-hot (empty / self / opponent)
          27-35:   9 cells × age encoding (1=latest … 3=oldest)
          36:      whose turn (1 = perspective player)
          37:      normalized step count
        """
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

        # Age encoding from piece order
        age_map: dict[str, int] = {}
        for pid, cells in po.items():
            for age, cid in enumerate(cells, 1):
                if cid:
                    age_map[cid] = age
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                feats.append(float(age_map.get(f"cell_{r}_{c}", 0)))

        feats.append(1.0 if cp == perspective_player_id else 0.0)
        feats.append(min(1.0, sc / 32.0))
        return np.asarray(feats, dtype=np.float32)

    def get_action_mask(self, state: dict) -> np.ndarray:
        """Build 9-dim binary action mask (1 = legal move)."""
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
        """Apply action and increment step count."""
        new_state = super().apply_action(state, action)
        new_state['env']['round'] = state['env'].get('round', 0) + 1
        return new_state
