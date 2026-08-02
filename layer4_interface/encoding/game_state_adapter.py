"""Unified GameState field accessor.

Supports both v4.1 (flat dict) and v5.0 (ground arrays + env) formats.
"""

from __future__ import annotations


class GameStateAdapter:
    """Centralizes access to GameState fields for both PPO and Binding."""

    def get_board(self, state: dict) -> list:
        """Get board array from state (v5.0 _arrays.board, or v4.1 _board)."""
        arrays = state.get('_arrays', {})
        board = arrays.get('board')
        if board is not None:
            return board
        return state.get('_board', state.get('board', []))

    def get_current_player(self, state: dict) -> str | None:
        """Get current player from state."""
        env = state.get('env', {})
        turn = env.get('turn')
        if turn is not None:
            return turn
        # v4.1 format
        turn_info = env.get('turn')  # might be dict
        if isinstance(turn_info, dict):
            return turn_info.get('currentPlayerId')
        return state.get('currentPlayerId')

    def get_piece_order(self, state: dict) -> list[dict] | None:
        """Get piece order records (v5.0 _arrays.pieceOrder, or v4.1 pieceOrder)."""
        arrays = state.get('_arrays', {})
        po = arrays.get('pieceOrder')
        if po is not None:
            return po
        return state.get('pieceOrder', {})

    def get_legal_actions(self, state: dict) -> list:
        """Get legal actions list."""
        return state.get('legalActions', [])

    def get_step_count(self, state: dict) -> int:
        """Get step/round count."""
        env = state.get('env', {})
        round_val = env.get('round', 0)
        if round_val:
            return int(round_val)
        return int(state.get('stepCount', 0))

    def is_terminal(self, state: dict) -> bool:
        """Check if state is terminal."""
        env = state.get('env', {})
        if env.get('phase') == 'game_over':
            return True
        if env.get('winner') is not None:
            return True
        status = state.get('status')
        return bool(status and status != 'running')
