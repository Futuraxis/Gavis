"""Unified GameState field accessor."""

from __future__ import annotations


class GameStateAdapter:
    """Centralizes access to GameState fields for both PPO and Binding."""

    def get_board(self, state: dict) -> list[list[str | None]]:
        return state.get("board", state.get("_board", []))

    def get_current_player(self, state: dict) -> str:
        return state.get("currentPlayerId", "player_x")

    def get_piece_order(self, state: dict) -> dict[str, list[dict]]:
        return state.get("pieceOrder", {})

    def get_legal_actions(self, state: dict) -> list[str]:
        return state.get("legalActions", [])

    def get_step_count(self, state: dict) -> int:
        return int(state.get("stepCount", 0))

    def is_terminal(self, state: dict) -> bool:
        status = state.get("status")
        return bool(status and status != "running")
