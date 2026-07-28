"""仅用于测试 Binding / Encoder / PPO 的最小月亮棋环境。"""

from __future__ import annotations

import random
from dataclasses import dataclass


PLAYER_TO_SYMBOL = {"player_x": "X", "player_o": "O"}


@dataclass(slots=True)
class RandomAgent:
    def act(self, state: dict) -> dict:
        actor_id = state["currentPlayerId"]
        target = random.choice(state["legalActions"])
        return {
            "actorId": actor_id,
            "actionType": "place_piece",
            "parameters": {"targetCellId": target},
        }


class MockMoonEnv:
    """最小规则实现，仅服务本模块联调，不替代正式游戏引擎。"""

    def __init__(self, max_steps: int = 32) -> None:
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> dict:
        self.game_id = "moon_demo_001"
        self.board = [[None, None, None] for _ in range(3)]
        self.current_player = "player_x"
        self.winner_id: str | None = None
        self.status = "running"
        self.step_count = 0
        self.global_seq = 0
        self.piece_order = {"player_x": [], "player_o": []}
        self.player_symbols = {"X": "player_x", "O": "player_o"}
        self._refresh_legal_actions()
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "gameId": self.game_id,
            "seq": self.global_seq,
            "currentPlayerId": self.current_player,
            "board": [row[:] for row in self.board],
            "pieceOrder": {
                player: [item.copy() for item in entries] for player, entries in self.piece_order.items()
            },
            "legalActions": self.legal_actions[:],
            "stepCount": self.step_count,
            "status": self.status,
            "winnerId": self.winner_id,
            "playerSymbols": self.player_symbols.copy(),
        }

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        if self.is_terminal():
            raise ValueError("终局后不能继续 step。")
        actor_id = action["actorId"]
        if actor_id != self.current_player:
            raise ValueError("当前不是该玩家回合。")
        target_cell = action["parameters"]["targetCellId"]
        if target_cell not in self.legal_actions:
            raise ValueError(f"非法动作: {target_cell}")

        row, col = _parse_cell_id(target_cell)
        symbol = PLAYER_TO_SYMBOL[actor_id]
        self.global_seq += 1
        self.board[row][col] = symbol
        self.piece_order[actor_id].append({"cellId": target_cell, "placedSeq": self.global_seq})
        if len(self.piece_order[actor_id]) > 3:
            removed = self.piece_order[actor_id].pop(0)
            remove_row, remove_col = _parse_cell_id(removed["cellId"])
            self.board[remove_row][remove_col] = None

        self.step_count += 1
        self.winner_id = self._check_winner()
        if self.winner_id is not None:
            self.status = "finished"
        elif self.step_count >= self.max_steps:
            self.status = "draw"
        self.current_player = "player_o" if self.current_player == "player_x" else "player_x"
        self._refresh_legal_actions()
        return self.get_state(), 0.0, self.is_terminal(), {}

    def is_terminal(self) -> bool:
        return self.status != "running"

    def get_winner_id(self) -> str | None:
        return self.winner_id

    def _refresh_legal_actions(self) -> None:
        if self.is_terminal():
            self.legal_actions = []
            return
        legal_actions: list[str] = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    legal_actions.append(f"cell_{row}_{col}")
        self.legal_actions = legal_actions

    def _check_winner(self) -> str | None:
        lines = []
        lines.extend(self.board)
        lines.extend([[self.board[row][col] for row in range(3)] for col in range(3)])
        lines.append([self.board[0][0], self.board[1][1], self.board[2][2]])
        lines.append([self.board[0][2], self.board[1][1], self.board[2][0]])
        for line in lines:
            if line[0] is not None and line[0] == line[1] == line[2]:
                return self.player_symbols[line[0]]
        return None


def _parse_cell_id(cell_id: str) -> tuple[int, int]:
    _, row, col = cell_id.split("_")
    return int(row), int(col)
