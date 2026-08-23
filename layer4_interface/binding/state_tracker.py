"""StateTracker — cross-frame state tracking for piece order inference.

Single frames cannot determine the placement order of pieces.
``StateTracker`` maintains a history of consecutive frames and infers
the FIFO piece order from transitions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .schemas import Observation


@dataclass
class StateChange:
    """A change between two consecutive frames."""

    added: list[tuple[int, int, str]] = field(default_factory=list)
    removed: list[tuple[int, int]] = field(default_factory=list)


class StateTracker:
    """Track board state across frames and infer piece ordering.

    Usage::

        tracker = StateTracker()
        obs1 = binding.parse_image("frame1.png")
        tracker.update(obs1)
        obs2 = binding.parse_image("frame2.png")
        change = tracker.update(obs2)
        # change.added → [(row, col, player), ...]
        # change.removed → [(row, col), ...]
    """

    def __init__(self):
        self._last_board: list[list[str | None]] | None = None
        self._history: list[Observation] = []
        # ThreadingHTTPServer 下多请求并发访问共享状态（审计 3.6 竞态）。
        self._lock = threading.Lock()

    def update(self, obs: Observation) -> StateChange:
        """Register a new observation and compute the state change.

        Returns a ``StateChange`` describing what changed since the
        previous frame.
        """
        with self._lock:
            change = StateChange()
            current = obs.boardObservation

            if self._last_board is not None:
                for r in range(len(current)):
                    for c in range(len(current[r])):
                        prev = (
                            self._last_board[r][c]
                            if r < len(self._last_board) and c < len(self._last_board[r])
                            else None
                        )
                        curr = current[r][c]
                        if prev != curr:
                            if curr is not None:
                                change.added.append((r, c, curr))
                            if prev is not None:
                                change.removed.append((r, c))

            self._last_board = [row[:] for row in current]
            self._history.append(obs)

            # Keep only last 100 frames
            if len(self._history) > 100:
                self._history.pop(0)

            return change

    def infer_piece_order(self, controlled_player: str = "player_x") -> dict[str, list[dict]]:
        """Infer FIFO piece order from observed frame history.

        Tracks the piece currently occupying each cell across frames: a
        cell that is emptied and later re-occupied (Moon Chess FIFO
        eviction + new placement) starts a fresh entry with the next
        ``placedSeq`` (C-05 — the old "first appearance ever" heuristic
        ignored re-placements and never saw the new piece).

        Returns a dict like ``{'player_x': [{'cellId': str, 'placedSeq': int}, ...]}``.

        注意（审查 Minor 2）：该形状与 ``MoonChessAdapter.load_state()``
        期望的 v4.1 格式（``{pid: [cell_id, ...]}``）不同，且 key 沿用遗留
        命名 ``player_x/player_o``——输出主要用于调试与回放，如需导入引擎
        请先按 load_state 的 v4.1/v5.0 形状转换。
        """
        with self._lock:
            piece_order: dict[str, list[dict]] = {controlled_player: []}
            # cell_id → (symbol, seq) of the piece currently sitting there
            current: dict[str, tuple[str, int]] = {}
            seq = 0
            for obs in list(self._history):
                board = obs.boardObservation
                for r in range(len(board)):
                    for c in range(len(board[r])):
                        cell = board[r][c]
                        cell_id = f"cell_{r}_{c}"
                        if cell is None:
                            # Piece left the cell (moved or FIFO-evicted).
                            current.pop(cell_id, None)
                            continue
                        symbol = str(cell)
                        entry = current.get(cell_id)
                        if entry is None or entry[0] != symbol:
                            # A (new) piece appeared in this cell — record it
                            # as the next placement in FIFO order.
                            seq += 1
                            current[cell_id] = (symbol, seq)
                            piece_order[controlled_player].append(
                                {
                                    "cellId": cell_id,
                                    "placedSeq": seq,
                                }
                            )
            return piece_order

    def reset(self) -> None:
        """Clear all tracked state."""
        with self._lock:
            self._last_board = None
            self._history.clear()
