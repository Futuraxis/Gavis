"""StateTracker — cross-frame state tracking for piece order inference.

Single frames cannot determine the placement order of pieces.
``StateTracker`` maintains a history of consecutive frames and infers
the FIFO piece order from transitions.
"""

from __future__ import annotations

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
        self._last_seq: int = -1
        self._history: list[Observation] = []

    def update(self, obs: Observation) -> StateChange:
        """Register a new observation and compute the state change.

        Returns a ``StateChange`` describing what changed since the
        previous frame.
        """
        change = StateChange()
        current = obs.boardObservation

        if self._last_board is not None:
            for r in range(len(current)):
                for c in range(len(current[r])):
                    prev = self._last_board[r][c] if r < len(self._last_board) and c < len(self._last_board[r]) else None
                    curr = current[r][c]
                    if prev != curr:
                        if curr is not None:
                            change.added.append((r, c, curr))
                        if prev is not None:
                            change.removed.append((r, c))

        self._last_board = [row[:] for row in current]
        self._last_seq = obs.frameSeq
        self._history.append(obs)

        # Keep only last 100 frames
        if len(self._history) > 100:
            self._history.pop(0)

        return change

    def infer_piece_order(self, controlled_player: str = "player_x") -> dict[str, list[dict]]:
        """Infer FIFO piece order from observed frame history.

        Returns a dict like ``{'player_x': [{'cellId': str, 'placedSeq': int}, ...]}``
        usable by ``MoonChessAdapter.load_state()``.
        """
        piece_order: dict[str, list[dict]] = {controlled_player: []}
        # Simple heuristic: pieces appear in order of first observation
        seen: set[str] = set()
        seq = 0
        for obs in self._history:
            board = obs.boardObservation
            for r in range(len(board)):
                for c in range(len(board[r])):
                    cell = board[r][c]
                    if cell is not None:
                        cell_id = f"cell_{r}_{c}"
                        if cell_id not in seen:
                            seen.add(cell_id)
                            seq += 1
                            piece_order[controlled_player].append({
                                "cellId": cell_id,
                                "placedSeq": seq,
                            })
        return piece_order

    def reset(self) -> None:
        """Clear all tracked state."""
        self._last_board = None
        self._last_seq = -1
        self._history.clear()
