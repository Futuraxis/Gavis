"""Trajectory capture for online learning (Layer 4).

Captures every decision — human and AI — of a live game session at
adapter level, so accumulated experiences can later drive solver
updates.  Everything here stays inside Layer 4: no ``layer3_solvers``
import, no layer violation (grep the package for layer3_solvers finds
nothing by construction).

Two capture points:

  - Human decisions: ``GameSession.step`` records the pre-move state,
    the parsed action, the info-set key and the legal action set right
    after ``parse_human_action`` returns and before the action is
    applied (the engine's states are immutable snapshots, so
    ``session.state`` still refers to the pre-move state at that point).
  - AI decisions: ``RecordingHandle`` wraps the session's
    ``SolverHandle``; the per-game ``run_ai`` closures call
    ``solver.select_action(state)`` unchanged, so every AI decision —
    including multi-action loops like Texas Hold'em or multi-seat
    mahjong — is recorded with zero ``GameSpec`` changes.

Records are buffered per match and written to the store atomically by
``TrajectoryRecorder.finish`` together with a terminal record carrying
each seat's final utility.  Matches that are abandoned (registry
eviction, server stop) never reach the store — no partial matches.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from layer2_engine.core.state_graph import ActionInstance

if TYPE_CHECKING:
    from layer2_engine.core.engine import GameEngine
    from layer4_interface.frontend.platform.session import GameSession

    from .store import LearningStore


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def jsonable(value: Any, _depth: int = 0) -> Any:
    """Convert arbitrary engine data into JSON-safe structures.

    Engine states are plain dicts of scalars/lists, but adapters may
    embed exotic values; anything unknown falls back to ``str()`` so a
    capture never fails mid-match.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, _depth + 1) for v in value]
    # numpy scalars/arrays and other opaque types — best-effort text.
    return str(value)


class LearningHooks(Protocol):
    """The minimal surface ``PlayManager`` needs from the learning manager.

    Implemented by :class:`~layer4_interface.online_learning.manager.LearningManager`;
    kept as a protocol here so ``PlayManager`` stays decoupled and the
    hook-in point is testable with a stub.
    """

    def wrap_handle(self, session: Any, solver: Any) -> Any:
        """Return a solver handle that records AI decisions for ``session``."""
        ...

    def on_finished(self, session: Any) -> None:
        """Called by ``PlayManager`` once a match has reached a terminal state."""
        ...


@dataclass
class TrajectoryRecorder:
    """Buffers one match's decisions and persists them on finish."""

    store: LearningStore
    game_id: str
    match_id: str
    started_at: str
    pending: list[dict] = field(default_factory=list)

    # ── Capture ────────────────────────────────────────────────────

    def record_human(self, session: GameSession, action: ActionInstance) -> None:
        """Record the human's decision (state is still the pre-move one)."""
        self.pending.append(self._decision(session, "human", session.player_pid, session.state, action))

    def record_ai(self, session: GameSession, state: dict, action: ActionInstance | None) -> None:
        """Record one AI decision (called by the recording handle)."""
        if action is None:
            return
        player = session.engine.get_current_player(state) or session.ai_pid
        self.pending.append(self._decision(session, "ai", player, state, action))

    def finish(self, session: GameSession) -> None:
        """Persist the buffered decisions plus a terminal record with utilities."""
        terminal = self._terminal(session)
        self.store.append_match(self.game_id, self.pending, terminal)
        self.pending.clear()

    # ── Internals ──────────────────────────────────────────────────

    def _decision(self, session: GameSession, actor: str, player: str, state: dict, action: ActionInstance) -> dict:
        engine: GameEngine = session.engine
        info_key = self._info_key(engine, state, player)
        legal = [a.canonical_key for a in engine.get_legal_actions(state)]
        # 隐私红线（审计 B7）：不落盘 god-view 全量状态（含对手底牌/手牌的
        # ``_arrays``）；只存决策者自己的信息集投影——对手建模（empirical
        # table）与信号转换（signals）消费的正是这一层，字段名保持 ``state``。
        observation = engine.project_observation(state, player)
        return {
            "match_id": self.match_id,
            "game_id": self.game_id,
            "step": len(self.pending) + 1,
            "actor": actor,
            "player": player,
            "state": jsonable(observation),
            "action": jsonable(asdict(action)),
            "info_key": info_key,
            "legal": legal,
        }

    def _terminal(self, session: GameSession) -> dict:
        engine: GameEngine = session.engine
        utilities: dict[str, float] = {}
        for pid in session.spec.seat_options:
            utilities[pid] = float(engine.get_utility(session.state, pid))
        return {
            "match_id": self.match_id,
            "game_id": self.game_id,
            "terminal": True,
            "winner": session.winner,
            "utilities": utilities,
            "human_pid": session.player_pid,
            "ai_pid": session.ai_pid,
            "difficulty": session.difficulty,
            "started_at": session.started_at,
            "finished_at": _now_iso(),
            "decisions": self.pending,
        }

    @staticmethod
    def _info_key(engine: GameEngine, state: dict, player: str) -> str | None:
        if not hasattr(engine, "get_info_set_key"):
            return None
        try:
            return engine.get_info_set_key(state, player)
        except (KeyError, TypeError, ValueError):
            return None


class RecordingHandle:
    """``SolverHandle`` wrapper that records every ``select_action`` call.

    Implements the same four members as :class:`~layer4_interface.solver_provider.SolverHandle`
    (``name`` / ``select_action`` / ``solve`` / ``train``) and delegates
    everything except decision recording to the inner handle, so the
    per-game ``run_ai`` closures keep working untouched.
    """

    def __init__(self, inner: Any, recorder: TrajectoryRecorder, session: Any) -> None:
        self._inner = inner
        self._recorder = recorder
        self._session = session

    @property
    def name(self) -> str:
        return self._inner.name

    def select_action(self, state: dict) -> Any | None:
        action = self._inner.select_action(state)
        self._recorder.record_ai(self._session, state, action)
        return action

    def solve(self, state: dict, **kwargs: Any) -> Any | None:
        return self._inner.solve(state, **kwargs)

    def train(self, episodes: int, **kwargs: Any) -> Any:
        return self._inner.train(episodes, **kwargs)


def records_to_json(records: list[dict]) -> str:
    """Serialize one match's records to a JSONL block (store helper)."""
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
