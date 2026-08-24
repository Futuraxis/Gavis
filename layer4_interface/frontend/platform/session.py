"""Generic human-vs-AI session management for the platform frontend.

``GameSession`` is game-agnostic: all per-game behaviour (action
parsing, chance resolution, AI turn loop, snapshot serialization) is
provided by the ``GameSpec`` registry in ``games.py``.  Finished
sessions are recorded into ``MatchHistory`` by ``PlayManager``.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ...online_learning.recorder import LearningHooks, TrajectoryRecorder
from ...solver_provider import SolverHandle, SolverProvider
from .games import GAMES, GameSpec, PlayError
from .history import MatchHistory


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class GameSession:
    """One human-vs-AI game, driven by its ``GameSpec``."""

    game_id: str
    spec: GameSpec
    player_pid: str
    difficulty: str
    engine: GameEngine
    solver: SolverHandle
    state: dict = field(init=False)
    last_ai_info: dict = field(default_factory=dict)  # {'move'|'vanish'|'action': ...} per game
    log: list[dict] = field(default_factory=list)  # move entries for history/replay
    started_at: str = field(default_factory=_now_iso)
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Online-learning capture hook (set by ``PlayManager`` when learning
    #: is enabled); records every human/AI decision at adapter level.
    recorder: TrajectoryRecorder | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    @property
    def ai_pid(self) -> str:
        return self.spec.seat_options[1] if self.player_pid == self.spec.seat_options[0] else self.spec.seat_options[0]

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def winner(self) -> str | None:
        return self.state["env"].get("winner")

    @property
    def current_player(self) -> str | None:
        return self.engine.get_current_player(self.state)

    # ── Play ─────────────────────────────────────────────────────

    def step(self, payload: dict) -> None:
        """Validate and apply the human's action, then run the AI reply."""
        if self.over:
            raise PlayError("本局已结束")
        action = self.spec.parse_human_action(self, payload)
        if self.recorder is not None:
            # ``self.state`` is still the pre-move snapshot here — engine
            # applies return new states, so the reference is safe.
            self.recorder.record_human(self, action)
        self.spec.apply_human(self, action)
        self.log.append(self._log_entry("human", action))
        self.spec.run_ai(self, self._record_ai_action)

    def _record_ai_action(self, action: ActionInstance) -> None:
        self.log.append(self._log_entry("ai", action))

    def _log_entry(self, actor: str, action: ActionInstance) -> dict:
        return {
            "step": len(self.log),
            "actor": actor,
            "action": self.spec.describe_action(action),
            "snapshot": self.spec.build_snapshot(self),
        }

    def snapshot(self) -> dict:
        """Public view of the session for the API."""
        return self.spec.build_snapshot(self)


class PlayManager:
    """Registry of active sessions (in-memory, single-process).

    Thread-safe: registry mutations and history writes are guarded by
    one lock; each session additionally owns a per-session lock so two
    concurrent ``move`` calls cannot interleave on the same game.
    """

    def __init__(
        self,
        provider: SolverProvider,
        history: MatchHistory | None = None,
        seed: int = 42,
        max_sessions: int = 128,
        learning: LearningHooks | None = None,
    ) -> None:
        self._provider = provider
        self._history = history
        self._seed = seed
        self._max_sessions = max_sessions
        self._learning = learning
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()

    def start(self, game_id: str, player_pid: str, difficulty: str, player_count: int = 2) -> GameSession:
        """Create a new session; resolves start chance nodes and lets the AI open."""
        spec = GAMES.get(game_id)
        if spec is None:
            raise PlayError(f"未知游戏: {game_id}")
        if player_count not in spec.player_counts:
            raise PlayError(f"该游戏不支持 {player_count} 人")
        if player_pid == "random":
            player_pid = spec.seat_options[0] if uuid.uuid4().int % 2 == 0 else spec.seat_options[1]
        if player_pid not in spec.seat_options:
            raise PlayError(f"未知{spec.seat_label}: {player_pid}")
        if difficulty not in spec.difficulty_budgets:
            raise PlayError(f"未知难度: {difficulty}")

        session_id = uuid.uuid4().hex[:8]
        if spec.player_counts != (2,):
            engine = spec.create_engine(self._seed, player_count=player_count)
        else:
            engine = spec.create_engine(self._seed)
        solver = spec.create_solver(self._provider, engine, self._seed, spec.difficulty_budgets[difficulty])
        session = GameSession(
            game_id=session_id,
            spec=spec,
            player_pid=player_pid,
            difficulty=difficulty,
            engine=engine,
            solver=solver,
        )
        if self._learning is not None:
            # Wrap the solver in a recording handle and attach a
            # per-match recorder; every AI decision (incl. multi-action
            # loops) is captured with zero GameSpec changes.
            session.solver = self._learning.wrap_handle(session, session.solver)
        spec.resolve_start(session)
        if spec.ai_opens(session):
            spec.run_ai(session, session._record_ai_action)
        with self._lock:
            self._evict_locked()
            self._sessions[session_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f"未知对局: {game_id}")
        return session

    def move(self, game_id: str, payload: dict) -> dict:
        """Apply a human move, run the AI reply, and return the snapshot.

        When the game ends, the session is recorded into history and
        removed from the registry.
        """
        session = self.get(game_id)
        with session.lock:
            session.step(payload)
            snapshot = session.snapshot()
            if session.over:
                if self._history is not None:
                    self._history.record(self._build_record(session))
                if self._learning is not None:
                    self._learning.on_finished(session)
                self.remove(game_id)
        return snapshot

    def remove(self, game_id: str) -> None:
        with self._lock:
            self._sessions.pop(game_id, None)

    # ── Internals ────────────────────────────────────────────────

    def _build_record(self, session: GameSession) -> dict:
        """Assemble the persisted match record (see history module)."""
        return {
            "match_id": session.game_id,
            "game_id": session.spec.game_id,
            "player_pid": session.player_pid,
            "ai_pid": session.ai_pid,
            "difficulty": session.difficulty,
            "seed": self._seed,
            "started_at": session.started_at,
            "finished_at": _now_iso(),
            "winner": session.winner,
            "over": True,
            "moves": session.log,
        }

    def _evict_locked(self) -> None:
        """Bound the registry by dropping the oldest unfinished session (FIFO)."""
        while len(self._sessions) >= self._max_sessions:
            _, oldest = next(iter(self._sessions.items()))
            self._sessions.pop(oldest.game_id, None)
