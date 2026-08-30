"""LearningManager — capture hooks + online-learning apply pipeline (Layer 4).

Capture side implements the :class:`LearningHooks` protocol consumed by
``PlayManager`` (``wrap_handle`` / ``on_finished``), creating one
:class:`TrajectoryRecorder` per match.

Apply side turns recorded matches into an empirical opponent table
(human decisions per info set), gates the candidate against the current
model with a short fixed-seed match (seats swapped), and publishes it on
pass — keeping the previous version on regression.  The whole pipeline
talks to solvers only through the ``SolverProvider`` protocol and to
games only through the Layer-2 adapters, so no ``layer3_solvers`` import
appears anywhere in this package.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Any

from layer4_interface.frontend.platform.games import GAMES

from ..solver_provider import SolverHandle, SolverProvider
from .models import OnlineModelStore
from .recorder import RecordingHandle, TrajectoryRecorder
from .store import LearningStore, LearningStoreError


@dataclass
class ApplyResult:
    """Outcome of one ``apply(game_id)`` run, serialized to the API."""

    game_id: str
    applied: bool
    reason: str  # ok | insufficient | unchanged | rejected | disabled | error
    version: int | None = None
    gate: dict | None = None
    samples: int = 0
    coverage: int = 0
    error: str | None = None


class LearningManager:
    """Coordinates capture, persistence and the apply pipeline."""

    def __init__(
        self,
        *,
        store: LearningStore,
        model_store: OnlineModelStore,
        provider: SolverProvider,
        seed: int = 42,
        min_samples: int = 30,
        gate_episodes: int = 20,
        gate_budget: int = 300,
        gate_tolerance: float = 0.03,
        default_enabled: tuple[str, ...] = ("texas_holdem",),
    ) -> None:
        self._store = store
        self._model_store = model_store
        self._provider = provider
        self._seed = seed
        self._min_samples = min_samples
        self._gate_episodes = gate_episodes
        self._gate_budget = gate_budget
        self._gate_tolerance = gate_tolerance
        self._enabled: dict[str, bool] = {}
        for game_id in default_enabled:
            self._enabled[game_id] = True
        self._auto_stop = threading.Event()
        self._auto_thread: threading.Thread | None = None

    # ── LearningHooks (consumed by PlayManager) ────────────────────

    def wrap_handle(self, session: Any, solver: SolverHandle) -> SolverHandle:
        """Create a per-match recorder and wrap the solver handle."""
        recorder = TrajectoryRecorder(
            store=self._store,
            game_id=session.spec.game_id,
            match_id=session.game_id,
            started_at=session.started_at,
        )
        session.recorder = recorder
        return RecordingHandle(solver, recorder, session)

    def on_finished(self, session: Any) -> None:
        """Persist the finished match's decisions + terminal record."""
        if session.recorder is None:
            # 开局时学习未启用的会话没有装配录制器（B5 门控按开局判定），
            # 整局从未采集——没有轨迹可落盘，静默返回。回归：旧实现在此
            # 无条件调 ``session.recorder.finish``，未启用学习的游戏（默认
            # 配置下的月亮棋）终局最后一手直接 500（NoneType.finish）。
            return
        session.recorder.finish(session)

    # ── Config ────────────────────────────────────────────────────

    def enabled(self, game_id: str) -> bool:
        return bool(self._enabled.get(game_id, False))

    def set_enabled(self, game_id: str, flag: bool) -> None:
        self._enabled[game_id] = bool(flag)

    # ── Apply pipeline ────────────────────────────────────────────

    def build_empirical_table(self, game_id: str) -> tuple[dict[str, dict[str, int]], int, int]:
        """Aggregate human decisions per info set into an empirical table.

        Returns ``(table, samples, coverage)``; only HUMAN-actor
        decisions are counted (the opponent in a human-vs-AI match).
        """
        counts: dict[str, dict[str, int]] = {}
        samples = 0
        for match in self._store.read_matches(game_id):
            for decision in match.get("decisions", []):
                if decision.get("actor") != "human":
                    continue
                info_key = decision.get("info_key")
                action_key = (decision.get("action") or {}).get("canonical_key")
                if info_key and action_key:
                    bucket = counts.setdefault(info_key, {})
                    bucket[action_key] = bucket.get(action_key, 0) + 1
                    samples += 1
        return counts, samples, len(counts)

    def gate(self, game_id: str, candidate: dict, baseline: dict | None) -> dict:
        """Short fixed-seed match: candidate vs baseline, seats swapped.

        Both sides are hybrid solvers in opponent-model search mode; the
        baseline is the current published table (or None → uniform).
        """
        spec = GAMES[game_id]
        seats = spec.seat_options[:2]
        c_wins = b_wins = draws = 0
        for ep in range(self._gate_episodes):
            c_first = ep % 2 == 0
            seed_c = self._seed + ep * 2
            seed_b = self._seed + ep * 2 + 1
            engine_c = spec.create_engine(seed_c)
            engine_b = spec.create_engine(seed_b)
            solver_c = self._provider.create_solver(
                game_id, "hybrid", engine_c, seed_c, self._gate_budget, empirical_table=candidate
            )
            solver_b = self._provider.create_solver(
                game_id, "hybrid", engine_b, seed_b, self._gate_budget, empirical_table=baseline
            )
            seat_c = seats[0] if c_first else seats[1]
            seat_b = seats[1] if c_first else seats[0]
            winner_tag, _moves = self._play_one(engine_c, engine_b, solver_c, solver_b, seat_c, seat_b, self._seed + ep)
            if winner_tag == "c":
                c_wins += 1
            elif winner_tag == "b":
                b_wins += 1
            else:
                draws += 1
        episodes = self._gate_episodes
        valid = max(1, episodes - 0)
        return {
            "episodes": episodes,
            "candidate_wins": c_wins,
            "baseline_wins": b_wins,
            "draws": draws,
            "candidate_win_rate": round(c_wins / valid, 4),
            "baseline_win_rate": round(b_wins / valid, 4),
            "draw_rate": round(draws / valid, 4),
            "budget": self._gate_budget,
        }

    def apply(self, game_id: str) -> ApplyResult:
        """Build, gate, and (on pass) publish the empirical model for a game."""
        if not self.enabled(game_id):
            return ApplyResult(game_id=game_id, applied=False, reason="disabled")
        try:
            table, samples, coverage = self.build_empirical_table(game_id)
            if samples < self._min_samples or coverage == 0:
                return ApplyResult(
                    game_id=game_id, applied=False, reason="insufficient", samples=samples, coverage=coverage
                )
            current = self._model_store.current(game_id)
            if current is not None and current.table == table:
                return ApplyResult(
                    game_id=game_id,
                    applied=False,
                    reason="unchanged",
                    version=current.version,
                    samples=samples,
                    coverage=coverage,
                )
            baseline = current.table if current is not None else None
            gate_result = self.gate(game_id, table, baseline)
            c_rate = gate_result["candidate_win_rate"]
            b_rate = gate_result["baseline_win_rate"]
            if c_rate + self._gate_tolerance >= b_rate:
                model = self._model_store.publish(game_id, table, samples=samples, coverage=coverage, gate=gate_result)
                # 留存上界（设计文档 §2 承诺，此前从未接线）：发布成功后
                # 收缩轨迹库至最新 N 局，防止 jsonl 无限增长。
                try:
                    self._store.trim(game_id)
                except (LearningStoreError, ValueError, OSError):
                    pass  # 留存收缩失败不影响发布结果
                return ApplyResult(
                    game_id=game_id,
                    applied=True,
                    reason="ok",
                    version=model.version,
                    gate=gate_result,
                    samples=samples,
                    coverage=coverage,
                )
            return ApplyResult(
                game_id=game_id, applied=False, reason="rejected", gate=gate_result, samples=samples, coverage=coverage
            )
        except Exception as exc:  # noqa: BLE001 - surface to status; never crash the server
            return ApplyResult(game_id=game_id, applied=False, reason="error", error=str(exc))

    def status(self, game_id: str) -> dict:
        """Public per-game status for the platform API and CLI."""
        counts = self._store.counts(game_id)
        model = self._model_store.status(game_id)
        human = counts["human_decisions"]
        return {
            "game_id": game_id,
            "enabled": self.enabled(game_id),
            **counts,
            "model": model,
            "min_samples": self._min_samples,
            "pending": model is None and human >= self._min_samples,
        }

    def status_all(self) -> list[dict]:
        """Status for every game with stored data or enabled learning."""
        games = set(self._store.game_ids()) | {g for g in self._enabled if self._enabled[g]}
        return [self.status(g) for g in sorted(games)]

    # ── Background auto-apply ────────────────────────────────────

    def start_auto(self, interval_seconds: float = 300.0) -> None:
        """Daemon loop: apply every enabled game once ``interval`` elapses.

        Idempotent — calling twice while running is a no-op.  Errors are
        swallowed per iteration (logged to stderr) so a bad game never
        kills the loop.
        """
        if self._auto_thread is not None and self._auto_thread.is_alive():
            return
        self._auto_stop.clear()

        def _loop() -> None:
            while not self._auto_stop.wait(interval_seconds):
                for game_id in list(self._enabled):
                    if self.enabled(game_id):
                        try:
                            result = self.apply(game_id)
                            if result.applied:
                                print(f"[online-learning] published {game_id} v{result.version} (gate pass)")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[online-learning] auto-apply failed for {game_id}: {exc}")

        self._auto_thread = threading.Thread(target=_loop, name="online-learning-auto", daemon=True)
        self._auto_thread.start()

    def stop_auto(self) -> None:
        self._auto_stop.set()
        if self._auto_thread is not None:
            self._auto_thread.join(timeout=2.0)
            self._auto_thread = None

    # ── Gate match loop ───────────────────────────────────────────

    @staticmethod
    def _play_one(
        engine_c: Any,
        engine_b: Any,
        solver_c: Any,
        solver_b: Any,
        seat_c: str,
        seat_b: str,
        rng_seed: int,
    ) -> tuple[str | None, int]:
        """One hybrid-vs-hybrid match; returns ``(winner_tag, moves)``.

        Mirrors ``BenchmarkRunner._play_one``: the engine belonging to
        the currently acting seat drives the state (both engines load the
        same rules), chance nodes are sampled from the acting engine.
        """
        rng = random.Random(rng_seed)
        engine = engine_c
        state = engine.create_initial_state()
        moves = 0
        while not engine.is_terminal(state) and moves < 200:
            node_type = engine.get_node_type(state)
            if node_type == "chance":
                _, state = engine.sample_chance(state)
                continue
            if node_type != "player":
                break
            current = engine.get_current_player(state)
            solver = solver_c if current == seat_c else solver_b
            engine = engine_c if current == seat_c else engine_b
            action = solver.select_action(state)
            if action is None:  # search found nothing — random fallback
                legal = engine.get_legal_actions(state)
                action = rng.choice(legal) if legal else None
            if action is None:
                break
            state = engine.apply_action(state, action)
            moves += 1
        winner = state["env"].get("winner")
        if winner == seat_c:
            return "c", moves
        if winner == seat_b:
            return "b", moves
        return None, moves
