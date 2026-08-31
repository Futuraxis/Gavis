"""Solver benchmark engine for the platform frontend.

Runs solver-vs-solver matches on a chosen game in one background thread
per job; clients poll ``status()`` for progress.  Each job owns its
engines, solvers and RNGs, so jobs can run concurrently.  Seats are
swapped on alternating iterations to cancel first-player advantage.
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from layer2_engine.core.engine import GameEngine

from ...solver_provider import SolverHandle, SolverProvider
from .games import GAMES

#: Solver compatibility matrix — the single source of truth for both the
#: API's ``solver_options`` and job validation.  CFR is excluded from
#: Texas Hold'em (imperfect-information tree explosion).
SOLVER_OPTIONS: dict[str, tuple[str, ...]] = {
    "moon_chess": ("mcts", "cfr", "hybrid", "random"),
    "stochastic_gomoku": ("mcts", "cfr", "hybrid", "random"),
    "texas_holdem": ("mcts", "hybrid", "random"),
    "mahjong_guangdong": ("mahjong", "random", "mcts", "maac"),
    "mahjong_hongzhong": ("mahjong", "random", "mcts", "maac"),
    "mahjong_blood": ("mahjong", "random", "mcts", "maac"),
    "mahjong_sichuan": ("mahjong", "random", "mcts", "maac"),
    "mahjong_changsha": ("mahjong", "random", "mcts", "maac"),
    "mahjong_taiwan": ("mahjong", "random", "mcts", "maac"),
    "mahjong_international": ("mahjong", "random", "mcts", "maac"),
}

#: Search budget per game (harder than the play tiers, bounded).
BENCHMARK_BUDGETS: dict[str, int] = {
    "moon_chess": 2000,
    "stochastic_gomoku": 3000,
    "texas_holdem": 1500,
    # 麻将 MCTS 基线预算：该游戏每决策 ~200ms/迭代，1000 迭代单决策需数分钟；
    # 登记 30（与 train-cli 评估预算一致，数据驱动的“可执行基线”语义）。
    "mahjong_guangdong": 30,
    "mahjong_hongzhong": 30,
    "mahjong_blood": 30,
    "mahjong_sichuan": 30,
    "mahjong_changsha": 30,
    "mahjong_taiwan": 30,
    "mahjong_international": 1000,
}

SOLVER_LABELS: dict[str, str] = {
    "mcts": "MCTS",
    "cfr": "CFR",
    "hybrid": "Hybrid",
    "random": "随机",
    "mahjong": "启发式",
    "maac": "MAAC(训练)",
}

#: 保留在内存中的 job 数上限（审计 3.6 资源泄漏：_jobs 此前无界增长）。
MAX_JOBS = 500


@dataclass
class BenchmarkJob:
    """State of one benchmark run, serialized to the frontend."""

    job_id: str
    game_id: str
    solver_a: str
    solver_b: str
    iterations: int
    status: str = "pending"  # pending | running | done | error
    progress: int = 0
    error: str | None = None
    results: dict | None = None
    started_at: str | None = None
    ended_at: str | None = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class BenchmarkRunner:
    """Runs benchmark jobs in background threads; results are polled."""

    def __init__(self, provider: SolverProvider, seed: int = 42, max_iters: int = 200) -> None:
        self._provider = provider
        self._seed = seed
        self._max_iters = max_iters
        self._jobs: dict[str, BenchmarkJob] = {}
        self._lock = threading.Lock()

    # ── Job API ──────────────────────────────────────────────────

    def start(
        self, game_id: str, solver_a: str, solver_b: str, iterations: int, budget: int | None = None
    ) -> BenchmarkJob:
        """Validate and launch a new job; returns the job."""
        if game_id not in GAMES:
            raise ValueError(f"未知游戏: {game_id}")
        options = SOLVER_OPTIONS.get(game_id, ())
        if solver_a not in options or solver_b not in options:
            raise ValueError(f"求解器与该游戏不兼容: {solver_a} vs {solver_b}")
        if solver_a == solver_b:
            raise ValueError("两个求解器不能相同")
        if not 1 <= iterations <= self._max_iters:
            raise ValueError(f"迭代次数须在 1..{self._max_iters} 之间")
        job = BenchmarkJob(
            job_id=uuid.uuid4().hex[:8],
            game_id=game_id,
            solver_a=solver_a,
            solver_b=solver_b,
            iterations=iterations,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._prune_locked()
        threading.Thread(target=self._run, args=(job, budget), daemon=True).start()
        return job

    def _prune_locked(self) -> None:
        """Drop finished jobs when the registry exceeds ``MAX_JOBS``.

        Called with ``self._lock`` held — daemon threads never clean up
        their own ``BenchmarkJob`` entries, so without a bound the dict
        grows without limit.
        """
        if len(self._jobs) <= MAX_JOBS:
            return
        finished = [jid for jid, j in self._jobs.items() if j.status in ("done", "error")]
        for jid in finished:
            self._jobs.pop(jid, None)

    def status(self, job_id: str) -> BenchmarkJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[BenchmarkJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.started_at or "", reverse=True)
        return jobs[:limit]

    # ── Job execution ────────────────────────────────────────────

    def _run(self, job: BenchmarkJob, budget: int | None) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = _now_iso()
        try:
            results = self._execute(job, budget)
            with self._lock:
                job.results = results
                job.status = "done"
                job.ended_at = _now_iso()
        except Exception as exc:  # surface any failure to the frontend
            with self._lock:
                job.error = str(exc)
                job.status = "error"
                job.ended_at = _now_iso()

    def _execute(self, job: BenchmarkJob, budget: int | None) -> dict:
        spec = GAMES[job.game_id]
        search_budget = budget if budget is not None else BENCHMARK_BUDGETS[job.game_id]
        # One engine + solver pair PER SOLVER (M-10): a shared engine with
        # identical seeds makes e.g. CFR-vs-CFR build the same strategy
        # table and play identical moves, so the benchmark measures
        # nothing.  Distinct seeds keep the two players independent.
        seed = self._seed + int(job.job_id, 16) % 1_000_000
        seed_a, seed_b = seed, seed + 1_000_003
        engine_a = spec.create_engine(seed_a)
        engine_b = spec.create_engine(seed_b)
        solver_a = self._provider.create_solver(job.game_id, job.solver_a, engine_a, seed_a, search_budget)
        solver_b = self._provider.create_solver(job.game_id, job.solver_b, engine_b, seed_b, search_budget)
        for solver, name, engine in (
            (solver_a, job.solver_a, engine_a),
            (solver_b, job.solver_b, engine_b),
        ):
            if name == "cfr":
                solver.solve(engine.create_initial_state(), verbose=False)

        a_wins = b_wins = draws = errors = 0
        total_moves = 0
        total_seconds = 0.0
        per_iteration: list[dict] = []
        for i in range(job.iterations):
            try:
                a_first = i % 2 == 0  # swap seats every iteration
                winner_tag, moves, seconds = self._play_one(
                    engine_a, engine_b, solver_a, solver_b, spec, a_first, seed + i
                )
                if winner_tag == "a":
                    a_wins += 1
                elif winner_tag == "b":
                    b_wins += 1
                else:
                    draws += 1
                total_moves += moves
                total_seconds += seconds
                per_iteration.append({"moves": moves, "seconds": round(seconds, 3), "winner": winner_tag})
            except Exception:
                errors += 1
                per_iteration.append({"moves": 0, "seconds": 0.0, "winner": None, "error": True})
            with self._lock:
                job.progress = i + 1

        total = job.iterations
        valid = total - errors  # 出错迭代不计入胜率分母（审查 P2-15）
        a_win_rate = round(a_wins / valid, 4) if valid else 0.0
        b_win_rate = round(b_wins / valid, 4) if valid else 0.0
        draw_rate = round(draws / valid, 4) if valid else 0.0
        return {
            "iterations": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_win_rate,
            "b_win_rate": b_win_rate,
            "draw_rate": draw_rate,
            "avg_moves": round(total_moves / max(1, total), 2),
            "avg_seconds_per_move": round(total_seconds / max(1, total_moves), 4),
            "errors": errors,
            "per_iteration": per_iteration,
        }

    @staticmethod
    def _play_one(
        engine_a: GameEngine,
        engine_b: GameEngine,
        solver_a: SolverHandle,
        solver_b: SolverHandle,
        spec,
        a_first: bool,
        rng_seed: int,
    ) -> tuple[str | None, int, float]:
        """Play one full match; returns (winner_tag, moves, seconds).

        The state is driven by the engine belonging to the currently
        acting solver (both engines load the same rules, so the state
        dict is interchangeable between them).

        Note: the runner swaps exactly two seats (A/B solvers); mahjong
        now runs the registered 4-player configuration, with the B-side
        solver driving all seats beyond the first seat pair.
        """
        seats = spec.seat_options[:2]
        seat_a = seats[0] if a_first else seats[1]
        seat_b = spec.seat_options[1] if a_first else spec.seat_options[0]
        rng = random.Random(rng_seed)
        engine = engine_a
        state = engine.create_initial_state()
        moves = 0
        t0 = time.time()
        # 步数护栏：月亮棋/五子棋几十步即终局；麻将自然局 ~1000 步（墙尽或
        # 胡牌），旧 200 上限会把麻将 benchmark 截断在发牌/早期弃牌阶段，
        # 每局都误判为无胜负——抬到 2000（仍远低于任何病理死循环）。
        while not engine.is_terminal(state) and moves < 2000:
            node_type = engine.get_node_type(state)
            if node_type == "chance":
                _, state = engine.sample_chance(state)
                continue
            if node_type != "player":
                break
            current = engine.get_current_player(state)
            solver = solver_a if current == seat_a else solver_b
            engine = engine_a if current == seat_a else engine_b
            legal = engine.get_legal_actions(state)
            if not legal:
                break
            if len(legal) == 1:
                action = legal[0]  # forced step (e.g. no-choice claim_pass)
            else:
                action = solver.select_action(state)
                if action is None:  # search found nothing — random fallback
                    action = rng.choice(legal)
            if action is None:
                break
            state = engine.apply_action(state, action)
            moves += 1
        seconds = time.time() - t0
        winner = state["env"].get("winner")
        if winner == seat_a:
            return "a", moves, seconds
        if winner == seat_b:
            return "b", moves, seconds
        return None, moves, seconds
