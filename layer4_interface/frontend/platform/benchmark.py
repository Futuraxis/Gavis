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
from typing import Optional

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter
from layer3_solvers import CFR, MCTS, CFRConfig, HybridConfig, HybridSolver, MCTSConfig, SolverBase, SolverConfig
from layer3_solvers.base import SolverMetrics, State

from .games import GAMES

#: Solver compatibility matrix — the single source of truth for both the
#: API's ``solver_options`` and job validation.  CFR is excluded from
#: Texas Hold'em (imperfect-information tree explosion).
SOLVER_OPTIONS: dict[str, tuple[str, ...]] = {
    "moon_chess": ("mcts", "cfr", "hybrid", "random"),
    "stochastic_gomoku": ("mcts", "cfr", "hybrid", "random"),
    "texas_holdem": ("mcts", "hybrid", "random"),
    "mahjong_guangdong": ("mahjong", "random"),
    "mahjong_hongzhong": ("mahjong", "random"),
    "mahjong_blood": ("mahjong", "random"),
}

#: Search budget per game (harder than the play tiers, bounded).
BENCHMARK_BUDGETS: dict[str, int] = {
    "moon_chess": 2000,
    "stochastic_gomoku": 3000,
    "texas_holdem": 1500,
    "mahjong_guangdong": 1000,
    "mahjong_hongzhong": 1000,
    "mahjong_blood": 1000,
}

SOLVER_LABELS: dict[str, str] = {
    "mcts": "MCTS",
    "cfr": "CFR",
    "hybrid": "Hybrid",
    "random": "随机",
    "mahjong": "启发式",
}


class RandomSolver(SolverBase):
    """Uniform random policy — the baseline for benchmark comparisons."""

    def __init__(self, adapter: SolverAdapter, seed: Optional[int] = None) -> None:
        super().__init__(adapter, SolverConfig(seed=seed))
        self._rng = random.Random(seed)

    def select_action(self, state: State) -> Optional[ActionInstance]:
        legal = self.adapter.get_legal_actions(state)
        return self._rng.choice(legal) if legal else None

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        return SolverMetrics(episodes=episodes)


def create_solver(game_id: str, name: str, engine: SolverAdapter, seed: int, budget: int) -> SolverBase:
    """Instantiate a benchmark solver by name; raises ValueError on mismatch."""
    if name == "mcts":
        return MCTS(engine, MCTSConfig(seed=seed, budget=budget))
    if name == "cfr":
        if game_id == "texas_holdem":
            raise ValueError("CFR 不适用于德州扑克（不完全信息）")
        return CFR(engine, CFRConfig(seed=seed, iterations=1000, depth_limit=8))
    if name == "hybrid":
        return HybridSolver(
            engine,
            HybridConfig(
                seed=seed,
                mode="search",
                imperfect_information=(game_id == "texas_holdem"),
                mcts_budget=budget,
                opponent_model="uniform",
            ),
        )
    if name == "random":
        return RandomSolver(engine, seed)
    if name == "mahjong":
        from layer3_solvers.mahjong.heuristic import MahjongHeuristicAI
        return MahjongHeuristicAI(engine, SolverConfig(seed=seed))
    raise ValueError(f"未知求解器: {name}")


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
    error: Optional[str] = None
    results: Optional[dict] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class BenchmarkRunner:
    """Runs benchmark jobs in background threads; results are polled."""

    def __init__(self, seed: int = 42, max_iters: int = 200) -> None:
        self._seed = seed
        self._max_iters = max_iters
        self._jobs: dict[str, BenchmarkJob] = {}
        self._lock = threading.Lock()

    # ── Job API ──────────────────────────────────────────────────

    def start(
        self, game_id: str, solver_a: str, solver_b: str, iterations: int, budget: Optional[int] = None
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
        threading.Thread(target=self._run, args=(job, budget), daemon=True).start()
        return job

    def status(self, job_id: str) -> Optional[BenchmarkJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[BenchmarkJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.started_at or "", reverse=True)
        return jobs[:limit]

    # ── Job execution ────────────────────────────────────────────

    def _run(self, job: BenchmarkJob, budget: Optional[int]) -> None:
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

    def _execute(self, job: BenchmarkJob, budget: Optional[int]) -> dict:
        spec = GAMES[job.game_id]
        search_budget = budget if budget is not None else BENCHMARK_BUDGETS[job.game_id]
        # One engine + solver pair per job: the thread owns them, so reuse
        # across iterations is safe, and CFR's table is warmed exactly once.
        seed = self._seed + int(job.job_id, 16) % 1_000_000
        engine = spec.create_engine(seed)
        solver_a = create_solver(job.game_id, job.solver_a, engine, seed, search_budget)
        solver_b = create_solver(job.game_id, job.solver_b, engine, seed, search_budget)
        for solver, name in ((solver_a, job.solver_a), (solver_b, job.solver_b)):
            if name == "cfr":
                solver.solve(engine.create_initial_state(), verbose=False)

        a_wins = b_wins = draws = errors = 0
        total_moves = 0
        total_seconds = 0.0
        per_iteration: list[dict] = []
        for i in range(job.iterations):
            try:
                a_first = i % 2 == 0  # swap seats every iteration
                winner_tag, moves, seconds = self._play_one(engine, solver_a, solver_b, spec, a_first, seed + i)
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
        return {
            "iterations": total,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": round(a_wins / total, 4),
            "b_win_rate": round(b_wins / total, 4),
            "draw_rate": round(draws / total, 4),
            "avg_moves": round(total_moves / max(1, total), 2),
            "avg_seconds_per_move": round(total_seconds / max(1, total_moves), 4),
            "errors": errors,
            "per_iteration": per_iteration,
        }

    @staticmethod
    def _play_one(
        engine: SolverAdapter, solver_a: SolverBase, solver_b: SolverBase, spec, a_first: bool, rng_seed: int
    ) -> tuple[Optional[str], int, float]:
        """Play one full match; returns (winner_tag, moves, seconds).

        Note: mahjong benchmarks run 2-player seats only (the runner
        swaps exactly two seats; 4-player mahjong is not benchmarked).
        """
        seats = spec.seat_options[:2]
        seat_a = seats[0] if a_first else seats[1]
        seat_b = spec.seat_options[1] if a_first else spec.seat_options[0]
        rng = random.Random(rng_seed)
        state = engine.create_initial_state()
        moves = 0
        t0 = time.time()
        while not engine.is_terminal(state) and moves < 200:
            node_type = engine.get_node_type(state)
            if node_type == "chance":
                _, state = engine.sample_chance(state)
                continue
            if node_type != "player":
                break
            current = engine.get_current_player(state)
            solver = solver_a if current == seat_a else solver_b
            action = solver.select_action(state)
            if action is None:  # search found nothing — random fallback
                legal = engine.get_legal_actions(state)
                action = rng.choice(legal) if legal else None
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
