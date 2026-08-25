"""Tests for the benchmark engine — RandomSolver, factory, job lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer4_interface.frontend.platform.benchmark import (
    BENCHMARK_BUDGETS,
    SOLVER_OPTIONS,
    BenchmarkRunner,
)
from layer4_interface.frontend.platform.games import GAMES
from train_cli import RandomSolver, create_solver, default_provider

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


class TestRandomSolver:
    def test_select_action_is_legal(self):
        engine = GameEngine(json.load(open(RULES_DIR / "moon_chess.json", encoding="utf-8")), seed=42)
        solver = RandomSolver(engine, seed=42)
        state = engine.create_initial_state()
        for _ in range(20):
            action = solver.select_action(state)
            assert action is not None
            assert action in engine.get_legal_actions(state)
            state = engine.apply_action(state, action)
            if engine.is_terminal(state):
                break

    def test_train_is_noop(self):
        solver = RandomSolver(
            GameEngine(json.load(open(RULES_DIR / "moon_chess.json", encoding="utf-8")), seed=42), seed=42
        )
        metrics = solver.train(episodes=10)
        assert metrics.episodes == 10


class TestCreateSolver:
    @pytest.mark.parametrize("game_id", list(SOLVER_OPTIONS))
    def test_options_instantiate(self, game_id: str):
        engine = GAMES[game_id].create_engine(42)
        for name in SOLVER_OPTIONS[game_id]:
            solver = create_solver(game_id, name, engine, 42, BENCHMARK_BUDGETS[game_id])
            assert solver is not None
            assert solver.name

    def test_cfr_rejected_for_texas(self):
        engine = GAMES["texas_holdem"].create_engine(42)
        with pytest.raises(ValueError):
            create_solver("texas_holdem", "cfr", engine, 42, 100)

    def test_unknown_solver_rejected(self):
        engine = GameEngine(json.load(open(RULES_DIR / "moon_chess.json", encoding="utf-8")), seed=42)
        with pytest.raises(ValueError):
            create_solver("moon_chess", "ppo", engine, 42, 100)


class TestRunner:
    def _wait(self, job, timeout: float = 120.0):
        deadline = time.time() + timeout
        while job.status in ("pending", "running") and time.time() < deadline:
            time.sleep(0.05)
        assert job.status == "done", job.error

    def test_job_lifecycle(self):
        runner = BenchmarkRunner(provider=default_provider, seed=42)
        job = runner.start("moon_chess", "mcts", "random", 2, budget=200)
        self._wait(job)
        results = job.results
        assert results["iterations"] == 2
        assert results["a_wins"] + results["b_wins"] + results["draws"] + results["errors"] == 2
        assert len(results["per_iteration"]) == 2
        assert 0.0 <= results["a_win_rate"] <= 1.0
        assert job.status == "done"
        assert job.progress == 2

    def test_concurrent_jobs(self):
        runner = BenchmarkRunner(provider=default_provider, seed=42)
        job_a = runner.start("moon_chess", "mcts", "random", 1, budget=100)
        job_b = runner.start("moon_chess", "random", "mcts", 1, budget=100)
        assert job_a.job_id != job_b.job_id
        self._wait(job_a)
        self._wait(job_b)
        assert runner.status(job_a.job_id).status == "done"
        assert runner.status(job_b.job_id).status == "done"

    def test_status_unknown_job(self):
        assert BenchmarkRunner(provider=default_provider).status("nope0000") is None

    def test_validation(self):
        runner = BenchmarkRunner(provider=default_provider)
        with pytest.raises(ValueError):
            runner.start("moon_chess", "mcts", "mcts", 2)  # identical solvers
        with pytest.raises(ValueError):
            runner.start("nope", "mcts", "random", 2)  # unknown game
        with pytest.raises(ValueError):
            runner.start("texas_holdem", "cfr", "mcts", 2)  # incompatible solver
        with pytest.raises(ValueError):
            runner.start("moon_chess", "mcts", "random", 0)  # iterations out of range
        with pytest.raises(ValueError):
            runner.start("moon_chess", "mcts", "random", 1000)  # beyond max_iters
