"""Tests for all four solvers (Layer 3).

Each solver is tested at the unit level (select_action, train API).
MCTS and CFR are tested on stochastic_gomoku (small board for CFR).
PPO and PSRO are tested on moon_chess via GameEngine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer3_solvers import CFR, MCTS, PSROSolver
from layer3_solvers.cfr import CFRConfig
from layer3_solvers.mcts import MCTSConfig
from layer3_solvers.psro import PSROConfig

try:
    from layer3_solvers.ppo import PPOConfig, PPOSolver

    _HAS_TORCH = True
except (ImportError, TypeError):
    PPOSolver = None
    PPOConfig = None
    _HAS_TORCH = False

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


@pytest.fixture
def moon_adapter() -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=42)


@pytest.fixture
def small_gomoku_engine() -> GameEngine:
    """3×3 gomoku for fast CFR testing (no chance for simplicity)."""
    path = RULES_DIR / "stochastic_gomoku.json"
    with open(path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    rules["constants"]["board_size"] = 3
    return GameEngine(rules, seed=42)


# ── SolverBase compliance ─────────────────────────────────────────


class TestSolverBaseCompliance:
    def test_all_have_required_methods(self, moon_adapter: GameEngine):
        for cls, cfg, _ in _all_solver_configs():
            if cls is None:
                continue
            solver = cls(moon_adapter, cfg)
            assert hasattr(solver, "select_action")
            assert hasattr(solver, "train")
            assert hasattr(solver, "name")

    def test_mcts_cfr_select_action(self, small_gomoku_engine: GameEngine):
        """MCTS and CFR on small gomoku (fast)."""
        state = small_gomoku_engine.create_initial_state()

        # MCTS
        mcts = MCTS(small_gomoku_engine, MCTSConfig(seed=42, budget=200))
        action = mcts.select_action(state)
        assert action is not None
        legal = {a.canonical_key for a in small_gomoku_engine.get_legal_actions(state)}
        assert action.canonical_key in legal

        # CFR (only if it can solve quickly on 3×3)
        cfr = CFR(small_gomoku_engine, CFRConfig(seed=42, iterations=20, depth_limit=4))
        cfr.solve(state, verbose=False)
        action = cfr.select_action(state)
        assert action is not None

    def test_ppo_psro_select_action(self, moon_adapter: GameEngine):
        """PPO and PSRO on moon chess (untrained — just check API)."""
        state = moon_adapter.create_initial_state()

        if _HAS_TORCH:
            ppo = PPOSolver(moon_adapter, PPOConfig(seed=42))
            action = ppo.select_action(state)
            assert action is not None

        psro = PSROSolver(moon_adapter, PSROConfig(seed=42, num_iters=1, num_steps_per_iter=100))
        action = psro.select_action(state)
        # Untrained PSRO may or may not return an action
        if action is not None:
            legal = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
            assert action.canonical_key in legal

    def test_all_can_play_moon_chess_short(self, moon_adapter: GameEngine):
        """Each solver plays a few moves on moon chess."""
        for cls, cfg, name in _all_solver_configs():
            if cls is None or name == "PSRO":
                continue  # PSRO needs training
            solver = cls(moon_adapter, cfg)
            state = moon_adapter.create_initial_state()
            if name == "CFR":
                continue  # CFR needs training on larger board
            for _ in range(5):
                if moon_adapter.is_terminal(state):
                    break
                nt = moon_adapter.get_node_type(state)
                if nt != "player":
                    break
                action = solver.select_action(state)
                if action is None:
                    break
                state = moon_adapter.apply_action(state, action)


# ── MCTS ──────────────────────────────────────────────────────────


class TestMCTS:
    def test_select_action_returns_valid(self, moon_adapter: GameEngine):
        solver = MCTS(moon_adapter, MCTSConfig(seed=42, budget=200))
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        assert action is not None
        legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys

    def test_plays_legal_moves_only(self, moon_adapter: GameEngine):
        solver = MCTS(moon_adapter, MCTSConfig(seed=42, budget=100))
        for _ in range(3):
            state = moon_adapter.create_initial_state()
            for _ in range(9):
                if moon_adapter.is_terminal(state):
                    break
                nt = moon_adapter.get_node_type(state)
                if nt != "player":
                    break
                action = solver.select_action(state)
                if action is None:
                    break
                legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
                assert action.canonical_key in legal_keys
                state = moon_adapter.apply_action(state, action)

    def test_blocks_immediate_three_in_a_row(self, moon_adapter: GameEngine):
        """Black 0,1 → the AI (white) must block cell_0_2.

        Regression: UCB selection did not flip the exploitation term for
        player children, so the search maximized the OPPONENT's utility
        and never learned to defend.
        """
        solver = MCTS(moon_adapter, MCTSConfig(seed=42, budget=1500))
        state = moon_adapter.create_initial_state()
        for cell in ("cell_0_0", "cell_1_2", "cell_0_1"):
            action = next(
                a for a in moon_adapter.get_legal_actions(state) if a.params.get("cell", {}).get("id", "") == cell
            )
            state = moon_adapter.apply_action(state, action)
        # Now black 0,1 on top row; white (AI) must play cell_0_2.
        assert moon_adapter.get_current_player(state) == "p_white"
        action = solver.select_action(state)
        assert action.params.get("cell", {}).get("id", "") == "cell_0_2"


# ── CFR ───────────────────────────────────────────────────────────


class TestCFR:
    def test_solve_returns_strategy(self, small_gomoku_engine: GameEngine):
        solver = CFR(small_gomoku_engine, CFRConfig(seed=42, iterations=30, depth_limit=4))
        state = small_gomoku_engine.create_initial_state()
        strategy = solver.solve(state)
        assert isinstance(strategy, dict)
        assert len(strategy) > 0
        assert abs(sum(strategy.values()) - 1.0) < 0.01

    def test_solve_creates_info_sets(self, small_gomoku_engine: GameEngine):
        solver = CFR(small_gomoku_engine, CFRConfig(seed=42, iterations=30, depth_limit=4))
        state = small_gomoku_engine.create_initial_state()
        solver.solve(state)
        assert len(solver.info_sets) > 0


# ── PPO ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestPPO:
    def test_select_action_no_training(self, moon_adapter: GameEngine):
        cfg = PPOConfig(seed=42, state_dim=38, action_dim=9)
        solver = PPOSolver(moon_adapter, cfg)
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        assert action is not None
        legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys

    def test_train_short(self, moon_adapter: GameEngine):
        cfg = PPOConfig(seed=42, state_dim=38, action_dim=9)
        solver = PPOSolver(moon_adapter, cfg)
        metrics = solver.train(episodes=3, verbose=False)
        assert metrics.episodes == 3

    def test_save_load(self, moon_adapter: GameEngine, tmp_path):
        cfg = PPOConfig(seed=42, state_dim=38, action_dim=9)
        solver = PPOSolver(moon_adapter, cfg)
        path = tmp_path / "ppo_test.pt"
        solver.save(str(path))
        assert path.exists()
        solver2 = PPOSolver(moon_adapter, cfg)
        solver2.load(str(path))


# ── PSRO ──────────────────────────────────────────────────────────


class TestPSRO:
    def test_train_short(self, moon_adapter: GameEngine):
        cfg = PSROConfig(seed=42, num_iters=2, num_steps_per_iter=200)
        solver = PSROSolver(moon_adapter, cfg)
        metrics = solver.train(episodes=2, verbose=False)
        assert metrics.extra["pool_size"] >= 1

    def test_select_action_after_training(self, moon_adapter: GameEngine):
        cfg = PSROConfig(seed=42, num_iters=2, num_steps_per_iter=200)
        solver = PSROSolver(moon_adapter, cfg)
        solver.train(episodes=2, verbose=False)
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        if action is not None:
            legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
            assert action.canonical_key in legal_keys


# ── Nash Solver (PSRO component) ─────────────────────────────────


class TestNashSolver:
    def test_solve_2x2(self):
        import numpy as np

        from layer3_solvers.psro.nash_solver import solve_nash

        reward_matrix = np.array([[0.0, -1.0], [1.0, 0.0]])
        nash = solve_nash(reward_matrix)
        assert nash.shape == (2,)
        assert abs(nash.sum() - 1.0) < 0.01
        assert all(nash >= 0)

    def test_solve_3x3_uniform(self):
        import numpy as np

        from layer3_solvers.psro.nash_solver import solve_nash

        reward_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        nash = solve_nash(reward_matrix)
        assert nash.shape == (3,)
        assert all(nash > 0.1)


# ── Helpers ───────────────────────────────────────────────────────


def _all_solver_configs():
    yield MCTS, MCTSConfig(seed=42, budget=100), "MCTS"
    yield CFR, CFRConfig(seed=42, iterations=10, depth_limit=3), "CFR"
    if _HAS_TORCH:
        yield PPOSolver, PPOConfig(seed=42), "PPO"
    yield PSROSolver, PSROConfig(seed=42, num_iters=1, num_steps_per_iter=100), "PSRO"
