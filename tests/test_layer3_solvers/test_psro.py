"""Focused regression tests for the PSRO implementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance
from layer3_solvers.psro.gym_adapter import GymAdapter
from layer3_solvers.psro.meta_game import estimate_reward, gamescape
from layer3_solvers.psro.solver import PSROConfig, PSROSolver

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def _moon(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


class CountingAgent:
    """Minimal agent that records how often it is asked to act."""

    def __init__(self) -> None:
        self.calls = 0

    def step(self, obs: int, amask: np.ndarray | None = None) -> int:
        self.calls += 1
        return 0


class TwoTurnEnvironment:
    """A tiny episode containing one turn for each player."""

    def __init__(self) -> None:
        self.turns = 0

    def reset(self) -> tuple[int, dict]:
        self.turns = 0
        return 0, {}

    def available_actions(self) -> np.ndarray:
        return np.array([True])

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        self.turns += 1
        done = self.turns >= 2
        reward = 1.0 if done else 0.0
        return 0, reward, done, False, {}


def test_estimate_reward_uses_both_players() -> None:
    """The two policies must alternate during a two-turn episode."""
    env = TwoTurnEnvironment()
    player_one = CountingAgent()
    player_two = CountingAgent()

    reward = estimate_reward(
        env,
        num_episodes=1,
        p1=player_one,
        p2=player_two,
        max_steps=2,
    )

    assert player_two.calls == 1
    assert player_one.calls == 1
    assert reward == 1.0


class PerspectiveAdapter:
    """Minimal two-player adapter used to check the reward perspective."""

    def __init__(self) -> None:
        self.rules = {"players": ["row", "column"]}
        self.utility_players: list[str] = []

    def create_initial_state(self) -> dict:
        return {"_board": [], "turn": 0}

    def get_current_player(self, state: dict) -> str | None:
        if state["turn"] >= 2:
            return None
        return "row" if state["turn"] == 0 else "column"

    def get_node_type(self, state: dict) -> str:
        return "terminal" if state["turn"] >= 2 else "player"

    def get_legal_actions(self, state: dict) -> list[ActionInstance]:
        return [
            ActionInstance(
                template_id="place",
                type="player",
                actor_id=self.get_current_player(state) or "",
                params={"cell": {"id": "cell_0_0"}},
                canonical_key="place:cell_0_0",
            )
        ]

    def apply_action(self, state: dict, action: ActionInstance) -> dict:
        return {"_board": [], "turn": state["turn"] + 1}

    def is_terminal(self, state: dict) -> bool:
        return state["turn"] >= 2

    def get_utility(self, state: dict, player: str) -> float:
        self.utility_players.append(player)
        if not self.is_terminal(state):
            return 0.0
        return 1.0 if player == "row" else -1.0


def test_gym_adapter_keeps_row_player_reward_perspective() -> None:
    """Rewards must not switch perspective when the second player acts."""
    adapter = PerspectiveAdapter()
    env = GymAdapter(adapter)

    env.reset()
    _, first_reward, first_done, _, _ = env.step(0)
    _, second_reward, second_done, _, _ = env.step(0)

    assert first_reward == 0.0
    assert first_done is False
    assert second_reward == 1.0
    assert second_done is True
    assert adapter.utility_players == ["row", "row"]


class CountingMatchEnvironment(TwoTurnEnvironment):
    """Environment that counts how many evaluation episodes are run."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0

    def reset(self) -> tuple[int, dict]:
        self.reset_calls += 1
        return super().reset()


def test_gamescape_reuses_previous_payoffs() -> None:
    """Adding one policy should evaluate only its new match-ups."""
    env = CountingMatchEnvironment()
    policy = np.ones((1, 1))

    first_matrix = gamescape(
        env,
        [policy, policy],
        Ne=2,
    )

    # Two policies have one unique match-up, evaluated for two episodes.
    assert env.reset_calls == 2

    expanded_matrix = gamescape(
        env,
        [policy, policy, policy],
        Ne=2,
        previous=first_matrix,
    )

    # The third policy creates only two new match-ups:
    # 2 old policies × 2 evaluation episodes = 4 additional resets.
    assert env.reset_calls == 6
    assert expanded_matrix.shape == (3, 3)
    assert np.array_equal(expanded_matrix[:2, :2], first_matrix)


def test_psro_save_load_preserves_payoff_matrix(tmp_path: Path) -> None:
    """A saved solver should restore its cached payoff matrix."""
    adapter = PerspectiveAdapter()
    config = PSROConfig(
        seed=42,
        num_iters=1,
        num_steps_per_iter=1,
    )

    solver = PSROSolver(adapter, config)
    expected_matrix = np.array(
        [
            [0.0, 0.5],
            [-0.5, 0.0],
        ]
    )
    solver._payoff_matrix = expected_matrix

    model_path = tmp_path / "psro_cache.npz"
    solver.save(str(model_path))

    restored_solver = PSROSolver(adapter, config)
    restored_solver.load(str(model_path))

    assert restored_solver._payoff_matrix is not None
    np.testing.assert_array_equal(
        restored_solver._payoff_matrix,
        expected_matrix,
    )


# ── 审查 P1-1/2/3 修复回归 ─────────────────────────────────────────


class _OpponentEndsEnv:
    """Row player's move is non-terminal (reward 0); the column player's
    reply ends the game with +1 from the row player's view."""

    observation_space = type("S", (), {"n": 2})()
    action_space = type("S", (), {"n": 2})()

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        return 0, {}

    def available_actions(self) -> np.ndarray:
        return np.array([True, True])

    def step(self, action: int):
        self.steps += 1
        if self.steps == 1:
            return 1, 0.0, False, False, {}
        return 0, 1.0, True, False, {}


def test_best_response_folds_opponent_terminal_payoff():
    """P1-1: 对手终局的收益必须折进 Q 更新（旧实现 update 在对手应手
    之前执行，对手终结的输局收益被整体丢弃，BR 退化为 1-ply 贪心）。"""
    from layer3_solvers.psro.agent import TabularQAgent

    env = _OpponentEndsEnv()
    agent = TabularQAgent(2, 2, epsilon=0.0, alpha=1.0, gamma=0.9)
    agent.reset_rng(0)
    obs, _ = env.reset()
    mask = env.available_actions()
    action = agent.select_action(obs, mask)
    next_obs, r1, done, _, _ = env.step(action)
    assert not done
    _, r2, done, _, _ = env.step(0)  # opponent's reply ends the game
    assert done
    # tabular_q_best_response 的固定循环：对手应手之后才 update，
    # reward = r1 + gamma * r2。
    agent.update(obs, action, r1 + 0.9 * r2, next_obs, done, next_mask=env.available_actions())
    assert agent.Q[obs, action] == pytest.approx(0.9)


def test_select_action_masks_from_passed_state():
    """P1-2: select_action 的 mask 必须来自传入 state（旧实现读 gym 的
    陈旧 _state → 全真 mask → 采到已占格 → 返回 None 违反契约）。"""
    adapter = _moon(3)
    solver = PSROSolver(adapter, PSROConfig(seed=1))
    solver._nash_mixture = np.full((19683, 9), 1.0 / 9)  # noqa: SLF001
    state = adapter.create_initial_state()
    state["_arrays"]["board"] = ["p_black"] * 4 + ["p_white"] * 4 + [None]
    action = solver.select_action(state)
    assert action is not None
    cell_id = action.params["cell"]["id"]
    assert cell_id == "cell_2_2", f"mixture picked an occupied cell: {cell_id}"


class _RowAlwaysLosesEnv:
    """Row player (the Nash mixture) always loses: step reward -1, terminal."""

    observation_space = type("S", (), {"n": 2})()
    action_space = type("S", (), {"n": 2})()
    players = ("row", "column")

    def reset(self, seed=None):
        return 0, {}

    def available_actions(self) -> np.ndarray:
        return np.array([True, True])

    def step(self, action: int):
        return 0, -1.0, True, False, {}

    def get_current_player(self):
        return None


def test_exploitability_measures_column_deviation():
    """P1-3: exploitability 必须测量"混合物作行玩家被池成员击败"的方向
    （旧实现方向相反 + max(v,0) 截断 → 恒 ≈ 0 的噪声指标）。"""
    from layer3_solvers.psro.meta_game import exploitability

    nash = np.full((2, 2), 0.5)
    pool = [np.eye(2)[0].repeat(2).reshape(2, 2), np.eye(2)[1].repeat(2).reshape(2, 2)]
    expl = exploitability(_RowAlwaysLosesEnv(), nash, pool, Ne=1, num_workers=1)
    # row loses every episode → values = [-1, -1] → expl = -mean = 1.0
    assert expl == pytest.approx(1.0)
