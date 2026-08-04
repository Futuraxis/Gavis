"""Focused regression tests for the PSRO implementation."""

from __future__ import annotations

import numpy as np

from layer2_engine.interfaces.solver_adapter import ActionInstance
from layer3_solvers.psro.gym_adapter import GymAdapter
from layer3_solvers.psro.meta_game import estimate_reward, gamescape


class CountingAgent:
    """Minimal agent that records how often it is asked to act."""

    def __init__(self) -> None:
        self.calls = 0

    def step(self, obs: int, Amask: np.ndarray | None = None) -> int:  # noqa: N803
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
