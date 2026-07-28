"""最小 PPO 训练脚本。"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable

import numpy as np

from algorithms.ppo_agent import PPOAgent
from encoding import GameStateAdapter, MoonStateEncoder
from tests.mock_moon_env import MockMoonEnv, RandomAgent


class GameEnvProtocol:
    def reset(self) -> dict: ...
    def get_state(self) -> dict: ...
    def step(self, action: dict) -> tuple[dict, float, bool, dict]: ...
    def is_terminal(self) -> bool: ...
    def get_winner_id(self) -> str | None: ...


def run_training(
    env_factory: Callable[[], GameEnvProtocol] | None = None,
    episodes: int = 20,
    save_path: str | None = None,
) -> tuple[PPOAgent, list[dict[str, float]]]:
    adapter = GameStateAdapter()
    encoder = MoonStateEncoder(adapter)
    env_factory = env_factory or MockMoonEnv
    env = env_factory()
    agent = PPOAgent(state_dim=encoder.FEATURE_DIM)
    logs: list[dict[str, float]] = []

    for episode in range(episodes):
        env = env_factory()
        state = env.reset()
        controlled_player = random.choice(["player_x", "player_o"])
        opponent = RandomAgent()
        episode_reward = 0.0
        agent_turn_steps = 0

        while not env.is_terminal():
            state = env.get_state()
            current_player = adapter.get_current_player(state)
            if current_player == controlled_player:
                state_vector = encoder.encode(state, perspective_player_id=controlled_player)
                action_mask = encoder.get_action_mask(state)
                action, log_prob, value = agent.select_action(
                    state_vector,
                    action_mask,
                    legal_actions=adapter.get_legal_actions(state),
                )
                action_payload = agent.build_action(controlled_player, action)
                next_state, _, done, _ = env.step(action_payload)
                reward = _terminal_reward(env.get_winner_id(), controlled_player) if done else 0.0
                next_value = 0.0 if done else agent.evaluate_value(
                    encoder.encode(next_state, perspective_player_id=controlled_player)
                )
                agent.record_transition(
                    state=state_vector,
                    action=action,
                    action_mask=action_mask,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    next_value=next_value,
                )
                episode_reward += reward
                agent_turn_steps += 1
            else:
                action_payload = opponent.act(state)
                next_state, _, done, _ = env.step(action_payload)
                if done and env.get_winner_id() == current_player:
                    episode_reward -= 1.0 if current_player != controlled_player else 0.0
            state = next_state

        metrics = agent.update()
        win_rate = 1.0 if env.get_winner_id() == controlled_player else 0.0
        log_row = {
            "episode": float(episode),
            "average_reward": episode_reward,
            "win_rate": win_rate,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "agent_turn_steps": float(agent_turn_steps),
        }
        logs.append(log_row)

    if save_path:
        agent.save(save_path)
    return agent, logs


def _terminal_reward(winner_id: str | None, controlled_player: str) -> float:
    if winner_id is None:
        return 0.0
    return 1.0 if winner_id == controlled_player else -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="artifacts/ppo_agent.pt")
    args = parser.parse_args()
    _, logs = run_training(episodes=args.episodes, save_path=args.save_path)
    for row in logs[-5:]:
        print(row)


if __name__ == "__main__":
    main()
