"""评估脚本。"""

from __future__ import annotations

import argparse

from algorithms.ppo_agent import PPOAgent
from encoding import GameStateAdapter, MoonStateEncoder
from tests.mock_moon_env import MockMoonEnv, RandomAgent


def evaluate(model_path: str, episodes: int = 10) -> dict[str, float]:
    adapter = GameStateAdapter()
    encoder = MoonStateEncoder(adapter)
    agent = PPOAgent(state_dim=encoder.FEATURE_DIM)
    agent.load(model_path)
    opponent = RandomAgent()

    wins = 0
    draws = 0
    for _ in range(episodes):
        env = MockMoonEnv()
        state = env.reset()
        controlled_player = "player_x"
        while not env.is_terminal():
            state = env.get_state()
            if adapter.get_current_player(state) == controlled_player:
                vector = encoder.encode(state, controlled_player)
                mask = encoder.get_action_mask(state)
                action_index, _, _ = agent.select_action(
                    vector,
                    mask,
                    legal_actions=adapter.get_legal_actions(state),
                )
                action = agent.build_action(controlled_player, action_index)
            else:
                action = opponent.act(state)
            state, _, _, _ = env.step(action)
        winner = env.get_winner_id()
        if winner == controlled_player:
            wins += 1
        elif winner is None:
            draws += 1
    return {"episodes": float(episodes), "win_rate": wins / episodes, "draw_rate": draws / episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    print(evaluate(args.model_path, args.episodes))


if __name__ == "__main__":
    main()
