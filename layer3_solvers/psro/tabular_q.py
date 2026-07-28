"""Tabular Q-learning best-response solver for PSRO.

Given an environment and a fixed opponent policy, train a new
policy via tabular Q-learning.
"""

from __future__ import annotations

import numpy as np

from .agent import TabularQAgent, Agent


def tabular_q_best_response(
    env,
    num_steps: int = 10000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eval_interval: int = -1,
    opponent_policy: np.ndarray | None = None,
) -> np.ndarray:
    """Train a best-response policy via tabular Q-learning.

    Parameters
    ----------
    env : GymAdapter
    num_steps : int
        Total training steps.
    epsilon, alpha, gamma : float
        Q-learning hyper-parameters.
    eval_interval : int
        Interval for evaluation (-1 = no evaluation).
    opponent_policy : np.ndarray, optional
        Fixed opponent policy. If None, uses random.

    Returns
    -------
    np.ndarray, shape (state_dim, action_dim)
        Greedy policy (one-hot) derived from trained Q-table.
    """
    obs_dim = env.observation_space.n if hasattr(env.observation_space, 'n') else env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = TabularQAgent(obs_dim, n_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    opponent = Agent(opponent_policy) if opponent_policy is not None else Agent()

    obs, _ = env.reset()
    for step in range(num_steps):
        mask = env.available_actions()
        action = agent.select_action(obs, mask)
        next_obs, reward, done, _, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, done)

        # Opponent's turn
        if not done:
            opp_mask = env.available_actions()
            opp_action = opponent.step(next_obs, Amask=opp_mask)
            next_obs, reward, done, _, _ = env.step(opp_action)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return agent.get_greedy_policy()
