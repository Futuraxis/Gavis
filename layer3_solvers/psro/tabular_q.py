"""Tabular Q-learning best-response solver for PSRO.

Given an environment and a fixed opponent policy, train a new
policy via tabular Q-learning.
"""

from __future__ import annotations

import numpy as np

from .agent import Agent, TabularQAgent


def tabular_q_best_response(
    env,
    num_steps: int = 10000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.99,
    opponent_policy: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Train a best-response policy via tabular Q-learning.

    Parameters
    ----------
    env : GymAdapter
    num_steps : int
        Total training steps.
    epsilon, alpha, gamma : float
        Q-learning hyper-parameters.
    opponent_policy : np.ndarray, optional
        Fixed opponent policy. If None, uses random.
    seed : int, optional
        Seed for the Q init / exploration RNGs (审查 P2-23).

    Returns
    -------
    np.ndarray, shape (state_dim, action_dim)
        Greedy policy (one-hot) derived from trained Q-table.
    """
    obs_dim = env.observation_space.n if hasattr(env.observation_space, "n") else env.observation_space.shape[0]
    n_actions = env.action_space.n

    if seed is not None:
        np.random.seed(seed)  # Q-table init uses the global RNG
    agent = TabularQAgent(obs_dim, n_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent.reset_rng(seed)
    opponent = (
        Agent(opponent_policy, action_dim=n_actions) if opponent_policy is not None else Agent(action_dim=n_actions)
    )
    opponent.reset_rng(seed)

    obs, _ = env.reset()
    for step in range(num_steps):
        mask = env.available_actions()
        action = agent.select_action(obs, mask)
        next_obs, reward, done, _, _ = env.step(action)

        # Opponent's turn, then update on the agent's OWN next decision
        # state.  Updating before the opponent's reply (the old order)
        # bootstrapped Q(s,a) from an opponent-turn state the agent never
        # acts in, and dropped the terminal ±1 payoff of opponent-ending
        # games entirely (审查 P1-1) — the learned "best response"
        # degenerated to 1-ply greediness.
        if not done:
            opp_mask = env.available_actions()
            opp_action = opponent.step(next_obs, amask=opp_mask)
            next_obs, r2, done, _, _ = env.step(opp_action)
            # Fold the opponent's immediate reward: nonzero only when the
            # opponent's move ends the game (row player's view).
            reward = reward + gamma * r2
        agent.update(obs, action, reward, next_obs, done, next_mask=env.available_actions())

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return agent.get_greedy_policy()
