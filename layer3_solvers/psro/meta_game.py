"""Meta-game utilities for PSRO — gamescape and exploitability."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .agent import Agent


def estimate_reward(env, num_episodes: int, p1: Agent, p2: Agent, max_steps: int = 200) -> float:
    """Estimate the expected reward of p1 vs p2 over ``num_episodes``.

    Parameters
    ----------
    env : GymAdapter
        Gym-style environment wrapper.
    num_episodes : int
        Number of episodes to average over.
    p1, p2 : Agent
        The two policies.

    Returns
    -------
    float
        Average reward for p1.
    """
    R = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            mask = env.available_actions()
            action = p1.step(obs, Amask=mask)
            obs, r, done, _, _ = env.step(action)
            R += r
            steps += 1
    return R / num_episodes


def gamescape(env, pi: list[np.ndarray], Ne: int = 10) -> np.ndarray:
    """Compute the payoff matrix for a set of policies.

    Parameters
    ----------
    env : GymAdapter
    pi : list of np.ndarray
        List of policies, each shape ``(state_dim, action_dim)``.
    Ne : int
        Episodes per match-up.

    Returns
    -------
    np.ndarray, shape (len(pi), len(pi))
        Payoff matrix (row player = pi[i], column player = pi[j]).
    """
    n = len(pi)
    R = np.zeros((n, n))
    for i in tqdm(range(n), desc="Gamescape", position=1, leave=False):
        for j in range(n):
            if j <= i:
                R[i, j] = -R[j, i]
                continue
            R[i, j] = estimate_reward(env, Ne, Agent(pi[i]), Agent(pi[j]))
    return R


def exploitability(env, nash_pi: np.ndarray, pi: list[np.ndarray], Ne: int = 50) -> float:
    """Compute the exploitability of a Nash mixture.

    Parameters
    ----------
    env : GymAdapter
    nash_pi : np.ndarray
        Nash mixture policy, shape ``(state_dim, action_dim)``.
    pi : list of np.ndarray
        Policy pool.
    Ne : int
        Episodes per exploitability estimate.

    Returns
    -------
    float
        Average exploitability.
    """
    nash_agent = Agent(nash_pi)
    total = 0.0
    for i in tqdm(range(len(pi)), desc="Exploitability", position=1, leave=False):
        total += max(estimate_reward(env, Ne, Agent(pi[i]), nash_agent), 0)
    return total / len(pi)
