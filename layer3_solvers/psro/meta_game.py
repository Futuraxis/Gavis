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
    total_reward = 0.0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Player one acts on even turns; player two acts on odd turns.
            actor = p1 if steps % 2 == 0 else p2
            mask = env.available_actions()
            action = actor.step(obs, Amask=mask)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

    return total_reward / num_episodes


def gamescape(
    env,
    pi: list[np.ndarray],
    Ne: int = 10,  # noqa: N803 — preserved for API compatibility
    previous: np.ndarray | None = None,
) -> np.ndarray:
    """Compute or incrementally expand the policy payoff matrix.

    Parameters
    ----------
    env : GymAdapter
        Environment used to evaluate policy match-ups.
    pi : list of np.ndarray
        Policies, each with shape ``(state_dim, action_dim)``.
    Ne : int
        Evaluation episodes per new match-up.
    previous : np.ndarray, optional
        Previously computed square payoff matrix. Existing entries are copied,
        so only match-ups involving newly added policies are evaluated.

    Returns
    -------
    np.ndarray
        Antisymmetric payoff matrix with shape ``(len(pi), len(pi))``.
    """
    n = len(pi)
    payoff_matrix = np.zeros((n, n))
    previous_size = 0

    if previous is not None:
        if previous.ndim != 2 or previous.shape[0] != previous.shape[1]:
            raise ValueError("previous payoff matrix must be square")

        previous_size = previous.shape[0]
        if previous_size > n:
            raise ValueError("previous payoff matrix is larger than the policy pool")

        payoff_matrix[:previous_size, :previous_size] = previous

    for i in tqdm(range(n), desc="Gamescape", position=1, leave=False):
        # Old-vs-old entries have already been copied. Start at the first
        # new policy, while still keeping j above the diagonal.
        first_opponent = max(i + 1, previous_size)

        for j in range(first_opponent, n):
            payoff = estimate_reward(
                env,
                Ne,
                Agent(pi[i]),
                Agent(pi[j]),
            )
            payoff_matrix[i, j] = payoff
            payoff_matrix[j, i] = -payoff

    return payoff_matrix


def exploitability(
    env,
    nash_pi: np.ndarray,
    pi: list[np.ndarray],
    Ne: int = 50,  # noqa: N803 — preserved for API compatibility
) -> float:
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
