"""Meta-game utilities for PSRO — gamescape and exploitability."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .agent import Agent


def _workers_for(num_workers: int | None, tasks: int) -> int:
    """Worker count for parallel match-up evaluation (1 = serial)."""
    if num_workers == 1:
        return 1
    return min(num_workers or (os.cpu_count() or 4), tasks)


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
            # Route by the environment's own current-player query — not by
            # turn parity (C-10): chance nodes, skipped turns and
            # multi-action turns all break a ``steps % 2`` assumption.
            # Generic Gym-like envs without such a query keep the parity
            # fallback (correct for strictly alternating two-player games).
            current = getattr(env, "get_current_player", lambda: None)()
            players = getattr(env, "players", None)
            if current is not None and players:
                actor = p1 if current == players[0] else p2
            else:
                actor = p1 if steps % 2 == 0 else p2
            mask = env.available_actions()
            action = actor.step(obs, amask=mask)

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
    num_workers: int | None = None,
) -> np.ndarray:
    """Compute or incrementally expand the policy payoff matrix.

    Parameters
    ----------
    env : GymAdapter
        Environment used to evaluate policy match-ups.  Needs a
        ``clone()`` method for parallel evaluation.
    pi : list of np.ndarray
        Policies, each with shape ``(state_dim, action_dim)``.
    Ne : int
        Evaluation episodes per new match-up.
    previous : np.ndarray, optional
        Previously computed square payoff matrix. Existing entries are copied,
        so only match-ups involving newly added policies are evaluated.
    num_workers : int, optional
        Parallel evaluation threads (None/0 = auto, 1 = serial).

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

    # Old-vs-old entries have already been copied. Start at the first
    # new policy, while still keeping j above the diagonal.
    pairs = [(i, j) for i in range(n) for j in range(max(i + 1, previous_size), n)]

    def eval_pair(env_local, ij: tuple[int, int]) -> float:
        i, j = ij
        return estimate_reward(env_local, Ne, Agent(pi[i]), Agent(pi[j]))

    # Parallel match-up evaluation (audit 3.6: match-ups were strictly
    # serial).  Each worker gets its own env clone and its own Agent
    # instances — no shared mutable state.  Envs without clone() (e.g.
    # custom test doubles) stay serial.
    env_factory = getattr(env, "clone", None)
    if env_factory is not None and num_workers != 1 and len(pairs) > 1:
        workers = _workers_for(num_workers, len(pairs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            payoffs = list(
                tqdm(
                    pool.map(lambda ij: eval_pair(env_factory(), ij), pairs),
                    total=len(pairs),
                    desc="Gamescape",
                    position=1,
                    leave=False,
                )
            )
    else:
        payoffs = [eval_pair(env, ij) for ij in tqdm(pairs, desc="Gamescape", position=1, leave=False)]

    for (i, j), payoff in zip(pairs, payoffs):
        payoff_matrix[i, j] = payoff
        payoff_matrix[j, i] = -payoff

    return payoff_matrix


def exploitability(
    env,
    nash_pi: np.ndarray,
    pi: list[np.ndarray],
    Ne: int = 50,  # noqa: N803 — preserved for API compatibility
    num_workers: int | None = None,
) -> float:
    """Compute the exploitability of a Nash mixture.

    Parameters
    ----------
    env : GymAdapter
        Needs a ``clone()`` method for parallel evaluation.
    nash_pi : np.ndarray
        Nash mixture policy, shape ``(state_dim, action_dim)``.
    pi : list of np.ndarray
        Policy pool.
    Ne : int
        Episodes per exploitability estimate.
    num_workers : int, optional
        Parallel evaluation threads (None/0 = auto, 1 = serial).

    Returns
    -------
    float
        Average exploitability.
    """
    env_factory = getattr(env, "clone", None)
    if env_factory is not None and num_workers != 1 and len(pi) > 1:
        workers = _workers_for(num_workers, len(pi))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(
                tqdm(
                    pool.map(
                        lambda i: estimate_reward(env_factory(), Ne, Agent(pi[i]), Agent(nash_pi)),
                        range(len(pi)),
                    ),
                    total=len(pi),
                    desc="Exploitability",
                    position=1,
                    leave=False,
                )
            )
    else:
        nash_agent = Agent(nash_pi)
        values = [
            estimate_reward(env, Ne, Agent(pi[i]), nash_agent)
            for i in tqdm(range(len(pi)), desc="Exploitability", position=1, leave=False)
        ]
    return sum(max(v, 0.0) for v in values) / len(pi)
