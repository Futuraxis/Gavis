"""Agent classes for PSRO.

The original code had a missing ``Agent`` dependency — this file
provides both the policy Agent and the Q-learning agent.
"""

from __future__ import annotations

import numpy as np


class Agent:
    """A policy agent that acts according to a fixed tabular policy.

    Parameters
    ----------
    policy : np.ndarray
        Shape ``(state_dim, action_dim)`` — one-hot or prob-distribution
        per state.
    """

    def __init__(self, policy: np.ndarray | None = None):
        self.policy = policy  # (state_dim, action_dim)
        self._rng = np.random.RandomState()

    def step(self, obs: int, Amask: np.ndarray | None = None) -> int:
        """Select an action given the observation index.

        Parameters
        ----------
        obs : int
            Encoded state index (0 … state_dim-1).
        Amask : np.ndarray, optional
            Boolean mask of legal actions.

        Returns
        -------
        int
            Action index.
        """
        if self.policy is None:
            # Random policy
            if Amask is not None:
                legal = np.where(Amask)[0]
                return int(self._rng.choice(legal)) if len(legal) > 0 else 0
            return int(self._rng.randint(9))

        probs = self.policy[obs].copy()
        if Amask is not None:
            probs = probs * Amask.astype(float)
            if probs.sum() == 0:
                legal = np.where(Amask)[0]
                return int(self._rng.choice(legal)) if len(legal) > 0 else 0
            probs /= probs.sum()
        return int(self._rng.choice(len(probs), p=probs))

    def reset_rng(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)


class TabularQAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ):
        self.Q = np.random.randn(state_dim, action_dim) * 1e-2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self._rng = np.random.RandomState()

    def select_action(self, obs: int, mask: np.ndarray | None = None) -> int:
        """Epsilon-greedy action selection."""
        if self._rng.random() < self.epsilon:
            if mask is not None:
                legal = np.where(mask)[0]
                return int(self._rng.choice(legal)) if len(legal) > 0 else 0
            return int(self._rng.randint(self.action_dim))
        # Greedy
        q_vals = self.Q[obs].copy()
        if mask is not None:
            q_vals[~mask] = -1e9
        return int(np.argmax(q_vals))

    def update(self, obs: int, action: int, reward: float, next_obs: int, done: bool) -> None:
        """Q-learning update."""
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_obs])
        self.Q[obs, action] += self.alpha * (target - self.Q[obs, action])

    def get_greedy_policy(self) -> np.ndarray:
        """Return one-hot greedy policy derived from Q."""
        return np.eye(self.action_dim)[self.Q.argmax(-1)]

    def reset_rng(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)
