"""Experience buffers for the MARL solvers.

- ``ReplayBuffer``: off-policy replay for QMix / MAAC, storing
  ``Transition`` records (one acting agent per entry).
- ``HAPPOTrajectories``: on-policy, per-agent rollout lists with the same
  GAE math as ``ppo/rollout_buffer.py`` but computed independently for
  each agent's own subsequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .env import Transition

# ── Off-policy replay (QMix / MAAC) ────────────────────────────────


@dataclass(slots=True)
class ReplayBatch:
    """A sampled batch of transitions (all tensors on ``device``)."""

    obs: torch.Tensor  # (B, obs_dim)
    masks: torch.Tensor  # (B, action_dim) float32 legal masks
    actions: torch.Tensor  # (B,) long
    rewards: torch.Tensor  # (B,)
    dones: torch.Tensor  # (B,) float32
    global_state: torch.Tensor  # (B, global_dim)
    player_idx: torch.Tensor  # (B,) long
    next_obs: torch.Tensor  # (B, obs_dim) acting player's view of s'
    next_global_state: torch.Tensor  # (B, global_dim)
    next_masks: torch.Tensor  # (B, action_dim) legal mask at s'


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer for off-policy MARL solvers."""

    def __init__(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._items: list[Transition] = []
        self._pos = 0

    def push(self, transition: Transition) -> None:
        if len(self._items) < self._capacity:
            self._items.append(transition)
        else:
            self._items[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Uniform sample of ``batch_size`` transitions (no replacement)."""
        idx = np.random.choice(len(self._items), size=min(batch_size, len(self._items)), replace=False)
        items = [self._items[i] for i in idx]

        def stack(arrs: list[np.ndarray], dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(np.stack(arrs), dtype=dtype, device=device)

        return ReplayBatch(
            obs=stack([t.obs for t in items], torch.float32),
            masks=stack([t.mask for t in items], torch.float32),
            actions=torch.as_tensor([t.action for t in items], dtype=torch.long, device=device),
            rewards=torch.as_tensor([t.reward for t in items], dtype=torch.float32, device=device),
            dones=torch.as_tensor([1.0 if t.done else 0.0 for t in items], dtype=torch.float32, device=device),
            global_state=stack([t.global_state for t in items], torch.float32),
            player_idx=torch.as_tensor([t.player_idx for t in items], dtype=torch.long, device=device),
            next_obs=stack([t.next_obs for t in items], torch.float32),
            next_global_state=stack([t.next_global_state for t in items], torch.float32),
            next_masks=stack([t.next_mask for t in items], torch.float32),
        )

    def __len__(self) -> int:
        return len(self._items)


# ── On-policy per-agent trajectories (HAPPO) ───────────────────────


@dataclass(slots=True)
class AgentBatch:
    obs: torch.Tensor  # (B, obs_dim)
    masks: torch.Tensor  # (B, action_dim)
    actions: torch.Tensor  # (B,)
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    global_state: torch.Tensor  # (B, global_dim)


class HAPPOTrajectories:
    """Per-agent rollout storage with per-agent GAE computation."""

    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        player_idx: int,
        *,
        obs: np.ndarray,
        mask: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        next_value: float,
        global_state: np.ndarray,
    ) -> None:
        self.obs[player_idx].append(np.asarray(obs, dtype=np.float32))
        self.masks[player_idx].append(np.asarray(mask, dtype=np.float32))
        self.actions[player_idx].append(int(action))
        self.log_probs[player_idx].append(float(log_prob))
        self.rewards[player_idx].append(float(reward))
        self.dones[player_idx].append(bool(done))
        self.values[player_idx].append(float(value))
        self.next_values[player_idx].append(float(next_value))
        self.global_states[player_idx].append(np.asarray(global_state, dtype=np.float32))

    def compute_returns_and_advantages(self, player_idx: int, gamma: float, gae_lambda: float) -> None:
        """GAE over one agent's own subsequence (PPO math, per agent)."""
        rewards = self.rewards[player_idx]
        size = len(rewards)
        advantages = np.zeros(size, dtype=np.float32)
        returns = np.zeros(size, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(size)):
            mask = 0.0 if self.dones[player_idx][step] else 1.0
            delta = rewards[step] + gamma * self.next_values[player_idx][step] * mask - self.values[player_idx][step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = gae + self.values[player_idx][step]
        self.advantages[player_idx] = advantages
        self.returns[player_idx] = returns

    def iter_minibatches(self, player_idx: int, batch_size: int, device: torch.device) -> Iterator[AgentBatch]:
        """Shuffle one agent's transitions into minibatches."""
        n = len(self.obs[player_idx])
        if n == 0:
            return
        indices = np.arange(n)
        np.random.shuffle(indices)

        def as_tensor(arrs: list, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(np.asarray(arrs), dtype=dtype, device=device)

        obs = as_tensor(self.obs[player_idx], torch.float32)
        masks = as_tensor(self.masks[player_idx], torch.float32)
        actions = as_tensor(self.actions[player_idx], torch.long)
        old_log_probs = as_tensor(self.log_probs[player_idx], torch.float32)
        returns = as_tensor(self.returns[player_idx], torch.float32)
        advantages = as_tensor(self.advantages[player_idx], torch.float32)
        global_state = as_tensor(self.global_states[player_idx], torch.float32)

        for start in range(0, n, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield AgentBatch(
                obs=obs[batch_indices],
                masks=masks[batch_indices],
                actions=actions[batch_indices],
                old_log_probs=old_log_probs[batch_indices],
                returns=returns[batch_indices],
                advantages=advantages[batch_indices],
                global_state=global_state[batch_indices],
            )

    def clear(self) -> None:
        self.obs: dict[int, list[np.ndarray]] = {}
        self.masks: dict[int, list[np.ndarray]] = {}
        self.actions: dict[int, list[int]] = {}
        self.log_probs: dict[int, list[float]] = {}
        self.rewards: dict[int, list[float]] = {}
        self.dones: dict[int, list[bool]] = {}
        self.values: dict[int, list[float]] = {}
        self.next_values: dict[int, list[float]] = {}
        self.global_states: dict[int, list[np.ndarray]] = {}
        self.advantages: dict[int, np.ndarray] = {}
        self.returns: dict[int, np.ndarray] = {}

    def ensure_agent(self, player_idx: int) -> None:
        """Initialize the per-agent lists on first use."""
        for attr in (
            "obs",
            "masks",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "values",
            "next_values",
            "global_states",
        ):
            getattr(self, attr).setdefault(player_idx, [])

    def __len__(self) -> int:
        return sum(len(v) for v in self.obs.values())
