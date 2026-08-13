"""PPO rollout buffer with GAE computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass(slots=True)
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Saves PPO trajectories and computes GAE."""

    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        next_value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.action_masks.append(np.asarray(action_mask, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self._tensors_valid = False

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> None:
        size = len(self.rewards)
        advantages = np.zeros(size, dtype=np.float32)
        returns = np.zeros(size, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(size)):
            mask = 0.0 if self.dones[step] else 1.0
            delta = self.rewards[step] + gamma * self.next_values[step] * mask - self.values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = gae + self.values[step]
        self.advantages = advantages
        self.returns = returns
        self._tensors_valid = False

    def iterate_minibatches(self, batch_size: int, device: torch.device) -> Iterator[RolloutBatch]:
        # Convert numpy → tensor ONCE per update (invalidated on add/GAE),
        # not once per epoch — the same data is replayed for every epoch.
        if not self._tensors_valid or self._tensor_device != device:
            self._tensor_device = device
            self._states_t = torch.as_tensor(np.asarray(self.states), dtype=torch.float32, device=device)
            self._actions_t = torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=device)
            self._masks_t = torch.as_tensor(np.asarray(self.action_masks), dtype=torch.float32, device=device)
            self._log_probs_t = torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device)
            self._returns_t = torch.as_tensor(self.returns, dtype=torch.float32, device=device)
            self._advantages_t = torch.as_tensor(self.advantages, dtype=torch.float32, device=device)
            self._values_t = torch.as_tensor(np.asarray(self.values), dtype=torch.float32, device=device)
            self._tensors_valid = True

        indices = np.arange(len(self.states))
        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                states=self._states_t[batch_indices],
                actions=self._actions_t[batch_indices],
                action_masks=self._masks_t[batch_indices],
                old_log_probs=self._log_probs_t[batch_indices],
                returns=self._returns_t[batch_indices],
                advantages=self._advantages_t[batch_indices],
                values=self._values_t[batch_indices],
            )

    def clear(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.action_masks: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.advantages = np.asarray([], dtype=np.float32)
        self.returns = np.asarray([], dtype=np.float32)
        self._tensor_device: torch.device | None = None
        self._tensors_valid = False

    def __len__(self) -> int:
        return len(self.states)
