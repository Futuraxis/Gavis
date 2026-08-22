"""PPO rollout buffer with GAE computation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional

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


class RolloutBuffer:
    """Saves PPO trajectories and computes GAE."""

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self._rng = rng or random.Random()
        self.max_size = max_size
        self.clear()

    @property
    def advantages(self) -> np.ndarray:
        return self._advantages

    @advantages.setter
    def advantages(self, value: np.ndarray) -> None:
        self._advantages = value
        self._tensors_valid = False

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @returns.setter
    def returns(self, value: np.ndarray) -> None:
        self._returns = value
        self._tensors_valid = False

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
        if done and next_value != 0.0:
            raise ValueError("done=True 时 next_value 应为 0.0，请检查调用方")
        if self.max_size is not None and len(self.states) >= self.max_size:
            raise RuntimeError(f"RolloutBuffer 已满（max_size={self.max_size}），请先 clear() 或调大容量")
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
        if len(self.states) == 0:
            raise ValueError("RolloutBuffer 为空，无法生成小批量")
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
            self._tensors_valid = True

        indices = np.arange(len(self.states))
        self._rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                states=self._states_t[batch_indices],
                actions=self._actions_t[batch_indices],
                action_masks=self._masks_t[batch_indices],
                old_log_probs=self._log_probs_t[batch_indices],
                returns=self._returns_t[batch_indices],
                advantages=self._advantages_t[batch_indices],
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
