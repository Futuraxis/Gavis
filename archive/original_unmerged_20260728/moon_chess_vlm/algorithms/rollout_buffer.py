"""PPO rollout buffer。"""

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
    """保存 PPO 训练所需轨迹，并负责计算 GAE。"""

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

    def iterate_minibatches(self, batch_size: int, device: torch.device) -> Iterator[RolloutBatch]:
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        states = torch.as_tensor(np.asarray(self.states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=device)
        action_masks = torch.as_tensor(np.asarray(self.action_masks), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device)
        returns = torch.as_tensor(self.returns, dtype=torch.float32, device=device)
        advantages = torch.as_tensor(self.advantages, dtype=torch.float32, device=device)
        values = torch.as_tensor(np.asarray(self.values), dtype=torch.float32, device=device)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield RolloutBatch(
                states=states[batch_indices],
                actions=actions[batch_indices],
                action_masks=action_masks[batch_indices],
                old_log_probs=old_log_probs[batch_indices],
                returns=returns[batch_indices],
                advantages=advantages[batch_indices],
                values=values[batch_indices],
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

    def __len__(self) -> int:
        return len(self.states)
