"""Actor-Critic network definition."""

from __future__ import annotations

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """Simple shared-backbone Actor-Critic network.

    forward() 要求 state 带 batch 维，例如 (N, input_dim)。
    输出 (logits, value)：
      - logits 形状 (N, action_dim)
      - value  形状 (N,)
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                "input_dim/action_dim/hidden_dim must be positive, "
                f"got input_dim={input_dim}, action_dim={action_dim}, "
                f"hidden_dim={hidden_dim}"
            )
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self._apply_initialization()

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape[-1] != self.input_dim:
            raise ValueError(
                f"ActorCriticNetwork expected input_dim={self.input_dim}, "
                f"but got last dim={state.shape[-1]} (state.shape={tuple(state.shape)})"
            )
        features = self.backbone(state)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def _apply_initialization(self) -> None:
        """正交初始化：主干 gain=√2，策略头 0.01，价值头 1.0。"""
        gain = nn.init.calculate_gain("relu")
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)
