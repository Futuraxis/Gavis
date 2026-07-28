"""Actor-Critic 网络定义。"""

from __future__ import annotations

import torch
from torch import nn


class ActorCriticNetwork(nn.Module):
    """简单共享骨干的 Actor-Critic 网络。"""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value
