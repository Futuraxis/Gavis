"""Networks for the MARL solvers (QMix / HAPPO / MAAC).

All networks are plain ``nn.Module`` MLPs mirroring ``ppo/networks.py``.
The QMix mixing network and the MAAC attention critic implement the two
centralized components of each algorithm; per-agent modules (Q-net,
actor, critic) are held in ``nn.ModuleDict`` by the solvers, one entry
per player.

--- Turn-based adaptation ---

Gavis games are turn-based: only one agent acts per timestep.  Central
networks therefore receive an ``acting_onehot`` mask (one-hot over
agents, from the transition's ``player_idx``) and zero out the Q /
contribution of non-acting agents; the hypernetwork / attention still
see the full joint state.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPActor(nn.Module):
    """Per-agent policy: obs → action logits."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))


class MLPCritic(nn.Module):
    """Per-agent value network on the joint state (HAPPO critic)."""

    def __init__(self, global_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(global_state)).squeeze(-1)


class QMixQNet(nn.Module):
    """Per-agent Q-network: obs → Q values over the action space."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))


class MixingNetwork(nn.Module):
    """QMix mixing network: joint state → weights of a monotone mixer.

    ``Q_tot = w2 · relu(w1 · Q + b1) + b2`` with ``abs()`` on the weights
    to guarantee monotonicity (the QMIX core).  Non-acting agents'
    Q-values are zeroed via ``acting_onehot`` so only the acting agent's
    contribution is mixed; the hypernetwork conditions on the full joint
    state (which carries the turn one-hot).
    """

    def __init__(self, global_dim: int, n_agents: int) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(global_dim, n_agents * n_agents)
        self.hyper_b1 = nn.Linear(global_dim, n_agents)
        self.hyper_w2 = nn.Linear(global_dim, n_agents)
        self.hyper_b2 = nn.Linear(global_dim, 1)

    def forward(
        self,
        global_state: torch.Tensor,
        q: torch.Tensor,
        acting_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Mix per-agent Q values into a joint Q.

        Parameters
        ----------
        global_state : (B, global_dim)
        q : (B, n_agents)  — per-agent Q at the chosen action
        acting_onehot : (B, n_agents)  — one-hot over the acting agent
        """
        batch = global_state.shape[0]
        n = self.n_agents
        w1 = torch.abs(self.hyper_w1(global_state).view(batch, n, n))
        b1 = self.hyper_b1(global_state).view(batch, n, 1)
        w2 = torch.abs(self.hyper_w2(global_state).view(batch, 1, n))
        b2 = self.hyper_b2(global_state).view(batch, 1, 1)

        q_eff = (q * acting_onehot).unsqueeze(-1)  # (B, n, 1)
        hidden = torch.relu(torch.bmm(w1, q_eff) + b1)
        q_tot = torch.bmm(w2, hidden) + b2
        return q_tot.squeeze(-1).squeeze(-1)


class AttentionCritic(nn.Module):
    """MAAC critic: per-agent Q with multi-head attention over agents.

    Every agent's (obs, action) is embedded; a shared multi-head
    attention layer mixes the embeddings across agents; each agent's Q
    is read out of the mixed embedding.  Non-acting agents pass a
    dedicated "no action" token so the attention sees the true joint.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_embed = nn.Embedding(action_dim + 1, hidden_dim, padding_idx=action_dim)
        # No-op token 行被 padding_idx 冻结在随机初值、永不学习 — 显式置零，
        # 让"未行动"标记是中性向量而非任意随机向量。
        with torch.no_grad():
            self.action_embed.weight.data[action_dim] = 0.0
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        acted: torch.Tensor,
    ) -> torch.Tensor:
        """Per-agent Q values.

        Parameters
        ----------
        obs : (B, n_agents, obs_dim)
        actions : (B, n_agents)  — action dims; ``action_dim`` = no-action token
        acted : (B, n_agents)  — 1.0 for agents that acted
        """
        action_emb = self.action_embed(actions)  # (B, n, d)
        obs_emb = self.obs_embed(obs)  # (B, n, d)
        emb = action_emb + obs_emb
        attn, _ = self.attention(emb, emb, emb)
        q = self.q_head(attn).squeeze(-1)  # (B, n_agents)
        return q * acted
