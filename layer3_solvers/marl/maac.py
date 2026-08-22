"""MAACSolver — Multi-Agent Actor-Critic with attention (Iqbal & Sha, 2019).

Each agent owns an actor (sampled during training, greedy at eval); a
shared per-agent attention critic embeds every agent's (obs, action) and
mixes the embeddings with multi-head attention before reading out each
agent's Q.  Training is off-policy (replay buffer) with soft target
updates and SAC-style entropy regularization at a fixed temperature.

--- Turn-based adaptation ---

Each replayed transition carries one acting agent.  The critic sees the
full joint state (all agents' observation slots, reconstructed from the
concatenated ``global_state``) with a no-op action token in the other
slots; the acting agent's Q is read out via a one-hot acting mask.  The
TD target bootstraps the acting agent's own target policy at the next
state (pymarl-style, as for Hanabi).
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter, State

from ..base import SolverBase, SolverConfig, SolverMetrics
from .action_space import ActionSpace
from .buffers import ReplayBuffer
from .encoders import GameEncoder
from .env import resolve_device, resolve_players, run_episode
from .networks import AttentionCritic, MLPActor


@dataclass
class MAACConfig(SolverConfig):
    gamma: float = 0.99
    learning_rate: float = 3e-4
    tau: float = 0.005  # soft target update
    entropy_temperature: float = 0.1  # fixed SAC-style temperature
    buffer_capacity: int = 50_000
    batch_size: int = 128
    train_interval: int = 4  # gradient steps per episode
    start_learning: int = 500  # min replay transitions before updating
    hidden_dim: int = 128
    attention_heads: int = 4


class MAACSolver(SolverBase):
    """MAAC solver for any ``SolverAdapter`` (moon_chess / mahjong / texas)."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None):
        super().__init__(adapter, config or MAACConfig())
        cfg = self.config
        if cfg.seed is not None:
            # 可复现性（审查 P2-27）：与 QMix 一致，种子化 torch/np 全局 RNG
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        self._players = resolve_players(adapter)
        self._encoder = GameEncoder.build_from_adapter(adapter, self._players)
        self._action_space = ActionSpace.build_from_adapter(adapter)
        self._player_idx = {p: i for i, p in enumerate(self._players)}
        self.device = resolve_device(cfg.device)

        obs_dim = self._encoder.obs_dim
        action_dim = self._action_space.dim
        n_agents = len(self._players)
        self._actors = nn.ModuleDict({p: MLPActor(obs_dim, action_dim, cfg.hidden_dim) for p in self._players}).to(
            self.device
        )
        self._critics = nn.ModuleDict(
            {
                p: AttentionCritic(obs_dim, action_dim, n_agents, cfg.hidden_dim, cfg.attention_heads)
                for p in self._players
            }
        ).to(self.device)
        self._actor_targets = deepcopy(self._actors)
        self._critic_targets = deepcopy(self._critics)

        self._optimizer = torch.optim.Adam(
            [*self._actors.parameters(), *self._critics.parameters()], lr=cfg.learning_rate
        )
        self._buffer = ReplayBuffer(cfg.buffer_capacity)
        self._rng = random.Random(cfg.seed)
        self._steps = 0

    @property
    def name(self) -> str:
        return f"MAAC({len(self._players)}p, dim={self._action_space.dim})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        """Greedy masked-argmax of the current player's actor (C-08: eval
        must be deterministic — sampling here would understate the trained
        policy's true strength)."""
        player = self.adapter.get_current_player(state)
        if player is None or player not in self._player_idx:
            return None
        legal = self.adapter.get_legal_actions(state)
        if not legal:
            return None
        # 复用已求值的 legal（legal_mask 支持传入，避免第二次引擎求值）
        mask = self._action_space.legal_mask(state, legal)
        with torch.no_grad():
            logits = self._actors[player](
                torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
            )
            masked = logits.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
            idx = int(masked.argmax(dim=1).item())
        return self._action_space.action_from_index(idx, legal)

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """Train MAAC via self-play over all agents.

        Parameters
        ----------
        episodes : int
        verbose : bool, optional
        """
        cfg = self.config
        verbose = kwargs.get("verbose", False)
        wins = 0
        total_payoff = 0.0

        for ep in range(episodes):
            traj = run_episode(
                self.adapter,
                self._players,
                self._rng,
                self._encoder,
                self._action_space,
                self._select_train,
                max_steps=4096,  # 病理局面步数上限（正常对局远低于此）
            )
            for t in traj.transitions:
                self._buffer.push(t)
                self._steps += 1
            if self._players[0] in traj.payoffs:
                payoff0 = traj.payoffs[self._players[0]]
                total_payoff += payoff0
                if payoff0 > 0:
                    wins += 1

            updates = min(len(traj.transitions), cfg.train_interval)
            for _ in range(updates):
                if len(self._buffer) < cfg.start_learning:
                    break
                self._gradient_step()

            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                win_pct = wins / (ep + 1) * 100
                print(f"  MAAC ep {ep + 1:4d}/{episodes}  win={win_pct:5.1f}%  replay={len(self._buffer)}")

        return SolverMetrics(
            episodes=episodes,
            win_rate=wins / max(1, episodes),
            avg_return=total_payoff / max(1, episodes),
            extra={"steps": self._steps},
        )

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actors": self._actors.state_dict(),
                "actor_targets": self._actor_targets.state_dict(),
                "critics": self._critics.state_dict(),
                "critic_targets": self._critic_targets.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "config": asdict(self.config),
                "steps": self._steps,
            },
            target,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._actors.load_state_dict(checkpoint["actors"])
        self._actor_targets.load_state_dict(checkpoint["actor_targets"])
        self._critics.load_state_dict(checkpoint["critics"])
        self._critic_targets.load_state_dict(checkpoint["critic_targets"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._steps = checkpoint.get("steps", 0)

    # ── Internal ────────────────────────────────────────────────────

    def _select_train(self, player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
        """Sample the acting agent's actor during training."""
        player = self._players[player_idx]
        logits = self._actors[player](
            torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
        )
        masked = logits.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        return int(dist.sample().item()), {}

    def _gradient_step(self) -> None:
        cfg = self.config
        batch = self._buffer.sample(cfg.batch_size, self.device)
        n_agents = len(self._players)
        acting = torch.eye(n_agents, device=self.device)[batch.player_idx]  # (B, N)

        # Joint inputs: obs slots from the concatenated global state
        obs_slots = batch.global_state.view(-1, n_agents, self._encoder.obs_dim)
        next_slots = batch.next_global_state.view(-1, n_agents, self._encoder.obs_dim)
        no_op = self._action_space.dim  # padding token for non-acting agents
        actions = no_op * torch.ones_like(batch.actions).unsqueeze(-1).expand(-1, n_agents)
        actions.scatter_(-1, batch.player_idx.unsqueeze(-1), batch.actions.unsqueeze(-1))

        # ── Critic ────────────────────────────────────────────────
        q_acting = torch.stack(
            [self._critics[p](obs_slots, actions, acting)[:, i] for i, p in enumerate(self._players)], dim=1
        )  # (B, N) — critic p reads slot p
        q_taken = (q_acting * acting).sum(-1)  # (B,)

        with torch.no_grad():
            # Bootstrap: acting agent's own target policy at s'
            next_actions = no_op * torch.ones_like(batch.actions).unsqueeze(-1).expand(-1, n_agents)
            next_log_probs = torch.zeros_like(batch.actions, dtype=torch.float32)
            for i, p in enumerate(self._players):
                sel = batch.player_idx == i
                if not sel.any():
                    continue
                logits_t = self._actor_targets[p](batch.next_obs[sel])
                masked_t = logits_t.masked_fill(batch.next_masks[sel] == 0, -1e9)
                dist_t = torch.distributions.Categorical(logits=masked_t)
                a_t = dist_t.sample()
                next_actions[sel, i] = a_t
                next_log_probs[sel] = dist_t.log_prob(a_t)
            q_next = torch.stack(
                [self._critic_targets[p](next_slots, next_actions, acting)[:, i] for i, p in enumerate(self._players)],
                dim=1,
            )  # (B, N)
            q_next = (q_next * acting).sum(-1)  # (B,)
            soft_q = q_next - cfg.entropy_temperature * next_log_probs
            target = batch.rewards + cfg.gamma * (1.0 - batch.dones) * soft_q

        critic_loss = nn.functional.mse_loss(q_taken, target)
        self._optimizer.zero_grad()
        critic_loss.backward()
        self._optimizer.step()

        # ── Actor (REINFORCE with entropy regularization) ──────────
        # Re-sample ``a ~ π(·|s)`` and score it (C-07).  The old
        # ``-(q_online − τ·log π)`` had two defects (审查 P1-14): with a
        # sampled discrete index the Q term carries no actor gradient (the
        # loss degenerated to a zero-mean log-prob random walk), and
        # ``q_online`` was not detached, so the shared optimizer's actor
        # step updated the critic a second time.  REINFORCE
        # ``∇[log π(a) · (Q − τ·log π(a))]`` fixes both.
        # 各 agent 子集按样本数加权（轮次制下不同 agent 的样本数天然不均衡，
        # 等权平均会让样本少的 agent 权重偏高）。
        actor_loss_sum = torch.zeros((), device=self.device)
        actor_total_n = 0
        for i, p in enumerate(self._players):
            sel = batch.player_idx == i
            n = int(sel.sum())
            if n == 0:
                continue
            logits = self._actors[p](batch.obs[sel])
            masked = logits.masked_fill(batch.masks[sel] == 0, -1e9)
            dist = torch.distributions.Categorical(logits=masked)
            a_new = dist.sample()
            log_prob = dist.log_prob(a_new)
            # Joint action tensor with the fresh action in the acting slot.
            new_joint = actions[sel].clone()
            new_joint[:, i] = a_new
            q_online = self._critics[p](obs_slots[sel], new_joint, acting[sel])[:, i]
            actor_loss_sum = actor_loss_sum - (log_prob * (q_online.detach() - cfg.entropy_temperature * log_prob)).sum()
            actor_total_n += n
        if actor_total_n > 0:
            actor_loss = actor_loss_sum / actor_total_n
            self._optimizer.zero_grad()
            actor_loss.backward()
            self._optimizer.step()

        self._soft_update(self._actors, self._actor_targets, cfg.tau)
        self._soft_update(self._critics, self._critic_targets, cfg.tau)

    @staticmethod
    def _soft_update(net: nn.ModuleDict, target: nn.ModuleDict, tau: float) -> None:
        """Polyak-averaged target update."""
        for n, t in zip(net.parameters(), target.parameters()):
            t.data.copy_(tau * n.data + (1.0 - tau) * t.data)
