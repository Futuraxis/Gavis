"""QMixSolver — Q-learning with a monotone mixing network (CTDE).

Each agent owns a Q-network; a mixing network (hypernetwork from the
joint state) combines per-agent Q values into a joint Q with
``abs()``-guaranteed monotonicity.  Training is centralized (mixing
network sees the joint state) while execution is decentralized (each
agent acts greedily on its own masked Q).

--- Turn-based adaptation ---

Only one agent acts per timestep, so each replayed transition carries
the acting agent's ``player_idx``; the mixer zeroes the other agents'
Q-contributions via a one-hot acting mask, and the TD target bootstraps
with the acting agent's own Q on the next state (pymarl-style, as used
for Hanabi — sequential games where the next state belongs to another
player's turn).
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
from .networks import MixingNetwork, QMixQNet


@dataclass
class QMixConfig(SolverConfig):
    gamma: float = 0.99
    learning_rate: float = 1e-3
    buffer_capacity: int = 50_000
    batch_size: int = 128
    target_update_interval: int = 200  # hard-copy steps
    double_q: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    train_interval: int = 4  # gradient steps per episode
    start_learning: int = 500  # min replay transitions before updating
    hidden_dim: int = 128
    max_grad_norm: float = 1.0


class QMixSolver(SolverBase):
    """QMix solver for any ``SolverAdapter`` (moon_chess / mahjong / texas)."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None):
        super().__init__(adapter, config or QMixConfig())
        cfg = self.config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
            np.random.seed(cfg.seed)
        self._players = resolve_players(adapter)
        self._encoder = GameEncoder.build_from_adapter(adapter, self._players)
        self._action_space = ActionSpace.build_from_adapter(adapter)
        self._player_idx = {p: i for i, p in enumerate(self._players)}
        self.device = resolve_device(cfg.device)

        obs_dim = self._encoder.obs_dim
        action_dim = self._action_space.dim
        n_agents = len(self._players)
        self._q_nets = nn.ModuleDict({p: QMixQNet(obs_dim, action_dim, cfg.hidden_dim) for p in self._players}).to(
            self.device
        )
        self._q_targets = deepcopy(self._q_nets)
        self._mixer = MixingNetwork(self._encoder.global_dim, n_agents).to(self.device)
        self._mixer_target = deepcopy(self._mixer)

        self._optimizer = torch.optim.Adam(
            [*self._q_nets.parameters(), *self._mixer.parameters()], lr=cfg.learning_rate
        )
        self._buffer = ReplayBuffer(cfg.buffer_capacity)
        self._rng = random.Random(cfg.seed)
        self._steps = 0
        self._grad_steps = 0
        self._epsilon = cfg.epsilon_start

    @property
    def name(self) -> str:
        return f"QMix({len(self._players)}p, dim={self._action_space.dim})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        """Greedy masked-argmax of the current player's Q (untrained-safe)."""
        player = self.adapter.get_current_player(state)
        if player is None or player not in self._player_idx:
            return None
        legal = self.adapter.get_legal_actions(state)
        if not legal:
            return None
        # 复用已求值的 legal（legal_mask 支持传入，避免第二次引擎求值）
        mask = self._action_space.legal_mask(state, legal)
        with torch.no_grad():
            q = self._q_nets[player](
                torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
            )
            masked = q.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
            idx = int(masked.argmax(dim=1).item())
        return self._action_space.action_from_index(idx, legal)

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """Train QMix via self-play over all agents.

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

            epsilon_denom = max(1, cfg.epsilon_decay_steps)  # 防止配置为 0 时除零
            self._epsilon = max(
                cfg.epsilon_end,
                cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * self._steps / epsilon_denom,
            )
            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                win_pct = wins / (ep + 1) * 100
                print(
                    f"  QMix ep {ep + 1:4d}/{episodes}  win={win_pct:5.1f}%  "
                    f"eps={self._epsilon:.3f}  replay={len(self._buffer)}"
                )

        return SolverMetrics(
            episodes=episodes,
            win_rate=wins / max(1, episodes),
            avg_return=total_payoff / max(1, episodes),
            extra={"steps": self._steps, "epsilon": self._epsilon},
        )

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_nets": self._q_nets.state_dict(),
                "q_targets": self._q_targets.state_dict(),
                "mixer": self._mixer.state_dict(),
                "mixer_target": self._mixer_target.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "config": asdict(self.config),
                "steps": self._steps,
                "grad_steps": self._grad_steps,
                "epsilon": self._epsilon,
            },
            target,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._q_nets.load_state_dict(checkpoint["q_nets"])
        self._q_targets.load_state_dict(checkpoint["q_targets"])
        self._mixer.load_state_dict(checkpoint["mixer"])
        self._mixer_target.load_state_dict(checkpoint["mixer_target"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._steps = checkpoint.get("steps", 0)
        self._grad_steps = checkpoint.get("grad_steps", 0)
        self._epsilon = checkpoint.get("epsilon", self.config.epsilon_start)

    # ── Internal ────────────────────────────────────────────────────

    def _select_train(self, player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
        """ε-greedy action selection during training."""
        player = self._players[player_idx]
        if self._rng.random() < self._epsilon:
            legal_idx = np.flatnonzero(mask).tolist()
            return int(self._rng.choice(legal_idx)), {}
        with torch.no_grad():
            q = self._q_nets[player](
                torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
            )
            masked = q.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
            return int(masked.argmax(dim=1).item()), {}

    def _gradient_step(self) -> None:
        cfg = self.config
        batch = self._buffer.sample(cfg.batch_size, self.device)
        n_agents = len(self._players)
        acting = torch.eye(n_agents, device=self.device)[batch.player_idx]  # (B, N)

        # Online: per-agent Q at the chosen action, non-acting zeroed
        q = torch.stack([self._q_nets[p](batch.obs) for p in self._players], dim=1)  # (B, N, AD)
        act_idx = batch.actions.unsqueeze(-1).unsqueeze(-1).expand(-1, n_agents, 1)  # (B, N, 1)
        q_chosen = q.gather(-1, act_idx).squeeze(-1)  # (B, N)
        q_tot = self._mixer(batch.global_state, q_chosen, acting)

        # Target: double-Q bootstrap with the acting agent's own network.
        # Illegal actions at s' are masked out before argmax/max so a
        # randomly-initialized Q on an illegal action can never be picked
        # as the bootstrap target (C-09).
        with torch.no_grad():
            invalid_next = batch.next_masks == 0  # (B, AD)
            q_target_next = torch.stack([self._q_targets[p](batch.next_obs) for p in self._players], dim=1)
            q_target_next = q_target_next.masked_fill(invalid_next.unsqueeze(1), -1e9)
            if cfg.double_q:
                q_online_next = torch.stack([self._q_nets[p](batch.next_obs) for p in self._players], dim=1)
                q_online_next = q_online_next.masked_fill(invalid_next.unsqueeze(1), -1e9)
                argmax_next = q_online_next.argmax(dim=-1)  # (B, N)
                q_next = q_target_next.gather(-1, argmax_next.unsqueeze(-1)).squeeze(-1)
            else:
                q_next = q_target_next.max(dim=-1).values
            q_next = (q_next * acting).sum(dim=-1)  # (B,)
            target = batch.rewards + cfg.gamma * (1.0 - batch.dones) * q_next

        loss = nn.functional.mse_loss(q_tot, target)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*self._q_nets.parameters(), *self._mixer.parameters()], cfg.max_grad_norm)
        self._optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % cfg.target_update_interval == 0:
            self._q_targets.load_state_dict(self._q_nets.state_dict())
            self._mixer_target.load_state_dict(self._mixer.state_dict())
