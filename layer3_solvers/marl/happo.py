"""HAPPOSolver — Heterogeneous-Agent PPO (sequential updates).

Each agent owns an actor (on its own observation) and a critic (on the
joint state).  After every episode the agents are updated **in random
order**, one at a time, with the standard clipped PPO surrogate — the
HAPPO sequential-update scheme that guarantees strict monotonic
improvement even with heterogeneous policies.  GAE is computed per
agent over its own subsequence of decisions.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter, State

from ..base import SolverBase, SolverConfig, SolverMetrics
from .action_space import ActionSpace
from .buffers import HAPPOTrajectories
from .encoders import GameEncoder
from .env import resolve_device, resolve_players, run_episode
from .networks import MLPActor, MLPCritic


@dataclass
class HAPPOConfig(SolverConfig):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32
    hidden_dim: int = 128


class HAPPOSolver(SolverBase):
    """HAPPO solver for any ``SolverAdapter`` (moon_chess / mahjong / texas)."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None):
        super().__init__(adapter, config or HAPPOConfig())
        cfg = self.config
        self._players = resolve_players(adapter)
        self._encoder = GameEncoder.build_from_adapter(adapter, self._players)
        self._action_space = ActionSpace.build_from_adapter(adapter)
        self._player_idx = {p: i for i, p in enumerate(self._players)}
        self.device = resolve_device(cfg.device)

        obs_dim = self._encoder.obs_dim
        action_dim = self._action_space.dim
        self._actors = nn.ModuleDict({p: MLPActor(obs_dim, action_dim, cfg.hidden_dim) for p in self._players}).to(
            self.device
        )
        self._critics = nn.ModuleDict(
            {p: MLPCritic(self._encoder.global_dim, cfg.hidden_dim) for p in self._players}
        ).to(self.device)

        self._optimizer = torch.optim.Adam(
            [*self._actors.parameters(), *self._critics.parameters()], lr=cfg.learning_rate
        )
        self._traj = HAPPOTrajectories()
        self._rng = random.Random(cfg.seed)

    @property
    def name(self) -> str:
        return f"HAPPO({len(self._players)}p, dim={self._action_space.dim})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        """Greedy masked-argmax of the current player's actor."""
        player = self.adapter.get_current_player(state)
        if player is None or player not in self._player_idx:
            return None
        legal = self.adapter.get_legal_actions(state)
        if not legal:
            return None
        mask = self._action_space.legal_mask(state)
        with torch.no_grad():
            logits = self._actors[player](
                torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
            )
            masked = logits.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
            idx = int(masked.argmax(dim=1).item())
        return self._action_space.action_from_index(idx, legal)

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """Train HAPPO via self-play over all agents.

        Parameters
        ----------
        episodes : int
        verbose : bool, optional
        """
        verbose = kwargs.get("verbose", False)
        wins = 0
        total_payoff = 0.0
        total_steps = 0

        for ep in range(episodes):
            traj = run_episode(
                self.adapter,
                self._players,
                self._rng,
                self._encoder,
                self._action_space,
                self._select_train,
                self._eval_next,
            )
            total_steps += len(traj.transitions)
            for t in traj.transitions:
                self._traj.ensure_agent(t.player_idx)
                self._traj.add(
                    t.player_idx,
                    obs=t.obs,
                    mask=t.mask,
                    action=t.action,
                    log_prob=t.log_prob,
                    reward=t.reward,
                    done=t.done,
                    value=t.value,
                    next_value=t.next_value,
                    global_state=t.global_state,
                )
            if self._players[0] in traj.payoffs:
                payoff0 = traj.payoffs[self._players[0]]
                total_payoff += payoff0
                if payoff0 > 0:
                    wins += 1

            self._update_all_agents()

            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                win_pct = wins / (ep + 1) * 100
                print(f"  HAPPO ep {ep + 1:4d}/{episodes}  win={win_pct:5.1f}%")

        return SolverMetrics(
            episodes=episodes,
            win_rate=wins / max(1, episodes),
            avg_return=total_payoff / max(1, episodes),
            extra={"steps": total_steps},
        )

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actors": self._actors.state_dict(),
                "critics": self._critics.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "config": asdict(self.config),
            },
            target,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self._actors.load_state_dict(checkpoint["actors"])
        self._critics.load_state_dict(checkpoint["critics"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])

    # ── Internal ────────────────────────────────────────────────────

    def _select_train(self, player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
        """Sample from the acting agent's actor, evaluate its critic."""
        player = self._players[player_idx]
        obs = torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
        logits = self._actors[player](obs)
        masked = logits.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
        global_state = torch.as_tensor(self._encoder.encode_global(state), device=self.device).unsqueeze(0)
        value = float(self._critics[player](global_state).item())
        return action, {"log_prob": log_prob, "value": value}

    def _eval_next(self, state: State, player_idx: int) -> float:
        """Critic value of the successor state for the acting player."""
        player = self._players[player_idx]
        global_state = torch.as_tensor(self._encoder.encode_global(state), device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self._critics[player](global_state).item())

    def _update_all_agents(self) -> None:
        """Sequential per-agent PPO updates in random order (HAPPO core)."""
        cfg = self.config
        order = list(self._players)
        self._rng.shuffle(order)
        for agent in order:
            aidx = self._player_idx[agent]
            if len(self._traj.obs.get(aidx, [])) == 0:
                continue
            self._traj.compute_returns_and_advantages(aidx, cfg.gamma, cfg.gae_lambda)
            adv = self._traj.advantages[aidx]
            if adv.size == 0:
                continue
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            self._traj.advantages[aidx] = adv

            for _ in range(cfg.update_epochs):
                for batch in self._traj.iter_minibatches(aidx, cfg.minibatch_size, self.device):
                    logits = self._actors[agent](batch.obs)
                    masked = logits.masked_fill(batch.masks == 0, -1e9)
                    dist = torch.distributions.Categorical(logits=masked)
                    new_log_probs = dist.log_prob(batch.actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - batch.old_log_probs)
                    unclipped = ratio * batch.advantages
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * batch.advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    values = self._critics[agent](batch.global_state)
                    value_loss = nn.functional.mse_loss(values, batch.returns)
                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                    self._optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [*self._actors[agent].parameters(), *self._critics[agent].parameters()],
                        cfg.max_grad_norm,
                    )
                    self._optimizer.step()
        self._traj.clear()
