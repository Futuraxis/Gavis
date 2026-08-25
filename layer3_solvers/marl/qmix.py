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
from typing import Any

import numpy as np
import torch
from torch import nn

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance, State

from ..base import SolverBase, SolverConfig, SolverMetrics
from .action_space import ActionSpace
from .buffers import ReplayBuffer
from .encoders import GameEncoder
from .env import resolve_device, resolve_players, run_episode
from .networks import MixingNetwork, QMixQNet
from .opponent_pool import (
    OpponentScheduleConfig,
    OpponentScheduler,
    OpponentSnapshot,
    SelectFn,
    build_selectors,
    eval_vs_random,
)


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
    # ── 训练对手编排（见 marl/opponent_pool.py；反对自博弈震荡、平滑曲线）──
    opponent_enabled: bool = False  # 开启后每局由池对手（冻结快照）陪练
    opponent_mode: str = "pfsp"  # self | uniform | pfsp | curriculum
    opponent_pool_capacity: int = 32
    opponent_checkpoint_interval: int = 25  # 每 N 局冻结一次当前策略入池
    opponent_warmup: int = 0  # 起步 N 局纯自博弈（池空自动退化）
    opponent_pfsp_alpha: float = 1.0  # 胜率权重指数
    opponent_pfsp_floor: float = 0.1  # 权重下限（保证池内全部对手可被抽中）
    opponent_pfsp_priority: str = "win"  # win → p∝win_rate^α；lose → p∝(1−win_rate)^α
    opponent_recency_decay: float = 0.9  # curriculum：p_i ∝ decay^age
    opponent_win_memory: int = 50  # 每对手滚动胜负窗口
    opponent_role_alternate: bool = True  # 2 人局学习器座位逐局轮换
    eval_interval: int = 0  # 每 N 局做一次 vs-random 曲线采样（0=关闭）
    eval_episodes: int = 5  # 每次曲线采样的局数


class QMixSolver(SolverBase):
    """QMix solver for any ``GameEngine`` (moon_chess / mahjong / texas)."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None):
        super().__init__(engine, config or QMixConfig())
        cfg = self.config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
            np.random.seed(cfg.seed)
        self._players = resolve_players(engine)
        self._encoder = GameEncoder.build_from_adapter(engine, self._players)
        self._action_space = ActionSpace.build_from_adapter(engine)
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

        # 训练对手编排：开启时为每个玩家持有一个对手池（见 opponent_pool.py）。
        self._opp: OpponentScheduler | None = None
        if cfg.opponent_enabled:
            self._opp = OpponentScheduler(OpponentScheduleConfig.from_flat(cfg), self._rng, self._players)
        self._shadow_q: dict[int, Any] = {}  # snp.id → 冻结对手 Q 网（懒创建）
        self._win_hist: list[float] = []  # 滚动胜负窗口（训练曲线用）

    @property
    def name(self) -> str:
        return f"QMix({len(self._players)}p, dim={self._action_space.dim})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        """Greedy masked-argmax of the current player's Q (untrained-safe)."""
        player = self.engine.get_current_player(state)
        if player is None or player not in self._player_idx:
            return None
        legal = self.engine.get_legal_actions(state)
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
        """Train QMix — 纯自博弈或对手编排（``opponent_enabled``）.

        Parameters
        ----------
        episodes : int
        verbose : bool, optional

        --- 对手编排模式 ---

        ``opponent_enabled=True`` 时每个玩家持有自己的对手池：每隔
        ``opponent_checkpoint_interval`` 局把当前策略冻结入池；每局轮换
        学习器座位，另一座位执行从池中按 ``opponent_mode`` 采样到的冻结快照
        （其 transitions 不入 buffer，保证 Q 学习只用学习器自身的 on-policy 数据）。
        池为空/warmup 时退化为纯自博弈（与旧行为一致）。
        """
        cfg = self.config
        verbose = kwargs.get("verbose", False)
        wins = 0
        total_payoff = 0.0
        curve_roll: list[dict[str, float | int | str]] = []
        curve_eval: list[dict[str, float | int | str]] = []

        for ep in range(episodes):
            scheduled = self._opp is not None and self._opp.config.mode != "self"
            learner = self._opp.learner_for(ep) if scheduled else None
            opp_snp: OpponentSnapshot | None = None
            if scheduled:
                self._opp.maybe_checkpoint(ep, self._capture_q)
                if learner is not None:
                    opp_snp = self._opp.sample(learner)
            frozen = self._frozen_select(opp_snp) if scheduled and learner is not None and opp_snp is not None else None
            selectors = build_selectors(
                learner if scheduled and learner is not None else None,
                len(self._players),
                self._select_train,
                frozen,
            )
            traj = run_episode(
                self.engine, self._players, self._rng, self._encoder, self._action_space, selectors, max_steps=4096
            )

            # 收集 transitions：编排模式只收学习器座位（保证 on-policy）。
            tracked = learner if (scheduled and learner is not None) else 0
            for t in traj.transitions:
                if t.player_idx != tracked:
                    continue
                self._buffer.push(t)
                self._steps += 1
            payoff = float(traj.payoffs.get(self._players[tracked], 0.0))
            total_payoff += payoff
            if payoff > 0:
                wins += 1
            if scheduled and learner is not None and opp_snp is not None:
                self._opp.record(learner, opp_snp, payoff)

            # 滚动胜率曲线（学习器视角，窗口 50）
            self._win_hist.append(1.0 if payoff > 0 else 0.0)
            if len(self._win_hist) > 50:
                self._win_hist.pop(0)
            if (ep + 1) % 25 == 0 or ep == episodes - 1:
                curve_roll.append({"ep": ep + 1, "roll_win": round(sum(self._win_hist) / len(self._win_hist), 4)})

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

            # vs-random 曲线采样（固定基线；与是否编排无关，基线与编排都可对比）
            if cfg.eval_interval > 0 and (ep + 1) % cfg.eval_interval == 0:
                res = eval_vs_random(
                    self.engine,
                    self._players,
                    self._encoder,
                    self._action_space,
                    self._greedy_select,
                    cfg.eval_episodes,
                    (cfg.seed or 0) + ep,
                )
                curve_eval.append({"ep": ep + 1, **{f"{p}_wr": round(v, 4) for p, v in res.items()}})

            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                win_pct = wins / (ep + 1) * 100
                pool_info = ""
                if scheduled:
                    pool_info = (
                        f"  pool={self._opp.pool_size(self._players[0])}/{self._opp.pool_size(self._players[1])}"
                    )
                print(
                    f"  QMix ep {ep + 1:4d}/{episodes}  win={win_pct:5.1f}%  "
                    f"eps={self._epsilon:.3f}  replay={len(self._buffer)}{pool_info}"
                )

        return SolverMetrics(
            episodes=episodes,
            win_rate=wins / max(1, episodes),
            avg_return=total_payoff / max(1, episodes),
            extra={
                "steps": self._steps,
                "epsilon": self._epsilon,
                "opponent_enabled": self._opp is not None,
                "curve_roll": curve_roll,
                "curve_eval": curve_eval,
            },
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

    def _capture_q(self, player: str) -> dict[str, Any]:
        """冻结指定玩家的 Q 网 state dict（深拷贝张量，供对手池入池）。"""
        return {"q_net": {k: v.detach().clone() for k, v in self._q_nets[player].state_dict().items()}}

    def _frozen_select(self, snp: OpponentSnapshot) -> SelectFn:
        """构造执行对手快照（学习者过去策略）的贪心选择回调。

        影子 Q 网按快照 id 懒创建并加载冻结参数——不触碰在线网，
        无需"换入换出"恢复。
        """
        shadow = self._shadow_q.get(snp.id)
        if shadow is None:
            shadow = QMixQNet(self._encoder.obs_dim, self._action_space.dim, self.config.hidden_dim).to(self.device)
            shadow.load_state_dict(snp.params["q_net"])
            shadow.eval()
            self._shadow_q[snp.id] = shadow

        def frozen(player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
            obs = torch.as_tensor(
                self._encoder.encode_obs(state, self._players[player_idx]), device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                q = shadow(obs)
                masked = q.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
                return int(masked.argmax(dim=1).item()), {}

        return frozen

    def _greedy_select(self, player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
        """贪心 masked-argmax（无探索）：对手座位 / vs-random 曲线评估用。"""
        player = self._players[player_idx]
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
