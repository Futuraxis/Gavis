"""HAPPOSolver — Heterogeneous-Agent PPO (sequential updates).

Each agent owns an actor (on its own observation) and a critic (on the
joint state).  After every episode the agents are updated **in a fixed
order**, one at a time, with the standard clipped PPO surrogate — the
HAPPO sequential-update scheme that guarantees strict monotonic
improvement even with heterogeneous policies.  GAE is computed per
agent over its own subsequence of decisions.
"""

from __future__ import annotations

import random
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
from .buffers import HAPPOTrajectories
from .encoders import GameEncoder
from .env import resolve_device, resolve_players, run_episode
from .networks import MLPActor, MLPCritic
from .opponent_pool import (
    OpponentScheduleConfig,
    OpponentScheduler,
    OpponentSnapshot,
    SelectFn,
    build_selectors,
    eval_vs_random,
)


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


class HAPPOSolver(SolverBase):
    """HAPPO solver for any ``GameEngine`` (moon_chess / mahjong / texas)."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None):
        super().__init__(engine, config or HAPPOConfig())
        cfg = self.config
        if cfg.seed is not None:
            # 可复现性（审查 P2-27）：与 QMix 一致，种子化 torch/np 全局 RNG
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        self._players = resolve_players(engine)
        self._encoder = GameEncoder.build_from_adapter(engine, self._players)
        self._action_space = ActionSpace.build_from_adapter(engine)
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

        # 训练对手编排：开启时为每个玩家持有一个对手池（见 opponent_pool.py）。
        self._opp: OpponentScheduler | None = None
        if cfg.opponent_enabled:
            self._opp = OpponentScheduler(OpponentScheduleConfig.from_flat(cfg), self._rng, self._players)
        self._shadow_actors: dict[int, MLPActor] = {}  # snp.id → 冻结 actor（懒创建）
        self._win_hist: list[float] = []  # 滚动胜负窗口（训练曲线用）

    @property
    def name(self) -> str:
        return f"HAPPO({len(self._players)}p, dim={self._action_space.dim})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> ActionInstance | None:
        """Greedy masked-argmax of the current player's actor."""
        player = self.engine.get_current_player(state)
        if player is None or player not in self._player_idx:
            return None
        legal = self.engine.get_legal_actions(state)
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
        """Train HAPPO — 纯自博弈或对手编排（``opponent_enabled``）.

        Parameters
        ----------
        episodes : int
        verbose : bool, optional

        --- 对手编排模式 ---

        ���启时（``opponent_enabled=True``）每局轮换学习器座位：学习器用当前
        actor 采样并记录（其 transitions 进入轨迹、用于自身的 PPO 更新），
        对手座位执行从池中采样到的冻结快照（贪心、无探索）；对手座位的
        transitions 不入轨迹——保证 HAPPO 的 on-policy 更新只吃学习器自身
        采样数据。池为空/warmup 时退化为纯自博弈（与旧行为一致）。
        """
        cfg = self.config
        verbose = kwargs.get("verbose", False)
        wins = 0
        total_payoff = 0.0
        total_steps = 0
        curve_roll: list[dict[str, float | int | str]] = []
        curve_eval: list[dict[str, float | int | str]] = []

        for ep in range(episodes):
            scheduled = self._opp is not None and self._opp.config.mode != "self"
            learner = self._opp.learner_for(ep) if scheduled else None
            opp_snp: OpponentSnapshot | None = None
            if scheduled:
                self._opp.maybe_checkpoint(ep, self._capture_actor)
                if learner is not None:
                    opp_snp = self._opp.sample(learner)
            frozen = self._frozen_select(opp_snp) if scheduled and learner is not None and opp_snp is not None else None
            selectors = build_selectors(
                learner if scheduled else None,
                len(self._players),
                self._select_train,
                frozen,
            )
            next_values = {learner: self._eval_next} if (scheduled and learner is not None) else None
            traj = run_episode(
                self.engine,
                self._players,
                self._rng,
                self._encoder,
                self._action_space,
                selectors,
                next_values,
                max_steps=4096,  # 病理局面步数上限（正常对局远低于此）
            )
            total_steps += len(traj.transitions)

            # 收集：编排模式只收学习器座位（保证 on-policy）。
            tracked = learner if (scheduled and learner is not None) else None
            for t in traj.transitions:
                if tracked is not None and t.player_idx != tracked:
                    continue
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
            payoff = float(traj.payoffs.get(self._players[tracked if tracked is not None else 0], 0.0))
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

            self._update_all_agents()

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
                print(f"  HAPPO ep {ep + 1:4d}/{episodes}  win={win_pct:5.1f}%{pool_info}")

        return SolverMetrics(
            episodes=episodes,
            win_rate=wins / max(1, episodes),
            avg_return=total_payoff / max(1, episodes),
            extra={
                "steps": total_steps,
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

    def _capture_actor(self, player: str) -> dict[str, Any]:
        """冻结指定玩家的 actor state dict（深拷贝张量，供对手池入池）。"""
        return {"actor": {k: v.detach().clone() for k, v in self._actors[player].state_dict().items()}}

    def _frozen_select(self, snp: OpponentSnapshot) -> SelectFn:
        """构造执行对手快照（学习者过去策略）的贪心选择回调。

        影子 actor 按快照 id 懒创建并加载冻结参数——不触碰在线网络。
        """
        shadow = self._shadow_actors.get(snp.id)
        if shadow is None:
            shadow = MLPActor(self._encoder.obs_dim, self._action_space.dim, self.config.hidden_dim).to(self.device)
            shadow.load_state_dict(snp.params["actor"])
            shadow.eval()
            self._shadow_actors[snp.id] = shadow

        def frozen(player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
            obs = torch.as_tensor(
                self._encoder.encode_obs(state, self._players[player_idx]), device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                logits = shadow(obs)
                masked = logits.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
                return int(masked.argmax(dim=1).item()), {}

        return frozen

    def _greedy_select(self, player_idx: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
        """贪心 masked-argmax（无探索）：对手座位 / vs-random 曲线评估用。"""
        player = self._players[player_idx]
        with torch.no_grad():
            logits = self._actors[player](
                torch.as_tensor(self._encoder.encode_obs(state, player), device=self.device).unsqueeze(0)
            )
            masked = logits.masked_fill(torch.as_tensor(mask, device=self.device).unsqueeze(0) == 0, -1e9)
            return int(masked.argmax(dim=1).item()), {}

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
        # 训练期 value 前向无梯度需求 — no_grad 省去构图开销
        with torch.no_grad():
            value = float(self._critics[player](global_state).item())
        return action, {"log_prob": log_prob, "value": value}

    def _eval_next(self, state: State, player_idx: int) -> float:
        """Critic value of the successor state for the acting player."""
        player = self._players[player_idx]
        global_state = torch.as_tensor(self._encoder.encode_global(state), device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self._critics[player](global_state).item())

    def _update_all_agents(self) -> None:
        """Sequential per-agent PPO updates in a fixed order (HAPPO core).

        The HAPPO monotonic-improvement guarantee requires each round of
        sequential updates to run in one consistent order; shuffling the
        order every episode (M-06) breaks the guarantee.
        """
        cfg = self.config
        order = list(self._players)
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
