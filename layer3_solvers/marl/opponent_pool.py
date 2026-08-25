"""OpponentPool — 训练对手编排机制（training opponent orchestration）.

纯自博弈（self-play）下学习 agent 的对手永远是"当前的自己"：两个网络互相
追赶、共同漂移，胜率曲线在噪声附近震荡（逼近陷阱 / approximation trap）。
对手编排把"这一局与谁打"从隐式（永远是当前镜像）变成显式决策：

- ``OpponentPool``：周期性把训练中的策略冻结成检查点快照，形成对手池；
- ``WinTracker``：逐池记录 (学习者, 对手快照) 的滚动胜负，供优先采样加权；
- 采样模式（``OpponentScheduleConfig.mode``）：
  - ``self`` — 纯自博弈（基线，无池）；对手 = 当前自身；
  - ``uniform`` — 虚构自博弈（FSP）：从池中均匀抽对手；
  - ``pfsp`` — 优先虚构自博弈（OpenAI Five 论文公式）：``p_i ∝ win_rate_i^α``
    （``priority="win"``，把训练时间集中在已能战胜、可稳定强化的对手上）；
    或 ``priority="lose"`` 时 ``p_i ∝ (1 − win_rate_i)^α``（集中在打不赢的对手）；
  - ``curriculum`` — 课程加权：``p_i ∝ decay^age``，越新的快照权重越高，
    从"旧弱对手"平滑过渡到"新强对手"；
- ``RoleScheduler``：2 人局中学习者座位逐局轮换——每局一个座位用当前策略
  训练（其 transitions 进入学习器），另一个座位执行从池中采样到的对手快照
  （其 transitions 不进入学习器，保证 HAPPO/MAAC/QMix 的 on-policy 性质）；
  池为空或 warmup 阶段时退化为自博弈（与旧行为完全一致）。

平滑训练曲线的原理：学习者不再只面对自己的即时镜像，而是面对**自身过去的
加权混合**——对手分布相对稳定，避免两个网络互相追逐导致的策略震荡；
``pfsp`` 的胜率加权进一步把训练资源稳定地投向"当前能赢的对手"，让
vs-random 评估曲线单调平滑上升。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from layer2_engine.core.state_graph import State

from .env import run_episode

#: 合法采样模式。
MODES: tuple[str, ...] = ("self", "uniform", "pfsp", "curriculum")
#: PFSP 优先方向。
PFSP_PRIORITIES: tuple[str, ...] = ("win", "lose")

#: 每局选择回调签名（与 ``env.SelectFn`` 一致）：(player_idx, state, mask) → (idx, info)。
SelectFn = Callable[[int, State, np.ndarray], tuple[int, dict]]


@dataclass(slots=True)
class OpponentSnapshot:
    """池中的一个冻结策略快照（学习者自己的某个历史检查点）。

    ``params`` 由求解器按自身网络结构填充（如 QMix 的 ``{"q_net": ...}``、
    HAPPO/MAAC 的 ``{"actor": ...}``）；池本身对结构不感知。
    """

    id: int  # 池内唯一 id（单调递增）
    episode: int  # 捕获时的训练局号（curriculum 模式按此排序）
    player: str  # 该策略属于哪个玩家（该玩家的池只存该玩家的快照）
    params: dict[str, Any]


@dataclass(slots=True)
class WinTracker:
    """(学习者, 对手快照) 滚动胜负跟踪，供 PFSP 加权。

    只保留最近 ``memory`` 次对战结果；胜负按 payoff 符号判定
    （>0 胜，==0 平，<0 负），与求解器 ``win_rate`` 口径一致。
    """

    memory: int = 50
    results: dict[int, list[float]] = field(default_factory=dict)

    def record(self, opp_id: int, payoff: float) -> None:
        outcome = 1.0 if payoff > 0 else (0.5 if payoff == 0 else 0.0)
        buf = self.results.setdefault(opp_id, [])
        buf.append(outcome)
        if len(buf) > self.memory:
            del buf[0]

    def win_rate(self, opp_id: int) -> float:
        buf = self.results.get(opp_id, [])
        return sum(buf) / len(buf) if buf else 0.0


@dataclass(slots=True)
class OpponentScheduleConfig:
    """对手编排配置（求解器 config 的扁平字段经 ``from_flat`` 聚合）。"""

    mode: str = "pfsp"  # self | uniform | pfsp | curriculum
    pool_capacity: int = 32  # 对手池容量（超出淘汰最旧快照）
    checkpoint_interval: int = 25  # 每隔多少局把当前学习策略冻结入池
    warmup_episodes: int = 0  # 起步 N 局纯自博弈（池为空时自动退化，无需显式设置）
    pfsp_alpha: float = 1.0  # PFSP 权重指数
    pfsp_floor: float = 0.1  # 权重下限（保证池内每个对手都有非零采样概率）
    pfsp_priority: str = "win"  # win → p∝win_rate^α；lose → p∝(1−win_rate)^α
    recency_decay: float = 0.9  # curriculum 模式：p_i ∝ decay^age
    win_memory: int = 50  # 每对手滚动胜负窗口
    role_alternate: bool = True  # 2 人局学习者座位逐局轮换
    eval_interval: int = 0  # 每 N 局做一次 vs-random 曲线采样（0=关闭）
    eval_episodes: int = 5  # 每次曲线采样的局数

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"未知对手编排模式: {self.mode}（可选 {MODES}）")
        if self.pfsp_priority not in PFSP_PRIORITIES:
            raise ValueError(f"未知 PFSP 优先方向: {self.pfsp_priority}（可选 {PFSP_PRIORITIES}）")

    @classmethod
    def from_flat(cls, cfg: Any) -> "OpponentScheduleConfig":
        """从求解器 config 的 ``opponent_*`` 扁平字段聚合（缺省保持默认）。

        扁平字段名 → 编排配置字段名的显式映射（前缀剥离对
        ``opponent_warmup → warmup_episodes`` 这类名字不成立，故用全表）。
        """
        mapping = {
            "opponent_mode": "mode",
            "opponent_pool_capacity": "pool_capacity",
            "opponent_checkpoint_interval": "checkpoint_interval",
            "opponent_warmup": "warmup_episodes",
            "opponent_pfsp_alpha": "pfsp_alpha",
            "opponent_pfsp_floor": "pfsp_floor",
            "opponent_pfsp_priority": "pfsp_priority",
            "opponent_recency_decay": "recency_decay",
            "opponent_win_memory": "win_memory",
            "opponent_role_alternate": "role_alternate",
        }
        kwargs = {dst: getattr(cfg, src) for src, dst in mapping.items() if hasattr(cfg, src)}
        for n in ("eval_interval", "eval_episodes"):
            if hasattr(cfg, n):
                kwargs[n] = getattr(cfg, n)
        return cls(**kwargs)


class OpponentPool:
    """对手池：冻结快照存储 + 按模式采样 + 胜负跟踪。

    Parameters
    ----------
    config : OpponentScheduleConfig
    rng : random.Random
    players : Sequence[str]
    """

    def __init__(self, config: OpponentScheduleConfig, rng: random.Random, players: Sequence[str]):
        self.config = config
        self._rng = rng
        self._players = list(players)
        self._snapshots: list[OpponentSnapshot] = []
        self._next_id = 1
        self._wins = WinTracker(config.win_memory)

    # ── 池管理 ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def win_tracker(self) -> WinTracker:
        return self._wins

    def checkpoint(self, episode: int, player: str, params: dict[str, Any]) -> OpponentSnapshot:
        """把当前策略冻结入池；超容量时淘汰最旧快照，返回快照。"""
        snp = OpponentSnapshot(id=self._next_id, episode=episode, player=player, params=params)
        self._next_id += 1
        if len(self._snapshots) >= self.config.pool_capacity:
            evicted = self._snapshots.pop(0)
            self._wins.results.pop(evicted.id, None)  # 被淘汰快照的胜负记录一并移除
        self._snapshots.append(snp)
        return snp

    def record_win(self, opp_id: int, payoff: float) -> None:
        self._wins.record(opp_id, payoff)

    # ── 采样 ────────────────────────────────────────────────────────

    def weights(self) -> list[float]:
        """按模式计算各快照的采样权重（未归一化）。"""
        snaps = self._snapshots
        if not snaps:
            return []
        mode = self.config.mode
        if mode == "uniform":
            return [1.0] * len(snaps)
        if mode == "curriculum":
            # 越新（age 越小）权重越高；decay=0.9 → 第 k 个新快照权重 0.9^k。
            return [self.config.recency_decay ** (len(snaps) - 1 - i) for i in range(len(snaps))]
        if mode == "pfsp":
            floor = self.config.pfsp_floor
            alpha = self.config.pfsp_alpha
            out = []
            for s in snaps:
                wr = self._wins.win_rate(s.id)
                base = wr if self.config.pfsp_priority == "win" else 1.0 - wr
                out.append(max(base, floor) ** alpha)
            return out
        raise ValueError(f"未实现的采样模式: {mode}")

    def sample(self) -> OpponentSnapshot | None:
        """按模式从池中抽一个对手快照；池为空返回 None。"""
        if not self._snapshots:
            return None
        weights = self.weights()
        total = sum(weights)
        if total <= 0:
            return self._rng.choice(self._snapshots)
        idx = self._rng.choices(range(len(self._snapshots)), weights=weights, k=1)[0]
        return self._snapshots[idx]

    def snapshot_of(self, player: str) -> OpponentSnapshot | None:
        """取该玩家最新的池快照（curriculum/退化用；无则 None）。"""
        for s in reversed(self._snapshots):
            if s.player == player:
                return s
        return None


class RoleScheduler:
    """2 人局学习者座位轮换：逐局在座位间交替谁是学习器。

    - ``alternate=True`` + 2 人局 → 每局一个学习器（另一座位执行池对手或自身）；
    - 否则（或 1 人/禁用编排）→ 所有座位都是学习器（纯自博弈，旧行为）。
    """

    def __init__(self, players: Sequence[str], alternate: bool = True):
        self._players = list(players)
        self._alternate = alternate

    def learner_for(self, episode: int) -> int | None:
        """返回本局学习器座位下标；None = 所有座位都是学习器。"""
        if not self._alternate or len(self._players) != 2:
            return None
        return episode % 2


class OpponentScheduler:
    """训练对手编排驱动器：每玩家一个对手池 + 座位轮换 + 检查点/胜负记录。

    求解器在 ``train()`` 循环里按局调用：

    - ``learner_for(ep)`` → 本局学习器座位（None = 所有座位学习 → 纯自博弈）；
    - ``maybe_checkpoint(ep, capture_fn)`` → 到期时把**每位玩家**的当前策略
      冻结入各自对手池（``capture_fn(player)`` 返回该玩家网络的冻结 state dict）；
      两个池同步增长，避免座位轮换造成只有偶数座位有池条目的不对称；
    - ``sample(learner_seat)`` → 从学习器的对手池抽一个快照（None = 池空）；
    - ``record(learner_seat, snp, payoff)`` → 更新 (学习者 vs 快照) 滚动胜负；
    - ``should_eval(ep)`` → 是否到 vs-random 曲线采样点。
    """

    def __init__(self, config: OpponentScheduleConfig, rng: random.Random, players: Sequence[str]):
        self.config = config
        self._rng = rng
        self._players = list(players)
        self._pools: dict[str, OpponentPool] = {p: OpponentPool(config, rng, players) for p in players}
        self._roles = RoleScheduler(players, config.role_alternate)

    def learner_for(self, episode: int) -> int | None:
        return self._roles.learner_for(episode)

    def pool_size(self, player: str) -> int:
        return len(self._pools[player])

    def maybe_checkpoint(self, episode: int, capture_fn: Callable[[str], dict[str, Any]]) -> list[OpponentSnapshot]:
        """按 interval 冻结**每位玩家**当前策略入各自对手池。

        warmup 内或未到期返回空列表；到期时两个池同步各入一个新快照
        （对称增长，避免座位轮换造成单边池为空的不对称陪练）。
        """
        if episode < self.config.warmup_episodes:
            return []
        if episode == 0 or episode % self.config.checkpoint_interval != 0:
            return []
        return [self._pools[p].checkpoint(episode, p, capture_fn(p)) for p in self._players]

    def sample(self, learner_seat: int) -> OpponentSnapshot | None:
        """从学习器自己的对手池抽一个快照（对手座位用其过去策略执行）。"""
        return self._pools[self._players[learner_seat]].sample()

    def record(self, learner_seat: int, snp: OpponentSnapshot | None, payoff: float) -> None:
        if snp is not None:
            self._pools[self._players[learner_seat]].record_win(snp.id, payoff)

    def should_eval(self, episode: int) -> bool:
        iv = self.config.eval_interval
        return iv > 0 and episode > 0 and episode % iv == 0


def build_selectors(
    learner_idx: int | None,
    n_players: int,
    learner_select: SelectFn,
    frozen_select: SelectFn | None,
) -> dict[int, SelectFn]:
    """组装本局各座位的选择回调。

    - ``learner_idx`` 为 None → 全部座位都用 ``learner_select``（纯自博弈）；
    - 否则：学习器座位用 ``learner_select``，另一座位用 ``frozen_select``
      （对手快照政策；为 None 时退化为学习器自身 → 自博弈退化路径）。
    """
    if learner_idx is None or frozen_select is None:
        return {i: learner_select for i in range(n_players)}
    return {i: (learner_select if i == learner_idx else frozen_select) for i in range(n_players)}


def eval_vs_random(
    engine,
    players: Sequence[str],
    encoder,
    action_space,
    greedy_select: SelectFn,
    episodes: int,
    base_seed: int,
) -> dict[str, float]:
    """每个座位 vs 均匀随机的胜率评估（固定对手基线的平滑曲线采样点）。

    每局轮换"own 座位"由 ``greedy_select``（贪心策略，无探索）执掌，其余座位
    均匀随机；返回 {玩家: win_rate∈[0,1]}（胜=1，平=0.5，负=0，按 payoff
    符号）。``episodes`` 内每玩家约一半局数。
    """
    per_player: dict[str, list[float]] = {p: [] for p in players}
    for ep in range(episodes):
        rng = random.Random(base_seed + ep * 131)
        owned = ep % len(players)

        def sel(pid: int, state: State, mask: np.ndarray) -> tuple[int, dict]:
            if pid == owned:
                return greedy_select(pid, state, mask)
            legal = np.flatnonzero(mask).tolist()
            return int(rng.choice(legal)), {}

        traj = run_episode(engine, list(players), rng, encoder, action_space, sel, max_steps=4096)
        payoff = float(traj.payoffs.get(players[owned], 0.0))
        per_player[players[owned]].append(1.0 if payoff > 0 else (0.5 if payoff == 0 else 0.0))
    return {p: (sum(v) / len(v) if v else 0.0) for p, v in per_player.items()}
