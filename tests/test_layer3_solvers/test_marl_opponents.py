"""Tests for the MARL training-opponent orchestration mechanism.

Covers the pure pool mechanics (sampling weights by mode, capacity
eviction, rolling win tracking) plus solver-level integration: scheduled
training alternates the learner seat, only learner transitions enter the
replay/trajectory store, and periodic vs-random curve samples land in
``extra['curve_eval']``.  Existing self-play behaviour must be untouched
when ``opponent_enabled=False`` (regression).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"

try:
    from layer3_solvers.marl.happo import HAPPOConfig, HAPPOSolver
    from layer3_solvers.marl.maac import MAACConfig, MAACSolver
    from layer3_solvers.marl.opponent_pool import (
        OpponentPool,
        OpponentScheduleConfig,
        OpponentScheduler,
        RoleScheduler,
        WinTracker,
        build_selectors,
        eval_vs_random,
    )
    from layer3_solvers.marl.qmix import QMixConfig, QMixSolver

    _HAS_TORCH = True
except (ImportError, TypeError):
    OpponentPool = OpponentScheduleConfig = OpponentScheduler = RoleScheduler = None
    WinTracker = build_selectors = eval_vs_random = None
    QMixSolver = HAPPOSolver = MAACSolver = None
    QMixConfig = HAPPOConfig = MAACConfig = None
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _mahjong(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "mahjong.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, variant="guangdong", player_count=2)


def _moon(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


# ── 池机制（纯组件）─────────────────────────────────────────────────


class TestWinTracker:
    def test_rolling_window_and_win_rate(self):
        wt = WinTracker(memory=3)
        for payoff in (1.0, 1.0, 0.0, -1.0):
            wt.record(7, payoff)
        # 窗口保留最后 3 次：1.0, 0.0, -1.0 → (1+0.5+0)/3 = 0.5
        assert wt.win_rate(7) == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert wt.win_rate(999) == 0.0  # 未见过的对手 → 0

    def test_draw_is_half_win(self):
        wt = WinTracker(memory=10)
        wt.record(1, 0.0)
        assert wt.win_rate(1) == 0.5


class TestOpponentPoolSampling:
    def _pool(self, mode: str, **kw) -> OpponentPool:
        cfg = OpponentScheduleConfig(mode=mode, pool_capacity=4, checkpoint_interval=10, **kw)
        return OpponentPool(cfg, random.Random(1), ["p0", "p1"])

    def test_uniform_weights(self):
        pool = self._pool("uniform")
        pool.checkpoint(10, "p0", {"w": 1.0})
        pool.checkpoint(20, "p0", {"w": 2.0})
        weights = pool.weights()
        assert weights == [1.0, 1.0]

    def test_curriculum_prefers_newer(self):
        pool = self._pool("curriculum", recency_decay=0.5)
        pool.checkpoint(10, "p0", {"w": 1.0})
        pool.checkpoint(20, "p0", {"w": 2.0})
        pool.checkpoint(30, "p0", {"w": 3.0})
        weights = pool.weights()
        # age 2/1/0 → 0.5^2, 0.5^1, 0.5^0
        assert weights == pytest.approx([0.25, 0.5, 1.0])

    def test_pfsp_win_prefers_beaten_opponents(self):
        pool = self._pool("pfsp", pfsp_priority="win", pfsp_floor=0.0)
        s1 = pool.checkpoint(10, "p0", {"w": 1.0})
        s2 = pool.checkpoint(20, "p0", {"w": 2.0})
        pool.record_win(s1.id, 1.0)  # 对 s1 全胜
        pool.record_win(s2.id, -1.0)  # 对 s2 全负
        weights = pool.weights()
        # win 模式：p ∝ win_rate → s1 权重 > s2
        assert weights[0] > weights[1]
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(0.0)

    def test_pfsp_lose_prefers_losses(self):
        pool = self._pool("pfsp", pfsp_priority="lose", pfsp_floor=0.0)
        s1 = pool.checkpoint(10, "p0", {"w": 1.0})
        s2 = pool.checkpoint(20, "p0", {"w": 2.0})
        pool.record_win(s1.id, 1.0)
        pool.record_win(s2.id, -1.0)
        weights = pool.weights()
        assert weights[1] > weights[0]

    def test_floor_keeps_all_samplable(self):
        pool = self._pool("pfsp", pfsp_floor=0.1)
        s1 = pool.checkpoint(10, "p0", {"w": 1.0})
        s2 = pool.checkpoint(20, "p0", {"w": 2.0})
        pool.record_win(s1.id, 1.0)
        pool.record_win(s2.id, -1.0)
        weights = pool.weights()
        assert all(w > 0 for w in weights)

    def test_capacity_evicts_oldest(self):
        pool = self._pool("uniform")  # capacity=4
        ids = [pool.checkpoint(10 * i, "p0", {"w": float(i)}).id for i in range(1, 6)]
        assert len(pool) == 4
        assert ids[0] not in [s.id for s in pool._snapshots]  # noqa: SLF001
        assert ids[-1] in [s.id for s in pool._snapshots]  # noqa: SLF001


class TestRoleScheduler:
    def test_alternates_two_players(self):
        rs = RoleScheduler(["p0", "p1"], alternate=True)
        assert [rs.learner_for(i) for i in range(4)] == [0, 1, 0, 1]

    def test_no_alternation_returns_none(self):
        rs = RoleScheduler(["p0", "p1"], alternate=False)
        assert rs.learner_for(3) is None

    def test_single_player_none(self):
        rs = RoleScheduler(["p0"], alternate=True)
        assert rs.learner_for(0) is None


class TestBuildSelectors:
    def test_no_learner_all_same(self):
        selectors = build_selectors(
            None,
            2,
            lambda p, s, m: (0, {}),
            lambda p, s, m: (1, {}),
        )
        assert selectors == {0: selectors[0], 1: selectors[0]}

    def test_learner_routed(self):
        learner = lambda p, s, m: (0, {})  # noqa: E731
        frozen = lambda p, s, m: (1, {})  # noqa: E731
        selectors = build_selectors(0, 2, learner, frozen)
        assert selectors[0] is learner
        assert selectors[1] is frozen


class TestEvalVsRandom:
    def test_mahjong_smoke(self):
        engine = _mahjong(7)
        from layer3_solvers.marl.action_space import ActionSpace
        from layer3_solvers.marl.encoders import GameEncoder
        from layer3_solvers.marl.env import resolve_players

        players = resolve_players(engine)
        encoder = GameEncoder.build_from_adapter(engine, players)
        action_space = ActionSpace.build_from_adapter(engine)

        def greedy(pid, state, mask):
            legal = np.flatnonzero(mask).tolist()
            return int(legal[0]), {}

        res = eval_vs_random(engine, players, encoder, action_space, greedy, episodes=4, base_seed=11)
        assert set(res) == set(players)
        assert all(0.0 <= v <= 1.0 for v in res.values())


# ── 求解器集成 ──────────────────────────────────────────────────────


SMALL_OPP: dict = {
    "opponent_enabled": True,
    "opponent_mode": "pfsp",
    "opponent_pool_capacity": 4,
    "opponent_checkpoint_interval": 10,
    "opponent_warmup": 0,
    "opponent_pfsp_floor": 0.1,
    "opponent_role_alternate": True,
    "eval_interval": 10,
    "eval_episodes": 2,
}
# 与 test_marl.py 同量级的小网络/小缓冲（按求解器：HAPPO 无 replay 缓冲字段）
SMALL_BASE: dict = {
    QMixSolver: {"hidden_dim": 16, "start_learning": 8, "batch_size": 16, "buffer_capacity": 1024},
    HAPPOSolver: {"hidden_dim": 16, "minibatch_size": 8},
    MAACSolver: {"hidden_dim": 16, "start_learning": 8, "batch_size": 16, "buffer_capacity": 1024},
}


@pytest.mark.parametrize(
    "solver_cls,cfg_cls",
    [(QMixSolver, QMixConfig), (HAPPOSolver, HAPPOConfig), (MAACSolver, MAACConfig)],
)
class TestScheduledTrainingIntegration:
    def test_pool_grows_and_curve_recorded(self, solver_cls, cfg_cls):
        engine = _moon(42)  # 快速棋类局，编排机制与游戏无关
        solver = solver_cls(engine, cfg_cls(seed=42, **SMALL_BASE[solver_cls], **SMALL_OPP))
        metrics = solver.train(episodes=30, verbose=False)
        extra = metrics.extra
        assert extra["opponent_enabled"] is True
        assert len(extra["curve_roll"]) > 0
        # 检查点：ep 10 / 20（ep 0 跳过、30 不在局内）→ 每个玩家池各 2 条
        assert solver._opp.pool_size("p_black") == 2  # noqa: SLF001
        assert solver._opp.pool_size("p_white") == 2  # noqa: SLF001
        # eval_interval=10 → ep 10/20/30 各一次 vs-random 采样
        assert len(extra["curve_eval"]) == 3
        assert all({"ep", "p_black_wr", "p_white_wr"} <= set(sample) for sample in extra["curve_eval"])

    def test_disabled_matches_self_play(self, solver_cls, cfg_cls):
        engine = _moon(42)
        solver = solver_cls(engine, cfg_cls(seed=42, **SMALL_BASE[solver_cls]))
        metrics = solver.train(episodes=3, verbose=False)
        assert metrics.extra["opponent_enabled"] is False
        assert metrics.episodes == 3


class TestScheduledMahjong:
    def test_mahjong_scheduled_train_smoke(self):
        solver = QMixSolver(_mahjong(42), QMixConfig(seed=42, **SMALL_BASE[QMixSolver], **SMALL_OPP))
        metrics = solver.train(episodes=22, verbose=False)
        assert metrics.episodes == 22
        assert len(metrics.extra["curve_roll"]) >= 1
        # 池内快照存在（ep 10/20 已入池）
        assert solver._opp.pool_size("p0") >= 1  # noqa: SLF001

    def test_happo_scheduled_mahjong_smoke(self):
        solver = HAPPOSolver(_mahjong(42), HAPPOConfig(seed=42, **SMALL_BASE[HAPPOSolver], **SMALL_OPP))
        metrics = solver.train(episodes=22, verbose=False)
        assert metrics.episodes == 22


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        OpponentScheduleConfig(mode="nonsense")

    with pytest.raises(ValueError):
        OpponentScheduleConfig(mode="pfsp", pfsp_priority="sideways")
