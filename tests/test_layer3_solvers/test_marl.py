"""Tests for the MARL solvers (QMix / HAPPO / MAAC, Layer 3).

Each solver is tested at the unit level (select_action, train API) on
moon_chess (fast testbed) and mahjong 2-player (the main multi-agent
target).  Action-space mapping is tested for all three games.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def _moon(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def _texas(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "texas_holdem.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def _mahjong_gd(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "mahjong.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, variant="guangdong", player_count=2)


def _mahjong_hz(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "mahjong.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, variant="hongzhong", player_count=2)


try:
    from layer3_solvers.marl import (
        ActionSpace,
        HAPPOConfig,
        HAPPOSolver,
        MAACConfig,
        MAACSolver,
        QMixConfig,
        QMixSolver,
        resolve_players,
        run_episode,
    )
    from layer3_solvers.marl.buffers import HAPPOTrajectories
    from layer3_solvers.marl.encoders import GameEncoder

    _HAS_TORCH = True
except (ImportError, TypeError):
    QMixSolver = HAPPOSolver = MAACSolver = None
    QMixConfig = HAPPOConfig = MAACConfig = None
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")

SMALL_Q = dict(start_learning=8, batch_size=16, buffer_capacity=1024)
SMALL_M = dict(start_learning=8, batch_size=16, buffer_capacity=1024)


@pytest.fixture
def moon_adapter() -> GameEngine:
    return _moon(42)


@pytest.fixture
def mahjong_adapter() -> GameEngine:
    return _mahjong_gd(42)


@pytest.fixture
def texas_adapter() -> GameEngine:
    return _texas(42)


def _advance_chance(adapter, state):
    """Advance through chance nodes until a player node (or terminal)."""
    while adapter.get_node_type(state) == "chance":
        outcomes = adapter.get_chance_outcomes(state)
        if not outcomes:
            break
        state = adapter.apply_chance(state, outcomes[0])
    return state


def _drive_to_claim(seed: int) -> tuple[GameEngine, dict] | None:
    """Play random legal mahjong until a claim phase (or give up).

    Returns (adapter, state-at-claim) or None.  Engine edge cases
    (degenerate chi chains) are treated as "no claim found".
    """
    adapter = _mahjong_gd(seed)
    rng = random.Random(seed)
    state = adapter.create_initial_state()
    for _ in range(800):
        try:
            node = adapter.get_node_type(state)
            if node == "chance":
                outcomes = adapter.get_chance_outcomes(state)
                if not outcomes:
                    return None
                state = adapter.apply_chance(state, outcomes[0])
                continue
            if node != "player":
                return None
            if state.get("env", {}).get("phase") == "claim":
                return adapter, state
            legal = adapter.get_legal_actions(state)
            if not legal:
                return None
            state = adapter.apply_action(state, rng.choice(legal))
        except Exception:
            return None  # engine edge case
    return None


# ── SolverBase compliance ─────────────────────────────────────────


class TestMARLCompliance:
    @pytest.mark.parametrize(
        "solver_cls,cfg_cls",
        [
            (QMixSolver, QMixConfig),
            (HAPPOSolver, HAPPOConfig),
            (MAACSolver, MAACConfig),
        ],
    )
    def test_select_action_untrained_moon_chess(self, moon_adapter, solver_cls, cfg_cls):
        solver = solver_cls(moon_adapter, cfg_cls(seed=42))
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        assert action is not None
        legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys

    @pytest.mark.parametrize(
        "solver_cls,cfg_cls",
        [
            (QMixSolver, QMixConfig),
            (HAPPOSolver, HAPPOConfig),
            (MAACSolver, MAACConfig),
        ],
    )
    def test_select_action_untrained_mahjong(self, mahjong_adapter, solver_cls, cfg_cls):
        solver = solver_cls(mahjong_adapter, cfg_cls(seed=42))
        state = _advance_chance(mahjong_adapter, mahjong_adapter.create_initial_state())
        action = solver.select_action(state)
        assert action is not None
        legal_keys = {a.canonical_key for a in mahjong_adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys

    @pytest.mark.parametrize(
        "solver_cls,cfg_cls",
        [
            (QMixSolver, QMixConfig),
            (HAPPOSolver, HAPPOConfig),
            (MAACSolver, MAACConfig),
        ],
    )
    def test_play_few_moves_moon_chess(self, moon_adapter, solver_cls, cfg_cls):
        solver = solver_cls(moon_adapter, cfg_cls(seed=42))
        state = moon_adapter.create_initial_state()
        for _ in range(5):
            if moon_adapter.is_terminal(state):
                break
            if moon_adapter.get_node_type(state) != "player":
                break
            action = solver.select_action(state)
            if action is None:
                break
            legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
            assert action.canonical_key in legal_keys
            state = moon_adapter.apply_action(state, action)

    @pytest.mark.parametrize(
        "solver_cls,cfg_cls",
        [
            (QMixSolver, QMixConfig),
            (HAPPOSolver, HAPPOConfig),
            (MAACSolver, MAACConfig),
        ],
    )
    def test_save_load_roundtrip(self, moon_adapter, tmp_path, solver_cls, cfg_cls):
        solver = solver_cls(moon_adapter, cfg_cls(seed=42))
        path = tmp_path / f"{solver_cls.__name__}_test.pt"
        solver.save(str(path))
        assert path.exists()
        solver2 = solver_cls(moon_adapter, cfg_cls(seed=1))
        solver2.load(str(path))


# ── QMix ───────────────────────────────────────────────────────────


class TestQMix:
    def test_train_short_moon_chess(self, moon_adapter):
        solver = QMixSolver(moon_adapter, QMixConfig(seed=42, **SMALL_Q))
        metrics = solver.train(episodes=3, verbose=False)
        assert metrics.episodes == 3
        assert metrics.extra["steps"] > 0

    def test_train_short_mahjong_2p(self, mahjong_adapter):
        solver = QMixSolver(mahjong_adapter, QMixConfig(seed=42, **SMALL_Q))
        metrics = solver.train(episodes=2, verbose=False)
        assert metrics.episodes == 2

    def test_runner_handles_chance(self, mahjong_adapter):
        players = resolve_players(mahjong_adapter)
        encoder = GameEncoder.build_from_adapter(mahjong_adapter, players)
        action_space = ActionSpace.build_from_adapter(mahjong_adapter)
        rng = random.Random(0)

        def random_policy(pid, state, mask):
            legal_idx = np.flatnonzero(mask).tolist()
            return int(rng.choice(legal_idx)), {}

        traj = run_episode(mahjong_adapter, players, rng, encoder, action_space, random_policy)
        assert len(traj.transitions) > 0
        assert set(traj.payoffs.keys()) == set(players)
        for t in traj.transitions:
            assert t.mask.sum() >= 1
            assert 0 <= t.action < action_space.dim
            # Reward is only ever placed on a done (terminal) transition
            assert t.reward == 0.0 or t.done

    def test_epsilon_decays(self, moon_adapter):
        cfg = QMixConfig(seed=42, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=5, **SMALL_Q)
        solver = QMixSolver(moon_adapter, cfg)
        solver.train(episodes=2, verbose=False)
        assert solver._steps >= cfg.epsilon_decay_steps
        assert solver._epsilon == cfg.epsilon_end


# ── HAPPO ──────────────────────────────────────────────────────────


class TestHAPPO:
    def test_train_short_moon_chess(self, moon_adapter):
        solver = HAPPOSolver(moon_adapter, HAPPOConfig(seed=42))
        metrics = solver.train(episodes=3, verbose=False)
        assert metrics.episodes == 3
        assert metrics.extra["steps"] > 0

    def test_train_short_mahjong_2p(self, mahjong_adapter):
        solver = HAPPOSolver(mahjong_adapter, HAPPOConfig(seed=42))
        metrics = solver.train(episodes=2, verbose=False)
        assert metrics.episodes == 2

    def test_select_action_after_training(self, moon_adapter):
        solver = HAPPOSolver(moon_adapter, HAPPOConfig(seed=42))
        solver.train(episodes=2, verbose=False)
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        assert action is not None
        legal_keys = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys

    def test_gae_returns_shapes(self):
        traj = HAPPOTrajectories()
        for p in range(2):
            traj.ensure_agent(p)
            for i in range(5):
                traj.add(
                    p,
                    obs=np.zeros(4, np.float32),
                    mask=np.ones(3, np.float32),
                    action=1,
                    log_prob=-0.5,
                    reward=1.0 if i == 4 else 0.0,
                    done=(i == 4),
                    value=0.1,
                    next_value=0.0,
                    global_state=np.zeros(8, np.float32),
                )
            traj.compute_returns_and_advantages(p, gamma=0.99, gae_lambda=0.95)
            assert traj.returns[p].shape == (5,)
            assert traj.advantages[p].shape == (5,)
            assert np.allclose(traj.returns[p], traj.advantages[p] + traj.values[p])


# ── MAAC ───────────────────────────────────────────────────────────


class TestMAAC:
    def test_train_short_moon_chess(self, moon_adapter):
        solver = MAACSolver(moon_adapter, MAACConfig(seed=42, **SMALL_M))
        metrics = solver.train(episodes=3, verbose=False)
        assert metrics.episodes == 3
        assert metrics.extra["steps"] > 0

    def test_train_short_mahjong_2p(self, mahjong_adapter):
        solver = MAACSolver(mahjong_adapter, MAACConfig(seed=42, **SMALL_M))
        metrics = solver.train(episodes=2, verbose=False)
        assert metrics.episodes == 2

    def test_soft_target_update(self, moon_adapter):
        import torch

        solver = MAACSolver(moon_adapter, MAACConfig(seed=42, **SMALL_M))
        solver.train(episodes=2, verbose=False)
        # With tau=0.005 and several gradient steps, targets must differ
        # from the online networks.
        for n, t in zip(solver._actors.parameters(), solver._actor_targets.parameters()):
            if not torch.equal(n, t):
                return
        raise AssertionError("actor targets never moved away from online nets")


# ── 审查 P1-13/14 修复回归 ─────────────────────────────────────────


class TestChanceRollForward:
    def test_next_mask_nonzero_after_chance(self, moon_adapter):
        """P1-13: 回合制后继状态前滚 chance 后，非终局 transition 的
        next_mask 必须非全零（旧实现：moon_chess 每步都接 vanish_check
        chance 节点 → next_mask 全零 → QMix/MAAC bootstrap 被掩成垃圾、
        QMix target ≈ −1e9 发散）。"""
        players = ["p_black", "p_white"]
        encoder = GameEncoder.build_from_adapter(moon_adapter, players)
        action_space = ActionSpace.build_from_adapter(moon_adapter)
        rng = random.Random(42)

        def rand_idx(pid, state, mask):
            idxs = np.flatnonzero(mask)
            return int(idxs[rng.randrange(len(idxs))]), {}

        traj = run_episode(moon_adapter, players, rng, encoder, action_space, rand_idx)
        assert traj.transitions
        for t in traj.transitions:
            if not t.done:
                assert t.next_mask.sum() >= 1, "all-zero next_mask on non-done transition"


class TestMAACPolicyGradientSeparation:
    def test_actor_step_does_not_touch_critic(self, moon_adapter):
        """P1-14: actor 步的 REINFORCE loss 经 q_online.detach() 后不再
        二次更新 critic（旧实现共享优化器 + 未 detach → critic 每步被
        步进两次且对 on-policy 动作向量做梯度上升）。"""
        cfg = MAACConfig(
            seed=1,
            start_learning=8,
            batch_size=16,
            buffer_capacity=1024,
            hidden_dim=16,
            learning_rate=0.01,
        )
        solver = MAACSolver(moon_adapter, cfg)
        rng = random.Random(1)
        encoder = solver._encoder  # noqa: SLF001
        action_space = solver._action_space  # noqa: SLF001
        players = solver._players  # noqa: SLF001

        def rand_idx(pid, state, mask):
            idxs = np.flatnonzero(mask)
            return int(idxs[rng.randrange(len(idxs))]), {}

        for _ in range(8):
            traj = run_episode(moon_adapter, players, rng, encoder, action_space, rand_idx)
            for t in traj.transitions:
                solver._buffer.push(t)  # noqa: SLF001
            if len(solver._buffer) >= cfg.batch_size:  # noqa: SLF001
                break
        assert len(solver._buffer) >= cfg.batch_size  # noqa: SLF001

        solver._gradient_step()  # noqa: SLF001
        for p, net in solver._critics.items():  # noqa: SLF001
            for param in net.parameters():
                assert param.grad is None, f"critic {p} received gradient from the actor step"
        actor_grads = [
            param.grad is not None
            for net in solver._actors.values()
            for param in net.parameters()  # noqa: SLF001
        ]
        assert any(actor_grads), "actor parameters received no gradient at all"


# ── Action space (unit, no training) ───────────────────────────────


class TestActionSpace:
    def test_moon_chess_mask_and_from_index(self, moon_adapter):
        action_space = ActionSpace.build_from_adapter(moon_adapter)
        assert action_space.dim == 9
        state = moon_adapter.create_initial_state()
        mask = action_space.legal_mask(state)
        assert mask.sum() == 9  # empty board
        legal = moon_adapter.get_legal_actions(state)
        for action in legal:
            idx = action_space.index_of(action)
            assert idx is not None
            back = action_space.action_from_index(idx, legal)
            assert back.canonical_key == action.canonical_key

    def test_mahjong_duplicates_set_one_bit(self, mahjong_adapter):
        action_space = ActionSpace.build_from_adapter(mahjong_adapter)
        assert action_space.dim == 227
        state = _advance_chance(mahjong_adapter, mahjong_adapter.create_initial_state())
        legal = mahjong_adapter.get_legal_actions(state)
        mask = action_space.legal_mask(state)
        unique = {a.canonical_key for a in legal}
        assert mask.sum() == len(unique)
        for action in legal:
            idx = action_space.index_of(action)
            assert idx is not None
            assert mask[idx] == 1.0

    def test_mahjong_claim_slots(self):
        found = None
        for seed in range(10):
            found = _drive_to_claim(seed)
            if found is not None:
                break
        assert found is not None, "no claim phase reached across seeds"
        adapter, state = found

        action_space = ActionSpace.build_from_adapter(adapter)
        legal = adapter.get_legal_actions(state)
        keys = {a.canonical_key for a in legal}
        assert any(k.startswith("claim_chi") for k in keys) or "claim_pass" in keys
        for action in legal:
            idx = action_space.index_of(action)
            assert idx is not None
            if action.template_id == "claim_pass":
                assert idx == 225
            if action.template_id == "claim_chi":
                assert 204 <= idx <= 224

    def test_texas_mask(self, texas_adapter):
        action_space = ActionSpace.build_from_adapter(texas_adapter)
        assert action_space.dim == 48
        state = _advance_chance(texas_adapter, texas_adapter.create_initial_state())
        mask = action_space.legal_mask(state)
        assert mask.sum() == len(texas_adapter.get_legal_actions(state))
        for action in texas_adapter.get_legal_actions(state):
            idx = action_space.index_of(action)
            assert idx is not None
            assert mask[idx] == 1.0
            assert 0 <= idx < 48
