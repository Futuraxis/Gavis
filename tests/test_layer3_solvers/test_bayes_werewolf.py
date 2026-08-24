"""Tests for the Bayesian Werewolf solver — belief tracking, joint
sampling and posterior-driven decisions (no network)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer3_solvers import BayesConfig, BayesSolver
from layer3_solvers.werewolf.belief import BeliefTracker, belief_obs

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "rules" / "werewolf.json"


def _engine(seed: int) -> GameEngine:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def _tracker(seed: int = 1) -> BeliefTracker:
    players = [f"p{i}" for i in range(9)]
    pool = ["wolf"] * 3 + ["villager"] * 3 + ["seer", "witch", "hunter"]
    return BeliefTracker(players, pool, "villager", rng=random.Random(seed))


def _obs_with(players, **kw):
    base = {
        "my_role": "villager",
        "alive": [1] * len(players),
        "deaths_arr": [],
        "dead_roles": {},
        "speech_log": [],
        "vote_log": [],
        "phase": "day_speech",
    }
    base.update(kw)
    return base


# ── 信念更新 ────────────────────────────────────────────────────


def test_prior_is_conditional():
    t = _tracker()
    # players[0] 是自己（villager），概率确定；其他 8 人从剩余池均等
    assert t.prob("p0", "villager") == 1.0
    assert t.wolf_prob("p0") == 0.0
    for p in ("p1", "p2"):
        assert abs(t.prob(p, "wolf") - 3 / 8) < 1e-9  # 3 狼 / 8 其他人
        assert t.prob(p, "seer") == pytest.approx(1 / 8)


def test_vote_signal_raises_target_wolf_prob():
    t = _tracker()
    before = t.wolf_prob("p3")
    t.update_from_observation(
        _obs_with(
            [f"p{i}" for i in range(9)],
            vote_log=[{"voter": "p1", "target": "p3", "round": 1}],
        )
    )
    assert t.wolf_prob("p3") > before


def test_accuse_signal_raises_target_wolf_prob():
    t = _tracker()
    before = t.wolf_prob("p5")
    t.update_from_observation(
        _obs_with(
            [f"p{i}" for i in range(9)],
            speech_log=[{"speaker": "p2", "intent": "accuse", "text": "我怀疑 p5 是狼", "round": 1}],
        )
    )
    assert t.wolf_prob("p5") > before


def test_death_reveals_role_exactly():
    t = _tracker()
    t.update_from_observation(
        _obs_with(
            [f"p{i}" for i in range(9)],
            dead_roles={"p4": "seer"},
        )
    )
    assert t.prob("p4", "seer") == 1.0
    assert t.wolf_prob("p4") == 0.0
    # 池中少一个 seer：其他人的 seer 概率下降
    assert t.prob("p1", "seer") == pytest.approx(0.0)  # 池里 seer 已用尽


def test_entropy_drops_with_evidence():
    t = _tracker()
    e0 = t.entropy("p3")
    t.update_from_observation(
        _obs_with(
            [f"p{i}" for i in range(9)],
            speech_log=[{"speaker": "p6", "intent": "accuse", "text": "p3 是狼", "round": 1}],
        )
    )
    assert t.entropy("p3") < e0


# ── 联合采样 ────────────────────────────────────────────────────


def test_sampling_respects_role_counts():
    t = _tracker()
    for _ in range(20):
        a = t.sample_assignment()
        from collections import Counter

        counts = Counter(a.values())
        assert counts["wolf"] == 3
        assert counts["villager"] == 2  # 9 人 3 民，自己占 1
        assert counts["seer"] == 1 and counts["witch"] == 1 and counts["hunter"] == 1
        assert "p0" not in a  # 自己不参与采样（已知角色）


# ── 决策层 ──────────────────────────────────────────────────────


def test_solver_vote_targets_most_suspicious():
    """投票决策：后验狼概率最高者被投。"""
    from dataclasses import replace

    adapter = _engine(3)
    solver = BayesSolver(adapter, BayesConfig(seed=1), player_id="p0")
    # 推进到投票阶段
    rng = random.Random(1)
    state = adapter.create_initial_state()
    for _ in range(400):
        nt = adapter.get_node_type(state)
        if nt == "player" and any(a.template_id == "vote" for a in adapter.get_legal_actions(state)):
            break
        if nt == "chance":
            outs = adapter.get_chance_outcomes(state)
            if not outs:
                break
            state = adapter.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
        elif nt == "player":
            legal = adapter.get_legal_actions(state)
            if not legal:
                break
            a = rng.choice(legal)
            if a.template_id == "speak":
                a = replace(a, params={**a.params, "text": "x"})
            state = adapter.apply_action(state, a)
        else:
            break
    legal = adapter.get_legal_actions(state)
    assert any(a.template_id == "vote" for a in legal)
    # 注入"p3 是狼"的强烈指控信号
    obs = belief_obs(adapter.project_observation(state, "p0"), "p0")
    obs["speech_log"] = list(obs["speech_log"]) + [
        {"speaker": "p1", "intent": "accuse", "text": "p3 是狼", "round": 1},
        {"speaker": "p2", "intent": "accuse", "text": "p3 是狼", "round": 1},
        {"speaker": "p4", "intent": "accuse", "text": "p3 是狼", "round": 1},
    ]
    solver._ensure_tracker(obs)
    solver._fold_incremental(obs)
    assert solver._tracker.wolf_prob("p3") > solver._tracker.wolf_prob("p1")
    action = solver._vote_action([a for a in legal if a.template_id == "vote"])
    assert action is not None
    assert action.params["target"]["id"] == "p3"


def test_solver_plays_full_game_randomly():
    """贝叶斯玩家在完整对局中动作合法、不卡死（与随机玩家对抗）。"""
    from dataclasses import replace

    adapter = _engine(5)
    solver = BayesSolver(adapter, BayesConfig(seed=5), player_id="p0")
    rng = random.Random(5)
    state = adapter.create_initial_state()
    for step in range(600):
        nt = adapter.get_node_type(state)
        if nt == "chance":
            outs = adapter.get_chance_outcomes(state)
            if not outs:
                break
            state = adapter.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
            continue
        if nt != "player":
            break
        cur = adapter.get_current_player(state)
        legal = adapter.get_legal_actions(state)
        if not legal:
            break
        if cur == "p0":
            a = solver.select_action(state)
            a = a if a is not None else rng.choice(legal)
        else:
            a = rng.choice(legal)
        if a.template_id == "speak" and not a.params.get("text"):
            a = replace(a, params={**a.params, "text": "x"})
        state = adapter.apply_action(state, a)
        if adapter.is_terminal(state):
            break
    winner = state["env"].get("winner")
    assert winner in ("wolf", "good"), f"game did not terminate properly: {winner}"
