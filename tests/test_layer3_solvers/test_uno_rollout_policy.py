"""UnoRolloutPolicy 单测 + 平台装配集成冒烟。

验证要点：
1. ``_split_card`` 正确解析各类 card id（数字 / 特殊 / wild / wild4）。
2. ``UnoRolloutPolicy`` 决策优先级：能赢即赢 > 反击叠加 > 特殊牌 > 大点数数字 > wild4 > wild。
3. 无牌可出时 draw 优先于 pass（尽快推进局面）。
4. ``_HIT_PROB`` 外回退 None（保持 rollout 探索性）。
5. 平台真实装配（hybrid + rollout_policy="uno"）跑一局，rollout_policy
   被注入、AI 出牌合法、无异常。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance
from layer3_solvers.uno.heuristic import _HIT_PROB, UnoRolloutPolicy, _play_score, _split_card
from train_cli import create_solver

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def _uno_engine(seed: int = 7) -> GameEngine:
    with open(RULES_DIR / "uno.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, player_count=4, variant="classic", allow_codegen=False)


def _craft(engine: GameEngine, *, hands: dict[str, list], top=("r", "5"), turn="p0") -> dict:
    """play 阶段手工状态（同 test_uno.py 的 _craft 但精简）。"""
    state = engine.create_initial_state()
    pids = engine._constants["player_ids"]  # noqa: SLF001
    for pid in pids:
        state["_arrays"][f"hand_{pid}"] = list(hands.get(pid, ["b1a"]))  # FILLER
    state["_arrays"]["discard"] = ["wild_1"]
    env = state["env"]
    env["phase"] = "play"
    env["turn"] = turn
    env["direction"] = 1
    env["topColor"], env["topSymbol"] = top
    return state


def _legal(engine: GameEngine, state: dict, tid: str, **params) -> ActionInstance | None:
    for a in engine.get_legal_actions(state):
        if a.template_id == tid and all(a.params.get(k) == v for k, v in params.items()):
            return a
    return None


def _action(tid: str, **params) -> ActionInstance:
    """构造 ActionInstance（_play_score 只读 template_id/params，其余字段占位）。"""
    return ActionInstance(
        template_id=tid,
        type=tid,
        actor_id="p0",
        params=dict(params),
        canonical_key=f"{tid}:{params}",
    )


# ── 解析 ──────────────────────────────────────────────────────────


class TestSplitCard:
    def test_number_with_instance(self) -> None:
        assert _split_card("r5a") == ("r", "5")

    def test_number_zero_no_instance(self) -> None:
        assert _split_card("b0") == ("b", "0")

    def test_special_skip(self) -> None:
        assert _split_card("rsa") == ("r", "s")

    def test_special_reverse(self) -> None:
        assert _split_card("gra") == ("g", "r")

    def test_special_draw2(self) -> None:
        assert _split_card("yda") == ("y", "d")

    def test_wild(self) -> None:
        assert _split_card("wild_2") == ("wild", "wild")

    def test_wild4(self) -> None:
        assert _split_card("wild4_3") == ("wild", "wild4")


# ── 出牌得分 ──────────────────────────────────────────────────────


class TestPlayScore:
    def test_stack2_highest(self) -> None:
        assert _play_score(_action("stack2")) == 200.0

    def test_take_penalty_low(self) -> None:
        assert _play_score(_action("take_penalty")) == 10.0

    def test_draw2_beats_skip(self) -> None:
        assert _play_score(_action("play", card="rda")) > _play_score(_action("play", card="rsa"))

    def test_digit_descending(self) -> None:
        assert _play_score(_action("play", card="r9a")) > _play_score(_action("play", card="r1a"))
        assert _play_score(_action("play", card="r1a")) > _play_score(_action("play", card="r0"))

    def test_wild4_beats_wild(self) -> None:
        assert _play_score(_action("play_wild", card="wild4_1")) > _play_score(_action("play_wild", card="wild_1"))

    def test_special_beats_digit(self) -> None:
        assert _play_score(_action("play", card="rsa")) > _play_score(_action("play", card="r9a"))


# ── 决策 ──────────────────────────────────────────────────────────


class TestDecision:
    def test_win_immediate(self) -> None:
        """手牌仅剩 1 张且能出 → 必出（不探索必胜）。"""
        engine = _uno_engine()
        state = _craft(engine, hands={"p0": ["r5a"]}, top=("r", "5"))
        actions = engine.get_legal_actions(state)
        policy = UnoRolloutPolicy(seed=0)
        # 固定 rng 也应 100% 命中（能赢路径绕过 HIT_PROB）
        chosen = policy(state, actions)
        assert chosen is not None
        assert chosen.template_id == "play"

    def test_no_play_prefers_draw(self) -> None:
        """无牌可出 → draw 优先于 pass（尽快推进局面）。"""
        engine = _uno_engine()
        # p0 全蓝牌，台面红 5 → 无可打，合法动作应含 draw
        state = _craft(engine, hands={"p0": ["b1a"]}, top=("r", "5"))
        actions = engine.get_legal_actions(state)
        assert any(a.template_id == "draw" for a in actions)
        policy = UnoRolloutPolicy(seed=0)
        chosen = policy(state, actions)
        assert chosen is not None
        assert chosen.template_id == "draw"

    def test_hit_prob_returns_none_sometimes(self) -> None:
        """命中率外回退 None（保持探索性）——多轮必出现 None。"""
        engine = _uno_engine()
        state = _craft(engine, hands={"p0": ["r5a", "r9a", "b1a"]}, top=("r", "5"))
        actions = engine.get_legal_actions(state)
        none_count = 0
        for seed in range(200):
            policy = UnoRolloutPolicy(seed=seed)
            if policy(state, actions) is None:
                none_count += 1
        # 期望约 (1 - _HIT_PROB) * 200 ≈ 60 次 None，容差放宽
        assert none_count > 30, f"探索性回退未触发（{none_count}/200 None）"
        assert none_count < 120, f"命中率过低（{none_count}/200 None，期望约 {int((1 - _HIT_PROB) * 200)}）"

    def test_prefers_special_over_digit(self) -> None:
        """命中时特殊牌优先于数字牌（draw2/skip > 9）。"""
        engine = _uno_engine()
        # p0 同时持有 skip 和 9，都合法（台面 skip=rsa 颜色 r）
        state = _craft(engine, hands={"p0": ["rsa", "r9a", "b1a"]}, top=("r", "5"))
        actions = engine.get_legal_actions(state)
        chosen = None
        # 跑到一次命中（非 None）
        for seed in range(50):
            p = UnoRolloutPolicy(seed=seed)
            c = p(state, actions)
            if c is not None:
                chosen = c
                break
        assert chosen is not None, "50 轮内未命中一次"
        assert chosen.params.get("card") == "rsa"  # skip(120) > r9a(9)

    def test_returns_legal_action(self) -> None:
        """策略返回的动作必须合法（不构造非法动作）。"""
        engine = _uno_engine()
        state = _craft(engine, hands={"p0": ["r5a", "rsa", "wild_1"]}, top=("r", "5"))
        actions = engine.get_legal_actions(state)
        legal_keys = {a.canonical_key for a in actions}
        policy = UnoRolloutPolicy(seed=42)
        for _ in range(20):
            chosen = policy(state, actions)
            if chosen is not None:
                assert chosen.canonical_key in legal_keys


# ── 平台装配集成冒烟 ──────────────────────────────────────────────


class TestPlatformAssemblySmoke:
    def test_rollout_policy_injected(self) -> None:
        """平台装配（hybrid + rollout_policy="uno"）后 hybrid.mcts.rollout_policy
        是 UnoRolloutPolicy 实例（而非 None / BoardHeuristicPolicy 链）。"""
        engine = _uno_engine()
        solver = create_solver(
            "uno", "hybrid", engine, 11, 25, cfr_iterations=100, cfr_depth_limit=4, rollout_policy="uno"
        )
        assert hasattr(solver, "mcts")
        assert isinstance(solver.mcts.rollout_policy, UnoRolloutPolicy)

    def test_full_game_no_error_legal_moves(self) -> None:
        """整局冒烟：40 步内无异常、AI 出牌合法、告警即失败。"""
        import warnings

        engine = _uno_engine(seed=11)
        solver = create_solver(
            "uno", "hybrid", engine, 11, 25, cfr_iterations=100, cfr_depth_limit=4, rollout_policy="uno"
        )
        rng = random.Random(11)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)

        steps = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            while engine.get_node_type(state) != "terminal" and steps < 40:
                if engine.get_node_type(state) == "chance":
                    _, state = engine.sample_chance(state)
                    continue
                action = solver.select_action(state)
                if action is None:
                    legal = engine.get_legal_actions(state)
                    if not legal:
                        break
                    action = rng.choice(legal)
                # 出牌必须合法
                legal_keys = {a.canonical_key for a in engine.get_legal_actions(state)}
                assert action.canonical_key in legal_keys, f"AI 出牌非法: {action.canonical_key}"
                state = engine.apply_action(state, action)
                steps += 1

        assert steps > 0, "未推进任何步"
