"""MCTS rollout 视角锚回归测试（2026-09 UNO 平台冒烟暴露）。

旧实现的 rollout 价值视角从状态推导（``root_player`` 读
``env.lastActor`` / ``env.turn``），而 UNO 把 lastActor 声明为
``initial: null`` —— ``env.get("lastActor", turn)`` 在 key 存在而值为
None 时不会回退到默认值，导致开局摸牌 chance 的整段 rollout 价值
归零并告警；中后期 lastActor 是陈旧的"最后出牌人"而非摸牌人，视角
静默错配（价值记错主人 → 回传符号翻转错向）。

修复后：视角锚 = 树内路径最深 player 节点（chance 子节点在
``_expand_player``/``_expand_chance`` 中继承父视角），由 ``_iterate``
显式传给 ``_rollout`` 与 ``_backpropagate`` 共用；``root_player`` 仅作
无树上下文的兜底，并修复其 None-key 陷阱。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from layer2_engine.core.engine import GameEngine
from layer3_solvers import MCTS
from layer3_solvers.mcts import MCTSConfig
from layer3_solvers.mcts.rollout_policy import root_player
from layer3_solvers.mcts.solver import MCTSNode

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"

#: 强制摸牌局：p0 蓝牌接不上红 5 → 唯一合法动作是 draw；其余人各持 1 张
#: 可打的牌，牌堆只剩 1 张蓝牌（b2a）。从 draw 后的 pick chance 出发的
#: rollout 会在数步内走到终局（p1 出完手牌获胜），且此时 lastActor 仍
#: 为 null（没人出过牌）——正是旧实现视角丢失的场景。
_FORCED_DRAW_HANDS = {"p0": ["b1a"], "p1": ["r2a"], "p2": ["g5a"], "p3": ["y9a"]}
_DECK_KEEP = {"b1a", "r2a", "g5a", "y9a", "b2a"}


def _uno_engine(seed: int = 7) -> GameEngine:
    with open(RULES_DIR / "uno.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed, player_count=4, variant="classic", allow_codegen=False)


def _small_gomoku_engine() -> GameEngine:
    with open(RULES_DIR / "stochastic_gomoku.json", "r", encoding="utf-8") as f:
        rules = json.load(f)
    rules["constants"]["board_size"] = 3
    return GameEngine(rules, seed=42)


def _forced_draw_state(engine: GameEngine) -> dict:
    """play 阶段、p0 被迫摸牌、牌堆仅剩 b2a 的"终局在望"状态。"""
    state = engine.create_initial_state()
    for pid, cards in _FORCED_DRAW_HANDS.items():
        state["_arrays"][f"hand_{pid}"] = list(cards)
    # 弃牌堆 = 除保留牌外的全部 → 牌堆（伪）= 108 − 手牌 4 − 弃牌 103 = 1（b2a）。
    state["_arrays"]["discard"] = [c for c in engine._constants["card_ids"] if c not in _DECK_KEEP]  # noqa: SLF001
    env = state["env"]
    env["phase"] = "play"
    env["turn"] = "p0"
    env["direction"] = 1
    env["topColor"], env["topSymbol"] = "r", "5"
    return state


def _pick_chance_state(engine: GameEngine, state: dict) -> dict:
    """应用 draw 后的 pick chance 状态（lastActor=null、turn=p0）。"""
    draw = next(a for a in engine.get_legal_actions(state) if a.template_id == "draw")
    chance = engine.apply_action(state, draw)
    assert engine.get_node_type(chance) == "chance"
    return chance


def _spy_utility(engine: GameEngine) -> tuple[list, callable]:
    """记录 get_utility 调用的 (state, player)；返回 (seen, restore)。"""
    seen: list = []
    orig = engine.get_utility

    def _spy(state, player):  # type: ignore[no-untyped-def]
        seen.append(player)
        return orig(state, player)

    engine.get_utility = _spy  # type: ignore[method-assign]
    return seen, orig


class TestRootPlayerChanceFallback:
    """root_player 的 chance 分支：lastActor 优先，null 时回退 turn。"""

    def test_null_lastactor_falls_back_to_turn(self) -> None:
        engine = _uno_engine()
        state = _forced_draw_state(engine)
        chance = _pick_chance_state(engine, state)
        assert chance["env"]["lastActor"] is None  # key 存在但值为 null
        assert chance["env"]["turn"] == "p0"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # 旧实现在此告警并返回 None
            assert root_player(chance, engine) == "p0"

    def test_lastactor_takes_precedence_when_set(self) -> None:
        # 随机五子棋落子后的 vanish chance：lastActor=落子者。实测该状态
        # turn 尚未轮转（switch_turn 在 chance 结算里），两者一致；手工把
        # turn 前推到对手，模拟"turn 先于 chance 轮转"的规则时序——
        # lastActor（动作主人）必须仍然胜出。
        engine = _small_gomoku_engine()
        state = engine.create_initial_state()
        state = engine.apply_action(state, engine.get_legal_actions(state)[0])
        assert engine.get_node_type(state) == "chance"
        assert state["env"]["lastActor"] == "p_black"
        assert state["env"]["turn"] == "p_black"  # 实测：switch_turn 在 chance 之后
        state["env"]["turn"] = "p_white"  # 模拟已轮转的时序
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert root_player(state, engine) == "p_black"


class TestPathPerspective:
    """_path_perspective：路径最深 player 节点 = 价值视角锚。"""

    def test_deepest_player_node_wins(self) -> None:
        root = MCTSNode(node_type="player", player="p0")
        chance = MCTSNode(node_type="chance", player="p0")
        deeper = MCTSNode(node_type="player", player="p2")
        assert MCTS._path_perspective([(None, root), (None, chance), (None, deeper)]) == "p2"
        assert MCTS._path_perspective([(None, root), (None, chance)]) == "p0"

    def test_no_player_node_returns_none(self) -> None:
        chance = MCTSNode(node_type="chance", player=None)
        assert MCTS._path_perspective([(None, chance)]) is None


class TestRolloutPerspectiveAnchor:
    """_rollout 的价值视角：树内锚优先，无锚时用修复后的兜底。"""

    def test_rollout_with_tree_anchor(self) -> None:
        engine = _uno_engine()
        state = _forced_draw_state(engine)
        chance = _pick_chance_state(engine, state)
        solver = MCTS(engine, MCTSConfig(seed=42, budget=10, rollout_depth=20))

        seen, orig = _spy_utility(engine)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                value = solver._rollout(chance, "p0")  # noqa: SLF001 — white-box
        finally:
            engine.get_utility = orig  # type: ignore[method-assign]

        # 旧实现：root_player(chance) → None → 直接返回 0.0，get_utility
        # 根本不被调用。修复后以锚 p0 评估终局（p1 获胜 → p0 视角 -1）。
        assert seen == ["p0"]
        assert value == -1.0

    def test_rollout_without_anchor_uses_fixed_fallback(self) -> None:
        engine = _uno_engine()
        state = _forced_draw_state(engine)
        chance = _pick_chance_state(engine, state)
        solver = MCTS(engine, MCTSConfig(seed=42, budget=10, rollout_depth=20))

        seen, orig = _spy_utility(engine)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)  # 旧实现在此告警
                value = solver._rollout(chance)  # noqa: SLF001 — white-box
        finally:
            engine.get_utility = orig  # type: ignore[method-assign]

        assert seen == ["p0"]  # turn 兜底生效
        assert value == -1.0


class TestSelectActionNoPerspectiveLoss:
    """端到端：强制摸牌局上 select_action 不再告警、终局价值正常流动。"""

    def test_select_action_no_warning_and_values_flow(self) -> None:
        engine = _uno_engine()
        state = _forced_draw_state(engine)
        solver = MCTS(engine, MCTSConfig(seed=42, budget=30, rollout_depth=20))

        seen, orig = _spy_utility(engine)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)  # 旧实现开局摸牌 rollout 全部告警
                action = solver.select_action(state)
        finally:
            engine.get_utility = orig  # type: ignore[method-assign]

        assert action is not None and action.template_id == "draw"
        # 终局效用以树内锚评估（旧实现：开局段价值归零、get_utility 不被调用）。
        assert seen, "rollout 从未到终局——价值没有流动"
        assert None not in seen
        assert set(seen) <= {"p0", "p1", "p2", "p3"}
