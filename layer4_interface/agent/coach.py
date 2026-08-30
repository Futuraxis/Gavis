"""coach — 教学对局的教练（Layer 4，"LLM + Skill" 的教学半边）.

教学对局（``teaching=True``）下，教练 Agent 能看到**玩家自己的牌**并进行
推理。融合进既有框架的三条设计红线：

1. **教练看的 = 玩家看的**。教练的唯一数据入口是
   ``engine.project_observation(state, player_pid)`` —— 玩家自己的投影
   （含玩家自己的手牌视图），绝不经由 ground ``_arrays``。教练因此从不
   比 player 知道更多：看不到 AI 的手牌、牌墙、未翻牌堆。这也是
   :func:`hidden_guard.assert_no_hidden` 仍然成立的原因——玩家自己的
   视图名（``hand_view_p0`` / ``sb_hole_view``）不在黑名单里。

2. **双脑分离**。对手脑（会话 ``solver``，在 AI 座位公平落子）与教练脑
   （本模块）互不相通：AI 自己的落子路径零改动；教练的"推理"是在**玩家
   回合**用同一求解器契约（``select_action`` 按状态当前行动者推理）替
   玩家算一手参考动作——求解器给玩家座位推理时读的正是玩家手牌。

3. **在线学习防污染**。参考动作必须走会话的 ``raw_solver``（未被
   ``RecordingHandle`` 包装的原始句柄）：教练替玩家算的动作不能混入
   AI 决策轨迹（那是"ai" actor 的训练数据）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ..solver_provider import SolverHandle
from .evaluation import evaluate
from .hidden_guard import assert_no_hidden
from .skills import SkillContext

#: 各游戏族「玩家自己手牌视图」的命名约定（Layer 4 表现层关注点，与
#: hidden_guard.infer_game_id 的按游戏分派同风格）：
#:   - 麻将 / UNO（含同族自定义游戏）: ``hand_view_<pid>``
#:   - 德州扑克: ``<seat>_hole_view``（pid ``p_sb`` → 视图 ``sb_hole_view``）
#: 棋类游戏没有手牌视图 → 提取为空，教练退化为纯局面讲解。
_HAND_VIEW_SUFFIXES = ("hand_view_{pid}", "{pid}_hole_view", "{seat}_hole_view")


@dataclass
class TeachContext(SkillContext):
    """教学上下文 —— 教练视角恰好是玩家自己的观测（含玩家手牌）.

    Attributes:
        teaching: 恒为 True（标记 ctx 走教练路径：教学系统提示 + 教学
            泄露扫描变体）。
        hand: 玩家自己的手牌 id 列表（从玩家投影的私有视图提取；棋类
            游戏为空列表）。
        legal_count: 当前合法动作数（给 LLM 讲"选择面"）。
        reference: 教练在玩家座位算出的参考动作（canonical key；
            ``None`` = 未计算或求解器无果；仅在玩家行动前的快照上计算）。
        player_action: 玩家实际动作的描述（``teach_move`` 讲评时填充）。
        matched: 玩家动作与参考动作是否一致（``None`` = 尚未讲评）。
    """

    teaching: bool = True
    hand: list[str] = field(default_factory=list)
    legal_count: int = 0
    reference: str | None = None
    player_action: str | None = None
    matched: bool | None = None


class Coach:
    """教学对局的确定性教练：只看玩家自己的投影，产出教学机械事实."""

    @staticmethod
    def build(
        state: dict[str, Any],
        player_pid: str,
        engine: Any,
        solver: SolverHandle | None = None,
    ) -> TeachContext:
        """从（玩家回合的）状态构建 :class:`TeachContext`.

        Args:
            state: 引擎状态 —— 必须是玩家行动时的状态（参考动作按当前
                行动者计算，传入非玩家回合会算成别人的动作）。
            player_pid: 人类玩家 id。
            engine: :class:`layer2_engine.core.engine.GameEngine`。
            solver: 求解器句柄（**必须传未被录制包装的 raw_solver**，
                教练的参考动作不进在线学习轨迹）；``None`` = 只读牌不算
                参考动作（``teach_turn`` 导读用，不剧透答案）。

        Returns:
            通过 :func:`assert_no_hidden` 校验的教学上下文。
        """
        observation = engine.project_observation(state, player_pid)
        legal_actions = engine.get_legal_actions(state)
        evaluation = evaluate(state, player_pid, engine)
        revealed = bool(engine.is_terminal(state)) and state.get("env", {}).get("last_action") == "showdown"
        ctx = TeachContext(
            human_pid=player_pid,
            observation=observation,
            legal_actions=legal_actions,
            evaluation=evaluation,
            revealed=revealed,
            hand=extract_hand(observation, player_pid),
            legal_count=len(legal_actions),
            reference=None,
        )
        if solver is not None:
            reference = Coach.reference_action(state, player_pid, engine, solver)
            if reference is not None:
                ctx.reference = getattr(reference, "canonical_key", None)
        assert_no_hidden(ctx)
        return ctx

    @staticmethod
    def reference_action(
        state: dict[str, Any],
        player_pid: str,
        engine: Any,
        solver: SolverHandle,
    ) -> Any | None:
        """在玩家座位上算一手参考动作（求解器按状态当前行动者推理）.

        失败（求解器异常 / 无合法动作 / 返回 ``None``）一律静默降级为
        "无参考"——教学讲解是增值项，绝不阻断对局主流程。
        """
        try:
            if engine.get_current_player(state) != player_pid:
                return None
            return solver.select_action(state)
        except Exception:  # noqa: BLE001 - 教练通道 fail-soft，不影响对局
            return None

    @staticmethod
    def review(
        pre: TeachContext,
        actual: Any,
        describe: Callable[[Any], str],
    ) -> TeachContext:
        """把玩家实际动作并入行动前的教学上下文（``teach_move`` 讲评用）.

        Args:
            pre: 玩家行动前构建的教学上下文（``hand`` / ``reference`` 都
                属于决策时刻）。
            actual: 玩家实际提交的 :class:`ActionInstance`。
            describe: 动作描述器（``GameSpec.describe_action``）。

        Returns:
            新上下文：``player_action`` 为人读描述，``matched`` 指明是否
            与教练参考一致。
        """
        actual_key = getattr(actual, "canonical_key", None)
        return replace(
            pre,
            player_action=describe(actual),
            matched=(pre.reference is not None and actual_key == pre.reference),
        )


def extract_hand(observation: dict[str, Any], player_pid: str) -> list[str]:
    """从**玩家自己的投影**里提取手牌 id（不是 ground arrays！）.

    依次尝试本游戏族的私有视图命名（见 :data:`_HAND_VIEW_SUFFIXES`）；
    视图实体带 ``id``（或 ``value``）字段才算命中。对手的视图实体在投影
    时已被 visibility 规则剥掉 ``id``，本函数也只按"自己的视图名"取，
    双保险不会把别人的牌当自己的讲。
    """
    seat = player_pid[2:] if player_pid.startswith("p_") else player_pid
    seen: set[str] = set()
    for template in _HAND_VIEW_SUFFIXES:
        key = template.format(pid=player_pid, seat=seat)
        if key in seen:
            continue
        seen.add(key)
        entities = observation.get(key)
        if not isinstance(entities, list):
            continue
        cards = [
            str(entity["id"] if entity.get("id") else entity["value"])
            for entity in entities
            if isinstance(entity, dict) and (entity.get("id") or entity.get("value"))
        ]
        if cards:
            return cards
    return []
