"""opponent — 二人非教练对局的「座内对手」陪伴（Layer 4，"LLM + Skill" 的对手半边）.

非教练二人局下，陪伴的身份从「站在玩家身后的啦啦队」翻成「你的对手」。
数据入口因此从玩家投影（``Skills.build`` → ``project_observation(state,
human_pid)``，与 ``Coach`` 同构）换成 **AI 投影**：

- ``observation = engine.project_observation(state, ai_pid)`` —— AI 自己的
  投影：含 AI 自己的手牌/底牌视图 + 公共牌 + 玩家公开动作序列。**不含**
  玩家底牌（visibility 规则本就不给 AI 玩家底牌）。
- ``ai_hand``：从 AI 投影的私有视图提取（复用 :func:`coach.extract_hand`
  的视图命名约定）。
- ``player_actions``：玩家本局公开动作序列（从 ``session.log`` 过滤
  ``actor == "human"``），供「读人」——真人对手也会基于公开下注/弃牌/
  摸打序列推断玩家意图，红线允许。
- 经 :func:`hidden_guard.assert_no_hidden` 校验：AI 投影里本就不该有玩家
  隐藏字段，守卫拦的是 ground-array 键名（``sb_hole`` / ``hand_p0`` …），
  AI 自己的视图名（``sb_hole_view`` / ``hand_view_p1``）不在黑名单。

红线镜像变体（与 ``coach`` 的 teaching 镜像）：

| 模式 | 自称 | 「你」指 | 可讲 | 仍拦 |
|------|------|----------|------|------|
| teaching | 教练「我」 | 玩家 | 玩家的牌（含具体牌面） | AI/对手的牌 |
| adversarial | AI 对手「我」 | 玩家 | **AI 的牌力措辞**（「一对K」等模糊） | **玩家的隐藏牌** + **具体花色点数**（黑桃4/♠A） |

AI 对手讲自己的**牌力**是其本分（模糊措辞如「我手里一对K」「这手同花」，
它本就看得到自己的牌），不向玩家泄露**新的**隐藏信息；但**具体花色点数**
（黑桃4 / ♠A）一律不报——报牌等于明牌，破坏二人博弈（banter 人设「不报
牌」）。玩家底牌仍由 visibility 规则 + adversarial scan 双保险拦住。
``OpponentContext.adversarial=True`` 标记驱动 :func:`hidden_guard.scan`
切换到 adversarial 模式（拦「你的/玩家的 + 牌面」+ 具体牌面、放行「我的
+ 牌力措辞」）。终局 showdown 后（``revealed=True``）双方牌公开，全放行，
可做完整复盘式对手点评。

在线学习不污染：对手说话是表达层，不影响 AI 决策轨迹；``raw_solver`` 仍
只在 coach 通道用，对手模式不碰训练数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .coach import extract_hand
from .evaluation import evaluate
from .hidden_guard import assert_no_hidden
from .skills import SkillContext


@dataclass
class OpponentContext(SkillContext):
    """对手上下文 —— AI 视角恰好是 AI 自己的观测（含 AI 手牌，不含玩家底牌）.

    Attributes:
        adversarial: 恒为 True（标记 ctx 走对手路径：对手系统提示 +
            adversarial 泄露扫描变体——放行 AI 自己的牌、拦玩家的隐藏牌）。
        ai_pid: AI 对手座位 pid（观测视角；``human_pid`` 基类字段同值，
            保留基类契约供 ``assert_no_hidden`` / ``_state_hash`` 使用）。
        ai_hand: AI 自己的手牌/底牌 id 列表（从 AI 投影的私有视图提取；
            棋类游戏为空列表）。
        player_actions: 玩家本局公开动作的人读描述序列（从
            ``session.log`` 过滤 ``actor == "human"``），供对手「读人」。
            仅基于公开动作，**不含**玩家未公开牌面。
    """

    adversarial: bool = True
    ai_pid: str = ""
    ai_hand: list[str] = field(default_factory=list)
    player_actions: list[str] = field(default_factory=list)


class Opponent:
    """二人非教练对局的确定性对手：只看 AI 自己的投影，产出对手机械事实."""

    @staticmethod
    def build(
        state: dict[str, Any],
        ai_pid: str,
        player_pid: str,
        engine: Any,
        log: list[dict],
    ) -> OpponentContext:
        """从状态构建 :class:`OpponentContext`（AI 投影入口，遵守红线）.

        Args:
            state: 引擎状态（AI 行动后或玩家行动后的快照均可——AI 投影
                始终是 AI 自己的视角）。
            ai_pid: AI 对手座位 id（观测视角）。
            player_pid: 人类玩家 id（仅用于过滤 ``log`` 里的玩家公开动作；
                AI 投影本就不含玩家底牌，此参数不参与观测构造）。
            engine: :class:`layer2_engine.core.engine.GameEngine`。
            log: 会话动作日志（``session.log``），用于提取玩家公开动作序列。

        Returns:
            通过 :func:`assert_no_hidden` 校验的对手上下文（``adversarial=True``）。
        """
        observation = engine.project_observation(state, ai_pid)
        legal_actions = engine.get_legal_actions(state)
        # 评估取 AI 视角（``viewer = ai_pid``）：对手说话以「我方」占位/落后
        # 措辞自洽。牌类游戏当前评估恒中性（C4，P3 缓解），不影响 opp_* 触发。
        evaluation = evaluate(state, ai_pid, engine)
        revealed = bool(engine.is_terminal(state)) and state.get("env", {}).get("last_action") == "showdown"
        player_actions = [
            str(entry.get("action", "")) for entry in log if isinstance(entry, dict) and entry.get("actor") == "human"
        ]
        ctx = OpponentContext(
            human_pid=ai_pid,  # 基类字段 = 观测视角（AI 自己）；保留契约兼容。
            observation=observation,
            legal_actions=legal_actions,
            evaluation=evaluation,
            revealed=revealed,
            adversarial=True,
            ai_pid=ai_pid,
            ai_hand=extract_hand(observation, ai_pid),
            player_actions=player_actions,
        )
        assert_no_hidden(ctx)
        return ctx
