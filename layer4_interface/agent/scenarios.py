"""Agent 场景常量（Layer 4 陪伴后端）.

``SCENARIOS`` 是对话引擎与技能层的场景枚举：对局会话在 move 后判定
命中的场景，再交给 :class:`layer4_interface.agent.dialogue_engine.DialogueEngine`
按人格成文。冻结契约，集成阶段与前端依赖此顺序与键名。

教学对局追加三个场景（**只在末尾追加**，既有键名与顺序不变）：

- ``teach_greet`` — 教学局开局：教练自我介绍 + 规则导读。
- ``teach_turn`` — 轮到玩家：读玩家自己的牌 + 讲选择面（不剧透参考）。
- ``teach_move`` — 玩家走后：对照教练参考动作的讲评。

对手模式（二人非教练）追加三个场景（**只在末尾追加**，与教学场景同列；
见 ``agent/opponent.py``）：

- ``opp_react`` — AI 行动后：对手视角的反应（看了自己刚出的牌 + 玩家
  的公开应对），是对手身份最自然的发声点。
- ``opp_read`` — 玩家行动后：对手基于玩家公开动作序列的「读人」（吐槽
  虚张声势 / 温和点节奏 / 高冷点关键），不依赖评分命中。
- ``opp_taunt`` — 按人设的「小心思」分寸：读人 ≠ 偷牌——基于公开下注/
  弃牌/摸打序列对玩家意图的合理推断，绝不报玩家未公开牌面。
"""

from __future__ import annotations

SCENARIOS = (
    "greet",
    "good_move",
    "blunder",
    "help",
    "ai_win",
    "ai_lose",
    "illegal",
    "idle",
    "game_over",
    # ── 教学对局（teaching=True 时启用；见 agent/coach.py）──
    "teach_greet",
    "teach_turn",
    "teach_move",
    # ── 对手模式（二人非教练；见 agent/opponent.py）──
    "opp_react",
    "opp_read",
    "opp_taunt",
)
