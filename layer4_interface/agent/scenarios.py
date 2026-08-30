"""Agent 场景常量（Layer 4 陪伴后端）.

``SCENARIOS`` 是对话引擎与技能层的场景枚举：对局会话在 move 后判定
命中的场景，再交给 :class:`layer4_interface.agent.dialogue_engine.DialogueEngine`
按人格成文。冻结契约，集成阶段与前端依赖此顺序与键名。

教学对局追加三个场景（**只在末尾追加**，既有键名与顺序不变）：

- ``teach_greet`` — 教学局开局：教练自我介绍 + 规则导读。
- ``teach_turn`` — 轮到玩家：读玩家自己的牌 + 讲选择面（不剧透参考）。
- ``teach_move`` — 玩家走后：对照教练参考动作的讲评。
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
)
