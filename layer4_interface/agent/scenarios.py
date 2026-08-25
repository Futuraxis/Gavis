"""Agent 九场景常量（Layer 4 陪伴后端）.

``SCENARIOS`` 是对话引擎与技能层的场景枚举：对局会话在 move 后判定
命中的场景，再交给 :class:`layer4_interface.agent.dialogue_engine.DialogueEngine`
按人格成文。冻结契约，集成阶段与前端依赖此顺序与键名。
"""

from __future__ import annotations

SCENARIOS = ("greet", "good_move", "blunder", "help", "ai_win", "ai_lose", "illegal", "idle", "game_over")
