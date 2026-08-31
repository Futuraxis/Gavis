"""layer4_interface.agent — Agent 陪伴后端（Layer 4，"LLM + Skill" 对话引擎）.

确定性半边：人格（Persona）+ 技能（Skills）+ 模板兜底 + 隐藏信息守卫；
LLM 半边（统一 ``layer2_engine.core.llm.LLMClient``）可选，失败回退兜底
台词。公开面由本模块导出，供集成阶段 / 复盘（C4）/ 前端（C5）按冻结契约
使用。

陪伴身份（按场景，见 ``docs/design/companion-redesign.md``）：

- **教练**（``Coach`` / ``TeachContext``）：教学对局，看玩家自己的牌，
  ``teaching`` 扫描放行玩家牌、拦 AI 牌。
- **对手**（``Opponent`` / ``OpponentContext``）：二人非教练，看 AI 自己
  的牌 + 玩家公开动作序列，``adversarial`` 扫描放行 AI 自己的牌、拦玩家
  的隐藏牌（与 teaching 镜像）。
- **啦啦队**（``Skills`` / ``SkillContext``）：默认 fallback，看玩家投影，
  默认扫描全拦（多人非教练在 P2 前的过渡身份）。
"""

from __future__ import annotations

from layer2_engine.core.llm import LLMClient

from .coach import Coach, TeachContext, extract_hand
from .dialogue_engine import AgentMessage, DialogueEngine
from .evaluation import evaluate
from .hidden_guard import assert_no_hidden, scan
from .opponent import Opponent, OpponentContext
from .persona import PERSONAS, Persona, persona_identity_block
from .scenarios import SCENARIOS
from .skills import SkillContext, Skills

# 兼容别名：旧名 ``OllamaClient`` 已并入统一客户端（LLM 统一改造）。
OllamaClient = LLMClient

__all__ = [
    "AgentMessage",
    "Coach",
    "DialogueEngine",
    "Opponent",
    "OpponentContext",
    "Persona",
    "PERSONAS",
    "persona_identity_block",
    "SCENARIOS",
    "SkillContext",
    "Skills",
    "TeachContext",
    "assert_no_hidden",
    "evaluate",
    "extract_hand",
    "scan",
    "LLMClient",
    "OllamaClient",
]
