"""layer4_interface.agent — Agent 陪伴后端（Layer 4，"LLM + Skill" 对话引擎）.

确定性半边：人格（Persona）+ 技能（Skills）+ 模板兜底 + 隐藏信息守卫；
LLM 半边（统一 ``layer2_engine.core.llm.LLMClient``）可选，失败回退兜底
台词。公开面由本模块导出，供集成阶段 / 复盘（C4）/ 前端（C5）按冻结契约
使用。
"""

from __future__ import annotations

from layer2_engine.core.llm import LLMClient

from .coach import Coach, TeachContext, extract_hand
from .dialogue_engine import AgentMessage, DialogueEngine
from .evaluation import evaluate
from .hidden_guard import assert_no_hidden, scan
from .persona import PERSONAS, Persona
from .scenarios import SCENARIOS
from .skills import SkillContext, Skills

# 兼容别名：旧名 ``OllamaClient`` 已并入统一客户端（LLM 统一改造）。
OllamaClient = LLMClient

__all__ = [
    "AgentMessage",
    "Coach",
    "DialogueEngine",
    "Persona",
    "PERSONAS",
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
