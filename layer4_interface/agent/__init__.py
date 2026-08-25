"""layer4_interface.agent — Agent 陪伴后端（Layer 4，"LLM + Skill" 对话引擎）.

确定性半边：人格（Persona）+ 技能（Skills）+ 模板兜底 + 隐藏信息守卫；
LLM 半边（OllamaClient）可选，失败回退兜底台词。公开面由本模块导出，
供集成阶段 / 复盘（C4）/ 前端（C5）按冻结契约使用。
"""

from __future__ import annotations

from .dialogue_engine import AgentMessage, DialogueEngine
from .evaluation import evaluate
from .hidden_guard import assert_no_hidden, scan
from .llm_client import OllamaClient
from .persona import PERSONAS, Persona
from .scenarios import SCENARIOS
from .skills import SkillContext, Skills

__all__ = [
    "AgentMessage",
    "DialogueEngine",
    "Persona",
    "PERSONAS",
    "SCENARIOS",
    "SkillContext",
    "Skills",
    "assert_no_hidden",
    "evaluate",
    "scan",
    "OllamaClient",
]
