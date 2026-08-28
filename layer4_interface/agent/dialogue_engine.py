"""dialogue_engine — 对话引擎（Layer 4，"LLM + Skill" 的成文半边）.

:class:`DialogueEngine.reply` 串行管线：静音开关 → LLM 成文（失败回退
:data:`Persona.fallback_lines`）→ 清洗（长度上限 + 剔控制字符）→
:func:`hidden_guard.scan` 后置泄露扫描 → 去重（``(scenario, persona.key,
状态哈希)`` 时间窗内不重复同一句）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from layer2_engine.core.llm import LLMClient, sanitize_text

from .hidden_guard import infer_game_id, scan
from .persona import Persona
from .skills import SkillContext

#: 允许的情绪标签集合（前端头像 / 表情据此渲染）。
_MOODS = ("happy", "thinking", "sorry", "neutral")

#: 场景 → 默认情绪。
_SCENARIO_MOODS = {
    "greet": "neutral",
    "good_move": "happy",
    "blunder": "thinking",
    "help": "thinking",
    "ai_win": "neutral",
    "ai_lose": "happy",
    "illegal": "sorry",
    "idle": "neutral",
    "game_over": "neutral",
}

@dataclass
class AgentMessage:
    """一条 Agent 消息."""

    text: str
    mood: str  # happy / thinking / sorry / neutral


class DialogueEngine:
    """按人格成文，LLM 失败回退兜底台词，串行清洗 / 扫描 / 去重."""

    def __init__(
        self,
        persona: Persona,
        llm: LLMClient | None = None,
        *,
        max_len: int = 100,
        dedup_window_s: float = 300,
    ) -> None:
        self.persona = persona
        self.llm = llm
        self.max_len = max_len
        self.dedup_window_s = dedup_window_s
        self.muted = False
        self._sent: dict[tuple[str, str, str], tuple[float, str]] = {}
        self._fallback_cursor: dict[str, int] = {}

    def set_muted(self, muted: bool) -> None:
        """开关静音；静音时 :meth:`reply` 返回空消息."""
        self.muted = muted

    def reply(self, ctx: SkillContext, scenario: str) -> AgentMessage:
        """生成一条场景消息（LLM 成文 → 失败回退兜底台词）.

        Args:
            ctx: 技能上下文（唯一数据入口产出）。
            scenario: 场景键（``SCENARIOS`` 之一）。

        Returns:
            清洗、扫描、去重后的 :class:`AgentMessage`。
        """
        if self.muted:
            return AgentMessage("", "neutral")

        text = self._generate(ctx, scenario)
        text = self._clean(text)
        game_id = infer_game_id(ctx.observation)
        text = scan(text, game_id)

        key = (scenario, self.persona.key, _state_hash(ctx))
        now = time.monotonic()
        previous = self._sent.get(key)
        if previous is not None and now - previous[0] < self.dedup_window_s:
            alternate = self._pick_fallback(scenario, avoid=previous[1])
            text = self._clean(alternate)
            text = scan(text, game_id)

        self._sent[key] = (now, text)
        return AgentMessage(text, _SCENARIO_MOODS.get(scenario, "neutral"))

    def _generate(self, ctx: SkillContext, scenario: str) -> str:
        """LLM 成文，失败或无 LLM 时回退兜底台词."""
        if self.llm is not None:
            system = self._system_prompt()
            user = self._user_prompt(ctx, scenario)
            try:
                text = self.llm.complete_chat(system, user, self.max_len)
            except Exception:
                text = ""
            if text:
                return text
        return self._pick_fallback(scenario, avoid=None)

    def _system_prompt(self) -> str:
        return (
            f"你是 Gavis 陪玩 Agent，性格：{self.persona.display_name}，语气：{self.persona.tone}。"
            "用中文回复，简洁，符合你的性格。"
            "严格遵守隐藏信息红线：不得提及德州底牌、麻将手牌、狼人身份、未翻开的牌。"
        )

    def _user_prompt(self, ctx: SkillContext, scenario: str) -> str:
        payload = _scenario_payload(ctx, scenario)
        return f"场景：{scenario}\n机械事实：{json.dumps(payload, ensure_ascii=False, default=str)}"

    def _pick_fallback(self, scenario: str, avoid: str | None) -> str:
        """轮换选择兜底台词；有备选时尽量避开 ``avoid``."""
        lines = self.persona.fallback_lines.get(scenario, [])
        if not lines:
            lines = self.persona.fallback_lines.get("idle", []) or ["……"]
        if avoid is not None and len(lines) > 1:
            candidates = [line for line in lines if line != avoid]
            if candidates:
                lines = candidates
        index = self._fallback_cursor.get(scenario, 0) % len(lines)
        self._fallback_cursor[scenario] = index + 1
        return lines[index]

    def _clean(self, text: str) -> str:
        """剔除控制字符并截断到 ``max_len`` 字符（统一清洗，layer2_engine.core.llm）。"""
        return sanitize_text(text, self.max_len).strip()


def _state_hash(ctx: SkillContext) -> str:
    """由投影观测生成稳定状态哈希（去重键的第三元）."""
    try:
        serialized = json.dumps(ctx.observation, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        serialized = repr(ctx.observation)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _scenario_payload(ctx: SkillContext, scenario: str) -> dict[str, Any]:
    """把场景与评估打包成机械事实（供 LLM 成文参考）."""
    score = float(ctx.evaluation.get("score", 0.0))
    payload: dict[str, Any] = {
        "scenario": scenario,
        "score": score,
        "summary": ctx.evaluation.get("summary", ""),
        "revealed": ctx.revealed,
    }
    kind_map = {
        "greet": "开局问候",
        "good_move": "玩家好棋",
        "blunder": "玩家失误",
        "help": "玩家请求帮助",
        "ai_win": "AI 获胜",
        "ai_lose": "AI 落败",
        "illegal": "玩家违规操作",
        "idle": "玩家长时间未操作",
        "game_over": "对局结束",
    }
    payload["kind"] = kind_map.get(scenario, scenario)
    return payload
