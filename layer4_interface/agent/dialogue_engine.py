"""dialogue_engine — 对话引擎（Layer 4，"LLM + Skill" 的成文半边）.

:class:`DialogueEngine.reply` 串行管线：静音开关 → LLM 成文（失败回退
:data:`Persona.fallback_lines`）→ 清洗（长度上限 + 剔控制字符）→
:func:`hidden_guard.scan` 后置泄露扫描 → 去重（``(scenario, persona.key,
状态哈希)`` 时间窗内不重复同一句）。

游戏知识注入（audit §5-4）：``reply`` 可带 ``game_id``，成文时把
``game_knowledge.game_knowledge_text`` 拼装的权威资料（注册表简介 +
玩法文档规则段）注入 user prompt，并在 system prompt 立红线——persona
聊天提到游戏玩法时依据资料作答，不再靠模型参数记忆（幻觉面与 chat
信息工具同源修复）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from layer2_engine.core.llm import LLMClient, sanitize_text

from ..frontend.engine_helpers import canonical_family_text, game_family, piece_names
from ..frontend.platform.game_knowledge import game_knowledge_text
from .hidden_guard import infer_game_id, scan
from .persona import Persona
from .skills import SkillContext

logger = logging.getLogger(__name__)

#: 思维链（reasoning）清洗上限（字符）——与 conversations 存档预算一致。
_REASONING_MAX = 4000

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
    # 教学对局（teaching=True）
    "teach_greet": "neutral",
    "teach_turn": "thinking",
    "teach_move": "thinking",
}

#: 教学场景 → 中文场景名（``_scenario_payload`` 的 kind 字段）。
_TEACH_KINDS = {
    "teach_greet": "教学局开局",
    "teach_turn": "轮到玩家读牌",
    "teach_move": "教学讲评",
}


@dataclass
class AgentMessage:
    """一条 Agent 消息."""

    text: str
    mood: str  # happy / thinking / sorry / neutral
    #: 思维链（模型思考过程；统一客户端 reasoning 透传，前端以折叠块展示）。
    reasoning: str = ""


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

    def reply(self, ctx: SkillContext, scenario: str, *, game_id: str = "") -> AgentMessage:
        """生成一条场景消息（LLM 成文 → 失败回退兜底台词）.

        Args:
            ctx: 技能上下文（唯一数据入口产出；教学对局下是
                :class:`~layer4_interface.agent.coach.TeachContext`，携带
                ``teaching=True`` 标记与教学事实）。
            scenario: 场景键（``SCENARIOS`` 之一）。
            game_id: 当前对局的注册表游戏 id（内置游戏）。携带时把权威
                资料（简介 + 玩法规则段）注入成文 prompt——persona 提到
                玩法时依据资料而非参数记忆；custom / 空值则不注入。

        Returns:
            清洗、扫描、去重后的 :class:`AgentMessage`。
        """
        if self.muted:
            return AgentMessage("", "neutral")

        teaching = bool(getattr(ctx, "teaching", False))
        text, reasoning = self._generate(ctx, scenario, game_id)
        text = self._clean(text)
        reasoning = self._clean_reasoning(reasoning)
        # 泄露扫描按观测形态自行推断游戏（不变更红线语义）；
        # game_id 参数只服务于知识注入。
        scan_game = infer_game_id(ctx.observation)
        text = scan(text, scan_game, teaching=teaching)

        key = (scenario, self.persona.key, _state_hash(ctx))
        now = time.monotonic()
        previous = self._sent.get(key)
        if previous is not None and now - previous[0] < self.dedup_window_s:
            alternate = self._pick_fallback(scenario, avoid=previous[1])
            text = self._clean(alternate)
            text = scan(text, scan_game, teaching=teaching)

        self._sent[key] = (now, text)
        return AgentMessage(text, _SCENARIO_MOODS.get(scenario, "neutral"), reasoning=reasoning)

    def _generate(self, ctx: SkillContext, scenario: str, game_id: str = "") -> tuple[str, str]:
        """LLM 成文，失败或无 LLM 时回退兜底台词；返回 ``(text, reasoning)``."""
        if self.llm is not None:
            system = self._system_prompt(bool(getattr(ctx, "teaching", False)))
            user = self._user_prompt(ctx, scenario, game_id)
            try:
                reply = self.llm.complete_chat_reply(system, user, self.max_len)
                text, reasoning = reply.text, reply.reasoning
            except Exception as exc:  # noqa: BLE001 — fail-soft 客户端一般不抛；兜测试注入
                logger.warning("对话 LLM 调用异常，回退兜底台词: %s", exc)
                text, reasoning = "", ""
            if text:
                return text, reasoning
            # 传输故障定性（统一客户端 fail-soft 时异常不抛出，真实原因在
            # last_error）；兜底台词是刻意设计，但失败必须可观测。
            last = getattr(self.llm, "last_error", None)
            if last is not None:
                logger.warning("对话 LLM 未产出内容（%s），回退兜底台词", last)
            else:
                logger.warning("对话 LLM 未产出内容（空回复），回退兜底台词")
        return self._pick_fallback(scenario, avoid=None), ""

    def _system_prompt(self, teaching: bool = False) -> str:
        if teaching:
            return (
                f"你是 Gavis 教练 Agent（教学对局），性格：{self.persona.display_name}，"
                f"语气：{self.persona.tone}。用中文回复，简洁，符合你的性格。"
                "教学对局：你能看到玩家自己的牌（与玩家所见完全一致），"
                "可以并且应该围绕它讲解思路、点评玩家刚才的打法。"
                "红线：绝不提及或猜测 AI/对手 的底牌、手牌、身份、未翻开的牌——你也看不到它们。"
                "游戏规则只依据资料栏，资料没有的细节不要编造。"
            )
        return (
            f"你是 Gavis 陪玩 Agent，性格：{self.persona.display_name}，语气：{self.persona.tone}。"
            "用中文回复，简洁，符合你的性格。"
            "严格遵守隐藏信息红线：不得提及德州底牌、麻将手牌、狼人身份、未翻开的牌。"
            "提到当前游戏的规则/玩法时，只依据资料栏给出的内容，资料没有的细节不要编造。"
        )

    def _user_prompt(self, ctx: SkillContext, scenario: str, game_id: str = "") -> str:
        payload = _scenario_payload(ctx, scenario, game_id=game_id)
        parts = [f"场景：{scenario}", f"机械事实：{json.dumps(payload, ensure_ascii=False, default=str)}"]
        # 权威资料（与 chat 信息工具同源）：内置游戏注入简介 + 规则段；
        # custom / 未知 id 返回空串 → 不注入（fail-soft）。
        knowledge = game_knowledge_text(game_id)
        if knowledge:
            parts.append(f"当前游戏资料（权威，玩法以此为准）：\n{knowledge}")
        return "\n".join(parts)

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

    def _clean_reasoning(self, reasoning: str) -> str:
        """思维链清洗：剔控制字符 + 上限（与前端展示/存档预算对齐）。"""
        return sanitize_text(reasoning, _REASONING_MAX).strip()


def _state_hash(ctx: SkillContext) -> str:
    """由投影观测生成稳定状态哈希（去重键的第三元）."""
    try:
        serialized = json.dumps(ctx.observation, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        serialized = repr(ctx.observation)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _payload_family(ctx: SkillContext, game_id: str) -> str:
    """教学载荷的牌名读法族：优先显式 ``game_id``，其次按观测推断.

    custom / 未知 id 时用 :func:`hidden_guard.infer_game_id` 的观测形态
    推断；推断不出返回 ``"unknown"``（上层 fail-soft 直出原 id）。
    """
    family = game_family(game_id)
    if family != "unknown":
        return family
    return game_family(infer_game_id(ctx.observation))


def _scenario_payload(ctx: SkillContext, scenario: str, *, game_id: str = "") -> dict[str, Any]:
    """把场景与评估打包成机械事实（供 LLM 成文参考）.

    Args:
        game_id: 会话游戏 id（custom / 空值时可从观测推断）；用于把
            手牌 id / 参考动作 canonical key 译成中文名 —— 这就是
            “传给 LLM 的信息不过分技术化”的对话侧出口。
    """
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
    if bool(getattr(ctx, "teaching", False)):
        # 教学事实（TeachContext；仅玩家自己的牌 + 参考动作对比——
        # 观测是玩家自己的投影，AI/对手的隐藏信息从来进不来）。
        payload["kind"] = _TEACH_KINDS.get(scenario, payload["kind"])
        family = _payload_family(ctx, game_id)
        hand = list(getattr(ctx, "hand", None) or [])
        if hand:
            # 手牌 id（s1…）→ 中文牌名（一条…）：LLM 读“一条”而不是“s1”。
            payload["player_hand"] = piece_names(family, hand)
        payload["legal_count"] = int(getattr(ctx, "legal_count", 0) or 0)
        reference = getattr(ctx, "reference", None)
        if reference is not None:
            payload["coach_reference"] = canonical_family_text(family, reference)
            payload["coach_reference_key"] = reference  # 机器键保留给校验/回放
        player_action = getattr(ctx, "player_action", None)
        if player_action is not None:
            payload["player_action"] = player_action
            payload["matched_reference"] = bool(getattr(ctx, "matched", None))
    return payload
