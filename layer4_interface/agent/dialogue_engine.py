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
from .persona import Persona, persona_identity_block
from .skills import SkillContext

logger = logging.getLogger(__name__)

#: 思维链（reasoning）清洗上限（字符）——与 conversations 存档预算一致。
_REASONING_MAX = 4000

#: LLM 成文的 token 预算（传给 ``complete_chat_reply`` 的 ``max_tokens``）。
#: 与 ``max_len``（可见正文清洗上限，字符）**解耦**：推理模型（qwen3 /
#: DeepSeek-R1 等）的 ``<think>…</think>`` 思考阶段也计入 ``max_tokens``，
#: 预算过小会把思考阶段截断在未闭合的 ``<think>`` 内 → 正文 ``content``
#: 为空、只产出 reasoning → 对话引擎误判「空回复」回退兜底台词。512 给
#: 思考 + 简短正文都留出空间；可见回复仍由 ``max_len`` 截到 100 字符内
#: （保持既有「简洁」约束）。非推理模型同样适用（多出的预算不影响）。
_REPLY_MAX_TOKENS = 1592

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
    # 对手模式（二人非教练；见 agent/opponent.py）
    "opp_react": "neutral",
    "opp_read": "thinking",
    "opp_taunt": "thinking",
}

#: 教学场景 → 中文场景名（``_scenario_payload`` 的 kind 字段）。
_TEACH_KINDS = {
    "teach_greet": "教学局开局",
    "teach_turn": "轮到玩家读牌",
    "teach_move": "教学讲评",
}

#: 对手场景 → 中文场景名（``_scenario_payload`` 的 kind 字段）。
_OPP_KINDS = {
    "opp_react": "对手反应",
    "opp_read": "对手读人",
    "opp_taunt": "对手小心思",
}

#: 终局场景——``_scenario_payload`` 为其注入 ``winner`` / ``winners`` /
#: ``outcome`` 事实。``game_over`` 的载荷原本只有位置评估 ``summary``（如
#: ``"p_white 获胜"``）与 ``kind="对局结束"``，不含「谁赢了 / 哪个 pid 是
#: AI」——LLM 无 pid→角色映射依据，会幻觉胜负（典型：温柔人设在 AI 真胜
#: 时反说「你赢了」）。注入面向说话身份的自包含 ``outcome`` 闭环此缺口。
_ENDGAME_SCENARIOS = ("ai_win", "ai_lose", "game_over")


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
        max_tokens: int | None = None,
    ) -> None:
        self.persona = persona
        self.llm = llm
        self.max_len = max_len
        self.dedup_window_s = dedup_window_s
        # LLM token 预算（与可见正文清洗上限 ``max_len`` 解耦，见
        # :data:`_REPLY_MAX_TOKENS`）。``None`` → 用模块默认；显式传入可按
        # 部署调（推理模型可调大、严格成本可调小）。
        self.max_tokens = _REPLY_MAX_TOKENS if max_tokens is None else max_tokens
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
                ``teaching=True`` 标记与教学事实；二人非教练对手模式下是
                :class:`~layer4_interface.agent.opponent.OpponentContext`，
                携带 ``adversarial=True`` 标记与对手机械事实）。
            scenario: 场景键（``SCENARIOS`` 之一）。
            game_id: 当前对局的注册表游戏 id（内置游戏）。携带时把权威
                资料（简介 + 玩法规则段）注入成文 prompt——persona 提到
                玩法时依据资料而非参数记忆；custom / 空值则不注入。

        Returns:
            清洗、扫描、去重后的 :class:`AgentMessage`。

        Note:
            泄露扫描按陪伴身份四态分派：``revealed``（揭底）优先级最高 →
            全放行；其次 ``adversarial``（对手，放行 AI 自己的牌、拦玩家
            牌）；其次 ``teaching``（教练，放行玩家牌、拦 AI 牌）；默认全拦。
            身份标记取自 ``ctx``（``adversarial`` / ``teaching`` /
            ``revealed``），非该身份的上下文按默认全拦。
        """
        if self.muted:
            return AgentMessage("", "neutral")

        teaching = bool(getattr(ctx, "teaching", False))
        adversarial = bool(getattr(ctx, "adversarial", False))
        revealed = bool(getattr(ctx, "revealed", False))
        text, reasoning = self._generate(ctx, scenario, game_id, teaching=teaching, adversarial=adversarial)
        text = self._clean(text)
        reasoning = self._clean_reasoning(reasoning)
        # 泄露扫描按观测形态自行推断游戏（不变更红线语义）；
        # game_id 参数只服务于知识注入。
        scan_game = infer_game_id(ctx.observation)
        text = scan(text, scan_game, teaching=teaching, adversarial=adversarial, revealed=revealed)

        key = (scenario, self.persona.key, _state_hash(ctx))
        now = time.monotonic()
        previous = self._sent.get(key)
        if previous is not None and now - previous[0] < self.dedup_window_s:
            alternate = self._pick_fallback(scenario, avoid=previous[1])
            text = self._clean(alternate)
            text = scan(text, scan_game, teaching=teaching, adversarial=adversarial, revealed=revealed)

        self._sent[key] = (now, text)
        return AgentMessage(text, _SCENARIO_MOODS.get(scenario, "neutral"), reasoning=reasoning)

    def _generate(
        self,
        ctx: SkillContext,
        scenario: str,
        game_id: str = "",
        *,
        teaching: bool = False,
        adversarial: bool = False,
    ) -> tuple[str, str]:
        """LLM 成文，失败或无 LLM 时回退兜底台词；返回 ``(text, reasoning)``."""
        if self.llm is not None:
            system = self._system_prompt(teaching=teaching, adversarial=adversarial)
            user = self._user_prompt(ctx, scenario, game_id)
            try:
                reply = self.llm.complete_chat_reply(system, user, self.max_tokens)
                text, reasoning = reply.text, reply.reasoning
            except Exception as exc:  # noqa: BLE001 — fail-soft 客户端一般不抛；兜测试注入
                logger.warning("对话 LLM 调用异常，回退兜底台词: %s", exc)
                text, reasoning = "", ""
            if text:
                return text, reasoning
            # 正文为空：区分「只产出思维链」与「全空」——前者典型是推理模型
            # 思考阶段耗尽 ``max_tokens``（``content`` 空、``reasoning`` 非空），
            # 给一条可操作诊断（调大 ``max_tokens``）而非泛化「空回复」告警。
            last = getattr(self.llm, "last_error", None)
            if last is not None:
                logger.warning("对话 LLM 未产出正文（%s），回退兜底台词", last)
            elif reasoning:
                logger.warning(
                    "对话 LLM 仅产出思维链未产出正文（reasoning 长度=%d），"
                    "可能 max_tokens=%d 不足截断了思考阶段，回退兜底台词",
                    len(reasoning),
                    self.max_tokens,
                )
            else:
                logger.warning("对话 LLM 未产出内容（空回复），回退兜底台词")
        return self._pick_fallback(scenario, avoid=None), ""

    def _system_prompt(self, *, teaching: bool = False, adversarial: bool = False) -> str:
        identity = persona_identity_block(self.persona)
        if teaching:
            return (
                f"你是 Gavis 教练 Agent（教学对局）。{identity}\n"
                "用中文回复，简洁，符合你的性格。"
                "教学对局：你能看到玩家自己的牌（与玩家所见完全一致），"
                "可以并且应该围绕它讲解思路、点评玩家刚才的打法。"
                "红线：绝不提及或猜测任何对手/其他玩家的未公开信息"
                "（手牌、身份、底牌、未翻开的牌等）——你也看不到它们。"
                "游戏规则只依据资料栏，资料没有的细节不要编造。"
            )
        if adversarial:
            return (
                f"你是 Gavis 对手 Agent（二人非教练对局）。{identity}\n"
                "用中文回复，简洁，符合你的性格。"
                "你是玩家的座内对手：你能看到**自己**的手牌/底牌（仅供你判断"
                "牌力、决定下注与是否虚张声势），可以围绕**牌力强弱**讲思路、"
                "对玩家刚才的公开动作做合理推断（读人）。"
                "红线一：绝不提及或猜测玩家的未公开信息（玩家的底牌、手牌、"
                "身份等）——你也看不到它们；只能基于玩家公开的下注/弃牌/"
                "摸打序列推断意图，不要报玩家未公开牌面。"
                "红线二：绝不报出**你自己**底牌的具体花色与点数（如「黑桃4」"
                "「♠A」「s10」），只能说「这手还行」「牌不大」「一对K」这类"
                "模糊牌力——报出具体牌面等于明牌，会直接毁掉这局。"
                "终局 showdown 揭底后双方牌公开，可做完整复盘式点评。"
                "游戏规则只依据资料栏，资料没有的细节不要编造。"
            )
        return (
            f"你是 Gavis 陪玩 Agent。{identity}\n"
            "用中文回复，简洁，符合你的性格。"
            "红线：不得编造任何对手/其他玩家的未公开信息"
            "（手牌、身份、底牌、未翻开的牌等）；"
            "需要局面细节时依据机械事实与玩家自己可见的信息。"
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


def _endgame_outcome(ctx: SkillContext) -> dict[str, Any]:
    """从观测派生终局胜负事实（防 LLM 幻觉「谁赢了」）.

    返回 ``{"winner", "winners", "outcome"}``。``outcome`` 面向说话身份、自
    包含——不依赖 pid 名称，模型无需做 pid→角色映射：

    - 对手模式（``adversarial``）：说话身份 = AI 对手，``ctx.ai_pid`` 即
      「你（AI）」；胜者等于它即「你（AI）获胜」。
    - 啦啦队 / 教练（默认）：说话身份 = 站在玩家身后的陪伴，「AI」指对局
      AI；玩家 id = ``ctx.human_pid``，玩家未胜即「AI 获胜（玩家落败）」。
    - 多胡局（``winners`` 列表）：玩家在 ``winners`` 中即计玩家胜，避免血
      战等多胡局误判玩家落败。

    终局事实取自公开 ``env``（``project_observation`` 投影 env 不被 visibility
    过滤），不触碰任何隐藏数组。
    """
    env = ctx.observation.get("env") if isinstance(ctx.observation, dict) else {}
    if not isinstance(env, dict):
        env = {}
    winner = env.get("winner")
    raw_winners = env.get("winners")
    winners = list(raw_winners) if isinstance(raw_winners, list) else []

    if bool(getattr(ctx, "adversarial", False)):
        ai_pid = getattr(ctx, "ai_pid", "") or ctx.human_pid
        if winner == ai_pid:
            outcome = "你（AI）获胜"
        elif winner is None and not winners:
            outcome = "平局"
        else:
            outcome = "你（AI）落败，玩家获胜"
    else:
        player_pid = ctx.human_pid
        player_won = winner == player_pid or player_pid in winners
        if player_won:
            outcome = "AI 落败（玩家获胜）"
        elif winner is None and not winners:
            outcome = "平局"
        else:
            outcome = "AI 获胜（玩家落败）"
    return {"winner": winner, "winners": winners, "outcome": outcome}


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
    # 终局场景注入胜负事实（防 LLM 幻觉「谁赢了」）：``outcome`` 面向说话
    # 身份自包含——对手模式「你（AI）」、啦啦队/教练「AI」对玩家——不依赖
    # pid 名称，模型无需 pid→角色映射。
    if scenario in _ENDGAME_SCENARIOS:
        payload.update(_endgame_outcome(ctx))
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
    if bool(getattr(ctx, "adversarial", False)):
        # 对手事实（OpponentContext；AI 自己的牌 + 玩家公开动作序列——
        # 观测是 AI 自己的投影，玩家的隐藏信息从来进不来）。
        payload["kind"] = _OPP_KINDS.get(scenario, payload["kind"])
        family = _payload_family(ctx, game_id)
        ai_hand = list(getattr(ctx, "ai_hand", None) or [])
        if ai_hand:
            # AI 自己的牌 id（s1…/hT…）→ 中文牌名：对手读「我手里一对 K」
            # 而不是「hK hK」。adversarial scan 放行「我的牌」。
            payload["ai_hand"] = piece_names(family, ai_hand)
        player_actions = list(getattr(ctx, "player_actions", None) or [])
        if player_actions:
            # 玩家公开动作序列（人读描述，如「跟注 2」「打出 一条」）——
            # 读人的唯一依据，绝不包含玩家未公开牌面。
            payload["player_actions"] = player_actions
    return payload
