"""hidden_guard — Agent 对话隐藏信息守卫（Layer 4，红线）.

两道防线把隐藏信息（德州底牌、麻将手牌、狼人身份、未翻牌堆）挡在每一条
Agent 发言之外：

1. :func:`assert_no_hidden` 拒绝投影观测里仍携带黑名单字段名的
   :class:`SkillContext` —— 拦截任何把 ``state["_arrays"]``（或原始
   snapshot）混入观测、而非经由 ``engine.project_observation`` 的路径。
2. :func:`scan` 对生成文本做后置令牌扫描，命中德州底牌记法（如
   ``♠A ♥K``）、"我的底牌" 等模式时把对应句改写为通用语。

黑名单字段名来自各游戏 ``rules/*.json`` 的 ``visibility`` 声明与平台
snapshot：投影观测用 *视图名*（``sb_hole_view`` / ``hand_view_p0`` /
``my_role``），永远不该出现这些原始隐藏键。

教学对局（``teaching=True``）下 :func:`scan` 切换到 *教学模式泄露
模式*：玩家自己的牌可以讨论（教练看的正是玩家自己的投影，见
``coach.py``），只有 **AI/对手的** 隐藏信息仍然拦截。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .skills import SkillContext

#: 投影观测里绝不允许出现的隐藏字段名（ground-array / snapshot 键）。
HIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        # 德州 —— 底牌与已发/未发牌堆
        "sb_hole",
        "bb_hole",
        "drawn",
        "my_hole",
        "ai_hole",
        # 麻将 —— 各家手牌、未摸牌墙、胡牌手牌
        "hand_p0",
        "hand_p1",
        "hand_p2",
        "hand_p3",
        "my_hand",
        "ai_hand",
        "win_hand",
        # 狼人杀 —— 身份分配与预言家的私密查验结果
        "roles",
        "seerResult",
    }
)

#: 泄露句被改写成的通用语（不透露任何牌面信息）。
_GENERIC_REWRITE = "这把牌先不细说。"

#: 单张牌记法：花色（符号或 S/H/D/C）+ 点数（10 或 2-9/J/Q/K/A）。
_CARD_TOKEN = r"(?:[♠♥♦♣]|[SHDC])(?:10|[2-9JQKA])"

_TEXAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?:{_CARD_TOKEN}\s*){{2,}}", re.IGNORECASE),
    re.compile(r"(?:我的|你的|对手的?|AI的?|他的|她的)?\s*底牌", re.IGNORECASE),
    re.compile(r"hole\s*cards?", re.IGNORECASE),
)

_MAHJONG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:[mpsz]\d+\s*){2,}", re.IGNORECASE),
    re.compile(r"(?:我的|你的|对手的?|AI的?)?\s*手牌"),
)

_WEREWOLF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:我的|你的|他的|她的)?\s*身份\s*(?:是|：)"),
    re.compile(r"我是(?:狼人|预言家|女巫|猎人|守卫|村民)"),
)

_PATTERNS_BY_GAME: dict[str, tuple[re.Pattern[str], ...]] = {
    "texas_holdem": _TEXAS_PATTERNS,
    "mahjong": _MAHJONG_PATTERNS,
    "mahjong_guangdong": _MAHJONG_PATTERNS,
    "mahjong_hongzhong": _MAHJONG_PATTERNS,
    "mahjong_blood": _MAHJONG_PATTERNS,
    "mahjong_sichuan": _MAHJONG_PATTERNS,
    "mahjong_changsha": _MAHJONG_PATTERNS,
    "mahjong_taiwan": _MAHJONG_PATTERNS,
    "werewolf": _WEREWOLF_PATTERNS,
}

#: 按中文句末标点 / 换行切句（保留标点）。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")

#: 「对手/AI 持牌」的中文所有格前缀（教学扫描用：AI 说话人自称"我"，
#: "你" 指玩家 —— 只有这些前缀指向的隐藏信息才是教练不可知的）。
_OPPONENT_POSSESSIVE = r"(?:我的|AI\s*的?|对手的?|庄家的?|上家的?|下家的?|对家的?|他的|她的)"

#: 教学对局的泄露模式：玩家自己的牌可以讲（玩家本来就看得到自己
#: 的牌），只有 **AI/对手的** 隐藏信息仍然拦截。这是教学模式下对
#: ``scan`` 红线的定向放宽 —— 不改动默认（非教学）行为。
_TEACHING_TEXAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_OPPONENT_POSSESSIVE}\s*底牌", re.IGNORECASE),
    re.compile(r"hole\s*cards?", re.IGNORECASE),
)
_TEACHING_MAHJONG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_OPPONENT_POSSESSIVE}\s*手牌", re.IGNORECASE),
    re.compile(rf"{_OPPONENT_POSSESSIVE}\s*听(?:牌|了)"),
)
_TEACHING_WEREWOLF_PATTERNS: tuple[re.Pattern[str], ...] = (
    # AI 教练自报身份 = 泄露（教练不该知道任何角色分配）。
    re.compile(r"我是(?:狼人|预言家|女巫|猎人|守卫|村民)"),
    re.compile(rf"{_OPPONENT_POSSESSIVE}\s*身份\s*(?:是|：)"),
)

#: 教学模式按游戏分派的泄露模式（只拦对手/AI 的隐藏信息）。
_TEACHING_PATTERNS_BY_GAME: dict[str, tuple[re.Pattern[str], ...]] = {
    "texas_holdem": _TEACHING_TEXAS_PATTERNS,
    "mahjong": _TEACHING_MAHJONG_PATTERNS,
    "mahjong_guangdong": _TEACHING_MAHJONG_PATTERNS,
    "mahjong_hongzhong": _TEACHING_MAHJONG_PATTERNS,
    "mahjong_blood": _TEACHING_MAHJONG_PATTERNS,
    "mahjong_sichuan": _TEACHING_MAHJONG_PATTERNS,
    "mahjong_changsha": _TEACHING_MAHJONG_PATTERNS,
    "mahjong_taiwan": _TEACHING_MAHJONG_PATTERNS,
    "werewolf": _TEACHING_WEREWOLF_PATTERNS,
}


def infer_game_id(observation: dict[str, Any]) -> str:
    """从投影观测的视图名推断 ``game_id``（供后置扫描按游戏分派规则）."""
    if "sb_hole_view" in observation or "community_view" in observation:
        return "texas_holdem"
    if any(key.startswith("hand_view_") for key in observation):
        return "mahjong"
    if "my_role" in observation or "dead_roles" in observation:
        return "werewolf"
    return "unknown"


def _walk_dict(node: Any, found: set[str]) -> None:
    """递归收集 ``node`` 中命中的黑名单键名."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key in HIDDEN_FIELDS:
                found.add(key)
            _walk_dict(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_dict(item, found)


def assert_no_hidden(ctx: "SkillContext") -> None:
    """校验 ``ctx.observation`` 不含隐藏字段，违规抛 ``ValueError``.

    Args:
        ctx: 技能上下文，其 ``observation`` 必须是投影观测。

    Raises:
        ValueError: 观测中出现了黑名单隐藏字段名（含字段名信息）。
    """
    found: set[str] = set()
    _walk_dict(ctx.observation, found)
    if found:
        raise ValueError(f"observation 泄露隐藏字段: {', '.join(sorted(found))}")


def scan(text: str, game_id: str, *, teaching: bool = False) -> str:
    """对生成文本做后置泄露令牌扫描，命中句改写为通用语.

    Args:
        text: 待清洗的生成文本。
        game_id: 游戏 id，决定启用的模式规则（至少覆盖德州底牌记法）。
        teaching: 教学对局模式 —— 放宽为"教学模式泄露模式"：玩家自己
            的牌（"你的底牌/手牌"）允许讨论（玩家本来就看得到），只有
            **AI/对手的** 隐藏信息（"我的/AI 的底牌"）仍然拦截。

    Returns:
        改写后的文本；未命中或 ``game_id`` 无规则时原样返回。
    """
    patterns = _TEACHING_PATTERNS_BY_GAME.get(game_id) if teaching else _PATTERNS_BY_GAME.get(game_id)
    if not patterns or not text:
        return text
    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part]
    rewritten: list[str] = []
    for sentence in sentences:
        if any(pattern.search(sentence) for pattern in patterns):
            rewritten.append(_GENERIC_REWRITE)
        else:
            rewritten.append(sentence)
    return "".join(rewritten)
