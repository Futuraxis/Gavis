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
    "mahjong_international": _MAHJONG_PATTERNS,
    "werewolf": _WEREWOLF_PATTERNS,
}

#: 按中文句末标点 / 换行切句（保留标点）。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


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


def scan(text: str, game_id: str) -> str:
    """对生成文本做后置泄露令牌扫描，命中句改写为通用语.

    Args:
        text: 待清洗的生成文本。
        game_id: 游戏 id，决定启用的模式规则（至少覆盖德州底牌记法）。

    Returns:
        改写后的文本；未命中或 ``game_id`` 无规则时原样返回。
    """
    patterns = _PATTERNS_BY_GAME.get(game_id)
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
