"""game_knowledge — 游戏权威资料的单一事实来源拼装（Layer 4 平台）.

chat 信息工具（``describe_game``）、无 LLM 兜底回答与陪伴对话引擎
（``agent/dialogue_engine.py`` 的知识注入）共用同一份资料拼装，避免
各自漂移。数据源全部 fail-soft：

- ``GAMES`` 注册表（``GameSpec.description`` / ``display_name`` /
  ``player_counts``）—— 一句话简介与元数据；
- ``docs/user/play_*.md`` 的规则段（``DOCS_RULES_SECTIONS`` 映射 +
  正则提取 + 缓存）—— 玩法要点，docs 更新即同步。

另外维护 ``GAME_ALIASES``（每款游戏的短名/别名表）：``_find_game`` 的
display_name/game_id 子串匹配接不住「UNO 的规则」（display_name 是
「UNO（经典）」）这类短名，别名 + 大小写不敏感匹配补上这个缺口。

依赖方向：本模块只依赖 ``games.py`` 与标准库——``agent`` 与
``platform`` 两侧都可安全导入，不构成循环。
"""

from __future__ import annotations

import re
from pathlib import Path

from .games import GAMES

#: game_id → (docs/user 文档, 要提取的 ``##`` 段标题)。规则段以文档为
#: 单一事实来源（docs 更新即同步）；映射外的游戏（custom 等）只拼
#: GameSpec.description。读取/提取失败一律回退空串（fail-soft）。
DOCS_RULES_SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "moon_chess": ("play_moon_chess.md", ("游戏规则",)),
    "stochastic_gomoku": ("play_gomoku.md", ("游戏规则",)),
    "texas_holdem": ("play_texas_holdem.md", ("游戏规则",)),
    "uno": ("play_uno.md", ("基础规则", "六种变体")),
    "uno_seven_zero": ("play_uno.md", ("基础规则", "六种变体")),
    "uno_jump_in": ("play_uno.md", ("基础规则", "六种变体")),
    "uno_stacking": ("play_uno.md", ("基础规则", "六种变体")),
    "uno_draw_until": ("play_uno.md", ("基础规则", "六种变体")),
    "uno_strict_wild4": ("play_uno.md", ("基础规则", "六种变体")),
    "mahjong_guangdong": ("play_mahjong.md", ("六种变体",)),
    "mahjong_hongzhong": ("play_mahjong.md", ("六种变体",)),
    "mahjong_blood": ("play_mahjong.md", ("六种变体",)),
    "mahjong_sichuan": ("play_mahjong.md", ("六种变体",)),
    "mahjong_changsha": ("play_mahjong.md", ("六种变体",)),
    "mahjong_taiwan": ("play_mahjong.md", ("六种变体",)),
    "undercover": ("play_undercover.md", ("规则",)),
    "werewolf": ("play_werewolf.md", ("规则",)),
}

_RULES_TEXT_MAX = 900
_RULES_TEXT_CACHE: dict[str, str] = {}

#: game_id → 短名/别名（供「X 的规则/玩X」的子串匹配）。刻意只收
#: 无歧义的用户口语短名；同一句里多个游戏命中时由最长匹配胜出
#: （如「UNO 7-0」同时命中 uno 的 "UNO" 与 seven_zero 的 "UNO 7-0"，
#: 取后者）。注意别加过短/过泛的别名（如裸 "70"）防误报。
GAME_ALIASES: dict[str, tuple[str, ...]] = {
    "moon_chess": ("月亮",),
    "stochastic_gomoku": ("五子棋", "gomoku"),
    "texas_holdem": ("德州", "德扑", "扑克", "texas", "holdem"),
    "mahjong_guangdong": ("广东麻将", "鸡胡", "广麻"),
    "mahjong_hongzhong": ("红中",),
    "mahjong_blood": ("血战",),
    "mahjong_sichuan": ("四川麻将", "川麻"),
    "mahjong_changsha": ("长沙麻将", "长麻"),
    "mahjong_taiwan": ("台湾麻将", "台麻", "16张"),
    "uno": ("UNO", "优诺"),
    "uno_seven_zero": ("UNO 7-0", "UNO7-0", "UNO 70", "换手"),
    "uno_jump_in": ("UNO 抢牌", "UNO抢牌", "抢牌", "jump in"),
    "uno_stacking": ("UNO 叠加", "UNO叠加", "叠加"),
    "uno_draw_until": ("UNO 摸到能打", "UNO摸到能打", "摸到能打"),
    "uno_strict_wild4": ("UNO 严格+4", "UNO严格+4", "严格"),
    "undercover": ("谁是卧底", "卧底"),
    "werewolf": ("狼人杀", "狼人", "werewolf"),
}


def game_rules_text(game_id: str) -> str:
    """Extract the rules section from ``docs/user/play_*.md`` (fail-soft).

    The docs are the single source of truth for how to play; missing
    mapping / unreadable file / absent section all return ``""`` so the
    caller falls back to ``GameSpec.description`` alone.
    """
    mapped = DOCS_RULES_SECTIONS.get(game_id)
    if mapped is None:
        return ""
    doc, titles = mapped
    if doc in _RULES_TEXT_CACHE:
        return _RULES_TEXT_CACHE[doc]
    try:
        path = Path(__file__).resolve().parents[3] / "docs" / "user" / doc
        md = path.read_text(encoding="utf-8")
    except OSError:
        _RULES_TEXT_CACHE[doc] = ""
        return ""
    chunks: list[str] = []
    for title in titles:
        m = re.search(rf"^## {re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)", md, re.S | re.M)
        if m:
            chunk = m.group(1).strip()
            if chunk:
                chunks.append(chunk)
    text = "\n\n".join(chunks)[:_RULES_TEXT_MAX]
    _RULES_TEXT_CACHE[doc] = text
    return text


def game_knowledge_text(game_id: str) -> str:
    """Assemble the authoritative knowledge text for one builtin game.

    ``describe_game`` 信息工具、无 LLM 兜底与陪伴对话注入共用本拼装：
    名字（id）+ 一句话简介 + 支持人数 + docs 规则段。未知 game_id
    （custom 游戏等）返回 ``""``，调用方各自 fail-soft。
    """
    spec = GAMES.get(game_id)
    if spec is None:
        return ""
    parts = [f"{spec.display_name}（{game_id}）"]
    if spec.description:
        parts.append(spec.description)
    counts = "、".join(str(c) for c in spec.player_counts)
    parts.append(f"支持人数: {counts} 人；难度: 简单/正常/困难")
    rules_md = game_rules_text(game_id)
    if rules_md:
        parts.append("规则要点:\n" + rules_md)
    return "\n".join(parts)
