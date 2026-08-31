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

:func:`scan` 四态互斥（按陪伴身份选其一）：

- **default**（全拦）：任何牌面表述都改写。啦啦队模式的既有行为。
- **teaching**（``teaching=True``）：教学模式泄露模式——玩家自己的牌
  可以讨论（教练看的正是玩家自己的投影，见 ``coach.py``），只有
  **AI/对手的** 隐藏信息仍然拦截。
- **adversarial**（``adversarial=True``）：对手模式泄露模式，与 teaching
  **镜像**——AI 对手讲**自己的牌力**允许（模糊措辞如「我这手还行」「一对
  K」，它本就看得到自己的牌），只有**玩家的** 隐藏信息仍然拦截；且**具体
  花色点数**（黑桃4 / ♠A）一律拦——报牌等于明牌，破坏二人博弈。二人非
  教练对手模式（见 ``opponent.py``）。
- **revealed**（``revealed=True``）：终局 showdown 揭底后双方牌公开，全
  放行——可做完整复盘式对手点评。优先级最高。
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
        # 麻将 / UNO —— 各家手牌、未摸牌墙、胡牌手牌（UNO 六变体最多 10 人，
        # 2-4 人之外还有 hand_p4…hand_p9 的隐藏数组，一并列入）
        "hand_p0",
        "hand_p1",
        "hand_p2",
        "hand_p3",
        "hand_p4",
        "hand_p5",
        "hand_p6",
        "hand_p7",
        "hand_p8",
        "hand_p9",
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

#: 单张牌面：花色（中文「黑桃/红桃/方块/梅花」、符号 ♠♥♦♣、或英文字母
#: S/H/D/C）+ 点数（10 或 2-9/J/Q/K/A）。覆盖 LLM 常见写法「黑桃4」「♠10」
#: 「sA」；不含「一对K」「同花」这类无花色牌力描述（不误伤）。
_CARD_TOKEN = r"(?:(?:黑桃|红桃|方块|梅花)|[♠♥♦♣]|[SHDC])(?:10|[2-9JQKA])"

#: 无花色前缀的**具体单牌持牌表述**（「我手里有张K」「拿着一张3」「握着
#: 5」）——只拦“持有一张具体点数”的报牌措辞，不误伤「一对K」「三条」
#: 「同花」这类牌力描述（它们不带 张/拿着/握着 等单牌持牌框架）。
_RANK_ONLY_HOLD = (
    r"(?:手里有张|手上有张|手里有|手上有|有一张|有张|拿着张|拿着一张|"
    r"攥着|握着|捏着|正好是张?)\s*(?:10|[2-9JQKA])"
)

_TEXAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"(?:{_CARD_TOKEN}\s*){{2,}}", re.IGNORECASE),
    re.compile(rf"{_RANK_ONLY_HOLD}", re.IGNORECASE),
    re.compile(r"(?:我的|你的|对手的?|AI的?|他的|她的)?\s*底牌", re.IGNORECASE),
    re.compile(r"hole\s*cards?", re.IGNORECASE),
)

#: UNO —— 六变体 2-10 人。具体牌面：数字牌「红5/蓝0」（色+数）、功能牌
#: 「绿禁止/红反转/黄+2」、万能「万能/万能四」。顶牌色（「红色」）等公开
#: 信息不含这些具体牌名记法，不误伤；「手牌」持牌措辞沿用麻将 2+ 张牌力
#: 表述规则。
_UNO_COLOR = r"(?:红|蓝|绿|黄)"
_UNO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_UNO_COLOR}\s*(?:10|[0-9])", re.IGNORECASE),
    re.compile(rf"{_UNO_COLOR}\s*(?:禁止|反转|\+2)", re.IGNORECASE),
    # 无颜色前缀的动作牌持牌表述（「我手里有张+2」「打出+2」）——+2 无歧义
    # （只可能是罚牌名），持牌动词框架（有张/有一张/拿到/摸到/打出…）里报
    # 具体动作牌同样是明牌；「禁止」「反转」「万能」单独出现可能是普通用语
    # （「禁止这样做」），不在此列，避免误伤。
    re.compile(r"(?:有张|有一张|拿到|摸到|打出|甩出|扔出)\s*\+2", re.IGNORECASE),
    re.compile(r"(?:万能(?:四)?牌?|万能四)", re.IGNORECASE),
    re.compile(r"(?:我的|你的|对手的?|AI的?|他的|她的)?\s*手牌", re.IGNORECASE),
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
    "uno": _UNO_PATTERNS,
    "uno_seven_zero": _UNO_PATTERNS,
    "uno_jump_in": _UNO_PATTERNS,
    "uno_stacking": _UNO_PATTERNS,
    "uno_draw_until": _UNO_PATTERNS,
    "uno_strict_wild4": _UNO_PATTERNS,
    "werewolf": _WEREWOLF_PATTERNS,
}

#: 按中文句末标点 / 换行切句（保留标点）。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")

#: 「对手/AI 持牌」的中文所有格前缀（教学扫描用：AI 说话人自称"我"，
#: "你" 指玩家 —— 只有这些前缀指向的隐藏信息才是教练不可知的）。
_OPPONENT_POSSESSIVE = r"(?:我的|AI\s*的?|对手的?|庄家的?|上家的?|下家的?|对家的?|他的|她的)"

#: 「玩家持牌」的中文所有格前缀（对手扫描用：AI 说话人自称"我"，
#: "你" 指玩家 —— 这些前缀指向的隐藏信息才是 AI 对手不可知的，即玩家
#: 的底牌/手牌）。与 :data:`_OPPONENT_POSSESSIVE` 对称定义：教学模式
#: 拦「对手/AI 的牌」、放行「玩家的牌」；对手模式镜像——拦「玩家的牌」、
#: 放行「我的/AI 的牌」。
_PLAYER_POSSESSIVE = r"(?:你的|玩家的?)"

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
#: 教学对局的 UNO 泄露模式：拦「我的/AI/对手的 + 手牌」（教练不可知对手牌），
#: 玩家自己的手牌「你的手牌」放行（教练看的正是玩家投影）。
_TEACHING_UNO_PATTERNS: tuple[re.Pattern[str], ...] = (re.compile(rf"{_OPPONENT_POSSESSIVE}\s*手牌", re.IGNORECASE),)

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
    "uno": _TEACHING_UNO_PATTERNS,
    "uno_seven_zero": _TEACHING_UNO_PATTERNS,
    "uno_jump_in": _TEACHING_UNO_PATTERNS,
    "uno_stacking": _TEACHING_UNO_PATTERNS,
    "uno_draw_until": _TEACHING_UNO_PATTERNS,
    "uno_strict_wild4": _TEACHING_UNO_PATTERNS,
    "werewolf": _TEACHING_WEREWOLF_PATTERNS,
}


# ── 对手模式（adversarial）：与 teaching 镜像 ──────────────────────
#
# 二人非教练对局下，陪伴是「座内对手」：AI 说话人自称"我"（= AI 自己），
# "你" 指玩家。AI 对手讲**自己的牌力**是其本分（它本就看得到自己的牌），
# 因此放行「我的/AI 的 + 牌力措辞」（如「我手里一对K」「这手同花不算大」）；
# 只有**玩家的**隐藏信息（"你的/玩家的 + 底牌/手牌"）仍然拦截——AI 本来
# 就没有玩家底牌（visibility 规则不给），scan 是双保险防 LLM 幻觉编造玩家
# 未公开牌面。但**具体花色点数**（黑桃4 / ♠A / s10）一律拦——banter 人设
# 明写「不报牌」，报出自己底牌的具体牌面等于明牌，破坏二人博弈；终局
# showdown 揭底后双方牌公开，由 ``revealed`` 全放行。无花色前缀的牌力描述
# （「一对K」「同花」「高牌」）不命中牌面正则，照旧放行。

#: 对手模式的德州泄露模式：拦「你的/玩家的 + 底牌」（玩家底牌本就不可见），
#: 拦英语 ``hole cards``；放行「我的底牌」这类**模糊**措辞（对手可自称牌力
#: 「我这手还行」「我手里一对K」——无花色点数，属人设「读牌/虚张」本分）；
#: 但**具体牌面**（黑桃4 / ♠A / s10）一律拦——banter 人设明写「不报牌」，
#: 报出具体花色点数等于明牌，破坏二人博弈（终局 showdown 揭底后由
#: ``revealed`` 全放行，可做完整复盘）。无花色的牌力描述（「一对K」「同花」
#: 「高牌」）不带花色前缀，不命中 ``_CARD_TOKEN``，照旧放行。
_ADVERSARIAL_TEXAS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_PLAYER_POSSESSIVE}\s*底牌", re.IGNORECASE),
    re.compile(r"hole\s*cards?", re.IGNORECASE),
    re.compile(rf"{_CARD_TOKEN}", re.IGNORECASE),
    # 无花色前缀的**具体单牌**表述（「我手里有张K」「拿着一张3」）也拦——
    # 报具体点数同样是明牌；「一对K」「同花」等牌力措辞不受影响。
    re.compile(rf"{_RANK_ONLY_HOLD}", re.IGNORECASE),
)
#: 对手模式的 UNO 泄露模式：拦「你的/玩家的 + 手牌/牌面」；拦具体牌面记法
#: （红5 / 蓝禁止 / +2 / 万能）；「我的/AI 的 + 手牌」这类模糊持牌措辞放行。
_ADVERSARIAL_UNO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_PLAYER_POSSESSIVE}\s*手牌", re.IGNORECASE),
    re.compile(rf"{_CARD_TOKEN}", re.IGNORECASE),
    re.compile(rf"{_RANK_ONLY_HOLD}", re.IGNORECASE),
    re.compile(rf"{_UNO_COLOR}\s*(?:10|[0-9])", re.IGNORECASE),
    re.compile(rf"{_UNO_COLOR}\s*(?:禁止|反转|\+2)", re.IGNORECASE),
    # 无颜色前缀的 +2 持牌表述（AI 报「我手里有张+2」同样是明牌）。
    re.compile(r"(?:有张|有一张|拿到|摸到|打出|甩出|扔出)\s*\+2", re.IGNORECASE),
    re.compile(r"(?:万能(?:四)?牌?|万能四)", re.IGNORECASE),
)
#: 对手模式的麻将泄露模式：拦「你的/玩家的 + 手牌/听牌」，放行「我的/
#: AI 的手牌/听牌」。
_ADVERSARIAL_MAHJONG_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"{_PLAYER_POSSESSIVE}\s*手牌", re.IGNORECASE),
    re.compile(rf"{_PLAYER_POSSESSIVE}\s*听(?:牌|了)"),
)
#: 对手模式的狼人杀泄露模式：拦「你是<角色>」「你的/玩家的 + 身份」
#: （AI 不知玩家身份，不得声称玩家角色）。AI 自报身份在对手模式下是
#: 允许的（它就是那个角色，本就该能讲自己）——与 teaching 镜像（teaching
#: 拦 AI 自报「我是<角色>」、放行玩家身份；adversarial 放行 AI 自报、
#: 拦玩家身份「你是<角色>」）。
_ADVERSARIAL_WEREWOLF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"你是(?:狼人|预言家|女巫|猎人|守卫|村民|平民)"),
    re.compile(rf"{_PLAYER_POSSESSIVE}\s*身份\s*(?:是|：)"),
)

#: 对手模式按游戏分派的泄露模式（只拦玩家的隐藏信息；与 teaching 镜像）。
_ADVERSARIAL_PATTERNS_BY_GAME: dict[str, tuple[re.Pattern[str], ...]] = {
    "texas_holdem": _ADVERSARIAL_TEXAS_PATTERNS,
    "mahjong": _ADVERSARIAL_MAHJONG_PATTERNS,
    "mahjong_guangdong": _ADVERSARIAL_MAHJONG_PATTERNS,
    "mahjong_hongzhong": _ADVERSARIAL_MAHJONG_PATTERNS,
    "mahjong_blood": _ADVERSARIAL_MAHJONG_PATTERNS,
    "mahjong_sichuan": _ADVERSARIAL_MAHJONG_PATTERNS,
    "mahjong_changsha": _ADVERSARIAL_MAHJONG_PATTERNS,
    "mahjong_taiwan": _ADVERSARIAL_MAHJONG_PATTERNS,
    "uno": _ADVERSARIAL_UNO_PATTERNS,
    "uno_seven_zero": _ADVERSARIAL_UNO_PATTERNS,
    "uno_jump_in": _ADVERSARIAL_UNO_PATTERNS,
    "uno_stacking": _ADVERSARIAL_UNO_PATTERNS,
    "uno_draw_until": _ADVERSARIAL_UNO_PATTERNS,
    "uno_strict_wild4": _ADVERSARIAL_UNO_PATTERNS,
    "werewolf": _ADVERSARIAL_WEREWOLF_PATTERNS,
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


def resolve_scan_game(game_id: str, observation: dict[str, Any]) -> str:
    """解析后置扫描的 ``game_id``：显式内置 id 优先，未知回退观测推断.

    观测视图名有歧义（UNO 与麻将都用 ``hand_view_*``，``infer_game_id``
    会把 UNO 误判成 ``mahjong``）且可能缺失（早期投影/自定义视图名），
    单靠推断会让扫描静默跳过——对手模式里 AI 报「黑桃K」就漏网。调用方
    手里通常有注册表 ``game_id``（``session.game_id``），当它在扫描规则
    表内时直接采用；否则（custom / 未知）退回 :func:`infer_game_id` 按
    观测形态兜底。
    """
    if game_id and (
        game_id in _PATTERNS_BY_GAME
        or game_id in _TEACHING_PATTERNS_BY_GAME
        or game_id in _ADVERSARIAL_PATTERNS_BY_GAME
    ):
        return game_id
    return infer_game_id(observation)


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


def scan(
    text: str,
    game_id: str,
    *,
    teaching: bool = False,
    adversarial: bool = False,
    revealed: bool = False,
) -> str:
    """对生成文本做后置泄露令牌扫描，命中句改写为通用语.

    四态互斥（调用方按陪伴身份选其一；``revealed`` 优先级最高）：

    - **default**（全拦）：任何牌面记法 / 持牌表述都改写。啦啦队模式的
      既有行为，不动。
    - **teaching**（定向放行玩家牌）：玩家自己的牌（"你的底牌/手牌"）
      允许讨论（玩家本来就看得到），只有 **AI/对手的** 隐藏信息
      （"我的/AI 的底牌"）仍然拦截。教练模式。
    - **adversarial**（定向放行 AI 自己的牌力、拦玩家牌与具体牌面，与
      teaching 镜像）：AI 对手讲**自己的牌力**（模糊措辞如「我这手还行」
      「一对K」）允许（它本就看得到自己的牌），只有**玩家的** 隐藏信息
      （"你的/玩家的 + 底牌/手牌"）仍然拦截；且**具体花色点数**（黑桃4 /
      ♠A）一律拦——报牌等于明牌，破坏二人博弈。二人非教练对手模式。
    - **revealed**（全放行）：终局 showdown 揭底后双方牌公开，文本原样
      返回——可做完整复盘式对手点评。

    Args:
        text: 待清洗的生成文本。
        game_id: 游戏 id，决定启用的模式规则（至少覆盖德州底牌记法）。
        teaching: 教学对局模式（见上）。
        adversarial: 对手模式（见上；与 ``teaching`` 互斥，同时为真时
            ``adversarial`` 生效——它是二人非教练的更具体身份）。
        revealed: 揭底模式（终局 showdown 后；优先级最高，为真即全放行）。

    Returns:
        改写后的文本；未命中或 ``game_id`` 无规则时原样返回。
    """
    if not text:
        return text
    if revealed:
        # 揭底后双方牌公开，全放行（不拦截任何牌面表述）。
        return text
    if adversarial:
        patterns = _ADVERSARIAL_PATTERNS_BY_GAME.get(game_id)
    elif teaching:
        patterns = _TEACHING_PATTERNS_BY_GAME.get(game_id)
    else:
        patterns = _PATTERNS_BY_GAME.get(game_id)
    if not patterns:
        return text
    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part]
    rewritten: list[str] = []
    for sentence in sentences:
        if any(pattern.search(sentence) for pattern in patterns):
            rewritten.append(_GENERIC_REWRITE)
        else:
            rewritten.append(sentence)
    return "".join(rewritten)
