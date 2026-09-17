"""Agent-style chat orchestrator — the *chat-first* backend of the platform.

One sentence from the user becomes one platform action.  ``chat_turn``
runs LLM function calling (unified ``LLMClient``, OpenAI-compatible
``tools`` schema) against a tool set built from the live registry and the
current session, validates the tool arguments against the authoritative
engine contract, and always fails soft to a deterministic regex fallback
when the LLM is missing.

Tool classes (function-calling audit 2026-09 + 对局/复盘信息源修复 + 卡片下线):

- **action tools** (play_game / make_move / …) map straight to a
  frontend intent — validated, fail-soft, one shot.  ``play_game`` 带
  **玩家偏好参数**（人数/难度/主题/性格/提示档/节奏/自适应/教学/座位）：
  模型从用户那句话里取，服务端按 ``GameSpec`` 校验后塞进 ``params.config``
  （前端不再有开局配置卡）。
- **local tools** are executed locally in a bounded loop and their result
  is fed back to the model as ``role: "tool"`` messages:

  - ``describe_game`` / ``list_games`` — registry + play docs, so
    knowledge questions (“月亮棋是什么？”) are answered from
    authoritative data instead of hallucinated; ``describe_game`` 同时给出
    **可配置面**（人数/难度/座位/主题），模型据此填 ``play_game`` 参数
    而不是猜；
  - ``create_game`` — **有副作用的本地工具**：把用户口述的规则/模板变体
    交给 L1 翻译器（``CustomGameRegistry.create``），成功即以 ``create``
    意图带 ``params.game`` 回前端（刷新目录 + 提示「玩X」），失败文本化
    上报（fail-soft）。同一回合只执行一次（防一次请求建出两个游戏）。
  - ``get_match_state`` — the *player-projected* live snapshot (board
    layout / own hand / pot / discards …): the in-match information
    source, the model pulls it instead of asking the user to describe
    the board;
  - ``ask_hint`` — the mechanical hint (direction / specific / demo),
    carried to the frontend as the ``hint`` intent with the hint dict
    in params; the model sees the hint and can explain it;
  - ``get_match_review`` — the latest (or given) match's timeline + key
    nodes + improvement, narrated by the model as the ``review``
    intent (the report travels in params — no more canned-only 复盘);
  - ``get_platform_help`` — per-feature platform help docs
    (``platform_knowledge`` 单一事实来源)：用户问**具体功能怎么用**时
    （“怎么创建游戏/在线学习怎么用/评测中心在哪/教学对局是什么/LLM
    配置/视觉识别…”）先取该主题的权威说明再回答——旧 ``help`` 工具只
    回一段泛泛总览，面对具体功能提问只能泛泛而谈或编造。

  The deterministic fallback answers the same classes from the same
  data when no LLM is available.

Intent contract (shared with the frontend ``ChatPage`` / ``useChatRuntime``):

=========  ========================================================
intent     params
=========  ========================================================
play       ``{game_id, config?, config_ignored?}``
                                   → 前端开新对局（``config`` = 已校验的
                                     玩家偏好；``config_ignored`` = 该游戏
                                     不支持、被丢弃的偏好项）
resume     ``{game_id}``            → 前端恢复活跃会话
move       ``{action}``             → 前端调 ``/match/move``
hint       ``{level, hint?}``       → 前端展示提示（``hint`` = 后端已算的机械提示 dict）
restart    ``{}``                   → 前端重开当前对局
history    ``{}``                   → 前端展示战绩
review     ``{match_id?, report?}`` → 前端展示复盘（``report`` = 后端已算的 ReviewReport）
create     ``{game_id?, game?, family?, diff_summary?, validation?}``
                                   → 创建结果通知（``game`` = 注册表条目）；
                                     **无 ``game``** 时前端打开创建游戏页
                                     （无 LLM 兜底路径：不臆造规则）
settings   ``{applied?, open_page?, chips?}``
                                   → 偏好已按白名单校验（``applied`` = profile
                                     字段 → 取值，前端写 ``/api/profile`` 并回执，
                                     **不跳页**）；``open_page=true`` = 用户明确
                                     要求打开设置页（此时才切页面）；
                                     什么都没说清时走 ``clarify`` + 风格 chips
platform   ``{}``                   → 前端切回完整平台界面
benchmark  ``{}``                   → 前端展示评测中心
learning   ``{}``                   → 前端展示在线学习
help       ``{}``                   → 前端展示帮助
chat       ``{game_id?, chips?}``   → 普通聊天回复（知识回答可带 chips）
clarify    ``{chips?: [str]}``      → 追问（附可点选项）
=========  ========================================================

Fail-soft rule: this module never raises for a misbehaving model — a
missing/empty LLM reply, an unknown tool name, or an invalid action all
land on a clarifying or canned reply.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from layer2_engine.core.llm import LLMClient

from ...agent.coach import extract_hand
from ...agent.hidden_guard import resolve_scan_game, scan
from ...agent.persona import PERSONAS, Persona, persona_identity_block
from ...result import player_won
from ...review import ReviewReport
from ...review import analyze as review_analyze
from ..engine_helpers import (
    build_seat_names,
    canonical_family_text,
    game_family,
    mahjong_tile_name,
    piece_names,
    seat_label,
    social_role_name,
    uno_card_name,
)
from .custom_games import CustomGameError, CustomGameRegistry
from .game_knowledge import (
    DIFFICULTY_LABELS,
    GAME_ALIASES,
    THEME_LABELS,
    game_knowledge_text,
    game_rules_text,
)
from .games import GAMES
from .platform_knowledge import (
    PLATFORM_TOPIC_KEYS,
    match_platform_topic,
    platform_help_index,
    platform_help_text,
)
from .session import _BUILTIN_FAMILY, PlayManager

logger = logging.getLogger(__name__)

GRID_BOARD_LEN = {
    "moon_chess": 3,
    # P2-18 修复：随机五子棋是 9×9（rules/stochastic_gomoku.json
    # board_size:9）。旧值 15 是错的 —— 且该字典只作 spec.board_size
    # 缺失时的兜底（见 _grid_cell_from_text，spec 是单一事实来源）。
    "stochastic_gomoku": 9,
}

#: 每轮随请求带来的对话历史上限（条数 / 总字符），防 prompt 无限膨胀。
_HISTORY_MAX_MESSAGES = 24
_HISTORY_MAX_CHARS = 6000

#: 一次用户消息最多几次 LLM 往返（信息工具可多轮取数后作答）。
#: 有界 agentic loop：防模型无限调用工具 / prompt 无限膨胀。
_MAX_TOOL_ROUNDS = 3

#: 单回合思维链（reasoning）流式上浮的总字符上限 —— 防 prompt/响应无限膨胀。
_REASONING_MAX_CHARS = 8000

#: 信息类工具 —— 后端就地执行、把结果以 ``role:"tool"`` 回传，模型基于
#: 资料组织最终回答。动作类工具仍走 intent 映射（fail-soft 校验）。
#: 其中 ``ask_hint`` 例外地非纯只读：会标记 ``session.hinted``（语义上
#: 用户确实要了提示——与 ``/match/hint`` 路由是同一调用）。
_INFO_TOOLS = (
    "describe_game",
    "list_games",
    "get_match_state",
    "ask_hint",
    "get_match_review",
    "get_platform_help",
)

#: 就地执行的工具全集 = 只读信息工具 + 有副作用的本地工具。工具循环用
#: 它挑「动作工具」（其余才算映射意图的一次性动作）；``create_game``
#: 因此不再走 ``_intent_from_tool`` 的空参映射，而是真正执行创建。
#: ``update_settings`` 同理：它要按白名单校验取值、并在「用户没说要改成
#: 哪种」时给出选项（空参映射会把这两种情况都压成同一句空话）。
_LOCAL_TOOLS = (*_INFO_TOOLS, "create_game", "update_settings")

#: get_match_state 的载荷预算（字符）。玩家投影快照除头部/棋盘/噪音
#: 字段外逐 key 序列化，超预算截断（fail-soft，宁缺毋滥）。
_STATE_MAX_CHARS = 1600
_STATE_PER_KEY_MAX = 240
#: 头部（over/turn）与棋盘渲染单独表达；其余属于会话元数据/伴生载荷，
#: 不进逐 key 的状态文本。
_STATE_SKIP_KEYS = frozenset(
    {
        "game_id",
        "player_pid",
        "ai_pid",
        "difficulty",
        "over",
        "winner",
        "turn",
        "family",
        "teaching",
        "chat",
        "evaluation",
        "board",
    }
)

#: get_match_review 的时间线上限（条）与总文本预算（字符）。
_REVIEW_TIMELINE_MAX = 60
_REVIEW_TEXT_MAX = 1800

#: 「X 是什么 / 怎么玩」类知识问句 —— 无 LLM 时也能用注册表 + 玩法
#: 文档给出确定性回答（零幻觉路径）。
_WHAT_IS_RE = re.compile(r"(?:是什么|什么叫|什么游戏|怎么玩|怎么下|怎么打|规则|玩法|介绍一?下|简介)")

#: game_id → (docs/user 文档, 要提取的 ``##`` 段标题) 的规则段映射与
#: 提取已抽到 ``game_knowledge``（与陪伴对话注入共用单一事实来源）；
#: 本模块只做消费。每款游戏的短名/别名表（``GAME_ALIASES``）同样在
#: ``game_knowledge`` 维护。

#: 历史消息角色白名单 —— system 由后端每轮现构（含实时对局上下文），
#: 绝不采信客户端传入的 system 角色。
_HISTORY_ROLES = ("user", "assistant")

#: 各规则族 make_move 的 action 形态（写入工具描述，帮助模型产出可校验参数）。
_ACTION_SHAPES = {
    "grid": '{"cell_index": 数字}  —— 0 基空格索引（描述里会给出当前可落子格）',
    "poker": '{"choice": "call"|"fold"|"raise"|"all_in", "amount": 数字 —— amount 仅 raise 时必填}',
    "mahjong": '{"type": "discard"|"chow"|"pong"|"kong"|"win", "tile": "牌 id"}  —— 只能用描述里列出的合法组合',
    "uno": (
        '{"type": "draw"|"pass"|"play"|"play_wild"|"play_drawn"|"play_drawn_wild"|"play7"'
        '|"jump_play"|"jump_pass"|"stack2"|"stack4"|"take_penalty", "card": "牌 id", '
        '"color": "r|b|g|y", "target": "玩家 id"}  —— 万能牌（wild_*/wild4_*）必带 color 选色；'
        "只能用描述里列出的合法动作（card 取其中的 card 值）"
    ),
    "social": '{"template_id": "发言模板 id", "text": "发言内容"}  —— 中文发言，text 必填',
}

_FALLBACK_REPLIES = {
    "play": "好，来一局！对局正在创建…",
    "resume": "继续上一局！",
    "move": "好，走这步！",
    "hint": "这一步的思路是…",
    "restart": "好，重新开一局！",
    "history": "这是你最近的战绩 👇",
    "review": "复盘已为你展开 👇",
    "create": "已为你打开创建游戏页 👇",
    "settings": "设置面板已为你展开 👇",
    "platform": "已为你打开完整平台界面 👇",
    "benchmark": "评测中心已为你展开 👇",
    "learning": "在线学习状态已为你展开 👇",
    "help": "我能帮你：开对局（如“玩月亮棋”）、继续对局、落子或要提示（“这步怎么走”）、看战绩“复盘”、创建游戏、改设置、打开平台界面。",
}

_HELP_TEXT = (
    "你可以直接用大白话跟我说话，例如：\n"
    "· “玩月亮棋” / “来一局德州扑克” —— 开对局\n"
    "· “继续上一局” —— 恢复进行中的对局\n"
    "· 对局中：“下第2行第3列” / “这步怎么走” / “提示我”\n"
    "· “看战绩” / “复盘上一局”\n"
    "· “创建一个新游戏” —— 直接说规则，我帮你生成（也可进「创建游戏」页）\n"
    "· “换个风格” / “换成高冷竞技” —— 换助手性格（立刻生效）\n"
    "· “打开平台界面” —— 切回完整界面\n"
    "· “设置” / “评测中心” / “在线学习” —— 各功能面板"
)

_JOIN_WORDS = ("加入", "来一局", "来一把", "玩", "下", "打", "开")
_PLAY_RE = re.compile(r"(?:玩|来一局|来一把|下|打|开局|对战|加入|开一局)")
_RESUME_RE = re.compile(r"(?:继续|接着|恢复|回到) *(?:上一局|对战|对局|游戏)")
_RESTART_RE = re.compile(
    r"(?:再来一局|重来|重新|重开|换一局|再来)"
)  # 注意: 与 play 有交集, 优先级低于 play 里的"来一局"判断
_HINT_RE = re.compile(r"(?:提示|怎么走|这步为什么|帮我想|下一步)")
_HISTORY_RE = re.compile(r"(?:战绩|历史|记录|胜率|输赢|数据)")
_REVIEW_RE = re.compile(r"(?:复盘|回放|重看|复盘一下)")
_CREATE_RE = re.compile(r"(?:创建|新建|自定义|设计一?个新?游戏|(?:做|写|弄|生成|搞)一?(?:个|款|套).{0,12}游戏)")
_SETTINGS_RE = re.compile(r"(?:设置|性格|声音|主题|偏好|选项)")
_PLATFORM_RE = re.compile(r"(?:平台界面|完整界面|平台模式|打开平台|回去|回平台)")
_BENCHMARK_RE = re.compile(r"(?:评测|benchmark|模拟对局|求解器对比)")
_LEARNING_RE = re.compile(r"(?:在线学习|学习状态|自动学习)")
_HELP_RE = re.compile(r"(?:帮助|能做什么|怎么用|你有什么功能|你会什么)")
_GRID_MOVE_RE = re.compile(r"(?:下|放|走)(?:第)?(\d{1,2})\s*行\s*(?:第)?(\d{1,2})\s*列")
_GRID_CELL_RE = re.compile(r"(?:下|放|走)\s*(?:第)?(\d{1,2})\s*(?:格|格位置|个空位)")
_CENTER_RE = re.compile(r"(?:中间|正中|中心)")

# ── 对话里改偏好（“换个风格” / “改设置”）────────────────────────
# 与前端 ``intents.ts`` 同表同口径。契约：明确说出取值 → ``settings`` +
# ``params.applied``（前端写档案并回执，**不跳页**）；只说要换、没说成哪种
# → ``clarify`` + 选项 chips。绝不替用户猜，也绝不把人从对话里静默甩到
# 设置页。触发词只认「性格/风格/主题」类名词与明确改动词，闲聊不误伤。

#: 用户明确要求打开设置页（只切页面，不改偏好）。
_OPEN_SETTINGS_RE = re.compile(r"(?:打开|进入|去|切到|回到|看看)\s*(?:一下)?\s*(?:设置|偏好)页?")
#: 改动词（“换/改/调/变…”）—— 与取值词同时出现才算改偏好请求。
_PREFERENCE_CHANGE_RE = re.compile(r"(?:换|改|调|变|设置|来)(?:成|一个|个|一下|点)?")
#: 性格/风格类名词。
_PERSONA_WORDS = ("性格", "人设", "风格", "语气", "口气", "人格", "说话方式")
#: 主题/外观类名词。
_THEME_WORDS = ("主题", "外观", "配色")
#: “温柔一点/高冷点”这类直接表态、没提名词的说法。
_PERSONA_ASK_RE = re.compile(r"(?:温柔|贴心|陪玩|吐槽|幽默|搞笑|高冷|严肃|竞技|认真|老师).{0,2}(?:点|一点|一些|些)")
#: 人格取值识别（“认真/教学”先判，避免“认真温柔”这类叠加句落到 gentle）。
_PERSONA_VALUE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("teacher", re.compile(r"(?:认真|教学|老师|讲道理)")),
    ("gentle", re.compile(r"(?:温柔|贴心|陪玩)")),
    ("banter", re.compile(r"(?:吐槽|幽默|搞笑)")),
    ("cold", re.compile(r"(?:高冷|严肃|竞技)")),
)
#: 界面主题取值识别。
_THEME_VALUE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dark", re.compile(r"(?:深色|暗色|夜间|黑夜|黑色主题|黑主题)")),
    ("light", re.compile(r"(?:浅色|亮色|日间|白色主题|白主题)")),
)
#: 没说成哪种时的选项 chips（点一下 = 当作一句话发回来 → 命中上面的取值规则）。
_PERSONA_STYLE_CHIPS = ("换成温柔陪伴", "换成认真教学", "换成轻松吐槽", "换成高冷竞技")
_THEME_CHIPS = ("换成深色主题", "换成浅色主题")
_OPEN_SETTINGS_CHIP = "打开设置页"
_PREFERENCE_CLARIFY_TEXT = "想换成哪种？挑一个我立刻改；也可以直接打开设置页自己调。"

#: update_settings 工具参数名 → profile 字段名。
_SETTING_ARG_FIELDS = {
    "persona": "default_persona",
    "hint_level": "hint_level",
    "difficulty": "default_difficulty",
    "theme": "theme",
}
#: profile 字段 → 合法取值 → 中文显示名（工具参数白名单 + 回执文案共用）。
_SETTING_VALUE_LABELS: dict[str, dict[str, str]] = {
    "default_persona": {"gentle": "温柔陪伴", "teacher": "认真教学", "banter": "轻松吐槽", "cold": "高冷竞技"},
    "hint_level": {"off": "关闭", "direction": "方向提示", "specific": "具体建议", "demo": "演示"},
    "default_difficulty": {"easy": "简单", "normal": "普通", "hard": "困难", "adaptive": "自适应"},
    "theme": {"light": "浅色", "dark": "深色"},
}
#: profile 字段 → 中文名（回执文案）。
_SETTING_FIELD_LABELS = {
    "default_persona": "助手性格",
    "hint_level": "提示档",
    "default_difficulty": "AI 难度",
    "theme": "界面主题",
}

#: 无 LLM 兜底的开局偏好解析（与前端 ``intents.ts::parseBattleOptions`` 同表）。
#: 每条 = ``(play_game 参数名, 取值, 触发正则)``，按序覆盖（同键后命中者胜）。
#: 刻意保守：只在措辞明确时命中，认不出的偏好一律不填——宁可走默认开局，
#: 也不要把「标准节奏」误判成难度档。
_BATTLE_COUNT_RE = re.compile(r"(\d{1,2}|[二两三四五六七八九十]{1,3})\s*个?\s*人")
#: 中文数词 → 阿拉伯数字（“三人局”“四人麻将”与“3 人”等价；与前端同表）。
_CN_NUMERALS: dict[str, int] = {
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}
_BATTLE_OPTION_RULES: tuple[tuple[str, Any, re.Pattern[str]], ...] = (
    ("difficulty", "easy", re.compile(r"(?:简单|容易)")),
    ("difficulty", "hard", re.compile(r"(?:困难|高难)")),
    ("difficulty", "normal", re.compile(r"(?:普通|中等)")),
    ("teaching", True, re.compile(r"(?:教学对局|教学|教练|带我打|带打)")),
    ("persona", "gentle", re.compile(r"(?:温柔|贴心|陪玩)")),
    ("persona", "banter", re.compile(r"(?:吐槽|幽默|搞笑)")),
    ("persona", "cold", re.compile(r"(?:高冷|严肃|竞技)")),
    ("hint_level", "demo", re.compile(r"(?:演示|示范)")),
    ("hint_level", "specific", re.compile(r"(?:具体建议|具体提示|详细建议|详细提示)")),
    ("hint_level", "direction", re.compile(r"(?:方向提示|给我方向|指个方向)")),
    ("hint_level", "off", re.compile(r"(?:关闭提示|不要提示|别提示)")),
    ("pacing", "fast", re.compile(r"(?:快棋|快节奏|快点下|下快点)")),
    ("pacing", "slow", re.compile(r"(?:慢棋|慢节奏|慢慢来)")),
    ("theme", "fruit", re.compile(r"水果")),
    ("theme", "food", re.compile(r"美食")),
    ("theme", "animal", re.compile(r"动物")),
    ("theme", "object", re.compile(r"物品")),
    ("theme", "place", re.compile(r"地点")),
    ("theme", "plant", re.compile(r"植物")),
    ("adaptive", True, re.compile(r"自适应")),
    ("adaptive", False, re.compile(r"(?:不要|不用|别|关闭).{0,4}自适应")),
)

_GOOD_WORDS = ("赢", "好", "棒", "哈", "谢", "厉害")
_BAD_WORDS = ("输", "难过", "唉", "可惜", "气", "烦")
_THINK_WORDS = ("为什么", "怎么", "提示", "不懂", "教")


@dataclass
class ChatTurnResult:
    """One chat turn's outcome — intent + agent reply + execution params."""

    intent: str
    text: str
    mood: str = "neutral"
    params: dict[str, Any] = field(default_factory=dict)


def _mood_for(text: str) -> str:
    """Cheap mood guess from the user's wording (happy/thinking/sorry/neutral)."""
    if any(w in text for w in _GOOD_WORDS):
        return "happy"
    if any(w in text for w in _BAD_WORDS):
        return "sorry"
    if any(w in text for w in _THINK_WORDS):
        return "thinking"
    return "neutral"


def _collect_games(custom: CustomGameRegistry | None) -> list[dict]:
    """Built-in + custom catalog for the ``play_game`` tool and fallback.

    Keeps each game's ``description`` — the one-line authoritative intro
    from ``GameSpec`` / the custom entry — plus its **可配置面**
    （人数档 / 难度档 / 座位 / 主题）：``describe_game`` 与系统提示据此
    让模型填 ``play_game`` 的偏好参数，不必猜合法取值。它 feeds the
    system prompt and the info tools, so the model never has to *guess*
    what a game is (the "月亮棋是什么？" hallucination class).
    """
    games: list[dict] = [
        {
            "game_id": spec.game_id,
            "display_name": spec.display_name,
            "description": spec.description,
            "kind": spec.kind,
            "family": _BUILTIN_FAMILY.get(spec.game_id),
            "player_counts": list(spec.player_counts),
            "difficulties": list(spec.difficulty_budgets),
            "seat_options": list(spec.seat_options),
            "seat_label": spec.seat_label,
            "seat_names": build_seat_names(_BUILTIN_FAMILY.get(spec.game_id) or "unknown", spec.seat_options),
            "variant_themes": list(spec.variant_themes) if spec.variant_themes else None,
        }
        for spec in GAMES.values()
    ]
    if custom is not None:
        for entry in custom.list_games():
            family = entry.get("family")
            games.append(
                {
                    "game_id": str(entry.get("game_id", "")),
                    "display_name": str(entry.get("display_name") or entry.get("game_id", "")),
                    "description": str(entry.get("description") or ""),
                    "kind": "board",
                    "family": family,
                    "custom": True,
                    "player_counts": list(entry.get("player_counts") or ()),
                    "difficulties": list(entry.get("difficulties") or ("easy", "normal", "hard")),
                    "seat_options": list(entry.get("seat_options") or ()),
                    "seat_label": entry.get("seat_label"),
                    "seat_names": build_seat_names(str(family or "unknown"), entry.get("seat_options") or ()),
                    "variant_themes": list(entry.get("variant_themes") or ()) or None,
                }
            )
    # 去重（自定义游戏可能覆盖内置 id）
    seen: set[str] = set()
    unique: list[dict] = []
    for g in games:
        if g["game_id"] and g["game_id"] not in seen:
            seen.add(g["game_id"])
            unique.append(g)
    return unique


def _game_brief(g: dict) -> str:
    """One catalog line for the system prompt: 名字(id)：一句话简介。"""
    desc = str(g.get("description") or "")
    return f"{g['display_name']}({g['game_id']})" + (f"：{desc}" if desc else "")


def _config_surface_text(game: dict) -> str:
    """目录条目 → 「可配置项」一行（人数/难度/座位/主题；缺项自动省略）.

    ``describe_game`` 的收尾行：模型据此知道该游戏**合法**的偏好取值，
    再把它从用户那句话里读到的偏好填进 ``play_game`` —— 不用猜、不用问。
    取值一律「中文名(机器 id)」双轨，与合法动作的表述口径一致。
    """
    parts: list[str] = []
    counts = game.get("player_counts") or []
    if counts:
        parts.append("人数: " + "、".join(f"{c}人" for c in counts))
    tiers = game.get("difficulties") or []
    if tiers:
        parts.append("难度: " + "、".join(f"{DIFFICULTY_LABELS.get(str(d), str(d))}({d})" for d in tiers))
    seats = game.get("seat_options") or []
    if seats:
        names = game.get("seat_names") or {}
        label = str(game.get("seat_label") or "座位")
        parts.append(f"{label}: " + "、".join(f"{names.get(str(s), str(s))}({s})" for s in seats))
    themes = game.get("variant_themes") or []
    if themes:
        parts.append("主题: " + "、".join(f"{THEME_LABELS.get(str(t), str(t))}({t})" for t in themes))
    if not parts:
        return ""
    return "可配置项 —— " + "；".join(parts)


#: ``play_game`` 偏好的「工具参数名 → 前端 BattleConfig 字段」映射
#: （见 platform-frontend/src/chat/battleConfig.ts 的白名单——两侧同表）。
_PLAY_CONFIG_FIELDS: dict[str, str] = {
    "player_count": "playerCount",
    "difficulty": "difficulty",
    "theme": "theme",
    "player_pid": "playerPid",
    "persona": "persona",
    "hint_level": "hintLevel",
    "pacing": "pacing",
    "adaptive": "adaptive",
    "teaching": "teaching",
}

#: 平台级枚举（与游戏无关的偏好档位；未命中的取值一律丢弃并如实上报）。
_PERSONA_VALUES = ("gentle", "teacher", "banter", "cold")
_HINT_LEVEL_VALUES = ("off", "direction", "specific", "demo")
_PACING_VALUES = ("fast", "standard", "slow")


def _validated_play_config(game: dict | None, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """校验 ``play_game`` 的偏好参数 → ``(config, ignored)``.

    - ``config`` —— 只含该游戏**真的支持**的项：人数在 ``player_counts``
      内、难度在 ``difficulties`` 内、座位在 ``seat_options`` 内、主题在
      ``variant_themes`` 内；性格/提示档/节奏为平台级枚举；两个布尔开关
      严格取 bool（字符串一律不认，避免把 ``"false"`` 当 True）。
    - ``ignored`` —— 用户提了但本游戏不支持的项（人类可读短句），交给
      前端在开局提示里说明：宁可说"这游戏不支持 9 人"，也不静默吞掉。

    任何拿不准的取值都**丢弃而不报错**：偏好非法绝不该让开局失败
    （fail-soft），缺省/全非法时 ``config`` 为空 dict。
    """
    config: dict[str, Any] = {}
    ignored: list[str] = []
    if not isinstance(arguments, dict):
        return config, ignored
    counts = [int(c) for c in (game or {}).get("player_counts") or ()]
    tiers = [str(d) for d in (game or {}).get("difficulties") or ()]
    seats = [str(s) for s in (game or {}).get("seat_options") or ()]
    themes = [str(t) for t in (game or {}).get("variant_themes") or ()]
    for raw_key, field_name in _PLAY_CONFIG_FIELDS.items():
        if raw_key not in arguments or arguments.get(raw_key) is None:
            continue
        value = arguments.get(raw_key)
        if field_name in ("adaptive", "teaching"):
            if isinstance(value, bool):
                config[field_name] = value
            continue
        if field_name == "playerCount":
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if counts and count not in counts:
                ignored.append(f"{count}人")
                continue
            config[field_name] = count
            continue
        text = str(value).strip()
        if not text:
            continue
        if field_name == "difficulty":
            if tiers and text not in tiers:
                ignored.append(f"难度 {DIFFICULTY_LABELS.get(text, text)}")
                continue
            config[field_name] = text
        elif field_name == "theme":
            # 只有**声明了主题**的游戏才接受主题：``/api/match/start`` 见到 theme 就会
            # 按 ``f"{theme}_{tier}"`` 拼 variant 传给引擎——给无主题的游戏塞主题会
            # 变成一个非法变体。拿不准就当"不支持"上报。
            if not themes or text not in themes:
                ignored.append(f"主题 {THEME_LABELS.get(text, text)}")
                continue
            config[field_name] = text
        elif field_name == "playerPid":
            if seats and text not in seats:
                ignored.append(f"座位 {text}")
                continue
            config[field_name] = text
        elif field_name == "persona":
            if text not in _PERSONA_VALUES:
                ignored.append(f"性格 {text}")
                continue
            config[field_name] = text
        elif field_name == "hintLevel":
            if text not in _HINT_LEVEL_VALUES:
                ignored.append(f"提示档 {text}")
                continue
            config[field_name] = text
        elif field_name == "pacing":
            if text not in _PACING_VALUES:
                ignored.append(f"节奏 {text}")
                continue
            config[field_name] = text
    return config, ignored


# ── 传给 LLM 的信息「不过分技术化」：快照/合法动作 → 中文读法 ──────
# 机器契约（快照 id、canonical key、工具参数）原样保留在别处；这里只
# 是 LLM 直面文本的出口 —— 牌/卡/角色/动作全走 engine_helpers 名称层。

#: 各族合法动作 payload 的“类型 → 中文名”映射（未知直出原 type）。
_LEGAL_TYPE_NAMES: dict[str, dict[str, str]] = {
    "mahjong": {
        "discard": "打出",
        "win_self": "自摸",
        "claim_win": "荣和",
        "claim_peng": "碰",
        "claim_gang": "明杠",
        "gang_concealed": "暗杠",
        "gang_added": "加杠",
        "claim_chi": "吃",
        "claim_pass": "过",
    },
    "social": {
        "speak": "发言",
        "vote": "投票",
        "kill": "击杀",
        "check": "查验",
        "shoot": "开枪",
        "heal": "救援",
        "poison": "下毒",
        "guard": "守护",
        "pass": "过",
    },
}
_UNO_FIXED_NAMES = {"draw": "摸牌", "pass": "过", "jump_pass": "放弃抢牌", "take_penalty": "吃下罚牌"}
_POKER_CHOICE_NAMES = {"call": "跟注", "fold": "弃牌", "raise": "加注", "all_in": "全下", "check": "过牌"}
_UNO_COLOR_LABELS = {"r": "红", "b": "蓝", "g": "绿", "y": "黄"}
_UNO_SYMBOL_LABELS = {"skip": "禁止", "reverse": "反转", "draw2": "+2", "wild": "万能", "wild4": "+4"}
_MAHJONG_PHASE_LABELS = {"action": "出牌", "claim": "响应", "discard": "打出"}
_MAHJONG_MELD_LABELS = {"chi": "吃", "peng": "碰", "gang": "杠", "concealed_gang": "暗杠", "added_gang": "加杠"}


def _legal_payload_text(family: str, payload: Any) -> str:
    """把一个合法动作项译成「中文（机器参数附注）」一句话.

    中文名让 LLM 读懂局面，括号里的 ``key=value`` 让模型能产出
    ``make_move`` 可校验的参数（双轨：读得懂 + 用得对）。
    未知形状直出 JSON（fail-soft）。
    """
    if not isinstance(payload, dict):
        return str(payload)
    ptype = str(payload.get("type", ""))
    if family == "mahjong":
        head = _LEGAL_TYPE_NAMES["mahjong"].get(ptype, ptype)
        tile = payload.get("tile")
        if tile not in (None, "", [], {}):
            return f"{head} {mahjong_tile_name(str(tile))}(tile={tile})"
        tiles = payload.get("tiles")
        if tiles not in (None, "", [], {}):
            names = "".join(mahjong_tile_name(str(t)) for t in tiles)
            return f"{head} {names}(tiles={tiles})"
        return head
    if family == "poker":
        choice = str(payload.get("choice", "") or "")
        if not choice:
            return json.dumps(payload, ensure_ascii=False)
        label = _POKER_CHOICE_NAMES.get(choice, choice)
        amount = payload.get("amount")
        if amount not in (None, 0, ""):
            return f"{label} {amount}(amount={amount})"
        return label
    if family == "uno":
        fixed = _UNO_FIXED_NAMES.get(ptype)
        if fixed:
            return fixed
        parts = [ptype]
        card = payload.get("card")
        if card not in (None, "", []):
            parts.append(f"{uno_card_name(str(card))}(card={card})")
        color = payload.get("color")
        if color not in (None, "", []):
            parts.append(f"{_UNO_COLOR_LABELS.get(str(color), str(color))}(color={color})")
        target = payload.get("target")
        if target not in (None, "", []):
            parts.append(f"{seat_label(str(target))}(target={target})")
        return " ".join(parts)
    if family == "social":
        label = _LEGAL_TYPE_NAMES["social"].get(ptype, ptype)
        target = payload.get("target")
        if target not in (None, "", []):
            return f"{label} {seat_label(str(target))}(target={target})"
        return label
    return json.dumps(payload, ensure_ascii=False)


def _mahjong_melds_text(rows: Any) -> str:
    """副露数组 → ``碰三条；吃一二三`` 一段中文（未知项跳过）。"""
    parts: list[str] = []
    for m in rows if isinstance(rows, list) else []:
        if not isinstance(m, dict):
            continue
        label = _MAHJONG_MELD_LABELS.get(str(m.get("type", "")), str(m.get("type", "") or ""))
        if not label:
            continue
        tiles = m.get("tiles") or []
        parts.append(label + "".join(mahjong_tile_name(str(t)) for t in tiles))
    return "；".join(parts)


def _humanize_snap(snap: dict, family: str) -> dict:
    """把玩家投影快照的值人化（键名保留 —— 前端/测试契约不变）.

    手牌/牌河/副露/公共牌/顶牌/身份等 id 一律换成中文名；数值型/枚举
    噪音字段（``last_action`` / ``raise_amounts`` 等）保持原样 ——
    它们不是“牌面”，原值对模型反而更精确。
    """
    out = dict(snap)
    if out.get("last_ai_action"):
        out["last_ai_action"] = canonical_family_text(family, str(out["last_ai_action"]))
    if family == "mahjong":
        for k in ("my_hand", "ai_hand"):
            if isinstance(out.get(k), list):
                out[k] = "、".join(piece_names("mahjong", out[k]))
        for k in ("melds", "discards"):
            container = out.get(k)
            if isinstance(container, dict):
                out[k] = {
                    pid: _mahjong_melds_text(rows) if k == "melds" else "、".join(piece_names("mahjong", rows))
                    for pid, rows in container.items()
                }
        for k in ("last_discard", "last_drawn"):
            if out.get(k):
                out[k] = mahjong_tile_name(str(out[k]))
        if out.get("last_discarder"):
            out["last_discarder"] = (
                "你" if out["last_discarder"] == out.get("player_pid") else str(out["last_discarder"])
            )
        if out.get("phase"):
            out["phase"] = _MAHJONG_PHASE_LABELS.get(str(out["phase"]), out["phase"])
        if isinstance(out.get("legal"), list):
            out["legal"] = [_legal_payload_text("mahjong", x) for x in out["legal"]]
    elif family == "poker":
        for k in ("community", "my_hole", "ai_hole"):
            if isinstance(out.get(k), list):
                out[k] = "、".join(piece_names("poker", out[k]))
        if isinstance(out.get("legal"), list):
            out["legal"] = [_legal_payload_text("poker", x) for x in out["legal"]]
        if out.get("last_actor"):
            out["last_actor"] = "你" if out["last_actor"] == out.get("player_pid") else str(out["last_actor"])
    elif family == "uno":
        for k in ("my_hand", "ai_hand"):
            if isinstance(out.get(k), list):
                out[k] = "、".join(piece_names("uno", out[k]))
        if out.get("top_color"):
            out["top_color"] = _UNO_COLOR_LABELS.get(str(out["top_color"]), out["top_color"])
        if out.get("top_symbol"):
            out["top_symbol"] = _UNO_SYMBOL_LABELS.get(str(out["top_symbol"]), out["top_symbol"])
        if out.get("discard_top"):
            out["discard_top"] = uno_card_name(str(out["discard_top"]))
        if isinstance(out.get("discard_recent"), list):
            out["discard_recent"] = "、".join(uno_card_name(str(c)) for c in out["discard_recent"])
        if out.get("penalty_target"):
            out["penalty_target"] = (
                "你" if out["penalty_target"] == out.get("player_pid") else str(out["penalty_target"])
            )
        if isinstance(out.get("legal"), list):
            out["legal"] = [_legal_payload_text("uno", x) for x in out["legal"]]
    elif family == "social":
        if out.get("my_role"):
            out["my_role"] = social_role_name(str(out["my_role"]))
        if isinstance(out.get("legal"), list):
            out["legal"] = [_legal_payload_text("social", x) for x in out["legal"]]
    return out


def _result_label(snap: dict) -> str:
    """终局结果称呼（绝不泄漏内部 pid 如 ``p_white``）。

    你 / AI 获胜 / 平局——与 ``_render_board`` 的"你/AI"口径一致。判定交给
    :func:`layer4_interface.result.player_won`：社交游戏的 ``winner`` 是**阵营名**
    ——直接 ``winner == player_pid`` 会把卧底获胜判成「AI 获胜」（实测 bug
    e7deb84b），先按 ``final_roles`` 身份表做阵营匹配（终局揭晓，公开信息）。
    ``snap`` 取 ``winner`` / ``player_pid`` / ``winners`` / ``final_roles``。
    """
    w = snap.get("winner")
    won = player_won(w, snap.get("player_pid"), snap.get("winners"), snap)
    if won is None:
        return "平局"
    return "你获胜" if won else "AI 获胜"


def _legal_context(session: Any) -> str:
    """Condensed, *already-projected* legal context for the model (hidden info red line).

    用 ``spec.build_snapshot``（玩家投影）而不是 ``session.snapshot()``
    —— 后者会 drain 掉待投递的陪伴消息（``drain_chat`` 副作用），
    用户对局中每聊一句就会吞掉队列里的 blunder/good_move 评语。
    """
    try:
        snap = session.spec.build_snapshot(session)
    except Exception:
        return ""
    parts: list[str] = []
    if snap.get("over"):
        parts.append(f"本局已结束，{_result_label(snap)}")
        return "；".join(parts)
    family = getattr(session, "family", None) or game_family(getattr(session, "game_id", ""))
    for key in ("legal", "legal_options", "legal_actions", "choices"):
        val = snap.get(key)
        if isinstance(val, list) and val:
            text = "；".join(_legal_payload_text(family, x) for x in val)
            parts.append("合法动作: " + text[:600])
            break
    board = snap.get("board")
    if isinstance(board, list) and board:
        empty = [i for i, p in enumerate(board) if p is None or p == 0]
        if empty:
            parts.append("可落子格(0基索引): " + json.dumps(empty))
    turn = snap.get("turn")
    player_pid = snap.get("player_pid")
    if turn is not None and player_pid is not None:
        parts.append("当前轮到: " + ("你" if turn == player_pid else "AI"))
    return "；".join(parts)


def _render_board(board: list, player_pid: Any) -> str:
    """Grid board → text rows（``你``/``AI``/``·``），供模型读局面。"""
    n = len(board)
    size = int(n**0.5)
    if size * size != n or size < 1:
        return ""
    rows: list[str] = []
    for r in range(size):
        cells: list[str] = []
        for c in range(size):
            v = board[r * size + c]
            if v in (None, 0, ""):
                cells.append("·")
            elif v == player_pid:
                cells.append("你")
            else:
                cells.append("AI")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _match_state_text(session: Any) -> str:
    """当前对局的实时状态文案（``get_match_state`` 载荷）——按陪伴身份分派.

    - **玩家视角**（默认/教学/自定义）：``spec.build_snapshot`` 就是发给
      玩家浏览器的那份投影（各族 reveal-gate 已挡掉隐藏信息），AI/对手的
      底牌/手牌不可见，红线天然满足。
    - **AI 视角**（二人非教练对手模式）：换成 AI 自己的投影——只含 AI 自己
      的牌 + 公开盘面，**绝不含玩家底牌/手牌**（对手本来就看不到玩家的牌）。
      —— 修 2026-08 对局记录泄露：对手模式聊天里 AI 报出「你现在手里是方块9
      和红桃3」，根因就是聊天通道把玩家投影直接喂给了扮演对手的模型。

    都不调用 ``session.snapshot()``——那会 drain 掉待投递的陪伴消息。
    """
    if bool(getattr(session, "is_opponent_mode", False)):
        return _ai_state_text(session)
    return _player_state_text(session)


def _player_state_text(session: Any) -> str:
    """玩家视角快照渲染（默认/教学/自定义共用；公开信息到玩家本人都可见）。"""
    try:
        snap = session.spec.build_snapshot(session)
    except Exception:
        return ""
    parts: list[str] = []
    family = getattr(session, "family", None) or game_family(getattr(session, "game_id", ""))
    snap = _humanize_snap(snap, family)
    player_pid = snap.get("player_pid")
    if snap.get("over"):
        parts.append(f"本局已结束，{_result_label(snap)}")
    else:
        turn = snap.get("turn")
        if turn is not None and player_pid is not None:
            parts.append("当前轮到: " + ("你" if turn == player_pid else "AI"))
    board = snap.get("board")
    if isinstance(board, list) and board:
        grid = _render_board(board, player_pid)
        if grid:
            parts.append("棋盘（·为空，你=你的子，AI=对方的子）:\n" + grid)
    for key, val in snap.items():
        if key in _STATE_SKIP_KEYS:
            continue
        if val is None or val == "" or val == [] or val == {}:
            continue
        try:
            rendered = json.dumps(val, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = str(val)
        if len(rendered) > _STATE_PER_KEY_MAX:
            rendered = rendered[:_STATE_PER_KEY_MAX] + "…"
        parts.append(f"{key}: {rendered}")
    text = "\n".join(parts)
    if len(text) > _STATE_MAX_CHARS:
        text = text[:_STATE_MAX_CHARS] + "…（已截断）"
    return text


def _ai_state_text(session: Any) -> str:
    """AI 视角快照渲染（二人非教练对手模式）——绝不包含玩家底牌/手牌.

    公开盘面（阶段/底池/公共牌/筹码/已投入/弃牌/合法动作）与玩家快照
    同源；隐藏字段按 AI 视角重写：「我的底牌/手牌」= AI 自己的牌（AI 本
    就看得到，仅供判断牌力），玩家底牌/手牌一律不出现（visibility 规则
    本就不给 AI）。网格等公开盘面族玩家的快照全是公开信息，直接复用玩家
    视角渲染。
    """
    try:
        snap = session.spec.build_snapshot(session)
    except Exception:
        return ""
    family = getattr(session, "family", None) or game_family(getattr(session, "game_id", ""))
    if family not in ("poker", "uno"):
        # 网格 / 社交等公开盘面族：玩家快照即公开信息（社交族无二人局）。
        return _player_state_text(session)
    parts: list[str] = []
    ai_pid = str(getattr(session, "ai_pid", "") or "")
    if snap.get("over"):
        parts.append(f"本局已结束，{_result_label(snap)}")
    else:
        turn = snap.get("turn")
        if turn is not None:
            parts.append("当前轮到: " + ("你（AI）" if turn == ai_pid else "玩家"))
    for key, label in (
        ("street_name", "阶段"),
        ("pot", "底池"),
        ("my_stack", "玩家筹码"),
        ("ai_stack", "你(AI)筹码"),
        ("my_committed", "玩家已投入"),
        ("ai_committed", "你(AI)已投入"),
        ("my_folded", "玩家已弃牌"),
        ("ai_folded", "你(AI)已弃牌"),
        ("call_to", "跟注额"),
    ):
        val = snap.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            parts.append(f"{label}: {'是' if val else '否'}")
        elif val != "" and val != []:
            parts.append(f"{label}: {val}")
    community = snap.get("community")
    if isinstance(community, list) and community:
        parts.append("公共牌: " + "、".join(piece_names("poker", community)))
    # AI 自己的牌（AI 投影视图提取；玩家底牌不出现在任何字段里）。
    try:
        obs = session.engine.project_observation(session.state, ai_pid)
        ai_hand = extract_hand(obs, ai_pid) if obs else []
    except Exception:
        ai_hand = []
    if family == "poker":
        own = "、".join(piece_names("poker", ai_hand)) if ai_hand else "（未发牌）"
        parts.append(f"我的底牌（AI 自己可见）: {own}")
        parts.append("玩家底牌: 不可见（隐藏）")
    elif family == "uno":
        own = "、".join(piece_names("uno", ai_hand)) if ai_hand else "（无）"
        parts.append(f"我的手牌（AI 自己可见）: {own}")
        parts.append("玩家手牌: 不可见（隐藏）")
        for key, label in (
            ("top_color", "顶牌颜色"),
            ("top_symbol", "顶牌符号"),
            ("discard_top", "顶牌名称"),
            ("deck_count", "牌堆余量"),
        ):
            val = snap.get(key)
            if val not in (None, "", []):
                parts.append(f"{label}: {val}")
    legal = snap.get("legal")
    if isinstance(legal, list) and legal:
        parts.append("玩家合法动作: " + "；".join(_legal_payload_text(family, x) for x in legal))
    raise_amts = snap.get("raise_amounts")
    if isinstance(raise_amts, list) and raise_amts:
        parts.append("加注档位: " + "、".join(str(a) for a in raise_amts))
    text = "\n".join(parts)
    if len(text) > _STATE_MAX_CHARS:
        text = text[:_STATE_MAX_CHARS] + "…（已截断）"
    return text


def _guard_text(text: str, session: Any) -> str:
    """聊天最终文本的隐藏信息后置扫描（按陪伴身份选态）.

    - **对手模式**（二人非教练）：adversarial——拦「玩家的隐藏牌」与
      「AI 自己的具体花色点数」（黑桃4 / ♠A / 红5…），放行 AI 模糊牌力
      （「这手还行」「一对K」）。**不向玩家的牌做任何具体表述**。
    - **教学**：teaching——放行玩家自己的牌（教练看的就是玩家投影），拦
      对手/AI 的隐藏牌。
    - **默认**（啦啦队/多人）：玩家自己可见的信息本就可被陪玩复述（
      get_match_state 给的就是玩家投影），不额外拦截；模型拿不到其它隐藏
      信息，扫描不改变既有默认行为。
    - custom / 无扫描规则的 id：原样返回（fail-soft）。
    """
    if not text or session is None:
        return text
    opponent = bool(getattr(session, "is_opponent_mode", False))
    teaching = bool(getattr(session, "teaching", False))
    if not opponent and not teaching:
        return text
    # session.game_id 是会话 uuid（如 b63c941e），扫描规则按**规则 JSON 的
    # builtin id**（session.spec.game_id，如 texas_holdem）分派——不要拿 uuid
    # 去查规则表。自定义局（custom_games 注册表）的 spec.game_id 不在表内时
    # 走 infer → unknown → 原样返回（fail-soft）。
    spec = getattr(session, "spec", None)
    spec_game = str(getattr(spec, "game_id", "") or "")
    scan_game = resolve_scan_game(spec_game, {})
    if scan_game == "unknown":
        return text
    return scan(text, scan_game, teaching=teaching, adversarial=opponent)


def _finalize_chat_result(result: ChatTurnResult, session: Any) -> ChatTurnResult:
    """对最终回复文本做模式感知隐藏信息扫描（对手/教学两态）."""
    if result.text:
        result.text = _guard_text(result.text, session)
    return result


def _latest_match(match_history: Any) -> dict | None:
    """Newest finished match meta (``None`` when history is off/empty).

    阵营胜者（undercover/werewolf 的 ``winner`` 是阵营名）无法从 meta 直接
    判定玩家胜负——补读完整记录最后一手的 ``snapshot.final_roles`` 并解析出
    ``_won``（``None`` = 平局/无法判定），供 ``_system_prompt`` 的
    「最近一局」行使用（否则卧底获胜的一局会被写成「AI 获胜」）。
    """
    if match_history is None:
        return None
    try:
        matches = match_history.list_matches(limit=1)
    except Exception:
        return None
    if not matches:
        return None
    meta = matches[0]
    out = dict(meta)
    try:
        full = match_history.get(str(meta.get("match_id") or ""))
        moves = full.get("moves") if isinstance(full, dict) else None
        if isinstance(moves, list) and moves and isinstance(moves[-1], dict):
            snap = moves[-1].get("snapshot")
            if isinstance(snap, dict):
                won = player_won(meta.get("winner"), meta.get("player_pid"), meta.get("winners"), snap)
                if won is not None:
                    out["_won"] = won
    except Exception:
        pass
    return out


def _review_text(match: dict, report: ReviewReport) -> str:
    """Deterministic review narration source (timeline + key nodes)."""
    moves = match.get("moves") if isinstance(match.get("moves"), list) else []
    kind_labels = {"turning_point": "转折点", "winning_move": "胜着", "blunder": "昏招"}
    lines = [f"对局 {match.get('match_id')}，{report.summary}", "走子时间线（1 基手数；你=人类座位，AI=AI座位）:"]
    for m in moves[:_REVIEW_TIMELINE_MAX]:
        if not isinstance(m, dict):
            continue
        actor = "你" if m.get("actor") == "human" else "AI"
        lines.append(f"{int(m.get('step', 0)) + 1}. {actor}: {m.get('action', '')}")
    if len(moves) > _REVIEW_TIMELINE_MAX:
        lines.append(f"…（共 {len(moves)} 手，已截断）")
    if report.key_nodes:
        lines.append("关键节点:")
        for node in report.key_nodes:
            label = kind_labels.get(node.kind, node.kind)
            what = f"（{node.what}）" if node.what else ""
            lines.append(f"- 第 {node.step + 1} 手 {label}{what} —— {node.why}")
    if report.improvement:
        lines.append(f"改进建议: {report.improvement}")
    return "\n".join(lines)[:_REVIEW_TEXT_MAX]


def _system_prompt(
    games: list[dict],
    session: Any,
    active: list[dict],
    latest: dict | None = None,
    *,
    persona: Persona | None = None,
) -> str:
    # 身份块（方向 C 人设统一）：平台助手与对局陪玩共用同一份 persona
    # 描述——用户在 profile 选一次人设后，两层身份连贯。persona=None
    # 时退回无性格的工具型助手（兼容旧调用方 / 测试）。
    head = persona_identity_block(persona) + "\n" if persona is not None else ""
    opponent = session is not None and bool(getattr(session, "is_opponent_mode", False))
    if opponent:
        # 二人非教练 = 座内对手：聊天模型就是牌桌对面的 AI 对手。它只能看
        # 自己的投影（get_match_state 返回 AI 视角），红线镜像 teaching——
        # 拦「玩家的隐藏牌」与「AI 自己的具体花色点数」，放行 AI 模糊牌力。
        # 修复 2026-08 对局记录：对手模式聊天把玩家投影直接喂给模型，AI 报出
        # 「你现在手里是方块9和红桃3」；且提示词无红线，AI 自报「黑桃K」。
        lines = [
            head + "你是玩家在本局牌桌对面的**座内对手**（二人非教练对局）。用户一句话 → 你选一个工具。",
            "规则：",
            "1. 用户没指明玩哪个游戏时，不要调用 play_game，直接回复询问（intent clarify）。",
            "2. 对局中的替玩家落子/发言只能用描述里给出的合法动作；含糊的话不调用动作工具，直接聊天。",
            "3. 用户一句话里带了开局偏好（人数/难度/主题/性格/提示档/节奏/自适应/教学/座位）时，"
            "把偏好填进 play_game 的对应参数；取值先看 describe_game 给出的可配置项，"
            "用户没说的**不要为了问而问**——直接省略按默认开局（前端不再有开局配置卡）。",
            "4. 用户要创建游戏：信息够了（规则描述，或基础模板 + 改动描述）直接调用 create_game，"
            "不要自己编造规则；缺规则或缺基础模板时先追问（intent clarify）。",
            "5. 你能看到**自己**的底牌/手牌（仅供判断牌力、决定下注与虚张），"
            "但**看不到玩家的底牌/手牌/身份**——那是玩家的隐藏信息。"
            "需要局面细节时调用 get_match_state（它返回**你(AI)自己可见的投影**："
            "你的底牌 + 公共牌 + 公开下注；绝不含玩家底牌——也不要猜测玩家底牌）。",
            "6. 红线一：绝不提及或猜测玩家的未公开信息（底牌、手牌、身份等）——只能基于玩家"
            "公开的下注/弃牌/摸打序列推断意图（读人），绝不报玩家未公开牌面。",
            "7. 红线二：绝不报出**你自己**底牌的具体花色与点数（如「黑桃4」「♠A」「s10」"
            "「红5」），只能说「这手还行」「牌不大」「一对K」这类模糊牌力——报出具体牌面"
            "等于明牌，会直接毁掉这局；终局 showdown 揭底后双方牌公开，可做完整复盘式点评。",
            "8. 知识红线：用户问游戏/平台知识（“X是什么/怎么玩/规则/有哪些游戏”）时，先调用 "
            "describe_game / list_games 取权威资料，只依据资料回答；资料里没有的细节不要编造。",
            "9. 平台功能提问：用户问**某功能怎么用/在哪**时，先调用 get_platform_help 取该主题的"
            "权威说明，再依据资料回答；不要泛泛而谈或编造功能细节。",
        ]
    else:
        lines = [
            head + "你是 Gavis 平台的对话助手（agent 聊天模式）。用户一句话 → 你选一个工具。",
            "规则：",
            "1. 用户没指明玩哪个游戏时，不要调用 play_game，直接回复询问（intent clarify）。",
            "2. 对局中的落子/发言只能用描述里给出的合法动作；含糊的话不调用动作工具，直接聊天。",
            "3. 用户一句话里带了开局偏好（人数/难度/主题/性格/提示档/节奏/自适应/教学/座位）时，"
            "把偏好填进 play_game 的对应参数；取值先看 describe_game 给出的可配置项，"
            "用户没说的**不要为了问而问**——直接省略按默认开局（前端不再有开局配置卡）。",
            "4. 用户要创建游戏：信息够了（规则描述，或基础模板 + 改动描述）直接调用 create_game，"
            "不要自己编造规则；缺规则或缺基础模板时先追问（intent clarify）。",
            "5. 隐藏信息红线：不得编造任何对手/其他玩家的未公开信息"
            "（手牌、身份、底牌、未翻开的牌、棋局评估等）——依据用户输入和给出的合法动作行事；"
            "需要局面细节时调用 get_match_state（它返回玩家自己可见的投影）。",
            "6. 知识红线：用户问游戏/平台知识（“X是什么/怎么玩/规则/有哪些游戏”）时，先调用 "
            "describe_game / list_games 取权威资料，只依据资料回答；资料里没有的细节不要编造，"
            "直接说不知道或建议开一局体验。",
            "7. 平台功能提问：用户问**某功能怎么用/在哪**（“怎么创建游戏”“在线学习怎么用”“评测中心"
            "在哪”“教学对局是什么”“视觉识别”“LLM配置”）时，先调用 get_platform_help 取该主题的"
            "权威说明，再依据资料回答；不要泛泛而谈或编造功能细节。",
        ]
    if games:
        catalog = "\n".join("- " + _game_brief(g) for g in games[:24])
        lines.append("可用游戏:\n" + catalog)
    if session is not None:
        name = session.spec.display_name
        ctx = _legal_context(session)
        lines.append(f"当前对局: {name}（{session.game_id}）")
        if ctx:
            lines.append(ctx)
        if session.persona:
            lines.append(f"陪伴角色: {session.persona}")
    if (session is None or session.over) and latest:
        # 终局后 session 已从注册表移除——用最近一局补上下文，让
        # “复盘一下”不必从零开始（get_match_review 从历史取数）。
        display = next((g["display_name"] for g in games if g.get("game_id") == latest.get("game_id")), None)
        label = display or str(latest.get("game_id") or "对局")
        # 阵营胜者已由 _latest_match 解析为 _won（玩家视角）；无 _won 时退回
        # pid 比较（常规对局）与"平局"（无胜者）。
        won = latest.get("_won")
        if won is not None:
            result = "你获胜" if won else "AI 获胜"
        else:
            winner = str(latest.get("winner") or "")
            if winner and winner == latest.get("player_pid"):
                result = "你获胜"
            elif winner:
                result = "AI 获胜"
            else:
                result = "平局"
        lines.append(
            f"最近一局: {label}（{latest.get('moves', '?')} 手，{result}）"
            "——用户想复盘/回顾上一局时调用 get_match_review"
        )
    if active:
        active_names = "、".join(f"{a.get('display_name') or a.get('game_id')}" for a in active[:8])
        lines.append(f"进行中的对局: {active_names}（用户说“继续/恢复”时用 resume_session）")
    return "\n".join(lines)


def build_tools(*, games: list[dict], session: Any, active: list[dict]) -> list[dict]:
    """Build the OpenAI ``tools`` list for the current context.

    Besides the action tools this always exposes the *local* tools — the
    ones the backend executes in-loop and feeds back as ``role: "tool"``
    messages: ``describe_game`` / ``list_games`` (the registry + play docs
    + **可配置面**), ``get_platform_help`` (per-feature platform help docs
    — answers “具体功能怎么用” from authoritative data),
    ``get_match_review`` (latest match timeline + key nodes),
    ``create_game`` (自然语言规则/模板变体 → 真正落盘的自定义游戏),
    ``update_settings`` (对话里说清的偏好 → 白名单校验过的变更 / 选项)
    and, mid-match, ``get_match_state`` (the player-projected live snapshot)
    + ``ask_hint`` (the mechanical hint).

    ``play_game`` 带**偏好参数**：模型从用户那句话里取（“三人局、困难、
    教学对局”），服务端按注册表校验后才传前端——前端不再有开局配置卡，
    所以这里的参数契约就是"配置界面"。
    """
    game_enum = [g["game_id"] for g in games if g["game_id"]]
    tools: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "play_game",
                "description": (
                    "用户想玩某款游戏 / 开局 / 对战。游戏没指明时不要调用。"
                    "用户一句话里带了偏好就一并填上：player_count 人数、difficulty 难度、"
                    "theme 主题（多主题游戏）、persona 陪玩性格、hint_level 提示档、"
                    "pacing 节奏、adaptive 自适应难度、teaching 教学对局、player_pid 座位。"
                    "取值必须来自 describe_game 给出的可配置项；不确定或用户没说就省略，"
                    "不要为了问而问——省略即按默认开局。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string", "enum": game_enum},
                        "player_count": {"type": "integer", "description": "人数（须在该游戏支持人数档内）"},
                        "difficulty": {"type": "string", "enum": ["easy", "normal", "hard"]},
                        "theme": {
                            "type": "string",
                            "description": "主题 id（仅 undercover 等多主题游戏；见 describe_game）",
                        },
                        "persona": {"type": "string", "enum": ["gentle", "teacher", "banter", "cold"]},
                        "hint_level": {"type": "string", "enum": ["off", "direction", "specific", "demo"]},
                        "pacing": {"type": "string", "enum": ["fast", "standard", "slow"]},
                        "adaptive": {"type": "boolean", "description": "自适应难度（按近期胜率自动升降）"},
                        "teaching": {"type": "boolean", "description": "教学对局（教练看你的牌带你打）"},
                        "player_pid": {"type": "string", "description": "人类座位 pid（见 describe_game）"},
                    },
                    "required": ["game_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_game",
                "description": (
                    "创建自定义游戏：用户描述规则（mode=from_scratch，rule_text 必填）或"
                    "基于已有游戏改变体（mode=variant，base_game_id + change_text 必填）。"
                    "用户没说规则/没说基础模板时**不要调用**，先追问（intent clarify）；"
                    "信息够了就调用，不要自己编造规则。默认走确定性模板翻译（快）；"
                    "只有用户明确要求用大模型翻译时才 use_llm=true（较慢）。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["from_scratch", "variant"]},
                        "rule_text": {"type": "string", "description": "从零描述的游戏规则（中文自然语言）"},
                        "base_game_id": {"type": "string", "description": "变体模式的基础模板 id"},
                        "change_text": {"type": "string", "description": "变体模式要做的改动描述"},
                        "game_name": {"type": "string", "description": "可选游戏名"},
                        "use_llm": {"type": "boolean", "description": "是否用 LLM 翻译规则（默认 false）"},
                    },
                    "required": ["mode"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_settings",
                "description": (
                    "把用户在对话里**明确说出**的平台偏好改成对应取值并回执：persona 助手性格、"
                    "hint_level 提示档、difficulty AI 难度、theme 界面主题。"
                    "取值必须来自用户那句话/上文明确点到的那个（“换成高冷一点”→cold）；"
                    "**不要替用户猜**：只说“换个风格/改设置”而没说成哪种时，仍然调用本工具但不要填任何"
                    "偏好参数——平台会给出选项让用户挑。"
                    "用户明确要求“打开设置页/进设置”时用 open_page=true（只切页面、不改偏好）。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "persona": {
                            "type": "string",
                            "enum": ["gentle", "teacher", "banter", "cold"],
                            "description": "助手/陪玩性格：gentle 温柔陪伴、teacher 认真教学、banter 轻松吐槽、cold 高冷竞技",
                        },
                        "hint_level": {"type": "string", "enum": ["off", "direction", "specific", "demo"]},
                        "difficulty": {"type": "string", "enum": ["easy", "normal", "hard", "adaptive"]},
                        "theme": {"type": "string", "enum": ["light", "dark"]},
                        "open_page": {
                            "type": "boolean",
                            "description": "用户明确要求打开「设置」页时设为 true（不改任何偏好）",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "describe_game",
                "description": (
                    "查询某款游戏的权威介绍（一句话简介、玩法规则要点、支持人数、难度档、"
                    "座位、主题等可配置项）。用户问“X是什么/怎么玩/规则”时先调用它；"
                    "需要知道某款游戏能怎么配置（人数/难度/主题）时也调用它，再据此填 play_game 参数。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"game_id": {"type": "string", "enum": game_enum}},
                    "required": ["game_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_games",
                "description": "列出平台全部游戏（按棋盘/扑克/麻将/UNO/自定义分组）。用户问“有哪些游戏/目录”时调用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_platform_help",
                "description": (
                    "查询平台某项功能的权威帮助文档（是什么/怎么说一句话触发/入口在哪/注意点）。"
                    "用户问**具体功能**怎么用时调用——如“怎么开局/继续对局/落子/要提示/看战绩/复盘/"
                    "创建游戏/改设置/评测中心/在线学习/教学对局/LLM配置/视觉识别”。"
                    "topic 拿不准时可省略以获取主题总览。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "enum": list(PLATFORM_TOPIC_KEYS),
                            "description": "要查询的功能主题 key；省略则返回全部主题总览。",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_match_review",
                "description": (
                    "获取最近一局（或指定 match_id）的复盘资料：完整走子时间线 + 关键节点"
                    "（转折点/胜着/昏招，含具体动作内容）+ 改进建议。用户说“复盘/回顾上一局/"
                    "讲讲关键手和失误点”时调用，依据资料用中文讲解。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"match_id": {"type": "string", "description": "对局 id；缺省取最近一局"}},
                    "required": [],
                },
            },
        },
    ]
    if active:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "resume_session",
                    "description": "用户说“继续上一局/回到对战/接着玩”时调用，恢复进行中的对局。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
    if session is not None and not session.over:
        family = session.family or _BUILTIN_FAMILY.get(session.game_id) or "grid"
        shape = _ACTION_SHAPES.get(family, _ACTION_SHAPES["grid"])
        ctx = _legal_context(session)
        try:
            your_turn = session.current_player == session.player_pid
        except Exception:
            your_turn = True
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "make_move",
                    "description": (
                        f"执行你（人类玩家）当前合法回合的动作。action 必须是 dict：{shape}。"
                        f"{'当前正轮到你。' if your_turn else '还没轮到你，不要调用。'} {ctx}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"action": {"type": "object", "description": "合法动作对象"}},
                        "required": ["action"],
                    },
                },
            }
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "get_match_state",
                    "description": (
                        "获取当前对局的实时状态"
                        + (
                            "（**你(AI)自己可见的投影**：你的底牌/手牌 + 公共牌/牌河 + "
                            "公开下注与筹码。玩家的底牌/手牌不可见——不要猜测玩家的隐藏牌，"
                            "也不要向玩家报出你自己底牌的具体花色点数）。"
                            if bool(getattr(session, "is_opponent_mode", False))
                            else "（棋盘布局/你的手牌/公共牌/牌河/各家张数等——都是玩家自己能看到的公开信息）。"
                        )
                        + "用户问“现在什么局面/我有什么牌/这步怎么走”或需要局面细节时调用，"
                        "依据返回内容回答，不要凭空猜测。"
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "ask_hint",
                    "description": (
                        "用户要提示/指导/“这步怎么走”时调用：返回该局的机械提示"
                        "（direction 方向 / specific 具体 / demo 演示），你结合局面给出讲解。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "enum": ["direction", "specific", "demo"],
                                "default": "direction",
                            }
                        },
                        "required": [],
                    },
                },
            }
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "restart_game",
                    "description": "用户说“再来一局/重新开始/换一局”且当前有对局时调用。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
    for name, description in [
        ("show_history", "用户想看战绩/历史/胜率时调用"),
        # 注意：create_game 上面已有**带参数**的同名工具（本地执行、真正落盘），
        # 这里绝不能再登记一次——工具名重复会被端点的 schema 校验整包拒绝
        # （DeepSeek：400 Tool names must be unique），全量工具随之失效、
        # 每个聊天回合都掉进正则兜底。见 test_build_tools_names_are_unique。
        (
            "open_platform",
            "用户想回到完整平台界面（大厅/对局/战绩等）时调用；"
            "用户只是要打开「设置」页时改用 update_settings(open_page=true)。",
        ),
        ("run_benchmark", "用户想看评测/求解器对比时调用"),
        ("show_learning", "用户想看在线学习状态时调用"),
        (
            "help",
            "用户**泛泛**问你能做什么/怎么用（未指向具体功能）时调用；"
            "指向具体功能（如“怎么创建游戏”“在线学习怎么用”）时改用 get_platform_help。",
        ),
    ]:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
    return tools


# ── Tool dispatch ─────────────────────────────────────────────────


def _find_game(text: str, games: list[dict]) -> dict | None:
    """Best-effort game match by display_name / game_id / alias substring.

    匹配大小写不敏感（拉丁短名 "uno"/"UNO" 等价），并纳入每款游戏的
    别名表（``GAME_ALIASES``）——display_name 带括注（「UNO（经典）」）
    或空格（「UNO 抢牌」）时，用户口语短名（"UNO"/"UNO抢牌"）也能命中。
    同一句命中多款时最长匹配胜出（"UNO 7-0" 优先于裸 "UNO"）；**等长
    平局时自定义游戏优先**——内置先入表、``>`` 严格大于曾让内置永久
    遮蔽同名自定义游戏（用户刚创建一款 display_name 与内置 game_id 撞串
    的游戏后，"玩X" 总是开内置而非他创建的那款）。
    """
    lowered = text.lower()
    best: dict | None = None
    best_len = 0
    best_is_custom = False
    for g in games:
        candidates = [str(g.get(key, "")) for key in ("display_name", "game_id")]
        candidates.extend(GAME_ALIASES.get(str(g.get("game_id", "")), ()))
        is_custom = bool(g.get("custom"))
        for name in candidates:
            if not name:
                continue
            lname = name.lower()
            if lname in lowered and (
                len(lname) > best_len or (len(lname) == best_len and is_custom and not best_is_custom)
            ):
                best = g
                best_len = len(lname)
                best_is_custom = is_custom
    return best


def _execute_local_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    games: list[dict],
    session: Any = None,
    manager: PlayManager | None = None,
    match_history: Any = None,
    custom: CustomGameRegistry | None = None,
    allow_create: bool = True,
) -> ChatTurnResult:
    """Run one *local* tool in-process (fail-soft).

    Returns the executed :class:`ChatTurnResult`: its ``text`` is the
    ``role: "tool"`` payload the model reads *and* doubles as the
    deterministic answer when the tool-round budget runs out; tools that
    map to a frontend intent (``ask_hint`` → ``hint`` with the hint
    dict, ``get_match_review`` → ``review`` with the report,
    ``create_game`` → ``create`` with the created registry entry) carry
    intent + params so the narration lands on the right intent.

    Deterministic, no LLM; read-only except ``ask_hint`` marking
    ``session.hinted`` (the user did ask for a hint — the same call the
    ``/match/hint`` route would make) and ``create_game``，它**有副作用**
    （翻译 + 校验 + 落盘一个自定义游戏），因此由 ``allow_create`` 闸门
    保证同一回合至多建一个（同一批次里的重复调用回一句说明而不是再建）。
    """
    if name == "describe_game":
        game_id = str(arguments.get("game_id", ""))
        # 内置游戏走共享拼装（与陪伴对话注入同源）；custom 游戏
        # （无 GameSpec）回落到目录条目 + description。两条路径都以
        # 「可配置项」一行收尾——play_game 的取值就来自这里。
        game = next((g for g in games if g["game_id"] == game_id), None)
        surface = _config_surface_text(game) if game is not None else ""
        knowledge = game_knowledge_text(game_id)
        if knowledge:
            body = knowledge + ("\n" + surface if surface else "")
            return ChatTurnResult(intent="chat", text=body, mood="thinking", params={})
        if game is None:
            body = f"未找到游戏 {game_id!r}。可用游戏请调用 list_games 查看。"
            return ChatTurnResult(intent="chat", text=body, mood="thinking", params={})
        parts = [f"{game['display_name']}（{game_id}）"]
        desc = str(game.get("description") or "")
        if desc:
            parts.append(desc)
        if surface:
            parts.append(surface)
        rules_md = game_rules_text(game_id)
        if rules_md:
            parts.append("规则要点:\n" + rules_md)
        return ChatTurnResult(intent="chat", text="\n".join(parts), mood="thinking", params={})
    if name == "list_games":
        kind_labels = {"board": "棋盘", "poker": "扑克", "mahjong": "麻将", "uno": "UNO"}
        by_label: dict[str, list[str]] = {}
        for g in games:
            if not g["game_id"]:
                continue
            label = kind_labels.get(str(g.get("kind")), "自定义" if g.get("family") else "其他")
            by_label.setdefault(label, []).append(f"{g['display_name']}({g['game_id']})")
        body = "\n".join(f"{label}: " + "、".join(items) for label, items in by_label.items())
        return ChatTurnResult(intent="chat", text=body, mood="thinking", params={})
    if name == "get_match_state":
        body = _match_state_text(session)
        return ChatTurnResult(intent="chat", text=body, mood="thinking", params={})
    if name == "get_platform_help":
        # 具体功能帮助：topic key → 权威文档；topic 缺省/未知 → 主题总览
        # （fail-soft：拿不准时给目录，模型可再指定主题或直接据此作答）。
        topic = str(arguments.get("topic") or "").strip().lower()
        body = platform_help_text(topic) if topic else platform_help_index()
        if not body:
            body = platform_help_index()
        return ChatTurnResult(
            intent="chat",
            text=body,
            mood="thinking",
            params={"topic": topic} if topic else {},
        )
    if name == "ask_hint":
        return _ask_hint_result(arguments, session, manager)
    if name == "get_match_review":
        return _match_review_result(match_history, arguments)
    if name == "create_game":
        if not allow_create:
            # 同回合并发重复调用：已经有创建在执行，别再建第二个。
            return ChatTurnResult(intent="chat", text="（同一回合只执行一次创建。）", mood="thinking", params={})
        return _create_game_result(arguments, custom)
    if name == "update_settings":
        # 对话里改偏好：白名单校验 + 回执（applied / open_page / 选项 chips）。
        return _update_settings_result(arguments)
    return ChatTurnResult(intent="chat", text="", params={})


def _create_error_text(exc: CustomGameError) -> str:
    """``CustomGameError`` → 给模型读的文本化失败原因（fail-soft，不抛）."""
    lines = [f"创建失败：{exc}"]
    validation = getattr(exc, "validation", None)
    errors = list(getattr(validation, "errors", ()) or ())
    warnings = list(getattr(validation, "warnings", ()) or ())
    if errors:
        lines.append("校验错误: " + "；".join(str(e) for e in errors[:6]))
    if warnings:
        lines.append("校验警告: " + "；".join(str(w) for w in warnings[:6]))
    lines.append("可以换一种说法重试，或改用平台「创建游戏」页手动创建。")
    return "\n".join(lines)


def _create_game_result(arguments: dict[str, Any], custom: CustomGameRegistry | None) -> ChatTurnResult:
    """Execute ``create_game``: 用户口述的规则/变体 → 落盘的自定义游戏.

    有副作用的本地工具（与 ``ask_hint`` 同列：就地执行 + 携带意图）。
    成功返回 ``create`` 意图 + ``params.game``（注册表条目，与
    ``/api/custom/games`` 的 ``game`` 字段同源）：前端据此刷新游戏目录
    （新游戏立刻能被「玩X」命中）并提示下一句怎么说。失败/注册表未启用
    一律文本化上报并回退 ``chat`` 意图，让模型解释原因——绝不抛异常毁掉
    一个聊天回合。

    ``use_llm`` 默认 False（确定性模板翻译，快）；只有用户明确要求用
    大模型翻译时才走 LLM（慢且有等待）。
    """
    if custom is None:
        return ChatTurnResult(
            intent="chat",
            text="自定义游戏注册表未启用，暂时不能在对话里创建游戏；可改用平台「创建游戏」页。",
            mood="thinking",
            params={},
        )
    mode = str(arguments.get("mode") or "from_scratch").strip()
    if mode not in ("from_scratch", "variant"):
        mode = "from_scratch"
    try:
        entry = custom.create(
            mode=mode,
            rule_text=arguments.get("rule_text"),
            base_game_id=arguments.get("base_game_id"),
            change_text=arguments.get("change_text"),
            game_name=arguments.get("game_name"),
            use_llm=bool(arguments.get("use_llm", False)),
        )
    except CustomGameError as exc:
        return ChatTurnResult(intent="chat", text=_create_error_text(exc), mood="sorry", params={})
    except Exception as exc:  # 翻译器/校验器任何异常都不该毁掉一个聊天回合
        logger.warning("create_game 工具执行失败: %s", exc)
        return ChatTurnResult(intent="chat", text=f"创建失败：{exc}", mood="sorry", params={})
    game_id = str(entry.get("game_id") or "")
    name = str(entry.get("display_name") or game_id)
    body = f"创建成功《{name}》（id: {game_id}，族: {entry.get('family') or '未知'}）。想玩就说“玩{name}”。"
    return ChatTurnResult(
        intent="create",
        text=body,
        mood="happy",
        params={
            "game_id": game_id,
            "game": entry,
            "family": entry.get("family"),
            "diff_summary": entry.get("diff_summary"),
            "validation": entry.get("validation"),
        },
    )


def _ask_hint_result(arguments: dict[str, Any], session: Any, manager: PlayManager | None) -> ChatTurnResult:
    """Execute ``ask_hint``: the mechanical hint, carried as the ``hint`` intent.

    ``level`` 透传（旧版动作工具直接丢弃），机械提示全文回传给模型
    （可讲解），并以 ``params.hint`` 随 intent 带给前端直接展示。
    """
    level = str(arguments.get("level") or "direction")
    if manager is None or session is None or session.over:
        return ChatTurnResult(intent="chat", text="当前没有可提示的对局。", mood="thinking", params={})
    try:
        hint = manager.hint(session.game_id, level)
    except Exception:
        return ChatTurnResult(
            intent="hint",
            text="提示暂时不可用，可参考描述里的可落子信息。",
            mood="thinking",
            params={"level": level},
        )
    if not isinstance(hint, dict):
        hint = {}
    body = f"机械提示（{level}）：{hint.get('hint') or hint.get('direction') or ''}"
    direction = str(hint.get("direction") or "")
    if direction and direction not in body:
        body += f"\n方向评估：{direction}"
    return ChatTurnResult(
        intent="hint",
        text=body,
        mood="thinking",
        params={"level": level, "hint": hint},
    )


def _match_review_result(match_history: Any, arguments: dict[str, Any]) -> ChatTurnResult:
    """Execute ``get_match_review``: latest (or given) match → narration source.

    复盘的 LLM 信息源：时间线（``moves[].action``）+ 关键节点 + 改进
    建议整体回传给模型讲解；``params.report`` 随 ``review`` intent 带
    给前端渲染复盘卡。没有 LLM 时同一段文本即确定性复盘。
    """
    if match_history is None:
        body = "对局历史未启用。"
        return ChatTurnResult(intent="chat", text=body, mood="thinking", params={})
    match_id = str(arguments.get("match_id") or "").strip()
    match: dict | None = None
    try:
        if not match_id:
            latest = match_history.list_matches(limit=1)
            match_id = str(latest[0].get("match_id") or "") if latest else ""
        if match_id:
            match = match_history.get(match_id)
    except Exception:
        match = None
    if not match_id or match is None:
        body = "还没有已结束的对局记录，先来一局，结束后就能复盘。"
        return ChatTurnResult(intent="chat", text=body, mood="thinking", params={})
    report = review_analyze(match)
    return ChatTurnResult(
        intent="review",
        text=_review_text(match, report),
        mood="thinking",
        params={"match_id": match_id, "report": asdict(report)},
    )


def _intent_from_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    games: list[dict],
    session: Any,
) -> ChatTurnResult:
    """Map one validated tool call to the intent contract (fail-soft on bad args).

    ``play_game`` 的偏好参数在这里按注册表校验：合法项进 ``params.config``，
    用户提了但本游戏不支持的进 ``params.config_ignored``（前端开局提示里
    明说，绝不静默）。**都不为真时不产生这两个键**——「玩月亮棋」这类
    零偏好请求的参数形状与旧版完全一致（既有断言不破）。
    """
    if name == "play_game":
        game_id = str(arguments.get("game_id", ""))
        game = next((g for g in games if g["game_id"] == game_id), None)
        if game is None:
            chips = [g["display_name"] for g in games[:8]]
            return ChatTurnResult(intent="clarify", text="想玩哪一款？", params={"chips": chips})
        config, ignored = _validated_play_config(game, arguments)
        params: dict[str, Any] = {"game_id": game_id}
        if config:
            params["config"] = config
        if ignored:
            params["config_ignored"] = ignored
        return ChatTurnResult(
            intent="play",
            text=f"好，来一局{game['display_name']}！对局正在创建…",
            mood="happy",
            params=params,
        )
    if name == "resume_session":
        if session is not None and not session.over:
            return ChatTurnResult(
                intent="resume",
                text=f"继续对局「{session.spec.display_name}」！",
                mood="happy",
                params={"game_id": session.game_id},
            )
        return ChatTurnResult(intent="chat", text="当前没有进行中的对局，想玩点什么？", params={})
    if name == "make_move":
        action = arguments.get("action")
        if session is None or session.over:
            return ChatTurnResult(intent="chat", text="当前没有可落子的对局。", params={})
        if not isinstance(action, dict):
            return ChatTurnResult(
                intent="clarify",
                text="这一步我没听懂，请直接点击棋盘/牌面操作，或说得更具体。",
                params={},
            )
        try:
            session.spec.parse_human_action(session, action)
        except Exception:
            return ChatTurnResult(
                intent="clarify",
                text="这一步当前不合法（看看我列出的合法动作），请直接点击棋盘/牌面操作。",
                mood="thinking",
                params={},
            )
        return ChatTurnResult(intent="move", text="好，走这步！", mood="happy", params={"action": action})
    # ask_hint / get_match_review / create_game 是本地工具（_execute_local_tool
    # 就地执行并携带 intent），不再走动作映射——这里不再有对应分支。
    if name == "restart_game":
        return ChatTurnResult(intent="restart", text=_FALLBACK_REPLIES["restart"], mood="neutral", params={})
    if name == "show_history":
        return ChatTurnResult(intent="history", text=_FALLBACK_REPLIES["history"], params={})
    if name == "create_game":
        # 兜底：create_game 是就地执行的本地工具，正常不会走到这里
        # （``_LOCAL_TOOLS`` 让它不进动作映射）；万一模型用错参数形状，
        # 仍然落到打开创建游戏页，而不是无声丢弃。
        return ChatTurnResult(intent="create", text=_FALLBACK_REPLIES["create"], params={})
    if name == "update_settings":
        # 正常不会走到：update_settings 是就地执行的本地工具（``_LOCAL_TOOLS``），
        # 由 ``_update_settings_result`` 按参数白名单执行。这里只为工具名/
        # 形状万一变化时留同一口径的 fail-soft 出口（绝不静默跳页）。
        return _update_settings_result(arguments)
    if name == "open_platform":
        return ChatTurnResult(intent="platform", text=_FALLBACK_REPLIES["platform"], params={})
    if name == "run_benchmark":
        return ChatTurnResult(intent="benchmark", text=_FALLBACK_REPLIES["benchmark"], params={})
    if name == "show_learning":
        return ChatTurnResult(intent="learning", text=_FALLBACK_REPLIES["learning"], params={})
    if name == "help":
        return ChatTurnResult(intent="help", text=_HELP_TEXT, params={})
    return ChatTurnResult(intent="chat", text="我在的，继续说？", params={})


# ── Deterministic fallback (no LLM) ───────────────────────────────


def _grid_cell_from_text(text: str, session: Any) -> dict | None:
    """Parse “下第X行第Y列 / 第N格 / 中间” into a grid action (best-effort)."""
    # P2-18 修复：spec.board_size 是单一事实来源（旧逻辑先查硬编码字典，
    # stochastic_gomoku 被按 15×15 解析 → “下第2行第3列”落到第 17 格、
    # 10-15 行被正则接受后再判非法）。
    size = getattr(session.spec, "board_size", None) or GRID_BOARD_LEN.get(session.game_id) or 9
    m = _GRID_MOVE_RE.search(text)
    if m:
        row, col = int(m.group(1)), int(m.group(2))
        if 1 <= row <= size and 1 <= col <= size:
            idx = (row - 1) * size + (col - 1)
            return {"cell_index": idx}
    m = _GRID_CELL_RE.search(text)
    if m:
        n = int(m.group(1))
        if 1 <= n <= size * size:
            return {"cell_index": n - 1}
    if _CENTER_RE.search(text) and size % 2 == 1:
        mid = size // 2
        return {"cell_index": mid * size + mid}
    return None


def _parse_battle_options(text: str) -> dict[str, Any]:
    """从一句话里抠开局偏好（无 LLM 兜底；与前端 ``parseBattleOptions`` 同表）.

    返回值就是 ``play_game`` 的**工具参数形状**，随后交给
    :func:`_validated_play_config` 按注册表校验 —— 无 LLM 路径与工具调用
    路径共用同一套校验与同一个 ``params.config`` 契约，不存在两套口径。
    只在措辞明确时命中；认不出的一律不填（默认开局），宁可少配不可误配。
    """
    out: dict[str, Any] = {}
    if not text:
        return out
    m = _BATTLE_COUNT_RE.search(text)
    if m:
        raw = m.group(1)
        count = int(raw) if raw.isdigit() else _CN_NUMERALS.get(raw)
        if count is not None:
            out["player_count"] = count
    for key, value, pattern in _BATTLE_OPTION_RULES:
        if pattern.search(text):
            out[key] = value
    return out


def _settings_applied_result(patch: dict[str, str]) -> ChatTurnResult:
    """已校验的偏好变更 → ``settings`` 意图（前端写档案 + 回执，不跳页）."""
    detail = "、".join(
        f"{_SETTING_FIELD_LABELS.get(profile_key, profile_key)}改成"
        f"「{_SETTING_VALUE_LABELS.get(profile_key, {}).get(value, value)}」"
        for profile_key, value in patch.items()
    )
    lines = [f"好，{detail} ✓"]
    if "default_persona" in patch:
        lines.append("以后我说话就按这个风格来。")
    return ChatTurnResult(
        intent="settings",
        text="\n".join(lines),
        mood="happy",
        params={"applied": patch, "chips": [_OPEN_SETTINGS_CHIP]},
    )


def _update_settings_result(arguments: dict[str, Any]) -> ChatTurnResult:
    """Execute ``update_settings``: 对话里说清的偏好 → 白名单校验过的变更.

    三个出口：``open_page``（用户明确要打开设置页）只切页面；给出了合法取值
    → ``settings`` + ``params.applied``（前端写档案并回执）；什么都没说清 →
    ``clarify`` + 选项 chips。

    「打开设置页」曾经是唯一出口：模型把「你能换个风格吗」也映射成它，于是页面
    跳走、话没回、设置一个没改——记在这里，别再退回那种行为。
    """
    if bool(arguments.get("open_page")):
        return ChatTurnResult(intent="settings", text=_FALLBACK_REPLIES["settings"], params={"open_page": True})
    patch: dict[str, str] = {}
    for arg_key, profile_key in _SETTING_ARG_FIELDS.items():
        value = arguments.get(arg_key)
        if isinstance(value, str) and value in _SETTING_VALUE_LABELS.get(profile_key, {}):
            patch[profile_key] = value
    if not patch:
        return ChatTurnResult(
            intent="clarify",
            text=_PREFERENCE_CLARIFY_TEXT,
            mood="thinking",
            params={"chips": [*_PERSONA_STYLE_CHIPS, _OPEN_SETTINGS_CHIP]},
        )
    return _settings_applied_result(patch)


def _preference_result(text: str) -> ChatTurnResult | None:
    """正则兜底：从一句话里抠出“改成哪种偏好”；与偏好无关的提问返回 ``None``.

    与前端 ``intents.ts::preferenceResult`` 同表同口径——断连时前端给出同一套
    chips 与同一份 ``applied`` 契约，两条路径的 UX 一致。
    """
    if _OPEN_SETTINGS_RE.search(text):
        return ChatTurnResult(intent="settings", text=_FALLBACK_REPLIES["settings"], params={"open_page": True})
    change = bool(_PREFERENCE_CHANGE_RE.search(text))
    persona_value = next((value for value, pattern in _PERSONA_VALUE_RULES if pattern.search(text)), "")
    theme_value = next((value for value, pattern in _THEME_VALUE_RULES if pattern.search(text)), "")
    wants_persona = (
        any(word in text for word in _PERSONA_WORDS)
        or bool(_PERSONA_ASK_RE.search(text))
        or (change and bool(persona_value))
    )
    wants_theme = any(word in text for word in _THEME_WORDS) or (change and bool(theme_value))
    if not wants_persona and not wants_theme:
        return None
    patch: dict[str, str] = {}
    if wants_persona and persona_value:
        patch["default_persona"] = persona_value
    if wants_theme and theme_value:
        patch["theme"] = theme_value
    if patch:
        return _settings_applied_result(patch)
    chips = [*_PERSONA_STYLE_CHIPS] if wants_persona else [*_THEME_CHIPS]
    chips.append(_OPEN_SETTINGS_CHIP)
    return ChatTurnResult(
        intent="clarify",
        text=_PREFERENCE_CLARIFY_TEXT,
        mood="thinking",
        params={"chips": chips},
    )


def fallback_intent(text: str, games: list[dict], session: Any) -> ChatTurnResult:
    """Deterministic regex routing used when the LLM is unavailable."""
    mood = _mood_for(text)
    game = _find_game(text, games)
    has_session = session is not None and not session.over

    # 「X 是什么/怎么玩/规则」→ 确定性知识回答（注册表 description +
    # 玩法文档规则段，零幻觉；必须先于 play —— “怎么下/怎么打”也含
    # 开局动词，语义却是问规则）。
    if _WHAT_IS_RE.search(text) and game is not None:
        desc = str(game.get("description") or "")
        name = game["display_name"]
        # description 多以名字开头（“3×3 经典月亮棋：…”），避免“月亮棋：3×3 经典月亮棋：…”式复读
        body = desc if (desc and name in desc) else (f"{name}：{desc}" if desc else f"{name}是平台支持的一款游戏。")
        rules_md = game_rules_text(game["game_id"])
        if rules_md:
            body += "\n" + rules_md
        body += "\n想试一试的话，说“玩" + game["display_name"] + "”即可开局。"
        return ChatTurnResult(
            intent="chat",
            text=body.strip(),
            mood="thinking",
            params={"game_id": game["game_id"], "chips": [f"玩{game['display_name']}"]},
        )

    # 对局中且是落子表达 → 只对 grid 族做最简解析（其余族请直接点击操作）
    if has_session and (session.family or _BUILTIN_FAMILY.get(session.game_id)) in ("grid", None):
        cell = _grid_cell_from_text(text, session)
        if cell is not None:
            try:
                session.spec.parse_human_action(session, cell)
                return ChatTurnResult(intent="move", text="好，走这步！", mood="happy", params={"action": cell})
            except Exception:
                return ChatTurnResult(intent="clarify", text="这个位置当前不能下，换个格试试。", mood=mood, params={})

    # 明确点名游戏 → play（优先级高于其它意图）。偏好同样在这里读：
    # 「三人局、困难、教学对局，玩谁是卧底」无需 LLM 也能落到 params.config
    # （与工具调用路径同一契约、同一校验）。
    if game is not None and _PLAY_RE.search(text):
        config, ignored = _validated_play_config(game, _parse_battle_options(text))
        params: dict[str, Any] = {"game_id": game["game_id"]}
        if config:
            params["config"] = config
        if ignored:
            params["config_ignored"] = ignored
        return ChatTurnResult(
            intent="play",
            text=f"好，来一局{game['display_name']}！对局正在创建…",
            mood="happy",
            params=params,
        )
    if _HINT_RE.search(text) and has_session:
        return ChatTurnResult(intent="hint", text=_FALLBACK_REPLIES["hint"], mood="thinking", params={})
    if _PLATFORM_RE.search(text):
        return ChatTurnResult(intent="platform", text=_FALLBACK_REPLIES["platform"], params={})
    if _CREATE_RE.search(text):
        return ChatTurnResult(intent="create", text=_FALLBACK_REPLIES["create"], params={})
    if _REVIEW_RE.search(text):
        return ChatTurnResult(intent="review", text=_FALLBACK_REPLIES["review"], params={})
    if _HISTORY_RE.search(text):
        return ChatTurnResult(intent="history", text=_FALLBACK_REPLIES["history"], params={})
    # 「换风格/改设置」：能识别取值就直接改（settings + params.applied），
    # 只说要换没说成哪种就给选项；只有明确“打开设置页”才切页面。
    preference = _preference_result(text)
    if preference is not None:
        return preference
    if _SETTINGS_RE.search(text):
        return ChatTurnResult(intent="settings", text=_FALLBACK_REPLIES["settings"], params={"open_page": True})
    if _BENCHMARK_RE.search(text):
        return ChatTurnResult(intent="benchmark", text=_FALLBACK_REPLIES["benchmark"], params={})
    if _LEARNING_RE.search(text):
        return ChatTurnResult(intent="learning", text=_FALLBACK_REPLIES["learning"], params={})
    if has_session and _RESTART_RE.search(text):
        return ChatTurnResult(
            intent="restart",
            text=_FALLBACK_REPLIES["restart"],
            params={"game_id": session.game_id},
        )
    if _RESUME_RE.search(text):
        if has_session:
            return ChatTurnResult(
                intent="resume",
                text=f"继续对局「{session.spec.display_name}」！",
                mood="happy",
                params={"game_id": session.game_id},
            )
        return ChatTurnResult(intent="chat", text="当前没有进行中的对局，想玩点什么？", params={})
    # 具体平台功能提问（未命中上面任一动作意图）→ 主题文档确定性回答。
    # “怎么改难度”“教学对局是什么”“视觉识别怎么用”：旧逻辑落到泛泛
    # 默认 chat，现在给对应主题的权威帮助（与 get_platform_help 同源，
    # 单一事实来源）。“你能做什么”这类泛泛问不命中主题，走原 _HELP_TEXT。
    topic = match_platform_topic(text)
    if topic is not None:
        body = platform_help_text(topic)
        if body:
            return ChatTurnResult(
                intent="help",
                text=body,
                mood="thinking",
                params={"topic": topic},
            )
    if _HELP_RE.search(text):
        return ChatTurnResult(intent="help", text=_HELP_TEXT, params={})
    if _PLAY_RE.search(text):
        chips = [g["display_name"] for g in games[:8]]
        return ChatTurnResult(intent="clarify", text="想玩哪一款？", mood=mood, params={"chips": chips})
    return ChatTurnResult(
        intent="chat",
        text="我在的，你可以试试：“玩月亮棋”“看战绩”“这步怎么走”…",
        mood=mood,
        params={},
    )


# ── Conversation history ──────────────────────────────────────────


def _sanitize_history(history: Any) -> list[dict[str, str]]:
    """Validate and bound client-supplied conversation history (fail-soft).

    Keeps only ``user``/``assistant`` turns with non-empty text, newest
    last, capped by ``_HISTORY_MAX_MESSAGES`` and ``_HISTORY_MAX_CHARS``.
    The system prompt is rebuilt by the backend every turn (live session
    context) and is never taken from the client.
    """
    if not isinstance(history, list):
        return []
    kept: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in _HISTORY_ROLES or not isinstance(content, str) or not content.strip():
            continue
        kept.append({"role": str(role), "content": content.strip()})
    kept = kept[-_HISTORY_MAX_MESSAGES:]
    total = 0
    trimmed: list[dict[str, str]] = []
    for msg in reversed(kept):
        total += len(msg["content"])
        if total > _HISTORY_MAX_CHARS and trimmed:
            break
        trimmed.append(msg)
    trimmed.reverse()
    return trimmed


# ── Main entry ────────────────────────────────────────────────────


def _prepare(
    manager: PlayManager,
    text: str,
    *,
    game_id: str | None = None,
    custom: CustomGameRegistry | None = None,
    match_history: Any = None,
) -> tuple[str, Any, list[dict], list[dict], dict | None, Persona]:
    """聊天回合共享前置：清洗文本、定位 session、收集目录/活跃会话/最近对局、
    解析全局人设.

    Returns:
        ``(text, session, games, active, latest, persona)`` —— ``chat_turn``
        与 ``chat_turn_stream`` 共用，保证两个出口的上下文与人设一致。
        persona 取全局默认（profile → gentle 兜底），平台助手据此注入身份块。
    """
    text = (text or "").strip()
    session = None
    if game_id:
        try:
            session = manager.get(str(game_id))
        except Exception:
            session = None
    games = _collect_games(custom)
    active = manager.active_sessions()[:16]
    # 终局后 session 已移除 → 用最近一局补 system prompt 上下文。
    latest = _latest_match(match_history) if session is None or session.over else None
    persona = PERSONAS.get(manager.default_persona()) or PERSONAS["gentle"]
    return text, session, games, active, latest, persona


def _assemble_llm_prompt(
    games: list[dict],
    session: Any,
    active: list[dict],
    latest: dict | None,
    history: list[dict[str, Any]] | None,
    text: str,
    *,
    persona: Persona | None = None,
) -> tuple[list[dict], list[dict[str, Any]]]:
    """拼 LLM 一回合的 ``(tools, messages)``（system 每轮现构，客户端 history 白名单清洗）。"""
    tools = build_tools(games=games, session=session, active=active)
    system = _system_prompt(games, session, active, latest=latest, persona=persona)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(_sanitize_history(history))
    messages.append({"role": "user", "content": text})
    return tools, messages


def _stream_events(result: ChatTurnResult) -> list[dict[str, Any]]:
    """把最终结果转成 SSE 事件序列（``intent`` + ``done``）。"""
    return [
        {"event": "intent", "data": asdict(result)},
        {"event": "done", "data": {}},
    ]


def _chat_turn_core(
    manager: PlayManager,
    text: str,
    *,
    llm: LLMClient | None = None,
    game_id: str | None = None,
    custom: CustomGameRegistry | None = None,
    history: list[dict[str, Any]] | None = None,
    match_history: Any = None,
) -> ChatTurnResult:
    """Turn one user message into a validated platform intent (fail-soft).

    ``history`` — prior ``user``/``assistant`` turns (newest last) sent by
    the frontend so the LLM sees the conversation context; sanitized and
    bounded here (``_sanitize_history``). Only used on the LLM path; the
    deterministic regex fallback routes on the current sentence alone.

    ``match_history`` — the platform :class:`MatchHistory`; backs the
    ``get_match_review`` info tool and the “最近一局” system-prompt line
    (post-match context instead of a blank slate once the session is
    removed from the registry).
    """
    text, session, games, active, latest, persona = _prepare(
        manager, text, game_id=game_id, custom=custom, match_history=match_history
    )
    if not text:
        return ChatTurnResult(intent="chat", text=_HELP_TEXT, params={})

    if llm is not None:
        tools, messages = _assemble_llm_prompt(games, session, active, latest, history, text, persona=persona)
        # Bounded tool loop: info tools are executed locally and their
        # result is fed back as a ``role: "tool"`` message so the model
        # can answer from authoritative data; action tools map to an
        # intent immediately.  ``last_tool_result`` doubles as the
        # deterministic answer if the budget runs out (fail-soft, zero
        # hallucination) or the model keeps requesting tools.
        #
        # 并行 tool_calls（audit §5-3）：一个回合里模型可能同时发起
        # 多个调用。信息类逐个就地执行、逐个以 ``role:"tool"`` 回传
        # （按 tool_call_id 成对关联）；动作类只取首个——intent 契约是
        # 单动作，「介绍一下血战到底然后来一局」这类复合请求里动作
        # 立即生效，其余调用不静默丢失语义。混合批次（信息 + 动作）
        # 时动作优先返回。
        #
        # 携带 intent 的信息工具（ask_hint → hint、get_match_review →
        # review）：模型成文后落在携带的 intent 上（文本换成模型亲笔），
        # 预算耗尽则直接用工具结果文本——两个出口都指向同一个前端
        # 意图，绝不再退回纯 chat 丢掉 hint/report 参数。
        last_tool_result = ""
        carried: ChatTurnResult | None = None
        call_seq = 0  # 端点未给 id 时的合成 tool_call_id 计数
        for _ in range(_MAX_TOOL_ROUNDS):
            try:
                reply = llm.complete_tools(messages, tools)
            except Exception:
                reply = None
            if reply is None:
                break
            if not reply.tool_calls:
                if reply.text:
                    if carried is None:
                        return ChatTurnResult(intent="chat", text=reply.text.strip(), mood=_mood_for(text), params={})
                    carried.text = reply.text.strip()
                    return carried
                break
            action = next((c for c in reply.tool_calls if str(c.name) not in _LOCAL_TOOLS), None)
            if action is not None:
                args = action.arguments if isinstance(action.arguments, dict) else {}
                result = _intent_from_tool(str(action.name), args, games=games, session=session)
                if reply.text:
                    result.text = reply.text.strip()  # 优先采用模型亲笔文案，兜底文案作后备
                return result
            tool_calls_payload: list[dict[str, Any]] = []
            results: list[str] = []
            created = False  # create_game 有副作用：同回合至多执行一次
            for call in reply.tool_calls:
                args = call.arguments if isinstance(call.arguments, dict) else {}
                call_seq += 1
                call_id = call.id or f"call_{call_seq}"
                name = str(call.name)
                allow_create = not created
                executed = _execute_local_tool(
                    name,
                    args,
                    games=games,
                    session=session,
                    manager=manager,
                    match_history=match_history,
                    custom=custom,
                    allow_create=allow_create,
                )
                if name == "create_game" and allow_create and executed.intent == "create":
                    created = True
                if executed.intent != "chat" or executed.params:
                    carried = executed  # 多个携带意图时最后一个生效（同回合并发 hint+review 无实际语义）
                results.append(executed.text)
                tool_calls_payload.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                    }
                )
            messages.append({"role": "assistant", "content": reply.text or "", "tool_calls": tool_calls_payload})
            for payload, result_text in zip(tool_calls_payload, results):
                messages.append({"role": "tool", "tool_call_id": payload["id"], "content": result_text})
            non_empty = [r for r in results if r]
            if non_empty:
                last_tool_result = "\n\n".join(non_empty)
        if carried is not None and (carried.text or last_tool_result):
            carried.text = (carried.text or last_tool_result).strip()[:1000]
            return carried
        if last_tool_result:
            return ChatTurnResult(
                intent="chat",
                text=last_tool_result.strip()[:800],
                mood="thinking",
                params={},
            )
    if llm is not None:
        # LLM 已装配（探测通过）但本回合无有效输出 → 走正则兜底。
        # 传输故障由 LLMClient 内记录并告警；这里补一条回合级日志，
        # 让「模型在线但持续不可用」可观测（不静默降级）。
        last = getattr(llm, "last_error", None)
        if last is not None:
            logger.warning("LLM 聊天回合失败: %s — 走正则兜底", last)
        elif carried is None:
            logger.warning("LLM 聊天回合未产出可用结果（空回复/工具循环耗尽）— 走正则兜底")
    return fallback_intent(text, games, session)


def chat_turn(
    manager: PlayManager,
    text: str,
    *,
    llm: LLMClient | None = None,
    game_id: str | None = None,
    custom: CustomGameRegistry | None = None,
    history: list[dict[str, Any]] | None = None,
    match_history: Any = None,
) -> ChatTurnResult:
    """Turn one user message into a validated platform intent (fail-soft).

    ``_chat_turn_core`` 之上包一层**隐藏信息后置扫描**（:func:`_guard_text`）：
    对手/教学模式的聊天正文不得泄露玩家隐藏牌或 AI 自己的具体牌面——
    2026-08 对局记录里对手模式聊天直接报「你现在手里是方块9和红桃3」、
    「我手里拿着黑桃K和黑桃3」，这次在出口统一收口。
    """
    result = _chat_turn_core(
        manager,
        text,
        llm=llm,
        game_id=game_id,
        custom=custom,
        history=history,
        match_history=match_history,
    )
    session = None
    if game_id:
        try:
            session = manager.get(str(game_id))
        except Exception:
            session = None
    return _finalize_chat_result(result, session)


def _chat_turn_stream_core(
    manager: PlayManager,
    text: str,
    *,
    llm: LLMClient | None = None,
    game_id: str | None = None,
    custom: CustomGameRegistry | None = None,
    history: list[dict[str, Any]] | None = None,
    match_history: Any = None,
) -> Iterator[dict[str, Any]]:
    """SSE 事件生成器 — ``chat_turn`` 的流式出口（``/api/chat`` 流式模式）.

    与 :func:`chat_turn` 共用 ``_prepare`` / ``_assemble_llm_prompt`` /
    工具循环语义，区别只在：LLM 走 ``complete_stream``，正文与思维链以
    增量事件上浮，最终意图以 ``intent`` 事件收口。

    事件契约（前端 ``chatTurnStream`` 消费）::

        event: reasoning   data: {"delta": str}   # 思维链增量（上限 _REASONING_MAX_CHARS）
        event: text        data: {"delta": str}   # 回复正文增量
        event: intent      data: ChatTurnResult   # 最终意图（text 为全量回复）
        event: error       data: {"error": str}   # 流中失败（fail-soft；随后仍给 intent 兜底）
        event: done        data: {}               # 结束

    失败语义与 JSON 模式一致：流中途传输失败 → ``error`` 事件 + 正则兜底
    的 ``intent`` 事件（已流出的增量由前端决定去留）；``llm=None`` /
    无 LLM 直接产出兜底 ``intent``（不产生任何增量）。
    """
    text, session, games, active, latest, persona = _prepare(
        manager, text, game_id=game_id, custom=custom, match_history=match_history
    )
    if not text:
        yield from _stream_events(ChatTurnResult(intent="chat", text=_HELP_TEXT, params={}))
        return
    if llm is None:
        yield from _stream_events(fallback_intent(text, games, session))
        return
    tools, messages = _assemble_llm_prompt(games, session, active, latest, history, text, persona=persona)
    last_tool_result = ""
    carried: ChatTurnResult | None = None
    call_seq = 0  # 端点未给 id 时的合成 tool_call_id 计数
    total_reasoning = 0
    for _ in range(_MAX_TOOL_ROUNDS):
        round_text: list[str] = []
        tool_calls: list[Any] = []
        failed = False
        fail_message = ""
        try:
            for chunk in llm.complete_stream(messages, tools=tools):
                if chunk.error:
                    failed = True
                    fail_message = chunk.error
                    break
                if chunk.reasoning:
                    total_reasoning += len(chunk.reasoning)
                    if total_reasoning <= _REASONING_MAX_CHARS:
                        yield {"event": "reasoning", "data": {"delta": chunk.reasoning}}
                if chunk.text:
                    round_text.append(chunk.text)
                    yield {"event": "text", "data": {"delta": chunk.text}}
                if chunk.done:
                    tool_calls = chunk.tool_calls
                    break
        except Exception:
            failed = True
        if failed:
            # 流中失败：真实原因在 client.last_error（或 fail_hard 抛出）；
            # 无 last_error 时用错误块自带信息；再没有走通用文案。
            last = getattr(llm, "last_error", None)
            if last is not None:
                message = str(last)
            elif fail_message:
                message = fail_message
            else:
                message = "LLM 流式调用中断"
            logger.warning("LLM 流式聊天回合失败: %s — 走正则兜底", message)
            yield {"event": "error", "data": {"error": message}}
            yield from _stream_events(fallback_intent(text, games, session))
            return
        if not tool_calls:
            # 纯文本终态：模型亲笔增量即最终回复；携带 intent（hint/review）
            # 时落在该 intent 上，绝不退回纯 chat 丢掉参数。
            full = "".join(round_text).strip()
            if carried is not None:
                carried.text = full if full else carried.text
                yield from _stream_events(carried)
                return
            if full:
                yield from _stream_events(ChatTurnResult(intent="chat", text=full, mood=_mood_for(text), params={}))
                return
            break  # 无文本无工具 → 走兜底（与 JSON 模式空回复语义一致）
        action = next((c for c in tool_calls if str(c.name) not in _LOCAL_TOOLS), None)
        if action is not None:
            args = action.arguments if isinstance(action.arguments, dict) else {}
            result = _intent_from_tool(str(action.name), args, games=games, session=session)
            model_text = "".join(round_text).strip()
            if model_text:
                result.text = model_text  # 优先采用模型亲笔文案（增量已在 text 事件发出）
            yield from _stream_events(result)
            return
        # 信息工具就地执行、逐个以 role:"tool" 回传（并行 tool_calls 语义与 chat_turn 一致）。
        tool_calls_payload: list[dict[str, Any]] = []
        results: list[str] = []
        created = False  # create_game 有副作用：同回合至多执行一次
        for call in tool_calls:
            args = call.arguments if isinstance(call.arguments, dict) else {}
            call_seq += 1
            call_id = call.id or f"call_{call_seq}"
            name = str(call.name)
            allow_create = not created
            executed = _execute_local_tool(
                name,
                args,
                games=games,
                session=session,
                manager=manager,
                match_history=match_history,
                custom=custom,
                allow_create=allow_create,
            )
            if name == "create_game" and allow_create and executed.intent == "create":
                created = True
            if executed.intent != "chat" or executed.params:
                carried = executed  # 多个携带意图时最后一个生效（同回合并发 hint+review 无实际语义）
            results.append(executed.text)
            tool_calls_payload.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            )
        messages.append({"role": "assistant", "content": "".join(round_text), "tool_calls": tool_calls_payload})
        for payload, result_text in zip(tool_calls_payload, results):
            messages.append({"role": "tool", "tool_call_id": payload["id"], "content": result_text})
        non_empty = [r for r in results if r]
        if non_empty:
            last_tool_result = "\n\n".join(non_empty)
    # 工具循环预算耗尽 / 空回复：与 chat_turn 相同的 fail-soft 收口。
    if carried is not None and (carried.text or last_tool_result):
        carried.text = (carried.text or last_tool_result).strip()[:1000]
        yield from _stream_events(carried)
        return
    if last_tool_result:
        yield from _stream_events(
            ChatTurnResult(
                intent="chat",
                text=last_tool_result.strip()[:800],
                mood="thinking",
                params={},
            )
        )
        return
    yield from _stream_events(fallback_intent(text, games, session))


def chat_turn_stream(
    manager: PlayManager,
    text: str,
    *,
    llm: LLMClient | None = None,
    game_id: str | None = None,
    custom: CustomGameRegistry | None = None,
    history: list[dict[str, Any]] | None = None,
    match_history: Any = None,
) -> Iterator[dict[str, Any]]:
    """SSE 事件生成器 — ``chat_turn`` 的流式出口（``/api/chat`` 流式模式）.

    ``_chat_turn_stream_core`` 之上包一层**隐藏信息后置扫描**：``intent``
    事件的最终正文按陪伴身份过 :func:`_guard_text`（对手/教学两态收口），
    增量 ``text``/``reasoning`` 事件原样上浮——前端以 ``intent`` 事件的全量
    文本为最终回复，泄露句在收口时被改写。
    """
    session = None
    if game_id:
        try:
            session = manager.get(str(game_id))
        except Exception:
            session = None
    for event in _chat_turn_stream_core(
        manager,
        text,
        llm=llm,
        game_id=game_id,
        custom=custom,
        history=history,
        match_history=match_history,
    ):
        if event["event"] == "intent" and isinstance(event.get("data"), dict):
            data = dict(event["data"])
            data["text"] = _guard_text(str(data.get("text", "")), session)
            event = {"event": "intent", "data": data}
        yield event
