"""Agent-style chat orchestrator — the *chat-first* backend of the platform.

One sentence from the user becomes one platform action.  ``chat_turn``
runs LLM function calling (unified ``LLMClient``, OpenAI-compatible
``tools`` schema) against a tool set built from the live registry and the
current session, validates the tool arguments against the authoritative
engine contract, and always fails soft to a deterministic regex fallback
when the LLM is missing.

Intent contract (shared with the frontend ``ChatPage`` / ``useChatRuntime``):

=========  ========================================================
intent     params
=========  ========================================================
play       ``{game_id}``            → 前端开新对局
resume     ``{game_id}``            → 前端恢复活跃会话
move       ``{action}``             → 前端调 ``/match/move``
hint       ``{level}``              → 前端调 ``/match/hint``
restart    ``{}``                   → 前端重开当前对局
history    ``{}``                   → 前端展示战绩
review     ``{}``                   → 前端展示复盘
create     ``{}``                   → 前端展示创建游戏面板
settings   ``{}``                   → 前端展示设置
platform   ``{}``                   → 前端切回完整平台界面
benchmark  ``{}``                   → 前端展示评测中心
learning   ``{}``                   → 前端展示在线学习
help       ``{}``                   → 前端展示帮助
chat       ``{}``                   → 普通聊天回复（无工具动作）
clarify    ``{chips?: [str]}``      → 追问（附可点选项）
=========  ========================================================

Fail-soft rule: this module never raises for a misbehaving model — a
missing/empty LLM reply, an unknown tool name, or an invalid action all
land on a clarifying or canned reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from layer2_engine.core.llm import LLMClient

from .custom_games import CustomGameRegistry
from .games import GAMES
from .session import _BUILTIN_FAMILY, PlayManager

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

#: 历史消息角色白名单 —— system 由后端每轮现构（含实时对局上下文），
#: 绝不采信客户端传入的 system 角色。
_HISTORY_ROLES = ("user", "assistant")

#: 各规则族 make_move 的 action 形态（写入工具描述，帮助模型产出可校验参数）。
_ACTION_SHAPES = {
    "grid": '{"cell_index": 数字}  —— 0 基空格索引（描述里会给出当前可落子格）',
    "poker": '{"choice": "call"|"fold"|"raise"|"all_in", "amount": 数字 —— amount 仅 raise 时必填}',
    "mahjong": '{"type": "discard"|"chow"|"pong"|"kong"|"win", "tile": "牌 id"}  —— 只能用描述里列出的合法组合',
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
    "create": "创建游戏面板已为你展开 👇",
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
    "· “创建一个新游戏” —— 用自然语言写规则\n"
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
_CREATE_RE = re.compile(r"(?:创建|新建|自定义|设计一?个新?游戏)")
_SETTINGS_RE = re.compile(r"(?:设置|性格|声音|主题|偏好|选项)")
_PLATFORM_RE = re.compile(r"(?:平台界面|完整界面|平台模式|打开平台|回去|回平台)")
_BENCHMARK_RE = re.compile(r"(?:评测|benchmark|模拟对局|求解器对比)")
_LEARNING_RE = re.compile(r"(?:在线学习|学习状态|自动学习)")
_HELP_RE = re.compile(r"(?:帮助|能做什么|怎么用|你有什么功能|你会什么)")
_GRID_MOVE_RE = re.compile(r"(?:下|放|走)(?:第)?(\d{1,2})\s*行\s*(?:第)?(\d{1,2})\s*列")
_GRID_CELL_RE = re.compile(r"(?:下|放|走)\s*(?:第)?(\d{1,2})\s*(?:格|格位置|个空位)")
_CENTER_RE = re.compile(r"(?:中间|正中|中心)")

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
    """Built-in + custom catalog for the ``play_game`` tool and fallback."""
    games: list[dict] = [
        {
            "game_id": spec.game_id,
            "display_name": spec.display_name,
            "kind": spec.kind,
            "family": _BUILTIN_FAMILY.get(spec.game_id),
        }
        for spec in GAMES.values()
    ]
    if custom is not None:
        for entry in custom.list_games():
            games.append(
                {
                    "game_id": str(entry.get("game_id", "")),
                    "display_name": str(entry.get("display_name") or entry.get("game_id", "")),
                    "kind": "board",
                    "family": entry.get("family"),
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


def _legal_context(session: Any) -> str:
    """Condensed, *already-projected* legal context for the model (hidden info red line)."""
    try:
        snap = session.snapshot()
    except Exception:
        return ""
    parts: list[str] = []
    if snap.get("over"):
        parts.append(f"本局已结束，胜方: {snap.get('winner') or '未知'}")
        return "；".join(parts)
    for key in ("legal", "legal_options", "legal_actions", "choices"):
        val = snap.get(key)
        if isinstance(val, list) and val:
            parts.append("合法动作: " + json.dumps(val, ensure_ascii=False)[:600])
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


def _system_prompt(games: list[dict], session: Any, active: list[dict]) -> str:
    lines = [
        "你是 Gavis 平台的对话助手（agent 聊天模式）。用户一句话 → 你选一个工具。",
        "规则：",
        "1. 用户没指明玩哪个游戏时，不要调用 play_game，直接回复询问（intent clarify）。",
        "2. 对局中的落子/发言只能用描述里给出的合法动作；含糊的话不调用动作工具，直接聊天。",
        "3. 隐藏信息红线：绝不编造其他玩家手牌/身份/棋局评估——依据用户输入和给出的合法动作行事。",
    ]
    if games:
        names = "、".join(f"{g['display_name']}({g['game_id']})" for g in games[:24])
        lines.append(f"可用游戏: {names}")
    if session is not None:
        name = session.spec.display_name
        ctx = _legal_context(session)
        lines.append(f"当前对局: {name}（{session.game_id}）")
        if ctx:
            lines.append(ctx)
        if session.persona:
            lines.append(f"陪伴角色: {session.persona}")
    if active:
        active_names = "、".join(f"{a.get('display_name') or a.get('game_id')}" for a in active[:8])
        lines.append(f"进行中的对局: {active_names}（用户说“继续/恢复”时用 resume_session）")
    return "\n".join(lines)


def build_tools(*, games: list[dict], session: Any, active: list[dict]) -> list[dict]:
    """Build the OpenAI ``tools`` list for the current context."""
    game_enum = [g["game_id"] for g in games if g["game_id"]]
    tools: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "play_game",
                "description": "用户想玩某款游戏 / 开局 / 对战。游戏没指明时不要调用。",
                "parameters": {
                    "type": "object",
                    "properties": {"game_id": {"type": "string", "enum": game_enum}},
                    "required": ["game_id"],
                },
            },
        }
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
                    "name": "ask_hint",
                    "description": "用户要提示/指导/“这步怎么走”时调用。",
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
        ("review_match", "用户想复盘/回放上一局时调用"),
        ("create_game", "用户想创建/自定义新游戏时调用"),
        ("update_settings", "用户想改设置/性格/主题时调用"),
        ("open_platform", "用户想回到完整平台界面时调用"),
        ("run_benchmark", "用户想看评测/求解器对比时调用"),
        ("show_learning", "用户想看在线学习状态时调用"),
        ("help", "用户问你能做什么/怎么用/帮助时调用"),
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
    """Best-effort game match by display_name / game_id substring."""
    best: dict | None = None
    best_len = 0
    for g in games:
        for key in ("display_name", "game_id"):
            name = str(g.get(key, ""))
            if name and name in text and len(name) > best_len:
                best = g
                best_len = len(name)
    return best


def _intent_from_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    games: list[dict],
    session: Any,
) -> ChatTurnResult:
    """Map one validated tool call to the intent contract (fail-soft on bad args)."""
    if name == "play_game":
        game_id = str(arguments.get("game_id", ""))
        game = next((g for g in games if g["game_id"] == game_id), None)
        if game is None:
            chips = [g["display_name"] for g in games[:8]]
            return ChatTurnResult(intent="clarify", text="想玩哪一款？", params={"chips": chips})
        return ChatTurnResult(
            intent="play",
            text=f"好，来一局{game['display_name']}！对局正在创建…",
            mood="happy",
            params={"game_id": game_id},
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
    if name == "ask_hint":
        return ChatTurnResult(intent="hint", text=_FALLBACK_REPLIES["hint"], mood="thinking", params={})
    if name == "restart_game":
        return ChatTurnResult(intent="restart", text=_FALLBACK_REPLIES["restart"], mood="neutral", params={})
    if name == "show_history":
        return ChatTurnResult(intent="history", text=_FALLBACK_REPLIES["history"], params={})
    if name == "review_match":
        return ChatTurnResult(intent="review", text=_FALLBACK_REPLIES["review"], params={})
    if name == "create_game":
        return ChatTurnResult(intent="create", text=_FALLBACK_REPLIES["create"], params={})
    if name == "update_settings":
        return ChatTurnResult(intent="settings", text=_FALLBACK_REPLIES["settings"], params={})
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


def fallback_intent(text: str, games: list[dict], session: Any) -> ChatTurnResult:
    """Deterministic regex routing used when the LLM is unavailable."""
    mood = _mood_for(text)
    game = _find_game(text, games)
    has_session = session is not None and not session.over

    # 对局中且是落子表达 → 只对 grid 族做最简解析（其余族请直接点击操作）
    if has_session and (session.family or _BUILTIN_FAMILY.get(session.game_id)) in ("grid", None):
        cell = _grid_cell_from_text(text, session)
        if cell is not None:
            try:
                session.spec.parse_human_action(session, cell)
                return ChatTurnResult(intent="move", text="好，走这步！", mood="happy", params={"action": cell})
            except Exception:
                return ChatTurnResult(intent="clarify", text="这个位置当前不能下，换个格试试。", mood=mood, params={})

    # 明确点名游戏 → play（优先级高于其它意图）
    if game is not None and _PLAY_RE.search(text):
        return ChatTurnResult(
            intent="play",
            text=f"好，来一局{game['display_name']}！对局正在创建…",
            mood="happy",
            params={"game_id": game["game_id"]},
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
    if _SETTINGS_RE.search(text):
        return ChatTurnResult(intent="settings", text=_FALLBACK_REPLIES["settings"], params={})
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


def chat_turn(
    manager: PlayManager,
    text: str,
    *,
    llm: LLMClient | None = None,
    game_id: str | None = None,
    custom: CustomGameRegistry | None = None,
    history: list[dict[str, Any]] | None = None,
) -> ChatTurnResult:
    """Turn one user message into a validated platform intent (fail-soft).

    ``history`` — prior ``user``/``assistant`` turns (newest last) sent by
    the frontend so the LLM sees the conversation context; sanitized and
    bounded here (``_sanitize_history``). Only used on the LLM path; the
    deterministic regex fallback routes on the current sentence alone.
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
    if not text:
        return ChatTurnResult(intent="chat", text=_HELP_TEXT, params={})

    if llm is not None:
        tools = build_tools(games=games, session=session, active=active)
        system = _system_prompt(games, session, active)
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(_sanitize_history(history))
        messages.append({"role": "user", "content": text})
        try:
            reply = llm.complete_tools(messages, tools)
        except Exception:
            reply = None
        if reply is not None and reply.tool_calls:
            call = reply.tool_calls[0]
            args = call.arguments if isinstance(call.arguments, dict) else {}
            result = _intent_from_tool(str(call.name), args, games=games, session=session)
            if reply.text:
                result.text = reply.text.strip()  # 优先采用模型亲笔文案，兜底文案作后备
            return result
        if reply is not None and reply.text:
            return ChatTurnResult(intent="chat", text=reply.text.strip(), mood=_mood_for(text), params={})
    return fallback_intent(text, games, session)
