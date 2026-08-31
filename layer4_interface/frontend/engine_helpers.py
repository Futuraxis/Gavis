"""Frontend assembly + display helpers (v5.2).

Game-agnostic assembly lives in Layer 4 (the user contract: nothing
game-specialized below the rules JSON / the frontend).  This module is
the single frontend home for:

  - rules loading and bare ``GameEngine`` construction (variants /
    player counts are declared inside each game's JSON)
  - ``resolve_all_chance`` — advance through pending chance nodes via
    the generic engine protocol
  - Texas / Mahjong *display* helpers (hand name, tile name) that
    evaluate rules-declared aliases through the engine's generic
    ``eval_expr``, plus the seat ids declared in ``rules/texas_holdem.json``
  - the **自然语言名称层** (natural-name layer): game piece ids
    (``s1`` / ``r7a`` / ``H14`` / ``wolf``) and engine canonical keys
    (``discard:s1`` / ``act:call:2``) → 中文称呼（一条 / 红7 / 红桃K /
    狼人 / 打出 一条 / 跟注 2）。凡是**直面 LLM 的文本**（对话引擎、
    技能提示、chat 信息工具、历史/复盘动作描述、狼人杀 LLM 提示）
    都必须经这一层，机器契约（快照 id、canonicalKey、工具参数形状）
    保持原样不动 —— 这是“传给 LLM 的信息不过分技术化”的单一出口。

``platform/games.py`` (GameSpec registry) and the standalone ``play_*``
apps import from here; no per-game adapter class exists in Layer 2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"

# Seats declared in ``rules/texas_holdem.json`` (display/assembly use).
TEXAS_SEATS = ("p_sb", "p_bb")

_TEXAS_HAND_NAMES = {
    0: "高牌",
    1: "一对",
    2: "两对",
    3: "三条",
    4: "顺子",
    5: "同花",
    6: "葫芦",
    7: "四条",
    8: "同花顺",
}

# ── 自然语言名称层 ─────────────────────────────────────────────────

#: 1-9 的中文读法（麻将一条…九条 / 一万…九万 / 一筒…九筒共用）。
_CN_NUMERALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九")
_MAHJONG_TILE_NAMES = {"m": "万", "p": "筒", "s": "条", "z": "字"}
_MAHJONG_Z_NAMES = {"1": "东", "2": "南", "3": "西", "4": "北", "5": "中", "6": "发", "7": "白"}
#: 麻将牌 id 的规整形态（m/p/s/z + 1-9），用于从残缺文本里提取牌 id。
_MAHJONG_TILE_ID_RE = re.compile(r"[mpsz][1-9]")

_UNO_COLOR_NAMES = {"r": "红", "b": "蓝", "g": "绿", "y": "黄"}
_UNO_SYMBOL_NAMES = {"s": "禁止", "r": "反转", "d": "+2"}
_POKER_SUIT_NAMES = {"s": "黑桃", "h": "红桃", "d": "方块", "c": "梅花"}
_POKER_RANK_NAMES = {"T": "10"}
_POKER_CHOICE_NAMES = {"call": "跟注", "fold": "弃牌", "raise": "加注", "all_in": "全下", "check": "过牌"}
_SOCIAL_ROLE_NAMES = {
    "wolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
    "civilian": "平民",
    "undercover": "卧底",
}

_GRID_CELL_RE = re.compile(r"cell_(\d+)_(\d+)")
_GRID_ROWCOL_RE = re.compile(r"(\d+),(\d+)$")


def game_family(game_id: str) -> str:
    """Rule-family label for an id: grid / poker / mahjong / uno / social.

    仅作展示/读法分派：内置游戏按注册表约定映射；自定义游戏按 id 前缀
    推断（``mahjong_*`` / ``uno_*``）；识别不了返回 ``"unknown"``
    （上层拿到 unknown 时按原文直出，不抛错）。
    """
    gid = str(game_id or "")
    if gid in ("moon_chess", "stochastic_gomoku"):
        return "grid"
    if gid == "texas_holdem":
        return "poker"
    if gid.startswith("mahjong"):
        return "mahjong"
    if gid == "uno" or gid.startswith("uno_"):
        return "uno"
    if gid in ("werewolf", "undercover") or "werewolf" in gid or "undercover" in gid:
        return "social"
    return "unknown"


def mahjong_tile_name(tile: str) -> str:
    """Chinese tile label: ``m3`` → '三万', ``s1`` → '一条', ``z5`` → '红中'.

    1-9 用中文数字读法（“一条/二条…”，不是“1条/2条…”）：麻将牌的
    点数名称本来就该这么写 —— 前端牌面与 LLM 直面文本统一走本函数。
    """
    if not tile or len(tile) < 2:
        return str(tile)
    suit, rank = tile[0], tile[1:]
    if suit == "z":
        return _MAHJONG_Z_NAMES.get(rank, tile)
    if suit not in _MAHJONG_TILE_NAMES:
        return str(tile)  # 未知花色（如 "joker"）原样兜底，不做拼接
    if rank.isdigit() and 1 <= int(rank) <= 9:
        numeral = _CN_NUMERALS[int(rank) - 1]
    else:
        numeral = rank
    return f"{numeral}{_MAHJONG_TILE_NAMES[suit]}"


def uno_card_name(card: str) -> str:
    """Chinese card label: ``r7a`` → '红7', ``gsa`` → '绿禁止', ``wild_1`` → '万能', ``wild4_1`` → '+4 万能'."""
    if not card:
        return str(card)
    if card.startswith("wild4"):
        return "+4 万能"
    if card.startswith("wild"):
        return "万能"
    if len(card) >= 2 and card[0] in _UNO_COLOR_NAMES:
        color = _UNO_COLOR_NAMES[card[0]]
        sym = card[1]
        return f"{color}{_UNO_SYMBOL_NAMES.get(sym, sym)}"
    return str(card)


def poker_card_name(card: str) -> str:
    """Chinese card label: ``hT`` → '红桃10', ``sA`` → '黑桃A', ``dK`` → '方块K'."""
    if not card or len(card) < 2:
        return str(card)
    suit, rank = card[0], card[1:]
    return f"{_POKER_SUIT_NAMES.get(suit, suit)}{_POKER_RANK_NAMES.get(rank, rank)}"


def social_role_name(role: str) -> str:
    """Chinese role label: ``wolf`` → '狼人', ``civilian`` → '平民', ``undercover`` → '卧底'."""
    return _SOCIAL_ROLE_NAMES.get(str(role or ""), str(role))


def piece_name(family: str, ident: str) -> str:
    """One piece id → Chinese name by family (unknown → raw id, fail-soft)."""
    ident = str(ident or "")
    if family == "mahjong":
        return mahjong_tile_name(ident)
    if family == "uno":
        return uno_card_name(ident)
    if family == "poker":
        return poker_card_name(ident)
    if family == "social":
        return social_role_name(ident)
    return ident


def piece_names(family: str, idents: list) -> list[str]:
    """A hand/array of piece ids → Chinese names (``None`` safe)."""
    return [piece_name(family, i) for i in (idents or [])]


def seat_label(pid: str, *, self_pid: str = "", ai_pid: str = "") -> str:
    """Player id → 中文称呼：自己 / AI / N号玩家 / 原文兜底."""
    pid = str(pid or "")
    if self_pid and pid == self_pid:
        return "你"
    if ai_pid and pid == ai_pid:
        return "AI"
    if pid.startswith("p") and pid[1:].isdigit():
        return f"{int(pid[1:]) + 1}号玩家"
    return f"玩家 {pid}"


#: 麻将座位称呼（庄家 + 顺时针下/对/上，4 人桌的东南西北位）。
_MAHJONG_SEAT_NAMES = {"p0": "庄家", "p1": "下家", "p2": "对家", "p3": "上家"}
#: 棋类座位（黑白）。
_GRID_SEAT_NAMES = {"p_black": "黑棋", "p_white": "白棋"}
#: 德州扑克座位（小盲 / 大盲）。
_POKER_SEAT_NAMES = {"p_sb": "小盲位", "p_bb": "大盲位"}


def build_seat_names(family: str, seat_options) -> dict[str, str]:
    """pid → 中文座位称呼（单一数据源，按规则族分派）。

    与 :func:`seat_label` 的"你 / AI / N号玩家"运行时称呼互补——本函数给
    的是每个座位的**固定展示名**，供前端徽章 / 下拉 / 战绩卡 / 复盘直接查表
    （后端 ``/games`` 与 ``/api/history`` 统一下发，前端零推导）。

    - grid：黑棋 / 白棋
    - poker：小盲位 / 大盲位
    - mahjong：庄家 / 下家 / 对家 / 上家（4 人桌座位文化）
    - uno / social / unknown：N号玩家（无庄家概念，与 seat_label 一致）

    未命中族映射的 ``pN`` 兜底"N号玩家"（``p0`` → 1号玩家），其余原样返回——
    谁是卧底 / 狼人杀 / UNO 这类多座社交游戏不再被套上"庄家"。
    """
    fam = str(family or "")
    if fam == "grid":
        table = _GRID_SEAT_NAMES
    elif fam == "poker":
        table = _POKER_SEAT_NAMES
    elif fam == "mahjong":
        table = _MAHJONG_SEAT_NAMES
    else:
        table = {}
    names: dict[str, str] = {}
    for pid in seat_options or []:
        pid = str(pid)
        if pid in table:
            names[pid] = table[pid]
        elif pid.startswith("p") and pid[1:].isdigit():
            names[pid] = f"{int(pid[1:]) + 1}号玩家"
        else:
            names[pid] = pid
    return names


_SOCIAL_ACTION_LABELS = {
    "speak": "发言",
    "vote": "投票",
    "kill": "击杀",
    "check": "查验",
    "shoot": "开枪",
    "heal": "救援",
    "poison": "下毒",
    "guard": "守护",
    "pass": "过",
}


def _social_key_text(key: str) -> str:
    template, sep, arg = key.partition(":")
    label = _SOCIAL_ACTION_LABELS.get(template, template)
    if not sep:
        return label
    if template == "speak":
        return f"发言（{arg}）"
    if arg == "pass":
        return "过"
    return f"{label} {seat_label(arg)}"


def canonical_family_text(family: str, key: str) -> str:
    """Engine canonical key → 自然中文动作描述（按规则族；未知原样返回）."""
    key = str(key or "")
    if family == "mahjong":
        return _mahjong_key_text(key)
    if family == "uno":
        return _uno_key_text(key)
    if family == "poker":
        return _poker_key_text(key)
    if family == "grid":
        return _grid_key_text(key)
    if family == "social":
        return _social_key_text(key)
    return key


def canonical_action_text(game_id: str, key: str) -> str:
    """Engine canonical key → 自然中文动作描述（LLM 直面 / 历史复述用）.

    规则 JSON 的 ``canonicalKey`` 是机器契约（``discard:s1`` /
    ``act:call:2`` / ``play:r7a``），原样读给 LLM 会逼它记“牌 id”。
    本函数按游戏族把它翻译成“打出 一条”“跟注 2”“打出 红7”等；
    未知形状原样返回（fail-soft，不抛错）。
    """
    return canonical_family_text(game_family(game_id), key)


def _mahjong_key_text(key: str) -> str:
    if key == "win_self":
        return "自摸"
    if key == "claim_pass" or key.startswith("claim_pass"):
        return "过"
    template, sep, arg = key.partition(":")
    if not sep:
        return key
    if template == "claim_chi":
        tiles = _MAHJONG_TILE_ID_RE.findall(arg)
        if tiles:
            return "吃 " + "".join(mahjong_tile_name(t) for t in tiles)
        return f"吃 {arg}"
    match = _MAHJONG_TILE_ID_RE.search(arg)
    tile_name = mahjong_tile_name(match.group(0)) if match else arg
    return {
        "discard": f"打出 {tile_name}",
        "claim_win": f"荣和 {tile_name}",
        "claim_peng": f"碰 {tile_name}",
        "claim_gang": f"明杠 {tile_name}",
        "gang_concealed": f"暗杠 {tile_name}",
        "gang_added": f"加杠 {tile_name}",
    }.get(template, f"{template} {tile_name}")


def _uno_key_text(key: str) -> str:
    fixed = {
        "draw": "摸牌",
        "pass": "过",
        "jump_pass": "放弃抢牌",
        "take_penalty": "吃下罚牌",
    }
    if key in fixed:
        return fixed[key]
    template, sep, rest = key.partition(":")
    if not sep:
        return key
    card, _, color = rest.partition(":")
    card_name = uno_card_name(card)
    if template == "play7":
        return f"出 7（{card_name}）与 {seat_label(color)} 换手"
    if template == "play_wild":
        return f"打出 {card_name} → {_UNO_COLOR_NAMES.get(color, color)}"
    if template == "play_drawn_wild":
        return f"打出刚摸的 {card_name} → {_UNO_COLOR_NAMES.get(color, color)}"
    if template == "stack2":
        return f"叠加 {card_name}（+2）"
    if template == "stack4":
        return f"叠加 {card_name}（+4）"
    labels = {"play": "打出", "play_drawn": "打出刚摸的", "jump_play": "抢出"}
    return f"{labels.get(template, template)} {card_name}"


def _poker_key_text(key: str) -> str:
    parts = key.split(":")
    if parts[0] == "act" and len(parts) >= 2:
        choice = parts[1]
        amount = parts[2] if len(parts) >= 3 else ""
        label = _POKER_CHOICE_NAMES.get(choice, choice)
        tail = f" {amount}" if amount not in ("", "0") else ""
        return f"{label}{tail}"
    return _POKER_CHOICE_NAMES.get(key, key)


def _grid_key_text(key: str) -> str:
    match = _GRID_CELL_RE.search(key)
    if match:
        row, col = int(match.group(1)) + 1, int(match.group(2)) + 1
        return f"落子 第{row}行第{col}列"
    match = _GRID_ROWCOL_RE.search(key)
    if match:
        row, col = int(match.group(1)) + 1, int(match.group(2)) + 1
        return f"落子 第{row}行第{col}列"
    return key


def load_rules(game_id: str) -> dict:
    """Load a game's rules JSON (pure data; the engine interprets it)."""
    with open(RULES_DIR / f"{game_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def engine_from_rules(game_id: str, seed: int | None = None, **kwargs) -> GameEngine:
    """Build a bare engine for ``game_id`` (variants selected as data)."""
    return GameEngine(load_rules(game_id), seed=seed, **kwargs)


def resolve_all_chance(engine: GameEngine, state: dict) -> dict:
    """Advance through all pending chance nodes (generic engine protocol)."""
    while engine.get_node_type(state) == "chance":
        _, state = engine.sample_chance(state)
    return state


def texas_hand_name(engine: GameEngine, cards: list) -> str | None:
    """Chinese name of the best hand (e.g. ``'葫芦'``) — rules ``best5`` alias."""
    if not cards:
        return None
    value = engine.eval_expr({"call": ["best5", {"const": list(cards)}]}, {"$cards": list(cards)})
    category = value[0] if isinstance(value, list) and value else None
    return _TEXAS_HAND_NAMES.get(category, "未知")
