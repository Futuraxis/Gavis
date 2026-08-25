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

``platform/games.py`` (GameSpec registry) and the standalone ``play_*``
apps import from here; no per-game adapter class exists in Layer 2.
"""

from __future__ import annotations

import json
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

_MAHJONG_TILE_NAMES = {"m": "万", "p": "筒", "s": "条", "z": "字"}
_MAHJONG_Z_NAMES = {"1": "东", "2": "南", "3": "西", "4": "北", "5": "中", "6": "发", "7": "白"}


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


def mahjong_tile_name(tile: str) -> str:
    """Chinese tile label, e.g. 'm3' → '三万', 'z5' → '红中' (display)."""
    if not tile or len(tile) < 2:
        return str(tile)
    suit, rank = tile[0], tile[1:]
    if suit == "z":
        return _MAHJONG_Z_NAMES.get(rank, tile)
    return f"{rank}{_MAHJONG_TILE_NAMES.get(suit, suit)}"
