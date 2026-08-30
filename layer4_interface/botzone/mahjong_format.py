"""Botzone Mahjong-Format-Test adapter.

The game sends line-based Chinese Standard Mahjong messages inside the
Botzone JSON envelope.  This adapter mirrors the official sample's
state reconstruction, with a deterministic discard heuristic so the bot
always emits legal text responses quickly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance
from layer3_solvers.base import SolverConfig
from layer3_solvers.mahjong.heuristic import MahjongHeuristicAI


@dataclass
class MahjongFormatState:
    player_id: int = -1
    quan: int = -1
    hand: list[str] = field(default_factory=list)
    melds: list[dict[str, Any]] = field(default_factory=list)
    peng_tiles: list[str] = field(default_factory=list)
    last_discard: str | None = None
    last_discard_player: int = -1


def is_mahjong_format_payload(payload: dict[str, Any]) -> bool:
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        return False
    first = requests[0]
    return isinstance(first, str) and first.strip().startswith("0 ")


def decide_mahjong_format(payload: dict[str, Any]) -> tuple[str, str]:
    """Return ``(response, debug)`` for Botzone Mahjong-Format-Test."""
    requests = [str(x) for x in payload.get("requests", [])]
    responses = [str(x) for x in payload.get("responses", [])]
    turn_id = len(responses)
    if turn_id < 2:
        return "PASS", f"mahjong-format: handshake turn={turn_id}"

    state = _reconstruct(requests, responses, turn_id)
    current = _tokens(requests[turn_id])
    if not current:
        return "PASS", "mahjong-format: empty request"
    code = current[0]

    layer3 = _decide_with_layer3(state, current)
    if layer3 is not None:
        response, debug = layer3
        return response, f"mahjong-international/layer3: {debug}"

    if code == "2" and len(current) >= 2:
        drawn = current[1]
        state.hand.append(drawn)
        gang = _choose_gang_after_draw(state)
        if gang:
            return gang, f"mahjong-format: draw {drawn}, {gang}"
        discard = _choose_discard(state.hand)
        _remove_one(state.hand, discard)
        return f"PLAY {discard}", f"mahjong-format: draw {drawn}, play {discard}"

    if code == "3" and len(current) >= 4:
        response = _decide_public_request(state, current)
        return response, f"mahjong-format: public {' '.join(current[:4])} -> {response}"

    return "PASS", f"mahjong-format: pass request={' '.join(current[:3])}"


def _reconstruct(requests: list[str], responses: list[str], turn_id: int) -> MahjongFormatState:
    state = MahjongFormatState()
    first = _tokens(requests[0])
    if len(first) >= 3 and first[0] == "0":
        state.player_id = int(first[1])
        state.quan = int(first[2])

    second = _tokens(requests[1])
    if len(second) >= 18 and second[0] == "1":
        # 1 hua0 hua1 hua2 hua3 Card1 ... Card13 flowers...
        state.hand = list(second[5:18])

    for i in range(2, min(turn_id, len(requests))):
        req = _tokens(requests[i])
        resp = _tokens(responses[i]) if i < len(responses) else []
        _apply_past_turn(state, req, resp)
    return state


def _apply_past_turn(state: MahjongFormatState, req: list[str], resp: list[str]) -> None:
    if not req:
        return
    if req[0] == "2" and len(req) >= 2:
        state.hand.append(req[1])
        _apply_own_response(state, resp)
        return
    if req[0] != "3" or len(req) < 3:
        return
    try:
        actor = int(req[1])
    except ValueError:
        return
    op = req[2]
    if actor == state.player_id:
        _apply_own_public_action(state, op, req)
    if op == "PLAY" and len(req) >= 4:
        state.last_discard = req[3]
        state.last_discard_player = actor
    elif op in {"PENG", "CHI"}:
        state.last_discard = req[4] if op == "CHI" and len(req) >= 5 else req[3] if len(req) >= 4 else None
        state.last_discard_player = actor
    elif op in {"GANG", "BUGANG", "DRAW", "BUHUA"}:
        state.last_discard = None
        state.last_discard_player = -1


def _apply_own_response(state: MahjongFormatState, resp: list[str]) -> None:
    if not resp:
        return
    op = resp[0]
    if op == "PLAY" and len(resp) >= 2:
        _remove_one(state.hand, resp[1])
    elif op == "GANG" and len(resp) >= 2:
        for _ in range(4):
            _remove_one(state.hand, resp[1])
        state.melds.append({"type": "concealed_gang", "tiles": [resp[1]] * 4})
    elif op == "BUGANG" and len(resp) >= 2:
        _remove_one(state.hand, resp[1])
        _promote_peng(state, resp[1])


def _apply_own_public_action(state: MahjongFormatState, op: str, req: list[str]) -> None:
    if op == "PLAY" and len(req) >= 4:
        _remove_one(state.hand, req[3])
    elif op == "PENG" and len(req) >= 4:
        tile = state.last_discard
        if tile is None:
            return
        for _ in range(2):
            _remove_one(state.hand, tile)
        state.peng_tiles.append(tile)
        state.melds.append({"type": "peng", "tiles": [tile] * 3})
        _remove_one(state.hand, req[3])
    elif op == "CHI" and len(req) >= 5:
        middle = req[3]
        for tile in _chi_hand_tiles(middle, state.last_discard):
            _remove_one(state.hand, tile)
        state.melds.append({"type": "chi", "tiles": _chi_sequence(middle)})
        _remove_one(state.hand, req[4])
    elif op == "GANG":
        # Public GANG may be concealed or exposed; exact tile is not
        # included in this notification, so prior response/public events
        # carry the hand mutation when available.
        return
    elif op == "BUGANG" and len(req) >= 4:
        _remove_one(state.hand, req[3])
        _promote_peng(state, req[3])


def _decide_with_layer3(state: MahjongFormatState, current: list[str]) -> tuple[str, str] | None:
    engine = _international_engine()
    gavis_state = _to_gavis_state(engine, state, current)
    if gavis_state is None:
        return None
    legal = engine.get_legal_actions(gavis_state)
    if not legal:
        return None
    solver = MahjongHeuristicAI(engine, SolverConfig(seed=0))
    action = solver.select_action(gavis_state)
    if action is None:
        return None
    response = _action_to_botzone_response(action, state, current)
    if response is None:
        return None
    return response, action.canonical_key


@lru_cache(maxsize=1)
def _international_engine() -> GameEngine:
    root = Path(__file__).resolve().parents[2]
    with open(root / "rules" / "mahjong.json", encoding="utf-8") as f:
        rules = json.load(f)
    return GameEngine(rules, seed=0, variant="international", player_count=4, allow_codegen=False)


def _to_gavis_state(engine: GameEngine, state: MahjongFormatState, current: list[str]) -> dict[str, Any] | None:
    pid = _gavis_player(state.player_id)
    arrays: dict[str, Any] = {
        f"hand_{pid}": [_to_gavis_tile(tile) for tile in state.hand],
        f"melds_{pid}": [_to_gavis_meld(meld) for meld in state.melds],
    }
    env: dict[str, Any] = {
        "turn": pid,
        "wall_count": 70,
        "phase": "action",
        "done": [],
        "winners": [],
        "payoffs": [0, 0, 0, 0],
    }
    if current[0] == "2" and len(current) >= 2:
        arrays[f"hand_{pid}"] = [*arrays[f"hand_{pid}"], _to_gavis_tile(current[1])]
        env["last_drawn"] = _to_gavis_tile(current[1])
        env["phase"] = "action"
    elif current[0] == "3" and len(current) >= 4 and current[2] == "PLAY":
        discarder = _gavis_player(_safe_int(current[1], -1))
        env.update(
            {
                "phase": "claim",
                "turn": pid,
                "actor": None,
                "last_discard": _to_gavis_tile(current[3]),
                "last_discarder": discarder,
                "claim_queue": [pid],
                "claim_index": 0,
            }
        )
    elif current[0] == "3" and len(current) >= 4 and current[2] == "BUGANG":
        # Robbing a kong maps to the same claim-win action shape.
        discarder = _gavis_player(_safe_int(current[1], -1))
        env.update(
            {
                "phase": "claim",
                "turn": pid,
                "actor": None,
                "last_discard": _to_gavis_tile(current[3]),
                "last_discarder": discarder,
                "claim_queue": [pid],
                "claim_index": 0,
            }
        )
    else:
        return None
    return engine.load_state({"_arrays": arrays, "env": env})


def _action_to_botzone_response(
    action: ActionInstance, state: MahjongFormatState, current: list[str]
) -> str | None:
    tid = action.template_id
    if tid in {"win_self", "claim_win"}:
        return "HU"
    if tid == "discard":
        return f"PLAY {_from_gavis_tile(action.params.get('tile'))}"
    if tid == "gang_concealed":
        return f"GANG {_from_gavis_tile(action.params.get('tile'))}"
    if tid == "gang_added":
        return f"BUGANG {_from_gavis_tile(action.params.get('tile'))}"
    if tid == "claim_gang":
        return "GANG"
    if tid == "claim_peng" and len(current) >= 4:
        claimed = current[3]
        remaining = list(state.hand)
        _remove_one(remaining, claimed)
        _remove_one(remaining, claimed)
        return f"PENG {_choose_discard(remaining)}"
    if tid == "claim_chi" and len(current) >= 4:
        actor = _safe_int(current[1], -1)
        if actor != (state.player_id - 1) % 4:
            return None
        tiles = [_from_gavis_tile(tile) for tile in action.params.get("tiles", [])]
        if len(tiles) != 3:
            return None
        middle = tiles[1]
        remaining = list(state.hand)
        for tile in tiles:
            if tile != current[3]:
                _remove_one(remaining, tile)
        return f"CHI {middle} {_choose_discard(remaining)}"
    if tid == "claim_pass":
        return "PASS"
    return None


def _choose_gang_after_draw(state: MahjongFormatState) -> str | None:
    counts = _counts(state.hand)
    for tile in sorted(counts):
        if counts[tile] >= 4:
            return f"GANG {tile}"
    for tile in sorted(state.peng_tiles):
        if counts.get(tile, 0) >= 1:
            return f"BUGANG {tile}"
    return None


def _decide_public_request(state: MahjongFormatState, req: list[str]) -> str:
    try:
        actor = int(req[1])
    except ValueError:
        return "PASS"
    op = req[2]
    if actor == state.player_id:
        return "PASS"
    if op == "BUGANG":
        # Robbing a kong requires full fan calculation; avoid unsafe HU.
        return "PASS"
    if op != "PLAY" or len(req) < 4:
        return "PASS"

    tile = req[3]
    counts = _counts(state.hand)
    if counts.get(tile, 0) >= 3:
        return "GANG"
    if counts.get(tile, 0) >= 2:
        remaining = list(state.hand)
        _remove_one(remaining, tile)
        _remove_one(remaining, tile)
        discard = _choose_discard(remaining)
        return f"PENG {discard}"

    if actor == (state.player_id - 1) % 4:
        chi = _best_chi_response(state.hand, tile)
        if chi is not None:
            middle, discard = chi
            return f"CHI {middle} {discard}"
    return "PASS"


def _best_chi_response(hand: list[str], claimed: str) -> tuple[str, str] | None:
    candidates: list[tuple[int, str, str]] = []
    for middle in _chi_middles_for_claim(claimed):
        needed = _chi_hand_tiles(middle, claimed)
        if not needed or not _has_tiles(hand, needed):
            continue
        remaining = list(hand)
        for tile in needed:
            _remove_one(remaining, tile)
        if not remaining:
            continue
        discard = _choose_discard(remaining)
        before = sum(_tile_score(tile, _counts(hand)) for tile in needed)
        after = _tile_score(discard, _counts(remaining))
        candidates.append((after - before, middle, discard))
    if not candidates:
        return None
    _, middle, discard = min(candidates)
    return middle, discard


def _choose_discard(hand: list[str]) -> str:
    if not hand:
        return "W1"
    counts: dict[str, int] = {}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    return min(hand, key=lambda tile: (_tile_score(tile, counts), tile))


def _tile_score(tile: str, counts: dict[str, int]) -> int:
    """Lower score means better to discard."""
    if len(tile) < 2:
        return 0
    suit = tile[0]
    rank = _rank(tile)
    n = counts.get(tile, 0)
    score = 0
    if n >= 3:
        score += 80
    elif n == 2:
        score += 40
    if suit not in {"W", "B", "T"}:
        return score - 30 if n == 1 else score
    if n == 1:
        score -= 10
    for delta in (-1, 1):
        if counts.get(f"{suit}{rank + delta}", 0):
            score += 20
    return score


def _rank(tile: str) -> int:
    try:
        return int(tile[1:])
    except ValueError:
        return 0


def _chi_side_tiles(middle: str) -> list[str]:
    if len(middle) < 2 or middle[0] not in {"W", "B", "T"}:
        return []
    rank = _rank(middle)
    return [f"{middle[0]}{rank - 1}", f"{middle[0]}{rank + 1}"]


def _chi_sequence(middle: str) -> list[str]:
    if len(middle) < 2 or middle[0] not in {"W", "B", "T"}:
        return []
    rank = _rank(middle)
    return [f"{middle[0]}{rank - 1}", middle, f"{middle[0]}{rank + 1}"]


def _chi_middles_for_claim(claimed: str) -> list[str]:
    if len(claimed) < 2 or claimed[0] not in {"W", "B", "T"}:
        return []
    suit = claimed[0]
    rank = _rank(claimed)
    middles = []
    for middle_rank in (rank - 1, rank, rank + 1):
        if 2 <= middle_rank <= 8 and middle_rank - 1 <= rank <= middle_rank + 1:
            middles.append(f"{suit}{middle_rank}")
    return middles


def _chi_hand_tiles(middle: str, claimed: str | None) -> list[str]:
    if len(middle) < 2 or middle[0] not in {"W", "B", "T"}:
        return []
    rank = _rank(middle)
    sequence = [f"{middle[0]}{rank - 1}", middle, f"{middle[0]}{rank + 1}"]
    if claimed not in sequence:
        return []
    remaining = list(sequence)
    remaining.remove(claimed)
    return remaining


def _has_tiles(hand: list[str], tiles: list[str]) -> bool:
    counts = _counts(hand)
    for tile in tiles:
        if counts.get(tile, 0) <= 0:
            return False
        counts[tile] -= 1
    return True


def _counts(hand: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    return counts


def _promote_peng(state: MahjongFormatState, tile: str) -> None:
    for meld in state.melds:
        if meld.get("type") == "peng" and meld.get("tiles") == [tile] * 3:
            meld["type"] = "added_gang"
            meld["tiles"] = [tile] * 4
            return
    state.melds.append({"type": "added_gang", "tiles": [tile] * 4})


def _gavis_player(player_id: int) -> str:
    return f"p{player_id}" if 0 <= player_id <= 3 else "p0"


def _to_gavis_tile(tile: str) -> str:
    if len(tile) < 2:
        return tile
    suit = tile[0]
    rank = tile[1:]
    if suit == "W":
        return f"m{rank}"
    if suit == "B":
        return f"p{rank}"
    if suit == "T":
        return f"s{rank}"
    if suit == "F":
        return f"z{rank}"
    if suit == "J":
        return f"z{int(rank) + 4}" if rank.isdigit() else tile
    return tile


def _from_gavis_tile(tile: Any) -> str:
    value = str(tile)
    if len(value) < 2:
        return value
    suit = value[0]
    rank = value[1:]
    if suit == "m":
        return f"W{rank}"
    if suit == "p":
        return f"B{rank}"
    if suit == "s":
        return f"T{rank}"
    if suit == "z":
        n = _safe_int(rank, 0)
        return f"F{n}" if 1 <= n <= 4 else f"J{n - 4}"
    return value


def _to_gavis_meld(meld: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": meld.get("type", "peng"),
        "tiles": [_to_gavis_tile(str(tile)) for tile in meld.get("tiles", [])],
        "from": meld.get("from"),
    }


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _remove_one(hand: list[str], tile: str) -> None:
    try:
        hand.remove(tile)
    except ValueError:
        pass


def _tokens(line: str) -> list[str]:
    return line.strip().split()
