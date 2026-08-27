"""Botzone Mahjong-Format-Test adapter.

The game sends line-based Chinese Standard Mahjong messages inside the
Botzone JSON envelope.  This adapter mirrors the official sample's
state reconstruction, with a deterministic discard heuristic so the bot
always emits legal text responses quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MahjongFormatState:
    player_id: int = -1
    quan: int = -1
    hand: list[str] = field(default_factory=list)
    peng_tiles: list[str] = field(default_factory=list)


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

    if code == "2" and len(current) >= 2:
        drawn = current[1]
        state.hand.append(drawn)
        discard = _choose_discard(state.hand)
        _remove_one(state.hand, discard)
        return f"PLAY {discard}", f"mahjong-format: draw {drawn}, play {discard}"

    # Public notifications and claim opportunities are acknowledged by a
    # conservative PASS.  This is valid for Mahjong-Format-Test and keeps
    # the first integration robust; richer PENG/CHI/HU mapping can be
    # added once the full scoring/claim contract is needed.
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


def _apply_own_response(state: MahjongFormatState, resp: list[str]) -> None:
    if not resp:
        return
    op = resp[0]
    if op == "PLAY" and len(resp) >= 2:
        _remove_one(state.hand, resp[1])
    elif op == "GANG" and len(resp) >= 2:
        for _ in range(4):
            _remove_one(state.hand, resp[1])
    elif op == "BUGANG" and len(resp) >= 2:
        _remove_one(state.hand, resp[1])


def _apply_own_public_action(state: MahjongFormatState, op: str, req: list[str]) -> None:
    if op == "PLAY" and len(req) >= 4:
        _remove_one(state.hand, req[3])
    elif op == "PENG" and len(req) >= 4:
        tile = req[3]
        for _ in range(2):
            _remove_one(state.hand, tile)
        state.peng_tiles.append(tile)
    elif op == "CHI" and len(req) >= 5:
        middle = req[3]
        for tile in _chi_side_tiles(middle):
            _remove_one(state.hand, tile)
    elif op == "GANG":
        # Public GANG may be concealed or exposed; exact tile is not
        # included in this notification, so prior response/public events
        # carry the hand mutation when available.
        return
    elif op == "BUGANG" and len(req) >= 4:
        _remove_one(state.hand, req[3])


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


def _remove_one(hand: list[str], tile: str) -> None:
    try:
        hand.remove(tile)
    except ValueError:
        pass


def _tokens(line: str) -> list[str]:
    return line.strip().split()
