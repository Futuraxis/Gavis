"""Botzone Texas Hold'em JSON adapter.

This module intentionally has no dependency on the Mahjong line protocol.
Botzone Texas Hold'em sends the current hand as a JSON object and expects one
integer response: ``-1`` fold, ``-2`` all-in, ``0`` call/check, or a positive
raise increment.  The adapter reconstructs only the public betting state
needed to prove that the selected response is legal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

SMALL_BLIND = 50
BIG_BLIND = 100
FOLD = -1
ALL_IN = -2
CALL_OR_CHECK = 0

_REQUIRED_REQUEST_FIELDS = frozenset(
    {"num_players", "dealer_id", "my_id", "my_chips", "my_cards", "public_cards", "history"}
)


class TexasHoldemFormatError(ValueError):
    """A Botzone Texas Hold'em request does not meet its public contract."""


@dataclass(frozen=True)
class BettingState:
    """Public betting facts reconstructed from the current request history."""

    round_id: int
    player_bets: tuple[int, ...]
    round_bet: int
    min_raise: int
    all_in_locked: bool


@dataclass(frozen=True)
class TexasHoldemRequest:
    """Validated, platform-specific input.  It remains in Layer 4."""

    num_players: int
    dealer_id: int
    my_id: int
    my_chips: int
    my_cards: tuple[int, int]
    public_cards: tuple[int, ...]
    history: tuple[dict[str, Any], ...]
    hand: int
    max_hand: int
    total_win_chips: tuple[int, ...]
    total_win_games: tuple[int, ...]


def is_texas_holdem_payload(payload: dict[str, Any]) -> bool:
    """Return true only for Botzone's Texas JSON shape, never Mahjong input."""
    try:
        current = _current_request(payload)
    except TexasHoldemFormatError:
        return False
    return _REQUIRED_REQUEST_FIELDS.issubset(current)


def decide_texas_holdem(payload: dict[str, Any]) -> tuple[int, str]:
    """Choose one strictly legal Botzone Texas Hold'em integer action."""
    request = parse_texas_holdem_request(payload)
    betting = reconstruct_betting_state(request)
    action = _heuristic_action(request, betting)
    legal = legal_responses(request, betting)
    if action not in legal:
        # This must always be a legal response.  Check/call is strategically
        # conservative; fold is the universal fallback if it is unavailable.
        action = CALL_OR_CHECK if CALL_OR_CHECK in legal else FOLD
    return action, _debug(request, betting, action)


def parse_texas_holdem_request(payload: dict[str, Any]) -> TexasHoldemRequest:
    """Extract and validate the latest Texas request from a Botzone envelope."""
    raw = _current_request(payload)
    missing = sorted(_REQUIRED_REQUEST_FIELDS - set(raw))
    if missing:
        raise TexasHoldemFormatError(f"missing Texas request fields: {', '.join(missing)}")

    players = _as_int(raw["num_players"], "num_players")
    dealer = _as_int(raw["dealer_id"], "dealer_id")
    mine = _as_int(raw["my_id"], "my_id")
    chips = _as_int(raw["my_chips"], "my_chips")
    if not 2 <= players <= 10:
        raise TexasHoldemFormatError("num_players must be in [2, 10]")
    if not 0 <= dealer < players or not 0 <= mine < players:
        raise TexasHoldemFormatError("dealer_id/my_id outside player range")
    if chips < 0:
        raise TexasHoldemFormatError("my_chips must not be negative")

    cards = _cards(raw["my_cards"], "my_cards", exact=2)
    public = _cards(raw["public_cards"], "public_cards", maximum=5)
    history_value = raw["history"]
    if not isinstance(history_value, list) or not all(isinstance(item, dict) for item in history_value):
        raise TexasHoldemFormatError("history must be a list of objects")
    history = tuple(dict(item) for item in history_value)

    return TexasHoldemRequest(
        num_players=players,
        dealer_id=dealer,
        my_id=mine,
        my_chips=chips,
        my_cards=cards,
        public_cards=public,
        history=history,
        hand=_as_int(raw.get("hand", 0), "hand"),
        max_hand=_as_int(raw.get("max_hand", 1), "max_hand"),
        total_win_chips=_int_tuple(raw.get("total_win_chips", []), "total_win_chips"),
        total_win_games=_int_tuple(raw.get("total_win_games", []), "total_win_games"),
    )


def reconstruct_betting_state(request: TexasHoldemRequest) -> BettingState:
    """Replay the current betting round using the judge's documented semantics."""
    current_round = max((_as_int(item.get("round", 0), "history.round") for item in request.history), default=0)
    if not 0 <= current_round <= 3:
        raise TexasHoldemFormatError("history.round must be in [0, 3]")

    bets = [0] * request.num_players
    if current_round == 0:
        bets[_next_player(request.dealer_id, 1, request.num_players)] = SMALL_BLIND
        bets[_next_player(request.dealer_id, 2, request.num_players)] = BIG_BLIND
        round_bet = BIG_BLIND
        min_raise = 2 * BIG_BLIND
    else:
        round_bet = 0
        min_raise = BIG_BLIND

    all_in_locked = False
    for event in request.history:
        if _as_int(event.get("round", 0), "history.round") != current_round:
            continue
        player = _as_int(event.get("player_id"), "history.player_id")
        if not 0 <= player < request.num_players:
            raise TexasHoldemFormatError("history.player_id outside player range")
        action = _as_int(event.get("action"), "history.action")
        action_type = event.get("action_type")
        if action_type == "fold":
            bets[player] = FOLD
        elif action_type == "allin":
            bets[player] = ALL_IN
            all_in_locked = True
        elif action_type in {"call", "check"}:
            if bets[player] >= 0:
                bets[player] = round_bet
        elif action_type == "raise":
            if action <= 0 or bets[player] < 0:
                raise TexasHoldemFormatError("invalid raise in history")
            bets[player] += action
            round_bet = max(round_bet, bets[player])
            min_raise = max(min_raise, 2 * action)
        else:
            raise TexasHoldemFormatError(f"unknown history action_type: {action_type!r}")

    return BettingState(
        round_id=current_round,
        player_bets=tuple(bets),
        round_bet=round_bet,
        min_raise=min_raise,
        all_in_locked=all_in_locked,
    )


def legal_responses(request: TexasHoldemRequest, betting: BettingState) -> frozenset[int]:
    """Return the subset of Botzone response integers safe to emit now."""
    mine = betting.player_bets[request.my_id]
    if mine < 0:
        return frozenset()
    legal = {FOLD, ALL_IN}
    if betting.all_in_locked:
        return frozenset(legal)
    to_call = betting.round_bet - mine
    if 0 <= to_call < request.my_chips:
        legal.add(CALL_OR_CHECK)
    # The official sample checks the raise increment against remaining chips;
    # keeping exactly this conservative minimum prevents malformed positives.
    if betting.min_raise > 0 and betting.min_raise < request.my_chips:
        legal.add(betting.min_raise)
    return frozenset(legal)


def _heuristic_action(request: TexasHoldemRequest, betting: BettingState) -> int:
    """Fast baseline policy; legality remains solely the responsibility of the adapter."""
    legal = legal_responses(request, betting)
    if not legal:
        return FOLD
    ranks = sorted((card // 4 + 2 for card in request.my_cards), reverse=True)
    suited = request.my_cards[0] % 4 == request.my_cards[1] % 4
    pair = ranks[0] == ranks[1]
    premium = pair and ranks[0] >= 10 or ranks[0] >= 13 and ranks[1] >= 10 or suited and ranks[0] >= 12 and ranks[1] >= 10
    strong = premium or pair or (ranks[0] >= 11 and ranks[1] >= 9) or suited and ranks[0] >= 10 and ranks[1] >= 8

    if betting.all_in_locked:
        return ALL_IN if premium and ALL_IN in legal else FOLD
    if betting.round_bet == betting.player_bets[request.my_id]:
        if premium and betting.min_raise in legal:
            return betting.min_raise
        return CALL_OR_CHECK
    to_call = betting.round_bet - betting.player_bets[request.my_id]
    # Avoid committing a large fraction of the stack with a weak hand.
    if not strong and to_call > max(BIG_BLIND * 2, request.my_chips // 12):
        return FOLD
    return CALL_OR_CHECK if CALL_OR_CHECK in legal else FOLD


def _current_request(payload: dict[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests")
    value: Any = requests[-1] if isinstance(requests, list) and requests else payload.get("request", payload)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TexasHoldemFormatError("Texas request string is not JSON") from exc
    if not isinstance(value, dict):
        raise TexasHoldemFormatError("Texas request must be a JSON object")
    return value


def _cards(value: Any, name: str, *, exact: int | None = None, maximum: int | None = None) -> tuple[int, ...]:
    cards = _int_tuple(value, name)
    if exact is not None and len(cards) != exact:
        raise TexasHoldemFormatError(f"{name} must contain exactly {exact} cards")
    if maximum is not None and len(cards) > maximum:
        raise TexasHoldemFormatError(f"{name} must contain at most {maximum} cards")
    if any(card < 0 or card > 51 for card in cards) or len(set(cards)) != len(cards):
        raise TexasHoldemFormatError(f"{name} contains invalid or duplicate cards")
    return cards


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TexasHoldemFormatError(f"{name} must be a list")
    return tuple(_as_int(item, name) for item in value)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TexasHoldemFormatError(f"{name} must be an integer")
    return value


def _next_player(player: int, offset: int, count: int) -> int:
    return (player + offset) % count


def _debug(request: TexasHoldemRequest, betting: BettingState, action: int) -> str:
    mine = betting.player_bets[request.my_id]
    to_call = betting.round_bet - mine if mine >= 0 else -1
    return f"texas-holdem: round={betting.round_id} to_call={to_call} action={action}"
