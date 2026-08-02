"""Texas Hold'em poker utilities — pure card/betting logic for the v5.0 engine.

Every function is deterministic over a game state dict (``_arrays`` + ``env``)
and registered in ``BUILTIN_FUNCTIONS``, so ``rules/texas_holdem.json`` can
call them from expressions and effectors (``{"call": [...]}``).

State contract (see the rules JSON):
  - arrays: ``sb_hole`` / ``bb_hole`` / ``community`` / ``drawn`` (card ids)
  - env: ``sb_*`` / ``bb_*`` per-player fields (stack, committed, folded,
    acted), plus ``street``, ``last_call_to``, ``last_raise_delta``, ``phase``
  - invariant: stack = stack_size - committed at all times
"""

from __future__ import annotations

import itertools
from typing import Optional

PLAYER_SB = 'p_sb'
PLAYER_BB = 'p_bb'

_RANK_VALUES = {'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


# ── Card primitives ──────────────────────────────────────────────────

def card_rank(card_id: str) -> int:
    """Numeric rank of a card id like ``"hA"`` (2..14, T=10 J=11 Q=12 K=13 A=14)."""
    rank = card_id[1] if len(card_id) > 1 else ''
    if rank in _RANK_VALUES:
        return _RANK_VALUES[rank]
    return int(rank) if rank.isdigit() else 0


def contains(items, item) -> bool:
    """Membership test used by rule queries (e.g. the ``undrawn_cards`` filter)."""
    return item in items


# ── Hand evaluation ──────────────────────────────────────────────────

def _eval_five(cards: list[str]) -> tuple:
    """Value of a 5-card hand: ``(category, tiebreaks...)``, bigger is better.

    Categories: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight,
    5 flush, 6 full house, 7 quads, 8 straight flush.
    """
    ranks = sorted((card_rank(c) for c in cards), reverse=True)
    suits = [c[0] for c in cards]
    flush = len(set(suits)) == 1

    uniq = sorted(set(ranks), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[-1] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:  # A-2-3-4-5 wheel
            straight_high = 5

    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    mult = [g[0] for g in groups]

    if flush and straight_high:
        return (8, straight_high)
    if groups[0][1] == 4:
        return (7, mult[0], mult[1])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, mult[0], mult[1])
    if flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if groups[0][1] == 3:
        return (3, mult[0], mult[1], mult[2])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, mult[0], mult[1], mult[2])
    if groups[0][1] == 2:
        return (1, mult[0], mult[1], mult[2], mult[3])
    return (0, *ranks)


def poker_hand_value(cards) -> tuple:
    """Best 5-card hand value from 2..7 card ids (chooses best of C(n,5))."""
    cards = list(cards)
    if len(cards) == 5:
        return _eval_five(cards)
    best: Optional[tuple] = None
    for combo in itertools.combinations(cards, 5):
        value = _eval_five(list(combo))
        if best is None or value > best:
            best = value
    return best or (0,)


# ── State helpers ────────────────────────────────────────────────────

def _other(player_id: str) -> str:
    return PLAYER_BB if player_id == PLAYER_SB else PLAYER_SB


def _own(env: dict, player_id: str, field: str):
    """Per-player env field: ``p_sb`` → ``sb_stack`` etc."""
    return env.get(f'{player_id[2:]}_{field}', 0)


def _stack_size(state: dict) -> int:
    return int(state.get('_constants', {}).get('stack_size', 100))


def _big_blind(state: dict) -> int:
    return int(state.get('_constants', {}).get('big_blind', 2))


# ── Betting logic ────────────────────────────────────────────────────

def poker_pot(state: dict) -> int:
    """Total chips committed to the pot (both players)."""
    env = state.get('env', {})
    return int(env.get('sb_committed', 0) + env.get('bb_committed', 0))


def poker_call_to(state: dict, player_id: str) -> int:
    """Total committed amount ``player_id`` reaches by calling (0 = check)."""
    env = state['env']
    opp = _other(player_id)
    own_committed = int(_own(env, player_id, 'committed'))
    opp_committed = int(_own(env, opp, 'committed'))
    own_total = own_committed + int(_own(env, player_id, 'stack'))
    if bool(_own(env, opp, 'folded')):
        return own_committed
    return max(own_committed, min(opp_committed, own_total))


def poker_min_raise_to(state: dict, player_id: str) -> int:
    """Minimum total for a raise by ``player_id`` (standard NLHE min-raise)."""
    env = state['env']
    if int(_own(env, player_id, 'stack')) <= 0:
        return 1 << 30  # no raise possible
    call_to = poker_call_to(state, player_id)
    delta = int(env.get('last_raise_delta', 0))
    return call_to + max(delta, _big_blind(state))


def poker_round_over(state: dict) -> bool:
    """True when the current betting round must end.

    Ends if: someone folded, both players are all-in, or all in-hand
    players have matched the current bet and acted this round.

    Note: a single all-in player does NOT end the round — heads-up the
    opponent still gets one fold-or-call decision (a raise is impossible
    because ``poker_min_raise_to`` exceeds the stack cap).
    """
    env = state['env']
    if bool(env.get('sb_folded')) or bool(env.get('bb_folded')):
        return True
    if int(env.get('sb_stack', 0)) <= 0 and int(env.get('bb_stack', 0)) <= 0:
        return True
    return (
        int(env.get('sb_committed', 0)) == int(env.get('bb_committed', 0))
        and bool(env.get('sb_acted')) and bool(env.get('bb_acted'))
    )


# ── Showdown / payoff ────────────────────────────────────────────────

def poker_winner(state: dict) -> Optional[str]:
    """Hand winner id, or None for a split pot (folded players never split)."""
    env = state['env']
    sb_folded = bool(env.get('sb_folded'))
    bb_folded = bool(env.get('bb_folded'))
    if sb_folded != bb_folded:
        return PLAYER_BB if sb_folded else PLAYER_SB
    if sb_folded:  # both folded — degenerate, engine should never reach it
        return None
    arrs = state.get('_arrays', {})
    community = list(arrs.get('community', []))
    sb_value = poker_hand_value([*arrs.get('sb_hole', []), *community])
    bb_value = poker_hand_value([*arrs.get('bb_hole', []), *community])
    if sb_value > bb_value:
        return PLAYER_SB
    if bb_value > sb_value:
        return PLAYER_BB
    return None


_HAND_NAMES = {
    0: '高牌', 1: '一对', 2: '两对', 3: '三条', 4: '顺子',
    5: '同花', 6: '葫芦', 7: '四条', 8: '同花顺',
}


def poker_hand_name(cards) -> str:
    """Chinese name of the best hand (e.g. ``'葫芦'``), for display."""
    return _HAND_NAMES.get(poker_hand_value(cards)[0], '未知')


def poker_payoff(state: dict, player_id: str) -> int:
    """Net chip change for ``player_id`` at a terminal state (pot minus cost).

    Handles folds, all-in over-calls (excess chips refunded to the
    over-committer) and split pots.  Zero-sum: payoffs sum to 0.
    """
    env = state['env']
    ca = int(env.get('sb_committed', 0))
    cb = int(env.get('bb_committed', 0))
    own = ca if player_id == PLAYER_SB else cb

    if bool(env.get('sb_folded')) or bool(env.get('bb_folded')):
        winner = poker_winner(state)
        if winner != player_id:
            return -own if winner is not None else 0
        return ca + cb - own

    min_c = min(ca, cb)
    main = 2 * min_c
    refund = (ca - min_c) if player_id == PLAYER_SB else (cb - min_c)
    winner = poker_winner(state)
    if winner is None:  # split pot (main is always even: 2 * min_c)
        return main // 2 - own + refund
    if winner == player_id:
        return main - own + refund
    return -own + refund
