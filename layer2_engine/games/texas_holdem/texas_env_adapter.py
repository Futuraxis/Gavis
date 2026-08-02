"""Texas Hold'em — thin GameEngine adapter (heads-up NLHE, v5.0).

The game logic lives entirely in ``rules/texas_holdem.json`` via
effectors + chance nodes.  This adapter only adds:
  1. Structured observations (``get_observation``) for UI/RL consumers
  2. ``resolve_chance`` — advance through pending chance nodes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ...core.engine import GameEngine
from ...core.poker_utils import poker_hand_name, poker_pot, PLAYER_SB, PLAYER_BB
from ...core.state_graph import clone_state

STREET_NAMES = {0: '翻前', 1: '翻牌', 2: '转牌', 3: '河牌'}


class TexasHoldemAdapter(GameEngine):
    """GameEngine subclass for heads-up Texas Hold'em."""

    STACK_SIZE = 100
    PLAYER_SB = PLAYER_SB
    PLAYER_BB = PLAYER_BB

    def __init__(self, seed: Optional[int] = None):
        rules_path = Path(__file__).resolve().parent.parent.parent.parent / 'rules' / 'texas_holdem.json'
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        super().__init__(rules, seed=seed)

    # ── Chance resolution ────────────────────────────────────────────

    def resolve_chance(self, state: dict) -> dict:
        """Advance through all pending chance nodes (deals, showdown)."""
        while self.get_node_type(state) == 'chance':
            _, state = self.sample_chance(state)
        return state

    # ── Hidden information (opponent-model search) ──────────────────

    def sample_hidden(self, state: dict) -> dict:
        """Complete a consistent world for the HybridSolver (PIMC).

        The acting player's hole cards are known to the searcher; the
        opponent's hole cards are re-sampled uniformly from the undealt
        deck so that each sampled world is internally consistent
        (no duplicate cards, ``drawn`` stays truthful).
        """
        env = state.get('env', {})
        actor = env.get('turn')
        if actor not in (PLAYER_SB, PLAYER_BB):
            return clone_state(state)
        opp = PLAYER_BB if actor == PLAYER_SB else PLAYER_SB
        arrs = state.get('_arrays', {})
        hole = list(arrs.get(f'{opp[2:]}_hole', []))
        if len(hole) != 2:
            return clone_state(state)  # dealing incomplete — nothing to hide yet
        drawn = set(arrs.get('drawn', []))
        remaining = [
            c for c in state.get('_constants', {}).get('card_ids', []) if c not in drawn
        ]
        if len(remaining) < 2:
            return clone_state(state)
        world = clone_state(state)
        world['_arrays'][f'{opp[2:]}_hole'] = self.rng.sample(remaining, 2)
        return world

    # ── Structured observation ───────────────────────────────────────

    def get_observation(self, state: dict, player_id: str = PLAYER_SB) -> dict:
        """Return a structured observation for ``player_id``.

        Opponent hole cards are only revealed at ``game_over`` (showdown);
        before that the observation carries the opponent's committed
        amount but no cards — mirroring the engine's visibility rules.
        """
        env = state.get('env', {})
        arrs = state.get('_arrays', {})

        def _stack(pid: str) -> int:
            return int(env.get(f'{pid[2:]}_stack', 0))

        def _committed(pid: str) -> int:
            return int(env.get(f'{pid[2:]}_committed', 0))

        def _folded(pid: str) -> bool:
            return bool(env.get(f'{pid[2:]}_folded'))

        opp = PLAYER_BB if player_id == PLAYER_SB else PLAYER_SB
        over = env.get('phase') == 'game_over'
        opp_hole = list(arrs.get(f'{opp[2:]}_hole', [])) if over else []

        return {
            'community': list(arrs.get('community', [])),
            'hole': list(arrs.get(f'{player_id[2:]}_hole', [])),
            'opponent_hole': opp_hole,
            'pot': poker_pot(state),
            'street': env.get('street', 0),
            'street_name': STREET_NAMES.get(env.get('street', 0), ''),
            'phase': env.get('phase', ''),
            'turn': env.get('turn'),
            'my_turn': env.get('turn') == player_id and env.get('phase') == 'betting',
            'last_actor': env.get('last_actor'),
            'last_action': env.get('last_action'),
            'last_call_to': int(env.get('last_call_to', 0)),
            'winner': env.get('winner'),
            'over': self.is_terminal(state),
            'my_stack': _stack(player_id),
            'opp_stack': _stack(opp),
            'my_committed': _committed(player_id),
            'opp_committed': _committed(opp),
            'my_folded': _folded(player_id),
            'opp_folded': _folded(opp),
            'hand_name': poker_hand_name(
                [*list(arrs.get(f'{player_id[2:]}_hole', [])), *list(arrs.get('community', []))]
            ) if over else None,
        }
