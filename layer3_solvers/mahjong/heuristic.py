"""MahjongHeuristicAI — pure-heuristic mahjong policy (v5.1).

Decision priority per phase:
  - claim: win (ron) > exposed gang > pung > chi (if it improves shape) > pass
  - action: self-win (tsumo) > gang (concealed/added) > discard

Discards are chosen by tile scoring: isolated honors first, then
isolated numbered tiles; pairs / triplets / run fragments are kept.
Deterministic for a fixed seed; no search (each decision < 1ms).

The engine's ``get_legal_actions`` is the single source of legality —
the solver never guesses.
"""

from __future__ import annotations

import random

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ..base import SolverBase, SolverConfig, SolverMetrics

_SUITS = "mpsz"


class MahjongHeuristicAI(SolverBase):
    """Heuristic mahjong policy (claim priority + tile-value discards)."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None):
        super().__init__(engine, config or SolverConfig(seed=0))
        self._rng = random.Random(self.config.seed)

    @property
    def name(self) -> str:
        return "mahjong_heuristic"

    def select_action(self, state: dict) -> ActionInstance | None:
        legal = self.engine.get_legal_actions(state)
        if not legal:
            return None
        phase = state.get("env", {}).get("phase", "")
        if phase == "claim":
            return self._pick_claim(legal, state)
        return self._pick_action(legal, state)

    # ── Phase policies ─────────────────────────────────────────────

    def _pick_claim(self, legal: list[ActionInstance], state: dict) -> ActionInstance:
        by_type: dict[str, list[ActionInstance]] = {}
        for a in legal:
            by_type.setdefault(a.template_id, []).append(a)
        if by_type.get("claim_win"):
            return by_type["claim_win"][0]
        if by_type.get("claim_gang"):
            return by_type["claim_gang"][0]
        if by_type.get("claim_peng"):
            return by_type["claim_peng"][0]
        if by_type.get("claim_chi"):
            # Chi only when it improves the hand (needs the run's tiles).
            chi = by_type["claim_chi"][0]
            if self._chi_improves(state, chi.params.get("tiles", [])):
                return chi
        if by_type.get("claim_pass"):
            return by_type["claim_pass"][0]
        return legal[0]

    def _pick_action(self, legal: list[ActionInstance], state: dict) -> ActionInstance:
        by_type: dict[str, list[ActionInstance]] = {}
        for a in legal:
            by_type.setdefault(a.template_id, []).append(a)
        if by_type.get("win_self"):
            return by_type["win_self"][0]
        if by_type.get("gang_concealed"):
            return by_type["gang_concealed"][0]
        if by_type.get("gang_added"):
            return by_type["gang_added"][0]
        if by_type.get("discard"):
            return self._pick_discard(by_type["discard"], state)
        return legal[0]

    # ── Discard scoring ────────────────────────────────────────────

    def _pick_discard(self, discards: list[ActionInstance], state: dict) -> ActionInstance:
        hand = state.get("_arrays", {}).get(f"hand_{self._my_pid(state)}", [])
        counts: dict[str, int] = {}
        for t in hand:
            counts[t] = counts.get(t, 0) + 1
        best = min(discards, key=lambda a: self._tile_value(a.params.get("tile"), counts, hand))
        return best

    @staticmethod
    def _tile_value(tile: str, counts: dict[str, int], hand: list[str]) -> float:
        """Lower = better to discard. Pairs/triplets/runs score negative."""
        if not tile or not isinstance(tile, str):
            return 0
        n = counts.get(tile, 0)
        suit = tile[0]
        rank = int(tile[1:]) if tile[1:].isdigit() else 0
        score = 0.0
        if n >= 3:
            score -= 80  # keep triplets / quads
        elif n == 2:
            score -= 40  # keep pairs
        if suit == "z":
            score += 30 if n == 1 else 0  # unpaired honors first out
            return score
        if n == 1:
            score += 10  # singles go before pairs, all else equal
        # run potential: adjacent tiles in hand (same suit, rank±1)
        for delta in (-1, 1):
            neighbor = f"{suit}{rank + delta}"
            if counts.get(neighbor, 0) >= 1:
                score -= 20
            elif counts.get(neighbor, 0) == 0 and 1 <= rank + delta <= 9:
                score += 0  # neutral
        return score

    def _chi_improves(self, state: dict, tiles: list) -> bool:
        """Chi when the run is not already covered by existing melds."""
        pid = self._my_pid(state)
        meld_tiles = []
        for meld in state.get("_arrays", {}).get(f"melds_{pid}", []):
            meld_tiles.extend(meld.get("tiles", []))
        return not all(t in meld_tiles for t in tiles)

    def _my_pid(self, state: dict) -> str:
        return state.get("env", {}).get("turn", "p0")

    # ── SolverBase contract ────────────────────────────────────────

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        return SolverMetrics(episodes=episodes)
