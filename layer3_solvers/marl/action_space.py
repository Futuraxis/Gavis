"""ActionSpace — fixed-size masked action spaces for MARL solvers.

The engine returns variable-length legal action lists per state; deep RL
solvers need a fixed-size output space.  An ``ActionSpace`` is an ordered
list of prototype actions (mirroring the rules' canonicalKey templates)
onto which each state's legal actions are projected as a boolean mask.

--- Layouts ---

- mahjong (227): discard / gang_concealed / gang_added / claim_peng /
  claim_gang / claim_win × 34 tiles, claim_chi × 21 runs, claim_pass,
  win_self
- moon_chess (9): ``place_piece`` at cell ``r*3+c``
- texas_holdem (48): ``act:{choice}:{amount}`` — 3 choices × 16 tiers
- unknown adapters: one prototype per observed template_id (fallback)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from layer2_engine.interfaces.solver_adapter import (
    ActionInstance,
    SolverAdapter,
    State,
)

# ── Game constants (mirror rules/*.json canonicalKey templates) ─────

MOON_CHESS_SIZE = 3

_MAHJONG_TILES = [f"{s}{r}" for s in "mps" for r in range(1, 10)] + [f"z{r}" for r in range(1, 8)]

TEXAS_CHOICES = ("fold", "call", "raise")
TEXAS_AMOUNTS = (0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 100)

# Mahjong prototype block layout: (template_id, base index)
_MAHJONG_BLOCKS = (
    ("discard", 0),
    ("gang_concealed", 34),
    ("gang_added", 68),
    ("claim_peng", 102),
    ("claim_gang", 136),
    ("claim_win", 170),
)
_MAHJONG_CHI_BASE = 204
_MAHJONG_CHI_SLOTS = 21
_MAHJONG_PASS_IDX = 225
_MAHJONG_WIN_SELF_IDX = 226

_RE_KEY = re.compile(r"^(\w+):(.+)$")


@dataclass(frozen=True)
class PrototypeAction:
    """A fixed (template_id, canonical-key shape) action slot."""

    template_id: str
    index: int


class ActionSpace:
    """Fixed action space: legal list ↔ index space projection.

    Parameters
    ----------
    prototypes : list[PrototypeAction]
        Ordered slot list; ``dim`` is its length.
    parser : Callable[[ActionInstance], int | None]
        Maps an action's canonical key to its prototype index (or None).
    legal_getter : Callable[[State], list[ActionInstance]], optional
        Returns the legal actions of a state; defaults to ``adapter.get_legal_actions``.
    """

    def __init__(
        self,
        prototypes: list[PrototypeAction],
        parser: Callable[[ActionInstance], int | None],
        legal_getter: Callable[[State], list[ActionInstance]] | None = None,
    ):
        self._prototypes = prototypes
        self._parser = parser
        self._legal_getter = legal_getter or (lambda state: [])
        # canonical_key → index 缓存（键与状态无关，训练热路径上避免重复正则解析）
        self._index_cache: dict[str, int | None] = {}

    # ── Meta ─────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """Number of action slots."""
        return len(self._prototypes)

    @property
    def prototypes(self) -> list[PrototypeAction]:
        return list(self._prototypes)

    # ── Mapping ──────────────────────────────────────────────────────

    def index_of(self, action: ActionInstance) -> int | None:
        """Index of ``action`` in this space, or None if unmapped."""
        key = action.canonical_key
        cached = self._index_cache.get(key, ...)
        if cached is not ...:
            return cached
        idx = self._parser(action)
        self._index_cache[key] = idx
        return idx

    def legal_mask(self, state: State, legal: list[ActionInstance] | None = None) -> np.ndarray:
        """Float32 mask over the space; duplicate actions set one bit.

        ``legal`` may be the caller's already-computed legal action list
        (run_episode computes it anyway) — avoids a second engine pass.
        """
        mask = np.zeros(self.dim, dtype=np.float32)
        actions = legal if legal is not None else self._legal_getter(state)
        for action in actions:
            idx = self.index_of(action)
            if idx is not None:
                mask[idx] = 1.0
        return mask

    def legal_indices(self, state: State) -> list[int]:
        """Sorted indices of all legal actions in ``state``."""
        return [i for i, v in enumerate(self.legal_mask(state)) if v > 0]

    def action_from_index(self, idx: int, legal: list[ActionInstance]) -> ActionInstance | None:
        """First legal action whose index is ``idx`` (fallback: ``legal[0]``)."""
        for action in legal:
            if self.index_of(action) == idx:
                return action
        return legal[0] if legal else None

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def build_from_adapter(cls, adapter: SolverAdapter) -> "ActionSpace":
        """Build an action space for ``adapter`` by dispatching on its class."""
        from layer2_engine.games.mahjong.mahjong_adapter import MahjongAdapter
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
        from layer2_engine.games.texas_holdem.texas_env_adapter import TexasHoldemAdapter

        if isinstance(adapter, MoonChessAdapter):
            return cls._build_moon_chess(adapter)
        if isinstance(adapter, MahjongAdapter):
            return cls._build_mahjong(adapter)
        if isinstance(adapter, TexasHoldemAdapter):
            return cls._build_texas(adapter)
        return cls._build_generic(adapter)

    # ── Per-game builders ────────────────────────────────────────────

    @classmethod
    def _build_moon_chess(cls, adapter: SolverAdapter) -> "ActionSpace":
        def parser(action: ActionInstance) -> int | None:
            m = re.match(r"^place:(\d+),(\d+)$", action.canonical_key)
            if m is None:
                return None
            x, y = int(m.group(1)), int(m.group(2))
            if x >= MOON_CHESS_SIZE or y >= MOON_CHESS_SIZE:
                return None
            return y * MOON_CHESS_SIZE + x

        prototypes = [
            PrototypeAction("place_piece", y * MOON_CHESS_SIZE + x)
            for y in range(MOON_CHESS_SIZE)
            for x in range(MOON_CHESS_SIZE)
        ]
        return cls(prototypes, parser, adapter.get_legal_actions)

    @classmethod
    def _build_mahjong(cls, adapter: SolverAdapter) -> "ActionSpace":
        constants = getattr(adapter, "_constants", {})
        tile_ids = constants.get("tile_ids") or []
        seen: list[str] = []
        for t in tile_ids:
            if t not in seen:
                seen.append(t)
        tiles = seen or _MAHJONG_TILES
        tile_index = {t: i for i, t in enumerate(tiles)}

        chi_runs = constants.get("chi_runs") or []
        chi_lookup = {tuple(sorted(run)): i for i, run in enumerate(chi_runs)}

        def tile_idx(tile: str) -> int | None:
            return tile_index.get(tile)

        def chi_idx(tiles_list: list[str]) -> int | None:
            """Match a claim_chi tile list to a run slot.

            Exact match against ``chi_runs`` first; falls back to a
            deterministic hash (covers wild-tile variants like hongzhong).
            """
            key = tuple(sorted(tiles_list))
            hit = chi_lookup.get(key)
            if hit is not None:
                return _MAHJONG_CHI_BASE + hit
            total = 0
            for t in tiles_list:
                ti = tile_index.get(t)
                if ti is None:
                    return None
                total += ti
            return _MAHJONG_CHI_BASE + (total % max(1, _MAHJONG_CHI_SLOTS))

        def parser(action: ActionInstance) -> int | None:
            key = action.canonical_key
            m = _RE_KEY.match(key)
            if m is None:
                if key == "claim_pass":
                    return _MAHJONG_PASS_IDX
                if key == "win_self":
                    return _MAHJONG_WIN_SELF_IDX
                return None
            name, rest = m.group(1), m.group(2)
            for template_id, base in _MAHJONG_BLOCKS:
                if name != template_id:
                    continue
                ti = tile_idx(rest)
                return base + ti if ti is not None else None
            if name == "claim_chi":
                try:
                    tiles_list = ast.literal_eval(rest)
                except (ValueError, SyntaxError):
                    return None
                if not isinstance(tiles_list, list):
                    return None
                return chi_idx(tiles_list)
            return None

        prototypes = [
            *[
                PrototypeAction(template_id, base + i)
                for template_id, base in _MAHJONG_BLOCKS
                for i in range(len(tiles))
            ],
            *[PrototypeAction("claim_chi", _MAHJONG_CHI_BASE + i) for i in range(_MAHJONG_CHI_SLOTS)],
            PrototypeAction("claim_pass", _MAHJONG_PASS_IDX),
            PrototypeAction("win_self", _MAHJONG_WIN_SELF_IDX),
        ]
        return cls(prototypes, parser, adapter.get_legal_actions)

    @classmethod
    def _build_texas(cls, adapter: SolverAdapter) -> "ActionSpace":
        amount_index = {amount: i for i, amount in enumerate(TEXAS_AMOUNTS)}
        choice_index = {choice: i for i, choice in enumerate(TEXAS_CHOICES)}

        def parser(action: ActionInstance) -> int | None:
            m = re.match(r"^act:(\w+):(\d+)$", action.canonical_key)
            if m is None:
                return None
            ci = choice_index.get(m.group(1))
            ai = amount_index.get(int(m.group(2)))
            if ci is None or ai is None:
                return None
            return ci * len(TEXAS_AMOUNTS) + ai

        prototypes = [
            PrototypeAction("act", ci * len(TEXAS_AMOUNTS) + ai)
            for ci in range(len(TEXAS_CHOICES))
            for ai in range(len(TEXAS_AMOUNTS))
        ]
        return cls(prototypes, parser, adapter.get_legal_actions)

    @classmethod
    def _build_generic(cls, adapter: SolverAdapter) -> "ActionSpace":
        """Fallback: one slot per distinct template_id observed."""
        template_ids = _probe_template_ids(adapter)
        by_template = {tid: i for i, tid in enumerate(template_ids)}

        def parser(action: ActionInstance) -> int | None:
            return by_template.get(action.template_id)

        return cls(
            [PrototypeAction(tid, i) for i, tid in enumerate(template_ids)],
            parser,
            adapter.get_legal_actions,
        )


def _probe_template_ids(adapter: SolverAdapter) -> list[str]:
    """Walk from the initial state until a player node, collecting template_ids."""
    state = adapter.create_initial_state()
    seen: list[str] = []
    for _ in range(64):
        legal = adapter.get_legal_actions(state)
        if legal:
            for a in legal:
                if a.template_id not in seen:
                    seen.append(a.template_id)
            return seen
        if adapter.get_node_type(state) == "chance":
            outcomes = adapter.get_chance_outcomes(state)
            if not outcomes:
                return seen
            state = adapter.apply_chance(state, outcomes[0])
            continue
        break
    return seen
