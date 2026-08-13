"""GameEncoder — per-game observation encoding for MARL solvers.

Turns each adapter's structured observation dict into a fixed-size
float32 vector.  ``encode_global`` concatenates every player's vector
in fixed ``rules['players']`` order, giving the CTDE joint state used by
the QMix mixing network and the HAPPO critic.

--- Layouts ---

- mahjong (214+N): 34-tile counts × (hand, meld, own discard, opp discard,
  last_discard one-hot, last_drawn one-hot), meld type counts, wall/136,
  phase one-hot, turn one-hot
- texas_holdem (383): hole + community one-hots, street/phase, stacks,
  committed, folded, call_to, last_action, pot
- moon_chess (38): delegates to ``adapter.get_feature_vector``
- unknown adapters: flattened numeric fields of the observation dict
"""

from __future__ import annotations

import numpy as np

from layer2_engine.interfaces.solver_adapter import SolverAdapter, State

TILE_IDS = [f"{s}{r}" for s in "mps" for r in range(1, 10)] + [f"z{r}" for r in range(1, 8)]
MAHJONG_PHASES = ("deal", "action", "claim", "draw", "gang_draw", "game_over")
TEXAS_PHASES = ("betting", "showdown", "game_over")
TEXAS_LAST_ACTIONS = ("fold", "call", "raise", "none")
MELD_TYPES = ("chi", "peng", "gang")


class GameEncoder:
    """Encodes observations and joint states for one adapter/game.

    Parameters
    ----------
    adapter : SolverAdapter
    players : list[str]
        Agent ids in fixed order (``rules['players']``).
    """

    def __init__(self, adapter: SolverAdapter, players: list[str]):
        self._adapter = adapter
        self._players = list(players)
        self._tile_index: dict[str, int] | None = None
        self._card_index: dict[str, int] | None = None

    # ── Meta ─────────────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Size of one player's observation vector."""
        raise NotImplementedError

    @property
    def global_dim(self) -> int:
        """Size of the joint state (``N * obs_dim``)."""
        return len(self._players) * self.obs_dim

    # ── Encoding ─────────────────────────────────────────────────────

    def encode_obs(self, state: State, player: str) -> np.ndarray:
        """Observation vector for ``player`` at ``state`` (float32)."""
        raise NotImplementedError

    def encode_global(self, state: State) -> np.ndarray:
        """Joint state: concatenated per-player observations."""
        return np.concatenate([self.encode_obs(state, p) for p in self._players]).astype(np.float32)

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def build_from_adapter(cls, adapter: SolverAdapter, players: list[str]) -> "GameEncoder":
        """Build the encoder matching ``adapter``'s game."""
        from layer2_engine.games.mahjong.mahjong_adapter import MahjongAdapter
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
        from layer2_engine.games.texas_holdem.texas_env_adapter import TexasHoldemAdapter

        if isinstance(adapter, MoonChessAdapter):
            return _MoonChessEncoder(adapter, players)
        if isinstance(adapter, MahjongAdapter):
            return _MahjongEncoder(adapter, players)
        if isinstance(adapter, TexasHoldemAdapter):
            return _TexasEncoder(adapter, players)
        return _GenericEncoder(adapter, players)


class _MoonChessEncoder(GameEncoder):
    """Delegates to ``MoonChessAdapter.get_feature_vector`` (38-dim)."""

    @property
    def obs_dim(self) -> int:
        return 38

    def encode_obs(self, state: State, player: str) -> np.ndarray:
        return np.asarray(self._adapter.get_feature_vector(state, player), dtype=np.float32)


class _MahjongEncoder(GameEncoder):
    """34-tile count / one-hot encoding built from ``get_observation``.

    Uses only observation fields and never ``obs['legal']`` (which the
    adapter leaves empty for the claimer during a claim phase).
    """

    def __init__(self, adapter: SolverAdapter, players: list[str]):
        super().__init__(adapter, players)
        constants = getattr(adapter, "_constants", {})
        tile_ids = constants.get("tile_ids") or []
        seen: list[str] = []
        for t in tile_ids:
            if t not in seen:
                seen.append(t)
        self._tiles = seen or TILE_IDS
        self._tile_index = {t: i for i, t in enumerate(self._tiles)}
        self._n_players = len(players)

    @property
    def obs_dim(self) -> int:
        # 6 × 34 tiles + 3 meld types + wall + 6 phases + N-player turn
        return 6 * len(self._tiles) + 3 + 1 + len(MAHJONG_PHASES) + self._n_players

    def encode_obs(self, state: State, player: str) -> np.ndarray:
        # 直读状态数组，绕开 get_observation（其 legal 部分会触发完整的
        # 规则求值——训练热路径上每步可省 ~15ms）
        arrs = state.get("_arrays", {})
        env = state.get("env", {})
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        n_tiles = len(self._tiles)

        def add_counts(slot: int, tiles: list, cap: float) -> None:
            for t in tiles:
                idx = self._tile_index.get(t)
                if idx is not None:
                    vec[slot + idx] = min(vec[slot + idx] + 1.0, cap)

        # Hand
        add_counts(0 * n_tiles, arrs.get(f"hand_{player}", []), 4.0)
        vec[0 * n_tiles : 1 * n_tiles] /= 4.0
        # Melds
        melds = arrs.get(f"melds_{player}", [])
        for meld in melds:
            if isinstance(meld, dict):
                for t in meld.get("tiles", []):
                    idx = self._tile_index.get(t)
                    if idx is not None:
                        vec[1 * n_tiles + idx] = min(vec[1 * n_tiles + idx] + 1.0, 4.0)
                mtype = str(meld.get("type", ""))
                mtype = mtype.replace("concealed_", "").replace("added_", "")
                if mtype in MELD_TYPES:
                    vec[2 * n_tiles + MELD_TYPES.index(mtype)] = min(
                        vec[2 * n_tiles + MELD_TYPES.index(mtype)] + 1.0, 4.0
                    )
        vec[1 * n_tiles : 2 * n_tiles] /= 4.0
        vec[2 * n_tiles : 2 * n_tiles + 3] /= 4.0
        # Own discards
        add_counts(3 * n_tiles, arrs.get(f"discard_{player}", []), 4.0)
        vec[3 * n_tiles : 4 * n_tiles] /= 4.0
        # Opponents' discards (summed, capped at 8)
        opp = np.zeros(n_tiles, dtype=np.float32)
        for p in self._players:
            if p == player:
                continue
            for t in arrs.get(f"discard_{p}", []):
                idx = self._tile_index.get(t)
                if idx is not None:
                    opp[idx] = min(opp[idx] + 1.0, 8.0)
        vec[4 * n_tiles : 5 * n_tiles] = opp / 8.0
        # Last discard / last drawn one-hots
        for slot, key in ((5 * n_tiles, "last_discard"), (6 * n_tiles - 34, "last_drawn")):
            tile = env.get(key)
            idx = self._tile_index.get(tile) if tile else None
            if idx is not None:
                vec[slot + idx] = 1.0
        # Wall
        wall = int(env.get("wall_count", 0) or 0)
        vec[6 * n_tiles] = wall / 136.0
        # Phase one-hot
        phase = str(env.get("phase", ""))
        pidx = MAHJONG_PHASES.index(phase) if phase in MAHJONG_PHASES else 0
        vec[6 * n_tiles + 1 + pidx] = 1.0
        # Turn one-hot
        turn = env.get("turn")
        if turn in self._players:
            vec[6 * n_tiles + 1 + len(MAHJONG_PHASES) + self._players.index(turn)] = 1.0
        return vec


class _TexasEncoder(GameEncoder):
    """One-hot card / scalar encoding for heads-up Texas Hold'em."""

    def __init__(self, adapter: SolverAdapter, players: list[str]):
        super().__init__(adapter, players)
        constants = getattr(adapter, "_constants", {})
        cards = constants.get("card_ids") or []
        self._cards = list(cards)
        self._card_index = {c: i for i, c in enumerate(self._cards)}

    @property
    def obs_dim(self) -> int:
        n_cards = len(self._cards)
        # hole(2×52) + community(5×52) + street(4) + phase(3) + 6 scalars
        # + call_to + last_action(4) + pot
        return 7 * n_cards + 4 + 3 + 6 + 1 + 4 + 1

    def encode_obs(self, state: State, player: str) -> np.ndarray:
        obs = self._adapter.get_observation(state, player)
        n_cards = len(self._cards)
        vec = np.zeros(self.obs_dim, dtype=np.float32)

        def add_one_hot(slot: int, cards: list) -> None:
            for c in cards:
                idx = self._card_index.get(c)
                if idx is not None:
                    vec[slot + idx] = 1.0

        add_one_hot(0 * n_cards, list(obs.get("hole", [])))
        add_one_hot(1 * n_cards, list(obs.get("community", [])))
        # Street / phase
        street = int(obs.get("street", 0) or 0)
        if 0 <= street < 4:
            vec[7 * n_cards + street] = 1.0
        phase = str(obs.get("phase", ""))
        pidx = TEXAS_PHASES.index(phase) if phase in TEXAS_PHASES else 0
        vec[7 * n_cards + 4 + pidx] = 1.0
        # Scalars
        base = 7 * n_cards + 4 + 3
        vec[base + 0] = float(obs.get("my_stack", 0) or 0) / 100.0
        vec[base + 1] = float(obs.get("opp_stack", 0) or 0) / 100.0
        vec[base + 2] = float(obs.get("my_committed", 0) or 0) / 100.0
        vec[base + 3] = float(obs.get("opp_committed", 0) or 0) / 100.0
        vec[base + 4] = 1.0 if obs.get("my_folded") else 0.0
        vec[base + 5] = 1.0 if obs.get("opp_folded") else 0.0
        vec[base + 6] = float(obs.get("last_call_to", 0) or 0) / 100.0
        # Last action one-hot
        last = str(obs.get("last_action") or "none")
        if last not in TEXAS_LAST_ACTIONS:
            last = "none"
        vec[base + 7 + TEXAS_LAST_ACTIONS.index(last)] = 1.0
        # Pot
        vec[base + 7 + len(TEXAS_LAST_ACTIONS)] = float(obs.get("pot", 0) or 0) / 100.0
        return vec


class _GenericEncoder(GameEncoder):
    """Fallback: flatten numeric / boolean fields of the obs dict."""

    def __init__(self, adapter: SolverAdapter, players: list[str]):
        super().__init__(adapter, players)
        probe = adapter.get_observation(adapter.create_initial_state(), players[0])
        self._fields = [k for k, v in probe.items() if isinstance(v, (int, float, bool))]

    @property
    def obs_dim(self) -> int:
        return len(self._fields)

    def encode_obs(self, state: State, player: str) -> np.ndarray:
        obs = self._adapter.get_observation(state, player)
        return np.asarray([float(obs.get(k, 0.0)) for k in self._fields], dtype=np.float32)
