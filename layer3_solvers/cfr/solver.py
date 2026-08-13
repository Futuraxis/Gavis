"""CFR (Counterfactual Regret Minimization) solver.

Uses External Sampling MC-CFR with depth-limited recursion and
rollout-based leaf evaluation.  Implements ``SolverBase``.

Player handling is generic: players are discovered from the rules'
utility section and mapped to dynamic reach-array indices — no
``p_black``/``p_white`` assumptions anywhere.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from layer2_engine.core.state_graph import clone_state
from layer2_engine.interfaces.solver_adapter import (
    ActionInstance,
    ChanceOutcome,
    SolverAdapter,
    State,
)

from ..base import SolverBase, SolverConfig, SolverMetrics


@dataclass
class CFRConfig(SolverConfig):
    iterations: int = 1000
    depth_limit: int = 8
    rollout_depth: int = 15
    use_cfr_plus: bool = True


class CFR(SolverBase):
    """External Sampling MC-CFR with depth-limited search + rollout."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None):
        # Coerce a plain SolverConfig into a CFRConfig so solver-specific
        # defaults apply (passing SolverConfig(seed=...) must not silently
        # drop the CFR parameter defaults).
        if config is None:
            config = CFRConfig()
        elif not isinstance(config, CFRConfig):
            config = CFRConfig(**vars(config))
        super().__init__(adapter, config)
        cfg = self.config
        self.iterations = cfg.iterations
        self.depth_limit = cfg.depth_limit
        self.rollout_depth = cfg.rollout_depth
        self.use_cfr_plus = cfg.use_cfr_plus
        self.rng = random.Random(cfg.seed)

        # info_set_key → {'regrets': defaultdict, 'strategy_sum': defaultdict}
        self.info_sets: dict[str, dict] = {}
        # Player list + reach-array index mapping, discovered once from the
        # adapter's rules (C-01: no hardcoded 'p_black'/'p_white').
        self._players: list[str] = self._discover_players()
        self._player_idx: dict[str, int] = {p: i for i, p in enumerate(self._players)}
        # Monotonic iteration counter for CFR+ weighted averaging.  Grows
        # across warm-started ``solve`` calls; reset() brings it back to 0.
        self._iter: int = 0

    @property
    def name(self) -> str:
        return f"CFR(it={self.iterations},cfr+={self.use_cfr_plus})"

    def select_action(self, state: State) -> Optional[ActionInstance]:
        """Deterministic best action from the average strategy."""
        strategy = self.get_strategy(state)
        if not strategy:
            return None
        best_key = max(strategy, key=strategy.get)
        for a in self.adapter.get_legal_actions(state):
            if a.canonical_key == best_key:
                return a
        return None

    def get_strategy(self, state: dict) -> dict[str, float]:
        """Average strategy for the current player at a state."""
        actions = self.adapter.get_legal_actions(state)
        if not actions:
            return {}

        cp = self.adapter.get_current_player(state)
        info_key = self.adapter.get_info_set_key(state, cp)
        info = self.info_sets.get(info_key)
        keys = [a.canonical_key for a in actions]

        if info is None:
            return {k: 1.0 / len(keys) for k in keys}

        ss = [info["strategy_sum"].get(k, 0.0) for k in keys]
        total = sum(ss)
        if total <= 0:
            return {k: 1.0 / len(keys) for k in keys}
        return {k: s / total for k, s in zip(keys, ss)}

    def solve(self, state: dict, iterations: Optional[int] = None, verbose: bool = False) -> dict[str, float]:
        """Run CFR iterations and return the root average strategy.

        Repeated calls warm-start from the existing info sets (regrets
        and strategy sums are kept).  Call :meth:`reset` first to start
        a clean training run.
        """
        iters = iterations if iterations is not None else self.iterations
        report_every = max(1, iters // 10)
        for _ in range(1, iters + 1):
            self._iter += 1
            for player in self._players:
                s = clone_state(state)
                self._walk(s, player, [1.0] * len(self._players), 0)
            if verbose and self._iter % report_every == 0:
                s = self.get_strategy(state)
                top = max(s.items(), key=lambda kv: kv[1]) if s else (None, 0)
                print(f"  CFR iter {self._iter:5d}  info_sets={len(self.info_sets):6d}  top={top[0]}:{top[1]:.3f}")
        return self.get_strategy(state)

    def reset(self) -> None:
        """Discard all learned info sets and the CFR+ iteration count.

        Call between independent training runs (or when the adapter's
        game changes); omitted otherwise for deliberate warm-starting.
        """
        self.info_sets.clear()
        self._iter = 0

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        """Run CFR for ``episodes`` iterations (0 → use config default)."""
        verbose = kwargs.get("verbose", False)
        iters = episodes if episodes > 0 else self.iterations
        state = self.adapter.create_initial_state()
        self.solve(state, iterations=iters, verbose=verbose)

        # Evaluate win rate vs random
        wins = 0
        total = min(100, episodes)
        for _ in range(total):
            s = clone_state(state)
            result = self._play_vs_random(s)
            if result == 1:
                wins += 1
        return SolverMetrics(
            episodes=iters,
            win_rate=wins / max(1, total),
            avg_return=wins / max(1, total),
            extra={"info_sets": len(self.info_sets)},
        )

    # ── Internal ──────────────────────────────────────────────────

    def _discover_players(self) -> list[str]:
        """Discover the player ids from the rules' utility section.

        Sorted for a deterministic reach-array index mapping (set
        iteration order is hash-dependent across processes).
        """
        players: set[str] = set()
        rules = getattr(self.adapter, "rules", None) or {}
        for rule in rules.get("utility", []):
            p = rule.get("player")
            if isinstance(p, str):
                players.add(p)
        return sorted(players) or ["p_black", "p_white"]

    def _get_players(self, state: dict) -> list[str]:
        """Player ids (kept for API compatibility; discovered at init)."""
        return list(self._players)

    def _walk(self, state: dict, updating_player: str, reach: list[float], depth: int) -> float:
        if self.adapter.is_terminal(state):
            return self.adapter.get_utility(state, updating_player)
        if depth >= self.depth_limit:
            return self._rollout(state, updating_player)

        nt = self.adapter.get_node_type(state)

        if nt == "chance":
            outcomes = self.adapter.get_chance_outcomes(state)
            if not outcomes:
                return 0.0
            o = self._sample_outcome(outcomes)
            return self._walk(
                self.adapter.apply_chance(state, o),
                updating_player,
                reach,
                depth + 1,
            )

        current_player = self.adapter.get_current_player(state)
        if current_player is None:
            return 0.0

        info_key = self.adapter.get_info_set_key(state, current_player)
        info = self._get_info(info_key)
        actions = self.adapter.get_legal_actions(state)
        if not actions:
            return 0.0

        keys = [a.canonical_key for a in actions]
        strategy = self._regret_matching(info, keys)
        is_updating = current_player == updating_player

        if is_updating:
            my_idx = self._player_idx.get(updating_player, 0)
            vals = {}
            for i, a in enumerate(actions):
                k = keys[i]
                nr = list(reach)
                nr[my_idx] *= strategy[k]
                vals[k] = self._walk(
                    self.adapter.apply_action(state, a),
                    updating_player,
                    nr,
                    depth + 1,
                )
            ev = sum(strategy[k] * vals[k] for k in keys)
            # Counterfactual reach of everyone except the updating player.
            opp_reach = math.prod(r for j, r in enumerate(reach) if j != my_idx)
            for k in keys:
                info["regrets"][k] += opp_reach * (vals[k] - ev)
                if self.use_cfr_plus:
                    info["regrets"][k] = max(0.0, info["regrets"][k])
            # CFR+ weights the average strategy linearly by iteration t
            # (M-01); vanilla CFR averages uniformly (weight 1).
            iter_weight = self._iter if self.use_cfr_plus else 1
            for k in keys:
                info["strategy_sum"][k] += iter_weight * reach[my_idx] * strategy[k]
            return ev
        else:
            k_list = list(strategy.keys())
            p_list = [strategy[k] for k in k_list]
            sampled_k = self.rng.choices(k_list, weights=p_list, k=1)[0]
            sampled_a = None
            for a in actions:
                if a.canonical_key == sampled_k:
                    sampled_a = a
                    break
            if sampled_a is None:
                sampled_a = actions[0]
            nr = list(reach)
            pi = self._player_idx.get(current_player, 0)
            nr[pi] *= strategy.get(sampled_a.canonical_key, 1.0)
            return self._walk(
                self.adapter.apply_action(state, sampled_a),
                updating_player,
                nr,
                depth + 1,
            )

    def _rollout(self, state: dict, player: str) -> float:
        s = clone_state(state)
        for _ in range(self.rollout_depth):
            if self.adapter.is_terminal(s):
                break
            nt = self.adapter.get_node_type(s)
            if nt == "player":
                actions = self.adapter.get_legal_actions(s)
                if not actions:
                    break
                s = self.adapter.apply_action(s, self._rollout_action(s, actions))
            elif nt == "chance":
                outcomes = self.adapter.get_chance_outcomes(s)
                if not outcomes:
                    break
                o = self._sample_outcome(outcomes)
                s = self.adapter.apply_chance(s, o)
            else:
                break
        if self.adapter.is_terminal(s):
            return self.adapter.get_utility(s, player)
        return self._leaf_heuristic(s, player)

    def _rollout_action(self, state: dict, actions: list) -> ActionInstance:
        """Sample a rollout move from the current regret-matching strategy.

        Falls back to uniform random for info sets never visited by the
        walk (equivalent — all regrets are still 0 there).  Sampling from
        the live strategy instead of uniform random lowers rollout
        variance as training progresses.
        """
        cp = self.adapter.get_current_player(state)
        if cp is not None:
            info = self.info_sets.get(self.adapter.get_info_set_key(state, cp))
            if info is not None:
                strategy = self._regret_matching(info, [a.canonical_key for a in actions])
                return self.rng.choices(actions, weights=[strategy[a.canonical_key] for a in actions], k=1)[0]
        return self.rng.choice(actions)

    def _leaf_heuristic(self, state: dict, player: str) -> float:
        """Threat-gap evaluation for depth-limited non-terminal leaves.

        ``get_utility`` returns 0.0 for non-terminal states, which would
        bias every truncated leaf toward zero (M-02).  On small square
        boards the number of "one move short of a win" lines is a cheap,
        informative estimate; elsewhere 0.0 is kept as the neutral prior.
        """
        board = state.get("_arrays", {}).get("board", [])
        if not board or len(board) > 25:
            return 0.0
        bs = int(len(board) ** 0.5)
        if bs * bs != len(board):
            return 0.0
        win_len = int(state.get("_constants", {}).get("win_length", 3))
        players = state.get("_players", [])
        ids = [p["id"] for p in players] if players and isinstance(players[0], dict) else players
        opponent = next((x for x in ids if x != player), None)
        if opponent is None:
            return 0.0

        mine = opp = 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for r in range(bs):
                for c in range(bs):
                    line = [(r + k * dr, c + k * dc) for k in range(win_len)]
                    if any(not (0 <= rr < bs and 0 <= cc < bs) for rr, cc in line):
                        continue
                    vals = [board[rr * bs + cc] for rr, cc in line]
                    if vals.count(None) != 1:
                        continue
                    if vals.count(player) == win_len - 1:
                        mine += 1
                    elif vals.count(opponent) == win_len - 1:
                        opp += 1
        gap = (mine - opp) / max(1, mine + opp) if mine + opp else 0.0
        return max(-1.0, min(1.0, gap)) * 0.5

    def _regret_matching(self, info: dict, keys: list[str]) -> dict[str, float]:
        for k in keys:
            info["regrets"].setdefault(k, 0.0)
        pos = [max(0.0, info["regrets"][k]) for k in keys]
        total = sum(pos)
        if total > 0:
            return {k: r / total for k, r in zip(keys, pos)}
        n = len(keys)
        return {k: 1.0 / n for k in keys}

    def _get_info(self, key: str) -> dict:
        if key not in self.info_sets:
            self.info_sets[key] = {
                "regrets": defaultdict(float),
                "strategy_sum": defaultdict(float),
            }
        return self.info_sets[key]

    def _sample_outcome(self, outcomes: list[ChanceOutcome]) -> ChanceOutcome:
        """Probability-weighted outcome sample (normalized — tolerates
        probability vectors that do not sum to exactly 1)."""
        total = sum(o.probability for o in outcomes)
        if total <= 0:
            return self.rng.choice(outcomes)
        r = self.rng.random() * total
        c = 0.0
        for o in outcomes:
            c += o.probability
            if r < c:
                return o
        return outcomes[-1]

    def _play_vs_random(self, state: dict) -> int:
        """Play one game: CFR as the first player vs random others. 1/0/-1."""
        s = clone_state(state)
        grng = random.Random()
        me = self._players[0]
        while not self.adapter.is_terminal(s):
            nt = self.adapter.get_node_type(s)
            if nt == "player":
                cp = self.adapter.get_current_player(s)
                if cp == me:
                    a = self.select_action(s)
                else:
                    acts = self.adapter.get_legal_actions(s)
                    a = grng.choice(acts) if acts else None
                if a is None:
                    break
                s = self.adapter.apply_action(s, a)
            elif nt == "chance":
                outcomes = self.adapter.get_chance_outcomes(s)
                if outcomes:
                    o = self._sample_outcome(outcomes)
                    s = self.adapter.apply_chance(s, o)
            else:
                break
        w = s["env"].get("winner")
        if w == me:
            return 1
        if w in self._players:
            return -1
        return 0
