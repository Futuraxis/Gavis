"""CFR (Counterfactual Regret Minimization) solver.

Uses External Sampling MC-CFR with depth-limited recursion and
rollout-based leaf evaluation.  Implements ``SolverBase``.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from layer2_engine.interfaces.solver_adapter import (
    SolverAdapter,
    State,
    ActionInstance,
    ChanceOutcome,
)
from layer2_engine.core.state_graph import clone_state
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
        super().__init__(adapter, config or CFRConfig())
        cfg = self.config
        self.iterations = getattr(cfg, 'iterations', 1000)
        self.depth_limit = getattr(cfg, 'depth_limit', 8)
        self.rollout_depth = getattr(cfg, 'rollout_depth', 15)
        self.use_cfr_plus = getattr(cfg, 'use_cfr_plus', True)
        self.rng = random.Random(cfg.seed)

        # info_set_key → {'regrets': defaultdict, 'strategy_sum': defaultdict}
        self.info_sets: dict[str, dict] = {}

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

        ss = [info['strategy_sum'].get(k, 0.0) for k in keys]
        total = sum(ss)
        if total <= 0:
            return {k: 1.0 / len(keys) for k in keys}
        return {k: s / total for k, s in zip(keys, ss)}

    def solve(self, state: dict, verbose: bool = False) -> dict[str, float]:
        """Run CFR iterations and return root average strategy."""
        report_every = max(1, self.iterations // 10)
        for i in range(1, self.iterations + 1):
            for player in self._get_players(state):
                s = clone_state(state)
                self._walk(s, player, [1.0, 1.0], 0)
            if verbose and i % report_every == 0:
                s = self.get_strategy(state)
                top = max(s.items(), key=lambda kv: kv[1]) if s else (None, 0)
                print(f'  CFR iter {i:5d}/{self.iterations}  '
                      f'info_sets={len(self.info_sets):6d}  '
                      f'top={top[0]}:{top[1]:.3f}')
        return self.get_strategy(state)

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        """Run CFR iterations.  ``episodes`` is interpreted as iterations."""
        verbose = kwargs.get('verbose', False)
        iters = episodes if episodes > 0 else self.iterations
        state = self.adapter.create_initial_state()
        self.solve(state, verbose=verbose)

        # Evaluate win rate vs random
        wins = 0
        total = min(100, episodes)
        for g in range(total):
            s = clone_state(state)
            result = self._play_vs_random(s)
            if result == 1:
                wins += 1
        return SolverMetrics(
            episodes=iters,
            win_rate=wins / max(1, total),
            avg_return=wins / max(1, total),
            extra={'info_sets': len(self.info_sets)},
        )

    # ── Internal ──────────────────────────────────────────────────

    def _get_players(self, state: dict) -> list[str]:
        """Discover players from the rules utility section."""
        players: set[str] = set()
        for rule in self.adapter.rules.get('utility', []) if hasattr(self.adapter, 'rules') else []:
            p = rule.get('player')
            if isinstance(p, str):
                players.add(p)
        return list(players) or ['p_black', 'p_white']

    def _walk(self, state: dict, updating_player: str, reach: list[float], depth: int) -> float:
        if self.adapter.is_terminal(state):
            return self.adapter.get_utility(state, updating_player)
        if depth >= self.depth_limit:
            return self._rollout(state, updating_player)

        nt = self.adapter.get_node_type(state)

        if nt == 'chance':
            outcomes = self.adapter.get_chance_outcomes(state)
            if not outcomes:
                return 0.0
            o = self._sample_outcome(outcomes)
            return self._walk(
                self.adapter.apply_chance(state, o),
                updating_player, reach, depth + 1,
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
        is_updating = (current_player == updating_player)

        if is_updating:
            vals = {}
            for i, a in enumerate(actions):
                k = keys[i]
                nr = list(reach)
                pi = 0 if updating_player == 'p_black' else 1
                nr[pi] *= strategy[k]
                vals[k] = self._walk(
                    self.adapter.apply_action(state, a),
                    updating_player, nr, depth + 1,
                )
            ev = sum(strategy[k] * vals[k] for k in keys)
            opp_idx = 1 if updating_player == 'p_black' else 0
            opp_reach = reach[opp_idx]
            for k in keys:
                info['regrets'][k] += opp_reach * (vals[k] - ev)
                if self.use_cfr_plus:
                    info['regrets'][k] = max(0.0, info['regrets'][k])
            my_idx = 0 if updating_player == 'p_black' else 1
            for k in keys:
                info['strategy_sum'][k] += reach[my_idx] * strategy[k]
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
            pi = 0 if current_player == 'p_black' else 1
            nr[pi] *= strategy.get(sampled_a.canonical_key, 1.0)
            return self._walk(
                self.adapter.apply_action(state, sampled_a),
                updating_player, nr, depth + 1,
            )

    def _rollout(self, state: dict, player: str) -> float:
        s = clone_state(state)
        for _ in range(self.rollout_depth):
            if self.adapter.is_terminal(s):
                break
            nt = self.adapter.get_node_type(s)
            if nt == 'player':
                actions = self.adapter.get_legal_actions(s)
                if not actions:
                    break
                s = self.adapter.apply_action(s, self.rng.choice(actions))
            elif nt == 'chance':
                outcomes = self.adapter.get_chance_outcomes(s)
                if outcomes:
                    o = self._sample_outcome(outcomes)
                    s = self.adapter.apply_chance(s, o)
            else:
                break
        return self.adapter.get_utility(s, player)

    def _regret_matching(self, info: dict, keys: list[str]) -> dict[str, float]:
        for k in keys:
            info['regrets'].setdefault(k, 0.0)
        pos = [max(0.0, info['regrets'][k]) for k in keys]
        total = sum(pos)
        if total > 0:
            return {k: r / total for k, r in zip(keys, pos)}
        n = len(keys)
        return {k: 1.0 / n for k in keys}

    def _get_info(self, key: str) -> dict:
        if key not in self.info_sets:
            self.info_sets[key] = {
                'regrets': defaultdict(float),
                'strategy_sum': defaultdict(float),
            }
        return self.info_sets[key]

    def _sample_outcome(self, outcomes: list[ChanceOutcome]) -> ChanceOutcome:
        r = self.rng.random()
        c = 0.0
        for o in outcomes:
            c += o.probability
            if r < c:
                return o
        return outcomes[-1]

    def _play_vs_random(self, state: dict) -> int:
        """Play one game with CFR as black vs random as white. Return 1/0/-1."""
        s = clone_state(state)
        grng = random.Random()
        while not self.adapter.is_terminal(s):
            nt = self.adapter.get_node_type(s)
            if nt == 'player':
                cp = self.adapter.get_current_player(s)
                if cp == 'p_black':
                    a = self.select_action(s)
                else:
                    acts = self.adapter.get_legal_actions(s)
                    a = grng.choice(acts) if acts else None
                if a is None:
                    break
                s = self.adapter.apply_action(s, a)
            elif nt == 'chance':
                outcomes = self.adapter.get_chance_outcomes(s)
                if outcomes:
                    o = self._sample_outcome(outcomes)
                    s = self.adapter.apply_chance(s, o)
            else:
                break
        w = s['env'].get('winner')
        if w == 'p_black':
            return 1
        elif w == 'p_white':
            return -1
        return 0
