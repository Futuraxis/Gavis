"""CFR (Counterfactual Regret Minimization) solver.

Uses External Sampling MC-CFR with depth-limited recursion and
rollout-based leaf evaluation for scalability.

  - At updating player's nodes: traverse ALL actions, compute regrets
  - At opponent / chance nodes: sample ONE outcome
  - At depth limit: random rollout to estimate terminal utility
  - CFR+ regret clamping for faster convergence
"""

from __future__ import annotations
import math
import random
from collections import defaultdict
from typing import Optional

from ..core.engine import GameEngine
from ..core.state_graph import ActionInstance, ChanceOutcome, clone_state


class CFR:
    """External Sampling MC-CFR with depth-limited search + rollout.

    Parameters
    ----------
    engine : GameEngine
    iterations : int
    depth_limit : int — max tree depth; beyond this, use rollouts
    rollout_depth : int — steps for random rollout at leaf
    use_cfr_plus : bool — CFR+ non-negative regret clamping
    seed : int or None
    """

    def __init__(
        self,
        engine: GameEngine,
        iterations: int = 1000,
        depth_limit: int = 8,
        rollout_depth: int = 15,
        use_cfr_plus: bool = True,
        seed: Optional[int] = None,
    ):
        self.engine = engine
        self.iterations = iterations
        self.depth_limit = depth_limit
        self.rollout_depth = rollout_depth
        self.use_cfr_plus = use_cfr_plus
        self.rng = random.Random(seed)

        # info_set_key → {'regrets': defaultdict, 'strategy_sum': defaultdict}
        self.info_sets: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, state: dict, verbose: bool = False) -> dict[str, float]:
        """Run CFR iterations and return root average strategy."""
        report_every = max(1, self.iterations // 10)

        for i in range(1, self.iterations + 1):
            for player in ['p_black', 'p_white']:
                s = clone_state(state)
                self._walk(s, player, [1.0, 1.0], 0)

            if verbose and i % report_every == 0:
                s = self.get_strategy(state)
                top = max(s.items(), key=lambda kv: kv[1]) if s else (None, 0)
                print(f'  CFR iter {i:5d}/{self.iterations}  '
                      f'info_sets={len(self.info_sets):6d}  '
                      f'top={top[0]}:{top[1]:.3f}')

        return self.get_strategy(state)

    def get_strategy(self, state: dict) -> dict[str, float]:
        """Average strategy for the current player at a state."""
        actions = self.engine.get_legal_actions(state)
        if not actions:
            return {}

        cp = self.engine.get_current_player(state)
        info_key = self.engine.get_info_set_key(state, cp)
        info = self.info_sets.get(info_key)
        keys = [a.canonical_key for a in actions]

        if info is None:
            return {k: 1.0 / len(keys) for k in keys}

        ss = [info['strategy_sum'].get(k, 0.0) for k in keys]
        total = sum(ss)
        if total <= 0:
            return {k: 1.0 / len(keys) for k in keys}
        return {k: s / total for k, s in zip(keys, ss)}

    def get_action(self, state: dict) -> Optional[ActionInstance]:
        """Deterministic best action from average strategy."""
        strategy = self.get_strategy(state)
        if not strategy:
            return None
        best_key = max(strategy, key=strategy.get)
        for a in self.engine.get_legal_actions(state):
            if a.canonical_key == best_key:
                return a
        return None

    # ------------------------------------------------------------------
    # Recursive walk
    # ------------------------------------------------------------------

    def _walk(
        self,
        state: dict,
        updating_player: str,
        reach: list[float],
        depth: int,
    ) -> float:
        """One CFR tree walk.  Returns counterfactual value for updating_player."""

        # ---- Terminal or depth limit → rollout ----
        if self.engine.is_terminal(state):
            return self.engine.get_utility(state, updating_player)

        if depth >= self.depth_limit:
            return self._rollout(state, updating_player)

        nt = self.engine.get_node_type(state)

        # ---- Chance node: sample one outcome ----
        if nt == 'chance':
            outcomes = self.engine.get_chance_outcomes(state)
            if not outcomes:
                return 0.0
            o = self._sample_outcome(outcomes)
            return self._walk(self.engine.apply_chance(state, o),
                              updating_player, reach, depth + 1)

        # ---- Player node ----
        current_player = self.engine.get_current_player(state)
        if current_player is None:
            return 0.0

        info_key = self.engine.get_info_set_key(state, current_player)
        info = self._get_info(info_key)
        actions = self.engine.get_legal_actions(state)
        if not actions:
            return 0.0

        keys = [a.canonical_key for a in actions]
        strategy = self._regret_matching(info, keys)

        is_updating = (current_player == updating_player)

        if is_updating:
            # ---- TRAVERSE ALL actions ----
            vals = {}
            for i, a in enumerate(actions):
                k = keys[i]
                nr = list(reach)
                pi = 0 if updating_player == 'p_black' else 1
                nr[pi] *= strategy[k]
                vals[k] = self._walk(
                    self.engine.apply_action(state, a),
                    updating_player, nr, depth + 1,
                )

            # Expected value
            ev = sum(strategy[k] * vals[k] for k in keys)

            # Regret update (weighted by opponent reach)
            opp_idx = 1 if updating_player == 'p_black' else 0
            opp_reach = reach[opp_idx]
            for k in keys:
                info['regrets'][k] += opp_reach * (vals[k] - ev)
                if self.use_cfr_plus:
                    info['regrets'][k] = max(0.0, info['regrets'][k])

            # Average strategy update
            my_idx = 0 if updating_player == 'p_black' else 1
            for k in keys:
                info['strategy_sum'][k] += reach[my_idx] * strategy[k]

            return ev

        else:
            # ---- SAMPLE one opponent action ----
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
                self.engine.apply_action(state, sampled_a),
                updating_player, nr, depth + 1,
            )

    # ------------------------------------------------------------------
    # Rollout (leaf evaluation)
    # ------------------------------------------------------------------

    def _rollout(self, state: dict, player: str) -> float:
        """Random rollout from a leaf state. Returns utility for `player`."""
        s = clone_state(state)
        for _ in range(self.rollout_depth):
            if self.engine.is_terminal(s):
                break
            nt = self.engine.get_node_type(s)
            if nt == 'player':
                actions = self.engine.get_legal_actions(s)
                if not actions:
                    break
                s = self.engine.apply_action(s, self.rng.choice(actions))
            elif nt == 'chance':
                _, s = self.engine.sample_chance(s)
            else:
                break
        return self.engine.get_utility(s, player)

    # ------------------------------------------------------------------
    # Regret matching
    # ------------------------------------------------------------------

    def _regret_matching(self, info: dict, keys: list[str]) -> dict[str, float]:
        for k in keys:
            info['regrets'].setdefault(k, 0.0)
        pos = [max(0.0, info['regrets'][k]) for k in keys]
        total = sum(pos)
        if total > 0:
            return {k: r / total for k, r in zip(keys, pos)}
        n = len(keys)
        return {k: 1.0 / n for k in keys}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Utility: exploitability estimation
# ------------------------------------------------------------------

def estimate_exploitability(
    engine: GameEngine,
    cfr: CFR,
    state: dict,
    n_games: int = 100,
    seed: int = 0,
) -> dict:
    """Play CFR (black) vs Random (white). Returns win-rate stats."""
    rng = random.Random(seed)
    wins = losses = draws = 0

    for g in range(n_games):
        s = clone_state(state)
        grng = random.Random(seed + g * 1000)

        while not engine.is_terminal(s):
            nt = engine.get_node_type(s)
            if nt == 'player':
                cp = engine.get_current_player(s)
                if cp == 'p_black':
                    a = cfr.get_action(s)
                else:
                    acts = engine.get_legal_actions(s)
                    a = grng.choice(acts) if acts else None

                if a is None:
                    break
                s = engine.apply_action(s, a)
            elif nt == 'chance':
                _, s = engine.sample_chance(s)
            else:
                break

        w = s['env'].get('winner')
        if w == 'p_black':
            wins += 1
        elif w == 'p_white':
            losses += 1
        else:
            draws += 1

    return {
        'cfr_wins': wins,
        'cfr_losses': losses,
        'draws': draws,
        'cfr_win_rate': wins / n_games,
    }
