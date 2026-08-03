"""Opponent model — how the hybrid solver simulates the adversary.

An opponent model answers one question: given the current state (from
the opponent's information set), what distribution over actions does the
opponent play?  The online search samples actions from this distribution
when it reaches an opponent node.

Three concrete models:
  - UniformModel: no knowledge (default / degenerate games).
  - CFRTableModel: a pre-trained CFR strategy table (equilibrium play).
  - PSROMixModel: a PSRO policy pool solved to a Nash mixture (targeted
    against a class of opponents).

The model is generic over any ``SolverAdapter``: it only needs the
adapter's ``get_legal_actions`` and, for table lookups, a strategy
mapping from the solver that produced it.
"""

from __future__ import annotations

import random
from typing import Optional, Protocol

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter, State


class OpponentModel(Protocol):
    """Protocol for opponent action distributions."""

    def action_distribution(self, adapter: SolverAdapter, state: State) -> dict[str, float]:
        """Return ``{canonical_key: probability}`` over legal actions.

        Keys must cover exactly the adapter's legal actions at ``state``.
        """


class UniformModel:
    """Opponent plays every legal action with equal probability."""

    def action_distribution(self, adapter: SolverAdapter, state: State) -> dict[str, float]:
        actions = adapter.get_legal_actions(state)
        prob = 1.0 / len(actions) if actions else 1.0
        return {a.canonical_key: prob for a in actions}


class TabularPolicyMember:
    """Callable wrapper making a PSRO tabular policy a pool member.

    ``PSROSolver._policy_pool`` stores members as ``np.ndarray`` policies
    of shape ``(state_dim, action_dim)`` keyed by the ``GymAdapter`` state
    encoding — not as callables.  This wrapper bridges the two: calling
    it with ``(adapter, state)`` encodes the state, samples one action
    from the policy via ``Agent.step``, and returns its canonical key.
    Illegal actions (policy outside the table) fall back to uniform.

    Encodes with the same ``GymAdapter`` that produced the policy, so the
    state index space always matches.
    """

    def __init__(self, policy, adapter: SolverAdapter, rng: random.Random):
        from ..psro.agent import Agent
        from ..psro.gym_adapter import GymAdapter

        self._policy = policy
        self._gym = GymAdapter(adapter)
        self._agent = Agent(policy)
        self._agent.reset_rng(rng.randrange(1 << 30))
        self._rng = rng

    def __call__(self, adapter: SolverAdapter, state: State) -> Optional[str]:
        """Sample one action; return its canonical key (None if no moves)."""
        actions = adapter.get_legal_actions(state)
        if not actions:
            return None
        self._gym._state = state  # noqa: SLF001 — point the encoder at the live state
        obs = self._gym._encode_state(state)  # noqa: SLF001 — shared with training
        mask = self._gym.available_actions()
        idx = self._agent.step(obs, Amask=mask)
        action = self._gym._int_to_action(idx, actions)  # noqa: SLF001
        if action is None:
            action = self._rng.choice(actions)
        return action.canonical_key


class CFRTableModel:
    """Opponent plays a pre-trained CFR strategy table.

    ``table`` maps ``info_set_key → {action_key: probability}`` (the
    layout produced by ``CFR.solve``).  Information-set lookup uses the
    adapter's ``get_info_set_key``, so hidden information is handled by
    the engine's visibility projection.
    """

    def __init__(self, table: dict[str, dict[str, float]], fallback: Optional[OpponentModel] = None):
        self._table = table
        self._fallback = fallback or UniformModel()

    def action_distribution(self, adapter: SolverAdapter, state: State) -> dict[str, float]:
        player = adapter.get_current_player(state)
        if player is None:
            return self._fallback.action_distribution(adapter, state)
        info_key = adapter.get_info_set_key(state, player)
        strategy = self._table.get(info_key)
        if not strategy:
            return self._fallback.action_distribution(adapter, state)
        return dict(strategy)


class PSROMixModel:
    """Opponent plays a PSRO pool solved to a Nash mixture.

    ``pool`` is a list of callables ``(adapter, state) → canonical_key``
    (each a member strategy); ``weights`` are the mixture probabilities
    from the meta-game Nash solution.  Each action samples one member
    strategy by weight, then lets it pick.
    """

    def __init__(self, pool: list, weights: list[float], rng: random.Random):
        self._pool = pool
        self._weights = weights
        self._rng = rng

    def action_distribution(self, adapter: SolverAdapter, state: State) -> dict[str, float]:
        """Monte-Carlo mixture estimate over a few samples per member."""
        actions = adapter.get_legal_actions(state)
        if not actions:
            return {}
        counts = {a.canonical_key: 0.0 for a in actions}
        samples_per_member = max(1, 8 // max(1, len(self._pool)))
        for weight, member in zip(self._weights, self._pool):
            for _ in range(samples_per_member):
                key = member(adapter, state)
                if key in counts:
                    counts[key] += weight / samples_per_member
        total = sum(counts.values())
        if total <= 0:
            return UniformModel().action_distribution(adapter, state)
        return {k: v / total for k, v in counts.items()}


def sample_action(
    distribution: dict[str, float],
    actions: list[ActionInstance],
    rng: random.Random,
) -> Optional[ActionInstance]:
    """Sample one action from a keyed distribution (``None`` if empty)."""
    if not actions:
        return None
    by_key = {a.canonical_key: a for a in actions}
    keys = list(distribution.keys())
    probs = [max(0.0, distribution[k]) for k in keys]
    total = sum(probs)
    if total <= 0:
        return rng.choice(actions)
    r = rng.random() * total
    cumsum = 0.0
    for key, prob in zip(keys, probs):
        cumsum += prob
        if r < cumsum:
            return by_key.get(key)
    return by_key.get(keys[-1])
