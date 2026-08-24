"""Opponent model — how the hybrid solver simulates the adversary.

An opponent model answers one question: given the current state (from
the opponent's information set), what distribution over actions does the
opponent play?  The online search samples actions from this distribution
when it reaches an opponent node.

Four concrete models:
  - UniformModel: no knowledge (default / degenerate games).
  - CFRTableModel: a pre-trained CFR strategy table (equilibrium play).
  - PSROMixModel: a PSRO policy pool solved to a Nash mixture (targeted
    against a class of opponents).
  - EmpiricalModel: counts of a real opponent's actions per info set,
    Laplace-smoothed — the online-learning consumer: it adapts the
    search to the human's observed tendencies.

The model is generic over any ``GameEngine``: it only needs the
engine's ``get_legal_actions`` and, for table lookups, a strategy
mapping from the solver that produced it.
"""

from __future__ import annotations

import random
from typing import Optional, Protocol

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance, State


class OpponentModel(Protocol):
    """Protocol for opponent action distributions."""

    def action_distribution(self, engine: GameEngine, state: State) -> dict[str, float]:
        """Return ``{canonical_key: probability}`` over legal actions.

        Keys must cover exactly the engine's legal actions at ``state``.
        """


class UniformModel:
    """Opponent plays every legal action with equal probability."""

    def action_distribution(self, engine: GameEngine, state: State) -> dict[str, float]:
        actions = engine.get_legal_actions(state)
        prob = 1.0 / len(actions) if actions else 1.0
        return {a.canonical_key: prob for a in actions}


class TabularPolicyMember:
    """Callable wrapper making a PSRO tabular policy a pool member.

    ``PSROSolver._policy_pool`` stores members as ``np.ndarray`` policies
    of shape ``(state_dim, action_dim)`` keyed by the ``GymAdapter`` state
    encoding — not as callables.  This wrapper bridges the two: calling
    it with ``(engine, state)`` encodes the state, samples one action
    from the policy via ``Agent.step``, and returns its canonical key.
    Illegal actions (policy outside the table) fall back to uniform.

    Encodes with the same ``GymAdapter`` that produced the policy, so the
    state index space always matches.
    """

    def __init__(self, policy, engine: GameEngine, rng: random.Random):
        from ..psro.agent import Agent
        from ..psro.gym_adapter import GymAdapter

        self._policy = policy
        self._gym = GymAdapter(engine)
        self._agent = Agent(policy)
        self._agent.reset_rng(rng.randrange(1 << 30))
        self._rng = rng

    def __call__(self, engine: GameEngine, state: State) -> Optional[str]:
        """Sample one action; return its canonical key (None if no moves)."""
        actions = engine.get_legal_actions(state)
        if not actions:
            return None
        self._gym._state = state  # noqa: SLF001 — point the encoder at the live state
        obs = self._gym._encode_state(state)  # noqa: SLF001 — shared with training
        mask = self._gym.available_actions()
        idx = self._agent.step(obs, amask=mask)
        action = self._gym._int_to_action(idx, actions)  # noqa: SLF001
        if action is None:
            action = self._rng.choice(actions)
        return action.canonical_key


class CFRTableModel:
    """Opponent plays a pre-trained CFR strategy table.

    ``table`` maps ``info_set_key → {action_key: probability}`` (the
    layout produced by ``CFR.solve``).  Information-set lookup uses the
    engine's ``get_info_set_key``, so hidden information is handled by
    the engine's visibility projection.
    """

    def __init__(self, table: dict[str, dict[str, float]], fallback: Optional[OpponentModel] = None):
        self._table = table
        self._fallback = fallback or UniformModel()

    def action_distribution(self, engine: GameEngine, state: State) -> dict[str, float]:
        player = engine.get_current_player(state)
        if player is None:
            return self._fallback.action_distribution(engine, state)
        info_key = engine.get_info_set_key(state, player)
        strategy = self._table.get(info_key)
        if not strategy:
            return self._fallback.action_distribution(engine, state)
        return dict(strategy)


class EmpiricalModel:
    """Opponent plays the Laplace-smoothed empirical distribution of real play.

    ``table`` maps ``info_set_key → {action_key: count}`` built from
    observed opponent decisions (online-learning signals).  The counts
    are smoothed with a Dirichlet prior (``prior_alpha`` per legal
    action), so a single observation never makes an action certain and
    unseen info sets fall back to ``UniformModel`` — the same fallback
    pattern as ``CFRTableModel``.

    Keys must cover exactly the engine's legal actions; the smoothing
    is computed over the legal set at query time, so actions recorded in
    counts but illegal at a state are handled naturally.
    """

    def __init__(
        self,
        table: dict[str, dict[str, int]],
        prior_alpha: float = 1.0,
        fallback: Optional[OpponentModel] = None,
    ) -> None:
        self._table = table
        self._prior_alpha = max(0.0, prior_alpha)
        self._fallback = fallback or UniformModel()

    def action_distribution(self, engine: GameEngine, state: State) -> dict[str, float]:
        player = engine.get_current_player(state)
        if player is None:
            return self._fallback.action_distribution(engine, state)
        info_key = engine.get_info_set_key(state, player)
        counts = self._table.get(info_key)
        if not counts:
            return self._fallback.action_distribution(engine, state)
        actions = engine.get_legal_actions(state)
        if not actions:
            return {}
        total = sum(counts.values())
        if total <= 0:
            return UniformModel().action_distribution(engine, state)
        alpha = self._prior_alpha
        legal = len(actions)
        dist: dict[str, float] = {}
        for action in actions:
            count = counts.get(action.canonical_key, 0)
            dist[action.canonical_key] = (count + alpha) / (total + alpha * legal)
        # Normalize over the legal set (sum is 1 by construction, but keep
        # the contract exact).
        total_prob = sum(dist.values())
        if total_prob <= 0:
            return UniformModel().action_distribution(engine, state)
        return {k: v / total_prob for k, v in dist.items()}

    def coverage(self) -> int:
        """Number of info sets with recorded counts."""
        return len(self._table)

    def merge(self, counts: dict[str, dict[str, int]]) -> None:
        """Merge additional per-info-set counts into the table (incremental)."""
        for info_key, actions in counts.items():
            bucket = self._table.setdefault(info_key, {})
            for action_key, n in actions.items():
                bucket[action_key] = bucket.get(action_key, 0) + int(n)


class PSROMixModel:
    """Opponent plays a PSRO pool solved to a Nash mixture.

    ``pool`` is a list of callables ``(engine, state) → canonical_key``
    (each a member strategy); ``weights`` are the mixture probabilities
    from the meta-game Nash solution.  Each action samples one member
    strategy by weight, then lets it pick.
    """

    def __init__(self, pool: list, weights: list[float], rng: random.Random):
        self._pool = pool
        self._weights = weights
        self._rng = rng

    def action_distribution(self, engine: GameEngine, state: State) -> dict[str, float]:
        """Monte-Carlo mixture estimate over a few samples per member."""
        actions = engine.get_legal_actions(state)
        if not actions:
            return {}
        counts = {a.canonical_key: 0.0 for a in actions}
        samples_per_member = max(1, 8 // max(1, len(self._pool)))
        for weight, member in zip(self._weights, self._pool):
            for _ in range(samples_per_member):
                key = member(engine, state)
                if key in counts:
                    counts[key] += weight / samples_per_member
        total = sum(counts.values())
        if total <= 0:
            return UniformModel().action_distribution(engine, state)
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
