"""MCTS solver with chance-node support.

Handles the full player→chance→player→... alternating structure
that stochastic games require.  Implements ``SolverBase``.

The search core is generic over ``SolverAdapter``; board-game rollout
heuristics live in ``rollout_policy.BoardHeuristicPolicy`` and are only
applied when the state carries a square ``_arrays.board`` (M-04).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from layer2_engine.core.state_graph import clone_state
from layer2_engine.interfaces.solver_adapter import (
    ActionInstance,
    ChanceOutcome,
    NodeType,
    SolverAdapter,
    State,
)

from ..base import SolverBase, SolverConfig, SolverMetrics
from .rollout_policy import BoardHeuristicPolicy, root_player


@dataclass
class MCTSConfig(SolverConfig):
    budget: int = 5000
    ucb_c: float = 1.414
    rollout_depth: int = 20


@dataclass
class MCTSNode:
    node_type: NodeType
    visits: int = 0
    total_value: float = 0.0
    children: dict = field(default_factory=dict)
    child_actions: dict = field(default_factory=dict)
    child_outcomes: dict = field(default_factory=dict)
    untried_actions: list = field(default_factory=list)
    untried_outcomes: list = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        if self.node_type == "player":
            return len(self.untried_actions) == 0
        return len(self.untried_outcomes) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def avg_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MCTS(SolverBase):
    """Monte Carlo Tree Search with chance-node handling."""

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None):
        # Coerce a plain SolverConfig into an MCTSConfig so solver-specific
        # defaults apply (M-05: `config or MCTSConfig()` silently ignored
        # the config class when a plain SolverConfig was passed).
        if config is None:
            config = MCTSConfig()
        elif not isinstance(config, MCTSConfig):
            config = MCTSConfig(**vars(config))
        super().__init__(adapter, config)
        cfg = self.config
        self.budget = cfg.budget
        self.ucb_c = cfg.ucb_c
        self.rollout_depth = cfg.rollout_depth
        self.rng = random.Random(cfg.seed)
        # Optional rollout policy hook (e.g. a CFR strategy prior from the
        # HybridSolver).  ``fn(state, actions) -> ActionInstance | None``;
        # None falls back to the board heuristic, then to random.
        self.rollout_policy = None
        # Built-in default policy for square-board line-connect games
        # (returns None for anything else).  Shares the search RNG so the
        # rollout random stream stays reproducible per seed.
        self._board_policy = BoardHeuristicPolicy(rng=self.rng)

    @property
    def name(self) -> str:
        return f"MCTS(b={self.budget})"

    def select_action(self, state: State) -> Optional[ActionInstance]:
        root_type = self.adapter.get_node_type(state)
        root = MCTSNode(node_type=root_type)

        if root_type == "player":
            root.untried_actions = self.adapter.get_legal_actions(state)
            if not root.untried_actions:
                return None
        elif root_type == "chance":
            root.untried_outcomes = list(self.adapter.get_chance_outcomes(state))

        for _ in range(self.budget):
            self._iterate(state, root)

        if root_type == "player":
            if not root.children:
                actions = self.adapter.get_legal_actions(state)
                return self.rng.choice(actions) if actions else None
            best_key = max(root.children, key=lambda k: root.children[k].visits)
            return root.child_actions.get(best_key)
        return None

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        """MCTS is a search algorithm — training is a no-op."""
        return SolverMetrics(episodes=0, win_rate=0.0, avg_return=0.0)

    # ── Internal ──────────────────────────────────────────────────

    def _iterate(self, root_state: dict, root: MCTSNode):
        state = clone_state(root_state)
        node = root
        path: list = [(None, node)]

        # Selection
        while not node.is_leaf() and node.is_fully_expanded():
            if node.node_type == "player":
                key = self._select_ucb1_key(node)
                if key is None:
                    break
                action = node.child_actions[key]
                state = self.adapter.apply_action(state, action)
                node = node.children[key]
                path.append((key, node))
            elif node.node_type == "chance":
                outcome = self._select_chance(node, state)
                if outcome is None:
                    break
                state = self.adapter.apply_chance(state, outcome)
                node = node.children[outcome.key]
                path.append((outcome.key, node))
            else:
                break

        # Expansion. _expand returns (child, child_state): the rollout must
        # start from the EXPANDED state, not the pre-expansion one — using
        # the old state shifts the value perspective by one ply, which
        # flips the UCB signal (winning rollouts get scored as losses).
        if node.node_type != "terminal" and not node.is_fully_expanded():
            expanded = self._expand(node, state)
            if expanded is not None:
                node, state = expanded
                path.append((None, node))

        # Simulation
        value = self._rollout(state)

        # Backpropagation
        self._backpropagate(path, value)

    def _select_ucb1_key(self, node: MCTSNode) -> Optional[str]:
        best_key = None
        best_ucb = -float("inf")

        for key, child in node.children.items():
            if child.visits == 0:
                return key
            exploitation = child.total_value / child.visits
            # Backprop stores each node's value from ITS OWN player's
            # perspective (flipping per player node).  In a zero-sum game
            # the parent's perspective on a player child is the negation —
            # without this flip, UCB maximizes the OPPONENT's utility and
            # the search actively helps the adversary.
            if child.node_type == "player":
                exploitation = -exploitation
            exploration = self.ucb_c * math.sqrt(math.log(node.visits + 1) / child.visits)
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_key = key

        return best_key

    def _select_chance(self, node: MCTSNode, state: dict) -> Optional[ChanceOutcome]:
        # Reuse the node's cached outcomes — avoids a full adapter re-query
        # (context build + probability eval) on every selection step.
        outcomes = list(node.child_outcomes.values())
        if not outcomes:
            outcomes = list(node.untried_outcomes)
        if not outcomes:
            outcomes = self.adapter.get_chance_outcomes(state)
        return self._sample_outcome(outcomes, self.rng)

    @staticmethod
    def _sample_outcome(outcomes: list[ChanceOutcome], rng: random.Random) -> Optional[ChanceOutcome]:
        """Probability-weighted sample, normalized — tolerates probability
        vectors that do not sum to exactly 1 (no tail bias toward the
        last outcome)."""
        if not outcomes:
            return None
        total = sum(o.probability for o in outcomes)
        if total <= 0:
            return rng.choice(outcomes)
        r = rng.random() * total
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return o
        return outcomes[-1]

    def _expand(self, node: MCTSNode, state: dict):
        """Expand one child; returns ``(child, child_state)`` or None."""
        if node.node_type == "player":
            return self._expand_player(node, state)
        elif node.node_type == "chance":
            return self._expand_chance(node, state)
        return None

    def _expand_player(self, node: MCTSNode, state: dict):
        if not node.untried_actions:
            return None
        action = node.untried_actions.pop()
        key = action.canonical_key
        new_state = self.adapter.apply_action(state, action)
        child_type = self.adapter.get_node_type(new_state)
        child = MCTSNode(node_type=child_type)
        if child_type == "player":
            child.untried_actions = self.adapter.get_legal_actions(new_state)
        elif child_type == "chance":
            child.untried_outcomes = list(self.adapter.get_chance_outcomes(new_state))
        node.children[key] = child
        node.child_actions[key] = action
        return child, new_state

    def _expand_chance(self, node: MCTSNode, state: dict):
        if not node.untried_outcomes:
            return None
        # Each outcome is applied exactly once: the child states built here
        # are cached and the sampled outcome descends into ITS cached state
        # (a second apply_chance could draw different internal randomness
        # and desync the tree from the sim).
        child_states: dict[str, dict] = {}
        for outcome in node.untried_outcomes:
            child_state = self.adapter.apply_chance(state, outcome)
            child_states[outcome.key] = child_state
            child_type = self.adapter.get_node_type(child_state)
            child = MCTSNode(node_type=child_type)
            if child_type == "player":
                child.untried_actions = self.adapter.get_legal_actions(child_state)
            elif child_type == "chance":
                child.untried_outcomes = list(self.adapter.get_chance_outcomes(child_state))
            node.children[outcome.key] = child
            node.child_outcomes[outcome.key] = outcome
        node.untried_outcomes.clear()

        # Sample which child the expansion descends into — from the node's
        # own outcome objects, no adapter re-query needed.
        chosen = self._sample_outcome(list(node.child_outcomes.values()), self.rng)
        if chosen is None:
            return None
        return node.children.get(chosen.key), child_states[chosen.key]

    def _rollout(self, state: dict) -> float:
        sim_state = clone_state(state)
        terminal = False
        for _ in range(self.rollout_depth):
            # get_node_type already folds in is_terminal — one call, not two.
            nt = self.adapter.get_node_type(sim_state)
            if nt == "player":
                actions = self.adapter.get_legal_actions(sim_state)
                if not actions:
                    break
                chosen = None
                if self.rollout_policy is not None:
                    chosen = self.rollout_policy(sim_state, actions)
                if chosen is None:
                    chosen = self._board_policy.choose(sim_state, actions)
                if chosen is None:
                    chosen = self.rng.choice(actions)
                sim_state = self.adapter.apply_action(sim_state, chosen)
            elif nt == "chance":
                outcomes = self.adapter.get_chance_outcomes(sim_state)
                chosen = self._sample_outcome(outcomes, self.rng)
                if chosen is None:
                    break
                sim_state = self.adapter.apply_chance(sim_state, chosen)
            else:
                terminal = True
                break

        if terminal or self.adapter.is_terminal(sim_state):
            player = root_player(state, self.adapter)
            if player is None:
                return 0.0
            return self.adapter.get_utility(sim_state, player)
        # Non-terminal at depth limit: board games get the threat-gap
        # heuristic (clamped to ±0.5); everything else gets the neutral 0.
        player = root_player(state, self.adapter)
        if player is None:
            return 0.0
        return self._board_policy.leaf_value(sim_state, player)

    def _backpropagate(self, path: list, value: float):
        for _key, node in reversed(path):
            node.visits += 1
            node.total_value += value
            if node.node_type == "player":
                value = -value
