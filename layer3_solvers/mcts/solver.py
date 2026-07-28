"""MCTS solver with chance-node support.

Handles the full player→chance→player→... alternating structure
that stochastic games require.  Implements ``SolverBase``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from layer2_engine.interfaces.solver_adapter import (
    SolverAdapter,
    State,
    ActionInstance,
    ChanceOutcome,
    NodeType,
)
from layer2_engine.core.state_graph import clone_state
from ..base import SolverBase, SolverConfig, SolverMetrics


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
        if self.node_type == 'player':
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
        super().__init__(adapter, config or MCTSConfig())
        cfg = self.config
        self.budget = getattr(cfg, 'budget', 5000)
        self.ucb_c = getattr(cfg, 'ucb_c', 1.414)
        self.rollout_depth = getattr(cfg, 'rollout_depth', 20)
        self.rng = random.Random(cfg.seed)

    @property
    def name(self) -> str:
        return f"MCTS(b={self.budget})"

    def select_action(self, state: State) -> Optional[ActionInstance]:
        root_type = self.adapter.get_node_type(state)
        root = MCTSNode(node_type=root_type)

        if root_type == 'player':
            root.untried_actions = self.adapter.get_legal_actions(state)
            if not root.untried_actions:
                return None
        elif root_type == 'chance':
            root.untried_outcomes = list(self.adapter.get_chance_outcomes(state))

        for _ in range(self.budget):
            self._iterate(state, root)

        if root_type == 'player':
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
            if node.node_type == 'player':
                key = self._select_ucb1_key(node)
                if key is None:
                    break
                action = node.child_actions[key]
                state = self.adapter.apply_action(state, action)
                node = node.children[key]
                path.append((key, node))
            elif node.node_type == 'chance':
                outcome = self._select_chance(node, state)
                if outcome is None:
                    break
                state = self.adapter.apply_chance(state, outcome)
                node = node.children[outcome.key]
                path.append((outcome.key, node))
            else:
                break

        # Expansion
        if node.node_type != 'terminal' and not node.is_fully_expanded():
            node = self._expand(node, state)
            if node is not None:
                path.append((None, node))

        # Simulation
        value = self._rollout(state)

        # Backpropagation
        self._backpropagate(path, value)

    def _select_ucb1_key(self, node: MCTSNode) -> Optional[str]:
        best_key = None
        best_ucb = -float('inf')

        for key, child in node.children.items():
            if child.visits == 0:
                return key
            exploitation = child.total_value / child.visits
            exploration = self.ucb_c * math.sqrt(
                math.log(node.visits + 1) / child.visits
            )
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_key = key

        return best_key

    def _select_chance(self, node: MCTSNode, state: dict) -> Optional[ChanceOutcome]:
        outcomes = self.adapter.get_chance_outcomes(state)
        r = self.rng.random()
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return o
        return outcomes[-1] if outcomes else None

    def _expand(self, node: MCTSNode, state: dict) -> Optional[MCTSNode]:
        if node.node_type == 'player':
            return self._expand_player(node, state)
        elif node.node_type == 'chance':
            return self._expand_chance(node, state)
        return None

    def _expand_player(self, node: MCTSNode, state: dict) -> Optional[MCTSNode]:
        if not node.untried_actions:
            return None
        action = node.untried_actions.pop()
        key = action.canonical_key
        new_state = self.adapter.apply_action(state, action)
        child_type = self.adapter.get_node_type(new_state)
        child = MCTSNode(node_type=child_type)
        if child_type == 'player':
            child.untried_actions = self.adapter.get_legal_actions(new_state)
        elif child_type == 'chance':
            child.untried_outcomes = list(self.adapter.get_chance_outcomes(new_state))
        node.children[key] = child
        node.child_actions[key] = action
        return child

    def _expand_chance(self, node: MCTSNode, state: dict) -> Optional[MCTSNode]:
        if not node.untried_outcomes:
            return None
        for outcome in node.untried_outcomes:
            child_state = self.adapter.apply_chance(state, outcome)
            child_type = self.adapter.get_node_type(child_state)
            child = MCTSNode(node_type=child_type)
            if child_type == 'player':
                child.untried_actions = self.adapter.get_legal_actions(child_state)
            elif child_type == 'chance':
                child.untried_outcomes = list(self.adapter.get_chance_outcomes(child_state))
            node.children[outcome.key] = child
            node.child_outcomes[outcome.key] = outcome
        node.untried_outcomes.clear()

        outcomes = self.adapter.get_chance_outcomes(state)
        r = self.rng.random()
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return node.children.get(o.key)
        last_key = outcomes[-1].key if outcomes else None
        return node.children.get(last_key) if last_key else None

    def _rollout(self, state: dict) -> float:
        sim_state = clone_state(state)
        for _ in range(self.rollout_depth):
            if self.adapter.is_terminal(sim_state):
                break
            nt = self.adapter.get_node_type(sim_state)
            if nt == 'player':
                actions = self.adapter.get_legal_actions(sim_state)
                if not actions:
                    break
                sim_state = self.adapter.apply_action(sim_state, self.rng.choice(actions))
            elif nt == 'chance':
                # Sample and apply
                outcomes = self.adapter.get_chance_outcomes(sim_state)
                if not outcomes:
                    break
                r = self.rng.random()
                c = 0.0
                chosen = outcomes[-1]
                for o in outcomes:
                    c += o.probability
                    if r < c:
                        chosen = o
                        break
                sim_state = self.adapter.apply_chance(sim_state, chosen)
            else:
                break

        if self.adapter.is_terminal(sim_state):
            current_player = state['env']['turn']['currentPlayerId']
            if self.adapter.get_node_type(state) == 'chance':
                current_player = state['env'].get('lastActor', current_player)
            return self.adapter.get_utility(sim_state, current_player)
        return 0.0

    def _backpropagate(self, path: list, value: float):
        for _key, node in reversed(path):
            node.visits += 1
            node.total_value += value
            if node.node_type == 'player':
                value = -value
