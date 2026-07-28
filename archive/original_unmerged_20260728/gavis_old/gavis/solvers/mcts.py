"""MCTS solver with chance-node support.

Handles the full player→chance→player→... alternating structure
that stochastic gomoku requires.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..core.engine import GameEngine
from ..core.state_graph import ActionInstance, ChanceOutcome, clone_state


@dataclass
class MCTSNode:
    """A node in the MCTS tree.

    Children are keyed by canonical string keys to avoid hashing issues
    with ActionInstance / ChanceOutcome dataclass objects.
    """
    node_type: str          # 'player' | 'chance' | 'terminal'
    visits: int = 0
    total_value: float = 0.0
    children: dict = field(default_factory=dict)       # canonical_key → MCTSNode
    child_actions: dict = field(default_factory=dict)  # canonical_key → ActionInstance (player only)
    child_outcomes: dict = field(default_factory=dict) # canonical_key → ChanceOutcome (chance only)
    untried_actions: list = field(default_factory=list)
    untried_outcomes: list = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        if self.node_type == 'player':
            return len(self.untried_actions) == 0
        elif self.node_type == 'chance':
            return len(self.untried_outcomes) == 0
        return True  # terminal

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def avg_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


class MCTS:
    """Monte Carlo Tree Search with chance-node handling.

    Parameters
    ----------
    engine : GameEngine
    budget : int — iterations per search
    ucb_c : float — UCB1 exploration constant
    rollout_depth : int — max depth for random rollout
    """

    def __init__(
        self,
        engine: GameEngine,
        budget: int = 5000,
        ucb_c: float = 1.414,
        rollout_depth: int = 20,
        seed: Optional[int] = None,
    ):
        self.engine = engine
        self.budget = budget
        self.ucb_c = ucb_c
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, state: dict) -> Optional[ActionInstance]:
        """Run MCTS from the given state, return the best action."""
        root_type = self.engine.get_node_type(state)
        root = MCTSNode(node_type=root_type)

        if root_type == 'player':
            root.untried_actions = self.engine.get_legal_actions(state)
            if not root.untried_actions:
                return None
        elif root_type == 'chance':
            root.untried_outcomes = list(self.engine.get_chance_outcomes(state))

        for _ in range(self.budget):
            self._iterate(state, root)

        if root_type == 'player':
            if not root.children:
                actions = self.engine.get_legal_actions(state)
                return self.rng.choice(actions) if actions else None
            # Find child with most visits
            best_key = max(root.children, key=lambda k: root.children[k].visits)
            return root.child_actions.get(best_key)
        return None

    def action_stats(self, state: dict) -> list[tuple]:
        """Return (action, visits, avg_value) for display/debug."""
        root = MCTSNode(node_type=self.engine.get_node_type(state))
        root_type = root.node_type
        if root_type == 'player':
            root.untried_actions = self.engine.get_legal_actions(state)
        elif root_type == 'chance':
            root.untried_outcomes = list(self.engine.get_chance_outcomes(state))

        for _ in range(self.budget):
            self._iterate(state, root)

        stats = []
        for key, child in root.children.items():
            action = root.child_actions.get(key) or root.child_outcomes.get(key)
            if action is not None:
                stats.append((action, child.visits, child.avg_value))
        stats.sort(key=lambda x: x[1], reverse=True)
        return stats

    # ------------------------------------------------------------------
    # MCTS iteration
    # ------------------------------------------------------------------

    def _iterate(self, root_state: dict, root: MCTSNode):
        """One full MCTS iteration: select → expand → simulate → backprop."""
        state = clone_state(root_state)

        # --- Selection ---
        node = root
        path: list = [(None, node)]  # (action_key, node)

        while not node.is_leaf() and node.is_fully_expanded():
            if node.node_type == 'player':
                key = self._select_ucb1_key(node)
                if key is None:
                    break
                action = node.child_actions[key]
                state = self.engine.apply_action(state, action)
                node = node.children[key]
                path.append((key, node))
            elif node.node_type == 'chance':
                outcome = self._select_chance(node, state)
                if outcome is None:
                    break
                state = self.engine.apply_chance(state, outcome)
                node = node.children[outcome.key]
                path.append((outcome.key, node))
            else:
                break

        # --- Expansion ---
        if node.node_type != 'terminal' and not node.is_fully_expanded():
            node = self._expand(node, state)
            if node is not None:
                path.append((None, node))

        # --- Simulation ---
        value = self._rollout(state)

        # --- Backpropagation ---
        self._backpropagate(path, value)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_ucb1_key(self, node: MCTSNode) -> Optional[str]:
        """Select child key using UCB1. Returns canonical_key string."""
        best_key = None
        best_ucb = -float('inf')

        for key, child in node.children.items():
            if child.visits == 0:
                return key  # prioritize unexplored
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
        """Sample a chance outcome from the prior probability distribution."""
        outcomes = self.engine.get_chance_outcomes(state)
        r = self.rng.random()
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return o
        return outcomes[-1] if outcomes else None

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode, state: dict) -> Optional[MCTSNode]:
        if node.node_type == 'player':
            return self._expand_player(node, state)
        elif node.node_type == 'chance':
            return self._expand_chance(node, state)
        return None

    def _expand_player(self, node: MCTSNode, state: dict) -> Optional[MCTSNode]:
        """Pick one untried action, apply it, and add as child."""
        if not node.untried_actions:
            return None

        action = node.untried_actions.pop()
        key = action.canonical_key

        new_state = self.engine.apply_action(state, action)
        child_type = self.engine.get_node_type(new_state)
        child = MCTSNode(node_type=child_type)

        if child_type == 'player':
            child.untried_actions = self.engine.get_legal_actions(new_state)
        elif child_type == 'chance':
            child.untried_outcomes = list(self.engine.get_chance_outcomes(new_state))

        node.children[key] = child
        node.child_actions[key] = action
        return child

    def _expand_chance(self, node: MCTSNode, state: dict) -> Optional[MCTSNode]:
        """Fully expand all chance outcomes at once (known distribution)."""
        if not node.untried_outcomes:
            return None

        for outcome in node.untried_outcomes:
            child_state = self.engine.apply_chance(state, outcome)
            child_type = self.engine.get_node_type(child_state)
            child = MCTSNode(node_type=child_type)

            if child_type == 'player':
                child.untried_actions = self.engine.get_legal_actions(child_state)
            elif child_type == 'chance':
                child.untried_outcomes = list(self.engine.get_chance_outcomes(child_state))

            node.children[outcome.key] = child
            node.child_outcomes[outcome.key] = outcome

        node.untried_outcomes.clear()

        # Return a random child (weighted by probability) for simulation
        outcomes = self.engine.get_chance_outcomes(state)
        r = self.rng.random()
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return node.children.get(o.key)
        last_key = outcomes[-1].key if outcomes else None
        return node.children.get(last_key) if last_key else None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _rollout(self, state: dict) -> float:
        """Random rollout. Returns utility from perspective of player
        who just acted at the original state."""
        sim_state = clone_state(state)

        for _ in range(self.rollout_depth):
            if self.engine.is_terminal(sim_state):
                break

            nt = self.engine.get_node_type(sim_state)

            if nt == 'player':
                actions = self.engine.get_legal_actions(sim_state)
                if not actions:
                    break
                sim_state = self.engine.apply_action(sim_state, self.rng.choice(actions))
            elif nt == 'chance':
                _, sim_state = self.engine.sample_chance(sim_state)
            else:
                break

        if self.engine.is_terminal(sim_state):
            current_player = state['env']['turn']['currentPlayerId']
            if self.engine.get_node_type(state) == 'chance':
                current_player = state['env'].get(
                    'lastActor', state['env']['turn']['currentPlayerId']
                )
            return self.engine.get_utility(sim_state, current_player)
        return 0.0

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, path: list, value: float):
        """Backpropagate value. Negate at player nodes (zero-sum),
        pass through at chance nodes."""
        for _key, node in reversed(path):
            node.visits += 1
            node.total_value += value
            if node.node_type == 'player':
                value = -value
