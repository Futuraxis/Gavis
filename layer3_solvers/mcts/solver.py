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
        # Optional rollout policy hook (e.g. a CFR strategy prior from the
        # HybridSolver).  ``fn(state, actions) -> ActionInstance | None``;
        # None falls back to the built-in heuristic/random choice.
        self.rollout_policy = None

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

        # Expansion. _expand returns (child, child_state): the rollout must
        # start from the EXPANDED state, not the pre-expansion one — using
        # the old state shifts the value perspective by one ply, which
        # flips the UCB signal (winning rollouts get scored as losses).
        if node.node_type != 'terminal' and not node.is_fully_expanded():
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
        best_ucb = -float('inf')

        for key, child in node.children.items():
            if child.visits == 0:
                return key
            exploitation = child.total_value / child.visits
            # Backprop stores each node's value from ITS OWN player's
            # perspective (flipping per player node).  In a zero-sum game
            # the parent's perspective on a player child is the negation —
            # without this flip, UCB maximizes the OPPONENT's utility and
            # the search actively helps the adversary.
            if child.node_type == 'player':
                exploitation = -exploitation
            exploration = self.ucb_c * math.sqrt(
                math.log(node.visits + 1) / child.visits
            )
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
        r = self.rng.random()
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return o
        return outcomes[-1] if outcomes else None

    def _expand(self, node: MCTSNode, state: dict):
        """Expand one child; returns ``(child, child_state)`` or None."""
        if node.node_type == 'player':
            return self._expand_player(node, state)
        elif node.node_type == 'chance':
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
        if child_type == 'player':
            child.untried_actions = self.adapter.get_legal_actions(new_state)
        elif child_type == 'chance':
            child.untried_outcomes = list(self.adapter.get_chance_outcomes(new_state))
        node.children[key] = child
        node.child_actions[key] = action
        return child, new_state

    def _expand_chance(self, node: MCTSNode, state: dict):
        if not node.untried_outcomes:
            return None
        sampled_state = None
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

        # Sample which child the expansion descends into — from the node's
        # own outcome objects, no adapter re-query needed.
        outcomes = list(node.child_outcomes.values())
        r = self.rng.random()
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return node.children.get(o.key), self.adapter.apply_chance(state, o)
        last_key = outcomes[-1].key if outcomes else None
        last_outcome = outcomes[-1] if outcomes else None
        if last_key is None or last_outcome is None:
            return None
        return node.children.get(last_key), self.adapter.apply_chance(state, last_outcome)

    def _rollout(self, state: dict) -> float:
        sim_state = clone_state(state)
        terminal = False
        for _ in range(self.rollout_depth):
            # get_node_type already folds in is_terminal — one call, not two.
            nt = self.adapter.get_node_type(sim_state)
            if nt == 'player':
                actions = self.adapter.get_legal_actions(sim_state)
                if not actions:
                    break
                chosen = None
                if self.rollout_policy is not None:
                    chosen = self.rollout_policy(sim_state, actions)
                if chosen is None:
                    chosen = self._rollout_choice(sim_state, actions)
                sim_state = self.adapter.apply_action(sim_state, chosen)
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
                terminal = True
                break

        if terminal or self.adapter.is_terminal(sim_state):
            current_player = state['env']['turn']
            if isinstance(current_player, dict):
                current_player = current_player.get('currentPlayerId', current_player)
            if self.adapter.get_node_type(state) == 'chance':
                current_player = state['env'].get('lastActor', current_player)
            return self.adapter.get_utility(sim_state, current_player)
        # Non-terminal at depth limit: return a threat heuristic instead of
        # 0. Pure-random/deep rollouts systematically under-value defensive
        # moves on small boards — blocking an immediate threat is decisive
        # but barely moves the terminal frequency, so the search never
        # learns to block.  The threat gap (my "N-1-in-a-line" lines minus
        # the opponent's) captures that signal; clamped to ±0.5 so terminal
        # utilities (±1) keep dominance.  Skipped on large boards, where
        # the full line scan is expensive and the signal is weak anyway.
        board = sim_state.get('_arrays', {}).get('board', [])
        if len(board) <= 25:
            return self._threat_gap(sim_state, state) * 0.5
        return 0.0

    def _threat_gap(self, sim_state: dict, root_state: dict) -> float:
        """Signed threat gap ([-1, 1]) from the rollout-start player's view."""
        current_player = root_state['env']['turn']
        if isinstance(current_player, dict):
            current_player = current_player.get('currentPlayerId', current_player)
        board = sim_state.get('_arrays', {}).get('board', [])
        if not board:
            return 0.0
        bs = int(len(board) ** 0.5)
        win_len = int(sim_state.get('_constants', {}).get('win_length', 3))
        players = sim_state.get('_players', [])
        ids = [p['id'] for p in players] if players and isinstance(players[0], dict) else players
        opponent = ids[1] if current_player == ids[0] else ids[0]

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
                    if vals.count(current_player) == win_len - 1:
                        mine += 1
                    elif vals.count(opponent) == win_len - 1:
                        opp += 1
        gap = (mine - opp) / max(1, mine + opp) if mine + opp else 0.0
        return max(-1.0, min(1.0, gap))

    def _rollout_choice(self, state: dict, actions: list) -> Optional[ActionInstance]:
        """Heuristic rollout move selection (any line-connect game).

        1) Play an immediate winning move if available.
        2) Otherwise block the opponent's immediate-win threat with
           probability 0.7 (residual randomness keeps the search
           exploratory — pure-greedy rollouts bias the value estimate).

        Both checks are pure board-line scans driven by ``win_length``
        from the game constants — no engine applies, no board-size
        special-casing.  Skipped for large action sets (the scan cost is
        proportional to the action count anyway).  Without the heuristic,
        random rollouts on small boards drown the value of defensive
        moves in noise (the opponent rarely wins by luck), so MCTS never
        learns to block.
        """
        if len(actions) > 16:
            return self.rng.choice(actions)

        board = state.get('_arrays', {}).get('board', [])
        bs = int(len(board) ** 0.5) if board else 0
        if bs == 0:
            return self.rng.choice(actions)
        win_len = int(state.get('_constants', {}).get('win_length', 3))
        turn = state['env'].get('turn')
        opponent = self._opponent_of(state, turn)

        my_win = self._scan_line_cells(board, bs, win_len, turn)
        if my_win:
            wins = [a for a in actions if self._action_index(a) in my_win]
            if wins:
                return self.rng.choice(wins)

        blocks = self._scan_line_cells(board, bs, win_len, opponent)
        if blocks and self.rng.random() < 0.7:
            block_actions = [a for a in actions if self._action_index(a) in blocks]
            if block_actions:
                return self.rng.choice(block_actions)

        return self.rng.choice(actions)

    @staticmethod
    def _scan_line_cells(board: list, bs: int, win_len: int, player: str) -> set[int]:
        """Cells that complete an ``win_len - 1`` run for ``player``.

        Scans every straight line of length ``win_len`` (4 directions)
        and returns the empty cell of each line that is one piece short
        of a win — the immediate-winning (or threat-blocking) moves.
        """
        cells: set[int] = set()
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for r in range(bs):
                for c in range(bs):
                    line = [(r + k * dr, c + k * dc) for k in range(win_len)]
                    if any(not (0 <= rr < bs and 0 <= cc < bs) for rr, cc in line):
                        continue
                    vals = [board[rr * bs + cc] for rr, cc in line]
                    if vals.count(player) != win_len - 1 or vals.count(None) != 1:
                        continue
                    er, ec = line[vals.index(None)]
                    cells.add(er * bs + ec)
        return cells

    @staticmethod
    def _action_index(action: ActionInstance) -> int:
        """Board index of an action's primary cell parameter.

        Uses the entity's ``_index`` field (produced by view
        materialization) — no cell-id format assumptions.
        """
        cell = action.params.get('cell', {})
        if isinstance(cell, dict) and '_index' in cell:
            return int(cell['_index'])
        return -1

    @staticmethod
    def _opponent_of(state: dict, player: str) -> str:
        players = state.get('_players', [])
        ids = [p['id'] for p in players] if players and isinstance(players[0], dict) else players
        return ids[1] if player == ids[0] else ids[0]

    def _backpropagate(self, path: list, value: float):
        for _key, node in reversed(path):
            node.visits += 1
            node.total_value += value
            if node.node_type == 'player':
                value = -value
