"""MCTS solver with chance-node support.

Handles the full player→chance→player→... alternating structure
that stochastic games require.  Implements ``SolverBase``.

The search core is generic over ``GameEngine``; board-game rollout
heuristics live in ``rollout_policy.BoardHeuristicPolicy`` and are only
applied when the state carries a square ``_arrays.board`` (M-04).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from layer2_engine.core.state_graph import ActionInstance, ChanceOutcome, NodeType, State, clone_state
from layer2_engine.core.engine import GameEngine

from ..base import SolverBase, SolverConfig, SolverMetrics
from .rollout_policy import BoardHeuristicPolicy, root_player


@dataclass
class MCTSConfig(SolverConfig):
    budget: int = 5000
    ucb_c: float = 1.414
    rollout_depth: int = 20
    time_limit: Optional[float] = None  # 秒；None 表示只按 budget
    max_nodes: Optional[int] = None  # 节点数上限；None 表示只按 budget/time_limit


@dataclass
class MCTSNode:
    node_type: NodeType
    player: Optional[str] = None  # 轮到谁行动；chance/terminal 节点保留 None
    visits: int = 0
    total_value: float = 0.0
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    child_actions: dict[str, ActionInstance] = field(default_factory=dict)
    child_outcomes: dict[str, ChanceOutcome] = field(default_factory=dict)
    untried_actions: list[ActionInstance] = field(default_factory=list)
    untried_outcomes: list[ChanceOutcome] = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        if self.node_type == "player":
            return len(self.untried_actions) == 0
        return len(self.untried_outcomes) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTS(SolverBase):
    """Monte Carlo Tree Search with chance-node handling."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None):
        # Coerce a plain SolverConfig into an MCTSConfig so solver-specific
        # defaults apply (M-05: `config or MCTSConfig()` silently ignored
        # the config class when a plain SolverConfig was passed).
        if config is None:
            config = MCTSConfig()
        elif not isinstance(config, MCTSConfig):
            config = MCTSConfig(**vars(config))
        super().__init__(engine, config)
        cfg = self.config
        self.budget = cfg.budget
        self.ucb_c = cfg.ucb_c
        self.rollout_depth = cfg.rollout_depth
        self.rng = random.Random(cfg.seed)
        self._nodes_created = 0
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
        """为玩家节点选棋；若根是 chance 节点则返回 None（随机事件请自行采样）。"""
        root_type = self.engine.get_node_type(state)
        root = MCTSNode(node_type=root_type)
        root.player = root_player(state, self.engine)
        self._nodes_created = 0

        if root_type == "player":
            root.untried_actions = sorted(self.engine.get_legal_actions(state), key=lambda a: a.canonical_key)
            if not root.untried_actions:
                return None
        elif root_type == "chance":
            root.untried_outcomes = sorted(self.engine.get_chance_outcomes(state), key=lambda o: o.key)

        _t0 = time.perf_counter()
        for _ in range(self.budget):
            self._iterate(state, root)
            if self.config.time_limit is not None and time.perf_counter() - _t0 > self.config.time_limit:
                break
            if self.config.max_nodes is not None and self._nodes_created >= self.config.max_nodes:
                break

        if root_type == "player":
            if not root.children:
                actions = sorted(self.engine.get_legal_actions(state), key=lambda a: a.canonical_key)
                return self.rng.choice(actions) if actions else None
            if getattr(self.config, "verbose", False):
                print(f"MCTS budget={self.budget} 根节点访问次数={root.visits}")
            best_key = max(root.children, key=lambda k: root.children[k].visits)
            return root.child_actions.get(best_key)
        return None

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        """MCTS 是搜索算法——训练为空操作，但保留 episodes 语义。"""
        return SolverMetrics(episodes=episodes, win_rate=0.0, avg_return=0.0)

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
                state = self.engine.apply_action(state, action)
                node = node.children[key]
                path.append((key, node))
            elif node.node_type == "chance":
                outcome = self._select_chance(node, state)
                if outcome is None:
                    break
                state = self.engine.apply_chance(state, outcome)
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

        unvisited = [k for k, c in node.children.items() if c.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        for key, child in node.children.items():
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
        # Reuse the node's cached outcomes — avoids a full engine re-query
        # (context build + probability eval) on every selection step.
        outcomes = sorted(node.child_outcomes.values(), key=lambda o: o.key)
        if not outcomes:
            outcomes = sorted(node.untried_outcomes, key=lambda o: o.key)
        if not outcomes:
            outcomes = sorted(self.engine.get_chance_outcomes(state), key=lambda o: o.key)
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
        new_state = self.engine.apply_action(state, action)
        child_type = self.engine.get_node_type(new_state)
        child = MCTSNode(node_type=child_type)
        if child_type == "player":
            child.player = self.engine.get_current_player(new_state)
            child.untried_actions = sorted(self.engine.get_legal_actions(new_state), key=lambda a: a.canonical_key)
        elif child_type == "chance":
            child.player = node.player  # 机会节点继承父节点视角
            child.untried_outcomes = sorted(self.engine.get_chance_outcomes(new_state), key=lambda o: o.key)
        node.children[key] = child
        node.child_actions[key] = action
        self._nodes_created += 1
        return child, new_state

    def _expand_chance(self, node: MCTSNode, state: dict):
        """机会结果一次性全部展开并缓存子状态，避免重复 apply_chance 的随机数不同步；
        结果集特别大时可改为逐次展开。"""
        if not node.untried_outcomes:
            return None
        # Each outcome is applied exactly once: the child states built here
        # are cached and the sampled outcome descends into ITS cached state
        # (a second apply_chance could draw different internal randomness
        # and desync the tree from the sim).
        child_states: dict[str, dict] = {}
        for outcome in node.untried_outcomes:
            child_state = self.engine.apply_chance(state, outcome)
            child_states[outcome.key] = child_state
            child_type = self.engine.get_node_type(child_state)
            child = MCTSNode(node_type=child_type)
            child.player = node.player  # 机会子节点继承父节点视角
            if child_type == "player":
                child.untried_actions = sorted(
                    self.engine.get_legal_actions(child_state), key=lambda a: a.canonical_key
                )
            elif child_type == "chance":
                child.untried_outcomes = sorted(self.engine.get_chance_outcomes(child_state), key=lambda o: o.key)
            node.children[outcome.key] = child
            node.child_outcomes[outcome.key] = outcome
            self._nodes_created += 1
        node.untried_outcomes.clear()

        # Sample which child the expansion descends into — from the node's
        # own outcome objects, no engine re-query needed.
        chosen = self._sample_outcome(sorted(node.child_outcomes.values(), key=lambda o: o.key), self.rng)
        if chosen is None:
            return None
        return node.children.get(chosen.key), child_states[chosen.key]

    def _rollout(self, state: dict) -> float:
        sim_state = clone_state(state)
        terminal = False
        for _ in range(self.rollout_depth):
            # get_node_type already folds in is_terminal — one call, not two.
            nt = self.engine.get_node_type(sim_state)
            if nt == "player":
                actions = sorted(self.engine.get_legal_actions(sim_state), key=lambda a: a.canonical_key)
                if not actions:
                    break
                chosen = None
                if self.rollout_policy is not None:
                    chosen = self.rollout_policy(sim_state, actions)
                if chosen is None:
                    chosen = self._board_policy.choose(sim_state, actions)
                if chosen is None:
                    chosen = self.rng.choice(actions)
                sim_state = self.engine.apply_action(sim_state, chosen)
            elif nt == "chance":
                outcomes = sorted(self.engine.get_chance_outcomes(sim_state), key=lambda o: o.key)
                chosen = self._sample_outcome(outcomes, self.rng)
                if chosen is None:
                    break
                sim_state = self.engine.apply_chance(sim_state, chosen)
            else:
                terminal = True
                break

        if terminal or self.engine.is_terminal(sim_state):
            player = root_player(state, self.engine)
            if player is None:
                return 0.0
            return self.engine.get_utility(sim_state, player)
        # Non-terminal at depth limit: board games get the threat-gap
        # heuristic (clamped to ±0.5); everything else gets the neutral 0.
        player = root_player(state, self.engine)
        if player is None:
            return 0.0
        return self._board_policy.leaf_value(sim_state, player)

    def _backpropagate(self, path: list, value: float):
        # value 以“推演起点（路径最深玩家节点）”的视角表示；沿路径向上，
        # 只在玩家确实变化时翻转视角，因此不依赖“双人严格轮流”的假设。
        perspective = None
        for _key, node in reversed(path):
            if node.node_type == "player" and node.player is not None:
                perspective = node.player
                break
        for _key, node in reversed(path):
            node.visits += 1
            if node.node_type == "player" and node.player is not None:
                if perspective is not None and node.player != perspective:
                    value = -value
                    perspective = node.player
            node.total_value += value
