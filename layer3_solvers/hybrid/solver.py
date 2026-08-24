"""HybridSolver — MCTS online search + CFR prior + PSRO opponent model.

A generic ``SolverBase`` that composes the three Layer-3 solvers:

  - CFR: offline equilibrium prior.  Its strategy table drives rollouts
    (instead of random play) and can serve as a pure table policy.
  - PSRO: offline policy pool + Nash mixture, consumed as an opponent
    model by the online search.
  - MCTS: online search.  In perfect-information games this is the
    standard search with a CFR-guided rollout; in imperfect-information
    games (rules declare ``hiddenWorld``) it becomes an opponent-model
    search over sampled worlds — the sampled-world (PIMC) completion is
    generic solver code driven purely by the rules' data spec.

Runtime ``mode`` selects the decision path:
  - 'search' (default): online search (MCTS or opponent-model search).
  - 'table': CFR strategy-table lookup (instant, no search).
  - 'pool': sample a PSRO member strategy by the Nash mixture.

Everything is generic over ``GameEngine`` — no game-specific logic.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance, ChanceOutcome, State, clone_state

from ..base import SolverBase, SolverConfig, SolverMetrics
from ..cfr.solver import CFR, CFRConfig
from ..mcts.solver import MCTS, MCTSConfig, MCTSNode
from .opponent_model import (
    CFRTableModel,
    EmpiricalModel,
    OpponentModel,
    PSROMixModel,
    TabularPolicyMember,
    UniformModel,
    sample_action,
)

try:
    # PSRO 是可选依赖（审查 P2-25）：psro extra 缺失时 HybridSolver 仍可
    # 导入，只有显式请求 psro 模式才报错。
    from ..psro.solver import PSROConfig, PSROSolver
except ImportError:  # pragma: no cover — exercised only in minimal installs
    PSROConfig = None
    PSROSolver = None


@dataclass
class HybridConfig(SolverConfig):
    mode: str = "search"  # 'search' | 'table' | 'pool'
    imperfect_information: bool = False  # enable the sampled-worlds path
    mcts_budget: int = 500
    mcts_rollout_depth: int = 20
    cfr_iterations: int = 1000
    cfr_depth_limit: int = 6  # bound CFR's tree walk (big games explode)
    psro_iters: int = 3
    psro_steps_per_iter: int = 2000
    opponent_model: str = "uniform"  # 'uniform' | 'cfr' | 'psro' | 'empirical'
    tree_samples: int = 50  # sampled worlds per search round
    cfr_table_path: Optional[str] = None  # load/save CFR strategy table
    #: Empirical opponent table (online learning) — either injected as a
    #: dict or loaded from ``empirical_table_path``.  Consumed only when
    #: ``opponent_model == 'empirical'``.
    empirical_table: Optional[dict] = None
    empirical_table_path: Optional[str] = None


class HybridSolver(SolverBase):
    """Composed solver: search online, prior from CFR, adversary from PSRO."""

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None):
        super().__init__(engine, config or HybridConfig())
        cfg: HybridConfig = self.config
        self.rng = random.Random(cfg.seed)

        self.mcts = MCTS(
            engine, MCTSConfig(seed=cfg.seed, budget=cfg.mcts_budget, rollout_depth=cfg.mcts_rollout_depth)
        )
        self.cfr = CFR(engine, CFRConfig(seed=cfg.seed, iterations=cfg.cfr_iterations, depth_limit=cfg.cfr_depth_limit))
        self.psro = None  # lazily built — see _ensure_psro
        if cfg.mode == "pool" or cfg.opponent_model == "psro":
            self._ensure_psro(engine)

        self._cfr_table: Optional[dict] = None
        self._pool: list = []
        self._pool_weights: list[float] = []
        self._opponent: OpponentModel = UniformModel()
        # id(child) → cached opponent reply; reset per opponent-model search.
        self._edge_replies: dict[int, Optional[ActionInstance]] = {}
        if cfg.cfr_table_path and Path(cfg.cfr_table_path).exists():
            self._load_cfr_table(cfg.cfr_table_path)
        if cfg.opponent_model == "empirical":
            table = cfg.empirical_table
            if table is None and cfg.empirical_table_path and Path(cfg.empirical_table_path).exists():
                with open(cfg.empirical_table_path, "r", encoding="utf-8") as f:
                    table = json.load(f)
            if table:
                self._opponent = EmpiricalModel(table)

    def _ensure_psro(self, engine: GameEngine) -> None:
        """Lazily build the PSRO component (pool mode / psro opponent model).

        Kept out of the main ``__init__`` path so search/table modes work
        for games PSRO cannot represent (e.g. 5×5 gomoku) and for minimal
        installs without the psro extra (审查 P2-25).
        """
        if self.psro is not None:
            return
        if PSROSolver is None:  # pragma: no cover — minimal install
            raise RuntimeError(
                "HybridSolver requires the 'psro' extra (gymnasium/tqdm) for pool mode / the psro opponent model"
            )
        cfg: HybridConfig = self.config
        self.psro = PSROSolver(
            engine, PSROConfig(seed=cfg.seed, num_iters=cfg.psro_iters, num_steps_per_iter=cfg.psro_steps_per_iter)
        )

    @property
    def name(self) -> str:
        return f"Hybrid({self.config.mode})"

    # ── SolverBase ──────────────────────────────────────────────────

    def select_action(self, state: State) -> Optional[ActionInstance]:
        cfg: HybridConfig = self.config
        if cfg.mode == "table":
            return self._select_from_table(state)
        if cfg.mode == "pool":
            return self._select_from_pool(state)
        return self._select_search(state)

    def train(self, episodes: int, **kwargs) -> SolverMetrics:
        """Train the components per config: CFR prior, then PSRO pool.

        The CFR table is built first (it guides both PSRO member play
        and online rollouts); PSRO then evolves a pool whose Nash
        mixture becomes the opponent model.
        """
        cfg: HybridConfig = self.config
        verbose = kwargs.get("verbose", False)

        # 1) CFR prior (equilibrium table).  ``solve`` only returns the
        # root info-set's strategy, so the full table is built from the
        # solver's normalized strategy sums.
        state = self.engine.create_initial_state()
        if verbose:
            print(f"  Hybrid: training CFR ({cfg.cfr_iterations} iters)...")
        self.cfr.solve(state, verbose=verbose)
        self._cfr_table = self._build_cfr_table()
        self.mcts.rollout_policy = self._cfr_rollout_policy
        if cfg.opponent_model == "cfr":
            self._opponent = CFRTableModel(self._cfr_table)
        if cfg.cfr_table_path:
            self._save_cfr_table(cfg.cfr_table_path)

        # 2) PSRO pool + Nash mixture (opponent model).
        if cfg.opponent_model == "psro" or cfg.mode == "pool":
            self._ensure_psro(self.engine)
            if verbose:
                print(f"  Hybrid: running PSRO ({cfg.psro_iters} iters)...")
            self.psro.train(episodes=max(1, cfg.psro_iters), verbose=verbose)
            # Pool members are tabular policies (ndarray keyed by the
            # GymAdapter encoding) — wrap them into callables so the
            # mixture model and pool-mode decisions can sample them.
            self._pool = [TabularPolicyMember(pi, self.engine, self.rng) for pi in self.psro._policy_pool]  # noqa: SLF001
            weights = np.asarray(self.psro._nash_weights, dtype=float)  # noqa: SLF001
            if weights.size != len(self._pool):
                weights = np.ones(len(self._pool)) / len(self._pool)
            self._pool_weights = list(weights / weights.sum())
            if cfg.opponent_model == "psro":
                self._opponent = PSROMixModel(self._pool, self._pool_weights, self.rng)

        return SolverMetrics(episodes=episodes, win_rate=0.0, avg_return=0.0)

    def solve(self, state: State, verbose: bool = False) -> dict[str, float]:
        self.train(1, verbose=verbose)
        strategy = self._select_from_table(state)
        if strategy is None:
            return {}
        return {strategy.canonical_key: 1.0}

    def learn_online(self, signals: list, **kwargs) -> dict[str, int]:
        """Incrementally merge opponent decisions from online-learning signals.

        ``signals`` is a sequence of objects with the ``OnlineLearningSignal``
        shape (``metadata[\"decisions\"]``: ``{actor, info_key, action:
        {canonical_key}}``) — duck-typed so Layer 3 never imports Layer 4
        types.  Only HUMAN-actor decisions are counted (the opponent in a
        human-vs-AI match); AI decisions are the solver's own play and
        must not pollute the opponent model.

        If the current opponent is an ``EmpiricalModel`` the counts are
        merged in place; otherwise a new empirical table is installed
        (and ``config.opponent_model`` is left unchanged — callers should
        construct the solver with ``opponent_model=\"empirical\"``).
        """
        counts: dict[str, dict[str, int]] = {}
        decisions = 0
        for signal in signals:
            metadata = getattr(signal, "metadata", {})
            for decision in metadata.get("decisions", []):
                if decision.get("actor") != "human":
                    continue
                info_key = decision.get("info_key")
                action = decision.get("action") or {}
                action_key = action.get("canonical_key")
                if info_key and action_key:
                    counts.setdefault(info_key, {}).setdefault(action_key, 0)
                    counts[info_key][action_key] += 1
                    decisions += 1
        if counts:
            if isinstance(self._opponent, EmpiricalModel):
                self._opponent.merge(counts)
            else:
                self._opponent = EmpiricalModel(counts)
        return {"info_keys": len(counts), "decisions": decisions}

    # ── Decision paths ──────────────────────────────────────────────

    def _select_search(self, state: State) -> Optional[ActionInstance]:
        cfg: HybridConfig = self.config
        if cfg.imperfect_information and self._has_hidden_world():
            return self._opponent_mcts(state)
        if self.mcts.rollout_policy is None and self._cfr_table:
            self.mcts.rollout_policy = self._cfr_rollout_policy
        return self.mcts.select_action(state)

    def _select_from_table(self, state: State) -> Optional[ActionInstance]:
        if not self._cfr_table:
            return None
        player = self.engine.get_current_player(state)
        if player is None:
            return None
        info_key = self.engine.get_info_set_key(state, player)
        strategy = self._cfr_table.get(info_key)
        if not strategy:
            return None
        actions = self.engine.get_legal_actions(state)
        return sample_action(strategy, actions, self.rng)

    def _select_from_pool(self, state: State) -> Optional[ActionInstance]:
        if not self._pool:
            return None
        actions = self.engine.get_legal_actions(state)
        if not actions:
            return None
        member = self.rng.choices(self._pool, weights=self._pool_weights, k=1)[0]
        key = member(self.engine, state)
        for a in actions:
            if a.canonical_key == key:
                return a
        return self.rng.choice(actions)

    # ── Opponent-model search (imperfect information) ───────────────

    def _has_hidden_world(self) -> bool:
        """True when the rules declare a ``hiddenWorld`` spec (pure data).

        The PIMC path is enabled by the rules JSON declaring which hidden
        arrays must be re-sampled — no engine/adaptor method is probed.
        """
        spec = (getattr(self.engine, "rules", {}) or {}).get("hiddenWorld") or {}
        return bool((spec.get("arrays") or {}))

    def _sample_hidden_world(self, state: State) -> State:
        """Complete one consistent hidden world from the rules spec.

        Generic solver-side PIMC completion: ``hiddenWorld.arrays`` lists
        each array to re-sample, its ``pool`` (constant name or literal
        list) and other arrays that must not be reused (``exclude``).
        Only arrays named in the spec are replaced — everything else is
        kept from the public/ground state.  Seeded via the engine RNG so
        world sampling is reproducible per game seed.
        """
        rules = getattr(self.engine, "rules", {}) or {}
        spec = (rules.get("hiddenWorld") or {}).get("arrays") or {}
        if not spec:
            return clone_state(state)
        world = clone_state(state)
        constants = getattr(self.engine, "_constants", {}) or {}
        arrs = world.setdefault("_arrays", {})
        for name, cfg in spec.items():
            pool_ref = cfg.get("pool")
            pool = constants.get(pool_ref) if isinstance(pool_ref, str) else (pool_ref or [])
            if not pool:
                continue
            taken = []
            for other in cfg.get("exclude") or []:
                taken.extend(arrs.get(other) or [])
            remaining = [x for x in pool if x not in taken]
            count = int(cfg.get("count", 1))
            if len(remaining) < count:
                continue
            arrs[name] = self.engine.rng.sample(remaining, count)
        return world

    def _opponent_mcts(self, state: State) -> Optional[ActionInstance]:
        """Opponent-model search (PIMC-style) over sampled worlds.

        Each simulation samples a complete hidden world consistent with
        the public observation (``hiddenWorld`` spec), then walks a
        tree containing only OUR decision nodes and chance nodes.
        Opponent nodes never enter the tree: every tree edge carries a
        cached opponent reply (sampled from the opponent model when the
        edge is first created, replayed on every descent), so each edge's
        subtree is deterministic within and across simulations.

        All values are stored from the root player's perspective (no
        sign flips); root children are aggregated by visit count.
        """
        cfg: HybridConfig = self.config
        root_player = self._our_player_id(state)
        self._edge_replies = {}  # fresh cache per search
        root = MCTSNode(node_type="player")
        root.untried_actions = self.engine.get_legal_actions(state)
        if not root.untried_actions:
            return None

        for _ in range(cfg.mcts_budget):
            world = self._sample_hidden_world(state)
            self._omcts_iterate(world, root, root_player)

        if not root.children:
            return self.rng.choice(root.untried_actions)
        best_key = max(root.children, key=lambda k: root.children[k].visits)
        return root.child_actions.get(best_key)

    def _omcts_iterate(self, world: State, root: MCTSNode, root_player: str) -> None:
        """One simulation: expand/descend our tree along cached replies."""
        sim = clone_state(world)
        node = root
        path: list[MCTSNode] = [root]
        guard = max(32, self.config.mcts_rollout_depth * 4)
        # M-11: refresh the root's untried actions per sampled world —
        # different hidden worlds can have different legal action sets, so
        # a one-shot snapshot from the public state would both miss legal
        # actions and try illegal ones.
        root.untried_actions = [a for a in self.engine.get_legal_actions(sim) if a.canonical_key not in root.children]

        for _ in range(guard):
            nt = self.engine.get_node_type(sim)
            if nt == "terminal":
                break
            if nt == "chance":
                if node.node_type != "chance":
                    break  # tree/sim out of sync — abandon this walk
                if not node.is_fully_expanded():
                    self._expand_omcts_chance(node, sim, root_player)
                    if not node.children:
                        break
                outcome = self._select_chance_child(node)
                if outcome is None:
                    break
                sim = self.engine.apply_chance(sim, outcome)
                node = node.children[outcome.key]
                sim = self._apply_cached_reply(sim, node, root_player)
                path.append(node)
                continue

            # Player node: must be OURS (opponent nodes are only crossed
            # at tree edges, where the cached reply handles them).
            if sim["env"].get("turn") != root_player:
                break
            if not node.is_fully_expanded():
                expanded = self._expand_omcts_player(node, sim, root_player)
                if expanded is None:
                    break
                node, sim = expanded
                path.append(node)
                break  # rollout starts from the expanded state
            if not node.children:
                break
            key = self._select_ucb1(node)
            if key is None:
                break
            sim = self.engine.apply_action(sim, node.child_actions[key])
            node = node.children[key]
            sim = self._apply_cached_reply(sim, node, root_player)
            path.append(node)

        value = self._rollout_prior(sim, root_player)
        self._backprop_omcts(path, value)

    # ── Opponent-reply caching ──────────────────────────────────────

    def _sample_opponent_reply(self, sim: State, root_player: str) -> Optional[ActionInstance]:
        """Sample one opponent-model action if it is the opponent's turn."""
        if self.engine.get_node_type(sim) != "player":
            return None
        if sim["env"].get("turn") == root_player:
            return None
        actions = self.engine.get_legal_actions(sim)
        dist = self._opponent.action_distribution(self.engine, sim)
        return sample_action(dist, actions, self.rng)

    def _apply_cached_reply(self, sim: State, child: MCTSNode, root_player: str) -> State:
        """Apply the child edge's cached opponent reply (sample + cache on first use).

        The cached reply is what makes the subtree under ``child``
        deterministic: expansion and every subsequent descent apply the
        same opponent action, so the sim always matches the child's type.
        """
        reply = self._edge_replies.get(id(child))
        if reply is None:
            reply = self._sample_opponent_reply(sim, root_player)
            self._edge_replies[id(child)] = reply
        if reply is not None:
            return self.engine.apply_action(sim, reply)
        return sim

    # ── Tree expansion ──────────────────────────────────────────────

    @staticmethod
    def _fill_child(child: MCTSNode, sim: State, engine: GameEngine) -> None:
        """Set untried actions/outcomes according to the child's node type."""
        child_type = engine.get_node_type(sim)
        child.node_type = child_type
        if child_type == "player":
            child.untried_actions = engine.get_legal_actions(sim)
        elif child_type == "chance":
            child.untried_outcomes = list(engine.get_chance_outcomes(sim))

    def _expand_omcts_player(self, node: MCTSNode, state: State, root_player: str) -> Optional[tuple[MCTSNode, State]]:
        """Expand one of OUR actions; the child is the node AFTER the
        opponent's (cached) reply — chance, our-player, or terminal."""
        if not node.untried_actions:
            return None
        action = node.untried_actions.pop()
        sim = self.engine.apply_action(state, action)
        child = MCTSNode(node_type=self.engine.get_node_type(sim))
        node.children[action.canonical_key] = child
        node.child_actions[action.canonical_key] = action
        sim = self._apply_cached_reply(sim, child, root_player)
        # The post-reply type may differ from the pre-reply one (e.g. an
        # opponent fold vs a call); re-derive it from the advanced state.
        self._fill_child(child, sim, self.engine)
        if child.node_type == "player" and sim["env"].get("turn") != root_player:
            # Degenerate edge (opponent still to act, e.g. a multi-action
            # turn): roll the expansion back so the action is not lost
            # (M-12).  Re-insert at the FRONT — untried_actions pops from
            # the end, so other actions get tried before it is retried.
            node.children.pop(action.canonical_key, None)
            node.child_actions.pop(action.canonical_key, None)
            node.untried_actions.insert(0, action)
            return None
        return child, sim

    def _expand_omcts_chance(self, node: MCTSNode, state: State, root_player: str) -> None:
        """Expand every chance outcome at once (mirrors plain MCTS)."""
        for outcome in node.untried_outcomes:
            sim = self.engine.apply_chance(state, outcome)
            child = MCTSNode(node_type=self.engine.get_node_type(sim))
            node.children[outcome.key] = child
            node.child_outcomes[outcome.key] = outcome
            sim = self._apply_cached_reply(sim, child, root_player)
            self._fill_child(child, sim, self.engine)
        node.untried_outcomes.clear()

    def _select_chance_child(self, node: MCTSNode) -> Optional[ChanceOutcome]:
        """Probability-weighted descent into an expanded chance node."""
        outcomes = list(node.child_outcomes.values())
        return self._sample_outcome(outcomes)

    def _sample_outcome(self, outcomes: list[ChanceOutcome]) -> Optional[ChanceOutcome]:
        """Probability-weighted sample, normalized — tolerates probability
        vectors that do not sum to exactly 1 (no tail bias)."""
        if not outcomes:
            return None
        total = sum(o.probability for o in outcomes)
        if total <= 0:
            return self.rng.choice(outcomes)
        r = self.rng.random() * total
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return o
        return outcomes[-1]

    def _select_ucb1(self, node: MCTSNode) -> Optional[str]:
        """UCB1 over our node's children (all values: root perspective)."""
        import math

        best_key = None
        best_ucb = -float("inf")
        for key, child in node.children.items():
            if child.visits == 0:
                return key
            exploitation = child.total_value / child.visits
            exploration = 1.414 * math.sqrt(math.log(node.visits + 1) / child.visits)
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_key = key
        return best_key

    def _backprop_omcts(self, path: list[MCTSNode], value: float) -> None:
        """Accumulate the root-perspective value along the path (no flips)."""
        for node in path:
            node.visits += 1
            node.total_value += value

    def _rollout_prior(self, sim: State, root_player: str) -> float:
        """Rollout guided by the CFR prior (fallback: uniform)."""
        depth = self.config.mcts_rollout_depth
        for _ in range(depth):
            nt = self.engine.get_node_type(sim)
            if nt == "terminal":
                break
            if nt == "chance":
                # C-06: ``sample_chance`` is not part of the GameEngine
                # Protocol — go through ``get_chance_outcomes`` and sample
                # locally so any compliant engine works here.
                outcomes = self.engine.get_chance_outcomes(sim)
                chosen = self._sample_outcome(outcomes)
                if chosen is None:
                    break
                sim = self.engine.apply_chance(sim, chosen)
                continue
            actions = self.engine.get_legal_actions(sim)
            if not actions:
                break
            if self._cfr_table is not None:
                player = self.engine.get_current_player(sim)
                info_key = self.engine.get_info_set_key(sim, player) if player else None
                strategy = self._cfr_table.get(info_key) if info_key else None
                chosen = sample_action(strategy, actions, self.rng) if strategy else None
                if chosen is not None:
                    sim = self.engine.apply_action(sim, chosen)
                    continue
            sim = self.engine.apply_action(sim, self.rng.choice(actions))

        if self.engine.is_terminal(sim):
            return self.engine.get_utility(sim, root_player)
        return 0.0

    @staticmethod
    def _our_player_id(world: State) -> str:
        player = world["env"].get("turn")
        if isinstance(player, dict):
            player = player.get("currentPlayerId", player)
        return player

    # ── CFR prior as rollout policy for the plain MCTS ──────────────

    def _cfr_rollout_policy(self, state: State, actions: list) -> Optional[ActionInstance]:
        if not self._cfr_table:
            return None
        player = self.engine.get_current_player(state)
        if player is None:
            return None
        info_key = self.engine.get_info_set_key(state, player)
        strategy = self._cfr_table.get(info_key)
        if not strategy:
            return None
        return sample_action(strategy, actions, self.rng)

    # ── Strategy table persistence ──────────────────────────────────

    def _build_cfr_table(self) -> dict[str, dict[str, float]]:
        """Normalize the CFR solver's strategy sums into a full table."""
        table: dict[str, dict[str, float]] = {}
        for info_key, info in self.cfr.info_sets.items():
            sums = info.get("strategy_sum", {})
            total = sum(sums.values())
            if total > 0:
                table[info_key] = {k: v / total for k, v in sums.items()}
        return table

    def _save_cfr_table(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cfr_table, f, ensure_ascii=False)

    def _load_cfr_table(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._cfr_table = json.load(f)
        self.mcts.rollout_policy = self._cfr_rollout_policy
        if self.config.opponent_model == "cfr":
            self._opponent = CFRTableModel(self._cfr_table)
