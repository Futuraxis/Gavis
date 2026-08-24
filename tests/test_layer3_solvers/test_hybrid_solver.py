"""Tests for the HybridSolver (MCTS + CFR prior + PSRO opponent model)."""

from __future__ import annotations

import json
import os
import random
import tempfile

import pytest

from layer2_engine.core.engine import GameEngine
from layer3_solvers import HybridConfig, HybridSolver
from layer3_solvers.mcts.solver import MCTSNode


@pytest.fixture
def moon_adapter() -> GameEngine:
    with open("rules/moon_chess.json", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=42)


def _small_gomoku() -> GameEngine:
    rules = json.load(open("rules/stochastic_gomoku.json", encoding="utf-8"))
    rules["constants"]["board_size"] = 5
    rules["constants"]["win_length"] = 3
    return GameEngine(rules, seed=42)


class TestHybridSearch:
    def test_moon_chess_search_legal(self, moon_adapter: GameEngine):
        solver = HybridSolver(moon_adapter, HybridConfig(seed=42, mcts_budget=200))
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        assert action is not None
        legal = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert action.canonical_key in legal

    def test_gomoku_search_legal(self):
        engine = _small_gomoku()
        solver = HybridSolver(engine, HybridConfig(seed=42, mcts_budget=100))
        action = solver.select_action(engine.create_initial_state())
        assert action is not None

    def test_search_plays_game(self, moon_adapter: GameEngine):
        solver = HybridSolver(moon_adapter, HybridConfig(seed=42, mcts_budget=100))
        state = moon_adapter.create_initial_state()
        for _ in range(60):
            if moon_adapter.is_terminal(state):
                break
            action = solver.select_action(state)
            if action is None:
                break
            state = moon_adapter.apply_action(state, action)
        # max_rounds=50 guarantees termination.
        assert moon_adapter.is_terminal(state)


class TestHybridCfrTable:
    def test_train_builds_full_table(self):
        engine = _small_gomoku()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfr.json")
            solver = HybridSolver(
                engine, HybridConfig(seed=42, cfr_iterations=30, cfr_depth_limit=4, cfr_table_path=path)
            )
            solver.train(1, verbose=False)
            assert solver._cfr_table
            # Table keys must be info-set keys: queryable via adapter.
            state = engine.create_initial_state()
            info_key = engine.get_info_set_key(state, "p_black")
            assert info_key in solver._cfr_table
            assert os.path.exists(path)

    def test_table_mode_reload(self):
        engine = _small_gomoku()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfr.json")
            trainer = HybridSolver(
                engine, HybridConfig(seed=42, cfr_iterations=30, cfr_depth_limit=4, cfr_table_path=path)
            )
            trainer.train(1, verbose=False)
            solver = HybridSolver(engine, HybridConfig(seed=42, mode="table", cfr_table_path=path))
            action = solver.select_action(engine.create_initial_state())
            assert action is not None


class TestHybridPsroPool:
    """PSRO pool integration: tabular policies wrapped as callables."""

    def test_pool_mode_returns_legal_action(self, moon_adapter: GameEngine):
        solver = HybridSolver(
            moon_adapter,
            HybridConfig(
                seed=42,
                mode="pool",
                opponent_model="psro",
                cfr_iterations=10,
                cfr_depth_limit=4,
                psro_iters=1,
                psro_steps_per_iter=200,
            ),
        )
        solver.train(1, verbose=False)
        assert solver._pool  # noqa: SLF001 — white-box
        assert abs(sum(solver._pool_weights) - 1.0) < 1e-9  # noqa: SLF001
        state = moon_adapter.create_initial_state()
        action = solver.select_action(state)
        legal = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert action is not None
        assert action.canonical_key in legal

    def test_psro_opponent_model_samples(self, moon_adapter: GameEngine):
        solver = HybridSolver(
            moon_adapter,
            HybridConfig(
                seed=42,
                mode="search",
                opponent_model="psro",
                cfr_iterations=10,
                cfr_depth_limit=4,
                mcts_budget=40,
                psro_iters=1,
                psro_steps_per_iter=200,
                imperfect_information=True,
            ),
        )
        solver.train(1, verbose=False)
        state = moon_adapter.create_initial_state()
        dist = solver._opponent.action_distribution(moon_adapter, state)  # noqa: SLF001
        legal = {a.canonical_key for a in moon_adapter.get_legal_actions(state)}
        assert set(dist) <= legal and sum(dist.values()) > 0


class TestHybridPoker:
    """Opponent-model search over sampled worlds on Texas Hold'em."""

    @pytest.fixture
    def texas(self) -> GameEngine:
        with open("rules/texas_holdem.json", encoding="utf-8") as f:
            return GameEngine(json.load(f), seed=42)

    @staticmethod
    def _resolve(engine: GameEngine, state: dict) -> dict:
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        return state

    @staticmethod
    def _solver(texas: GameEngine, budget: int = 100) -> HybridSolver:
        return HybridSolver(texas, HybridConfig(seed=42, imperfect_information=True, mcts_budget=budget))

    def test_sample_hidden_consistency(self, texas: GameEngine):
        """v5.2: world completion is rules-driven (``hiddenWorld`` section) —
        the acting player's cards survive; the opponent's are re-drawn from
        the undealt deck with no collisions."""
        state = self._resolve(texas, texas.create_initial_state())
        solver = self._solver(texas)
        world = solver._sample_hidden_world(state)  # noqa: SLF001 — white-box
        assert world["_arrays"]["sb_hole"] == state["_arrays"]["sb_hole"]
        assert len(world["_arrays"]["bb_hole"]) == 2
        drawn = set(state["_arrays"]["drawn"])
        assert not (set(world["_arrays"]["bb_hole"]) & drawn)
        assert not (set(world["_arrays"]["bb_hole"]) & set(state["_arrays"]["sb_hole"]))
        # Repeated samples explore the deck, not a fixed hand
        hands = {tuple(solver._sample_hidden_world(state)["_arrays"]["bb_hole"]) for _ in range(30)}  # noqa: SLF001
        assert len(hands) > 1

    def test_poker_search_legal(self, texas: GameEngine):
        solver = self._solver(texas)
        state = self._resolve(texas, texas.create_initial_state())
        action = solver.select_action(state)
        assert action is not None
        legal = {a.canonical_key for a in texas.get_legal_actions(state)}
        assert action.canonical_key in legal

    def test_poker_chance_expansion(self, texas: GameEngine):
        """The tree must grow through deal (chance) nodes, not stall at them."""
        solver = self._solver(texas, budget=50)
        state = self._resolve(texas, texas.create_initial_state())
        root = MCTSNode(node_type="player")
        root.untried_actions = texas.get_legal_actions(state)
        root_player = state["env"]["turn"]
        for _ in range(200):
            world = solver._sample_hidden_world(state)  # noqa: SLF001 — white-box
            solver._omcts_iterate(world, root, root_player)  # noqa: SLF001 — white-box
        assert root.visits == 200
        chance_nodes = [c for c in root.children.values() if c.node_type == "chance"]
        assert chance_nodes, "no deal nodes reached under any root action"
        assert any(c.children for c in chance_nodes), "chance nodes were never expanded"
        # utilities are chip payoffs in [-100, 100] per simulation
        assert all(abs(c.total_value) <= c.visits * 100 + 1e-9 for c in root.children.values())

    def test_poker_hybrid_plays_full_hand(self, texas: GameEngine):
        """Hybrid (imperfect info) drives a full hand to a zero-sum terminal."""
        solver = self._solver(texas, budget=80)
        state = self._resolve(texas, texas.create_initial_state())
        rng = random.Random(0)
        guard = 0
        while not texas.is_terminal(state) and guard < 60:
            if texas.get_node_type(state) == "player":
                if state["env"]["turn"] == "p_sb":
                    action = solver.select_action(state)
                else:
                    actions = texas.get_legal_actions(state)
                    action = rng.choice(actions) if actions else None
                if action is None:
                    break
                state = texas.apply_action(state, action)
            else:
                while texas.get_node_type(state) == "chance":
                    outs = texas.get_chance_outcomes(state)
                    if not outs:
                        break
                    state = texas.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
            guard += 1
        assert texas.is_terminal(state)
        assert texas.get_utility(state, "p_sb") + texas.get_utility(state, "p_bb") == 0.0

    def test_poker_cfr_prior(self, texas: GameEngine):
        """CFR prior training works on the imperfect-info game and the
        resulting table serves table-mode decisions."""
        solver = HybridSolver(texas, HybridConfig(seed=42, cfr_iterations=15, cfr_depth_limit=10))
        solver.train(1, verbose=False)
        assert solver._cfr_table  # noqa: SLF001 — white-box
        state = self._resolve(texas, texas.create_initial_state())
        action = solver._select_from_table(state)  # noqa: SLF001 — white-box
        legal = {a.canonical_key for a in texas.get_legal_actions(state)}
        if action is not None:
            assert action.canonical_key in legal


class TestEmpiricalModel:
    """Online-learning opponent model: Laplace-smoothed action counts."""

    @pytest.fixture
    def texas(self) -> GameEngine:
        with open("rules/texas_holdem.json", encoding="utf-8") as f:
            return GameEngine(json.load(f), seed=7)

    def _state(self, texas: GameEngine):
        state = texas.create_initial_state()
        while texas.get_node_type(state) == "chance":
            _, state = texas.sample_chance(state)
        return state

    def test_distribution_sums_to_one_and_is_smoothed(self, texas: GameEngine):
        from layer3_solvers.hybrid import EmpiricalModel

        state = self._state(texas)
        player = texas.get_current_player(state)
        info_key = texas.get_info_set_key(state, player)
        legal = texas.get_legal_actions(state)
        table = {info_key: {"act:fold:0": 8, "act:call:2": 2}}
        dist = EmpiricalModel(table).action_distribution(texas, state)
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        # Laplace prior: fold dominates but is never 1.0
        assert 0.0 < dist["act:fold:0"] < 1.0
        # every legal action has positive probability (smoothing)
        assert all(k in dist and dist[k] > 0.0 for k in (a.canonical_key for a in legal))

    def test_unseen_info_set_falls_back_to_uniform(self, texas: GameEngine):
        from layer3_solvers.hybrid import EmpiricalModel

        state = self._state(texas)
        dist = EmpiricalModel({}).action_distribution(texas, state)
        actions = texas.get_legal_actions(state)
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert len(dist) == len(actions)

    def test_merge_accumulates_counts(self, texas: GameEngine):
        from layer3_solvers.hybrid import EmpiricalModel

        state = self._state(texas)
        player = texas.get_current_player(state)
        info_key = texas.get_info_set_key(state, player)
        model = EmpiricalModel({info_key: {"act:fold:0": 8}})
        model.merge({info_key: {"act:fold:0": 2, "act:call:2": 3}})
        assert model._table[info_key] == {"act:fold:0": 10, "act:call:2": 3}  # noqa: SLF001
        assert model.coverage() == 1

    def test_hybrid_learn_online_merges_human_decisions(self, texas: GameEngine):
        """learn_online counts only actor='human' decisions (duck-typed
        signal shape: metadata['decisions']), never AI plays."""

        state = self._state(texas)
        player = texas.get_current_player(state)
        info_key = texas.get_info_set_key(state, player)
        solver = HybridSolver(
            texas,
            HybridConfig(seed=1, opponent_model="empirical", empirical_table={info_key: {"act:fold:0": 1}}),
        )

        class FakeSignal:
            def __init__(self, decisions):
                self.metadata = {"decisions": decisions}

        sig = FakeSignal(
            [
                {"actor": "human", "info_key": info_key, "action": {"canonical_key": "act:fold:0"}},
                {"actor": "human", "info_key": info_key, "action": {"canonical_key": "act:raise:4"}},
                {"actor": "ai", "info_key": info_key, "action": {"canonical_key": "act:call:2"}},
            ]
        )
        metrics = solver.learn_online([sig])
        assert metrics["decisions"] == 2
        assert solver._opponent._table[info_key] == {"act:fold:0": 2, "act:raise:4": 1}  # noqa: SLF001

    def test_hybrid_empirical_search_plays_full_hand(self, texas: GameEngine):
        """Opponent-model search with an empirical table drives a full
        hand (the online-learning consumption path)."""
        state = texas.create_initial_state()
        while texas.get_node_type(state) == "chance":
            _, state = texas.sample_chance(state)
        player = texas.get_current_player(state)
        info_key = texas.get_info_set_key(state, player)
        solver = HybridSolver(
            texas,
            HybridConfig(
                seed=7,
                mode="search",
                imperfect_information=True,
                mcts_budget=60,
                opponent_model="empirical",
                empirical_table={info_key: {"act:fold:0": 1, "act:call:2": 9}},
            ),
        )
        action = solver.select_action(state)
        assert action is not None
        legal = {a.canonical_key for a in texas.get_legal_actions(state)}
        assert action.canonical_key in legal
