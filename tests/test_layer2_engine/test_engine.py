"""Tests for Layer 2: GameEngine (stochastic gomoku, v5.0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import layer2_engine
from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import (
    clone_state,
    create_initial_state,
)
from layer2_engine.interfaces.solver_adapter import ActionInstance

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


@pytest.fixture
def gomoku_rules() -> dict:
    path = RULES_DIR / "stochastic_gomoku.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def engine(gomoku_rules: dict) -> GameEngine:
    return GameEngine(gomoku_rules, seed=42)


# ── State basics ──────────────────────────────────────────────────


class TestStateBasics:
    def test_create_initial_state(self, engine: GameEngine):
        state = engine.create_initial_state()
        board = state["_arrays"]["board"]
        assert len(board) == 81  # 9×9 default
        assert all(c is None for c in board)
        assert state["env"]["phase"] == "playing"
        assert state["env"]["turn"] == "p_black"

    def test_get_node_type_initial(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert engine.get_node_type(state) == "player"

    def test_get_current_player_initial(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert engine.get_current_player(state) == "p_black"

    def test_get_legal_actions_count(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        assert len(actions) == 81  # all cells empty

    def test_is_terminal_initial(self, engine: GameEngine):
        state = engine.create_initial_state()
        assert not engine.is_terminal(state)

    def test_clone_state(self):
        schema = {
            "groundState": {
                "board": {"type": "array", "length": 25, "element": "string?"},
                "env": {"type": "env", "fields": {"phase": {"type": "string", "initial": "playing"}}},
            },
            "derivedViews": {},
        }
        state = create_initial_state(schema)
        board = state["_arrays"]["board"]
        cloned = clone_state(state)
        assert len(cloned["_arrays"]["board"]) == 25
        assert cloned["_arrays"]["board"] == board
        # Mutating clone should not affect original
        cloned["_arrays"]["board"][0] = "p_black"
        assert board[0] is None


# ── Actions ───────────────────────────────────────────────────────


class TestActions:
    def test_action_instance_single_source(self, engine: GameEngine) -> None:
        """Review M-2: engine-produced actions are the SAME class as the
        Layer 2↔3 contract type (solver_adapter) — isinstance must hold."""
        state = engine.create_initial_state()
        action = engine.get_legal_actions(state)[0]

        assert isinstance(action, layer2_engine.ActionInstance)
        assert isinstance(action, ActionInstance)
        assert type(action) is ActionInstance

    def test_apply_action_returns_new_state(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        assert len(actions) > 0
        new_state = engine.apply_action(state, actions[0])
        # Should be a different object
        assert new_state is not state
        # Board should have changed
        board = new_state["_arrays"]["board"]
        placed = sum(1 for c in board if c is not None)
        assert placed == 1

    def test_place_black_then_white(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)

        # Black places
        state = engine.apply_action(state, actions[0])
        board = state["_arrays"]["board"]
        assert board[0] == "p_black"

        # Chance node (vanish check)
        assert engine.get_node_type(state) == "chance"
        outcomes = engine.get_chance_outcomes(state)
        assert len(outcomes) == 2
        outcome, state = engine.sample_chance(state)

        # Now white's turn (unless game over)
        if engine.get_node_type(state) == "player":
            assert engine.get_current_player(state) == "p_white"

    def test_legal_actions_reduce_after_placement(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        assert len(actions) == 81

        # Place at index 0
        state = engine.apply_action(state, actions[0])
        # Resolve chance
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)

        # If still playing, should have fewer legal actions
        if not engine.is_terminal(state):
            actions2 = engine.get_legal_actions(state)
            assert len(actions2) <= 80

    def test_apply_chance_vanish(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)

        # Black places at center
        center_action = actions[40]
        state = engine.apply_action(state, center_action)

        # Get chance outcomes
        outcomes = engine.get_chance_outcomes(state)
        assert len(outcomes) == 2

        # Apply vanish
        vanish = [o for o in outcomes if o.key == "vanish"][0]
        new_state = engine.apply_chance(state, vanish)
        assert new_state["_arrays"]["board"][40] is None

    def test_apply_chance_keep(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        state = engine.apply_action(state, actions[40])

        outcomes = engine.get_chance_outcomes(state)
        keep = [o for o in outcomes if o.key == "keep"][0]
        new_state = engine.apply_chance(state, keep)
        assert new_state["_arrays"]["board"][40] == "p_black"

    def test_sample_chance(self, engine: GameEngine):
        """Over many samples, verify approx 50% vanish rate."""
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        state = engine.apply_action(state, actions[0])

        vanish_count = 0
        n = 200
        for i in range(n):
            s = clone_state(state)
            engine.rng.seed(i * 100)
            outcome, _ = engine.sample_chance(s)
            if outcome.key == "vanish":
                vanish_count += 1

        # Should be roughly 50% (allow ±15%)
        rate = vanish_count / n
        assert 0.35 <= rate <= 0.65, f"Vanish rate {rate} too far from 0.5"


# ── Utility and terminal ──────────────────────────────────────────


class TestUtility:
    def test_get_utility_win(self, engine: GameEngine):
        """Simulate five in a row directly on the board."""
        state = engine.create_initial_state()
        # Manually set five in a row for p_black
        board = state["_arrays"]["board"]
        for i in range(5):
            board[i] = "p_black"
        state["env"]["winner"] = "p_black"
        state["env"]["phase"] = "game_over"

        assert engine.is_terminal(state)
        assert engine.get_utility(state, "p_black") == 1.0
        assert engine.get_utility(state, "p_white") == -1.0

    def test_get_utility_draw(self, engine: GameEngine):
        state = engine.create_initial_state()
        state["env"]["winner"] = None
        assert engine.get_utility(state, "p_black") == 0.0
        assert engine.get_utility(state, "p_white") == 0.0

    def test_observation_structure(self, engine: GameEngine):
        """get_observation returns projected views (derived views + env)."""
        state = engine.create_initial_state()
        obs = engine.get_observation(state, "p_black")
        # Should have at least 'cell' view (from derivedViews) and 'env'
        assert "cell" in obs
        assert "env" in obs
        # cell view should have 81 entities (9×9 board)
        assert len(obs["cell"]) == 81
        assert obs["env"]["turn"] == "p_black"

    def test_info_set_key(self, engine: GameEngine):
        state = engine.create_initial_state()
        key1 = engine.get_info_set_key(state, "p_black")
        key2 = engine.get_info_set_key(state, "p_white")
        assert isinstance(key1, str)
        assert len(key1) > 0
        # Perfect information game: both players see the same thing
        assert key1 == key2

    def test_load_state(self, engine: GameEngine):
        """load_state fills in default schema from any state dict."""
        ext_state = {
            "_arrays": {"board": ["p_black"] + [None] * 80},
            "env": {"phase": "playing", "turn": "p_white", "winner": None},
        }
        loaded = engine.load_state(ext_state)
        assert loaded["_arrays"]["board"][0] == "p_black"
        assert loaded["env"]["turn"] == "p_white"
        assert loaded["env"]["phase"] == "playing"


# ── Expr evaluator ────────────────────────────────────────────────


class TestExprEvaluator:
    def test_const(self, engine: GameEngine):
        assert engine.expr.eval({"const": 42}, {}) == 42

    def test_eq(self, engine: GameEngine):
        assert engine.expr.eval({"eq": [{"const": 1}, {"const": 1}]}, {})
        assert not engine.expr.eval({"eq": [{"const": 1}, {"const": 2}]}, {})

    def test_and_or_not(self, engine: GameEngine):
        assert engine.expr.eval({"and": [{"const": True}, {"const": True}]}, {})
        assert not engine.expr.eval({"and": [{"const": True}, {"const": False}]}, {})
        assert engine.expr.eval({"or": [{"const": False}, {"const": True}]}, {})
        assert engine.expr.eval({"not": {"const": False}}, {})

    def test_var(self, engine: GameEngine):
        ctx = {"$env": {"turn": "p_black"}}
        val = engine.expr.eval({"var": "$env.turn"}, ctx)
        assert val == "p_black"

    def test_template(self, engine: GameEngine):
        ctx = {"cell": {"x": 3, "y": 5}}
        val = engine.expr.eval({"template": "place:{$cell.x},{$cell.y}"}, ctx)
        # Note: template uses full var path inside {}
        assert val == "place:3,5"

    def test_count(self, engine: GameEngine):
        # count works on evaluator result
        result = engine.expr.eval({"count": [1, 2, 3]}, {})
        assert result == 3

    def test_switch(self, engine: GameEngine):
        expr = {
            "switch": [
                {"case": "p_black", "then": "black"},
                {"case": "p_white", "then": "white"},
            ],
            "input": {"const": "p_white"},
        }
        assert engine.expr.eval(expr, {}) == "white"

    def test_switch_default(self, engine: GameEngine):
        expr = {
            "switch": [
                {"case": "p_black", "then": "black"},
                {"case": "p_white", "then": "white"},
                {"then": "unknown"},
            ],
            "input": {"const": "p_red"},
        }
        assert engine.expr.eval(expr, {}) == "unknown"

    def test_arithmetic_expr(self, engine: GameEngine):
        ctx = {"board_size": 9, "y": 3, "x": 4}
        # The ExprEvaluator._eval_arithmetic resolves var names from context
        result = engine.expr.eval({"expr": "y * board_size + x"}, ctx)
        assert result == 31  # 3 * 9 + 4


# ── Different board sizes ─────────────────────────────────────────


class TestBoardSizes:
    @pytest.mark.parametrize("size", [3, 5, 7, 9, 11])
    def test_various_sizes(self, gomoku_rules: dict, size: int):
        rules = dict(gomoku_rules)
        rules["constants"] = dict(rules["constants"])
        rules["constants"]["board_size"] = size
        eng = GameEngine(rules, seed=42)
        state = eng.create_initial_state()
        board = state["_arrays"]["board"]
        assert len(board) == size * size
        actions = eng.get_legal_actions(state)
        assert len(actions) == size * size
