"""Tests for Layer 2: GameEngine (stochastic gomoku)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import (
    ActionInstance,
    ChanceOutcome,
    create_gomoku_state,
    clone_state,
    check_five_in_row,
)

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
        assert state["board_size"] == 9
        assert len(state["_board"]) == 81
        assert all(c is None for c in state["_board"])
        assert state["env"]["phase"] == "playing"
        assert state["env"]["turn"]["currentPlayerId"] == "p_black"

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
        state = create_gomoku_state(5)
        cloned = clone_state(state)
        assert cloned["board_size"] == 5
        assert cloned["_board"] == state["_board"]
        # Mutating clone should not affect original
        cloned["_board"][0] = "p_black"
        assert state["_board"][0] is None


# ── Actions ───────────────────────────────────────────────────────


class TestActions:
    def test_apply_action_returns_new_state(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        assert len(actions) > 0
        new_state = engine.apply_action(state, actions[0])
        # Should be a different object
        assert new_state is not state
        # Board should have changed
        placed = sum(1 for c in new_state["_board"] if c is not None)
        assert placed == 1

    def test_place_black_then_white(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)

        # Black places
        state = engine.apply_action(state, actions[0])
        assert state["_board"][0] == "p_black"

        # Chance node (vanish check)
        assert engine.get_node_type(state) == "chance"
        outcomes = engine.get_chance_outcomes(state)
        assert len(outcomes) == 2
        assert outcomes[0].key in ("vanish", "keep")
        outcome, state = engine.sample_chance(state)

        # Now white's turn
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
        assert new_state["_board"][40] is None

    def test_apply_chance_keep(self, engine: GameEngine):
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        state = engine.apply_action(state, actions[40])

        outcomes = engine.get_chance_outcomes(state)
        keep = [o for o in outcomes if o.key == "keep"][0]
        new_state = engine.apply_chance(state, keep)
        assert new_state["_board"][40] == "p_black"

    def test_sample_chance(self, engine: GameEngine):
        """Over many samples, verify approx 50% vanish rate."""
        state = engine.create_initial_state()
        actions = engine.get_legal_actions(state)
        state = engine.apply_action(state, actions[0])

        vanish_count = 0
        n = 200
        for i in range(n):
            s = clone_state(state)
            # Use seed to get deterministic sampling
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
        for i in range(5):
            state["_board"][i] = "p_black"
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
        state = engine.create_initial_state()
        obs = engine.get_observation(state, "p_black")
        assert "board" in obs
        assert "board_size" in obs
        assert "current_player" in obs
        assert obs["board_size"] == 9
        assert obs["current_player"] == "p_black"
        assert len(obs["board"]) == 81

    def test_info_set_key(self, engine: GameEngine):
        state = engine.create_initial_state()
        key1 = engine.get_info_set_key(state, "p_black")
        key2 = engine.get_info_set_key(state, "p_white")
        assert isinstance(key1, str)
        assert key1 != key2

    def test_load_state_valid(self, engine: GameEngine):
        state = engine.create_initial_state()
        state["_board"][0] = "p_black"
        loaded = engine.load_state(state)
        assert loaded["_board"][0] == "p_black"
        assert "cell_0_0" in loaded["nodes"]

    def test_load_state_invalid_board(self, engine: GameEngine):
        with pytest.raises(ValueError, match="_board"):
            engine.load_state({"_board": [None] * 10, "env": {}})


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
        ctx = {"$env": {"turn": {"currentPlayerId": "p_black"}}}
        val = engine.expr.eval({"var": "$env.turn.currentPlayerId"}, ctx)
        assert val == "p_black"

    def test_template(self, engine: GameEngine):
        ctx = {"cell": {"props": {"x": 3, "y": 5}}}
        val = engine.expr.eval(
            {"template": "place:{cell.props.x},{cell.props.y}"}, ctx
        )
        assert val == "place:3,5"

    def test_count(self, engine: GameEngine):
        assert engine.expr.eval({"count": [1, 2, 3]}, {}) == 3


# ── Different board sizes ─────────────────────────────────────────


class TestBoardSizes:
    @pytest.mark.parametrize("size", [3, 5, 7, 9, 11])
    def test_various_sizes(self, gomoku_rules: dict, size: int):
        rules = dict(gomoku_rules)
        rules["constants"] = dict(rules["constants"])
        rules["constants"]["board_size"] = size
        eng = GameEngine(rules, seed=42)
        state = eng.create_initial_state()
        assert state["board_size"] == size
        assert len(state["_board"]) == size * size
        actions = eng.get_legal_actions(state)
        assert len(actions) == size * size
