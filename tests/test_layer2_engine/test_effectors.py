"""Tests for the v5.1 effector ops (Layer 2).

Covers ``remove`` (multiset difference), ``setArray`` (wholesale
replacement, env-list clone-sharing safe), and expression ``array``
names, plus the two regression traps from the mahjong plan:
  - a field name that collides with an expression key must not break
    append's value-dict detection (``_EXPR_KEYS`` completeness)
  - env lists are shared by reference across cloned states — effectors
    must rebind them, never mutate in place
"""

from __future__ import annotations

from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import clone_state

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


@pytest.fixture
def engine() -> GameEngine:
    return _test_engine()


def _test_engine() -> GameEngine:
    rules = {
        "constants": {"board_size": 3},
        "groundState": {
            "hand": {"type": "array", "mutable": True, "element": "string"},
            "hand_p0": {"type": "array", "mutable": True, "element": "string"},
            "melds": {"type": "array", "mutable": True, "element": "dict"},
            "env": {
                "type": "env",
                "fields": {
                    "phase": {"type": "string", "initial": "playing"},
                    "claim_queue": {"type": "list", "initial": ["a", "b", "c"]},
                },
            },
        },
        "derivedViews": {},
        "effectors": {
            "do_remove": {
                "ops": [
                    {"op": "remove", "array": "hand", "value": {"var": "$target"}, "count": {"var": "$howmany"}},
                ]
            },
            "do_remove_one": {
                "ops": [
                    {"op": "remove", "array": "hand", "value": {"var": "$target"}},
                ]
            },
            "do_remove_expr": {
                "ops": [
                    {"op": "remove", "array": {"template": "hand_{$pid}"}, "value": {"var": "$target"}},
                ]
            },
            "do_remove_env": {
                "ops": [
                    {"op": "remove", "array": "claim_queue", "value": {"const": "b"}},
                ]
            },
            "do_set_array": {
                "ops": [
                    {
                        "op": "setArray",
                        "array": "melds",
                        "value": {"map": {"list": {"var": "$list"}, "expr": {"const": {"type": "peng"}}}},
                    },
                ]
            },
            "do_set_env_list": {
                "ops": [
                    {"op": "setArray", "array": "claim_queue", "value": {"const": ["x", "y"]}},
                ]
            },
            "do_append_expr": {
                "ops": [
                    {"op": "append", "array": {"template": "hand_{$pid}"}, "value": {"var": "$tile"}},
                ]
            },
            "do_append_expr_value": {
                "ops": [
                    {"op": "append", "array": "melds", "value": {"sum": {"var": "$list"}}},
                ]
            },
            "do_append_value_dict": {
                "ops": [
                    {"op": "append", "array": "melds", "value": {"type": "chi", "count": {"var": "$n"}}},
                ]
            },
        },
        "actions": [],
        "phases": [],
        "chance": [],
        "terminal": [],
        "utility": [],
        "visibility": {"default": "public"},
    }
    return GameEngine(rules, seed=1)


def _state(engine: GameEngine, **env) -> dict:
    state = engine.create_initial_state()
    state["env"].update(env)
    return state


def _run(engine: GameEngine, state: dict, effector: str, **params) -> dict:
    ctx = engine._build_context(state)  # noqa: SLF001 — engine-layer test
    ctx.update(params)
    engine._execute_effector(effector, ctx, state)  # noqa: SLF001
    return state


# ── remove ─────────────────────────────────────────────────────────────


class TestRemove:
    def test_remove_one_match(self, engine):
        state = _state(engine)
        state["_arrays"]["hand"] = ["m1", "m2", "m1", "m3"]
        _run(engine, state, "do_remove_one", target="m1")
        assert state["_arrays"]["hand"] == ["m2", "m1", "m3"]

    def test_remove_count(self, engine):
        state = _state(engine)
        state["_arrays"]["hand"] = ["m1", "m2", "m1", "m1", "m3"]
        _run(engine, state, "do_remove", target="m1", howmany=2)
        assert state["_arrays"]["hand"] == ["m2", "m1", "m3"]

    def test_remove_no_match(self, engine):
        state = _state(engine)
        state["_arrays"]["hand"] = ["m1", "m2"]
        _run(engine, state, "do_remove_one", target="p9")
        assert state["_arrays"]["hand"] == ["m1", "m2"]

    def test_remove_expression_array_name(self, engine):
        state = _state(engine)
        state["_arrays"]["hand_p0"] = ["m1", "m1", "m2"]
        _run(engine, state, "do_remove_expr", pid="p0", target="m1")
        assert state["_arrays"]["hand_p0"] == ["m1", "m2"]

    def test_remove_missing_array_ignored(self, engine):
        state = _state(engine)
        _run(engine, state, "do_remove_one", target="m1")  # 'hand' empty
        assert state["_arrays"]["hand"] == []


# ── setArray ───────────────────────────────────────────────────────────


class TestSetArray:
    def test_replace_ground_array(self, engine):
        state = _state(engine)
        state["_arrays"]["melds"] = [{"type": "chi"}]
        _run(engine, state, "do_set_array", list=["a", "b"])
        assert state["_arrays"]["melds"] == [{"type": "peng"}, {"type": "peng"}]

    def test_set_env_list_rebinds(self, engine):
        """Env lists are shared across clones — setArray must rebind."""
        state = _state(engine)
        _run(engine, state, "do_set_env_list")
        assert state["env"]["claim_queue"] == ["x", "y"]

    def test_remove_env_list_does_not_leak_into_original(self, engine):
        """The clone-sharing regression: mutating a clone's env list must
        leave the original state untouched."""
        state = _state(engine)
        original = clone_state(state)
        clone = clone_state(state)
        _run(engine, clone, "do_remove_env")
        assert clone["env"]["claim_queue"] == ["a", "c"]
        assert original["env"]["claim_queue"] == ["a", "b", "c"]


# ── append: expression value vs value-dict detection ───────────────────


class TestAppendDetection:
    def test_expression_value(self, engine):
        """append with an expression-shaped value evaluates it."""
        state = _state(engine)
        _run(engine, state, "do_append_expr_value", list=[1, 2, 3])
        assert state["_arrays"]["melds"] == [6]

    def test_value_dict_with_expr_keys(self, engine):
        """A field named like an expression key ('count') must still be a
        value-dict field — the _EXPR_KEYS regression trap."""
        state = _state(engine)
        _run(engine, state, "do_append_value_dict", n=3)
        assert state["_arrays"]["melds"] == [{"type": "chi", "count": 3}]

    def test_append_expression_array_name(self, engine):
        state = _state(engine)
        _run(engine, state, "do_append_expr", pid="p0", tile="m1")
        assert state["_arrays"]["hand_p0"] == ["m1"]
