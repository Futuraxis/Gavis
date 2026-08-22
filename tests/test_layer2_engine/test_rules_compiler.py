"""Tests for the rules compiler (codegen) — compiled artifacts must behave
identically to the interpreter across probe states."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.rules_compiler import _Gen

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def _mini_rules(phase: str = "c1", explicit: list[dict] | None = None) -> dict:
    """Synthetic rules with a single explicit chance template (compilable)."""
    return {
        "constants": {},
        "players": [{"id": "p0"}],
        "groundState": {
            "arrays": {},
            "env": {"type": "env", "fields": {"phase": {"type": "str", "initial": phase}}},
        },
        "derivedViews": {},
        "phases": [{"id": phase, "actions": []}],
        "actions": [],
        "effectors": {},
        "chance": [{"phases": [phase], "probability": {"explicit": explicit}}],
        "terminal": [],
        "utility": [],
        "visibility": {},
        "queries": {},
        "functions": {},
    }


def _load(game: str) -> GameEngine:
    with open(RULES_DIR / f"{game}.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=42)


def _collect_states(engine: GameEngine, n: int = 6) -> list[dict]:
    """Advance n random-legal-action steps (resolving chance), collecting states."""
    states = [engine.create_initial_state()]
    state = states[0]
    for _ in range(n):
        nt = engine.get_node_type(state)
        if nt == "player":
            actions = engine.get_legal_actions(state)
            if not actions:
                break
            state = engine.apply_action(state, actions[0])
        elif nt == "chance":
            _, state = engine.sample_chance(state)
        else:
            break
        states.append(state)
    return states


@pytest.mark.parametrize("game", ["stochastic_gomoku", "moon_chess"])
def test_compiler_artifacts_present(game: str):
    engine = _load(game)
    compiled = engine._compiled
    assert compiled is not None, "rules compiler should produce artifacts"
    assert compiled.is_terminal is not None
    assert compiled.legal_actions is not None
    assert compiled.materialize is not None
    assert {"cell", "player"} <= set(compiled._views.keys())


@pytest.mark.parametrize("game", ["stochastic_gomoku", "moon_chess"])
def test_compiled_matches_interpreter(game: str):
    """Compiled hot paths must agree with the interpreter on all states."""
    engine = _load(game)
    compiled = engine._compiled
    assert compiled is not None

    for state in _collect_states(engine):
        # Terminal
        assert compiled.is_terminal(state) == engine._interp_is_terminal(state)
        # Legal actions (order-sensitive canonical keys)
        mine = [a.canonical_key for a in compiled.legal_actions(state)]
        theirs = [a.canonical_key for a in engine._interp_legal_actions(state)]
        assert mine == theirs
        # Chance outcomes
        if compiled.chance_outcomes is not None:
            mine = [(o.key, o.probability, o.effect_ref, o.canonical_key) for o in compiled.chance_outcomes(state)]
            theirs = [
                (o.key, o.probability, o.effect_ref, o.canonical_key) for o in engine._interp_chance_outcomes(state)
            ]
            assert mine == theirs
        # Views
        for vname, fn in compiled._views.items():
            assert fn(state) == engine._view_engine.materialize(state, vname)


def test_switch_falsy_branch_value():
    """P1-10: compiled switch must return a matched branch's falsy ``then``
    value instead of falling through the old ``or``-chain."""
    gen = _Gen({}, {}, {}, {})
    src = gen.expr({"switch": [{"case": 1, "then": 0}, {"case": 2, "then": "x"}], "input": {"const": 1}})
    ns: dict = {}
    assert eval(compile(src, "<switch>", "eval"), ns) == 0
    # Empty-string branch value is also preserved (previously fell to default).
    src2 = gen.expr({"switch": [{"case": 1, "then": ""}, {"case": 2, "then": "x"}], "input": {"const": 1}})
    assert eval(compile(src2, "<switch>", "eval"), {}) == ""
    # Unmatched input → default / None, mirroring the interpreter.
    src3 = gen.expr({"switch": [{"case": 1, "then": 0}, {"default": True, "then": "d"}], "input": {"const": 9}})
    assert eval(compile(src3, "<switch>", "eval"), {}) == "d"


def test_chance_multi_template_first_match():
    """P1-11: when two explicit chance templates share a phase, the compiled
    path must return the first matching template (interpreter semantics),
    not a union of both."""
    rules = _mini_rules(explicit=[{"outcome": "a", "prob": 0.5}])
    rules["chance"].append({"phases": ["c1"], "probability": {"explicit": [{"outcome": "b", "prob": 0.5}]}})
    engine = GameEngine(rules, seed=1)
    state = engine.create_initial_state()
    compiled = engine._compiled
    assert compiled is not None and compiled.chance_outcomes is not None
    assert [o.key for o in compiled.chance_outcomes(state)] == ["a"]
    assert [o.key for o in engine._interp_chance_outcomes(state)] == ["a"]


def test_sample_chance_normalizes_probabilities():
    """P1-12: non-normalized explicit tables (0.3 + 0.4) must sample with
    normalized weights instead of biasing toward the last outcome."""
    engine = GameEngine(
        _mini_rules(explicit=[{"outcome": "a", "prob": 0.3}, {"outcome": "b", "prob": 0.4}]),
        seed=1,
    )
    state = engine.create_initial_state()
    counts = Counter(engine.sample_chance(state)[0].key for _ in range(2000))
    # 3/7 vs 4/7 → 857/1143 within sampling noise.
    assert abs(counts["a"] - 2000 * 3 / 7) < 90
    assert abs(counts["b"] - 2000 * 4 / 7) < 90


def test_sample_chance_empty_outcomes_raises():
    """P1-12: a chance node that expands to zero outcomes is a rules bug and
    must raise a descriptive ValueError (previously IndexError on [-1])."""
    engine = GameEngine(_mini_rules(explicit=[]), seed=1)
    with pytest.raises(ValueError, match="phase"):
        engine.sample_chance(engine.create_initial_state())


def test_construction_does_not_consume_rng():
    """Engine construction (incl. probe validation) must not shift the rng stream."""
    a = _load("stochastic_gomoku")
    b = _load("stochastic_gomoku")
    # Same seed → identical first chance draw.
    sa = a.create_initial_state()
    sb = b.create_initial_state()
    outcome_a, _ = a.sample_chance(sa) if a.get_node_type(sa) != "player" else (None, sa)
    # Force a chance node: place a stone first.
    if outcome_a is None:
        action = a.get_legal_actions(sa)[0]
        sa = a.apply_action(sa, action)
        sb = b.apply_action(sb, b.get_legal_actions(sb)[0])
    outcome_a, _ = a.sample_chance(sa)
    outcome_b, _ = b.sample_chance(sb)
    assert outcome_a.key == outcome_b.key
