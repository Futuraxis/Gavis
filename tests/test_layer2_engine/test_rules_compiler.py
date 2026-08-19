"""Tests for the rules compiler (codegen) — compiled artifacts must behave
identically to the interpreter across probe states."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


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
