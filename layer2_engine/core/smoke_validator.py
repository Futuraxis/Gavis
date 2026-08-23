"""Engine smoke-validation service for rule producers.

Owned by Layer 2: the question "can this rules JSON boot a ``GameEngine``
and expose basic dynamics" belongs to the engine, not to the translator.
``layer1_translator.engine_validator`` consumes this service through its
single authorized L1→L2 channel (review round 2026-08).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine import GameEngine


@dataclass
class SmokeValidation:
    """Errors and warnings from one smoke-validation run."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def smoke_validate(rules: dict[str, Any], seed: int = 42) -> SmokeValidation:
    """Boot the engine once and probe one initial transition.

    Construction and state progression are captured separately so a
    real engine defect surfaces as a distinct, typed error instead of
    being collapsed into a generic "validation failed".
    """
    result = SmokeValidation()
    try:
        engine = GameEngine(rules, seed=seed)
    except Exception as exc:
        result.errors.append(f"GameEngine 构造失败: {type(exc).__name__}: {exc}")
        return result
    try:
        state = engine.create_initial_state()
        node_type = engine.get_node_type(state)
        if node_type == "player":
            actions = engine.get_legal_actions(state)
            if not actions:
                result.warnings.append("初始 player 节点没有合法动作")
            else:
                engine.apply_action(state, actions[0])
        elif node_type == "chance":
            outcomes = engine.get_chance_outcomes(state)
            if not outcomes:
                result.errors.append("初始 chance 节点没有 chance outcomes")
            else:
                engine.apply_chance(state, outcomes[0])
        elif node_type != "terminal":
            result.errors.append(f"未知节点类型: {node_type}")
    except Exception as exc:
        result.errors.append(f"Engine smoke validation failed: {type(exc).__name__}: {exc}")
    return result
