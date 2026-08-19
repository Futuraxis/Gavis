"""Engine-level smoke validation for translated rules.

This module is the only Layer 1 component that touches Layer 2. It does
not import solvers or frontends; it simply asks ``GameEngine`` whether a
candidate rules JSON can create a state and expose basic game dynamics.
"""

from __future__ import annotations

from typing import Any

from layer2_engine.core.engine import GameEngine

from .protocol import ValidationResult
from .schema_validator import SchemaValidator


class EngineValidator:
    """Validate translated rules by loading them into ``GameEngine``."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def validate(self, rules: dict[str, Any]) -> ValidationResult:
        """Run schema validation plus a minimal engine smoke test."""
        schema_result = SchemaValidator.validate(rules)
        errors = list(schema_result.errors)
        warnings = list(schema_result.warnings)
        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        try:
            engine = GameEngine(rules, seed=self.seed)
            state = engine.create_initial_state()
            node_type = engine.get_node_type(state)
            if node_type == "player":
                actions = engine.get_legal_actions(state)
                if not actions:
                    warnings.append("初始 player 节点没有合法动作")
                else:
                    engine.apply_action(state, actions[0])
            elif node_type == "chance":
                outcomes = engine.get_chance_outcomes(state)
                if not outcomes:
                    errors.append("初始 chance 节点没有 chance outcomes")
                else:
                    engine.apply_chance(state, outcomes[0])
            elif node_type != "terminal":
                errors.append(f"未知节点类型: {node_type}")
        except Exception as exc:
            errors.append(f"Engine smoke validation failed: {exc}")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
