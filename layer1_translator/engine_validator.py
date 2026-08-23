"""Engine-level smoke validation for translated rules.

This module is the **only** Layer 1 component that touches Layer 2 — the
authorized L1→L2 validation channel: Layer 1 produces rules JSON that
Layer 2 consumes, so Layer 2 owns the engine smoke-validation service
(``layer2_engine.core.smoke_validator``) and this module delegates to it.
It does not import solvers or frontends.
"""

from __future__ import annotations

from typing import Any

from layer2_engine.core.smoke_validator import smoke_validate

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

        smoke = smoke_validate(rules, seed=self.seed)
        errors.extend(smoke.errors)
        warnings.extend(smoke.warnings)
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
