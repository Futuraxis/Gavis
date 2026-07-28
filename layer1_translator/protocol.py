"""Translator Protocol — the sole contract between Layer 1 and Layer 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class TranslateRequest:
    """Input: a description of a strategy game's rules."""
    rule_text: str
    source_lang: str = "zh"          # language hint for the translator
    game_name: str | None = None     # optional known game name


@dataclass
class ValidationResult:
    """Result of validating a translated ``rules.json`` against the engine."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TranslateResponse:
    """Output: a validated ``rules.json`` dict ready for GameEngine."""
    rules_json: dict
    confidence: float = 0.0           # 0.0 … 1.0
    validation: ValidationResult | None = None


@runtime_checkable
class TranslatorProtocol(Protocol):
    """A translator that converts game-rule text into executable rules.json.

    Implementations may use LLM calls, template filling, or any other
    approach.  The only requirement is that the output passes
    ``SchemaValidator.validate()``.
    """

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        """Translate game-rule text into a structured rules.json."""
        ...
