"""Layer 1: Translator — LLM-driven rule translation layer.

Translates natural-language strategy game rules into structured
``rules.json`` (v4.1 format) that Layer 2 (Env/Engine) can consume.

This layer is currently a placeholder: the Protocol is defined but
concrete implementations (LLM-based, template-based) are future work.
"""

from .protocol import (
    TranslatorProtocol,
    TranslateRequest,
    TranslateResponse,
    ValidationResult,
)
from .schema_validator import SchemaValidator

__all__ = [
    "TranslatorProtocol",
    "TranslateRequest",
    "TranslateResponse",
    "ValidationResult",
    "SchemaValidator",
]
