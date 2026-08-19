"""Layer 1: Translator — rule translation layer.

Translates natural-language strategy game rules into structured
``rules.json`` that Layer 2 (Env/Engine) can consume.

The current implementation is deterministic and template-based for known
Gavis games. Future LLM translators can implement the same protocol.
"""

from .engine_validator import EngineValidator
from .protocol import (
    TranslateRequest,
    TranslateResponse,
    TranslatorProtocol,
    ValidationResult,
)
from .rule_parser import ParsedRuleRequest, RuleParser
from .schema_validator import SchemaValidator
from .template_translator import TemplateTranslator

__all__ = [
    "TranslatorProtocol",
    "TranslateRequest",
    "TranslateResponse",
    "ValidationResult",
    "EngineValidator",
    "SchemaValidator",
    "ParsedRuleRequest",
    "RuleParser",
    "TemplateTranslator",
]
