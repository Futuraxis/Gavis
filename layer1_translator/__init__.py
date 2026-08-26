"""Layer 1: Translator — rule translation layer.

Translates natural-language strategy game rules into structured
``rules.json`` that Layer 2 (Env/Engine) can consume.

The current implementation is deterministic: known Gavis games use
templates, while supported unknown variants can be generated from rule
families. Future LLM translators can implement the same protocol.
"""

from __future__ import annotations

from .datasets import (
    RuleExample,
    RuleJsonDataset,
    build_synthetic_examples,
    dump_examples_json,
    load_jsonl_examples,
)
from .engine_validator import EngineValidator
from .external_frontend_reader import ExternalFrontendRuleReader, ExternalRuleInput
from .llm_translator import LLMRuleTranslator
from .local_client import (
    LLMTranslatorError,
    LocalTransformersRuleClient,
    OpenAICompatibleRuleClient,
    RuleLLMClient,
)
from .natural_language_translator import NaturalLanguageRuleTranslator, translate_rules_json
from .prompt_builder import RulePromptBuilder
from .protocol import (
    TranslateRequest,
    TranslateResponse,
    TranslatorProtocol,
    ValidationResult,
)
from .rule_family_builder import RuleFamilyBuilder, RuleFamilySpec
from .rule_parser import ParsedRuleRequest, RuleParser
from .schema_validator import SchemaValidator
from .template_translator import TemplateTranslator
from .variant_translator import VariantTranslator, translate_variant_rules

__all__ = [
    "TranslatorProtocol",
    "TranslateRequest",
    "TranslateResponse",
    "ValidationResult",
    "EngineValidator",
    "ExternalFrontendRuleReader",
    "ExternalRuleInput",
    "NaturalLanguageRuleTranslator",
    "LLMRuleTranslator",
    "LLMTranslatorError",
    "RulePromptBuilder",
    "LocalTransformersRuleClient",
    "OpenAICompatibleRuleClient",
    "RuleLLMClient",
    "RuleExample",
    "RuleJsonDataset",
    "build_synthetic_examples",
    "dump_examples_json",
    "load_jsonl_examples",
    "SchemaValidator",
    "ParsedRuleRequest",
    "RuleFamilyBuilder",
    "RuleFamilySpec",
    "RuleParser",
    "TemplateTranslator",
    "VariantTranslator",
    "translate_rules_json",
    "translate_variant_rules",
]
