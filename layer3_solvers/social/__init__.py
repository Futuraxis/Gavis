"""Social/language games — LLM policy hooks.

Entry point for language-driven games (Werewolf, Undercover, poker
banter): a policy decides what to SAY and whom to VOTE for, backed by
an LLM client or a template fallback.
"""

from __future__ import annotations

from .base import LanguageObservation, LanguagePolicy
from .llm_policy import LLMClient, LLMPolicy, OpenAICompatibleClient
from .template_policy import TemplatePolicy

__all__ = [
    "LanguageObservation",
    "LanguagePolicy",
    "LLMClient",
    "LLMPolicy",
    "OpenAICompatibleClient",
    "TemplatePolicy",
]
