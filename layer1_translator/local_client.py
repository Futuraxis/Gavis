"""Layer 1 LLM client surface — unified via ``layer2_engine.core.llm``.

Layer 1 keeps only the protocol it injects into translators (so tests can
pass deterministic fakes) plus the translator-level error type.  The
concrete transport is the project's single LLM client
(``layer2_engine.core.llm.LLMClient``) — the previously separate
``LocalTransformersRuleClient`` (torch/transformers local model) and
``OpenAICompatibleRuleClient`` copies were removed in the LLM unification;
both are covered by the same OpenAI-compatible endpoint.
"""

from __future__ import annotations

from typing import Protocol

from layer2_engine.core.llm import LLMClient as _UnifiedLLMClient

LLMClient = _UnifiedLLMClient  # 统一客户端别名（Layer 1 消费者经此引用）


class RuleLLMClient(Protocol):
    """Minimal chat-completion surface for rule translation (injection point).

    ``layer2_engine.core.llm.LLMClient`` satisfies this protocol; tests use
    deterministic in-memory fakes with the same shape.
    """

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Return the assistant's reply text for ``messages``."""


class LLMTranslatorError(Exception):
    """LLM translation failed before a valid candidate could be validated."""


__all__ = ["LLMClient", "LLMTranslatorError", "RuleLLMClient"]