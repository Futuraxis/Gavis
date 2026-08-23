"""Natural-language rule translator facade for Layer 1.

This module provides the public Layer 1 entrypoint for callers that want
to submit rule text, plus optional externally collected frontend hints,
and receive executable ``rules.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .llm_translator import LLMRuleTranslator, RuleLLMClient
from .protocol import TranslateRequest, TranslateResponse, TranslatorProtocol
from .template_translator import TemplateTranslator

logger = logging.getLogger(__name__)


class NaturalLanguageRuleTranslator:
    """Translate natural-language game rules into executable rules JSON."""

    def __init__(self, backend: TranslatorProtocol | None = None) -> None:
        self.backend = backend or TemplateTranslator()

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        """Translate ``request`` into a structured ``rules.json`` response."""
        return self.backend.translate(request)

    def translate_text(
        self,
        rule_text: str,
        *,
        source_lang: str = "zh",
        game_name: str | None = None,
        external_frontend: dict[str, Any] | None = None,
    ) -> TranslateResponse:
        """Translate raw rule text and optional collected frontend hints."""
        return self.translate(
            TranslateRequest(
                rule_text=rule_text,
                source_lang=source_lang,
                game_name=game_name,
                external_frontend=external_frontend,
            )
        )


def translate_rules_json(
    rule_text: str,
    *,
    source_lang: str = "zh",
    game_name: str | None = None,
    external_frontend: dict[str, Any] | None = None,
    run_engine_validation: bool = True,
    use_llm: bool = False,
    llm_client: RuleLLMClient | None = None,
    llm_model_path: str | Path | None = None,
) -> TranslateResponse:
    """Translate natural-language rule input into a ``rules.json`` response.

    When ``use_llm`` is False, ``llm_client`` and ``llm_model_path`` are
    ignored; a log warning is emitted so callers are not silently misled.
    """
    if not use_llm and (llm_client is not None or llm_model_path is not None):
        logger.warning("translate_rules_json: use_llm=False，忽略传入的 llm_client/llm_model_path")
    if use_llm:
        translator = NaturalLanguageRuleTranslator(
            LLMRuleTranslator(
                llm_client,
                model_path=llm_model_path,
                run_engine_validation=run_engine_validation,
            )
        )
        return translator.translate_text(
            rule_text,
            source_lang=source_lang,
            game_name=game_name,
            external_frontend=external_frontend,
        )

    translator = NaturalLanguageRuleTranslator(
        TemplateTranslator(run_engine_validation=run_engine_validation),
    )
    return translator.translate_text(
        rule_text,
        source_lang=source_lang,
        game_name=game_name,
        external_frontend=external_frontend,
    )
