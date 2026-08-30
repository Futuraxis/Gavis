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
    strict_llm: bool = False,
    llm_client: RuleLLMClient | None = None,
    llm_model: str | None = None,
    llm_model_path: str | Path | None = None,
) -> TranslateResponse:
    """Translate natural-language rule input into a ``rules.json`` response.

    ``llm_model`` names the model for the unified LLM client (e.g.
    ``"qwen3:8b"``).  ``llm_model_path`` is a deprecated alias (previously
    a local model directory); when provided it is echoed as the model name
    and a warning is logged.

    When ``use_llm`` is False, ``llm_client`` / ``llm_model`` / legacy
    ``llm_model_path`` are ignored; a log warning is emitted so callers are
    not silently misled.

    ``strict_llm=True`` (meaningful only with ``use_llm=True``) turns any
    LLM failure — API 4xx/5xx, unreachable endpoint, timeout, unparseable
    output — into a failed response with the real error in
    ``validation.errors``, instead of silently falling back to templates.
    Callers that explicitly request LLM translation (e.g. the platform
    custom-game flow) should enable it so an API error cannot produce a
    silently wrong template game.
    """
    if llm_model_path is not None:
        logger.warning("translate_rules_json: llm_model_path 已废弃，请改用 llm_model（模型名）")
        llm_model = llm_model or str(llm_model_path)
    if not use_llm and (llm_client is not None or llm_model is not None):
        logger.warning("translate_rules_json: use_llm=False，忽略传入的 llm_client/llm_model")
    if use_llm:
        translator = NaturalLanguageRuleTranslator(
            LLMRuleTranslator(
                llm_client,
                llm_model=llm_model,
                run_engine_validation=run_engine_validation,
                strict_llm=strict_llm,
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
