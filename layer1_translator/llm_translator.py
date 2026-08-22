"""LLM-backed natural-language rule translator orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine_validator import EngineValidator
from .local_client import (
    DEFAULT_LOCAL_MODEL_DIR,
    LLMTranslatorError,
    LocalTransformersRuleClient,
    OpenAICompatibleRuleClient,
    RuleLLMClient,
)
from .prompt_builder import CONTROL_CHARS_RE, RulePromptBuilder, system_prompt
from .protocol import TranslateRequest, TranslateResponse, ValidationResult
from .schema_validator import SchemaValidator
from .template_translator import TemplateTranslator

MAX_LLM_REPLY_LEN = 512_000


class LLMRuleTranslator:
    """Translate natural-language rules with an LLM and validation loop."""

    def __init__(
        self,
        client: RuleLLMClient | None = None,
        *,
        model_path: str | Path | None = None,
        run_engine_validation: bool = True,
        fallback: TemplateTranslator | None = None,
        max_repair_attempts: int = 1,
        max_tokens: int = 8192,
        strict_llm: bool = False,
        prompt_builder: RulePromptBuilder | None = None,
    ) -> None:
        self.client = client
        self.model_path = model_path or DEFAULT_LOCAL_MODEL_DIR
        self.run_engine_validation = run_engine_validation
        self.fallback = fallback
        self.max_repair_attempts = max(0, max_repair_attempts)
        self.max_tokens = max_tokens
        self.strict_llm = strict_llm
        self.prompt_builder = prompt_builder or RulePromptBuilder()
        self.engine_validator = EngineValidator()

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        """Return LLM-generated rules JSON, optionally falling back to templates."""
        warnings: list[str] = []
        try:
            client = self.client or LocalTransformersRuleClient(model_path=self.model_path)
        except LLMTranslatorError as exc:
            warnings.append("本地 LLM 不可用，已使用模板兜底")
            return self._fallback_or_error(request, ValidationResult(valid=False, errors=[str(exc)]), warnings)

        messages = self.prompt_builder.build_initial_messages(request)
        attempts = self.max_repair_attempts + 1
        last_rules: dict[str, Any] = {}
        last_validation = ValidationResult(valid=False, errors=["LLM 未返回可验证的 rules JSON"])

        for attempt in range(attempts):
            try:
                raw = client.complete(messages, max_tokens=self.max_tokens)
                rules = self._parse_rules(raw)
            except LLMTranslatorError as exc:
                last_validation = ValidationResult(valid=False, errors=[str(exc)])
                warnings.append("LLM 生成失败，尝试模板兜底")
                break

            last_rules = rules
            last_validation = self._validate(rules)
            if last_validation.valid:
                last_validation.warnings.extend(warnings)
                last_validation.warnings.append("使用 LLM 生成 rules.json")
                return TranslateResponse(
                    rules_json=rules,
                    confidence=self._confidence(last_validation, llm_valid=True),
                    validation=last_validation,
                )
            if attempt < attempts - 1:
                messages = self.prompt_builder.build_repair_messages(request, rules, last_validation)

        fallback_response = self._fallback_or_error(request, last_validation, warnings)
        if fallback_response.rules_json:
            return fallback_response
        return TranslateResponse(
            rules_json=last_rules,
            confidence=0.2 if last_rules else fallback_response.confidence,
            validation=fallback_response.validation,
        )

    @staticmethod
    def _system_prompt() -> str:
        """Compatibility wrapper for training code and older callers."""
        return system_prompt()

    @classmethod
    def _parse_rules(cls, raw: str) -> dict[str, Any]:
        text = CONTROL_CHARS_RE.sub("", raw or "")[:MAX_LLM_REPLY_LEN].strip()
        if not text:
            raise LLMTranslatorError("LLM 返回为空")
        parsed = cls._decode_json_object(text)
        if "rules_json" in parsed and isinstance(parsed["rules_json"], dict):
            parsed = parsed["rules_json"]
        if not isinstance(parsed, dict):
            raise LLMTranslatorError("LLM 输出不是 JSON object")
        return parsed

    @staticmethod
    def _decode_json_object(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        candidates = [text]
        if "```" in text:
            candidates.extend(part.strip("` \n") for part in text.split("```") if "{" in part)
        for candidate in candidates:
            stripped = candidate.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                return value
            for index, char in enumerate(stripped):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(stripped[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
        raise LLMTranslatorError("无法从 LLM 输出中解析 JSON object")

    def _validate(self, rules: dict[str, Any]) -> ValidationResult:
        if self.run_engine_validation:
            return self.engine_validator.validate(rules)
        return SchemaValidator.validate(rules)

    def _fallback_or_error(
        self,
        request: TranslateRequest,
        last_validation: ValidationResult,
        warnings: list[str],
    ) -> TranslateResponse:
        if self.strict_llm:
            last_validation.warnings.extend(warnings)
            return TranslateResponse(rules_json={}, confidence=0.0, validation=last_validation)

        fallback = self.fallback or TemplateTranslator(run_engine_validation=self.run_engine_validation)
        fallback_response = fallback.translate(request)
        if fallback_response.validation is None:
            fallback_response.validation = ValidationResult(valid=bool(fallback_response.rules_json))
        fallback_response.validation.warnings.extend(warnings)
        fallback_response.validation.warnings.extend(
            [
                "LLM 输出未通过校验，已使用模板兜底",
                *last_validation.errors,
                *last_validation.warnings,
            ]
        )
        return fallback_response

    @staticmethod
    def _confidence(validation: ValidationResult, *, llm_valid: bool) -> float:
        if not llm_valid or not validation.valid:
            return 0.2
        penalty = min(0.2, len(validation.warnings) * 0.03)
        return round(0.85 - penalty, 2)


__all__ = [
    "DEFAULT_LOCAL_MODEL_DIR",
    "LLMRuleTranslator",
    "LLMTranslatorError",
    "LocalTransformersRuleClient",
    "OpenAICompatibleRuleClient",
    "RuleLLMClient",
]
