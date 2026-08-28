"""LLM-backed natural-language rule translator orchestration."""

from __future__ import annotations

import json
from typing import Any

from layer2_engine.core.llm import LLMClient

from .engine_validator import EngineValidator
from .local_client import LLMTranslatorError, RuleLLMClient
from .prompt_builder import CONTROL_CHARS_RE, RulePromptBuilder
from .protocol import TranslateRequest, TranslateResponse, ValidationResult
from .schema_validator import SchemaValidator
from .template_translator import TemplateTranslator

MAX_LLM_REPLY_LEN = 512_000


class LLMRuleTranslator:
    """Translate natural-language rules with an LLM and validation loop.

    The LLM transport is the project's unified client
    (``layer2_engine.core.llm.LLMClient``).  ``llm_model`` names the model
    (e.g. ``"qwen3:8b"``) for the default client; an explicit ``client``
    (any ``RuleLLMClient``) wins.  Anything unavailable / empty / invalid
    falls back to templates, exactly like the pre-unification behaviour.
    """

    def __init__(
        self,
        client: RuleLLMClient | None = None,
        *,
        llm_model: str | None = None,
        run_engine_validation: bool = True,
        fallback: TemplateTranslator | None = None,
        max_repair_attempts: int = 1,
        max_tokens: int = 8192,
        strict_llm: bool = False,
        prompt_builder: RulePromptBuilder | None = None,
    ) -> None:
        self.client = client
        self.llm_model = llm_model
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
        client = self.client or LLMClient(model=self.llm_model)
        messages = self.prompt_builder.build_initial_messages(request)
        attempts = self.max_repair_attempts + 1
        last_validation = ValidationResult(valid=False, errors=["LLM 未返回可验证的 rules JSON"])

        for attempt in range(attempts):
            try:
                raw = client.complete(messages, max_tokens=self.max_tokens)
            except Exception as exc:  # noqa: BLE001 — 网络/推理异常统一进入兜底
                last_validation = ValidationResult(valid=False, errors=[str(exc)])
                warnings.append("LLM 生成失败，尝试模板兜底")
                break
            if not raw:
                warnings.append("本地 LLM 不可用（未返回内容），已使用模板兜底")
                break
            try:
                rules = self._parse_rules(raw)
            except Exception as exc:  # noqa: BLE001 — 解析/校验异常进入兜底
                last_validation = ValidationResult(valid=False, errors=[str(exc)])
                warnings.append("LLM 生成失败，尝试模板兜底")
                break

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
        merged = self._merged_failure_validation(last_validation, fallback_response, warnings)
        return TranslateResponse(rules_json={}, confidence=0.0, validation=merged)

    @staticmethod
    def _merged_failure_validation(
        last_validation: ValidationResult,
        fallback_response: TranslateResponse,
        warnings: list[str],
    ) -> ValidationResult:
        """Merge LLM and template failure explanations into one result.

        The returned ``rules_json`` is empty, so the validation describes
        exactly that artifact. Previously the last invalid LLM output was
        paired with template-failure validation — a mismatch between
        ``rules_json`` and ``validation``.
        """
        errors = LLMRuleTranslator._unique(
            list(last_validation.errors)
            + (list(fallback_response.validation.errors) if fallback_response.validation is not None else [])
        )
        warns = LLMRuleTranslator._unique(
            list(warnings)
            + list(last_validation.warnings)
            + (list(fallback_response.validation.warnings) if fallback_response.validation is not None else [])
        )
        return ValidationResult(valid=False, errors=errors, warnings=warns)

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        """Return ``items`` in order with duplicates removed."""
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

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
    "LLMRuleTranslator",
    "LLMTranslatorError",
    "RuleLLMClient",
]