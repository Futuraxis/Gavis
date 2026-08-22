"""Prompt construction for Layer 1 LLM rule translation."""

from __future__ import annotations

import json
import re
from typing import Any

from .external_frontend_reader import ExternalFrontendRuleReader
from .protocol import TranslateRequest, ValidationResult

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_RULE_TEXT_LEN = 12000


def sanitize_rule_text(text: str) -> str:
    """Remove control characters and cap user rule text."""
    return CONTROL_CHARS_RE.sub("", text or "")[:MAX_RULE_TEXT_LEN].strip()


def system_prompt() -> str:
    """Return the stable system prompt shared by training and inference."""
    return (
        "你是 Gavis Layer 1 规则翻译器。输出必须是单个 JSON object，不要 Markdown、不要解释。"
        "目标方言为 Gavis v5.x：顶层至少包含 meta、players、groundState、derivedViews、constants、"
        "actions、effectors、terminal、utility；chance、queries、functions 可按需加入。"
        "actions 每项必须有 id、params、legal、effectRef；effectRef 必须指向 effectors 中的 key。"
        "自由文本动作参数使用 {\"type\":\"text\"}，不可枚举。"
        "表达式只使用规则 JSON 内已有数学原语和 alias，不要引用外部 Python 函数或 BUILTIN。"
        "如果规则太复杂，生成一个保守但可运行的近似规则，并在 meta.description 说明简化点。"
    )


class RulePromptBuilder:
    """Build initial and repair prompts for rule translation."""

    def __init__(self, external_reader: ExternalFrontendRuleReader | None = None) -> None:
        self.external_reader = external_reader or ExternalFrontendRuleReader()

    def build_initial_messages(self, request: TranslateRequest) -> list[dict[str, str]]:
        """Build the first-pass translation prompt."""
        context = {
            "source_lang": request.source_lang,
            "game_name": request.game_name,
            "rule_text": sanitize_rule_text(request.rule_text),
            "external_frontend": self._normalized_external_frontend(request.external_frontend),
        }
        return [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": (
                    "请把以下游戏规则翻译为 Gavis v5.x rules.json。\n"
                    f"{json.dumps(context, ensure_ascii=False)}"
                ),
            },
        ]

    def build_repair_messages(
        self,
        request: TranslateRequest,
        rules: dict[str, Any],
        validation: ValidationResult,
    ) -> list[dict[str, str]]:
        """Build a repair prompt from validator feedback."""
        repair_context = {
            "source_lang": request.source_lang,
            "game_name": request.game_name,
            "rule_text": sanitize_rule_text(request.rule_text),
            "candidate_rules_json": rules,
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
        }
        return [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": (
                    "上一次输出没有通过 Gavis 校验。请只返回修正后的完整 rules.json 对象。\n"
                    f"{json.dumps(repair_context, ensure_ascii=False)}"
                ),
            },
        ]

    def _normalized_external_frontend(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        rule_input = self.external_reader.read(payload)
        return {
            "game_id": rule_input.game_id,
            "family": rule_input.family,
            "rule_text": rule_input.rule_text,
            "parameters": rule_input.parameters,
            "source": rule_input.source,
            "warnings": rule_input.warnings,
        }
