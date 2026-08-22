#!/usr/bin/env python3
"""Evaluate the Layer 1 local rule translator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layer1_translator import LLMRuleTranslator, TranslateRequest
from layer1_translator.datasets import RuleExample, build_synthetic_examples, load_jsonl_examples


def evaluate_example(translator: LLMRuleTranslator, example: RuleExample) -> dict:
    """Evaluate one rule translation example."""
    response = translator.translate(
        TranslateRequest(
            rule_text=example.rule_text,
            source_lang=example.source_lang,
            game_name=example.game_name,
        )
    )
    validation = response.validation
    warnings = validation.warnings if validation is not None else []
    errors = validation.errors if validation is not None else []
    used_fallback = any("模板兜底" in warning for warning in warnings)
    return {
        "id": example.example_id,
        "family": example.family,
        "game_name": example.game_name,
        "json_parse": bool(response.rules_json),
        "schema_pass": validation.valid if validation is not None else False,
        "engine_pass": validation.valid if validation is not None and not errors else False,
        "repair_success": any("使用 LLM 生成" in warning for warning in warnings) and not used_fallback,
        "used_fallback": used_fallback,
        "confidence": response.confidence,
        "errors": errors,
        "warnings": warnings,
    }


def summarize(results: list[dict], heldout_family: str | None) -> dict:
    """Aggregate evaluation metrics."""
    total = len(results) or 1
    heldout = [row for row in results if heldout_family is not None and row["family"] == heldout_family]
    heldout_total = len(heldout) or 1
    return {
        "examples": len(results),
        "json_parse_rate": round(sum(row["json_parse"] for row in results) / total, 4),
        "schema_pass_rate": round(sum(row["schema_pass"] for row in results) / total, 4),
        "engine_pass_rate": round(sum(row["engine_pass"] for row in results) / total, 4),
        "repair_success_rate": round(sum(row["repair_success"] for row in results) / total, 4),
        "fallback_rate": round(sum(row["used_fallback"] for row in results) / total, 4),
        "heldout_family": heldout_family,
        "heldout_rule_family_generalization": round(sum(row["engine_pass"] for row in heldout) / heldout_total, 4)
        if heldout
        else None,
    }


def load_examples(path: Path | None) -> list[RuleExample]:
    """Load eval examples, defaulting to deterministic bootstrap examples."""
    if path is None:
        return build_synthetic_examples()
    return load_jsonl_examples(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Layer1 Rule Translator")
    parser.add_argument("--model-path", type=Path, default=Path("models/layer1-rule-llm"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--heldout-family", default=None)
    parser.add_argument("--strict-llm", action="store_true", help="关闭模板兜底，真实评估模型")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    translator = LLMRuleTranslator(model_path=args.model_path, strict_llm=args.strict_llm)
    results = [evaluate_example(translator, example) for example in load_examples(args.data)]
    report = {"summary": summarize(results, args.heldout_family), "results": results}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
