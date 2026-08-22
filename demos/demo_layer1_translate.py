#!/usr/bin/env python3
"""CLI demo for Layer 1 natural-language rule translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layer1_translator import translate_rules_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer1 Demo: 自然语言游戏规则 -> Gavis rules.json")
    parser.add_argument("rule_text", nargs="?", default="9x9 棋盘，双方轮流落子，五连获胜，落子后 20% 概率消失")
    parser.add_argument("--game-name", default=None)
    parser.add_argument("--model-path", type=Path, default=Path("models/layer1-rule-llm"))
    parser.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--schema-only", action="store_true", help="只做 schema 校验，不跑 engine smoke")
    args = parser.parse_args()

    response = translate_rules_json(
        args.rule_text,
        game_name=args.game_name,
        use_llm=args.use_llm,
        llm_model_path=args.model_path,
        run_engine_validation=not args.schema_only,
    )
    print(
        json.dumps(
            {
                "confidence": response.confidence,
                "validation": response.validation.__dict__ if response.validation is not None else None,
                "rules_json": response.rules_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
