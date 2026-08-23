"""Datasets and bootstrap examples for Layer 1 rule-LLM training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .local_client import format_messages
from .prompt_builder import RulePromptBuilder
from .protocol import TranslateRequest
from .template_translator import TemplateTranslator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleExample:
    """One supervised rule-translation example."""

    rule_text: str
    rules_json: dict[str, Any]
    game_name: str | None = None
    source_lang: str = "zh"
    family: str | None = None
    example_id: str | None = None


class RuleJsonDataset:
    """Causal-LM dataset with labels masked over the prompt portion."""

    def __init__(self, examples: list[RuleExample], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_builder = RulePromptBuilder()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        messages = self.prompt_builder.build_initial_messages(
            TranslateRequest(
                rule_text=example.rule_text,
                source_lang=example.source_lang,
                game_name=example.game_name,
            )
        )
        prompt = format_messages(messages, self.tokenizer)
        target = json.dumps(example.rules_json, ensure_ascii=False, separators=(",", ":"))
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(
            prompt + target + (self.tokenizer.eos_token or ""),
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))
        if prompt_len >= len(full_ids):
            logger.warning(
                "RuleJsonDataset 样本 %d 的 prompt 已达 max_length=%d，labels 将全为 -100（无监督信号）",
                index,
                self.max_length,
            )
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def build_synthetic_examples() -> list[RuleExample]:
    """Generate bootstrap examples from deterministic Layer 1 templates."""
    prompts = [
        ("月亮棋，3x3 棋盘，每方三枚棋子，三连成线获胜", "moon_chess", "moon_chess"),
        ("4x4 月亮棋，每方4枚棋子，四连获胜", "moon_chess", "moon_chess"),
        ("随机五子棋，9x9 棋盘，五子连珠获胜，50% 消失", None, "board_alignment"),
        ("connect4 是一个 7x7 棋盘，四连成线获胜", None, "board_alignment"),
        ("广东麻将，2人局", None, "mahjong"),
        ("血战麻将，4人局", None, "mahjong"),
        ("德州扑克，盲注 1/2，筹码 100", None, "texas_holdem"),
        ("狼人杀 9人局，3狼1预言家1女巫1猎人", None, "werewolf"),
    ]
    translator = TemplateTranslator(run_engine_validation=False)
    examples: list[RuleExample] = []
    for index, (rule_text, game_name, family) in enumerate(prompts):
        response = translator.translate(TranslateRequest(rule_text=rule_text, game_name=game_name))
        if response.validation is not None and response.validation.valid:
            examples.append(
                RuleExample(
                    rule_text=rule_text,
                    game_name=game_name,
                    family=family,
                    example_id=f"synthetic_{index:04d}",
                    rules_json=response.rules_json,
                )
            )
    return examples


def load_jsonl_examples(path: Path) -> list[RuleExample]:
    """Load user-provided JSONL training/evaluation examples."""
    examples: list[RuleExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 需要 JSON object 行，得到 {type(row).__name__}")
            rules_json = row.get("rules_json")
            if not isinstance(rules_json, dict):
                raise ValueError(f"{path}:{line_number} 缺少 dict 类型 rules_json")
            examples.append(
                RuleExample(
                    rule_text=str(row.get("rule_text", "")),
                    game_name=row.get("game_name"),
                    source_lang=str(row.get("source_lang", "zh")),
                    family=row.get("family"),
                    example_id=row.get("id") or row.get("example_id"),
                    rules_json=rules_json,
                )
            )
    return examples


def dump_examples_json(path: Path, examples: list[RuleExample]) -> None:
    """Write examples as JSON for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([example.__dict__ for example in examples], f, ensure_ascii=False, indent=2)
