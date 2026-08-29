"""Deterministic natural-language translator for known Gavis games.

The template translator keeps Layer 1 deterministic: it maps natural
language hints, known ``game_name`` values, or externally collected
frontend rule payloads to validated ``rules.json``. Future LLM translators
can share the same ``TranslatorProtocol`` without changing callers.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .engine_validator import EngineValidator
from .external_frontend_reader import ExternalFrontendRuleReader
from .protocol import TranslateRequest, TranslateResponse, ValidationResult
from .rule_family_builder import RuleFamilyBuilder
from .rule_parser import TEMPLATE_FILES, ParsedRuleRequest, RuleParser
from .schema_validator import SchemaValidator

_RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


class TemplateTranslator:
    """Translate natural-language rule requests with deterministic templates."""

    def __init__(
        self,
        *,
        run_engine_validation: bool = True,
        rules_dir: Path | None = None,
        parser: RuleParser | None = None,
        family_builder: RuleFamilyBuilder | None = None,
        external_reader: ExternalFrontendRuleReader | None = None,
    ) -> None:
        self.run_engine_validation = run_engine_validation
        self.rules_dir = rules_dir or _RULES_DIR
        self.engine_validator = EngineValidator()
        self.parser = parser or RuleParser()
        self.family_builder = family_builder or RuleFamilyBuilder(rules_dir=self.rules_dir, parser=self.parser)
        self.external_reader = external_reader or ExternalFrontendRuleReader()

    def translate(self, request: TranslateRequest) -> TranslateResponse:
        """Return a template-derived rules JSON for ``request``."""
        if request.external_frontend:
            return self._translate_external_frontend(request)

        parsed = self.parser.parse(rule_text=request.rule_text, game_name=request.game_name)
        if parsed is None:
            return self._translate_rule_family(request)

        rules = self._load_template(parsed.game_id)
        warnings = self._apply_parameters(rules, parsed)

        validation = self._validate(rules)
        validation.warnings.extend(warnings)
        confidence = 0.95 if validation.valid else 0.4
        return TranslateResponse(rules_json=rules, confidence=confidence, validation=validation)

    def _translate_external_frontend(self, request: TranslateRequest) -> TranslateResponse:
        rule_input = self.external_reader.read(request.external_frontend or {})
        spec = self.family_builder.parse_external(rule_input)
        if spec is not None:
            rules = self.family_builder.build(spec)
            validation = self._validate(rules)
            validation.warnings.extend([f"使用外部前端规则族生成: {spec.family_id}", *spec.warnings])
            confidence = 0.8 if validation.valid else 0.35
            return TranslateResponse(rules_json=rules, confidence=confidence, validation=validation)

        if rule_input.rule_text:
            fallback_request = TranslateRequest(
                rule_text=rule_input.rule_text,
                source_lang=request.source_lang,
                game_name=request.game_name or rule_input.game_id,
            )
            response = self.translate(fallback_request)
            if response.validation is not None:
                response.validation.warnings.extend(rule_input.warnings)
            return response

        validation = ValidationResult(
            valid=False,
            errors=["无法从外部前端读取规则信息"],
            warnings=rule_input.warnings,
        )
        return TranslateResponse(rules_json={}, confidence=0.0, validation=validation)

    def _translate_rule_family(self, request: TranslateRequest) -> TranslateResponse:
        spec = self.family_builder.parse(rule_text=request.rule_text, game_name=request.game_name)
        if spec is None:
            validation = ValidationResult(valid=False, errors=["无法识别游戏类型"])
            return TranslateResponse(rules_json={}, confidence=0.0, validation=validation)

        rules = self.family_builder.build(spec)
        validation = self._validate(rules)
        validation.warnings.extend([f"使用规则族生成: {spec.family_id}", *spec.warnings])
        confidence = 0.8 if validation.valid else 0.35
        return TranslateResponse(rules_json=rules, confidence=confidence, validation=validation)

    def _load_template(self, game_id: str) -> dict[str, Any]:
        file_name = TEMPLATE_FILES[game_id]
        with open(self.rules_dir / file_name, "r", encoding="utf-8") as f:
            return copy.deepcopy(json.load(f))

    def _apply_parameters(self, rules: dict[str, Any], parsed: ParsedRuleRequest) -> list[str]:
        if parsed.game_id == "stochastic_gomoku":
            self._apply_constant_params(rules, parsed.parameters)
            return []
        if parsed.game_id == "moon_chess":
            self._apply_constant_params(rules, parsed.parameters)
            self._sync_grid_cols(rules)
            return []
        if parsed.game_id == "mahjong":
            return self._apply_mahjong_params(rules, parsed.parameters)
        if parsed.game_id == "werewolf":
            return self._apply_werewolf_params(rules, parsed.parameters)
        if parsed.game_id == "texas_holdem":
            self._apply_texas_holdem_params(rules, parsed.parameters)
            return []
        if parsed.game_id == "uno":
            return self._apply_uno_params(rules, parsed.parameters)
        return []

    @staticmethod
    def _apply_constant_params(rules: dict[str, Any], params: dict[str, Any]) -> None:
        constants = rules.setdefault("constants", {})
        constants.update(params)

    @staticmethod
    def _sync_grid_cols(rules: dict[str, Any]) -> None:
        board_size = rules.get("constants", {}).get("board_size")
        cell_view = rules.get("derivedViews", {}).get("cell", {})
        source = cell_view.get("from", {})
        if isinstance(board_size, int) and isinstance(source, dict):
            source["cols"] = {"var": "$constants.board_size"}

    @staticmethod
    def _apply_mahjong_params(rules: dict[str, Any], params: dict[str, Any]) -> list[str]:
        """Apply mahjong ``variant`` / ``player_count`` to the declarative variants spec.

        与 VariantTranslator._apply_mahjong_params 同实现（T1 修复：旧版
        写 constants 无运行时效果 —— 引擎 _resolve_variants 覆写
        constants.variant/player_count，"红中麻将 2人" 实际仍跑 guangdong/4人）。
        只改 ``rules["variants"]`` 规约默认值，人数裁剪由规约的
        player_ids map / trim_players / trim_utility 表达。
        """
        spec = rules.setdefault("variants", {})
        warnings: list[str] = []
        options = spec.get("options", {}) or {}
        variant = params.get("variant")
        if variant is not None:
            if variant in options:
                spec["variant"] = variant
            else:
                warnings.append(
                    f"麻将变体 {variant!r} 未声明（可选 {sorted(options)}），已保留默认 {spec.get('variant')!r}"
                )
        player_count = params.get("player_count")
        if player_count is not None:
            if player_count in (2, 4):
                spec["player_count"] = player_count
            else:
                warnings.append(f"麻将模板仅支持 2 或 4 人，已保留默认 player_count={spec.get('player_count')}")
        return warnings

    @staticmethod
    def _apply_werewolf_params(rules: dict[str, Any], params: dict[str, Any]) -> list[str]:
        if not params:
            return []
        constants = rules.setdefault("constants", {})
        current_pool = list(constants.get("role_pool", []))
        current_players = list(constants.get("player_ids", []))
        expected_pool = TemplateTranslator._werewolf_role_pool(params, current_pool, current_players)
        if expected_pool != current_pool:
            return [
                "狼人杀模板的阶段和发牌结构目前固定为 "
                f"{len(current_players)} 人 / {TemplateTranslator._describe_role_pool(current_pool)}，"
                "已保留默认模板",
            ]
        constants["player_ids"] = [f"p{i}" for i in range(len(expected_pool))]
        rules["players"] = constants["player_ids"]
        rules["utility"] = [u for u in rules.get("utility", []) if u.get("player") in constants["player_ids"]]
        return []

    @staticmethod
    def _werewolf_role_pool(
        params: dict[str, Any],
        current_pool: list[str],
        current_players: list[str],
    ) -> list[str]:
        players = int(params.get("players", len(current_players) or len(current_pool)))
        wolves = int(params.get("wolves", current_pool.count("wolf")))
        seers = int(params.get("seers", current_pool.count("seer")))
        with_witch = bool(params.get("with_witch", "witch" in current_pool))
        with_hunter = bool(params.get("with_hunter", "hunter" in current_pool))
        with_guard = bool(params.get("with_guard", "guard" in current_pool))
        extras = ["seer"] * seers
        extras.extend(
            role for role, enabled in (("witch", with_witch), ("hunter", with_hunter), ("guard", with_guard)) if enabled
        )
        villagers = int(params.get("villagers", players - wolves - len(extras)))
        return ["wolf"] * wolves + ["villager"] * villagers + extras

    @staticmethod
    def _describe_role_pool(role_pool: list[str]) -> str:
        labels = [
            ("wolf", "狼"),
            ("villager", "村民"),
            ("seer", "预言家"),
            ("witch", "女巫"),
            ("hunter", "猎人"),
            ("guard", "守卫"),
        ]
        return " / ".join(f"{role_pool.count(role)}{label}" for role, label in labels if role_pool.count(role))

    @staticmethod
    def _apply_texas_holdem_params(rules: dict[str, Any], params: dict[str, Any]) -> None:
        constants = rules.setdefault("constants", {})
        constants.update(params)
        stack_size = constants.get("stack_size")
        if isinstance(stack_size, int) and isinstance(constants.get("raise_grid"), list):
            grid = [value for value in constants["raise_grid"] if isinstance(value, int) and value <= stack_size]
            if stack_size not in grid:
                grid.append(stack_size)
            constants["raise_grid"] = sorted(set(grid))

    @staticmethod
    def _apply_uno_params(rules: dict[str, Any], params: dict[str, Any]) -> list[str]:
        """Apply UNO ``variant`` / ``player_count`` to the declarative variants spec.

        与 VariantTranslator._apply_uno_params 同实现（P1-5：此前无 uno 分支，
        解析出的变体/人数被静默丢弃）。UNO 是声明式变体游戏，引擎构造期纯数据
        解析，故只改 ``rules["variants"]`` 规约默认值，不做 constants 注入。
        """
        spec = rules.setdefault("variants", {})
        warnings: list[str] = []
        options = spec.get("options", {}) or {}
        variant = params.get("variant")
        if variant is not None:
            if variant in options:
                spec["variant"] = variant
            else:
                warnings.append(
                    f"UNO 变体 {variant!r} 未声明（可选 {sorted(options)}），已保留默认 {spec.get('variant')!r}"
                )
        player_count = params.get("player_count")
        if player_count is not None:
            if isinstance(player_count, int) and 2 <= player_count <= 10:
                spec["player_count"] = player_count
            else:
                warnings.append(f"UNO 仅支持 2-10 人，已保留默认 player_count={spec.get('player_count')}")
        return warnings

    def _validate(self, rules: dict[str, Any]) -> ValidationResult:
        if self.run_engine_validation:
            return self.engine_validator.validate(rules)
        return SchemaValidator.validate(rules)
