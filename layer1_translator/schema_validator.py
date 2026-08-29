"""Schema validator for Gavis rules JSON.

The validator performs structural checks only. It accepts both legacy
v4.1-style rules using effects and current v5.x-style rules using
effectors so Layer 1 can validate generated rules without coupling to
solver or frontend code.
"""

from __future__ import annotations

from typing import Any

from .protocol import ValidationResult

LEGACY_REQUIRED_TOP_LEVEL = {"constants", "actions", "effects", "phases", "terminal", "utility"}
V5_REQUIRED_TOP_LEVEL = {
    "meta",
    "players",
    "groundState",
    "derivedViews",
    "constants",
    "actions",
    "effectors",
    "terminal",
    "utility",
}


class SchemaValidator:
    """Validate a rules JSON dict against supported Gavis rule shapes."""

    @staticmethod
    def validate(rules: dict[str, Any]) -> ValidationResult:
        """Return structural validation errors and warnings for rules."""
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(rules, dict):
            return ValidationResult(valid=False, errors=["rules must be a dict"])

        dialect = SchemaValidator._detect_dialect(rules)
        required = V5_REQUIRED_TOP_LEVEL if dialect == "v5" else LEGACY_REQUIRED_TOP_LEVEL
        missing = required - set(rules.keys())
        if missing:
            errors.append(f"缺少必需顶层字段: {', '.join(sorted(missing))}")

        if dialect == "v5":
            SchemaValidator._validate_v5_metadata(rules, errors)
            # P2-25 修复：v5.2 声明式变体此前完全未被 schema 校验 —— 未知
            # 默认 variant / 坏 options 结构要等引擎运行时才暴露（连 smoke
            # 也会被同一次异常吞掉标签）。这里是结构层第一道闸。
            SchemaValidator._validate_variants(rules, errors)

        actions = SchemaValidator._expect_list(rules, "actions", errors)
        SchemaValidator._validate_actions(actions, errors)

        effect_key = "effectors" if dialect == "v5" else "effects"
        effectors = rules.get(effect_key, {})
        if effectors is None:
            errors.append(f"{effect_key} 必须是对象，不能为 None")
            effectors = {}
        elif not isinstance(effectors, dict):
            errors.append(f"{effect_key} 必须是对象")
            effectors = {}
        SchemaValidator._validate_action_effect_refs(actions, effectors or {}, effect_key, errors)

        phases = SchemaValidator._collect_phase_ids(rules.get("phases", []), warnings)
        SchemaValidator._validate_phase_refs(actions, phases, "action", warnings)

        chances = SchemaValidator._expect_list(rules, "chance", errors, default=[])
        SchemaValidator._validate_chance(chances, effectors or {}, phases, effect_key, errors, warnings)

        terminal = SchemaValidator._expect_list(rules, "terminal", errors)
        SchemaValidator._validate_terminal(terminal, errors)

        utility = SchemaValidator._expect_list(rules, "utility", errors)
        SchemaValidator._validate_utility(utility, errors)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def _detect_dialect(rules: dict[str, Any]) -> str:
        if "effectors" in rules or "groundState" in rules or "derivedViews" in rules:
            return "v5"
        return "v4"

    @staticmethod
    def _expect_list(
        rules: dict[str, Any],
        key: str,
        errors: list[str],
        *,
        default: list[Any] | None = None,
    ) -> list[Any]:
        value = rules.get(key, default if default is not None else [])
        if value is None:
            errors.append(f"{key} 必须是列表，不能为 None")
            return []
        if not isinstance(value, list):
            errors.append(f"{key} 必须是列表")
            return []
        return value

    @staticmethod
    def _validate_v5_metadata(rules: dict[str, Any], errors: list[str]) -> None:
        meta = rules.get("meta", {})
        if not isinstance(meta, dict):
            errors.append("meta 必须是对象")
        elif not meta.get("gameId") and not meta.get("name"):
            errors.append("meta 缺少 gameId/name")

        players = rules.get("players", [])
        if not isinstance(players, list) or not players:
            errors.append("players 必须是非空列表")

        for key in ("groundState", "derivedViews"):
            value = rules.get(key, {})
            if value is not None and not isinstance(value, dict):
                errors.append(f"{key} 必须是对象")

    @staticmethod
    def _validate_variants(rules: dict[str, Any], errors: list[str]) -> None:
        """v5.2 声明式变体节的结构校验（与引擎解析语义对齐）。

        P2-25 修复：变体节此前完全未被 schema 校验 —— 未知默认 variant /
        坏 options 结构要等引擎运行时才暴露（variant-aware smoke 也会被
        同一异常吞掉标签）。这里是结构层第一道闸。
        """
        variants = rules.get("variants")
        if variants is None:
            return
        if not isinstance(variants, dict):
            errors.append("variants 必须是对象")
            return
        options = variants.get("options", {})
        if options is None:
            errors.append("variants.options 不能为 None")
            options = {}
        if not isinstance(options, dict):
            errors.append("variants.options 必须是对象")
            options = {}
        if not isinstance(options, dict) or not options:
            errors.append("variants.options 不能为空（至少声明默认变体）")
            return
        for name, option in options.items():
            if not isinstance(name, str) or not name:
                errors.append("variants.options 的键必须是非空字符串")
                continue
            if not isinstance(option, dict):
                errors.append(f"variants.options.{name} 必须是对象")
                continue
            constants = option.get("constants")
            if constants is not None and not isinstance(constants, dict):
                errors.append(f"variants.options.{name}.constants 必须是对象")
        default = variants.get("variant")
        if default is not None:
            if not isinstance(default, str):
                errors.append("variants.variant 必须是字符串")
            elif default not in options:
                errors.append(f"variants.variant={default!r} 未在 variants.options 中声明")
        count = variants.get("player_count")
        if count is not None and not (isinstance(count, int) and count >= 1):
            errors.append("variants.player_count 必须是 ≥1 的整数")
        for key in ("trim_players", "trim_utility"):
            value = variants.get(key)
            if value is not None and not isinstance(value, bool):
                errors.append(f"variants.{key} 必须是布尔值")
        for key in ("player_ids", "deal_target"):
            value = variants.get(key)
            if value is not None and not isinstance(value, dict):
                errors.append(f"variants.{key} 必须是表达式对象")

    @staticmethod
    def _validate_actions(actions: list[Any], errors: list[str]) -> None:
        action_ids: set[str] = set()
        for i, act in enumerate(actions):
            if not isinstance(act, dict):
                errors.append(f"actions[{i}] 必须是对象")
                continue
            aid = act.get("id")
            if not aid:
                errors.append(f"actions[{i}] 缺少 'id'")
            elif aid in action_ids:
                errors.append(f"actions 中存在重复 id: {aid}")
            else:
                action_ids.add(str(aid))
            if "effectRef" not in act:
                errors.append(f"action '{aid}' 缺少 'effectRef'")

    @staticmethod
    def _validate_action_effect_refs(
        actions: list[Any],
        effectors: dict[str, Any],
        effect_key: str,
        errors: list[str],
    ) -> None:
        for act in actions:
            if not isinstance(act, dict):
                continue
            ref = act.get("effectRef")
            if ref and ref not in effectors:
                errors.append(f"action '{act.get('id')}' 引用不存在的 {effect_key} '{ref}'")

    @staticmethod
    def _collect_phase_ids(phases_value: Any, warnings: list[str]) -> set[str]:
        if not isinstance(phases_value, list):
            if phases_value not in (None, []):
                warnings.append("phases 不是列表，跳过 phase 引用检查")
            return set()
        return {str(p["id"]) for p in phases_value if isinstance(p, dict) and "id" in p}

    @staticmethod
    def _validate_phase_refs(items: list[Any], phases: set[str], label: str, warnings: list[str]) -> None:
        if not phases:
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            for phase in item.get("phases", []):
                if phase not in phases:
                    warnings.append(f"{label} '{item.get('id')}' 引用了不存在的 phase '{phase}'")

    @staticmethod
    def _validate_chance(
        chances: list[Any],
        effectors: dict[str, Any],
        phases: set[str],
        effect_key: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        for ct in chances:
            if not isinstance(ct, dict):
                errors.append("chance 节点必须是对象")
                continue
            if "id" not in ct:
                errors.append("chance 节点缺少 'id'")
            prob = ct.get("probability", {})
            if "explicit" not in prob:
                warnings.append(f"chance '{ct.get('id')}' 使用非 explicit 概率，跳过检查")
            for ref in ct.get("effectMap", {}).values():
                if ref not in effectors:
                    errors.append(f"chance '{ct.get('id')}' 引用不存在的 {effect_key} '{ref}'")
        SchemaValidator._validate_phase_refs(chances, phases, "chance", warnings)

    @staticmethod
    def _validate_terminal(terminal: list[Any], errors: list[str]) -> None:
        for i, term in enumerate(terminal):
            if not isinstance(term, dict):
                errors.append(f"terminal[{i}] 必须是对象")
                continue
            if "condition" not in term:
                errors.append(f"terminal[{i}] 缺少 'condition'")
            if "id" not in term:
                errors.append(f"terminal[{i}] 缺少 'id'")

    @staticmethod
    def _validate_utility(utility: list[Any], errors: list[str]) -> None:
        for i, util in enumerate(utility):
            if not isinstance(util, dict):
                errors.append(f"utility[{i}] 必须是对象")
                continue
            if "player" not in util:
                errors.append(f"utility[{i}] 缺少 'player'")
            if "value" not in util:
                errors.append(f"utility[{i}] 缺少 'value'")
