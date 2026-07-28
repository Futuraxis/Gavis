"""Schema validator for ``rules.json`` v4.1 format.

Checks structural correctness (required keys, types, references)
without executing the rules.  Execution-level validation (e.g. "are
there unreachable phases?") is delegated to a later simulation pass.
"""

from __future__ import annotations

from .protocol import ValidationResult

REQUIRED_TOP_LEVEL = {"constants", "actions", "effects", "phases", "terminal", "utility"}


class SchemaValidator:
    """Validate a ``rules.json`` dict against the v4.1 schema."""

    @staticmethod
    def validate(rules: dict) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # --- top-level keys ---
        missing = REQUIRED_TOP_LEVEL - set(rules.keys())
        if missing:
            errors.append(f"缺少必需顶层字段: {', '.join(sorted(missing))}")

        # --- actions ---
        actions = rules.get("actions", [])
        action_ids: set[str] = set()
        for i, act in enumerate(actions):
            aid = act.get("id")
            if not aid:
                errors.append(f"actions[{i}] 缺少 'id'")
            elif aid in action_ids:
                errors.append(f"actions 中存在重复 id: {aid}")
            else:
                action_ids.add(aid)
            if "effectRef" not in act:
                errors.append(f"action '{aid}' 缺少 'effectRef'")

        # --- effects ---
        effects = rules.get("effects", {})
        for act in actions:
            ref = act.get("effectRef")
            if ref and ref not in effects:
                errors.append(f"action '{act.get('id')}' 引用不存在的 effect '{ref}'")

        # --- chance ---
        chances = rules.get("chance", [])
        for ct in chances:
            if "id" not in ct:
                errors.append("chance 节点缺少 'id'")
            prob = ct.get("probability", {})
            if "explicit" not in prob:
                warnings.append(f"chance '{ct.get('id')}' 使用非 explicit 概率，跳过检查")

        # --- phases ---
        phases = {p["id"] for p in rules.get("phases", []) if "id" in p}
        for act in actions:
            for ph in act.get("phases", []):
                if ph not in phases:
                    warnings.append(f"action '{act.get('id')}' 引用了不存在的 phase '{ph}'")
        for ct in chances:
            for ph in ct.get("phases", []):
                if ph not in phases:
                    warnings.append(f"chance '{ct.get('id')}' 引用了不存在的 phase '{ph}'")

        # --- terminal ---
        for i, term in enumerate(rules.get("terminal", [])):
            if "condition" not in term:
                errors.append(f"terminal[{i}] 缺少 'condition'")
            if "id" not in term:
                errors.append(f"terminal[{i}] 缺少 'id'")

        # --- utility ---
        for i, util in enumerate(rules.get("utility", [])):
            if "player" not in util:
                errors.append(f"utility[{i}] 缺少 'player'")
            if "value" not in util:
                errors.append(f"utility[{i}] 缺少 'value'")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
