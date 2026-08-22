"""External frontend payload reader for Layer 1.

This module accepts DOM/config/storage data that has already been collected
by another component and normalizes rule hints for deterministic translators.
It does not fetch pages or control browsers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .rule_parser import RuleParser

_GAME_ID_KEYS = ("game_id", "gameId", "data-game-id", "id")
_FAMILY_KEYS = ("family", "gameFamily", "game_family", "data-game-family")
_PARAMETER_ALIASES = {
    "board_size": ("board_size", "boardSize", "data-board-size"),
    "win_length": ("win_length", "winLength", "connectN", "connect_n", "data-win-length"),
    "vanish_probability": ("vanish_probability", "vanishProbability", "vanishChance", "data-vanish-probability"),
}
_STORAGE_KEYS = ("localStorage", "sessionStorage")


@dataclass(frozen=True)
class ExternalRuleInput:
    """Normalized rule information extracted from an external frontend payload."""

    game_id: str | None
    family: str | None
    rule_text: str
    parameters: dict[str, Any]
    source: str
    warnings: list[str]


class ExternalFrontendRuleReader:
    """Read already-collected frontend data into Layer 1 rule hints."""

    def read(self, payload: dict[str, Any]) -> ExternalRuleInput:
        """Normalize a frontend payload without performing any collection."""
        warnings: list[str] = []
        if not isinstance(payload, dict):
            return ExternalRuleInput(
                game_id=None,
                family=None,
                rule_text="",
                parameters={},
                source="empty",
                warnings=["external_frontend 必须是 dict"],
            )

        config = self._dict_at(payload, "config", warnings)
        attributes = self._dict_at(payload, "attributes", warnings)
        storage = self._read_storage(payload, warnings)
        text = self._read_text(payload, warnings)
        text_params = RuleParser().parse_grid_family_parameters(text) if text else {}

        sources = (("config", config), ("attributes", attributes), ("storage", storage))
        game_id, game_id_source = self._first_string(sources, _GAME_ID_KEYS)
        family, family_source = self._first_string(sources, _FAMILY_KEYS)
        parameters: dict[str, Any] = {}
        parameter_sources: list[str] = []

        for parameter_name, aliases in _PARAMETER_ALIASES.items():
            parsed_value, source = self._first_parsed_value(sources, aliases, parameter_name, warnings)
            if source is None and parameter_name in text_params:
                parsed_value = text_params[parameter_name]
                source = "text"
            if source is not None:
                parameters[parameter_name] = parsed_value
                parameter_sources.append(source)

        source = self._source(game_id_source, family_source, parameter_sources, bool(text))
        return ExternalRuleInput(
            game_id=game_id,
            family=family,
            rule_text=text,
            parameters=parameters,
            source=source,
            warnings=warnings,
        )

    @staticmethod
    def _dict_at(payload: dict[str, Any], key: str, warnings: list[str]) -> dict[str, Any]:
        value = payload.get(key)
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        warnings.append(f"{key} 不是 dict，已忽略")
        return {}

    def _read_storage(self, payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        storage: dict[str, Any] = {}
        for storage_key in _STORAGE_KEYS:
            section = self._dict_at(payload, storage_key, warnings)
            self._merge_storage_section(storage, section, storage_key, warnings)
        return storage

    def _merge_storage_section(
        self,
        storage: dict[str, Any],
        section: dict[str, Any],
        storage_key: str,
        warnings: list[str],
    ) -> None:
        for key, value in section.items():
            if key not in storage:
                storage[key] = value
            if isinstance(value, str):
                parsed = self._parse_json_object(value, f"{storage_key}.{key}", warnings)
                if parsed is not None:
                    storage.update(
                        {
                            nested_key: nested_value
                            for nested_key, nested_value in parsed.items()
                            if nested_key not in storage
                        }
                    )
            elif isinstance(value, dict):
                storage.update(
                    {
                        nested_key: nested_value
                        for nested_key, nested_value in value.items()
                        if nested_key not in storage
                    }
                )

    @staticmethod
    def _parse_json_object(raw: str, label: str, warnings: list[str]) -> dict[str, Any] | None:
        stripped = raw.strip()
        if not stripped.startswith("{"):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            warnings.append(f"{label} 不是合法 JSON，已忽略")
            return None
        if not isinstance(parsed, dict):
            warnings.append(f"{label} JSON 不是 object，已忽略")
            return None
        return parsed

    @staticmethod
    def _read_text(payload: dict[str, Any], warnings: list[str]) -> str:
        value = payload.get("text", "")
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        warnings.append("text 不是 string，已忽略")
        return ""

    @staticmethod
    def _first_string(
        sources: tuple[tuple[str, dict[str, Any]], ...],
        keys: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        for source_name, values in sources:
            for key in keys:
                value = values.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip(), source_name
        return None, None

    def _first_parsed_value(
        self,
        sources: tuple[tuple[str, dict[str, Any]], ...],
        keys: tuple[str, ...],
        parameter_name: str,
        warnings: list[str],
    ) -> tuple[Any | None, str | None]:
        for source_name, values in sources:
            for key in keys:
                if key not in values:
                    continue
                parsed = self._parse_value(values[key], parameter_name, f"{source_name}.{key}", warnings)
                if parsed is not None:
                    return parsed, source_name
        return None, None

    @staticmethod
    def _parse_value(value: Any, parameter_name: str, label: str, warnings: list[str]) -> Any | None:
        if parameter_name in ("board_size", "win_length"):
            return ExternalFrontendRuleReader._parse_int(value, label, warnings)
        if parameter_name == "vanish_probability":
            return ExternalFrontendRuleReader._parse_float(value, label, warnings)
        return value

    @staticmethod
    def _parse_int(value: Any, label: str, warnings: list[str]) -> int | None:
        if isinstance(value, bool):
            warnings.append(f"{label} 不能解析为整数，已忽略")
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[+-]?\d+", stripped):
                return int(stripped)
        warnings.append(f"{label} 不能解析为整数，已忽略")
        return None

    @staticmethod
    def _parse_float(value: Any, label: str, warnings: list[str]) -> float | bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in ("true", "false"):
                return stripped == "true"
            if stripped.endswith("%"):
                number = stripped.removesuffix("%").strip()
                try:
                    return float(number) / 100.0
                except ValueError:
                    warnings.append(f"{label} 不能解析为概率，已忽略")
                    return None
            try:
                return float(stripped)
            except ValueError:
                warnings.append(f"{label} 不能解析为概率，已忽略")
                return None
        warnings.append(f"{label} 不能解析为概率，已忽略")
        return None

    @staticmethod
    def _source(
        game_id_source: str | None,
        family_source: str | None,
        parameter_sources: list[str],
        has_text: bool,
    ) -> str:
        for candidate in ("config", "attributes", "storage", "text"):
            if candidate in (game_id_source, family_source) or candidate in parameter_sources:
                return candidate
        return "text" if has_text else "empty"
