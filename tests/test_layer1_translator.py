"""Tests for Layer 1: Translator (protocol, schema validator, dataclasses)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer1_translator import (
    SchemaValidator,
    TranslateRequest,
    TranslateResponse,
    TranslatorProtocol,
    ValidationResult,
)

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def valid_rules() -> dict:
    return {
        "constants": {"board_size": 3},
        "actions": [
            {
                "id": "place_piece",
                "phases": ["playing"],
                "effectRef": "do_place",
                "params": {},
                "legal": {"const": True},
            }
        ],
        "effects": {
            "do_place": {"ops": [{"op": "set", "path": "x", "value": {"const": 1}}]},
        },
        "phases": [{"id": "playing", "actions": ["place_piece"]}],
        "terminal": [{"id": "game_over", "condition": {"const": False}}],
        "utility": [{"player": "p_black", "value": {"const": 0}}],
    }


@pytest.fixture
def valid_rules_with_chance(valid_rules: dict) -> dict:
    rules = dict(valid_rules)
    rules["chance"] = [
        {
            "id": "vanish",
            "phases": ["playing"],
            "probability": {
                "explicit": [
                    {"outcome": "vanish", "prob": 0.5},
                    {"outcome": "keep", "prob": 0.5},
                ]
            },
            "effectMap": {"vanish": "do_vanish", "keep": "do_keep"},
        }
    ]
    rules["effects"] = dict(rules["effects"])
    rules["effects"]["do_vanish"] = {"ops": []}
    rules["effects"]["do_keep"] = {"ops": []}
    return rules


@pytest.fixture
def gomoku_rules() -> dict:
    path = RULES_DIR / "stochastic_gomoku.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def moon_chess_rules() -> dict:
    path = RULES_DIR / "moon_chess.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── ValidationResult ──────────────────────────────────────────────


class TestValidationResult:
    def test_create_valid(self) -> None:
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_create_invalid_with_errors(self) -> None:
        result = ValidationResult(valid=False, errors=["错误1", "错误2"])
        assert result.valid is False
        assert len(result.errors) == 2
        assert result.warnings == []

    def test_create_with_warnings(self) -> None:
        result = ValidationResult(valid=True, warnings=["警告1"])
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_create_full(self) -> None:
        result = ValidationResult(
            valid=False,
            errors=["严重错误"],
            warnings=["轻微警告"],
        )
        assert not result.valid
        assert result.errors == ["严重错误"]
        assert result.warnings == ["轻微警告"]

    def test_default_factory_errors(self) -> None:
        result = ValidationResult(valid=True)
        result.errors.append("新错误")
        assert len(result.errors) == 1
        result2 = ValidationResult(valid=True)
        assert len(result2.errors) == 0

    def test_default_factory_warnings(self) -> None:
        result = ValidationResult(valid=True)
        result.warnings.append("新警告")
        assert len(result.warnings) == 1
        result2 = ValidationResult(valid=True)
        assert len(result2.warnings) == 0


# ── TranslateRequest ───────────────────────────────────────────────


class TestTranslateRequest:
    def test_create(self) -> None:
        req = TranslateRequest(rule_text="3×3 棋盘，三子连珠获胜")
        assert req.rule_text == "3×3 棋盘，三子连珠获胜"
        assert req.source_lang == "zh"
        assert req.game_name is None

    def test_create_with_game_name(self) -> None:
        req = TranslateRequest(rule_text="...", game_name="moon_chess")
        assert req.game_name == "moon_chess"

    def test_create_custom_lang(self) -> None:
        req = TranslateRequest(rule_text="...", source_lang="en")
        assert req.source_lang == "en"

    def test_create_empty_text(self) -> None:
        req = TranslateRequest(rule_text="")
        assert req.rule_text == ""

    def test_create_all_fields(self) -> None:
        req = TranslateRequest(
            rule_text="完整规则文本",
            source_lang="zh",
            game_name="test_game",
        )
        assert req.rule_text == "完整规则文本"
        assert req.source_lang == "zh"
        assert req.game_name == "test_game"


# ── TranslateResponse ─────────────────────────────────────────────


class TestTranslateResponse:
    def test_create_minimal(self) -> None:
        rules_json = {"actions": []}
        resp = TranslateResponse(rules_json=rules_json)
        assert resp.rules_json == rules_json
        assert resp.confidence == 0.0
        assert resp.validation is None

    def test_create_with_confidence(self) -> None:
        resp = TranslateResponse(rules_json={}, confidence=0.95)
        assert resp.confidence == 0.95

    def test_create_with_validation(self) -> None:
        validation = ValidationResult(valid=True)
        resp = TranslateResponse(rules_json={}, validation=validation)
        assert resp.validation is not None
        assert resp.validation.valid is True

    def test_create_full(self) -> None:
        rules_json = {"constants": {}}
        validation = ValidationResult(valid=True, warnings=["注意"])
        resp = TranslateResponse(
            rules_json=rules_json,
            confidence=1.0,
            validation=validation,
        )
        assert resp.rules_json == {"constants": {}}
        assert resp.confidence == 1.0
        assert resp.validation is not None
        assert resp.validation.valid is True
        assert resp.validation.warnings == ["注意"]

    def test_confidence_boundary_zero(self) -> None:
        resp = TranslateResponse(rules_json={}, confidence=0.0)
        assert resp.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        resp = TranslateResponse(rules_json={}, confidence=1.0)
        assert resp.confidence == 1.0

    def test_validation_none_default(self) -> None:
        resp = TranslateResponse(rules_json={})
        assert resp.validation is None


# ── TranslatorProtocol ────────────────────────────────────────────


class TestTranslatorProtocol:
    def test_isinstance_with_valid_implementation(self) -> None:
        class SimpleTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                return TranslateResponse(rules_json={})

        translator = SimpleTranslator()
        assert isinstance(translator, TranslatorProtocol)

    def test_isinstance_with_incomplete_implementation(self) -> None:
        class IncompleteTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                return TranslateResponse(rules_json={})

        translator = IncompleteTranslator()
        assert isinstance(translator, TranslatorProtocol)

    def test_not_isinstance_without_translate(self) -> None:
        class NoTranslate:
            pass

        obj = NoTranslate()
        assert not isinstance(obj, TranslatorProtocol)

    def test_protocol_with_request_response_types(self) -> None:
        class TypedTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                result = SchemaValidator.validate({"constants": {}})
                return TranslateResponse(
                    rules_json={"constants": {}},
                    confidence=0.5,
                    validation=result,
                )

        translator = TypedTranslator()
        req = TranslateRequest(rule_text="测试规则")
        resp = translator.translate(req)
        assert isinstance(resp, TranslateResponse)
        assert resp.rules_json == {"constants": {}}
        assert resp.validation is not None

    def test_multiple_translators_all_valid(self) -> None:
        class TranslatorA:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                return TranslateResponse(rules_json={"a": 1})

        class TranslatorB:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                return TranslateResponse(rules_json={"b": 2})

        assert isinstance(TranslatorA(), TranslatorProtocol)
        assert isinstance(TranslatorB(), TranslatorProtocol)


# ── SchemaValidator: basic valid / invalid ────────────────────────


class TestSchemaValidatorBasics:
    def test_valid_rules_pass(self, valid_rules: dict) -> None:
        result = SchemaValidator.validate(valid_rules)
        assert result.valid
        assert len(result.errors) == 0

    def test_empty_dict_fails(self) -> None:
        result = SchemaValidator.validate({})
        assert not result.valid
        assert any("缺少必需顶层字段" in e for e in result.errors)

    def test_missing_top_level_keys(self) -> None:
        result = SchemaValidator.validate({})
        assert not result.valid
        expected_keys = {"constants", "actions", "effects", "phases", "terminal", "utility"}
        for key in expected_keys:
            assert any(key in e for e in result.errors)

    def test_partial_top_level_keys(self) -> None:
        rules = {"constants": {}, "actions": []}
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("缺少必需顶层字段" in e for e in result.errors)

    def test_valid_rules_no_warnings(self, valid_rules: dict) -> None:
        result = SchemaValidator.validate(valid_rules)
        assert len(result.warnings) == 0


# ── SchemaValidator: actions ──────────────────────────────────────


class TestSchemaValidatorActions:
    def test_missing_effect_ref(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [{"id": "bad_action", "phases": ["playing"]}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("effectRef" in e for e in result.errors)

    def test_missing_action_id(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [{"phases": ["playing"], "effectRef": "do_place"}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("缺少 'id'" in e for e in result.errors)

    def test_duplicate_action_id(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "dup", "phases": ["playing"], "effectRef": "do_place"},
            {"id": "dup", "phases": ["playing"], "effectRef": "do_place"},
        ]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("重复" in e for e in result.errors)

    def test_reference_nonexistent_effect(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "bad", "phases": ["playing"], "effectRef": "nonexistent"},
        ]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("nonexistent" in e for e in result.errors)

    def test_empty_actions_list(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = []
        result = SchemaValidator.validate(rules)
        assert result.valid
        assert len(result.errors) == 0

    def test_multiple_valid_actions(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "a1", "phases": ["playing"], "effectRef": "do_place"},
            {"id": "a2", "phases": ["playing"], "effectRef": "do_place"},
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_warning_for_nonexistent_phase_ref(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {
                "id": "orphan",
                "phases": ["nonexistent_phase"],
                "effectRef": "do_place",
            }
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid
        assert len(result.warnings) > 0
        assert any("nonexistent_phase" in w for w in result.warnings)


# ── SchemaValidator: effects ──────────────────────────────────────


class TestSchemaValidatorEffects:
    def test_empty_effects_with_no_actions(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = []
        rules["effects"] = {}
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_effect_used_by_multiple_actions(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "a1", "phases": ["playing"], "effectRef": "do_place"},
            {"id": "a2", "phases": ["playing"], "effectRef": "do_place"},
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid


# ── SchemaValidator: phases ───────────────────────────────────────


class TestSchemaValidatorPhases:
    def test_action_with_multiple_valid_phases(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["phases"] = [
            {"id": "playing", "actions": ["place_piece"]},
            {"id": "ended", "actions": []},
        ]
        rules["actions"] = [
            {"id": "act1", "phases": ["playing", "ended"], "effectRef": "do_place"},
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_action_with_one_valid_one_invalid_phase(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {
                "id": "act1",
                "phases": ["playing", "ghost_phase"],
                "effectRef": "do_place",
            },
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid
        assert any("ghost_phase" in w for w in result.warnings)


# ── SchemaValidator: terminal ──────────────────────────────────────


class TestSchemaValidatorTerminal:
    def test_terminal_missing_condition(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["terminal"] = [{"id": "no_condition"}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("condition" in e for e in result.errors)

    def test_terminal_missing_id(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["terminal"] = [{"condition": {"const": False}}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("id" in e for e in result.errors)

    def test_terminal_both_missing(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["terminal"] = [{}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert len(result.errors) >= 2

    def test_multiple_terminal_conditions(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["terminal"] = [
            {"id": "t1", "condition": {"const": False}},
            {"id": "t2", "condition": {"const": False}},
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_empty_terminal_list(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["terminal"] = []
        result = SchemaValidator.validate(rules)
        assert result.valid


# ── SchemaValidator: utility ──────────────────────────────────────


class TestSchemaValidatorUtility:
    def test_utility_missing_player(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["utility"] = [{"value": {"const": 1}}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("player" in e for e in result.errors)

    def test_utility_missing_value(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["utility"] = [{"player": "p_black"}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("value" in e for e in result.errors)

    def test_utility_both_missing(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["utility"] = [{}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert len(result.errors) >= 2

    def test_multiple_utilities(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["utility"] = [
            {"player": "p_black", "value": {"const": 1}},
            {"player": "p_white", "value": {"const": -1}},
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_empty_utility_list(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["utility"] = []
        result = SchemaValidator.validate(rules)
        assert result.valid


# ── SchemaValidator: chance ────────────────────────────────────────


class TestSchemaValidatorChance:
    def test_valid_chance_node(self, valid_rules_with_chance: dict) -> None:
        result = SchemaValidator.validate(valid_rules_with_chance)
        assert result.valid

    def test_chance_missing_id(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["chance"] = [
            {
                "phases": ["playing"],
                "probability": {"explicit": []},
            }
        ]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("chance" in e and "id" in e for e in result.errors)

    def test_chance_non_explicit_probability_warning(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["chance"] = [
            {
                "id": "mystery",
                "phases": ["playing"],
                "probability": {"calculated": True},
            }
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid
        assert len(result.warnings) > 0
        assert any("explicit" in w for w in result.warnings)

    def test_chance_ref_nonexistent_phase_warning(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["chance"] = [
            {
                "id": "bad_chance",
                "phases": ["no_such_phase"],
                "probability": {"explicit": []},
            }
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid
        assert any("no_such_phase" in w for w in result.warnings)

    def test_empty_chance_list(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["chance"] = []
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_multiple_chance_nodes(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["chance"] = [
            {
                "id": "c1",
                "phases": ["playing"],
                "probability": {"explicit": []},
            },
            {
                "id": "c2",
                "phases": ["playing"],
                "probability": {"explicit": []},
            },
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid


# ── SchemaValidator: multiple errors / warnings ────────────────────


class TestSchemaValidatorMultipleIssues:
    def test_multiple_errors_simultaneously(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "a1", "phases": ["playing"]},
            {"id": "a1", "phases": ["playing"]},
        ]
        rules["terminal"] = [{}]
        rules["utility"] = [{}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert len(result.errors) >= 5

    def test_errors_and_warnings_mixed(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "a1", "phases": ["ghost_phase"]},
        ]
        rules["terminal"] = [{"id": "t1"}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert len(result.errors) >= 1
        assert len(result.warnings) >= 1

    def test_warnings_only_no_errors(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "a1", "phases": ["ghost_phase"], "effectRef": "do_place"},
        ]
        result = SchemaValidator.validate(rules)
        assert result.valid
        assert len(result.warnings) > 0
        assert len(result.errors) == 0


# ── SchemaValidator: real rule files (v5.0 format) ─────────────────


class TestSchemaValidatorRealRules:
    def test_gomoku_has_effectors_not_effects(self, gomoku_rules: dict) -> None:
        assert "effectors" in gomoku_rules
        assert "effects" not in gomoku_rules

    def test_moon_chess_has_effectors_not_effects(self, moon_chess_rules: dict) -> None:
        assert "effectors" in moon_chess_rules
        assert "effects" not in moon_chess_rules

    def test_gomoku_fails_v41_validation(self, gomoku_rules: dict) -> None:
        result = SchemaValidator.validate(gomoku_rules)
        assert not result.valid
        assert any("effects" in e for e in result.errors)

    def test_moon_chess_fails_v41_validation(self, moon_chess_rules: dict) -> None:
        result = SchemaValidator.validate(moon_chess_rules)
        assert not result.valid
        assert any("effects" in e for e in result.errors)

    def test_gomoku_structure_unchanged(self, gomoku_rules: dict) -> None:
        snapshot = json.dumps(gomoku_rules, sort_keys=True)
        SchemaValidator.validate(gomoku_rules)
        after = json.dumps(gomoku_rules, sort_keys=True)
        assert snapshot == after

    def test_moon_chess_structure_unchanged(self, moon_chess_rules: dict) -> None:
        snapshot = json.dumps(moon_chess_rules, sort_keys=True)
        SchemaValidator.validate(moon_chess_rules)
        after = json.dumps(moon_chess_rules, sort_keys=True)
        assert snapshot == after


# ── SchemaValidator: param / edge cases ────────────────────────────


class TestSchemaValidatorEdgeCases:
    @pytest.mark.parametrize(
        "field",
        ["constants", "actions", "effects", "phases", "terminal", "utility"],
    )
    def test_missing_each_top_level_field_individually(
        self, valid_rules: dict, field: str
    ) -> None:
        rules = {k: v for k, v in valid_rules.items() if k != field}
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any(field in e for e in result.errors)

    def test_rules_with_extra_unknown_keys(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["unknown_field"] = "should_be_ignored"
        rules["another_extra"] = {"nested": True}
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_none_values_in_rules_top_level(self) -> None:
        rules = {
            "constants": None,
            "actions": [],
            "effects": {},
            "phases": [],
            "terminal": [],
            "utility": [],
        }
        result = SchemaValidator.validate(rules)
        assert result.valid

    def test_none_actions_raises_type_error(self) -> None:
        rules = {
            "constants": {},
            "actions": None,
            "effects": {},
            "phases": [],
            "terminal": [],
            "utility": [],
        }
        with pytest.raises(TypeError):
            SchemaValidator.validate(rules)

    def test_deeply_nested_effects(self, valid_rules: dict) -> None:
        rules = dict(valid_rules)
        rules["effects"] = {
            "do_place": {
                "ops": [
                    {"op": "setIndex", "array": "board", "value": {"var": "$env.turn"}},
                    {"op": "branch", "if": {"const": True}, "then": [], "else": []},
                ]
            }
        }
        result = SchemaValidator.validate(rules)
        assert result.valid


# ── End-to-end workflow ───────────────────────────────────────────


class TestEndToEndWorkflow:
    def test_request_to_response_pipeline(self, valid_rules: dict) -> None:
        class MockTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                result = SchemaValidator.validate(valid_rules)
                return TranslateResponse(
                    rules_json=valid_rules,
                    confidence=0.9,
                    validation=result,
                )

        translator = MockTranslator()
        assert isinstance(translator, TranslatorProtocol)

        req = TranslateRequest(rule_text="测试规则")
        resp = translator.translate(req)

        assert isinstance(resp, TranslateResponse)
        assert resp.rules_json == valid_rules
        assert resp.confidence == 0.9
        assert resp.validation is not None
        assert resp.validation.valid is True

    def test_invalid_rules_pipeline(self, valid_rules: dict) -> None:
        class MockTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                bad_rules = dict(valid_rules)
                bad_rules["actions"] = []
                bad_rules["terminal"] = [{}]
                result = SchemaValidator.validate(bad_rules)
                return TranslateResponse(
                    rules_json=bad_rules,
                    confidence=0.3,
                    validation=result,
                )

        translator = MockTranslator()
        req = TranslateRequest(rule_text="测试规则")
        resp = translator.translate(req)

        assert not resp.validation.valid
        assert resp.confidence == 0.3
        assert len(resp.validation.errors) > 0

    def test_response_without_validation(self, valid_rules: dict) -> None:
        class MockTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                return TranslateResponse(
                    rules_json=valid_rules,
                    confidence=1.0,
                )

        translator = MockTranslator()
        req = TranslateRequest(rule_text="测试规则")
        resp = translator.translate(req)

        assert resp.validation is None
        assert resp.confidence == 1.0
        assert isinstance(resp, TranslateResponse)

    def test_protocol_compliance_check(self, valid_rules: dict) -> None:
        class CompliantTranslator:
            def translate(self, request: TranslateRequest) -> TranslateResponse:
                result = SchemaValidator.validate(valid_rules)
                return TranslateResponse(
                    rules_json=valid_rules,
                    confidence=0.85,
                    validation=result,
                )

        translator = CompliantTranslator()
        assert isinstance(translator, TranslatorProtocol)

        req = TranslateRequest(rule_text="游戏规则文本", source_lang="zh", game_name="test")
        assert req.game_name == "test"

        resp = translator.translate(req)
        assert resp.validation is not None
        assert resp.validation.valid
        assert resp.confidence == 0.85
        assert resp.rules_json == valid_rules