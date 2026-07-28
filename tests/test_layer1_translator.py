"""Tests for Layer 1: Translator."""

from __future__ import annotations

import pytest

from layer1_translator import SchemaValidator, TranslateRequest, TranslateResponse


class TestSchemaValidator:
    @pytest.fixture
    def valid_rules(self) -> dict:
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

    def test_valid_rules_pass(self, valid_rules: dict):
        result = SchemaValidator.validate(valid_rules)
        assert result.valid
        assert len(result.errors) == 0

    def test_missing_top_level_keys(self):
        result = SchemaValidator.validate({})
        assert not result.valid
        assert any("缺少必需顶层字段" in e for e in result.errors)

    def test_missing_effect_ref(self, valid_rules: dict):
        rules = dict(valid_rules)
        rules["actions"] = [{"id": "bad_action", "phases": ["playing"]}]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("effectRef" in e for e in result.errors)

    def test_reference_nonexistent_effect(self, valid_rules: dict):
        rules = dict(valid_rules)
        rules["actions"] = [{
            "id": "bad",
            "phases": ["playing"],
            "effectRef": "nonexistent",
        }]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("nonexistent" in e for e in result.errors)

    def test_terminal_missing_condition(self, valid_rules: dict):
        rules = dict(valid_rules)
        rules["terminal"] = [{"id": "no_condition"}]
        result = SchemaValidator.validate(rules)
        assert not result.valid

    def test_utility_missing_player(self, valid_rules: dict):
        rules = dict(valid_rules)
        rules["utility"] = [{"value": {"const": 1}}]
        result = SchemaValidator.validate(rules)
        assert not result.valid

    def test_duplicate_action_id(self, valid_rules: dict):
        rules = dict(valid_rules)
        rules["actions"] = [
            {"id": "dup", "phases": ["playing"], "effectRef": "do_place"},
            {"id": "dup", "phases": ["playing"], "effectRef": "do_place"},
        ]
        result = SchemaValidator.validate(rules)
        assert not result.valid
        assert any("重复" in e for e in result.errors)

    def test_warning_for_missing_phase(self, valid_rules: dict):
        rules = dict(valid_rules)
        rules["actions"] = [{
            "id": "orphan",
            "phases": ["nonexistent_phase"],
            "effectRef": "do_place",
        }]
        result = SchemaValidator.validate(rules)
        # Should still be valid (warning, not error)
        assert result.valid
        assert len(result.warnings) > 0


class TestTranslateRequest:
    def test_create(self):
        req = TranslateRequest(rule_text="3×3 棋盘，三子连珠获胜")
        assert req.rule_text == "3×3 棋盘，三子连珠获胜"
        assert req.source_lang == "zh"
        assert req.game_name is None

    def test_create_with_game_name(self):
        req = TranslateRequest(rule_text="...", game_name="moon_chess")
        assert req.game_name == "moon_chess"
