"""Tests for the LLM incremental patch protocol (``rule_patch``).

Covers RFC-6902-style parse/apply semantics over Gavis rules dicts:
replace/add/remove ops, dotted paths with list indices, deep-copy
isolation, op-order application, and every documented failure mode.
"""

from __future__ import annotations

import copy

import pytest

from layer1_translator.rule_patch import MAX_PATCH_OPS, PatchError, apply_patch, parse_patch


def _base() -> dict:
    return {
        "meta": {"gameId": "toy", "version": "5.1"},
        "constants": {"board_size": 3, "win_length": 5, "list": ["a", "b", "c"]},
        "actions": [{"id": "a1", "legal": {"const": True}}, {"id": "a2", "legal": {"const": False}}],
    }


# ── parse_patch ────────────────────────────────────────────────────


class TestParsePatch:
    def test_envelope_valid(self) -> None:
        ops = parse_patch({"patch": [{"op": "replace", "path": "a.b", "value": 1}]})
        assert ops == [{"op": "replace", "path": "a.b", "value": 1}]

    def test_missing_patch_field(self) -> None:
        with pytest.raises(PatchError, match="patch"):
            parse_patch({"rules": {}})

    def test_patch_not_list(self) -> None:
        with pytest.raises(PatchError, match="数组"):
            parse_patch({"patch": {"op": "replace"}})

    def test_unknown_op(self) -> None:
        with pytest.raises(PatchError, match="op"):
            parse_patch({"patch": [{"op": "upsert", "path": "a", "value": 1}]})

    def test_empty_or_missing_path(self) -> None:
        with pytest.raises(PatchError, match="path"):
            parse_patch({"patch": [{"op": "replace", "path": "", "value": 1}]})
        with pytest.raises(PatchError, match="path"):
            parse_patch({"patch": [{"op": "remove"}]})

    def test_replace_without_value(self) -> None:
        with pytest.raises(PatchError, match="value"):
            parse_patch({"patch": [{"op": "replace", "path": "a"}]})

    def test_op_count_cap(self) -> None:
        ops = [{"op": "remove", "path": f"k{i}"} for i in range(MAX_PATCH_OPS + 1)]
        with pytest.raises(PatchError, match="上限"):
            parse_patch({"patch": ops})

    def test_non_dict_op(self) -> None:
        with pytest.raises(PatchError, match="对象"):
            parse_patch({"patch": ["nope"]})


# ── apply_patch ────────────────────────────────────────────────────


class TestApplyPatch:
    def test_replace_nested_key(self) -> None:
        rules = apply_patch(_base(), [{"op": "replace", "path": "constants.board_size", "value": 9}])
        assert rules["constants"]["board_size"] == 9
        assert _base()["constants"]["board_size"] == 3  # base untouched

    def test_add_creates_new_key(self) -> None:
        rules = apply_patch(_base(), [{"op": "add", "path": "constants.vanish_probability", "value": 0.2}])
        assert rules["constants"]["vanish_probability"] == 0.2

    def test_add_overwrites_existing(self) -> None:
        rules = apply_patch(_base(), [{"op": "add", "path": "constants.win_length", "value": 7}])
        assert rules["constants"]["win_length"] == 7

    def test_remove_deletes_key(self) -> None:
        rules = apply_patch(_base(), [{"op": "remove", "path": "constants.win_length"}])
        assert "win_length" not in rules["constants"]

    def test_list_index_replace(self) -> None:
        rules = apply_patch(_base(), [{"op": "replace", "path": "actions.0.legal", "value": {"const": False}}])
        assert rules["actions"][0]["legal"] == {"const": False}

    def test_list_index_remove(self) -> None:
        rules = apply_patch(_base(), [{"op": "remove", "path": "actions.1.id"}])
        assert "id" not in rules["actions"][1]

    def test_ordered_application(self) -> None:
        # later op addresses a key created by an earlier one
        rules = apply_patch(
            _base(),
            [
                {"op": "add", "path": "constants.extra", "value": 1},
                {"op": "replace", "path": "constants.extra", "value": 2},
            ],
        )
        assert rules["constants"]["extra"] == 2

    def test_deep_copy_isolation(self) -> None:
        base = _base()
        rules = apply_patch(base, [{"op": "replace", "path": "actions.0.legal", "value": {"const": False}}])
        assert base["actions"][0]["legal"] == {"const": True}
        assert rules is not base

    def test_replace_missing_target(self) -> None:
        with pytest.raises(PatchError, match="不存在"):
            apply_patch(_base(), [{"op": "replace", "path": "constants.nope", "value": 1}])

    def test_remove_missing_target(self) -> None:
        with pytest.raises(PatchError, match="不存在"):
            apply_patch(_base(), [{"op": "remove", "path": "constants.nope"}])

    def test_parent_not_dict(self) -> None:
        with pytest.raises(PatchError, match="str"):
            apply_patch(_base(), [{"op": "add", "path": "meta.gameId.x", "value": 1}])

    def test_list_index_out_of_range(self) -> None:
        with pytest.raises(PatchError, match="越界"):
            apply_patch(_base(), [{"op": "replace", "path": "actions.9.id", "value": "x"}])

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(PatchError, match="空 path"):
            apply_patch(_base(), [{"op": "replace", "path": "", "value": 1}])

    def test_null_value_allowed(self) -> None:
        rules = apply_patch(_base(), [{"op": "add", "path": "constants.nullable", "value": None}])
        assert "nullable" in rules["constants"] and rules["constants"]["nullable"] is None

    def test_failed_patch_leaves_input_untouched(self) -> None:
        base = _base()
        snapshot = copy.deepcopy(base)
        with pytest.raises(PatchError):
            apply_patch(
                base, [{"op": "replace", "path": "constants.ok", "value": 1}, {"op": "remove", "path": "nope.a"}]
            )
        assert base == snapshot
