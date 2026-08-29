"""Tests for ``layer2_engine.core.smoke_validator`` — engine smoke-validation.

Covers the v5.5 variant-aware extension: ``smoke_validate(..., variants="all")``
boots every declared ``variants.options`` (plus the default selection) and
labels per-variant errors, so a variant whose ``constants`` patch breaks
construction or the initial transition is caught at rule-production time.
"""

from __future__ import annotations

from layer2_engine.core.smoke_validator import SmokeValidation, smoke_validate


def _rules(**variants_patch) -> dict:
    """Minimal bootable v5 rules with a declarative variants section.

    ``variants_patch`` merges into ``variants["options"]`` so tests can
    inject broken options (e.g. a non-dict ``constants`` that makes
    ``dict.update`` raise at engine construction).
    """
    options = {
        "classic": {},
        "quick": {"constants": {"board_size": 5}},
    }
    options.update(variants_patch)
    return {
        "constants": {"board_size": 3},
        "groundState": {
            "env": {
                "type": "env",
                "fields": {"phase": {"type": "string", "initial": "playing"}},
            }
        },
        "derivedViews": {},
        "players": [],
        "actions": [],
        "phases": [],
        "chance": [],
        "terminal": [],
        "utility": [],
        "visibility": {"default": "public"},
        "variants": {
            "variant": "classic",
            "player_count": 2,
            "options": options,
        },
    }


class TestSmokeValidate:
    def test_base_only_without_variants_arg(self) -> None:
        result = smoke_validate(_rules())
        assert result.errors == []
        assert result.warnings == []

    def test_all_probes_every_variant(self) -> None:
        result = smoke_validate(_rules(), variants="all")
        assert result.errors == []
        assert result.warnings == []

    def test_explicit_option_list(self) -> None:
        result = smoke_validate(_rules(), variants=["quick"])
        assert result.errors == []

    def test_no_variants_section_ignores_all(self) -> None:
        rules = _rules()
        rules.pop("variants")
        result = smoke_validate(rules, variants="all")
        assert result.errors == []
        assert result.warnings == []

    def test_broken_option_labels_construction_error(self) -> None:
        result = smoke_validate(_rules(broken={"constants": "not-a-dict"}), variants="all")
        # classic/quick 仍过；broken 的构造失败必须带 [variant=broken] 标签。
        assert any(e.startswith("[variant=broken]") and "构造失败" in e for e in result.errors)
        assert not any(e.startswith("[variant=classic]") for e in result.errors)
        assert not any(e.startswith("[variant=quick]") for e in result.errors)

    def test_unknown_named_option_fails_with_label(self) -> None:
        result = smoke_validate(_rules(), variants=["nope"])
        assert any(e.startswith("[variant=nope]") for e in result.errors)

    def test_broken_default_variant_fails_at_base_probe(self) -> None:
        rules = _rules()
        rules["variants"]["variant"] = "ghost"
        result = smoke_validate(rules, variants="all")
        # 引擎在每次选择（含 base）都会尝试解析默认变体 → 全部标失败。
        assert result.errors
        assert all("构造失败" in e for e in result.errors)


class TestSmokeValidationDataclass:
    def test_default_empty(self) -> None:
        result = SmokeValidation()
        assert result.errors == []
        assert result.warnings == []
