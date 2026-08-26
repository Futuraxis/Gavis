"""Tests for Layer 1: variant rule translation (VariantTranslator).

Covers the L1 variant contract:

- deterministic parameter path (template + parsed ``change_text`` params)
- LLM path, including repair loop and deterministic fallback
- total-failure semantics (never returns unvalidated artifacts)
- ``GameEngine(allow_codegen=False)`` pure-interpreter switch (Layer 2)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer1_translator import VariantTranslator, translate_variant_rules
from layer2_engine.core.engine import GameEngine

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


@pytest.fixture
def gomoku_rules() -> dict:
    with open(RULES_DIR / "stochastic_gomoku.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── Deterministic parameter path ─────────────────────────────────


class TestDeterministicPath:
    @pytest.mark.parametrize(
        ("base_game_id", "change_text", "expected_constants"),
        [
            ("stochastic_gomoku", "五子棋 15x15 连五", {"board_size": 15, "win_length": 5}),
            ("stochastic_gomoku", "15x15", {"board_size": 15}),
            (
                "stochastic_gomoku",
                "随机五子棋 9x9 五连获胜 25% 消失",
                {"board_size": 9, "win_length": 5, "vanish_probability": 0.25},
            ),
            ("moon_chess", "月亮棋 4x4 每方4枚 四连获胜", {"board_size": 4, "win_length": 4, "max_pieces": 4}),
            ("texas_holdem", "德州扑克，盲注 1/2，筹码 80", {"small_blind": 1, "big_blind": 2, "stack_size": 80}),
            ("mahjong", "红中麻将 2人", {"variant": "hongzhong", "player_count": 2}),
            ("werewolf", "狼人杀 9人局，3狼，1预言家，1女巫，1猎人", {"player_count_hint": 9}),
        ],
    )
    def test_deterministic_variants_pass_engine_validation(
        self, base_game_id: str, change_text: str, expected_constants: dict
    ) -> None:
        response = translate_variant_rules(base_game_id, change_text, use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert response.validation.errors == []
        assert response.rules_json, "确定性路径必须产出非空 rules_json"
        constants = response.rules_json["constants"]
        for key, value in expected_constants.items():
            if key == "player_count_hint":
                continue
            assert constants[key] == value, f"constants.{key} 应为 {value!r}，实为 {constants.get(key)!r}"

    def test_gomoku_board_size_15(self) -> None:
        response = translate_variant_rules("stochastic_gomoku", "五子棋 15x15 连五", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15
        assert response.rules_json["constants"]["win_length"] == 5
        assert response.confidence == 0.95

    def test_equivalent_change_text_with_clear_base_id(self) -> None:
        # 等价形式：change_text 只含变化点，base_game_id 指定模板
        response = translate_variant_rules("stochastic_gomoku", "15x15", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15
        # 未指定的参数保持模板默认值
        assert response.rules_json["constants"]["win_length"] == 5

    def test_base_id_alias_resolution(self) -> None:
        response = translate_variant_rules("五子棋", "15x15", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["meta"]["gameId"] == "stochastic_gomoku"
        assert response.rules_json["constants"]["board_size"] == 15

    def test_moon_chess_syncs_grid_cols(self) -> None:
        response = translate_variant_rules("moon_chess", "4x4 月亮棋", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 4
        assert response.rules_json["derivedViews"]["cell"]["from"]["cols"] == {"var": "$constants.board_size"}

    def test_texas_raise_grid_capped_by_stack(self) -> None:
        response = translate_variant_rules("texas_holdem", "德州扑克，筹码 80", use_llm=False)

        constants = response.rules_json["constants"]
        assert constants["stack_size"] == 80
        assert max(constants["raise_grid"]) == 80

    def test_unparseable_change_fails_loudly(self) -> None:
        # 无法参数化的变更文本不得静默返回未改动的模板（用户会误以为改动生效）
        response = translate_variant_rules("texas_holdem", "移除加注上限改为每手翻三倍底池分彩", use_llm=False)

        assert response.validation is not None
        assert not response.validation.valid
        assert not response.rules_json
        assert any("未解析出任何可应用的模板参数" in error for error in response.validation.errors)

    def test_empty_change_fails_loudly(self) -> None:
        response = translate_variant_rules("texas_holdem", "", use_llm=False)

        assert response.validation is not None
        assert not response.validation.valid
        assert not response.rules_json

    def test_mahjong_player_shape_updated(self) -> None:
        response = translate_variant_rules("mahjong", "红中麻将 2人", use_llm=False)

        constants = response.rules_json["constants"]
        assert constants["variant"] == "hongzhong"
        assert constants["player_count"] == 2
        assert constants["player_ids"] == ["p0", "p1"]
        assert constants["deal_target"] == 27
        assert response.rules_json["players"] == ["p0", "p1"]

    def test_werewolf_matching_composition_keeps_template(self) -> None:
        response = translate_variant_rules("werewolf", "狼人杀 9人局，3狼，1预言家，1女巫，1猎人", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        role_pool = response.rules_json["constants"]["role_pool"]
        assert len(role_pool) == 9
        assert role_pool.count("wolf") == 3
        assert len(response.rules_json["constants"]["player_ids"]) == 9
        # 配比与模板一致 → 无“结构固定”警告
        assert not any("固定" in w for w in response.validation.warnings)

    def test_werewolf_unsupported_composition_warns_and_keeps_template(self) -> None:
        response = translate_variant_rules("werewolf", "狼人杀 6人局，2狼1预言家", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert len(response.rules_json["constants"]["role_pool"]) == 9
        assert any("固定" in w for w in response.validation.warnings)

    def test_schema_only_mode(self) -> None:
        response = translate_variant_rules(
            "moon_chess",
            "4x4 月亮棋，每方4枚，四连获胜",
            use_llm=False,
            run_engine_validation=False,
        )

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 4

    def test_facade_keyword_only_signature(self) -> None:
        # 契约签名：base_game_id/change_text 位置参数，其余 keyword-only
        response = translate_variant_rules(
            base_game_id="stochastic_gomoku",
            change_text="15x15",
            source_lang="zh",
            game_name="五子棋",
            use_llm=False,
            llm_client=None,
            llm_model_path=None,
            run_engine_validation=True,
        )

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15

    def test_variant_translator_class_direct(self) -> None:
        translator = VariantTranslator(run_engine_validation=True)
        response = translator.translate("stochastic_gomoku", "五子棋 15x15 连五", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15


# ── Total failure ────────────────────────────────────────────────


class TestTotalFailure:
    def test_unknown_base_without_llm(self) -> None:
        response = translate_variant_rules("unknown_game", "随便什么文本", use_llm=False)

        assert response.rules_json == {}
        assert response.confidence == 0.0
        assert response.validation is not None
        assert not response.validation.valid
        assert response.validation.errors
        assert any("基础游戏模板" in e or "无法" in e for e in response.validation.errors)

    def test_unknown_base_with_llm_merges_reasons(self) -> None:
        class ShouldNotBeCalledClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                raise AssertionError("基础模板不可识别时不应调用 LLM")

        response = translate_variant_rules(
            "unknown_game",
            "随便什么文本",
            use_llm=True,
            llm_client=ShouldNotBeCalledClient(),
        )

        assert response.rules_json == {}
        assert response.validation is not None
        assert not response.validation.valid
        # 合并 LLM 无基线 + 确定性无法识别的两类中文原因
        assert any("基础模板" in e for e in response.validation.errors)

    def test_never_returns_unvalidated_product(self) -> None:
        # 不变量：rules_json 非空 ⟺ validation.valid
        for base, text in [("stochastic_gomoku", "15x15"), ("texas_holdem", "筹码 60"), ("unknown_game", "x")]:
            response = translate_variant_rules(base, text, use_llm=False)
            assert response.validation is not None
            assert bool(response.rules_json) == response.validation.valid


# ── LLM path ─────────────────────────────────────────────────────


class TestLLMPath:
    def test_llm_success(self, gomoku_rules: dict) -> None:
        class FakeClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                rules = json.loads(json.dumps(gomoku_rules))
                rules["constants"]["board_size"] = 15
                return "```json\n" + json.dumps(rules, ensure_ascii=False) + "\n```"

        response = translate_variant_rules("stochastic_gomoku", "15x15", use_llm=True, llm_client=FakeClient())

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15
        assert any("LLM" in w for w in response.validation.warnings)

    def test_llm_repairs_invalid_output(self, gomoku_rules: dict) -> None:
        class RepairClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                self.calls += 1
                if self.calls == 1:
                    return "{}"  # 无效输出 → 触发 repair（错误回喂）
                rules = json.loads(json.dumps(gomoku_rules))
                rules["constants"]["board_size"] = 15
                return json.dumps(rules, ensure_ascii=False)

        client = RepairClient()
        response = translate_variant_rules("stochastic_gomoku", "15x15", use_llm=True, llm_client=client)

        assert client.calls == 2
        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15

    def test_llm_never_valid_falls_back_to_deterministic(self, gomoku_rules: dict) -> None:
        class NeverValidClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                self.calls += 1
                rules = json.loads(json.dumps(gomoku_rules))
                rules.pop("actions", None)  # 破坏 schema，永远无法通过校验
                return json.dumps(rules, ensure_ascii=False)

        client = NeverValidClient()
        response = translate_variant_rules("stochastic_gomoku", "五子棋 15x15 连五", use_llm=True, llm_client=client)

        assert client.calls == 2  # 初始 + 1 次 repair
        assert response.validation is not None
        assert response.validation.valid  # 回退确定性路径
        assert response.rules_json["constants"]["board_size"] == 15
        assert any("LLM 输出未通过校验" in w for w in response.validation.warnings)

    def test_llm_model_missing_falls_back(self, tmp_path: Path) -> None:
        response = translate_variant_rules(
            "stochastic_gomoku",
            "五子棋 15x15 连五",
            use_llm=True,
            llm_model_path=str(tmp_path / "missing-variant-llm"),
        )

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15
        assert any("LLM 不可用" in w for w in response.validation.warnings)

    def test_bad_llm_client_falls_back(self) -> None:
        class BrokenClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                raise RuntimeError("inference down")

        response = translate_variant_rules(
            "stochastic_gomoku",
            "五子棋 15x15 连五",
            use_llm=True,
            llm_client=BrokenClient(),
        )

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15
        assert any("LLM 生成失败" in w for w in response.validation.warnings)

    def test_use_llm_false_ignores_llm_args(self) -> None:
        class ShouldNotBeCalledClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                raise AssertionError("use_llm=False 时不应调用 LLM")

        response = translate_variant_rules(
            "stochastic_gomoku",
            "15x15",
            use_llm=False,
            llm_client=ShouldNotBeCalledClient(),
            llm_model_path="/nonexistent/model",
        )

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15


# ── Layer 2 engine switch (A1) ───────────────────────────────────


class TestEngineAllowCodegen:
    def test_allow_codegen_false_pure_interpreter(self, gomoku_rules: dict) -> None:
        engine = GameEngine(gomoku_rules, seed=42, allow_codegen=False)

        assert engine._compiled is None
        state = engine.create_initial_state()
        assert state["env"]["phase"] == "playing"
        assert state["env"]["turn"] == "p_black"

        actions = engine.get_legal_actions(state)
        assert actions

        new_state = engine.apply_action(state, actions[0])
        # 落子后进入 vanish_check（chance 节点）或继续游戏
        assert engine.get_node_type(new_state) in ("player", "chance", "terminal")

    def test_allow_codegen_default_still_compiles(self, gomoku_rules: dict) -> None:
        engine = GameEngine(gomoku_rules, seed=42)

        assert engine._compiled is not None

    def test_codegen_and_interpreter_parity_on_initial_state(self, gomoku_rules: dict) -> None:
        compiled = GameEngine(gomoku_rules, seed=42)
        interpreter = GameEngine(gomoku_rules, seed=42, allow_codegen=False)

        s1 = compiled.create_initial_state()
        s2 = interpreter.create_initial_state()
        acts1 = sorted(a.canonical_key for a in compiled.get_legal_actions(s1))
        acts2 = sorted(a.canonical_key for a in interpreter.get_legal_actions(s2))
        assert acts1 == acts2
        assert compiled.get_node_type(s1) == interpreter.get_node_type(s2)

    def test_allow_codegen_false_full_playout(self, gomoku_rules: dict) -> None:
        engine = GameEngine(gomoku_rules, seed=42, allow_codegen=False)
        state = engine.create_initial_state()
        moves = 0
        for _ in range(8):
            node_type = engine.get_node_type(state)
            if node_type == "player":
                actions = engine.get_legal_actions(state)
                if not actions:
                    break
                state = engine.apply_action(state, actions[0])
                moves += 1
            elif node_type == "chance":
                _, state = engine.sample_chance(state)
            else:
                break
        assert moves >= 1
        assert engine.get_node_type(state) in ("player", "chance", "terminal")

    def test_allow_codegen_false_texas_holdem(self) -> None:
        with open(RULES_DIR / "texas_holdem.json", "r", encoding="utf-8") as f:
            rules = json.load(f)
        engine = GameEngine(rules, seed=42, allow_codegen=False)

        assert engine._compiled is None
        state = engine.create_initial_state()
        node_type = engine.get_node_type(state)
        if node_type == "player":
            assert engine.get_legal_actions(state)
        elif node_type == "chance":
            assert engine.get_chance_outcomes(state)
