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
        """T1 修复后的断言面：翻译产物写 ``variants`` 规约（声明式），
        而非 constants —— 引擎 ``_resolve_variants`` 构造期才把
        variant/player_count 合并进运行时 constants。旧断言
        ``constants.variant == "hongzhong"`` 检查的是一个从未生效的表面
        （运行时仍按 guangdong/4 人装配）。"""
        response = translate_variant_rules("mahjong", "红中麻将 2人", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        spec = response.rules_json["variants"]
        assert spec["variant"] == "hongzhong"
        assert spec["player_count"] == 2
        # 运行时验证：GameEngine 按 variants 规约装配（真正的判据）。
        engine = GameEngine(response.rules_json)
        assert engine._constants["variant"] == "hongzhong"  # noqa: SLF001
        assert engine._constants["player_count"] == 2  # noqa: SLF001
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":  # deal chance 发牌
            _, state = engine.sample_chance(state)
        assert len(state["_arrays"]["hand_p0"]) == 14  # 2 人 13+1 发牌
        assert len(state["_arrays"]["hand_p1"]) == 13

    @pytest.mark.parametrize(
        ("change_text", "expected_variant"),
        [
            ("四川麻将 4人", "sichuan"),
            ("血战到底 2人", "sichuan"),
            ("血流成河 4人", "blood"),
            ("长沙麻将 2人", "changsha"),
            ("台湾麻将 4人", "taiwan"),
            ("红中麻将 2人", "hongzhong"),
            ("广东麻将 4人", "guangdong"),
            ("鸡胡 2人", "guangdong"),
        ],
    )
    def test_mahjong_variant_keywords(self, change_text: str, expected_variant: str) -> None:
        """T2 修复：四川/血战到底/血流成河/长沙/台湾 关键词全部解析到
        声明的变体（旧版只认 红中/血战/广东，且 血战→blood 映射错误）。"""
        response = translate_variant_rules("mahjong", change_text, use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        engine = GameEngine(response.rules_json)
        assert engine._constants["variant"] == expected_variant  # noqa: SLF001

    def test_mahjong_unknown_variant_warns_and_keeps_default(self) -> None:
        response = translate_variant_rules("mahjong", "麻将 2人", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        spec = response.rules_json["variants"]
        assert spec["variant"] == "guangdong"  # 未识别变体 → 模板默认
        assert spec["player_count"] == 2

    def test_mahjong_bad_player_count_warns(self) -> None:
        response = translate_variant_rules("mahjong", "红中麻将 3人", use_llm=False)

        assert response.validation is not None
        assert response.validation.valid
        spec = response.rules_json["variants"]
        assert spec["player_count"] == 4  # 非法人数 → 模板默认 4
        assert any("2 或 4 人" in w for w in response.validation.warnings)

    def test_oversized_template_uses_patch_protocol_not_rewrite(self) -> None:
        """T3 护栏 → 补丁协议（v5.5）：mahjong 模板 ≈87k 字符不能全量改写
        （回复上限装不下），但**增量补丁**输出极小 —— 改走补丁协议调用 LLM；
        LLM 传输不可用时仍确定性兜底（patch_mode=False 保留旧护栏行为）。"""

        class ExplodingClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, max_tokens=None):  # noqa: ANN001, ANN202
                self.calls += 1
                raise RuntimeError("inference down")

        client = ExplodingClient()
        response = translate_variant_rules("mahjong", "红中麻将 2人", use_llm=True, llm_client=client, patch_mode=False)

        assert client.calls == 0
        assert response.validation is not None
        assert response.validation.valid, "护栏短路后确定性路径应产出有效规则"
        assert any("过大" in w for w in response.validation.warnings)

        # 补丁协议（默认/自动模式）下大模板不再短路 —— LLM 被调用（重试一次），
        # 失败后确定性兜底。
        auto_client = ExplodingClient()
        response = translate_variant_rules("mahjong", "红中麻将 2人", use_llm=True, llm_client=auto_client)
        assert auto_client.calls == 2  # complete_with_retry 立即重试一次
        assert response.validation is not None
        assert response.validation.valid
        assert any("LLM 生成失败" in w for w in response.validation.warnings)

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


# ── LLM 增量补丁协议（v5.5）───────────────────────────────────────


class TestLLMPatchPath:
    """完整 LLM 增量补丁协议：模型输出 ``{"patch": [...]}`` 操作而非完整
    rules JSON，由 ``rule_patch`` 应用到基础模板。

    覆盖：补丁成功 / 补丁格式修复循环 / 补丁始终无效回退确定性 / 自动
    模式按模板尺寸选补丁 / 巨型模板（mahjong）经补丁协议可走 LLM。
    """

    def test_patch_success(self, gomoku_rules: dict) -> None:
        class PatchClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                payload = {"patch": [{"op": "replace", "path": "constants.board_size", "value": 15}]}
                return json.dumps(payload, ensure_ascii=False)

        response = translate_variant_rules(
            "stochastic_gomoku", "15x15", use_llm=True, llm_client=PatchClient(), patch_mode=True
        )

        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15
        assert any("增量补丁" in w for w in response.validation.warnings)

    def test_patch_repairs_invalid_format(self, gomoku_rules: dict) -> None:
        class RepairPatchClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                self.calls += 1
                if self.calls == 1:
                    # 非补丁格式（未知 op）→ 修复循环回喂错误
                    return json.dumps({"patch": [{"op": "upsert", "path": "x", "value": 1}]}, ensure_ascii=False)
                return json.dumps(
                    {"patch": [{"op": "replace", "path": "constants.board_size", "value": 15}]},
                    ensure_ascii=False,
                )

        client = RepairPatchClient()
        response = translate_variant_rules(
            "stochastic_gomoku", "15x15", use_llm=True, llm_client=client, patch_mode=True
        )

        assert client.calls == 2
        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15

    def test_patch_repairs_validation_failure(self, gomoku_rules: dict) -> None:
        class ValidationRepairClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                self.calls += 1
                if self.calls == 1:
                    # 合法格式但破坏 schema（删掉 actions）→ 校验失败进修复循环
                    return json.dumps({"patch": [{"op": "remove", "path": "actions"}]}, ensure_ascii=False)
                return json.dumps(
                    {"patch": [{"op": "replace", "path": "constants.board_size", "value": 15}]},
                    ensure_ascii=False,
                )

        client = ValidationRepairClient()
        response = translate_variant_rules(
            "stochastic_gomoku", "15x15", use_llm=True, llm_client=client, patch_mode=True
        )

        assert client.calls == 2
        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15

    def test_patch_never_valid_falls_back_to_deterministic(self) -> None:
        class NeverValidPatchClient:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                self.calls += 1
                # 每次都破坏 schema 的补丁
                return json.dumps({"patch": [{"op": "remove", "path": "actions"}]}, ensure_ascii=False)

        client = NeverValidPatchClient()
        response = translate_variant_rules(
            "stochastic_gomoku", "五子棋 15x15 连五", use_llm=True, llm_client=client, patch_mode=True
        )

        assert client.calls == 2  # 初始 + 1 次 repair
        assert response.validation is not None
        assert response.validation.valid  # 回退确定性路径
        assert response.rules_json["constants"]["board_size"] == 15
        assert any("增量补丁" in w or "补丁" in w for w in response.validation.warnings)

    def test_patch_message_carries_base_template_and_patch_instruction(self) -> None:
        captured: list[list[dict[str, str]]] = []

        class CapturingClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                captured.append(messages)
                return json.dumps({"patch": []}, ensure_ascii=False)

        translate_variant_rules(
            "stochastic_gomoku", "15x15", use_llm=True, llm_client=CapturingClient(), patch_mode=True
        )

        assert captured, "补丁路径必须调用 LLM"
        assert captured[0][0]["role"] == "system"
        assert "增量补丁" in captured[0][0]["content"]
        assert "base_rules_json" in captured[0][1]["content"]
        assert '"patch"' in captured[0][1]["content"]

    def test_auto_mode_uses_patch_for_oversized_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """自动模式：模板超过全量改写护栏 → 补丁协议（无需显式 patch_mode）。"""
        import layer1_translator.variant_translator as vt

        monkeypatch.setattr(vt, "_MAX_LLM_TEMPLATE_CHARS", 1)  # 任何模板都“超大”

        class PatchClient:
            calls = 0

            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                PatchClient.calls += 1
                return json.dumps(
                    {"patch": [{"op": "replace", "path": "constants.board_size", "value": 15}]},
                    ensure_ascii=False,
                )

        client = PatchClient()
        response = translate_variant_rules("stochastic_gomoku", "15x15", use_llm=True, llm_client=client)

        assert client.calls == 1  # 只走补丁，不重复调用
        assert response.validation is not None
        assert response.validation.valid
        assert response.rules_json["constants"]["board_size"] == 15

    def test_mahjong_oversized_translates_via_patch(self) -> None:
        """补齐暂缓项：巨型模板（mahjong ≈87k 字符）经增量补丁协议走 LLM ——
        旧护栏只允许确定性路径。"""
        from layer1_translator.variant_translator import _MAX_LLM_TEMPLATE_CHARS

        with open(RULES_DIR / "mahjong.json", "r", encoding="utf-8") as f:
            template = json.load(f)
        assert len(json.dumps(template, ensure_ascii=False)) > _MAX_LLM_TEMPLATE_CHARS, "fixture 前提"

        class PatchClient:
            def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
                # 只改声明式变体节的默认变体 —— 极小的一处补丁
                return json.dumps(
                    {"patch": [{"op": "replace", "path": "variants.variant", "value": "hongzhong"}]},
                    ensure_ascii=False,
                )

        response = translate_variant_rules("mahjong", "红中麻将", use_llm=True, llm_client=PatchClient())

        assert response.validation is not None
        assert response.validation.valid, response.validation.errors
        assert response.rules_json["variants"]["variant"] == "hongzhong"
        # 运行时装配验证：补丁真的生效了（声明式变体经引擎解析）
        engine = GameEngine(response.rules_json)
        assert engine._constants["variant"] == "hongzhong"  # noqa: SLF001


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
