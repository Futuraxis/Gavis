"""Difficulty/pacing effectiveness tests for OllamaSolver + undercover transfer.

「难度形同虚设」的回归防线：mock LLM 只捕获 prompt 与 temperature，不真正
联网，验证：

  1. 谁是卧底 prompt 按 difficulty 换 ``_UNDERCOVER_HINT`` 档；
  2. 狼人杀 prompt 按 (role, difficulty) 换 ``ROLE_GUIDE`` 档；
  3. pacing 决定 ``_ask_model`` 的温度（fast=0.9 / standard=0.7 / slow=0.5）；
  4. 平台传递链：difficulty/theme → variant → word_pairs 档（easy 差异大 /
     hard 极易混淆），自适应锚定 normal 档。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance
from layer3_solvers import OllamaConfig, OllamaSolver
from layer3_solvers.llm.ollama_solver import _PACING_TEMP, _UNDERCOVER_HINT, ROLE_GUIDE

RULES_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def _wolf_engine(seed: int = 7) -> GameEngine:
    with open(RULES_DIR / "werewolf.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def _solver(engine: GameEngine, difficulty: str = "normal", pacing: str = "standard") -> OllamaSolver:
    return OllamaSolver(engine, OllamaConfig(difficulty=difficulty, pacing=pacing), "p0")


def _uc_obs(word: str = "苹果") -> dict:
    # 谁是卧底 describe：只看自己的词（my_role 隐藏），无 seer/witch 字段。
    return {"phase": "describe", "alive": [1, 1, 1, 1, 1, 1, 1, 1], "my_word": word, "round": 1, "deaths": []}


def _uc_legal() -> list[ActionInstance]:
    return [ActionInstance("speak", "action", "p0", {"text": ""}, "speak")]


def _wolf_obs(role: str = "wolf") -> dict:
    return {"phase": "day_speech", "alive": [1] * 9, "my_role": role, "round": 1, "deaths": []}


def _wolf_legal() -> list[ActionInstance]:
    return [ActionInstance("speak", "action", "p0", {"intent": {"id": "claim"}, "text": ""}, "speak:claim")]


class TestUndercoverHint:
    """谁是卧底：difficulty → _UNDERCOVER_HINT 档（发言强度提示两维之一）。"""

    @pytest.mark.parametrize("difficulty", ["easy", "normal", "hard"])
    def test_hint_matches_difficulty(self, difficulty: str):
        prompt = _solver(_wolf_engine(), difficulty=difficulty)._build_prompt(_uc_obs(), _uc_legal())
        assert _UNDERCOVER_HINT[difficulty] in prompt
        for other in ("easy", "normal", "hard"):
            if other != difficulty:
                assert _UNDERCOVER_HINT[other] not in prompt

    def test_default_is_normal(self):
        prompt = _solver(_wolf_engine())._build_prompt(_uc_obs(), _uc_legal())
        assert _UNDERCOVER_HINT["normal"] in prompt
        assert _UNDERCOVER_HINT["hard"] not in prompt


class TestWerewolfGuide:
    """狼人杀：difficulty → ROLE_GUIDE 按角色分档的策略提示。"""

    @pytest.mark.parametrize("role", ["wolf", "seer", "witch", "hunter", "villager"])
    @pytest.mark.parametrize("difficulty", ["easy", "normal", "hard"])
    def test_guide_matches_role_and_difficulty(self, role: str, difficulty: str):
        prompt = _solver(_wolf_engine(), difficulty=difficulty)._build_prompt(_wolf_obs(role), _wolf_legal())
        assert ROLE_GUIDE[role][difficulty] in prompt
        for other in ("easy", "normal", "hard"):
            if other != difficulty:
                assert ROLE_GUIDE[role][other] not in prompt

    def test_easy_and_hard_wolf_guides_differ(self):
        obs, legal = _wolf_obs("wolf"), _wolf_legal()
        easy = _solver(_wolf_engine(), difficulty="easy")._build_prompt(obs, legal)
        hard = _solver(_wolf_engine(), difficulty="hard")._build_prompt(obs, legal)
        assert ROLE_GUIDE["wolf"]["easy"] in easy
        assert ROLE_GUIDE["wolf"]["hard"] in hard
        assert easy != hard


class TestPacingTemperature:
    """pacing → _ask_model 温度（快节奏发散易露馅 / 慢节奏精准强伪装）。"""

    @pytest.mark.parametrize("pacing,expected", [("fast", 0.9), ("standard", 0.7), ("slow", 0.5)])
    def test_temperature_follows_pacing(self, pacing: str, expected: float):
        calls: list[float | None] = []
        solver = _solver(_wolf_engine(), pacing=pacing)

        class _CapLLM:
            def complete_chat(self, system: str, prompt: str, temperature: float | None = None) -> str:
                calls.append(temperature)
                return ""

        solver._llm = _CapLLM()  # noqa: SLF001
        solver._ask_model("prompt")
        assert calls == [expected]

    def test_default_standard_temperature(self):
        calls: list[float | None] = []
        solver = _solver(_wolf_engine())

        class _CapLLM:
            def complete_chat(self, system: str, prompt: str, temperature: float | None = None) -> str:
                calls.append(temperature)
                return ""

        solver._llm = _CapLLM()  # noqa: SLF001
        solver._ask_model("prompt")
        assert calls == [_PACING_TEMP["standard"]]


class TestTransferChainUndercover:
    """平台传递链：difficulty/theme → variant(f"{theme}_{tier}") → word_pairs 档。"""

    @staticmethod
    def _spec():
        from layer4_interface.frontend.platform.families.social import build_spec

        with open(RULES_DIR / "undercover.json", "r", encoding="utf-8") as f:
            rules = json.load(f)
        return build_spec("undercover", rules)

    def test_explicit_theme_tier_variant(self):
        eng = self._spec().create_engine(seed=42, player_count=8, variant="animal_hard", difficulty="hard")
        assert eng.variant == "animal_hard"
        wp = eng._constants["word_pairs"]  # noqa: SLF001
        assert any("猎豹" in p and "花豹" in p for p in wp)

    def test_difficulty_alone_picks_default_theme_tier(self):
        eng = self._spec().create_engine(seed=42, player_count=8, variant=None, difficulty="hard")
        assert eng.variant == "fruit_hard"
        wp = eng._constants["word_pairs"]  # noqa: SLF001
        assert any("菠萝" in p and "凤梨" in p for p in wp)

    def test_easy_pairs_differ_obviously_hard_pairs_are_confusable(self):
        spec = self._spec()
        easy = spec.create_engine(seed=42, player_count=8, variant=None, difficulty="easy")._constants["word_pairs"]  # noqa: SLF001
        hard = spec.create_engine(seed=42, player_count=8, variant=None, difficulty="hard")._constants["word_pairs"]  # noqa: SLF001
        assert any("苹果" in p and "香蕉" in p for p in easy)
        assert any("苹果" in p and "沙果" in p for p in hard)
        assert easy != hard

    def test_adaptive_anchors_normal_tier(self):
        eng = self._spec().create_engine(seed=42, player_count=8, variant=None, difficulty="adaptive")
        assert eng.variant == "fruit_normal"