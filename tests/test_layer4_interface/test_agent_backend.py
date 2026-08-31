"""Tests for the Agent companionship backend (C2, layer4_interface.agent)."""

from __future__ import annotations

import pytest

from layer2_engine.core.llm import ChatReply
from layer4_interface.agent import (
    PERSONAS,
    SCENARIOS,
    AgentMessage,
    DialogueEngine,
    LLMClient,
    SkillContext,
    Skills,
    assert_no_hidden,
    scan,
)
from layer4_interface.frontend.engine_helpers import engine_from_rules, resolve_all_chance

_MOODS = {"happy", "thinking", "sorry", "neutral"}


def _make_ctx() -> SkillContext:
    """Build a minimal deterministic SkillContext (no hidden fields)."""
    return SkillContext(
        human_pid="p0",
        observation={"env": {"phase": "betting"}},
        legal_actions=[],
        evaluation={"score": 0.0, "summary": "局面胶着", "mechanical_text": "p0 当前评估 +0.00，局面胶着"},
        revealed=False,
    )


class FakeLLM:
    """Deterministic in-memory LLM stub for pipeline tests."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete_chat(self, system: str, user: str, max_tokens: int) -> str:
        return self.reply

    def complete_chat_reply(self, system: str, user: str, max_tokens: int) -> ChatReply:
        return ChatReply(text=self.complete_chat(system, user, max_tokens))


class PromptCaptureLLM:
    """Fake LLM recording the exact (system, user) prompts handed over."""

    def __init__(self, reply: str = "好的") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete_chat(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append((system, user))
        return self.reply

    def complete_chat_reply(self, system: str, user: str, max_tokens: int) -> ChatReply:
        self.calls.append((system, user))
        return ChatReply(text=self.reply)


@pytest.fixture
def moon_engine():
    return engine_from_rules("moon_chess", seed=42)


@pytest.fixture
def texas_engine():
    return engine_from_rules("texas_holdem", seed=42)


class TestPersonaCoverage:
    def test_every_scenario_has_fallback_for_every_persona(self):
        for persona in PERSONAS.values():
            for scenario in SCENARIOS:
                assert persona.fallback_lines.get(scenario), f"{persona.key}/{scenario} 缺少兜底台词"


class TestDialogueEngine:
    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_reply_nonempty_for_all_personas(self, scenario: str):
        ctx = _make_ctx()
        for persona in PERSONAS.values():
            engine = DialogueEngine(persona)
            message = engine.reply(ctx, scenario)
            assert message.text, f"{persona.key}/{scenario} 兜底为空"
            assert message.mood in _MOODS

    def test_llm_none_falls_back_to_persona_lines(self):
        persona = PERSONAS["gentle"]
        engine = DialogueEngine(persona, llm=None)
        message = engine.reply(_make_ctx(), "greet")
        assert message.text in persona.fallback_lines["greet"]

    def test_cleaning_trims_length_and_strips_control_chars(self):
        llm = FakeLLM("前缀\x00\x01\x1f内容" + "啊" * 300)
        engine = DialogueEngine(PERSONAS["gentle"], llm=llm, max_len=100)
        message = engine.reply(_make_ctx(), "greet")
        assert len(message.text) <= 100
        assert "\x00" not in message.text
        assert "\x01" not in message.text

    def test_dedup_within_window_does_not_repeat_same_line(self):
        persona = PERSONAS["gentle"]
        engine = DialogueEngine(persona, llm=None)
        ctx = _make_ctx()
        first = engine.reply(ctx, "greet")
        second = engine.reply(ctx, "greet")
        assert first.text != second.text
        assert second.text in persona.fallback_lines["greet"]

    def test_mute_returns_empty_neutral(self):
        engine = DialogueEngine(PERSONAS["gentle"])
        engine.set_muted(True)
        assert engine.reply(_make_ctx(), "greet") == AgentMessage("", "neutral")


class TestDialogueKnowledgeInjection:
    """陪伴聊天的游戏知识注入（audit §5-4）：persona 提到玩法时依据
    权威资料作答，不再靠模型参数记忆。"""

    def test_game_knowledge_injected_into_prompt(self):
        llm = PromptCaptureLLM()
        engine = DialogueEngine(PERSONAS["gentle"], llm=llm)
        engine.reply(_make_ctx(), "good_move", game_id="moon_chess")
        assert len(llm.calls) == 1
        system, user = llm.calls[0]
        # user prompt 带权威资料（注册表简介 + docs 规则段，与 chat 信息工具同源）
        assert "游戏资料" in user
        assert "3×3" in user
        assert "规则要点" in user
        # system prompt 立知识红线（提到玩法只依据资料）
        assert "资料" in system

    def test_unknown_game_injects_nothing(self):
        """custom / 未知 id → 不注入（fail-soft），旧 prompt 结构不变。"""
        llm = PromptCaptureLLM()
        engine = DialogueEngine(PERSONAS["gentle"], llm=llm)
        engine.reply(_make_ctx(), "good_move", game_id="custom_thing")
        system, user = llm.calls[0]
        assert "游戏资料" not in user

    def test_no_game_id_backward_compatible(self):
        """旧调用面（不传 game_id）行为不变：无资料注入，正常成文。"""
        llm = PromptCaptureLLM()
        engine = DialogueEngine(PERSONAS["gentle"], llm=llm)
        message = engine.reply(_make_ctx(), "greet")
        assert message.mood in _MOODS
        system, user = llm.calls[0]
        assert "游戏资料" not in user


class TestHiddenGuard:
    def test_assert_no_hidden_rejects_my_hole(self):
        ctx = SkillContext("p0", {"my_hole": ["♠A"]}, [], {"score": 0.0}, False)
        with pytest.raises(ValueError):
            assert_no_hidden(ctx)

    def test_assert_no_hidden_rejects_hand_p0(self):
        ctx = SkillContext("p0", {"hand_p0": ["m1"]}, [], {"score": 0.0}, False)
        with pytest.raises(ValueError):
            assert_no_hidden(ctx)

    def test_assert_no_hidden_passes_clean_observation(self):
        assert_no_hidden(_make_ctx())

    def test_scan_rewrites_texas_hole_notation(self):
        output = scan("我的底牌是 ♠A ♥K，这局稳了。", "texas_holdem")
        assert "♠A" not in output
        assert "不细说" in output

    def test_scan_leaves_safe_text_untouched(self):
        text = "这步棋下得不错。"
        assert scan(text, "texas_holdem") == text


class TestSkills:
    def test_build_moon_chess_context(self, moon_engine):
        state = moon_engine.create_initial_state()
        ctx = Skills.build(state, "p_black", moon_engine)
        assert ctx.human_pid == "p_black"
        assert ctx.revealed is False
        assert "score" in ctx.evaluation

    def test_build_texas_projects_without_hidden_leak(self, texas_engine):
        state = resolve_all_chance(texas_engine, texas_engine.create_initial_state())
        ctx = Skills.build(state, "p_sb", texas_engine)
        assert_no_hidden(ctx)
        assert "bb_hole" not in ctx.observation
        assert "hand_p0" not in ctx.observation

    def test_deterministic_skill_apis(self, moon_engine):
        ctx = _make_ctx()
        assert "score" in Skills.evaluate_position(ctx, moon_engine)
        assert Skills.detect_good_move(ctx, moon_engine) is None
        assert Skills.detect_blunder(ctx, moon_engine) is None
        assert "hint" in Skills.suggest_hint(ctx, "direction", None, moon_engine)
        assert "hint" in Skills.suggest_hint(ctx, "specific", None, moon_engine)
        assert "hint" in Skills.suggest_hint(ctx, "demo", None, moon_engine)
        result = Skills.summarize_result(ctx, moon_engine, "p_black", "p_black")
        assert result["won"] is True
        assert "reason" in Skills.explain_illegal(ctx, moon_engine, {"choice": "fold"})
        assert "summary" in Skills.idle_reminder(ctx)
        greet = Skills.greet(ctx, {"nickname": "阿远", "recent": {"moon_chess": {"wins": 3, "plays": 5}}})
        assert greet["nickname"] == "阿远"


class TestLLMClient:
    def test_complete_fails_soft_when_unreachable(self):
        client = LLMClient(base_url="http://127.0.0.1:59999", timeout_s=0.5)
        assert client.complete_chat("sys", "usr", 8) == ""


class TestEvaluationPidFree:
    """evaluate 的 summary/mechanical_text 不得含原始 pid。

    pid（如 ``p_sb``）会经 ``_scenario_payload`` 的 ``summary`` 字段渗入对话
    LLM 的「机械事实」，被角色扮演的 AI 复述成「p_sb 赢了」（用户实测）。
    摘要须为 viewer 相对、pid 无关的措辞（「本方获胜」等）。
    """

    class _TermEngine:
        """Stub：恒终局，viewer=p_sb 时 utility=+1（本方获胜），否则 -1。"""

        def is_terminal(self, state):
            return True

        def get_utility(self, state, viewer):
            return 1.0 if viewer == "p_sb" else -1.0

    def test_terminal_summary_has_no_pid(self):
        from layer4_interface.agent.evaluation import evaluate

        eng = self._TermEngine()
        for viewer in ("p_sb", "p_bb", "p0"):
            res = evaluate({}, viewer, eng)
            assert viewer not in res["summary"], res
            assert viewer not in res["mechanical_text"], res
            assert res["summary"] in ("本方获胜", "本方落败", "平局"), res

    def test_nonterminal_summary_has_no_pid(self, moon_engine):
        from layer4_interface.agent.evaluation import evaluate

        state = moon_engine.create_initial_state()
        for viewer in ("p_black", "p_white"):
            res = evaluate(state, viewer, moon_engine)
            assert viewer not in res["summary"], res
            assert viewer not in res["mechanical_text"], res
