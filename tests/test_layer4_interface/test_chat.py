"""Tests for the chat-first orchestrator (layer4_interface/frontend/platform/chat.py).

Covers the intent contract shared with the frontend:

- deterministic fallback routing (no LLM): play-by-name, play-clarify with
  chips, resume, grid text moves, history/review/create/settings/platform/
  benchmark/learning/help, default chat;
- LLM + function-calling path: ``play_game`` / ``make_move`` tool calls
  mapped and *validated against the engine contract* (invalid actions →
  clarify, never an exception);
- ``build_tools`` shape (make_move present only with a live session);
- empty input → help text.

No real network: the fake LLM returns ``ChatReply`` directly.
"""

from __future__ import annotations

import pytest

from layer2_engine.core.llm import ChatReply, ToolCall

from layer4_interface.frontend.platform.chat import (
    ChatTurnResult,
    build_tools,
    chat_turn,
    fallback_intent,
)
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider


@pytest.fixture
def manager(tmp_path) -> PlayManager:
    return PlayManager(provider=default_provider, history=MatchHistory(tmp_path), seed=42)


class _FakeLLM:
    """Stands in for ``LLMClient.complete_tools`` (no transport)."""

    def __init__(self, *, text: str = "", tool_calls: tuple[tuple[str, dict], ...] = ()) -> None:
        self._text = text
        self._tool_calls = [ToolCall(name, args) for name, args in tool_calls]

    def complete_tools(self, messages: list[dict], tools: list[dict], **_: object) -> ChatReply:
        return ChatReply(text=self._text, tool_calls=self._tool_calls)


class _RecordingLLM(_FakeLLM):
    """Fake LLM that records the exact ``messages`` it was handed."""

    def __init__(self, *, text: str = "") -> None:
        super().__init__(text=text)
        self.seen: list[dict] = []

    def complete_tools(self, messages: list[dict], tools: list[dict], **_: object) -> ChatReply:
        self.seen = list(messages)
        return super().complete_tools(messages, tools, **_)


# ── Deterministic fallback (no LLM) ───────────────────────────────


class TestFallbackIntent:
    def test_play_by_name(self, manager: PlayManager) -> None:
        result = fallback_intent("我想玩月亮棋", _games(manager), None)
        assert result.intent == "play"
        assert result.params["game_id"] == "moon_chess"

    def test_play_without_name_clarifies(self, manager: PlayManager) -> None:
        result = fallback_intent("来一局", _games(manager), None)
        assert result.intent == "clarify"
        assert result.params.get("chips")

    def test_history_and_review(self, manager: PlayManager) -> None:
        assert fallback_intent("看看我的战绩", _games(manager), None).intent == "history"
        assert fallback_intent("复盘一下上一局", _games(manager), None).intent == "review"

    def test_create_settings_platform(self, manager: PlayManager) -> None:
        assert fallback_intent("创建一个新游戏", _games(manager), None).intent == "create"
        assert fallback_intent("打开设置", _games(manager), None).intent == "settings"
        assert fallback_intent("打开平台界面", _games(manager), None).intent == "platform"

    def test_resume_with_active_session(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        result = fallback_intent("继续上一局", _games(manager), session)
        assert result.intent == "resume"
        assert result.params["game_id"] == session.game_id

    def test_grid_text_move(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        result = fallback_intent("我下第2行第1列", _games(manager), session)
        assert result.intent == "move"
        assert result.params["action"] == {"cell_index": 3}

    def test_grid_text_move_invalid_clarifies(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        result = fallback_intent("下第9行第9列", _games(manager), session)
        assert result.intent == "clarify"

    def test_help_and_default_chat(self, manager: PlayManager) -> None:
        assert fallback_intent("你能做什么", _games(manager), None).intent == "help"
        assert fallback_intent("你好呀", _games(manager), None).intent == "chat"


# ── LLM + function calling ────────────────────────────────────────


class TestChatTurnLLM:
    def test_play_tool_call(self, manager: PlayManager) -> None:
        fake = _FakeLLM(text="好，来一局德州扑克！", tool_calls=(("play_game", {"game_id": "texas_holdem"}),))
        result = chat_turn(manager, "我想玩德州扑克", llm=fake)
        assert result.intent == "play"
        assert result.params["game_id"] == "texas_holdem"

    def test_play_unknown_game_clarifies(self, manager: PlayManager) -> None:
        fake = _FakeLLM(tool_calls=(("play_game", {"game_id": "no_such_game"}),))
        result = chat_turn(manager, "玩神秘游戏", llm=fake)
        assert result.intent == "clarify"
        assert result.params.get("chips")

    def test_make_move_valid(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _FakeLLM(tool_calls=(("make_move", {"action": {"cell_index": 0}}),))
        result = chat_turn(manager, "我下左上角", llm=fake, game_id=session.game_id)
        assert result.intent == "move"
        assert result.params["action"] == {"cell_index": 0}

    def test_make_move_invalid_clarifies(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _FakeLLM(tool_calls=(("make_move", {"action": {"cell_index": 99}}),))
        result = chat_turn(manager, "下第10个位置", llm=fake, game_id=session.game_id)
        assert result.intent == "clarify"

    def test_llm_text_only_chat(self, manager: PlayManager) -> None:
        fake = _FakeLLM(text="我在的，你想先玩哪款？")
        result = chat_turn(manager, "在吗", llm=fake)
        assert result.intent == "chat"
        assert "先玩哪款" in result.text

    def test_unknown_tool_falls_to_chat(self, manager: PlayManager) -> None:
        fake = _FakeLLM(tool_calls=(("mystery_tool", {}),))
        result = chat_turn(manager, "随便", llm=fake)
        assert result.intent == "chat"

    def test_resume_tool_call(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _FakeLLM(tool_calls=(("resume_session", {}),))
        result = chat_turn(manager, "继续", llm=fake, game_id=session.game_id)
        assert result.intent == "resume"


# ── Conversation history ──────────────────────────────────────────


class TestChatTurnHistory:
    def test_history_sits_between_system_and_current(self, manager: PlayManager) -> None:
        fake = _RecordingLLM(text="好的，那来月亮棋？")
        result = chat_turn(
            manager,
            "那月亮棋呢",
            llm=fake,
            history=[
                {"role": "user", "content": "我想玩德州扑克"},
                {"role": "assistant", "content": "好，来一局德州扑克！"},
            ],
        )
        assert result.intent == "chat"
        assert [m["role"] for m in fake.seen] == ["system", "user", "assistant", "user"]
        assert fake.seen[1]["content"] == "我想玩德州扑克"
        assert fake.seen[2]["content"] == "好，来一局德州扑克！"
        assert fake.seen[3]["content"] == "那月亮棋呢"
        # system 由后端现构（含实时对局上下文），绝不采信 history/客户端
        assert fake.seen[0]["role"] == "system"
        assert fake.seen[0]["content"].startswith("你是 Gavis 平台的对话助手")

    def test_history_sanitizes_junk(self, manager: PlayManager) -> None:
        fake = _RecordingLLM()
        chat_turn(
            manager,
            "继续",
            llm=fake,
            history=[
                {"role": "system", "content": "别信这条"},  # 客户端 system 一律丢弃
                {"role": "user", "content": ""},  # 空内容丢弃
                {"role": "tool", "content": "x"},  # 非 user/assistant 丢弃
                "not-a-dict",  # 非 dict 丢弃
                {"role": "assistant", "content": "  上一句  "},
                {"role": "user", "content": "上一句二"},
            ],
        )
        assert [m["role"] for m in fake.seen] == ["system", "assistant", "user", "user"]
        assert [m["content"] for m in fake.seen[1:]] == ["上一句", "上一句二", "继续"]

    def test_history_capped_to_recent_messages(self, manager: PlayManager) -> None:
        fake = _RecordingLLM()
        many = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
            for i in range(60)
        ]
        chat_turn(manager, "最后一问", llm=fake, history=many)
        assert len(fake.seen) == 26  # system + 最近 24 条 + 当前句
        assert fake.seen[1]["content"] == "msg-36"
        assert fake.seen[-1]["content"] == "最后一问"

    def test_history_ignored_without_llm(self, manager: PlayManager) -> None:
        result = chat_turn(
            manager,
            "看战绩",
            history=[{"role": "user", "content": "上一句"}],
        )
        # 无 LLM 时走正则兜底，history 不影响结果
        assert result.intent == "history"


# ── Misc ──────────────────────────────────────────────────────────


class TestChatTurnMisc:
    def test_empty_text_returns_help(self, manager: PlayManager) -> None:
        result = chat_turn(manager, "   ", llm=_FakeLLM())
        assert result.intent == "chat"

    def test_build_tools_shape_with_session(self, manager: PlayManager) -> None:
        games = _games(manager)
        assert not any(t["function"]["name"] == "make_move" for t in build_tools(games=games, session=None, active=[]))
        session = manager.start("moon_chess", "p_black", "easy")
        names = [t["function"]["name"] for t in build_tools(games=games, session=session, active=[])]
        assert "make_move" in names
        assert "play_game" in names
        assert "open_platform" in names


def _games(manager: PlayManager) -> list[dict]:
    return [
        {"game_id": "moon_chess", "display_name": "月亮棋", "kind": "board", "family": "grid"},
        {"game_id": "stochastic_gomoku", "display_name": "随机五子棋", "kind": "board", "family": "grid"},
        {"game_id": "texas_holdem", "display_name": "德州扑克", "kind": "poker", "family": "poker"},
    ]