"""Tests for the unified LLM client (layer2_engine/core/llm.py).

Covers the fail-soft contract every layer depends on: unreachable /
malformed endpoints return ``""`` / empty tool calls instead of raising;
function-calling replies parse into ``ToolCall``; API-key resolution keeps
the audit-3.6 precedence; ``sanitize_text`` does the shared cleaning.
Transport responses are injected via monkeypatched ``urlopen`` so no real
network or model is needed.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from layer2_engine.core.llm import LLMClient, LLMConfig, ToolCall, sanitize_text


class _FakeResponse:
    """Minimal ``urlopen``-style context-managed response."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False


class TestSanitizeText:
    def test_strips_control_chars_and_caps(self) -> None:
        text = "a\x00b\x01\x1f" + "啊" * 300
        cleaned = sanitize_text(text, 100)
        assert len(cleaned) == 100
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert cleaned.startswith("ab啊")

    def test_no_cap_keeps_length(self) -> None:
        assert len(sanitize_text("你好" * 50)) == 100


class TestLLMClientFailSoft:
    def test_unreachable_complete_returns_empty(self) -> None:
        client = LLMClient(base_url="http://127.0.0.1:59999", timeout_s=0.3)
        assert client.complete([{"role": "user", "content": "hi"}]) == ""

    def test_unreachable_tools_returns_empty_reply(self) -> None:
        client = LLMClient(base_url="http://127.0.0.1:59998", timeout_s=0.3)
        reply = client.complete_tools([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        assert reply.text == ""
        assert reply.tool_calls == []

    def test_unreachable_complete_chat_returns_empty(self) -> None:
        client = LLMClient(base_url="http://127.0.0.1:59997", timeout_s=0.3)
        assert client.complete_chat("sys", "usr", 8) == ""

    def test_available_probe_false_when_unreachable(self) -> None:
        assert LLMClient.available(base_url="http://127.0.0.1:59996") is False


class TestLLMClientParsing:
    def test_complete_parses_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({"choices": [{"message": {"content": "回复文本"}}]}).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            return _FakeResponse(body)

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient()
        assert client.complete([{"role": "user", "content": "hi"}], max_tokens=64) == "回复文本"

    def test_tools_parses_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "play_game", "arguments": '{"game_id": "moon_chess"}'}}
                            ],
                        }
                    }
                ]
            }
        ).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            return _FakeResponse(body)

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient()
        reply = client.complete_tools([{"role": "user", "content": "我想玩月亮棋"}], tools=[{"type": "function"}])
        assert reply.text == ""
        assert len(reply.tool_calls) == 1
        call = reply.tool_calls[0]
        assert isinstance(call, ToolCall)
        assert call.name == "play_game"
        assert call.arguments == {"game_id": "moon_chess"}

    def test_malformed_body_fails_soft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req, timeout=None):
            return _FakeResponse(b"not json")

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient()
        assert client.complete([{"role": "user", "content": "hi"}]) == ""

    def test_transport_exception_fails_soft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(req, timeout=None):
            raise urllib.error.URLError("down")

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", boom)
        client = LLMClient()
        assert client.complete([{"role": "user", "content": "hi"}]) == ""


class TestLLMClientApiKey:
    def test_precedence_param_env_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        assert LLMClient(api_key="param-key").api_key == "param-key"
        assert LLMClient().api_key == "env-key"
        monkeypatch.delenv("LLM_API_KEY")
        assert LLMClient().api_key == "ollama"  # 本地默认

    def test_config_object_honored(self) -> None:
        client = LLMClient(LLMConfig(model="custom-model", base_url="http://host:11434"))
        assert client.model == "custom-model"
        assert client.base_url == "http://host:11434"