"""Tests for the unified LLM client (layer2_engine/core/llm.py).

Covers the fail-soft contract every layer depends on: unreachable /
malformed endpoints return ``""`` / empty tool calls instead of raising;
function-calling replies parse into ``ToolCall``; API-key resolution keeps
the audit-3.6 precedence; ``sanitize_text`` does the shared cleaning.
Transport responses are injected via monkeypatched ``urlopen`` so no real
network or model is needed.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from layer2_engine.core.llm import LLMClient, LLMClientError, LLMConfig, ToolCall, sanitize_text


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

    def test_tools_keeps_call_id_for_tool_result_pairing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """多轮工具循环需要 ``tool_call_id`` 把 role:"tool" 结果消息关联回
        发起调用的 assistant 消息 —— 解析层不得丢弃端点给出的 id。"""
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "function": {"name": "describe_game", "arguments": '{"game_id": "moon_chess"}'},
                                }
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
        reply = client.complete_tools([{"role": "user", "content": "月亮棋是什么"}], tools=[{"type": "function"}])
        assert len(reply.tool_calls) == 1
        assert reply.tool_calls[0].id == "call_abc123"

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


def _http_error(url: str, code: int, body: str) -> urllib.error.HTTPError:
    """Construct a real ``HTTPError`` carrying an API error body."""
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(body.encode("utf-8")))


class TestLLMClientErrorClassification:
    """审查（LLM 兜底系统性排查）：API 错误必须分类记录并上浮，而非统一空串。

    Fail-soft 默认行为不变（返回 ``""``），但真实原因记录在
    ``last_error`` 并写日志；``fail_hard=True`` 则把 API 4xx/5xx /
    传输失败升级为 :class:`LLMClientError`。
    """

    def test_http_401_fails_soft_but_records_cause(self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        def fake_urlopen(req, timeout=None):
            raise _http_error(req.full_url, 401, '{"error": "invalid api key"}')

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient()
        assert client.complete([{"role": "user", "content": "hi"}]) == ""
        assert client.last_error is not None
        message = str(client.last_error)
        assert "401" in message
        assert "invalid api key" in message
        assert any("401" in record.message for record in caplog.records)

    def test_http_error_detail_includes_server_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req, timeout=None):
            raise _http_error(req.full_url, 429, '{"message": "rate limit exceeded"}')

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient()
        assert client.complete([{"role": "user", "content": "hi"}]) == ""
        assert "429" in str(client.last_error)
        assert "rate limit" in str(client.last_error)

    def test_fail_hard_raises_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req, timeout=None):
            raise _http_error(req.full_url, 401, '{"error": "invalid api key"}')

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient(fail_hard=True)
        with pytest.raises(LLMClientError) as excinfo:
            client.complete([{"role": "user", "content": "hi"}])
        assert "401" in str(excinfo.value)
        assert "invalid api key" in str(excinfo.value)

    def test_fail_hard_via_config_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req, timeout=None):
            raise _http_error(req.full_url, 500, '{"error": "boom"}')

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient(LLMConfig(fail_hard=True))
        with pytest.raises(LLMClientError) as excinfo:
            client.complete_tools([{"role": "user", "content": "hi"}], tools=[])
        assert "500" in str(excinfo.value)

    def test_fail_hard_raises_on_transport_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", boom)
        client = LLMClient(fail_hard=True)
        with pytest.raises(LLMClientError) as excinfo:
            client.complete([{"role": "user", "content": "hi"}])
        assert "connection refused" in str(excinfo.value)

    def test_fail_hard_raises_on_malformed_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_urlopen(req, timeout=None):
            class Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self, n=-1):
                    return b"not json"

            return Resp()

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient(fail_hard=True)
        with pytest.raises(LLMClientError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_last_error_cleared_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(req.full_url, 401, '{"error": "invalid api key"}')
            body = json.dumps({"choices": [{"message": {"content": "回复"}}]}).encode("utf-8")
            return _FakeResponse(body)

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        client = LLMClient()
        assert client.complete([{"role": "user", "content": "hi"}]) == ""
        assert client.last_error is not None
        assert client.complete([{"role": "user", "content": "hi"}]) == "回复"
        assert client.last_error is None
