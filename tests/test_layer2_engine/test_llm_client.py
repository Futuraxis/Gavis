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
from typing import Iterator

import pytest

from layer2_engine.core.llm import (
    LLMClient,
    LLMClientError,
    LLMConfig,
    StreamChunk,
    ToolCall,
    sanitize_text,
)


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


class TestLLMClientEndpointModelResolution:
    """Endpoint/model 统一优先级：显式配置 > LLM_BASE_URL/LLM_MODEL > 内置默认。"""

    def test_env_overrides_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "http://env-host:9999")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        assert LLMClient().base_url == "http://env-host:9999"
        assert LLMClient().model == "env-model"

    def test_explicit_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "http://env-host:9999")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        client = LLMClient(base_url="http://explicit:8888", model="explicit-model")
        assert client.base_url == "http://explicit:8888"
        assert client.model == "explicit-model"

    def test_blank_env_falls_back_to_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        client = LLMClient()
        assert client.base_url == "http://127.0.0.1:11434"
        assert client.model == "qwen3:8b"

    def test_available_probe_uses_env_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """探测端点与实例化同源：``available()`` 也吃 LLM_BASE_URL。"""
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:59995")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert LLMClient.available() is False  # 该端口必然不可达

    def test_available_probe_sends_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """无参 ``available()`` 必须带上 LLM_API_KEY 环境变量里的密钥——
        否则配置了云端鉴权端点（DeepSeek/GLM/OpenAI）的平台会话在
        ``/v1/models`` 探测时 401 → 静默退化为 random 求解器，AI 不跟着
        平台配置走 API（用户反馈的「AI 还是不发言」根因）。"""
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:59994")
        monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
        captured: dict[str, str] = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization", "")
            raise urllib.error.URLError("probe-stub")

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", fake_urlopen)
        assert LLMClient.available() is False  # stub 抛 URLError → False
        assert captured["url"].rstrip("/").endswith("/v1/models")
        assert captured["auth"] == "Bearer sk-test-123"


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


class _FakeStreamResponse:
    """urlopen 风格的 SSE 响应：按行 ``readline()`` 喂数据."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]
        self._idx = 0

    def readline(self) -> bytes:
        if self._idx >= len(self._lines):
            return b""
        line = self._lines[self._idx]
        self._idx += 1
        return line

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False


def _sse(payload: object) -> str:
    """一行 OpenAI 系 SSE ``data:`` 帧（含换行）。"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n"


class TestCompleteStream:
    """流式传输：SSE 增量、三种终态、思维链双来源、工具分片、失败收尾."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, resp: object) -> None:
        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", lambda req, timeout=None: resp)

    def _collect(self, chunks: Iterator[StreamChunk]) -> tuple[str, str, list[ToolCall], str]:
        """收拢 (text, reasoning, tool_calls, error)。"""
        text = ""
        reasoning = ""
        tool_calls: list[ToolCall] = []
        error = ""
        for chunk in chunks:
            text += chunk.text
            reasoning += chunk.reasoning
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            if chunk.error:
                error = chunk.error
            assert chunk.done or chunk.text or chunk.reasoning or chunk.tool_calls or chunk.error
        return text, reasoning, tool_calls, error

    def test_delta_text_and_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            _sse({"choices": [{"delta": {"content": "你"}}]}),
            _sse({"choices": [{"delta": {"content": "好"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, reasoning, tool_calls, error = self._collect(
            LLMClient().complete_stream([{"role": "user", "content": "hi"}])
        )
        assert text == "你好"
        assert reasoning == ""
        assert tool_calls == []
        assert error == ""

    def test_done_marker_terminates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [_sse({"choices": [{"delta": {"content": "嗨"}}]}), "data: [DONE]\n"]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, _, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "嗨"

    def test_ollama_native_done_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            _sse({"choices": [{"delta": {"content": "哈"}}], "done": False}),
            _sse({"choices": [], "done": True}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, _, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "哈"

    def test_reasoning_content_delta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            _sse({"choices": [{"delta": {"reasoning_content": "先看中心"}}]}),
            _sse({"choices": [{"delta": {"content": "建议占中心。"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, reasoning, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "建议占中心。"
        assert reasoning == "先看中心"

    def test_reasoning_fallback_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            _sse({"choices": [{"delta": {"reasoning": "兜底键"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        _, reasoning, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert reasoning == "兜底键"

    def test_think_tags_across_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 标签被三个 chunk 切开（``<thi | nk>… | nk>…``），不得丢字。
        lines = [
            _sse({"choices": [{"delta": {"content": "开场<thi"}}]}),
            _sse({"choices": [{"delta": {"content": "nk>先想</thi"}}]}),
            _sse({"choices": [{"delta": {"content": "nk>收尾"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, reasoning, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "开场收尾"
        assert reasoning == "先想"

    def test_unclosed_close_at_eof_kept_in_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [_sse({"choices": [{"delta": {"content": "开场<think>思考"}}]})]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, reasoning, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "开场"
        assert reasoning == "思考"

    def test_unclosed_open_at_eof_as_literal_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [_sse({"choices": [{"delta": {"content": "开场<thi"}}]})]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, _, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "开场<thi"

    def test_stray_close_without_open_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            _sse({"choices": [{"delta": {"content": "一句</think>两句"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, reasoning, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "一句两句"
        assert reasoning == ""

    def test_non_data_lines_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            ": keep-alive 注释\n",
            "\n",
            _sse({"choices": [{"delta": {"content": "正常"}}]}),
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, _, _, _ = self._collect(LLMClient().complete_stream([{"role": "user", "content": "hi"}]))
        assert text == "正常"

    def test_tool_fragments_accumulated_parsed_at_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines = [
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "id": "call_1", "function": {"name": "play", "arguments": '{"game'}}
                                ]
                            }
                        }
                    ]
                }
            ),
            _sse(
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '_id": "moon_chess"}'}}]}}
                    ]
                }
            ),
            _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        ]
        self._patch(monkeypatch, _FakeStreamResponse(lines))
        text, _, tool_calls, _ = self._collect(
            LLMClient().complete_stream([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        )
        assert text == ""
        assert len(tool_calls) == 1
        call = tool_calls[0]
        assert call.name == "play"
        assert call.arguments == {"game_id": "moon_chess"}
        assert call.id == "call_1"

    def test_bad_json_line_fails_soft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, _FakeStreamResponse(["data: not json\n"]))
        client = LLMClient()
        _, _, _, error = self._collect(client.complete_stream([{"role": "user", "content": "hi"}]))
        assert error
        assert client.last_error is not None

    def test_http_error_fails_soft_with_error_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(req, timeout=None):
            raise _http_error(req.full_url, 401, '{"error": "invalid api key"}')

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", boom)
        client = LLMClient()
        _, _, _, error = self._collect(client.complete_stream([{"role": "user", "content": "hi"}]))
        assert "401" in error
        assert "invalid api key" in error
        assert client.last_error is not None

    def test_fail_hard_stream_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("layer2_engine.core.llm.urllib.request.urlopen", boom)
        client = LLMClient(fail_hard=True)
        with pytest.raises(LLMClientError) as excinfo:
            list(client.complete_stream([{"role": "user", "content": "hi"}]))
        assert "connection refused" in str(excinfo.value)


class TestChatReplyReasoning:
    """非流式路径的思维链提取（complete_chat_reply → ChatReply.reasoning）."""

    def test_reasoning_field_and_think_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "直接想法",
                            "content": "开场<think>先想</think>答案",
                        }
                    }
                ]
            }
        ).encode("utf-8")
        monkeypatch.setattr(
            "layer2_engine.core.llm.urllib.request.urlopen", lambda req, timeout=None: _FakeResponse(body)
        )
        reply = LLMClient().complete_chat_reply("sys", "usr", 64)
        assert reply.text == "开场答案"
        assert reply.reasoning == "直接想法先想"

    def test_no_reasoning_keeps_text_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = json.dumps({"choices": [{"message": {"content": "普通回复"}}]}).encode("utf-8")
        monkeypatch.setattr(
            "layer2_engine.core.llm.urllib.request.urlopen", lambda req, timeout=None: _FakeResponse(body)
        )
        reply = LLMClient().complete_chat_reply("sys", "usr", 64)
        assert reply.text == "普通回复"
        assert reply.reasoning == ""
