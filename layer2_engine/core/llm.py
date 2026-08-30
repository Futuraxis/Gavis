"""Unified LLM client — the single authority for LLM access in Gavis.

Consolidates the previously scattered clients into one OpenAI-compatible
chat/completions client:

  - Layer 1: ``local_client.py`` (local transformers + OpenAI-compatible
    rule clients) and ``datasets.py`` (Layer 1's own rule-LLM training)
  - Layer 3: ``llm/ollama_solver.py`` inline urllib call and
    ``social/llm_policy.py`` OpenAICompatibleClient
  - Layer 4: ``agent/llm_client.py`` OllamaClient

It lives in Layer 2 core because that is the lowest layer Layer 1/3/4 may
all legally import (same precedent as ``api_key.py``).  A single transport
— ``{base_url}/v1/chat/completions`` — works against local Ollama
(``/v1``), vLLM, DeepSeek and any other OpenAI-compatible endpoint, and
also carries function calling (``tools``) for the agent-chat turn.

Semantics stay fail-soft (matching every previous client): transport /
parse failures return ``""`` / empty ``tool_calls`` instead of raising, so
callers keep their existing template / random fallback contracts.  No
third-party HTTP dependency (stdlib ``urllib`` only).

Error observability (审查: LLM 兜底系统性排查): every swallowed failure
is now classified and logged — HTTP 4xx/5xx (``HTTPError``) with its
status + error body, transport failures with their reason — and recorded
on ``LLMClient.last_error`` (thread-safe).  ``LLMConfig.fail_hard`` opts a
client into raising :class:`LLMClientError` on any failure instead of
returning ``""``, so callers that *require* LLM output (e.g. explicit
``use_llm=True`` rule translation) surface the real API error instead of
silently degrading.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .api_key import resolve_api_key

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 1.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

#: 控制字符清洗（C0 + DEL），原来是 L1/L3/L4 各写一份 —— 统一到此处。
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str, max_len: int | None = None) -> str:
    """Strip control characters and cap length (unified prompt-injection guard)."""
    cleaned = _CONTROL_CHARS_RE.sub("", str(text or ""))
    return cleaned[:max_len] if max_len is not None else cleaned


class LLMClientError(Exception):
    """LLM request/response failure.

    Raised by transport when ``LLMConfig.fail_hard`` is set (API 4xx/5xx,
    unreachable endpoint, timeout, malformed envelope), and by callers on
    empty/malformed LLM output.  With the default fail-soft config the
    transport never raises — failures surface as ``""`` plus
    ``LLMClient.last_error`` and a warning log.
    """


@dataclass
class LLMConfig:
    """One configuration shape for every LLM consumer in the project."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    temperature: float = 0.2
    max_tokens: int = 2048
    #: 严格模式：任何调用失败（API 4xx/5xx、端点不可达、超时、畸形响应）
    #: 抛 :class:`LLMClientError` 而不是返回空串。默认 False 保持 fail-soft。
    fail_hard: bool = False


@dataclass
class ToolCall:
    """One function-calling invocation returned by the model.

    ``id`` echoes the endpoint's ``tool_calls[i].id`` (fallback: ``""``).
    Multi-turn tool loops need it to pair the ``role: "tool"`` result
    message back with the assistant message that requested the call.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class ChatReply:
    """A chat completion with optional tool calls."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    """OpenAI-compatible chat client (local Ollama /v1, vLLM, DeepSeek, ...).

    Construction never fails (fail-soft): an unreachable or missing
    backend surfaces as ``""`` from :meth:`complete` /
    :meth:`complete_tools`, letting every caller keep its fallback path.
    Each failed call is logged with its cause and recorded on
    :attr:`last_error`; ``LLMConfig.fail_hard`` switches the transport to
    raise :class:`LLMClientError` instead.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fail_hard: bool | None = None,
    ) -> None:
        """Accept a :class:`LLMConfig` or individual overrides (drop-in for
        the removed per-layer clients that took flat kwargs)."""
        self.config = config or LLMConfig()
        if model is not None:
            self.config = LLMConfig(
                model=model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout_s=self.config.timeout_s,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                fail_hard=self.config.fail_hard,
            )
        overrides = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_s": timeout_s,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "fail_hard": fail_hard,
        }
        for name, value in overrides.items():
            if value is not None:
                setattr(self.config, name, value)
        # 统一密钥读取流程（audit 3.6）：显式参数 > LLM_API_KEY 环境变量 > 默认。
        self.api_key = resolve_api_key(self.config.api_key, "LLM_API_KEY", default="ollama")
        self.model = self.config.model
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout_s = self.config.timeout_s
        #: 最近一次调用的失败原因（无失败时为 None；线程安全、每次调用刷新）。
        self._last_error: Exception | None = None
        self._last_error_lock = threading.Lock()

    # ── Public surface ─────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the assistant reply text, or ``""`` on transport failure."""
        return self._chat(messages, max_tokens=max_tokens, temperature=temperature).text

    def complete_chat(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Convenience wrapper for ``[system, user]`` callers (Layer 4 dialogue)."""
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )

    def complete_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatReply:
        """Chat with function calling; returns text + parsed tool calls (fail-soft).

        Models/endpoints without tool support simply reply with text and
        an empty ``tool_calls`` list — the caller decides the fallback.

        ``messages`` may include full agentic-loop shapes (assistant
        messages carrying ``tool_calls`` and ``role: "tool"`` result
        messages), not just plain text turns.
        """
        return self._chat(messages, tools=tools, max_tokens=max_tokens, temperature=temperature)

    @staticmethod
    def available(base_url: str | None = None) -> bool:
        """Probe whether an LLM endpoint answers; never raises."""
        url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        req = urllib.request.Request(f"{url}/v1/models", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_S) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

    # ── Transport ──────────────────────────────────────────────────

    @property
    def last_error(self) -> Exception | None:
        """失败原因 of the most recent :meth:`_chat` call (``None`` on success)."""
        with self._last_error_lock:
            return self._last_error

    def _set_last_error(self, exc: Exception | None) -> None:
        with self._last_error_lock:
            self._last_error = exc

    def _fail(self, message: str, exc: Exception | None) -> ChatReply:
        """Record + log a transport failure; fail-soft or raise per config.

        ``last_error`` 存携带完整格式化信息（HTTP 状态码 + 错误体片段）的
        :class:`LLMClientError`，而不是裸 ``HTTPError``（其 ``str()`` 只有
        状态行）——调用方（``complete_with_retry``、chat、solver）据此拿到
        可用的失败原因。
        """
        recorded: Exception = LLMClientError(message)
        self._set_last_error(recorded)
        logger.warning("LLM 调用失败: %s (base_url=%s, model=%s)", message, self.base_url, self.model)
        if self.config.fail_hard:
            raise LLMClientError(message) from exc
        return ChatReply(text="", tool_calls=[])

    @staticmethod
    def _http_error_detail(exc: urllib.error.HTTPError) -> str:
        """Extract a short snippet of the API error body (best effort)."""
        try:
            snippet = exc.read(300).decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 — body 可能已被读取/非字节流，放弃提取
            return ""
        return f": {snippet}" if snippet else ""

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        # 失败分类（审查：LLM 兜底系统性排查）——API 4xx/5xx、端点不可达/
        # 超时、响应超限、畸形响应分别记录真实原因，而不是统一吞成空串。
        # ``HTTPError`` 是 ``URLError`` 的子类，必须先于它捕获。
        self._set_last_error(None)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            return self._fail(f"LLM API 错误 HTTP {exc.code} {exc.reason}{self._http_error_detail(exc)}", exc)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return self._fail(f"LLM 端点不可达/超时: {exc}", exc)
        if len(raw) > _MAX_RESPONSE_BYTES:
            return self._fail(f"LLM 响应超过 {_MAX_RESPONSE_BYTES} 字节上限", None)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._fail(f"LLM 响应不是有效 JSON: {exc}", exc)
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            return self._fail("LLM 响应缺少 choices[0].message 结构", exc)
        text = sanitize_text(str(message.get("content") or ""))
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            if not name:
                continue
            args_raw = str(fn.get("arguments") or "{}")
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {}
            if isinstance(args, dict):
                tool_calls.append(ToolCall(name=name, arguments=args, id=str(call.get("id") or "")))
        return ChatReply(text=text, tool_calls=tool_calls)


__all__ = [
    "ChatReply",
    "LLMClient",
    "LLMClientError",
    "LLMConfig",
    "ToolCall",
    "sanitize_text",
]
