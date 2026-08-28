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
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .api_key import resolve_api_key

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
    """Raised by callers on empty/malformed LLM output; transport stays fail-soft."""


@dataclass
class LLMConfig:
    """One configuration shape for every LLM consumer in the project."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S
    temperature: float = 0.2
    max_tokens: int = 2048


@dataclass
class ToolCall:
    """One function-calling invocation returned by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


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
            )
        overrides = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_s": timeout_s,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for name, value in overrides.items():
            if value is not None:
                setattr(self.config, name, value)
        # 统一密钥读取流程（audit 3.6）：显式参数 > LLM_API_KEY 环境变量 > 默认。
        self.api_key = resolve_api_key(self.config.api_key, "LLM_API_KEY", default="ollama")
        self.model = self.config.model
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout_s = self.config.timeout_s

    # ── Public surface ─────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatReply:
        """Chat with function calling; returns text + parsed tool calls (fail-soft).

        Models/endpoints without tool support simply reply with text and
        an empty ``tool_calls`` list — the caller decides the fallback.
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

    def _chat(
        self,
        messages: list[dict[str, str]],
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
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return ChatReply(text="", tool_calls=[])
                body = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return ChatReply(text="", tool_calls=[])
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return ChatReply(text="", tool_calls=[])
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
                tool_calls.append(ToolCall(name=name, arguments=args))
        return ChatReply(text=text, tool_calls=tool_calls)


__all__ = [
    "ChatReply",
    "LLMClient",
    "LLMClientError",
    "LLMConfig",
    "ToolCall",
    "sanitize_text",
]