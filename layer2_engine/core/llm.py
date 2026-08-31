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

Streaming + chain-of-thought (2026 流式/思维链改造): :meth:`complete_stream`
is the SSE transport — payload ``stream: true``, parsed line by line from
the OpenAI-compatible endpoint.  It yields :class:`StreamChunk` deltas
(``text`` / ``reasoning`` / ``tool_calls`` fragments / ``done``), failing
fail-soft like the rest of the client (an error chunk with ``done=True``
terminates the iteration, never a hang).  Reasoning is extracted
defensively from whichever shape the endpoint uses — ``reasoning_content``
or ``reasoning`` in the message/delta — and Ollama-legacy paired
``<think>…</think>`` spans inside ``content`` are split off into the
reasoning channel (strictly paired tags, safe on ordinary text).  The
non-stream path mirrors the same extraction into ``ChatReply.reasoning``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

from .api_key import resolve_api_key

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 1.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _env_or(name: str) -> str | None:
    """Strip-whitespace env read; unset/blank yields ``None`` (falls through)."""
    value = os.environ.get(name, "").strip()
    return value or None


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
    """One configuration shape for every LLM consumer in the project.

    ``model`` / ``base_url`` default to ``None`` — resolution happens in
    :class:`LLMClient` with the unified precedence
    **显式配置 > 环境变量 (``LLM_BASE_URL`` / ``LLM_MODEL``) > 内置默认**
    (``DEFAULT_BASE_URL`` / ``DEFAULT_MODEL``), matching the api-key flow
    (``LLM_API_KEY``).  Platform-persisted settings layer on top of that
    by passing explicit values here.
    """

    model: str | None = None
    base_url: str | None = None
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
    """A chat completion with optional tool calls and chain-of-thought.

    ``reasoning`` is the model's thinking output (``reasoning_content`` /
    ``reasoning`` field, or the inner text of legacy ``<think>…</think>``
    spans).  Empty when the endpoint/model doesn't emit reasoning — all
    existing consumers are unaffected.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class StreamChunk:
    """One delta from :meth:`LLMClient.complete_stream` (SSE transport).

    ``text`` / ``reasoning`` are *increments* — concatenate them to get the
    full reply.  ``done`` marks stream end; on a transport failure
    ``done=True`` with a non-empty ``error`` (fail-soft, matches the
    ``complete`` contract — callers check ``last_error`` for the cause).
    ``tool_calls`` is only populated on the terminal chunk when the model
    ended with ``finish_reason: "tool_calls"`` (fragments are accumulated
    internally across chunks and parsed at stream end).
    """

    text: str = ""
    reasoning: str = ""
    done: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str = ""


#: Ollama / llama.cpp legacy 思考标签（严格配对才剥离，防止误伤普通文本；
#: 新端点一般走 ``reasoning_content`` 字段，此处是内容包装形态的兜底）。
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
#: ``思考标签`` 的部分前缀（跨 chunk 分片时需把尾部挂起等待补全）。
_THINK_TAIL_PREFIXES = (
    "<",
    "<t",
    "<th",
    "<thi",
    "<thin",
    "<think",
    "</",
    "</t",
    "</th",
    "</thi",
    "</thin",
    "</think",
)


class _ThinkTagSplitter:
    """把 ``<think>…</think>`` 夹层从内容流里剥出来（支持跨 chunk 分片）.

    状态机按完整标签切分：open 之前的文本 → ``(text, "")``，夹层内文本 →
    ``("", reasoning)``。分片结尾是不完整标签前缀时把该尾段挂起，等下一块
    补全；普通文本不含标签时零开销（整段原样回 text）。``flush`` 在流结束
    时把残留缓冲吐出。
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """处理一块增量，返回 ``[(text, reasoning)]`` 切分结果。"""
        if not delta:
            return []
        self._buf += delta
        out: list[tuple[str, str]] = []
        while True:
            if self._in_think:
                idx = self._buf.find(_THINK_CLOSE)
                if idx >= 0:
                    out.append(("", self._buf[:idx]))
                    self._buf = self._buf[idx + len(_THINK_CLOSE) :]
                    if self._buf.startswith(">"):  # 对称剥掉 ``</think>`` 形态的收尾 ``>``
                        self._buf = self._buf[1:]
                    self._in_think = False
                    continue
                hold = _hold_tag_tail(self._buf)
                if hold:
                    tail, self._buf = self._buf[-hold:], self._buf[:-hold]
                    if self._buf:
                        out.append(("", self._buf))
                    self._buf = tail
                elif self._buf:
                    out.append(("", self._buf))
                    self._buf = ""
                break
            idx = self._buf.find(_THINK_OPEN)
            if idx >= 0:
                if idx > 0:
                    out.append((self._buf[:idx], ""))
                self._buf = self._buf[idx + len(_THINK_OPEN) :]
                if self._buf.startswith(">"):  # 剥掉 ``<think>`` 形态的收尾 ``>``
                    self._buf = self._buf[1:]
                self._in_think = True
                continue
            # 不配对 stray close（不在 think 态）→ 当作普通文本的一截，跳过标签。
            stray = self._buf.find(_THINK_CLOSE)
            if stray >= 0:
                self._buf = self._buf[:stray] + self._buf[stray + len(_THINK_CLOSE) :]
                continue
            hold = _hold_tag_tail(self._buf)
            if hold:
                tail, self._buf = self._buf[-hold:], self._buf[:-hold]
                if self._buf:
                    out.append((self._buf, ""))
                self._buf = tail
            elif self._buf:
                out.append((self._buf, ""))
                self._buf = ""
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        """流结束时吐出残留缓冲（未闭合标签按字面文本处理）。"""
        if not self._buf:
            return []
        parts = [(self._buf, "")] if not self._in_think else [("", self._buf)]
        self._buf = ""
        return parts


def _hold_tag_tail(buf: str) -> int:
    """返回需要挂起等待补全的尾部字节数（不完整标签前缀），否则 0."""
    for prefix in _THINK_TAIL_PREFIXES:
        if buf.endswith(prefix) and len(prefix) < len(_THINK_OPEN):
            return len(prefix)
        if buf.endswith(prefix) and len(prefix) < len(_THINK_CLOSE):
            return len(prefix)
    return 0


def _extract_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    """从一条非流式 ``message`` 对象取出 ``(text, reasoning)``。

    reasoning 双来源：``reasoning_content`` / ``reasoning`` 字段（DeepSeek/
    Qwen 思维模型形态），以及 content 里 Ollama / llama.cpp legacy 的
    ``<think>``…``</think>`` 夹层（严格配对才剥离）。普通文本模型两种来源
    都为空 → 零开销原样返回。
    """
    content = str(message.get("content") or "")
    direct = message.get("reasoning_content")
    if direct is None:
        direct = message.get("reasoning")
    reasoning = str(direct or "")
    splitter = _ThinkTagSplitter()
    text_parts: list[str] = []
    reasoning_parts: list[str] = [reasoning] if reasoning else []
    for text, r_text in splitter.feed(content):
        if text:
            text_parts.append(text)
        if r_text:
            reasoning_parts.append(r_text)
    for text, r_text in splitter.flush():
        if text:
            text_parts.append(text)
        if r_text:
            reasoning_parts.append(r_text)
    return "".join(text_parts), "".join(reasoning_parts)


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
        # 端点/模型同优先级：显式配置 > LLM_BASE_URL / LLM_MODEL 环境变量 > 内置默认。
        self.model = self.config.model or _env_or("LLM_MODEL") or DEFAULT_MODEL
        self.base_url = (self.config.base_url or _env_or("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
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

    def complete_chat(
        self, system: str, user: str, max_tokens: int | None = None, temperature: float | None = None
    ) -> str:
        """Convenience wrapper for ``[system, user]`` callers (Layer 4 dialogue).

        ``temperature`` overrides the client default when given (e.g. pacing
        presets for social games: fast=发散, slow=精准)。
        """
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_chat_reply(self, system: str, user: str, max_tokens: int | None = None) -> ChatReply:
        """``complete_chat`` 的 :class:`ChatReply` 版本（带 reasoning，供成文侧透传思维链）。"""
        return self._chat(
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

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamChunk]:
        """Streaming chat (SSE) — yields :class:`StreamChunk` deltas.

        Same OpenAI-compatible endpoint as :meth:`_chat` with
        ``stream: true``.  ``text`` / ``reasoning`` are *increments*
        (concatenate for the full reply); tool-call fragments are
        accumulated internally and the terminal chunk (``done=True``)
        carries the parsed :class:`ToolCall` list when the model ended
        with ``finish_reason: "tool_calls"``.

        Fail-soft contract matches the rest of the client: transport /
        parse failures yield one final chunk with ``done=True`` and
        ``error`` set (cause also on ``last_error``) — the iteration
        always terminates and never raises, unless ``fail_hard`` is set
        (raises :class:`LLMClientError` at the failure point).

        Reasoning deltas are taken from ``reasoning_content`` /
        ``reasoning`` in the delta, and Ollama-legacy paired
        ``<think…</think`` spans inside ``content`` are split into
        the reasoning channel across chunk boundaries.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
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
        self._set_last_error(None)
        splitter = _ThinkTagSplitter()
        tool_frags: dict[int, dict[str, str]] = {}
        total = 0
        pending_text = ""
        pending_reasoning = ""
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    total += len(line)
                    if total > _MAX_RESPONSE_BYTES:
                        yield self._fail_stream(f"LLM 流式响应超过 {_MAX_RESPONSE_BYTES} 字节上限", None)
                        return
                    text_line = line.decode("utf-8", "replace").strip()
                    if not text_line or text_line.startswith(":"):
                        continue
                    if not text_line.startswith("data:"):
                        continue
                    data_raw = text_line[len("data:") :].strip()
                    if data_raw == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_raw)
                    except (ValueError, TypeError) as exc:
                        yield self._fail_stream(f"LLM 流式响应不是有效 JSON: {exc}", exc)
                        return
                    finished = False
                    choices = obj.get("choices") if isinstance(obj.get("choices"), list) else []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta") or choice.get("message") or {}
                        if isinstance(delta, dict):
                            pending_text, pending_reasoning = self._absorb_delta(
                                delta, splitter, tool_frags, pending_text, pending_reasoning
                            )
                        if choice.get("finish_reason") in ("stop", "tool_calls"):
                            finished = True
                    if obj.get("done") is True:
                        finished = True
                    if finished:
                        break
                    if pending_text or pending_reasoning:
                        yield StreamChunk(text=pending_text, reasoning=pending_reasoning)
                        pending_text = ""
                        pending_reasoning = ""
        except urllib.error.HTTPError as exc:
            yield self._fail_stream(f"LLM API 错误 HTTP {exc.code} {exc.reason}{self._http_error_detail(exc)}", exc)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            yield self._fail_stream(f"LLM 端点不可达/超时: {exc}", exc)
            return
        # 终态（正常/异常统一收尾）：残留 splitter 缓冲 + 工具分片解析。
        tail_text: list[str] = []
        tail_reasoning: list[str] = []
        for text, r_text in splitter.flush():
            if text:
                tail_text.append(text)
            if r_text:
                tail_reasoning.append(r_text)
        if pending_text:
            tail_text.append(pending_text)
        if pending_reasoning:
            tail_reasoning.append(pending_reasoning)
        yield StreamChunk(
            text="".join(tail_text),
            reasoning="".join(tail_reasoning),
            done=True,
            tool_calls=self._finalize_tool_calls(tool_frags),
        )

    def _absorb_delta(
        self,
        delta: dict[str, Any],
        splitter: _ThinkTagSplitter,
        tool_frags: dict[int, dict[str, str]],
        pending_text: str,
        pending_reasoning: str,
    ) -> tuple[str, str]:
        """吸收一块 ``delta``：content 经 splitter 切分，reasoning 直取，工具分片累积."""
        content = delta.get("content")
        if isinstance(content, str) and content:
            for text, r_text in splitter.feed(content):
                if text:
                    pending_text += sanitize_text(text)
                if r_text:
                    pending_reasoning += sanitize_text(r_text)
        direct = delta.get("reasoning_content")
        if direct is None:
            direct = delta.get("reasoning")
        if isinstance(direct, str) and direct:
            pending_reasoning += sanitize_text(direct)
        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                index = call.get("index")
                try:
                    index = int(index) if index is not None else len(tool_frags)
                except (TypeError, ValueError):
                    index = len(tool_frags)
                frag = tool_frags.setdefault(index, {"name": "", "arguments": "", "id": ""})
                fn = call.get("function") or {}
                if isinstance(fn, dict):
                    name = fn.get("name")
                    if isinstance(name, str) and name:
                        frag["name"] += name
                    args = fn.get("arguments")
                    if isinstance(args, str) and args:
                        frag["arguments"] += args
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    frag["id"] = call_id
        return pending_text, pending_reasoning

    @staticmethod
    def _finalize_tool_calls(tool_frags: dict[int, dict[str, str]]) -> list[ToolCall]:
        """把按 ``index`` 累积的工具分片解析成 :class:`ToolCall` 列表."""
        out: list[ToolCall] = []
        for index in sorted(tool_frags):
            frag = tool_frags[index]
            name = str(frag.get("name") or "")
            if not name:
                continue
            args_raw = str(frag.get("arguments") or "")
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            out.append(ToolCall(name=name, arguments=args, id=str(frag.get("id") or "")))
        return out

    @staticmethod
    def available(base_url: str | None = None, api_key: str = "", timeout_s: float | None = None) -> bool:
        """Probe whether an LLM endpoint answers; never raises.

        ``base_url`` falls back to ``LLM_BASE_URL`` env then
        ``DEFAULT_BASE_URL``; ``api_key`` falls back to ``LLM_API_KEY`` env
        when not explicitly passed, so the no-arg ``available()`` call used
        by the social-family solver probe authenticates against a configured
        cloud endpoint (DeepSeek / GLM / OpenAI) — otherwise ``/v1/models``
        401s and the session silently degrades to the random solver, which
        is exactly "AI 不跟着平台配置走 API".
        """
        url = (base_url or _env_or("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        key = api_key or _env_or("LLM_API_KEY") or ""
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{url}/v1/models", method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or _PROBE_TIMEOUT_S) as resp:
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

    def _record_failure(self, message: str) -> None:
        """记录 + 日志一次 LLM 失败（fail-soft 与 fail_hard 共用）。"""
        recorded: Exception = LLMClientError(message)
        self._set_last_error(recorded)
        logger.warning("LLM 调用失败: %s (base_url=%s, model=%s)", message, self.base_url, self.model)

    def _fail(self, message: str, exc: Exception | None) -> ChatReply:
        """Record + log a transport failure; fail-soft or raise per config.

        ``last_error`` 存携带完整格式化信息（HTTP 状态码 + 错误体片段）的
        :class:`LLMClientError`，而不是裸 ``HTTPError``（其 ``str()`` 只有
        状态行）——调用方（``complete_with_retry``、chat、solver）据此拿到
        可用的失败原因。
        """
        self._record_failure(message)
        if self.config.fail_hard:
            raise LLMClientError(message) from exc
        return ChatReply(text="", tool_calls=[])

    def _fail_stream(self, message: str, exc: Exception | None) -> StreamChunk:
        """流式失败收尾块（fail-soft：``done=True`` + ``error``；``fail_hard`` 抛错）。"""
        self._record_failure(message)
        if self.config.fail_hard:
            raise LLMClientError(message) from exc
        return StreamChunk(done=True, error=message)

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
        text, reasoning = _extract_reasoning(message)
        text = sanitize_text(text)
        reasoning = sanitize_text(reasoning)
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
        return ChatReply(text=text, tool_calls=tool_calls, reasoning=reasoning)


__all__ = [
    "ChatReply",
    "LLMClient",
    "LLMClientError",
    "LLMConfig",
    "StreamChunk",
    "ToolCall",
    "sanitize_text",
]
