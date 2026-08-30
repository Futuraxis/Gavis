"""Layer 1 LLM client surface — unified via ``layer2_engine.core.llm``.

Layer 1 keeps only the protocol it injects into translators (so tests can
pass deterministic fakes) plus the translator-level error type.  The
concrete transport is the project's single LLM client
(``layer2_engine.core.llm.LLMClient``) — the previously separate
``LocalTransformersRuleClient`` (torch/transformers local model) and
``OpenAICompatibleRuleClient`` copies were removed in the LLM unification;
both are covered by the same OpenAI-compatible endpoint.
"""

from __future__ import annotations

import logging
from typing import Protocol

from layer2_engine.core.llm import LLMClient as _UnifiedLLMClient

LLMClient = _UnifiedLLMClient  # 统一客户端别名（Layer 1 消费者经此引用）

logger = logging.getLogger(__name__)

#: 规则翻译的 LLM 采样温度 —— 必须 0（确定性/可复现：同一规则文本跨次产出
#: 相同 rules.json，"它就是能用" 的复现性前提；闲聊/发言等创作场景不受影响）。
RULE_LLM_TEMPERATURE = 0.0


class RuleLLMClient(Protocol):
    """Minimal chat-completion surface for rule translation (injection point).

    ``layer2_engine.core.llm.LLMClient`` satisfies this protocol; tests use
    deterministic in-memory fakes with the same shape.
    """

    def complete(self, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
        """Return the assistant's reply text for ``messages``."""


class LLMTranslatorError(Exception):
    """LLM translation failed before a valid candidate could be validated."""


def complete_with_retry(
    client: RuleLLMClient,
    messages: list[dict[str, str]],
    max_tokens: int = 8192,
    retries: int = 1,
) -> tuple[str, Exception | None]:
    """P2-23 修复：传输失败/空回复先立即重试一次，再让调用方兜底。

    冷启动 Ollama 的典型形态是首次调用超时或空回复（模型仍在加载），
    立即重试往往即可用；只有重试仍失败才回退确定性路径。修复前传输
    失败/冷启动不重试（只有"校验失败"进修复循环），网络抖动即触发兜底。

    Returns:
        ``(raw, error)``：成功时 ``(reply, None)``；持久传输异常时
        ``("", 最后一次异常)``；持久空回复时 ``("", 客户端 last_error)``
        （统一客户端 fail-soft 时异常不抛出，真实原因记录在
        ``client.last_error``，调用方据此定性"LLM 不可用"而非笼统报空）。
    """
    error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            raw = client.complete(messages, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — 传输异常统一进入重试/兜底
            if attempt < retries:
                logger.warning("LLM 传输失败（%s），立即重试", exc)
            error = exc
            continue
        if raw:
            return raw, None
        # fail-soft 客户端不抛异常：取它记录的真实失败原因（HTTP 4xx/5xx、
        # 端点不可达等），None 表示"确实无错误信息"。
        error = getattr(client, "last_error", None) or None
    return "", error


__all__ = ["LLMClient", "LLMTranslatorError", "RULE_LLM_TEMPERATURE", "RuleLLMClient", "complete_with_retry"]
