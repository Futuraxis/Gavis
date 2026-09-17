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
import os
from typing import Protocol

from layer2_engine.core.llm import LLMClient as _UnifiedLLMClient

LLMClient = _UnifiedLLMClient  # 统一客户端别名（Layer 1 消费者经此引用）

logger = logging.getLogger(__name__)

#: 规则翻译的 LLM 采样温度 —— 必须 0（确定性/可复现：同一规则文本跨次产出
#: 相同 rules.json，"它就是能用" 的复现性前提；闲聊/发言等创作场景不受影响）。
RULE_LLM_TEMPERATURE = 0.0

#: 规则翻译的输出预算（tokens）。8192 对**推理模型**是不够的：思维链也计入
#: ``max_tokens``，实测 deepseek-flash 一次规则翻译就能把 8192 全烧在
#: reasoning 上、正文为空（``finish_reason=length``），表现为「LLM 不可用
#: （未返回内容）」→ 创建游戏失败。32768 留出思考 + 完整 rules.json 的空间。
#: 端点拒绝该预算时统一客户端会自动降档到 8192 重试（老模型上限）。
#: 环境变量 ``LLM_MAX_TOKENS`` 可覆盖（无需改代码即可按模型调）。
RULE_LLM_MAX_TOKENS = 32768

#: 规则翻译的传输超时（秒）。默认 30s 是聊天尺度：推理模型生成一份完整
#: rules.json 常需 1-3 分钟，30s 必然超时 → 用户等一分钟只拿到「端点不可达/
#: 超时」。翻译路径单独用长超时。环境变量 ``LLM_TIMEOUT_S`` 可覆盖。
RULE_LLM_TIMEOUT_S = 300.0

#: 一次规则翻译的**总**预算（秒，含「校验失败 → 修复重试」的第二次调用）。
#: 单次超时管不住总时长：修复重试会再来一次 300s，端点抽风时用户可能等上
#: 十分钟（实测有一次 486s）。有了总预算，第二次调用只拿到剩余时间，超了就
#: 停止重试、改走确定性模板（平台给出降级告警）。环境变量
#: ``LLM_CREATE_DEADLINE_S`` 可覆盖。
RULE_LLM_DEADLINE_S = 300.0

#: 预算/超时的合法区间（防御坏环境变量把翻译打瘸）。
_MIN_MAX_TOKENS = 1024
_MAX_MAX_TOKENS = 262_144
_MIN_TIMEOUT_S = 5.0
_MAX_TIMEOUT_S = 3600.0
_MIN_DEADLINE_S = 30.0
_MAX_DEADLINE_S = 3600.0


def _env_float(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r 不是数字，忽略（用默认 %s）", name, raw, default)
        return default
    return min(high, max(low, value))


def rule_llm_max_tokens() -> int:
    """规则翻译的输出预算（``LLM_MAX_TOKENS`` 环境变量可覆盖）。"""
    return int(_env_float("LLM_MAX_TOKENS", float(RULE_LLM_MAX_TOKENS), _MIN_MAX_TOKENS, _MAX_MAX_TOKENS))


def rule_llm_timeout_s() -> float:
    """规则翻译的传输超时秒数（``LLM_TIMEOUT_S`` 环境变量可覆盖）。"""
    return _env_float("LLM_TIMEOUT_S", RULE_LLM_TIMEOUT_S, _MIN_TIMEOUT_S, _MAX_TIMEOUT_S)


def rule_llm_deadline_s() -> float:
    """一次规则翻译的总时间预算（``LLM_CREATE_DEADLINE_S`` 环境变量可覆盖）。"""
    return _env_float("LLM_CREATE_DEADLINE_S", RULE_LLM_DEADLINE_S, _MIN_DEADLINE_S, _MAX_DEADLINE_S)


def build_rule_llm_client(
    *,
    model: str | None = None,
    temperature: float = RULE_LLM_TEMPERATURE,
    fail_hard: bool = False,
    timeout_s: float | None = None,
) -> LLMClient:
    """构造规则翻译用的统一客户端（temperature=0 + 长超时）。

    超时必须在**构造时**给出：聊天尺度默认 30s 会让规则翻译必然超时。
    ``timeout_s`` 显式给出时优先（修复重试只拿剩余预算，见
    :func:`rule_llm_deadline_s`）。
    """
    return LLMClient(
        model=model,
        temperature=temperature,
        fail_hard=fail_hard,
        timeout_s=timeout_s if timeout_s is not None else rule_llm_timeout_s(),
    )


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


__all__ = [
    "LLMClient",
    "LLMTranslatorError",
    "RULE_LLM_DEADLINE_S",
    "RULE_LLM_MAX_TOKENS",
    "RULE_LLM_TEMPERATURE",
    "RULE_LLM_TIMEOUT_S",
    "RuleLLMClient",
    "build_rule_llm_client",
    "complete_with_retry",
    "rule_llm_deadline_s",
    "rule_llm_max_tokens",
    "rule_llm_timeout_s",
]
