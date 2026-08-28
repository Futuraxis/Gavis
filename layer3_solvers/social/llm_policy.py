"""LLM-backed language policy — the pluggable API hook.

``LLMPolicy`` turns a :class:`LanguageObservation` into a prompt and
asks an LLM client for the utterance/vote.  The client is injected so
any OpenAI-compatible endpoint works (Qwen, DeepSeek, local vLLM, ...);
the default concrete implementation is the project's unified LLM client
(``layer2_engine.core.llm.LLMClient``) — the former local
``OpenAICompatibleClient`` copy was removed in the LLM unification.

No LLM available?  ``TemplatePolicy`` (template_policy.py) keeps the
game playable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from layer2_engine.core.llm import LLMClient as _UnifiedLLMClient, sanitize_text

from .base import LanguageObservation

#: 发言清洗（审计 3.6 prompt 注入）：长度上限与控制字符剔除（统一清洗）。
MAX_SPEECH_LEN = 200


class LLMClient(Protocol):
    """Minimal chat-completion surface (injection point for fakes)."""

    def complete(self, messages: list[dict], max_tokens: int = 200) -> str:
        """Return the assistant's reply text for ``messages``."""


# 兼容别名：旧 ``OpenAICompatibleClient`` 实现已并入统一客户端。
OpenAICompatibleClient = _UnifiedLLMClient


class LLMError(Exception):
    """LLM request/response failure."""


class LLMPolicy:
    """Language policy backed by an LLM client.

    Prompts are assembled per phase ('speech' | 'vote'); private info,
    public context, and the transcript are included so the model can
    bluff, deduce, and argue — the point of social-deduction play.
    """

    def __init__(self, client: LLMClient, system_prompt: str | None = None):
        self._client = client
        self._system_prompt = system_prompt or (
            "你是一个社会推理游戏玩家。基于你的身份和场上信息，用中文给出符合你身份目标的发言或投票，简洁有力。"
        )

    def decide_speech(self, obs: LanguageObservation) -> str:
        messages = self._build_messages(obs, instruction="发言（直接给出你要说的话，不要解释）")
        return self._complete(messages)

    def decide_vote(self, obs: LanguageObservation) -> str:
        targets = "、".join(obs.legal_targets) if obs.legal_targets else "（可投票对象）"
        instruction = f"投票（从以下对象中选择一个：{targets}，只输出对象名）"
        messages = self._build_messages(obs, instruction=instruction)
        return self._complete(messages)

    def _build_messages(self, obs: LanguageObservation, instruction: str) -> list[dict]:
        transcript = "\n".join(
            f"[第{h.get('round', '?')}轮] {h.get('speaker', '?')}: {h.get('text', '')}" for h in obs.history[-12:]
        )
        context = {
            "phase": obs.phase,
            "role": obs.role,
            "private_info": obs.private_info,
            "public_context": obs.public_context,
            "transcript": transcript,
        }
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"{json.dumps(context, ensure_ascii=False)}\n\n{instruction}"},
        ]

    def _complete(self, messages: list[dict]) -> str:
        text = self._client.complete(messages)
        text = sanitize_text(text, MAX_SPEECH_LEN).strip()
        # 空结果返回 "" — LanguagePolicy 空值契约："" = 沉默/弃权
        return text