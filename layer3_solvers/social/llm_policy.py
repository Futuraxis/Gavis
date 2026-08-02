"""LLM-backed language policy — the pluggable API hook.

``LLMPolicy`` turns a :class:`LanguageObservation` into a prompt and
asks an LLM client for the utterance/vote.  The client is injected so
any OpenAI-compatible endpoint works (Qwen, DeepSeek, local vLLM, ...);
``OpenAICompatibleClient`` is the default HTTP implementation.

No LLM available?  ``TemplatePolicy`` (template_policy.py) keeps the
game playable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .base import LanguageObservation


class LLMClient(Protocol):
    """Minimal chat-completion surface."""

    def complete(self, messages: list[dict], max_tokens: int = 200) -> str:
        """Return the assistant's reply text for ``messages``."""


@dataclass
class OpenAICompatibleClient:
    """OpenAI-compatible ``/chat/completions`` HTTP client.

    Config via env vars: ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``
    (defaults target a local endpoint so it fails loudly, not silently).
    """

    base_url: str = 'http://127.0.0.1:11434/v1'   # ollama-style default
    api_key: str = 'ollama'
    model: str = 'qwen2.5:7b'
    timeout_s: float = 30.0

    def complete(self, messages: list[dict], max_tokens: int = 200) -> str:
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': 0.8,
        }
        req = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f'LLM request failed: {exc}') from exc
        try:
            return body['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f'Unexpected LLM response: {body!r}') from exc


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
            '你是一个社会推理游戏玩家。基于你的身份和场上信息，'
            '用中文给出符合你身份目标的发言或投票，简洁有力。'
        )

    def decide_speech(self, obs: LanguageObservation) -> str:
        messages = self._build_messages(obs, instruction='发言（直接给出你要说的话，不要解释）')
        return self._complete(messages)

    def decide_vote(self, obs: LanguageObservation) -> str:
        targets = '、'.join(obs.legal_targets) if obs.legal_targets else '（可投票对象）'
        instruction = f'投票（从以下对象中选择一个：{targets}，只输出对象名）'
        messages = self._build_messages(obs, instruction=instruction)
        return self._complete(messages)

    def _build_messages(self, obs: LanguageObservation, instruction: str) -> list[dict]:
        transcript = '\n'.join(
            f"[第{h.get('round', '?')}轮] {h.get('speaker', '?')}: {h.get('text', '')}"
            for h in obs.history[-12:]
        )
        context = {
            'phase': obs.phase,
            'role': obs.role,
            'private_info': obs.private_info,
            'public_context': obs.public_context,
            'transcript': transcript,
        }
        return [
            {'role': 'system', 'content': self._system_prompt},
            {'role': 'user', 'content': f'{json.dumps(context, ensure_ascii=False)}\n\n{instruction}'},
        ]

    def _complete(self, messages: list[dict]) -> str:
        text = self._client.complete(messages)
        return text.strip() or '（沉默）'
