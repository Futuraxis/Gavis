"""Tests for the language-game policy hooks (LLM + template fallback)."""

from __future__ import annotations

from layer3_solvers.social import (
    LanguageObservation,
    LLMPolicy,
    TemplatePolicy,
)


class _FakeClient:
    def __init__(self, reply: str = "我是狼人，我觉得他很可疑。"):
        self._reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], max_tokens: int = 200) -> str:
        self.calls.append(messages)
        return self._reply


def _obs(**kwargs) -> LanguageObservation:
    base = dict(
        role="狼人",
        phase="speech",
        public_context={"round": 1},
        private_info={"id": "p2"},
        history=[{"speaker": "p1", "text": "我是村民", "round": 1}],
        legal_targets=["p1", "p2", "p3"],
    )
    base.update(kwargs)
    return LanguageObservation(**base)


class TestTemplatePolicy:
    def test_speech_not_empty(self):
        policy = TemplatePolicy()
        text = policy.decide_speech(_obs())
        assert isinstance(text, str) and len(text) > 0

    def test_vote_targets_legal(self):
        policy = TemplatePolicy()
        vote = policy.decide_vote(_obs(phase="vote"))
        assert vote in {"p1", "p2", "p3"}

    def test_vote_empty_without_targets(self):
        policy = TemplatePolicy()
        vote = policy.decide_vote(_obs(phase="vote", legal_targets=[]))
        assert vote == ""


class TestLLMPolicy:
    def test_speech_uses_client_reply(self):
        client = _FakeClient()
        policy = LLMPolicy(client)
        text = policy.decide_speech(_obs())
        assert text == client._reply

    def test_prompt_contains_role_and_history(self):
        client = _FakeClient()
        policy = LLMPolicy(client)
        policy.decide_speech(_obs())
        assert client.calls, "client must be called"
        messages = client.calls[0]
        assert messages[0]["role"] == "system"
        joined = messages[1]["content"]
        assert "狼人" in joined
        assert "p1" in joined
        assert "发言" in joined

    def test_vote_instruction_lists_targets(self):
        client = _FakeClient()
        policy = LLMPolicy(client)
        policy.decide_vote(_obs(phase="vote"))
        assert "p1" in client.calls[0][1]["content"]
