"""Tests for the language-game policy hooks (LLM + template fallback)."""

from __future__ import annotations

from layer3_solvers.social import (
    BeliefDrivenPolicy,
    LanguageObservation,
    LLMPolicy,
    SocialEventParser,
    SocialSituationAnalyzer,
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


class TestSocialEventParser:
    def test_parses_accuse_claim_and_vote_events(self):
        obs = _obs(
            history=[
                {"speaker": "p1", "intent": "claim", "text": "我是预言家", "round": 1},
                {"speaker": "p2", "text": "我怀疑 p3 是狼", "round": 1},
                {"voter": "p4", "target": "p3", "round": 1},
            ],
        )

        events = SocialEventParser().parse(obs)

        assert [event.event_type for event in events] == ["claim", "accuse", "vote"]
        assert events[0].claimed_role == "seer"
        assert events[1].target == "p3"
        assert events[2].actor == "p4"


class TestSocialSituationAnalyzer:
    def test_analyzer_builds_belief_and_suspicion_state(self):
        obs = _obs(
            role="villager",
            private_info={"id": "p0"},
            public_context={
                "round": 2,
                "players": ["p0", "p1", "p2", "p3"],
                "role_pool": ["wolf", "villager", "seer", "witch"],
            },
            history=[
                {"speaker": "p1", "text": "我怀疑 p3 是狼", "round": 1},
                {"voter": "p2", "target": "p3", "round": 1},
            ],
            legal_targets=["p1", "p2", "p3"],
        )

        state = SocialSituationAnalyzer().analyze(obs)

        assert state.player_id == "p0"
        assert state.round == 2
        assert state.belief.suspicion_matrix["p1"]["p3"] > 0
        assert state.belief.team_beliefs["p3"]["wolf"] > state.belief.team_beliefs["p1"]["wolf"]


class TestBeliefDrivenPolicy:
    def test_vote_uses_belief_target(self):
        obs = _obs(
            role="villager",
            private_info={"id": "p0"},
            public_context={"players": ["p0", "p1", "p2", "p3"]},
            history=[
                {"speaker": "p1", "text": "我怀疑 p3 是狼", "round": 1},
                {"speaker": "p2", "text": "p3 是狼", "round": 1},
            ],
            legal_targets=["p1", "p2", "p3"],
        )

        policy = BeliefDrivenPolicy()

        assert policy.decide_vote(obs) == "p3"
        assert policy.last_plan is not None
        assert policy.last_plan.vote_target == "p3"

    def test_speech_realizes_strategy_plan_without_llm(self):
        obs = _obs(
            role="villager",
            private_info={"id": "p0"},
            public_context={"players": ["p0", "p1", "p2", "p3"]},
            history=[{"speaker": "p1", "text": "我怀疑 p3 是狼", "round": 1}],
            legal_targets=["p1", "p2", "p3"],
        )

        text = BeliefDrivenPolicy().decide_speech(obs)

        assert "p3" in text
        assert "怀疑" in text

    def test_llm_speech_receives_structured_plan(self):
        client = _FakeClient("我认为 p3 的投票很可疑。")
        policy = BeliefDrivenPolicy(speech_client=client)
        text = policy.decide_speech(
            _obs(
                role="villager",
                private_info={"id": "p0"},
                history=[{"speaker": "p1", "text": "我怀疑 p3 是狼", "round": 1}],
                legal_targets=["p1", "p2", "p3"],
            )
        )

        assert text == "我认为 p3 的投票很可疑。"
        assert client.calls
        assert "plan" in client.calls[0][1]["content"]
