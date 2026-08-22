"""Belief-driven social policy with optional LLM speech realization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .analyzer import SocialSituationAnalyzer
from .base import LanguageObservation, LanguagePolicy
from .llm_policy import LLMClient
from .planner import HeuristicSocialPlanner
from .state import SocialAgentState, StrategyPlan

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
MAX_SPEECH_LEN = 200


@dataclass(frozen=True)
class BeliefDrivenConfig:
    """Config for belief-driven social policy."""

    use_llm_speech: bool = False
    max_speech_len: int = MAX_SPEECH_LEN


class BeliefDrivenPolicy(LanguagePolicy):
    """Social policy that separates analysis, planning, speech and voting."""

    def __init__(
        self,
        *,
        analyzer: SocialSituationAnalyzer | None = None,
        planner: HeuristicSocialPlanner | None = None,
        speech_client: LLMClient | None = None,
        config: BeliefDrivenConfig | None = None,
    ) -> None:
        self.analyzer = analyzer or SocialSituationAnalyzer()
        self.planner = planner or HeuristicSocialPlanner()
        self.speech_client = speech_client
        self.config = config or BeliefDrivenConfig(use_llm_speech=speech_client is not None)
        self.last_state: SocialAgentState | None = None
        self.last_plan: StrategyPlan | None = None

    def decide_speech(self, obs: LanguageObservation) -> str:
        """Analyze the situation, plan a speech act, then realize it."""
        state = self.analyzer.analyze(obs)
        plan = self.planner.plan(state)
        self.last_state = state
        self.last_plan = plan
        if self.config.use_llm_speech and self.speech_client is not None:
            return self._llm_speech(state, plan)
        return self._template_speech(state, plan)

    def decide_vote(self, obs: LanguageObservation) -> str:
        """Vote according to the current strategy plan, not raw LLM text."""
        state = self.analyzer.analyze(obs)
        plan = self.planner.plan(state)
        self.last_state = state
        self.last_plan = plan
        if plan.vote_target in obs.legal_targets:
            return str(plan.vote_target)
        return next(
            (target for target in obs.legal_targets if target != state.player_id),
            obs.legal_targets[0] if obs.legal_targets else "",
        )

    def _llm_speech(self, state: SocialAgentState, plan: StrategyPlan) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个社会推理游戏玩家。你只负责把给定策略计划表达成自然发言，"
                    "不要改变目标，不要泄露未要求泄露的私有信息。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "player_id": state.player_id,
                        "role": state.role,
                        "phase": state.phase,
                        "round": state.round,
                        "plan": plan.__dict__,
                        "memory_summary": state.belief.memory_summary,
                        "tactical_notes": state.belief.tactical_notes[-5:],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        text = self.speech_client.complete(messages, max_tokens=120) if self.speech_client is not None else ""
        return self._sanitize(text) or self._template_speech(state, plan)

    def _template_speech(self, state: SocialAgentState, plan: StrategyPlan) -> str:
        target = plan.target
        if plan.speech_act == "stay_low":
            return "我先低调听一轮，大家把自己的判断说清楚。"
        if plan.speech_act in {"claim", "fake_claim"} and plan.claim:
            if target:
                return f"我是{self._role_label(plan.claim)}，我会重点观察{target}。"
            return f"我是{self._role_label(plan.claim)}，目前先保留更多判断。"
        if plan.speech_act == "accuse" and target:
            evidence = "，".join(plan.evidence[:2]) if plan.evidence else "他的发言和投票不太自然"
            return f"我怀疑{target}，理由是{evidence}。这一轮可以重点票他。"
        if plan.speech_act == "defend" and target:
            return f"我暂时相信{target}，他目前的行为不像最优先出的对象。"
        if plan.speech_act == "support" and target:
            return f"我倾向支持{target}的判断，但还要看后续投票。"
        if plan.speech_act == "deflect" and target:
            return f"现在焦点不该只放在我身上，{target}的逻辑更需要解释。"
        if target:
            return f"我想听{target}解释一下上一轮的发言和投票。"
        return "我先听一轮，暂时不急着下结论。"

    def _sanitize(self, text: str) -> str:
        return _CONTROL_CHARS_RE.sub("", str(text or ""))[: self.config.max_speech_len].strip()

    @staticmethod
    def _role_label(role: str) -> str:
        labels = {
            "wolf": "普通村民",
            "villager": "村民",
            "seer": "预言家",
            "witch": "女巫",
            "hunter": "猎人",
            "guard": "守卫",
        }
        return labels.get(role, role)
