"""Social/language games — policy hooks and belief-driven agents.

Entry point for language-driven games (Werewolf, Undercover, poker
banter): a policy decides what to SAY and whom to VOTE for, backed by
an LLM client, a template fallback, or structured social reasoning.
"""

from __future__ import annotations

from .analyzer import SocialSituationAnalyzer
from .base import LanguageObservation, LanguagePolicy
from .belief_policy import BeliefDrivenConfig, BeliefDrivenPolicy
from .event_parser import SocialEventParser
from .llm_policy import LLMClient, LLMPolicy, OpenAICompatibleClient
from .planner import HeuristicSocialPlanner
from .state import PlayerModel, SocialAgentState, SocialBeliefState, SocialEvent, StrategyPlan
from .template_policy import TemplatePolicy

__all__ = [
    "LanguageObservation",
    "LanguagePolicy",
    "SocialEvent",
    "PlayerModel",
    "SocialBeliefState",
    "StrategyPlan",
    "SocialAgentState",
    "SocialEventParser",
    "SocialSituationAnalyzer",
    "HeuristicSocialPlanner",
    "BeliefDrivenConfig",
    "BeliefDrivenPolicy",
    "LLMClient",
    "LLMPolicy",
    "OpenAICompatibleClient",
    "TemplatePolicy",
]
