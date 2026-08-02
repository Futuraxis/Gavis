"""Template language policy — rule-based fallback when no LLM is wired.

Keeps social games playable offline: speeches are assembled from the
agent's role and recent transcript; votes follow a simple heuristic
(different role → vote the most suspicious recent speaker).
"""

from __future__ import annotations

import random
from typing import Optional

from .base import LanguageObservation, LanguagePolicy


class TemplatePolicy(LanguagePolicy):
    """Deterministic-ish fallback policy (needs no network)."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def decide_speech(self, obs: LanguageObservation) -> str:
        role = obs.role
        recent = obs.history[-3:]
        if not recent:
            return f'我是{role}，希望大家先都介绍一下自己的身份，我再判断。'
        if role == '狼人' or role == '卧底':
            return f'我是普通村民。我觉得{recent[-1].get("speaker", "有人")}发言有点可疑，想听他再多说两句。'
        return f'我暂时没有太多信息，但我会认真观察每个人的发言。'

    def decide_vote(self, obs: LanguageObservation) -> str:
        targets = obs.legal_targets
        if not targets:
            return ''
        # Naive heuristic: vote the last speaker who isn't obviously us.
        for entry in reversed(obs.history):
            speaker = entry.get('speaker', '')
            if speaker in targets and speaker != obs.private_info.get('id'):
                return speaker
        return self._rng.choice(targets)
