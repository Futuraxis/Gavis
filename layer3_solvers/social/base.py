"""Language-game policy abstractions.

Social deduction games (Werewolf, Undercover, ...) and real-card
scenarios share a structure the standard SolverAdapter cannot express:
the "actions" are natural-language utterances and votes over players,
driven by private roles and public speech history.  These protocols
define the policy surface; concrete policies (LLM-backed, template,
hybrid) live alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LanguageObservation:
    """What a language-game policy sees at one decision point.

    ``history`` entries are ``{'speaker': str, 'text': str, 'round': int}``
    (or vote records); ``private_info`` holds the agent's own role/hand.
    """

    role: str
    phase: str                       # 'speech' | 'vote' | 'reveal' | ...
    public_context: dict[str, Any] = field(default_factory=dict)
    private_info: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    legal_targets: list[str] = field(default_factory=list)   # votable players


class LanguagePolicy(Protocol):
    """Policy for language/social games."""

    def decide_speech(self, obs: LanguageObservation) -> str:
        """Return the agent's utterance for the current speech round."""

    def decide_vote(self, obs: LanguageObservation) -> str:
        """Return the target player id this agent votes for."""
