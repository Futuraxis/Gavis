"""Structured state objects for social-deduction policies.

The lightweight ``LanguageObservation`` protocol is still the external
entrypoint.  These dataclasses are the richer internal representation used
by belief-driven and trainable social agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SpeechAct = Literal[
    "claim",
    "fake_claim",
    "accuse",
    "defend",
    "question",
    "support",
    "deflect",
    "stay_low",
]


@dataclass(frozen=True)
class SocialEvent:
    """One normalized public/private social-game event."""

    round: int
    phase: str
    actor: str
    event_type: str
    target: str | None = None
    text: str = ""
    intent: str | None = None
    claimed_role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerModel:
    """Simple behavioral profile used by opponent modeling."""

    aggression: float = 0.0
    consistency: float = 1.0
    follow_rate: float = 0.0
    claim_frequency: float = 0.0
    bluff_likelihood: float = 0.0
    vote_stability: float = 1.0


@dataclass
class SocialBeliefState:
    """Beliefs and social relations inferred from public history."""

    role_beliefs: dict[str, dict[str, float]] = field(default_factory=dict)
    team_beliefs: dict[str, dict[str, float]] = field(default_factory=dict)
    trust_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    suspicion_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    claim_table: dict[str, list[SocialEvent]] = field(default_factory=dict)
    vote_intentions: dict[str, dict[str, float]] = field(default_factory=dict)
    player_models: dict[str, PlayerModel] = field(default_factory=dict)
    memory_summary: str = ""
    tactical_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyPlan:
    """Intermediate strategic intent before natural-language realization."""

    speech_act: SpeechAct
    target: str | None
    vote_target: str | None
    stance: Literal["low", "medium", "strong"] = "medium"
    claim: str | None = None
    evidence: list[str] = field(default_factory=list)
    desired_effect: str = ""
    risk_level: float = 0.0


@dataclass
class SocialAgentState:
    """Richer social-agent state derived from one observation."""

    player_id: str
    role: str
    phase: str
    round: int
    alive_players: list[str]
    legal_targets: list[str]
    private_info: dict[str, Any]
    public_context: dict[str, Any]
    events: list[SocialEvent]
    belief: SocialBeliefState


def uniform_role_beliefs(
    players: list[str], roles: list[str], self_id: str, self_role: str
) -> dict[str, dict[str, float]]:
    """Create a conservative role prior for all players."""
    unique_roles = sorted(set(roles or [self_role or "unknown"]))
    if not unique_roles:
        unique_roles = ["unknown"]
    prior = 1.0 / len(unique_roles)
    beliefs: dict[str, dict[str, float]] = {}
    for player in players:
        if player == self_id and self_role:
            beliefs[player] = {self_role: 1.0}
        else:
            beliefs[player] = {role: prior for role in unique_roles}
    return beliefs
