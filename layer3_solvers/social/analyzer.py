"""Situation analysis and lightweight belief modeling for social agents."""

from __future__ import annotations

from .base import LanguageObservation
from .event_parser import SocialEventParser
from .state import PlayerModel, SocialAgentState, SocialBeliefState, SocialEvent, uniform_role_beliefs

_WOLF_MARKERS = ("wolf", "werewolf", "狼", "狼人", "卧底", "undercover", "traitor")
_GOOD_ROLES = {"villager", "seer", "witch", "hunter", "guard", "good"}


class SocialSituationAnalyzer:
    """Build richer social-agent state from a raw language observation."""

    def __init__(self, parser: SocialEventParser | None = None) -> None:
        self.parser = parser or SocialEventParser()

    def analyze(self, obs: LanguageObservation) -> SocialAgentState:
        """Return a structured state with beliefs and social relation scores."""
        events = self.parser.parse(obs)
        player_id = str(obs.private_info.get("id") or obs.public_context.get("player_id") or "")
        players = self._players(obs, events, player_id)
        roles = self._roles(obs)
        belief = SocialBeliefState(
            role_beliefs=uniform_role_beliefs(players, roles, player_id, obs.role),
            team_beliefs={},
            trust_matrix={p: {q: 0.0 for q in players if q != p} for p in players},
            suspicion_matrix={p: {q: 0.0 for q in players if q != p} for p in players},
            claim_table={p: [] for p in players},
            vote_intentions={p: {} for p in players},
            player_models={p: PlayerModel() for p in players},
        )
        self._apply_events(events, belief, players)
        self._derive_team_beliefs(belief)
        return SocialAgentState(
            player_id=player_id,
            role=obs.role,
            phase=obs.phase,
            round=int(obs.public_context.get("round", 0) or 0),
            alive_players=self._alive_players(obs, players),
            legal_targets=list(obs.legal_targets),
            private_info=obs.private_info,
            public_context=obs.public_context,
            events=events,
            belief=belief,
        )

    def _apply_events(self, events: list[SocialEvent], belief: SocialBeliefState, players: list[str]) -> None:
        last_vote: dict[str, str] = {}
        for event in events:
            if event.actor not in players:
                continue
            model = belief.player_models[event.actor]
            if event.event_type == "claim":
                model.claim_frequency += 1.0
                belief.claim_table.setdefault(event.actor, []).append(event)
                if event.claimed_role:
                    self._boost_role(belief, event.actor, event.claimed_role, amount=0.25)
            elif event.event_type == "accuse" and event.target in players:
                model.aggression += 1.0
                self._add_relation(belief.suspicion_matrix, event.actor, event.target, 0.35)
                self._boost_role(belief, event.target, "wolf", amount=0.15)
                belief.tactical_notes.append(f"{event.actor} accused {event.target}")
            elif event.event_type in {"defend", "support"} and event.target in players:
                self._add_relation(belief.trust_matrix, event.actor, event.target, 0.25)
                self._boost_role(belief, event.target, "wolf", amount=-0.08)
            elif event.event_type == "vote" and event.target in players:
                model.aggression += 0.5
                if last_vote.get(event.actor) and last_vote[event.actor] != event.target:
                    model.vote_stability = max(0.0, model.vote_stability - 0.2)
                last_vote[event.actor] = event.target
                belief.vote_intentions.setdefault(event.actor, {})[event.target] = 1.0
                self._add_relation(belief.suspicion_matrix, event.actor, event.target, 0.2)
                self._boost_role(belief, event.target, "wolf", amount=0.08)
            elif event.event_type == "death" and event.claimed_role:
                belief.role_beliefs[event.actor] = {event.claimed_role: 1.0}
        self._mark_claim_conflicts(belief)

    @staticmethod
    def _add_relation(matrix: dict[str, dict[str, float]], actor: str, target: str, amount: float) -> None:
        matrix.setdefault(actor, {})
        matrix[actor][target] = max(-1.0, min(1.0, matrix[actor].get(target, 0.0) + amount))

    @staticmethod
    def _boost_role(belief: SocialBeliefState, player: str, role: str, amount: float) -> None:
        post = belief.role_beliefs.get(player)
        if not post:
            return
        if role not in post:
            post[role] = 0.0
        post[role] = max(0.0, post[role] + amount)
        total = sum(post.values())
        if total <= 0:
            return
        for key in post:
            post[key] /= total

    @staticmethod
    def _mark_claim_conflicts(belief: SocialBeliefState) -> None:
        claims_by_role: dict[str, list[str]] = {}
        for player, claims in belief.claim_table.items():
            for claim in claims:
                if claim.claimed_role:
                    claims_by_role.setdefault(claim.claimed_role, []).append(player)
        for role, claimers in claims_by_role.items():
            if role in {"seer", "witch", "hunter", "guard"} and len(set(claimers)) > 1:
                for player in set(claimers):
                    belief.player_models[player].bluff_likelihood += 0.35
                    SocialSituationAnalyzer._boost_role(belief, player, "wolf", amount=0.12)
                    belief.tactical_notes.append(f"conflicting {role} claim by {player}")

    @staticmethod
    def _derive_team_beliefs(belief: SocialBeliefState) -> None:
        for player, post in belief.role_beliefs.items():
            wolf_prob = sum(prob for role, prob in post.items() if any(marker in role for marker in ("wolf", "狼")))
            good_prob = sum(prob for role, prob in post.items() if role in _GOOD_ROLES or role not in {"wolf"})
            total = wolf_prob + good_prob
            if total <= 0:
                belief.team_beliefs[player] = {"wolf": 0.0, "good": 0.0}
            else:
                belief.team_beliefs[player] = {"wolf": wolf_prob / total, "good": good_prob / total}

    @staticmethod
    def _roles(obs: LanguageObservation) -> list[str]:
        for key in ("role_pool", "roles"):
            value = obs.public_context.get(key) or obs.private_info.get(key)
            if isinstance(value, list):
                return [str(role) for role in value]
        return ["wolf", "villager", "seer", "witch", "hunter"] if _is_werewolf_like(obs.role) else [obs.role or "unknown"]

    @staticmethod
    def _players(obs: LanguageObservation, events: list[SocialEvent], self_id: str) -> list[str]:
        candidates = set(obs.legal_targets)
        if self_id:
            candidates.add(self_id)
        for key in ("players", "alive_players"):
            value = obs.public_context.get(key) or obs.private_info.get(key)
            if isinstance(value, list):
                candidates.update(str(player) for player in value)
        for event in events:
            if event.actor:
                candidates.add(event.actor)
            if event.target:
                candidates.add(event.target)
        return sorted(str(player) for player in candidates)

    @staticmethod
    def _alive_players(obs: LanguageObservation, players: list[str]) -> list[str]:
        alive = obs.public_context.get("alive_players")
        if isinstance(alive, list):
            return [str(player) for player in alive]
        alive_mask = obs.public_context.get("alive")
        if isinstance(alive_mask, list):
            return [f"p{i}" for i, value in enumerate(alive_mask) if value == 1]
        return players


def _is_werewolf_like(role: str) -> bool:
    normalized = str(role).casefold()
    return any(marker.casefold() in normalized for marker in _WOLF_MARKERS) or role in _GOOD_ROLES
