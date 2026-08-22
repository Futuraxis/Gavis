"""Strategy planning for belief-driven social policies."""

from __future__ import annotations

from .state import SocialAgentState, StrategyPlan

_WOLF_MARKERS = ("wolf", "werewolf", "狼", "狼人", "卧底", "undercover", "traitor")
_SEER_MARKERS = ("seer", "预言家")


class HeuristicSocialPlanner:
    """Convert analyzed social state into an abstract strategy plan."""

    def plan(self, state: SocialAgentState) -> StrategyPlan:
        """Select a phase-aware social plan."""
        target = self._primary_target(state)
        vote_target = target if target in state.legal_targets else None
        if self._is_wolf(state.role):
            return self._wolf_plan(state, target, vote_target)
        if self._is_seer(state.role):
            return self._seer_plan(state, target, vote_target)
        return self._good_plan(state, target, vote_target)

    def _wolf_plan(self, state: SocialAgentState, target: str | None, vote_target: str | None) -> StrategyPlan:
        if target is None:
            return StrategyPlan(
                speech_act="stay_low",
                target=None,
                vote_target=vote_target,
                stance="low",
                desired_effect="avoid exposing wolf alignment",
                risk_level=0.2,
            )
        return StrategyPlan(
            speech_act="accuse",
            target=target,
            vote_target=vote_target,
            stance="medium",
            claim="villager",
            evidence=[f"{target} 的发言/投票有可疑点"],
            desired_effect=f"raise suspicion on {target} while appearing villager",
            risk_level=0.55,
        )

    def _seer_plan(self, state: SocialAgentState, target: str | None, vote_target: str | None) -> StrategyPlan:
        checked = state.private_info.get("seer_result") or state.public_context.get("seer_result")
        if checked and checked in state.alive_players:
            return StrategyPlan(
                speech_act="accuse",
                target=str(checked),
                vote_target=str(checked) if checked in state.legal_targets else vote_target,
                stance="strong",
                claim="seer",
                evidence=[f"night check result points to {checked}"],
                desired_effect=f"coordinate votes onto {checked}",
                risk_level=0.65,
            )
        return StrategyPlan(
            speech_act="claim",
            target=target,
            vote_target=vote_target,
            stance="medium",
            claim="seer",
            evidence=["需要保留查验信息并观察对跳"],
            desired_effect="build credibility without overexposing detailed information",
            risk_level=0.5,
        )

    def _good_plan(self, state: SocialAgentState, target: str | None, vote_target: str | None) -> StrategyPlan:
        suspicion = self._suspicion_score(state, target) if target else 0.0
        if target is None or suspicion < 0.25:
            return StrategyPlan(
                speech_act="question",
                target=target,
                vote_target=vote_target,
                stance="low",
                evidence=["当前证据不足"],
                desired_effect="elicit more information before committing",
                risk_level=0.2,
            )
        return StrategyPlan(
            speech_act="accuse",
            target=target,
            vote_target=vote_target,
            stance="strong" if suspicion >= 0.5 else "medium",
            evidence=[f"{target} 被多次怀疑或投票"],
            desired_effect=f"test and pressure {target}",
            risk_level=0.35,
        )

    def _primary_target(self, state: SocialAgentState) -> str | None:
        candidates = [target for target in state.legal_targets if target != state.player_id]
        if not candidates:
            candidates = [player for player in state.alive_players if player != state.player_id]
        if not candidates:
            return None
        if self._is_wolf(state.role):
            return min(candidates, key=lambda player: self._wolf_prob(state, player))
        return max(candidates, key=lambda player: self._suspicion_score(state, player))

    def _suspicion_score(self, state: SocialAgentState, player: str | None) -> float:
        if not player:
            return 0.0
        incoming = sum(row.get(player, 0.0) for row in state.belief.suspicion_matrix.values())
        model = state.belief.player_models.get(player)
        bluff = model.bluff_likelihood if model is not None else 0.0
        return self._wolf_prob(state, player) + incoming * 0.2 + bluff * 0.2

    @staticmethod
    def _wolf_prob(state: SocialAgentState, player: str) -> float:
        team = state.belief.team_beliefs.get(player, {})
        if "wolf" in team:
            return team["wolf"]
        return state.belief.role_beliefs.get(player, {}).get("wolf", 0.0)

    @staticmethod
    def _is_wolf(role: str) -> bool:
        normalized = str(role).casefold()
        return any(marker.casefold() in normalized for marker in _WOLF_MARKERS)

    @staticmethod
    def _is_seer(role: str) -> bool:
        normalized = str(role).casefold()
        return any(marker.casefold() in normalized for marker in _SEER_MARKERS)
