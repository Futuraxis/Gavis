"""Normalize language-game histories into structured social events."""

from __future__ import annotations

import re
from typing import Any

from .base import LanguageObservation
from .state import SocialEvent

_ROLE_MARKERS = {
    "wolf": ("狼", "狼人", "wolf", "werewolf", "undercover", "卧底"),
    "seer": ("预言家", "seer", "查验", "验了", "验人"),
    "witch": ("女巫", "witch", "解药", "毒药"),
    "hunter": ("猎人", "hunter", "开枪"),
    "guard": ("守卫", "guard", "守护"),
    "villager": ("村民", "平民", "好人", "villager", "good"),
}
_ACCUSE_MARKERS = ("怀疑", "可疑", "是狼", "投", "票", "accuse", "suspect")
_DEFEND_MARKERS = ("相信", "保", "不像狼", "好人", "defend", "trust")
_QUESTION_MARKERS = ("为什么", "解释", "说清楚", "question", "?")


class SocialEventParser:
    """Parse raw history dictionaries into ``SocialEvent`` objects."""

    def parse(self, obs: LanguageObservation) -> list[SocialEvent]:
        """Parse all known history entries in an observation."""
        players = self._players(obs)
        events: list[SocialEvent] = []
        for entry in obs.history:
            events.extend(self._parse_entry(entry, obs.phase, players))
        for death in self._deaths(obs):
            events.append(death)
        return events

    def _parse_entry(self, entry: dict[str, Any], default_phase: str, players: list[str]) -> list[SocialEvent]:
        round_no = int(entry.get("round", 0) or 0)
        phase = str(entry.get("phase") or default_phase)
        if "voter" in entry or entry.get("event_type") == "vote":
            return [
                SocialEvent(
                    round=round_no,
                    phase=phase,
                    actor=str(entry.get("voter") or entry.get("actor") or ""),
                    event_type="vote",
                    target=str(entry.get("target") or "") or None,
                    metadata=dict(entry),
                )
            ]

        actor = str(entry.get("speaker") or entry.get("actor") or "")
        text = str(entry.get("text") or "")
        intent = str(entry.get("intent") or "") or None
        target = self.find_target(text, players, exclude=actor)
        claimed_role = self.find_claimed_role(text)
        event_type = self._classify(text, intent, target, claimed_role)
        return [
            SocialEvent(
                round=round_no,
                phase=phase,
                actor=actor,
                event_type=event_type,
                target=target,
                text=text,
                intent=intent,
                claimed_role=claimed_role,
                metadata=dict(entry),
            )
        ]

    def _deaths(self, obs: LanguageObservation) -> list[SocialEvent]:
        dead_roles = obs.public_context.get("dead_roles", {})
        if not isinstance(dead_roles, dict):
            return []
        round_no = int(obs.public_context.get("round", 0) or 0)
        return [
            SocialEvent(
                round=round_no,
                phase=obs.phase,
                actor=str(player),
                event_type="death",
                claimed_role=str(role) if role else None,
                metadata={"role": role},
            )
            for player, role in dead_roles.items()
        ]

    @staticmethod
    def _classify(text: str, intent: str | None, target: str | None, claimed_role: str | None) -> str:
        normalized = text.casefold()
        if intent in {"accuse", "defend", "question", "claim", "support"}:
            return intent
        if claimed_role is not None and ("我是" in text or "i am" in normalized or "claim" in normalized):
            return "claim"
        if target is not None and any(marker in normalized for marker in _ACCUSE_MARKERS):
            return "accuse"
        if target is not None and any(marker in normalized for marker in _DEFEND_MARKERS):
            return "defend"
        if any(marker in normalized for marker in _QUESTION_MARKERS):
            return "question"
        return "speak"

    @staticmethod
    def find_claimed_role(text: str) -> str | None:
        normalized = text.casefold()
        for role, markers in _ROLE_MARKERS.items():
            if any(marker.casefold() in normalized for marker in markers):
                return role
        return None

    @staticmethod
    def find_target(text: str, players: list[str], exclude: str = "") -> str | None:
        tokens = [token.casefold() for token in re.findall(r"[A-Za-z0-9_]+", text)]
        for player in players:
            if player != exclude and player.casefold() in tokens:
                return player
        return None

    @staticmethod
    def _players(obs: LanguageObservation) -> list[str]:
        candidates: list[str] = []
        for key in ("players", "alive_players"):
            value = obs.public_context.get(key) or obs.private_info.get(key)
            if isinstance(value, list):
                candidates.extend(str(player) for player in value)
        candidates.extend(str(target) for target in obs.legal_targets)
        if obs.private_info.get("id"):
            candidates.append(str(obs.private_info["id"]))
        for entry in obs.history:
            for key in ("speaker", "actor", "voter", "target"):
                if entry.get(key):
                    candidates.append(str(entry[key]))
        return sorted(set(candidates))
