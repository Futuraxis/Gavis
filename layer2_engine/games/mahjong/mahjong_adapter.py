"""Mahjong — thin GameEngine adapter (v5.1, one JSON for all variants).

``rules/mahjong.json`` is a single rule set; variant and player count are
injected into ``constants`` here:
  - ``variant``: guangdong (鸡胡) / hongzhong (红中万能) / blood (血战到底)
  - ``player_count``: 2 or 4 → ``player_ids``, ``deal_target`` (13N+1),
    ``players`` and the utility table are trimmed to N seats

The adapter also adds a structured ``get_observation`` (hand / melds /
discards / wall / legal actions) for AI and UI consumers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ...core.engine import GameEngine

VARIANTS = ("guangdong", "hongzhong", "blood")
TILE_NAMES = {
    "m": "万",
    "p": "筒",
    "s": "条",
    "z": "字",
}


class MahjongAdapter(GameEngine):
    """GameEngine subclass for mahjong (variant × player count)."""

    def __init__(self, variant: str = "guangdong", player_count: int = 2, seed: Optional[int] = None):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown mahjong variant: {variant!r}")
        if player_count not in (2, 4):
            raise ValueError(f"player_count must be 2 or 4, got {player_count}")
        self.variant = variant
        self.player_count = player_count
        rules_path = Path(__file__).resolve().parent.parent.parent.parent / "rules" / "mahjong.json"
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        rules["constants"] = dict(rules["constants"])
        rules["constants"]["variant"] = variant
        rules["constants"]["player_count"] = player_count
        rules["constants"]["player_ids"] = [f"p{i}" for i in range(player_count)]
        rules["constants"]["deal_target"] = 13 * player_count + 1
        rules["players"] = rules["constants"]["player_ids"]
        rules["utility"] = [u for u in rules["utility"] if u["player"] in rules["constants"]["player_ids"]]
        super().__init__(rules, seed=seed)

    # ── Current player ────────────────────────────────────────────────

    def get_current_player(self, state: dict) -> Optional[str]:
        """During a claim the actor is the queue head, not ``env.turn``
        (which still names the discarder)."""
        env = state.get("env", {})
        if env.get("phase") == "claim":
            queue = env.get("claim_queue") or []
            idx = int(env.get("claim_index", 0))
            if 0 <= idx < len(queue):
                return queue[idx]
            return None
        return super().get_current_player(state)

    # ── Structured observation ───────────────────────────────────────

    def get_observation(self, state: dict, player_id: str) -> dict:
        """Hand / melds / discards / wall / legal actions for ``player_id``."""
        arrs = state.get("_arrays", {})
        env = state.get("env", {})
        pid = player_id

        legal = []
        if not self.is_terminal(state) and env.get("turn") == player_id:
            for action in self.get_legal_actions(state):
                legal.append(
                    {
                        "type": action.template_id,
                        "params": action.params,
                        "key": action.canonical_key,
                    }
                )

        return {
            "hand": list(arrs.get(f"hand_{pid}", [])),
            "melds": list(arrs.get(f"melds_{pid}", [])),
            "discards": list(arrs.get(f"discard_{pid}", [])),
            "wall_count": int(env.get("wall_count", 0)),
            "last_discard": env.get("last_discard"),
            "last_drawn": env.get("last_drawn"),
            "phase": env.get("phase"),
            "turn": env.get("turn"),
            "my_turn": env.get("turn") == player_id and env.get("phase") in ("action", "claim"),
            "done": list(env.get("done", [])),
            "winners": list(env.get("winners", [])),
            "payoffs": list(env.get("payoffs", [])),
            "legal": legal,
        }

    # ── Display helpers ──────────────────────────────────────────────

    @staticmethod
    def tile_name(tile: str) -> str:
        """Chinese tile label, e.g. 'm3' → '三万', 'z5' → '红中'."""
        if not tile or len(tile) < 2:
            return str(tile)
        suit, rank = tile[0], tile[1:]
        if suit == "z":
            names = {"1": "东", "2": "南", "3": "西", "4": "北", "5": "中", "6": "发", "7": "白"}
            return names.get(rank, tile)
        return f"{rank}{TILE_NAMES.get(suit, suit)}"
