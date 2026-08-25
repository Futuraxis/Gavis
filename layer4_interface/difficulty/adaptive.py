"""Adaptive difficulty — win-rate driven search-budget selection (Layer 4).

``AdaptiveController`` is a deterministic, stateless helper that floats the
AI search budget between the tiers declared in ``platform/games.py``
(``GameSpec.difficulty_budgets``) based on the player's recent win rate.
It never imports Layer 3: budgets are plain integers handed to
``SolverProvider.create_solver``.

Pacing presets map the player's preferred think-time ceiling to a budget
scale factor (``PACING`` / ``pacing_scale``); actual time capping is
enforced by the integration phase, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

#: 自适应模式缺省锚定档位（与 profile 的 ``default_difficulty`` 一致）。
_DEFAULT_TIER = "normal"


@dataclass(frozen=True)
class PacingSpec:
    """节奏预设：AI 思考上限与对应的预算缩放系数。

    Attributes:
        max_seconds: AI 思考时间上限（秒）。
        budget_scale: 搜索预算缩放系数（近似控制搜索规模，非真实计时）。
    """

    max_seconds: float
    budget_scale: float


#: 节奏 → 思考上限 / 预算缩放：快棋 ≤1s（0.2x）、标准 ≤5s（1.0x）、慢棋 ≤15s（3.0x）。
PACING: dict[str, PacingSpec] = {
    "fast": PacingSpec(max_seconds=1.0, budget_scale=0.2),
    "standard": PacingSpec(max_seconds=5.0, budget_scale=1.0),
    "slow": PacingSpec(max_seconds=15.0, budget_scale=3.0),
}


def pacing_scale(pacing: str) -> float:
    """Return the budget scale factor for ``pacing`` (fast/standard/slow)."""
    return _pacing(pacing).budget_scale


def pacing_seconds(pacing: str) -> float:
    """Return the AI think-time ceiling (seconds) for ``pacing``."""
    return _pacing(pacing).max_seconds


def _pacing(pacing: str) -> PacingSpec:
    """Resolve a pacing preset; raise ``ValueError`` on unknown names."""
    try:
        return PACING[pacing]
    except KeyError:
        raise ValueError(f"未知节奏: {pacing!r}") from None


@lru_cache(maxsize=1)
def _budgets_map() -> dict[str, dict[str, int]]:
    """Difficulty budgets, loaded lazily from the platform game registry.

    Mirrors ``GameSpec.difficulty_budgets`` in
    ``layer4_interface/frontend/platform/games.py`` (the single source of
    truth) and must stay in sync with it.
    """
    from layer4_interface.frontend.platform.games import GAMES

    return {spec.game_id: dict(spec.difficulty_budgets) for spec in GAMES.values()}


def _tiers(game_id: str) -> dict[str, int]:
    """Return the ``{tier: budget}`` map for ``game_id`` (raise on unknown)."""
    budgets = _budgets_map().get(game_id)
    if budgets is None:
        raise ValueError(f"未知游戏: {game_id}")
    return budgets


def _has_strength_knob(tiers: dict[str, int]) -> bool:
    """True when tiers differ in budget (i.e. strength is tunable)."""
    return len(set(tiers.values())) > 1


def _ordered_tiers(tiers: dict[str, int]) -> list[str]:
    """Tiers ordered by budget ascending (easy < normal < hard)."""
    return sorted(tiers, key=tiers.get)


def _tier_budget(tiers: dict[str, int], tier: str) -> int:
    """Return the budget of an explicit tier (raise on unknown)."""
    if tier not in tiers:
        raise ValueError(f"未知难度档位: {tier!r}")
    return tiers[tier]


def _anchor_tier(recent: list[dict], ordered: list[str]) -> str:
    """Most recent concrete tier in ``recent``, defaulting to ``normal``.

    ``recent`` is oldest-first, so it is scanned backwards for the last
    entry whose ``difficulty`` is an explicit tier (easy/normal/hard).
    """
    valid = set(ordered)
    for entry in reversed(recent):
        tier = entry.get("difficulty")
        if tier in valid:
            return tier
    return _DEFAULT_TIER if _DEFAULT_TIER in valid else ordered[0]


def _win_rate(recent: list[dict], window: int) -> float | None:
    """Player win rate over the most recent ``window`` matches.

    ``recent`` is oldest-first, so the window is its last ``window``
    entries.  A match counts as a win when ``winner == player_pid``
    (draws/unknown winners count against).  Returns ``None`` when empty.
    """
    matches = recent[-window:]
    if not matches:
        return None
    wins = 0
    for entry in matches:
        winner = entry.get("winner")
        player_pid = entry.get("player_pid")
        if winner is not None and player_pid is not None and winner == player_pid:
            wins += 1
    return wins / len(matches)


class AdaptiveController:
    """Win-rate → budget controller (stateless; deterministic)."""

    def __init__(self, *, target_lo: float = 0.40, target_hi: float = 0.60, window: int = 10) -> None:
        self.target_lo = target_lo
        self.target_hi = target_hi
        self.window = window

    def pick_budget(self, game_id: str, difficulty: str, recent: list[dict]) -> int:
        """Return the search budget for the next game.

        ``recent`` is the player's recent matches, oldest-first, as
        ``[{"winner": str | None, "player_pid": str, "difficulty": str}, ...]``.

        Rules:
            - ``difficulty`` is an explicit tier (easy/normal/hard) → the
              player locked that tier; return its original budget unchanged.
            - ``difficulty == "adaptive"`` → win rate below ``target_lo``
              drops one tier (easier), above ``target_hi`` raises one tier
              (harder), inside the band keeps the anchor tier; clamped to
              min/max.  Anchor = the most recent concrete tier in ``recent``
              (default ``normal``); empty ``recent`` returns the anchor.
            - Games whose tiers share one budget (mahjong heuristics) have
              no strength knob and always return that budget.
        """
        tiers = _tiers(game_id)
        if not _has_strength_knob(tiers):
            return next(iter(tiers.values()))
        if difficulty != "adaptive":
            return _tier_budget(tiers, difficulty)
        ordered = _ordered_tiers(tiers)
        anchor = _anchor_tier(recent, ordered)
        index = ordered.index(anchor)
        win_rate = _win_rate(recent, self.window)
        if win_rate is None:
            return tiers[anchor]
        if win_rate < self.target_lo:
            index -= 1
        elif win_rate > self.target_hi:
            index += 1
        index = max(0, min(index, len(ordered) - 1))
        return tiers[ordered[index]]

    def strength_explain(self, game_id: str, old_budget: int, new_budget: int) -> str:
        """Return a one-line Chinese explanation of a strength change.

        Note: the win-rate percentage is not available at this call site
        (the frozen signature carries only budgets), so the wording reports
        the budget move; callers wanting the exact win rate should format it
        from ``recent`` themselves.
        """
        if not _has_strength_knob(_tiers(game_id)):
            return "麻将为展示档位，强度暂不自动调整"
        if new_budget < old_budget:
            return f"AI 强度从 {old_budget} 降到 {new_budget}，先让你找回手感"
        if new_budget > old_budget:
            return f"AI 强度从 {old_budget} 提高到 {new_budget}，给你来点挑战"
        return "AI 强度保持"
