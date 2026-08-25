"""Tests for AdaptiveController — win-rate driven budget selection."""

from __future__ import annotations

import pytest

from layer4_interface.difficulty import PACING, AdaptiveController, pacing_scale, pacing_seconds


def _game(winner: str | None, player_pid: str = "p_black", difficulty: str = "normal") -> dict:
    """One recent-match entry for the human player ``player_pid``."""
    return {"winner": winner, "player_pid": player_pid, "difficulty": difficulty}


@pytest.fixture
def controller() -> AdaptiveController:
    return AdaptiveController()


class TestAdaptiveFloat:
    def test_low_win_rate_drops_a_tier(self, controller: AdaptiveController) -> None:
        recent = [_game("p_white") for _ in range(10)]  # 10 losses
        assert controller.pick_budget("moon_chess", "adaptive", recent) == 200

    def test_high_win_rate_raises_a_tier(self, controller: AdaptiveController) -> None:
        recent = [_game("p_black") for _ in range(10)]  # 10 wins
        assert controller.pick_budget("moon_chess", "adaptive", recent) == 2000

    def test_in_range_keeps_tier(self, controller: AdaptiveController) -> None:
        recent = [_game("p_black" if i < 5 else "p_white") for i in range(10)]  # 50%
        assert controller.pick_budget("moon_chess", "adaptive", recent) == 800

    def test_empty_recent_returns_anchor_tier(self, controller: AdaptiveController) -> None:
        assert controller.pick_budget("moon_chess", "adaptive", []) == 800

    def test_clamps_at_min_and_max(self, controller: AdaptiveController) -> None:
        bottom = [_game("p_white", difficulty="easy") for _ in range(10)]
        assert controller.pick_budget("moon_chess", "adaptive", bottom) == 200
        top = [_game("p_black", difficulty="hard") for _ in range(10)]
        assert controller.pick_budget("moon_chess", "adaptive", top) == 2000

    def test_anchors_from_most_recent_concrete_tier(self, controller: AdaptiveController) -> None:
        # 早期 easy、最近 hard，全胜 → 锚定最近档 hard 并 clamp 在 hard。
        recent = [_game("p_black", difficulty="easy") for _ in range(3)] + [
            _game("p_black", difficulty="hard") for _ in range(7)
        ]
        assert controller.pick_budget("moon_chess", "adaptive", recent) == 2000


class TestLockedDifficulty:
    def test_explicit_tier_ignores_recent(self, controller: AdaptiveController) -> None:
        losses = [_game("p_white") for _ in range(10)]
        wins = [_game("p_black") for _ in range(10)]
        assert controller.pick_budget("moon_chess", "normal", losses) == 800
        assert controller.pick_budget("moon_chess", "normal", wins) == 800
        assert controller.pick_budget("moon_chess", "normal", []) == 800

    def test_each_explicit_tier(self, controller: AdaptiveController) -> None:
        assert controller.pick_budget("moon_chess", "easy", []) == 200
        assert controller.pick_budget("moon_chess", "hard", []) == 2000
        assert controller.pick_budget("texas_holdem", "normal", []) == 500


class TestMahjongDisplayOnly:
    def test_mahjong_adaptive_returns_one(self, controller: AdaptiveController) -> None:
        recent = [_game("p_black") for _ in range(10)]
        assert controller.pick_budget("mahjong_guangdong", "adaptive", recent) == 1

    def test_mahjong_explicit_returns_one(self, controller: AdaptiveController) -> None:
        assert controller.pick_budget("mahjong_hongzhong", "hard", []) == 1


class TestStrengthExplain:
    def test_down(self, controller: AdaptiveController) -> None:
        assert "降到" in controller.strength_explain("moon_chess", 800, 200)

    def test_up(self, controller: AdaptiveController) -> None:
        assert "提高" in controller.strength_explain("moon_chess", 800, 2000)

    def test_hold(self, controller: AdaptiveController) -> None:
        assert controller.strength_explain("moon_chess", 800, 800) == "AI 强度保持"

    def test_mahjong(self, controller: AdaptiveController) -> None:
        assert "麻将" in controller.strength_explain("mahjong_guangdong", 1, 1)


class TestPacing:
    def test_scale(self) -> None:
        assert pacing_scale("fast") == 0.2
        assert pacing_scale("standard") == 1.0
        assert pacing_scale("slow") == 3.0

    def test_seconds(self) -> None:
        assert pacing_seconds("fast") == 1.0
        assert pacing_seconds("standard") == 5.0
        assert pacing_seconds("slow") == 15.0

    def test_unknown_pacing_raises(self) -> None:
        with pytest.raises(ValueError):
            pacing_scale("turbo")

    def test_pacing_constant(self) -> None:
        assert set(PACING) == {"fast", "standard", "slow"}
