"""Smoke tests for the standalone Texas Hold'em play app (M-1 regression).

M-1: ``play_texas_holdem.session`` referenced a nonexistent
``layer2_engine.core.poker_utils`` module, so the whole app failed to
import.  These tests pin the import path and the basic start/move flow
against the real engine (seats, hand/payoff lookups via the adapter).
"""

from __future__ import annotations

import pytest

from demos.solver_provider import default_provider
from layer4_interface.frontend.play_texas_holdem.session import PlayError, PlayManager


@pytest.fixture
def manager() -> PlayManager:
    return PlayManager(provider=default_provider, seed=42, max_sessions=16)


class TestTexasPlaySession:
    def test_import_and_start(self, manager: PlayManager):
        session = manager.start("p_sb", "easy")
        assert session.over is False
        snap = session.snapshot()
        assert snap["player_pid"] == "p_sb"
        assert snap["ai_pid"] == "p_bb"
        assert snap["pot"] >= 0
        assert "payoff" not in snap or snap["payoff"] is None  # 未结束无结算

    def test_human_fold_ends_hand(self, manager: PlayManager):
        session = manager.start("p_sb", "easy")
        # SB（翻前先行动）直接 fold：本局结束，对方胜。
        session.human_move("fold")
        assert session.over
        assert session.winner == "p_bb"
        assert session.snapshot()["payoff"] is not None

    def test_unknown_seat_rejected(self, manager: PlayManager):
        with pytest.raises(PlayError):
            manager.start("p_zz", "easy")
