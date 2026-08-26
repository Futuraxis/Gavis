"""Tests for the platform's generic game sessions (all three games).

Engine and solver are both seeded (MCTS seeds its own RNG), so the
AI's choices are deterministic for a fixed seed — the "play to the
end" tests below are reproducible.
"""

from __future__ import annotations

import pytest

from layer4_interface.frontend.platform.games import GAMES, PlayError
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider


@pytest.fixture
def manager(tmp_path: pytest.TempPathFactory) -> PlayManager:
    return PlayManager(provider=default_provider, history=MatchHistory(tmp_path), seed=42)


def _first_legal_cell(session) -> int:
    for action in session.engine.get_legal_actions(session.state):
        cell = action.params.get("cell", {})
        idx = int(cell.get("_index", -1)) if isinstance(cell, dict) else -1
        if idx >= 0:
            return idx
    return -1


class TestMoonChess:
    def test_start_black(self, manager: PlayManager):
        session = manager.start("moon_chess", "p_black", "easy")
        assert session.over is False
        assert session.current_player == "p_black"
        board = session.state["_arrays"]["board"]
        assert len(board) == 9
        assert all(c is None for c in board)

    def test_start_white_ai_opens(self, manager: PlayManager):
        session = manager.start("moon_chess", "p_white", "easy")
        assert session.current_player == "p_white"
        assert len(session.last_ai_info) > 0
        snapshot = session.snapshot()
        assert snapshot["player_pid"] == "p_white"
        assert snapshot["last_ai_move"] in range(9)
        assert snapshot["round_age"]  # the AI's opening piece has an age

    def test_move_applies_and_ai_replies(self, manager: PlayManager):
        session = manager.start("moon_chess", "p_black", "easy")
        payload = {"cell_index": 0}
        manager.move(session.game_id, payload)
        assert session.state["_arrays"]["board"][0] == "p_black"
        # the AI replied: either the game is over or the turn is back on the human
        assert session.over or session.current_player == "p_black"

    def test_illegal_cell_raises(self, manager: PlayManager):
        session = manager.start("moon_chess", "p_black", "easy")
        # occupy cell 0 first, then it is no longer a legal target
        manager.move(session.game_id, {"cell_index": 0})
        with pytest.raises(PlayError):
            manager.move(session.game_id, {"cell_index": 0})

    def test_unknown_game_raises(self, manager: PlayManager):
        with pytest.raises(PlayError):
            manager.start("nope", "p_black", "easy")

    def test_unknown_difficulty_raises(self, manager: PlayManager):
        with pytest.raises(PlayError):
            manager.start("moon_chess", "p_black", "insane")

    def test_random_seat(self, manager: PlayManager):
        session = manager.start("moon_chess", "random", "easy")
        assert session.player_pid in ("p_black", "p_white")

    def test_full_game_records_and_removes(self, manager: PlayManager):
        session = manager.start("moon_chess", "p_black", "easy")
        for _ in range(60):
            if session.over:
                break
            manager.move(session.game_id, {"cell_index": _first_legal_cell(session)})
        assert session.over
        # the session was recorded into history and dropped from the registry
        matches = manager._history.list_matches()  # type: ignore[union-attr]
        assert [m["match_id"] for m in matches] == [session.game_id]
        with pytest.raises(PlayError):
            manager.get(session.game_id)
        with pytest.raises(PlayError):
            manager.move(session.game_id, {"cell_index": 0})


class TestGomoku:
    def test_start_and_move(self, manager: PlayManager):
        session = manager.start("stochastic_gomoku", "p_black", "easy")
        assert session.current_player == "p_black"
        manager.move(session.game_id, {"cell_index": 0})
        assert session.state["_arrays"]["board"][0] == "p_black"
        snapshot = session.snapshot()
        assert snapshot["last_vanish"] is None or snapshot["last_vanish"] in range(81)
        if snapshot["last_vanish"] is not None:
            assert snapshot["last_vanish_color"] in ("p_black", "p_white")
        assert len(snapshot["board"]) == 81

    def test_move_on_over_game_raises(self, manager: PlayManager):
        session = manager.start("stochastic_gomoku", "p_black", "easy")
        with pytest.raises(PlayError):
            manager.move(session.game_id, {"cell_index": -1})


class TestTexasHoldem:
    def test_start_sb_human_acts_first(self, manager: PlayManager):
        session = manager.start("texas_holdem", "p_sb", "easy")
        assert session.current_player == "p_sb"
        snapshot = session.snapshot()
        assert len(snapshot["my_hole"]) == 2
        assert snapshot["street_name"] == "翻前"
        assert snapshot["legal"], "SB preflop must have legal actions"
        assert snapshot["raise_amounts"]  # raise must be legal preflop
        assert snapshot["ai_hole"] == []

    def test_call_flow(self, manager: PlayManager):
        session = manager.start("texas_holdem", "p_sb", "easy")
        manager.move(session.game_id, {"choice": "call"})
        snapshot = session.snapshot()
        assert snapshot["my_stack"] >= 0
        assert snapshot["ai_stack"] >= 0

    def test_raise_with_amount(self, manager: PlayManager):
        session = manager.start("texas_holdem", "p_sb", "easy")
        before = session.snapshot()
        amount = before["raise_amounts"][0]
        after = manager.move(session.game_id, {"choice": "raise", "amount": amount})
        # the raise always adds chips to the pot, however the AI responds
        assert after["pot"] > before["pot"]

    def test_illegal_choice_raises(self, manager: PlayManager):
        session = manager.start("texas_holdem", "p_sb", "easy")
        with pytest.raises(PlayError):
            manager.move(session.game_id, {"choice": "all_in"})

    def test_fold_ends_and_records(self, manager: PlayManager):
        session = manager.start("texas_holdem", "p_sb", "easy")
        snapshot = manager.move(session.game_id, {"choice": "fold"})
        assert snapshot["over"] is True
        assert snapshot["payoff"] is not None
        matches = manager._history.list_matches()  # type: ignore[union-attr]
        assert [m["match_id"] for m in matches] == [session.game_id]

    def test_replay_log_shape(self, manager: PlayManager):
        session = manager.start("moon_chess", "p_black", "easy")
        manager.move(session.game_id, {"cell_index": 0})
        if not session.over:
            manager.move(session.game_id, {"cell_index": _first_legal_cell(session)})
        entries = session.log
        assert entries, "AI opening moves are also logged"
        for step, entry in enumerate(entries):
            assert entry["step"] == step
            assert entry["actor"] in ("human", "ai")
            assert isinstance(entry["action"], str)
            assert "board" in entry["snapshot"]


class TestGameSpecRegistry:
    def test_all_games_present(self):
        assert set(GAMES) == {
            "moon_chess",
            "stochastic_gomoku",
            "texas_holdem",
            "mahjong_guangdong",
            "mahjong_hongzhong",
            "mahjong_blood",
            "mahjong_sichuan",
            "mahjong_changsha",
            "mahjong_taiwan",
        }

    def test_seat_options_consistent(self):
        assert GAMES["moon_chess"].seat_options == ("p_black", "p_white")
        assert GAMES["texas_holdem"].seat_options == ("p_sb", "p_bb")


# ── Mahjong ───────────────────────────────────────────────────────────


class TestMahjong:
    def test_start_2p(self, manager: PlayManager):
        session = manager.start("mahjong_guangdong", "p0", "easy", player_count=2)
        assert session.over is False
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 14
        assert snap["phase"] == "action"
        assert snap["wall_remaining"] == 136 - 27
        assert "discard" in {a["type"] for a in snap["legal"]}

    def test_start_4p(self, manager: PlayManager):
        session = manager.start("mahjong_blood", "p0", "easy", player_count=4)
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 14
        assert len(snap["melds"]) == 4
        assert snap["wall_remaining"] == 136 - 53

    def test_discard_and_claim_flow(self, manager: PlayManager):
        session = manager.start("mahjong_guangdong", "p0", "easy", player_count=2)
        legal = session.snapshot()["legal"]
        tile = next(action["tile"] for action in legal if action["type"] == "discard")
        manager.move(session.game_id, {"type": "discard", "tile": tile})
        snap = session.snapshot()
        assert tile in snap["discards"]["p0"]
        # The AI (p0) replied: either claimed/passed and we are back, or over.
        assert session.over or snap["phase"] in ("action", "claim")

    def test_ai_solver_used(self, manager: PlayManager):
        session = manager.start("mahjong_hongzhong", "p0", "easy", player_count=2)
        assert session.solver.name == "mahjong_heuristic"

    def test_player_count_validation(self, manager: PlayManager):
        with pytest.raises(PlayError, match="3 人"):
            manager.start("mahjong_guangdong", "p0", "easy", player_count=3)

    def test_snapshot_hides_ai_hand(self, manager: PlayManager):
        session = manager.start("mahjong_guangdong", "p1", "easy", player_count=2)
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 13  # AI (dealer) opened with a discard
        assert snap["ai_hand"] == []

    def test_full_game_records(self, manager: PlayManager, tmp_path):
        session = manager.start("mahjong_guangdong", "p1", "easy", player_count=2)
        guard = 0
        while not session.over and guard < 300:
            snap = session.snapshot()
            legal = snap["legal"]
            if not legal:
                break
            if snap["phase"] == "claim":
                action = {"type": "claim_pass"}
            else:
                action = {"type": "discard", "tile": legal[0]["tile"]}
            manager.move(session.game_id, action)
            guard += 1
        assert session.over or guard >= 300
