"""Tests for MatchHistory — atomic persistence of finished matches."""

from __future__ import annotations

import json

import pytest

from layer4_interface.frontend.platform.history import HistoryError, MatchHistory


def _match(
    match_id: str = "aabbccdd",
    game_id: str = "moon_chess",
    moves: int = 3,
    started_at: str = "2026-08-03T10:00:00+08:00",
) -> dict:
    return {
        "match_id": match_id,
        "game_id": game_id,
        "player_pid": "p_black",
        "ai_pid": "p_white",
        "difficulty": "normal",
        "seed": 42,
        "started_at": started_at,
        "finished_at": "2026-08-03T10:01:00+08:00",
        "winner": "p_black",
        "over": True,
        "moves": [{"step": i, "actor": "ai", "action": f"cell_{i}", "snapshot": {"board": []}} for i in range(moves)],
    }


class TestRecord:
    def test_record_writes_file(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        match_id = store.record(_match())
        assert (tmp_path / f"{match_id}.json").is_file()

    def test_record_embeds_meta(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        store.record(_match(moves=3))
        record = json.loads((tmp_path / "aabbccdd.json").read_text(encoding="utf-8"))
        assert record["meta"]["match_id"] == "aabbccdd"
        assert record["meta"]["moves"] == 3
        assert record["meta"]["winner"] == "p_black"

    def test_record_missing_match_id(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        with pytest.raises(HistoryError):
            store.record({"game_id": "moon_chess", "moves": []})


class TestList:
    def test_list_newest_first(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        store.record(_match("aa000001", started_at="2026-08-03T09:00:00+08:00"))
        store.record(_match("aa000002", started_at="2026-08-03T11:00:00+08:00"))
        metas = store.list_matches()
        assert [m["match_id"] for m in metas] == ["aa000002", "aa000001"]

    def test_list_filter_and_limit(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        store.record(_match("aa000001", game_id="moon_chess"))
        store.record(_match("aa000002", game_id="texas_holdem"))
        store.record(_match("aa000003", game_id="moon_chess"))
        assert len(store.list_matches(game_id="moon_chess")) == 2
        assert len(store.list_matches(limit=1)) == 1

    def test_list_skips_corrupt_file(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        store.record(_match("aa000001"))
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert len(store.list_matches()) == 1


class TestGet:
    def test_get_round_trip(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        store.record(_match(moves=2))
        record = store.get("aabbccdd")
        assert record["winner"] == "p_black"
        assert len(record["moves"]) == 2

    def test_get_missing_raises(self, tmp_path: pytest.TempPathFactory):
        with pytest.raises(HistoryError):
            MatchHistory(tmp_path).get("nope0000")

    def test_get_corrupt_raises(self, tmp_path: pytest.TempPathFactory):
        (tmp_path / "badbad01.json").write_text("garbage", encoding="utf-8")
        with pytest.raises(HistoryError):
            MatchHistory(tmp_path).get("badbad01")


class TestDelete:
    def test_delete_removes_file(self, tmp_path: pytest.TempPathFactory):
        store = MatchHistory(tmp_path)
        store.record(_match("aa000001"))
        store.delete("aa000001")
        with pytest.raises(HistoryError):
            store.delete("aa000001")
