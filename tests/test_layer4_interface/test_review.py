"""Tests for the Layer 4 post-match review analyzer (C4)."""

from __future__ import annotations

import pytest

import layer4_interface.review.analyzer as analyzer_mod
from layer4_interface.review import KeyNode, ReviewReport, analyze


def _moon_match() -> dict:
    """A moon-chess record where the human (p_black) loses to the AI."""
    boards = [
        ["p_black", None, None, None, None, None, None, None, None],
        ["p_black", None, "p_white", None, None, None, None, None, None],
        ["p_black", "p_black", "p_white", None, None, None, None, None, None],
        ["p_black", "p_black", "p_white", None, None, "p_white", None, None, None],
        ["p_black", "p_black", "p_white", "p_black", None, "p_white", None, None, None],
        ["p_black", "p_black", "p_white", "p_black", None, "p_white", None, None, "p_white"],
    ]
    moves = []
    for i, board in enumerate(boards):
        over = i == len(boards) - 1
        moves.append(
            {
                "step": i,
                "actor": "human" if i % 2 == 0 else "ai",
                "action": f"cell_{i}",
                "snapshot": {
                    "player_pid": "p_black",
                    "board": board,
                    "turn": "p_white" if i % 2 == 0 else "p_black",
                    "winner": "p_white" if over else None,
                    "over": over,
                    "round": i + 1,
                },
            }
        )
    return _record("moon_chess", "p_black", "p_white", "p_white", moves)


def _texas_match() -> dict:
    """A heads-up showdown where the human (p_sb) loses on the river."""
    moves = [
        {
            "step": 0,
            "actor": "human",
            "action": "act:call:2",
            "snapshot": _poker_snapshot(over=False, winner=None, payoff=None, revealed=False, ai_hole=[]),
        },
        {
            "step": 1,
            "actor": "ai",
            "action": "act:call:2",
            "snapshot": _poker_snapshot(over=False, winner=None, payoff=None, revealed=False, ai_hole=[]),
        },
        {
            "step": 2,
            "actor": "human",
            "action": "act:call:2",
            "snapshot": _poker_snapshot(over=True, winner="p_bb", payoff=-1, revealed=True, ai_hole=["sA", "sK"]),
        },
    ]
    return _record("texas_holdem", "p_sb", "p_bb", "p_bb", moves)


def _poker_snapshot(*, over: bool, winner: str | None, payoff: int | None, revealed: bool, ai_hole: list[str]) -> dict:
    """Build one poker snapshot (player perspective, public fields)."""
    return {
        "player_pid": "p_sb",
        "ai_pid": "p_bb",
        "over": over,
        "winner": winner,
        "phase": "game_over" if over else "betting",
        "pot": 4,
        "community": ["d9", "h3", "cJ"] if over else [],
        "my_hole": ["hA", "hK"],
        "ai_hole": ai_hole,
        "revealed": revealed,
        "payoff": payoff,
        "last_action": "call",
    }


def _record(game_id: str, player_pid: str, ai_pid: str, winner: str, moves: list[dict]) -> dict:
    """Wrap moves into the ``MatchHistory.get`` record shape."""
    return {
        "match_id": "review0001",
        "game_id": game_id,
        "player_pid": player_pid,
        "ai_pid": ai_pid,
        "difficulty": "easy",
        "winner": winner,
        "over": True,
        "moves": moves,
        "meta": {
            "match_id": "review0001",
            "game_id": game_id,
            "player_pid": player_pid,
            "ai_pid": ai_pid,
            "difficulty": "easy",
            "winner": winner,
            "over": True,
            "moves": len(moves),
        },
    }


class TestAnalyzeMoonChess:
    def test_returns_key_nodes_and_summary(self) -> None:
        report = analyze(_moon_match())
        assert isinstance(report, ReviewReport)
        assert len(report.key_nodes) >= 1
        assert any(node.kind == "turning_point" for node in report.key_nodes)
        assert report.improvement != ""
        assert "月亮棋" in report.summary
        assert "AI 获胜" in report.summary

    def test_detects_winning_move(self) -> None:
        report = analyze(_moon_match())
        winning = [node for node in report.key_nodes if node.kind == "winning_move"]
        assert winning
        assert winning[0].step == 5  # AI's final move (index 5)

    def test_empty_moves_degrades_gracefully(self) -> None:
        report = analyze(_record("moon_chess", "p_black", "p_white", "p_white", []))
        assert report.key_nodes == []
        assert report.improvement == "继续巩固优势，稳扎稳打"
        assert "共 0 手" in report.summary


class TestAnalyzeTexasNoLeak:
    def test_key_nodes_present(self) -> None:
        report = analyze(_texas_match())
        assert len(report.key_nodes) >= 1

    def test_no_opponent_hole_cards_in_text(self) -> None:
        report = analyze(_texas_match())
        text = report.improvement + report.summary + " ".join(node.why for node in report.key_nodes)
        for secret in ("sA", "sK", "ai_hole", "_bb_hole"):
            assert secret not in text

    def test_blunder_not_fabricated_without_midgame_signal(self) -> None:
        """非 grid 族没有中盘评分信号：不再把玩家的终局手机械标成昏招。

        旧版给中盘快照伪造 0.0 分，唯一的「评分落差」永远出现在终局
        结算，于是无论最后一手是 fold 还是 call，报告都把「玩家最后
        一手」指为昏招——空有形式、毫无对局内容。现在无信号 → 无昏招
        节点；对具体失误的讲评交给 get_match_review 的 LLM 叙事（时间
        线里有完整动作与牌面）。
        """
        report = analyze(_texas_match())
        assert not any(node.kind == "blunder" for node in report.key_nodes)


class TestFallbackPath:
    def test_fallback_when_c2_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(analyzer_mod, "_try_import_c2_evaluate", lambda: None)
        report = analyze(_moon_match())
        assert len(report.key_nodes) >= 1
        assert report.improvement != ""


class TestC2Preference:
    def test_prefers_c2_evaluate_when_state_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_evaluate = lambda state, viewer, engine: {"score": 0.9}  # noqa: E731
        monkeypatch.setattr(analyzer_mod, "_try_import_c2_evaluate", lambda: fake_evaluate)
        match = _moon_match()
        for move in match["moves"]:
            move["snapshot"]["state"] = {"env": {}}
            move["snapshot"]["engine"] = object()
        report = analyze(match)
        # Constant 0.9 scores → no jump, so the turning point degenerates
        # to step 0 (proving C2's score was used instead of the board proxy).
        turning = next(node for node in report.key_nodes if node.kind == "turning_point")
        assert turning.step == 0


class TestKeyNode:
    def test_dataclass_shape(self) -> None:
        node = KeyNode(step=3, kind="blunder", why="x")
        assert node.step == 3
        assert node.kind == "blunder"
        assert node.why == "x"
        assert node.what == ""  # 默认空（旧记录 / 无动作文本时不炸）

    def test_nodes_carry_recorded_action(self) -> None:
        """关键手要带动作内容（what）——复盘卡与 LLM 讲解都能说出「哪一手」。"""
        report = analyze(_moon_match())
        assert report.key_nodes
        assert all(node.what for node in report.key_nodes)  # 每个节点都有 moves[].action
        winning = next(node for node in report.key_nodes if node.kind == "winning_move")
        assert winning.what == "cell_5"
