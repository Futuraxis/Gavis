"""Tests for the platform's generic game sessions (all three games).

Engine and solver are both seeded (MCTS seeds its own RNG), so the
AI's choices are deterministic for a fixed seed — the "play to the
end" tests below are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layer4_interface.frontend.platform.games import GAMES, PlayError
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import _BUILTIN_FAMILY, PlayManager
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

    def test_fold_does_not_leak_ai_hand_name(self, manager: PlayManager):
        """P1-2 回归：弃牌局（over=True、revealed=False）不得泄露 AI 牌型类别。

        修复前 ``ai_hand_name`` 以 ``over`` 为门，弃牌时仍据 AI 隐藏底牌+公共牌
        计算并返回（暴露口袋对/成牌类别），击穿 reveal-gate 红线。修复后以
        ``revealed``（= over 且 last_action=="showdown"）为门。
        """
        session = manager.start("texas_holdem", "p_sb", "easy")
        snapshot = manager.move(session.game_id, {"choice": "fold"})
        assert snapshot["over"] is True
        assert snapshot["revealed"] is False
        assert snapshot["ai_hole"] == []  # 原始底牌已 withheld（本来就对）
        assert snapshot["ai_hand_name"] is None  # 牌型类别也必须 withheld（P1-2 修复）

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
        # 15 款 = 月亮棋/随机五子棋/德州 + 麻将六变种 + UNO 六变体
        # （uno 基类 + seven_zero/jump_in/stacking/draw_until/strict_wild4）。
        # UNO 前端 FAMILY_BOARDS 尚无 "uno" 条目：大厅对 uno 族置灰标注
        # 「前端界面开发中」，后端契约先按 15 款锁定（与
        # test_platform_server.py::TestGames::test_list_games 同步维护）。
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
            "uno",
            "uno_seven_zero",
            "uno_jump_in",
            "uno_stacking",
            "uno_draw_until",
            "uno_strict_wild4",
        }

    def test_seat_options_consistent(self):
        assert GAMES["moon_chess"].seat_options == ("p_black", "p_white")
        assert GAMES["texas_holdem"].seat_options == ("p_sb", "p_bb")

    def test_registry_covers_all_declared_mahjong_variants(self):
        """平台注册表必须覆盖 rules/mahjong.json 声明的全部变体（v5.2 声明式）。

        防漂移守卫：注册清单由平台 games.py / train-cli/games.py / 文档各自
        手工维护时，最容易漏挂新变体 —— 曾漏挂 sichuan/changsha/taiwan
        （文档承诺六变种、大厅只有三个）。此断言把「注册表 ⊇ rules variants」
        变成自动化约束：改 rules/mahjong.json 的 variants.options 而不同步
        平台注册表，这里会立即变红。
        """
        rules = json.loads((Path("rules") / "mahjong.json").read_text(encoding="utf-8"))
        declared = set(rules["variants"]["options"].keys())
        platform_mahjong = {g for g in GAMES if g.startswith("mahjong_")}
        assert platform_mahjong == {f"mahjong_{variant}" for variant in declared}

    def test_builtin_family_covers_every_registry_game(self):
        """`_BUILTIN_FAMILY` 必须全量覆盖平台注册表（家族映射防缺项回归）。

        缺项会让 `GameInfo.family` 与快照 `family` 为 None：前端 InlineBoard
        的分发曾因 mahjong_sichuan / changsha / taiwan 缺映射而把麻将快照
        误路由到 grid 棋盘，在 `board.length` 上崩掉整个对话页。新增平台游戏
        忘记登记家族时，此断言立即变红（与 ``test_all_games_present`` 的
        9 游戏契约同步维护）。
        """
        assert set(_BUILTIN_FAMILY) == set(GAMES)


# ── Mahjong ───────────────────────────────────────────────────────────


class TestMahjong:
    def test_start_default_4p(self, manager: PlayManager):
        # 未传 player_count → 注册表默认 4 人（麻将标准人数，2 人不再兜底）。
        session = manager.start("mahjong_guangdong", "p0", "easy")
        assert session.over is False
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 14
        assert len(snap["hand_counts"]) == 4
        assert snap["phase"] == "action"
        assert snap["wall_remaining"] == 136 - 53
        assert "discard" in {a["type"] for a in snap["legal"]}

    def test_start_4p(self, manager: PlayManager):
        session = manager.start("mahjong_blood", "p0", "easy", player_count=4)
        snap = session.snapshot()
        assert len(snap["my_hand"]) == 14
        assert len(snap["melds"]) == 4
        assert snap["wall_remaining"] == 136 - 53

    def test_discard_and_claim_flow(self, manager: PlayManager):
        session = manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        legal = session.snapshot()["legal"]
        tile = next(action["tile"] for action in legal if action["type"] == "discard")
        manager.move(session.game_id, {"type": "discard", "tile": tile})
        snap = session.snapshot()
        assert tile in snap["discards"]["p0"]
        # The AI (p0) replied: either claimed/passed and we are back, or over.
        assert session.over or snap["phase"] in ("action", "claim")

    def test_ai_solver_used(self, manager: PlayManager):
        session = manager.start("mahjong_hongzhong", "p0", "easy", player_count=4)
        assert session.solver.name == "mahjong_heuristic"

    def test_player_count_validation(self, manager: PlayManager):
        with pytest.raises(PlayError, match="3 人"):
            manager.start("mahjong_guangdong", "p0", "easy", player_count=3)

    def test_snapshot_hides_ai_hand(self, manager: PlayManager):
        session = manager.start("mahjong_guangdong", "p1", "easy", player_count=4)
        snap = session.snapshot()
        # AI（庄家 p0）已先开一张；无选择 claim 被自动过后轮到 p1 摸牌
        # （14 张 / action 阶段）；若碰/杠可选则停在 claim（13 张）。
        assert len(snap["my_hand"]) in (13, 14)
        assert snap["ai_hand"] == []

    def test_snapshot_carries_family_for_every_variant(self, manager: PlayManager):
        """六个麻将变体的会话快照都必须携带 ``family == \"mahjong\"``。

        防漂移守卫：快照 family 是前端渲染分发的第一优先来源（快照自描述，
        不依赖游戏目录是否已加载）。sichuan/changsha/taiwan 曾因 `_BUILTIN_FAMILY`
        缺项而 family 为 None，导致前端把麻将快照误路由到 grid 棋盘崩溃。
        """
        for game_id in (
            "mahjong_guangdong",
            "mahjong_hongzhong",
            "mahjong_blood",
            "mahjong_sichuan",
            "mahjong_changsha",
            "mahjong_taiwan",
        ):
            session = manager.start(game_id, "p0", "easy", player_count=4)
            snap = session.snapshot()
            assert snap["family"] == "mahjong", game_id
            assert "board" not in snap, f"{game_id} 是非 grid 快照，不应含 board"

    def test_full_game_records(self, manager: PlayManager, tmp_path):
        session = manager.start("mahjong_guangdong", "p1", "easy", player_count=4)
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

    def test_no_choice_claims_auto_passed(self, manager: PlayManager):
        """无选择的 claim 回合由 run_ai 自动「过」掉，只把真实决策交给玩家。

        回归背景：麻将 claim 阶段按 胡→碰/杠→吃 三档轮询每个响应者，
        旧实现把只剩「过」的回合也浮给玩家——AI 每打一张牌玩家都要连点
        三次「过」，而且出牌按钮在 claim 阶段仍可点，产生
        「非法动作: discard {'type': 'discard', 'tile': ...}」。
        修复后：凡是轮到人类的 claim 阶段，合法动作必有碰/杠/胡/吃之一。
        """
        session = manager.start("mahjong_sichuan", "p0", "easy")
        guard = 0
        while not session.over and guard < 400:
            snap = session.snapshot()
            legal = snap["legal"]
            if not legal:
                break
            if snap["phase"] == "claim":
                types = {action["type"] for action in legal}
                assert types != {"claim_pass"}, "只剩「过」的 claim 不应浮给玩家"
                action = {"type": "claim_pass"}
            else:
                tiles = [a["tile"] for a in legal if a["type"] == "discard"]
                assert tiles, "出牌阶段手牌里的每张牌都应可打"
                action = {"type": "discard", "tile": tiles[0]}
            manager.move(session.game_id, action)
            guard += 1
        assert session.over or guard >= 400

    def test_snapshot_carries_last_discarder(self, manager: PlayManager):
        """快照携带 last_discarder（claim 操作条提示「谁打了哪张牌」）。"""
        session = manager.start("mahjong_sichuan", "p0", "easy")
        tile = next(action["tile"] for action in session.snapshot()["legal"] if action["type"] == "discard")
        snap = manager.move(session.game_id, {"type": "discard", "tile": tile})
        assert snap["last_discard"] is not None
        assert snap["last_discarder"] in {"p0", "p1", "p2", "p3"}


# ── UNO（P1-4/P1-6 平台接入）─────────────────────────────────────────


class TestUno:
    def test_start_and_snapshot_shape(self, manager: PlayManager):
        """开局快照：4 手各 7 张、family=uno、牌堆守恒、隐藏信息红线。"""
        session = manager.start("uno", "p0", "easy")
        snap = session.snapshot()
        assert snap["family"] == "uno"
        assert snap["over"] is False
        assert snap["direction"] in (1, -1)
        # 起手各 7 张；首张特殊牌可能改写先手 → AI 座位开局即行动（打出/摸牌）。
        assert snap["hand_counts"]["p0"] == 7
        assert all(1 <= c <= 8 for c in snap["hand_counts"].values())
        assert len(snap["my_hand"]) == 7
        # 隐藏信息红线：他人手牌只暴露张数；AI 手牌终局前不可见。
        assert snap["ai_hand"] == []
        # 换手 scratch 字段不得进入快照（P1-3 双保险）。
        assert "handsSnapshot" not in snap
        # 牌堆守恒：108 = 4×7 手牌 + ≥1 弃牌 + 牌堆（首张翻牌可能有后续效果）。
        assert 0 < snap["deck_count"] <= 79

    def test_move_applies_and_ai_replies(self, manager: PlayManager):
        """人类出第一张合法牌后 AI 座位轮转行动，快照正常返回。"""
        session = manager.start("uno", "p0", "easy")
        guard = 0
        moved = False
        while not session.over and guard < 20:
            snap = session.snapshot()
            if snap["turn"] == "p0" and snap["legal"]:
                action = snap["legal"][0]
                payload = {"type": action["type"]}
                for key in ("card", "color", "target"):
                    if key in action:
                        payload[key] = action[key]
                after = manager.move(session.game_id, payload)
                moved = True
                assert after["game_id"] == session.game_id
                break
            guard += 1
        assert moved, "p0 必须在某步获得合法动作（发牌后轮转）"

    def test_play_wild_requires_color_param(self, manager: PlayManager):
        """万能牌动作带 color 参数（play_wild 枚举即具体色），前端照 payload 下发。"""
        session = manager.start("uno", "p0", "easy")
        guard = 0
        while not session.over and guard < 40:
            snap = session.snapshot()
            wilds = [a for a in snap["legal"] if a.get("type") == "play_wild"]
            if snap["turn"] == "p0" and wilds:
                for w in wilds:
                    assert w.get("color") in ("r", "b", "g", "y"), "万能牌合法动作必须带具体 color"
                break
            if snap["turn"] == "p0" and snap["legal"]:
                action = snap["legal"][0]
                payload = {"type": action["type"]}
                for key in ("card", "color", "target"):
                    if key in action:
                        payload[key] = action[key]
                manager.move(session.game_id, payload)
            guard += 1

    def test_illegal_action_raises(self, manager: PlayManager):
        session = manager.start("uno", "p0", "easy")
        with pytest.raises(PlayError, match="还没轮到你|非法动作|本局已结束"):
            manager.move(session.game_id, {"type": "play", "card": "not_a_card"})

    def test_snapshot_carries_family_for_every_variant(self, manager: PlayManager):
        """六个 UNO 变体的会话快照都必须携带 ``family == \"uno\"``（前端分发第一优先来源）。"""
        for game_id in (
            "uno",
            "uno_seven_zero",
            "uno_jump_in",
            "uno_stacking",
            "uno_draw_until",
            "uno_strict_wild4",
        ):
            session = manager.start(game_id, "p0", "easy")
            snap = session.snapshot()
            assert snap["family"] == "uno", game_id
            assert "board" not in snap, f"{game_id} 是非 grid 快照，不应含 board"

    def test_engine_uses_interpreter_path(self, manager: PlayManager):
        """UNO 引擎必须固定解释器路径（allow_codegen=False）。

        编译快路径在 draw_result 等阶段与解释器分叉（compiled=1 vs interp=2），
        并在牌堆耗尽后返回 0 合法动作导致对局卡死（2026-09 平台接入实测，
        审查报告 P2-6~10 编译/解释器静默分叉组的现实表现）。
        """
        session = manager.start("uno", "p0", "easy")
        assert session.engine._compiled is None  # noqa: SLF001 — 解释器路径守卫


class TestSessionSeed:
    """每局种子派生（审计 B2：固定 seed=42 → 「再来一局」永远同牌）。"""

    def test_first_session_uses_base_seed(self, manager: PlayManager):
        session = manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        assert session.seed == 42, "首局必须等于 base seed（既有测试的确定性锚点）"

    def test_second_session_deals_different_wall(self, manager: PlayManager):
        first = manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        second = manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        assert second.seed == first.seed + 1
        assert first.snapshot()["my_hand"] != second.snapshot()["my_hand"], "第二局必须换牌墙"

    def test_record_persists_actual_seed(self, manager: PlayManager):
        # 终局才落盘历史记录，这里直接检查记录构造器带上的实际种子。
        first = manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        second = manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        assert manager._build_record(first)["seed"] == 42  # noqa: SLF001
        assert manager._build_record(second)["seed"] == 43  # noqa: SLF001
